"""
Distributed Topological Synapse — Multi-GPU landmark sharing via NCCL.

Scales the warp-cortex agent pool beyond a single GPU by:
  1. Partitioning agents across GPUs (each GPU owns a slice of the pool).
  2. Keeping a coherent shared landmark buffer via periodic all_gather.
  3. Using local writes + global reads so each GPU can update its own
     partition with zero contention, then broadcast changes cheaply.

Inherits from synapse.TopologicalSynapse — all local APIs (update_landmarks,
evict_stale, inject_embedding, topo_features, etc.) work identically.
sync() is the only addition: it merges landmarks across GPUs.

Requirements:
  - PyTorch >= 2.0
  - NCCL backend (Linux) or Gloo backend (Windows / fallback)
  - Launch via torchrun / torch.distributed.launch

Usage:
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')

    synapse = DistributedSynapse(
        dim=896, max_landmarks=128,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    # Each rank writes its own landmarks normally...
    synapse.update_landmarks(keys, values, attn_scores)
    # ...then call sync to merge across GPUs
    synapse.sync()
    # All ranks now see the globally best landmarks
    k, v = synapse.get_context()
"""

import torch
import time
from typing import Optional, cast

from .synapse import TopologicalSynapse


class DistributedSynapse(TopologicalSynapse):
    """
    Multi-GPU version of TopologicalSynapse.

    Each GPU maintains a *local* partition of landmarks and periodically
    merges with other GPUs via ``sync()``.  The merge keeps the top-k
    landmarks globally (by importance score), so memory stays bounded
    regardless of world size.
    """

    def __init__(
        self,
        dim: int,
        max_landmarks: int = 128,
        world_size: int = 1,
        rank: int = 0,
        device: str = 'cuda',
        adaptive_k: bool = True,
        k_min: int = 16,
        k_max: int = 128,
        ttl_seconds: float = 300.0,
        backend: str = 'nccl',
    ):
        super().__init__(
            dim=dim, max_landmarks=max_landmarks, device=device,
            adaptive_k=adaptive_k, k_min=k_min, k_max=k_max,
            ttl_seconds=ttl_seconds,
        )
        self.world_size = world_size
        self.rank = rank
        self.backend = backend

        # Gather buffers (allocated once to avoid repeated allocation)
        self._gather_keys: Optional[torch.Tensor] = None
        self._gather_values: Optional[torch.Tensor] = None
        self._gather_scores: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Distributed sync — merge landmarks across GPUs
    # ------------------------------------------------------------------

    def sync(self):
        """
        All-gather local landmarks from every rank, then keep the global
        top-k by importance score.

        After ``sync()``, every rank has the same coherent landmark buffer
        containing the best landmarks from all GPUs.
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return  # single-GPU — nothing to do

        if not dist.is_initialized() or self.world_size <= 1:
            return

        # Pad locals to max_landmarks
        n = self.max_landmarks
        dim = cast(int, self.dim)
        local_k = self.landmark_keys[:n].contiguous()
        local_v = self.landmark_values[:n].contiguous()
        local_s = self.landmark_scores[:n].contiguous()

        # Allocate gather buffers once
        if self._gather_keys is None:
            self._gather_keys = torch.zeros(
                self.world_size, n, dim, device=self.device
            )
            self._gather_values = torch.zeros_like(self._gather_keys)
            self._gather_scores = torch.zeros(
                self.world_size, n, device=self.device
            )

        gather_keys = cast(torch.Tensor, self._gather_keys)
        gather_values = cast(torch.Tensor, self._gather_values)
        gather_scores = cast(torch.Tensor, self._gather_scores)

        # All-gather across ranks
        dist.all_gather_into_tensor(
            gather_keys.view(self.world_size * n, dim),
            local_k,
        )
        dist.all_gather_into_tensor(
            gather_values.view(self.world_size * n, dim),
            local_v,
        )
        dist.all_gather_into_tensor(
            gather_scores.view(self.world_size * n),
            local_s,
        )

        # Merge: keep global top-k
        all_keys = gather_keys.view(-1, dim)     # [W*N, D]
        all_values = gather_values.view(-1, dim)
        all_scores = gather_scores.view(-1)            # [W*N]

        # Filter zero-score (empty) slots
        nonzero = all_scores > 0
        if nonzero.sum() == 0:
            self.count = 0
            return

        valid_scores = all_scores[nonzero]
        valid_keys = all_keys[nonzero]
        valid_values = all_values[nonzero]

        top_k = min(self.max_landmarks, valid_scores.shape[0])
        top_scores, top_idx = torch.topk(valid_scores, top_k)

        self.landmark_keys[:top_k] = valid_keys[top_idx]
        self.landmark_values[:top_k] = valid_values[top_idx]
        self.landmark_scores[:top_k] = top_scores
        self.landmark_timestamps[:top_k] = time.time()
        self.count = top_k

        # Zero out the rest
        if top_k < self.max_landmarks:
            self.landmark_keys[top_k:] = 0
            self.landmark_values[top_k:] = 0
            self.landmark_scores[top_k:] = 0
            self.landmark_timestamps[top_k:] = 0

    @property
    def stats(self):
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "count": self.count,
            "max_landmarks": self.max_landmarks,
        }
