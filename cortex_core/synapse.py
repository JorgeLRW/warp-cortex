"""
Unified Topological Synapse — shared landmark memory for warp-cortex.

Replaces three separate systems with ONE manifold:
  - topological_synapse.py  (adaptive-k, LRU, topo_features)
  - cortex_engine.py inline TopologicalSynapse  (KV-tuple landmarks)
  - cortex_attention.py SynapseBuffer  (ring buffer)

All landmarks now live in one structure.  Stream-injected embeddings
enter as first-class landmarks alongside attention-derived ones.
The TopologicalSynapse computes topology features (density, spread,
coverage) over the full landmark manifold, feeding the CortexAttention
gate with real geometric information.

Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │  TopologicalSynapse                                       │
    │                                                           │
    │  ┌─────────────────────┐   ┌────────────────────────┐    │
    │  │  Attention Landmarks │   │  Injection Landmarks    │    │
    │  │  [max_landmarks, D]  │   │  [max_injections, D]    │    │
    │  │  from update_landmarks│  │  from inject_embedding   │    │
    │  └─────────┬───────────┘   └──────────┬─────────────┘    │
    │            │                           │                  │
    │            └───────────┬───────────────┘                  │
    │                        ▼                                  │
    │              topo_features()                               │
    │              (density, spread, coverage)                   │
    │                                                           │
    │  ┌──────────────────────┐   ┌──────────────────────┐     │
    │  │  KV-cache landmarks   │   │  Thought memory       │     │
    │  │  (tuple of KV slices) │   │  (text, vector) pairs │     │
    │  │  for side agents      │   │  for referential inject│     │
    │  └──────────────────────┘   └──────────────────────┘     │
    └───────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Optional, List


class TopologicalSynapse:
    """
    Shared memory for landmarks — token-level from attention AND
    embedding-level from verification / delegation workers.

    Two landmark pools (unified manifold, partitioned storage):
      attention landmarks — set by update_landmarks() from model attention
      injection landmarks — set by inject_embedding() from verified claims

    Both pools contribute to topo_features (density, spread, coverage).
    CortexAttention cross-attends to injection landmarks via the gate.
    Side agents get KV-cache context via get_landmarks().

    Also maintains:
      - KV-cache landmarks (tuple format for side agent coordination)
      - Thought memory (text-level injection, push / read)
      - Adaptive k: landmark count scales with attention entropy
      - LRU eviction: stale landmarks evicted after ttl_seconds
    """

    def __init__(self, dim=None, max_landmarks=128, max_injections=128,
                 device='cuda', adaptive_k=True, k_min=16, k_max=128,
                 ttl_seconds=300.0):
        self.dim = dim
        self.max_landmarks = max_landmarks
        self.max_injections = max_injections
        self.device = device

        # --- Attention landmarks (from model attention scores) ---
        if dim is not None:
            self.landmark_keys = torch.zeros(max_landmarks, dim, device=device)
            self.landmark_values = torch.zeros(max_landmarks, dim, device=device)
            self.landmark_scores = torch.zeros(max_landmarks, device=device)
            self.landmark_timestamps = torch.zeros(max_landmarks, dtype=torch.float64)
        self.count = 0

        # --- Injection landmarks (from stream injection / delegation) ---
        if dim is not None:
            self.injection_keys = torch.zeros(max_injections, dim, device=device)
            self.injection_values = torch.zeros(max_injections, dim, device=device)
            self.injection_scores = torch.zeros(max_injections, device=device)
            self.injection_timestamps = torch.zeros(max_injections, dtype=torch.float64)
        self.injection_count = 0

        # --- KV-cache landmarks (for engine side agents) ---
        self._kv_landmarks = None

        # --- Thought memory (text-level) ---
        self.thought_memory = []

        # --- Adaptive-k ---
        self.adaptive_k = adaptive_k
        self.k_min = k_min
        self.k_max = k_max

        # --- LRU ---
        self.ttl_seconds = ttl_seconds

    # ==================================================================
    # Adaptive k
    # ==================================================================

    def compute_adaptive_k(self, attention_scores) -> int:
        """
        Determine how many landmarks to keep based on attention entropy.
        High entropy (diffuse) → more landmarks; low entropy → fewer.
        """
        if not self.adaptive_k:
            return self.max_landmarks

        if attention_scores.dim() >= 3:
            probs = F.softmax(attention_scores.float().sum(dim=(0, 1, 2)), dim=-1)
        elif attention_scores.dim() == 1:
            probs = F.softmax(attention_scores.float(), dim=-1)
        else:
            probs = F.softmax(attention_scores.float().flatten(), dim=-1)

        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        max_entropy = torch.tensor(probs.shape[-1], dtype=torch.float).log().item()
        norm_entropy = entropy / max(max_entropy, 1e-10)

        k = int(self.k_min + (self.k_max - self.k_min) * norm_entropy)
        return max(self.k_min, min(self.k_max, k))

    # ==================================================================
    # Attention landmarks (flat tensor [B, Seq, Dim])
    # ==================================================================

    def update_landmarks(self, keys, values, attention_scores):
        """
        From main model attention.  Identifies high-importance tokens
        and promotes them to landmarks.

        keys, values: [Batch, Seq, Dim]
        attention_scores: [Batch, Heads, Seq, Seq] or [Seq]
        """
        k = self.compute_adaptive_k(attention_scores)
        k = min(k,
                attention_scores.shape[-1] if attention_scores.dim() > 1
                else attention_scores.shape[0])

        token_importance = (
            attention_scores.sum(dim=(0, 1, 2))
            if attention_scores.dim() >= 3
            else attention_scores
        )

        actual_k = min(k, token_importance.shape[0], self.max_landmarks)
        top_scores, top_indices = torch.topk(token_importance, actual_k)

        self.landmark_keys[:actual_k] = keys[0, top_indices, :]
        self.landmark_values[:actual_k] = values[0, top_indices, :]
        self.landmark_scores[:actual_k] = top_scores
        self.landmark_timestamps[:actual_k] = time.time()
        self.count = actual_k

    # ==================================================================
    # KV-cache landmarks (for engine side agents)
    # ==================================================================

    def update_kv_landmarks(self, past_key_values, query_states=None,
                            keep_ratio=0.1):
        """
        Compress full KV cache into landmarks for side agents.

        past_key_values: tuple(layers) of tuple(key[B,H,S,D], value[B,H,S,D])
        query_states: [B,H,1,D] for attention-based selection (optional)
        """
        if past_key_values is None:
            return

        new_kv = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            seq_len = k.shape[2]
            if seq_len < 100:
                new_kv.append((k, v))
                continue

            if query_states is not None:
                attn_scores = torch.matmul(
                    query_states, k.transpose(-1, -2)
                ).squeeze(2)  # [B, H, S]
                global_scores = attn_scores.sum(dim=1)  # [B, S]
                k_val = min(64, seq_len)
                _, top_indices = torch.topk(global_scores, k_val, dim=-1)
                indices, _ = torch.sort(top_indices, dim=-1)
                indices = indices.squeeze(0)
            else:
                indices = torch.cat([
                    torch.arange(0, 10, device=self.device),
                    torch.arange(10, seq_len - 50, 10, device=self.device),
                    torch.arange(seq_len - 50, seq_len, device=self.device),
                ]).long()

            indices = indices[indices < seq_len]
            k_selected = k.index_select(2, indices)
            v_selected = v.index_select(2, indices)
            new_kv.append((k_selected, v_selected))

        self._kv_landmarks = tuple(new_kv)

    def get_landmarks(self):
        """Side agents call this to get KV-cache context."""
        return self._kv_landmarks

    # ==================================================================
    # Injection landmarks (from stream injection / delegation)
    # ==================================================================

    def inject_embedding(self, embedding, score=1.0):
        """
        Push a verified claim / delegation result as an injection landmark.

        When the buffer is full, evicts the LOWEST-SCORE injection
        (ties broken by oldest timestamp).  High-confidence landmarks
        (score=1.0) stubbornly resist eviction; low-confidence ones
        (score=0.5 for failed claims, 0.4 for speculative thoughts)
        are overwritten first.  This keeps the k-landmarks biased
        toward ground truth.
        """
        vec = embedding.detach()
        if vec.device != torch.device(self.device):
            vec = vec.to(self.device)

        if self.injection_count < self.max_injections:
            idx = self.injection_count
            self.injection_count += 1
        else:
            # Score-weighted LRU: evict weakest landmark.
            # Composite = score + recency_bonus (0..1 normalised by TTL).
            scores = self.injection_scores[:self.injection_count]
            ages = time.time() - self.injection_timestamps[:self.injection_count]
            recency = 1.0 - (ages / max(self.ttl_seconds, 1.0)).clamp(0, 1)
            composite = scores + 0.2 * recency  # score dominates
            idx = int(composite.argmin().item())

        self.injection_keys[idx] = vec
        self.injection_values[idx] = vec  # key == value for injections
        self.injection_scores[idx] = score
        self.injection_timestamps[idx] = time.time()

    def get_injection_context(self):
        """
        Return injection landmarks for CortexAttention cross-attention.
        Auto-evicts stale entries before returning.

        Returns (keys, values): each [count, dim], or (None, None) if empty.
        """
        self._evict_stale_injections()
        if self.injection_count == 0:
            return None, None
        return (self.injection_keys[:self.injection_count],
                self.injection_values[:self.injection_count])

    def _evict_stale_injections(self):
        """Remove injection landmarks older than ttl_seconds."""
        if self.injection_count == 0 or self.ttl_seconds <= 0:
            return 0
        now = time.time()
        ages = now - self.injection_timestamps[:self.injection_count]
        mask = ages < self.ttl_seconds
        kept = int(mask.sum().item())
        if kept == self.injection_count:
            return 0
        evicted = self.injection_count - kept
        if kept > 0:
            keep_idx = mask.nonzero(as_tuple=True)[0]
            self.injection_keys[:kept] = self.injection_keys[keep_idx].clone()
            self.injection_values[:kept] = self.injection_values[keep_idx].clone()
            self.injection_scores[:kept] = self.injection_scores[keep_idx].clone()
            self.injection_timestamps[:kept] = self.injection_timestamps[keep_idx].clone()
        self.injection_keys[kept:self.injection_count] = 0
        self.injection_values[kept:self.injection_count] = 0
        self.injection_scores[kept:self.injection_count] = 0
        self.injection_timestamps[kept:self.injection_count] = 0
        self.injection_count = kept
        return evicted

    # ==================================================================
    # Thought memory (text-level injection)
    # ==================================================================

    def push_thought(self, text, vector=None):
        self.thought_memory.append((text, vector))

    def read_thought(self):
        if self.thought_memory:
            return self.thought_memory.pop(0)
        return None, None

    # ==================================================================
    # LRU eviction for attention landmarks
    # ==================================================================

    def evict_stale(self):
        """Remove attention landmarks older than ttl_seconds."""
        if self.count == 0 or self.ttl_seconds <= 0:
            return 0
        now = time.time()
        ages = now - self.landmark_timestamps[:self.count]
        mask = ages < self.ttl_seconds
        kept = int(mask.sum().item())
        if kept == self.count:
            return 0
        evicted = self.count - kept
        if kept > 0:
            keep_idx = mask.nonzero(as_tuple=True)[0]
            self.landmark_keys[:kept] = self.landmark_keys[keep_idx].clone()
            self.landmark_values[:kept] = self.landmark_values[keep_idx].clone()
            self.landmark_scores[:kept] = self.landmark_scores[keep_idx].clone()
            self.landmark_timestamps[:kept] = self.landmark_timestamps[keep_idx].clone()
        self.landmark_keys[kept:self.count] = 0
        self.landmark_values[kept:self.count] = 0
        self.landmark_scores[kept:self.count] = 0
        self.landmark_timestamps[kept:self.count] = 0
        self.count = kept
        return evicted

    # ==================================================================
    # Full context (union of attention + injection landmarks)
    # ==================================================================

    def get_context(self):
        """
        Return ALL landmarks (attention + injection) as [N, dim] tensors.
        Auto-evicts stale from both pools.
        """
        self.evict_stale()
        self._evict_stale_injections()

        parts_k, parts_v = [], []
        if self.count > 0:
            parts_k.append(self.landmark_keys[:self.count])
            parts_v.append(self.landmark_values[:self.count])
        if self.injection_count > 0:
            parts_k.append(self.injection_keys[:self.injection_count])
            parts_v.append(self.injection_values[:self.injection_count])

        if not parts_k:
            d = self.dim or 1
            return (torch.zeros(0, d, device=self.device),
                    torch.zeros(0, d, device=self.device))

        return torch.cat(parts_k, dim=0), torch.cat(parts_v, dim=0)

    # ==================================================================
    # Topology features — computed over the full landmark manifold
    # ==================================================================

    def topo_features(self):
        """
        Compute topology descriptors from ALL landmarks (attention + injection).

        Returns (density, spread, coverage):
          density:  mean pairwise cosine similarity (how clustered)
          spread:   std of landmark L2 norms (how uniform the energy is)
          coverage: total_count / total_capacity (how full the buffer is)

        Cost: O(k²·d) where k = total landmark count.  Negligible for k ≤ 256.
        """
        total = self.count + self.injection_count
        total_capacity = self.max_landmarks + self.max_injections
        coverage = total / max(total_capacity, 1)

        if total < 2:
            return 0.0, 0.0, coverage

        parts = []
        if self.count > 0:
            parts.append(self.landmark_keys[:self.count])
        if self.injection_count > 0:
            parts.append(self.injection_keys[:self.injection_count])
        keys = torch.cat(parts, dim=0)

        # Density: mean pairwise cosine similarity
        normed = F.normalize(keys, dim=-1)
        sim_matrix = normed @ normed.T
        k = keys.shape[0]
        density = (sim_matrix.sum() - k) / max(k * (k - 1), 1)
        density = density.item()

        # Spread: std of L2 norms
        spread = keys.norm(dim=-1).std().item()

        return density, spread, coverage


class TopologicalSideAgent(nn.Module):
    """
    A Side Agent that attends ONLY to the Topological Synapse landmarks.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x, synapse: TopologicalSynapse):
        B, _, D = x.shape
        k_mem, v_mem = synapse.get_context()
        if k_mem.shape[0] == 0:
            return x
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_mem.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)
        v = v_mem.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(B, 1, D)
        return self.o_proj(context)
