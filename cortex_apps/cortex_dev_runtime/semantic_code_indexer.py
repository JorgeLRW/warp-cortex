"""
Multi-Aspect Semantic Code Indexer (Z).
======================================
Projects code files, symbols, and tests into multi-aspect semantic representations:
  - Band 0 (Architecture): class signatures, module docstrings, public interfaces
  - Band 1 (Logic & Flow): internal implementations, control flow, AST calls
  - Band 2 (Invariants & Testing): assertions, test assertions, error checks
  - Band 3 (Workload & Perf): loop structures, configuration params, batch sizes

Maintains a unified embedding tensor matrix for rapid semantic retrieval,
topological-semantic coupling, and similarity search without external API dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cortex_apps.cortex_dev_runtime.dev_runtime_api import CodeSymbol, FileNode, TestNode


class MultiAspectCodeIndexer:
    """
    Computes and maintains multi-aspect semantic representations Z for code entities.
    Vector dimension: d = 64 per band x 4 bands = 256 total dimensions.
    """

    def __init__(self, d_band: int = 64, device: str = "cpu"):
        self.d_band = d_band
        self.total_dim = d_band * 4
        self.device = torch.device(device)
        
        # Entity keys -> index in matrix
        self.entity_ids: List[str] = []
        self.entity_index: Dict[str, int] = {}
        
        # Z tensor: [N, 4, d_band]
        self.Z: Optional[torch.Tensor] = None

    def _hash_to_features(self, text: str, band_seed: int, dim: int) -> torch.Tensor:
        """Deterministically hashes n-grams and tokens to a normalized feature vector."""
        vec = torch.zeros(dim, dtype=torch.float32)
        if not text:
            return vec
        
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for i, tok in enumerate(tokens):
            h_int = int(hashlib.md5(f"{band_seed}:{tok}".encode("utf-8")).hexdigest(), 16)
            pos = h_int % dim
            sign = 1.0 if (h_int % 2 == 0) else -1.0
            vec[pos] += sign

            # Bigrams
            if i + 1 < len(tokens):
                bigram = f"{tok}_{tokens[i+1]}"
                h_bi = int(hashlib.md5(f"{band_seed}:{bigram}".encode("utf-8")).hexdigest(), 16)
                pos_bi = h_bi % dim
                sign_bi = 1.0 if (h_bi % 2 == 0) else -1.0
                vec[pos_bi] += sign_bi * 0.5

        norm = torch.norm(vec, p=2)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def extract_bands_from_file(self, file_node: FileNode) -> torch.Tensor:
        """
        Extracts 4 functional aspect bands from a file:
          0: Architecture (imports, class declarations, exports)
          1: Logic (functions, method signatures, call sequences)
          2: Invariants (asserts, exceptions, type hints)
          3: Performance/Configuration (literals, configs, loop counts)
        Returns tensor of shape [4, d_band].
        """
        lines = file_node.content.splitlines() if file_node.content else []
        
        arch_lines = []
        logic_lines = []
        inv_lines = []
        perf_lines = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue
            
            # Band 0: Architecture
            if (
                trimmed.startswith("import ")
                or trimmed.startswith("from ")
                or trimmed.startswith("class ")
                or trimmed.startswith("__all__")
            ):
                arch_lines.append(trimmed)
            # Band 2: Invariants / Testing
            elif (
                "assert " in trimmed
                or "raise " in trimmed
                or "Exception" in trimmed
                or "test_" in trimmed
            ):
                inv_lines.append(trimmed)
            # Band 3: Performance / Config
            elif (
                "batch" in trimmed
                or "dim" in trimmed
                or "size" in trimmed
                or "for " in trimmed
                or "while " in trimmed
                or "timeout" in trimmed
            ):
                perf_lines.append(trimmed)
            # Band 1: Logic
            else:
                logic_lines.append(trimmed)

        # Include symbols docstrings in architecture
        for sym in file_node.symbols.values():
            if sym.docstring:
                arch_lines.append(sym.docstring)
            logic_lines.append(f"{sym.name} {sym.kind}")

        b0 = self._hash_to_features(" ".join(arch_lines), band_seed=101, dim=self.d_band)
        b1 = self._hash_to_features(" ".join(logic_lines), band_seed=202, dim=self.d_band)
        b2 = self._hash_to_features(" ".join(inv_lines), band_seed=303, dim=self.d_band)
        b3 = self._hash_to_features(" ".join(perf_lines), band_seed=404, dim=self.d_band)

        return torch.stack([b0, b1, b2, b3], dim=0)

    def index_files(self, files: Dict[str, FileNode]) -> None:
        """Indexes an entire dictionary of files into the multi-aspect Z matrix."""
        self.entity_ids = sorted(files.keys())
        self.entity_index = {fid: i for i, fid in enumerate(self.entity_ids)}

        tensors = []
        for fid in self.entity_ids:
            tensors.append(self.extract_bands_from_file(files[fid]))

        if tensors:
            self.Z = torch.stack(tensors, dim=0).to(self.device)  # [N, 4, d_band]
        else:
            self.Z = torch.empty((0, 4, self.d_band), device=self.device)

    def update_file(self, file_node: FileNode) -> None:
        """Updates or appends a single file embedding in Z."""
        fid = file_node.file_path
        new_vec = self.extract_bands_from_file(file_node).unsqueeze(0).to(self.device)  # [1, 4, d_band]

        if fid in self.entity_index:
            idx = self.entity_index[fid]
            self.Z[idx] = new_vec[0]
        else:
            idx = len(self.entity_ids)
            self.entity_ids.append(fid)
            self.entity_index[fid] = idx
            if self.Z is None or self.Z.size(0) == 0:
                self.Z = new_vec
            else:
                self.Z = torch.cat([self.Z, new_vec], dim=0)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        band_weights: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Queries the multi-aspect index.
        band_weights: (w_arch, w_logic, w_inv, w_perf). Defaults to equal weighting.
        """
        if self.Z is None or len(self.entity_ids) == 0:
            return []

        if band_weights is None:
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=self.device)
        else:
            weights = torch.tensor(band_weights, device=self.device)
            weights = weights / (weights.sum() + 1e-8)

        # Extract query vector for each band
        q0 = self._hash_to_features(query_text, band_seed=101, dim=self.d_band)
        q1 = self._hash_to_features(query_text, band_seed=202, dim=self.d_band)
        q2 = self._hash_to_features(query_text, band_seed=303, dim=self.d_band)
        q3 = self._hash_to_features(query_text, band_seed=404, dim=self.d_band)
        q = torch.stack([q0, q1, q2, q3], dim=0).to(self.device)  # [4, d_band]

        # Compute cosine similarity per band: [N, 4]
        # self.Z is [N, 4, d_band], q is [4, d_band]
        # (Z * q).sum(dim=-1) -> [N, 4]
        band_sims = (self.Z * q.unsqueeze(0)).sum(dim=-1)  # [N, 4]
        
        # Weighted aggregate score
        scores = (band_sims * weights.unsqueeze(0)).sum(dim=-1)  # [N]

        k = min(top_k, len(self.entity_ids))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.entity_ids[idx], float(s)))
        return results

    def get_coupling(self, file_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Finds semantically coupled files (high multi-aspect similarity)."""
        if file_path not in self.entity_index or self.Z is None:
            return []
        
        idx = self.entity_index[file_path]
        target_vec = self.Z[idx]  # [4, d_band]

        # [N, 4]
        band_sims = (self.Z * target_vec.unsqueeze(0)).sum(dim=-1)
        scores = band_sims.mean(dim=-1)  # [N]

        k = min(top_k + 1, len(self.entity_ids))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for s, i in zip(top_scores.tolist(), top_indices.tolist()):
            fid = self.entity_ids[i]
            if fid != file_path:
                results.append((fid, float(s)))
            if len(results) >= top_k:
                break
        return results

    def memory_bytes(self) -> int:
        """Returns physical tensor memory bytes occupied by Z."""
        if self.Z is None:
            return 0
        return self.Z.element_size() * self.Z.nelement()
