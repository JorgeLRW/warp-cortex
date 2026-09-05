"""
Memory & Duplicated Byte Profiler.
===================================
Rigorous memory profiling using Python's tracemalloc and deep object inspection
to measure real heap allocations, tensor memory, and duplicate bytes across
data structures (aspect vectors, graphs, and status tables).
"""

from __future__ import annotations

import sys
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import torch


@dataclass
class MemoryProfileResult:
    arch_name: str
    heap_allocated_mb: float
    heap_peak_mb: float
    total_tensor_bytes: int
    unique_tensor_bytes: int
    duplicate_tensor_bytes: int
    duplicate_tensor_ratio: float
    graph_edge_copies: int
    status_entry_copies: int


def inspect_tensor_duplication(obj: Any) -> Tuple[int, int, int]:
    """
    Traverses an architecture object to find all torch.Tensor instances.
    Identifies total bytes, unique bytes (by tensor content hash / data ptr),
    and duplicate bytes.
    """
    seen_ids: Set[int] = set()
    all_tensors: List[torch.Tensor] = []

    def _walk(o: Any, depth: int = 0):
        if depth > 10:
            return
        if id(o) in seen_ids:
            return
        seen_ids.add(id(o))

        if isinstance(o, torch.Tensor):
            all_tensors.append(o)
            return

        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for item in o:
                _walk(item, depth + 1)
        elif hasattr(o, "__dict__"):
            for k, v in vars(o).items():
                if not k.startswith("__"):
                    _walk(v, depth + 1)

    _walk(obj)

    total_bytes = sum(t.numel() * t.element_size() for t in all_tensors)

    # Group by tensor data pointer (shared storage) and content hash
    unique_data_ptrs: Set[int] = set()
    unique_content_hashes: Set[Tuple[int, Tuple[int, ...], float]] = set()
    unique_bytes = 0

    for t in all_tensors:
        ptr = t.data_ptr()
        if ptr not in unique_data_ptrs:
            unique_data_ptrs.add(ptr)
            # Content signature: dtype, shape, sum of elements
            sig = (int(t.dtype.is_floating_point), tuple(t.shape), float(t.sum().item()) if t.numel() > 0 else 0.0)
            if sig not in unique_content_hashes:
                unique_content_hashes.add(sig)
                unique_bytes += t.numel() * t.element_size()

    duplicate_bytes = max(0, total_bytes - unique_bytes)
    return total_bytes, unique_bytes, duplicate_bytes


def profile_architecture_memory(
    arch_name: str,
    arch: Any,
) -> MemoryProfileResult:
    """Profiles heap and duplicate state for an architecture instance."""
    # Tracemalloc snapshot
    curr, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
    heap_mb = curr / (1024 * 1024)
    peak_mb = peak / (1024 * 1024)

    total_t_bytes, uniq_t_bytes, dup_t_bytes = inspect_tensor_duplication(arch)
    dup_ratio = (dup_t_bytes / total_t_bytes) if total_t_bytes > 0 else 0.0

    # Count graph edge copies
    graph_edge_copies = 0
    if hasattr(arch, "forward_adj"):
        graph_edge_copies += sum(len(v) for v in getattr(arch, "forward_adj").values())
    if hasattr(arch, "frontier_forward_adj"):
        graph_edge_copies += sum(len(v) for v in getattr(arch, "frontier_forward_adj").values())

    # Count status table copies
    status_copies = 0
    if hasattr(arch, "state"):
        status_copies += len(getattr(arch, "state"))
    if hasattr(arch, "state_table"):
        status_copies += len(getattr(arch, "state_table"))
    if hasattr(arch, "state_cache"):
        status_copies += len(getattr(arch, "state_cache"))

    return MemoryProfileResult(
        arch_name=arch_name,
        heap_allocated_mb=heap_mb,
        heap_peak_mb=peak_mb,
        total_tensor_bytes=total_t_bytes,
        unique_tensor_bytes=uniq_t_bytes,
        duplicate_tensor_bytes=dup_t_bytes,
        duplicate_tensor_ratio=dup_ratio,
        graph_edge_copies=graph_edge_copies,
        status_entry_copies=status_copies,
    )
