"""
Entity Scale Curve Benchmark: Memory, Latency & Representation Duplication.
===========================================================================
Scales enterprise research worlds from N = 100 to N = 5,000 entities/documents.
Measures:
  - Exact Tracemalloc Heap Usage (MB)
  - Duplicate Tensor Memory (MB)
  - Ingestion Latency (ms/event)
  - Context Selection Latency (ms/query)
  - Explain-Risk Query Latency (ms/query)

Compares:
  - Unified Context Substrate (Single persistent representation)
  - Representation-Matched Modular Monolith (Disjoint module materializations)
"""

from __future__ import annotations

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from cortex_apps.multi_service_substrate.memory_profiler import profile_architecture_memory
from cortex_apps.multi_service_substrate.representation_matched_monolith import RepresentationMatchedMonolith
from cortex_apps.multi_service_substrate.scale_catalog import build_scalable_research_world
from cortex_apps.multi_service_substrate.service7_explain_risk import (
    explain_risk_representation_matched_monolith,
    explain_risk_unified,
)
from cortex_apps.multi_service_substrate.substrate_api import TelemetryEvent
from cortex_apps.multi_service_substrate.unified_substrate import UnifiedContextSubstrate


@dataclass
class ScaleMetricRow:
    n_entities: int
    architecture: str
    heap_mb: float
    duplicate_mb: float
    duplicate_pct: float
    ingest_latency_ms: float
    context_latency_ms: float
    explain_latency_ms: float


def run_scale_curve_benchmark(
    scales: List[int] = [100, 500, 1000, 2500, 5000],
    seed: int = 42,
) -> List[ScaleMetricRow]:
    """Runs scaling benchmark across N entities."""
    rows: List[ScaleMetricRow] = []

    print(f"Executing Scale Curve Benchmark across N = {scales} entities...")

    for n in scales:
        print(f"\n--- Generating Enterprise Catalog with N={n} entities ---")
        catalog = build_scalable_research_world(n_entities=n, seed=seed)

        # 1. Evaluate Representation-Matched Monolith
        tracemalloc.start()
        t_init = time.perf_counter()
        mono = RepresentationMatchedMonolith(catalog)
        mem_mono = profile_architecture_memory("Representation-Matched Monolith", mono)

        # Ingest 10 events
        ingest_times_mono = []
        for i in range(10):
            ev = TelemetryEvent(f"ev_{i}", 100.0 + i, "SENSOR_SHOCK" if i % 3 == 0 else "HEARTBEAT", "inst_quadrupole_ms", "shock")
            t0 = time.perf_counter()
            mono.ingest(ev)
            ingest_times_mono.append((time.perf_counter() - t0) * 1000.0)

        # Context query
        ctx_times_mono = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = mono.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)
            ctx_times_mono.append((time.perf_counter() - t0) * 1000.0)

        # Explain-risk query
        exp_times_mono = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = explain_risk_representation_matched_monolith(mono, "ds_proteomics_spectra")
            exp_times_mono.append((time.perf_counter() - t0) * 1000.0)

        tracemalloc.stop()

        rows.append(ScaleMetricRow(
            n_entities=n,
            architecture="Representation-Matched Monolith",
            heap_mb=mem_mono.heap_allocated_mb,
            duplicate_mb=mem_mono.duplicate_tensor_bytes / (1024 * 1024),
            duplicate_pct=mem_mono.duplicate_tensor_ratio * 100.0,
            ingest_latency_ms=float(np.mean(ingest_times_mono)),
            context_latency_ms=float(np.mean(ctx_times_mono)),
            explain_latency_ms=float(np.mean(exp_times_mono)),
        ))

        # 2. Evaluate Unified Context Substrate
        tracemalloc.start()
        sub = UnifiedContextSubstrate(catalog)
        mem_sub = profile_architecture_memory("Unified Context Substrate", sub)

        ingest_times_sub = []
        for i in range(10):
            ev = TelemetryEvent(f"ev_{i}", 100.0 + i, "SENSOR_SHOCK" if i % 3 == 0 else "HEARTBEAT", "inst_quadrupole_ms", "shock")
            t0 = time.perf_counter()
            sub.ingest(ev)
            ingest_times_sub.append((time.perf_counter() - t0) * 1000.0)

        ctx_times_sub = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sub.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)
            ctx_times_sub.append((time.perf_counter() - t0) * 1000.0)

        exp_times_sub = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = explain_risk_unified(sub, "ds_proteomics_spectra")
            exp_times_sub.append((time.perf_counter() - t0) * 1000.0)

        tracemalloc.stop()

        rows.append(ScaleMetricRow(
            n_entities=n,
            architecture="Unified Context Substrate",
            heap_mb=mem_sub.heap_allocated_mb,
            duplicate_mb=mem_sub.duplicate_tensor_bytes / (1024 * 1024),
            duplicate_pct=mem_sub.duplicate_tensor_ratio * 100.0,
            ingest_latency_ms=float(np.mean(ingest_times_sub)),
            context_latency_ms=float(np.mean(ctx_times_sub)),
            explain_latency_ms=float(np.mean(exp_times_sub)),
        ))

    return rows


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = run_scale_curve_benchmark(scales=[100, 500, 1000, 2000], seed=42)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 145)
    print("ENTITY SCALE CURVE BENCHMARK (N = 100 .. 2,000 ENTITIES)")
    print("Evaluating Heap Footprint, Duplicate Tensor Bytes, Ingestion & Query Latencies as Entities Scale")
    print("=" * 145)
    print(f"{'Entities (N)':<14} | {'Architecture':<34} | {'Heap (MB)':<12} | {'Dup Tensors':<16} | {'Ingest Lat':<14} | {'Context Lat':<14} | {'Explain Lat':<14}")
    print("-" * 145)

    for r in res:
        print(f"{r.n_entities:<14} | {r.architecture:<34} | {r.heap_mb:8.2f} MB  | {r.duplicate_mb:6.2f} MB ({r.duplicate_pct:4.1f}%) | {r.ingest_latency_ms:6.2f} ms     | {r.context_latency_ms:6.2f} ms     | {r.explain_latency_ms:6.2f} ms")

    print("=" * 145)
    print(f"Benchmark completed in {elapsed:.2f} seconds.")
