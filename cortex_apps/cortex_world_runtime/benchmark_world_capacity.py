"""
Benchmark B: World Capacity & 3-Clock Real-Time Latency Benchmark.
==================================================================
Evaluates:
  - Entity scaling from N = 1,000 to N = 100,000 entities.
  - Memory footprint: bytes/entity, bytes/edge, total heap MB.
  - Clock 1 Frame Tick latency (Target: < 1.0 ms).
  - State lookup latency (Target: < 0.1 ms).
  - Graph BFS reachability latency (Target: < 2.0 ms).
  - Indexed semantic candidate search latency (Target: < 5.0 ms).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import FastWorldSubstrate


def evaluate_scale_point(n_entities: int, n_trials: int = 100) -> Dict[str, Any]:
    print(f"\nEvaluating Scale Point N = {n_entities:,} entities...")
    substrate = FastWorldSubstrate(num_clusters=32)
    t0 = time.perf_counter()
    substrate.populate_synthetic_world(num_entities=n_entities, edges_per_entity=4)
    init_time_s = time.perf_counter() - t0
    print(f"  Initialized in {init_time_s:.2f}s")

    # 1. Memory Footprint
    mem = substrate.memory_footprint_bytes()
    total_mb = mem["total_bytes"] / (1024 * 1024)
    bytes_per_entity = mem["total_bytes"] / n_entities
    print(f"  Memory Footprint: {total_mb:.2f} MB ({bytes_per_entity:.1f} bytes/entity)")

    # 2. Clock 1 Frame Tick Benchmark (Batch state delta ingestion)
    clock1_latencies: List[float] = []
    for i in range(n_trials):
        # 10 entity delta updates per frame
        deltas = [(f"ent_{((i * 10 + j) % n_entities):06d}", {"health": 90, "resource_units": j}) for j in range(10)]
        dur_ms = substrate.clock1_tick(deltas)
        clock1_latencies.append(dur_ms)

    # 3. State Lookup Benchmark (O(1))
    state_latencies: List[float] = []
    snapshot = substrate.current_snapshot()
    for i in range(200):
        eid = f"ent_{(i * 37 % n_entities):06d}"
        t_start = time.perf_counter()
        node = snapshot.get_entity(eid)
        val = node.state.get("health") if node else None
        state_latencies.append((time.perf_counter() - t_start) * 1000.0)

    # 4. Local Graph BFS Traversal Benchmark (max_depth=2, max_nodes=25)
    bfs_latencies: List[float] = []
    for i in range(n_trials):
        start_id = f"ent_{(i * 101 % n_entities):06d}"
        t_start = time.perf_counter()
        nbrs = snapshot.bfs(start_id, max_depth=2, max_nodes=25)
        bfs_latencies.append((time.perf_counter() - t_start) * 1000.0)

    # 5. Indexed Semantic Candidate Search Benchmark (Clock 2, top_k=5)
    sem_latencies: List[float] = []
    for i in range(n_trials):
        q_vec = torch.randn(64)
        q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=0)
        t_start = time.perf_counter()
        matches = snapshot.search_semantics_indexed(q_vec, top_k=5)
        sem_latencies.append((time.perf_counter() - t_start) * 1000.0)

    stats = {
        "n_entities": n_entities,
        "init_time_s": init_time_s,
        "total_mb": total_mb,
        "bytes_per_entity": bytes_per_entity,
        "clock1_p50": float(np.percentile(clock1_latencies, 50)),
        "clock1_p95": float(np.percentile(clock1_latencies, 95)),
        "clock1_p99": float(np.percentile(clock1_latencies, 99)),
        "state_lookup_p99_us": float(np.percentile(state_latencies, 99) * 1000.0),
        "bfs_p50": float(np.percentile(bfs_latencies, 50)),
        "bfs_p95": float(np.percentile(bfs_latencies, 95)),
        "semantic_p50": float(np.percentile(sem_latencies, 50)),
        "semantic_p95": float(np.percentile(sem_latencies, 95)),
    }

    print(f"  Clock 1 Frame Tick:  p50={stats['clock1_p50']:.3f} ms, p95={stats['clock1_p95']:.3f} ms, p99={stats['clock1_p99']:.3f} ms")
    print(f"  State Lookup (S):    p99={stats['state_lookup_p99_us']:.2f} us")
    print(f"  Local BFS (G):       p50={stats['bfs_p50']:.3f} ms, p95={stats['bfs_p95']:.3f} ms")
    print(f"  Indexed Sem (Z):     p50={stats['semantic_p50']:.3f} ms, p95={stats['semantic_p95']:.3f} ms")
    return stats


def run_benchmark_b():
    print("=" * 80)
    print("BENCHMARK B: WORLD CAPACITY & 3-CLOCK REAL-TIME LATENCY BENCHMARK")
    print("Sweeping N = 1,000 to N = 100,000 entities")
    print("=" * 80)

    scale_points = [1000, 10000, 100000]
    all_results = []

    for n in scale_points:
        res = evaluate_scale_point(n, n_trials=100)
        all_results.append(res)

    print("\n" + "=" * 80)
    print("SUMMARY OF REAL-TIME CAPACITY & LATENCY SCALING")
    print("=" * 80)
    print(f"{'Entities (N)':<14} {'Heap (MB)':<12} {'Bytes/Ent':<12} {'Clock 1 p95':<14} {'BFS p95':<12} {'Sem p95':<12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['n_entities']:<14,d} {r['total_mb']:>9.2f} MB {r['bytes_per_entity']:>10.1f} B  {r['clock1_p95']:>10.3f} ms {r['bfs_p95']:>9.3f} ms {r['semantic_p95']:>9.3f} ms")

    # Evaluate Hard Kill Criteria
    n100k = all_results[-1]
    clock1_pass = n100k["clock1_p95"] < 1.0
    clock2_pass = n100k["semantic_p95"] < 5.0

    print("\n" + "=" * 80)
    print("GAME-RUNTIME HARD KILL CRITERIA EVALUATION (N = 100,000):")
    print(f"  1. Clock 1 Delta Ingestion (< 1.0 ms):  {n100k['clock1_p95']:.3f} ms -> [{'PASS' if clock1_pass else 'FAIL'}]")
    print(f"  2. Clock 2 Semantic Lookup (< 5.0 ms):  {n100k['semantic_p95']:.3f} ms -> [{'PASS' if clock2_pass else 'FAIL'}]")
    print(f"  3. Local BFS Graph Query (< 2.0 ms):   {n100k['bfs_p95']:.3f} ms -> [{'PASS' if n100k['bfs_p95'] < 2.0 else 'FAIL'}]")
    print(f"  4. Memory at 100k entities:            {n100k['total_mb']:.2f} MB (~{n100k['bytes_per_entity']:.1f} bytes/entity)")
    print("=" * 80)

    # Save artifact
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_b_world_capacity_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    run_benchmark_b()
