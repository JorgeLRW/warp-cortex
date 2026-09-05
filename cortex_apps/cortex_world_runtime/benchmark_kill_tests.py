"""
Benchmark Kill Tests: The Four Decisive Systems Kill Tests.
============================================================
1. Kill-Test 1: Indexed Semantic Recall@k vs. Candidate Budget (Ground truth: exhaustive N=100k).
2. Kill-Test 2: AI-Tick Mixed Workload Saturation (20 Hz / 50 ms budget across A=32..2048).
3. Kill-Test 3: Batched Clock 1 Frame Ingestion (E=1..10,000 deltas vs 1.0 ms budget).
4. Kill-Test 4: Sharded Regional Writes vs. Global Monolithic Versioning (Conflict elimination).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import FastWorldSubstrate
from cortex_apps.cortex_world_runtime.sharded_world_substrate import ShardedWorldSubstrate
from cortex_apps.cortex_world_runtime.skill_registry import (
    SkillDefinition,
    SkillInvocationEvent,
    SkillRegistry,
    SkillSelectionMode,
    SkillSelector,
)


def run_kill_test_1_semantic_recall(n_entities: int = 100000, n_queries: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"KILL-TEST 1: INDEXED SEMANTIC RECALL@K VS EXHAUSTIVE SCAN (N = {n_entities:,})")
    print("=" * 80)

    substrate = FastWorldSubstrate(num_clusters=32)
    print("Populating world...")
    substrate.populate_synthetic_world(num_entities=n_entities, edges_per_entity=4)
    snapshot = substrate.current_snapshot()

    # Pre-stack all embeddings for fast exhaustive baseline
    print("Stacking full embedding matrix for ground truth evaluation...")
    all_eids = list(snapshot.entities.keys())
    all_embeddings = torch.stack([snapshot.entities[eid].aspect_vector for eid in all_eids])

    candidate_budgets = [32, 64, 128, 256, 512]
    budget_results = {}

    # Sample test queries
    queries = []
    for _ in range(n_queries):
        q = torch.randn(64)
        q = torch.nn.functional.normalize(q, p=2, dim=0)
        queries.append(q)

    # Compute Ground Truth Exhaustive Top-10 for each query
    print("Computing exhaustive ground truth...")
    ground_truths: List[List[str]] = []
    exhaustive_latencies: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        sims = torch.matmul(all_embeddings, q)
        top10_vals, top10_idx = torch.topk(sims, k=10)
        exhaustive_latencies.append((time.perf_counter() - t0) * 1000.0)
        gt_ids = [all_eids[idx.item()] for idx in top10_idx]
        ground_truths.append(gt_ids)

    exhaustive_p50 = float(np.percentile(exhaustive_latencies, 50))
    exhaustive_p95 = float(np.percentile(exhaustive_latencies, 95))
    print(f"Exhaustive Baseline: p50={exhaustive_p50:.2f} ms, p95={exhaustive_p95:.2f} ms")

    for B in candidate_budgets:
        r1_list, r5_list, r10_list = [], [], []
        latencies: List[float] = []

        for q, gt in zip(queries, ground_truths):
            t0 = time.perf_counter()
            retrieved = snapshot.search_semantics_indexed(q, top_k=10, candidate_budget=B)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            ret_ids = [eid for eid, _ in retrieved]
            # Recall@1: is gt[0] in ret[:1]?
            r1 = 1.0 if (len(ret_ids) > 0 and ret_ids[0] == gt[0]) else 0.0
            # Recall@5: fraction of gt[:5] present in ret[:5]
            r5 = len(set(ret_ids[:5]).intersection(set(gt[:5]))) / 5.0
            # Recall@10: fraction of gt[:10] present in ret[:10]
            r10 = len(set(ret_ids[:10]).intersection(set(gt[:10]))) / 10.0

            r1_list.append(r1)
            r5_list.append(r5)
            r10_list.append(r10)

        budget_results[B] = {
            "candidate_budget": B,
            "recall_at_1": float(np.mean(r1_list) * 100.0),
            "recall_at_5": float(np.mean(r5_list) * 100.0),
            "recall_at_10": float(np.mean(r10_list) * 100.0),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
        }
        b_res = budget_results[B]
        print(f"  Budget B={B:>3d} | R@1={b_res['recall_at_1']:>5.1f}% | R@5={b_res['recall_at_5']:>5.1f}% | R@10={b_res['recall_at_10']:>5.1f}% | p50={b_res['latency_p50']:>5.3f}ms | p95={b_res['latency_p95']:>5.3f}ms")

    return {
        "exhaustive_p50": exhaustive_p50,
        "exhaustive_p95": exhaustive_p95,
        "budgets": budget_results,
    }


def run_kill_test_2_ai_tick_saturation(n_entities: int = 10000) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("KILL-TEST 2: AI-TICK WORKLOAD CAPACITY (20 Hz / 50 ms Deadline)")
    print("Workload Mix: 60% S (lookup), 20% G (BFS), 15% Z (semantic), 5% H/K (skill)")
    print("=" * 80)

    substrate = FastWorldSubstrate(num_clusters=16)
    substrate.populate_synthetic_world(num_entities=n_entities, edges_per_entity=4)
    snapshot = substrate.current_snapshot()

    registry = SkillRegistry()
    registry.register(SkillDefinition(
        skill_id="patrol_route", version="v1", name="Patrol Route",
        description="Inspect nearby sector nodes", aspect_tags=["PATROL", "SECTOR"],
    ))
    selector = SkillSelector(registry, mode=SkillSelectionMode.SHARED_CORTEX_LEDGER)

    agent_scales = [32, 128, 512, 1024, 2048]
    tick_results = {}

    def simulate_agent_tick_request(agent_idx: int) -> str:
        # Roll workload mix
        r = (agent_idx * 17) % 100
        target_eid = f"ent_{(agent_idx * 23 % n_entities):06d}"

        if r < 60:
            # 60% State Lookup (S)
            node = snapshot.get_entity(target_eid)
            _ = node.state.get("health") if node else None
            return "state"
        elif r < 80:
            # 20% Graph Neighborhood (G)
            _ = snapshot.bfs(target_eid, max_depth=2, max_nodes=15)
            return "graph"
        elif r < 95:
            # 15% Indexed Semantic Search (Z)
            q = torch.randn(64)
            q = torch.nn.functional.normalize(q, p=2, dim=0)
            _ = snapshot.search_semantics_indexed(q, top_k=5, candidate_budget=64)
            return "semantic"
        else:
            # 5% Skill / History Query (H/K)
            _ = selector.select_skill("inspect patrol route", snapshot, agent_id=f"a_{agent_idx}", top_k=1)
            return "skill"

    for A in agent_scales:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, A)) as executor:
            futures = [executor.submit(simulate_agent_tick_request, i) for i in range(A)]
            concurrent.futures.wait(futures)
        tick_duration_ms = (time.perf_counter() - t0) * 1000.0
        service_qps = A / max(0.001, (tick_duration_ms / 1000.0))
        fits_50ms = tick_duration_ms <= 50.0

        tick_results[A] = {
            "agent_count": A,
            "tick_duration_ms": tick_duration_ms,
            "service_qps": service_qps,
            "fits_in_50ms_tick": fits_50ms,
        }
        print(f"  Agents A={A:>4d} | Tick Time: {tick_duration_ms:>6.2f} ms | QPS: {service_qps:>8,.0f} | Deadline (<50ms): [{'PASS' if fits_50ms else 'FAIL'}]")

    return tick_results


def run_kill_test_3_batched_frame_ingestion(n_entities: int = 100000, n_trials: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"KILL-TEST 3: BATCHED CLOCK-1 FRAME INGESTION CAPACITY (N = {n_entities:,})")
    print("Target: Find E* = max E such that p95(T_ingest(E)) < 1.0 ms")
    print("=" * 80)

    substrate = FastWorldSubstrate(num_clusters=32)
    substrate.populate_synthetic_world(num_entities=n_entities, edges_per_entity=4)

    batch_sizes = [1, 10, 100, 500, 1000, 2500, 5000, 10000]
    results = {}

    for E in batch_sizes:
        durations: List[float] = []
        for trial in range(n_trials):
            deltas = [
                (f"ent_{((trial * E + j) % n_entities):06d}", {"health": 95, "resource_units": j % 20})
                for j in range(E)
            ]
            dur_ms = substrate.clock1_tick(deltas)
            durations.append(dur_ms)

        p50 = float(np.percentile(durations, 50))
        p95 = float(np.percentile(durations, 95))
        p99 = float(np.percentile(durations, 99))
        throughput_events_sec = (E / (p50 / 1000.0)) if p50 > 0 else 0
        fits_1ms = p95 < 1.0

        results[E] = {
            "events_per_frame": E,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "throughput_events_sec": throughput_events_sec,
            "fits_in_1ms_budget": fits_1ms,
        }
        print(f"  E = {E:>5,d} deltas/frame | p50={p50:>6.3f} ms | p95={p95:>6.3f} ms | p99={p99:>6.3f} ms | Ingest Rate={throughput_events_sec:>11,.0f} ev/s | [{'PASS' if fits_1ms else 'FAIL'}]")

    e_star = max([E for E, r in results.items() if r["fits_in_1ms_budget"]], default=0)
    print(f"\nE* Capacity: {e_star:,} state deltas can be ingested within the 1.0 ms frame budget at p95!")
    return {"results": results, "e_star": e_star}


def run_kill_test_4_sharded_writes(n_entities: int = 10000, n_commits: int = 500) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("KILL-TEST 4: SHARDED REGIONAL WRITES VS MONOLITHIC GLOBAL VERSIONING")
    print(f"Testing A = 128 and A = 512 concurrent writers across R = 1, 16, 64 regions")
    print("=" * 80)

    comparison_results = {}

    for num_regions in [1, 16, 64]:
        sharded_sub = ShardedWorldSubstrate(num_regions=num_regions, num_clusters=16)
        sharded_sub.populate_synthetic_world(num_entities=n_entities, edges_per_entity=4)

        for A in [128, 512]:
            snapshot = sharded_sub.current_snapshot()
            # Distribute agents across regions
            agent_regions = [i % num_regions for i in range(A)]
            agent_expected_versions = [snapshot.get_region_version(agent_regions[i]) for i in range(A)]

            commits_succeeded = 0
            conflicts = 0
            t0 = time.perf_counter()

            # Execute serialized transaction attempts
            for c_idx in range(n_commits):
                agent_idx = c_idx % A
                r_id = agent_regions[agent_idx]
                exp_v = agent_expected_versions[agent_idx]
                target_eid = f"ent_{(c_idx * 13 % n_entities):06d}"

                ok, new_v, msg = sharded_sub.commit_intent_regional(
                    agent_id=f"agent_{agent_idx}",
                    region_id=r_id,
                    expected_region_version=exp_v,
                    intent_deltas=[(target_eid, {"resource_units": c_idx % 10})],
                )
                if ok:
                    commits_succeeded += 1
                    agent_expected_versions[agent_idx] = new_v
                else:
                    conflicts += 1
                    # Update to current observed version to simulate standard retry backoff
                    agent_expected_versions[agent_idx] = new_v

            dur_s = time.perf_counter() - t0
            conflict_rate_pct = (conflicts / n_commits) * 100.0
            throughput = commits_succeeded / max(0.001, dur_s)

            key = f"R={num_regions}_A={A}"
            comparison_results[key] = {
                "regions": num_regions,
                "agents": A,
                "commits_attempted": n_commits,
                "commits_succeeded": commits_succeeded,
                "conflicts": conflicts,
                "conflict_rate_pct": conflict_rate_pct,
                "commit_throughput_sec": throughput,
            }
            print(f"  Config R={num_regions:>2d} regions | A={A:>3d} agents | Success: {commits_succeeded:>3d}/{n_commits} | Conflicts: {conflicts:>3d} ({conflict_rate_pct:>5.1f}%) | Commit QPS: {throughput:>8,.0f}")

    return comparison_results


def run_all_kill_tests():
    out_dir = os.path.dirname(__file__)

    # 1. Semantic Recall
    res1 = run_kill_test_1_semantic_recall(n_entities=100000, n_queries=50)

    # 2. AI-Tick Saturation
    res2 = run_kill_test_2_ai_tick_saturation(n_entities=10000)

    # 3. Batched Frame Ingestion
    res3 = run_kill_test_3_batched_frame_ingestion(n_entities=100000, n_trials=50)

    # 4. Sharded Writes
    res4 = run_kill_test_4_sharded_writes(n_entities=10000, n_commits=500)

    all_kill_results = {
        "kill_test_1_semantic_recall": res1,
        "kill_test_2_ai_tick_saturation": res2,
        "kill_test_3_batched_frame_ingestion": res3,
        "kill_test_4_sharded_writes": res4,
    }

    out_file = os.path.join(out_dir, "benchmark_kill_tests_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_kill_results, f, indent=2)
    print(f"\nSaved all Kill-Test results to {out_file}")


if __name__ == "__main__":
    run_all_kill_tests()
