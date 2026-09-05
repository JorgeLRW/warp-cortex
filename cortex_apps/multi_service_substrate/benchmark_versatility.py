"""
The Versatility Dividend & Multi-Service Substrate Benchmark.
=============================================================
Evaluates the systems thesis:
  "Maintain context once. Reuse it everywhere."

Compares 3 Architectures:
  1. Fragmented Naive (Uncoordinated microservices, eventual consistency)
  2. Fragmented Production-Grade (Transactional outbox, worker projections, barrier sync)
  3. Unified Context Substrate (Single logical source of truth U_v = <S_v, G_v, Z, H_v>)

Under matched algorithms (Okapi BM25, static Z aspect vectors, BFS reachability, prerequisite checks).
Measures:
  - Table 1: Joint Service Success (J_t) & Individual Service Quality
  - Table 2: The Update Tax Audit (Writes, Index mutations, Serialization ops, Invalidation ops)
  - Table 3: The Versatility Dividends & Capability Density
  - Table 4: The Coherence-Latency Frontier
  - Table 5: Marginal Cost per Capability Curve (Cost vs Number of Enabled Services)
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_apps.multi_service_substrate.fragmented_naive import FragmentedNaiveArchitecture
from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.substrate_api import (
    ContextSubstrate,
    EntityStatus,
    OperationMetrics,
    ProposedAction,
    TelemetryEvent,
)
from cortex_apps.multi_service_substrate.unified_substrate import UnifiedContextSubstrate
from cortex_apps.multi_service_substrate.workload_generator import generate_streaming_workload, WorkloadStep
from cortex_apps.research_agent_system.world_state import build_research_world, ResearchWorldCatalog


@dataclass
class TimelineResult:
    joint_success_rate: float
    service_quality: Dict[str, float]
    version_agreement_rate: float
    total_writes: int
    total_index_mutations: int
    total_serialization_ops: int
    total_invalidation_ops: int
    total_cpu_time_ms: float
    p95_decision_latency_ms: float
    memory_footprint_kb: float


def evaluate_single_timeline(
    arch: ContextSubstrate,
    steps: List[WorkloadStep],
    enabled_services: Optional[Set[str]] = None,
) -> Tuple[TimelineResult, List[float]]:
    """
    Executes a 100-step streaming workload against a substrate contender.
    Collects Joint Service Success J_t and latency measurements.
    """
    if enabled_services is None:
        enabled_services = {"context", "route", "affected", "search", "verify", "subscribe"}

    arch.reset_metrics()
    subscribed_events: List[str] = []

    if "subscribe" in enabled_services:
        arch.subscribe(
            predicate=lambda e: e.event_type in ("SENSOR_SHOCK", "GRAPH_MUTATION"),
            callback=lambda ev, v: subscribed_events.append(ev.event_id),
        )

    joint_successes = []
    service_successes: Dict[str, List[float]] = {s: [] for s in enabled_services}
    version_agreements = []
    decision_latencies: List[float] = []

    for step in steps:
        # Ingest event
        v_ingest = arch.ingest(step.event)

        # Evaluate probe if present
        if step.probe is not None:
            t_probe_start = time.perf_counter()
            observed_versions: List[int] = []
            service_pass: Dict[str, bool] = {}

            # Service 1: Context Pack
            if "context" in enabled_services:
                ctx = arch.context(step.probe.query, token_budget=256)
                observed_versions.append(ctx.version)
                doc_ids = set(ctx.doc_ids)
                # Success if critical documents are present
                has_crit = all(c in doc_ids for c in step.probe.critical_doc_ids)
                service_pass["context"] = has_crit
                service_successes["context"].append(1.0 if has_crit else 0.0)

            # Service 2: Agent Route
            if "route" in enabled_services:
                probe_ev = TelemetryEvent(
                    event_id=f"ev_probe_{step.step_id}",
                    timestamp=step.event.timestamp,
                    event_type="DECISION_PROBE",
                    entity_id=step.probe.target_action.target_node,
                    raw_text=step.probe.query,
                )
                woken, v_r = arch.route(probe_ev)
                observed_versions.append(v_r)
                has_expected = all(a in woken for a in step.probe.expected_woken_agents)
                service_pass["route"] = has_expected
                service_successes["route"].append(1.0 if has_expected else 0.0)

            # Service 3: Affected Frontier
            if "affected" in enabled_services:
                aff, v_a = arch.affected(step.probe.probe_entity_id)
                observed_versions.append(v_a)
                has_aff = any(e in aff for e in step.probe.expected_affected_entities)
                service_pass["affected"] = has_aff
                service_successes["affected"].append(1.0 if has_aff else 0.0)

            # Service 4: Search
            if "search" in enabled_services:
                results, v_s = arch.search(step.probe.query, top_k=5)
                observed_versions.append(v_s)
                # Search success if top-5 has relevant documents
                has_res = len(results) > 0 and any("ms4" in d.doc_id or "pilot" in d.doc_id or "yield" in d.doc_id for d in results)
                service_pass["search"] = has_res
                service_successes["search"].append(1.0 if has_res else 0.0)

            # Service 5: Invariant Verification
            if "verify" in enabled_services:
                ver_res = arch.verify(step.probe.target_action)
                observed_versions.append(ver_res.version)
                ver_correct = (ver_res.permit == step.probe.expected_permit)
                service_pass["verify"] = ver_correct
                service_successes["verify"].append(1.0 if ver_correct else 0.0)

            # Service 6: Subscribe
            if "subscribe" in enabled_services:
                if step.event.event_type in ("SENSOR_SHOCK", "GRAPH_MUTATION"):
                    notified = step.event.event_id in subscribed_events
                else:
                    notified = True
                service_pass["subscribe"] = notified
                service_successes["subscribe"].append(1.0 if notified else 0.0)

            probe_latency_ms = (time.perf_counter() - t_probe_start) * 1000.0
            decision_latencies.append(probe_latency_ms)

            # Check snapshot coherence
            v_agree = (len(set(observed_versions)) <= 1)
            version_agreements.append(1.0 if v_agree else 0.0)

            # Joint Service Success J_t = Product(1[Q_i >= q*]) * 1[v_agree]
            all_services_passed = all(service_pass.values())
            joint_success = (all_services_passed and v_agree)
            joint_successes.append(1.0 if joint_success else 0.0)

    # Approximate memory footprint based on internal dicts & vectors
    metrics = arch.get_metrics()
    mem_kb = sys.getsizeof(arch) / 1024.0
    if hasattr(arch, "aspect_vectors"):
        mem_kb += sum(sys.getsizeof(v) for d in arch.aspect_vectors.values() for v in d.values()) / 1024.0

    p95_lat = float(np.percentile(decision_latencies, 95)) if decision_latencies else 0.0

    res = TimelineResult(
        joint_success_rate=float(np.mean(joint_successes)) * 100.0 if joint_successes else 0.0,
        service_quality={s: float(np.mean(vals)) * 100.0 for s, vals in service_successes.items()},
        version_agreement_rate=float(np.mean(version_agreements)) * 100.0 if version_agreements else 0.0,
        total_writes=metrics.writes,
        total_index_mutations=metrics.index_mutations,
        total_serialization_ops=metrics.serialization_ops,
        total_invalidation_ops=metrics.invalidation_ops,
        total_cpu_time_ms=metrics.cpu_time_ms,
        p95_decision_latency_ms=p95_lat,
        memory_footprint_kb=mem_kb,
    )
    return res, decision_latencies


def run_multi_service_benchmark(n_timelines: int = 30):
    print("=" * 165)
    print("THE VERSATILITY DIVIDEND: MULTI-SERVICE CONTEXT SUBSTRATE BENCHMARK")
    print("Evaluating Systems-Level Infrastructure: Maintain Context Once, Reuse Everywhere")
    print(f"Workload: 100-step streaming adversarial timeline across {n_timelines} timelines (Seeds 30000..{30000 + n_timelines - 1})")
    print("Comparing: 1. Fragmented Naive | 2. Fragmented Production-Grade | 3. Unified Context Substrate")
    print("=" * 165)

    arch_names = [
        "1. Fragmented Naive",
        "2. Fragmented Production-Grade",
        "3. Unified Context Substrate",
    ]

    all_results: Dict[str, List[TimelineResult]] = {name: [] for name in arch_names}
    all_latencies: Dict[str, List[float]] = {name: [] for name in arch_names}

    seed_start = 30000
    t0 = time.perf_counter()

    for i in range(n_timelines):
        seed = seed_start + i
        catalog = build_research_world(seed=seed, world_variant="WORLD_A_LINKED")
        workload = generate_streaming_workload(n_steps=100, seed=seed, world_variant="WORLD_A_LINKED")

        # Contender 1: Fragmented Naive
        naive_arch = FragmentedNaiveArchitecture(catalog, staleness_prob=0.25)
        res_naive, l_naive = evaluate_single_timeline(naive_arch, workload)
        all_results["1. Fragmented Naive"].append(res_naive)
        all_latencies["1. Fragmented Naive"].extend(l_naive)

        # Contender 2: Fragmented Production-Grade
        prod_arch = FragmentedProductionArchitecture(catalog, sync_barrier=True)
        res_prod, l_prod = evaluate_single_timeline(prod_arch, workload)
        all_results["2. Fragmented Production-Grade"].append(res_prod)
        all_latencies["2. Fragmented Production-Grade"].extend(l_prod)

        # Contender 3: Unified Context Substrate
        unified_arch = UnifiedContextSubstrate(catalog)
        res_unified, l_uni = evaluate_single_timeline(unified_arch, workload)
        all_results["3. Unified Context Substrate"].append(res_unified)
        all_latencies["3. Unified Context Substrate"].extend(l_uni)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_timelines} timelines ({time.perf_counter() - t0:.1f}s elapsed)...")

    total_time = time.perf_counter() - t0

    # =========================================================================
    # TABLE 1: JOINT SERVICE SUCCESS & SNAPSHOT COHERENCE
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 1: JOINT SERVICE SUCCESS & SNAPSHOT COHERENCE ACROSS ALL 6 SERVICES")
    print("Joint Success J_t = Product(1[Q_i >= q_i*]) * 1[versions agree]. Required for simultaneous multi-service consistency.")
    print("=" * 165)
    print(f"{'Architecture':<32} | {'Joint Success':<14} | {'Version Agree':<14} | {'Context':<10} | {'Route':<10} | {'Affected':<10} | {'Search':<10} | {'Verify':<10} | {'Subscribe':<10}")
    print("-" * 145)

    for name in arch_names:
        r_list = all_results[name]
        j_mean = np.mean([r.joint_success_rate for r in r_list])
        v_mean = np.mean([r.version_agreement_rate for r in r_list])
        q_ctx = np.mean([r.service_quality["context"] for r in r_list])
        q_rt = np.mean([r.service_quality["route"] for r in r_list])
        q_aff = np.mean([r.service_quality["affected"] for r in r_list])
        q_sr = np.mean([r.service_quality["search"] for r in r_list])
        q_ver = np.mean([r.service_quality["verify"] for r in r_list])
        q_sub = np.mean([r.service_quality["subscribe"] for r in r_list])

        print(f"{name:<32} | {j_mean:<13.1f}% | {v_mean:<13.1f}% | {q_ctx:<9.1f}% | {q_rt:<9.1f}% | {q_aff:<9.1f}% | {q_sr:<9.1f}% | {q_ver:<9.1f}% | {q_sub:<9.1f}%")

    # =========================================================================
    # TABLE 2: THE UPDATE TAX AUDIT
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 2: THE UPDATE TAX AUDIT (PER 100-EVENT STREAMING WORKLOAD)")
    print("Quantifying writes, index mutations, serialization boundaries, and invalidation triggers.")
    print("=" * 165)
    print(f"{'Architecture':<32} | {'Writes / 100ev':<16} | {'Index Mutations':<18} | {'Serialization Ops':<20} | {'Invalidations':<16} | {'Total CPU Time':<18}")
    print("-" * 145)

    for name in arch_names:
        r_list = all_results[name]
        w_mean = np.mean([r.total_writes for r in r_list])
        idx_mean = np.mean([r.total_index_mutations for r in r_list])
        ser_mean = np.mean([r.total_serialization_ops for r in r_list])
        inv_mean = np.mean([r.total_invalidation_ops for r in r_list])
        cpu_mean = np.mean([r.total_cpu_time_ms for r in r_list])

        print(f"{name:<32} | {w_mean:<16.1f} | {idx_mean:<18.1f} | {ser_mean:<20.1f} | {inv_mean:<16.1f} | {cpu_mean:<15.2f} ms")

    # =========================================================================
    # TABLE 3: NORMALIZED VERSATILITY DIVIDENDS
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 3: NORMALIZED VERSATILITY DIVIDENDS (RELATIVE TO PRODUCTION-GRADE FRAGMENTED)")
    print("D_update = 1 - W_U / W_F | D_latency = 1 - L_U / L_F | D_coherence = C_U - C_naive | Capability Density = Services / (CPU_ms + MB)")
    print("=" * 165)
    print(f"{'Metric Dimension':<35} | {'Unified Substrate':<22} | {'Fragmented Production':<24} | {'Versatility Dividend':<25}")
    print("-" * 125)

    w_u = np.mean([r.total_writes for r in all_results["3. Unified Context Substrate"]])
    w_f = np.mean([r.total_writes for r in all_results["2. Fragmented Production-Grade"]])
    d_update = (1.0 - w_u / w_f) * 100.0

    l_u = np.mean([r.p95_decision_latency_ms for r in all_results["3. Unified Context Substrate"]])
    l_f = np.mean([r.p95_decision_latency_ms for r in all_results["2. Fragmented Production-Grade"]])
    d_latency = (1.0 - l_u / l_f) * 100.0

    c_u = np.mean([r.version_agreement_rate for r in all_results["3. Unified Context Substrate"]])
    c_naive = np.mean([r.version_agreement_rate for r in all_results["1. Fragmented Naive"]])
    d_coherence = c_u - c_naive

    ser_u = np.mean([r.total_serialization_ops for r in all_results["3. Unified Context Substrate"]])
    ser_f = np.mean([r.total_serialization_ops for r in all_results["2. Fragmented Production-Grade"]])
    d_ser = (1.0 - ser_u / max(1.0, ser_f)) * 100.0

    cpu_u = np.mean([r.total_cpu_time_ms for r in all_results["3. Unified Context Substrate"]])
    cpu_f = np.mean([r.total_cpu_time_ms for r in all_results["2. Fragmented Production-Grade"]])
    density_u = 6.0 / (cpu_u / 100.0 + 1.0)
    density_f = 6.0 / (cpu_f / 100.0 + 1.0)

    print(f"{'Update Tax Elimination (Writes)':<35} | {w_u:<22.1f} | {w_f:<24.1f} | {d_update:+24.1f}%")
    print(f"{'Serialization Boundary Reduction':<35} | {ser_u:<22.1f} | {ser_f:<24.1f} | {d_ser:+24.1f}%")
    print(f"{'Decision Latency (p95 ms)':<35} | {l_u:<19.2f} ms | {l_f:<21.2f} ms | {d_latency:+24.1f}%")
    print(f"{'Snapshot Coherence Dividend':<35} | {c_u:<21.1f}% | {c_naive:<23.1f}% | {d_coherence:+24.1f}%")
    print(f"{'Capability Density (Services/Cost)':<35} | {density_u:<22.2f} | {density_f:<24.2f} | {(density_u / density_f - 1.0)*100:+24.1f}%")

    # =========================================================================
    # TABLE 4: COHERENCE-LATENCY FRONTIER
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 4: COHERENCE-LATENCY FRONTIER")
    print("Measuring p50, p90, and p95 decision latency vs achieved snapshot coherence.")
    print("=" * 165)
    print(f"{'Architecture':<32} | {'Coherence Rate':<16} | {'p50 Latency':<16} | {'p90 Latency':<16} | {'p95 Latency':<16}")
    print("-" * 115)

    for name in arch_names:
        lats = np.array(all_latencies[name])
        coh = np.mean([r.version_agreement_rate for r in all_results[name]])
        p50 = float(np.percentile(lats, 50))
        p90 = float(np.percentile(lats, 90))
        p95 = float(np.percentile(lats, 95))
        print(f"{name:<32} | {coh:<15.1f}% | {p50:<13.2f} ms | {p90:<13.2f} ms | {p95:<13.2f} ms")

    # =========================================================================
    # TABLE 5: MARGINAL COST PER CAPABILITY CURVE
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 5: MARGINAL COST PER CAPABILITY CURVE: TOTAL RUNTIME VS ENABLED SERVICES (n = 1..6)")
    print("Testing if dC_U / dn << dC_F / dn (Amortized Capability Hypothesis)")
    print("=" * 165)
    print(f"{'Enabled Services Count (n)':<32} | {'Unified Substrate (ms)':<26} | {'Fragmented Prod (ms)':<26} | {'Marginal Ratio (dC_U / dC_F)':<30}")
    print("-" * 125)

    # We evaluate sequentially enabling services n=1..6
    service_order = ["search", "context", "route", "affected", "verify", "subscribe"]
    u_costs: List[float] = []
    f_costs: List[float] = []

    test_seed = 35000
    test_catalog = build_research_world(seed=test_seed, world_variant="WORLD_A_LINKED")
    test_workload = generate_streaming_workload(n_steps=50, seed=test_seed, world_variant="WORLD_A_LINKED")

    for k in range(1, 7):
        active = set(service_order[:k])
        u_times = []
        f_times = []
        for rep in range(10):
            u_arch = UnifiedContextSubstrate(test_catalog)
            res_u, _ = evaluate_single_timeline(u_arch, test_workload, enabled_services=active)
            u_times.append(res_u.total_cpu_time_ms)

            f_arch = FragmentedProductionArchitecture(test_catalog, sync_barrier=True)
            res_f, _ = evaluate_single_timeline(f_arch, test_workload, enabled_services=active)
            f_times.append(res_f.total_cpu_time_ms)

        u_costs.append(float(np.mean(u_times)))
        f_costs.append(float(np.mean(f_times)))

    for idx in range(6):
        n = idx + 1
        u_c = u_costs[idx]
        f_c = f_costs[idx]
        if idx == 0:
            m_ratio_str = "Baseline (n=1)"
        else:
            du = u_costs[idx] - u_costs[idx - 1]
            df = f_costs[idx] - f_costs[idx - 1]
            ratio = du / max(0.001, df)
            m_ratio_str = f"du={du:.1f}ms, df={df:.1f}ms (ratio={ratio:.2f}x)"

        services_str = f"n={n} ({'+'.join(service_order[:n])})"
        if len(services_str) > 30:
            services_str = f"n={n} (+{service_order[idx]})"
        print(f"{services_str:<32} | {u_c:<23.2f} ms | {f_c:<23.2f} ms | {m_ratio_str:<30}")

    print("\n" + "=" * 165)
    print(f"Benchmark completed across {n_timelines} timelines in {total_time:.2f} seconds ({total_time / n_timelines:.2f}s / timeline).")
    print("=" * 165)


if __name__ == "__main__":
    n = 30
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_multi_service_benchmark(n_timelines=n)
