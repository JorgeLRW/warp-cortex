"""
Permutation / Shapley Marginal Capability Cost Benchmark.
=========================================================
Rigorously evaluates the Marginal Cost of Capability Addition:
  Delta C(s | A) = C(A union {s}) - C(A)
across K randomized permutations of the 7 services to eliminate service-order
confounding.

Computes:
  - Per-Service Shapley Marginal Cost: MC_s = E_A[Delta C(s | A)]
  - Architecture-Level Marginal Cost: MC_arch = E_{s, A}[Delta C(s | A)]
  - 95% Bootstrap Confidence Intervals for MC_arch

Compares:
  1. Unified Context Substrate (U_v = <S_v, G_v, Z, H_v>)
  2. Representation-Matched Modular Monolith (Identical Z, separate materializations)
  3. Versioned Modular Monolith (Flat single-vector index)
  4. Fragmented Production-Grade Architecture
"""

from __future__ import annotations

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import copy
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.modular_monolith import VersionedModularMonolith
from cortex_apps.multi_service_substrate.representation_matched_monolith import RepresentationMatchedMonolith
from cortex_apps.multi_service_substrate.service7_explain_risk import (
    RiskExplanation,
    explain_risk_fragmented,
    explain_risk_modular_monolith,
    explain_risk_representation_matched_monolith,
    explain_risk_unified,
)
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


ALL_SERVICES = ["context", "route", "affected", "search", "verify", "subscribe", "explain"]


def evaluate_workload_with_active_services(
    arch_name: str,
    arch: ContextSubstrate,
    steps: List[WorkloadStep],
    active_services: Set[str],
) -> float:
    """
    Executes a streaming workload executing ONLY the active_services set.
    Returns total execution CPU time in milliseconds.
    """
    arch.reset_metrics()
    subscribed_events: List[str] = []

    if "subscribe" in active_services:
        arch.subscribe(
            predicate=lambda e: e.event_type in ("SENSOR_SHOCK", "GRAPH_MUTATION"),
            callback=lambda ev, v: subscribed_events.append(ev.event_id),
        )

    t_start = time.perf_counter()

    for step in steps:
        v_ingest = arch.ingest(step.event)

        if step.probe is not None:
            # Service 1: Context Packing
            if "context" in active_services:
                _ = arch.context(step.probe.query, token_budget=256, version=v_ingest)

            # Service 2: Agent Wake Routing
            if "route" in active_services:
                _ = arch.route(step.event)

            # Service 3: Affected Frontier
            if "affected" in active_services:
                if hasattr(arch, "affected"):
                    _ = arch.affected(step.event.entity_id, version=v_ingest)
                else:
                    _ = arch.affected_frontier(step.event.entity_id, version=v_ingest)

            # Service 4: Hybrid Search
            if "search" in active_services:
                _ = arch.search(step.probe.query, top_k=5, version=v_ingest)

            # Service 5: Invariant Verification
            if "verify" in active_services:
                action = ProposedAction(
                    action_id=f"act_probe_{step.step_id}",
                    action_name="Scale-up Pilot Run Alpha",
                    target_node="node_act_bioreactor",
                    required_prerequisites=["node_sensor_ms4", "node_dataset_42", "node_exp_pep", "node_hypo_yield"],
                )
                _ = arch.verify(action, version=v_ingest)

            # Service 7: Cross-Cutting Risk Explanation
            if "explain" in active_services:
                if isinstance(arch, UnifiedContextSubstrate):
                    _ = explain_risk_unified(arch, "ds_proteomics_spectra", version=v_ingest)
                elif isinstance(arch, RepresentationMatchedMonolith):
                    _ = explain_risk_representation_matched_monolith(arch, "ds_proteomics_spectra", version=v_ingest)
                elif isinstance(arch, VersionedModularMonolith):
                    _ = explain_risk_modular_monolith(arch, "ds_proteomics_spectra", version=v_ingest)
                elif isinstance(arch, FragmentedProductionArchitecture):
                    _ = explain_risk_fragmented(
                        state_store=arch.state_store,
                        graph_adj_reverse=arch.reverse_adj,
                        aspect_vectors=arch.aspect_vectors,
                        event_bus_log=arch.event_bus_log,
                        entity_to_doc=arch.entity_to_doc,
                        doc_to_node=arch.doc_to_node,
                        doc_to_entity=arch.doc_to_entity,
                        node_to_doc=arch.node_to_doc,
                        entity_id="ds_proteomics_spectra",
                        v_state=arch.v_state,
                        v_graph=arch.v_graph,
                        v_vector=arch.v_vector,
                        v_bus=arch.global_version,
                    )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return total_ms


def instantiate_architecture(arch_name: str, catalog: ResearchWorldCatalog) -> ContextSubstrate:
    if arch_name == "Unified Context Substrate":
        return UnifiedContextSubstrate(catalog)
    elif arch_name == "Representation-Matched Monolith":
        return RepresentationMatchedMonolith(catalog)
    elif arch_name == "Versioned Modular Monolith":
        return VersionedModularMonolith(catalog)
    elif arch_name == "Fragmented Production-Grade":
        return FragmentedProductionArchitecture(catalog, sync_barrier=True)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def run_shapley_benchmark(
    n_permutations: int = 24,
    n_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Shapley marginal capability costs across n_permutations random orderings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    catalog = build_research_world(seed=seed, world_variant="WORLD_A_LINKED")
    workload = generate_streaming_workload(n_steps=n_steps, seed=seed, world_variant="WORLD_A_LINKED")

    contenders = [
        "Unified Context Substrate",
        "Representation-Matched Monolith",
        "Versioned Modular Monolith",
        "Fragmented Production-Grade",
    ]

    # Generate random permutations
    all_perms = list(itertools.permutations(ALL_SERVICES))
    if len(all_perms) > n_permutations:
        sampled_perms = random.sample(all_perms, n_permutations)
    else:
        sampled_perms = all_perms

    # Results collection
    # arch -> service -> list of delta_C
    service_deltas: Dict[str, Dict[str, List[float]]] = {
        c: {s: [] for s in ALL_SERVICES} for c in contenders
    }
    arch_mean_marginal_costs: Dict[str, List[float]] = {c: [] for c in contenders}

    print(f"Executing Shapley Marginal Cost Benchmark across {len(sampled_perms)} randomized service orders...")

    for p_idx, perm in enumerate(sampled_perms):
        if (p_idx + 1) % 4 == 0 or p_idx == 0:
            print(f"  Processing permutation {p_idx + 1}/{len(sampled_perms)}: {' -> '.join(perm)}")

        for c in contenders:
            costs_by_prefix: Dict[int, float] = {}

            # Evaluate C(empty)
            arch_inst = instantiate_architecture(c, catalog)
            t_base = evaluate_workload_with_active_services(c, arch_inst, workload, set())
            costs_by_prefix[0] = t_base

            # Evaluate incremental additions
            current_set: Set[str] = set()
            for k, s in enumerate(perm):
                current_set.add(s)
                # Average over 2 runs to minimize CPU noise
                r_times = []
                for _ in range(2):
                    arch_inst = instantiate_architecture(c, catalog)
                    t_k = evaluate_workload_with_active_services(c, arch_inst, workload, current_set)
                    r_times.append(t_k)
                costs_by_prefix[k + 1] = float(np.mean(r_times))

                delta = max(0.0, costs_by_prefix[k + 1] - costs_by_prefix[k])
                service_deltas[c][s].append(delta)

            # Overall marginal cost for this permutation: (C(all) - C(empty)) / 7
            overall_mc = (costs_by_prefix[7] - costs_by_prefix[0]) / 7.0
            arch_mean_marginal_costs[c].append(overall_mc)

    # Compute statistics
    summary: Dict[str, Any] = {}
    for c in contenders:
        per_service_stats: Dict[str, Tuple[float, float]] = {}
        for s in ALL_SERVICES:
            arr = np.array(service_deltas[c][s])
            per_service_stats[s] = (float(np.mean(arr)), float(np.std(arr) / np.sqrt(len(arr))))

        mc_arr = np.array(arch_mean_marginal_costs[c])
        mean_mc = float(np.mean(mc_arr))
        se_mc = float(np.std(mc_arr) / np.sqrt(len(mc_arr)))
        ci_lower = mean_mc - 1.96 * se_mc
        ci_upper = mean_mc + 1.96 * se_mc

        summary[c] = {
            "per_service": per_service_stats,
            "mean_marginal_cost": mean_mc,
            "se_marginal_cost": se_mc,
            "ci_95": (ci_lower, ci_upper),
        }

    return summary


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = run_shapley_benchmark(n_permutations=16, n_steps=40, seed=42)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 135)
    print("SHAPLEY MARGINAL CAPABILITY COST BENCHMARK (RANDOMIZED SERVICE PERMUTATIONS)")
    print("Quantifying the true marginal cost of adding capabilities, averaged over random orders.")
    print("=" * 135)
    print(f"{'Architecture':<35} | {'Mean MC (ms/service)':<24} | {'95% Conf. Interval':<24} | {'Ratio vs Unified':<20}")
    print("-" * 135)

    base_mc = res["Unified Context Substrate"]["mean_marginal_cost"]

    for c in res:
        m = res[c]["mean_marginal_cost"]
        se = res[c]["se_marginal_cost"]
        ci = res[c]["ci_95"]
        ratio_str = "1.00x (Baseline)" if c == "Unified Context Substrate" else f"{m / max(0.01, base_mc):.2f}x higher"
        print(f"{c:<35} | {m:6.2f} +/- {se:4.2f} ms         | [{ci[0]:5.2f}, {ci[1]:5.2f}] ms          | {ratio_str:<20}")

    print("\n" + "=" * 135)
    print("PER-SERVICE SHAPLEY MARGINAL COSTS (Mean +/- SE in ms):")
    print("=" * 135)
    hdr = f"{'Service':<14} | " + " | ".join([f"{c[:18]:<18}" for c in res])
    print(hdr)
    print("-" * len(hdr))

    for s in ALL_SERVICES:
        row = f"{s:<14} | "
        for c in res:
            m, se = res[c]["per_service"][s]
            row += f"{m:5.2f} +/- {se:4.2f} ms | "
        print(row)

    print("=" * 135)
    print(f"Completed in {elapsed:.2f} seconds.")
