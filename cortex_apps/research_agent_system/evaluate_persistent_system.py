"""
End-to-End Evaluation: Persistent Research-Workflow Paired-World Factorial Benchmark.
=====================================================================================
Definitive Factorial Evaluation isolating where Warp Cortex earns its architectural complexity:

Factor 1: Topology Completeness:
  - Explicit Path Complete: Causal DAG has full explicit path MS-4 -> Dataset 42 -> Exp Pep -> Yield Model -> Bioreactor Pilot.
  - Explicit Path Incomplete: Link (Dataset 42 -> Exp Pep) is severed in explicit DAG, leaving semantic evidence intact.

Factor 2: Downstream Evidentiary Necessity via Paired Worlds:
  - World A (Linked / Anomaly Matters):
    Quadrupole MS-4 is drifted (TAINTED). Dataset 42 was acquired on MS-4.
    Query: "Is Pilot Run Alpha still scientifically justified based on current empirical evidence?"
    Ground Truth: HALT.
  - World B (Unlinked / Anomaly Irrelevant):
    Quadrupole MS-4 is identically broken (TAINTED), but Dataset 42 was acquired on independent nominal instrument MS-2.
    Query: "Is Pilot Run Alpha still scientifically justified based on current empirical evidence?"
    Ground Truth: COMMIT (PERMIT).
  - Decision Metric: Joint Paired Accuracy = 1[Decision_A == HALT and Decision_B == COMMIT].
    Blindly halting on MS-4 gives 50% accuracy. Discovering/verifying empirical provenance is strictly necessary.

Sanity Control: Task A (Direct Root Query):
  - Query: "What is the current calibration status and operational readiness of Quadrupole MS-4?"
  - Ground Truth: FLAG_ANOMALY. Downstream topology modifications have zero effect on Task A.

Contenders (All equipped with fair Okapi BM25 + dense vector hybrid search):
  1. Stateless Hybrid RAG (Baseline P(x | q))
  2. Status-Aware State Store (Relational DB table + status boost on abnormal documents)
  3. Directed Recursive Graph Store (Multi-hop BFS prefix-preserving reachability: b_G(x) = gamma^d(a, x) * 0.90)
  4. Undirected Graph State Store (Undirected structural BFS reachability)
  5. Cortex Prior Dynamic RAG (Continuous reaction-diffusion field h_t over multi-aspect semantic manifold)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_apps.research_agent_system.world_state import build_research_world, ResearchWorldCatalog
from cortex_apps.research_agent_system.memory_baselines import (
    StatelessRAG,
    StatusAwareStateStoreRAG,
    StatusAwareStaticZExpansionRAG,
    StaticSemanticGraphPropagationRAG,
    DirectedRecursiveGraphStateStoreRAG,
    UndirectedGraphStateStoreRAG,
    CortexPriorRAG,
    TieredContextUnionRAG,
    SemanticQueryEncoder,
    RetrievalResult,
)
from cortex_apps.research_agent_system.agent_council import (
    ExecutiveScaleUpAgent,
    DataIntegrityMonitorAgent,
    AgentDecision,
)
from cortex_apps.research_agent_system.event_stream import generate_research_timeline, StreamEvent


@dataclass
class TrialMetrics:
    first_correct_lag: float
    sustained_coherence_lag: float
    root_incident_rank: float
    downstream_consequence_rank: float
    sop_rank: float
    stale_commit_count: int
    total_scaleup_probes: int
    total_tokens_consumed: int
    post_remediation_accuracy: float
    accuracy_by_budget_task_a: Dict[int, float] = field(default_factory=dict)
    accuracy_a_by_budget_task_b: Dict[int, float] = field(default_factory=dict)
    accuracy_b_by_budget_task_b: Dict[int, float] = field(default_factory=dict)
    joint_accuracy_by_budget_task_b: Dict[int, float] = field(default_factory=dict)
    root_recall_by_budget: Dict[int, float] = field(default_factory=dict)
    downstream_recall_by_budget: Dict[int, float] = field(default_factory=dict)
    false_reach_by_budget: Dict[int, float] = field(default_factory=dict)
    selectivity_by_budget: Dict[int, float] = field(default_factory=dict)
    field_activation_pr: List[Tuple[float, float, float, int, int]] = field(default_factory=list)


def run_single_factorial_trial(
    trial_seed: int,
    complete_topology: bool,
    budgets: List[int] = [128, 256, 512, 1024],
    hidden_dim: int = 64,
    noise_count: int = 50,
) -> Dict[str, TrialMetrics]:
    # World A: Linked (MS-4 -> Dataset 42 -> Exp -> Yield -> Pilot). Ground truth: HALT
    cat_a = build_research_world(
        hidden_dim=hidden_dim, seed=trial_seed, complete_topology=complete_topology, world_variant="WORLD_A_LINKED"
    )
    timeline_a = generate_research_timeline(cat_a, seed=trial_seed, noise_event_count=noise_count)

    # World B: Unlinked (MS-2 -> Dataset 42 -> Exp -> Yield -> Pilot). Ground truth: COMMIT
    cat_b = build_research_world(
        hidden_dim=hidden_dim, seed=trial_seed, complete_topology=complete_topology, world_variant="WORLD_B_UNLINKED"
    )
    timeline_b = generate_research_timeline(cat_b, seed=trial_seed, noise_event_count=noise_count)

    scaleup_agent = ExecutiveScaleUpAgent()
    monitor_agent = DataIntegrityMonitorAgent()
    query_encoder_a = SemanticQueryEncoder(cat_a.band_anchors, cat_a.hidden_dim)
    query_encoder_b = SemanticQueryEncoder(cat_b.band_anchors, cat_b.hidden_dim)

    contenders_a = {
        "1. Stateless Hybrid RAG": StatelessRAG(cat_a),
        "2. Status-Aware State Store": StatusAwareStateStoreRAG(cat_a, beta=1.0),
        "3. Status + Static Z (1-Hop)": StatusAwareStaticZExpansionRAG(cat_a, beta=1.0, alpha_z=0.50, tau_z=0.50),
        "4. Status + Static PPR Graph": StaticSemanticGraphPropagationRAG(cat_a, beta=1.0, alpha_ppr=0.50, steps=2),
        "5. Directed Recursive Graph Store": DirectedRecursiveGraphStateStoreRAG(cat_a, beta=1.0, gamma=0.75),
        "6. Cortex Prior Dynamic RAG": CortexPriorRAG(cat_a, alpha=0.40),
        "7. Tiered Context Union": TieredContextUnionRAG(cat_a, beta=1.0, gamma=0.75, alpha=0.40),
    }
    contenders_b = {
        "1. Stateless Hybrid RAG": StatelessRAG(cat_b),
        "2. Status-Aware State Store": StatusAwareStateStoreRAG(cat_b, beta=1.0),
        "3. Status + Static Z (1-Hop)": StatusAwareStaticZExpansionRAG(cat_b, beta=1.0, alpha_z=0.50, tau_z=0.50),
        "4. Status + Static PPR Graph": StaticSemanticGraphPropagationRAG(cat_b, beta=1.0, alpha_ppr=0.50, steps=2),
        "5. Directed Recursive Graph Store": DirectedRecursiveGraphStateStoreRAG(cat_b, beta=1.0, gamma=0.75),
        "6. Cortex Prior Dynamic RAG": CortexPriorRAG(cat_b, alpha=0.40),
        "7. Tiered Context Union": TieredContextUnionRAG(cat_b, beta=1.0, gamma=0.75, alpha=0.40),
    }

    results: Dict[str, TrialMetrics] = {
        name: TrialMetrics(
            first_correct_lag=float(noise_count + 30),
            sustained_coherence_lag=float(noise_count + 30),
            root_incident_rank=999.0,
            downstream_consequence_rank=999.0,
            sop_rank=999.0,
            stale_commit_count=0,
            total_scaleup_probes=0,
            total_tokens_consumed=0,
            post_remediation_accuracy=0.0,
            accuracy_by_budget_task_a={b: 0.0 for b in budgets},
            accuracy_a_by_budget_task_b={b: 0.0 for b in budgets},
            accuracy_b_by_budget_task_b={b: 0.0 for b in budgets},
            joint_accuracy_by_budget_task_b={b: 0.0 for b in budgets},
            root_recall_by_budget={b: 0.0 for b in budgets},
            downstream_recall_by_budget={b: 0.0 for b in budgets},
            false_reach_by_budget={b: 0.0 for b in budgets},
            selectivity_by_budget={b: 0.0 for b in budgets},
        )
        for name in contenders_a
    }

    shock_step = -1
    shock_active = False
    shock_probe_history: Dict[str, List[Tuple[int, str]]] = {name: [] for name in contenders_a}

    task_a_query_text = "What is the current calibration status and operational readiness of Quadrupole MS-4?"
    task_a_query_vec = query_encoder_a.encode(task_a_query_text)

    task_b_query_text = "Is Pilot Run Alpha still scientifically justified based on current empirical evidence?"
    task_b_query_vec_a = query_encoder_a.encode(task_b_query_text)
    task_b_query_vec_b = query_encoder_b.encode(task_b_query_text)

    for ev_a, ev_b in zip(timeline_a, timeline_b):
        if not ev_a.is_query_probe:
            if ev_a.is_shock:
                shock_active = True
                if shock_step < 0:
                    shock_step = ev_a.step
            elif ev_a.is_remediation:
                shock_active = False

            for name in contenders_a:
                contenders_a[name].record_raw_event(ev_a.event_id, ev_a.text, ev_a.embedding, ev_a.step)
                contenders_b[name].record_raw_event(ev_b.event_id, ev_b.text, ev_b.embedding, ev_b.step)

        elif ev_a.is_query_probe:
            # High-stakes audit sweep evaluated during active unresolved shock
            if ev_a.event_id == "probe_unresolved_high_stakes":
                # --- Task A: Direct Root Calibration Status Probe ---
                for b in budgets:
                    for name in contenders_a:
                        res_a_root = contenders_a[name].query(task_a_query_text, task_a_query_vec, token_budget=b)
                        dec_a_root = monitor_agent.evaluate_sensor_telemetry(res_a_root, ground_truth_has_drift=True)
                        results[name].accuracy_by_budget_task_a[b] = 1.0 if dec_a_root.is_correct else 0.0

                # --- Task B: Downstream Scale-Up Justification Probe across Paired Worlds ---
                for b in budgets:
                    for name in contenders_a:
                        # World A (Linked -> Ground Truth: HALT)
                        res_b_a = contenders_a[name].query(task_b_query_text, task_b_query_vec_a, token_budget=b)
                        dec_b_a = scaleup_agent.evaluate_scaleup_request(res_b_a, ground_truth_status="TAINTED_UNRESOLVED")
                        is_corr_a = dec_b_a.is_correct

                        # World B (Unlinked -> Ground Truth: COMMIT)
                        res_b_b = contenders_b[name].query(task_b_query_text, task_b_query_vec_b, token_budget=b)
                        dec_b_b = scaleup_agent.evaluate_scaleup_request(res_b_b, ground_truth_status="NOMINAL")
                        is_corr_b = dec_b_b.is_correct

                        # Joint Accuracy: 1[Correct_A and Correct_B]
                        is_joint = (is_corr_a and is_corr_b)

                        results[name].accuracy_a_by_budget_task_b[b] = 1.0 if is_corr_a else 0.0
                        results[name].accuracy_b_by_budget_task_b[b] = 1.0 if is_corr_b else 0.0
                        results[name].joint_accuracy_by_budget_task_b[b] = 1.0 if is_joint else 0.0

                        # Evidentiary Reach and Selectivity Metrics
                        results[name].root_recall_by_budget[b] = 1.0 if res_b_a.root_in_context else 0.0
                        results[name].downstream_recall_by_budget[b] = 1.0 if res_b_a.downstream_in_context else 0.0
                        results[name].false_reach_by_budget[b] = res_b_a.false_reach_rate
                        results[name].selectivity_by_budget[b] = res_b_a.selectivity

                        if b == 512:
                            results[name].root_incident_rank = float(res_b_a.root_incident_rank)
                            results[name].downstream_consequence_rank = float(res_b_a.downstream_consequence_rank)
                            results[name].sop_rank = float(res_b_a.sop_rank)

                # Collect continuous reaction field PR curve from Cortex
                cortex_prior = contenders_a["6. Cortex Prior Dynamic RAG"]
                results["6. Cortex Prior Dynamic RAG"].field_activation_pr = cortex_prior.compute_field_activation_pr()

            # Timeline operational tracking on World A
            ground_truth_status = "TAINTED_UNRESOLVED" if shock_active else ("REMEDIATED" if shock_step > 0 else "NOMINAL")
            probe_text = ev_a.probe_query_text or "Evaluate authorization and release $250k capital for Bioreactor Pilot Run Alpha"
            probe_vec = query_encoder_a.encode(probe_text)

            for name in contenders_a:
                res = contenders_a[name].query(probe_text, probe_vec, token_budget=512)
                decision = scaleup_agent.evaluate_scaleup_request(res, ground_truth_status=ground_truth_status)

                m = results[name]
                m.total_scaleup_probes += 1
                m.total_tokens_consumed += res.total_tokens

                if decision.is_stale_commit:
                    m.stale_commit_count += 1

                if shock_active:
                    elapsed = ev_a.step - shock_step
                    shock_probe_history[name].append((elapsed, decision.action))

                if ev_a.event_id == "probe_post_remediation":
                    m.post_remediation_accuracy = 1.0 if decision.action == "COMMIT" else 0.0

    # First-Correct Lag & Sustained Coherence Lag
    never_val = float(noise_count + 30)
    for name in contenders_a:
        hist = shock_probe_history[name]
        if not hist:
            continue

        first_correct = never_val
        for elapsed, action in hist:
            if action == "HALT":
                first_correct = float(elapsed)
                break
        results[name].first_correct_lag = first_correct

        sustained = never_val
        for idx in range(len(hist)):
            all_halt = all(act == "HALT" for _, act in hist[idx:])
            if all_halt:
                sustained = float(hist[idx][0])
                break
        results[name].sustained_coherence_lag = sustained

    return results


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def mcnemar_test(cortex_accs: np.ndarray, baseline_accs: np.ndarray) -> Tuple[int, int, float]:
    b = int(np.sum((cortex_accs > 0.5) & (baseline_accs <= 0.5)))  # Cortex correct, baseline wrong
    c = int(np.sum((cortex_accs <= 0.5) & (baseline_accs > 0.5)))  # Cortex wrong, baseline correct
    n = b + c
    if n == 0:
        return b, c, 1.0
    from scipy import stats
    try:
        res = stats.binomtest(min(b, c), n, 0.5, alternative="two-sided")
        p_val = res.pvalue
    except Exception:
        z = (abs(b - c) - 1) / math.sqrt(n)
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return b, c, float(p_val)


def run_decisive_2x2_benchmark(n_trials: int = 500):
    print("=" * 165)
    print("WARP CORTEX: THE DECISIVE PAIRED-WORLD FACTORIAL BENCHMARK")
    print(f"Evaluating 4 Memory Architectures across 2 Topologies (Complete vs Incomplete) over {n_trials} Matched Timelines")
    print("Direct-Root Sanity Test (Task A) vs Downstream Evidentiary Necessity Paired Worlds (Task B: World A vs World B)")
    print("=" * 165)

    budgets = [128, 256, 512, 1024]
    topologies = [True, False]
    top_names = {True: "Complete Topology", False: "Incomplete Topology"}
    contender_names = [
        "1. Stateless Hybrid RAG",
        "2. Status-Aware State Store",
        "3. Status + Static Z (1-Hop)",
        "4. Status + Static PPR Graph",
        "5. Directed Recursive Graph Store",
        "6. Cortex Prior Dynamic RAG",
        "7. Tiered Context Union",
    ]

    all_data: Dict[bool, Dict[str, Dict[str, List[float]]]] = {
        top: {
            name: {
                "first_correct_lag": [],
                "sustained_coherence_lag": [],
                "root_incident_rank": [],
                "downstream_consequence_rank": [],
                "sop_rank": [],
                "stale_commit_rate": [],
                "tokens_consumed": [],
                "post_remediation_acc": [],
                **{f"acc_a_{b}": [] for b in budgets},
                **{f"acc_b_wA_{b}": [] for b in budgets},
                **{f"acc_b_wB_{b}": [] for b in budgets},
                **{f"acc_b_joint_{b}": [] for b in budgets},
                **{f"root_rec_{b}": [] for b in budgets},
                **{f"down_rec_{b}": [] for b in budgets},
                **{f"false_reach_{b}": [] for b in budgets},
                **{f"selectivity_{b}": [] for b in budgets},
            }
            for name in contender_names
        }
        for top in topologies
    }

    cortex_pr_curves: Dict[bool, List[List[Tuple[float, float, float, int, int]]]] = {
        top: [] for top in topologies
    }

    test_seed_start = 20000
    t0 = time.perf_counter()

    for top in topologies:
        print(f"\n[Starting Evaluation on: {top_names[top].upper()} across {n_trials} timelines (Seeds {test_seed_start}..{test_seed_start + n_trials - 1})]")
        for i in range(n_trials):
            trial_seed = test_seed_start + i
            trial_results = run_single_factorial_trial(
                trial_seed=trial_seed,
                complete_topology=top,
                budgets=budgets,
            )

            for name, m in trial_results.items():
                d = all_data[top][name]
                d["first_correct_lag"].append(m.first_correct_lag)
                d["sustained_coherence_lag"].append(m.sustained_coherence_lag)
                d["root_incident_rank"].append(m.root_incident_rank)
                d["downstream_consequence_rank"].append(m.downstream_consequence_rank)
                d["sop_rank"].append(m.sop_rank)
                stale_rate = (m.stale_commit_count / max(1, m.total_scaleup_probes)) * 100.0
                d["stale_commit_rate"].append(stale_rate)
                d["tokens_consumed"].append(float(m.total_tokens_consumed))
                d["post_remediation_acc"].append(m.post_remediation_accuracy * 100.0)

                for b in budgets:
                    d[f"acc_a_{b}"].append(m.accuracy_by_budget_task_a[b] * 100.0)
                    d[f"acc_b_wA_{b}"].append(m.accuracy_a_by_budget_task_b[b] * 100.0)
                    d[f"acc_b_wB_{b}"].append(m.accuracy_b_by_budget_task_b[b] * 100.0)
                    d[f"acc_b_joint_{b}"].append(m.joint_accuracy_by_budget_task_b[b] * 100.0)
                    d[f"root_rec_{b}"].append(m.root_recall_by_budget[b] * 100.0)
                    d[f"down_rec_{b}"].append(m.downstream_recall_by_budget[b] * 100.0)
                    d[f"false_reach_{b}"].append(m.false_reach_by_budget[b] * 100.0)
                    d[f"selectivity_{b}"].append(m.selectivity_by_budget[b] * 100.0)

                if name == "6. Cortex Prior Dynamic RAG" and m.field_activation_pr:
                    cortex_pr_curves[top].append(m.field_activation_pr)

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {i + 1}/{n_trials} timelines ({elapsed:.1f}s elapsed)...")

    total_sec = time.perf_counter() - t0

    # =========================================================================
    # TABLE 1: TASK A DIRECT-ROOT SANITY TEST ACCURACY (%)
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 1: TASK A DIRECT-ROOT SANITY TEST ACCURACY (%) ACROSS CONTEXT BUDGETS")
    print("Query: 'What is the current calibration status and operational readiness of Quadrupole MS-4?'")
    print(f"Untouched test seeds ({test_seed_start}..{test_seed_start + n_trials - 1}), {n_trials} matched timelines per cell. Fair BM25 + dense hybrid search.")
    print("=" * 165)

    for top in [True, False]:
        t_title = top_names[top].upper()
        print(f"\n====================== TOPOLOGY: {t_title} ======================")
        print(f"{'Memory Architecture':<36} | {'128 Tokens':<14} | {'256 Tokens':<14} | {'512 Tokens':<14} | {'1024 Tokens':<14}")
        print("-" * 105)
        for name in contender_names:
            d = all_data[top][name]
            acc_128 = np.mean(d["acc_a_128"])
            acc_256 = np.mean(d["acc_a_256"])
            acc_512 = np.mean(d["acc_a_512"])
            acc_1024 = np.mean(d["acc_a_1024"])
            print(f"{name:<36} | {acc_128:<13.1f}% | {acc_256:<13.1f}% | {acc_512:<13.1f}% | {acc_1024:<13.1f}%")

    # =========================================================================
    # TABLE 2: TASK B JOINT PAIRED DECISION ACCURACY (%)
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 2: TASK B JOINT PAIRED DECISION ACCURACY (%) ACROSS CONTEXT BUDGETS")
    print("Query: 'Is Pilot Run Alpha still scientifically justified based on current empirical evidence?'")
    print("Joint Paired Accuracy = 1[Decision_A == HALT and Decision_B == COMMIT]. Evaluated across paired identical-query worlds.")
    print("=" * 165)

    for top in [True, False]:
        t_title = top_names[top].upper()
        print(f"\n====================== TOPOLOGY: {t_title} (Task B Joint Paired Accuracy) ======================")
        print(f"{'Memory Architecture':<36} | {'128 Tokens':<14} | {'256 Tokens':<14} | {'512 Tokens':<14} | {'1024 Tokens':<14} | {'World A (HALT)':<16} | {'World B (PERMIT)':<18}")
        print("-" * 145)
        for name in contender_names:
            d = all_data[top][name]
            j_128 = np.mean(d["acc_b_joint_128"])
            j_256 = np.mean(d["acc_b_joint_256"])
            j_512 = np.mean(d["acc_b_joint_512"])
            j_1024 = np.mean(d["acc_b_joint_1024"])
            wA_512 = np.mean(d["acc_b_wA_512"])
            wB_512 = np.mean(d["acc_b_wB_512"])
            print(f"{name:<36} | {j_128:<13.1f}% | {j_256:<13.1f}% | {j_512:<13.1f}% | {j_1024:<13.1f}% | {wA_512:<15.1f}% | {wB_512:<17.1f}%")

    # =========================================================================
    # TABLE 3: DOWNSTREAM EVIDENTIARY RECALL & SELECTIVITY AUDIT (TASK B)
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 3: EVIDENTIARY REACH & SELECTIVITY AUDIT ON TASK B")
    print("Auditing: Root Recall (MS-4 in context), Downstream Recall (Data42 in context), False Reach Rate, and Selectivity")
    print("=" * 165)

    for top in [True, False]:
        t_title = top_names[top].upper()
        print(f"\n====================== TOPOLOGY: {t_title} (Task B @ 128 and 256 Tokens) ======================")
        print(f"{'Memory Architecture':<36} | {'RootRec@128':<12} | {'DownRec@128':<12} | {'FalseReach@128':<15} | {'Selectivity@128':<16} | {'RootRec@256':<12} | {'DownRec@256':<12} | {'Selectivity@256':<16}")
        print("-" * 160)
        for name in contender_names:
            d = all_data[top][name]
            r128 = np.mean(d["root_rec_128"])
            d128 = np.mean(d["down_rec_128"])
            fr128 = np.mean(d["false_reach_128"])
            sel128 = np.mean(d["selectivity_128"])
            r256 = np.mean(d["root_rec_256"])
            d256 = np.mean(d["down_rec_256"])
            sel256 = np.mean(d["selectivity_256"])
            print(f"{name:<36} | {r128:<11.1f}% | {d128:<11.1f}% | {fr128:<14.1f}% | {sel128:<15.1f}% | {r256:<11.1f}% | {d256:<11.1f}% | {sel256:<15.1f}%")

    # =========================================================================
    # TABLE 4: CONTINUOUS REACTION FIELD PR EXPLOSION AUDIT (CORTEX)
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 4: CONTINUOUS REACTION FIELD ACTIVATION PR EXPLOSION AUDIT (CORTEX)")
    print("Auditing: Reaction Field Activation Precision(theta) vs Activation Recall(theta) across theta in [0.05..0.95]")
    print("=" * 165)

    for top in [True, False]:
        t_title = top_names[top].upper()
        curves = cortex_pr_curves[top]
        if not curves:
            continue

        n_pts = len(curves[0])
        print(f"\n--- Reaction Field PR Curve ({t_title}) averaged over {len(curves)} timelines ---")
        print(f"{'Theta Threshold':<18} | {'Activation Recall':<20} | {'Activation Precision':<22} | {'Active Entities':<18} | {'Relevant Hits':<16}")
        print("-" * 105)
        for pt_idx in range(n_pts):
            th = curves[0][pt_idx][0]
            avg_rec = np.mean([c[pt_idx][1] for c in curves]) * 100.0
            avg_prec = np.mean([c[pt_idx][2] for c in curves]) * 100.0
            avg_active = np.mean([c[pt_idx][3] for c in curves])
            avg_hits = np.mean([c[pt_idx][4] for c in curves])
            print(f"{th:<18.2f} | {avg_rec:<19.1f}% | {avg_prec:<21.1f}% | {avg_active:<18.1f} | {avg_hits:<16.1f}")

    # =========================================================================
    # TABLE 5: PAIRED DIFFERENCE ANALYSIS ON TASK B JOINT ACCURACY
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 5: PAIRED DIFFERENCE ANALYSIS ON TASK B JOINT ACCURACY")
    print(f"Delta Joint Accuracy = Cortex - Baseline across {n_trials} matched timelines with 95% Bootstrap CIs & McNemar Tests")
    print("=" * 165)
    print(f"{'Topology & Baseline Comparison':<45} | {'Budget':<8} | {'Cortex':<9} | {'Baseline':<10} | {'Delta':<10} | {'95% Bootstrap CI':<22} | {'McNemar (b/c, p)':<24}")
    print("-" * 165)

    for top in [True, False]:
        t_label = "Complete" if top else "Incomplete"
        cortex_data = all_data[top]["6. Cortex Prior Dynamic RAG"]
        union_data = all_data[top]["7. Tiered Context Union"]
        graph_data = all_data[top]["5. Directed Recursive Graph Store"]

        baselines = [
            ("Status-Aware State Store", all_data[top]["2. Status-Aware State Store"]),
            ("Status + Static Z (1-Hop)", all_data[top]["3. Status + Static Z (1-Hop)"]),
            ("Status + Static PPR Graph", all_data[top]["4. Status + Static PPR Graph"]),
            ("Directed Recursive Graph Store", graph_data),
        ]

        for base_name, b_data in baselines:
            comp_label = f"[{t_label}] Cortex vs {base_name}"
            for b in budgets:
                c_acc = np.array(cortex_data[f"acc_b_joint_{b}"])
                s_acc = np.array(b_data[f"acc_b_joint_{b}"])
                diff = c_acc - s_acc

                mean_c = np.mean(c_acc)
                mean_s = np.mean(s_acc)
                mean_diff = np.mean(diff)

                ci_low, ci_high = bootstrap_ci(diff, n_boot=10000)
                b_wins, c_wins, p_val = mcnemar_test(c_acc, s_acc)

                p_str = f"b={b_wins}, c={c_wins}, p={p_val:.2e}" if p_val < 0.001 else f"b={b_wins}, c={c_wins}, p={p_val:.4f}"
                ci_str = f"[{ci_low:+.1f}%, {ci_high:+.1f}%]"

                print(f"{comp_label:<45} | {b:<8} | {mean_c:<8.1f}% | {mean_s:<9.1f}% | {mean_diff:<+9.1f}% | {ci_str:<22} | {p_str:<24}")
            print("-" * 165)

        # Also evaluate Tiered Context Union vs Directed Graph
        union_label = f"[{t_label}] Tiered Union vs Directed Graph"
        for b in budgets:
            u_acc = np.array(union_data[f"acc_b_joint_{b}"])
            g_acc = np.array(graph_data[f"acc_b_joint_{b}"])
            diff = u_acc - g_acc
            mean_u = np.mean(u_acc)
            mean_g = np.mean(g_acc)
            mean_diff = np.mean(diff)
            ci_low, ci_high = bootstrap_ci(diff, n_boot=10000)
            b_wins, c_wins, p_val = mcnemar_test(u_acc, g_acc)
            p_str = f"b={b_wins}, c={c_wins}, p={p_val:.2e}" if p_val < 0.001 else f"b={b_wins}, c={c_wins}, p={p_val:.4f}"
            ci_str = f"[{ci_low:+.1f}%, {ci_high:+.1f}%]"
            print(f"{union_label:<45} | {b:<8} | {mean_u:<8.1f}% | {mean_g:<9.1f}% | {mean_diff:<+9.1f}% | {ci_str:<22} | {p_str:<24}")
        print("-" * 165)

    # =========================================================================
    # TABLE 6: TIMELINE OPERATIONAL METRICS
    # =========================================================================
    print("\n" + "=" * 165)
    print("TABLE 6: TIMELINE OPERATIONAL & HYSTERESIS METRICS (500 TIMELINES)")
    print("=" * 165)
    print(f"{'Topology & Architecture':<40} | {'1st-Correct Lag':<16} | {'Coherence Lag':<18} | {'Stale Commit %':<16} | {'Post-Remediation %':<20}")
    print("-" * 165)

    for top in [True, False]:
        t_label = "Complete" if top else "Incomplete"
        for name in contender_names:
            d = all_data[top][name]
            f_lag = np.mean(d["first_correct_lag"])
            s_lag = np.mean(d["sustained_coherence_lag"])
            stale = np.mean(d["stale_commit_rate"])
            post = np.mean(d["post_remediation_acc"])

            f_str = f"{f_lag:.1f} ev" if f_lag < 70 else ">70 (NEVER)"
            s_str = f"{s_lag:.1f} ev" if s_lag < 70 else ">70 (NEVER)"
            disp_name = f"[{t_label}] {name}"
            print(f"{disp_name:<40} | {f_str:<16} | {s_str:<18} | {stale:<15.1f}% | {post:<19.1f}%")
        print("-" * 165)

    print(f"\nAll {n_trials * 2} Timelines Completed in {total_sec:.2f} seconds ({total_sec / (n_trials * 2):.3f}s / timeline).")


if __name__ == "__main__":
    n = 500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_decisive_2x2_benchmark(n_trials=n)
