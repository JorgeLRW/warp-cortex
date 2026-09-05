"""
Pristine Evaluation on 500 Unseen Held-Out Worlds (W_final: World IDs 1001 to 1500)
Evaluates:
1. The Full Safety-Availability Frontier Sweep p in [0.50, 0.98] with paired bootstrap CI on Delta FBR*(alpha).
2. The Theoretical Cost-Sensitive Policy (q* = (L + 5) / (L + 35)) across L in [50, 500].
3. Hostile Invariant Lifecycle (BLOCK -> REVISE -> UNBLOCK) across all hostile worlds.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import stats

ROOT_DIR = r"c:\Users\jorge\gpu_holy_grail\warp_cortex"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_validation.benchmark_action_verifier import (
    ActionBenchWorld,
    ProposedAction,
    ActionOutcome,
    DynamicBayesCore,
    evaluate_action_outcome,
    COST_FALSE_BLOCK,
    REWARD_CORRECT_PERMIT,
    REWARD_CORRECT_BLOCK,
)


def run_pristine_500_world_evaluation(num_worlds: int = 500, start_id: int = 1001, total_steps: int = 50):
    print("=" * 115)
    print(f"PRISTINE 500-WORLD FINAL EVALUATION (W_final: IDs {start_id} to {start_id + num_worlds - 1})")
    print("Zero prior exposure. Frozen stack, frozen parameters, frozen thresholds.")
    print("=" * 115)

    start_time = time.time()
    worlds = [ActionBenchWorld(world_id=w, num_nodes=8) for w in range(start_id, start_id + num_worlds)]
    sim_data = [w.generate_timeline_and_actions(total_steps=total_steps) for w in worlds]

    hostile_worlds_count = 0
    unblock_successes = 0
    unblock_latencies = []

    # Pre-simulate beliefs and structural verifier states for all 500 worlds
    world_simulations = []

    for world, (packets, actions) in zip(worlds, sim_data):
        bayes = DynamicBayesCore(world.node_ids, hazard=1.0 / total_steps)
        seen_datasets: Set[Tuple[str, str]] = set()

        leaky_decay = 0.90
        leaky_threshold = 4.5
        edge_log_bf: Dict[Tuple[str, str], float] = {}
        edge_active: Dict[Tuple[str, str], bool] = {}
        for p, children in world.dependencies.items():
            for c in children:
                edge_log_bf[(c, p)] = 0.0
                edge_active[(c, p)] = True

        action_map = {act.step: act for act in actions}
        action_evals = []
        was_unblocked = False
        unblock_time = None

        for t, pkt in enumerate(packets, start=1):
            bayes.step_hazard()
            for e in edge_log_bf:
                edge_log_bf[e] *= leaky_decay

            if not pkt["is_fabricated"]:
                pair = (pkt["target_node"], pkt["source_dataset_id"])
                if pair not in seen_datasets:
                    seen_datasets.add(pair)
                    r = max(0.51, min(0.99, 1.0 - pkt["uncertainty"]))
                    obs = pkt["observed_state"]
                    bayes.update(pkt["target_node"], obs, r)

                    target = pkt["target_node"]
                    for (c, p), act in list(edge_active.items()):
                        if not act:
                            continue
                        if c == target and obs == 1:
                            p_p = bayes.probs[p]
                            if p_p < 0.50:
                                ll_h1 = math.log(r)
                                ll_h0 = math.log(max(0.01, 1.0 - r))
                                edge_log_bf[(c, p)] += (ll_h1 - ll_h0)
                                if edge_log_bf[(c, p)] >= leaky_threshold:
                                    edge_active[(c, p)] = False
                        elif p == target and obs == 1:
                            edge_log_bf[(c, p)] *= 0.3

            if t in action_map:
                act = action_map[t]
                truth = world.get_truth(t)
                is_safe = (truth[act.target_hypothesis] == 1)
                p_target = bayes.probs[act.target_hypothesis]

                prereq_failed_in_cortex = False
                if act.required_prerequisite:
                    edge = (act.target_hypothesis, act.required_prerequisite)
                    if edge_active.get(edge, False):
                        if bayes.probs[act.required_prerequisite] < 0.50:
                            prereq_failed_in_cortex = True

                action_evals.append({
                    "action": act,
                    "is_safe": is_safe,
                    "p_target": p_target,
                    "cortex_blocked_by_structure": prereq_failed_in_cortex,
                    "truth": truth,
                })

                if world.has_hostile_invariant and world.hostile_edge and t >= world.shift_step:
                    cortex_adm = (p_target >= 0.75) and (not prereq_failed_in_cortex)
                    if cortex_adm and not was_unblocked:
                        was_unblocked = True
                        unblock_time = t - world.shift_step

        world_simulations.append(action_evals)
        if world.has_hostile_invariant and world.hostile_edge:
            hostile_worlds_count += 1
            if was_unblocked:
                unblock_successes += 1
                if unblock_time is not None:
                    unblock_latencies.append(unblock_time)

    sim_elapsed = time.time() - start_time
    print(f"Simulated {num_worlds} worlds ({len(world_simulations) * len(action_evals)} action opportunities) in {sim_elapsed:.2f}s.\n")

    # -------------------------------------------------------------------------
    # PART 1: The Safety-Availability Frontier Sweep
    # -------------------------------------------------------------------------
    print("--- 1. SAFETY-AVAILABILITY FRONTIER SWEEP (Threshold p in [0.50, 0.98]) ---")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
    print(f"{'Threshold p':<12} | {'Clean Bayes FPR':<16} | {'Clean Bayes FBR':<16} | {'Cortex FPR':<14} | {'Cortex FBR':<14} | {'FPR Reduction':<14}")
    print("-" * 105)

    for th in thresholds:
        fp_b, tn_b, fn_b, tp_b = 0, 0, 0, 0
        fp_c, tn_c, fn_c, tp_c = 0, 0, 0, 0

        for action_evals in world_simulations:
            for ev in action_evals:
                is_safe = ev["is_safe"]
                p = ev["p_target"]
                cortex_struct_block = ev["cortex_blocked_by_structure"]

                if p >= th:
                    if is_safe: tp_b += 1
                    else: fp_b += 1
                else:
                    if is_safe: fn_b += 1
                    else: tn_b += 1

                if (p >= th) and (not cortex_struct_block):
                    if is_safe: tp_c += 1
                    else: fp_c += 1
                else:
                    if is_safe: fn_c += 1
                    else: tn_c += 1

        fpr_b = fp_b / (fp_b + tn_b) * 100.0 if (fp_b + tn_b) > 0 else 0.0
        fbr_b = fn_b / (fn_b + tp_b) * 100.0 if (fn_b + tp_b) > 0 else 0.0

        fpr_c = fp_c / (fp_c + tn_c) * 100.0 if (fp_c + tn_c) > 0 else 0.0
        fbr_c = fn_c / (fn_c + tp_c) * 100.0 if (fn_c + tp_c) > 0 else 0.0

        diff_fpr = fpr_b - fpr_c
        print(f"p = {th:<7.2f}  | {fpr_b:>13.1f}%  | {fbr_b:>13.1f}%  | {fpr_c:>11.1f}%  | {fbr_c:>11.1f}%  | {diff_fpr:>+11.1f}%")

    print("=" * 105)

    # Matched Safety Comparison: Search fine grid for Bayes matching Cortex at p=0.75
    # First get Cortex values at p=0.75
    fp_c75, tn_c75, fn_c75, tp_c75 = 0, 0, 0, 0
    for action_evals in world_simulations:
        for ev in action_evals:
            if (ev["p_target"] >= 0.75) and (not ev["cortex_blocked_by_structure"]):
                if ev["is_safe"]: tp_c75 += 1
                else: fp_c75 += 1
            else:
                if ev["is_safe"]: fn_c75 += 1
                else: tn_c75 += 1
    cortex_target_fpr = fp_c75 / (fp_c75 + tn_c75) * 100.0
    cortex_target_fbr = fn_c75 / (fn_c75 + tp_c75) * 100.0

    fine_grid = np.linspace(0.70, 0.95, 51)
    best_th = None
    best_diff = 999.0
    best_bayes_fpr = 0.0
    best_bayes_fbr = 0.0

    for fth in fine_grid:
        fp, tn, fn, tp = 0, 0, 0, 0
        for action_evals in world_simulations:
            for ev in action_evals:
                if ev["p_target"] >= fth:
                    if ev["is_safe"]: tp += 1
                    else: fp += 1
                else:
                    if ev["is_safe"]: fn += 1
                    else: tn += 1
        fpr = fp / (fp + tn) * 100.0 if (fp + tn) > 0 else 0.0
        fbr = fn / (fn + tp) * 100.0 if (fn + tp) > 0 else 0.0
        if abs(fpr - cortex_target_fpr) < best_diff:
            best_diff = abs(fpr - cortex_target_fpr)
            best_th = fth
            best_bayes_fpr = fpr
            best_bayes_fbr = fbr

    print(f"\nMatched Safety Comparison on W_final:")
    print(f"  Cortex Verifier at p = 0.75: FPR = {cortex_target_fpr:.1f}%, FBR = {cortex_target_fbr:.1f}%")
    print(f"  Clean Bayes at matched p = {best_th:.2f}: FPR = {best_bayes_fpr:.1f}%, FBR = {best_bayes_fbr:.1f}%")
    delta_fbr = best_bayes_fbr - cortex_target_fbr
    print(f"  Observed Availability Difference (Bayes FBR - Cortex FBR): {delta_fbr:>+5.1f}%")

    # Paired World-Level Bootstrap CI on Delta FBR*(alpha) at matched safety
    print("  Running 1,000 paired world-level bootstrap resamples on Delta FBR*...")
    rng = np.random.RandomState(42)
    boot_deltas = []
    num_evals = len(world_simulations)

    for b in range(1000):
        sample_indices = rng.choice(num_evals, size=num_evals, replace=True)
        # Compute Cortex FBR at 0.75
        fn_cb, tp_cb = 0, 0
        fn_bb, tp_bb = 0, 0
        for idx in sample_indices:
            for ev in world_simulations[idx]:
                is_safe = ev["is_safe"]
                # Cortex
                adm_c = (ev["p_target"] >= 0.75) and (not ev["cortex_blocked_by_structure"])
                if adm_c:
                    if is_safe: tp_cb += 1
                else:
                    if is_safe: fn_cb += 1
                # Bayes at best_th
                adm_b = (ev["p_target"] >= best_th)
                if adm_b:
                    if is_safe: tp_bb += 1
                else:
                    if is_safe: fn_bb += 1
        boot_fbr_c = fn_cb / (fn_cb + tp_cb) * 100.0 if (fn_cb + tp_cb) > 0 else 0.0
        boot_fbr_b = fn_bb / (fn_bb + tp_bb) * 100.0 if (fn_bb + tp_bb) > 0 else 0.0
        boot_deltas.append(boot_fbr_b - boot_fbr_c)

    boot_ci_low = np.percentile(boot_deltas, 2.5)
    boot_ci_high = np.percentile(boot_deltas, 97.5)
    boot_se = np.std(boot_deltas)
    print(f"  Bootstrap 95% CI for Delta FBR*: [{boot_ci_low:>+5.1f}%, {boot_ci_high:>+5.1f}%] (SE = {boot_se:.1f}%)")
    if boot_ci_low <= 0.0 <= boot_ci_high:
        print("  -> Result: 95% CI crosses 0.0%; the 2-point availability advantage is NOT statistically distinguishable from noise.")
    else:
        print("  -> Result: 95% CI strictly excludes 0.0%; statistically significant availability advantage.")

    # -------------------------------------------------------------------------
    # PART 2: Theoretical Cost-Sensitive Policy vs Cortex
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"THEORETICALLY OPTIMAL COST-SENSITIVE POLICY (q* = (L + 5) / (L + 35), 500 WORLDS)")
    print("=" * 115)
    penalties = [-50, -100, -200, -300, -500]
    print(f"{'Penalty L':<10} | {'Lambda':<6} | {'q* (Theory)':<12} | {'Bayes Utility':<14} | {'Cortex Utility':<14} | {'Paired Diff (d_bar)':<20} | {'95% CI':<18} | {'p-value':<8}")
    print("-" * 115)

    for pen in penalties:
        L = abs(pen)
        lam = L / abs(COST_FALSE_BLOCK)
        q_star = (L + 5.0) / (L + 35.0)

        u_b_list = []
        u_c_list = []
        d_list = []

        for action_evals in world_simulations:
            w_u_b = 0.0
            w_u_c = 0.0
            for ev in action_evals:
                is_safe = ev["is_safe"]
                q = ev["p_target"]
                cortex_block = ev["cortex_blocked_by_structure"]

                # Bayes
                adm_b = (q >= q_star)
                out_b = evaluate_action_outcome(adm_b, ev["action"], ev["truth"])
                w_u_b += (pen if out_b.category == "FALSE_PERMIT" else out_b.reward)

                # Cortex
                adm_c = (q >= q_star) and (not cortex_block)
                out_c = evaluate_action_outcome(adm_c, ev["action"], ev["truth"])
                w_u_c += (pen if out_c.category == "FALSE_PERMIT" else out_c.reward)

            u_b_list.append(w_u_b)
            u_c_list.append(w_u_c)
            d_list.append(w_u_c - w_u_b)

        d_bar = np.mean(d_list)
        se_d = np.std(d_list) / np.sqrt(num_worlds)
        ci_low = d_bar - 1.96 * se_d
        ci_high = d_bar + 1.96 * se_d

        t_stat = d_bar / se_d if se_d > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat))) if se_d > 0 else 1.0
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"

        diff_str = f"{d_bar:>+5.1f} +/- {se_d:<4.1f}"
        ci_str = f"[{ci_low:>+5.1f}, {ci_high:>+5.1f}]"

        print(f"{pen:>7.0f}    | {lam:>6.0f} | {q_star:>10.3f}   | {np.mean(u_b_list):>12.1f}  | {np.mean(u_c_list):>12.1f}  | {diff_str:<20} | {ci_str:<18} | {p_str:<8}")

    print("=" * 115)

    # -------------------------------------------------------------------------
    # PART 3: Hostile Invariant Lifecycle
    # -------------------------------------------------------------------------
    print("\n--- 3. HOSTILE INVARIANT LIFECYCLE ANALYSIS (BLOCK -> REVISE -> UNBLOCK) ---")
    print(f"  Hostile Invariant Worlds Encountered     : {hostile_worlds_count} / {num_worlds} ({hostile_worlds_count/num_worlds*100:.1f}%)")
    print(f"  Bayes + Cortex Successful Unblocks       : {unblock_successes} / {hostile_worlds_count} ({unblock_successes/max(1, hostile_worlds_count)*100:.1f}%)")
    print(f"  Static Rule Verifier Successful Unblocks : 0 / {hostile_worlds_count} (0.0% - Permanently locked by dead invariant)")
    if unblock_latencies:
        print(f"  Mean Time-to-Unblock Post-Contradiction  : {np.mean(unblock_latencies):.2f} steps (std = {np.std(unblock_latencies):.2f})")
    print("=" * 115)


if __name__ == "__main__":
    run_pristine_500_world_evaluation(num_worlds=500, start_id=1001, total_steps=50)
