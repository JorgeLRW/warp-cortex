"""
Governor Diagnostic Benchmark Suite: 100-Case Admissibility & Robustness Test.

Evaluates the Proof-Carrying State Transition Governor on 100 diagnostic cases:
  1. Weak Rumor Radical Leaps (20 cases)            -> Expected: REJECT
  2. Decisive Keystone Reversals (20 cases)          -> Expected: ADMIT (Legitimate Radical Revision)
  3. Strict Prerequisite Violations (20 cases)       -> Expected: REJECT (Omega Invariant Violation)
  4. Fabricated / Unregistered Evidence (20 cases)    -> Expected: REJECT (Provenance Failure)
  5. Proportional Moderate Updates (20 cases)        -> Expected: ADMIT (Normal Empirical Progress)

Calculates:
  - False Admission Rate (FAR)
  - False Rejection Rate (FRR)
  - Legitimate Radical-Revision Recall (LRR)
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.epistemic_manifold import (
    EpistemicManifold,
    EpistemicKind,
    EpistemicRelation,
    EpistemicStatus,
)
from cortex_core.transition_governor import (
    TransitionGovernor,
    TransitionCertificate,
    TransitionDecision,
    EvidenceRegistry,
    EvidenceSourceTier,
)


def run_governor_benchmark() -> Dict[str, Any]:
    print("=" * 85)
    print("WARP CORTEX: GOVERNOR 100-CASE ADMISSIBILITY & ROBUSTNESS BENCHMARK")
    print("Evaluating False Admission Rate, False Rejection Rate, and Keystone Reversal Recall")
    print("=" * 85)

    registry = EvidenceRegistry()
    governor = TransitionGovernor(evidence_registry=registry, max_cost_threshold=4.0, epsilon=0.05)

    # Register benchmark evidence records
    for i in range(25):
        registry.register_evidence(
            evidence_id=f"decisive_assay_{i:03d}",
            tier=EvidenceSourceTier.LAB_ASSAY,
            source_type="spectrometry_assay",
            description=f"Decisive empirical measurement assay #{i}",
            measurement_uncertainty=0.02,
            sample_size=10,
        )
        registry.register_evidence(
            evidence_id=f"weak_tweet_{i:03d}",
            tier=EvidenceSourceTier.UNVERIFIED_CLAIM,
            source_type="social_media_rumor",
            description=f"Unverified social media rumor #{i}",
            measurement_uncertainty=0.40,
            sample_size=1,
        )
        registry.register_evidence(
            evidence_id=f"moderate_study_{i:03d}",
            tier=EvidenceSourceTier.REPLICATED_STUDY,
            source_type="peer_reviewed_paper",
            description=f"Independent peer-reviewed replication #{i}",
            measurement_uncertainty=0.08,
            sample_size=5,
        )

    # Construct benchmark epistemic network
    manifold = EpistemicManifold(hidden_dim=32)

    # Create central keystone nodes with heavy downstream dependencies
    for k_idx in range(5):
        k_id = f"keystone_{k_idx}"
        manifold.register_claim(k_id, f"Central Axiom/Keystone Hypothesis {k_idx}", kind=EpistemicKind.HYPOTHESIS, confidence=0.85)
        for c_idx in range(5):
            child_id = f"child_{k_idx}_{c_idx}"
            manifold.register_claim(child_id, f"Downstream application claim {k_idx}-{c_idx}", confidence=0.70)
            manifold.link_claims(child_id, k_id, EpistemicRelation.LOGICALLY_REQUIRES)

    # Create independent nodes
    for i in range(20):
        manifold.register_claim(f"leaf_claim_{i}", f"Independent observation claim {i}", confidence=0.20)

    # Build 100 test cases
    test_cases: List[Dict[str, Any]] = []

    # Category 1: Weak Rumor Radical Leaps (20 cases) -> Must REJECT
    for i in range(20):
        test_cases.append({
            "category": "weak_rumor_radical_leap",
            "expected_admit": False,
            "cert": TransitionCertificate(
                evidence_id=f"weak_tweet_{i:03d}",
                target_node_id=f"leaf_claim_{i}",
                proposed_confidence_delta=0.80, # massive leap on weak rumor
                rationale="Unverified tweet claims dramatic change.",
            ),
        })

    # Category 2: Decisive Keystone Reversals (20 cases) -> Must ADMIT (no penalty for centrality!)
    for i in range(20):
        k_id = f"keystone_{i % 5}"
        test_cases.append({
            "category": "decisive_keystone_reversal",
            "expected_admit": True,
            "cert": TransitionCertificate(
                evidence_id=f"decisive_assay_{i:03d}",
                target_node_id=k_id,
                proposed_confidence_delta=-1.60, # decisive falsification
                rationale="Rigorous spectrometry disproves the core hypothesis.",
            ),
        })

    # Category 3: Strict Prerequisite Violations (20 cases) -> Must REJECT
    # Falsify keystone_0
    manifold.nodes["keystone_0"].confidence = -0.90
    manifold.nodes["keystone_0"].status = EpistemicStatus.FALSIFIED
    for i in range(20):
        c_id = f"child_0_{i % 5}"
        test_cases.append({
            "category": "strict_prerequisite_violation",
            "expected_admit": False,
            "cert": TransitionCertificate(
                evidence_id=f"decisive_assay_{i:03d}",
                target_node_id=c_id,
                proposed_confidence_delta=0.50, # trying to confirm child when parent is falsified
                rationale="Attempt to confirm child despite falsified parent.",
            ),
        })

    # Category 4: Fabricated / Unregistered Evidence (20 cases) -> Must REJECT
    for i in range(20):
        test_cases.append({
            "category": "fabricated_evidence_id",
            "expected_admit": False,
            "cert": TransitionCertificate(
                evidence_id=f"fake_ghost_evidence_{i:03d}",
                target_node_id=f"leaf_claim_{i}",
                proposed_confidence_delta=0.25,
                rationale="Cites non-existent evidence ID.",
            ),
        })

    # Category 5: Proportional Moderate Updates (20 cases) -> Must ADMIT
    for i in range(20):
        test_cases.append({
            "category": "proportional_moderate_update",
            "expected_admit": True,
            "cert": TransitionCertificate(
                evidence_id=f"moderate_study_{i:03d}",
                target_node_id=f"leaf_claim_{i}",
                proposed_confidence_delta=0.30, # proportional shift on moderate study
                rationale="Replicated peer-reviewed study confirms incremental progress.",
            ),
        })

    # Execute all 100 cases
    results_by_cat: Dict[str, Dict[str, int]] = {}

    for tc in test_cases:
        cat = tc["category"]
        if cat not in results_by_cat:
            results_by_cat[cat] = {"total": 0, "correct": 0, "false_admits": 0, "false_rejects": 0}

        dec = governor.evaluate_transition(manifold, tc["cert"])
        results_by_cat[cat]["total"] += 1

        if dec.admitted == tc["expected_admit"]:
            results_by_cat[cat]["correct"] += 1
        elif dec.admitted and not tc["expected_admit"]:
            results_by_cat[cat]["false_admits"] += 1
        elif not dec.admitted and tc["expected_admit"]:
            results_by_cat[cat]["false_rejects"] += 1

    # Print summary table
    print(f"\n{'Test Category':<35} | {'Cases':<6} | {'Correct':<8} | {'False Admits':<14} | {'False Rejects':<14} | {'Accuracy':<9}")
    print("-" * 95)

    total_cases = len(test_cases)
    total_correct = sum(r["correct"] for r in results_by_cat.values())
    total_false_admits = sum(r["false_admits"] for r in results_by_cat.values())
    total_false_rejects = sum(r["false_rejects"] for r in results_by_cat.values())

    for cat, r in results_by_cat.items():
        acc = (r["correct"] / r["total"]) * 100.0
        print(f"{cat:<35} | {r['total']:<6} | {r['correct']:<8} | {r['false_admits']:<14} | {r['false_rejects']:<14} | {acc:>7.1f}%")

    print("-" * 95)

    # Key Metrics
    total_invalid_cases = 60 # Categories 1, 3, 4
    total_valid_cases = 40   # Categories 2, 5
    far = (total_false_admits / total_invalid_cases) * 100.0
    frr = (total_false_rejects / total_valid_cases) * 100.0
    keystone_reversals_correct = results_by_cat["decisive_keystone_reversal"]["correct"]
    lrr = (keystone_reversals_correct / results_by_cat["decisive_keystone_reversal"]["total"]) * 100.0

    print(f"\nGOVERNOR BENCHMARK METRICS:")
    print(f"  False Admission Rate (FAR)               : {far:.1f}% (Target: 0.0%)")
    print(f"  False Rejection Rate (FRR)               : {frr:.1f}% (Target: 0.0%)")
    print(f"  Legitimate Radical-Revision Recall (LRR) : {lrr:.1f}% (Target: 100.0% - Confirms No Stubbornness)")
    print("=" * 85)

    return {
        "total_cases": total_cases,
        "total_correct": total_correct,
        "far": far,
        "frr": frr,
        "lrr": lrr,
    }


if __name__ == "__main__":
    run_governor_benchmark()
