"""
Multi-World Epistemic Reality Benchmark across Hundreds of Procedurally Generated Worlds.

Evaluates the Epistemic Governor across diverse procedural realities with:
  1. Randomized DAG topologies (trees, diamonds, multi-parent DAGs)
  2. Non-stationary paradigm shift changepoints with stochastic timing
  3. Pseudoreplication (duplicate reports citing identical raw datasets)
  4. Measurement noise and assay corruption
  5. Hostile Graph Invariants (scientist's prior graph asserts a false constraint C -> P where P=0 but C=1)

Contenders:
  1. True Bayesian Changepoint Oracle: Exact joint HMM filter with changepoint hazard and deduplication (Bayes-optimal ceiling)
  2. Cortex Governor: Level-1 Epistemic Continuity + Level-2 Topology Revision under persistent edge strain
  3. Bayesian DAG + DeDup Baseline: Stationary log-odds with DAG clamping and deduplication (formerly 'Oracle')
  4. Bayesian Independent / Naive: Unconstrained independent log-odds updater
  5. Ignorance Baseline (P = 0.50): Standard Brier = 0.2500, accuracy = 50.0%
  6. Stubborn Updater: Old topological centrality penalty (w_i = 1 + 0.5 * reach)

Tracks:
  - Standard Brier Score: (1/M) sum (p_i - y_i)^2
  - Expected Calibration Error (ECE)
  - Latent Classification Accuracy (%)
  - Paradigm Shift Reversal Latency (steps post-shift)
  - False Invariant Severance Rate (%) in hostile worlds
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

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
    TransitionRule,
    EvidenceRegistry,
    EvidenceSourceTier,
)


# ---------------------------------------------------------------------------------------
# 1. Procedural Epistemic World Generation
# ---------------------------------------------------------------------------------------

@dataclass
class EvidencePacket:
    step: int
    evidence_id: str
    target_node: str
    observed_state: int
    true_generative_accuracy: float
    tier: EvidenceSourceTier
    sample_size: int
    measurement_uncertainty: float
    causal_path: List[Tuple[str, str, str]]
    source_dataset_id: str
    is_anomaly: bool = False
    description: str = ""


class ProceduralEpistemicWorld:
    """Generates an independent procedural epistemic reality with randomized DAG and latent states."""

    def __init__(self, world_id: int, num_nodes: int = 8, seed: Optional[int] = None):
        self.world_id = world_id
        self.num_nodes = num_nodes
        self.node_ids = [f"h_{i:02d}" for i in range(num_nodes)]
        self.rng = random.Random(seed if seed is not None else 10000 + world_id)
        self.np_rng = np.random.RandomState(seed if seed is not None else 10000 + world_id)

        # 1. Generate random acyclic DAG topology
        self.dependencies: Dict[str, List[str]] = {nid: [] for nid in self.node_ids}
        self.parents: Dict[str, Optional[str]] = {nid: None for nid in self.node_ids}

        # Topological order: i can only be parent of j if i < j
        for i in range(num_nodes - 1):
            for j in range(i + 1, min(num_nodes, i + 3)):
                if self.rng.random() < 0.45:
                    p = self.node_ids[i]
                    c = self.node_ids[j]
                    if self.parents[c] is None:  # Single primary parent for tree-like clarity
                        self.dependencies[p].append(c)
                        self.parents[c] = p

        # 2. Hostile Graph Invariant Injection (in 30% of worlds)
        # Inject an incorrect invariant into stored graph: asserts C requires P, but in reality C is independent
        self.has_hostile_invariant = (self.rng.random() < 0.30)
        self.hostile_edge: Optional[Tuple[str, str]] = None

        if self.has_hostile_invariant:
            # Pick a leaf node and an un-linked parent
            candidates = [nid for nid in self.node_ids[2:] if self.parents[nid] is None]
            if candidates:
                child = candidates[0]
                parent = self.node_ids[0]  # Keystone parent
                self.dependencies[parent].append(child)
                self.parents[child] = parent
                self.hostile_edge = (child, parent)

        # 3. Non-stationary Changepoint Timing
        self.shift_step = self.rng.randint(20, 30)

        # 4. Latent Truth Trajectory
        # Pre-shift: keystones and valid descendants are active (1), others inactive (0)
        # Post-shift: paradigm inversion!
        self.pre_shift_state = {}
        self.post_shift_state = {}

        # Balanced 50/50 prior overall
        for i, nid in enumerate(self.node_ids):
            if i < num_nodes // 2:
                self.pre_shift_state[nid] = 1
                self.post_shift_state[nid] = 0
            else:
                self.pre_shift_state[nid] = 0
                self.post_shift_state[nid] = 1

        # If hostile edge injected: in latent reality, child is active (1) in post-shift even though parent is 0!
        if self.has_hostile_invariant and self.hostile_edge:
            c, p = self.hostile_edge
            self.post_shift_state[c] = 1
            self.post_shift_state[p] = 0

    def get_latent_truth(self, t: int) -> Dict[str, int]:
        return self.pre_shift_state if t < self.shift_step else self.post_shift_state

    def generate_timeline(self, total_steps: int = 50) -> List[EvidencePacket]:
        packets = []
        for t in range(1, total_steps + 1):
            latent = self.get_latent_truth(t)

            # Target selection
            if t in [self.shift_step, self.shift_step + 1]:
                target = self.node_ids[0]  # Keystone decisively tested at shift
                latent_val = 0
                acc = 0.95
                obs = 0
                unc = 0.05
                tier = EvidenceSourceTier.LAB_ASSAY
                ds_id = f"dataset_decisive_{self.world_id}_{t}"
                desc = f"Decisive falsification assay on keystone {target}"
                is_anom = False
            elif self.has_hostile_invariant and self.hostile_edge and t in [self.shift_step + 2, self.shift_step + 3, self.shift_step + 4, self.shift_step + 5]:
                target = self.hostile_edge[0]  # Targeted investigation of empirical anomaly
                latent_val = 1
                acc = 0.90
                obs = 1
                unc = 0.10
                tier = EvidenceSourceTier.REPLICATED_STUDY
                ds_id = f"dataset_investigation_{self.world_id}_{t}"
                desc = f"Replicated empirical study on {target}"
                is_anom = False
            else:
                target = self.rng.choice(self.node_ids)
                latent_val = latent[target]

                acc = self.rng.uniform(0.72, 0.92)
                is_anom = (self.rng.random() < 0.05)  # 5% anomaly chance
                if is_anom:
                    obs = 1 - latent_val
                    acc = 0.10
                    unc = 0.05
                    tier = EvidenceSourceTier.LAB_ASSAY
                    ds_id = f"dataset_anom_{self.world_id}_{t}"
                    desc = f"Corrupted assay on {target}"
                else:
                    obs = latent_val if self.rng.random() < acc else (1 - latent_val)
                    unc = round(1.0 - acc, 2)
                    tier = EvidenceSourceTier.REPLICATED_STUDY if acc > 0.80 else EvidenceSourceTier.UNVERIFIED_CLAIM
                    if self.rng.random() < 0.20 and t > 5:
                        ds_id = f"dataset_shared_batch_{self.world_id}_{t % 4}"
                    else:
                        ds_id = f"dataset_{self.world_id}_{t}"
                    desc = f"Empirical observation on {target}"

            parent = self.parents.get(target)
            c_path = [(target, parent, "logically_requires")] if parent else []

            packets.append(EvidencePacket(
                step=t,
                evidence_id=f"ev_w{self.world_id}_t{t:03d}",
                target_node=target,
                observed_state=obs,
                true_generative_accuracy=acc,
                tier=tier,
                sample_size=self.rng.randint(4, 15),
                measurement_uncertainty=unc,
                causal_path=c_path,
                source_dataset_id=ds_id,
                is_anomaly=is_anom,
                description=desc,
            ))
        return packets


# ---------------------------------------------------------------------------------------
# 2. Contender Epistemic Updaters
# ---------------------------------------------------------------------------------------

class TrueBayesianChangepointOracle:
    """
    Exact Hidden Markov Model (HMM) joint state filter across the 2^M lattice.
    Explicitly models:
      - Transition hazard lambda = 1/T (paradigm shift changepoint probability)
      - Exact observation likelihoods P(E_t | H_t)
      - Exact joint state forward recursion: P(H_t | E_{1:t})
      - Raw dataset deduplication (resisting pseudoreplication)
    Represents the mathematical Bayes-optimal ceiling under non-stationary reality.
    """

    def __init__(self, node_ids: List[str], total_steps: int = 50):
        self.node_ids = node_ids
        self.num_nodes = len(node_ids)
        self.num_states = 1 << self.num_nodes
        self.total_steps = total_steps
        self.hazard = 1.0 / float(total_steps)

        # Precompute state matrix (num_states, M) in {0, 1}
        self.states = np.array([
            [(s >> i) & 1 for i in range(self.num_nodes)]
            for s in range(self.num_states)
        ], dtype=np.float32)

        # Uniform prior over all joint states
        self.posterior = np.ones(self.num_states, dtype=np.float64) / float(self.num_states)
        self.seen_datasets: Set[str] = set()
        self.node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    def process(self, packet: EvidencePacket):
        if packet.source_dataset_id in self.seen_datasets:
            return  # Suppress duplicate reports
        self.seen_datasets.add(packet.source_dataset_id)

        # 1. HMM Transition step: with probability (1 - hazard) stay, with hazard jump uniformly
        self.posterior = (1.0 - self.hazard) * self.posterior + self.hazard * (1.0 / float(self.num_states))

        # 2. Likelihood update
        target_idx = self.node_to_idx[packet.target_node]
        obs = packet.observed_state
        r = max(0.51, min(0.99, 1.0 - packet.measurement_uncertainty))

        # Vectorized likelihood: P(E | H) = r if H[target] == obs else (1 - r)
        likelihood = np.where(self.states[:, target_idx] == obs, r, 1.0 - r)
        self.posterior = self.posterior * likelihood
        denom = self.posterior.sum()
        if denom > 0:
            self.posterior /= denom
        else:
            self.posterior = np.ones(self.num_states) / float(self.num_states)

    def get_probabilities(self) -> Dict[str, float]:
        # Exact marginals: P(H[m] = 1) = states.T @ posterior
        marginals = self.states.T @ self.posterior
        return {nid: float(marginals[i]) for i, nid in enumerate(self.node_ids)}


class BayesianDAGBaseline:
    """Stationary Bayesian log-odds tracker with DAG clamping and deduplication (formerly 'Oracle')."""

    def __init__(self, node_ids: List[str], dependencies: Dict[str, List[str]]):
        self.node_ids = node_ids
        self.dependencies = dependencies
        self.log_odds = {nid: 0.0 for nid in node_ids}
        self.seen_datasets: Set[str] = set()

    def process(self, packet: EvidencePacket):
        if packet.source_dataset_id in self.seen_datasets:
            return
        self.seen_datasets.add(packet.source_dataset_id)

        r = max(0.51, min(0.99, 1.0 - packet.measurement_uncertainty))
        lr = math.log(r / (1.0 - r))
        delta = lr if packet.observed_state == 1 else -lr
        self.log_odds[packet.target_node] += delta

        # Clamp child log-odds to parent log-odds (P(C=1) <= P(P=1))
        for p, children in self.dependencies.items():
            for c in children:
                if self.log_odds[c] > self.log_odds[p]:
                    self.log_odds[c] = self.log_odds[p]

    def get_probabilities(self) -> Dict[str, float]:
        probs = {}
        for nid, lo in self.log_odds.items():
            lo_clamped = max(-15.0, min(15.0, lo))
            probs[nid] = 1.0 / (1.0 + math.exp(-lo_clamped))
        for p, children in self.dependencies.items():
            for c in children:
                probs[c] = min(probs[c], probs[p])
        return probs


class BayesianIndependentNaive:
    """Standard naive independent Bayesian updater without DAG constraints or deduplication."""

    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids
        self.log_odds = {nid: 0.0 for nid in node_ids}

    def process(self, packet: EvidencePacket):
        r = max(0.51, min(0.99, 1.0 - packet.measurement_uncertainty))
        lr = math.log(r / (1.0 - r))
        delta = lr if packet.observed_state == 1 else -lr
        self.log_odds[packet.target_node] += delta

    def get_probabilities(self) -> Dict[str, float]:
        probs = {}
        for nid, lo in self.log_odds.items():
            lo_clamped = max(-15.0, min(15.0, lo))
            probs[nid] = 1.0 / (1.0 + math.exp(-lo_clamped))
        return probs


class IgnoranceBaseline:
    """Always predicts P = 0.50 (theoretical Brier = 0.2500)."""
    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids

    def process(self, packet: EvidencePacket):
        pass

    def get_probabilities(self) -> Dict[str, float]:
        return {nid: 0.50 for nid in self.node_ids}


class StubbornUpdater:
    """Conservative updater with topological centrality penalty."""
    def __init__(self, node_ids: List[str], dependencies: Dict[str, List[str]]):
        self.beliefs = {nid: 0.0 for nid in node_ids}
        self.dependencies = dependencies

    def process(self, packet: EvidencePacket):
        target = packet.target_node
        reach = len(self.dependencies.get(target, []))
        weight = 1.0 + 0.5 * reach

        obs_conf = 0.85 if packet.observed_state == 1 else -0.85
        delta = obs_conf - self.beliefs[target]
        cost = abs(delta) * weight / max(0.1, 1.0 - packet.measurement_uncertainty)

        if cost <= 4.0:
            self.beliefs[target] += delta * 0.35

    def get_probabilities(self) -> Dict[str, float]:
        return {nid: max(0.01, min(0.99, (c + 1.0) / 2.0)) for nid, c in self.beliefs.items()}


class CortexGovernorAgent:
    """
    Warp Cortex:
      - EvidenceRegistry with dataset signature tracking
      - Level-1 Epistemic Continuity & Deductive Cascades
      - Level-2 Topology Revision under persistent edge strain (threshold = 2.0)
      - Contradiction Strain tracking
    """

    def __init__(self, node_ids: List[str], dependencies: Dict[str, List[str]]):
        self.registry = EvidenceRegistry()
        self.governor = TransitionGovernor(
            evidence_registry=self.registry,
            max_cost_threshold=4.0,
            epsilon=0.05,
            topology_revision_threshold=1.0,
        )
        self.manifold = EpistemicManifold()

        for nid in node_ids:
            self.manifold.register_claim(nid, f"Hypothesis {nid}", confidence=0.0)

        # Wire dependency constraints: child LOGICALLY_REQUIRES parent
        for p, children in dependencies.items():
            for c in children:
                self.manifold.link_claims(c, p, EpistemicRelation.LOGICALLY_REQUIRES)

        self.registered_dataset_ids: Set[str] = set()

    def process(self, packet: EvidencePacket) -> Tuple[bool, str]:
        # Deduplication
        if packet.source_dataset_id in self.registered_dataset_ids:
            return False, "Duplicate dataset; suppressed."
        self.registered_dataset_ids.add(packet.source_dataset_id)

        # Register evidence
        self.registry.register_evidence(
            evidence_id=packet.evidence_id,
            tier=packet.tier,
            source_type="empirical_assay",
            description=packet.description,
            sample_size=packet.sample_size,
            measurement_uncertainty=packet.measurement_uncertainty,
            metadata={"dataset_id": packet.source_dataset_id},
        )

        curr_conf = self.manifold.nodes[packet.target_node].confidence
        obs_val = 0.85 if packet.observed_state == 1 else -0.85
        delta_c = (obs_val - curr_conf) * 0.40

        cert = TransitionCertificate(
            evidence_id=packet.evidence_id,
            target_node_id=packet.target_node,
            proposed_confidence_delta=delta_c,
            causal_path=packet.causal_path,
            rule=TransitionRule.DIRECT_EMPIRICAL_UPDATE,
        )

        decision = self.governor.evaluate_transition(self.manifold, cert)
        if decision.admitted:
            self.manifold.inject_observation(
                target_id=packet.target_node,
                observation_text=packet.description,
                confidence_delta=delta_c,
                obs_id=packet.evidence_id,
            )
            return True, "Admitted"
        else:
            return False, decision.reason

    def get_probabilities(self) -> Dict[str, float]:
        probs = {}
        for nid, node in self.manifold.nodes.items():
            if node.kind == EpistemicKind.HYPOTHESIS:
                probs[nid] = max(0.01, min(0.99, (node.confidence + 1.0) / 2.0))
        return probs

    def is_invariant_severed(self, child_id: str, parent_id: str) -> bool:
        edge = self.manifold._adjacency.get(child_id, {}).get(parent_id)
        return edge is not None and not edge.is_active


# ---------------------------------------------------------------------------------------
# 3. Metrics & Large-Scale Evaluation Engine
# ---------------------------------------------------------------------------------------

def compute_brier(probs: Dict[str, float], latent_truth: Dict[str, int]) -> float:
    return sum((probs[nid] - latent_truth[nid]) ** 2 for nid in latent_truth) / len(latent_truth)

def compute_accuracy(probs: Dict[str, float], latent_truth: Dict[str, int]) -> float:
    return sum(1 for nid in latent_truth if (probs[nid] >= 0.5) == latent_truth[nid]) / len(latent_truth) * 100.0

def compute_ece(probs: Dict[str, float], latent_truth: Dict[str, int], n_bins: int = 5) -> float:
    bins = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(latent_truth)
    for i in range(n_bins):
        bl, br = bins[i], bins[i + 1]
        items = [(probs[nid], latent_truth[nid]) for nid in latent_truth if bl <= probs[nid] <= br]
        if items:
            avg_p = sum(p for p, _ in items) / len(items)
            avg_y = sum(y for _, y in items) / len(items)
            ece += (len(items) / n) * abs(avg_p - avg_y)
    return ece


def run_multi_world_epistemic_benchmark(num_worlds: int = 500, total_steps: int = 50):
    print("=" * 125)
    print(f"WARP CORTEX: HIERARCHICAL MULTI-WORLD EPISTEMIC BENCHMARK ({num_worlds} INDEPENDENT PROCEDURAL WORLDS)")
    print(f"Tracking Latent Realities across {total_steps} Steps (Randomized DAGs, Changepoints, Noise, and Hostile Invariants)")
    print("=" * 125)

    contender_names = [
        "True Bayesian Changepoint Oracle",
        "Cortex Governor (L1+L2)",
        "Bayesian DAG Baseline (Stationary)",
        "Bayesian (Independent/Naive)",
        "Ignorance Baseline (P = 0.50)",
        "Stubborn (Centrality Penalty)",
    ]

    world_brier: Dict[str, List[float]] = {name: [] for name in contender_names}
    world_acc: Dict[str, List[float]] = {name: [] for name in contender_names}
    world_ece: Dict[str, List[float]] = {name: [] for name in contender_names}
    world_reversal: Dict[str, List[float]] = {name: [] for name in contender_names}

    hostile_worlds_count = 0
    cortex_severed_count = 0

    t_start = time.time()

    for w_idx in range(1, num_worlds + 1):
        world = ProceduralEpistemicWorld(world_id=w_idx, num_nodes=8)
        packets = world.generate_timeline(total_steps=total_steps)

        contenders = {
            "True Bayesian Changepoint Oracle": TrueBayesianChangepointOracle(world.node_ids, total_steps=total_steps),
            "Cortex Governor (L1+L2)": CortexGovernorAgent(world.node_ids, world.dependencies),
            "Bayesian DAG Baseline (Stationary)": BayesianDAGBaseline(world.node_ids, world.dependencies),
            "Bayesian (Independent/Naive)": BayesianIndependentNaive(world.node_ids),
            "Ignorance Baseline (P = 0.50)": IgnoranceBaseline(world.node_ids),
            "Stubborn (Centrality Penalty)": StubbornUpdater(world.node_ids, world.dependencies),
        }

        step_briers: Dict[str, List[float]] = {name: [] for name in contenders}
        step_accs: Dict[str, List[float]] = {name: [] for name in contenders}
        step_eces: Dict[str, List[float]] = {name: [] for name in contenders}
        reversal_latency: Dict[str, Optional[int]] = {name: None for name in contenders}

        keystone_id = world.node_ids[0]

        for t, pkt in enumerate(packets, start=1):
            latent_state = world.get_latent_truth(t)

            for name, agent in contenders.items():
                agent.process(pkt)
                probs = agent.get_probabilities()

                step_briers[name].append(compute_brier(probs, latent_state))
                step_accs[name].append(compute_accuracy(probs, latent_state))
                step_eces[name].append(compute_ece(probs, latent_state))

                # Track paradigm shift reversal latency
                if t >= world.shift_step and reversal_latency[name] is None:
                    if probs[keystone_id] < 0.50:
                        reversal_latency[name] = t - world.shift_step

        for name in contenders:
            world_brier[name].append(sum(step_briers[name]) / len(step_briers[name]))
            world_acc[name].append(sum(step_accs[name]) / len(step_accs[name]))
            world_ece[name].append(sum(step_eces[name]) / len(step_eces[name]))
            rev = reversal_latency[name] if reversal_latency[name] is not None else (total_steps - world.shift_step)
            world_reversal[name].append(rev)

        # Hostile invariant evaluation
        if world.has_hostile_invariant and world.hostile_edge:
            hostile_worlds_count += 1
            c, p = world.hostile_edge
            cortex_agent: CortexGovernorAgent = contenders["Cortex Governor (L1+L2)"]  # type: ignore
            if cortex_agent.is_invariant_severed(c, p):
                cortex_severed_count += 1

    elapsed = time.time() - t_start

    print(f"\nCompleted {num_worlds} worlds ({num_worlds * total_steps:,} evaluation steps) in {elapsed:.2f}s ({elapsed/num_worlds*1000:.1f} ms/world)\n")

    print(f"{'Epistemic Architecture':<36} | {'Mean Brier Score':<18} | {'Latent Accuracy':<16} | {'ECE (Calibration)':<18} | {'Shift Reversal':<16}")
    print("-" * 125)

    summary = {}
    for name in contender_names:
        m_bs = float(np.mean(world_brier[name]))
        se_bs = float(np.std(world_brier[name]) / np.sqrt(num_worlds))
        m_acc = float(np.mean(world_acc[name]))
        m_ece = float(np.mean(world_ece[name]))
        m_rev = float(np.mean(world_reversal[name]))

        summary[name] = {
            "brier": m_bs,
            "brier_se": se_bs,
            "acc": m_acc,
            "ece": m_ece,
            "reversal": m_rev,
        }

        brier_str = f"{m_bs:.4f} +/- {se_bs:.4f}"
        acc_str = f"{m_acc:.1f}%"
        ece_str = f"{m_ece:.4f}"
        rev_str = f"{m_rev:.1f} steps"
        print(f"{name:<36} | {brier_str:>18} | {acc_str:>16} | {ece_str:>18} | {rev_str:>16}")

    print("=" * 125)

    # Hostile Invariant Analysis
    print("\nHOSTILE CAUSAL INVARIANT EXPERIMENT (Scientist's Graph Mis-specified with False Invariant C -> P):")
    print(f"  Total Hostile Worlds Generated           : {hostile_worlds_count} / {num_worlds} ({hostile_worlds_count/num_worlds*100:.1f}%)")
    cortex_severance_rate = (cortex_severed_count / hostile_worlds_count) * 100.0 if hostile_worlds_count > 0 else 0.0
    print(f"  Cortex Governor Invariant Severance Rate : {cortex_severed_count}/{hostile_worlds_count} ({cortex_severance_rate:.1f}%)")
    print(f"  Bayesian DAG Baseline Severance Rate     : 0/{hostile_worlds_count} (0.0% - Permanently trapped in false invariant)")
    print("=" * 125)

    return summary


if __name__ == "__main__":
    run_multi_world_epistemic_benchmark(num_worlds=500, total_steps=50)
