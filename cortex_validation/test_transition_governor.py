"""
Unit Test Suite for Proof-Carrying State Transition Governor (Independent Provenance & Topology Weighting).

Tests:
1. Rejection of transitions citing unregistered or fabricated evidence IDs.
2. Rejection of illegal transitions violating hard invariants (admissible region Omega).
3. Rejection of ungrounded radical leaps (high displacement on weak evidence tier).
4. Admission of radical transitions when backed by decisive empirical evidence (LAB_ASSAY).
5. Topology-weighted displacement: keystone node with downstream dependents costs more than a leaf node.
6. Rejection of transitions citing non-existent causal paths in the graph.
7. End-to-end commit pipeline updating the persistent epistemic state.
"""

import os
import sys
import unittest

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
    TransitionWitness,
    TransitionDecision,
    EvidenceRegistry,
    EvidenceSourceTier,
)


class TestTransitionGovernor(unittest.TestCase):
    def setUp(self):
        self.manifold = EpistemicManifold(hidden_dim=32)
        self.registry = EvidenceRegistry()
        self.governor = TransitionGovernor(
            evidence_registry=self.registry,
            max_cost_threshold=4.0,
            epsilon=0.05,
        )

        # Register standard evidence records
        self.registry.register_evidence(
            evidence_id="assay_spectrometry_001",
            tier=EvidenceSourceTier.LAB_ASSAY,
            source_type="spectrometry",
            description="NMR Spectrometry binding assay Kd = 1.2 nM",
            custom_reliability=0.95,
        )
        self.registry.register_evidence(
            evidence_id="claim_tweet_002",
            tier=EvidenceSourceTier.UNVERIFIED_CLAIM,
            source_type="social_media",
            description="Tweet asserting unverified result",
            custom_reliability=0.10,
        )
        self.registry.register_evidence(
            evidence_id="study_nature_003",
            tier=EvidenceSourceTier.REPLICATED_STUDY,
            source_type="peer_reviewed_paper",
            description="Independent multicenter trial",
            custom_reliability=0.85,
        )

        # Build test graph: Parent -> Child -> Leaf
        self.parent = self.manifold.register_claim(
            node_id="parent_catalyst",
            statement="Catalyst remains stable at pH 3.5",
            confidence=0.80,
        )
        self.child = self.manifold.register_claim(
            node_id="child_continuous_flow",
            statement="Continuous flow reactor operates at pH 3.5",
            confidence=0.75,
        )
        self.leaf = self.manifold.register_claim(
            node_id="leaf_packaging",
            statement="Final product packaged in glass vials",
            confidence=0.50,
        )
        self.manifold.link_claims("child_continuous_flow", "parent_catalyst", EpistemicRelation.LOGICALLY_REQUIRES)
        self.manifold.link_claims("leaf_packaging", "child_continuous_flow", EpistemicRelation.LOGICALLY_REQUIRES)

        # Falsify the parent
        self.parent.confidence = -0.90
        self.parent.status = EpistemicStatus.FALSIFIED

    def test_unregistered_evidence_rejected(self):
        """Proposals citing fabricated or unregistered evidence IDs are rejected immediately."""
        witness = TransitionWitness(
            evidence_id="fabricated_id_999",
            target_node_id="leaf_packaging",
            proposed_confidence_delta=0.20,
            rationale="Cites fake evidence ID.",
        )
        decision = self.governor.evaluate_transition(self.manifold, witness)
        print(f"\n[Governor Test 1] Unregistered Evidence Rejection: Admitted={decision.admitted}, Reason={decision.reason}")
        self.assertFalse(decision.admitted, "Fabricated evidence IDs must cause rejection")

    def test_invariant_violation_rejected(self):
        """Child cannot be increased or set to confirmed when strict prerequisite is falsified."""
        witness = TransitionWitness(
            evidence_id="assay_spectrometry_001",
            target_node_id="child_continuous_flow",
            proposed_confidence_delta=0.20,
            rationale="Model argues the reactor will work anyway without the catalyst.",
        )
        decision = self.governor.evaluate_transition(self.manifold, witness)
        print(f"[Governor Test 2] Invariant Rejection: Admitted={decision.admitted}, Reason={decision.reason}")
        self.assertFalse(decision.admitted, "Transition violating LOGICALLY_REQUIRES must be rejected")
        self.assertTrue(len(decision.violated_invariants) > 0)

    def test_unwarranted_radical_leap_rejected(self):
        """Weak unverified evidence cannot justify a massive epistemic displacement."""
        self.manifold.register_claim("hypo_fusion", "Cold fusion achieved in kitchen", confidence=0.0)

        # Weak tweet evidence (reliability 0.10) proposing massive confidence shift (0.85)
        witness = TransitionWitness(
            evidence_id="claim_tweet_002",
            target_node_id="hypo_fusion",
            proposed_confidence_delta=0.85,
            rationale="Tweet claimed fusion worked.",
        )
        decision = self.governor.evaluate_transition(self.manifold, witness)
        print(f"[Governor Test 3] Radical Leap Rejection: Admitted={decision.admitted}, Cost={decision.transition_cost}")
        self.assertFalse(decision.admitted, "Ungrounded radical leap must be rejected")
        self.assertGreater(decision.transition_cost, 4.0)

    def test_decisive_radical_leap_admitted(self):
        """Decisive empirical evidence (reliability 0.95) can cause large valid displacement."""
        self.manifold.register_claim("hypo_assay", "Compound binds target receptor", confidence=0.0)

        witness = TransitionWitness(
            evidence_id="assay_spectrometry_001",
            target_node_id="hypo_assay",
            proposed_confidence_delta=0.85,
            rationale="Spectrometry confirmed binding with Kd = 1.2 nM.",
        )
        decision = self.governor.evaluate_transition(self.manifold, witness)
        print(f"[Governor Test 4] Decisive Update: Admitted={decision.admitted}, Cost={decision.transition_cost}")
        self.assertTrue(decision.admitted, "Radical update with decisive evidence must be admitted")
        self.assertLessEqual(decision.transition_cost, 5.0)

    def test_decoupled_truth_gate_and_blast_radius(self):
        """
        Truth Gate != Blast Radius Gate:
        Modifying a keystone has identical epistemic transition cost as a leaf node
        given identical evidence, but accurately triggers a larger blast radius.
        """
        # Node with downstream dependents (parent_catalyst governs child)
        # Reset parent for test
        self.parent.confidence = 0.50
        self.parent.status = EpistemicStatus.UNVERIFIED

        witness_keystone = TransitionWitness(
            evidence_id="study_nature_003",
            target_node_id="parent_catalyst",
            proposed_confidence_delta=-0.80, # flipping keystone
        )
        witness_leaf = TransitionWitness(
            evidence_id="study_nature_003",
            target_node_id="leaf_packaging",
            proposed_confidence_delta=-0.80, # identical delta on leaf
        )

        dec_keystone = self.governor.evaluate_transition(self.manifold, witness_keystone)
        dec_leaf = self.governor.evaluate_transition(self.manifold, witness_leaf)

        print(f"[Governor Test 5] Decoupled Truth vs Blast Radius: Keystone Cost={dec_keystone.transition_cost:.2f}, Leaf Cost={dec_leaf.transition_cost:.2f}, Keystone Blast={dec_keystone.blast_radius}, Leaf Blast={dec_leaf.blast_radius}")
        self.assertAlmostEqual(
            dec_keystone.transition_cost,
            dec_leaf.transition_cost,
            places=4,
            msg="Truth gate cost must NOT penalize keystones: identical delta with identical evidence must have identical epistemic cost",
        )
        self.assertGreater(
            dec_keystone.blast_radius,
            dec_leaf.blast_radius,
            "Keystone blast radius must exceed leaf blast radius due to topological downstream reach",
        )

    def test_fake_causal_path_rejected(self):
        """Witness declaring non-existent causal links is rejected."""
        self.manifold.register_claim("hypo_unrelated", "Unrelated claim", confidence=0.2)
        witness = TransitionWitness(
            evidence_id="study_nature_003",
            target_node_id="hypo_unrelated",
            proposed_confidence_delta=0.30,
            causal_path=[("hypo_unrelated", "non_existent_node", "logically_requires")],
            rationale="Fabricated step in reasoning.",
        )
        decision = self.governor.evaluate_transition(self.manifold, witness)
        print(f"[Governor Test 6] Fake Path Rejection: Admitted={decision.admitted}, Reason={decision.reason}")
        self.assertFalse(decision.admitted, "Non-existent causal edges in warrant must cause rejection")

    def test_commit_pipeline(self):
        """Admitted witness commits state changes to the persistent manifold."""
        self.manifold.register_claim("hypo_test", "Valid test hypothesis", confidence=0.10)
        witness = TransitionWitness(
            evidence_id="assay_spectrometry_001",
            target_node_id="hypo_test",
            proposed_confidence_delta=0.65,
            rationale="Rigorous validation confirmed hypothesis.",
        )
        committed, decision = self.governor.commit_if_admitted(self.manifold, witness)
        self.assertTrue(committed)
        self.assertEqual(self.manifold.nodes["hypo_test"].confidence, 0.75)
        self.assertEqual(self.manifold.nodes["hypo_test"].status, EpistemicStatus.CONFIRMED)

    def test_level2_topology_revision_under_persistent_strain(self):
        """
        Level 2 Topology Revision:
        When authenticated empirical evidence repeatedly contradicts a structural invariant,
        persistent edge strain suspends the invariant edge rather than dogmatically rejecting reality.
        """
        # Parent catalyst is falsified
        self.manifold.nodes["parent_catalyst"].confidence = -0.90
        self.manifold.nodes["parent_catalyst"].status = EpistemicStatus.FALSIFIED

        # Register repeated authoritative assays for child_continuous_flow
        for i in range(1, 4):
            self.registry.register_evidence(
                evidence_id=f"authoritative_assay_{i}",
                tier=EvidenceSourceTier.LAB_ASSAY,
                source_type="spectrometry",
                description=f"Empirical assay {i} proving child works without parent.",
                sample_size=20,
                measurement_uncertainty=0.10,
                custom_reliability=0.90,
            )

        edge = self.manifold._adjacency["child_continuous_flow"]["parent_catalyst"]
        self.assertTrue(edge.is_active)
        self.assertEqual(edge.edge_strain, 0.0)

        # Observation 1: Strain = 0.90 < 2.0 -> Rejected by invariant
        w1 = TransitionWitness(
            evidence_id="authoritative_assay_1",
            target_node_id="child_continuous_flow",
            proposed_confidence_delta=0.40,
            rationale="Assay 1",
        )
        d1 = self.governor.evaluate_transition(self.manifold, w1)
        self.assertFalse(d1.admitted)
        self.assertTrue(edge.is_active)
        self.assertGreater(edge.edge_strain, 1.0)

        # Observation 2: Strain >= 2.0 -> Level 2 Topology Revision: Invariant suspended!
        w2 = TransitionWitness(
            evidence_id="authoritative_assay_2",
            target_node_id="child_continuous_flow",
            proposed_confidence_delta=0.40,
            rationale="Assay 2",
        )
        d2 = self.governor.evaluate_transition(self.manifold, w2)
        self.assertTrue(d2.admitted, "Second authoritative assay must trigger topology revision and be admitted!")
        self.assertFalse(edge.is_active, "Invariant edge must be suspended under persistent strain!")
        print(f"[Governor Test 7] Topology Revision: Admitted={d2.admitted}, Invariant Active={edge.is_active}, Accumulated Strain={edge.edge_strain:.2f}")


if __name__ == "__main__":
    unittest.main()

