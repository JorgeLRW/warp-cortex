"""
Validation Test Suite for Continuous Epistemic Manifold.

Simulates a scientific research project:
- Foundational Axiom -> Keystone Hypothesis -> Downstream Applications.
- Tests Tarjan bridge detection for keystone identification.
- Tests empirical cascade collapse when a parent hypothesis is refuted.
- Tests epistemic contradiction strain and active research frontier detection.
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
)


class TestContinuousEpistemicManifold(unittest.TestCase):
    def setUp(self):
        self.manifold = EpistemicManifold(hidden_dim=64)

        # 1. Register project nodes
        self.axiom = self.manifold.register_claim(
            node_id="axiom_thermo",
            statement="Thermodynamics allows spontaneous catalytic conversion at room temp.",
            kind=EpistemicKind.AXIOM,
            confidence=1.0,
        )

        self.keystone = self.manifold.register_claim(
            node_id="hypo_catalyst_x",
            statement="Catalyst X lowers activation energy by 40% via dual-site binding.",
            kind=EpistemicKind.HYPOTHESIS,
            confidence=0.80,
        )

        self.sub_yield = self.manifold.register_claim(
            node_id="hypo_yield_90",
            statement="Reaction achieves > 90% purity yield in under 30 minutes.",
            kind=EpistemicKind.HYPOTHESIS,
            confidence=0.75,
        )

        self.sub_scale = self.manifold.register_claim(
            node_id="hypo_industrial_scale",
            statement="Process scales economically to a 1000L continuous flow reactor.",
            kind=EpistemicKind.HYPOTHESIS,
            confidence=0.65,
        )

        self.competing = self.manifold.register_claim(
            node_id="hypo_enzyme_alt",
            statement="Enzyme Y provides an ambient biological pathway replacing synthetic Catalyst X.",
            kind=EpistemicKind.HYPOTHESIS,
            confidence=0.50,
        )

        # 2. Link topological constraints
        # Keystone depends on axiom
        self.manifold.link_claims("hypo_catalyst_x", "axiom_thermo", EpistemicRelation.DEPENDS_ON)
        # Downstream yields depend on keystone
        self.manifold.link_claims("hypo_yield_90", "hypo_catalyst_x", EpistemicRelation.DEPENDS_ON)
        # Scaling depends on yield
        self.manifold.link_claims("hypo_industrial_scale", "hypo_yield_90", EpistemicRelation.DEPENDS_ON)
        # Competing alternative refutes/blocks keystone
        self.manifold.link_claims("hypo_enzyme_alt", "hypo_catalyst_x", EpistemicRelation.REFUTES, weight=1.0)

    def test_keystone_articulation_point(self):
        """Verify that Tarjan bridge analysis identifies Catalyst X as the structural keystone."""
        keystones = self.manifold.find_keystone_hypotheses()
        print(f"\n[Keystone Detection] Identified keystone hypotheses: {keystones}")
        self.assertIn("hypo_catalyst_x", keystones, "Catalyst X must be recognized as keystone articulation point")

    def test_empirical_cascade_collapse(self):
        """
        Inject refuting empirical observation into Keystone.
        Assert that dependent downstream hypotheses (yield, scale) automatically collapse,
        while foundational axiom remains unaffected.
        """
        print("\n--- Injecting Empirical Refutation into Keystone ---")
        res = self.manifold.inject_observation(
            target_id="hypo_catalyst_x",
            observation_text="Spectrometry Run #42: Catalyst X degrades irreversibly within 3 minutes; yield = 1.8%.",
            confidence_delta=-1.60,
        )

        # 1. Keystone should be falsified
        keystone_node = self.manifold.nodes["hypo_catalyst_x"]
        print(f"  Keystone new confidence: {keystone_node.confidence:.2f} (falsified: {keystone_node.is_falsified()})")
        self.assertTrue(keystone_node.is_falsified(), "Catalyst X should be falsified")
        self.assertLessEqual(keystone_node.confidence, -0.5)

        # 2. Downstream dependent hypotheses must automatically collapse
        yield_node = self.manifold.nodes["hypo_yield_90"]
        scale_node = self.manifold.nodes["hypo_industrial_scale"]
        print(f"  Downstream Yield confidence: {yield_node.confidence:.2f}")
        print(f"  Downstream Scale confidence: {scale_node.confidence:.2f}")
        self.assertLessEqual(yield_node.confidence, keystone_node.confidence)
        self.assertLessEqual(scale_node.confidence, yield_node.confidence)

        # 3. Axiom must remain intact
        axiom_node = self.manifold.nodes["axiom_thermo"]
        print(f"  Foundational Axiom confidence: {axiom_node.confidence:.2f} (intact)")
        self.assertEqual(axiom_node.confidence, 1.0, "Foundational axiom must not be corrupted by downward child failure")

    def test_active_research_frontier(self):
        """Verify that the manifold highlights unresolved claims and strain as top research frontier."""
        frontier = self.manifold.get_active_frontier(top_k=3)
        print("\n--- Active Research Frontier ---")
        for item in frontier:
            print(f"  [{item['kind'].upper()}] {item['node_id']}: priority={item['priority_score']:.2f}, conf={item['confidence']:.2f}, strain={item['strain']:.2f}")

        # Top frontier should surface claims needing experimental resolution
        top_ids = [item["node_id"] for item in frontier]
        self.assertIn("hypo_enzyme_alt", top_ids, "Alternative hypothesis with open tension must be in active frontier")

    def test_contradiction_energy_mutually_rejected(self):
        """Verify that two mutually refuting claims both rejected (C_i < 0, C_j < 0) produce 0 contradiction energy."""
        m = EpistemicManifold(hidden_dim=32)
        m.register_claim("theory_a", "Phlogiston theory", confidence=-0.9)
        m.register_claim("theory_b", "Caloric theory", confidence=-0.9)
        m.link_claims("theory_a", "theory_b", EpistemicRelation.REFUTES, weight=1.0)

        strain_res = m.calculate_contradiction_energy()
        print(f"\n[Contradiction Energy Test] Both rejected theories energy: {strain_res['contradiction_energy']}")
        self.assertEqual(strain_res["contradiction_energy"], 0.0, "Two rejected competing claims must have zero contradiction energy!")

    def test_bifurcated_justification_vs_logical_requirement(self):
        """
        Verify that:
        1. LOGICALLY_REQUIRES forces child to falsified when parent fails.
        2. EVIDENCE_DEPENDS_ON resets child to UNSUPPORTED (confidence 0.0), NOT falsified!
        """
        m = EpistemicManifold(hidden_dim=32)
        
        # Branch 1: Deductive requirement
        m.register_claim("pre_req", "Catalyst stable at pH 3", confidence=0.8)
        m.register_claim("deductive_child", "Continuous flow reactor operates at pH 3", confidence=0.8)
        m.link_claims("deductive_child", "pre_req", EpistemicRelation.LOGICALLY_REQUIRES)

        # Branch 2: Justificatory evidence
        m.register_claim("evidence_x", "Spectrometry run 4 indicates room-temp superconductivity", confidence=0.9)
        m.register_claim("inductive_child", "Material is room-temp superconductor", confidence=0.7)
        m.link_claims("inductive_child", "evidence_x", EpistemicRelation.EVIDENCE_DEPENDS_ON)

        # Invalidate both parents
        m.inject_observation("pre_req", "Catalyst disintegrates at pH 3", confidence_delta=-1.6)
        m.inject_observation("evidence_x", "Run 4 equipment was uncalibrated", confidence_delta=-1.8)

        deductive_node = m.nodes["deductive_child"]
        inductive_node = m.nodes["inductive_child"]

        print(f"\n[Bifurcation Test] Deductive child: status={deductive_node.status.value}, conf={deductive_node.confidence:.2f}")
        print(f"[Bifurcation Test] Inductive child: status={inductive_node.status.value}, conf={inductive_node.confidence:.2f}")

        # Deductive child must be falsified
        self.assertTrue(deductive_node.is_falsified())
        self.assertLessEqual(deductive_node.confidence, -0.5)

        # Inductive child must NOT be falsified! It must be UNSUPPORTED with confidence 0.0
        self.assertTrue(inductive_node.is_unsupported())
        self.assertEqual(inductive_node.confidence, 0.0)
        self.assertFalse(inductive_node.is_falsified(), "Inductive child must not be falsified when evidence is invalidated!")


if __name__ == "__main__":
    unittest.main()

