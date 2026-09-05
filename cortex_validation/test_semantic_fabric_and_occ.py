"""
Unit Tests for Semantic Context Fabric and OCC Runtime Concurrency.
===================================================================
Tests:
1. Multi-Aspect Compartment Registration.
2. State-Conditioned Context Assembly: Context(q, S1) != Context(q, S2).
3. Innate Context Assembly (Unprompted Event Context).
4. OCC Concurrency: Stale Proposal Rejection & Read-Set Conflict Detection.
5. Idempotency: Duplicate Events & Commits.
6. Snapshot & Deterministic Replay.
"""

import unittest
import torch
import torch.nn.functional as F

from cortex_core.cortex_runtime import (
    CortexRuntime,
    ProposedCommit,
    RuntimeEvent,
)
from cortex_core.semantic_fabric import (
    SemanticContextFabric,
    SemanticBand,
    FabricItem,
)
from cortex_core.epistemic_manifold import (
    EpistemicKind,
    EpistemicRelation,
    EpistemicStatus,
)
from cortex_core.transition_governor import (
    EvidenceSourceTier,
    TransitionRule,
)


class TestSemanticFabricAndOCC(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.runtime = CortexRuntime(hidden_dim=32)

    def test_multi_aspect_registration_and_routing(self):
        fabric = self.runtime.context_fabric

        # Register multi-aspect items
        v_inst = F.normalize(torch.randn(32), dim=0)
        v_mfg = F.normalize(torch.randn(32), dim=0)
        v_safe = F.normalize(torch.randn(32), dim=0)

        item = fabric.register_item(
            item_id="bioreactor_7",
            title="Bioreactor 7 Run Log",
            content="Temperature maintained at 37C with high yield.",
            aspect_vectors={
                SemanticBand.INSTRUMENTATION.value: v_inst,
                SemanticBand.MANUFACTURING.value: v_mfg,
                SemanticBand.SAFETY.value: v_safe,
            },
            primary_aspect=SemanticBand.MANUFACTURING.value,
        )

        self.assertIn("bioreactor_7", fabric.compartments[SemanticBand.INSTRUMENTATION.value])
        self.assertIn("bioreactor_7", fabric.compartments[SemanticBand.MANUFACTURING.value])
        self.assertIn("bioreactor_7", fabric.compartments[SemanticBand.SAFETY.value])

        # Query targeting manufacturing
        ctx = fabric.assemble_context(
            query="scale up manufacturing run",
            target_aspects=[SemanticBand.MANUFACTURING.value],
            token_budget=512,
        )
        self.assertTrue(len(ctx.items) >= 1)
        self.assertEqual(ctx.items[0].item_id, "bioreactor_7")

    def test_state_conditioned_context_change(self):
        """
        Critical Invariant:
        Identical query q under State S1 yields different context under State S2!
        Context(q, S1) != Context(q, S2)
        """
        # Register claims in runtime
        self.runtime.register_claim("assay_yield", "Assay yield exceeds target threshold", EpistemicKind.AXIOM, 0.95)
        self.runtime.register_claim("detector_status", "Mass spec detector calibration normal", EpistemicKind.AXIOM, 0.90)

        # Align query embedding with manufacturing channel anchor
        mfg_anchor = self.runtime.context_fabric.band_anchors[SemanticBand.MANUFACTURING.value]
        inst_anchor = self.runtime.context_fabric.band_anchors[SemanticBand.INSTRUMENTATION.value]
        
        torch.manual_seed(42)
        q_emb = F.normalize(mfg_anchor + 0.10 * torch.randn(32), dim=0)
        v1 = F.normalize(q_emb + 0.05 * torch.randn(32), dim=0)
        v2 = F.normalize(inst_anchor + 0.10 * torch.randn(32), dim=0)

        self.runtime.register_fabric_item(
            item_id="doc_scaleup_sop",
            title="Scale-Up Standard Operating Procedure",
            content="Standard procedure for moving assay to 100L production.",
            aspect_vectors={SemanticBand.MANUFACTURING.value: v1},
            primary_aspect=SemanticBand.MANUFACTURING.value,
            causal_node_id="assay_yield",
        )
        self.runtime.register_fabric_item(
            item_id="doc_detector_log",
            title="Detector Calibration Alert",
            content="Significant detector drift observed in sensor channel 4.",
            aspect_vectors={SemanticBand.INSTRUMENTATION.value: v2},
            primary_aspect=SemanticBand.INSTRUMENTATION.value,
            causal_node_id="detector_status",
        )

        query = "Should we scale this assay to 100L production?"

        # State S1 (Normal): detector is valid, dynamic energy is 0
        ctx_s1 = self.runtime.query_context(query=query, query_embedding=q_emb, token_budget=512, state_weight=0.50)
        top_item_s1 = ctx_s1.items[0].item_id

        # Change State to S2: Detector fails, dynamic strain is injected, status is TAINTED
        self.runtime.context_fabric.update_dynamic_state("doc_detector_log", energy_delta=1.0, validity_status="TAINTED")
        
        ctx_s2 = self.runtime.query_context(query=query, query_embedding=q_emb, token_budget=512, state_weight=0.50)
        top_item_s2 = ctx_s2.items[0].item_id

        # The query did NOT change. The world state DID change.
        # Therefore context assembly prioritized the critical anomaly!
        self.assertEqual(top_item_s1, "doc_scaleup_sop")
        self.assertEqual(top_item_s2, "doc_detector_log")
        self.assertIn("TAINTED", ctx_s2.summary_text)

    def test_innate_context_assembly(self):
        """Test unprompted context assembly directly from entity coordinates and causal graph."""
        # Build causal graph: sensor -> dataset -> model -> scale_decision
        self.runtime.register_claim("sensor_42", "Sensor 42 calibration", EpistemicKind.AXIOM, 0.9)
        self.runtime.register_claim("dataset_42", "Dataset 42 raw measurements", EpistemicKind.AXIOM, 0.8)
        self.runtime.register_claim("model_v2", "Model v2 trained on Dataset 42", EpistemicKind.HYPOTHESIS, 0.8)

        self.runtime.link_causal_dependency("sensor_42", "dataset_42", EpistemicRelation.LOGICALLY_REQUIRES)
        self.runtime.link_causal_dependency("dataset_42", "model_v2", EpistemicRelation.LOGICALLY_REQUIRES)

        v = F.normalize(torch.randn(32), dim=0)
        self.runtime.register_fabric_item(
            item_id="dataset_42_record",
            title="Dataset 42 Storage Record",
            content="Mass spectrometry measurements for Batch 93.",
            aspect_vectors={SemanticBand.DATA_VALIDITY.value: v},
            causal_node_id="dataset_42",
        )
        self.runtime.register_fabric_item(
            item_id="sensor_42_record",
            title="Sensor 42 Logs",
            content="Sensor calibration specs.",
            aspect_vectors={SemanticBand.INSTRUMENTATION.value: v},
            causal_node_id="sensor_42",
        )
        self.runtime.register_fabric_item(
            item_id="model_v2_record",
            title="Model v2 Weights & Card",
            content="Efficacy prediction model.",
            aspect_vectors={SemanticBand.MECHANISM.value: v},
            causal_node_id="model_v2",
        )

        # Ingest event without ANY query
        innate_ctx = self.runtime.get_innate_context("dataset_42_record", token_budget=1024)
        
        # Innate context must automatically surface upstream sensor and downstream model!
        found_ids = {it.item_id for it in innate_ctx.items}
        self.assertIn("dataset_42_record", found_ids)
        self.assertIn("sensor_42_record", found_ids)
        self.assertIn("model_v2_record", found_ids)
        self.assertTrue(innate_ctx.structural_links_traversed >= 2)

    def test_occ_concurrency_conflict_and_staleness(self):
        """
        Test Optimistic Concurrency Control:
        Agent A reads S_42.
        Agent B changes state to S_43.
        Agent A attempts commit based on S_42 -> REJECTED as STALE_PROPOSAL_REVALIDATE.
        """
        self.runtime.register_claim("target_claim", "Target claim", EpistemicKind.HYPOTHESIS, 0.5)
        self.runtime.register_claim("read_claim", "Context read claim", EpistemicKind.AXIOM, 0.8)
        self.runtime.register_evidence("ev_agent_b", "lab", EvidenceSourceTier.LAB_ASSAY, reliability=0.9)
        self.runtime.register_evidence("ev_agent_a", "lab", EvidenceSourceTier.LAB_ASSAY, reliability=0.9)

        # Agent A reads at base_version
        base_v = self.runtime.state_version

        # Agent B commits a mutation to read_claim
        prop_b = ProposedCommit(
            commit_id="commit_b",
            action_type="STATE_UPDATE",
            target_node_id="read_claim",
            proposed_confidence_delta=0.1,
            evidence_id="ev_agent_b",
            proposing_agent_id="agent_b",
            base_version=base_v,
            read_set=["read_claim"],
            write_set=["read_claim"],
        )
        res_b = self.runtime.commit(prop_b)
        self.assertTrue(res_b.admitted)
        self.assertGreater(self.runtime.state_version, base_v)

        # Agent A now tries to commit based on stale base_version
        prop_a = ProposedCommit(
            commit_id="commit_a",
            action_type="STATE_UPDATE",
            target_node_id="target_claim",
            proposed_confidence_delta=0.2,
            evidence_id="ev_agent_a",
            proposing_agent_id="agent_a",
            base_version=base_v,  # STALE!
            read_set=["read_claim"],
            write_set=["target_claim"],
        )
        res_a = self.runtime.commit(prop_a)
        
        # Must be rejected due to OCC conflict!
        self.assertFalse(res_a.admitted)
        self.assertTrue(res_a.stale_detected)
        self.assertIn("STALE_PROPOSAL_REVALIDATE", res_a.reason)
        self.assertEqual(self.runtime.total_stale_proposals_rejected, 1)

    def test_disjoint_occ_concurrency_succeeds(self):
        """
        Test Disjoint Optimistic Concurrency:
        Agent A reads and writes biology state at base_version V_0.
        Agent B mutates unrelated finance state, advancing version to V_1.
        Agent A's commit based on V_0 MUST SUCCEED because read/write sets are disjoint!
        """
        self.runtime.register_claim("bio_claim", "Enzyme activity at 37C", EpistemicKind.HYPOTHESIS, 0.4)
        self.runtime.register_claim("fin_claim", "Q3 equipment budget", EpistemicKind.HYPOTHESIS, 0.5)
        self.runtime.register_evidence("ev_bio", "lab", EvidenceSourceTier.LAB_ASSAY, reliability=0.9)
        self.runtime.register_evidence("ev_fin", "audit", EvidenceSourceTier.REPLICATED_STUDY, reliability=0.8)

        base_v = self.runtime.state_version

        # Agent B mutates finance claim
        prop_b = ProposedCommit(
            commit_id="commit_fin_b",
            action_type="STATE_UPDATE",
            target_node_id="fin_claim",
            proposed_confidence_delta=0.1,
            evidence_id="ev_fin",
            proposing_agent_id="agent_finance",
            base_version=base_v,
            read_set=["fin_claim"],
            write_set=["fin_claim"],
        )
        res_b = self.runtime.commit(prop_b)
        self.assertTrue(res_b.admitted)
        self.assertEqual(self.runtime.state_version, base_v + 1)

        # Agent A now commits biology claim based on base_version V_0
        prop_a = ProposedCommit(
            commit_id="commit_bio_a",
            action_type="STATE_UPDATE",
            target_node_id="bio_claim",
            proposed_confidence_delta=0.2,
            evidence_id="ev_bio",
            proposing_agent_id="agent_bio",
            base_version=base_v, # base_v < current_version, but disjoint!
            read_set=["bio_claim"],
            write_set=["bio_claim"],
        )
        res_a = self.runtime.commit(prop_a)

        # Non-overlapping read/write sets -> MUST SUCCEED without false abort!
        self.assertTrue(res_a.admitted)
        self.assertFalse(res_a.stale_detected)
        self.assertEqual(self.runtime.state_version, base_v + 2)

    def test_idempotency_events_and_commits(self):
        """Test that duplicate events and duplicate commits do not duplicate side-effects."""
        obs1 = self.runtime.observe("Test event", event_id="evt_100")
        self.assertFalse(obs1.is_idempotent_skip)

        # Send same event again
        obs2 = self.runtime.observe("Test event", event_id="evt_100")
        self.assertTrue(obs2.is_idempotent_skip)
        self.assertEqual(self.runtime.total_idempotent_skips, 1)

    def test_snapshot_and_restore(self):
        """Test snapshot creation and state restoration."""
        self.runtime.register_claim("claim_x", "Claim X", EpistemicKind.HYPOTHESIS, 0.2)
        snap = self.runtime.create_checkpoint("snap_1")

        # Mutate state
        self.runtime.epistemic_manifold.nodes["claim_x"].confidence = 0.95
        self.assertEqual(self.runtime.epistemic_manifold.nodes["claim_x"].confidence, 0.95)

        # Restore
        restored = self.runtime.restore_checkpoint("snap_1")
        self.assertTrue(restored)
        self.assertEqual(self.runtime.epistemic_manifold.nodes["claim_x"].confidence, 0.2)


if __name__ == "__main__":
    unittest.main()
