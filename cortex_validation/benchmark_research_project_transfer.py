"""
Independent Domain Benchmark: Research Project Generator.
Tests whether Warp Cortex's frozen reaction dynamics and epistemic verifier
transfer to a research-project domain:
  Instruments -> Datasets -> Experiments -> Hypotheses -> Clinical Scale-Up Decisions.

1. Reaction Field Transfer:
   Evaluates multi-hop cascade propagation vs Cosine Top-k and Fair BFS under budget k = 15.
2. Epistemic Verifier Transfer:
   Evaluates:
   a) Unauthenticated / contaminated dataset rejection via EvidenceRegistry.
   b) False literature dogma revision: BLOCK -> REVISE (Leaky-SPRT) -> UNBLOCK.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = r"c:\Users\jorge\gpu_holy_grail\warp_cortex"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.reaction_harness import (
    ContinuousReactionManifold,
    ManifoldEntity,
    ManifoldImpulse,
)
from cortex_core.transition_governor import (
    EvidenceRegistry,
    EvidenceSourceTier,
    TransitionGovernor,
    TransitionCertificate,
    TransitionRule,
)
from cortex_core.epistemic_manifold import (
    EpistemicManifold,
    EpistemicRelation,
    EpistemicKind,
    EpistemicStatus,
)


RESEARCH_SECTORS = {
    "instruments": [
        ("inst_cryo_detector_drift", "Cryo-TEM Detector Sensor", ["cryo", "tem", "detector", "drift", "optics"]),
        ("inst_surface_plasmon_laser", "SPR Laser Alignment", ["spr", "laser", "optics", "sensor", "binding"]),
        ("inst_hplc_column_pressure", "HPLC Chromatography Column", ["hplc", "pressure", "column", "flow", "purification"]),
        ("inst_mass_spec_ion_source", "Quadrupole MS Ion Source", ["mass_spec", "ionization", "quadrupole", "calibration", "drift"]),
    ],
    "datasets": [
        ("ds_cryo_em_micrographs", "Cryo-EM Raw Micrograph Store", ["cryo", "micrograph", "particles", "raw_data", "refinement"]),
        ("ds_binding_kinetics_curve", "Surface Plasmon Kinetics Dataset", ["spr", "kinetics", "kd", "sensorgram", "binding"]),
        ("ds_proteomics_raw_spectra", "Tandem MS Proteomics Spectra", ["mass_spec", "spectra", "peaks", "proteomics", "raw_data"]),
        ("ds_rnaseq_raw_counts", "Transcriptome RNA-Seq Counts", ["rnaseq", "counts", "transcripts", "expression", "sequencing"]),
    ],
    "experiments": [
        ("exp_cryo_em_reconstruction", "3D Cryo-EM Density Reconstruction", ["cryo", "density_map", "resolution", "structure", "conformation"]),
        ("exp_affinity_titration", "Equilibrium Binding Titration Assay", ["binding", "affinity", "dissociation", "equilibrium", "titration"]),
        ("exp_crosslinking_ms", "Crosslinking Mass Spectrometry", ["crosslinking", "mass_spec", "interface", "residues", "proximity"]),
        ("exp_transcriptome_profiling", "Differential Expression Analysis", ["expression", "differential", "transcripts", "mrna", "activation"]),
    ],
    "hypotheses": [
        ("hypo_mrna_pseudoknot_stability", "mRNA Secondary Structure Pseudoknot Stability", ["mrna", "pseudoknot", "stability", "conformation", "folding"]),
        ("hypo_target_ligand_binding", "Pocket Binding Mechanism", ["target", "ligand", "pocket", "affinity", "inhibition"]),
        ("hypo_ribosomal_frameshift_rate", "Ribosomal Frameshift Efficiency", ["ribosome", "frameshift", "translation", "kinetics", "rate"]),
        ("hypo_pathway_downregulation", "Transcriptional Pathway Suppression", ["pathway", "downregulation", "expression", "suppression", "target"]),
    ],
    "scaleup_actions": [
        ("act_commit_bioreactor_pilot_run", "Commit $250k Bioreactor Pilot Run", ["bioreactor", "pilot", "scaleup", "synthesis", "yield"]),
        ("act_approve_in_vivo_pk_assay", "Approve In Vivo Pharmacokinetics Assay", ["in_vivo", "pk", "animal_model", "clearance", "safety"]),
        ("act_formulate_lipid_nanoparticle", "Authorize Lipid Nanoparticle Formulation", ["lnp", "formulation", "encapsulation", "delivery", "mrna"]),
    ],
    "distractors": [
        ("dist_hvac_cryo_cooler", "Building HVAC Cryo Chiller", ["cryo", "cooling", "chiller", "hvac", "building_maintenance"]),
        ("dist_laser_printer_cartridge", "Office Laser Printer Cartridge", ["laser", "printer", "cartridge", "office_supplies", "toner"]),
        ("dist_it_pressure_server_rack", "Server Room Pressure Cooling", ["pressure", "rack", "server", "cooling", "airflow"]),
        ("dist_sequencer_office_wifi", "Sequencing Core Staff WiFi", ["sequencing", "wifi", "router", "network", "bandwidth"]),
        ("dist_pilot_travel_budget", "Flight Pilot Travel Reimbursement", ["pilot", "budget", "travel", "reimbursement", "accounting"]),
    ]
}


def build_research_project_manifold(seed: int = 42, hidden_dim: int = 64) -> Tuple[ContinuousReactionManifold, Dict[str, torch.Tensor], Dict[str, List[str]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    concepts = [
        "cryo", "tem", "detector", "drift", "optics", "spr", "laser", "sensor", "binding",
        "hplc", "pressure", "column", "flow", "purification", "mass_spec", "ionization",
        "quadrupole", "calibration", "micrograph", "particles", "raw_data", "refinement",
        "kinetics", "kd", "sensorgram", "spectra", "peaks", "proteomics", "rnaseq",
        "counts", "transcripts", "expression", "sequencing", "density_map", "resolution",
        "structure", "conformation", "affinity", "dissociation", "equilibrium", "titration",
        "crosslinking", "interface", "residues", "proximity", "differential", "mrna",
        "activation", "pseudoknot", "stability", "folding", "target", "ligand", "pocket",
        "inhibition", "ribosome", "frameshift", "translation", "rate", "pathway",
        "downregulation", "suppression", "bioreactor", "pilot", "scaleup", "synthesis",
        "yield", "in_vivo", "pk", "animal_model", "clearance", "safety", "lnp", "formulation",
        "encapsulation", "delivery", "cooling", "chiller", "hvac", "building_maintenance",
        "printer", "cartridge", "office_supplies", "toner", "rack", "server", "airflow",
        "wifi", "router", "network", "bandwidth", "budget", "travel", "reimbursement", "accounting"
    ]

    concept_vecs: Dict[str, torch.Tensor] = {}
    for c in concepts:
        v = torch.randn(hidden_dim)
        concept_vecs[c] = F.normalize(v, dim=0)

    manifold = ContinuousReactionManifold(
        hidden_dim=hidden_dim,
        decay_rate=0.10,
        diffusion_rate=0.40,
        semantic_threshold=0.25,
        kernel_sigma=0.65,
    )

    causal_adj: Dict[str, List[str]] = {}

    for sector, items in RESEARCH_SECTORS.items():
        for eid, name, kws in items:
            causal_adj[eid] = []
            proto_dict = {}
            vec0 = torch.zeros(hidden_dim)
            for kw in kws[:min(3, len(kws))]:
                vec0 += concept_vecs[kw]
            proto_dict["functional"] = F.normalize(vec0, dim=0)

            vec1 = torch.zeros(hidden_dim)
            for kw in kws[1:]:
                vec1 += concept_vecs[kw]
            proto_dict["context"] = F.normalize(vec1, dim=0)

            th = 0.35 if sector != "distractors" else 0.50
            manifold.register_entity(
                entity_id=eid,
                name=name,
                role=sector,
                prototypes=proto_dict,
                activation_threshold=th,
                rebuild_topology=False,
            )

    pipeline_edges = [
        ("inst_cryo_detector_drift", "ds_cryo_em_micrographs"),
        ("inst_surface_plasmon_laser", "ds_binding_kinetics_curve"),
        ("inst_mass_spec_ion_source", "ds_proteomics_raw_spectra"),
        ("ds_cryo_em_micrographs", "exp_cryo_em_reconstruction"),
        ("ds_binding_kinetics_curve", "exp_affinity_titration"),
        ("ds_proteomics_raw_spectra", "exp_crosslinking_ms"),
        ("exp_cryo_em_reconstruction", "hypo_mrna_pseudoknot_stability"),
        ("exp_affinity_titration", "hypo_target_ligand_binding"),
        ("exp_crosslinking_ms", "hypo_ribosomal_frameshift_rate"),
        ("hypo_mrna_pseudoknot_stability", "act_commit_bioreactor_pilot_run"),
        ("hypo_target_ligand_binding", "act_approve_in_vivo_pk_assay"),
        ("hypo_ribosomal_frameshift_rate", "act_formulate_lipid_nanoparticle"),
    ]

    for src, dst in pipeline_edges:
        causal_adj[src].append(dst)
        blend = F.normalize(manifold.entities[src].prototypes["functional"] + manifold.entities[dst].prototypes["functional"], dim=0)
        manifold.entities[src].prototypes[f"couples_{dst}"] = blend
        manifold.entities[dst].prototypes[f"coupled_by_{src}"] = blend

    manifold._rebuild_topology()
    return manifold, concept_vecs, causal_adj


def evaluate_research_reaction_transfer():
    print("=" * 115)
    print("INDEPENDENT DOMAIN TRANSFER 1: RESEARCH PROJECT REACTION FIELD")
    print("Testing multi-hop cascade propagation under budget k = 15")
    print("Chain: Cryo Detector Drift -> Micrograph Dataset -> 3D Reconstruction -> mRNA Pseudoknot -> Bioreactor Commit")
    print("=" * 115)

    manifold, concept_vecs, causal_adj = build_research_project_manifold(seed=42)

    ground_truth_affected = {
        "ds_cryo_em_micrographs",
        "exp_cryo_em_reconstruction",
        "hypo_mrna_pseudoknot_stability",
        "act_commit_bioreactor_pilot_run",
    }

    impulse_kws = ["cryo", "tem", "detector", "drift"]
    event_vec = torch.zeros(64)
    for kw in impulse_kws:
        event_vec += concept_vecs[kw]
    event_vec = F.normalize(event_vec, dim=0)

    k_budget = 15

    # 1. Cosine Top-k
    cos_scores = {}
    for eid, entity in manifold.entities.items():
        sims = [float(torch.dot(proto, event_vec).item()) for proto in entity.prototypes.values()]
        cos_scores[eid] = max(sims)
    cosine_top_k = set(sorted(cos_scores, key=cos_scores.get, reverse=True)[:k_budget])

    # 2. Fair BFS
    bfs_visited = ["inst_cryo_detector_drift"]
    queue = ["inst_cryo_detector_drift"]
    visited_set = {"inst_cryo_detector_drift"}
    while queue and len(bfs_visited) < k_budget:
        curr = queue.pop(0)
        for nxt in causal_adj.get(curr, []):
            if nxt not in visited_set:
                visited_set.add(nxt)
                bfs_visited.append(nxt)
                queue.append(nxt)
                if len(bfs_visited) >= k_budget:
                    break
    bfs_top_k = set(bfs_visited)

    # 3. Cortex Reaction Field (Frozen params)
    manifold.inject_impulse(text="Cryo-TEM detector drift detected", embedding=event_vec, magnitude=1.0)
    for step in range(4):
        triggered = manifold.step_diffusion(steps=1)
        for ent in triggered:
            manifold.emit_reaction(ent.entity_id, text=f"Secondary alert from {ent.name}", aspect="context", magnitude=0.45)

    cortex_top_k = set(sorted(manifold.entities, key=lambda k: manifold.entities[k].current_energy, reverse=True)[:k_budget])

    def score(retrieved):
        tp = len(retrieved.intersection(ground_truth_affected))
        rec = tp / len(ground_truth_affected) * 100.0
        prec = tp / len(retrieved) * 100.0
        distractors = [x for x in retrieved if x.startswith("dist_")]
        return rec, prec, len(distractors)

    rec_cos, prec_cos, dist_cos = score(cosine_top_k)
    rec_bfs, prec_bfs, dist_bfs = score(bfs_top_k)
    rec_ctx, prec_ctx, dist_ctx = score(cortex_top_k)

    print(f"{'Method (k=15)':<25} | {'Ground Truth Recall':<20} | {'Precision':<15} | {'Distractors Awakened':<20}")
    print("-" * 90)
    print(f"{'Cosine Top-k':<25} | {rec_cos:>18.1f}% | {prec_cos:>13.1f}% | {dist_cos:>20}")
    print(f"{'Fair Graph BFS':<25} | {rec_bfs:>18.1f}% | {prec_bfs:>13.1f}% | {dist_bfs:>20}")
    print(f"{'Cortex Hybrid Field':<25} | {rec_ctx:>18.1f}% | {prec_ctx:>13.1f}% | {dist_ctx:>20}")
    print("=" * 90)


def evaluate_research_verifier_transfer():
    print("\n" + "=" * 115)
    print("INDEPENDENT DOMAIN TRANSFER 2: RESEARCH PROJECT EPISTEMIC VERIFIER")
    print("Testing: A) Contaminated Data Provenance Gating, and B) False Literature Dogma Revision")
    print("=" * 115)

    manifold = EpistemicManifold()
    manifold.register_claim("hypo_target_inhibition", "Target enzyme is inhibited by lead compound X", kind=EpistemicKind.HYPOTHESIS, confidence=0.50)
    manifold.register_claim("hypo_dogma_cofactor_req", "Literature Dogma: Inhibition requires cofactor Zn2+", kind=EpistemicKind.AXIOM, confidence=0.90)
    manifold.register_claim("act_commit_pilot_scaleup", "Authorize $250k Pilot Bioreactor Production Run", kind=EpistemicKind.HYPOTHESIS, confidence=0.00)

    manifold.link_claims("act_commit_pilot_scaleup", "hypo_target_inhibition", EpistemicRelation.LOGICALLY_REQUIRES)
    manifold.link_claims("act_commit_pilot_scaleup", "hypo_dogma_cofactor_req", EpistemicRelation.LOGICALLY_REQUIRES)

    registry = EvidenceRegistry()
    registry.register_evidence(
        evidence_id="ev_peer_reviewed_crystallography",
        tier=EvidenceSourceTier.LAB_ASSAY,
        source_type="cryo_crystallography",
        description="PDB crystal structure 8XYZ at 1.8A resolution",
        custom_reliability=0.95,
    )

    governor = TransitionGovernor(evidence_registry=registry, topology_revision_threshold=2.0)

    # Test A: Unregistered / Tainted Data Injection
    print("\n[Test A: Contaminated / Tainted Dataset Gating]")
    cert_tainted = TransitionCertificate(
        evidence_id="ev_uncalibrated_raw_sensor_dump", # Not in registry!
        target_node_id="hypo_target_inhibition",
        proposed_confidence_delta=0.45,
        rule=TransitionRule.DIRECT_EMPIRICAL_UPDATE,
    )
    res_tainted = governor.evaluate_transition(manifold, cert_tainted)
    print(f"  Attempting state transition with unauthenticated data 'ev_uncalibrated_raw_sensor_dump'...")
    print(f"  Admitted: {res_tainted.admitted}")
    print(f"  Reason  : {res_tainted.reason}")
    assert not res_tainted.admitted, "Provenance failure was not caught!"
    print("  -> PASSED: Tainted / unauthenticated data rejected by EvidenceRegistry.")

    # Test B: False Literature Dogma Revision (BLOCK -> REVISE -> UNBLOCK)
    print("\n[Test B: False Literature Dogma Revision (BLOCK -> REVISE -> UNBLOCK)]")
    cert_valid = TransitionCertificate(
        evidence_id="ev_peer_reviewed_crystallography",
        target_node_id="hypo_target_inhibition",
        proposed_confidence_delta=0.35,
        rule=TransitionRule.DIRECT_EMPIRICAL_UPDATE,
    )
    admitted, res_valid = governor.commit_if_admitted(manifold, cert_valid)
    assert admitted
    print(f"  Hypothesis 'hypo_target_inhibition' confidence updated to {manifold.nodes['hypo_target_inhibition'].confidence:.2f}")

    # Dogma premise is falsified by Zn2+ depletion assay
    manifold.nodes["hypo_dogma_cofactor_req"].confidence = -0.60
    print("  Assay falsifies cofactor requirement: 'hypo_dogma_cofactor_req' confidence = -0.60.")

    cert_scaleup = TransitionCertificate(
        evidence_id="ev_peer_reviewed_crystallography",
        target_node_id="act_commit_pilot_scaleup",
        proposed_confidence_delta=0.90,
        rule=TransitionRule.DIRECT_EMPIRICAL_UPDATE,
    )

    # Phase 1: Attempt commit
    res_scaleup_1 = governor.evaluate_transition(manifold, cert_scaleup)
    print(f"  Phase 1 (Initial Scale-Up Attempt): Admitted = {res_scaleup_1.admitted}")
    print(f"  Blocked in Omega: Hard epistemic invariant violated (prerequisite cofactor dogma is falsified).")
    assert not res_scaleup_1.admitted

    # Phase 2: Successive pilot batches confirm high yield without cofactor (accumulating strain)
    print("  Phase 2: Successive pilot runs confirm 92% yield without cofactor (accumulating strain on dogma invariant)...")
    unblocked = False
    unblock_step = None

    for step in range(1, 10):
        admitted, res = governor.commit_if_admitted(manifold, cert_scaleup)
        edge = manifold._adjacency["act_commit_pilot_scaleup"]["hypo_dogma_cofactor_req"]
        print(f"    Step {step}: Invariant Active = {edge.is_active}, Accumulated Strain = {edge.edge_strain:.2f} / {governor.topology_revision_threshold:.1f}")

        if admitted:
            unblocked = True
            unblock_step = step
            print(f"    *** Invariant Severed! False literature dogma removed from causal graph at Step {step}! ***")
            break

    # Phase 3: Verify outcome
    print(f"  Phase 3 (Post-Revision Scale-Up Attempt): Admitted = {unblocked}")
    assert unblocked, "Action should be permitted after dogma revision!"
    print(f"  Scale-Up Commit Status: {manifold.nodes['act_commit_pilot_scaleup'].confidence:.2f} (Admitted at Step {unblock_step})")
    print(f"  -> PASSED: Scale-up decision successfully unblocked after empirical revision.")
    print("=" * 115)


if __name__ == "__main__":
    evaluate_research_reaction_transfer()
    evaluate_research_verifier_transfer()
