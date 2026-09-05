"""
World State: Persistent Biological Research Environment.
========================================================
Defines the entities, documents, physical dependencies, and causal topology
for a multi-agent pharmaceutical research laboratory.

Entities follow a rigorous dependency hierarchy:
  Instruments -> Datasets -> Experiments/Pipelines -> Hypotheses -> Commitments
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from cortex_core.semantic_fabric import SemanticBand
from cortex_core.epistemic_manifold import (
    EpistemicKind,
    EpistemicRelation,
    EpistemicStatus,
)


@dataclass
class ResearchDocument:
    doc_id: str
    title: str
    content: str
    band: str
    entity_id: str
    embedding: torch.Tensor
    causal_node_id: Optional[str] = None
    tokens: int = 35
    version: int = 1
    state_tag: str = "VALID"
    metadata: Dict[str, Any] = field(default_factory=dict)
    aspect_vectors: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class ResearchWorldCatalog:
    documents: Dict[str, ResearchDocument]
    entity_to_doc: Dict[str, str]
    causal_dependencies: List[Tuple[str, str, str]]  # (source_node, target_node, relation)
    band_anchors: Dict[str, torch.Tensor]
    hidden_dim: int = 64


def build_research_world(
    hidden_dim: int = 64,
    seed: int = 42,
    complete_topology: bool = True,
    world_variant: str = "WORLD_A_LINKED",
) -> ResearchWorldCatalog:
    """
    Constructs an identical, realistic scientific research repository.
    Includes instruments, datasets, experiments, hypotheses, scale-up commitments,
    and ambient distractor documents.
    
    Supports:
      - world_variant = "WORLD_A_LINKED": Dataset 42 acquired from Quadrupole MS-4 (compromised when MS-4 drifts).
      - world_variant = "WORLD_B_UNLINKED": Dataset 42 acquired from nominal Quadrupole MS-2 (unaffected by MS-4 drift).
      - complete_topology: If False, omits the explicit dependency edge between Dataset 42 and downstream experiment.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # 1. Aspect Band Anchors
    bands = [
        SemanticBand.INSTRUMENTATION.value,
        SemanticBand.DATA_VALIDITY.value,
        SemanticBand.MECHANISM.value,
        SemanticBand.MANUFACTURING.value,
        SemanticBand.UNIT_ECONOMICS.value,
        SemanticBand.SAFETY.value,
    ]
    band_anchors: Dict[str, torch.Tensor] = {}
    for i, b in enumerate(bands):
        torch.manual_seed(seed + i * 17)
        band_anchors[b] = F.normalize(torch.randn(hidden_dim), dim=0)

    docs: Dict[str, ResearchDocument] = {}
    entity_to_doc: Dict[str, str] = {}
    dependencies: List[Tuple[str, str, str]] = []

    # 2. Core Research Entities
    # --- Causal Path 1: Quadrupole MS -> Dataset 42 -> Peptide Fingerprinting -> Yield Model v4 -> $250k Bioreactor Commit
    # --- Causal Path 2: Cryo-TEM -> Dataset 18 -> 3D Reconstruction -> Binding Hypothesis -> In Vivo PK Commit

    is_unlinked = (world_variant == "WORLD_B_UNLINKED")
    d42_content = (
        "Raw high-resolution mass spectra collection Dataset 42 acquired directly from Quadrupole MS-2 for peptide yield validation."
        if is_unlinked else
        "Raw high-resolution mass spectra collection Dataset 42 acquired directly from Quadrupole MS-4 for peptide yield validation."
    )

    specs = [
        # Path 1: Yield & Bioreactor
        {
            "doc_id": "doc_inst_ms4",
            "entity_id": "inst_quadrupole_ms",
            "causal_node": "node_sensor_ms4",
            "band": SemanticBand.INSTRUMENTATION.value,
            "title": "Quadrupole MS-4 Calibration Log",
            "content": "Quadrupole mass spectrometer MS-4 ion source operating calibration. Nominal transmission efficiency 99.1%.",
            "tokens": 30,
        },
        {
            "doc_id": "doc_inst_ms2",
            "entity_id": "inst_quadrupole_ms2",
            "causal_node": "node_sensor_ms2",
            "band": SemanticBand.INSTRUMENTATION.value,
            "title": "Quadrupole MS-2 Calibration Log",
            "content": "Quadrupole mass spectrometer MS-2 ion source operating calibration. Nominal transmission efficiency 99.4%.",
            "tokens": 30,
        },
        {
            "doc_id": "doc_ds_data42",
            "entity_id": "ds_proteomics_spectra",
            "causal_node": "node_dataset_42",
            "band": SemanticBand.DATA_VALIDITY.value,
            "title": "Dataset 42: Tandem MS Proteomics Spectra",
            "content": d42_content,
            "tokens": 35,
        },
        {
            "doc_id": "doc_exp_pep_fingerprint",
            "entity_id": "exp_ms_fingerprinting",
            "causal_node": "node_exp_pep",
            "band": SemanticBand.MECHANISM.value,
            "title": "Peptide Mass Fingerprint Assay Report",
            "content": "Analysis of mass fingerprinting derived from Dataset 42 spectra. Confirms target recombinant sequence integrity.",
            "tokens": 40,
        },
        {
            "doc_id": "doc_hypo_yield_v4",
            "entity_id": "hypo_yield_model_v4",
            "causal_node": "node_hypo_yield",
            "band": SemanticBand.MANUFACTURING.value,
            "title": "Target Yield Prediction Model v4",
            "content": "Bioreactor expression yield forecast model v4. Validates estimated >2.8g/L yield contingent upon Dataset 42 peptide stability.",
            "tokens": 35,
        },
        {
            "doc_id": "doc_act_bioreactor_pilot",
            "entity_id": "act_bioreactor_pilot",
            "causal_node": "node_act_bioreactor",
            "band": SemanticBand.MANUFACTURING.value,
            "title": "Scale-up Commitment SOP: $250k Bioreactor Pilot Run Alpha",
            "content": "Standard operating procedure authorizing release of $250k capital for 100L pilot bioreactor fermentation Alpha. Requires verified Yield Model v4.",
            "tokens": 45,
        },

        # Path 2: Cryo-EM & In Vivo PK
        {
            "doc_id": "doc_inst_cryo1",
            "entity_id": "inst_cryo_tem",
            "causal_node": "node_inst_cryo",
            "band": SemanticBand.INSTRUMENTATION.value,
            "title": "Titan Krios Cryo-TEM Detector Status",
            "content": "Direct electron detector calibration for Titan Krios 300kV Cryo-TEM. Beam alignment within 0.1 mrad tolerance.",
            "tokens": 30,
        },
        {
            "doc_id": "doc_ds_cryo18",
            "entity_id": "ds_cryo_micrographs",
            "causal_node": "node_ds_cryo18",
            "band": SemanticBand.DATA_VALIDITY.value,
            "title": "Dataset 18: Cryo-EM Micrograph Store",
            "content": "Raw movie stacks and motion-corrected micrographs Dataset 18 collected on Titan Krios for macromolecular reconstruction.",
            "tokens": 35,
        },
        {
            "doc_id": "doc_exp_cryo_recon",
            "entity_id": "exp_cryo_reconstruction",
            "causal_node": "node_exp_recon",
            "band": SemanticBand.MECHANISM.value,
            "title": "3D Density Reconstruction Map 2.4A",
            "content": "Single-particle 3D density map reconstructed from Dataset 18 micrographs. Resolves allosteric binding pocket architecture.",
            "tokens": 40,
        },
        {
            "doc_id": "doc_hypo_allosteric",
            "entity_id": "hypo_allosteric_binding",
            "causal_node": "node_hypo_allosteric",
            "band": SemanticBand.MECHANISM.value,
            "title": "Allosteric Binding Mechanism Hypothesis",
            "content": "Mechanistic model demonstrating pocket cavity gating and cooperativity, verified against 2.4A density map.",
            "tokens": 35,
        },
        {
            "doc_id": "doc_act_invivo_pk",
            "entity_id": "act_in_vivo_pk",
            "causal_node": "node_act_pk",
            "band": SemanticBand.SAFETY.value,
            "title": "Clinical Assay SOP: $180k In Vivo PK Study Authorization",
            "content": "Protocol approving $180k animal cohort enrollment for In Vivo pharmacokinetics. Requires verified allosteric target engagement.",
            "tokens": 45,
        },
    ]

    # Register dependencies:
    root_sensor = "node_sensor_ms2" if is_unlinked else "node_sensor_ms4"
    dependencies.append((root_sensor, "node_dataset_42", EpistemicRelation.LOGICALLY_REQUIRES.value))
    if complete_topology:
        # Complete topology includes explicit edge from Dataset 42 to Peptide Assay
        dependencies.append(("node_dataset_42", "node_exp_pep", EpistemicRelation.LOGICALLY_REQUIRES.value))
    dependencies.extend([
        ("node_exp_pep", "node_hypo_yield", EpistemicRelation.LOGICALLY_REQUIRES.value),
        ("node_hypo_yield", "node_act_bioreactor", EpistemicRelation.LOGICALLY_REQUIRES.value),

        ("node_inst_cryo", "node_ds_cryo18", EpistemicRelation.LOGICALLY_REQUIRES.value),
        ("node_ds_cryo18", "node_exp_recon", EpistemicRelation.LOGICALLY_REQUIRES.value),
        ("node_exp_recon", "node_hypo_allosteric", EpistemicRelation.LOGICALLY_REQUIRES.value),
        ("node_hypo_allosteric", "node_act_pk", EpistemicRelation.LOGICALLY_REQUIRES.value),
    ])

    for item in specs:
        anchor = band_anchors[item["band"]]
        vec = F.normalize(anchor + 0.15 * torch.randn(hidden_dim), dim=0)
        docs[item["doc_id"]] = ResearchDocument(
            doc_id=item["doc_id"],
            title=item["title"],
            content=item["content"],
            band=item["band"],
            entity_id=item["entity_id"],
            embedding=vec,
            causal_node_id=item["causal_node"],
            tokens=item["tokens"],
        )
        entity_to_doc[item["entity_id"]] = item["doc_id"]

    # Configure multi-aspect coordinates for Dataset 42
    # In World A: acquired from MS-4. Secondary aspect is MS-4 instrumentation.
    # In World B: acquired from MS-2. Secondary aspect is MS-2 instrumentation.
    ms_inst_doc_id = "doc_inst_ms2" if is_unlinked else "doc_inst_ms4"
    docs["doc_ds_data42"].aspect_vectors = {
        SemanticBand.DATA_VALIDITY.value: docs["doc_ds_data42"].embedding,
        SemanticBand.INSTRUMENTATION.value: docs[ms_inst_doc_id].embedding,
    }

    # 3. Add 40 Ambient Research / Infrastructure Distractor Documents
    # (Covers maintenance, HR, cleaning, unrelated batch logs, chemical inventories)
    distractor_templates = [
        ("Facility HVAC coolant loop inspection report", SemanticBand.SAFETY.value),
        ("Autoclave sterilization log for glassware unit 3", SemanticBand.SAFETY.value),
        ("HPLC solvent delivery pump seal replacement", SemanticBand.INSTRUMENTATION.value),
        ("General lab chemical hygiene and MSDS audit", SemanticBand.SAFETY.value),
        ("Monthly nitrogen dewar refill certification", SemanticBand.UNIT_ECONOMICS.value),
        ("Annual pipette volumetric calibration checklist", SemanticBand.INSTRUMENTATION.value),
        ("Routine peptide buffer pH stability testing", SemanticBand.DATA_VALIDITY.value),
        ("Centrifuge rotor balance speed safety inspection", SemanticBand.SAFETY.value),
        ("Office ethernet switch maintenance downtime notice", SemanticBand.MANUFACTURING.value),
        ("Staff biosafety level 2 refresher training record", SemanticBand.SAFETY.value),
    ]

    for idx in range(40):
        d_id = f"doc_ambient_{idx}"
        title_tmpl, band = distractor_templates[idx % len(distractor_templates)]
        title = f"{title_tmpl} (Log #{1000 + idx})"
        content = f"Standard routine documentation regarding {band} procedures for facility sub-unit {idx // 4}. All parameters within normal baseline range."
        anchor = band_anchors[band]
        vec = F.normalize(anchor + 0.30 * torch.randn(hidden_dim), dim=0)

        docs[d_id] = ResearchDocument(
            doc_id=d_id,
            title=title,
            content=content,
            band=band,
            entity_id=f"ambient_{idx}",
            embedding=vec,
            causal_node_id=None,
            tokens=30,
        )

    return ResearchWorldCatalog(
        documents=docs,
        entity_to_doc=entity_to_doc,
        causal_dependencies=dependencies,
        band_anchors=band_anchors,
        hidden_dim=hidden_dim,
    )
