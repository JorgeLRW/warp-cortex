"""
Scale Catalog Generator: Multi-Entity Enterprise Graph & Document Synthesis.
=============================================================================
Synthesizes scalable enterprise research worlds for arbitrary entity count N
with realistic causal DAG topologies, multi-aspect vector manifolds, and text metadata.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cortex_core.semantic_fabric import SemanticBand
from cortex_core.epistemic_manifold import EpistemicRelation
from cortex_apps.research_agent_system.world_state import (
    ResearchDocument,
    ResearchWorldCatalog,
    build_research_world,
)


def build_scalable_research_world(
    n_entities: int = 100,
    hidden_dim: int = 64,
    seed: int = 42,
) -> ResearchWorldCatalog:
    """
    Constructs a scalable enterprise catalog with N entities.
    Features:
      - 20% Instruments, 20% Datasets, 20% Assays, 20% Models, 20% Commitments.
      - Causal chains linking Instrument -> Dataset -> Assay -> Model -> Commitment.
      - Multi-aspect vectors for datasets and assays (linking instrumentation and mechanism).
      - Cross-chain distractors and peripheral entities.
    """
    if n_entities <= 50:
        return build_research_world(hidden_dim=hidden_dim, seed=seed)

    torch.manual_seed(seed)
    random.seed(seed)

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

    # First add the canonical 10 core entities from build_research_world so all standard test probes work
    base_catalog = build_research_world(hidden_dim=hidden_dim, seed=seed)
    docs.update(base_catalog.documents)
    entity_to_doc.update(base_catalog.entity_to_doc)
    dependencies.extend(base_catalog.causal_dependencies)

    remaining = n_entities - len(docs)
    n_chains = max(1, remaining // 5)
    remainder = remaining % 5 if remaining > 5 else 0

    # Generate synthetic 5-step causal chains
    for c in range(n_chains):
        inst_id = f"inst_gen_{c}"
        ds_id = f"ds_gen_{c}"
        exp_id = f"exp_gen_{c}"
        hypo_id = f"hypo_gen_{c}"
        act_id = f"act_gen_{c}"

        inst_doc = f"doc_inst_gen_{c}"
        ds_doc = f"doc_ds_gen_{c}"
        exp_doc = f"doc_exp_gen_{c}"
        hypo_doc = f"doc_hypo_gen_{c}"
        act_doc = f"doc_act_gen_{c}"

        inst_node = f"node_inst_gen_{c}"
        ds_node = f"node_ds_gen_{c}"
        exp_node = f"node_exp_gen_{c}"
        hypo_node = f"node_hypo_gen_{c}"
        act_node = f"node_act_gen_{c}"

        chain_specs = [
            (inst_doc, inst_id, inst_node, SemanticBand.INSTRUMENTATION.value, f"Automated Sensor Platform {c}", f"Calibration logs for automated diagnostic sensor platform {c}."),
            (ds_doc, ds_id, ds_node, SemanticBand.DATA_VALIDITY.value, f"Telemetry Stream Dataset {c}", f"Raw continuous diagnostic stream dataset {c} acquired from sensor {c}."),
            (exp_doc, exp_id, exp_node, SemanticBand.MECHANISM.value, f"Empirical Assay Pipeline {c}", f"Kinetic assay protocol verifying molecular interaction rates based on dataset {c}."),
            (hypo_doc, hypo_id, hypo_node, SemanticBand.MANUFACTURING.value, f"Process Optimization Model {c}", f"Yield and stability computational model {c} parameterized by assay {c}."),
            (act_doc, act_id, act_node, SemanticBand.SAFETY.value, f"Capital Deployment Action {c}", f"Capital allocation authorization committing production scaling under model {c}."),
        ]

        # Add dependencies along the chain
        dependencies.append((inst_node, ds_node, EpistemicRelation.LOGICALLY_REQUIRES.value))
        dependencies.append((ds_node, exp_node, EpistemicRelation.LOGICALLY_REQUIRES.value))
        dependencies.append((exp_node, hypo_node, EpistemicRelation.LOGICALLY_REQUIRES.value))
        dependencies.append((hypo_node, act_node, EpistemicRelation.LOGICALLY_REQUIRES.value))

        for d_id, e_id, node_id, band_name, title, content in chain_specs:
            anchor = band_anchors[band_name]
            vec = F.normalize(anchor + 0.15 * torch.randn(hidden_dim), dim=0)
            docs[d_id] = ResearchDocument(
                doc_id=d_id,
                title=title,
                content=content,
                band=band_name,
                entity_id=e_id,
                embedding=vec,
                causal_node_id=node_id,
                tokens=35,
            )
            entity_to_doc[e_id] = d_id

        # Multi-aspect vectors for dataset linking sensor
        docs[ds_doc].aspect_vectors = {
            SemanticBand.DATA_VALIDITY.value: docs[ds_doc].embedding,
            SemanticBand.INSTRUMENTATION.value: docs[inst_doc].embedding,
        }

    # Add any remainder entities as distractor/ambient documents
    for r in range(remainder):
        r_id = f"doc_scale_ambient_{r}"
        e_id = f"scale_ambient_ent_{r}"
        band = random.choice(bands)
        vec = F.normalize(band_anchors[band] + 0.25 * torch.randn(hidden_dim), dim=0)
        docs[r_id] = ResearchDocument(
            doc_id=r_id,
            title=f"Ambient Environmental Log {r}",
            content=f"Facility temperature and pressure monitoring log record {r}.",
            band=band,
            entity_id=e_id,
            embedding=vec,
            tokens=25,
        )
        entity_to_doc[e_id] = r_id

    return ResearchWorldCatalog(
        documents=docs,
        entity_to_doc=entity_to_doc,
        causal_dependencies=dependencies,
        band_anchors=band_anchors,
        hidden_dim=hidden_dim,
    )
