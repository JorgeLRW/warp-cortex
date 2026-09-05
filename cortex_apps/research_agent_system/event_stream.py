"""
Event Stream: Asynchronous Laboratory Timeline Generator.
==========================================================
Generates realistic multi-phase timelines for pharmaceutical research operations:
  Phase 1: Baseline nominal operations (Steps 0-20).
  Phase 2: Unheralded physical shock (Step 21: MS-4 Quadrupole drift invalidates Dataset 42).
  Phase 3: Ambient operational noise (Steps 22-120: 100 unrelated facility/maintenance events).
  Phase 4: High-stakes scale-up commitment query (Step 121: $250k Bioreactor pilot run).
  Phase 5: Remediation event (Step 125: MS-4 recalibration verified & certified).
  Phase 6: Post-remediation scale-up query (Step 130).

Provides both:
  - Realistic ambiguous raw event text
  - Clean structured event metadata (entity_id, status_update) for fair 2x6 factorial testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cortex_apps.research_agent_system.world_state import ResearchWorldCatalog
from cortex_core.semantic_fabric import SemanticBand


@dataclass
class StreamEvent:
    step: int
    event_id: str
    text: str
    embedding: torch.Tensor
    band: str
    magnitude: float
    entity_id: Optional[str] = None
    status_update: Optional[str] = None
    target_doc_id: Optional[str] = None
    is_shock: bool = False
    is_remediation: bool = False
    is_query_probe: bool = False
    probe_query_text: Optional[str] = None
    probe_query_vec: Optional[torch.Tensor] = None
    ground_truth_expected_action: Optional[str] = None  # "HALT" or "COMMIT"


def generate_research_timeline(
    catalog: ResearchWorldCatalog,
    seed: int = 1337,
    noise_event_count: int = 100,
) -> List[StreamEvent]:
    random.seed(seed)
    torch.manual_seed(seed)
    hidden_dim = catalog.hidden_dim

    events: List[StreamEvent] = []
    step = 0

    mfg_anchor = catalog.band_anchors[SemanticBand.MANUFACTURING.value]
    inst_anchor = catalog.band_anchors[SemanticBand.INSTRUMENTATION.value]
    safety_anchor = catalog.band_anchors[SemanticBand.SAFETY.value]
    data_anchor = catalog.band_anchors[SemanticBand.DATA_VALIDITY.value]

    scaleup_query_text = "Evaluate authorization and release $250k capital for Bioreactor Pilot Run Alpha"
    scaleup_query_vec = F.normalize(mfg_anchor + 0.08 * torch.randn(hidden_dim), dim=0)

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Nominal Operations (Steps 0-20)
    # -------------------------------------------------------------------------
    for i in range(20):
        step += 1
        ev_id = f"ev_nominal_{i}"
        text = f"Routine operational log: Facility sub-unit {i % 5} reports normal baseline readings."
        vec = F.normalize(safety_anchor + 0.25 * torch.randn(hidden_dim), dim=0)
        events.append(StreamEvent(
            step=step,
            event_id=ev_id,
            text=text,
            embedding=vec,
            band=SemanticBand.SAFETY.value,
            magnitude=0.3,
            entity_id=f"ambient_{i}",
            status_update="NOMINAL",
        ))

    # Probe query during nominal phase (should COMMIT)
    step += 1
    events.append(StreamEvent(
        step=step,
        event_id="probe_nominal_pre_shock",
        text="Pre-shock scale-up authorization audit probe",
        embedding=scaleup_query_vec,
        band=SemanticBand.MANUFACTURING.value,
        magnitude=0.1,
        is_query_probe=True,
        probe_query_text=scaleup_query_text,
        probe_query_vec=scaleup_query_vec,
        ground_truth_expected_action="COMMIT",
    ))

    # -------------------------------------------------------------------------
    # Phase 2: Unheralded Shock Event (Step 22) - Ambiguous Raw Telemetry
    # -------------------------------------------------------------------------
    step += 1
    shock_text = (
        "ANOMALOUS TELEMETRY: Mass spectrometer ion transmission efficiency dropped 5.2% "
        "following ambient chiller pressure fluctuation. Calibration boundary breached."
    )
    # The shock embedding resides in INSTRUMENTATION aspect space (near ms4 instrument)
    ms4_doc = catalog.documents["doc_inst_ms4"]
    shock_vec = F.normalize(ms4_doc.embedding + 0.08 * torch.randn(hidden_dim), dim=0)

    events.append(StreamEvent(
        step=step,
        event_id="ev_shock_ms4_drift",
        text=shock_text,
        embedding=shock_vec,
        band=SemanticBand.INSTRUMENTATION.value,
        magnitude=2.0,
        entity_id="inst_quadrupole_ms",
        status_update="TAINTED",
        target_doc_id="doc_inst_ms4",
        is_shock=True,
    ))

    # Immediate probe query right after shock (Coherence probe 1: should HALT)
    step += 1
    events.append(StreamEvent(
        step=step,
        event_id="probe_coherence_t1",
        text="Immediate post-shock scale-up probe (t+1)",
        embedding=scaleup_query_vec,
        band=SemanticBand.MANUFACTURING.value,
        magnitude=0.1,
        is_query_probe=True,
        probe_query_text=scaleup_query_text,
        probe_query_vec=scaleup_query_vec,
        ground_truth_expected_action="HALT",
    ))

    # -------------------------------------------------------------------------
    # Phase 3: Ambient Laboratory Noise (Steps 25 to 25 + noise_count)
    # -------------------------------------------------------------------------
    ambient_topics = [
        ("Centrifuge rotor balance verification", SemanticBand.SAFETY.value),
        ("Autoclave steam pressure temperature cycle", SemanticBand.SAFETY.value),
        ("Nitrogen tank level quarterly replenishment", SemanticBand.UNIT_ECONOMICS.value),
        ("General glassware chemical wash log", SemanticBand.SAFETY.value),
        ("Staff biosafety training certification", SemanticBand.SAFETY.value),
        ("Routine peptide buffer pH calibration", SemanticBand.DATA_VALIDITY.value),
    ]

    for i in range(noise_event_count):
        step += 1
        topic, band = ambient_topics[i % len(ambient_topics)]
        ev_id = f"ev_ambient_{i}"
        text = f"Facility ambient event #{i}: {topic} for sector {i % 4}. Completed nominally."
        vec = F.normalize(catalog.band_anchors[band] + 0.25 * torch.randn(hidden_dim), dim=0)
        events.append(StreamEvent(
            step=step,
            event_id=ev_id,
            text=text,
            embedding=vec,
            band=band,
            magnitude=0.2,
            entity_id=f"ambient_{i}",
            status_update="NOMINAL",
        ))

        # Periodic coherence probes every 20 noise steps
        if (i + 1) % 20 == 0:
            step += 1
            events.append(StreamEvent(
                step=step,
                event_id=f"probe_coherence_noise_{i+1}",
                text=f"Ambient noise probe at noise step {i+1}",
                embedding=scaleup_query_vec,
                band=SemanticBand.MANUFACTURING.value,
                magnitude=0.1,
                is_query_probe=True,
                probe_query_text=scaleup_query_text,
                probe_query_vec=scaleup_query_vec,
                ground_truth_expected_action="HALT",
            ))

    # -------------------------------------------------------------------------
    # Phase 4: High-Stakes Scale-Up Commitment Query (Unresolved shock still active)
    # -------------------------------------------------------------------------
    step += 1
    events.append(StreamEvent(
        step=step,
        event_id="probe_unresolved_high_stakes",
        text="High-stakes $250k Bioreactor Authorization Query (Unresolved Anomaly)",
        embedding=scaleup_query_vec,
        band=SemanticBand.MANUFACTURING.value,
        magnitude=0.1,
        is_query_probe=True,
        probe_query_text=scaleup_query_text,
        probe_query_vec=scaleup_query_vec,
        ground_truth_expected_action="HALT",
    ))

    # -------------------------------------------------------------------------
    # Phase 5: Emergency Remediation (Step ~130)
    # -------------------------------------------------------------------------
    step += 1
    remed_text = (
        "MAINTENANCE RESOLVED: Mass spectrometer ion source serviced and recalibration "
        "certificate verified at 99.7% efficiency. Calibration restored."
    )
    remed_vec = F.normalize(ms4_doc.embedding + 0.05 * torch.randn(hidden_dim), dim=0)
    events.append(StreamEvent(
        step=step,
        event_id="ev_remediation_ms4",
        text=remed_text,
        embedding=remed_vec,
        band=SemanticBand.INSTRUMENTATION.value,
        magnitude=1.5,
        entity_id="inst_quadrupole_ms",
        status_update="VALID",
        target_doc_id="doc_inst_ms4",
        is_remediation=True,
    ))

    # -------------------------------------------------------------------------
    # Phase 6: Post-Remediation Scale-Up Query (Should now COMMIT)
    # -------------------------------------------------------------------------
    step += 1
    events.append(StreamEvent(
        step=step,
        event_id="probe_post_remediation",
        text="Post-remediation scale-up authorization audit probe",
        embedding=scaleup_query_vec,
        band=SemanticBand.MANUFACTURING.value,
        magnitude=0.1,
        is_query_probe=True,
        probe_query_text=scaleup_query_text,
        probe_query_vec=scaleup_query_vec,
        ground_truth_expected_action="COMMIT",
    ))

    return events
