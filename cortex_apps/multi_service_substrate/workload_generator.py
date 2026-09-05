"""
Workload Generator for the Multi-Service Substrate Benchmark.
=============================================================
Generates realistic, adversarial streaming event timelines:
  - Ambient lab telemetry & routine instrument logs
  - Sudden operational shocks (sensor drift, beam misalignment)
  - Remediation & re-calibration events
  - Dynamic workflow graph mutations (new dependencies added/removed)
  - Interleaved decision probes & action verification requests
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cortex_apps.multi_service_substrate.substrate_api import (
    ProposedAction,
    TelemetryEvent,
)


@dataclass
class WorkloadProbe:
    """
    Evaluation probe attached to an event point:
      - query: query string for context & search
      - target_action: proposed action to verify
      - expected_permit: ground-truth permission boolean
      - expected_woken_agents: set of agents that must wake
      - expected_affected_entities: entities that must appear in affected frontier
      - critical_doc_ids: documents required in bounded context
    """
    probe_id: str
    query: str
    target_action: ProposedAction
    expected_permit: bool
    expected_woken_agents: List[str]
    expected_affected_entities: List[str]
    critical_doc_ids: List[str]
    probe_entity_id: str = "inst_quadrupole_ms"


@dataclass
class WorkloadStep:
    step_id: int
    event: TelemetryEvent
    probe: Optional[WorkloadProbe] = None


def generate_streaming_workload(
    n_steps: int = 100,
    seed: int = 42,
    world_variant: str = "WORLD_A_LINKED",
) -> List[WorkloadStep]:
    """
    Builds a reproducible, adversarial streaming sequence of 100 events.
    Includes shocks, remediations, graph mutations, ambient noise, and interleaved decision probes.
    """
    rng = random.Random(seed)
    steps: List[WorkloadStep] = []

    # Standard proposed scale-up action
    scaleup_action = ProposedAction(
        action_id="act_pilot_alpha_commit",
        action_name="Commit $250k Capital for 100L Bioreactor Pilot Fermentation Alpha",
        target_node="node_act_bioreactor",
        required_prerequisites=["node_sensor_ms4", "node_dataset_42", "node_exp_pep", "node_hypo_yield"],
        payload={"budget": 250000, "run_type": "GMP_PILOT"},
    )

    is_linked = (world_variant == "WORLD_A_LINKED")
    ms4_is_tainted = False

    # Ambient distractor entities
    ambient_entities = [
        "ambient_0", "ambient_1", "ambient_2", "ambient_3", "ambient_4",
        "ambient_5", "ambient_6", "ambient_7", "ambient_8", "ambient_9",
    ]

    for step_idx in range(n_steps):
        # Determine event type based on timeline script
        if step_idx == 15:
            # SHOCK 1: MS-4 ion source drift
            ev_type = "SENSOR_SHOCK"
            ent_id = "inst_quadrupole_ms"
            raw_text = "CRITICAL ALERT: Quadrupole MS-4 ion transmission dropped to 64.2%. Severe calibration drift detected."
            meta = {"severity": "CRITICAL", "sensor": "MS-4"}
            ms4_is_tainted = True

        elif step_idx == 35:
            # REMEDIATION 1: MS-4 recalibrated
            ev_type = "REMEDIATION"
            ent_id = "inst_quadrupole_ms"
            raw_text = "REMEDIATION: Quadrupole MS-4 ion source cleaned and re-tuned. Calibration certified nominal at 99.2%."
            meta = {"severity": "INFO", "sensor": "MS-4"}
            ms4_is_tainted = False

        elif step_idx == 45:
            # GRAPH MUTATION 1: Explicit edge added between Dataset 42 and Peptide Assay
            ev_type = "GRAPH_MUTATION"
            ent_id = "node_dataset_42"
            raw_text = "TOPOLOGY UPDATE: Formal verified link registered: Dataset 42 -> Peptide Fingerprint Assay."
            meta = {"source_node": "node_dataset_42", "target_node": "node_exp_pep", "action": "ADD"}

        elif step_idx == 60:
            # SHOCK 2 (Secondary oscillation): MS-4 thermal drift
            ev_type = "SENSOR_SHOCK"
            ent_id = "inst_quadrupole_ms"
            raw_text = "WARNING: Quadrupole MS-4 quadrupole rod temperature excursion. Drift detected."
            meta = {"severity": "WARNING", "sensor": "MS-4"}
            ms4_is_tainted = True

        elif step_idx == 80:
            # REMEDIATION 2
            ev_type = "REMEDIATION"
            ent_id = "inst_quadrupole_ms"
            raw_text = "REMEDIATION: MS-4 thermal controller reset. Operational calibration verified."
            meta = {"severity": "INFO", "sensor": "MS-4"}
            ms4_is_tainted = False

        else:
            # Ambient lab noise
            ev_type = "AMBIENT_NOISE"
            ent_id = rng.choice(ambient_entities)
            raw_text = f"Routine operational telemetry log for facility component {ent_id}. All baseline parameters nominal."
            meta = {"category": "ROUTINE"}

        event = TelemetryEvent(
            event_id=f"ev_{step_idx:04d}",
            timestamp=1000.0 + float(step_idx) * 10.0,
            event_type=ev_type,
            entity_id=ent_id,
            raw_text=raw_text,
            metadata=meta,
        )

        # Attach evaluation probe every 5 steps or during critical events
        probe: Optional[WorkloadProbe] = None
        if step_idx in [16, 20, 30, 36, 46, 61, 70, 81, 90, 99]:
            # Ground truth expected permit:
            # In World A: if MS-4 is tainted, permit must be False (blocked)
            # In World B: Dataset 42 came from MS-2, so MS-4 drift does NOT invalidate the action (permit True)
            expected_permit = not (ms4_is_tainted and is_linked)

            expected_agents = ["agent_fermentation_scaleup"]
            if ms4_is_tainted:
                expected_agents.append("agent_instrumentation")
                expected_agents.append("agent_executive_safety")

            affected_ents = ["inst_quadrupole_ms", "node_sensor_ms4"]
            if is_linked:
                affected_ents.extend(["ds_proteomics_spectra", "node_dataset_42", "node_exp_pep", "act_bioreactor_pilot"])

            crit_docs = ["doc_inst_ms4"]
            if is_linked:
                crit_docs.append("doc_ds_data42")

            probe = WorkloadProbe(
                probe_id=f"probe_{step_idx:04d}",
                query="Is Pilot Run Alpha still scientifically justified based on current empirical evidence?",
                target_action=scaleup_action,
                expected_permit=expected_permit,
                expected_woken_agents=expected_agents,
                expected_affected_entities=affected_ents,
                critical_doc_ids=crit_docs,
            )

        steps.append(WorkloadStep(step_id=step_idx, event=event, probe=probe))

    return steps
