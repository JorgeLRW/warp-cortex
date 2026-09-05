"""
Unit Tests for Multi-Service Substrate Package.
===============================================
Verifies that all three contenders (Unified, Fragmented Naive, Fragmented Production)
correctly implement the Substrate API and adhere to functional invariant contracts.
"""

import pytest

from cortex_apps.multi_service_substrate.fragmented_naive import FragmentedNaiveArchitecture
from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.substrate_api import (
    EntityStatus,
    ProposedAction,
    TelemetryEvent,
)
from cortex_apps.multi_service_substrate.unified_substrate import UnifiedContextSubstrate
from cortex_apps.multi_service_substrate.workload_generator import generate_streaming_workload
from cortex_apps.research_agent_system.world_state import build_research_world


@pytest.fixture
def catalog():
    return build_research_world(seed=42, world_variant="WORLD_A_LINKED")


@pytest.mark.parametrize("arch_cls", [
    UnifiedContextSubstrate,
    FragmentedNaiveArchitecture,
    FragmentedProductionArchitecture,
])
def test_substrate_lifecycle_and_verification(catalog, arch_cls):
    sub = arch_cls(catalog)
    sub.reset_metrics()

    # 1. Action definition
    action = ProposedAction(
        action_id="act_pilot_test",
        action_name="Scale-up Commit Test",
        target_node="node_act_bioreactor",
        required_prerequisites=["node_sensor_ms4", "node_dataset_42"],
    )

    # Initial state should permit action
    res_initial = sub.verify(action)
    assert res_initial.permit is True, f"{arch_cls.__name__} failed initial permit"

    # 2. Ingest SENSOR_SHOCK on MS-4
    shock_event = TelemetryEvent(
        event_id="ev_shock_1",
        timestamp=100.0,
        event_type="SENSOR_SHOCK",
        entity_id="inst_quadrupole_ms",
        raw_text="Quadrupole MS-4 severe calibration drift.",
    )
    v_shock = sub.ingest(shock_event)
    assert v_shock > 1

    # 3. Action must now be blocked
    res_shock = sub.verify(action)
    assert res_shock.permit is False, f"{arch_cls.__name__} failed to block on shock"

    # 4. Routing test: safety and instrumentation agents must wake
    woken, v_route = sub.route(shock_event)
    assert "agent_instrumentation" in woken or "agent_executive_safety" in woken

    # 5. Context selection test: MS-4 document must be in packed context
    ctx = sub.context("Is pilot run justified?", token_budget=256)
    doc_ids = [d.doc_id for d in ctx.documents]
    assert "doc_inst_ms4" in doc_ids

    # 6. Affected frontier: Dataset 42 should be surfaced
    affected_ents, _ = sub.affected("inst_quadrupole_ms")
    assert any("42" in e or "spectra" in e for e in affected_ents)

    # 7. Remediation event
    remed_event = TelemetryEvent(
        event_id="ev_remed_1",
        timestamp=200.0,
        event_type="REMEDIATION",
        entity_id="inst_quadrupole_ms",
        raw_text="Quadrupole MS-4 recalibrated nominal.",
    )
    v_remed = sub.ingest(remed_event)
    assert v_remed > v_shock

    # 8. Action should now be permitted again
    res_remed = sub.verify(action)
    assert res_remed.permit is True, f"{arch_cls.__name__} failed to restore permit after remediation"

    metrics = sub.get_metrics()
    assert metrics.writes > 0
    assert metrics.cpu_time_ms >= 0.0


def test_subscription_callback(catalog):
    sub = UnifiedContextSubstrate(catalog)
    received_events = []

    def callback(ev, v):
        received_events.append((ev.event_id, v))

    sub.subscribe(lambda e: e.event_type == "SENSOR_SHOCK", callback)

    sub.ingest(TelemetryEvent("ev_1", 1.0, "AMBIENT_NOISE", "ambient_0", "routine"))
    assert len(received_events) == 0

    sub.ingest(TelemetryEvent("ev_2", 2.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "drift"))
    assert len(received_events) == 1
    assert received_events[0][0] == "ev_2"
