"""
Unit Tests for Systems Kill-Test Suite.
=======================================
Verifies the Versioned Modular Monolith, real TCP socket communication,
and the Service-7 Explain-Risk implementations.
"""

import pytest

from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.modular_monolith import VersionedModularMonolith
from cortex_apps.multi_service_substrate.network_ipc_service import FragmentedNetworkArchitecture
from cortex_apps.multi_service_substrate.service7_explain_risk import (
    explain_risk_fragmented,
    explain_risk_unified,
)
from cortex_apps.multi_service_substrate.substrate_api import (
    ProposedAction,
    TelemetryEvent,
)
from cortex_apps.multi_service_substrate.unified_substrate import UnifiedContextSubstrate
from cortex_apps.research_agent_system.world_state import build_research_world


@pytest.fixture
def catalog():
    return build_research_world(seed=42, world_variant="WORLD_A_LINKED")


def test_modular_monolith_lifecycle(catalog):
    mono = VersionedModularMonolith(catalog)
    mono.reset_metrics()

    action = ProposedAction(
        action_id="act_test",
        action_name="Scale-up Commit Test",
        target_node="node_act_bioreactor",
        required_prerequisites=["node_sensor_ms4", "node_dataset_42"],
    )

    # Initial permit
    assert mono.verify(action).permit is True

    # Shock MS-4
    shock = TelemetryEvent("ev_shock", 100.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "drift")
    mono.ingest(shock)

    # Blocked
    assert mono.verify(action).permit is False

    # Route wakes safety agent
    woken, _ = mono.route(shock)
    assert "agent_instrumentation" in woken or "agent_executive_safety" in woken


def test_network_ipc_loopback(catalog):
    net_arch = FragmentedNetworkArchitecture(catalog)
    try:
        shock = TelemetryEvent("ev_net_1", 100.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "drift")
        v = net_arch.ingest(shock)
        assert v > 1
        assert net_arch.actual_wire_bytes > 0
        assert net_arch.socket_syscall_count > 0

        action = ProposedAction("act_net", "Commit", "node_act_bioreactor", ["node_sensor_ms4"])
        ver = net_arch.verify(action)
        assert ver.permit is False
    finally:
        net_arch.shutdown()


def test_service7_explain_risk(catalog):
    # Setup shock on unified substrate
    sub_u = UnifiedContextSubstrate(catalog)
    shock = TelemetryEvent("ev_s7", 100.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "severe calibration drift")
    sub_u.ingest(shock)

    # Explain risk on downstream dataset
    exp_u = explain_risk_unified(sub_u, "ds_proteomics_spectra")
    assert exp_u.data_stores_queried == 1
    assert exp_u.root_anomaly_id == "node_sensor_ms4"
    assert exp_u.trigger_event_id == "ev_s7"

    # Setup shock on fragmented architecture
    arch_f = FragmentedProductionArchitecture(catalog)
    arch_f.ingest(shock)
    exp_f = explain_risk_fragmented(
        state_store=arch_f.state_store,
        graph_adj_reverse=arch_f.reverse_adj,
        aspect_vectors=arch_f.aspect_vectors,
        event_bus_log=arch_f.event_bus_log,
        entity_to_doc=arch_f.entity_to_doc,
        doc_to_node=arch_f.doc_to_node,
        doc_to_entity=arch_f.doc_to_entity,
        node_to_doc=arch_f.node_to_doc,
        entity_id="ds_proteomics_spectra",
        v_state=arch_f.v_state,
        v_graph=arch_f.v_graph,
        v_vector=arch_f.v_vector,
        v_bus=arch_f.global_version,
    )
    assert exp_f.data_stores_queried == 4
    assert exp_f.root_anomaly_id == "node_sensor_ms4"
    assert exp_f.trigger_event_id == "ev_s7"
