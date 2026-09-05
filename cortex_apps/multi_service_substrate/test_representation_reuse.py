"""
Test Suite: Representation Reuse & Scale Validation.
=====================================================
Verifies:
  1. RepresentationMatchedMonolith decision equivalence to UnifiedContextSubstrate.
  2. Scale catalog DAG synthesis.
  3. Memory profiling tensor tracking.
  4. Fault injection lag detection.
"""

import pytest

from cortex_apps.multi_service_substrate.fault_injection_test import (
    LaggingFragmentedArchitecture,
    check_explanation_consistency,
)
from cortex_apps.multi_service_substrate.memory_profiler import (
    inspect_tensor_duplication,
    profile_architecture_memory,
)
from cortex_apps.multi_service_substrate.representation_matched_monolith import (
    RepresentationMatchedMonolith,
)
from cortex_apps.multi_service_substrate.scale_catalog import (
    build_scalable_research_world,
)
from cortex_apps.multi_service_substrate.substrate_api import (
    ProposedAction,
    TelemetryEvent,
)
from cortex_apps.multi_service_substrate.unified_substrate import (
    UnifiedContextSubstrate,
)
from cortex_apps.research_agent_system.world_state import build_research_world


@pytest.fixture
def catalog():
    return build_research_world(seed=42, world_variant="WORLD_A_LINKED")


def test_representation_matched_equivalence(catalog):
    mono = RepresentationMatchedMonolith(catalog)
    sub = UnifiedContextSubstrate(catalog)

    # Initial context selection
    pack_mono = mono.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)
    pack_sub = sub.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)

    assert [d.doc_id for d in pack_mono.documents] == [d.doc_id for d in pack_sub.documents]

    # Ingest shock
    shock = TelemetryEvent("ev_test", 100.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "drift")
    mono.ingest(shock)
    sub.ingest(shock)

    # Post-shock context selection: both foreground abnormal doc
    post_mono = mono.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)
    post_sub = sub.context("Is Pilot Run Alpha scientifically justified?", token_budget=256)

    assert [d.doc_id for d in post_mono.documents] == [d.doc_id for d in post_sub.documents]
    assert "doc_inst_ms4" in [d.doc_id for d in post_mono.documents]


def test_scale_catalog_topology():
    cat = build_scalable_research_world(n_entities=100, seed=42)
    assert len(cat.documents) >= 100
    assert len(cat.causal_dependencies) > 10
    # Check that aspect vectors exist on datasets
    ds_docs = [d for d in cat.documents.values() if d.aspect_vectors is not None]
    assert len(ds_docs) >= 5


def test_memory_profiler(catalog):
    mono = RepresentationMatchedMonolith(catalog)
    sub = UnifiedContextSubstrate(catalog)

    mem_mono = profile_architecture_memory("Representation-Matched Monolith", mono)
    mem_sub = profile_architecture_memory("Unified Context Substrate", sub)

    assert mem_mono.total_tensor_bytes > 0
    assert mem_sub.total_tensor_bytes > 0
    # Representation-matched monolith duplicates aspect vectors across search and router modules
    assert mem_mono.duplicate_tensor_bytes > mem_sub.duplicate_tensor_bytes


def test_fault_injection_logic(catalog):
    lag_arch = LaggingFragmentedArchitecture(catalog, tau_graph=2, tau_bus=2)
    shock = TelemetryEvent("ev_shock_1", 100.0, "SENSOR_SHOCK", "inst_quadrupole_ms", "drift")
    lag_arch.ingest(shock)

    # State is updated immediately to tainted, but bus and graph lag
    assert lag_arch.v_state > lag_arch.v_bus
    assert len(lag_arch.event_bus_log) == 0  # Still in pending queue
