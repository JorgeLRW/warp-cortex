"""
Unit Tests: Research Agent System Components.
=============================================
Verifies initialization, event ingestion (both raw and structured), retrieval interfaces,
and agent decision logic across all 6 architectures.
"""

import pytest
import torch
import torch.nn.functional as F

from cortex_apps.research_agent_system.world_state import build_research_world
from cortex_apps.research_agent_system.memory_baselines import (
    StatelessRAG,
    EventLogRAG,
    PeriodicSummarizedMemory,
    TemporalGraphRAG,
    ConventionalStateStoreRAG,
    CortexPriorRAG,
)
from cortex_apps.research_agent_system.agent_council import (
    ExecutiveScaleUpAgent,
    DataIntegrityMonitorAgent,
)
from cortex_apps.research_agent_system.event_stream import generate_research_timeline


def test_research_world_creation():
    catalog = build_research_world(hidden_dim=64, seed=42)
    assert len(catalog.documents) >= 45
    assert "doc_inst_ms4" in catalog.documents
    assert "doc_act_bioreactor_pilot" in catalog.documents
    assert len(catalog.causal_dependencies) >= 8


def test_memory_baselines_raw_ingestion():
    catalog = build_research_world(hidden_dim=64, seed=42)
    q_vec = F.normalize(torch.randn(64), dim=0)
    q_text = "Evaluate scale-up authorization"

    # 1. Stateless RAG
    stateless = StatelessRAG(catalog)
    stateless.record_raw_event("e1", "MS-4 drift alert", q_vec, timestamp=1)
    res1 = stateless.query(q_text, q_vec, token_budget=256)
    assert len(res1.items) > 0

    # 2. Event-Log RAG
    event_log = EventLogRAG(catalog)
    event_log.record_raw_event("e1", "ANOMALOUS TELEMETRY: Mass spectrometer drift observed", q_vec, timestamp=1)
    res2 = event_log.query(q_text, q_vec, token_budget=256)
    assert len(res2.items) > 0

    # 3. Periodic Summarized Memory
    periodic = PeriodicSummarizedMemory(catalog, summarize_interval=5)
    for i in range(6):
        periodic.record_raw_event(f"e_{i}", f"Log {i} with alert", q_vec, timestamp=i + 1)
    res3 = periodic.query(q_text, q_vec, token_budget=256)
    assert len(res3.items) > 0
    assert any(it.doc_id == "summary_scratchpad" for it in res3.items)

    # 4. Temporal GraphRAG
    temporal_graph = TemporalGraphRAG(catalog)
    temporal_graph.record_raw_event("e1", "ANOMALOUS TELEMETRY: Mass spectrometer drift observed", q_vec, timestamp=1)
    res4 = temporal_graph.query(q_text, q_vec, token_budget=256)
    assert len(res4.items) > 0

    # 5. Conventional State Store RAG
    state_store = ConventionalStateStoreRAG(catalog)
    state_store.record_raw_event("e1", "ANOMALOUS TELEMETRY: Mass spectrometer drift observed", q_vec, timestamp=1)
    res5 = state_store.query(q_text, q_vec, token_budget=256)
    assert len(res5.items) > 0

    # 6. CortexPriorRAG
    cortex_rag = CortexPriorRAG(catalog)
    cortex_rag.record_raw_event("e1", "ANOMALOUS TELEMETRY: Mass spectrometer drift observed", q_vec, timestamp=1)
    res6 = cortex_rag.query(q_text, q_vec, token_budget=256)
    assert len(res6.items) > 0


def test_memory_baselines_structured_ingestion():
    catalog = build_research_world(hidden_dim=64, seed=42)
    q_vec = F.normalize(torch.randn(64), dim=0)
    q_text = "Evaluate scale-up authorization"

    # All 6 contenders support structured events
    contenders = [
        StatelessRAG(catalog),
        EventLogRAG(catalog),
        PeriodicSummarizedMemory(catalog),
        TemporalGraphRAG(catalog),
        ConventionalStateStoreRAG(catalog),
        CortexPriorRAG(catalog),
    ]

    for c in contenders:
        c.record_structured_event(
            event_id="e_struct_1",
            entity_id="inst_quadrupole_ms",
            status="TAINTED",
            text="MS-4 sensor calibration drift",
            embedding=q_vec,
            timestamp=5,
        )
        res = c.query(q_text, q_vec, token_budget=256)
        assert len(res.items) > 0


def test_event_stream_generation():
    catalog = build_research_world(hidden_dim=64, seed=42)
    events = generate_research_timeline(catalog, seed=42, noise_event_count=50)
    assert len(events) > 50
    shocks = [e for e in events if e.is_shock]
    remeds = [e for e in events if e.is_remediation]
    probes = [e for e in events if e.is_query_probe]
    assert len(shocks) >= 1
    assert len(remeds) >= 1
    assert len(probes) >= 4
    # Check that structured fields exist
    assert shocks[0].entity_id == "inst_quadrupole_ms"
    assert shocks[0].status_update == "TAINTED"


def test_executive_agent_decision():
    agent = ExecutiveScaleUpAgent()
    catalog = build_research_world(hidden_dim=64, seed=42)
    stateless = StatelessRAG(catalog)
    q_vec = catalog.documents["doc_act_bioreactor_pilot"].embedding

    # Pre-shock query (nominal)
    res = stateless.query("Scale-up authorization", q_vec, token_budget=512)
    dec = agent.evaluate_scaleup_request(res, ground_truth_status="NOMINAL")
    assert dec.action == "COMMIT"
    assert dec.is_correct is True
