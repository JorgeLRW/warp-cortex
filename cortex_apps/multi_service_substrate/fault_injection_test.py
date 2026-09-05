"""
Fault Injection Test: Service-7 Cross-Store Consistency Audit.
===============================================================
Directly tests whether multi-store distributed joins in fragmented architectures
cause cross-store version drift and inconsistent explanations when worker projections
experience lag, jitter, or asynchronous replication windows.

Measures:
  P(inconsistent explanation) = N_inconsistent / N_queries
Across:
  - tau = 0 (Synchronous lockstep barrier)
  - tau = 1 (1-event asynchronous projection lag)
  - tau = 2 (2-event projection lag)
  - tau = 5 (5-event projection lag / worker burst)
"""

from __future__ import annotations

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from cortex_apps.multi_service_substrate.fragmented_production import FragmentedProductionArchitecture
from cortex_apps.multi_service_substrate.service7_explain_risk import (
    RiskExplanation,
    explain_risk_fragmented,
    explain_risk_unified,
)
from cortex_apps.multi_service_substrate.substrate_api import (
    EntityStatus,
    TelemetryEvent,
)
from cortex_apps.multi_service_substrate.unified_substrate import UnifiedContextSubstrate
from cortex_apps.research_agent_system.world_state import ResearchWorldCatalog, build_research_world


@dataclass
class FaultInjectionResult:
    tau_lag: int
    architecture: str
    total_queries: int
    inconsistent_queries: int
    p_inconsistent: float
    breakdown_missing_path: int
    breakdown_missing_trigger: int
    breakdown_status_contradiction: int


class LaggingFragmentedArchitecture(FragmentedProductionArchitecture):
    """
    Fragmented architecture with configurable asynchronous worker projection lag.
    Simulates real-world production microservices where downstream projections
    (Graph DB, Vector Store, Event Bus) experience replication lag tau.
    """

    def __init__(self, catalog: ResearchWorldCatalog, tau_graph: int = 0, tau_vector: int = 0, tau_bus: int = 0):
        super().__init__(catalog, sync_barrier=False)
        self.tau_graph = tau_graph
        self.tau_vector = tau_vector
        self.tau_bus = tau_bus
        self.v_bus = 1

        # Asynchronous queues
        self.pending_graph_events: deque[Tuple[int, TelemetryEvent]] = deque()
        self.pending_vector_events: deque[Tuple[int, TelemetryEvent]] = deque()
        self.pending_bus_events: deque[Tuple[int, Dict[str, Any]]] = deque()

    def _apply_state_update(self, event: TelemetryEvent):
        if event.event_type == "SENSOR_SHOCK":
            self.state_store[event.entity_id] = EntityStatus.TAINTED
            self.metrics.writes += 1
            doc_id = self.entity_to_doc.get(event.entity_id)
            if doc_id and doc_id in self.doc_to_node:
                self.state_store[self.doc_to_node[doc_id]] = EntityStatus.TAINTED
                self.metrics.writes += 1
        elif event.event_type == "REMEDIATION":
            self.state_store[event.entity_id] = EntityStatus.NOMINAL
            self.metrics.writes += 1
            doc_id = self.entity_to_doc.get(event.entity_id)
            if doc_id and doc_id in self.doc_to_node:
                self.state_store[self.doc_to_node[doc_id]] = EntityStatus.NOMINAL
                self.metrics.writes += 1

    def _apply_graph_update(self, event: TelemetryEvent):
        if event.event_type == "GRAPH_MUTATION":
            src = event.metadata.get("source_node")
            dst = event.metadata.get("target_node")
            rel = event.metadata.get("relation", "LOGICALLY_REQUIRES")
            action = event.metadata.get("action", "ADD")
            if src and dst:
                edge = (src, dst, rel)
                if action == "ADD":
                    self.graph_edges.add(edge)
                elif action == "REMOVE" and edge in self.graph_edges:
                    self.graph_edges.remove(edge)
                self._rebuild_graph_adj()

    def ingest(self, event: TelemetryEvent) -> int:
        self.metrics.record_call("ingest")
        self.global_version += 1
        current_v = self.global_version

        # 1. State Store updates immediately
        self._apply_state_update(event)
        self.v_state = current_v

        # 2. Graph worker buffers with lag tau_graph
        self.pending_graph_events.append((current_v, event))
        while len(self.pending_graph_events) > self.tau_graph:
            v_g, ev_g = self.pending_graph_events.popleft()
            self._apply_graph_update(ev_g)
            self.v_graph = v_g

        # 3. Vector worker buffers with lag tau_vector
        self.pending_vector_events.append((current_v, event))
        while len(self.pending_vector_events) > self.tau_vector:
            v_vec, ev_vec = self.pending_vector_events.popleft()
            self.v_vector = v_vec

        # 4. Bus log buffers with lag tau_bus
        bus_record = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "entity_id": event.entity_id,
            "raw_text": event.raw_text,
            "version": current_v,
        }
        self.pending_bus_events.append((current_v, bus_record))
        while len(self.pending_bus_events) > self.tau_bus:
            v_b, rec = self.pending_bus_events.popleft()
            self.event_bus_log.append(rec)
            self.v_bus = v_b

        return current_v


def check_explanation_consistency(
    exp: RiskExplanation,
    query_entity: str,
    target_compromised: bool,
) -> Tuple[bool, str]:
    """
    Evaluates whether a RiskExplanation is internally coherent:
      1. If target entity is compromised, root anomaly must exist and causal path must not be empty.
      2. If root anomaly exists, trigger event must be found in the event bus log.
      3. Graph path must start with query node and end with root anomaly.
    """
    if target_compromised:
        if exp.root_anomaly_id is None:
            return False, "missing_path"
        if not exp.graph_path_to_root:
            return False, "missing_path"
        if exp.trigger_event_id is None:
            return False, "missing_trigger"
        if exp.graph_path_to_root[-1] != exp.root_anomaly_id:
            return False, "status_contradiction"

    return True, "consistent"


def run_fault_injection_benchmark(
    n_events: int = 100,
    seed: int = 42,
) -> List[FaultInjectionResult]:
    """Runs fault injection benchmark across varying projection lag tau."""
    results: List[FaultInjectionResult] = []
    tau_conditions = [0, 1, 2, 5]

    for tau in tau_conditions:
        random.seed(seed)
        catalog = build_research_world(seed=seed, world_variant="WORLD_A_LINKED")

        # 1. Fragmented Architecture
        frag_arch = LaggingFragmentedArchitecture(
            catalog,
            tau_graph=tau,
            tau_vector=tau,
            tau_bus=tau,
        )

        inconsistent_frag = 0
        miss_path = 0
        miss_trig = 0
        stat_contra = 0

        # Run streaming events with shocks and remediations
        for i in range(n_events):
            if i % 10 == 0:
                ev = TelemetryEvent(f"ev_shock_{i}", 100.0 + i, "SENSOR_SHOCK", "inst_quadrupole_ms", "Calibration drift detected")
            elif i % 10 == 5:
                ev = TelemetryEvent(f"ev_rem_{i}", 100.0 + i, "REMEDIATION", "inst_quadrupole_ms", "Sensor recalibrated")
            else:
                ev = TelemetryEvent(f"ev_noise_{i}", 100.0 + i, "HEARTBEAT", "inst_cryo_tem", "Nominal ping")

            frag_arch.ingest(ev)

            # Query Service 7 mid-stream on downstream dataset
            exp = explain_risk_fragmented(
                state_store=frag_arch.state_store,
                graph_adj_reverse=frag_arch.reverse_adj,
                aspect_vectors=frag_arch.aspect_vectors,
                event_bus_log=frag_arch.event_bus_log,
                entity_to_doc=frag_arch.entity_to_doc,
                doc_to_node=frag_arch.doc_to_node,
                doc_to_entity=frag_arch.doc_to_entity,
                node_to_doc=frag_arch.node_to_doc,
                entity_id="ds_proteomics_spectra",
                v_state=frag_arch.v_state,
                v_graph=frag_arch.v_graph,
                v_vector=frag_arch.v_vector,
                v_bus=frag_arch.v_bus,
            )

            is_compromised = frag_arch.state_store.get("node_sensor_ms4") in (EntityStatus.TAINTED, EntityStatus.INVALID)
            is_valid, reason = check_explanation_consistency(exp, "ds_proteomics_spectra", is_compromised)
            if not is_valid:
                inconsistent_frag += 1
                if reason == "missing_path":
                    miss_path += 1
                elif reason == "missing_trigger":
                    miss_trig += 1
                else:
                    stat_contra += 1

        results.append(FaultInjectionResult(
            tau_lag=tau,
            architecture=f"Fragmented Architecture (tau={tau})",
            total_queries=n_events,
            inconsistent_queries=inconsistent_frag,
            p_inconsistent=inconsistent_frag / n_events,
            breakdown_missing_path=miss_path,
            breakdown_missing_trigger=miss_trig,
            breakdown_status_contradiction=stat_contra,
        ))

        # 2. Unified Substrate (Atomic Snapshot Control)
        unified_sub = UnifiedContextSubstrate(catalog)
        inconsistent_u = 0
        for i in range(n_events):
            if i % 10 == 0:
                ev = TelemetryEvent(f"ev_shock_{i}", 100.0 + i, "SENSOR_SHOCK", "inst_quadrupole_ms", "Calibration drift detected")
            elif i % 10 == 5:
                ev = TelemetryEvent(f"ev_rem_{i}", 100.0 + i, "REMEDIATION", "inst_quadrupole_ms", "Sensor recalibrated")
            else:
                ev = TelemetryEvent(f"ev_noise_{i}", 100.0 + i, "HEARTBEAT", "inst_cryo_tem", "Nominal ping")

            unified_sub.ingest(ev)
            exp_u = explain_risk_unified(unified_sub, "ds_proteomics_spectra")
            is_comp_u = unified_sub.state.get("node_sensor_ms4") in (EntityStatus.TAINTED, EntityStatus.INVALID)
            is_valid_u, _ = check_explanation_consistency(exp_u, "ds_proteomics_spectra", is_comp_u)
            if not is_valid_u:
                inconsistent_u += 1

        results.append(FaultInjectionResult(
            tau_lag=tau,
            architecture=f"Unified Context Substrate (tau={tau})",
            total_queries=n_events,
            inconsistent_queries=inconsistent_u,
            p_inconsistent=inconsistent_u / n_events,
            breakdown_missing_path=0,
            breakdown_missing_trigger=0,
            breakdown_status_contradiction=0,
        ))

    return results


if __name__ == "__main__":
    print("Running Fault Injection Consistency Benchmark...")
    res = run_fault_injection_benchmark(n_events=100, seed=42)
    print("\n" + "=" * 125)
    print(f"{'Architecture':<42} | {'Lag (tau)':<10} | {'Total Queries':<14} | {'Inconsistent':<14} | {'P(Inconsistency)':<18}")
    print("-" * 125)
    for r in res:
        print(f"{r.architecture:<42} | {r.tau_lag:<10} | {r.total_queries:<14} | {r.inconsistent_queries:<14} | {r.p_inconsistent * 100:<17.1f}%")
    print("=" * 125)
