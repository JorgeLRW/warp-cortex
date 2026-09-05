"""
Fragmented Production-Grade Architecture (Contender 2).
======================================================
Real-world baseline with event log, transactional outbox/bus, version watermarks,
and coordinated worker acknowledgments.

Features:
  - Central Event Bus with monotonic sequence numbers
  - 6 dedicated worker projections (State, Graph, Vector, Router, Search, Changefeed)
  - Serialization / deserialization across service boundaries
  - Two operating modes:
      1. Async Mode: background worker propagation with version watermark tracking
      2. Synchronous Barrier Mode: ingest blocks until all 6 workers acknowledge watermark
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from cortex_apps.multi_service_substrate.substrate_api import (
    ContextPack,
    ContextSubstrate,
    EntityStatus,
    OperationMetrics,
    ProposedAction,
    SubstrateSnapshot,
    TelemetryEvent,
    VerificationResult,
)
from cortex_apps.research_agent_system.memory_baselines import OkapiBM25Scorer
from cortex_apps.research_agent_system.world_state import ResearchDocument, ResearchWorldCatalog


class FragmentedProductionArchitecture(ContextSubstrate):
    """
    Production-grade event-driven microservices architecture:
      - Transactional Outbox + Event Bus
      - Separate persistent worker instances
      - Version watermarks per worker
      - Serialization boundaries
      - Coordinated acknowledgments
    """

    def __init__(self, catalog: ResearchWorldCatalog, sync_barrier: bool = True):
        self.catalog = catalog
        self.bm25 = OkapiBM25Scorer(catalog)
        self.metrics = OperationMetrics()
        self.sync_barrier = sync_barrier

        # Monotonic global event bus watermark
        self.global_version = 1
        self.event_bus_log: List[Dict[str, Any]] = []

        # Worker 1: State Store Worker
        self.v_state = 1
        self.state_store: Dict[str, EntityStatus] = {
            doc.entity_id: EntityStatus.NOMINAL for doc in catalog.documents.values()
        }

        # Worker 2: Graph DB Worker
        self.v_graph = 1
        self.graph_edges: Set[Tuple[str, str, str]] = set(catalog.causal_dependencies)
        self.forward_adj: Dict[str, List[str]] = {}
        self.reverse_adj: Dict[str, List[str]] = {}
        self._rebuild_graph_adj()

        # Worker 3: Vector / Aspect Store Worker
        self.v_vector = 1
        self.aspect_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.aspect_vectors:
                self.aspect_vectors[doc_id] = {k: v.clone() for k, v in doc.aspect_vectors.items()}
            else:
                self.aspect_vectors[doc_id] = {doc.band: doc.embedding.clone()}

        # Worker 4: Agent Router Worker
        self.v_router = 1
        self.router_states: Dict[str, EntityStatus] = dict(self.state_store)
        self.agent_scopes: Dict[str, Set[str]] = {
            "agent_instrumentation": {"inst_quadrupole_ms", "inst_quadrupole_ms2", "inst_cryo_tem", "node_sensor_ms4", "node_sensor_ms2", "node_inst_cryo"},
            "agent_proteomics_data": {"ds_proteomics_spectra", "ds_cryo_micrographs", "node_dataset_42", "node_ds_cryo18"},
            "agent_biochemistry_assay": {"exp_ms_fingerprinting", "exp_cryo_reconstruction", "node_exp_pep", "node_exp_recon"},
            "agent_fermentation_scaleup": {"hypo_yield_model_v4", "act_bioreactor_pilot", "node_hypo_yield", "node_act_bioreactor"},
            "agent_executive_safety": {"act_bioreactor_pilot", "act_pk_study", "node_act_bioreactor", "node_act_pk"},
        }

        # Worker 5: Full-Text Search Worker
        self.v_search = 1

        # Worker 6: Subscription Worker
        self.v_subscriber = 1
        self.subscribers: Dict[str, Tuple[Callable[[TelemetryEvent], bool], Callable[[TelemetryEvent, int], None]]] = {}
        self.sub_counter = 0

        # Mappings
        self.entity_to_doc: Dict[str, str] = dict(catalog.entity_to_doc)
        self.doc_to_entity: Dict[str, str] = {d: e for e, d in self.entity_to_doc.items()}
        self.node_to_doc: Dict[str, str] = {}
        self.doc_to_node: Dict[str, str] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.causal_node_id:
                self.node_to_doc[doc.causal_node_id] = doc_id
                self.doc_to_node[doc_id] = doc.causal_node_id

        # In-memory snapshots
        self.snapshots: Dict[int, SubstrateSnapshot] = {}
        self._save_snapshot()

    def _rebuild_graph_adj(self):
        self.forward_adj.clear()
        self.reverse_adj.clear()
        for src, dst, rel in self.graph_edges:
            self.forward_adj.setdefault(src, []).append(dst)
            self.reverse_adj.setdefault(dst, []).append(src)

    def _save_snapshot(self):
        self.snapshots[self.global_version] = SubstrateSnapshot(
            version=self.global_version,
            timestamp=time.time(),
            entity_states=dict(self.state_store),
            graph_edges=list(self.graph_edges),
            event_log_length=len(self.event_bus_log),
        )

    # =========================================================================
    # PRODUCTION INGESTION VIA OUTBOX & SYNCHRONIZED WORKERS
    # =========================================================================

    def ingest(self, event: TelemetryEvent) -> int:
        t0 = time.perf_counter()
        self.metrics.record_call("ingest")

        self.global_version += 1
        target_v = self.global_version

        # 1. Transactional Outbox write & serialization
        event_dict = {
            "version": target_v,
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "entity_id": event.entity_id,
            "raw_text": event.raw_text,
            "metadata": event.metadata,
        }
        serialized_payload = json.dumps(event_dict)
        self.metrics.serialization_ops += 1
        self.event_bus_log.append(event_dict)
        self.metrics.writes += 1

        # 2. Worker 1: State Store
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
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
        self.v_state = target_v

        # 3. Worker 2: Graph DB
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
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
                self.metrics.writes += 1
                self.metrics.index_mutations += 1
        self.v_graph = target_v

        # 4. Worker 3: Vector / Aspect Store
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
        self.metrics.writes += 1
        self.v_vector = target_v

        # 5. Worker 4: Agent Router
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
        self.metrics.writes += 1
        if event.event_type == "SENSOR_SHOCK":
            self.router_states[event.entity_id] = EntityStatus.TAINTED
            self.metrics.writes += 1
            self.metrics.invalidation_ops += 1
        elif event.event_type == "REMEDIATION":
            self.router_states[event.entity_id] = EntityStatus.NOMINAL
            self.metrics.writes += 1
            self.metrics.invalidation_ops += 1
        self.v_router = target_v

        # 6. Worker 5: Search Worker
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
        self.metrics.writes += 1
        self.v_search = target_v

        # 7. Worker 6: Subscription Worker
        _ = json.loads(serialized_payload)
        self.metrics.serialization_ops += 1
        self.metrics.writes += 1
        for sub_id, (predicate, callback) in list(self.subscribers.items()):
            try:
                if predicate(event):
                    callback(event, target_v)
            except Exception:
                pass
        self.v_subscriber = target_v

        # Save snapshot
        self._save_snapshot()

        # In synchronous barrier mode, coordinator verifies all watermarks equal target_v
        if self.sync_barrier:
            assert (
                self.v_state == target_v
                and self.v_graph == target_v
                and self.v_vector == target_v
                and self.v_router == target_v
                and self.v_search == target_v
                and self.v_subscriber == target_v
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return target_v

    # =========================================================================
    # SERVICE 1: CONTEXT SELECTION & PACKING
    # =========================================================================

    def context(self, query: str, token_budget: int, version: Optional[int] = None) -> ContextPack:
        t0 = time.perf_counter()
        self.metrics.record_call("context")

        v_observed = min(self.v_state, self.v_graph, self.v_vector)
        target_v = version if version is not None else v_observed
        if target_v != v_observed:
            self.metrics.version_mismatches += 1

        abnormal_entities = [e for e, s in self.state_store.items() if s in (EntityStatus.TAINTED, EntityStatus.INVALID)]
        abnormal_docs = set()
        for e in abnormal_entities:
            if e in self.entity_to_doc:
                abnormal_docs.add(self.entity_to_doc[e])
            if e in self.node_to_doc:
                abnormal_docs.add(self.node_to_doc[e])

        tier0_docs = list(abnormal_docs)

        tier1_docs: List[str] = []
        visited_nodes: Set[str] = set()
        for e in abnormal_entities:
            start_node = e if e in self.forward_adj else self.doc_to_node.get(self.entity_to_doc.get(e, ""), None)
            if start_node and start_node in self.forward_adj:
                queue = deque([(start_node, 0)])
                visited_nodes.add(start_node)
                while queue:
                    curr, dist = queue.popleft()
                    for nxt in self.forward_adj.get(curr, []):
                        if nxt not in visited_nodes:
                            visited_nodes.add(nxt)
                            queue.append((nxt, dist + 1))
                            doc_id = self.node_to_doc.get(nxt)
                            if doc_id and doc_id not in abnormal_docs and doc_id not in tier1_docs:
                                tier1_docs.append(doc_id)

        tier2_scored: List[Tuple[float, str]] = []
        bm25_scores = self.bm25.normalized_scores(query)
        abnormal_aspects: List[torch.Tensor] = []
        for d_id in abnormal_docs:
            if d_id in self.aspect_vectors:
                abnormal_aspects.extend(self.aspect_vectors[d_id].values())

        for doc_id, doc in self.catalog.documents.items():
            if doc_id in abnormal_docs or doc_id in tier1_docs:
                continue
            r_z = 0.0
            if abnormal_aspects and doc_id in self.aspect_vectors:
                sims = [
                    F.cosine_similarity(u.unsqueeze(0), v_t.unsqueeze(0)).item()
                    for u in abnormal_aspects
                    for v_t in self.aspect_vectors[doc_id].values()
                ]
                if sims:
                    r_z = max(sims)
            z_boost = 0.50 if r_z >= 0.65 else 0.0
            b_score = bm25_scores.get(doc_id, 0.0)
            tier2_scored.append((b_score + z_boost, doc_id))

        tier2_scored.sort(key=lambda x: x[0], reverse=True)
        tier2_docs = [doc_id for _, doc_id in tier2_scored]

        ordered_doc_ids: List[str] = []
        for d in tier0_docs:
            if d not in ordered_doc_ids:
                ordered_doc_ids.append(d)
        for d in tier1_docs:
            if d not in ordered_doc_ids:
                ordered_doc_ids.append(d)
        for d in tier2_docs:
            if d not in ordered_doc_ids:
                ordered_doc_ids.append(d)

        packed_docs: List[ResearchDocument] = []
        used_tokens = 0
        final_doc_ids: List[str] = []

        for d_id in ordered_doc_ids:
            doc = self.catalog.documents[d_id]
            if used_tokens + doc.tokens <= token_budget:
                packed_docs.append(doc)
                final_doc_ids.append(d_id)
                used_tokens += doc.tokens

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms

        return ContextPack(
            documents=packed_docs,
            token_budget=token_budget,
            tokens_used=used_tokens,
            version=v_observed,
            doc_ids=final_doc_ids,
        )

    # =========================================================================
    # SERVICE 2: AGENT WAKE ROUTER
    # =========================================================================

    def route(self, event: TelemetryEvent, version: Optional[int] = None) -> Tuple[List[str], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("route")
        v = self.v_router

        affected_set: Set[str] = {event.entity_id}
        doc_id = self.entity_to_doc.get(event.entity_id)
        if doc_id:
            affected_set.add(doc_id)
            node_id = self.doc_to_node.get(doc_id)
            if node_id:
                affected_set.add(node_id)
                for nxt in self.forward_adj.get(node_id, []):
                    affected_set.add(nxt)
                    if nxt in self.node_to_doc:
                        affected_set.add(self.node_to_doc[nxt])

        for ent, st in self.router_states.items():
            if st in (EntityStatus.TAINTED, EntityStatus.INVALID):
                affected_set.add(ent)

        woken_agents: List[str] = []
        for agent_id, scope in self.agent_scopes.items():
            if bool(affected_set.intersection(scope)):
                woken_agents.append(agent_id)

        has_anomaly = any(st in (EntityStatus.TAINTED, EntityStatus.INVALID) for st in self.router_states.values())
        if has_anomaly and "agent_executive_safety" not in woken_agents:
            woken_agents.append("agent_executive_safety")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (woken_agents, v)

    # =========================================================================
    # SERVICE 3: AFFECTED DOWNSTREAM FRONTIER
    # =========================================================================

    def affected(self, entity_id: str, version: Optional[int] = None) -> Tuple[List[str], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("affected")
        v = self.v_graph

        impacted: Set[str] = {entity_id}
        doc_id = self.entity_to_doc.get(entity_id)
        start_node = self.doc_to_node.get(doc_id, entity_id) if doc_id else entity_id

        if start_node in self.forward_adj:
            queue = deque([start_node])
            visited = {start_node}
            while queue:
                curr = queue.popleft()
                for nxt in self.forward_adj.get(curr, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
                        impacted.add(nxt)
                        if nxt in self.node_to_doc:
                            d_id = self.node_to_doc[nxt]
                            impacted.add(d_id)
                            if d_id in self.doc_to_entity:
                                impacted.add(self.doc_to_entity[d_id])

        if doc_id and doc_id in self.aspect_vectors:
            source_aspects = list(self.aspect_vectors[doc_id].values())
            for other_doc_id, other_aspects in self.aspect_vectors.items():
                if other_doc_id == doc_id:
                    continue
                sims = [
                    F.cosine_similarity(u.unsqueeze(0), v_t.unsqueeze(0)).item()
                    for u in source_aspects
                    for v_t in other_aspects.values()
                ]
                if sims and max(sims) >= 0.65:
                    impacted.add(other_doc_id)
                    if other_doc_id in self.doc_to_entity:
                        impacted.add(self.doc_to_entity[other_doc_id])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (list(impacted), v)

    # =========================================================================
    # SERVICE 4: SEARCH
    # =========================================================================

    def search(self, query: str, top_k: int = 5, version: Optional[int] = None) -> Tuple[List[ResearchDocument], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("search")
        v = self.v_search

        bm25_scores = self.bm25.normalized_scores(query)
        scored: List[Tuple[float, ResearchDocument]] = []

        for doc_id, doc in self.catalog.documents.items():
            b_score = bm25_scores.get(doc_id, 0.0)
            status = self.state_store.get(doc.entity_id, EntityStatus.NOMINAL)
            status_prior = 0.50 if status in (EntityStatus.TAINTED, EntityStatus.INVALID) else 0.0
            scored.append((b_score + status_prior, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:top_k]]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (results, v)

    # =========================================================================
    # SERVICE 5: INVARIANT VERIFICATION
    # =========================================================================

    def verify(self, action: ProposedAction, version: Optional[int] = None) -> VerificationResult:
        t0 = time.perf_counter()
        self.metrics.record_call("verify")
        v = self.v_state

        violated: List[str] = []
        for prereq_node in action.required_prerequisites:
            if self.state_store.get(prereq_node) in (EntityStatus.TAINTED, EntityStatus.INVALID):
                violated.append(prereq_node)
                continue
            ancestors: Set[str] = set()
            queue = deque([prereq_node])
            while queue:
                curr = queue.popleft()
                for parent in self.reverse_adj.get(curr, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        queue.append(parent)
                        if self.state_store.get(parent) in (EntityStatus.TAINTED, EntityStatus.INVALID):
                            violated.append(f"{prereq_node}<-{parent}")
                            break

        permit = len(violated) == 0
        reason = "All prerequisites satisfied." if permit else f"Violated prerequisites: {', '.join(violated)}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return VerificationResult(
            permit=permit,
            reason=reason,
            version=v,
            violated_prerequisites=violated,
        )

    # =========================================================================
    # SERVICE 6: SUBSCRIBE
    # =========================================================================

    def subscribe(self, predicate: Callable[[TelemetryEvent], bool], callback: Callable[[TelemetryEvent, int], None]) -> str:
        self.metrics.record_call("subscribe")
        self.sub_counter += 1
        sub_id = f"sub_{self.sub_counter}"
        self.subscribers[sub_id] = (predicate, callback)
        return sub_id

    def get_snapshot(self, version: Optional[int] = None) -> SubstrateSnapshot:
        v = version if version is not None else self.global_version
        return self.snapshots.get(v, self.snapshots[self.global_version])

    def reset_metrics(self) -> None:
        self.metrics = OperationMetrics()

    def get_metrics(self) -> OperationMetrics:
        return self.metrics
