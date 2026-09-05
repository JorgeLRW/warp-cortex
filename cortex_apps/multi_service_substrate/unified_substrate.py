"""
Unified Context Substrate Implementation.
=========================================
Implements the Unified Context Substrate U_v = <S_v, G_v, Z, H_v>.
All 6 services are direct zero-IPC, zero-synchronization functional projections
over an explicitly versioned snapshot v.
"""

from __future__ import annotations

import copy
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


class UnifiedContextSubstrate(ContextSubstrate):
    """
    Unified Context Substrate maintaining:
      - S_v: Operational entity status table
      - G_v: Structural causal / workflow dependency graph
      - Z: Multi-aspect semantic manifold (static aspect tensors)
      - H_v: Chronological event provenance log
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.bm25 = OkapiBM25Scorer(catalog)
        self.metrics = OperationMetrics()

        self.version = 1
        self.state: Dict[str, EntityStatus] = {}
        self.entity_to_doc: Dict[str, str] = dict(catalog.entity_to_doc)
        self.doc_to_entity: Dict[str, str] = {d: e for e, d in self.entity_to_doc.items()}

        # Initialize entity states
        for doc_id, doc in catalog.documents.items():
            self.state[doc.entity_id] = EntityStatus.NOMINAL

        # Initialize Graph
        self.graph_edges: Set[Tuple[str, str, str]] = set(catalog.causal_dependencies)
        self.forward_adj: Dict[str, List[str]] = {}
        self.reverse_adj: Dict[str, List[str]] = {}
        self._rebuild_graph_adj()

        # Aspect vectors (Z)
        self.aspect_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.aspect_vectors:
                self.aspect_vectors[doc_id] = {k: v.clone() for k, v in doc.aspect_vectors.items()}
            else:
                self.aspect_vectors[doc_id] = {doc.band: doc.embedding.clone()}

        # Node ID to Doc ID mapping
        self.node_to_doc: Dict[str, str] = {}
        self.doc_to_node: Dict[str, str] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.causal_node_id:
                self.node_to_doc[doc.causal_node_id] = doc_id
                self.doc_to_node[doc_id] = doc.causal_node_id

        # Agent scopes for wake routing
        self.agent_scopes: Dict[str, Set[str]] = {
            "agent_instrumentation": {"inst_quadrupole_ms", "inst_quadrupole_ms2", "inst_cryo_tem", "node_sensor_ms4", "node_sensor_ms2", "node_inst_cryo"},
            "agent_proteomics_data": {"ds_proteomics_spectra", "ds_cryo_micrographs", "node_dataset_42", "node_ds_cryo18"},
            "agent_biochemistry_assay": {"exp_ms_fingerprinting", "exp_cryo_reconstruction", "node_exp_pep", "node_exp_recon"},
            "agent_fermentation_scaleup": {"hypo_yield_model_v4", "act_bioreactor_pilot", "node_hypo_yield", "node_act_bioreactor"},
            "agent_executive_safety": {"act_bioreactor_pilot", "act_pk_study", "node_act_bioreactor", "node_act_pk"},
        }

        # Event log and subscribers
        self.event_log: List[TelemetryEvent] = []
        self.subscribers: Dict[str, Tuple[Callable[[TelemetryEvent], bool], Callable[[TelemetryEvent, int], None]]] = {}
        self.sub_counter = 0

        # Saved snapshots
        self.snapshots: Dict[int, SubstrateSnapshot] = {}
        self._save_snapshot()

    def _rebuild_graph_adj(self):
        self.forward_adj.clear()
        self.reverse_adj.clear()
        for src, dst, rel in self.graph_edges:
            self.forward_adj.setdefault(src, []).append(dst)
            self.reverse_adj.setdefault(dst, []).append(src)

    def _save_snapshot(self):
        snap = SubstrateSnapshot(
            version=self.version,
            timestamp=time.time(),
            entity_states=dict(self.state),
            graph_edges=list(self.graph_edges),
            event_log_length=len(self.event_log),
        )
        self.snapshots[self.version] = snap

    # =========================================================================
    # CORE INGESTION
    # =========================================================================

    def ingest(self, event: TelemetryEvent) -> int:
        t0 = time.perf_counter()
        self.metrics.record_call("ingest")

        # 1. Append to event log (1 write)
        self.event_log.append(event)
        self.metrics.writes += 1

        # 2. Update state or topology
        if event.event_type == "SENSOR_SHOCK":
            target_entity = event.entity_id
            if target_entity in self.state:
                self.state[target_entity] = EntityStatus.TAINTED
                self.metrics.writes += 1
            # Also taint associated causal node if present
            doc_id = self.entity_to_doc.get(target_entity)
            if doc_id and doc_id in self.doc_to_node:
                self.state[self.doc_to_node[doc_id]] = EntityStatus.TAINTED
                self.metrics.writes += 1

        elif event.event_type == "REMEDIATION":
            target_entity = event.entity_id
            if target_entity in self.state:
                self.state[target_entity] = EntityStatus.NOMINAL
                self.metrics.writes += 1
            doc_id = self.entity_to_doc.get(target_entity)
            if doc_id and doc_id in self.doc_to_node:
                self.state[self.doc_to_node[doc_id]] = EntityStatus.NOMINAL
                self.metrics.writes += 1

        elif event.event_type == "GRAPH_MUTATION":
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

        # 3. Advance version watermark
        self.version += 1
        self._save_snapshot()

        # 4. Notify matching subscribers
        for sub_id, (predicate, callback) in list(self.subscribers.items()):
            try:
                if predicate(event):
                    callback(event, self.version)
            except Exception:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return self.version

    # =========================================================================
    # SERVICE 1: CONTEXT SELECTION & PACKING (Tiered Context Union)
    # =========================================================================

    def context(self, query: str, token_budget: int, version: Optional[int] = None) -> ContextPack:
        t0 = time.perf_counter()
        self.metrics.record_call("context")
        v = version if version is not None else self.version

        # Active anomalies in snapshot v
        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state

        abnormal_entities = [e for e, s in states.items() if s in (EntityStatus.TAINTED, EntityStatus.INVALID)]
        abnormal_docs = set()
        for e in abnormal_entities:
            if e in self.entity_to_doc:
                abnormal_docs.add(self.entity_to_doc[e])
            if e in self.node_to_doc:
                abnormal_docs.add(self.node_to_doc[e])

        # Tier 0: Mandatory Hard Current State
        tier0_docs = list(abnormal_docs)

        # Tier 1: Explicit Graph Reachability
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

        # Tier 2: Static Z Semantic Frontier + Hybrid BM25
        tier2_scored: List[Tuple[float, str]] = []
        bm25_scores = self.bm25.normalized_scores(query)

        # Gather abnormal aspect prototypes
        abnormal_aspects: List[torch.Tensor] = []
        for d_id in abnormal_docs:
            if d_id in self.aspect_vectors:
                abnormal_aspects.extend(self.aspect_vectors[d_id].values())

        for doc_id, doc in self.catalog.documents.items():
            if doc_id in abnormal_docs or doc_id in tier1_docs:
                continue
            # Static Z similarity to abnormal prototypes
            r_z = 0.0
            if abnormal_aspects and doc_id in self.aspect_vectors:
                sims = []
                for u in abnormal_aspects:
                    for v_t in self.aspect_vectors[doc_id].values():
                        sims.append(F.cosine_similarity(u.unsqueeze(0), v_t.unsqueeze(0)).item())
                if sims:
                    r_z = max(sims)

            z_boost = 0.50 if r_z >= 0.65 else 0.0
            b_score = bm25_scores.get(doc_id, 0.0)
            score = b_score + z_boost
            tier2_scored.append((score, doc_id))

        tier2_scored.sort(key=lambda x: x[0], reverse=True)
        tier2_docs = [doc_id for _, doc_id in tier2_scored]

        # Assemble packed context
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
            version=v,
            doc_ids=final_doc_ids,
        )

    # =========================================================================
    # SERVICE 2: AGENT WAKE ROUTER
    # =========================================================================

    def route(self, event: TelemetryEvent, version: Optional[int] = None) -> Tuple[List[str], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("route")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state

        # Find directly affected entity, graph neighbors, and static Z neighbors
        affected_set: Set[str] = {event.entity_id}
        doc_id = self.entity_to_doc.get(event.entity_id)
        if doc_id:
            affected_set.add(doc_id)
            node_id = self.doc_to_node.get(doc_id)
            if node_id:
                affected_set.add(node_id)
                # Add 1-hop graph neighbors
                for nxt in self.forward_adj.get(node_id, []):
                    affected_set.add(nxt)
                    if nxt in self.node_to_doc:
                        affected_set.add(self.node_to_doc[nxt])

        # Add any actively tainted entities in snapshot v
        for ent, st in states.items():
            if st in (EntityStatus.TAINTED, EntityStatus.INVALID):
                affected_set.add(ent)

        # Match against agent scopes
        woken_agents: List[str] = []
        for agent_id, scope in self.agent_scopes.items():
            if bool(affected_set.intersection(scope)):
                woken_agents.append(agent_id)

        # Always wake safety agent if an active anomaly exists
        has_anomaly = any(st in (EntityStatus.TAINTED, EntityStatus.INVALID) for st in states.values())
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
        v = version if version is not None else self.version

        impacted_entities: Set[str] = {entity_id}
        doc_id = self.entity_to_doc.get(entity_id)
        start_node = self.doc_to_node.get(doc_id, entity_id) if doc_id else entity_id

        # 1. Graph Reachability
        if start_node in self.forward_adj:
            queue = deque([start_node])
            visited = {start_node}
            while queue:
                curr = queue.popleft()
                for nxt in self.forward_adj.get(curr, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
                        impacted_entities.add(nxt)
                        if nxt in self.node_to_doc:
                            d_id = self.node_to_doc[nxt]
                            impacted_entities.add(d_id)
                            if d_id in self.doc_to_entity:
                                impacted_entities.add(self.doc_to_entity[d_id])

        # 2. Static Z Frontier
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
                    impacted_entities.add(other_doc_id)
                    if other_doc_id in self.doc_to_entity:
                        impacted_entities.add(self.doc_to_entity[other_doc_id])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (list(impacted_entities), v)

    # =========================================================================
    # SERVICE 4: HYBRID BM25 + DENSE SEARCH WITH STATUS PRIORS
    # =========================================================================

    def search(self, query: str, top_k: int = 5, version: Optional[int] = None) -> Tuple[List[ResearchDocument], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("search")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state

        bm25_scores = self.bm25.normalized_scores(query)
        scored_docs: List[Tuple[float, ResearchDocument]] = []

        for doc_id, doc in self.catalog.documents.items():
            b_score = bm25_scores.get(doc_id, 0.0)
            status = states.get(doc.entity_id, EntityStatus.NOMINAL)
            status_prior = 0.50 if status in (EntityStatus.TAINTED, EntityStatus.INVALID) else 0.0
            total_score = b_score + status_prior
            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored_docs[:top_k]]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (results, v)

    # =========================================================================
    # SERVICE 5: INVARIANT & PREREQUISITE VERIFICATION
    # =========================================================================

    def verify(self, action: ProposedAction, version: Optional[int] = None) -> VerificationResult:
        t0 = time.perf_counter()
        self.metrics.record_call("verify")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state

        violated: List[str] = []
        for prereq_node in action.required_prerequisites:
            # Check if prerequisite itself is tainted
            if states.get(prereq_node) in (EntityStatus.TAINTED, EntityStatus.INVALID):
                violated.append(prereq_node)
                continue
            # Also check if any upstream ancestor in G_v is tainted
            ancestors: Set[str] = set()
            queue = deque([prereq_node])
            while queue:
                curr = queue.popleft()
                for parent in self.reverse_adj.get(curr, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        queue.append(parent)
                        if states.get(parent) in (EntityStatus.TAINTED, EntityStatus.INVALID):
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
    # SERVICE 6: SUBSCRIPTION CHANGEFEED
    # =========================================================================

    def subscribe(self, predicate: Callable[[TelemetryEvent], bool], callback: Callable[[TelemetryEvent, int], None]) -> str:
        self.metrics.record_call("subscribe")
        self.sub_counter += 1
        sub_id = f"sub_{self.sub_counter}"
        self.subscribers[sub_id] = (predicate, callback)
        return sub_id

    # =========================================================================
    # SNAPSHOTS & METRICS
    # =========================================================================

    def get_snapshot(self, version: Optional[int] = None) -> SubstrateSnapshot:
        v = version if version is not None else self.version
        return self.snapshots.get(v, self.snapshots[self.version])

    def reset_metrics(self) -> None:
        self.metrics = OperationMetrics()

    def get_metrics(self) -> OperationMetrics:
        return self.metrics
