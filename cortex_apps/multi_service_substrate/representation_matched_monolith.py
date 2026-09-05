"""
Representation-Matched Versioned Modular Monolith (The Pure Reuse Baseline).
=============================================================================
A decisive systems baseline: A single in-memory address space with an authoritative
atomic version boundary, using the EXACT SAME multi-aspect semantic manifold (Z),
aspect prototypes, and max-pooling math as the Unified Context Substrate, BUT structured
as separate modules maintaining their own derived indexes, caches, and internal state copies.

This isolates:
  - Disjoint materializations & module boundaries
  versus
  - Shared in-place representation reuse (U_v = <S_v, G_v, Z, H_v>).
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
from cortex_core.semantic_fabric import SemanticBand


class RepresentationMatchedMonolith(ContextSubstrate):
    """
    Modular Monolith with identical representation expressiveness (S, G, Z, H),
    but separate module materializations:
      - Module 1: State Store & Status Cache
      - Module 2: Graph Adjacency Store & Reachability Cache
      - Module 3: Search Service (independent copy of Z aspect vectors + BM25)
      - Module 4: Agent Router (independent copy of entity & capability Z vectors)
      - Module 5: Frontier Service (independent copy of causal graph + frontier cache)
      - Module 6: Verification Service (independent copy of action prerequisites + status)
      - Module 7: Event Log & In-Memory Subscription Registry
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.metrics = OperationMetrics()
        self.version = 1

        # Mappings
        self.entity_to_doc: Dict[str, str] = dict(catalog.entity_to_doc)
        self.doc_to_entity: Dict[str, str] = {d: e for e, d in self.entity_to_doc.items()}
        self.node_to_doc: Dict[str, str] = {}
        self.doc_to_node: Dict[str, str] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.causal_node_id:
                self.node_to_doc[doc.causal_node_id] = doc_id
                self.doc_to_node[doc_id] = doc.causal_node_id

        # ---------------------------------------------------------------------
        # MODULE 1: State Store & Status Cache
        # ---------------------------------------------------------------------
        self.state_table: Dict[str, EntityStatus] = {
            doc.entity_id: EntityStatus.NOMINAL for doc in catalog.documents.values()
        }
        self.state_cache: Dict[str, EntityStatus] = dict(self.state_table)

        # ---------------------------------------------------------------------
        # MODULE 2: Graph Store & Reachability Cache
        # ---------------------------------------------------------------------
        self.graph_edges: Set[Tuple[str, str, str]] = set(catalog.causal_dependencies)
        self.forward_adj: Dict[str, List[str]] = {}
        self.reverse_adj: Dict[str, List[str]] = {}
        self._rebuild_graph_adj()
        self.graph_reachability_cache: Dict[str, Set[str]] = {}

        # ---------------------------------------------------------------------
        # MODULE 3: Search Service (Separate Materialization of Aspect Vectors Z)
        # ---------------------------------------------------------------------
        self.search_bm25 = OkapiBM25Scorer(catalog)
        self.search_aspect_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.aspect_vectors:
                self.search_aspect_vectors[doc_id] = {
                    b: v.clone() for b, v in doc.aspect_vectors.items()
                }
            else:
                self.search_aspect_vectors[doc_id] = {
                    doc.band: doc.embedding.clone()
                }

        # ---------------------------------------------------------------------
        # MODULE 4: Agent Router (Separate Materialization of Capability & Entity Z)
        # ---------------------------------------------------------------------
        self.router_aspect_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        for doc_id, doc in catalog.documents.items():
            if doc.aspect_vectors:
                self.router_aspect_vectors[doc_id] = {
                    b: v.clone() for b, v in doc.aspect_vectors.items()
                }
            else:
                self.router_aspect_vectors[doc_id] = {
                    doc.band: doc.embedding.clone()
                }

        self.agent_capabilities: Dict[str, Dict[str, torch.Tensor]] = {
            "agent_instrumentation": {
                SemanticBand.INSTRUMENTATION.value: catalog.band_anchors[SemanticBand.INSTRUMENTATION.value].clone(),
                SemanticBand.DATA_VALIDITY.value: catalog.band_anchors[SemanticBand.DATA_VALIDITY.value].clone(),
            },
            "agent_proteomics_data": {
                SemanticBand.DATA_VALIDITY.value: catalog.band_anchors[SemanticBand.DATA_VALIDITY.value].clone(),
                SemanticBand.MECHANISM.value: catalog.band_anchors[SemanticBand.MECHANISM.value].clone(),
            },
            "agent_biochemistry_assay": {
                SemanticBand.MECHANISM.value: catalog.band_anchors[SemanticBand.MECHANISM.value].clone(),
                SemanticBand.MANUFACTURING.value: catalog.band_anchors[SemanticBand.MANUFACTURING.value].clone(),
            },
            "agent_fermentation_scaleup": {
                SemanticBand.MANUFACTURING.value: catalog.band_anchors[SemanticBand.MANUFACTURING.value].clone(),
                SemanticBand.UNIT_ECONOMICS.value: catalog.band_anchors[SemanticBand.UNIT_ECONOMICS.value].clone(),
            },
            "agent_executive_safety": {
                SemanticBand.SAFETY.value: catalog.band_anchors[SemanticBand.SAFETY.value].clone(),
                SemanticBand.UNIT_ECONOMICS.value: catalog.band_anchors[SemanticBand.UNIT_ECONOMICS.value].clone(),
            },
        }

        # ---------------------------------------------------------------------
        # MODULE 5: Frontier Service (Separate Causal Graph Copy)
        # ---------------------------------------------------------------------
        self.frontier_forward_adj: Dict[str, List[str]] = copy.deepcopy(self.forward_adj)
        self.frontier_cache: Dict[str, List[str]] = {}

        # ---------------------------------------------------------------------
        # MODULE 6: Verification Service (Separate Action Prerequisite Copy)
        # ---------------------------------------------------------------------
        self.verification_prereqs: Dict[str, List[str]] = {
            "node_act_bioreactor": ["node_sensor_ms4", "node_dataset_42", "node_exp_pep", "node_hypo_yield"],
            "node_act_pk": ["node_inst_cryo", "node_ds_cryo18", "node_exp_recon", "node_hypo_allosteric"],
        }
        self.verification_cache: Dict[str, bool] = {}

        # ---------------------------------------------------------------------
        # MODULE 7: Event Log & Subscribers
        # ---------------------------------------------------------------------
        self.event_log: List[TelemetryEvent] = []
        self.subscribers: Dict[str, Tuple[Callable[[TelemetryEvent], bool], Callable[[TelemetryEvent, int], None]]] = {}
        self.sub_counter = 0

        # Snapshot store
        self.snapshots: Dict[int, SubstrateSnapshot] = {}
        self._save_snapshot()

    def _rebuild_graph_adj(self):
        self.forward_adj.clear()
        self.reverse_adj.clear()
        for src, dst, rel in self.graph_edges:
            self.forward_adj.setdefault(src, []).append(dst)
            self.reverse_adj.setdefault(dst, []).append(src)

    def _save_snapshot(self):
        self.snapshots[self.version] = SubstrateSnapshot(
            version=self.version,
            timestamp=time.time(),
            entity_states=dict(self.state_table),
            graph_edges=list(self.graph_edges),
            event_log_length=len(self.event_log),
        )

    # =========================================================================
    # INGESTION (Update Fan-out across all 7 module materializations)
    # =========================================================================

    def ingest(self, event: TelemetryEvent) -> int:
        t0 = time.perf_counter()
        self.metrics.record_call("ingest")

        # 1. Update Event Log
        self.event_log.append(event)
        self.metrics.writes += 1

        # 2. Fan-out Update to State Module
        if event.event_type == "SENSOR_SHOCK":
            self.state_table[event.entity_id] = EntityStatus.TAINTED
            self.state_cache[event.entity_id] = EntityStatus.TAINTED
            self.metrics.writes += 1
            doc_id = self.entity_to_doc.get(event.entity_id)
            if doc_id and doc_id in self.doc_to_node:
                node = self.doc_to_node[doc_id]
                self.state_table[node] = EntityStatus.TAINTED
                self.state_cache[node] = EntityStatus.TAINTED
                self.metrics.writes += 1

        elif event.event_type == "REMEDIATION":
            self.state_table[event.entity_id] = EntityStatus.NOMINAL
            self.state_cache[event.entity_id] = EntityStatus.NOMINAL
            self.metrics.writes += 1
            doc_id = self.entity_to_doc.get(event.entity_id)
            if doc_id and doc_id in self.doc_to_node:
                node = self.doc_to_node[doc_id]
                self.state_table[node] = EntityStatus.NOMINAL
                self.state_cache[node] = EntityStatus.NOMINAL
                self.metrics.writes += 1

        # 3. Fan-out Update to Graph Module & Frontier Module
        elif event.event_type == "GRAPH_MUTATION":
            src = event.metadata.get("source_node")
            dst = event.metadata.get("target_node")
            rel = event.metadata.get("relation", "LOGICALLY_REQUIRES")
            action = event.metadata.get("action", "ADD")
            if src and dst:
                edge = (src, dst, rel)
                if action == "ADD":
                    self.graph_edges.add(edge)
                    self.frontier_forward_adj.setdefault(src, []).append(dst)
                elif action == "REMOVE" and edge in self.graph_edges:
                    self.graph_edges.remove(edge)
                    if src in self.frontier_forward_adj and dst in self.frontier_forward_adj[src]:
                        self.frontier_forward_adj[src].remove(dst)
                self._rebuild_graph_adj()
                self.metrics.writes += 1

        # Invalidate module-level derived caches
        self.graph_reachability_cache.clear()
        self.frontier_cache.clear()
        self.verification_cache.clear()

        self.version += 1
        self._save_snapshot()

        # Notify subscribers
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
    # SERVICE 1: CONTEXT SELECTION & PACKING (Representation-Matched)
    # Uses IDENTICAL multi-aspect max pooling to Unified Substrate
    # =========================================================================

    def context(self, query: str, token_budget: int, version: Optional[int] = None) -> ContextPack:
        t0 = time.perf_counter()
        self.metrics.record_call("context")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state_table

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

        # Tier 2: Static Z Semantic Frontier (Identical multi-aspect max-pooling math)
        tier2_scored: List[Tuple[float, str]] = []
        bm25_scores = self.search_bm25.normalized_scores(query)

        # Gather abnormal aspect prototypes from Search Module's copy
        abnormal_aspects: List[torch.Tensor] = []
        for d_id in abnormal_docs:
            if d_id in self.search_aspect_vectors:
                abnormal_aspects.extend(self.search_aspect_vectors[d_id].values())

        for doc_id, doc in self.catalog.documents.items():
            if doc_id in abnormal_docs or doc_id in tier1_docs:
                continue

            r_z = 0.0
            if abnormal_aspects and doc_id in self.search_aspect_vectors:
                sims = []
                for u in abnormal_aspects:
                    for v_t in self.search_aspect_vectors[doc_id].values():
                        sims.append(F.cosine_similarity(u.unsqueeze(0), v_t.unsqueeze(0)).item())
                if sims:
                    r_z = max(sims)

            z_boost = 0.50 if r_z >= 0.65 else 0.0
            b_score = bm25_scores.get(doc_id, 0.0)
            score = b_score + z_boost
            tier2_scored.append((score, doc_id))

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
            version=v,
            doc_ids=final_doc_ids,
        )

    # =========================================================================
    # SERVICE 2: AGENT WAKE ROUTING (Representation-Matched)
    # Uses Router Module's Z vectors
    # =========================================================================

    def route(self, event: TelemetryEvent) -> Tuple[List[str], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("route")

        ev_vec: Optional[torch.Tensor] = None
        doc_id = self.entity_to_doc.get(event.entity_id)
        if doc_id and doc_id in self.router_aspect_vectors:
            aspects = list(self.router_aspect_vectors[doc_id].values())
            if aspects:
                ev_vec = aspects[0]

        if ev_vec is None:
            ev_vec = self.catalog.band_anchors.get(
                SemanticBand.INSTRUMENTATION.value,
                torch.zeros(self.catalog.hidden_dim),
            )

        woken: List[str] = []
        for agent_id, cap_dict in self.agent_capabilities.items():
            max_sim = 0.0
            for cap_band, cap_vec in cap_dict.items():
                sim = F.cosine_similarity(ev_vec.unsqueeze(0), cap_vec.unsqueeze(0)).item()
                if sim > max_sim:
                    max_sim = sim
            if max_sim >= 0.55:
                woken.append(agent_id)

        if not woken:
            woken = ["agent_executive_safety"]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return woken, self.version

    # =========================================================================
    # SERVICE 3: AFFECTED DOWNSTREAM FRONTIER
    # Uses Frontier Module's separate causal graph
    # =========================================================================

    def affected(self, entity_id: str, version: Optional[int] = None) -> Tuple[List[str], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("affected")
        v = version if version is not None else self.version

        # Check cache
        if entity_id in self.frontier_cache:
            res = list(self.frontier_cache[entity_id])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.cpu_time_ms += elapsed_ms
            return (res, v)

        doc_id = self.entity_to_doc.get(entity_id)
        start_node = self.doc_to_node.get(doc_id, entity_id) if doc_id else entity_id

        affected_ents: List[str] = []
        queue = deque([start_node])
        visited = {start_node}
        while queue:
            curr = queue.popleft()
            for nxt in self.frontier_forward_adj.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    target_ent = self.doc_to_entity.get(self.node_to_doc.get(nxt, ""), nxt)
                    affected_ents.append(target_ent)

        self.frontier_cache[entity_id] = list(affected_ents)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.cpu_time_ms += elapsed_ms
        return (affected_ents, v)

    def affected_frontier(self, entity_id: str, version: Optional[int] = None) -> List[str]:
        res, _ = self.affected(entity_id, version)
        return res

    # =========================================================================
    # SERVICE 4: HYBRID SEARCH
    # Uses Search Module's BM25 and aspect vectors
    # =========================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        version: Optional[int] = None,
    ) -> Tuple[List[ResearchDocument], int]:
        t0 = time.perf_counter()
        self.metrics.record_call("search")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state_table

        bm25_scores = self.search_bm25.normalized_scores(query)
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
    # SERVICE 5: INVARIANT VERIFICATION
    # Uses Verification Module's prerequisite copy & State Module's status
    # =========================================================================

    def verify(self, action: ProposedAction, version: Optional[int] = None) -> VerificationResult:
        t0 = time.perf_counter()
        self.metrics.record_call("verify")
        v = version if version is not None else self.version

        snap = self.snapshots.get(v)
        states = snap.entity_states if snap else self.state_table

        violated: List[str] = []
        for prereq_node in action.required_prerequisites:
            if states.get(prereq_node) in (EntityStatus.TAINTED, EntityStatus.INVALID):
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

    def subscribe(
        self,
        predicate: Callable[[TelemetryEvent], bool],
        callback: Callable[[TelemetryEvent, int], None],
    ) -> str:
        self.metrics.record_call("subscribe")
        self.sub_counter += 1
        sub_id = f"sub_mono_{self.sub_counter}"
        self.subscribers[sub_id] = (predicate, callback)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        if subscription_id in self.subscribers:
            del self.subscribers[subscription_id]
            return True
        return False

    def get_snapshot(self, version: Optional[int] = None) -> SubstrateSnapshot:
        v = version if version is not None else self.version
        return self.snapshots.get(v, self.snapshots[self.version])

    def reset_metrics(self) -> None:
        self.metrics = OperationMetrics()

    def get_metrics(self) -> OperationMetrics:
        return self.metrics
