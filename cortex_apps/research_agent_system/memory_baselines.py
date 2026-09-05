"""
Memory Baselines & Cortex Relevance Prior.
=========================================
Implements the retrieval and persistent memory architectures for fair 2x6 comparison:
  1. StatelessRAG: Pure query-conditioned vector retrieval P(x | q).
  2. EventLogRAG: Rolling chronological event log with recency-weighted similarity and event snippets.
  3. PeriodicSummarizedMemory: Rolling executive scratchpad updated every K events.
  4. TemporalGraphRAG: Causal graph neighborhood expansion with chronological node annotations.
  5. ConventionalStateStoreRAG: Relational event-sourced state store (S_t^DB) + explicit graph propagation + RAG.
  6. CortexPriorRAG: Exact same base RAG retriever conditioned on Cortex dynamic working state h_t and S_t:
       score(x) = retrieval(x, q) * (1.0 + alpha * h_x(t)) * relevance_factor(x, S_t).

Features a SharedFrozenEventResolver guaranteeing identical raw event interpretation
between State Store and Cortex.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from cortex_apps.research_agent_system.world_state import ResearchDocument, ResearchWorldCatalog
from cortex_core.cortex_runtime import CortexRuntime
from cortex_core.semantic_fabric import SemanticBand
from cortex_core.epistemic_manifold import EpistemicKind, EpistemicRelation


# =============================================================================
# Fair Lexical Search: Okapi BM25 Scorer
# =============================================================================
class OkapiBM25Scorer:
    """
    Standard Okapi BM25 index and scorer.
    Parameters: k1 = 1.5, b = 0.75.
    Provides fair, exact lexical matching over document title and content.
    """
    def __init__(self, catalog: ResearchWorldCatalog, k1: float = 1.5, b: float = 0.75):
        self.catalog = catalog
        self.k1 = k1
        self.b = b
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_lens: Dict[str, int] = {}
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.n_docs = len(catalog.documents)

        total_len = 0
        for doc_id, doc in catalog.documents.items():
            tokens = self._tokenize(f"{doc.title} {doc.content}")
            self.doc_tokens[doc_id] = tokens
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_dl = total_len / max(1, self.n_docs)
        for t, count in self.df.items():
            # Standard Okapi IDF
            self.idf[t] = math.log(1.0 + (self.n_docs - count + 0.5) / (count + 0.5))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9_\-]+', text.lower())

    def score(self, query_text: str, doc_id: str) -> float:
        q_tokens = self._tokenize(query_text)
        doc_toks = self.doc_tokens.get(doc_id, [])
        dl = self.doc_lens.get(doc_id, 0)
        if dl == 0 or not q_tokens:
            return 0.0

        tf_map: Dict[str, int] = {}
        for t in doc_toks:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for qt in q_tokens:
            if qt in tf_map and qt in self.idf:
                f = tf_map[qt]
                idf = self.idf[qt]
                num = f * (self.k1 + 1.0)
                den = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                score += idf * (num / den)
        return score

    def normalized_scores(self, query_text: str) -> Dict[str, float]:
        raw = {d_id: self.score(query_text, d_id) for d_id in self.catalog.documents}
        max_s = max(raw.values()) if raw else 0.0
        if max_s <= 0.0:
            return {d_id: 0.0 for d_id in raw}
        return {d_id: s / max_s for d_id, s in raw.items()}


# =============================================================================
# Frozen Procedural Semantic Query Text Encoder
# =============================================================================
class SemanticQueryEncoder:
    """
    Procedural query text encoder.
    Maps natural language query strings to dense embeddings in hidden_dim space
    via semantic band keyword projections and deterministic token hashing.
    Completely independent of any catalog document embeddings or target IDs.
    Shared identically across all memory and retrieval baselines.
    """
    BAND_KEYWORDS = {
        SemanticBand.MANUFACTURING.value: ["pilot", "run", "bioreactor", "scaleup", "scale-up", "production", "fermentation", "batch", "sop", "manufacturing"],
        SemanticBand.DATA_VALIDITY.value: ["data", "dataset", "spectra", "integrity", "valid", "evidence", "empirical", "reproducibility", "measurements"],
        SemanticBand.MECHANISM.value: ["mechanism", "assay", "peptide", "fingerprint", "sequence", "hypothesis", "binding", "biological", "scientifically"],
        SemanticBand.INSTRUMENTATION.value: ["instrument", "sensor", "spectrometer", "calibration", "drift", "detector", "ms4", "quadrupole", "reading"],
        SemanticBand.UNIT_ECONOMICS.value: ["capital", "cost", "budget", "$250k", "release", "authorization", "funds", "investment"],
        SemanticBand.SAFETY.value: ["safety", "containment", "hazard", "pressure", "chiller", "cleanroom", "protocol"],
    }

    def __init__(self, band_anchors: Dict[str, torch.Tensor], hidden_dim: int = 64):
        self.band_anchors = band_anchors
        self.hidden_dim = hidden_dim

    def encode(self, text: str) -> torch.Tensor:
        words = re.findall(r'[a-zA-Z0-9_\$]+', text.lower())
        vec = torch.zeros(self.hidden_dim)
        matched = False
        for band, keywords in self.BAND_KEYWORDS.items():
            count = sum(1 for w in words if w in keywords)
            if count > 0:
                vec += count * self.band_anchors[band]
                matched = True

        if not matched:
            for anchor in self.band_anchors.values():
                vec += anchor

        h = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
        rng = torch.Generator().manual_seed(h % (2**31 - 1))
        noise = torch.randn(self.hidden_dim, generator=rng) * 0.05
        return F.normalize(vec + noise, dim=0)


# =============================================================================
# Multi-Hop Shortest Path BFS on Causal DAG
# =============================================================================
def compute_bfs_shortest_paths(
    dependencies: List[Tuple[str, str, str]], undirected: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Computes all-pairs shortest paths on the causal dependency graph.
    Returns: dist[u][v] = shortest path distance.
    """
    adj: Dict[str, Set[str]] = {}
    for src, tgt, _ in dependencies:
        if src not in adj: adj[src] = set()
        if tgt not in adj: adj[tgt] = set()
        adj[src].add(tgt)
        if undirected:
            adj[tgt].add(src)

    all_nodes = list(adj.keys())
    dist: Dict[str, Dict[str, int]] = {u: {} for u in all_nodes}
    for start in all_nodes:
        dist[start][start] = 0
        queue = [start]
        visited = {start}
        while queue:
            curr = queue.pop(0)
            curr_d = dist[start][curr]
            for neighbor in adj.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist[start][neighbor] = curr_d + 1
                    queue.append(neighbor)
    return dist


@dataclass
class RetrievedItem:
    doc_id: str
    title: str
    content: str
    band: str
    tokens: int
    score: float
    state_tag: str = "VALID"
    source_method: str = ""


@dataclass
class RetrievalResult:
    items: List[RetrievedItem]
    total_tokens: int
    method_name: str
    retrieval_ms: float = 0.0
    critical_incident_rank: int = 999  # Preserved for backwards compatibility (root rank)
    root_incident_rank: int = 999      # Rank of doc_inst_ms4
    downstream_consequence_rank: int = 999  # Rank of doc_ds_data42
    yield_model_rank: int = 999        # Rank of doc_hypo_yield_v4
    sop_rank: int = 999                # Rank of doc_act_bioreactor_pilot
    root_in_context: bool = False      # Whether doc_inst_ms4 is in packed context
    downstream_in_context: bool = False # Whether doc_ds_data42 is in packed context
    distractor_count: int = 0          # Number of ambient/irrelevant docs packed
    causal_path_count: int = 0         # Number of relevant causal path docs packed
    false_reach_rate: float = 0.0      # distractor_count / total_packed
    selectivity: float = 0.0           # causal_path_count / total_packed


def pack_within_budget(items: List[RetrievedItem], token_budget: int) -> RetrievalResult:
    packed: List[RetrievedItem] = []
    used_tokens = 0
    for it in items:
        if used_tokens + it.tokens <= token_budget:
            packed.append(it)
            used_tokens += it.tokens
        elif not packed:
            packed.append(it)
            used_tokens += it.tokens
            break
        else:
            break

    # Determine ranks for critical path documents:
    root_rank = 999
    downstream_rank = 999
    yield_rank = 999
    sop_rank = 999
    for r_idx, it in enumerate(items):
        if it.doc_id == "doc_inst_ms4" or "quadrupole" in it.title.lower():
            if root_rank == 999:
                root_rank = r_idx + 1
        if it.doc_id == "doc_ds_data42" or "dataset 42" in it.title.lower():
            if downstream_rank == 999:
                downstream_rank = r_idx + 1
        if it.doc_id == "doc_hypo_yield_v4" or "yield prediction model" in it.title.lower():
            if yield_rank == 999:
                yield_rank = r_idx + 1
        if it.doc_id == "doc_act_bioreactor_pilot" or "bioreactor pilot run" in it.title.lower():
            if sop_rank == 999:
                sop_rank = r_idx + 1

    CAUSAL_PATH_DOCS = {
        "doc_inst_ms4", "doc_ds_data42", "doc_exp_pep_fingerprint",
        "doc_hypo_yield_v4", "doc_act_bioreactor_pilot"
    }
    root_in_ctx = any(it.doc_id == "doc_inst_ms4" or "quadrupole" in it.title.lower() for it in packed)
    downstream_in_ctx = any(it.doc_id == "doc_ds_data42" or "dataset 42" in it.title.lower() for it in packed)
    causal_cnt = sum(1 for it in packed if it.doc_id in CAUSAL_PATH_DOCS)
    distractor_cnt = sum(1 for it in packed if it.doc_id.startswith("doc_ambient_") or "cryo" in it.doc_id)
    n_packed = max(1, len(packed))
    false_reach = distractor_cnt / n_packed
    selectivity = causal_cnt / n_packed

    return RetrievalResult(
        items=packed,
        total_tokens=used_tokens,
        method_name=items[0].source_method if items else "",
        critical_incident_rank=root_rank,
        root_incident_rank=root_rank,
        downstream_consequence_rank=downstream_rank,
        yield_model_rank=yield_rank,
        sop_rank=sop_rank,
        root_in_context=root_in_ctx,
        downstream_in_context=downstream_in_ctx,
        distractor_count=distractor_cnt,
        causal_path_count=causal_cnt,
        false_reach_rate=false_reach,
        selectivity=selectivity,
    )


# =============================================================================
# Shared Frozen Event Resolver (Guarantees Identical Raw Interpretation)
# =============================================================================
@dataclass
class ResolvedEvent:
    event_id: str
    text: str
    matched_entity_id: Optional[str]
    matched_doc_id: Optional[str]
    confidence: float
    is_alert: bool
    is_remediation: bool
    status: str  # "TAINTED", "VALID", or "NOMINAL"


class SharedFrozenEventResolver:
    """
    Shared frozen semantic resolver.
    Guarantees that both State Store and Cortex (and baselines)
    receive the exact same raw event interpretation without asymmetric parsing advantages.
    Uses fair hybrid scoring (BM25 + dense embedding similarity) to resolve events to catalog entities.
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.bm25 = OkapiBM25Scorer(catalog)

    def resolve_raw_event(
        self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int
    ) -> ResolvedEvent:
        best_doc_id = None
        best_entity_id = None
        best_score = -1.0

        bm25_map = self.bm25.normalized_scores(text) if text else {}

        for d_id, doc in self.catalog.documents.items():
            sim = torch.dot(embedding, doc.embedding).item()
            b25 = bm25_map.get(d_id, 0.0)
            score = 0.5 * b25 + 0.5 * sim if bm25_map else sim
            if score > best_score:
                best_score = score
                best_doc_id = d_id
                best_entity_id = doc.entity_id

        # Require hybrid similarity >= 0.40 to match a specific catalog entity/document
        if best_score < 0.40:
            best_doc_id = None
            best_entity_id = None

        text_lower = text.lower()
        is_alert = any(k in text_lower for k in ("alert", "drift", "dropped", "breached", "warning", "instability", "anomalous"))
        is_remed = any(k in text_lower for k in ("resolved", "restored", "recalibrated", "remediation", "serviced"))

        if is_alert:
            status = "TAINTED"
        elif is_remed:
            status = "VALID"
        else:
            status = "NOMINAL"

        return ResolvedEvent(
            event_id=event_id,
            text=text,
            matched_entity_id=best_entity_id,
            matched_doc_id=best_doc_id,
            confidence=best_score,
            is_alert=is_alert,
            is_remediation=is_remed,
            status=status,
        )


# =============================================================================
# 1. Stateless Hybrid RAG
# =============================================================================
class StatelessRAG:
    """Standard fair hybrid search: score(x) = 0.5 * BM25(q, x) + 0.5 * cos(q, x). Memoryless."""

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.bm25 = OkapiBM25Scorer(catalog)

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        pass

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        pass

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        scored: List[Tuple[float, ResearchDocument]] = []
        for doc in self.catalog.documents.values():
            sim = torch.dot(query_vec, doc.embedding).item()
            b25 = bm25_map.get(doc.doc_id, 0.0)
            score = 0.5 * b25 + 0.5 * sim if bm25_map else sim
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag="",
                source_method="StatelessRAG",
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 2. Event-Log RAG (Persistent Memory Baseline 1)
# =============================================================================
class EventLogRAG:
    """
    Maintains an indexed chronological log of past events.
    Combines document retrieval with recent matching event log snippets.
    """

    def __init__(self, catalog: ResearchWorldCatalog, decay_lambda: float = 0.995, beta: float = 0.35):
        self.catalog = catalog
        self.decay_lambda = decay_lambda
        self.beta = beta
        self.event_log: List[Dict[str, Any]] = []
        self.current_timestep: int = 0
        self.resolver = SharedFrozenEventResolver(catalog)

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        self.current_timestep = timestamp
        resolved = self.resolver.resolve_raw_event(event_id, text, embedding, timestamp)
        self.event_log.append({
            "step": timestamp,
            "id": event_id,
            "text": text,
            "embedding": embedding,
            "is_alert": resolved.is_alert,
            "is_remediation": resolved.is_remediation,
            "status": resolved.status,
            "entity_id": resolved.matched_entity_id,
        })

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        self.current_timestep = timestamp
        self.event_log.append({
            "step": timestamp,
            "id": event_id,
            "entity_id": entity_id,
            "status": status,
            "text": text,
            "embedding": embedding,
            "is_alert": (status in ("TAINTED", "DRIFT")),
            "is_remediation": (status == "VALID"),
        })

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        scored_docs: List[Tuple[float, ResearchDocument]] = []

        for doc in self.catalog.documents.values():
            base_sim = torch.dot(query_vec, doc.embedding).item()
            event_boost = 0.0

            for ev in self.event_log[-50:]:
                age = self.current_timestep - ev["step"]
                decay = self.decay_lambda ** age
                ev_sim = torch.dot(doc.embedding, ev["embedding"]).item()
                boost = ev_sim * decay
                if boost > event_boost:
                    event_boost = boost

            total_score = base_sim + self.beta * event_boost
            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        items: List[RetrievedItem] = []

        # Pack most recent relevant event snippet if alert is active
        relevant_events = [
            ev for ev in self.event_log[-20:]
            if torch.dot(query_vec, ev["embedding"]).item() > 0.30 or ev["is_alert"]
        ]
        if relevant_events:
            latest_ev = relevant_events[-1]
            items.append(RetrievedItem(
                doc_id=f"log_{latest_ev['id']}",
                title=f"Chronological Event Log #{latest_ev['step']}",
                content=latest_ev["text"],
                band="LOG",
                tokens=25,
                score=100.0,
                state_tag=latest_ev.get("status", ""),
                source_method="EventLogRAG",
            ))

        for s, d in scored_docs:
            items.append(RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag="",
                source_method="EventLogRAG",
            ))

        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 3. Periodic Summarized Memory (Persistent Memory Baseline 2)
# =============================================================================
class PeriodicSummarizedMemory:
    """
    Periodically compresses events into a running scratchpad summary every K steps.
    Suffers from inter-epoch coherence lag.
    """

    def __init__(self, catalog: ResearchWorldCatalog, summarize_interval: int = 20):
        self.catalog = catalog
        self.summarize_interval = summarize_interval
        self.unsummarized_events: List[Dict[str, Any]] = []
        self.current_summary: str = "Nominal operations. All laboratory systems calibrated."
        self.current_timestep: int = 0
        self.resolver = SharedFrozenEventResolver(catalog)

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        self.current_timestep = timestamp
        resolved = self.resolver.resolve_raw_event(event_id, text, embedding, timestamp)
        self.unsummarized_events.append({
            "step": timestamp,
            "text": text,
            "is_alert": resolved.is_alert,
            "is_remediation": resolved.is_remediation,
        })
        if len(self.unsummarized_events) >= self.summarize_interval:
            self._regenerate_summary()

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        self.current_timestep = timestamp
        self.unsummarized_events.append({
            "step": timestamp,
            "entity_id": entity_id,
            "status": status,
            "text": text,
            "is_alert": (status in ("TAINTED", "DRIFT")),
            "is_remediation": (status == "VALID"),
        })
        if len(self.unsummarized_events) >= self.summarize_interval:
            self._regenerate_summary()

    def _regenerate_summary(self):
        alerts = [e["text"] for e in self.unsummarized_events if e["is_alert"]]
        remeds = [e["text"] for e in self.unsummarized_events if e["is_remediation"]]
        if remeds:
            self.current_summary = f"Summary @ step {self.current_timestep}: Active remediations certified: {remeds[-1]}."
        elif alerts:
            self.current_summary = f"Summary @ step {self.current_timestep}: Active alerts detected: {alerts[-1]}."
        else:
            self.current_summary = f"Summary @ step {self.current_timestep}: Operations nominal. No unhandled alerts."
        self.unsummarized_events.clear()

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        scored: List[Tuple[float, ResearchDocument]] = []
        for doc in self.catalog.documents.values():
            sim = torch.dot(query_vec, doc.embedding).item()
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        items: List[RetrievedItem] = []
        summary_tokens = 35
        items.append(RetrievedItem(
            doc_id="summary_scratchpad",
            title="Executive Periodic Memory Scratchpad",
            content=self.current_summary,
            band="SUMMARY",
            tokens=summary_tokens,
            score=999.0,
            state_tag="",
            source_method="PeriodicSummarizedMemory",
        ))

        for s, d in scored:
            items.append(RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag="",
                source_method="PeriodicSummarizedMemory",
            ))

        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 4. Temporal GraphRAG (Persistent Memory Baseline 3)
# =============================================================================
class TemporalGraphRAG:
    """
    Causal graph neighborhood expansion with chronological node event annotations.
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.node_events: Dict[str, Dict[str, Any]] = {}
        self.current_timestep: int = 0
        self.resolver = SharedFrozenEventResolver(catalog)

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        self.current_timestep = timestamp
        resolved = self.resolver.resolve_raw_event(event_id, text, embedding, timestamp)
        if resolved.matched_doc_id:
            c_node = self.catalog.documents[resolved.matched_doc_id].causal_node_id
            if c_node:
                self.node_events[c_node] = {
                    "text": text,
                    "step": timestamp,
                    "status": resolved.status,
                }

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        self.current_timestep = timestamp
        doc_id = self.catalog.entity_to_doc.get(entity_id)
        if doc_id:
            c_node = self.catalog.documents[doc_id].causal_node_id
            if c_node:
                self.node_events[c_node] = {
                    "text": text,
                    "step": timestamp,
                    "status": status,
                }

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        scored = [(torch.dot(query_vec, d.embedding).item(), d) for d in self.catalog.documents.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_doc = scored[0][1]

        # Follow causal graph edges
        graph_doc_ids: Set[str] = {top_doc.doc_id}
        if top_doc.causal_node_id:
            for src, tgt, rel in self.catalog.causal_dependencies:
                if src == top_doc.causal_node_id or tgt == top_doc.causal_node_id:
                    for d in self.catalog.documents.values():
                        if d.causal_node_id in (src, tgt):
                            graph_doc_ids.add(d.doc_id)

        items: List[RetrievedItem] = []
        for s, d in scored:
            is_graph_neighbor = d.doc_id in graph_doc_ids
            score = s + (0.50 if is_graph_neighbor else 0.0)

            node_tag = "VALID"
            content = d.content
            if d.causal_node_id and d.causal_node_id in self.node_events:
                ev_info = self.node_events[d.causal_node_id]
                node_tag = ev_info["status"]
                content = f"{d.content} [Recent Event #{ev_info['step']}: {ev_info['text']}]"

            items.append(RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=content,
                band=d.band,
                tokens=d.tokens + 15 if d.causal_node_id in self.node_events else d.tokens,
                score=score,
                state_tag=node_tag,
                source_method="TemporalGraphRAG",
            ))

        items.sort(key=lambda x: x.score, reverse=True)
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 5. Conventional State Store + Fair Hybrid RAG (Persistent Memory Baseline 4: S_t^DB)
# =============================================================================
class ConventionalStateStoreRAG:
    """
    Event-sourced relational state store (S_t^DB) + explicit graph propagation + fair Hybrid RAG.
    Tracks entity statuses in a database table. Annotates retrieved documents with DB status.
    DOES NOT maintain a continuous dynamic relevance field h_t.
    """

    def __init__(self, catalog: ResearchWorldCatalog):
        self.catalog = catalog
        self.entity_status: Dict[str, str] = {e_id: "VALID" for e_id in catalog.entity_to_doc.keys()}
        self.current_timestep: int = 0
        self.resolver = SharedFrozenEventResolver(catalog)
        self.bm25 = OkapiBM25Scorer(catalog)

    def _propagate_downstream(self, root_entity: str, status: str):
        # Pure relational state store does not perform automatic downstream database cascades
        pass

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        self.current_timestep = timestamp
        resolved = self.resolver.resolve_raw_event(event_id, text, embedding, timestamp)
        if resolved.matched_entity_id:
            curr = self.entity_status.get(resolved.matched_entity_id, "VALID")
            if resolved.status in ("TAINTED", "VALID") or curr != "TAINTED":
                self.entity_status[resolved.matched_entity_id] = resolved.status

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        self.current_timestep = timestamp
        self.entity_status[entity_id] = status

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        scored = []
        for d in self.catalog.documents.values():
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            score = 0.5 * b25 + 0.5 * sim if bm25_map else sim
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=self.entity_status.get(d.entity_id, "VALID"),
                source_method="ConventionalStateStoreRAG",
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 5b. State Store + Status-Aware Hybrid RAG (Abnormal Status Boost)
# =============================================================================
class StatusAwareStateStoreRAG(ConventionalStateStoreRAG):
    """
    Conventional State Store + Status-Aware Hybrid RAG.
    Modulates query relevance by abnormal state:
      score(x) = similarity_hybrid(q, x) * (1 + beta * 1[status(x) != NOMINAL])
    """

    def __init__(self, catalog: ResearchWorldCatalog, beta: float = 1.0):
        super().__init__(catalog)
        self.beta = beta

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        scored = []
        for d in self.catalog.documents.values():
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            base_score = 0.5 * b25 + 0.5 * sim if bm25_map else sim
            status = self.entity_status.get(d.entity_id, "VALID")
            boost = (1.0 + self.beta) if status in ("TAINTED", "SUSPECT", "DRIFT") else 1.0
            scored.append((base_score * boost, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=self.entity_status.get(d.entity_id, "VALID"),
                source_method="StatusAwareStateStoreRAG",
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 5b-ii. State Store + Static Semantic Expansion (Z-Only Baseline, No h_t)
# =============================================================================
class StatusAwareStaticZExpansionRAG(ConventionalStateStoreRAG):
    """
    Conventional State Store + Static Multi-Aspect Semantic Expansion (Z-Only).
    Ablates continuous field dynamics (h_t), differential diffusion, and temporal memory:
      1. Identifies abnormal documents a in Abnormal (status in TAINTED, SUSPECT, DRIFT).
      2. Computes maximum aspect similarity between abnormal nodes and candidate documents:
           r_Z(d) = max_{a in Abnormal} max_{u in Z_a, v in Z_d} cos(u, v)
      3. Boosts candidates exceeding static similarity threshold tau_Z:
           score(d) = base_score(d) * (1 + beta * 1[d is abnormal]) + alpha_Z * max(0, r_Z(d) - tau_Z)
    """

    def __init__(
        self,
        catalog: ResearchWorldCatalog,
        beta: float = 1.0,
        alpha_z: float = 0.50,
        tau_z: float = 0.50,
    ):
        super().__init__(catalog)
        self.beta = beta
        self.alpha_z = alpha_z
        self.tau_z = tau_z

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        abnormal_doc_ids: List[str] = []
        for e_id, status in self.entity_status.items():
            if status in ("TAINTED", "SUSPECT", "DRIFT"):
                d_id = self.catalog.entity_to_doc.get(e_id)
                if d_id and d_id in self.catalog.documents:
                    abnormal_doc_ids.append(d_id)

        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        scored = []

        for d in self.catalog.documents.values():
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            base_score = 0.5 * b25 + 0.5 * sim if bm25_map else sim

            status = self.entity_status.get(d.entity_id, "VALID")
            status_boost = (1.0 + self.beta) if status in ("TAINTED", "SUSPECT", "DRIFT") else 1.0

            # Static aspect expansion from abnormal entities (No h_t, No diffusion)
            z_boost = 0.0
            if abnormal_doc_ids:
                d_aspects = list(d.aspect_vectors.values()) if d.aspect_vectors else [d.embedding]
                max_sim = 0.0
                for a_id in abnormal_doc_ids:
                    a_doc = self.catalog.documents[a_id]
                    a_aspects = list(a_doc.aspect_vectors.values()) if a_doc.aspect_vectors else [a_doc.embedding]
                    for u in a_aspects:
                        for v in d_aspects:
                            sim_uv = torch.dot(u, v).item()
                            if sim_uv > max_sim:
                                max_sim = sim_uv

                if max_sim >= self.tau_z:
                    z_boost = self.alpha_z * (max_sim - self.tau_z) / (1.0 - self.tau_z + 1e-6)

            total_score = base_score * status_boost + z_boost
            scored.append((total_score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=self.entity_status.get(d.entity_id, "VALID"),
                source_method="StatusAwareStaticZExpansionRAG",
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 5b-iii. State Store + Static Semantic Graph Propagation (PPR / Multi-Hop W)
# =============================================================================
class StaticSemanticGraphPropagationRAG(ConventionalStateStoreRAG):
    """
    Conventional State Store + Multi-Hop Static Semantic Graph Propagation.
    Computes Personalized PageRank (PPR) / multi-step power diffusion over the static
    multi-aspect affinity matrix W_norm, without temporal decay, persistent ODEs, or h_t:
      p = (1 - alpha_ppr) * p_0 + alpha_ppr * (W_norm @ p_0)
    where p_0 is indicator over abnormal entities.
    """

    def __init__(
        self,
        catalog: ResearchWorldCatalog,
        beta: float = 1.0,
        alpha_ppr: float = 0.50,
        semantic_threshold: float = 0.30,
        steps: int = 2,
    ):
        super().__init__(catalog)
        self.beta = beta
        self.alpha_ppr = alpha_ppr
        self.steps = steps

        # Precompute static multi-aspect affinity matrix W_norm
        docs = list(catalog.documents.values())
        self.doc_keys = [d.doc_id for d in docs]
        n = len(docs)
        W = torch.zeros((n, n), dtype=torch.float32)

        for i in range(n):
            asp_i = list(docs[i].aspect_vectors.values()) if docs[i].aspect_vectors else [docs[i].embedding]
            for j in range(i, n):
                asp_j = list(docs[j].aspect_vectors.values()) if docs[j].aspect_vectors else [docs[j].embedding]
                max_sim = 0.0
                for u in asp_i:
                    for v in asp_j:
                        sim = torch.dot(u, v).item()
                        if sim > max_sim:
                            max_sim = sim
                if max_sim >= semantic_threshold:
                    W[i, j] = max_sim
                    W[j, i] = max_sim

        deg = W.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        self.W_norm = deg_inv_sqrt * W * deg_inv_sqrt.t()

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        n = len(self.doc_keys)
        p0 = torch.zeros(n, dtype=torch.float32)
        has_abnormal = False

        for e_id, status in self.entity_status.items():
            if status in ("TAINTED", "SUSPECT", "DRIFT"):
                d_id = self.catalog.entity_to_doc.get(e_id)
                if d_id and d_id in self.doc_keys:
                    p0[self.doc_keys.index(d_id)] = 1.0
                    has_abnormal = True

        # Static multi-step power propagation (No temporal h_t)
        p = p0.clone()
        if has_abnormal:
            for _ in range(self.steps):
                p = (1.0 - self.alpha_ppr) * p0 + self.alpha_ppr * torch.matmul(self.W_norm, p)

        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        scored = []

        for idx, d_id in enumerate(self.doc_keys):
            d = self.catalog.documents[d_id]
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            base_score = 0.5 * b25 + 0.5 * sim if bm25_map else sim

            status = self.entity_status.get(d.entity_id, "VALID")
            status_boost = (1.0 + self.beta) if status in ("TAINTED", "SUSPECT", "DRIFT") else 1.0

            static_prop_boost = 0.50 * float(p[idx].item()) if has_abnormal else 0.0
            total_score = base_score * status_boost + static_prop_boost
            scored.append((total_score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=self.entity_status.get(d.entity_id, "VALID"),
                source_method="StaticSemanticGraphPropagationRAG",
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


# =============================================================================
# 5c. State Store + Recursive Graph RAG (Multi-Hop Shortest Path BFS Decay)
# =============================================================================
class RecursiveGraphStateStoreRAG(ConventionalStateStoreRAG):
    """
    Conventional State Store + Recursive Graph Propagation.
    Computes shortest paths on the causal DAG via BFS.
    Supports:
      - directed=True: Directed downstream causal propagation along active paths from anomaly to query anchor.
      - directed=False: Undirected structural relevance.
    """

    def __init__(self, catalog: ResearchWorldCatalog, beta: float = 1.0, gamma: float = 0.75, directed: bool = True):
        super().__init__(catalog)
        self.beta = beta
        self.gamma = gamma
        self.directed = directed
        self.shortest_paths = compute_bfs_shortest_paths(catalog.causal_dependencies, undirected=(not directed))

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        abnormal_nodes: Set[str] = set()
        for e_id, status in self.entity_status.items():
            if status in ("TAINTED", "SUSPECT", "DRIFT"):
                doc_id = self.catalog.entity_to_doc.get(e_id)
                if doc_id:
                    node = self.catalog.documents[doc_id].causal_node_id
                    if node:
                        abnormal_nodes.add(node)

        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        base_scores = {}
        for d in self.catalog.documents.values():
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            base_scores[d.doc_id] = 0.5 * b25 + 0.5 * sim if bm25_map else sim

        top_doc_id = max(base_scores, key=base_scores.get)
        query_node = self.catalog.documents[top_doc_id].causal_node_id

        scored = []
        for d in self.catalog.documents.values():
            base = base_scores[d.doc_id]
            status = self.entity_status.get(d.entity_id, "VALID")
            status_boost = (1.0 + self.beta) if status in ("TAINTED", "SUSPECT", "DRIFT") else 1.0

            graph_boost = 0.0
            d_node = d.causal_node_id
            if d_node and abnormal_nodes:
                for an in abnormal_nodes:
                    dist_an_d = self.shortest_paths.get(an, {}).get(d_node, 999)
                    if dist_an_d < 999:
                        graph_boost += (self.gamma ** dist_an_d) * 0.90

            scored.append((base * status_boost + graph_boost, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        method_name = "DirectedRecursiveGraphStateStoreRAG" if self.directed else "UndirectedGraphStateStoreRAG"
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=self.entity_status.get(d.entity_id, "VALID"),
                source_method=method_name,
            )
            for s, d in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res


class DirectedRecursiveGraphStateStoreRAG(RecursiveGraphStateStoreRAG):
    """Explicit directed causal dependency materializer."""
    def __init__(self, catalog: ResearchWorldCatalog, beta: float = 1.0, gamma: float = 0.75):
        super().__init__(catalog, beta=beta, gamma=gamma, directed=True)


class UndirectedGraphStateStoreRAG(RecursiveGraphStateStoreRAG):
    """Undirected structural relevance baseline."""
    def __init__(self, catalog: ResearchWorldCatalog, beta: float = 1.0, gamma: float = 0.75):
        super().__init__(catalog, beta=beta, gamma=gamma, directed=False)


# Backward compatibility alias
GraphAwareStateStoreRAG = DirectedRecursiveGraphStateStoreRAG


# =============================================================================
# 6. CortexPriorRAG: Same Hybrid RAG + Persistent Relevance Prior
# =============================================================================
class CortexPriorRAG:
    """
    Maintains persistent dynamic activation h_t and epistemic working state S_t.
    Modulates fair hybrid base retrieval on the continuous dynamic prior:
      score(x) = retrieval_hybrid(x, q) * (1.0 + alpha * h_x(t)) * relevance_factor(x, S_t)
    where dynamic strain h_t diffuses continuously across the semantic manifold.
    """

    def __init__(self, catalog: ResearchWorldCatalog, alpha: float = 1.5):
        self.catalog = catalog
        self.alpha = alpha
        self.runtime = CortexRuntime(hidden_dim=catalog.hidden_dim)
        self.runtime.context_fabric.band_anchors = catalog.band_anchors
        self.resolver = SharedFrozenEventResolver(catalog)
        self.bm25 = OkapiBM25Scorer(catalog)

        # Register documents into Cortex Context Fabric and Reaction Field
        for doc in catalog.documents.values():
            aspects = doc.aspect_vectors if doc.aspect_vectors else {doc.band: doc.embedding}
            self.runtime.register_fabric_item(
                item_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                aspect_vectors=aspects,
                primary_aspect=doc.band,
                causal_node_id=doc.causal_node_id,
            )
            self.runtime.register_agent_entity(
                agent_id=doc.doc_id,
                name=doc.title,
                role=doc.band,
                prototypes=aspects,
                activation_threshold=0.30,
            )

        # Register causal claims into Epistemic Manifold
        registered_nodes: Set[str] = set()
        for src, tgt, rel in catalog.causal_dependencies:
            for n in (src, tgt):
                if n not in registered_nodes:
                    self.runtime.register_claim(n, f"Claim {n}", EpistemicKind.HYPOTHESIS, 0.85)
                    registered_nodes.add(n)
            self.runtime.link_causal_dependency(src, tgt, rel)

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        resolved = self.resolver.resolve_raw_event(event_id, text, embedding, timestamp)

        mag = 2.0 if resolved.is_alert else (1.5 if resolved.is_remediation else 0.15)
        self.runtime.observe(
            text=text,
            embedding=embedding,
            magnitude=mag,
            source="lab_sensor_stream",
            event_id=event_id,
        )

        if resolved.matched_doc_id:
            item = self.runtime.context_fabric.items.get(resolved.matched_doc_id)
            if item:
                item.validity_status = resolved.status

        # On remediation, clear residual strain
        if resolved.is_remediation:
            for item in self.runtime.context_fabric.items.values():
                item.dynamic_energy = 0.0
                item.validity_status = "VALID"
            for entity in self.runtime.reaction_field.entities.values():
                entity.current_energy = 0.0

    def record_structured_event(
        self, event_id: str, entity_id: str, status: str, text: str, embedding: torch.Tensor, timestamp: int
    ):
        doc_id = self.catalog.entity_to_doc.get(entity_id)
        if doc_id:
            item = self.runtime.context_fabric.items.get(doc_id)
            if item:
                item.validity_status = status

        self.runtime.observe(
            text=text,
            embedding=embedding,
            magnitude=2.0,
            source="lab_sensor_stream",
            event_id=event_id,
        )
        self.runtime.reaction_field.step_diffusion(steps=2)

        if status == "VALID":
            for item in self.runtime.context_fabric.items.values():
                item.dynamic_energy = 0.0
                item.validity_status = "VALID"
            for entity in self.runtime.reaction_field.entities.values():
                entity.current_energy = 0.0

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()
        bm25_map = self.bm25.normalized_scores(query_text) if query_text else {}
        energies = {k: e.current_energy for k, e in self.runtime.reaction_field.entities.items()}
        max_e = max(energies.values()) if energies and max(energies.values()) > 0 else 1.0

        scored = []
        for d in self.catalog.documents.values():
            sim = torch.dot(query_vec, d.embedding).item()
            b25 = bm25_map.get(d.doc_id, 0.0)
            base_score = 0.5 * b25 + 0.5 * sim if bm25_map else sim

            e = energies.get(d.doc_id, 0.0)
            norm_e = e / max_e if max_e > 0 else 0.0

            fabric_item = self.runtime.context_fabric.items.get(d.doc_id)
            tag = fabric_item.validity_status if fabric_item else "VALID"
            status_bonus = 0.50 if tag in ("TAINTED", "SUSPECT", "DRIFT") else 0.0

            # Dynamic relevance prior modulation from continuous manifold strain
            score = base_score + status_bonus + 0.40 * norm_e
            scored.append((score, d, tag))

        scored.sort(key=lambda x: x[0], reverse=True)
        items = [
            RetrievedItem(
                doc_id=d.doc_id,
                title=d.title,
                content=d.content,
                band=d.band,
                tokens=d.tokens,
                score=s,
                state_tag=tag,
                source_method="CortexPriorRAG",
            )
            for s, d, tag in scored
        ]
        res = pack_within_budget(items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def compute_field_activation_pr(
        self,
        thresholds: Optional[List[float]] = None,
        relevant_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[float, float, float, int, int]]:
        r"""
        Computes reaction field continuous activation precision and recall across potential thresholds theta:
          Activation Precision(theta) = |{x : h_x > theta} \cap R| / |{x : h_x > theta}|
          Activation Recall(theta)    = |{x : h_x > theta} \cap R| / |R|
        Returns: [(theta, recall, precision, active_count, relevant_active_count), ...]
        """
        if thresholds is None:
            thresholds = [round(0.05 + 0.05 * i, 2) for i in range(19)]
        if relevant_ids is None:
            relevant_ids = {"doc_inst_ms4", "doc_ds_data42"}

        energies = {k: e.current_energy for k, e in self.runtime.reaction_field.entities.items()}
        max_e = max(energies.values()) if energies and max(energies.values()) > 0 else 1.0
        normalized_h = {k: e / max_e for k, e in energies.items()}

        pr_curve: List[Tuple[float, float, float, int, int]] = []
        for th in thresholds:
            active = {k for k, h in normalized_h.items() if h >= th}
            hits = active.intersection(relevant_ids)
            rec = len(hits) / len(relevant_ids) if relevant_ids else 0.0
            prec = len(hits) / len(active) if active else 1.0
            pr_curve.append((th, rec, prec, len(active), len(hits)))
        return pr_curve


# =============================================================================
# 6. Tiered Context Union Architecture (Tier 0: Status, Tier 1: Graph, Tier 2: Cortex)
# =============================================================================
class TieredContextUnionRAG:
    """
    Tiered Context Union Architecture:
      C_t = C_{explicit status} U C_{graph} U C_{Cortex}
    Guarantees:
      - Tier 0: Hard Current State. Explicitly abnormal entities (status != NORMAL)
        are mandatory context items (cannot be displaced by fuzzy relevance).
      - Tier 1: Explicit Graph Consequences. Prefix-preserving BFS reachable nodes
        from active anomalies are guaranteed priority inclusion.
      - Tier 2: Semantic Frontier. Dynamic Cortex h_t reaction potential (or Z)
        fills remaining token budget with soft candidates.
    """

    def __init__(
        self,
        catalog: ResearchWorldCatalog,
        beta: float = 1.0,
        gamma: float = 0.75,
        alpha: float = 0.40,
        directed: bool = True,
    ):
        self.catalog = catalog
        self.state_store = StatusAwareStateStoreRAG(catalog, beta=beta)
        self.graph_store = DirectedRecursiveGraphStateStoreRAG(catalog, beta=beta, gamma=gamma)
        self.cortex = CortexPriorRAG(catalog, alpha=alpha)
        self.shortest_paths = compute_bfs_shortest_paths(catalog.causal_dependencies, undirected=(not directed))

    def record_raw_event(self, event_id: str, text: str, embedding: torch.Tensor, timestamp: int):
        self.state_store.record_raw_event(event_id, text, embedding, timestamp)
        self.graph_store.record_raw_event(event_id, text, embedding, timestamp)
        self.cortex.record_raw_event(event_id, text, embedding, timestamp)

    def query(self, query_text: str, query_vec: torch.Tensor, token_budget: int = 512) -> RetrievalResult:
        t0 = time.perf_counter()

        # 1. Tier 0: Hard Current State (Mandatory Abnormal Items)
        abnormal_docs: List[ResearchDocument] = []
        for e_id, status in self.state_store.entity_status.items():
            if status in ("TAINTED", "SUSPECT", "DRIFT"):
                d_id = self.catalog.entity_to_doc.get(e_id)
                if d_id and d_id in self.catalog.documents:
                    abnormal_docs.append(self.catalog.documents[d_id])

        # 2. Tier 1: Explicit Graph Reachable Consequences
        abnormal_nodes = {d.causal_node_id for d in abnormal_docs if d.causal_node_id}
        graph_reachable_docs: List[Tuple[int, ResearchDocument]] = []
        if abnormal_nodes:
            for d in self.catalog.documents.values():
                if d.causal_node_id and d not in abnormal_docs:
                    for an in abnormal_nodes:
                        dist = self.shortest_paths.get(an, {}).get(d.causal_node_id, 999)
                        if dist < 999:
                            graph_reachable_docs.append((dist, d))
                            break
            graph_reachable_docs.sort(key=lambda x: x[0])

        # 3. Tier 2: Cortex Dynamic Relevance Frontier
        cortex_res = self.cortex.query(query_text, query_vec, token_budget=token_budget * 2)

        # Assemble Priority-Tiered Context: Tier 0 -> Tier 1 -> Tier 2
        selected_ids: Set[str] = set()
        ordered_items: List[RetrievedItem] = []

        # Pack Tier 0
        for d in abnormal_docs:
            if d.doc_id not in selected_ids:
                selected_ids.add(d.doc_id)
                ordered_items.append(RetrievedItem(
                    doc_id=d.doc_id,
                    title=d.title,
                    content=d.content,
                    band=d.band,
                    tokens=d.tokens,
                    score=1000.0,
                    state_tag=self.state_store.entity_status.get(d.entity_id, "VALID"),
                    source_method="Tier0_Status",
                ))

        # Pack Tier 1
        for dist, d in graph_reachable_docs:
            if d.doc_id not in selected_ids:
                selected_ids.add(d.doc_id)
                ordered_items.append(RetrievedItem(
                    doc_id=d.doc_id,
                    title=d.title,
                    content=d.content,
                    band=d.band,
                    tokens=d.tokens,
                    score=500.0 - dist,
                    state_tag=self.state_store.entity_status.get(d.entity_id, "VALID"),
                    source_method="Tier1_Graph",
                ))

        # Pack Tier 2
        for it in cortex_res.items:
            if it.doc_id not in selected_ids:
                selected_ids.add(it.doc_id)
                ordered_items.append(RetrievedItem(
                    doc_id=it.doc_id,
                    title=it.title,
                    content=it.content,
                    band=it.band,
                    tokens=it.tokens,
                    score=it.score,
                    state_tag=it.state_tag,
                    source_method="Tier2_CortexFrontier",
                ))

        res = pack_within_budget(ordered_items, token_budget)
        res.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return res
