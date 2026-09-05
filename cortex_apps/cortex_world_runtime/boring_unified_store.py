"""
Boring Unified Store (U_0) kill test + physical D measurement + retrieval scaling.

Scope (evaluation only -- architecture is frozen):
  U_0 = {entity table, adjacency arrays, embedding matrix, event log}
  in one process, one consistency domain, no Cortex abstraction. Same S, G, Z,
  H data and same retrieval algorithms as the Cortex substrate. Questions:

  1. Retrieval parity: does U_0 recover the identical premise sets as Cortex?
     If yes, Cortex adds no retrieval magic over the obvious unified store.
  2. Physical D: measure REAL bytes (not the 27.2% model constant) for
     Cortex substrate vs U_0 vs a genuinely separate 4-store Modular-C
     materialization, plus real join costs (CPU ms, copies, marshal bytes).
     Does D_Cortex < D_Modular hold physically? Does D_Cortex < D_U0?
  3. Scaling: R_prov(N) full/partial provenance recall at N = 2k..1M with a
     fixed candidate budget, sparse data-driven graph wiring, and bounded
     prompt tokens.

Honest limitations (read before citing):
  - Lean synthetic background (not 250k real workspace tokens); premises are
    the 40 benchmark-authored synthetics from unseen_synthesis_suite.
  - Premise-premise edges are forbidden so premises must bridge via
    background (mirrors the dispersed-facts scenario; stated, not hidden).
  - Large-N graph wiring is sparse ring+random (O(N)), premise edges are
    top-m data-driven links computed in blocks (O(40*N)); the dense O(N^2)
    k-NN build used at small N is infeasible past ~10k and is NOT used there.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# U_0: the dumbest competent unified store imaginable
# ---------------------------------------------------------------------------

class BoringUnifiedStore:
    """One process, one consistency domain. Four plain containers, no abstraction."""

    def __init__(self):
        self.states: Dict[str, Dict[str, Any]] = {}   # S: entity table
        self.adj: Dict[str, Set[str]] = {}             # G: adjacency arrays
        self.emb: Dict[str, torch.Tensor] = {}         # Z: embedding matrix (per-row)
        self.log: List[Any] = []                       # H: event log
        self.clusters: Dict[int, List[str]] = {}       # index partition
        self.centroids: Optional[torch.Tensor] = None  # index centroids

    def __len__(self):
        return len(self.states)

    # -- same algorithms as WorldSnapshot (deliberately, not cleverly) ------
    def vector_search(self, qvec: torch.Tensor, top_k: int = 5,
                      candidate_budget: int = 400) -> List[Tuple[str, float]]:
        candidates: List[str] = []
        if self.centroids is not None:
            sims = torch.matmul(self.centroids, qvec)
            for c_idx in torch.argsort(sims, descending=True):
                cid = int(c_idx.item())
                candidates.extend(self.clusters.get(cid, []))
                if len(candidates) >= candidate_budget:
                    break
        else:
            for c in self.clusters.values():
                candidates.extend(c)
                if len(candidates) >= candidate_budget:
                    break
        candidates = candidates[:candidate_budget]
        scored = [(eid, torch.dot(qvec, self.emb[eid]).item())
                  for eid in candidates if eid in self.emb]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def bfs(self, start_id: str, max_depth: int = 3,
            max_nodes: Optional[int] = 25) -> List[str]:
        if start_id not in self.adj:
            return []
        visited = {start_id}
        queue = deque([(start_id, 0)])
        result = []
        while queue and (max_nodes is None or len(result) < max_nodes):
            curr, depth = queue.popleft()
            if depth > 0:
                result.append(curr)
            if depth < max_depth:
                # sorted(): set iteration order depends on PYTHONHASHSEED;
                # traversal must not.
                for nbr in sorted(self.adj.get(curr, ())):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1))
        return result

    def get_state(self, eid: str) -> Optional[Dict[str, Any]]:
        return self.states.get(eid)

    def rank_candidates(self, qvec: torch.Tensor,
                        candidate_budget: int) -> List[Tuple[str, float]]:
        """Full score-sorted candidate ranking (for Recall@k / MRR)."""
        candidates: List[str] = []
        if self.centroids is not None:
            sims = torch.matmul(self.centroids, qvec)
            for c_idx in torch.argsort(sims, descending=True):
                candidates.extend(self.clusters.get(int(c_idx.item()), []))
                if len(candidates) >= candidate_budget:
                    break
        else:
            for c in self.clusters.values():
                candidates.extend(c)
                if len(candidates) >= candidate_budget:
                    break
        candidates = candidates[:candidate_budget]
        scored = [(eid, torch.dot(qvec, self.emb[eid]).item())
                  for eid in candidates if eid in self.emb]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def bfs_depths(self, start_id: str, max_depth: int = 6) -> Dict[str, int]:
        """Order-independent shortest-path depths from start (sorted traversal)."""
        if start_id not in self.adj:
            return {}
        depths = {start_id: 0}
        queue = deque([start_id])
        while queue:
            curr = queue.popleft()
            if depths[curr] >= max_depth:
                continue
            for nbr in sorted(self.adj.get(curr, ())):
                if nbr not in depths:
                    depths[nbr] = depths[curr] + 1
                    queue.append(nbr)
        return depths


def wire_dense_knn(src: BoringUnifiedStore, k: int = 4, sim_threshold: float = 0.45,
                   block: int = 1000):
    """Harvester-style dense k-NN wiring over ALL entities (data-driven, blocked
    to avoid O(N^2) memory). Only feasible to ~10k; beyond that use sparse."""
    eids = list(src.emb.keys())
    mats = torch.stack([src.emb[e] for e in eids])
    n = len(eids)
    added = 0
    for i0 in range(0, n, block):
        chunk = mats[i0:i0 + block]
        sims = chunk @ mats.t()
        # exclude self
        for r in range(sims.shape[0]):
            sims[r, i0 + r] = -1.0
        kval = min(k, n - 1)
        vals, idx = torch.topk(sims, k=kval, dim=1)
        for r in range(sims.shape[0]):
            s = eids[i0 + r]
            for c in range(kval):
                if float(vals[r, c]) >= sim_threshold:
                    d = eids[int(idx[r, c])]
                    if d not in src.adj[s]:
                        src.adj[s].add(d)
                        src.adj[d].add(s)
                        added += 1
    return added


# ---------------------------------------------------------------------------
# Modular-C: genuinely separate 4-store materialization (independent copies)
# ---------------------------------------------------------------------------

class ModularFourStore:
    """4 decoupled stores, each with its own index tables and owned copies."""

    def __init__(self):
        self.vec_index: Dict[str, int] = {}   # VectorStore: own id table
        self.vec_embs: Dict[str, torch.Tensor] = {}
        self.vec_clusters: Dict[int, List[str]] = {}
        self.vec_centroids: Optional[torch.Tensor] = None
        self.graph_index: Dict[str, int] = {}  # GraphStore: own id table
        self.graph_adj: Dict[str, Set[str]] = {}
        self.doc_index: Dict[str, int] = {}    # DocumentStore: own id table
        self.doc_states: Dict[str, Dict[str, Any]] = {}
        self.hist_log: List[Any] = []          # HistoryStore: own log copy
        self.hist_validity: Dict[str, bool] = {}
        # join accounting (measured, not asserted)
        self.marshal_bytes_total = 0
        self.marshal_calls_total = 0
        self.copy_objects_total = 0

    def _marshal(self, payload: Any) -> Any:
        """Simulated store-API boundary: JSON serialize + deserialize."""
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.marshal_bytes_total += len(blob)
        self.marshal_calls_total += 1
        return json.loads(blob.decode("utf-8"))

    def vector_search(self, qvec: torch.Tensor, top_k: int = 5,
                      candidate_budget: int = 400) -> List[Tuple[str, float]]:
        candidates: List[str] = []
        if self.vec_centroids is not None:
            sims = torch.matmul(self.vec_centroids, qvec)
            for c_idx in torch.argsort(sims, descending=True):
                cid = int(c_idx.item())
                candidates.extend(self.vec_clusters.get(cid, []))
                if len(candidates) >= candidate_budget:
                    break
        candidates = candidates[:candidate_budget]
        scored = [(eid, torch.dot(qvec, self.vec_embs[eid]).item())
                  for eid in candidates if eid in self.vec_embs]
        scored.sort(key=lambda x: x[1], reverse=True)
        hits = [e for e, _ in scored[:top_k]]
        return [(e, s) for e, s in scored[:top_k] if e in self._marshal(hits)]

    def get_neighbors(self, eid: str) -> Set[str]:
        nbrs = self._marshal(sorted(self.graph_adj.get(eid, ())))
        self.copy_objects_total += len(nbrs)
        return set(nbrs)

    def get_state(self, eid: str) -> Optional[Dict[str, Any]]:
        st = self.doc_states.get(eid)
        if st is None:
            return None
        out = self._marshal(st)
        self.copy_objects_total += len(out)
        return out

    def check_valid(self, eids: List[str]) -> bool:
        flags = self._marshal({e: bool(self.hist_validity.get(e, True)) for e in eids})
        return all(flags.values())


def materialize_modular(source: BoringUnifiedStore) -> ModularFourStore:
    """Build 4 independent stores from the same logical data (owned copies)."""
    m = ModularFourStore()
    for i, eid in enumerate(source.states):
        m.vec_index[eid] = i
        m.graph_index[eid] = i
        m.doc_index[eid] = i
    for eid, t in source.emb.items():
        m.vec_embs[eid] = t.clone()
    for cid, members in source.clusters.items():
        m.vec_clusters[cid] = list(members)
    if source.centroids is not None:
        m.vec_centroids = source.centroids.clone()
    for eid, nbrs in source.adj.items():
        m.graph_adj[eid] = set(nbrs)
    for eid, st in source.states.items():
        m.doc_states[eid] = copy.deepcopy(st)
    m.hist_log = copy.deepcopy(source.log)
    for eid in source.states:
        m.hist_validity[eid] = True
    return m


def materialize_cortex_substrate(source: BoringUnifiedStore, num_clusters: int):
    """Load the same logical data into the real Cortex FastWorldSubstrate."""
    from cortex_apps.cortex_world_runtime.fast_world_substrate import (
        EntityNode, FastWorldSubstrate,
    )
    sub = FastWorldSubstrate(num_clusters=num_clusters)
    for cid, members in source.clusters.items():
        sub.clusters[cid] = list(members)
    for eid in source.states:
        sub.entities[eid] = EntityNode(
            entity_id=eid,
            state=dict(source.states[eid]),
            neighbors=set(source.adj.get(eid, ())),
            aspect_vector=source.emb[eid].clone(),
            cluster_id=0,
            version_modified=1,
        )
    if source.centroids is not None:
        sub.centroids = source.centroids.clone()
    return sub


# ---------------------------------------------------------------------------
# Memory accounting (one method applied to every representation)
# ---------------------------------------------------------------------------

def _sizer(obj: Any, seen: Set[int], depth: int = 0) -> int:
    if depth > 6 or id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, torch.Tensor):
        return total + obj.element_size() * obj.nelement()
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += _sizer(k, seen, depth + 1) + _sizer(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            total += _sizer(v, seen, depth + 1)
    elif isinstance(obj, (set, frozenset)):
        for v in obj:
            total += _sizer(v, seen, depth + 1)
    elif isinstance(obj, str):
        pass  # getsizeof already counts str bytes
    elif hasattr(obj, "__dict__"):
        # dataclass / plain objects (e.g. EntityNode): recurse into fields.
        # Without this, whole object graphs hide behind one getsizeof.
        for v in vars(obj).values():
            total += _sizer(v, seen, depth + 1)
    return total


def measure_container_bytes(*containers: Any) -> int:
    seen: Set[int] = set()
    return sum(_sizer(c, seen) for c in containers)


def rss_bytes() -> int:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Scale-world builder (lean synthetic background + 40 real synthetic premises)
# ---------------------------------------------------------------------------

def build_scale_source(n_background: int, num_clusters: int = 16, seed: int = 7,
                       premise_top_m: int = 4, bg_edges_per_node: int = 3):
    """Source-of-truth plain data. Returns (BoringUnifiedStore without index,
    tasks, encoder). Premise edges: top-m data-driven background links only
    (no premise-premise edges); background: ring + seeded intra-cluster edges.
    """
    from cortex_apps.cortex_world_runtime.unseen_synthesis_suite import (
        build_20_unseen_tasks, premise_eid,
    )
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )

    torch.manual_seed(seed)
    dim = 64
    tasks = build_20_unseen_tasks()
    encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)
    # NOTE: the encoder reseeds torch internally (fixed projection seed 42 by
    # design -- the encoder itself stays frozen). Re-seed here so background
    # geometry actually varies per world seed; otherwise multi-seed CIs are
    # vacuous (all worlds share identical embeddings).
    torch.manual_seed(seed)

    src = BoringUnifiedStore()
    # background embeddings: clustered noise (batched)
    centroids = F.normalize(torch.randn(num_clusters, dim), p=2, dim=1)
    eids = [f"bg_{i:07d}" for i in range(n_background)]
    cids = [i % num_clusters for i in range(n_background)]
    embs = F.normalize(
        centroids[[cids]] + 0.15 * torch.randn(n_background, dim), p=2, dim=1
    )
    projs = ["warp_cortex", "warp_align", "inference_wedge", "project_2521"]
    for i, eid in enumerate(eids):
        src.states[eid] = {"project": projs[i % 4], "idx": i, "kind": "BACKGROUND"}
        src.emb[eid] = embs[i]
        src.adj[eid] = set()
        src.clusters.setdefault(cids[i], []).append(eid)
    # sparse background graph: ring + seeded intra-cluster links
    rng = torch.Generator().manual_seed(seed)
    for i, eid in enumerate(eids):
        src.adj[eid].add(eids[(i + 1) % n_background])
        eids[(i + 1) % n_background] and src.adj[eids[(i + 1) % n_background]].add(eid)
        for _ in range(bg_edges_per_node - 1):
            j = int(torch.randint(0, n_background, (1,), generator=rng).item())
            tgt = eids[j]
            src.adj[eid].add(tgt)
            src.adj[tgt].add(eid)
    # inject the 40 premises as frozen entities (own cluster slots)
    premise_ids: List[str] = []
    for t in tasks:
        for doc_key in ("doc_a", "doc_b"):
            eid = premise_eid(t.task_id, doc_key)
            text = t.context_docs[doc_key]
            try:
                vec = encoder.encode(f"{t.visible_query} {text}")
            except Exception:
                vec = F.normalize(torch.randn(dim), p=2, dim=0)
            vec = F.normalize(vec, p=2, dim=0)
            src.states[eid] = {
                "project": "synthetic_benchmark", "task_id": t.task_id,
                "doc_key": doc_key, "premise_text": text,
                "origin": "benchmark_authored_synthetic",
            }
            src.emb[eid] = vec
            src.adj[eid] = set()
            cid = (len(eids) + len(premise_ids)) % num_clusters
            src.clusters.setdefault(cid, []).append(eid)
            premise_ids.append(eid)
    # data-driven premise edges: top-m background neighbors, blocked O(40N)
    bg_mat = torch.stack([src.emb[e] for e in eids])  # [N, 64]
    for j in range(0, len(premise_ids), 8):
        chunk = torch.stack([src.emb[e] for e in premise_ids[j:j + 8]])
        sims = chunk @ bg_mat.t()
        top = torch.topk(sims, k=premise_top_m, dim=1).indices
        for k, eid in enumerate(premise_ids[j:j + 8]):
            for nb in top[k].tolist():
                tgt = eids[nb]
                src.adj[eid].add(tgt)
                src.adj[tgt].add(eid)
    # index centroids = cluster means (same index quality for every rep)
    cent_rows = []
    for cid in range(num_clusters):
        members = src.clusters.get(cid, [])
        if members:
            cent_rows.append(torch.stack([src.emb[e] for e in members]).mean(dim=0))
        else:
            cent_rows.append(torch.zeros(dim))
    src.centroids = F.normalize(torch.stack(cent_rows), p=2, dim=1)
    src.log = [{"version": 1, "note": "scale-world freeze"}]
    return src, tasks, encoder


# ---------------------------------------------------------------------------
# Retrieval over each representation (unified mode: vector seed + BFS bridge)
# ---------------------------------------------------------------------------

def retrieve_unified_u0(src: BoringUnifiedStore, encoder, task, top_k: int = 5,
                      max_nodes: Optional[int] = 25):
    t0 = time.perf_counter()
    qvec = encoder.encode(task.visible_query)
    hits = src.vector_search(qvec, top_k=top_k, candidate_budget=400)
    wanted = set(task.required_eids)
    retrieved: List[str] = []
    if hits:
        for eid in src.bfs(hits[0][0], max_depth=3, max_nodes=max_nodes):
            if eid in wanted and eid not in retrieved:
                retrieved.append(eid)
    for eid, _ in hits:
        if eid in wanted and eid not in retrieved:
            retrieved.append(eid)
    return retrieved, (time.perf_counter() - t0) * 1000.0


def retrieve_unified_cortex(sub, encoder, task, top_k: int = 5,
                            max_nodes: Optional[int] = 25):
    t0 = time.perf_counter()
    snap = sub.current_snapshot()
    qvec = encoder.encode(task.visible_query)
    hits = snap.search_semantics_indexed(qvec, top_k=top_k, candidate_budget=400)
    wanted = set(task.required_eids)
    retrieved: List[str] = []
    if hits:
        # NOTE: WorldSnapshot.bfs iterates neighbor SETS, so with max_nodes set
        # the truncation is PYTHONHASHSEED-dependent (production finding, kept
        # as-is: architecture frozen). max_nodes=None explores full depth-3,
        # which is order-independent and is what the parity test uses.
        for eid in snap.bfs(hits[0][0], max_depth=3,
                            max_nodes=(10**9 if max_nodes is None else max_nodes)):
            if eid in wanted and eid not in retrieved:
                retrieved.append(eid)
    for eid, _ in hits:
        if eid in wanted and eid not in retrieved:
            retrieved.append(eid)
    return retrieved, (time.perf_counter() - t0) * 1000.0


def retrieve_unified_modular(m: ModularFourStore, encoder, task, top_k: int = 5,
                             max_nodes: Optional[int] = 25):
    t0 = time.perf_counter()
    qvec = encoder.encode(task.visible_query)
    hits = m.vector_search(qvec, top_k=top_k, candidate_budget=400)
    wanted = set(task.required_eids)
    retrieved: List[str] = []
    if hits:
        # full BFS through the marshaled neighbor API
        visited = {hits[0][0]}
        queue = deque([(hits[0][0], 0)])
        found: List[str] = []
        cap = 10**9 if max_nodes is None else max_nodes
        while queue and len(found) < cap:
            curr, depth = queue.popleft()
            if depth > 0:
                found.append(curr)
            if depth < 3:
                for nbr in sorted(m.get_neighbors(curr)):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, depth + 1))
        for eid in found:
            if eid in wanted and eid not in retrieved:
                retrieved.append(eid)
    for eid, _ in hits:
        if eid in wanted and eid not in retrieved:
            retrieved.append(eid)
    # state fetch + validity through doc/history stores (join cost accounted)
    for eid in list(retrieved):
        _ = m.get_state(eid)
    _ = m.check_valid(retrieved)
    return retrieved, (time.perf_counter() - t0) * 1000.0


def retrieved_texts(src_states: Dict[str, Dict[str, Any]], eids: List[str]) -> str:
    parts = []
    for eid in eids:
        st = src_states.get(eid, {})
        txt = st.get("premise_text", "")
        parts.append(f"[{eid}]\n{txt}")
    return "\n\n".join(parts) if parts else "(no premises retrieved)"


def prompt_token_count(encoder, context: str, query: str) -> int:
    tok = getattr(encoder, "tokenizer", None)
    if tok is not None:
        try:
            return len(tok.encode(f"Context:\n{context}\n\nQuestion: {query}"))
        except Exception:
            pass
    return max(1, (len(context) + len(query)) // 4)
