"""recall(query): nodes + edge paths + event seqs + degradation metadata.

Returns a contextual object (RecallResult), not bare strings. Every hit
carries provenance; the result additionally exposes the operational budget
and how much of the world was actually examined, so callers can see
retrieval degradation instead of silently receiving thin context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RecallHit:
    entity_id: str
    state: Dict[str, Any]
    score: float
    edge_path: List[str]
    event_seq: int
    provenance: List[str]
    version: int


@dataclass
class RecallResult:
    hits: List[RecallHit]
    snapshot_version: int
    candidate_budget: int
    candidates_examined: int
    top_k: int
    note: str = ""


def _all_embeddings(store):
    rows = store.db.execute(
        "SELECT id, aspect, cluster, updated_seq, version FROM nodes "
        "WHERE aspect IS NOT NULL ORDER BY id").fetchall()
    ids = [r[0] for r in rows]
    mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, mat / norms, [r[2] for r in rows]


def _centroids(store, dim: int):
    rows = store.db.execute("SELECT DISTINCT cluster FROM nodes ORDER BY cluster").fetchall()
    cents = []
    cids = []
    for (cid,) in rows:
        erows = store.db.execute(
            "SELECT aspect FROM nodes WHERE cluster=? AND aspect IS NOT NULL", (cid,)).fetchall()
        if not erows:
            continue
        m = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in erows]).astype(np.float64)
        c = m.mean(axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        cents.append(c / n)
        cids.append(cid)
    import numpy as _np
    return cids, (_np.stack(cents) if cents else _np.zeros((0, dim)))


def recall(store, query_vec, top_k: Optional[int] = None,
           candidate_budget: Optional[int] = None) -> RecallResult:
    """recall(query) -> RecallResult. Budgets default to manifest (operational,
    not guarantees). Provenance per hit: BFS edge path + node updated_seq."""
    from cortex_apps.cortex_world_runtime.cortex_world.graph import bfs
    b = store.manifest.get("budgets", {})
    top_k = b.get("top_k", 5) if top_k is None else top_k
    candidate_budget = b.get("semantic_candidates", 400) if candidate_budget is None else candidate_budget

    q = np.asarray(
        query_vec.tolist() if hasattr(query_vec, "tolist") else query_vec, dtype=np.float64)
    n = float(np.linalg.norm(q)) or 1.0
    q = q / n

    ids, mat, clusters = _all_embeddings(store)
    if not ids:
        return RecallResult([], store.version, candidate_budget, 0, top_k,
                            note="empty world")
    cids, cents = _centroids(store, q.shape[0])
    order = np.argsort(cents @ q)[::-1] if len(cents) else []
    pool: List[str] = []
    by_cluster: Dict[int, List[str]] = {}
    for eid, cid in zip(ids, clusters):
        by_cluster.setdefault(cid, []).append(eid)
    for oi in order:
        pool.extend(by_cluster.get(cids[int(oi)], []))
        if len(pool) >= candidate_budget:
            break
    pool = pool[:candidate_budget]
    idx = {e: i for i, e in enumerate(ids)}
    scored = sorted(((e, float(mat[idx[e]] @ q)) for e in pool if e in idx),
                    key=lambda x: x[1], reverse=True)
    hits: List[RecallHit] = []
    for eid, s in scored[:top_k]:
        node = store.get_node(eid)
        if node is None:
            continue
        path = bfs(store, eid, max_depth=1, max_nodes=6)
        prov = [f"node:{eid}@v{node['version']}",
                f"event_seq:{node['updated_seq']}"]
        if path:
            prov.append(f"bridge:{eid}->{'->'.join(path[:3])}")
        hits.append(RecallHit(entity_id=eid, state=node["state"], score=s,
                              edge_path=path, event_seq=node["updated_seq"],
                              provenance=prov, version=node["version"]))
    note = ""
    if len(pool) < candidate_budget:
        note = "candidate pool smaller than budget (small world)"
    return RecallResult(hits=hits, snapshot_version=store.version,
                        candidate_budget=candidate_budget,
                        candidates_examined=len(pool), top_k=top_k, note=note)
