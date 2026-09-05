"""harvest_file(path): watcher-style ingest -- chunk, frozen-encode, upsert, kNN.

Uses the frozen task-agnostic encoder (encoder id recorded in the manifest;
aspect_dim + encoder_id are manifest fields so worlds survive encoder swaps).
Default kNN edges k=4, sim>=0.45 (same defaults as the research harvester).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + max_chars])
        i += max_chars - overlap
    return [c for c in chunks if c.strip()]


def harvest_file(store, path: str, encoder=None, project: str = "default",
                 k: int = 4, sim_threshold: float = 0.45) -> List[str]:
    """Ingest one file into the portable world. Returns chunk entity ids."""
    if encoder is None:
        from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
            GenericFrozenAspectEncoder,
        )
        encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    rel = os.path.relpath(os.path.abspath(path), os.path.dirname(store.root))
    eids = []
    for ci, chunk in enumerate(chunk_text(text)):
        eid = f"file::{rel}::chunk{ci:03d}"
        vec = encoder.encode(chunk)
        arr = np.asarray(vec.tolist() if hasattr(vec, "tolist") else vec, dtype="<f4")
        seq = store.commit("ingest", {"entity": eid, "file": rel, "chunk": ci})
        store.upsert_node(eid, {"project": project, "file": rel, "chunk": ci,
                                "text": chunk[:400], "type": "FILE_CHUNK"},
                          aspect_vec=arr, event_seq=seq, mirror_md=False)
        eids.append(eid)
    _knn_link(store, eids, k=k, sim_threshold=sim_threshold)
    # md mirrors for ingested chunks (cheap; state+edges only)
    for eid in eids:
        node = store.get_node(eid)
        if node is not None:
            store._write_md_mirror(eid, node["state"], node["version"])
    return eids


def _knn_link(store, eids: List[str], k: int = 4, sim_threshold: float = 0.45):
    vecs = {}
    for eid in eids:
        node = store.get_node(eid)
        if node is not None and node["aspect"] is not None:
            v = node["aspect"].astype(np.float64)
            n = float(np.linalg.norm(v)) or 1.0
            vecs[eid] = v / n
    ids = sorted(vecs.keys())
    if len(ids) < 2:
        return
    mat = np.stack([vecs[e] for e in ids])
    sims = mat @ mat.T
    from cortex_apps.cortex_world_runtime.cortex_world.graph import EDGE_TYPES
    assert "mentions" in EDGE_TYPES
    for i, eid in enumerate(ids):
        order = np.argsort(sims[i])[::-1]
        added = 0
        for j in order:
            if ids[j] == eid:
                continue
            if float(sims[i, j]) >= sim_threshold:
                store.add_edge(eid, ids[j], "mentions")
                added += 1
                if added >= k:
                    break
