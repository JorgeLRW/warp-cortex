"""Tests for the portable cortex_world package (product path, tmp dirs)."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from cortex_apps.cortex_world_runtime.cortex_world.store import (
    PortableWorld, open_world, FORMAT_VERSION,
)
from cortex_apps.cortex_world_runtime.cortex_world import graph as G
from cortex_apps.cortex_world_runtime.cortex_world.recall import recall
from cortex_apps.cortex_world_runtime.cortex_world import skills as SK
from cortex_apps.cortex_world_runtime.cortex_world.ingest import harvest_file


@pytest.fixture(scope="module")
def encoder():
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )
    return GenericFrozenAspectEncoder(d_out=64, seed=42)


def _vec(rng, dim=64):
    v = rng.standard_normal(dim).astype("<f4")
    return v / float(np.linalg.norm(v))


def test_open_manifest_budgets(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    try:
        assert w.manifest["format"] == FORMAT_VERSION
        assert w.manifest["budgets"]["semantic_candidates"] == 400
        assert w.manifest["encoder"]["aspect_dim"] == 64
        assert w.manifest["history"]["hot_events"] == 2000
    finally:
        w.close()


def test_commit_upsert_bfs_recall(tmp_path, encoder):
    rng = np.random.default_rng(0)
    w = open_world(str(tmp_path / "proj"))
    try:
        s1 = w.commit("note", {"text": "cache tile size thirty three"})
        w.upsert_node("a", {"project": "p", "title": "tile", "text": "cache tile 33"},
                      aspect_vec=_vec(rng), event_seq=s1)
        s2 = w.commit("note", {"text": "bank count thirty two"})
        w.upsert_node("b", {"project": "p", "title": "banks", "text": "banks 32"},
                      aspect_vec=_vec(rng), event_seq=s2)
        w.add_edge("a", "b", "depends_on")
        assert w.neighbors("a") == ["b"]
        assert G.bfs(w, "a") == ["b"]
        res = recall(w, encoder.encode("cache tile size"))
        assert res.candidate_budget == 400
        assert res.candidates_examined == 2
        assert {h.entity_id for h in res.hits} == {"a", "b"}
        hit = {h.entity_id: h for h in res.hits}["a"]
        assert hit.event_seq == s1
        assert any(p.startswith("node:a@v") for p in hit.provenance)
        # md mirror written, sqlite canonical
        assert os.path.exists(os.path.join(w.root, "entities", "a.md"))
        assert "NOT recoverable" in open(
            os.path.join(w.root, "entities", "a.md")).read()
    finally:
        w.close()


def test_dag_ops_and_cascade(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    try:
        for eid in ("p1", "p2", "app", "note"):
            w.upsert_node(eid, {"project": "p"}, mirror_md=False)
        w.add_edge("app", "p1", "depends_on")
        w.add_edge("app", "p2", "depends_on")
        w.add_edge("app", "note", "mentions")
        w.add_edge("p2", "p1", "derived_from")
        assert G.ancestors(w, "p1") == {"app", "p2"}
        assert G.descendants(w, "app") == {"p1", "p2", "note"}
        order = G.topo_sort(w)
        assert order.index("app") < order.index("p1")
        # cascade follows causal edges only: mentions never invalidates
        assert G.cascade_invalidate(w, "p1") == {"app", "p2"}
        assert G.cascade_invalidate(w, "note") == set()
        with pytest.raises(ValueError):
            w.add_edge("a", "b", "vibes")
    finally:
        w.close()


def test_articulation_points(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    try:
        for eid in ("hub", "x", "y"):
            w.upsert_node(eid, {"project": "p"}, mirror_md=False)
        w.add_edge("hub", "x", "mentions")
        w.add_edge("hub", "y", "mentions")
        assert G.articulation_points(w) == {"hub"}
    finally:
        w.close()


def test_history_cap_and_snapshots(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    try:
        m = w._read_manifest()
        m["history"]["hot_events"] = 50
        w._write_manifest(m)
        w.manifest = m
        for i in range(300):
            w.commit("tick", {"i": i})
        n = w.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        snaps = w.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        assert n <= 101  # hot + one window
        assert snaps >= 1
    finally:
        w.close()


def test_crash_recovery_verify(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    w.commit("note", {"a": 1})
    w.upsert_node("a", {"project": "p"}, mirror_md=False)
    w.close()
    w2 = open_world(str(tmp_path / "proj"))  # startup verify runs here
    try:
        assert w2.get_node("a")["state"] == {"project": "p"}
    finally:
        w2.close()


def test_skills_scoped_and_mirrored(tmp_path):
    w = open_world(str(tmp_path / "proj"))
    try:
        md = "# Repair\nRepairs bridges."
        SK.register_skill(w, "repair", "v1", md)
        assert os.path.exists(os.path.join(w.root, "skills", "repair", "SKILL.md"))
        SK.record_invocation(w, "repair", "v1", True, project="projA")
        SK.record_invocation(w, "repair", "v1", False, project="projB")
        rank_a = SK.select_skill(w, "repair bridges", project="projA")
        assert rank_a and rank_a[0][0] == "repair"
        # project scoping: projB history must not leak into projA ranking input
        ha = SK._history(w, "repair", "projA")
        hb = SK._history(w, "repair", "projB")
        assert len(ha) == len(hb) == 1 and ha[0]["success"] != hb[0]["success"]
    finally:
        w.close()


def test_harvest_file_roundtrip(tmp_path, encoder):
    src = tmp_path / "notes.txt"
    src.write_text("Cache tile size M must be coprime with bank count B. " * 40)
    w = open_world(str(tmp_path / "proj"))
    try:
        eids = harvest_file(w, str(src), encoder=encoder, project="projA")
        assert len(eids) >= 1
        res = recall(w, encoder.encode("coprime tile bank"))
        assert res.candidates_examined >= 1
        assert res.hits
    finally:
        w.close()


def test_cli_roundtrip(tmp_path):
    from cortex_apps.cortex_world_runtime.cortex_world.cli import main
    proj = str(tmp_path / "proj")
    assert main(["open", proj]) == 0
    assert main(["bfs", proj, "ghost"]) == 0
    assert main(["select-skill", proj, "repair"]) == 0
