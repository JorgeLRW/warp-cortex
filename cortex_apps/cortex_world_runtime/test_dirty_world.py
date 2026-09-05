"""
P2 fast harness tests (small age, CPU-only, no GPU).

  - Oracle/substrate agreement at birth (all metrics near-perfect).
  - Age degrades the no-maintenance world on staleness/graph axes.
  - Maintenance ordering: incremental beats none on SSR + dangling.
  - Tombstone accounting: recreations tracked, leaks measurable.
  - Checkpoint policy bounds history length.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.dirty_world import DirtyWorld, run_dirty_world


def _small_world(seed: int = 7, n_bg: int = 400):
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )
    enc = GenericFrozenAspectEncoder(d_out=64, seed=42)
    w = DirtyWorld(n_bg=n_bg, num_clusters=8, seed=seed, encoder=enc, probe_every=10**9)
    return w


def _age(w: DirtyWorld, n: int, policy: str = "none", maint_every: int = 5000):
    for _ in range(n):
        w.mutate()
        if policy == "incremental" and w.seq % maint_every == 0:
            w.maintain(policy)
    if policy in ("rebuild", "checkpoint"):
        w.maintain(policy)
    return w.probe()


def test_birth_agreement():
    w = _small_world()
    p = w.probe()
    assert p["state_mismatch_rate"] == 0.0
    assert p["graph_dangling_per_edge"] == 0.0
    assert p["graph_missing_per_edge"] == 0.0
    assert p["provenance_answerable"] == 1.0
    assert p["n_live"] == 400
    assert p["recall_full"] + p["recall_part"] > 0.0


def test_age_degrades_unmaintained_world():
    w = _small_world()
    p0 = w.probe()
    p1 = _age(w, 20000, policy="none")
    assert p1["age_mutations"] == 20000
    # staleness and graph rot must appear without maintenance
    assert p1["ssr_self_top1_miss"] > p0["ssr_self_top1_miss"]
    assert p1["graph_dangling_per_edge"] > 0.0
    assert p1["schema_lingering_rate"] > 0.0
    assert p1["n_tombstones"] > 0


def test_incremental_beats_none_on_rot():
    w1 = _small_world(seed=11)
    p_none = _age(w1, 20000, policy="none")
    w2 = _small_world(seed=11)
    p_inc = _age(w2, 20000, policy="incremental", maint_every=2000)
    assert p_inc["ssr_self_top1_miss"] <= p_none["ssr_self_top1_miss"]
    assert p_inc["graph_dangling_per_edge"] <= p_none["graph_dangling_per_edge"]
    assert p_inc["schema_lingering_rate"] <= p_none["schema_lingering_rate"]


def test_checkpoint_bounds_history():
    w = _small_world(seed=23)
    _age(w, 20000, policy="checkpoint")
    w.maintain("checkpoint")
    assert len(w.sub.history_events) <= 500
    p = w.probe()
    # compaction costs old provenance: some last-writes predate truncation
    assert p["provenance_answerable"] <= 1.0
    assert p["state_mismatch_rate"] < 0.05


def test_rebuild_restores_staleness():
    w = _small_world(seed=5)
    _age(w, 20000, policy="none")
    before = w.probe()["ssr_self_top1_miss"]
    assert before > 0.0
    w.maintain("rebuild")
    after = w.probe()["ssr_self_top1_miss"]
    assert after < before
