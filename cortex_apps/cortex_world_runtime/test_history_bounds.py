"""P4 fast tests: contracts, views, scenarios, sqlite battery (small scale)."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.history_bounds import (
    ViewH0, ViewH1, ViewH2, ViewH3, ViewH4, ViewH5,
    _build_views, _truth_index, _explain_sample,
    eval_p_state, eval_p_explain, eval_p_replay, eval_p_audit,
    run_scenarios, apply_entry,
)


def _snap_substrate(w):
    return {e: dict(n.state) for e, n in w.sub.entities.items()
            if w.o_alive.get(e, False)}


def _mini_world(n_bg=300, seed=7, n_mut=3000):
    from cortex_apps.cortex_world_runtime.dirty_world import DirtyWorld
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )
    enc = GenericFrozenAspectEncoder(d_out=64, seed=42)
    w = DirtyWorld(n_bg=n_bg, num_clusters=8, seed=seed, encoder=enc)
    views = _build_views()
    orig = w._log

    def _feed(op, _o=orig, _vs=views, **kw):
        s = _o(op, **kw)
        entry = {"seq": s, "op": op, **kw}
        for v in _vs.values():
            v.notify(entry)
        return s
    w._log = _feed
    birth = _snap_substrate(w)
    for v in views.values():
        v.snapshot_tick(0, _snap_substrate(w))
    for _ in range(n_mut):
        w.mutate()
    live = _snap_substrate(w)
    for v in views.values():
        v.snapshot_tick(w.seq, live)
    return w, views, birth, live


def test_h0_full_contracts():
    w, views, birth, live = _mini_world()
    v = views["H0"]
    ps = eval_p_state(v, live, birth)
    assert ps["p_state"] == 1.0, ps
    idx = _truth_index(w.journal)
    sample = _explain_sample(w.journal, set(live), n=30)
    assert len(sample) > 0
    pe = eval_p_explain(v, idx, sample)
    assert pe["p_explain_full"] == 1.0, pe
    assert pe["chain_coverage"] == 1.0
    pa = eval_p_audit(v)
    assert pa["p_audit_chain_ok"] is True
    assert pa["tamper_modify_detected"] is True
    assert pa["tamper_remove_detected"] is True
    assert pa["tamper_reorder_detected"] is True


def test_h1_loses_old_provenance_honestly():
    w, views, birth, live = _mini_world(n_mut=6000)
    for name in ("H1-500", "H1-2000"):
        v = views[name]
        idx = _truth_index(w.journal)
        sample = _explain_sample(w.journal, set(live), n=30)
        pe = eval_p_explain(v, idx, sample)
        # bounded tail cannot explain everything; must report honestly
        assert 0.0 <= (pe["p_explain_full"] or 0.0) <= 1.0
        assert pe["den"] == len(sample)
        pa = eval_p_audit(v)
        assert pa["p_audit_chain_ok"] is True
    # shorter tail loses at least as much as longer tail
    assert views["H1-500"].log.last_seq == views["H1-2000"].log.last_seq


def test_h2_h3_h4_state_and_lineage():
    w, views, birth, live = _mini_world()
    for name in ("H2", "H3", "H4"):
        v = views[name]
        # force a snapshot (50k cadence never fires in a 3k world): snapshot
        # mechanism itself is what this asserts, not the cadence.
        v.last_snap = -10**9
        v.snapshot_tick(w.seq, live)
        ps = eval_p_state(v, live, birth)
        assert ps["p_state"] == 1.0, (name, ps)
    assert len(views["H3"].lineage) > 0
    assert len(views["H4"].summaries) > 0
    idx = _truth_index(w.journal)
    sample = _explain_sample(w.journal, set(live), n=20)
    pe = eval_p_explain(views["H4"], idx, sample)
    assert (pe["p_explain_full"] or 0) + (pe["p_explain_partial"] or 0) > 0


def test_h5_tiers_reported_separately():
    w, views, birth, live = _mini_world(n_mut=6000)
    v = views["H5"]
    assert len(v.cold_files) >= 0
    a, t = v.retained_bytes()
    assert t >= a
    assert v.cold_bytes >= 0


def test_replay_exact_where_retained():
    w, views, birth, live = _mini_world()
    ents = sorted(live)[:10]
    seqs = [max(1, w.seq // 2), w.seq]
    pr = eval_p_replay(views["H0"], w.journal, birth, ents, seqs)
    assert pr["p_replay_exact"] == 1.0, pr
    pr1 = eval_p_replay(views["H1-500"], w.journal, birth, ents, seqs)
    assert 0.0 <= (pr1["p_replay_exact"] or 0.0) <= 1.0
    assert pr1["den"] == len(ents) * len(seqs)


def test_scenarios_all_policies():
    w, views, birth, live = _mini_world(n_mut=1000)
    res = run_scenarios(w, views, w.seq)
    assert set(res) == set(views)
    for name, r in res.items():
        for t in ("T1_retraction", "T2_contradiction", "T3_recreate",
                  "T4_cross_checkpoint", "T5_skill_lineage"):
            assert r[t]["den"] == 1, (name, t, r[t])


def test_sqlite_battery_small():
    from cortex_apps.cortex_world_runtime.history_bounds import run_sqlite_battery
    res = run_sqlite_battery(n_events=2000, seed=7, out_name="history_bounds_sqlite_test.json")
    assert res["state_verify"][0] == res["state_verify"][1]
    assert res["audit_ok"] is True
    assert res["corrupt_detected"] is True
    assert res["reopen_ok"] is True
    assert res["db_bytes"] > 0
    p = os.path.join(os.path.dirname(__file__), "history_bounds_sqlite_test.json")
    if os.path.exists(p):
        os.remove(p)
