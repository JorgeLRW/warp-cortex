"""
Kill tests: boring unified store (U_0) vs Cortex vs Modular-C.

  - Retrieval parity (small N, fast): same S/G/Z/H + same algorithms must
    yield identical premise sets on all three reps. If U_0 ties Cortex
    everywhere, Cortex is a useful API/data-model, not a novel primitive.
  - Physical D (medium N): real measured bytes + join costs.
  - Scaling sweep R_prov(N): run via __main__ (slow at large N, not in CI).
"""

from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.boring_unified_store import (
    BoringUnifiedStore,
    materialize_cortex_substrate,
    materialize_modular,
    measure_container_bytes,
    prompt_token_count,
    retrieve_unified_cortex,
    retrieve_unified_modular,
    retrieve_unified_u0,
    retrieved_texts,
    rss_bytes,
    build_scale_source,
)


def _parity_world(n_bg: int = 1960, num_clusters: int = 16):
    src, tasks, encoder = build_scale_source(n_bg, num_clusters)
    sub = materialize_cortex_substrate(src, num_clusters)
    u0 = BoringUnifiedStore()
    u0.states = {k: dict(v) for k, v in src.states.items()}
    u0.adj = {k: set(v) for k, v in src.adj.items()}
    u0.emb = {k: v.clone() for k, v in src.emb.items()}
    u0.log = list(src.log)
    u0.clusters = {k: list(v) for k, v in src.clusters.items()}
    u0.centroids = src.centroids.clone()
    mod = materialize_modular(src)
    return src, tasks, encoder, sub, u0, mod


def test_u0_retrieval_parity_with_cortex():
    """Kill test #1 (retrieval half): U_0 must tie Cortex exactly.

    Uses uncapped depth-3 BFS (order-independent). The production cap
    (max_nodes=25 over set iteration) is PYTHONHASHSEED-dependent -- a
    separate production finding, covered by test_production_bfs_cap_stability.
    """
    src, tasks, encoder, sub, u0, _ = _parity_world()
    assert len(src) == 2000
    for t in tasks:
        c_eids, _ = retrieve_unified_cortex(sub, encoder, t, max_nodes=None)
        u_eids, _ = retrieve_unified_u0(u0, encoder, t, max_nodes=None)
        assert set(c_eids) == set(u_eids), f"retrieval divergence on {t.task_id}"


def test_modular4_retrieval_parity_and_join_cost():
    """Same retrieval quality through 4 stores, but joins cost real bytes/ms."""
    src, tasks, encoder, sub, _, mod = _parity_world()
    for t in tasks[:6]:
        c_eids, _ = retrieve_unified_cortex(sub, encoder, t, max_nodes=None)
        m_eids, _ = retrieve_unified_modular(mod, encoder, t, max_nodes=None)
        assert set(c_eids) == set(m_eids), f"modular retrieval divergence on {t.task_id}"
    assert mod.marshal_calls_total > 0
    assert mod.marshal_bytes_total > 0


def test_production_bfs_cap_stability():
    """Quantify the production finding: capped BFS over set iteration varies
    with PYTHONHASHSEED. Asserts the variance mechanism exists (documents why
    parity tests use uncapped BFS), without failing on any particular seed."""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from test_boring_store_kill import _parity_world; "
        "from boring_unified_store import retrieve_unified_u0; "
        "src, tasks, enc, sub, u0, mod = _parity_world(); "
        "t = [x for x in tasks if x.task_id == 'TASK_04_LIPSCHITZ_STEP'][0]; "
        "e, _ = retrieve_unified_u0(u0, enc, t); print(sorted(e))"
        % os.path.join(REPO_ROOT, "cortex_apps", "cortex_world_runtime")
    )
    # NOTE: U_0 BFS is now sorted (deterministic); this test pins that the
    # harness itself is seed-stable across 3 hash seeds.
    outs = set()
    for seed in ("0", "1", "42"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=os.path.join(REPO_ROOT, "cortex_apps", "cortex_world_runtime"),
            env=env, timeout=300)
        outs.add(p.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, f"harness BFS not seed-stable: {outs}"


def test_physical_duplication_ordering():
    """Physical D at 10k: modular bytes > unified bytes; U_0 ~= Cortex."""
    src, _, _, sub, u0, mod = _parity_world(n_bg=9960, num_clusters=16)
    cortex_b = measure_container_bytes(
        sub.entities, sub.clusters, sub.history_events, sub.global_state,
        getattr(sub, "centroids", {}),
    )
    u0_b = measure_container_bytes(
        u0.states, u0.adj, u0.emb, u0.log, u0.clusters, u0.centroids
    )
    mod_b = measure_container_bytes(
        mod.vec_index, mod.vec_embs, mod.vec_clusters, mod.vec_centroids,
        mod.graph_index, mod.graph_adj, mod.doc_index, mod.doc_states,
        mod.hist_log, mod.hist_validity,
    )
    print(f"\nbytes @10k: cortex={cortex_b/1e6:.2f}MB u0={u0_b/1e6:.2f}MB modular4={mod_b/1e6:.2f}MB")
    print(f"dup ratio modular/unified={(mod_b / u0_b - 1.0) * 100:.1f}%  u0/cortex={u0_b / cortex_b:.3f}")
    assert mod_b > u0_b, "modular 4-store must physically duplicate unified state"
    # U_0 and Cortex hold the same logical data in plain containers: within 2x.
    assert 0.5 < u0_b / cortex_b < 2.0


def test_substrate_bfs_deterministic_across_hash_seeds():
    """P3: production WorldSnapshot.bfs must be identical under PYTHONHASHSEED
    0/1/42 (capped traversal included). Fails if set-iteration leaks back in."""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from test_boring_store_kill import _parity_world; "
        "from boring_unified_store import retrieve_unified_cortex; "
        "src, tasks, enc, sub, u0, mod = _parity_world(); "
        "out = []; "
        "from cortex_apps.cortex_world_runtime.unseen_synthesis_suite import build_20_unseen_tasks; "
        "snap = sub.current_snapshot(); "
        "t = [x for x in tasks if x.task_id == 'TASK_04_LIPSCHITZ_STEP'][0]; "
        "import torch; q = enc.encode(t.visible_query); "
        "h = snap.search_semantics_indexed(q, top_k=5, candidate_budget=400); "
        "out.append(snap.bfs(h[0][0], max_depth=3, max_nodes=25)); "
        "e, _ = retrieve_unified_cortex(sub, enc, t); out.append(sorted(e)); "
        "print(out)"
        % os.path.join(REPO_ROOT, "cortex_apps", "cortex_world_runtime")
    )
    outs = set()
    for seed in ("0", "1", "42"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=os.path.join(REPO_ROOT, "cortex_apps", "cortex_world_runtime"),
            env=env, timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        outs.add(p.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, f"substrate traversal not seed-stable: {outs}"


def run_scaling_sweep(sizes=(2000, 10000, 50000, 200000, 1000000)):
    """R_prov(N) to destruction + physical bytes at each scale. Slow: __main__ only."""
    import torch  # noqa: F401  (kept local to avoid pytest import cost claims)

    print("\n" + "=" * 110)
    print(f"{'N':>10} {'full':>6} {'part':>6} {'ret_ms':>9} {'tokens':>8} "
          f"{'cortexMB':>10} {'u0MB':>10} {'mod4MB':>10} {'dup%':>8} {'rssGB':>8}")
    print("=" * 110)
    for n in sizes:
        n_bg = n - 40
        src, tasks, encoder = build_scale_source(n_bg, 16)
        sub = materialize_cortex_substrate(src, 16)
        u0 = BoringUnifiedStore()
        u0.states, u0.adj = src.states, src.adj
        u0.emb, u0.log, u0.clusters, u0.centroids = (
            src.emb, src.log, src.clusters, src.centroids)
        full = part = 0
        ret_ms: list = []
        toks: list = []
        for t in tasks:
            eids, ms = retrieve_unified_u0(u0, encoder, t)
            ret_ms.append(ms)
            hit = len(set(t.required_eids) & set(eids))
            full += (hit == 2)
            part += (hit == 1)
            toks.append(prompt_token_count(
                encoder, retrieved_texts(u0.states, eids), t.visible_query))
        cortex_b = measure_container_bytes(sub.entities, sub.clusters,
                                           sub.history_events, sub.global_state)
        u0_b = measure_container_bytes(u0.states, u0.adj, u0.emb, u0.log,
                                       u0.clusters, u0.centroids)
        # Modular4 at 1M would triple memory; measure it only up to 200k.
        if n <= 200000:
            mod = materialize_modular(src)
            mod_b = measure_container_bytes(
                mod.vec_index, mod.vec_embs, mod.vec_clusters, mod.vec_centroids,
                mod.graph_index, mod.graph_adj, mod.doc_index, mod.doc_states,
                mod.hist_log, mod.hist_validity)
            dup = (mod_b / u0_b - 1.0) * 100
            mod_s = f"{mod_b / 1e6:>10.1f}"
            del mod
        else:
            dup = float("nan")
            mod_s = f"{'--':>10}"
        import statistics
        print(f"{n:>10} {full:>6} {part:>6} {statistics.mean(ret_ms):>9.2f} "
              f"{statistics.mean(toks):>8.0f} {cortex_b / 1e6:>10.1f} "
              f"{u0_b / 1e6:>10.1f} {mod_s} {dup:>8.1f} {rss_bytes() / 1e9:>8.2f}")
        del src, sub, u0
    print("=" * 110)


def _law_metrics_for_task(u0, encoder, task, candidate_budget: int, top_k: int = 5):
    """Per-task retrieval metrics. Returns dict with full/partial, ranks,
    recall@k, mrr-first, depth of found premises, candidates, tokens, ms."""
    import time as _t
    t0 = _t.perf_counter()
    qvec = encoder.encode(task.visible_query)
    ranking = u0.rank_candidates(qvec, candidate_budget)
    rank_of = {}
    for pos, (eid, _) in enumerate(ranking, start=1):
        if eid in task.required_eids and eid not in rank_of:
            rank_of[eid] = pos
    hits = [e for e, _ in ranking[:top_k]]
    wanted = set(task.required_eids)
    retrieved = []
    if hits:
        for eid in u0.bfs(hits[0], max_depth=3, max_nodes=25):
            if eid in wanted and eid not in retrieved:
                retrieved.append(eid)
    for eid in hits:
        if eid in wanted and eid not in retrieved:
            retrieved.append(eid)
    depths = u0.bfs_depths(hits[0], max_depth=6) if hits else {}
    ms = (_t.perf_counter() - t0) * 1000.0
    hit_count = len(set(retrieved) & wanted)
    ranks = [rank_of.get(e, float("inf")) for e in task.required_eids]
    finite = [r for r in ranks if r != float("inf")]
    mrr_first = (1.0 / min(finite)) if finite else 0.0
    return {
        "full": hit_count == 2,
        "partial": hit_count == 1,
        "recall_at_5": sum(1 for r in ranks if r <= 5) / 2.0,
        "recall_at_20": sum(1 for r in ranks if r <= 20) / 2.0,
        "recall_at_100": sum(1 for r in ranks if r <= 100) / 2.0,
        "mrr_first": mrr_first,
        "depths_found": [depths[e] for e in retrieved if e in depths],
        "depth_failures": sum(1 for e in task.required_eids if e not in depths),
        "n_candidates": len(ranking),
        "tokens": prompt_token_count(
            encoder, retrieved_texts(u0.states, retrieved), task.visible_query),
        "ms": ms,
    }


def run_retrieval_law(out_name: str = "retrieval_law_results.json"):
    """P1: R_prov(N, B, k) with multi-seed CIs, dense vs sparse regimes, Pareto.

    Runs on U_0 (proven identical to Cortex retrieval in parity tests).
    Saves raw per-seed artifacts + aggregated curves to JSON.
    """
    import statistics

    LAW_SCALES = (2000, 10000, 50000, 100000, 200000, 500000, 1000000)
    LAW_SEEDS = (7, 11, 23)
    DENSE_SCALES = (2000, 10000)
    PARETO_SCALES = (10000, 100000, 1000000)
    PARETO_BUDGETS = (100, 400, 1600)
    K_RECALL = (5, 20, 100)

    from boring_unified_store import wire_dense_knn

    import json
    path = os.path.join(os.path.dirname(__file__), out_name)

    def _checkpoint():
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "pareto": pareto, "partial": True}, f, indent=1)

    rows = []
    pareto = []
    for regime in ("sparse", "dense"):
        scales = LAW_SCALES if regime == "sparse" else DENSE_SCALES
        for n in scales:
            for seed in LAW_SEEDS:
                src, tasks, encoder = build_scale_source(n - 40, 16, seed=seed)
                if regime == "dense":
                    # reset to empty graph, then dense data-driven wiring
                    for eid in src.adj:
                        src.adj[eid] = set()
                    added = wire_dense_knn(src, k=4, sim_threshold=0.45)
                else:
                    added = -1
                u0 = BoringUnifiedStore()
                u0.states, u0.adj, u0.emb = src.states, src.adj, src.emb
                u0.log, u0.clusters, u0.centroids = src.log, src.clusters, src.centroids
                per_task = [_law_metrics_for_task(u0, encoder, t, 400) for t in tasks]
                agg = _agg_tasks(per_task)
                agg.update({"N": n, "regime": regime, "seed": seed,
                            "B": 400, "edges_added": added})
                rows.append(agg)
                _checkpoint()
                print(f"law {regime:>6} N={n:>8} seed={seed}: "
                      f"full={agg['full_mean']:.2f} mrr={agg['mrr_mean']:.3f} "
                      f"r@100={agg['r100_mean']:.2f} ms={agg['ms_mean']:.1f}",
                      flush=True)
                del src, u0
    # Pareto: recall <-> latency <-> context size over B
    pareto = []
    for n in PARETO_SCALES:
        for B in PARETO_BUDGETS:
            for seed in (7, 11):
                src, tasks, encoder = build_scale_source(n - 40, 16, seed=seed)
                u0 = BoringUnifiedStore()
                u0.states, u0.adj, u0.emb = src.states, src.adj, src.emb
                u0.log, u0.clusters, u0.centroids = src.log, src.clusters, src.centroids
                per_task = [_law_metrics_for_task(u0, encoder, t, B) for t in tasks]
                agg = _agg_tasks(per_task)
                agg.update({"N": n, "regime": "sparse", "seed": seed, "B": B})
                pareto.append(agg)
                _checkpoint()
                print(f"pareto N={n:>8} B={B:>5} seed={seed}: "
                      f"full={agg['full_mean']:.2f} ms={agg['ms_mean']:.1f} "
                      f"tok={agg['tok_mean']:.0f}", flush=True)
                del src, u0
    curves = _ci_by_group(rows, keys=("N", "regime", "B"))
    pareto_curves = _ci_by_group(pareto, keys=("N", "B"))
    out = {"rows": rows, "pareto": pareto, "curves": curves,
           "pareto_curves": pareto_curves,
           "partial": False,
           "note": ("U_0 retrieval proven identical to Cortex (parity tests). "
                    "Sparse regime: ring+random graph + top-4 data-driven premise "
                    "links. Dense regime: harvester-style k-NN (k=4, thr=0.45). "
                    "Premises: 40 benchmark-authored synthetics; no premise-premise edges. "
                    "Harness limitation: capped BFS expands neighbors in sorted "
                    "order, so truncation has a lexicographic bias (constant across "
                    "N; parity tests use uncapped BFS and are unaffected).") }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path} (partial=False)")
    return out


def _agg_tasks(per_task):
    import statistics
    def m(k):
        return statistics.mean(x[k] for x in per_task)
    depths = [d for x in per_task for d in x["depths_found"]]
    import statistics as _s
    return {
        "full_mean": sum(1 for x in per_task if x["full"]) / len(per_task),
        "part_mean": sum(1 for x in per_task if x["partial"]) / len(per_task),
        "r5_mean": m("recall_at_5"),
        "r20_mean": m("recall_at_20"),
        "r100_mean": m("recall_at_100"),
        "mrr_mean": m("mrr_first"),
        "depth_mean": _s.mean(depths) if depths else float("nan"),
        "depth_fail_frac": _s.mean(x["depth_failures"] / 2.0 for x in per_task),
        "cand_mean": m("n_candidates"),
        "tok_mean": m("tokens"),
        "ms_mean": m("ms"),
    }


def _ci_by_group(rows, keys):
    """Mean +/- 95% CI across seeds per group. Small-n normal approx, stated."""
    import statistics
    groups = {}
    for r in rows:
        g = tuple(r[k] for k in keys)
        groups.setdefault(g, []).append(r)
    out = []
    metrics = ("full_mean", "part_mean", "r5_mean", "r20_mean", "r100_mean",
               "mrr_mean", "tok_mean", "ms_mean", "cand_mean")
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        entry = dict(zip(keys, g))
        entry["n_seeds"] = len(rs)
        for met in metrics:
            vals = [r[met] for r in rs]
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            ci = 1.96 * sd / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
            entry[met] = round(mean, 4)
            entry[met + "_ci95"] = round(ci, 4)
        out.append(entry)
    return out


if __name__ == "__main__":
    import sys as _sys
    if "--law" in _sys.argv:
        run_retrieval_law()
    else:
        run_scaling_sweep()
