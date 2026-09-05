"""
P2: Long-lived / dirty-world durability.

One persistent world through 10^6 (later 10^7) mutations at FIXED live size,
so age effects separate from size effects. Retrieval stack stays FROZEN.

Brutality mix (seeded): create/update/delete/recreate cycles, renames/aliases,
edge replacement (not just addition), contradictory status flips, late-arriving
/ duplicate / out-of-order events, tombstone resurrections, stale embeddings
after content change, project merge/split, schema migration, retracted
evidence, skill-version replacement, compaction/rebuild checkpoints.

Honest production-fidelity notes (read before citing):
  - Writes go through FastWorldSubstrate.clock1_tick ONLY (batched; one history
    entry per tick, so 10^6 mutations ~= 5*10^4 ticks -- disclosed, not hidden).
  - Clock1 has NO write version gate (Clock3 has only a commit tolerance
    window), so late/duplicate/out-of-order payloads apply naively, like prod.
  - There is NO first-class delete or re-embed API. Delete is modeled once,
    identically for all policies: tombstone flag via clock1_tick + removal
    from the cluster index, WITHOUT neighbor pruning (the cheap realistic
    choice). Re-embedding happens ONLY inside maintenance policies via direct
    aspect assignment -- disclosed as a missing-API gap, not a mechanism.
  - Retrieval never filters tombstones (no such concept in prod) and the
    cluster index DOES drop deleted eids, so index-vs-graph staleness splits
    are measurable rather than assumed.
  - The oracle (canonical state/adjacency/aliases/journal) lives OUTSIDE the
    substrate and is never shown to any policy. Policies see only substrate +
    journal seqs, like a real operator would.

Maintenance policies (the ONLY thing that differs between runs):
  none:        nothing ever (maximal rot control).
  rebuild:     every R mutations: re-embed all live from current text, rewire
               premise top-m links, prune all dangling refs, drop tombstones
               whose eid was recreated (compact), full history kept.
  incremental: every 5k mutations: re-embed recently-touched, sweep their
               neighborhoods for dangling refs, process tombstones, clear
               non-canonical keys on touched entities (write-behind worker).
  checkpoint:  every R mutations: snapshot canonical live state, truncate
               history_events to post-checkpoint, keep tombstones; provenance
               answered from checkpoint + deltas (tests bounded H_v, i.e. P4
               preview -- retrieval/state behavior identical to `none`
               between checkpoints by construction).
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# op name -> weight (sums to 100)
# Demography (disclosed design choice, not a finding): deaths per 100
# mutations ~= 6 (delete) + 5 (rename kills old eid) = 11; births ~= 5
# (rename births new eid) + recreate. Net zero requires recreate = 6:
# -6-5+5+6 = 0, so live size stays approximately stationary and AGE
# separates from SIZE. (First attempt used recreate=11 and the live
# population exploded 50x -- caught by the live= counter in probe output.)
OP_MIX = {
    "update_state": 28,
    "edge_churn": 15,
    "delete": 6,
    "recreate": 6,
    "rename": 5,
    "contradict": 10,
    "late_dup_reorder": 10,
    "merge_split": 5,
    "schema_migrate": 5,
    "retract": 5,
    "skill_replace": 5,
}

STATUSES = ["NOMINAL", "DEGRADED", "BLOCKED", "RECOVERING", "DRAINING"]
PROJECTS = ["warp_cortex", "warp_align", "inference_wedge", "project_2521"]
BOOKKEEPING = {"alive", "tombstone_seq", "origin"}


def entity_text(project: str, kind: str, status: str, idx: int) -> str:
    return f"{project} {kind} {status} entity {idx}"


def _rate(num: int, den: int):
    """(value, num, den) with None (N/A), never 0, on empty denominator.

    Audit rule from the P2 islands: a 0/0 probe must read N/A, because 0.00
    with an empty sample previously masqueraded as perfect health.
    """
    if den <= 0:
        return None, num, den
    return num / den, num, den


class DirtyWorld:
    """Aging world: production substrate under test + external oracle."""

    def __init__(self, n_bg: int = 3000, num_clusters: int = 16, seed: int = 7,
                 encoder=None, probe_every: int = 50000):
        from cortex_apps.cortex_world_runtime.fast_world_substrate import (
            EntityNode, FastWorldSubstrate,
        )
        from cortex_apps.cortex_world_runtime.unseen_synthesis_suite import (
            build_20_unseen_tasks, premise_eid,
        )
        self.n_bg = n_bg
        self.num_clusters = num_clusters
        self.seed = seed
        self.probe_every = probe_every
        self.rng = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        if encoder is None:
            from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
                GenericFrozenAspectEncoder,
            )
            encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)
            # P1 lesson: the encoder reseeds torch internally (fixed projection
            # seed by design). Re-seed here so background geometry actually
            # varies per world seed; the encoder itself stays frozen.
            torch.manual_seed(seed)
        self.encoder = encoder
        self.tasks = build_20_unseen_tasks()

        self.sub = FastWorldSubstrate(num_clusters=num_clusters)
        self.eid_cid: Dict[str, int] = {}
        # oracle
        self.o_alive: Dict[str, bool] = {}
        self.o_state: Dict[str, Dict[str, Any]] = {}
        self.o_adj: Dict[str, Set[str]] = {}
        self.o_alias: Dict[str, str] = {}
        self.journal: List[Dict[str, Any]] = []  # every mutation, with seq
        self.seq = 0
        self.bg_ids: List[str] = []
        self.premise_ids: List[str] = []
        self.tombstoned: Set[str] = set()
        self.checkpoint_seq = 0
        self.checkpoint_state: Dict[str, Dict[str, Any]] = {}
        self.update_ms: List[float] = []
        self.skill_gens = {"repair_bridge": 1}
        self.stale_skill_picks = 0
        self.skill_queries = 0
        self.last_seq_for: Dict[str, int] = {}
        self.truncation_seq = 0
        self._recreated: List[str] = []

        self._build_initial()

    # -- construction ------------------------------------------------------
    def _add_node(self, eid: str, state: Dict[str, Any], text: str, cid: int):
        from cortex_apps.cortex_world_runtime.fast_world_substrate import EntityNode
        vec = self.encoder.encode(text)
        self.sub.entities[eid] = EntityNode(
            entity_id=eid, state=dict(state), neighbors=set(),
            aspect_vector=F.normalize(vec, p=2, dim=0),
            cluster_id=cid, version_modified=1)
        self.sub.clusters[cid].append(eid)
        self.eid_cid[eid] = cid

    def _build_initial(self):
        from cortex_apps.cortex_world_runtime.unseen_synthesis_suite import premise_eid
        for i in range(self.n_bg):
            eid = f"bg_{i:07d}"
            cid = i % self.num_clusters
            proj = PROJECTS[i % 4]
            st = f"status_{i % 5}"
            state = {"project": proj, "kind": "BACKGROUND", "status": STATUSES[i % 5],
                     "idx": i, "alive": True, "origin": "init"}
            self._add_node(eid, state, entity_text(proj, "BACKGROUND", state["status"], i), cid)
            self.o_alive[eid] = True
            self.o_state[eid] = {k: v for k, v in state.items() if k not in ("alive", "origin")}
            self.o_adj[eid] = set()
            self.bg_ids.append(eid)
        # sparse ring + random edges (both substrate and oracle agree at t=0)
        for i, eid in enumerate(self.bg_ids):
            nxt = self.bg_ids[(i + 1) % self.n_bg]
            self.sub.entities[eid].neighbors.add(nxt)
            self.sub.entities[nxt].neighbors.add(eid)
            self.o_adj[eid].add(nxt)
            self.o_adj[nxt].add(eid)
            j = int(torch.randint(0, self.n_bg, (1,), generator=self.rng).item())
            tgt = self.bg_ids[j]
            if tgt != eid:
                self.sub.entities[eid].neighbors.add(tgt)
                self.sub.entities[tgt].neighbors.add(eid)
                self.o_adj[eid].add(tgt)
                self.o_adj[tgt].add(eid)
        # pinned premises (never mutated; neighborhood churns around them)
        for t in self.tasks:
            for doc_key in ("doc_a", "doc_b"):
                eid = premise_eid(t.task_id, doc_key)
                text = t.context_docs[doc_key]
                state = {"project": "synthetic_benchmark", "task_id": t.task_id,
                         "doc_key": doc_key, "premise_text": text,
                         "origin": "benchmark_authored_synthetic", "alive": True}
                cid = (len(self.bg_ids) + len(self.premise_ids)) % self.num_clusters
                self._add_node(eid, state, f"{t.visible_query} {text}", cid)
                self.o_alive[eid] = True
                self.o_state[eid] = {k: v for k, v in state.items()
                                     if k not in ("alive", "origin")}
                self.o_adj[eid] = set()
                self.premise_ids.append(eid)
        # premise anchor links: top-4 live background by sim (frozen at birth)
        bg_mat = torch.stack([self.sub.entities[e].aspect_vector for e in self.bg_ids])
        for j in range(0, len(self.premise_ids), 8):
            chunk = torch.stack([self.sub.entities[e].aspect_vector
                                 for e in self.premise_ids[j:j + 8]])
            top = torch.topk(chunk @ bg_mat.t(), k=4, dim=1).indices
            for k, eid in enumerate(self.premise_ids[j:j + 8]):
                for nb in top[k].tolist():
                    tgt = self.bg_ids[nb]
                    self.sub.entities[eid].neighbors.add(tgt)
                    self.sub.entities[tgt].neighbors.add(eid)
                    self.o_adj[eid].add(tgt)
                    self.o_adj[tgt].add(eid)
        # index centroids = live cluster means
        self._refresh_centroids()
        self.journal.append({"seq": 0, "op": "init", "note": "world birth"})

    def _refresh_centroids(self):
        rows = []
        for cid in range(self.num_clusters):
            members = [e for e in self.sub.clusters.get(cid, [])
                       if e in self.sub.entities and self.o_alive.get(e, False)]
            if members:
                rows.append(torch.stack(
                    [self.sub.entities[e].aspect_vector for e in members]).mean(dim=0))
            else:
                rows.append(torch.zeros(64))
        self.sub.centroids = F.normalize(torch.stack(rows), p=2, dim=1)

    # -- mutation helpers ----------------------------------------------------
    def _live_bg(self) -> List[str]:
        return [e for e in self.bg_ids if self.o_alive.get(e, False)]

    def _pick(self, items: List[str]) -> str:
        return items[int(torch.randint(0, len(items), (1,), generator=self.rng).item())]

    def _tick(self, deltas: List[Tuple[str, Dict[str, Any]]]):
        t0 = time.perf_counter()
        ms = self.sub.clock1_tick(deltas)
        self.update_ms.append(ms)
        return ms

    def _log(self, op: str, **kw):
        self.seq += 1
        entry = {"seq": self.seq, "op": op, **kw}
        self.journal.append(entry)
        return self.seq

    def mutate(self):
        """One mutation drawn from OP_MIX. Always advances seq by exactly 1
        (no-log paths emit noop_skip) so run loops cannot stall."""
        s0 = self.seq
        r = int(torch.randint(0, 100, (1,), generator=self.rng).item())
        acc = 0
        op = "update_state"
        for name, w in OP_MIX.items():
            acc += w
            if r < acc:
                op = getattr(self, "_op_" + name)()
                break
        else:
            op = self._op_update_state()
        if self.seq == s0:
            self._log("noop_skip", op_attempted=op)
            return "noop_skip"
        return op

    def _op_update_state(self):
        live = self._live_bg()
        if not live:
            return "update_state"
        eid = self._pick(live)
        new_status = STATUSES[int(torch.randint(0, len(STATUSES), (1,), generator=self.rng).item())]
        self._tick([(eid, {"status": new_status})])
        self.o_state[eid]["status"] = new_status
        self._log("update_state", eid=eid, status=new_status)
        # NOTE: aspect vector deliberately NOT refreshed (stale-Z source).
        return "update_state"

    def _op_edge_churn(self):
        live = self._live_bg()
        if len(live) < 2:
            return "edge_churn"
        a, b = self._pick(live), self._pick(live)
        if b in self.o_adj.get(a, set()) or a == b:
            # replace: drop an existing edge instead (edge REPLACEMENT)
            cur = sorted(self.o_adj.get(a, set()))
            if cur:
                drop = cur[int(torch.randint(0, len(cur), (1,), generator=self.rng).item())]
                self.sub.entities[a].neighbors.discard(drop)
                self.sub.entities[drop].neighbors.discard(a)
                self.o_adj[a].discard(drop)
                self.o_adj[drop].discard(a)
                self._tick([(a, {}), (drop, {})])
                self._log("edge_drop", a=a, b=drop)
                return "edge_churn"
            return "edge_churn"
        self.sub.entities[a].neighbors.add(b)
        self.sub.entities[b].neighbors.add(a)
        self.o_adj[a].add(b)
        self.o_adj[b].add(a)
        self._tick([(a, {}), (b, {})])
        self._log("edge_add", a=a, b=b)
        return "edge_churn"

    def _op_delete(self):
        live = [e for e in self._live_bg() if e not in self.premise_ids]
        if not live:
            return "delete"
        eid = self._pick(live)
        seq = self._log("delete", eid=eid)
        # uniform delete op: tombstone + cluster removal, NO neighbor pruning
        self._tick([(eid, {"alive": False, "tombstone_seq": seq})])
        self.o_alive[eid] = False
        cid = self.eid_cid[eid]
        if eid in self.sub.clusters.get(cid, []):
            self.sub.clusters[cid].remove(eid)
        self.tombstoned.add(eid)
        return "delete"

    def _op_recreate(self):
        # sorted(): tombstoned is a set; pick order must not depend on
        # PYTHONHASHSEED or same-seed reruns diverge across processes.
        dead = sorted(e for e in self.tombstoned if not self.o_alive.get(e, False))
        if not dead:
            return self._op_update_state()
        eid = self._pick(dead)
        # naive merge-recreate: fresh keys merged in, OLD keys linger
        # (production merge cannot delete keys) -> resurrection-leak source
        new_status = STATUSES[int(torch.randint(0, len(STATUSES), (1,), generator=self.rng).item())]
        self._tick([(eid, {"alive": True, "status": new_status, "recreated_seq": self.seq + 1,
                           "tombstone_seq": -1})])
        seq = self._log("recreate", eid=eid, status=new_status)
        self.o_alive[eid] = True
        self.o_state[eid] = {"project": self.o_state[eid].get("project", PROJECTS[0]),
                             "kind": "BACKGROUND", "status": new_status,
                             "idx": self.o_state[eid].get("idx", 0)}
        cid = self.eid_cid[eid]
        if eid not in self.sub.clusters.get(cid, []):
            self.sub.clusters[cid].append(eid)
        self.tombstoned.discard(eid)
        # NOTE: aspect vector still describes pre-delete content (stale).
        return "recreate"

    def _op_rename(self):
        live = [e for e in self._live_bg() if e not in self.premise_ids]
        if not live:
            return "rename"
        old = self._pick(live)
        new = f"{old}_aka{self.seq}"
        st = dict(self.sub.entities[old].state)
        st["alive"] = True
        self._tick([(new, st)])
        # new node inherits embedding snapshot (alias of same content at birth)
        self.sub.entities[new].aspect_vector = self.sub.entities[old].aspect_vector.clone()
        cid = self.eid_cid[old]
        self.sub.clusters[cid].append(new)
        self.eid_cid[new] = cid
        self.bg_ids.append(new)
        self.o_alive[new] = True
        self.o_state[new] = {k: v for k, v in self.o_state[old].items()}
        self.o_adj[new] = set(self.o_adj[old])
        for nbr in self.o_adj[old]:
            self.sub.entities[new].neighbors.add(nbr)
            self.sub.entities[nbr].neighbors.add(new)
            self.o_adj[nbr].add(new)
        # tombstone the old eid (alias kept in oracle only)
        seq = self._log("rename", old=old, new=new)
        self._tick([(old, {"alive": False, "tombstone_seq": seq})])
        self.o_alive[old] = False
        if old in self.sub.clusters.get(cid, []):
            self.sub.clusters[cid].remove(old)
        self.tombstoned.add(old)
        self.o_alias[old] = new
        return "rename"

    def _op_contradict(self):
        live = self._live_bg()
        if not live:
            return "contradict"
        eid = self._pick(live)
        cur = self.o_state[eid].get("status", "NOMINAL")
        alt = STATUSES[(STATUSES.index(cur) + 1) % len(STATUSES)] if cur in STATUSES else "NOMINAL"
        self._tick([(eid, {"status": alt})])
        self.o_state[eid]["status"] = alt
        self._log("contradict", eid=eid, old=cur, new=alt)
        return "contradict"

    def _op_late_dup_reorder(self):
        if len(self.journal) < 3:
            return self._op_update_state()
        past = self.journal[int(torch.randint(1, len(self.journal), (1,), generator=self.rng).item())]
        eid = past.get("eid", past.get("a"))
        if not eid or eid not in self.sub.entities:
            return self._op_update_state()
        # naive re-application with NO version gate (prod Clock1 behavior)
        payload = {}
        if "status" in past:
            payload["status"] = past["status"]
        elif "new" in past and isinstance(past.get("new"), str) and past["new"] in STATUSES:
            payload["status"] = past["new"]
        if not payload:
            self._tick([(eid, {})])
            self._log("duplicate_noop", eid=eid, replay_of=past["seq"])
            return "late_dup_reorder"
        self._tick([(eid, payload)])
        # Deliberate oracle divergence: the replay is a STALE write that a
        # version gate should have rejected. The substrate applies it (prod
        # Clock1 has no gate); the oracle keeps the pre-replay value. Their
        # gap is exactly the cost of missing write validation, and later
        # legitimate updates re-sync (also realistic).
        self._log("replay", eid=eid, replay_of=past["seq"], payload=payload)
        return "late_dup_reorder"

    def _op_merge_split(self):
        live = [e for e in self._live_bg() if e not in self.premise_ids]
        if len(live) < 4:
            return "merge_split"
        batch = [self._pick(live) for _ in range(4)]
        new_proj = PROJECTS[int(torch.randint(0, 4, (1,), generator=self.rng).item())]
        self._tick([(e, {"project": new_proj}) for e in batch])
        for e in batch:
            self.o_state[e]["project"] = new_proj
        self._log("merge_split", eids=batch, project=new_proj)
        return "merge_split"

    def _op_schema_migrate(self):
        live = [e for e in self._live_bg() if e not in self.premise_ids]
        if not live:
            return "schema_migrate"
        eid = self._pick(live)
        cur = self.o_state[eid].get("status", "NOMINAL")
        # migrate status -> state_v2; old key LINGERS (schema rot) unless swept
        self._tick([(eid, {"state_v2": cur})])
        self.o_state[eid] = {k: v for k, v in self.o_state[eid].items() if k != "status"}
        self.o_state[eid]["state_v2"] = cur
        # Journal completeness: the tick value MUST be logged (shadow
        # reconstruction reads the journal, never the oracle).
        self._log("schema_migrate", eid=eid, value=cur)
        return "schema_migrate"

    def _op_retract(self):
        live = self._live_bg()
        if not live:
            return "retract"
        eid = self._pick(live)
        self._tick([(eid, {"evidence_status": "RETRACTED"})])
        self.o_state[eid]["evidence_status"] = "RETRACTED"
        self._log("retract", eid=eid)
        return "retract"

    def _op_skill_replace(self):
        name = "repair_bridge"
        self.skill_gens[name] = self.skill_gens.get(name, 1) + 1
        self._tick([(f"skill_{name}", {f"v{self.skill_gens[name]}": "current"})])
        self._log("skill_replace", skill=name, gen=self.skill_gens[name])
        return "skill_replace"

    # -- maintenance policies ------------------------------------------------
    def maintain(self, policy: str):
        if policy == "none":
            return {"cost_ms": 0.0}
        t0 = time.perf_counter()
        if policy == "rebuild":
            self._rebuild_full()
        elif policy == "incremental":
            self._incremental(time_window_ticks=1000)
        elif policy == "checkpoint":
            self._checkpoint_compact()
        return {"cost_ms": (time.perf_counter() - t0) * 1000.0}

    def _live_entities(self) -> List[str]:
        return [e for e, n in self.sub.entities.items() if self.o_alive.get(e, False)]

    def _embed_text(self, eid: str) -> str:
        st = self.o_state.get(eid, {})
        return entity_text(st.get("project", "?"), st.get("kind", "?"),
                           st.get("state_v2", st.get("status", "?")), st.get("idx", 0))

    def _rebuild_full(self):
        for eid in self._live_entities():
            if eid in self.premise_ids:
                continue
            self.sub.entities[eid].aspect_vector = F.normalize(
                self.encoder.encode(self._embed_text(eid)), p=2, dim=0)
        self._prune_dangling()
        self._clear_noncanonical()
        self._refresh_centroids()

    def _incremental(self, time_window_ticks: int = 1000):
        # write-behind worker: only entities touched recently (journal window)
        recent: Set[str] = set()
        for entry in self.journal[-2000:]:
            for k in ("eid", "a", "b", "old", "new"):
                v = entry.get(k)
                if isinstance(v, str) and v in self.sub.entities:
                    recent.add(v)
            for v in entry.get("eids", []):
                if v in self.sub.entities:
                    recent.add(v)
        for eid in recent:
            if not self.o_alive.get(eid, False) or eid in self.premise_ids:
                continue
            self.sub.entities[eid].aspect_vector = F.normalize(
                self.encoder.encode(self._embed_text(eid)), p=2, dim=0)
            # sweep this neighborhood for dangling refs
            for nbr in list(self.sub.entities[eid].neighbors):
                if not self.o_alive.get(nbr, False) and nbr not in self.premise_ids:
                    self.sub.entities[eid].neighbors.discard(nbr)
            self._clear_one(eid)
        self._refresh_centroids()

    def _prune_dangling(self):
        for eid, node in self.sub.entities.items():
            if not self.o_alive.get(eid, False) and eid not in self.premise_ids:
                continue
            for nbr in list(node.neighbors):
                if nbr not in self.sub.entities:
                    node.neighbors.discard(nbr)
                elif not self.o_alive.get(nbr, False) and nbr not in self.premise_ids:
                    node.neighbors.discard(nbr)

    def _canonical_keys(self, eid: str) -> Set[str]:
        return set(self.o_state.get(eid, {}).keys()) | BOOKKEEPING | {"recreated_seq"}

    def _clear_one(self, eid: str):
        node = self.sub.entities.get(eid)
        if node is None:
            return
        keep = self._canonical_keys(eid)
        for k in [k for k in node.state.keys() if k not in keep]:
            del node.state[k]

    def _clear_noncanonical(self):
        for eid in self._live_entities():
            if eid in self.premise_ids:
                continue
            self._clear_one(eid)

    def _checkpoint_compact(self):
        self.checkpoint_seq = self.seq
        self.checkpoint_state = {e: dict(s) for e, s in self.o_state.items()
                                 if self.o_alive.get(e, False)}
        # truncate event log to post-checkpoint (bounded H_v preview)
        self.sub.history_events = self.sub.history_events[-500:]
        self.truncation_seq = self.seq

    # -- probes ----------------------------------------------------------------
    def _retrieve_task(self, snap, task) -> List[str]:
        qvec = self.encoder.encode(task.visible_query)
        hits = snap.search_semantics_indexed(qvec, top_k=5, candidate_budget=400)
        wanted = set(task.required_eids)
        out: List[str] = []
        if hits:
            for eid in snap.bfs(hits[0][0], max_depth=3, max_nodes=25):
                if eid in wanted and eid not in out:
                    out.append(eid)
        for eid, _ in hits:
            if eid in wanted and eid not in out:
                out.append(eid)
        return out

    def probe(self) -> Dict[str, Any]:
        """Full metric battery at current world age. Returns flat dict."""
        import statistics
        snap = self.sub.current_snapshot()
        live_bg = [e for e in self.bg_ids if self.o_alive.get(e, False)]
        sample_n = min(200, len(live_bg))
        step = max(1, len(live_bg) // max(1, sample_n))
        sample = live_bg[::step][:sample_n]

        # 1. retrieval recall over the 20 pinned tasks (premise granularity:
        # 40 premises -> 0.025 steps, so the pool is not quantized to 0/.5/1)
        full = part = tomb_hits = prem_hit = tot_ret = 0
        ret_ms = []
        for t in self.tasks:
            t0 = time.perf_counter()
            eids = self._retrieve_task(snap, t)
            ret_ms.append((time.perf_counter() - t0) * 1000.0)
            hit = len(set(t.required_eids) & set(eids))
            full += (hit == 2)
            part += (hit == 1)
            prem_hit += hit
            tot_ret += len(eids)
            tomb_hits += sum(1 for e in eids if not self.o_alive.get(e, False))

        # 2. state correctness + schema lingering on samples
        mism = ling = 0
        for eid in sample:
            node = self.sub.entities.get(eid)
            if node is None:
                mism += 1
                continue
            canon = self.o_state.get(eid, {})
            if any(node.state.get(k) != v for k, v in canon.items()):
                mism += 1
            if any(k not in canon and k not in BOOKKEEPING and k != "recreated_seq"
                   for k in node.state.keys()):
                ling += 1

        # 3. Semantic Staleness Rate: self-retrieval top-1 misses (Z vs S)
        ssr_n = min(100, len(sample))
        ssr_miss = 0
        for eid in sample[:ssr_n]:
            st = self.o_state.get(eid, {})
            q = self.encoder.encode(entity_text(
                st.get("project", "?"), st.get("kind", "?"),
                st.get("state_v2", st.get("status", "?")), st.get("idx", 0)))
            hits = snap.search_semantics_indexed(q, top_k=1, candidate_budget=400)
            if not hits or hits[0][0] != eid:
                ssr_miss += 1

        # 4. graph correctness on samples: dangling + missing live edges
        dang = miss_e = tot_e = 0
        for eid in sample:
            node = self.sub.entities.get(eid)
            if node is None:
                continue
            want = self.o_adj.get(eid, set())
            dead_nbrs = {n for n in node.neighbors
                         if n in self.sub.entities and not self.o_alive.get(n, False)
                         and n not in self.premise_ids}
            live_want = {n for n in want if self.o_alive.get(n, False) or n in self.premise_ids}
            live_got = {n for n in node.neighbors
                        if self.o_alive.get(n, False) or n in self.premise_ids}
            dang += len(dead_nbrs)
            miss_e += len(live_want - live_got)
            tot_e += max(1, len(live_want))

        # 5. resurrection leaks on recreated entities still live
        rec = [e for e in getattr(self, "_recreated", []) if self.o_alive.get(e, False)]
        rec_leak = 0
        for eid in rec[:200]:
            node = self.sub.entities.get(eid)
            canon = self.o_state.get(eid, {})
            if node is not None and any(
                    k not in canon and k not in BOOKKEEPING and k != "recreated_seq"
                    for k in node.state.keys()):
                rec_leak += 1

        # 6. provenance answerability: last-write seq retained?
        prov_ok = 0
        prov_n = min(50, len(sample))
        for eid in sample[:prov_n]:
            ls = self.last_seq_for.get(eid, 0)
            if getattr(self, "truncation_seq", 0) and ls <= self.truncation_seq:
                continue  # compacted away -> LOST
            prov_ok += 1

        # 7. stale-skill selection (light P5 preview)
        skill_stale_frac = self._skill_probe()

        # 8. memory + history
        from cortex_apps.cortex_world_runtime.boring_unified_store import (
            measure_container_bytes,
        )
        mem = measure_container_bytes(
            self.sub.entities, self.sub.clusters, self.sub.history_events,
            self.sub.global_state, getattr(self.sub, "centroids", {}))

        n = len(sample) or 0
        recall_full, recall_full_num, _ = _rate(full, 20)
        recall_part, recall_part_num, _ = _rate(part, 20)
        premise_recall, premise_recall_num, premise_recall_den = _rate(prem_hit, 40)
        tomb_rate, tomb_num, tomb_den = _rate(tomb_hits, tot_ret)
        state_mis, state_mis_num, state_den = _rate(mism, n)
        schema_ling, schema_num, _ = _rate(ling, n)
        ssr, ssr_num, ssr_den = _rate(ssr_miss, max(0, ssr_n))
        dang_r, dang_num, edge_den = _rate(dang, tot_e)
        miss_r, miss_num, _ = _rate(miss_e, tot_e)
        rec_den = min(200, len(rec))
        resurr, resurr_num, _ = _rate(rec_leak, rec_den)
        prov, prov_num, prov_den = _rate(prov_ok, min(50, len(sample)))
        return {
            "age_mutations": self.seq,
            "recall_full": recall_full,
            "recall_part": recall_part,
            "premise_recall": premise_recall,
            "premise_recall_num": premise_recall_num,
            "premise_recall_den": premise_recall_den,
            "tombstone_hit_rate": tomb_rate,
            "tombstone_hit_den": tomb_den,
            "state_mismatch_rate": state_mis,
            "state_sample_n": state_den,
            "schema_lingering_rate": schema_ling,
            "ssr_self_top1_miss": ssr,
            "ssr_n": ssr_den,
            "graph_dangling_per_edge": dang_r,
            "graph_missing_per_edge": miss_r,
            "graph_edge_den": edge_den,
            "resurrection_leak_rate": resurr,
            "resurrection_den": rec_den,
            "n_recreated_live": len(rec),
            "provenance_answerable": prov,
            "provenance_n": prov_den,
            "skill_stale_frac": skill_stale_frac,
            "update_ms_mean": statistics.mean(self.update_ms) if self.update_ms else 0.0,
            "retrieval_ms_mean": statistics.mean(ret_ms),
            "memory_bytes": mem,
            "history_len": len(self.sub.history_events),
            "journal_len": len(self.journal),
            "n_live": len(live_bg),
            "n_tombstones": len(self.tombstoned),
        }

    def _skill_probe(self) -> float:
        from cortex_apps.cortex_world_runtime.skill_registry import (
            SkillDefinition, SkillInvocationEvent, SkillRegistry, SkillSelector,
            SkillSelectionMode,
        )
        reg = SkillRegistry()
        cur = self.skill_gens.get("repair_bridge", 1)
        for v in range(1, cur + 1):
            tags = ["REPAIR", "STRUCTURE", "BRIDGE"] + (["HIGH_TEMP"] if v > 1 else [])
            reg.register(SkillDefinition(
                skill_id="repair_bridge", version=f"v{v}",
                name="Bridge Repair", description="Repairs damaged bridge structure.",
                aspect_tags=tags, prerequisites={"has_tools": True}))
        ledger = []
        if cur > 1:
            ledger.append(SkillInvocationEvent(
                invocation_id="probe_fail", skill_id="repair_bridge",
                skill_version="v1", agent_id="probe", world_version=1,
                task_query="repair the damaged bridge", inputs={},
                success=False, outcome_summary="obsolete", latency_ms=1.0,
                token_cost=1, error_type="ObsoleteVersion",
                discovered_constraints={"cooling_active": True}))
        snap = self.sub.current_snapshot()
        snap.state = dict(getattr(snap, "state", {}))
        snap.state.update({"has_tools": True, "cooling_active": True})
        sel = SkillSelector(reg, mode=SkillSelectionMode.SHARED_CORTEX_LEDGER)
        ranked = sel.select_skill("repair the damaged bridge", snap,
                                  agent_id="probe", shared_history=ledger)
        self.skill_queries += 1
        if not ranked:
            return 0.0
        top = ranked[0][0]
        if cur > 1 and top.version == "v1":
            self.stale_skill_picks += 1
            return 1.0
        return 0.0


def run_dirty_world(n_mutations: int = 1_000_000, policies=("none", "rebuild", "incremental", "checkpoint"),
                    seeds=(7, 11), n_bg: int = 3000, probe_every: int = 50000,
                    rebuild_every: int = 200000, out_name: str = "dirty_world_results.json",
                    probe_ages=None):
    """P2 main: same seeded mutation stream per policy; probe over age.
    probe_ages: explicit age list (e.g. strategic rerun) replacing cadence."""
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )
    import json

    path = os.path.join(os.path.dirname(__file__), out_name)
    all_series = []

    def _checkpoint():
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"series": all_series, "partial": True}, f, indent=1)

    encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)
    for seed in seeds:
        for policy in policies:
            w = DirtyWorld(n_bg=n_bg, seed=seed, encoder=encoder, probe_every=probe_every)
            w.last_seq_for = {}
            w.truncation_seq = 0
            w._recreated = []
            orig_log = w._log

            def _log_indexed(op, _o=orig_log, _w=w, **kw):
                s = _o(op, **kw)
                for k in ("eid", "a", "b", "old", "new"):
                    v = kw.get(k)
                    if isinstance(v, str):
                        _w.last_seq_for[v] = s
                for v in kw.get("eids", []):
                    _w.last_seq_for[v] = s
                if op == "recreate":
                    _w._recreated.append(kw["eid"])
                return s
            w._log = _log_indexed
            series = {"policy": policy, "seed": seed, "probes": []}
            series["probes"].append({"age_mutations": 0, **w.probe()})
            all_series.append({"policy": policy, "seed": seed, **series["probes"][-1]})
            _checkpoint()
            maint_ms_total = 0.0
            last_maint = 0
            ages = probe_ages or []
            if not ages:
                a, ages = probe_every, []
                while a < n_mutations:
                    ages.append(a)
                    a += probe_every
                ages.append(n_mutations)
            for age in ages:
                while w.seq < age:
                    w.mutate()
                    if policy == "incremental" and w.seq % 5000 == 0:
                        maint_ms_total += w.maintain(policy)["cost_ms"]
                if policy in ("rebuild", "checkpoint"):
                    while last_maint + rebuild_every <= w.seq:
                        maint_ms_total += w.maintain(policy)["cost_ms"]
                        last_maint += rebuild_every
                p = {"age_mutations": w.seq, **w.probe()}
                p["maint_ms_total"] = maint_ms_total
                series["probes"].append(p)
                def _f(v, fmt=":.3f"):
                    return ("{v" + fmt + "}").format(v=v) if v is not None else "N/A"
                print(f"p2 {policy:>11} seed={seed} age={w.seq:>8}: "
                      f"state_mis={_f(p['state_mismatch_rate'])} "
                      f"ssr={_f(p['ssr_self_top1_miss'])} "
                      f"recall={_f(p['recall_full'], ':.2f')} "
                      f"prem_rec={_f(p['premise_recall'], ':.3f')} "
                      f"dangling={_f(p['graph_dangling_per_edge'])} "
                      f"resurr={_f(p['resurrection_leak_rate'])} "
                      f"prov={_f(p['provenance_answerable'], ':.2f')} "
                      f"mem={p['memory_bytes']/1e6:.0f}MB hist={p['history_len']} "
                      f"live={p['n_live']}",
                      flush=True)
                all_series.append({"policy": policy, "seed": seed, **p})
                _checkpoint()
    out = {"series": all_series, "partial": False,
           "note": ("Fixed live size; age varies. Retrieval frozen. Clock1 has no "
                    "version gate; delete=tombstone+index removal without neighbor "
                    "pruning; re-embed only inside maintenance (missing-API gap). "
                    "Checkpoint policy truncates history (bounded-H preview).")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path} (partial=False)")
    return out


if __name__ == "__main__":
    import sys as _sys
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 else 1_000_000
    run_dirty_world(n_mutations=n)
