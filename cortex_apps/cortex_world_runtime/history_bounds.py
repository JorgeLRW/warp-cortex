"""
P4: How much history must a persistent world retain?

Information-retention / bounded-memory experiment. Architecture frozen;
this is purely an H-retention representation policy comparison.

Design (paired streams):
  - ONE DirtyWorld mutation stream per seed is shared by ALL policies, so
    differences reflect retention, never the stream. Retention views are
    passive: they never touch S/G/Z (P1/P2 stay frozen).
  - The full journal is harness-side ground truth, never shown to views.

Policies:
  H0 full log (upper-bound provenance control).
  H1 hard hot-tail K in {500, 2000, 10000}, no summarization (naive failure).
  H2 periodic exact snapshot + bounded tail (intervals/tails varied).
  H3 H2 + per-entity lineage heads (last event, prev root, sources).
  H4 H3 + compact causal summaries (justification, sources, retraction
     markers, checkpoint linkage); raw intermediates dropped.
  H5 H2 + tiered cold archive (gzip segments; active vs total reported
     separately; NOT called bounded if total grows linearly).

Contracts (separate, never one ambiguous prov metric):
  A. P_state: retained data reconstructs exact current S_v.
  B. P_explain: terminal source event (+predecessor chain where retained).
  C. P_replay: exact state at sampled historical versions.
  D. P_audit: hash-chain / checkpoint-root verification + tamper detection.

Every ratio: (value, num, den), null on empty den. Measured, never modeled.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _rate(num: int, den: int):
    if den <= 0:
        return None, num, den
    return num / den, num, den


def _chain(prev: str, seq: int, op: str, payload: str) -> str:
    return hashlib.sha256(f"{prev}|{seq}|{op}|{payload}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Shadow-S: reconstruct live state ONLY from retained events (merge semantics
# mirror the substrate tick: updates merge, keys are never deleted by writes).
# ---------------------------------------------------------------------------

def apply_entry(shadow: Dict[str, Dict[str, Any]], alive: Set[str], entry: Dict[str, Any]):
    op = entry.get("op")
    if op == "update_state" and entry.get("status") is not None:
        shadow.setdefault(entry["eid"], {})["status"] = entry["status"]
        alive.add(entry["eid"])
    elif op == "contradict" and entry.get("new") is not None:
        # contradict journals old/new (tick payload was {"status": new})
        shadow.setdefault(entry["eid"], {})["status"] = entry["new"]
        alive.add(entry["eid"])
    elif op == "replay":
        eid, payload = entry.get("eid"), entry.get("payload") or {}
        if eid and payload.get("status") is not None:
            shadow.setdefault(eid, {})["status"] = payload["status"]
            alive.add(eid)
    elif op == "merge_split":
        for e in entry.get("eids", []):
            shadow.setdefault(e, {})["project"] = entry.get("project")
            alive.add(e)
    elif op == "schema_migrate":
        eid = entry.get("eid")
        if eid and entry.get("value") is not None:
            # substrate tick merges {"state_v2": cur}, keeping old status
            # (lingering by design); value is journaled (completeness rule).
            shadow.setdefault(eid, {})["state_v2"] = entry["value"]
            alive.add(eid)
    elif op == "retract":
        eid = entry.get("eid")
        if eid:
            shadow.setdefault(eid, {})["evidence_status"] = "RETRACTED"
            alive.add(eid)
    elif op == "scenario_revise":
        eid = entry.get("eid")
        if eid and entry.get("claim") is not None:
            shadow.setdefault(eid, {})["claim"] = entry["claim"]
            alive.add(eid)
    elif op == "delete":
        eid = entry.get("eid")
        if eid:
            shadow.setdefault(eid, {}).update(
                {"alive": False, "tombstone_seq": entry.get("seq", -1)})
            alive.discard(eid)
    elif op == "recreate":
        eid = entry.get("eid")
        if eid:
            # tick ran before logging with recreated_seq = the seq this entry
            # receives, so the value is recoverable exactly.
            shadow.setdefault(eid, {}).update(
                {"alive": True, "status": entry.get("status"),
                 "recreated_seq": entry.get("seq", -1), "tombstone_seq": -1})
            alive.add(eid)
    elif op == "rename":
        old, new = entry.get("old"), entry.get("new")
        if old and new:
            shadow[new] = dict(shadow.get(old, {}))
            shadow[new]["alive"] = True
            if old in shadow:
                shadow[old].update({"alive": False})
            alive.discard(old)
            alive.add(new)
    elif op == "scenario_support":
        eid = entry.get("eid")
        if eid:
            shadow.setdefault(eid, {}).update(
                {"supported_by": entry.get("evidence"), "claim": entry.get("claim")})
            alive.add(eid)
    elif op in ("scenario_untrusted", "scenario_trusted"):
        eid = entry.get("eid")
        if eid:
            shadow.setdefault(eid, {}).update(
                {"val": entry.get("val"), "trust": entry.get("trust")})
            alive.add(eid)
    elif op == "scenario_cause":
        eid = entry.get("eid")
        if eid:
            shadow.setdefault(eid, {})["cause_marker"] = entry.get("marker")
            alive.add(eid)
    # edge ops / skill_replace / init / noop / scenario_skill_outcome: no S effect.


# ---------------------------------------------------------------------------
# Retention views
# ---------------------------------------------------------------------------

class RetentionView:
    name = "base"

    def notify(self, entry: Dict[str, Any]):
        raise NotImplementedError

    def snapshot_tick(self, seq: int, live_state: Dict[str, Dict[str, Any]]):
        """Optional periodic hook (exact live S offered; views may copy)."""

    def retained_bytes(self) -> Tuple[int, int]:
        """(active_bytes, total_bytes) serialized retained payload."""
        raise NotImplementedError

    # -- contract primitives -------------------------------------------------
    def terminal_source(self, eid: str, key: str):
        """-> (kind, seq|snapshot-id|None, detail). kind in
        exact/snapshot/summary/miss."""
        raise NotImplementedError

    def touches_for(self, eid: str, upto_seq: int):
        """Retained entries touching eid with seq <= upto_seq, ascending."""
        raise NotImplementedError

    def verify_chain(self) -> Tuple[bool, int]:
        """(ok, entries_checked) over retained hash-chained log."""
        raise NotImplementedError


class _ChainedLog:
    """Append-time hash-chained retained log shared by views."""

    def __init__(self):
        self.entries: deque = deque()
        self.head = "GENESIS"
        # exact O(1) indexes (no scans): last touch per eid; contiguous seqs
        self.last_touch: Dict[str, Dict[str, Any]] = {}
        self.first_seq: Optional[int] = None
        self.last_seq: int = 0

    def append(self, entry: Dict[str, Any]):
        blob = json.dumps(entry, sort_keys=True, default=str)
        self.head = _chain(self.head, int(entry.get("seq", 0)),
                           str(entry.get("op")), blob)
        self.entries.append({"entry": entry, "link": self.head})
        for eid in _entry_eids(entry):
            self.last_touch[eid] = entry
        if self.first_seq is None:
            self.first_seq = int(entry.get("seq", 0))
        self.last_seq = int(entry.get("seq", 0))

    def evict_left(self):
        """Pop oldest; keep last_touch exact (invalidate only if it WAS oldest)."""
        rec = self.entries.popleft()
        e = rec["entry"]
        for eid in _entry_eids(e):
            if self.last_touch.get(eid, {}).get("seq") == e.get("seq"):
                del self.last_touch[eid]
        if self.entries:
            self.first_seq = int(self.entries[0]["entry"].get("seq", 0))
        return rec

    def verify(self) -> Tuple[bool, int]:
        h = "GENESIS"
        n = 0
        for rec in self.entries:
            e = rec["entry"]
            h = _chain(h, int(e.get("seq", 0)), str(e.get("op")),
                       json.dumps(e, sort_keys=True, default=str))
            if h != rec["link"]:
                return False, n
            n += 1
        return True, n

    def bytes(self) -> int:
        return len(json.dumps(
            [{"e": r["entry"], "l": r["link"]} for r in self.entries], default=str))


class ViewH0(RetentionView):
    name = "H0-full"

    def __init__(self):
        self.log = _ChainedLog()

    def notify(self, entry):
        self.log.append(entry)

    def retained_bytes(self):
        b = self.log.bytes()
        return b, b

    def touches_for(self, eid, upto_seq):
        out = []
        for rec in self.log.entries:
            e = rec["entry"]
            if int(e.get("seq", 0)) > upto_seq:
                break
            if _touches(e, eid):
                out.append(e)
        return out

    def terminal_source(self, eid, key):
        # exact fast path: last touch is cached and (eviction-synced) retained
        cached = self.log.last_touch.get(eid)
        if cached is not None and _changes_key(cached, key):
            return ("exact", cached.get("seq"), cached.get("op"))
        for rec in reversed(self.log.entries):
            e = rec["entry"]
            if _touches(e, eid) and _changes_key(e, key):
                return ("exact", e.get("seq"), e.get("op"))
        return ("miss", None, None)

    def verify_chain(self):
        return self.log.verify()


class ViewH1(RetentionView):
    name = "H1-tail"

    def __init__(self, keep: int):
        self.keep = keep
        self.name = f"H1-tail{keep}"
        self.log = _ChainedLog()
        # chain must restart honestly at truncation: anchor records the
        # dropped head so verification covers the RETAINED window only.
        self.anchor = "GENESIS"

    def notify(self, entry):
        self.log.append(entry)
        while len(self.log.entries) > self.keep:
            dropped = self.log.evict_left()
            self.anchor = dropped["link"]

    def retained_bytes(self):
        b = self.log.bytes()
        return b, b

    def touches_for(self, eid, upto_seq):
        out = []
        for rec in self.log.entries:
            e = rec["entry"]
            if int(e.get("seq", 0)) > upto_seq:
                break
            if _touches(e, eid):
                out.append(e)
        return out

    def terminal_source(self, eid, key):
        cached = self.log.last_touch.get(eid)
        if cached is not None and _changes_key(cached, key):
            return ("exact", cached.get("seq"), cached.get("op"))
        for rec in reversed(self.log.entries):
            e = rec["entry"]
            if _touches(e, eid) and _changes_key(e, key):
                return ("exact", e.get("seq"), e.get("op"))
        return ("miss", None, None)

    def verify_chain(self):
        # verify retained window against the truncation anchor
        h = self.anchor
        n = 0
        for rec in self.log.entries:
            e = rec["entry"]
            h = _chain(h, int(e.get("seq", 0)), str(e.get("op")),
                       json.dumps(e, sort_keys=True, default=str))
            if h != rec["link"]:
                return False, n
            n += 1
        return True, n


class ViewH2(RetentionView):
    name = "H2-snap-tail"

    def __init__(self, tail: int = 2000, snap_every: int = 50000):
        self.tail_view = ViewH1(tail)
        self.tail = tail
        self.snap_every = snap_every
        self.name = f"H2-snap{snap_every}-tail{tail}"
        self.snapshots: List[Dict[str, Any]] = []  # {seq, state, live, hash}
        self.last_snap = 0

    def notify(self, entry):
        self.tail_view.notify(entry)

    def snapshot_tick(self, seq, live_state):
        if seq - self.last_snap < self.snap_every:
            return
        blob = json.dumps(live_state, sort_keys=True, default=str)
        self.snapshots.append({
            "seq": seq,
            "state": copy.deepcopy(live_state),
            "live": sorted(live_state.keys()),
            "hash": hashlib.sha256(blob.encode()).hexdigest(),
        })
        self.last_snap = seq

    def retained_bytes(self):
        b = self.tail_view.log.bytes() + len(json.dumps(self.snapshots, default=str))
        return b, b

    def touches_for(self, eid, upto_seq):
        return self.tail_view.touches_for(eid, upto_seq)

    def terminal_source(self, eid, key):
        exact = self.tail_view.terminal_source(eid, key)
        if exact[0] == "exact":
            return exact
        for snap in reversed(self.snapshots):
            if eid in snap["state"] and key in snap["state"][eid]:
                return ("snapshot", snap["seq"], f"value={snap['state'][eid][key]}")
        return ("miss", None, None)

    def verify_chain(self):
        ok, n = self.tail_view.verify_chain()
        if not ok:
            return False, n
        for s in self.snapshots:
            blob = json.dumps(
                {k: s["state"][k] for k in s["live"] if k in s["state"]},
                sort_keys=True, default=str)
            # snapshot covers its live set; hash must match record
            full = json.dumps(s["state"], sort_keys=True, default=str)
            if hashlib.sha256(full.encode()).hexdigest() != s["hash"]:
                return False, n
        return True, n + len(self.snapshots)


class ViewH3(ViewH2):
    name = "H3-lineage"

    def __init__(self, tail: int = 2000, snap_every: int = 50000):
        super().__init__(tail, snap_every)
        self.name = f"H3-lineage-snap{snap_every}-tail{tail}"
        self.lineage: Dict[str, Dict[str, Any]] = {}

    def notify(self, entry):
        super().notify(entry)
        seq = int(entry.get("seq", 0))
        eids = _entry_eids(entry)
        snap_root = self.snapshots[-1]["seq"] if self.snapshots else 0
        for eid in eids:
            prev = self.lineage.get(eid, {})
            self.lineage[eid] = {
                "last_seq": seq,
                "last_kind": entry.get("op"),
                "prev_root": prev.get("last_seq", snap_root),
                "sources": sorted(set(prev.get("sources", [])) | set(_entry_sources(entry))),
            }

    def retained_bytes(self):
        a, t = super().retained_bytes()
        b = len(json.dumps(self.lineage, default=str))
        return a + b, t + b

    def terminal_source(self, eid, key):
        exact = self.tail_view.terminal_source(eid, key)
        if exact[0] == "exact":
            return exact
        lin = self.lineage.get(eid)
        if lin is not None:
            return ("lineage", lin["last_seq"],
                    f"{lin['last_kind']} prev_root={lin['prev_root']}")
        return super().terminal_source(eid, key)


class ViewH4(ViewH3):
    name = "H4-summary"

    def __init__(self, tail: int = 2000, snap_every: int = 50000, per_entity_raw: int = 5):
        super().__init__(tail, snap_every)
        self.name = f"H4-summary-snap{snap_every}-tail{tail}"
        self.per_entity_raw = per_entity_raw
        self.summaries: Dict[str, Dict[str, Any]] = {}
        self.skill_agg: Dict[str, Dict[str, int]] = {}

    def notify(self, entry):
        super().notify(entry)
        seq = int(entry.get("seq", 0))
        for eid in _entry_eids(entry):
            s = self.summaries.get(eid, {"sources": [], "retractions": 0,
                                         "contradictions": 0, "checkpoint_link": 0})
            op = entry.get("op")
            if op == "retract":
                s["retractions"] += 1
            if op == "contradict":
                s["contradictions"] += 1
            src = _entry_sources(entry)
            s["sources"] = sorted(set(s["sources"]) | set(src))[-20:]
            if self.snapshots:
                s["checkpoint_link"] = self.snapshots[-1]["seq"]
            s["last_seq"] = seq
            s["last_kind"] = op
            just = _justify(entry)
            if just:
                s["justification"] = just
            self.summaries[eid] = s
        if entry.get("op") == "skill_replace":
            sk = "repair_bridge"
            self.skill_agg.setdefault(sk, {"gens_seen": 0})
            self.skill_agg[sk]["gens_seen"] = max(
                self.skill_agg[sk]["gens_seen"], int(entry.get("gen", 1)))
        if entry.get("op") == "scenario_skill_outcome":
            sk = entry.get("skill", "sk1")
            a = self.skill_agg.setdefault(sk, {"ok": 0, "n": 0, "gens_seen": 1})
            a["ok"] = a.get("ok", 0) + int(bool(entry.get("success")))
            a["n"] = a.get("n", 0) + 1

    def record_skill_outcome(self, skill: str, success: bool):
        a = self.skill_agg.setdefault(skill, {"ok": 0, "n": 0, "gens_seen": 1})
        a["ok"] = a.get("ok", 0) + int(success)
        a["n"] = a.get("n", 0) + 1

    def retained_bytes(self):
        a, t = super().retained_bytes()
        b = len(json.dumps([self.summaries, self.skill_agg], default=str))
        return a + b, t + b

    def terminal_source(self, eid, key):
        exact = self.tail_view.terminal_source(eid, key)
        if exact[0] == "exact":
            return exact
        s = self.summaries.get(eid)
        if s is not None and "justification" in s:
            return ("summary", s.get("last_seq"),
                    f"{s.get('last_kind')}: {s['justification']}")
        return super().terminal_source(eid, key)


class ViewH5(ViewH2):
    name = "H5-tiered"

    def __init__(self, tail: int = 2000, snap_every: int = 50000,
                 seal_every: int = 50000, cold_dir: Optional[str] = None):
        super().__init__(tail, snap_every)
        self.name = f"H5-tiered-snap{snap_every}-tail{tail}"
        self.seal_every = seal_every
        self.cold_dir = cold_dir
        self.cold_files: List[str] = []
        self.cold_bytes = 0
        self.cold_index: Dict[str, List[str]] = {}
        self.sealed_ranges: List[Tuple[int, int]] = []
        self._pending: List[Dict[str, Any]] = []
        self.last_seal = 0

    def notify(self, entry):
        super().notify(entry)
        self._pending.append(entry)
        if int(entry.get("seq", 0)) - self.last_seal >= self.seal_every:
            self._seal(int(entry.get("seq", 0)))

    def _seal(self, seq: int):
        import tempfile
        blob = json.dumps(self._pending, default=str).encode()
        data = gzip.compress(blob, compresslevel=6)
        if self.cold_dir is None:
            self.cold_dir = tempfile.mkdtemp(prefix="cortex_cold_")
        path = os.path.join(self.cold_dir, f"seg_{self.last_seal}_{seq}.jsonl.gz")
        with open(path, "wb") as f:
            f.write(data)
        self.cold_files.append(path)
        self.cold_bytes += len(data)
        lo = int(self._pending[0].get("seq", seq)) if self._pending else seq
        self.sealed_ranges.append((lo, seq))
        for e in self._pending:
            for eid in _entry_eids(e):
                self.cold_index.setdefault(eid, []).append(path)
        self._pending = []
        self.last_seal = seq

    def cold_touches_for(self, eid: str, upto_seq: int) -> List[Dict[str, Any]]:
        out = []
        seen_files = set()
        for path in self.cold_index.get(eid, []):
            if path in seen_files:
                continue
            seen_files.add(path)
            with gzip.open(path, "rt", encoding="utf-8") as f:
                seg = json.loads(f.read())
            for e in seg:
                if int(e.get("seq", 0)) > upto_seq:
                    break
                if _touches(e, eid):
                    out.append(e)
        return out

    def retained_bytes(self):
        a, _ = super().retained_bytes()
        return a, a + self.cold_bytes + len(json.dumps(self._pending, default=str))


# ---------------------------------------------------------------------------
# Journal entry helpers (mirror DirtyWorld._log fields)
# ---------------------------------------------------------------------------

def _entry_eids(entry: Dict[str, Any]) -> List[str]:
    out = []
    for k in ("eid", "a", "b"):
        v = entry.get(k)
        if isinstance(v, str) and " " not in v:
            out.append(v)
    # old/new are entity ids ONLY for renames; for contradict they are
    # status strings and must not pollute lineage (caught in review).
    if entry.get("op") == "rename":
        for k in ("old", "new"):
            v = entry.get(k)
            if isinstance(v, str):
                out.append(v)
    for v in entry.get("eids", []) or []:
        if isinstance(v, str):
            out.append(v)
    # skill entities live in substrate too
    if entry.get("op") == "skill_replace":
        out.append("skill_repair_bridge")
    return sorted(set(out))


def _touches(entry: Dict[str, Any], eid: str) -> bool:
    return eid in _entry_eids(entry)


def _changes_key(entry: Dict[str, Any], key: str) -> bool:
    op = entry.get("op")
    if op in ("update_state", "contradict") and key == "status":
        return True
    if op == "replay" and key == "status" and (entry.get("payload") or {}).get("status") is not None:
        return True
    if op == "merge_split" and key == "project":
        return True
    if op == "schema_migrate" and key in ("status", "state_v2"):
        return True
    if op == "retract" and key == "evidence_status":
        return True
    if op in ("delete", "recreate", "rename"):
        return True  # lifecycle touches all keys
    return False


def _entry_sources(entry: Dict[str, Any]) -> List[str]:
    src = []
    for k in ("a", "b", "old", "eids"):
        v = entry.get(k)
        if isinstance(v, str):
            src.append(v)
        elif isinstance(v, list):
            src.extend([x for x in v if isinstance(x, str)])
    if entry.get("replay_of") is not None:
        src.append(f"seq:{entry['replay_of']}")
    return sorted(set(src))


def _justify(entry: Dict[str, Any]) -> Optional[str]:
    op = entry.get("op")
    if op in ("update_state", "contradict"):
        return f"{op} status={entry.get('status', entry.get('new'))}"
    if op == "replay":
        return f"replay of seq {entry.get('replay_of')}"
    if op == "retract":
        return "evidence RETRACTED"
    if op in ("delete", "recreate", "rename", "merge_split", "schema_migrate"):
        return op
    return None


# ---------------------------------------------------------------------------
# Contract evaluators (truth journal kept harness-side, never shown to views)
# ---------------------------------------------------------------------------

def _justify(entry: Dict[str, Any]) -> Optional[str]:
    op = entry.get("op")
    if op in ("update_state", "contradict"):
        return f"{op} status={entry.get('status', entry.get('new'))}"
    if op == "replay":
        return f"replay of seq {entry.get('replay_of')}"
    if op == "retract":
        return "evidence RETRACTED"
    if op in ("delete", "recreate", "rename", "merge_split", "schema_migrate"):
        return op
    return None


def eval_p_state(view: RetentionView, live_substrate: Dict[str, Dict[str, Any]],
                 birth_substrate: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A: rebuild current S strictly from retained data, compare exactly.

    Target is the SUBSTRATE's live state (S_v as stored, lingering keys
    included); the oracle's canonical view is P2's domain. Shadow mirrors
    production merge semantics and is always seeded from the birth snapshot
    (genesis = legitimate checkpoint-at-0, available to every policy).
    """
    _KEEP = ("alive", "tombstone_seq", "recreated_seq", "origin")
    shadow: Dict[str, Dict[str, Any]] = {}
    alive: Set[str] = set()
    if isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        base = view.snapshots[-1] if view.snapshots else None
        if base is not None:
            for eid in base["live"]:
                if eid in base["state"]:
                    shadow[eid] = dict(base["state"][eid])
                    alive.add(eid)
            # only post-snapshot tail: re-applying older events on top of a
            # snapshot is NOT idempotent across lifecycle ops (delete then
            # recreate would re-kill). Snapshot covers seq <= base["seq"].
            for rec in view.tail_view.log.entries:
                if int(rec["entry"].get("seq", 0)) > base["seq"]:
                    apply_entry(shadow, alive, rec["entry"])
        else:
            for eid, st in birth_substrate.items():
                shadow[eid] = dict(st)
                alive.add(eid)
            for rec in view.tail_view.log.entries:
                apply_entry(shadow, alive, rec["entry"])
    else:
        for eid, st in birth_substrate.items():
            shadow[eid] = dict(st)
            alive.add(eid)
        log = view.log.entries if isinstance(view, ViewH0) else view.log.entries
        for rec in log:
            apply_entry(shadow, alive, rec["entry"])
    match = mismatch = 0
    for eid, canon in live_substrate.items():
        got = shadow.get(eid)
        if got is not None and all(got.get(k) == v for k, v in canon.items()) \
                and all(k in canon or k in _KEEP for k in got.keys()):
            match += 1
        else:
            mismatch += 1
    v, _, _ = _rate(match, match + mismatch)
    return {"p_state": v, "num": match, "den": match + mismatch}


def eval_p_explain(view: RetentionView, truth_touches: Dict[str, List[Dict[str, Any]]],
                   sample: List[Tuple[str, str]]) -> Dict[str, Any]:
    """B: terminal source event per sampled (eid,key); chain coverage."""
    full = part = miss = 0
    chain_found = chain_true = 0
    kinds: Dict[str, int] = {}
    for eid, key in sample:
        truth = [e for e in truth_touches.get(eid, []) if _changes_key(e, key)]
        true_seq = truth[-1].get("seq") if truth else None
        kind, seq, _ = view.terminal_source(eid, key)
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "exact" and seq == true_seq:
            full += 1
        elif kind in ("snapshot", "lineage", "summary"):
            part += 1
        else:
            miss += 1
        # predecessor chain while retained (cap walk at 8)
        if truth:
            t = 0
            for e in reversed(truth[:-1][-8:]):
                chain_true += 1
                if _retained_has(view, e):
                    chain_found += 1
                t += 1
                if t >= 8:
                    break
    n = len(sample)
    fv, _, _ = _rate(full, n)
    pv, _, _ = _rate(part, n)
    cv, cn, cd = _rate(chain_found, chain_true)
    compr = (1.0 - (cn / cd)) if cd else None
    return {"p_explain_full": fv, "full_num": full, "p_explain_partial": pv,
            "part_num": part, "miss_num": miss, "den": n,
            "chain_coverage": cv, "chain_num": cn, "chain_den": cd,
            "chain_compression": compr, "source_kinds": kinds}


def _retained_has(view: RetentionView, entry: Dict[str, Any]) -> bool:
    seq = entry.get("seq")
    log = None
    if isinstance(view, ViewH0):
        log = view.log
    elif isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        if any(s["seq"] >= seq for s in view.snapshots):
            return True
        if isinstance(view, ViewH5) and any(lo <= seq <= hi for lo, hi in view.sealed_ranges):
            return True
        log = view.tail_view.log
    else:
        log = view.log
    if log.first_seq is None:
        return False
    return log.first_seq <= seq <= log.last_seq


def eval_p_replay(view: RetentionView, truth_log: List[Dict[str, Any]],
                  birth_state: Dict[str, Dict[str, Any]],
                  entities: List[str], seqs: List[int]) -> Dict[str, Any]:
    """C: exact entity state at sampled historical seqs, from retained data."""
    exact = soft_num = 0
    soft_den = 0
    n = 0
    for eid in entities:
        for seq in seqs:
            n += 1
            # truth at seq: birth + full-journal touches <= seq, substrate-
            # merge semantics via apply_entry (the state that truly existed,
            # lingering keys included -- replay must reproduce storage, not
            # the oracle's canonical view).
            truth_shadow: Dict[str, Dict[str, Any]] = {}
            talive: Set[str] = set()
            if eid in birth_state:
                truth_shadow[eid] = dict(birth_state[eid])
                talive.add(eid)
            for e in truth_log:
                if int(e.get("seq", 0)) > seq:
                    break
                if _touches(e, eid):
                    apply_entry(truth_shadow, talive, e)
            truth_st = truth_shadow.get(eid, {})
            # retained reconstruction
            shadow: Dict[str, Dict[str, Any]] = {}
            alive: Set[str] = set()
            if isinstance(view, ViewH0):
                src = [r["entry"] for r in view.log.entries if int(r["entry"].get("seq", 0)) <= seq]
                base = dict(birth_state.get(eid, {}))
                if eid in birth_state:
                    shadow[eid] = dict(base)
                    alive.add(eid)
            elif isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
                snaps = [s for s in view.snapshots if s["seq"] <= seq]
                base = snaps[-1] if snaps else None
                base_seq = 0
                if base is not None and eid in base["state"]:
                    shadow[eid] = dict(base["state"][eid])
                    alive.add(eid)
                    base_seq = base["seq"]
                elif eid in birth_state and not snaps:
                    shadow[eid] = dict(birth_state[eid])
                    alive.add(eid)
                # post-base tail only (same non-idempotency rule as P_state)
                src = [r["entry"] for r in view.tail_view.log.entries
                       if base_seq < int(r["entry"].get("seq", 0)) <= seq]
                if isinstance(view, ViewH5):
                    src = sorted(view.cold_touches_for(eid, seq) + src,
                                 key=lambda e: int(e.get("seq", 0)))
            else:
                src = [r["entry"] for r in view.log.entries if int(r["entry"].get("seq", 0)) <= seq]
                if eid in birth_state:
                    shadow[eid] = dict(birth_state[eid])
                    alive.add(eid)
            for e in src:
                if _touches(e, eid):
                    if eid not in shadow:
                        shadow[eid] = {}
                    apply_entry(shadow, alive, e)
            got = shadow.get(eid, {})
            keys = (set(truth_st) | set(got)) - {"alive"}
            if not keys:
                exact += 1
                continue
            hit = sum(1 for k in keys if got.get(k) == truth_st.get(k))
            soft_num += hit
            soft_den += len(keys)
            if hit == len(keys):
                exact += 1
    ev, _, _ = _rate(exact, n)
    sv, _, _ = _rate(soft_num, soft_den)
    return {"p_replay_exact": ev, "exact_num": exact, "den": n,
            "p_replay_soft": sv, "soft_num": soft_num, "soft_den": soft_den}


def _apply_truth_touch(state: Dict[str, Any], entry: Dict[str, Any]):
    """Ground-truth application (full information) for replay targets."""
    op = entry.get("op")
    if op in ("update_state", "contradict") and entry.get("status") is not None:
        state["status"] = entry["status"]
    elif op == "replay" and (entry.get("payload") or {}).get("status") is not None:
        state["status"] = entry["payload"]["status"]
    elif op == "merge_split":
        state["project"] = entry.get("project")
    elif op == "schema_migrate":
        if "status" in state:
            state["state_v2"] = state.pop("status")
    elif op == "retract":
        state["evidence_status"] = "RETRACTED"
    elif op == "delete":
        state["alive"] = False
    elif op == "recreate":
        state["alive"] = True
        if entry.get("status") is not None:
            state["status"] = entry["status"]


def eval_p_audit(view: RetentionView) -> Dict[str, Any]:
    """D: chain verifies; tamper/remove/reorder on copies detected.

    Tamper battery runs on a middle 1000-entry window (same primitive as the
    full chain; full-log deepcopies at 1M would be pure overhead, stated)."""
    ok, n = view.verify_chain()
    import copy as _copy
    if isinstance(view, ViewH0):
        recs = list(view.log.entries)
        anchor = "GENESIS"
    elif isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        recs = list(view.tail_view.log.entries)
        anchor = view.tail_view.anchor
    else:
        recs = list(view.log.entries)
        anchor = view.anchor
    det = {"modify": None, "remove": None, "reorder": None}
    if len(recs) >= 3:
        if len(recs) > 1200:
            mid0 = len(recs) // 2 - 500
            window = recs[mid0:mid0 + 1000]
            anchor = recs[mid0 - 1]["link"] if mid0 > 0 else anchor
        else:
            window = recs
        mid = len(window) // 2
        mod = _copy.deepcopy([{"entry": dict(r["entry"]), "link": r["link"]} for r in window])
        mod[mid]["entry"]["_tamper"] = "x"
        det["modify"] = not _verify_recs(mod, anchor)
        rem = [r for i, r in enumerate(window) if i != mid]
        rem = [{"entry": dict(r["entry"]), "link": r["link"]} for r in rem]
        det["remove"] = not _verify_recs(rem, anchor)
        reo = [{"entry": dict(r["entry"]), "link": r["link"]} for r in window]
        reo[mid], reo[mid + 1] = reo[mid + 1], reo[mid]
        det["reorder"] = not _verify_recs(reo, anchor)
    return {"p_audit_chain_ok": ok, "chain_n": n,
            "tamper_modify_detected": det["modify"],
            "tamper_remove_detected": det["remove"],
            "tamper_reorder_detected": det["reorder"]}


def _anchor(view: RetentionView) -> str:
    if isinstance(view, ViewH1):
        return view.anchor
    if isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        return view.tail_view.anchor
    return "GENESIS"


def _verify_recs(recs, anchor: str) -> bool:
    h = anchor
    for rec in recs:
        e = rec["entry"]
        h = _chain(h, int(e.get("seq", 0)), str(e.get("op")),
                   json.dumps(e, sort_keys=True, default=str))
        if h != rec["link"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Temporal scenario battery (scripted, identical ops for every policy)
# ---------------------------------------------------------------------------

def run_scenarios(world, views: Dict[str, RetentionView], at_seq: int) -> Dict[str, Any]:
    """T1..T5 injected through the standard tick+log funnel. Returns per-policy
    pass maps with denominators."""
    results: Dict[str, Any] = {}
    live = [e for e in world.bg_ids if world.o_alive.get(e, False)]
    if len(live) < 6:
        return {name: {"skipped": True} for name in views}
    import torch as _t
    # world.rng: same picks for every policy sharing this world (paired).
    pick = lambda xs: xs[int(_t.randint(0, len(xs), (1,), generator=world.rng).item())]
    x, y, z, c1 = pick(live), pick(live), pick(live), pick(live)

    # T1 retraction cascade: X supported_by=E, conclusion C cites X; retract E
    world._tick([(x, {"supported_by": "ev1", "claim": "B"})])
    s1 = world._log("scenario_support", eid=x, claim="B", evidence="ev1")
    world.o_state[x].update({"supported_by": "ev1", "claim": "B"})
    world._tick([(x, {"evidence_status": "RETRACTED", "claim": "B revised"})])
    s2 = world._log("retract", eid=x)
    world._log("scenario_revise", eid=x, claim="B revised")
    world.o_state[x].update({"evidence_status": "RETRACTED", "claim": "B revised"})

    # T2 contradiction: untrusted 1 then trusted 2 (trust journaled: completeness)
    world._tick([(y, {"val": 1, "trust": "low"})])
    world._log("scenario_untrusted", eid=y, val=1, trust="low")
    world.o_state[y].update({"val": 1, "trust": "low"})
    world._tick([(y, {"val": 2, "trust": "high"})])
    s3 = world._log("scenario_trusted", eid=y, val=2, trust="high")
    world.o_state[y].update({"val": 2, "trust": "high"})

    # T3 delete/recreate: Z deleted then recreated with NEW content only
    world._tick([(z, {"alive": False, "tombstone_seq": world.seq + 1})])
    world._log("delete", eid=z)
    world.o_alive[z] = False
    world.tombstoned.add(z)
    cid = world.eid_cid[z]
    if z in world.sub.clusters.get(cid, []):
        world.sub.clusters[cid].remove(z)
    world._tick([(z, {"alive": True, "status": "NOMINAL", "tombstone_seq": -1})])
    world._log("recreate", eid=z, status="NOMINAL")
    world.o_alive[z] = True
    world.o_state[z] = {"project": world.o_state[z].get("project", PROJECTS0),
                        "kind": "BACKGROUND", "status": "NOMINAL",
                        "idx": world.o_state[z].get("idx", 0)}
    world.tombstoned.discard(z)
    if z not in world.sub.clusters.get(cid, []):
        world.sub.clusters[cid].append(z)

    # T4 cross-checkpoint causality: cause on c1 now, consequence later is
    # evaluated by lineage (cause seq recorded here)
    world._tick([(c1, {"cause_marker": at_seq})])
    s4 = world._log("scenario_cause", eid=c1, marker=at_seq)
    world.o_state[c1]["cause_marker"] = at_seq

    # T5 skill aggregate: synthetic outcomes for sk1 v1/v2 (same for all)
    scen_sk = getattr(world, "_scen_skill", {"ok": 0, "n": 0})
    for i, ok in enumerate([True, True, False, True]):
        world._log("scenario_skill_outcome", skill="sk1",
                   version="v2" if i > 1 else "v1", success=ok)
        scen_sk["ok"] += int(ok)
        scen_sk["n"] += 1
    world._scen_skill = scen_sk

    for name, view in views.items():
        t1 = _t1_score(view, x, s2)
        t2 = _t2_score(view, y, s3)
        t3 = _t3_score(world, z)
        t4 = _t4_score(view, c1, s4)
        t5 = _t5_score(view, scen_sk)
        results[name] = {"T1_retraction": t1, "T2_contradiction": t2,
                         "T3_recreate": t3, "T4_cross_checkpoint": t4,
                         "T5_skill_lineage": t5}
    return results


PROJECTS0 = "warp_cortex"


def _t1_score(view, x, retract_seq):
    kind, seq, _ = view.terminal_source(x, "evidence_status")
    if kind == "exact" and seq == retract_seq:
        return {"pass": True, "num": 1, "den": 1}
    if kind in ("snapshot", "lineage", "summary"):
        return {"pass": None, "num": 0, "den": 1, "note": f"partial:{kind}"}
    return {"pass": False, "num": 0, "den": 1}


def _t2_score(view, y, trusted_seq):
    # trusted update is an exact event iff its seq is retained
    if _view_has_seq(view, trusted_seq):
        return {"pass": True, "num": 1, "den": 1}
    kind, _, _ = view.terminal_source(y, "status")
    if kind in ("snapshot", "lineage", "summary"):
        return {"pass": None, "num": 0, "den": 1, "note": f"partial:{kind}"}
    return {"pass": False, "num": 0, "den": 1}


def _view_knows_key(view, eid, key):
    kind, _, _ = view.terminal_source(eid, key)
    return kind != "miss"


def _view_has_seq(view, seq):
    if isinstance(view, ViewH0):
        return 0 <= seq <= view.log.last_seq
    if isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        log = view.tail_view.log
        if log.first_seq is not None and log.first_seq <= seq <= log.last_seq:
            return True
        return False
    log = view.log
    if log.first_seq is None:
        return False
    return log.first_seq <= seq <= log.last_seq


def _t3_score(world, z):
    node = world.sub.entities.get(z)
    canon = world.o_state.get(z, {})
    if node is None:
        return {"pass": False, "num": 0, "den": 1, "note": "missing node"}
    ok = all(node.state.get(k) == v for k, v in canon.items())
    extra = [k for k in node.state.keys()
             if k not in canon and k not in ("alive", "tombstone_seq", "recreated_seq", "origin")]
    return {"pass": bool(ok and not extra), "num": int(ok and not extra), "den": 1,
            "extra_keys": extra[:5]}


def _t4_score(view, c1, cause_seq):
    if _view_has_seq(view, cause_seq):
        return {"pass": True, "num": 1, "den": 1}
    kind, seq, _ = view.terminal_source(c1, "cause_marker") \
        if _view_knows_key(view, c1, "cause_marker") else ("miss", None, None)
    if kind in ("snapshot", "lineage", "summary"):
        return {"pass": None, "num": 0, "den": 1, "note": f"partial:{kind}"}
    return {"pass": False, "num": 0, "den": 1}


def _t5_score(view, scen_sk):
    if isinstance(view, ViewH4):
        agg = view.skill_agg.get("sk1", {})
        # H4 summaries track gens, not these synthetic outcomes (recorded only
        # in the log funnel views share); score aggregate availability:
        if agg.get("n", 0) == scen_sk["n"] and agg.get("ok", -1) == scen_sk["ok"]:
            return {"pass": True, "num": 1, "den": 1}
        return {"pass": False, "num": 0, "den": 1, "note": "aggregate mismatch"}
    # other views: outcomes live in retained log iff window covers them
    found = 0
    if isinstance(view, ViewH0):
        recs = list(view.log.entries)
    elif isinstance(view, (ViewH2, ViewH3, ViewH4, ViewH5)):
        recs = list(view.tail_view.log.entries)
    else:
        recs = list(view.log.entries)
    for r in recs:
        e = r["entry"]
        if e.get("op") == "scenario_skill_outcome" and e.get("skill") == "sk1":
            found += 1
    if found == scen_sk["n"]:
        return {"pass": True, "num": 1, "den": 1}
    if found > 0:
        return {"pass": None, "num": 0, "den": 1, "note": f"partial:{found}/{scen_sk['n']}"}
    return {"pass": False, "num": 0, "den": 1}


# ---------------------------------------------------------------------------
# Runner: shared seeded streams, all views fed identically
# ---------------------------------------------------------------------------

_KEY_FOR_OP = {
    "update_state": "status", "contradict": "status", "replay": "status",
    "merge_split": "project", "schema_migrate": "state_v2",
    "retract": "evidence_status",
}


def _build_views(cold_parent: Optional[str] = None):
    import tempfile
    views: Dict[str, RetentionView] = {
        "H0": ViewH0(),
        "H1-500": ViewH1(500),
        "H1-2000": ViewH1(2000),
        "H1-10000": ViewH1(10000),
        "H2": ViewH2(tail=2000, snap_every=50000),
        "H3": ViewH3(tail=2000, snap_every=50000),
        "H4": ViewH4(tail=2000, snap_every=50000),
    }
    cold_dir = tempfile.mkdtemp(prefix="cortex_cold_") if cold_parent is None else cold_parent
    views["H5"] = ViewH5(tail=2000, snap_every=50000, seal_every=50000, cold_dir=cold_dir)
    return views


def _truth_index(journal: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for e in journal:
        for eid in _entry_eids(e):
            idx.setdefault(eid, []).append(e)
    return idx


def _explain_sample(journal: List[Dict[str, Any]], live: Set[str], n: int = 50):
    """(eid,key) pairs from recent truth touches, spread across entities."""
    out, seen = [], set()
    for e in reversed(journal[-20000:]):
        op = e.get("op")
        if op not in _KEY_FOR_OP:
            continue
        for eid in _entry_eids(e):
            if eid not in live or (eid, _KEY_FOR_OP[op]) in seen:
                continue
            seen.add((eid, _KEY_FOR_OP[op]))
            out.append((eid, _KEY_FOR_OP[op]))
            if len(out) >= n:
                return out
    return out


def run_history_bounds(ages=(10000, 100000, 250000, 500000, 1000000),
                       seeds=(7, 11, 23), n_bg: int = 3000,
                       out_name: str = "history_bounds_results.json"):
    """P4 main: one shared stream per seed; every view sees identical events."""
    from dirty_world import DirtyWorld
    from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
        GenericFrozenAspectEncoder,
    )
    from cortex_apps.cortex_world_runtime.boring_unified_store import (
        measure_container_bytes,
    )
    import json

    path = os.path.join(os.path.dirname(__file__), out_name)
    rows = []

    def _checkpoint():
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "partial": True}, f, indent=1)

    encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)
    for seed in seeds:
        world = DirtyWorld(n_bg=n_bg, seed=seed, encoder=encoder)
        views = _build_views()
        birth_substrate = {e: dict(n.state) for e, n in world.sub.entities.items()}
        orig_log = world._log

        def _feed(op, _o=orig_log, _vs=views, **kw):
            s = _o(op, **kw)
            entry = {"seq": s, "op": op, **kw}
            for v in _vs.values():
                v.notify(entry)
            return s
        world._log = _feed
        # snapshot cadence hook: H2+ snapshot every 50k during advance
        sub_mem = 0
        for age in ages:
            while world.seq < age:
                world.mutate()
                if world.seq % 50000 == 0:
                    # snapshots capture SUBSTRATE state (S_v as stored,
                    # lingering keys included) -- that is what a real snapshot
                    # does. Oracle states here would contaminate every
                    # snapshot-derived metric (caught in review).
                    live_s = {e: dict(world.sub.entities[e].state)
                              for e in world.o_state
                              if world.o_alive.get(e, False) and e in world.sub.entities}
                    for v in views.values():
                        v.snapshot_tick(world.seq, live_s)
            # scenarios inject identical ops for every policy (shared world)
            scen = run_scenarios(world, views, world.seq)
            live_oracle = {e: dict(world.o_state[e]) for e in world.o_state
                           if world.o_alive.get(e, False)}
            live_substrate = {e: dict(world.sub.entities[e].state)
                              for e in live_oracle if e in world.sub.entities}
            for v in views.values():
                v.snapshot_tick(world.seq, live_substrate)
            t0 = time.perf_counter()
            truth_idx = _truth_index(world.journal)
            live_set = set(live_oracle)
            sample = _explain_sample(world.journal, live_set)
            rep_entities = sorted(live_set)[:20]
            rep_seqs = [max(1, age // 10), max(1, age // 2), max(1, age * 9 // 10)]
            sub_mem = measure_container_bytes(
                world.sub.entities, world.sub.clusters, world.sub.history_events,
                world.sub.global_state, getattr(world.sub, "centroids", {}))
            for name, view in views.items():
                t1 = time.perf_counter()
                ps = eval_p_state(view, live_substrate, birth_substrate)
                t_state = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                pe = eval_p_explain(view, truth_idx, sample)
                t_expl = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                pr = eval_p_replay(view, world.journal, birth_substrate, rep_entities, rep_seqs)
                t_repl = (time.perf_counter() - t1) * 1000.0
                t1 = time.perf_counter()
                pa = eval_p_audit(view)
                t_audit = (time.perf_counter() - t1) * 1000.0
                ab, tb = view.retained_bytes()
                row = {"seed": seed, "age": world.seq, "policy": name,
                       "p_state": ps["p_state"], "p_state_den": ps["den"],
                       "p_explain_full": pe["p_explain_full"],
                       "p_explain_partial": pe["p_explain_partial"],
                       "explain_den": pe["den"],
                       "chain_coverage": pe["chain_coverage"],
                       "chain_compression": pe["chain_compression"],
                       "source_kinds": pe["source_kinds"],
                       "p_replay_exact": pr["p_replay_exact"],
                       "p_replay_soft": pr["p_replay_soft"],
                       "replay_den": pr["den"],
                       "p_audit_ok": pa["p_audit_chain_ok"],
                       "audit_n": pa["chain_n"],
                       "tamper_modify": pa["tamper_modify_detected"],
                       "tamper_remove": pa["tamper_remove_detected"],
                       "tamper_reorder": pa["tamper_reorder_detected"],
                       "active_bytes": ab, "total_bytes": tb,
                       "substrate_bytes": sub_mem,
                       "scenarios": scen.get(name, {}),
                       "ms_state": round(t_state, 1), "ms_explain": round(t_expl, 1),
                       "ms_replay": round(t_repl, 1), "ms_audit": round(t_audit, 1)}
                rows.append(row)
                _checkpoint()
            r0 = [r for r in rows if r["seed"] == seed and r["age"] == world.seq and r["policy"] == "H0"][0]
            rh = [r for r in rows if r["seed"] == seed and r["age"] == world.seq and r["policy"] == "H4"][0]
            print(f"p4 seed={seed} age={world.seq:>8}: H0 state={r0['p_state']} expl={r0['p_explain_full']} "
                  f"replay={r0['p_replay_exact']} audit={r0['p_audit_ok']} active={r0['active_bytes']/1e6:.1f}MB | "
                  f"H4 state={rh['p_state']} expl={rh['p_explain_full']}+{rh['p_explain_partial']} "
                  f"replay={rh['p_replay_exact']} active={rh['active_bytes']/1e6:.2f}MB",
                  flush=True)
    out = {"rows": rows, "partial": False,
           "note": ("Shared seeded streams per seed; retention passive (never touches "
                    "S/G/Z). Truth journal harness-side. Ratios null on empty dens.")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path} (partial=False)")
    return out


def run_sqlite_battery(n_events: int = 100000, seed: int = 7,
                       out_name: str = "history_bounds_sqlite.json"):
    """Section 6: portable store under real SQLite/WAL bytes + crash/corrupt tests."""
    import tempfile
    from cortex_apps.cortex_world_runtime.cortex_world.store import open_world
    import json

    tmp = tempfile.mkdtemp(prefix="cortex_p4_")
    proj = os.path.join(tmp, "proj")
    t0 = time.perf_counter()
    w = open_world(proj)
    t_open = (time.perf_counter() - t0) * 1000.0
    truth: Dict[str, Dict[str, Any]] = {}
    commit_ms = []
    for i in range(n_events):
        eid = f"ent_{i % 500:04d}"
        st = truth.get(eid, {})
        st = dict(st)
        st["v"] = i
        st["project"] = "p4"
        truth[eid] = st
        t1 = time.perf_counter()
        seq = w.commit("tick", {"entity": eid, "v": i})
        w.upsert_node(eid, st, mirror_md=False)
        commit_ms.append((time.perf_counter() - t1) * 1000.0)
        if i and i % 20000 == 0:
            print(f"  sqlite {i}/{n_events}", flush=True)
    import statistics
    db_b = os.path.getsize(w.db_path)
    wal = w.db_path + "-wal"
    wal_b = os.path.getsize(wal) if os.path.exists(wal) else 0
    snap_b = sum(len(r[0]) for r in w.db.execute("SELECT state_json FROM snapshots").fetchall())
    n_ev = w.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_sn = w.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    # state verify sample
    vyr, vym = 0, 0
    for eid in sorted(truth)[:100]:
        node = w.get_node(eid)
        vym += 1
        if node is not None and node["state"].get("v") == truth[eid]["v"]:
            vyr += 1
    # explain sample via SQL scan (retained provenance contracts)
    t1 = time.perf_counter()
    ex_ok = ex_n = 0
    for eid in sorted(truth)[:50]:
        row = w.db.execute(
            "SELECT seq FROM events WHERE payload_json LIKE ? ORDER BY seq DESC LIMIT 1",
            (f"%{eid}%",)).fetchone()
        ex_n += 1
        if row is not None:
            ex_ok += 1
    explain_ms = (time.perf_counter() - t1) * 1000.0 / max(1, ex_n)
    # chain audit timing
    t1 = time.perf_counter()
    audit_ok, audit_n = w.verify_chain()
    audit_ms = (time.perf_counter() - t1) * 1000.0
    w.close()
    # reopen = recovery timing + startup verify
    t1 = time.perf_counter()
    w2 = open_world(proj)
    recovery_ms = (time.perf_counter() - t1) * 1000.0
    node = w2.get_node("ent_0000")
    reopen_ok = node is not None
    w2.close()
    # corrupt one retained event payload -> reopen must raise
    import sqlite3 as _sq
    con = _sq.connect(os.path.join(proj, ".cortex", "cortex.sqlite"))
    target = con.execute("SELECT seq FROM events ORDER BY seq DESC LIMIT 5 OFFSET 2").fetchone()[0]
    con.execute("UPDATE events SET payload_json=? WHERE seq=?", ('{"tampered": true}', target))
    con.commit()
    con.close()
    corrupted_detected = False
    try:
        open_world(proj)
    except RuntimeError:
        corrupted_detected = True
    res = {"n_events": n_events, "open_ms": round(t_open, 1),
           "commit_ms_p50": round(statistics.median(commit_ms), 3),
           "commit_ms_p99": round(sorted(commit_ms)[int(len(commit_ms) * 0.99)], 3),
           "db_bytes": db_b, "wal_bytes": wal_b, "snapshot_bytes": snap_b,
           "events_retained": n_ev, "snapshots": n_sn,
           "state_verify": [vyr, vym], "explain_found": [ex_ok, ex_n],
           "explain_ms_each": round(explain_ms, 2),
           "audit_ok": audit_ok, "audit_n": audit_n, "audit_ms": round(audit_ms, 1),
           "recovery_ms": round(recovery_ms, 1), "reopen_ok": reopen_ok,
           "corrupt_event_seq": target, "corrupt_detected": corrupted_detected,
           "bytes_per_event": round((db_b + wal_b) / max(1, n_events), 1)}
    with open(os.path.join(os.path.dirname(__file__), out_name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("sqlite battery:", json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys as _sys
    if "--sqlite" in _sys.argv:
        run_sqlite_battery()
    else:
        run_history_bounds()

