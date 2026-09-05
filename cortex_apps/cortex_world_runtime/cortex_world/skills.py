"""Project-scoped skills: registry in sqlite + SKILL.md mirrors (agentskills-v1).

Scope rule: SHARED selection is scoped to ONE project world (this sqlite).
Cross-project learning is never default; another layer must explicitly allow
and move it. Events record project + world_version at invocation time.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def register_skill(store, skill_id: str, version: str, skill_md: str) -> str:
    h = hashlib.sha256(f"{skill_id}@{version}|{skill_md}".encode()).hexdigest()
    store.db.execute(
        "INSERT OR REPLACE INTO skills(id, version, skill_md, hash) VALUES (?,?,?,?)",
        (skill_id, version, skill_md, h))
    store.db.commit()
    try:
        os.fsync(store.db.execute("PRAGMA wal_checkpoint(TRUNCATE);").connection.fileno())
    except Exception:
        pass
    d = os.path.join(store.root, "skills", skill_id)
    os.makedirs(d, exist_ok=True)
    tmp, dst = os.path.join(d, "SKILL.md.tmp"), os.path.join(d, "SKILL.md")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(skill_md)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    return h


def record_invocation(store, skill_id: str, version: str, success: bool,
                      latency_ms: float = 0.0, error: str = "",
                      constraints: Optional[Dict[str, Any]] = None,
                      project: str = "default") -> int:
    """record_invocation(...): appends to invocations AND a chained H event."""
    seq = store.commit("skill_invocation", {
        "skill_id": skill_id, "version": version, "success": success,
        "project": project})
    store.db.execute(
        "INSERT INTO invocations(skill_id, version, project, success, latency_ms,"
        " error, constraints_json, world_version, seq)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (skill_id, version, project, int(success), latency_ms, error,
         json.dumps(constraints or {}), store.version, seq))
    store.db.commit()
    return seq


def _history(store, skill_id: str, project: str) -> List[Dict[str, Any]]:
    rows = store.db.execute(
        "SELECT version, success, latency_ms, error, constraints_json, world_version"
        " FROM invocations WHERE skill_id=? AND project=? ORDER BY id",
        (skill_id, project)).fetchall()
    return [{"version": r[0], "success": bool(r[1]), "latency_ms": r[2],
             "error": r[3], "constraints": json.loads(r[4]), "world_version": r[5]}
            for r in rows]


def select_skill(store, query: str, project: str = "default", top_k: int = 3,
                 query_vec=None) -> List[Tuple[str, str, float, str]]:
    """select_skill(query): rank skill versions by (Z similarity + H win rate).
    Returns [(skill_id, version, score, explanation)]. Scoped to project."""
    import numpy as np
    regs = store.db.execute("SELECT id, version, skill_md FROM skills ORDER BY id, version"
                            ).fetchall()
    qterms = set(query.lower().split())
    ranked = []
    for sid, ver, md in regs:
        terms = set(md.lower().split())
        overlap = len(qterms & terms)
        if overlap == 0 and query_vec is None:
            continue
        z = 0.0
        if query_vec is not None:
            z = float(overlap)  # lexical proxy unless caller supplies Z match
        hist = _history(store, sid, project)
        ver_hist = [h for h in hist if h["version"] == ver]
        h_score = 0.0
        note = "no history"
        if ver_hist:
            wr = sum(1 for h in ver_hist if h["success"]) / len(ver_hist)
            h_score = (wr - 0.5) * 4.0
            note = f"{sum(1 for h in ver_hist if h['success'])}W/{len(ver_hist)}@{ver}"
        score = overlap * 1.0 + z + h_score
        ranked.append((sid, ver, score, f"overlap={overlap} H={h_score:.2f} [{note}]"))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked[:top_k]
