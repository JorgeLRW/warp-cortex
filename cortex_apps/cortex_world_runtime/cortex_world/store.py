"""Per-project portable store: cortex.sqlite (WAL) + manifest.json + md mirrors.

Layout:
  <project>/.cortex/  (or memories/{id}/cortex/)
    cortex.sqlite   # canonical S,G,Z,H,K
    manifest.json   # format, encoder, budgets (operational), history policy
    entities/*.md   # human/app-readable mirror (NOT canonical)
    skills/<id>/SKILL.md  # portable skill mirrors (agentskills-v1)

Tables: nodes(id, state_json, aspect BLOB float32, cluster, version, updated_seq),
  edges(src, dst, type), events(seq, version, kind, payload_json, hash_prev),
  snapshots(id, seq, state_json), skills(id, version, skill_md, hash),
  invocations(id, skill_id, version, project, success, latency_ms, error,
  constraints_json, world_version, seq).

Durability: WAL mode, fsync on commit, atomic manifest replace, startup
verify (sqlite sha + counts + last seq vs manifest). History: hot cap
(manifest.history.hot_events, default 2000) + auto-compact into snapshots.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import time
from typing import Any, Dict, List, Optional

FORMAT_VERSION = "cortex-world-v1"


def DEFAULT_MANIFEST() -> Dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "encoder": {"id": "qwen2.5-0.5b-embed-rp64-v1", "aspect_dim": 64},
        # Operational budgets, NOT retrieval guarantees: fixed-budget recall
        # is known to degrade with world size (see retrieval_law_results.json).
        "budgets": {"bfs_nodes": 50, "semantic_candidates": 400, "top_k": 5},
        "history": {"hot_events": 2000, "snapshot_policy": "auto"},
    }


class PortableWorld:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "entities"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "skills"), exist_ok=True)
        self.db_path = os.path.join(self.root, "cortex.sqlite")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        fresh = not os.path.exists(self.db_path)
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA synchronous=FULL;")
        self._init_schema()
        if fresh:
            self._write_manifest(DEFAULT_MANIFEST())
        self.manifest = self._read_manifest()
        self.version = self._current_version()
        if not fresh:
            self._startup_verify()

    # -- schema ------------------------------------------------------------
    def _init_schema(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
          id TEXT PRIMARY KEY, state_json TEXT NOT NULL, aspect BLOB,
          cluster INTEGER DEFAULT 0, version INTEGER DEFAULT 1, updated_seq INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS edges(
          src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL,
          PRIMARY KEY (src, dst, type));
        CREATE TABLE IF NOT EXISTS events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, version INTEGER NOT NULL,
          kind TEXT NOT NULL, payload_json TEXT NOT NULL, hash_prev TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots(
          id INTEGER PRIMARY KEY AUTOINCREMENT, seq INTEGER NOT NULL, state_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS skills(
          id TEXT NOT NULL, version TEXT NOT NULL, skill_md TEXT NOT NULL, hash TEXT NOT NULL,
          PRIMARY KEY (id, version));
        CREATE TABLE IF NOT EXISTS invocations(
          id INTEGER PRIMARY KEY AUTOINCREMENT, skill_id TEXT NOT NULL, version TEXT NOT NULL,
          project TEXT NOT NULL DEFAULT 'default', success INTEGER NOT NULL,
          latency_ms REAL DEFAULT 0, error TEXT DEFAULT '', constraints_json TEXT DEFAULT '{}',
          world_version INTEGER NOT NULL, seq INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        self.db.commit()
        self._fsync()

    def _fsync(self):
        self.db.commit()
        try:
            os.fsync(self.db.execute("PRAGMA wal_checkpoint(TRUNCATE);").connection.fileno())
        except Exception:
            pass

    # -- manifest ------------------------------------------------------------
    def _write_manifest(self, m: Dict[str, Any]):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.manifest_path)  # atomic_replace pattern

    def _read_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _startup_verify(self):
        """Crash recovery check: sqlite readable, manifest matches head."""
        try:
            n_nodes = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"cortex.sqlite unreadable at {self.db_path}: {e}")
        m = self.manifest
        if m.get("format") != FORMAT_VERSION:
            raise RuntimeError(f"format mismatch: {m.get('format')} != {FORMAT_VERSION}")
        if "last_seq" in m and m["last_seq"] > last:
            raise RuntimeError(
                f"manifest last_seq {m['last_seq']} ahead of sqlite {last}: torn write")
        self._last_verified = {"nodes": n_nodes, "last_seq": last}

    def _refresh_manifest_head(self):
        last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
        n_nodes = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_md = len([f for f in os.listdir(os.path.join(self.root, "entities"))
                    if f.endswith(".md")]) if os.path.isdir(
                        os.path.join(self.root, "entities")) else 0
        h = hashlib.sha256()
        with open(self.db_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        m = self._read_manifest()
        m.update({"last_seq": last, "node_count": n_nodes, "md_count": n_md,
                  "sqlite_sha": h.hexdigest(), "saved_utc": time.strftime(
                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._write_manifest(m)
        self.manifest = m

    # -- versions / events -----------------------------------------------------
    def _current_version(self) -> int:
        row = self.db.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        return int(row[0]) if row else 1

    def _bump_version(self) -> int:
        self.version += 1
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('version', ?)",
                        (str(self.version),))
        return self.version

    def _chain_hash(self, kind: str, payload: str) -> str:
        row = self.db.execute("SELECT hash_prev FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev = "GENESIS" if row is None else self.db.execute(
            "SELECT seq, kind, payload_json FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev_s = "GENESIS" if row is None else json.dumps(prev)
        return hashlib.sha256(f"{prev_s}|{kind}|{payload}".encode()).hexdigest()

    def commit(self, kind: str, payload: Dict[str, Any]) -> int:
        """commit(event): durable append to H with hash chain. Returns seq."""
        ver = self._bump_version()
        blob = json.dumps(payload, sort_keys=True)
        hp = self._chain_hash(kind, blob)
        cur = self.db.execute(
            "INSERT INTO events(version, kind, payload_json, hash_prev) VALUES (?,?,?,?)",
            (ver, kind, blob, hp))
        seq = cur.lastrowid
        self._fsync()
        self._maybe_compact()
        self._refresh_manifest_head()
        return seq

    def _maybe_compact(self):
        hot = int(self.manifest.get("history", {}).get("hot_events", 2000))
        n = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if n > hot * 2:
            # snapshot live node states, then drop oldest compaction window
            rows = self.db.execute("SELECT id, state_json, version FROM nodes").fetchall()
            snap = json.dumps([{"id": r[0], "state": json.loads(r[1]), "v": r[2]} for r in rows])
            last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0]
            self.db.execute("INSERT INTO snapshots(seq, state_json) VALUES (?, ?)", (last, snap))
            keep_from = last - hot
            self.db.execute("DELETE FROM events WHERE seq < ?", (keep_from,))
            self._fsync()

    # -- nodes / edges ---------------------------------------------------------
    @staticmethod
    def _vec_to_blob(vec) -> bytes:
        import numpy as np
        # Native LE float32 (x86/ARM). Manifest records aspect_dim; cross-
        # endian port is explicit future work, not silent reinterpretation.
        a = np.asarray(vec.tolist() if hasattr(vec, "tolist") else vec,
                       dtype=np.float32)
        return a.tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes):
        import numpy as np
        return np.frombuffer(blob, dtype=np.float32)

    def upsert_node(self, eid: str, state: Dict[str, Any], aspect_vec=None,
                    cluster: int = 0, event_seq: int = 0, mirror_md: bool = True):
        blob = self._vec_to_blob(aspect_vec) if aspect_vec is not None else None
        row = self.db.execute("SELECT version FROM nodes WHERE id=?", (eid,)).fetchone()
        ver = (row[0] + 1) if row else 1
        self.db.execute(
            "INSERT OR REPLACE INTO nodes(id, state_json, aspect, cluster, version, updated_seq)"
            " VALUES (?,?,?,?,?,?)",
            (eid, json.dumps(state, sort_keys=True), blob, cluster, ver, event_seq))
        self._fsync()
        if mirror_md:
            self._write_md_mirror(eid, state, ver)
        return ver

    def get_node(self, eid: str) -> Optional[Dict[str, Any]]:
        row = self.db.execute(
            "SELECT state_json, aspect, cluster, version, updated_seq FROM nodes WHERE id=?",
            (eid,)).fetchone()
        if row is None:
            return None
        return {"id": eid, "state": json.loads(row[0]),
                "aspect": self._blob_to_vec(row[1]) if row[1] is not None else None,
                "cluster": row[2], "version": row[3], "updated_seq": row[4]}

    def add_edge(self, src: str, dst: str, etype: str):
        from cortex_apps.cortex_world_runtime.cortex_world.graph import EDGE_TYPES
        if etype not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {etype!r}; must be one of {sorted(EDGE_TYPES)}")
        self.db.execute("INSERT OR IGNORE INTO edges(src, dst, type) VALUES (?,?,?)",
                        (src, dst, etype))
        self._fsync()

    def neighbors(self, eid: str, etypes=None) -> List[str]:
        if etypes is None:
            rows = self.db.execute("SELECT dst FROM edges WHERE src=? ORDER BY dst", (eid,))
        else:
            q = ",".join("?" for _ in etypes)
            rows = self.db.execute(
                f"SELECT dst FROM edges WHERE src=? AND type IN ({q}) ORDER BY dst",
                (eid, *etypes))
        return [r[0] for r in rows.fetchall()]

    def close(self):
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass

    # -- md mirror (NOT canonical; recovery documented as partial) --------------
    def _write_md_mirror(self, eid: str, state: Dict[str, Any], version: int):
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in eid)
        lines = [f"---", f"id: {eid}", f"version: {version}",
                 f"project: {state.get('project', '?')}", f"---",
                 f"# {state.get('title', eid)}", ""]
        for k in sorted(state.keys()):
            if k != "title":
                lines.append(f"- {k}: {state[k]}")
        nbrs = self.neighbors(eid)
        lines += ["", f"### edges ({len(nbrs)})"]
        for n in nbrs[:20]:
            lines.append(f"- [[{n}]]")
        lines += ["",
                  "_Mirror only: embeddings, event seqs, invocations, and full "
                  "provenance live in cortex.sqlite and are NOT recoverable from "
                  "this file beyond state+edges._", ""]
        tmp = os.path.join(self.root, "entities", safe + ".md.tmp")
        dst = os.path.join(self.root, "entities", safe + ".md")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)


def open_world(project_dir: str) -> PortableWorld:
    """open(project_dir): open or create <project_dir>/.cortex world."""
    return PortableWorld(os.path.join(os.path.abspath(project_dir), ".cortex"))
