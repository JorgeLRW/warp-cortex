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
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

FORMAT_VERSION = "cortex-world-v1"
STATUS_SCHEMA_VERSION = "cortex-status-v1"


def DEFAULT_MANIFEST() -> Dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "status_schema": STATUS_SCHEMA_VERSION,
        "encoder": {"id": "qwen2.5-0.5b-embed-rp64-v1", "aspect_dim": 64},
        # Operational budgets, NOT retrieval guarantees: fixed-budget recall
        # is known to degrade with world size (see retrieval_law_results.json).
        "budgets": {"bfs_nodes": 50, "semantic_candidates": 400, "top_k": 5},
        "history": {"hot_events": 2000, "snapshot_policy": "auto"},
    }


def _world_presence(root: str) -> Dict[str, bool]:
    return {
        "manifest_present": os.path.isfile(os.path.join(root, "manifest.json")),
        "sqlite_present": os.path.isfile(os.path.join(root, "cortex.sqlite")),
        "entities_present": os.path.isdir(os.path.join(root, "entities")),
        "skills_present": os.path.isdir(os.path.join(root, "skills")),
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
        self._refresh_manifest_head()

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
    @contextmanager
    def _manifest_lock(self):
        lock_path = self.manifest_path + ".lock"
        with open(lock_path, "a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write_manifest(self, m: Dict[str, Any], _locked: bool = False):
        def write():
            tmp = f"{self.manifest_path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.manifest_path)

        if _locked:
            write()
        else:
            with self._manifest_lock():
                write()

    def _read_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if "status_schema" not in manifest:
            manifest["status_schema"] = STATUS_SCHEMA_VERSION
        return manifest

    def _startup_verify(self):
        """Crash recovery check: chain validity + snapshot linkage over retained
        history. Cost is O(retained events); measured as recovery latency.
        Any failure raises: silent rewrites must not open cleanly."""
        m = self.manifest
        if m.get("format") != FORMAT_VERSION:
            raise RuntimeError(f"format mismatch: {m.get('format')} != {FORMAT_VERSION}")
        try:
            n_nodes = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"cortex.sqlite unreadable at {self.db_path}: {e}")
        if "last_seq" in m and int(m["last_seq"]) > last:
            raise RuntimeError(
                f"manifest last_seq {m['last_seq']} ahead of sqlite {last}: torn write")
        ok, checked = self.verify_chain()
        if not ok:
            raise RuntimeError("event hash chain broken: retained history rewritten")
        for sid, seq, blob in self.db.execute("SELECT id, seq, state_json FROM snapshots").fetchall():
            try:
                env = json.loads(blob)
                assert isinstance(env, dict) and "state_hash" in env
            except Exception:
                raise RuntimeError(f"snapshot {sid}: unreadable envelope (rewritten?)")
            if int(env.get("seq", -1)) != seq:
                raise RuntimeError(f"snapshot {sid}: seq mismatch (rewritten?)")
            st = json.dumps(env.get("state"), sort_keys=True)
            if hashlib.sha256(st.encode()).hexdigest() != env.get("state_hash"):
                raise RuntimeError(f"snapshot {sid}: state hash mismatch (rewritten?)")
        self._last_verified = {"nodes": n_nodes, "last_seq": last,
                               "chain_checked": checked}

    def _meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def verify_chain(self) -> tuple:
        """Recompute hash_prev links over retained events from the stored
        truncation anchor (GENESIS if never compacted). Returns (ok, n)."""
        anchor = self._meta("chain_anchor", "GENESIS")
        prev_desc = anchor
        rows = self.db.execute(
            "SELECT seq, kind, payload_json, hash_prev FROM events ORDER BY seq").fetchall()
        for seq, kind, payload, stored in rows:
            expect = hashlib.sha256(f"{prev_desc}|{kind}|{payload}".encode()).hexdigest()
            if expect != stored:
                return False, seq
            # next descriptor must BYTE-match _chain_hash's json.dumps(row tuple)
            prev_desc = json.dumps([seq, kind, payload])
        return True, len(rows)

    def _refresh_manifest_head(self):
        with self._manifest_lock():
            self._refresh_manifest_head_locked()

    def _refresh_manifest_head_locked(self):
        last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0] or 0
        n_nodes = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_skills = self.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        n_md = len([f for f in os.listdir(os.path.join(self.root, "entities"))
                    if f.endswith(".md")]) if os.path.isdir(
                        os.path.join(self.root, "entities")) else 0
        n_skill_md = len([
            name for name in os.listdir(os.path.join(self.root, "skills"))
            if os.path.isdir(os.path.join(self.root, "skills", name))
            and os.path.isfile(os.path.join(self.root, "skills", name, "SKILL.md"))
        ]) if os.path.isdir(os.path.join(self.root, "skills")) else 0
        h = hashlib.sha256()
        with open(self.db_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        head = self.db.execute(
            "SELECT hash_prev FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        m = self._read_manifest()
        m.update({"status_schema": STATUS_SCHEMA_VERSION,
                  "last_seq": last, "node_count": n_nodes,
                  "edge_count": n_edges, "skill_count": n_skills,
                  "md_count": n_md, "skill_md_count": n_skill_md,
                  "event_head_hash": head[0] if head else "GENESIS",
                  "sqlite_sha": h.hexdigest(), "saved_utc": time.strftime(
                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self._write_manifest(m, _locked=True)
        self.manifest = m

    def refresh_manifest(self) -> Dict[str, Any]:
        """Refresh derived manifest metadata after an external table update."""
        self._refresh_manifest_head()
        return dict(self.manifest)

    @staticmethod
    def _mirror_name(entity_id: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in entity_id)

    @staticmethod
    def _state_hash(state: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def _edge_hash(self, eid: str) -> str:
        return hashlib.sha256(
            json.dumps(self.neighbors(eid), sort_keys=True).encode()
        ).hexdigest()

    def _snapshot_errors(self) -> List[str]:
        errors = []
        for sid, seq, blob in self.db.execute(
            "SELECT id, seq, state_json FROM snapshots"
        ).fetchall():
            try:
                env = json.loads(blob)
                if not isinstance(env, dict) or "state_hash" not in env:
                    raise ValueError("missing state_hash")
            except Exception:
                errors.append(f"snapshot {sid}: unreadable envelope")
                continue
            if int(env.get("seq", -1)) != seq:
                errors.append(f"snapshot {sid}: seq mismatch")
                continue
            state = json.dumps(env.get("state"), sort_keys=True)
            if hashlib.sha256(state.encode()).hexdigest() != env.get("state_hash"):
                errors.append(f"snapshot {sid}: state hash mismatch")
        return errors

    def _projection_status(self) -> Dict[str, Any]:
        nodes = {
            row[0]: {"version": row[1], "updated_seq": row[2],
                     "state": json.loads(row[3])}
            for row in self.db.execute(
                "SELECT id, version, updated_seq, state_json FROM nodes"
            ).fetchall()
        }
        node_ids = set(nodes)
        entity_dir = os.path.join(self.root, "entities")
        entity_files = {
            name[:-3] for name in os.listdir(entity_dir) if name.endswith(".md")
        } if os.path.isdir(entity_dir) else set()
        expected_entity_files = {self._mirror_name(eid) for eid in node_ids}
        stale_entity_files = []
        for eid, node in nodes.items():
            safe = self._mirror_name(eid)
            path = os.path.join(entity_dir, safe + ".md")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    actual = f.read()
                expected = self._render_md_mirror(
                    eid, node["state"], node["version"],
                    event_seq=node["updated_seq"],
                )
                if actual != expected:
                    stale_entity_files.append(safe)
            except (OSError, TypeError, ValueError):
                stale_entity_files.append(safe)

        skill_ids = {
            row[0] for row in self.db.execute("SELECT DISTINCT id FROM skills").fetchall()
        }
        skill_rows = self.db.execute(
            "SELECT id, version, hash FROM skills"
        ).fetchall()
        skill_hashes = {}
        for skill_id, version, stored_hash in skill_rows:
            skill_hashes.setdefault(skill_id, []).append((version, stored_hash))
        skill_dir = os.path.join(self.root, "skills")
        skill_files = {
            name for name in os.listdir(skill_dir)
            if os.path.isfile(os.path.join(skill_dir, name, "SKILL.md"))
        } if os.path.isdir(skill_dir) else set()
        stale_skill_files = []
        for skill_id in skill_ids & skill_files:
            path = os.path.join(skill_dir, skill_id, "SKILL.md")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                valid = any(
                    hashlib.sha256(f"{skill_id}@{version}|{content}".encode()).hexdigest()
                    == stored_hash
                    for version, stored_hash in skill_hashes.get(skill_id, [])
                )
                if not valid:
                    stale_skill_files.append(skill_id)
            except OSError:
                stale_skill_files.append(skill_id)

        def classify(expected: set, actual: set) -> str:
            if not expected and not actual:
                return "empty"
            if expected == actual:
                return "complete"
            if not actual:
                return "absent"
            return "partial"

        entity_status = classify(expected_entity_files, entity_files)
        if entity_status == "complete" and stale_entity_files:
            entity_status = "stale"
        skill_status = classify(skill_ids, skill_files)
        if skill_status == "complete" and stale_skill_files:
            skill_status = "stale"

        return {
            "entities": {
                "status": entity_status,
                "expected_count": len(expected_entity_files),
                "actual_count": len(entity_files),
                "missing": sorted(expected_entity_files - entity_files)[:20],
                "unexpected": sorted(entity_files - expected_entity_files)[:20],
                "stale": sorted(set(stale_entity_files))[:20],
            },
            "skills": {
                "status": skill_status,
                "expected_count": len(skill_ids),
                "actual_count": len(skill_files),
                "missing": sorted(skill_ids - skill_files)[:20],
                "unexpected": sorted(skill_files - skill_ids)[:20],
                "stale": sorted(set(stale_skill_files))[:20],
            },
        }

    def status(self, verify: bool = True) -> Dict[str, Any]:
        """Return a versioned, content-free health and provenance snapshot."""
        manifest = self._read_manifest()
        errors: List[str] = []
        try:
            node_count = int(self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
            edge_count = int(self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
            skill_count = int(self.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0])
            last_seq = int(self.db.execute(
                "SELECT MAX(seq) FROM events"
            ).fetchone()[0] or 0)
            head = self.db.execute(
                "SELECT hash_prev FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            sqlite_sha = self._sqlite_sha256()
        except sqlite3.DatabaseError as exc:
            return {
                "status_schema": STATUS_SCHEMA_VERSION,
                "root": self.root,
                "presence": _world_presence(self.root),
                "lifecycle": "damaged",
                "canonical": None,
                "projections": {},
                "consistency": {"status": "damaged", "errors": [str(exc)]},
            }

        chain_ok, checked = self.verify_chain() if verify else (None, None)
        if verify and not chain_ok:
            errors.append(f"event hash chain broken at seq {checked}")
        snapshot_errors = self._snapshot_errors() if verify else []
        errors.extend(snapshot_errors)
        event_head_hash = head[0] if head else "GENESIS"
        canonical = {
            "event_seq": last_seq,
            "verified_seq": last_seq if chain_ok is True else None,
            "event_head_hash": event_head_hash,
            "node_count": node_count,
            "edge_count": edge_count,
            "skill_count": skill_count,
            "sqlite_sha256": sqlite_sha,
            "chain_status": (
                "verified" if chain_ok is True else
                "unverified" if chain_ok is None else "broken"
            ),
        }
        projections = self._projection_status()

        manifest_fields = {
            "last_seq": last_seq,
            "node_count": node_count,
            "edge_count": edge_count,
            "skill_count": skill_count,
            "event_head_hash": event_head_hash,
            "sqlite_sha": sqlite_sha,
        }
        manifest_mismatches = [
            key for key, actual in manifest_fields.items()
            if key in manifest and str(manifest[key]) != str(actual)
        ]
        manifest_mismatches.extend(
            f"missing:{key}" for key in manifest_fields if key not in manifest
        )
        if manifest_mismatches:
            errors.append("manifest mismatch: " + ", ".join(manifest_mismatches))

        projection_statuses = [item["status"] for item in projections.values()]
        mirrors_complete = all(status in ("empty", "complete") for status in projection_statuses)
        if errors and any("chain" in error or "snapshot" in error for error in errors):
            consistency_status = "damaged"
            lifecycle = "damaged"
        elif manifest_mismatches:
            consistency_status = "manifest_stale"
            lifecycle = "degraded"
        elif not mirrors_complete:
            consistency_status = "partial_mirror"
            lifecycle = "degraded"
        elif last_seq == 0 and node_count == 0 and edge_count == 0 and skill_count == 0:
            consistency_status = "consistent"
            lifecycle = "initialized_empty"
        else:
            consistency_status = "consistent"
            lifecycle = "ready"

        if errors and consistency_status not in ("damaged", "manifest_stale"):
            consistency_status = "degraded"
            lifecycle = "degraded"

        return {
            "status_schema": STATUS_SCHEMA_VERSION,
            "root": self.root,
            "presence": _world_presence(self.root),
            "format": manifest.get("format"),
            "workspace_id": manifest.get("workspace_id"),
            "lifecycle": lifecycle,
            "canonical": canonical,
            "projections": projections,
            "consistency": {
                "status": consistency_status,
                "manifest_verified": not manifest_mismatches,
                "errors": errors,
            },
        }

    def _sqlite_sha256(self) -> str:
        digest = hashlib.sha256()
        with open(self.db_path, "rb") as f:
            while chunk := f.read(65536):
                digest.update(chunk)
        return digest.hexdigest()


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
            rows = self.db.execute("SELECT id, state_json, version FROM nodes").fetchall()
            state = [{"id": r[0], "state": json.loads(r[1]), "v": r[2]} for r in rows]
            last = self.db.execute("SELECT MAX(seq) FROM events").fetchone()[0]
            keep_from = last - hot
            # anchor = exact running descriptor at keep_from-1, so the retained
            # tail still verifies after deletion (read BEFORE deleting).
            anchor_row = self.db.execute(
                "SELECT seq, kind, payload_json FROM events WHERE seq < ?"
                " ORDER BY seq DESC LIMIT 1", (keep_from,)).fetchone()
            anchor = json.dumps(list(anchor_row)) if anchor_row else "GENESIS"
            anchor_seq = int(anchor_row[0]) if anchor_row else 0
            env = {"seq": last, "state": state,
                   "state_hash": hashlib.sha256(
                       json.dumps(state, sort_keys=True).encode()).hexdigest(),
                   "anchor": anchor, "anchor_seq": anchor_seq}
            self.db.execute("INSERT INTO snapshots(seq, state_json) VALUES (?, ?)",
                            (last, json.dumps(env)))
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES "
                            "('chain_anchor', ?), ('chain_anchor_seq', ?)",
                            (anchor, str(anchor_seq)))
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
            self._write_md_mirror(eid, state, ver, event_seq=event_seq)
        self._refresh_manifest_head()
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
        source = self.get_node(src)
        if source is not None:
            self._write_md_mirror(src, source["state"], source["version"],
                                   event_seq=source["updated_seq"])
        self._refresh_manifest_head()

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
    def _render_md_mirror(self, eid: str, state: Dict[str, Any], version: int,
                          event_seq: int = 0) -> str:
        safe = self._mirror_name(eid)
        lines = [f"---", f"id: {eid}", f"version: {version}",
                 f"updated_seq: {event_seq}",
             f"state_hash: {self._state_hash(state)}",
             f"edge_hash: {self._edge_hash(eid)}",
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
        return "\n".join(lines)

    def _write_md_mirror(self, eid: str, state: Dict[str, Any], version: int,
                         event_seq: int = 0):
        safe = self._mirror_name(eid)
        content = self._render_md_mirror(eid, state, version, event_seq=event_seq)
        tmp = os.path.join(self.root, "entities", safe + ".md.tmp")
        dst = os.path.join(self.root, "entities", safe + ".md")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)


def open_world(project_dir: str) -> PortableWorld:
    """open(project_dir): open or create <project_dir>/.cortex world."""
    return PortableWorld(os.path.join(os.path.abspath(project_dir), ".cortex"))


def inspect_world(project_dir: str, verify: bool = True) -> Dict[str, Any]:
    """Inspect a world without creating directories, a database, or a manifest."""
    root = os.path.join(os.path.abspath(project_dir), ".cortex")
    base = {"status_schema": STATUS_SCHEMA_VERSION, "root": root,
            "presence": _world_presence(root)}
    if not os.path.isdir(root):
        return {**base, "lifecycle": "absent", "canonical": None,
                "projections": {},
                "consistency": {"status": "absent", "manifest_verified": False,
                                 "errors": ["world directory is absent"]}}

    manifest_path = os.path.join(root, "manifest.json")
    db_path = os.path.join(root, "cortex.sqlite")
    missing = [path for path in (manifest_path, db_path) if not os.path.isfile(path)]
    if missing:
        return {**base, "lifecycle": "damaged", "canonical": None,
                "projections": {},
                "consistency": {
                    "status": "damaged", "manifest_verified": False,
                    "errors": [f"missing required file: {path}" for path in missing],
                }}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "lifecycle": "damaged", "canonical": None,
                "projections": {},
                "consistency": {"status": "damaged", "manifest_verified": False,
                                 "errors": [f"manifest unreadable: {exc}"]}}

    if manifest.get("format") != FORMAT_VERSION:
        return {**base, "format": manifest.get("format"),
                "lifecycle": "damaged", "canonical": None,
                "projections": {},
                "consistency": {
                    "status": "damaged", "manifest_verified": False,
                    "errors": [
                        f"format mismatch: {manifest.get('format')} != {FORMAT_VERSION}"
                    ],
                }}

    try:
        uri = f"file:{db_path.replace(chr(92), '/')}?mode=ro"
        db = sqlite3.connect(uri, uri=True)
    except sqlite3.DatabaseError as exc:
        return {**base, "format": manifest.get("format"),
                "lifecycle": "damaged", "canonical": None,
                "projections": {},
                "consistency": {"status": "damaged", "manifest_verified": False,
                                 "errors": [f"sqlite unreadable: {exc}"]}}

    probe = object.__new__(PortableWorld)
    probe.root = root
    probe.db_path = db_path
    probe.manifest_path = manifest_path
    probe.db = db
    probe.manifest = manifest
    try:
        result = probe.status(verify=verify)
    finally:
        db.close()
    return result
