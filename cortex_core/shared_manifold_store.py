from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import torch


class SQLiteSharedManifoldStore:
    """Durable shared-manifold backing store with SQLite WAL semantics."""

    def __init__(self, db_path: str, *, timeout_seconds: float = 30.0):
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.timeout_seconds, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self):
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    score REAL NOT NULL,
                    source TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    agent_id TEXT,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_nodes_timestamp ON shared_nodes(timestamp DESC)"
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(shared_nodes)").fetchall()
            }
            if "node_id" not in columns:
                conn.execute("ALTER TABLE shared_nodes ADD COLUMN node_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shared_nodes_node_id ON shared_nodes(node_id)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS shared_hot_cache (cache_key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
            )

    @staticmethod
    def _tensor_to_blob(tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(tensor.detach().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _blob_to_tensor(blob: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(blob), map_location="cpu")

    @staticmethod
    def _payload_to_blob(payload: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    @staticmethod
    def _blob_to_payload(blob: bytes) -> Dict[str, Any]:
        return torch.load(io.BytesIO(blob), map_location="cpu")

    def _insert_node(self, conn: sqlite3.Connection, node: Any):
        metadata_json = json.dumps(getattr(node, "metadata", {}) or {}, sort_keys=True)
        conn.execute(
            """
            INSERT INTO shared_nodes(node_id, text, embedding, score, source, node_type, agent_id, timestamp, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                getattr(node, "node_id", None),
                node.text,
                sqlite3.Binary(self._tensor_to_blob(node.embedding)),
                float(node.score),
                node.source,
                node.node_type,
                node.agent_id,
                float(node.timestamp),
                metadata_json,
            ),
        )

    def _prune_overflow(self, conn: sqlite3.Connection, *, capacity: int):
        overflow = conn.execute("SELECT COUNT(*) AS count FROM shared_nodes").fetchone()["count"] - int(capacity)
        if overflow > 0:
            conn.execute(
                """
                DELETE FROM shared_nodes
                WHERE id IN (
                    SELECT id FROM shared_nodes
                    ORDER BY score ASC, timestamp ASC
                    LIMIT ?
                )
                """,
                (int(overflow),),
            )

    def append_node(self, node: Any, *, capacity: int):
        with self._connection() as conn:
            self._insert_node(conn, node)
            self._prune_overflow(conn, capacity=capacity)

    def upsert_node(self, node: Any, *, capacity: int):
        with self._connection() as conn:
            node_id = str(getattr(node, "node_id", "") or "").strip()
            if node_id:
                conn.execute("DELETE FROM shared_nodes WHERE node_id = ?", (node_id,))
            self._insert_node(conn, node)
            self._prune_overflow(conn, capacity=capacity)

    def list_nodes(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT node_id, text, embedding, score, source, node_type, agent_id, timestamp, metadata_json "
            "FROM shared_nodes ORDER BY timestamp ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            result.append({
                "node_id": row["node_id"],
                "text": row["text"],
                "embedding": self._blob_to_tensor(row["embedding"]),
                "score": float(row["score"]),
                "source": row["source"],
                "node_type": row["node_type"],
                "agent_id": row["agent_id"],
                "timestamp": float(row["timestamp"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
            })
        return result

    def write_hot_cache(self, payload: Dict[str, Any], *, cache_key: str = "default"):
        blob = sqlite3.Binary(self._payload_to_blob(payload))
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO shared_hot_cache(cache_key, payload) VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload = excluded.payload
                """,
                (cache_key, blob),
            )

    def read_hot_cache(self, *, cache_key: str = "default") -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT payload FROM shared_hot_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return self._blob_to_payload(row["payload"])