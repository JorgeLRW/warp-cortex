"""Portable project world package (Cortex v1 product path, research frozen).

Authority (explicit, per spec freeze):
  cortex.sqlite = CANONICAL machine state (S,G,Z,H,K).
  entities/*.md + skills/*/SKILL.md = portable/exportable MIRRORS.
Recovery from mirrors restores state + edges; embeddings are re-encodable
via the manifest's encoder_id, but versions, event seqs, invocations, and
provenance chains are NOT losslessly recoverable from Markdown. Anything
claiming otherwise is a bug in the docs, not a feature.

Public surface (deliberately tiny):
  open(project_dir) / ingest(file) / commit(event) / recall(query) /
  bfs(entity_id) / select_skill(query) / record_invocation(...)
"""

from cortex_apps.cortex_world_runtime.cortex_world.store import (
    PortableWorld,
    open_world,
  inspect_world,
    FORMAT_VERSION,
  STATUS_SCHEMA_VERSION,
    DEFAULT_MANIFEST,
)
from cortex_apps.cortex_world_runtime.cortex_world.recall import (
    RecallHit,
    RecallResult,
)
from cortex_apps.cortex_world_runtime.cortex_world.graph import (
    EDGE_TYPES,
    CAUSAL_EDGE_TYPES,
)

__all__ = [
    "PortableWorld",
    "open_world",
    "inspect_world",
    "FORMAT_VERSION",
    "STATUS_SCHEMA_VERSION",
    "DEFAULT_MANIFEST",
    "RecallHit",
    "RecallResult",
    "EDGE_TYPES",
    "CAUSAL_EDGE_TYPES",
]
