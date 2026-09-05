# Cortex World Status Contract

The portable world exposes a content-free health and provenance view for
workspace integrations. Consumers should call:

```bash
python -m cortex_apps.cortex_world_runtime.cortex_world.cli status <project_dir>
```

The command never creates `.cortex`, `cortex.sqlite`, or `manifest.json`.

## Schema

The top-level `status_schema` is `cortex-status-v1`.

The `presence` object explicitly reports `manifest_present`, `sqlite_present`,
`entities_present`, and `skills_present`. These fields describe filesystem
presence only; they do not imply that a present file is valid or current.

### Lifecycle

- `absent`: no `.cortex` directory exists.
- `initialized_empty`: the canonical world is valid but has no events, nodes,
  edges, or skills.
- `ready`: canonical SQLite state, retained event history, manifest metadata,
  and all deterministic projections agree.
- `degraded`: canonical SQLite state is readable, but manifest metadata or one
  or more projections are stale, missing, or partial.
- `damaged`: SQLite cannot be read, the manifest format is unsupported, the
  retained event chain is broken, or a snapshot envelope is invalid.

### Canonical fields

`canonical` describes `cortex.sqlite` only:

- `event_seq`: the highest retained event sequence. This is event-log progress,
  not a general-purpose world revision.
- `verified_seq`: the highest sequence covered by a successful retained-chain
  verification. It is `null` when verification is disabled or fails.
- `event_head_hash`: the stored hash for the latest event, or `GENESIS` for an
  empty log.
- `node_count`, `edge_count`, `skill_count`: counts in canonical SQLite tables.
- `sqlite_sha256`: SHA-256 of the SQLite database file after the normal WAL
  checkpoint path.
- `chain_status`: `verified`, `unverified`, or `broken`.

`last_seq` remains in `manifest.json` for backward compatibility, but new
consumers should use `canonical.event_seq` and `canonical.verified_seq` from
this status response.

### Projection fields

`projections.entities` verifies every expected entity mirror by rendering the
canonical state and outgoing edges and comparing the complete file contents.
Its status is one of `empty`, `complete`, `stale`, `partial`, or `absent`.

`projections.skills` verifies that each `SKILL.md` exists and matches a hash
for one of the corresponding canonical skill versions. Its status uses the
same vocabulary.

Markdown and `SKILL.md` files remain projections, not recovery authorities.
A `complete` projection means the current deterministic projection agrees with
SQLite; it does not make the projection canonical or lossless.

### Consistency

`consistency.status` is one of:

- `consistent`: canonical state and projections agree.
- `partial_mirror`: canonical state is verified but a projection is missing,
  stale, or incomplete.
- `manifest_stale`: canonical state is verified but derived manifest fields no
  longer match it.
- `damaged`: canonical verification failed.
- `absent`: no world exists.

`workspace_id` is optional metadata passed through from the manifest. The
status API does not infer identity from an absolute filesystem path, so a
portable world can move between machines without changing identity policy.

## Migration and write ordering

Opening a valid `cortex-world-v1` world upgrades a missing status schema and
refreshes derived manifest fields. Read-only `status` inspection does not write
that migration; it reports missing derived fields as `manifest_stale`.

SQLite commits use WAL and full synchronous mode. Manifest replacement and
each mirror replacement use temporary files followed by atomic replacement.
There is intentionally no single transaction spanning SQLite and every mirror.
A consumer must therefore treat `degraded` and `partial_mirror` as normal
crash-recovery states and wait for the next projection refresh rather than
reading stale mirrors as canonical data.

## Integration rule

Unoctu should consume only this status response and should not inspect
`cortex.sqlite` contents or depend on internal table layout. In particular,
`event_seq` and `node_count` should not be presented as trustworthy freshness
claims unless `chain_status` is `verified` and `consistency.status` is
`consistent` or the consumer explicitly accepts a degraded world.

## Inference cache boundary

The inference runtime now routes auto-compaction through
`cortex_core.cache_control.PythonKVCacheAdapter`. This adapter supports
Hugging Face-style tuple and `DynamicCache` objects and reports
`backend=huggingface-python-kv`.

It does **not** claim:

- ownership of vLLM or SGLang paged block tables;
- native in-place page mutation;
- copy-on-write cache branching;
- scheduler-level interruption or preemption;
- witness-complex or persistent-homology routing.

A future backend adapter must expose its capabilities explicitly before the
runtime can make any of those claims.
