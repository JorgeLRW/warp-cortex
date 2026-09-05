"""Stable CLI Unoctu will call: status/open/ingest/recall/bfs/select-skill/record-invocation.

Usage:
    python -m cortex_apps.cortex_world_runtime.cortex_world.cli status <project_dir>
  python -m cortex_apps.cortex_world_runtime.cortex_world.cli open <project_dir>
  ... ingest <project_dir> <file>
  ... recall <project_dir> <query text...>
  ... bfs <project_dir> <entity_id> [--depth 2] [--nodes 50] [--types depends_on,blocks]
  ... select-skill <project_dir> <query text...>
  ... record-invocation <project_dir> <skill_id> <version> <0|1> [error]
"""

from __future__ import annotations

import json
import sys


def _open(project_dir: str):
    from cortex_apps.cortex_world_runtime.cortex_world.store import open_world
    return open_world(project_dir)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "status":
        from cortex_apps.cortex_world_runtime.cortex_world.store import inspect_world
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("project_dir")
        p.add_argument("--no-verify", action="store_true")
        a = p.parse_args(rest)
        print(json.dumps(inspect_world(a.project_dir, verify=not a.no_verify), indent=1))
    elif cmd == "open":
        w = _open(rest[0])
        print(json.dumps({"root": w.root, "version": w.version,
                          "manifest": w.manifest}, indent=1))
        w.close()
    elif cmd == "ingest":
        from cortex_apps.cortex_world_runtime.cortex_world.ingest import harvest_file
        w = _open(rest[0])
        eids = harvest_file(w, rest[1])
        print(json.dumps({"ingested_chunks": len(eids), "eids": eids[:10]}))
        w.close()
    elif cmd == "recall":
        from cortex_apps.cortex_world_runtime.cortex_world.recall import recall
        from cortex_apps.cortex_world_runtime.workspace_knowledge_harvester import (
            GenericFrozenAspectEncoder,
        )
        w = _open(rest[0])
        enc = GenericFrozenAspectEncoder(d_out=64, seed=42)
        res = recall(w, enc.encode(" ".join(rest[1:])))
        print(json.dumps({
            "snapshot_version": res.snapshot_version,
            "candidate_budget": res.candidate_budget,
            "candidates_examined": res.candidates_examined,
            "note": res.note,
            "hits": [{"entity": h.entity_id, "score": round(h.score, 4),
                      "path": h.edge_path, "seq": h.event_seq,
                      "provenance": h.provenance} for h in res.hits]}, indent=1))
        w.close()
    elif cmd == "bfs":
        from cortex_apps.cortex_world_runtime.cortex_world.graph import bfs
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("project_dir")
        p.add_argument("entity_id")
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--nodes", type=int, default=50)
        p.add_argument("--types", default="")
        a = p.parse_args(rest)
        w = _open(a.project_dir)
        print(json.dumps(bfs(w, a.entity_id, max_depth=a.depth, max_nodes=a.nodes,
                             etypes=a.types.split(",") if a.types else None)))
        w.close()
    elif cmd == "select-skill":
        from cortex_apps.cortex_world_runtime.cortex_world.skills import select_skill
        w = _open(rest[0])
        print(json.dumps(select_skill(w, " ".join(rest[1:])), indent=1))
        w.close()
    elif cmd == "record-invocation":
        from cortex_apps.cortex_world_runtime.cortex_world.skills import record_invocation
        w = _open(rest[0])
        seq = record_invocation(w, rest[1], rest[2], bool(int(rest[3])),
                                error=rest[4] if len(rest) > 4 else "")
        print(json.dumps({"seq": seq}))
        w.close()
    else:
        print(f"unknown command {cmd!r}\n" + __doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
