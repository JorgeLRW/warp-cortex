"""Typed graph ops over the portable store (edge-type aware throughout)."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set

EDGE_TYPES = frozenset({
    "depends_on", "blocks", "supports", "refutes", "mentions", "derived_from",
})

# cascade_invalidate propagates ONLY through explicitly causal/epistemic
# types. A `mentions` (or `supports`/`refutes`) edge never invalidates.
CAUSAL_EDGE_TYPES = frozenset({"depends_on", "blocks", "derived_from"})


def bfs(store, start_id: str, max_depth: int = 3, max_nodes: int = 50,
        etypes=None, direction: str = "out") -> List[str]:
    """Specialized BFS: sorted (deterministic), edge-type filter, direction."""
    if store.get_node(start_id) is None:
        return []
    visited = {start_id}
    queue = deque([(start_id, 0)])
    result = []
    while queue and len(result) < max_nodes:
        curr, depth = queue.popleft()
        if depth > 0:
            result.append(curr)
        if depth < max_depth:
            nbrs = _neighbors(store, curr, etypes, direction)
            for nbr in sorted(nbrs):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
    return result


def _neighbors(store, eid: str, etypes, direction: str) -> Set[str]:
    out = set(store.neighbors(eid, etypes))
    if direction in ("in", "both"):
        rows = store.db.execute("SELECT src FROM edges WHERE dst=? ORDER BY src", (eid,))
        ins = {r[0] for r in rows.fetchall()}
        if etypes is not None:
            q = ",".join("?" for _ in etypes)
            rows = store.db.execute(
                f"SELECT src FROM edges WHERE dst=? AND type IN ({q}) ORDER BY src",
                (eid, *etypes))
            ins = {r[0] for r in rows.fetchall()}
        out |= ins
    if direction == "in":
        out -= set(store.neighbors(eid, etypes))
    return out


def ancestors(store, eid: str, etypes=None) -> Set[str]:
    """All nodes that can reach eid (reverse traversal)."""
    seen, queue = {eid}, deque([eid])
    while queue:
        curr = queue.popleft()
        for p in _neighbors(store, curr, etypes, "in"):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    seen.discard(eid)
    return seen


def descendants(store, eid: str, etypes=None) -> Set[str]:
    seen, queue = {eid}, deque([eid])
    while queue:
        curr = queue.popleft()
        for c in _neighbors(store, curr, etypes, "out"):
            if c not in seen:
                seen.add(c)
                queue.append(c)
    seen.discard(eid)
    return seen


def topo_sort(store, etypes=None) -> List[str]:
    """Kahn's algorithm over (optionally type-filtered) edges. Deterministic."""
    nodes = [r[0] for r in store.db.execute("SELECT id FROM nodes ORDER BY id").fetchall()]
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    rows = store.db.execute("SELECT src, dst, type FROM edges ORDER BY src, dst").fetchall()
    for s, d, t in rows:
        if etypes is not None and t not in etypes:
            continue
        if s in adj and d in indeg and d not in adj[s]:
            adj[s].add(d)
            indeg[d] += 1
    import heapq
    heap = [n for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        n = heapq.heappop(heap)
        order.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    if len(order) != len(nodes):
        raise ValueError("topo_sort: cycle detected in filtered graph")
    return order


def articulation_points(store, etypes=None) -> Set[str]:
    """Tarjan articulation points on the undirected projection. Deterministic."""
    adj: Dict[str, Set[str]] = {}
    rows = store.db.execute("SELECT src, dst, type FROM edges ORDER BY src, dst").fetchall()
    for s, d, t in rows:
        if etypes is not None and t not in etypes:
            continue
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    arts: Set[str] = set()
    clock = [0]

    def dfs(u: str, parent: Optional[str]):
        disc[u] = low[u] = clock[0]
        clock[0] += 1
        children = 0
        for v in sorted(adj.get(u, ())):
            if v not in disc:
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent is not None and low[v] >= disc[u]:
                    arts.add(u)
            elif v != parent:
                low[u] = min(low[u], disc[v])
        if parent is None and children > 1:
            arts.add(u)

    for n in sorted(adj.keys()):
        if n not in disc:
            dfs(n, None)
    return arts


def cascade_invalidate(store, premise_id: str) -> Set[str]:
    """Nodes invalidated by premise failure, via CAUSAL edges only
    (depends_on / blocks / derived_from), following dependents (reverse)."""
    seen, queue = {premise_id}, deque([premise_id])
    while queue:
        curr = queue.popleft()
        for dep in _neighbors(store, curr, CAUSAL_EDGE_TYPES, "in"):
            if dep not in seen:
                seen.add(dep)
                queue.append(dep)
    seen.discard(premise_id)
    return seen
