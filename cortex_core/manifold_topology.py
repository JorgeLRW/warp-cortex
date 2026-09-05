from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def canonicalize_tokens(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if len(token) < 3 or token in _STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def normalize_entity_refs(raw: Any) -> Set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return canonicalize_tokens(raw)

    values: Set[str] = set()
    if isinstance(raw, dict):
        iterable = raw.values()
    elif isinstance(raw, (list, tuple, set)):
        iterable = raw
    else:
        iterable = [raw]

    for item in iterable:
        values.update(canonicalize_tokens(str(item)))
    return values


def overlap_score(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def sequence_index(metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("sequence_index")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_RELATION_METADATA_KEYS = {
    "depends_on": "depends_on",
    "supports": "supports",
    "caused_by": "caused_by",
    "blocks": "blocks",
    "related_to": "related_to",
    "projection_node_ids": "projection_member",
    "projection_bridge_node_ids": "projection_bridge",
    "component_node_ids": "component_member",
}


def normalize_node_ids(raw: Any) -> Set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {part.strip() for part in re.split(r"[;,]+", raw) if part.strip()}

    values: Set[str] = set()
    if isinstance(raw, dict):
        iterable = raw.values()
    elif isinstance(raw, (list, tuple, set)):
        iterable = raw
    else:
        iterable = [raw]

    for item in iterable:
        text = str(item).strip()
        if text:
            values.add(text)
    return values


def relation_ids(metadata: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    if not isinstance(metadata, dict):
        return {}

    relations: Dict[str, Set[str]] = {}
    for key in _RELATION_METADATA_KEYS:
        values = normalize_node_ids(metadata.get(key))
        if values:
            relations[key] = values
    return relations


@dataclass
class ManifoldTopologyView:
    adjacency: Dict[int, Set[int]]
    components: List[List[int]]
    component_index: Dict[int, int]
    bridge_nodes: Set[int]
    keyword_sets: List[Set[str]]
    entity_sets: List[Set[str]]
    node_ids: List[str]
    edge_types: Dict[tuple[int, int], List[str]]
    edge_strengths: Dict[tuple[int, int], float]


def _connected_components(adjacency: Dict[int, Set[int]], node_count: int) -> List[List[int]]:
    seen: Set[int] = set()
    components: List[List[int]] = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        component: List[int] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component.sort()
        components.append(component)
    components.sort(key=lambda item: (-len(item), item[0] if item else -1))
    return components


def _articulation_points(adjacency: Dict[int, Set[int]], node_count: int) -> Set[int]:
    discovery = [-1] * node_count
    low = [-1] * node_count
    parent = [-1] * node_count
    bridges: Set[int] = set()
    timer = 0

    def dfs(node: int):
        nonlocal timer
        discovery[node] = timer
        low[node] = timer
        timer += 1
        child_count = 0

        for neighbor in adjacency.get(node, set()):
            if discovery[neighbor] == -1:
                parent[neighbor] = node
                child_count += 1
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])

                if parent[node] == -1 and child_count > 1:
                    bridges.add(node)
                if parent[node] != -1 and low[neighbor] >= discovery[node]:
                    bridges.add(node)
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(node_count):
        if discovery[node] == -1:
            dfs(node)
    return bridges


def build_manifold_topology(
    nodes: Sequence[Any],
    *,
    semantic_enabled: bool,
    max_neighbors: int = 6,
) -> ManifoldTopologyView:
    node_count = len(nodes)
    if node_count == 0:
        return ManifoldTopologyView(
            adjacency={},
            components=[],
            component_index={},
            bridge_nodes=set(),
            keyword_sets=[],
            entity_sets=[],
            node_ids=[],
            edge_types={},
            edge_strengths={},
        )

    keyword_sets: List[Set[str]] = []
    entity_sets: List[Set[str]] = []
    node_ids: List[str] = []
    relation_sets: List[Dict[str, Set[str]]] = []
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        node_id = str(
            getattr(node, "node_id", "")
            or metadata.get("node_id")
            or f"node_{len(node_ids)}"
        ).strip()
        node_ids.append(node_id)
        raw_keywords = metadata.get("keywords")
        if raw_keywords:
            keywords = normalize_entity_refs(raw_keywords)
        else:
            keywords = canonicalize_tokens(getattr(node, "text", ""))
        keyword_sets.append(keywords)

        raw_entities = metadata.get("entity_refs") or metadata.get("entities") or metadata.get("entity_ref")
        entity_sets.append(normalize_entity_refs(raw_entities))
        relation_sets.append(relation_ids(metadata))

    semantic_scores = None
    if semantic_enabled and node_count > 1:
        matrix = torch.stack([
            F.normalize(getattr(node, "embedding").detach().float().reshape(-1).cpu(), dim=0)
            for node in nodes
        ], dim=0)
        semantic_scores = matrix @ matrix.T

    candidate_edges: Dict[int, List[tuple[float, int, Set[str]]]] = {index: [] for index in range(node_count)}
    for left in range(node_count):
        left_node = nodes[left]
        left_agent = getattr(left_node, "agent_id", None)
        left_time = float(getattr(left_node, "timestamp", 0.0) or 0.0)
        left_meta = getattr(left_node, "metadata", {}) or {}
        left_sequence = sequence_index(left_meta)
        left_node_id = node_ids[left]

        for right in range(left + 1, node_count):
            right_node = nodes[right]
            right_agent = getattr(right_node, "agent_id", None)
            right_time = float(getattr(right_node, "timestamp", 0.0) or 0.0)
            right_meta = getattr(right_node, "metadata", {}) or {}
            right_sequence = sequence_index(right_meta)
            right_node_id = node_ids[right]

            semantic = 0.0
            if semantic_scores is not None:
                semantic = max(float(semantic_scores[left, right].item()), 0.0)

            lexical = overlap_score(keyword_sets[left], keyword_sets[right])
            entity = overlap_score(entity_sets[left], entity_sets[right])
            has_explicit_entities = bool(entity_sets[left]) and bool(entity_sets[right])

            temporal = 0.0
            if left_agent and left_agent == right_agent:
                if left_sequence is not None and right_sequence is not None:
                    delta = abs(left_sequence - right_sequence)
                    if delta <= 1.0:
                        temporal = 1.0
                    elif delta <= 3.0:
                        temporal = 0.5
                else:
                    delta_time = abs(left_time - right_time)
                    if delta_time <= 30.0:
                        temporal = 0.35
                    elif delta_time <= 300.0:
                        temporal = 0.15

            structural_labels: Set[str] = set()
            for relation_key, relation_label in _RELATION_METADATA_KEYS.items():
                if right_node_id in relation_sets[left].get(relation_key, set()):
                    structural_labels.add(relation_label)
                if left_node_id in relation_sets[right].get(relation_key, set()):
                    structural_labels.add(relation_label)
            structural = 1.0 if structural_labels else 0.0

            strength = 0.35 * semantic + 0.20 * lexical + 0.15 * entity + 0.10 * temporal + 0.20 * structural
            if structural_labels:
                linked = True
            elif has_explicit_entities:
                linked = entity >= 0.5
            else:
                linked = (
                    entity >= 0.5
                    or lexical >= 0.45
                    or (semantic_enabled and semantic >= 0.78)
                    or strength >= 0.42
                )
            if not linked:
                continue

            candidate_edges[left].append((strength, right, set(structural_labels)))
            candidate_edges[right].append((strength, left, set(structural_labels)))

    adjacency: Dict[int, Set[int]] = {index: set() for index in range(node_count)}
    edge_types: Dict[tuple[int, int], List[str]] = {}
    edge_strengths: Dict[tuple[int, int], float] = {}
    for node, edges in candidate_edges.items():
        edges.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        for strength, neighbor, labels in edges[:max_neighbors]:
            adjacency[node].add(neighbor)
            adjacency[neighbor].add(node)
            edge_key = (min(node, neighbor), max(node, neighbor))
            existing_labels = set(edge_types.get(edge_key, []))
            merged_labels = sorted(existing_labels | labels)
            if merged_labels:
                edge_types[edge_key] = merged_labels
            edge_strengths[edge_key] = max(edge_strengths.get(edge_key, 0.0), float(strength))

    components = _connected_components(adjacency, node_count)
    component_index = {
        node_index: component_id
        for component_id, component in enumerate(components)
        for node_index in component
    }
    bridge_nodes = _articulation_points(adjacency, node_count)

    return ManifoldTopologyView(
        adjacency=adjacency,
        components=components,
        component_index=component_index,
        bridge_nodes=bridge_nodes,
        keyword_sets=keyword_sets,
        entity_sets=entity_sets,
        node_ids=node_ids,
        edge_types=edge_types,
        edge_strengths=edge_strengths,
    )