"""
Sharded World Substrate: Regional Vector Versioning.
=====================================================
Partitions entities across R regions/shards:
    U_v -> { U^(r)_{v_r} }_{r=1}^R,    v = (v_1, v_2, ..., v_R)

Eliminates global optimistic conflict bottlenecks so concurrent agents
operating in distinct regions (e.g. distinct dungeon rooms, spatial tiles,
subsystems) commit independently without aborting each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from cortex_apps.cortex_world_runtime.fast_world_substrate import (
    EntityNode,
    FastWorldSubstrate,
    WorldSnapshot,
)


@dataclass
class ShardedWorldSnapshot:
    """Immutable view of the sharded world at a specific vector version."""
    vector_version: Dict[int, int]
    global_event_seq: int
    entities: Dict[str, EntityNode]
    clusters: Dict[int, List[str]]
    cluster_centroids: torch.Tensor
    entity_to_region: Dict[str, int]
    region_entities: Dict[int, List[str]]

    def get_region_version(self, region_id: int) -> int:
        return self.vector_version.get(region_id, 1)

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        return self.entities.get(entity_id)

    def get_entity_region(self, entity_id: str) -> int:
        return self.entity_to_region.get(entity_id, 0)

    def bfs(self, start_id: str, max_depth: int = 2, max_nodes: int = 25) -> List[str]:
        if start_id not in self.entities:
            return []
        visited: Set[str] = {start_id}
        queue: List[Tuple[str, int]] = [(start_id, 0)]
        result: List[str] = []
        while queue and len(result) < max_nodes:
            curr, depth = queue.pop(0)
            result.append(curr)
            if depth < max_depth:
                node = self.entities.get(curr)
                if node:
                    # sorted(): see fast_world_substrate.WorldSnapshot.bfs.
                    for nbr in sorted(node.neighbors):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append((nbr, depth + 1))
        return result

    def search_semantics_indexed(
        self,
        query_aspect_vec: torch.Tensor,
        top_k: int = 5,
        candidate_budget: int = 128,
    ) -> List[Tuple[str, float]]:
        if len(self.entities) == 0:
            return []
        cluster_sims = torch.matmul(self.cluster_centroids, query_aspect_vec)
        best_clusters = torch.argsort(cluster_sims, descending=True)
        candidates: List[str] = []
        for c_idx in best_clusters:
            c_int = int(c_idx.item())
            ents = self.clusters.get(c_int, [])
            candidates.extend(ents)
            if len(candidates) >= candidate_budget:
                break
        candidates = candidates[:candidate_budget]
        if not candidates:
            return []
        cand_tensors = torch.stack([self.entities[eid].aspect_vector for eid in candidates])
        sims = torch.matmul(cand_tensors, query_aspect_vec)
        k = min(top_k, len(candidates))
        top_vals, top_indices = torch.topk(sims, k=k)
        return [(candidates[idx.item()], float(val.item())) for idx, val in zip(top_indices, top_vals)]


class ShardedWorldSubstrate:
    """
    World substrate partitioned into R independent regions with local versioning.
    """
    def __init__(self, num_regions: int = 16, num_clusters: int = 32):
        self.num_regions = num_regions
        self.num_clusters = num_clusters
        self.region_versions: Dict[int, int] = {r: 1 for r in range(num_regions)}
        self.global_event_seq: int = 0

        self.entities: Dict[str, EntityNode] = {}
        self.entity_to_region: Dict[str, int] = {}
        self.region_entities: Dict[int, List[str]] = {r: [] for r in range(num_regions)}

        self.clusters: Dict[int, List[str]] = {c: [] for c in range(num_clusters)}
        self.cluster_centroids = torch.randn(num_clusters, 64)
        self.cluster_centroids = torch.nn.functional.normalize(self.cluster_centroids, p=2, dim=1)

        self.event_log: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}

        # Regional locks for fine-grained concurrency
        self._region_locks: List[threading.Lock] = [threading.Lock() for _ in range(num_regions)]
        self._global_lock = threading.Lock()

    def populate_synthetic_world(
        self,
        num_entities: int = 10000,
        edges_per_entity: int = 4,
    ) -> None:
        raw_embs = torch.randn(num_entities, 64)
        raw_embs = torch.nn.functional.normalize(raw_embs, p=2, dim=1)
        cluster_assignments = torch.argmax(torch.matmul(raw_embs, self.cluster_centroids.T), dim=1)

        for i in range(num_entities):
            eid = f"ent_{i:06d}"
            r_id = i % self.num_regions
            c_id = int(cluster_assignments[i].item())
            self.entity_to_region[eid] = r_id
            self.region_entities[r_id].append(eid)
            self.clusters[c_id].append(eid)

            out_edges = [
                f"ent_{((i + j * 7 + 1) % num_entities):06d}"
                for j in range(edges_per_entity)
            ]
            self.entities[eid] = EntityNode(
                entity_id=eid,
                state={"health": 100, "resource_units": 10, "status": "ACTIVE", "region": r_id},
                aspect_vector=raw_embs[i],
                cluster_id=c_id,
                neighbors=set(out_edges),
            )

    def current_snapshot(self) -> ShardedWorldSnapshot:
        with self._global_lock:
            return ShardedWorldSnapshot(
                vector_version=dict(self.region_versions),
                global_event_seq=self.global_event_seq,
                entities=self.entities,
                clusters=self.clusters,
                cluster_centroids=self.cluster_centroids,
                entity_to_region=self.entity_to_region,
                region_entities=self.region_entities,
            )

    def clock1_tick_batch(self, deltas_by_region: Dict[int, List[Tuple[str, Dict[str, Any]]]]) -> float:
        """Batch ingestion partitioned across regions in Clock 1."""
        t0 = time.perf_counter()
        with self._global_lock:
            for r_id, deltas in deltas_by_region.items():
                for eid, state_patch in deltas:
                    node = self.entities.get(eid)
                    if node:
                        node.state.update(state_patch)
                self.region_versions[r_id] += 1
                self.global_event_seq += 1
        return (time.perf_counter() - t0) * 1000.0

    def commit_intent_regional(
        self,
        agent_id: str,
        region_id: int,
        expected_region_version: int,
        intent_deltas: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[bool, int, str]:
        """
        Commits action intent targeting a single region with local verification.
        Mutations in region r1 DO NOT conflict with mutations in region r2.
        """
        with self._region_locks[region_id]:
            curr_v = self.region_versions[region_id]
            if curr_v != expected_region_version:
                return False, curr_v, f"Conflict: region {region_id} version changed ({expected_region_version} -> {curr_v})"

            # Apply delta to region
            for eid, state_patch in intent_deltas:
                node = self.entities.get(eid)
                if node:
                    node.state.update(state_patch)

            self.region_versions[region_id] += 1
            new_v = self.region_versions[region_id]

            with self._global_lock:
                self.global_event_seq += 1
                self.event_log.append({
                    "seq": self.global_event_seq,
                    "agent_id": agent_id,
                    "region_id": region_id,
                    "region_version": new_v,
                    "delta_count": len(intent_deltas),
                })

            return True, new_v, "Success: regional commit accepted"

    def commit_intent_multi_region(
        self,
        agent_id: str,
        expected_versions: Dict[int, int],
        intent_deltas: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[bool, Dict[int, int], str]:
        """
        Commits action intent spanning multiple regions using canonical 2-phase lock ordering.
        """
        sorted_regions = sorted(expected_versions.keys())
        # Acquire locks in canonical order to prevent deadlocks
        acquired = [self._region_locks[r] for r in sorted_regions]
        for lk in acquired:
            lk.acquire()

        try:
            # Phase 1: Verify all expected region versions
            for r, exp_v in expected_versions.items():
                if self.region_versions[r] != exp_v:
                    return False, dict(self.region_versions), f"Multi-region conflict on region {r}"

            # Phase 2: Commit deltas across regions
            for eid, state_patch in intent_deltas:
                node = self.entities.get(eid)
                if node:
                    node.state.update(state_patch)

            for r in sorted_regions:
                self.region_versions[r] += 1

            with self._global_lock:
                self.global_event_seq += 1
                self.event_log.append({
                    "seq": self.global_event_seq,
                    "agent_id": agent_id,
                    "regions": sorted_regions,
                    "delta_count": len(intent_deltas),
                })

            return True, dict(self.region_versions), "Multi-region commit success"
        finally:
            for lk in reversed(acquired):
                lk.release()
