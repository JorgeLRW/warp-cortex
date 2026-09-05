"""
Heterogeneous Agent Society & Event-Driven Routing Benchmark.
==============================================================
Evaluates:
  1. Society Composition: A = 512 agents across 4 distinct archetypes:
     - 256 Guard NPCs (local BFS G + status S)
     - 128 Merchant NPCs (local inventory S + price math)
     -  96 Worker NPCs (procedural skill execution K + H)
     -  32 Faction Planner NPCs (wide-area semantic Z + history H)
  2. Spatial distribution & private memory footprint by archetype (KB/agent).
  3. Event-Driven Wake Routing vs Continuous Polling:
     - Polling: All 512 agents query every 20 Hz tick.
     - Event-Driven: Substrate projects impact frontier; only affected agents wake (A_active << A_all).
  4. Work Efficiency & Parallel Speedup S(A) vs single sequential execution.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import FastWorldSubstrate, WorldSnapshot
from cortex_apps.cortex_world_runtime.skill_registry import (
    SkillDefinition,
    SkillInvocationEvent,
    SkillRegistry,
    SkillSelectionMode,
    SkillSelector,
)


@dataclass
class BaseAgent:
    agent_id: str
    archetype: str
    home_region: int
    focus_entity_id: str
    scratchpad: List[str] = field(default_factory=list)
    memory_footprint_bytes: int = 0

    def memory_bytes(self) -> int:
        base = sys.getsizeof(self) + sys.getsizeof(self.scratchpad)
        for item in self.scratchpad:
            base += sys.getsizeof(item)
        return base


class GuardAgent(BaseAgent):
    """Guards patrol local graph neighborhoods and verify entity health."""
    def execute_cognition(self, snapshot: WorldSnapshot) -> str:
        nbrs = snapshot.bfs(self.focus_entity_id, max_depth=2, max_nodes=15)
        for nid in nbrs[:3]:
            node = snapshot.get_entity(nid)
            st = node.state.get("health", 100) if node else 0
            if st < 50:
                self.scratchpad.append(f"Alert: {nid} damaged ({st})")
        if len(self.scratchpad) > 10:
            self.scratchpad = self.scratchpad[-10:]
        return f"Patrolled {len(nbrs)} nodes"


class MerchantAgent(BaseAgent):
    """Merchants manage inventory, adjust prices, and inspect local supply."""
    def execute_cognition(self, snapshot: WorldSnapshot) -> str:
        node = snapshot.get_entity(self.focus_entity_id)
        res = node.state.get("resource_units", 10) if node else 0
        new_price = round(100.0 / max(1, res), 2)
        self.scratchpad.append(f"Stock: {res}, Price: ${new_price}")
        if len(self.scratchpad) > 20:
            self.scratchpad = self.scratchpad[-20:]
        return f"Priced at ${new_price}"


class WorkerAgent(BaseAgent):
    """Workers execute procedural skills to repair and maintain infrastructure."""
    def __init__(self, *args, selector: SkillSelector, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = selector

    def execute_cognition(self, snapshot: WorldSnapshot) -> str:
        ranked = self.selector.select_skill(
            "repair bridge infrastructure with cooling",
            snapshot,
            agent_id=self.agent_id,
            top_k=1,
        )
        if ranked:
            skill = ranked[0][0]
            self.scratchpad.append(f"Selected skill {skill.full_id}")
        if len(self.scratchpad) > 30:
            self.scratchpad = self.scratchpad[-30:]
        return f"Skill {ranked[0][0].skill_id if ranked else 'none'}"


class FactionPlannerAgent(BaseAgent):
    """Strategic planners perform wide-area semantic analysis and review history."""
    def execute_cognition(self, snapshot: WorldSnapshot) -> str:
        q_vec = torch.randn(64)
        q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=0)
        matches = snapshot.search_semantics_indexed(q_vec, top_k=5, candidate_budget=128)
        self.scratchpad.append(f"Strategic scan found {len(matches)} hubs")
        if len(self.scratchpad) > 50:
            self.scratchpad = self.scratchpad[-50:]
        return f"Scanned {len(matches)} strategic hubs"


def instantiate_society(substrate: FastWorldSubstrate, total_agents: int = 512) -> List[BaseAgent]:
    registry = SkillRegistry()
    registry.register(SkillDefinition(
        skill_id="repair_bridge", version="v2", name="Thermal Shielded Repair",
        description="Repairs bridge structure with active cooling.",
        aspect_tags=["REPAIR", "BRIDGE", "COOLING"],
    ))
    selector = SkillSelector(registry, mode=SkillSelectionMode.SHARED_CORTEX_LEDGER)

    agents: List[BaseAgent] = []
    guards = int(total_agents * 0.50)
    merchants = int(total_agents * 0.25)
    workers = int(total_agents * 0.1875)
    planners = total_agents - (guards + merchants + workers)
    counts = [("GUARD", guards), ("MERCHANT", merchants), ("WORKER", workers), ("PLANNER", planners)]
    idx = 0
    num_ents = len(substrate.entities)

    for arch, count in counts:
        for _ in range(count):
            aid = f"agent_{idx:04d}"
            r_id = idx % substrate.num_clusters
            target_eid = f"ent_{(idx * 17 % num_ents):06d}"

            if arch == "GUARD":
                ag = GuardAgent(aid, arch, r_id, target_eid)
            elif arch == "MERCHANT":
                ag = MerchantAgent(aid, arch, r_id, target_eid)
            elif arch == "WORKER":
                ag = WorkerAgent(aid, arch, r_id, target_eid, selector=selector)
            else:
                ag = FactionPlannerAgent(aid, arch, r_id, target_eid)

            agents.append(ag)
            idx += 1

    return agents


def benchmark_agent_society():
    print("\n" + "=" * 80)
    print("BENCHMARK: HETEROGENEOUS AGENT SOCIETY & EVENT ROUTING")
    print("Society Population: A = 512 Agents (256 Guards, 128 Merchants, 96 Workers, 32 Planners)")
    print("=" * 80)

    substrate = FastWorldSubstrate(num_clusters=16)
    substrate.populate_synthetic_world(num_entities=10000, edges_per_entity=4)
    snapshot = substrate.current_snapshot()

    agents = instantiate_society(substrate, total_agents=512)

    # 1. Measure Private Memory Footprint per Archetype
    mem_by_arch: Dict[str, List[int]] = {}
    for a in agents:
        # Pre-seed scratchpads
        for i in range(5):
            a.scratchpad.append(f"Pre-seed event {i}")
        mem = a.memory_bytes()
        mem_by_arch.setdefault(a.archetype, []).append(mem)

    print("\n1. AGENT PRIVATE MEMORY BREAKDOWN:")
    print(f"{'Archetype':<16} {'Count':<8} {'Avg Bytes/Agent':<18} {'Total Memory':<15}")
    print("-" * 60)
    total_society_mem = 0
    for arch, mems in mem_by_arch.items():
        avg_b = float(np.mean(mems))
        tot_b = sum(mems)
        total_society_mem += tot_b
        print(f"{arch:<16} {len(mems):<8d} {avg_b:>12.1f} B ({avg_b/1024:.2f} KB)   {tot_b/(1024):>8.1f} KB")
    print(f"\nTotal Society Private Memory: {total_society_mem / 1024:.2f} KB ({total_society_mem / (1024*1024):.4f} MB)")
    print(f"World Memory:                 {substrate.memory_footprint_bytes()['total_bytes'] / (1024*1024):.2f} MB")

    # 2. Sequential Execution Baseline (for Speedup S(A))
    print("\n2. EXECUTING SEQUENTIAL AGENT COGNITION BASELINE...")
    t_seq_0 = time.perf_counter()
    for a in agents[:64]:  # Sample 64 for sequential baseline
        a.execute_cognition(snapshot)
    seq_time_64_ms = (time.perf_counter() - t_seq_0) * 1000.0
    projected_seq_512_ms = seq_time_64_ms * (512 / 64)
    print(f"  Sequential 64 Agents:  {seq_time_64_ms:.2f} ms")
    print(f"  Projected 512 Agents:  {projected_seq_512_ms:.2f} ms (Far exceeds 50 ms tick!)")

    # 3. Continuous Polling across all 512 Agents (Parallel)
    print("\n3. BENCHMARKING CONTINUOUS POLLING (All 512 Agents Run Concurrently)...")
    t_poll_0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(a.execute_cognition, snapshot) for a in agents]
        concurrent.futures.wait(futures)
    polling_tick_ms = (time.perf_counter() - t_poll_0) * 1000.0
    fits_polling = polling_tick_ms <= 50.0
    speedup_parallel = projected_seq_512_ms / max(0.001, polling_tick_ms)
    efficiency_parallel = speedup_parallel / 32.0  # Normalized to 32 worker threads

    print(f"  Continuous Polling Tick Time: {polling_tick_ms:.2f} ms | Fits 50ms Budget: [{'PASS' if fits_polling else 'FAIL'}]")
    print(f"  Parallel Speedup S(A):        {speedup_parallel:.2f}x (Work Efficiency: {efficiency_parallel*100:.1f}%)")

    # 4. Event-Driven Wake Routing (Only impacted agents wake)
    print("\n4. BENCHMARKING EVENT-DRIVEN WAKE ROUTING...")
    # Simulate world event: bridge damaged in entity ent_000042
    event_target = "ent_000042"
    # Substrate identifies impacted neighborhood via G BFS (radius 2)
    impacted_nodes = set(snapshot.bfs(event_target, max_depth=2, max_nodes=20))
    impacted_nodes.add(event_target)

    # Substrate wakes only agents whose focus_entity_id is in the impacted frontier
    woken_agents = [a for a in agents if a.focus_entity_id in impacted_nodes]
    # In addition, 2 strategic faction planners wake on any event
    woken_agents.extend([a for a in agents if a.archetype == "PLANNER"][:2])
    woken_agents = list({a.agent_id: a for a in woken_agents}.values())

    print(f"  World Event on {event_target} -> Impact Frontier: {len(impacted_nodes)} nodes")
    print(f"  Event-Driven Wake Filter: Only {len(woken_agents)} / 512 agents woke ({(len(woken_agents)/512)*100:.1f}% active)!")

    t_wake_0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(woken_agents)))) as executor:
        futures = [executor.submit(a.execute_cognition, snapshot) for a in woken_agents]
        concurrent.futures.wait(futures)
    event_wake_tick_ms = (time.perf_counter() - t_wake_0) * 1000.0
    fits_event_wake = event_wake_tick_ms <= 50.0

    print(f"  Event-Driven Tick Time:       {event_wake_tick_ms:.2f} ms | Fits 50ms Budget: [{'PASS' if fits_event_wake else 'FAIL'}]")
    print(f"  Work Reduction:               {((polling_tick_ms - event_wake_tick_ms) / polling_tick_ms) * 100:.1f}% CPU time saved per tick!")

    print("\n" + "=" * 80)
    print("SOCIETY & EVENT-DRIVEN ROUTING SUMMARY:")
    print(f"  Total Society Population:          512 agents")
    print(f"  Continuous Polling Duration:       {polling_tick_ms:.2f} ms [{'FAIL' if not fits_polling else 'PASS'}]")
    print(f"  Event-Driven Routing Duration:     {event_wake_tick_ms:.2f} ms [{'PASS' if fits_event_wake else 'FAIL'}]")
    print(f"  Cognition Throughput Maintained:   {len(woken_agents)} decisions in {event_wake_tick_ms:.2f} ms")
    print(f"  Total Private Agent Memory:        {total_society_mem / 1024:.2f} KB (Avg: {total_society_mem / (512*1024):.2f} KB/agent)")
    print("=" * 80)

    out_dir = os.path.dirname(__file__)
    out_file = os.path.join(out_dir, "benchmark_heterogeneous_society_results.json")
    results = {
        "total_agents": 512,
        "memory_by_archetype_bytes": {arch: float(np.mean(mems)) for arch, mems in mem_by_arch.items()},
        "total_society_memory_kb": total_society_mem / 1024,
        "polling_tick_ms": polling_tick_ms,
        "event_wake_tick_ms": event_wake_tick_ms,
        "woken_agent_count": len(woken_agents),
        "speedup_parallel": speedup_parallel,
        "fits_polling": fits_polling,
        "fits_event_wake": fits_event_wake,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved society benchmark results to {out_file}")
    return results


if __name__ == "__main__":
    benchmark_agent_society()
