"""
Unit Tests for Cortex World Runtime: Skills, Fast Engine & 3 Clocks.
=====================================================================
"""

import pytest
import torch

from cortex_apps.cortex_world_runtime.fast_world_substrate import (
    FastWorldSubstrate,
    WorldSnapshot,
)
from cortex_apps.cortex_world_runtime.skill_registry import (
    SkillDefinition,
    SkillInvocationEvent,
    SkillRegistry,
    SkillSelectionMode,
    SkillSelector,
)


def test_skill_registry_and_selection():
    registry = SkillRegistry()

    # Skill 1: Bridge Repair v1 (flawed: missing cooling constraint)
    s1_v1 = SkillDefinition(
        skill_id="repair_bridge",
        version="v1",
        name="Bridge Repair Basic",
        description="Repairs damaged bridge structure using standard steel beams.",
        aspect_tags=["REPAIR", "STRUCTURE", "BRIDGE"],
        prerequisites={"has_tools": True},
    )
    # Skill 1: Bridge Repair v2 (fixed: requires cooling lock)
    s1_v2 = SkillDefinition(
        skill_id="repair_bridge",
        version="v2",
        name="Bridge Repair High-Temp",
        description="Repairs high-temperature load-bearing bridge structures.",
        aspect_tags=["REPAIR", "STRUCTURE", "BRIDGE", "HIGH_TEMP"],
        prerequisites={"has_tools": True},
    )
    registry.register(s1_v1)
    registry.register(s1_v2)

    substrate = FastWorldSubstrate(num_clusters=4)
    substrate.global_state["has_tools"] = True
    substrate.global_state["cooling_active"] = True
    snapshot = substrate.current_snapshot()

    # In STATIC mode: both score similarly based on description
    selector_static = SkillSelector(registry, mode=SkillSelectionMode.STATIC)
    ranked_static = selector_static.select_skill("repair the damaged bridge", snapshot, agent_id="agent_1")
    assert len(ranked_static) > 0

    # In SHARED_CORTEX_LEDGER mode:
    shared_ledger: list = []
    selector_shared = SkillSelector(registry, mode=SkillSelectionMode.SHARED_CORTEX_LEDGER)

    # Simulate Agent 1 trying v1 and failing due to thermal failure
    ev_fail = SkillInvocationEvent(
        invocation_id="inv_001",
        skill_id="repair_bridge",
        skill_version="v1",
        agent_id="agent_1",
        world_version=1,
        task_query="repair the damaged bridge",
        inputs={"target": "bridge_east"},
        success=False,
        outcome_summary="Structural failure: thermal cracking occurred",
        latency_ms=12.0,
        token_cost=50,
        error_type="ThermalCrackingError",
        discovered_constraints={"cooling_active": True},
    )
    selector_shared.record_invocation(ev_fail, shared_ledger)

    # Now Agent 2 queries the shared selector for the same problem!
    ranked_agent2 = selector_shared.select_skill(
        "repair the damaged bridge", snapshot, agent_id="agent_2", shared_history=shared_ledger
    )

    # Assert that Agent 2 automatically ranks v2 or learned constraint satisfaction higher than v1!
    top_skill, top_score, explanation = ranked_agent2[0]
    assert "Learned constraint satisfied" in explanation or top_skill.version == "v2"
    assert top_score > 0


def test_fast_world_substrate_clocks():
    substrate = FastWorldSubstrate(num_clusters=4)
    substrate.populate_synthetic_world(num_entities=1000, edges_per_entity=3)

    # 1. Clock 1 Frame Tick (< 1 ms target)
    deltas = [
        ("ent_000010", {"status": "DAMAGED", "health": 40}),
        ("ent_000011", {"status": "ALERT", "resource_units": 100}),
    ]
    dur1 = substrate.clock1_tick(deltas)
    assert dur1 < 5.0, f"Clock 1 tick took {dur1:.3f} ms, expected < 5 ms"
    assert substrate.entities["ent_000010"].state["status"] == "DAMAGED"

    # 2. Clock 2 AI Cognition Tick (< 5 ms target)
    q_vec = torch.randn(64)
    q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=0)
    sem_matches, nbrs, dur2 = substrate.clock2_tick(q_vec, focus_entity_id="ent_000010", top_k=5)
    assert dur2 < 15.0, f"Clock 2 tick took {dur2:.3f} ms"
    assert len(sem_matches) == 5
    assert len(nbrs) > 0

    # 3. Clock 3 Intent Commit
    snapshot = substrate.current_snapshot()
    ok, new_v, msg = substrate.clock3_commit_intent(
        agent_id="agent_99",
        expected_version=snapshot.version,
        intent_deltas=[("ent_000010", {"status": "REPAIRED", "health": 100})],
    )
    assert ok is True
    assert substrate.entities["ent_000010"].state["status"] == "REPAIRED"

    # 4. Obsidian Inspector Card
    card = snapshot.inspect("ent_000010")
    assert "# Entity Card: ent_000010" in card
    assert "Operational Status" in card
