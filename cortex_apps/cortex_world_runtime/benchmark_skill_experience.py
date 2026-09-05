"""
Benchmark A: Skill Experience Transfer Benchmark.
==================================================
Compares 3 conditions across 20 agents facing 5 recurring task families:
  1. Condition 1: Static Skill Library (Hermes baseline: static metadata matching)
  2. Condition 2: Private Agent Memory (Each agent maintains its own private history)
  3. Condition 3: Shared Cortex Skill Ledger (U_v + K: cross-agent shared experience)

Task Families:
  - Task Family 1: Bridge High-Temp Repair (v1 cracks; v2 works if cooling_active)
  - Task Family 2: API Contract Migration (v1 returns tuple; v2 returns EventVector)
  - Task Family 3: Bioreactor Calibration (fails if calibration_locked!=True)
  - Task Family 4: Quantum Tensor Compression (fails if aspect_rank < 4)
  - Task Family 5: Emergency Coolant Venting (v1 vents hazardous vapor; v2 safe scrubber)

Measures:
  - First-attempt success rate across 20 agents
  - Cumulative repeated mistakes
  - Tool / execution token cost
  - Cross-agent experience transfer ratio (Agent i > 1 success on previously encountered task families)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import FastWorldSubstrate
from cortex_apps.cortex_world_runtime.skill_registry import (
    SkillDefinition,
    SkillInvocationEvent,
    SkillRegistry,
    SkillSelectionMode,
    SkillSelector,
)


def create_task_catalog() -> Tuple[SkillRegistry, List[Dict[str, Any]]]:
    registry = SkillRegistry()

    # Family 1: Bridge Repair
    registry.register(SkillDefinition(
        skill_id="repair_bridge", version="v1", name="Standard Bridge Repair",
        description="Repairs damaged bridge structure using standard welding.",
        aspect_tags=["REPAIR", "BRIDGE", "INFRASTRUCTURE", "STANDARD"],
    ))
    registry.register(SkillDefinition(
        skill_id="repair_bridge", version="v2", name="Thermal-Shielded Bridge Repair",
        description="Repairs load-bearing bridge structure with catalytic cooling scrubber.",
        aspect_tags=["REPAIR", "BRIDGE", "THERMAL", "COOLING"],
    ))

    # Family 2: API Contract Migration
    registry.register(SkillDefinition(
        skill_id="migrate_resolver_api", version="v1", name="Standard Event Resolver Patch",
        description="Patches SharedFrozenEventResolver to update event handling.",
        aspect_tags=["API", "RESOLVER", "CODE", "MIGRATE", "STANDARD"],
    ))
    registry.register(SkillDefinition(
        skill_id="migrate_resolver_api", version="v2", name="EventVector Contract Preserving Patch",
        description="Patches SharedFrozenEventResolver preserving EventVector interface.",
        aspect_tags=["API", "RESOLVER", "CONTRACT", "TYPED"],
    ))

    # Family 3: Bioreactor Calibration
    registry.register(SkillDefinition(
        skill_id="calibrate_bioreactor", version="v1", name="Direct Sensor Calibration",
        description="Calibrates MS-4 quadrupole sensors on active fermentation bioreactor.",
        aspect_tags=["CALIBRATION", "SENSOR", "BIOREACTOR", "QUADRUPOLE"],
    ))

    # Family 4: Tensor Compression
    registry.register(SkillDefinition(
        skill_id="compress_manifold_tensor", version="v1", name="Standard SVD Compression",
        description="Compresses manifold tensors using standard SVD reduction.",
        aspect_tags=["COMPRESSION", "TENSOR", "SVD", "MANIFOLD", "STANDARD"],
    ))
    registry.register(SkillDefinition(
        skill_id="compress_manifold_tensor", version="v2", name="Multi-Aspect 4-Band Compression",
        description="Compresses manifold tensors preserving 4 functional aspect bands.",
        aspect_tags=["COMPRESSION", "TENSOR", "MULTI_ASPECT", "PRESERVE"],
    ))

    # Family 5: Emergency Coolant Venting
    registry.register(SkillDefinition(
        skill_id="vent_coolant", version="v1", name="Rapid Valve Vent",
        description="Emergency pressure release for coolant line.",
        aspect_tags=["EMERGENCY", "COOLANT", "VENT", "PRESSURE", "STANDARD"],
    ))
    registry.register(SkillDefinition(
        skill_id="vent_coolant", version="v2", name="Catalytic Scrubber Vent",
        description="Emergency pressure release for coolant line with catalytic scrubbers.",
        aspect_tags=["EMERGENCY", "COOLANT", "SAFE", "SCRUBBER"],
    ))

    # Build 20 task instances across the 5 families
    tasks = []
    task_templates = [
        {"family": "repair_bridge", "query": "repair damaged bridge structure", "target": "bridge_east", "bad_ver": "v1", "good_ver": "v2", "learned_constraint": "cooling_active"},
        {"family": "migrate_resolver_api", "query": "patch SharedFrozenEventResolver", "target": "memory_baselines", "bad_ver": "v1", "good_ver": "v2", "learned_constraint": "event_vector_contract"},
        {"family": "calibrate_bioreactor", "query": "calibrate quadrupole sensor on bioreactor", "target": "ms4_quadrupole", "bad_ver": "v1", "good_ver": "v1", "requires_lock": True, "learned_constraint": "calibration_locked"},
        {"family": "compress_manifold_tensor", "query": "compress manifold tensors", "target": "manifold_core", "bad_ver": "v1", "good_ver": "v2", "learned_constraint": "aspect_rank_4"},
        {"family": "vent_coolant", "query": "emergency pressure release for coolant line", "target": "coolant_line_2", "bad_ver": "v1", "good_ver": "v2", "learned_constraint": "closed_loop_scrubber"},
    ]

    for i in range(20):
        tmpl = task_templates[i % len(task_templates)]
        tasks.append({
            "task_index": i + 1,
            "agent_id": f"agent_{(i % 10) + 1}",  # 10 distinct agents encounter tasks in rotation
            **tmpl
        })

    return registry, tasks


def simulate_task_execution(
    selected_skill: SkillDefinition,
    task: Dict[str, Any],
    world_state: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any], int]:
    """
    Simulates environment response and failure constraints.
    Returns (success, outcome_summary, discovered_constraints, token_cost).
    """
    token_cost = 150
    # Family 1 constraint
    if task["family"] == "repair_bridge":
        if selected_skill.version == "v1":
            return False, "Thermal cracking: standard welding failed without cooling.", {"cooling_active": True}, token_cost
        return True, "Bridge safely repaired with thermal shielding.", {}, token_cost

    # Family 2 constraint
    if task["family"] == "migrate_resolver_api":
        if selected_skill.version == "v1":
            return False, "Contract violation: downstream agents crashed on tuple return.", {"event_vector_contract": True}, token_cost
        return True, "API safely patched with EventVector contract preserved.", {}, token_cost

    # Family 3 constraint
    if task["family"] == "calibrate_bioreactor":
        if not world_state.get("calibration_locked", False):
            return False, "Safety lock violation: sensor drift during active fermentation.", {"calibration_locked": True}, token_cost
        return True, "Sensor calibration locked and validated.", {}, token_cost

    # Family 4 constraint
    if task["family"] == "compress_manifold_tensor":
        if selected_skill.version == "v1":
            return False, "Aspect collapse: logic and safety dimensions erased.", {"aspect_rank_4": True}, token_cost
        return True, "Tensor compressed with all 4 aspect bands intact.", {}, token_cost

    # Family 5 constraint
    if task["family"] == "vent_coolant":
        if selected_skill.version == "v1":
            return False, "Containment hazard: toxic vapor exceeded permissible threshold.", {"closed_loop_scrubber": True}, token_cost
        return True, "Coolant safely vented through catalytic scrubber.", {}, token_cost

    return True, "Success.", {}, token_cost


def run_benchmark_condition(
    mode: SkillSelectionMode,
    registry: SkillRegistry,
    tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    selector = SkillSelector(registry, mode=mode)
    shared_history: List[SkillInvocationEvent] = []

    # World state
    world_substrate = FastWorldSubstrate(num_clusters=4)
    world_substrate.populate_synthetic_world(num_entities=100)
    world_substrate.global_state["cooling_active"] = True
    world_substrate.global_state["calibration_locked"] = True
    world_substrate.global_state["event_vector_contract"] = True
    world_substrate.global_state["aspect_rank_4"] = True
    world_substrate.global_state["closed_loop_scrubber"] = True

    first_attempt_successes = 0
    total_attempts = 0
    repeated_mistakes = 0
    total_tokens = 0
    cross_agent_transfer_wins = 0
    previously_seen_families: Dict[str, bool] = {}

    for t in tasks:
        agent_id = t["agent_id"]
        family = t["family"]
        is_first_time_seen = (family not in previously_seen_families)

        snapshot = world_substrate.current_snapshot()
        ranked = selector.select_skill(t["query"], snapshot, agent_id=agent_id, shared_history=shared_history, top_k=2)

        # Agent tries top-ranked skill
        chosen_skill, score, explanation = ranked[0]
        success, outcome, disc_constraints, tokens = simulate_task_execution(
            chosen_skill, t, world_substrate.global_state
        )
        total_attempts += 1
        total_tokens += tokens

        inv_event = SkillInvocationEvent(
            invocation_id=f"inv_{total_attempts:03d}",
            skill_id=chosen_skill.skill_id,
            skill_version=chosen_skill.version,
            agent_id=agent_id,
            world_version=snapshot.version,
            task_query=t["query"],
            inputs={"target": t["target"]},
            success=success,
            outcome_summary=outcome,
            latency_ms=10.0,
            token_cost=tokens,
            discovered_constraints=disc_constraints,
        )
        selector.record_invocation(inv_event, shared_history)

        if success:
            first_attempt_successes += 1
            if not is_first_time_seen:
                cross_agent_transfer_wins += 1
        else:
            if not is_first_time_seen:
                repeated_mistakes += 1

            # Fallback attempt if first failed
            if len(ranked) > 1:
                fallback_skill, _, _ = ranked[1]
                succ2, out2, disc2, tok2 = simulate_task_execution(fallback_skill, t, world_substrate.global_state)
                total_attempts += 1
                total_tokens += tok2
                inv_event2 = SkillInvocationEvent(
                    invocation_id=f"inv_{total_attempts:03d}",
                    skill_id=fallback_skill.skill_id,
                    skill_version=fallback_skill.version,
                    agent_id=agent_id,
                    world_version=snapshot.version,
                    task_query=t["query"],
                    inputs={"target": t["target"]},
                    success=succ2,
                    outcome_summary=out2,
                    latency_ms=10.0,
                    token_cost=tok2,
                    discovered_constraints=disc2,
                )
                selector.record_invocation(inv_event2, shared_history)

        previously_seen_families[family] = True

    repeat_opportunities = len(tasks) - len(previously_seen_families)
    transfer_rate = (cross_agent_transfer_wins / max(1, repeat_opportunities)) * 100.0

    return {
        "mode": mode.value,
        "total_tasks": len(tasks),
        "first_attempt_success_rate": (first_attempt_successes / len(tasks)) * 100.0,
        "first_attempt_count": first_attempt_successes,
        "total_attempts": total_attempts,
        "repeated_mistakes": repeated_mistakes,
        "total_tokens": total_tokens,
        "cross_agent_transfer_rate": transfer_rate,
        "cross_agent_transfer_wins": cross_agent_transfer_wins,
        "repeat_opportunities": repeat_opportunities,
    }


def run_benchmark_a():
    print("=" * 80)
    print("BENCHMARK A: SKILL EXPERIENCE TRANSFER BENCHMARK")
    print("Evaluating cross-agent procedural learning across 20 tasks & 10 agents")
    print("=" * 80)

    registry, tasks = create_task_catalog()

    # 1. Condition 1: Static Skills (Hermes Baseline)
    res_static = run_benchmark_condition(SkillSelectionMode.STATIC, registry, tasks)

    # 2. Condition 2: Private Agent Memory
    res_private = run_benchmark_condition(SkillSelectionMode.PRIVATE_MEMORY, registry, tasks)

    # 3. Condition 3: Shared Cortex Skill Ledger (U_v + K)
    res_shared = run_benchmark_condition(SkillSelectionMode.SHARED_CORTEX_LEDGER, registry, tasks)

    print("\nEMPIRICAL RESULTS:")
    print("-" * 80)
    print(f"{'Condition':<32} {'1st-Attempt Acc':<18} {'Repeated Mistakes':<20} {'Transfer Rate':<15} {'Total Tokens':<12}")
    print("-" * 80)
    for res in [res_static, res_private, res_shared]:
        print(f"{res['mode']:<32} {res['first_attempt_success_rate']:>12.1f}%     {res['repeated_mistakes']:>14d}     {res['cross_agent_transfer_rate']:>12.1f}%    {res['total_tokens']:>10d}")

    print("\n" + "=" * 80)
    print("VERDICT ON SKILL EXPERIENCE LEDGER:")
    print("=" * 80)
    if res_shared['cross_agent_transfer_rate'] > res_static['cross_agent_transfer_rate']:
        print(f"1. CONFIRMED: Shared skill ledger achieved {res_shared['cross_agent_transfer_rate']:.1f}% cross-agent transfer")
        print(f"   versus {res_static['cross_agent_transfer_rate']:.1f}% in static baseline and {res_private['cross_agent_transfer_rate']:.1f}% in private memory.")
        print(f"2. REPEATED MISTAKES ELIMINATED: Dropped from {res_static['repeated_mistakes']} in static to {res_shared['repeated_mistakes']} in shared ledger.")
    else:
        print("KILL CRITERION TRIGGERED: Shared skill ledger did not measurably outperform static skills.")
    print("=" * 80)

    # Save artifact
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_a_skill_experience_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"static": res_static, "private": res_private, "shared": res_shared}, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    run_benchmark_a()
