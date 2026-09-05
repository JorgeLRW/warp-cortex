"""Programmatic scaling helpers for shared-manifold benchmark tasks.

These expansions preserve the existing benchmark families while producing many
templated variants via exact substitutions over names, ids, locations, and
field values. This increases breadth without pretending the scaled tasks are
independent human-authored benchmarks.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence, TypeVar

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_benchmarks.benchmark_shared_manifold import (
    SharedManifoldCodingTask,
    SharedManifoldEnergyReuseTask,
    SharedManifoldNecessityTask,
    SharedManifoldRecallTask,
    SharedManifoldScenario,
    SharedManifoldTopologyTask,
    default_coding_tasks,
    default_real_coding_tasks,
    default_real_energy_reuse_tasks,
    default_real_necessity_tasks,
    default_real_recall_tasks,
    default_real_topology_tasks,
    default_scenarios,
    default_topology_tasks,
)

T = TypeVar("T")

PERSONS: Sequence[str] = (
    "Avery",
    "Bianca",
    "Callum",
    "Daria",
    "Elias",
    "Farah",
    "Gavin",
    "Helena",
    "Iris",
    "Jonah",
    "Kira",
    "Luca",
    "Mira",
    "Nolan",
    "Opal",
    "Pavel",
    "Quinn",
    "Rhea",
    "Soren",
    "Talia",
    "Uma",
    "Vera",
    "Willa",
    "Yara",
    "Zane",
)

COLORS: Sequence[str] = (
    "red",
    "blue",
    "green",
    "yellow",
    "teal",
    "amber",
    "violet",
    "silver",
    "orange",
    "black",
    "white",
    "crimson",
)


def _pick(sequence: Sequence[str], variant_index: int, offset: int = 0) -> str:
    position = (variant_index - 1 + offset) % len(sequence)
    return sequence[position]


def _replace_text(text: str, replacements: Dict[str, str]) -> str:
    updated = text
    for old in sorted(replacements, key=len, reverse=True):
        updated = updated.replace(old, replacements[old])
    return updated


def _apply_replacements(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_text(value, replacements)
    if isinstance(value, list):
        return [_apply_replacements(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            _apply_replacements(key, replacements): _apply_replacements(item, replacements)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _apply_replacements(getattr(value, item.name), replacements)
                for item in fields(value)
            }
        )
    return value


def _clone_task(task: T, replacements: Dict[str, str], suffix: str) -> T:
    clone = _apply_replacements(task, replacements)
    return replace(clone, name=f"{task.name}__{suffix}")


def _expand_tasks(
    base_tasks: Iterable[T],
    scale_multiplier: int,
    builder: Callable[[T, int], Dict[str, str]],
) -> List[T]:
    task_list = list(base_tasks)
    if scale_multiplier <= 1:
        return task_list

    expanded = list(task_list)
    for variant_index in range(1, scale_multiplier):
        for task in task_list:
            expanded.append(
                _clone_task(
                    task,
                    builder(task, variant_index),
                    suffix=f"s{variant_index:03d}",
                )
            )
    return expanded


def _jenny_family_values(variant_index: int) -> Dict[str, str]:
    return {
        "target_name": _pick(PERSONS, variant_index, 0),
        "target_color": _pick(COLORS, variant_index, 0),
        "locker": f"locker {14 + variant_index}",
        "distractor_one": _pick(PERSONS, variant_index, 3),
        "distractor_two": _pick(PERSONS, variant_index, 6),
        "distractor_three": _pick(PERSONS, variant_index, 9),
        "distractor_color_one": _pick(COLORS, variant_index, 4),
        "distractor_color_two": _pick(COLORS, variant_index, 7),
        "distractor_color_three": _pick(COLORS, variant_index, 10),
    }


def _build_real_coding_replacements(task: SharedManifoldCodingTask, variant_index: int) -> Dict[str, str]:
    if task.name == "retry_replay_token_simple":
        token_id = 100 + variant_index
        return {
            "Replay-Safety-Token": f"Replay-Safety-Token-{token_id}",
            "ch_123": f"ch_{token_id}",
            "rp-42": f"rp-{token_id}",
        }
    if task.name == "rotation_key_field":
        token_id = 200 + variant_index
        return {
            "rotation_key_version": f"rotation_key_version_{token_id}",
            "k2": f"k{token_id}",
        }
    if task.name == "session_trace_field":
        token_id = 300 + variant_index
        return {
            "session_trace_id": f"session_trace_id_{token_id}",
            "sess-9": f"sess-{token_id}",
        }
    raise ValueError(f"Unsupported real coding task for scaling: {task.name}")


def _build_scenario_replacements(task: SharedManifoldScenario, variant_index: int) -> Dict[str, str]:
    case_id = 50 + variant_index
    if task.name == "payment_retry":
        return {
            "Use idempotency keys on payment retries to avoid duplicate captures.": f"Use idempotency keys on payment retries to avoid duplicate captures in checkout lane P-{case_id}.",
            "Emit retry telemetry so duplicate payment attempts can be debugged quickly.": f"Emit retry telemetry so duplicate payment attempts in checkout lane P-{case_id} can be debugged quickly.",
            "Implement payment retry safety in checkout.": f"Implement payment retry safety in checkout lane P-{case_id}.",
            "Need to stop duplicate capture when a flaky network retries payment.": f"Need to stop duplicate capture when a flaky network retries payment in checkout lane P-{case_id}.",
        }
    if task.name == "schema_backfill":
        return {
            "Backfills must keep old index names unique until cutover completes.": f"Backfills must keep old index names unique until cutover C-{case_id} completes.",
            "Dual writes should remain enabled during the migration backfill window.": f"Dual writes should remain enabled during the migration backfill window B-{case_id}.",
            "Review the migration rollout for index backfill safety.": f"Review the migration rollout for index backfill safety in wave B-{case_id}.",
            "The rollout plan changes an index while the backfill is still running.": f"The rollout plan changes an index while the backfill for wave B-{case_id} is still running.",
        }
    if task.name == "token_rotation":
        return {
            "During token rotation, keep both old and new keys valid for one deployment window.": f"During token rotation, keep both old and new keys valid for deployment window W-{case_id}.",
            "Log the active key version with every auth failure for rollback triage.": f"Log the active key version with every auth failure during rollback wave W-{case_id} for triage.",
            "Prepare the production token rotation runbook.": f"Prepare the production token rotation runbook for wave W-{case_id}.",
            "Auth failures increased after rotating credentials during deployment.": f"Auth failures increased after rotating credentials during deployment wave W-{case_id}.",
        }
    raise ValueError(f"Unsupported scenario for scaling: {task.name}")


def _build_deterministic_coding_replacements(task: SharedManifoldCodingTask, variant_index: int) -> Dict[str, str]:
    case_id = 60 + variant_index
    if task.name == "payment_retry_repair":
        return {
            "Repair the checkout retry helper so retries are safe and observable.": f"Repair the checkout retry helper for lane P-{case_id} so retries are safe and observable.",
            "A flaky checkout path is issuing duplicate captures after retries.": f"A flaky checkout lane P-{case_id} is issuing duplicate captures after retries.",
            "ch_123": f"ch_{case_id}",
            "idem-42": f"idem-{case_id}",
            "checkout": f"checkout-{case_id}",
        }
    if task.name == "schema_backfill_repair":
        return {
            "Backfills must keep old index names unique until cutover completes.": f"Backfills must keep old index names unique until cutover C-{case_id} completes.",
            "Dual writes should remain enabled during the migration backfill window.": f"Dual writes should remain enabled during migration backfill window B-{case_id}.",
            "Repair the migration cutover helper so the backfill window stays safe.": f"Repair the migration cutover helper so backfill window B-{case_id} stays safe.",
            "orders_v1": f"orders_{case_id}_v1",
            "orders_v2": f"orders_{case_id}_v2",
        }
    if task.name == "token_rotation_repair":
        return {
            "During token rotation, keep both old and new keys valid for one deployment window.": f"During token rotation, keep both old and new keys valid for deployment window W-{case_id}.",
            "Repair the token rotation helpers so the rollout keeps compatibility and debuggability.": f"Repair the token rotation helpers so rollout wave W-{case_id} keeps compatibility and debuggability.",
            "k1": f"k{case_id}a",
            "k2": f"k{case_id}b",
        }
    raise ValueError(f"Unsupported deterministic coding task for scaling: {task.name}")


def _build_deterministic_topology_replacements(task: SharedManifoldTopologyTask, variant_index: int) -> Dict[str, str]:
    if task.name == "payment_region_isolation":
        ticket_id = 17 + variant_index
        return {
            "PX-17": f"PX-{ticket_id}",
            "px17": f"px{ticket_id}",
        }
    if task.name == "bridge_recall_vs_flat_leakage":
        packet_id = 9 + variant_index
        return {
            "PX-9": f"PX-{packet_id}",
        }
    raise ValueError(f"Unsupported deterministic topology task for scaling: {task.name}")


def _build_real_recall_replacements(task: SharedManifoldRecallTask, variant_index: int) -> Dict[str, str]:
    jenny_values = _jenny_family_values(variant_index)
    if task.name in {"jenny_boots_red", "jenny_boots_locker"}:
        return {
            "Jenny": jenny_values["target_name"],
            "red": jenny_values["target_color"],
            "locker 14": jenny_values["locker"],
        }
    if task.name == "cedar_compass_chain":
        compass_id = 400 + variant_index
        return {
            "Eli": _pick(PERSONS, variant_index, 2),
            "Nora": _pick(PERSONS, variant_index, 5),
            "bronze compass": f"bronze compass C-{compass_id}",
            "cedar drawer": f"cedar drawer {compass_id}",
        }
    if task.name == "silver_keycard_chain":
        keycard_id = 500 + variant_index
        return {
            "Priya": _pick(PERSONS, variant_index, 1),
            "Omar": _pick(PERSONS, variant_index, 8),
            "silver keycard": f"silver keycard S-{keycard_id}",
            "locker 12": f"locker {112 + variant_index}",
        }
    if task.name == "color_distractor_mix":
        return {
            "Jenny": jenny_values["target_name"],
            "red": jenny_values["target_color"],
            "Marta": jenny_values["distractor_one"],
            "Theo": jenny_values["distractor_two"],
            "Nina": jenny_values["distractor_three"],
            "green": jenny_values["distractor_color_one"],
            "blue": jenny_values["distractor_color_two"],
            "yellow": jenny_values["distractor_color_three"],
        }
    raise ValueError(f"Unsupported real recall task for scaling: {task.name}")


def _build_real_necessity_replacements(task: SharedManifoldNecessityTask, variant_index: int) -> Dict[str, str]:
    if task.name == "vx17_badge_locker":
        ticket_id = 17 + variant_index
        return {
            "VX-17": f"VX-{ticket_id}",
            "teal": _pick(COLORS, variant_index, 4),
            "locker 42": f"locker {42 + variant_index}",
        }
    if task.name == "rq91_parking_chain":
        route_id = 91 + variant_index
        distractor_id = 73 + variant_index
        return {
            "RQ-91": f"RQ-{route_id}",
            "RQ-73": f"RQ-{distractor_id}",
            "Mina": _pick(PERSONS, variant_index, 0),
            "Jules": _pick(PERSONS, variant_index, 7),
            "Oren": _pick(PERSONS, variant_index, 12),
            "bay 6": f"bay {6 + variant_index}",
            "bay 3": f"bay {103 + variant_index}",
        }
    if task.name == "cedar88_drawer":
        specimen_id = 88 + variant_index
        distractor_id = 20 + variant_index
        return {
            "Cedar-88": f"Cedar-{specimen_id}",
            "Maple-20": f"Maple-{distractor_id}",
            "Nora": _pick(PERSONS, variant_index, 4),
            "drawer cedar-3": f"drawer cedar-{3 + variant_index}",
            "tray 9": f"tray {9 + variant_index}",
        }
    raise ValueError(f"Unsupported real necessity task for scaling: {task.name}")


def _build_real_topology_replacements(task: SharedManifoldTopologyTask, variant_index: int) -> Dict[str, str]:
    if task.name == "real_payment_retry_fields":
        ticket_id = 17 + variant_index
        upper_id = f"PX-{ticket_id}"
        lower_id = f"px{ticket_id}"
        return {
            "PX-17": upper_id,
            "px17": lower_id,
            "X-Payment-Retry-Key": f"X-Payment-Retry-Key-{ticket_id}",
            "replay_token_px17": f"replay_token_{lower_id}",
            "X-Bridge-Retry-Key": f"X-Bridge-Retry-Key-{ticket_id}",
            "bridge_manifest_px17": f"bridge_manifest_{lower_id}",
        }
    if task.name == "real_bridge_route_chain":
        packet_id = 9 + variant_index
        upper_id = f"PX-{packet_id}"
        lower_id = f"px{packet_id}"
        return {
            "PX-9": upper_id,
            "px9": lower_id,
            "beta-seam": f"beta-seam-{packet_id}",
            "cedar-checkpoint": f"cedar-checkpoint-{packet_id}",
            "cold-archive": f"cold-archive-{packet_id}",
        }
    raise ValueError(f"Unsupported real topology task for scaling: {task.name}")


def _build_real_energy_reuse_replacements(task: SharedManifoldEnergyReuseTask, variant_index: int) -> Dict[str, str]:
    case_id = 600 + variant_index
    if task.name == "gateway_route_reuse":
        return {
            "build_gateway_event": f"build_gateway_event_{case_id}",
            "route_gateway_field": f"route_gateway_field_{case_id}",
            "checkpoint_gateway_field": f"checkpoint_gateway_field_{case_id}",
            "metadata_gateway_field": f"metadata_gateway_field_{case_id}",
            "retry_gateway_field": f"retry_gateway_field_{case_id}",
        }
    if task.name == "handoff_trail_reuse":
        return {
            "build_handoff_event": f"build_handoff_event_{case_id}",
            "trail_handoff_field": f"trail_handoff_field_{case_id}",
            "marker_handoff_field": f"marker_handoff_field_{case_id}",
            "audit_handoff_field": f"audit_handoff_field_{case_id}",
            "token_handoff_field": f"token_handoff_field_{case_id}",
        }
    if task.name == "ledger_seal_reuse":
        return {
            "build_ledger_event": f"build_ledger_event_{case_id}",
            "seal_ledger_field": f"seal_ledger_field_{case_id}",
            "checkpoint_ledger_field": f"checkpoint_ledger_field_{case_id}",
            "audit_ledger_field": f"audit_ledger_field_{case_id}",
            "token_ledger_field": f"token_ledger_field_{case_id}",
        }
    raise ValueError(f"Unsupported real energy reuse task for scaling: {task.name}")


def build_scaled_real_coding_tasks(scale_multiplier: int) -> List[SharedManifoldCodingTask]:
    return _expand_tasks(default_real_coding_tasks(), scale_multiplier, _build_real_coding_replacements)


def build_scaled_scenarios(scale_multiplier: int) -> List[SharedManifoldScenario]:
    return _expand_tasks(default_scenarios(), scale_multiplier, _build_scenario_replacements)


def build_scaled_deterministic_coding_tasks(scale_multiplier: int) -> List[SharedManifoldCodingTask]:
    return _expand_tasks(default_coding_tasks(), scale_multiplier, _build_deterministic_coding_replacements)


def build_scaled_deterministic_topology_tasks(scale_multiplier: int) -> List[SharedManifoldTopologyTask]:
    return _expand_tasks(default_topology_tasks(), scale_multiplier, _build_deterministic_topology_replacements)


def build_scaled_real_recall_tasks(scale_multiplier: int) -> List[SharedManifoldRecallTask]:
    return _expand_tasks(default_real_recall_tasks(), scale_multiplier, _build_real_recall_replacements)


def build_scaled_real_necessity_tasks(scale_multiplier: int) -> List[SharedManifoldNecessityTask]:
    return _expand_tasks(default_real_necessity_tasks(), scale_multiplier, _build_real_necessity_replacements)


def build_scaled_real_topology_tasks(scale_multiplier: int) -> List[SharedManifoldTopologyTask]:
    return _expand_tasks(default_real_topology_tasks(), scale_multiplier, _build_real_topology_replacements)


def build_scaled_real_energy_reuse_tasks(scale_multiplier: int) -> List[SharedManifoldEnergyReuseTask]:
    return _expand_tasks(default_real_energy_reuse_tasks(), scale_multiplier, _build_real_energy_reuse_replacements)