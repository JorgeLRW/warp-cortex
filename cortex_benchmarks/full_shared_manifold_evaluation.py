"""Consolidated shared-manifold evaluation runner.

Runs the deterministic probe slices once and the real local-model slices multiple
times, then writes both a JSON artifact and a compact Markdown summary. The goal
is breadth and honesty: preserve default task inventories, report repeat
stability, and state the evaluation limits explicitly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Callable, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_benchmarks.benchmark_shared_manifold import (  # noqa: E402
    compare_coding_slice,
    compare_pipeline,
    compare_real_coding_slice,
    compare_real_energy_reuse_slice,
    compare_real_handoff_slice,
    compare_real_necessity_slice,
    compare_real_recall_handoff_slice,
    compare_real_topology_slice,
    compare_topology_slice,
    default_coding_tasks,
    default_real_coding_tasks,
    default_real_energy_reuse_tasks,
    default_real_necessity_tasks,
    default_real_recall_tasks,
    default_real_topology_tasks,
    default_scenarios,
    default_topology_tasks,
)
from cortex_benchmarks.scaled_shared_manifold_tasks import (  # noqa: E402
    build_scaled_deterministic_coding_tasks,
    build_scaled_deterministic_topology_tasks,
    build_scaled_real_coding_tasks,
    build_scaled_real_energy_reuse_tasks,
    build_scaled_real_necessity_tasks,
    build_scaled_real_recall_tasks,
    build_scaled_real_topology_tasks,
    build_scaled_scenarios,
)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _compact_probe_compare(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": {
            "aggregate": _round_floats(report["enabled"]["aggregate"]),
            "scenarios": [
                {
                    "name": item["name"],
                    "prompt_hit": item["prompt_hit"],
                    "refresh_hit": item["refresh_hit"],
                    "matched_terms": list(item["matched_terms"]),
                }
                for item in report["enabled"]["scenarios"]
            ],
        },
        "disabled": {
            "aggregate": _round_floats(report["disabled"]["aggregate"]),
            "scenarios": [
                {
                    "name": item["name"],
                    "prompt_hit": item["prompt_hit"],
                    "refresh_hit": item["refresh_hit"],
                    "matched_terms": list(item["matched_terms"]),
                }
                for item in report["disabled"]["scenarios"]
            ],
        },
    }


def _compact_coding_compare(report: Dict[str, Any]) -> Dict[str, Any]:
    def compact_side(side: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "aggregate": _round_floats(side["aggregate"]),
            "tasks": [
                {
                    "name": item["name"],
                    "prompt_hit": item["prompt_hit"],
                    "passed": item["passed"],
                    "matched_terms": list(item["matched_terms"]),
                }
                for item in side["tasks"]
            ],
        }

    return {
        "enabled": compact_side(report["enabled"]),
        "disabled": compact_side(report["disabled"]),
    }


def _compact_topology_probe(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "topology_expected_recall": _round_floats(item["topology"]["expected_recall"]),
                "flat_expected_recall": _round_floats(item["flat"]["expected_recall"]),
                "topology_bridge_recall": _round_floats(item["topology"]["bridge_recall"]),
                "flat_bridge_recall": _round_floats(item["flat"]["bridge_recall"]),
                "topology_leakage_rate": _round_floats(item["topology"]["leakage_rate"]),
                "flat_leakage_rate": _round_floats(item["flat"]["leakage_rate"]),
            }
            for item in report["tasks"]
        ],
    }


def _compact_real_coding(report: Dict[str, Any]) -> Dict[str, Any]:
    def compact_side(side: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "aggregate": _round_floats(side["aggregate"]),
            "tasks": [
                {
                    "name": item["name"],
                    "prompt_hit": item["prompt_hit"],
                    "passed": item["passed"],
                    "matched_terms": list(item["matched_terms"]),
                }
                for item in side["tasks"]
            ],
        }

    return {
        "model_id": report.get("enabled", {}).get("model_id"),
        "device": report.get("enabled", {}).get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "enabled": compact_side(report["enabled"]),
        "disabled": compact_side(report["disabled"]),
    }


def _compact_real_energy_reuse(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "device": report.get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "expected_task_id": item["expected_task_id"],
                "followup_selected_task_id": item["followup_selected_task_id"],
                "followup_target_hit": item["followup_target_hit"],
                "followup_patch_hit": item["followup_patch_hit"],
            }
            for item in report["tasks"]
        ],
    }


def _compact_real_handoff(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "device": report.get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "context_match": item["context_match"],
                "output_match": item["output_match"],
                "fresh_passed": item["fresh_reader"]["passed"],
                "loaded_passed": item["loaded_reader"]["passed"],
            }
            for item in report["tasks"]
        ],
    }


def _compact_real_recall_handoff(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "device": report.get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "context_match": item["context_match"],
                "writer_passed": item["writer"]["passed"],
                "fresh_passed": item["fresh_reader"]["passed"],
                "loaded_passed": item["loaded_reader"]["passed"],
                "loaded_fields": dict(item["loaded_reader"]["parsed_fields"]),
            }
            for item in report["tasks"]
        ],
    }


def _compact_real_necessity(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "device": report.get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "isolated_passed": item["isolated_reader"]["passed"],
                "manifold_passed": item["manifold_reader"]["passed"],
                "oracle_passed": item["oracle_reader"]["passed"],
                "manifold_fields": dict(item["manifold_reader"]["parsed_fields"]),
            }
            for item in report["tasks"]
        ],
    }


def _compact_real_topology(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": report.get("model_id"),
        "device": report.get("device"),
        "energy_feedback_enabled": bool(report.get("energy_feedback_enabled", False)),
        "aggregate": _round_floats(report["aggregate"]),
        "tasks": [
            {
                "name": item["name"],
                "component_count": item["shared_manifold_stats"]["component_count"],
                "active_region_size": item["topology_retrieval"]["active_region_size"],
                "topology_expected_recall": _round_floats(item["topology_retrieval"]["expected_recall"]),
                "flat_expected_recall": _round_floats(item["flat_retrieval"]["expected_recall"]),
                "topology_passed": item["topology_reader"]["passed"],
                "flat_passed": item["flat_reader"]["passed"],
                "topology_fields": dict(item["topology_reader"]["parsed_fields"]),
            }
            for item in report["tasks"]
        ],
    }


def _run_repeated(
    name: str,
    repeats: int,
    runner: Callable[[], Dict[str, Any]],
    compact: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    signatures: List[str] = []
    for index in range(repeats):
        print(f"[real] {name}: run {index + 1}/{repeats}")
        report = compact(runner())
        runs.append(report)
        signatures.append(json.dumps(report, sort_keys=True))
    stability_assessed = repeats > 1
    return {
        "repeats": repeats,
        "stability_assessed": stability_assessed,
        "stable": len(set(signatures)) == 1 if stability_assessed else None,
        "runs": runs,
    }


def _latest_metric_view(report: Dict[str, Any]) -> Dict[str, Any]:
    if "aggregate" in report:
        return _round_floats(report["aggregate"])
    if "enabled" in report and "disabled" in report:
        return {
            "enabled": _round_floats(report["enabled"]["aggregate"]),
            "disabled": _round_floats(report["disabled"]["aggregate"]),
        }
    return _round_floats(report)


def _numeric_delta(before: Any, after: Any) -> Any:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return round(float(after) - float(before), 6)
    if isinstance(before, dict) and isinstance(after, dict):
        delta: Dict[str, Any] = {}
        for key in sorted(set(before.keys()) & set(after.keys())):
            value = _numeric_delta(before[key], after[key])
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            delta[key] = value
        return delta
    return None


def _run_energy_ablation(
    name: str,
    repeats: int,
    runner_off: Callable[[], Dict[str, Any]],
    runner_on: Callable[[], Dict[str, Any]],
    compact: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    off = _run_repeated(f"{name}_energy_off", repeats, runner_off, compact)
    on = _run_repeated(f"{name}_energy_on", repeats, runner_on, compact)
    latest_off = _latest_metric_view(off["runs"][-1])
    latest_on = _latest_metric_view(on["runs"][-1])
    return {
        "repeats": repeats,
        "off": off,
        "on": on,
        "latest_off": latest_off,
        "latest_on": latest_on,
        "delta_latest": _numeric_delta(latest_off, latest_on) or {},
    }


def _markdown_section(title: str, content: List[str]) -> str:
    return "\n".join([f"## {title}", *content, ""])


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Shared-Manifold Full Evaluation")
    lines.append("")
    lines.append(f"Generated: {report['meta']['generated_at_utc']}")
    lines.append(f"Device: {report['meta']['device']}")
    lines.append(f"Real repeats: {report['meta']['real_repeats']}")
    lines.append(f"Energy ablation enabled: {report['meta']['energy_ablation_enabled']}")
    lines.append(f"Energy ablation repeats: {report['meta']['energy_ablation_repeats']}")
    lines.append(f"Deterministic scale multiplier: {report['meta']['deterministic_scale_multiplier']}")
    lines.append(f"Real scale multiplier: {report['meta']['real_scale_multiplier']}")
    lines.append(f"Matrix item count: {report['meta']['matrix_item_count']}")
    if report['meta']['energy_ablation_item_count']:
        lines.append(f"Energy ablation item count: {report['meta']['energy_ablation_item_count']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Deterministic probe slices were run once because they are engine-stub evaluations.")
    if report["meta"]["deterministic_scale_multiplier"] > 1:
        lines.append("- Deterministic task inventories were expanded from the default fixtures using exact templated substitutions over ids, names, and sample values.")
    lines.append("- Real local-model slices were repeated to check stability, not just one-off wins.")
    if report["meta"]["real_scale_multiplier"] > 1:
        lines.append("- Real task inventories were expanded from the default fixtures using exact templated substitutions over ids, names, locations, and field values.")
    else:
        lines.append("- Task inventories are the default tasks already defined in benchmark_shared_manifold.py.")
    lines.append("- This remains proof-of-mechanism evaluation, not a large external benchmark.")
    lines.append("")

    counts = report["meta"]["task_counts"]
    lines.append("## Task Inventory")
    lines.append("")
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    det = report["deterministic"]
    lines.append("## Deterministic Slices")
    lines.append("")
    lines.append(f"- Prompt probe enabled prompt_hit_rate: {det['prompt_probe']['enabled']['aggregate']['prompt_hit_rate']}")
    lines.append(f"- Prompt probe disabled prompt_hit_rate: {det['prompt_probe']['disabled']['aggregate']['prompt_hit_rate']}")
    lines.append(f"- Coding probe enabled pass_rate: {det['coding_probe']['enabled']['aggregate']['pass_rate']}")
    lines.append(f"- Coding probe disabled pass_rate: {det['coding_probe']['disabled']['aggregate']['pass_rate']}")
    lines.append(f"- Topology probe topology_win_rate: {det['topology_probe']['aggregate']['topology_win_rate']}")
    lines.append("")

    lines.append("## Real Slices")
    lines.append("")
    for key, payload in report["real"].items():
        latest = payload["runs"][-1]
        lines.append(f"### {key}")
        lines.append("")
        if payload.get("stability_assessed"):
            lines.append(f"- Stable across repeats: {payload['stable']}")
        else:
            lines.append("- Stable across repeats: not assessed (single run)")
        if "aggregate" in latest:
            for metric_key, metric_value in latest["aggregate"].items():
                lines.append(f"- {metric_key}: {metric_value}")
        elif "enabled" in latest and "disabled" in latest:
            lines.append(f"- enabled aggregate: {json.dumps(latest['enabled']['aggregate'], sort_keys=True)}")
            lines.append(f"- disabled aggregate: {json.dumps(latest['disabled']['aggregate'], sort_keys=True)}")
        lines.append("")

    if report.get("energy_ablation"):
        lines.append("## Energy Ablation")
        lines.append("")
        lines.append("- Automatic manifold-energy feedback remains off in the main real-slice report unless this ablation is explicitly enabled.")
        lines.append("- Each ablation reruns the same real slices twice: once with automatic energy feedback off and once with it on.")
        lines.append("")
        for key, payload in report["energy_ablation"].items():
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"- energy off latest: {json.dumps(payload['latest_off'], sort_keys=True)}")
            lines.append(f"- energy on latest: {json.dumps(payload['latest_on'], sort_keys=True)}")
            lines.append(f"- delta (on - off): {json.dumps(payload['delta_latest'], sort_keys=True)}")
            lines.append("")

    lines.append("## Honesty Notes")
    lines.append("")
    for note in report["honesty_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a consolidated shared-manifold evaluation matrix.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--real-repeats", type=int, default=2)
    parser.add_argument("--include-energy-ablation", action="store_true")
    parser.add_argument("--energy-ablation-repeats", type=int, default=1)
    parser.add_argument("--deterministic-scale-multiplier", type=int, default=1)
    parser.add_argument("--real-scale-multiplier", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "..", "local_artifacts", "warp_cortex_eval"),
    )
    parser.add_argument("--coding-max-tokens", type=int, default=160)
    parser.add_argument("--recall-max-tokens", type=int, default=48)
    args = parser.parse_args()

    generated_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.deterministic_scale_multiplier < 1:
        raise ValueError("--deterministic-scale-multiplier must be at least 1")
    if args.real_scale_multiplier < 1:
        raise ValueError("--real-scale-multiplier must be at least 1")
    if args.include_energy_ablation and args.energy_ablation_repeats < 1:
        raise ValueError("--energy-ablation-repeats must be at least 1 when energy ablation is enabled")

    scenario_tasks = build_scaled_scenarios(args.deterministic_scale_multiplier)
    deterministic_coding_tasks = build_scaled_deterministic_coding_tasks(args.deterministic_scale_multiplier)
    deterministic_topology_tasks = build_scaled_deterministic_topology_tasks(args.deterministic_scale_multiplier)
    real_coding_tasks = build_scaled_real_coding_tasks(args.real_scale_multiplier)
    real_energy_reuse_tasks = build_scaled_real_energy_reuse_tasks(args.real_scale_multiplier)
    real_recall_tasks = build_scaled_real_recall_tasks(args.real_scale_multiplier)
    real_necessity_tasks = build_scaled_real_necessity_tasks(args.real_scale_multiplier)
    real_topology_tasks = build_scaled_real_topology_tasks(args.real_scale_multiplier)

    task_counts = {
        "probe_scenarios": len(scenario_tasks),
        "probe_coding_tasks": len(deterministic_coding_tasks),
        "probe_topology_tasks": len(deterministic_topology_tasks),
        "real_coding_tasks": len(real_coding_tasks),
        "real_coding_handoff_tasks": len(real_coding_tasks),
        "real_energy_reuse_tasks": len(real_energy_reuse_tasks),
        "real_recall_tasks": len(real_recall_tasks),
        "real_necessity_tasks": len(real_necessity_tasks),
        "real_topology_tasks": len(real_topology_tasks),
    }
    matrix_item_count = sum(task_counts.values())
    real_slice_item_count = (
        task_counts["real_coding_tasks"]
        + task_counts["real_coding_handoff_tasks"]
        + task_counts["real_energy_reuse_tasks"]
        + task_counts["real_recall_tasks"]
        + task_counts["real_necessity_tasks"]
        + task_counts["real_topology_tasks"]
    )
    energy_ablation_item_count = real_slice_item_count * 2 if args.include_energy_ablation else 0

    print("[deterministic] prompt probe")
    prompt_probe = _compact_probe_compare(compare_pipeline(scenario_tasks))
    print("[deterministic] coding probe")
    coding_probe = _compact_coding_compare(compare_coding_slice(deterministic_coding_tasks))
    print("[deterministic] topology probe")
    topology_probe = _compact_topology_probe(compare_topology_slice(deterministic_topology_tasks))

    real_report = {
        "coding_compare": _run_repeated(
            "coding_compare",
            args.real_repeats,
            lambda: compare_real_coding_slice(
                tasks=real_coding_tasks,
                model_id=args.model_id,
                device=args.device,
                max_tokens=args.coding_max_tokens,
            ),
            _compact_real_coding,
        ),
        "energy_reuse": _run_repeated(
            "energy_reuse",
            args.real_repeats,
            lambda: compare_real_energy_reuse_slice(
                tasks=real_energy_reuse_tasks,
                model_id=args.model_id,
                device=args.device,
            ),
            _compact_real_energy_reuse,
        ),
        "coding_handoff": _run_repeated(
            "coding_handoff",
            args.real_repeats,
            lambda: compare_real_handoff_slice(
                tasks=real_coding_tasks,
                model_id=args.model_id,
                device=args.device,
                max_tokens=args.coding_max_tokens,
            ),
            _compact_real_handoff,
        ),
        "recall_handoff": _run_repeated(
            "recall_handoff",
            args.real_repeats,
            lambda: compare_real_recall_handoff_slice(
                tasks=real_recall_tasks,
                model_id=args.model_id,
                device=args.device,
                max_tokens=args.recall_max_tokens,
            ),
            _compact_real_recall_handoff,
        ),
        "necessity": _run_repeated(
            "necessity",
            args.real_repeats,
            lambda: compare_real_necessity_slice(
                tasks=real_necessity_tasks,
                model_id=args.model_id,
                device=args.device,
                max_tokens=args.recall_max_tokens,
            ),
            _compact_real_necessity,
        ),
        "topology": _run_repeated(
            "topology",
            args.real_repeats,
            lambda: compare_real_topology_slice(
                tasks=real_topology_tasks,
                model_id=args.model_id,
                device=args.device,
                max_tokens=args.recall_max_tokens,
            ),
            _compact_real_topology,
        ),
    }

    energy_ablation_report = None
    if args.include_energy_ablation:
        energy_ablation_report = {
            "coding_compare": _run_energy_ablation(
                "coding_compare",
                args.energy_ablation_repeats,
                lambda: compare_real_coding_slice(
                    tasks=real_coding_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.coding_max_tokens,
                ),
                lambda: compare_real_coding_slice(
                    tasks=real_coding_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.coding_max_tokens,
                ),
                _compact_real_coding,
            ),
            "energy_reuse": _run_energy_ablation(
                "energy_reuse",
                args.energy_ablation_repeats,
                lambda: compare_real_energy_reuse_slice(
                    tasks=real_energy_reuse_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                ),
                lambda: compare_real_energy_reuse_slice(
                    tasks=real_energy_reuse_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                ),
                _compact_real_energy_reuse,
            ),
            "coding_handoff": _run_energy_ablation(
                "coding_handoff",
                args.energy_ablation_repeats,
                lambda: compare_real_handoff_slice(
                    tasks=real_coding_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.coding_max_tokens,
                ),
                lambda: compare_real_handoff_slice(
                    tasks=real_coding_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.coding_max_tokens,
                ),
                _compact_real_handoff,
            ),
            "recall_handoff": _run_energy_ablation(
                "recall_handoff",
                args.energy_ablation_repeats,
                lambda: compare_real_recall_handoff_slice(
                    tasks=real_recall_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                lambda: compare_real_recall_handoff_slice(
                    tasks=real_recall_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                _compact_real_recall_handoff,
            ),
            "necessity": _run_energy_ablation(
                "necessity",
                args.energy_ablation_repeats,
                lambda: compare_real_necessity_slice(
                    tasks=real_necessity_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                lambda: compare_real_necessity_slice(
                    tasks=real_necessity_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                _compact_real_necessity,
            ),
            "topology": _run_energy_ablation(
                "topology",
                args.energy_ablation_repeats,
                lambda: compare_real_topology_slice(
                    tasks=real_topology_tasks,
                    enable_energy_feedback=False,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                lambda: compare_real_topology_slice(
                    tasks=real_topology_tasks,
                    enable_energy_feedback=True,
                    model_id=args.model_id,
                    device=args.device,
                    max_tokens=args.recall_max_tokens,
                ),
                _compact_real_topology,
            ),
        }

    report = {
        "meta": {
            "generated_at_utc": generated_at,
            "device": args.device,
            "model_id": args.model_id or "default",
            "real_repeats": args.real_repeats,
            "energy_ablation_enabled": bool(args.include_energy_ablation),
            "energy_ablation_repeats": args.energy_ablation_repeats if args.include_energy_ablation else 0,
            "deterministic_scale_multiplier": args.deterministic_scale_multiplier,
            "real_scale_multiplier": args.real_scale_multiplier,
            "matrix_item_count": matrix_item_count,
            "energy_ablation_item_count": energy_ablation_item_count,
            "task_counts": task_counts,
        },
        "deterministic": {
            "prompt_probe": prompt_probe,
            "coding_probe": coding_probe,
            "topology_probe": topology_probe,
        },
        "real": real_report,
        "energy_ablation": energy_ablation_report,
        "honesty_notes": [
            "These are default benchmark tasks defined inside benchmark_shared_manifold.py, not external public benchmark suites.",
            "If deterministic_scale_multiplier > 1, the larger deterministic sweep is created by templated substitutions over the base fixtures rather than by adding new benchmark families.",
            "If real_scale_multiplier > 1, the larger run is created by templated substitutions over the base fixtures rather than by adding new human-authored benchmark families.",
            "The topology slices are hand-constructed proof-of-mechanism tasks; bridge labels come from benchmark metadata rather than external annotation.",
            "Real slices use the local Qwen/Qwen2.5-0.5B-Instruct runtime on the requested device and should be interpreted as mechanism validation rather than large-scale benchmark coverage.",
            "The energy_reuse slice is an intentionally targeted follow-up benchmark that measures whether prompt-time energy keeps a previously used task-board neighborhood selected under a blended follow-up query.",
            "Repeat stability is reported explicitly so one-off wins are distinguishable from stable behavior.",
            "When --include-energy-ablation is enabled, the runner replays the same real slices with automatic manifold-energy feedback toggled off and on; this is an internal ablation, not a separate benchmark family.",
        ],
    }

    scale_parts = []
    if args.deterministic_scale_multiplier > 1:
        scale_parts.append(f"d{args.deterministic_scale_multiplier}")
    if args.real_scale_multiplier > 1:
        scale_parts.append(f"r{args.real_scale_multiplier}")
    scale_tag = "_" + "_".join(scale_parts) if scale_parts else ""
    stem = f"shared_manifold_full_eval{scale_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = os.path.join(args.output_dir, stem + ".json")
    md_path = os.path.join(args.output_dir, stem + ".md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_build_markdown(report))

    print(f"[report] json={json_path}")
    print(f"[report] markdown={md_path}")
    summary = {
        "deterministic": {
            "prompt_probe": prompt_probe["enabled"]["aggregate"],
            "coding_probe": coding_probe["enabled"]["aggregate"],
            "topology_probe": topology_probe["aggregate"],
        },
        "real_stability": {key: value["stable"] for key, value in real_report.items()},
        "real_stability_assessed": {key: value["stability_assessed"] for key, value in real_report.items()},
        "real_latest": {
            key: value["runs"][-1]["aggregate"] if "aggregate" in value["runs"][-1] else {
                "enabled": value["runs"][-1]["enabled"]["aggregate"],
                "disabled": value["runs"][-1]["disabled"]["aggregate"],
            }
            for key, value in real_report.items()
        },
    }
    if energy_ablation_report is not None:
        summary["energy_ablation_latest"] = {
            key: {
                "off": value["latest_off"],
                "on": value["latest_on"],
                "delta": value["delta_latest"],
            }
            for key, value in energy_ablation_report.items()
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())