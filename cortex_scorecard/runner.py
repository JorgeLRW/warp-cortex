from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .backends import BackendContext, BackendUnavailable, DeterministicBackend, LocalHFBackend, OpenAIBackend, WarpBitNetBackend
from .schema import AttemptResult, BackendResponse, CandidateSpec, CaseResult, ScorecardConfig, TraceCase, ValidationResult
from .tasks import builtin_trace_cases, limit_cases, load_trace_cases
from .validators import validate_output


def run_scorecard(
    *,
    config: ScorecardConfig,
    candidate_names: Iterable[str],
    candidate_specs: Optional[Iterable[CandidateSpec]] = None,
    trace_file: str = "",
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    api_model: str = "gpt-4o-mini",
    write_artifacts: bool = True,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    created_at = _utc_timestamp()
    cases = load_trace_cases(trace_file) if trace_file else builtin_trace_cases()
    cases = limit_cases(cases, config.limit)
    candidates = list(candidate_specs) if candidate_specs is not None else parse_candidate_specs(candidate_names, local_model=local_model, api_model=api_model)

    backend_context = BackendContext(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        device=config.device,
        hf_home=config.hf_home,
        offline=config.offline,
        timeout_seconds=config.timeout_seconds,
    )
    backend_cache: Dict[Tuple[str, str], Any] = {}
    results: List[CaseResult] = []

    for candidate in candidates:
        for case in cases:
            results.append(_run_case(candidate, case, backend_context, backend_cache))

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "suite": config.suite,
        "trace_file": trace_file,
        "config": {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
            "device": config.device,
            "offline": config.offline,
            "limit": config.limit,
        },
        "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
        "cases": [_case_to_dict(case) for case in cases],
        "results": [result.to_dict() for result in results],
        "aggregate": _aggregate_results(cases, candidates, results),
        "manifest": _build_manifest(local_model=local_model, api_model=api_model),
    }
    report["policy"] = _compile_policy(report)

    if write_artifacts:
        out_dir = Path(config.out_dir)
        _write_artifacts(report, out_dir)
        evidence_db = config.evidence_db or str(out_dir / "evidence.sqlite")
        _persist_evidence(report, evidence_db)
        report["artifacts"] = {
            "out_dir": str(out_dir),
            "scorecard_json": str(out_dir / "scorecard.json"),
            "scorecard_md": str(out_dir / "scorecard.md"),
            "failures_jsonl": str(out_dir / "failures.jsonl"),
            "policy_yaml": str(out_dir / "policy.yaml"),
            "manifest_json": str(out_dir / "manifest.json"),
            "evidence_db": evidence_db,
        }
        _write_json(out_dir / "scorecard.json", report)

    return report


def parse_candidate_specs(
    candidate_names: Iterable[str],
    *,
    local_model: str,
    api_model: str,
) -> List[CandidateSpec]:
    names = [name.strip() for name in candidate_names if str(name).strip()]
    if not names:
        names = ["local"]

    specs = []
    for name in names:
        if name == "deterministic":
            specs.append(
                CandidateSpec(
                    name=name,
                    strategy="single",
                    primary_backend="deterministic",
                    primary_model="fixture",
                    metadata={"class": "fixture", "description": "Deterministic validation fixture."},
                )
            )
        elif name == "deterministic_bad":
            specs.append(
                CandidateSpec(
                    name=name,
                    strategy="single",
                    primary_backend="deterministic_bad",
                    primary_model="fixture_bad",
                    metadata={"class": "fixture", "description": "Intentionally broken deterministic validation fixture."},
                )
            )
        elif name == "hybrid_demo":
            specs.append(
                CandidateSpec(
                    name=name,
                    strategy="hybrid",
                    primary_backend="deterministic_bad",
                    primary_model="fixture_bad",
                    fallback_backend="deterministic",
                    fallback_model="fixture",
                    metadata={"class": "fixture", "description": "Hybrid deterministic demo with a forced fallback."},
                )
            )
        elif name == "local":
            specs.append(
                CandidateSpec(
                    name="local_hf",
                    strategy="single",
                    primary_backend="local_hf",
                    primary_model=local_model,
                    primary_cost_model={"kind": "local_time", "usd_per_hour": 1.25},
                    metadata={"class": "local", "description": "Legacy local Hugging Face candidate.", "pricing_assumption": "1.25 USD per GPU hour"},
                )
            )
        elif name == "api":
            specs.append(
                CandidateSpec(
                    name="api_single",
                    strategy="single",
                    primary_backend="api_openai",
                    primary_model=api_model,
                    primary_cost_model=_default_api_cost_model(api_model),
                    metadata={"class": "api", "description": "Legacy hosted OpenAI-compatible API candidate."},
                )
            )
        elif name == "hybrid":
            specs.append(
                CandidateSpec(
                    name="hybrid_repair",
                    strategy="hybrid",
                    primary_backend="local_hf",
                    primary_model=local_model,
                    fallback_backend="api_openai",
                    fallback_model=api_model,
                    primary_cost_model={"kind": "local_time", "usd_per_hour": 1.25},
                    fallback_cost_model=_default_api_cost_model(api_model),
                    metadata={"class": "hybrid", "description": "Legacy local-first candidate with API fallback."},
                )
            )
        else:
            raise ValueError(f"Unknown scorecard candidate: {name}")
    return specs


def _default_api_cost_model(model_id: str) -> Dict[str, Any]:
    per_million: Dict[str, Tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
    }
    in_rate, out_rate = per_million.get(model_id, (0.0, 0.0))
    if in_rate == 0.0 and out_rate == 0.0:
        return {}
    return {
        "kind": "openai_tokens",
        "input_per_million": in_rate,
        "output_per_million": out_rate,
    }


def _run_case(
    candidate: CandidateSpec,
    case: TraceCase,
    context: BackendContext,
    backend_cache: Dict[Tuple[str, str], Any],
) -> CaseResult:
    attempts: List[AttemptResult] = []
    primary_attempt = _run_attempt(
        candidate.primary_backend,
        candidate.primary_model,
        case,
        context,
        backend_cache,
        backend_options=candidate.primary_backend_options,
        cost_model=candidate.primary_cost_model,
    )
    attempts.append(primary_attempt)
    repaired_primary_attempt = _maybe_run_repair_attempt(
        candidate.primary_backend,
        candidate.primary_model,
        case,
        context,
        backend_cache,
        backend_options=candidate.primary_backend_options,
        cost_model=candidate.primary_cost_model,
        prior_attempt=primary_attempt,
    )
    if repaired_primary_attempt is not None:
        attempts.append(repaired_primary_attempt)
        primary_attempt = repaired_primary_attempt

    if candidate.strategy == "hybrid" and primary_attempt.status == "ok":
        primary_validation = primary_attempt.validation
        if primary_validation is not None and primary_validation.passed:
            return _case_result(candidate, case, attempts, fallback_used=False)
        if candidate.fallback_backend:
            fallback_attempt = _run_attempt(
                candidate.fallback_backend,
                candidate.fallback_model,
                case,
                context,
                backend_cache,
                backend_options=candidate.fallback_backend_options,
                cost_model=candidate.fallback_cost_model,
            )
            attempts.append(fallback_attempt)
            repaired_fallback_attempt = _maybe_run_repair_attempt(
                candidate.fallback_backend,
                candidate.fallback_model,
                case,
                context,
                backend_cache,
                backend_options=candidate.fallback_backend_options,
                cost_model=candidate.fallback_cost_model,
                prior_attempt=fallback_attempt,
            )
            if repaired_fallback_attempt is not None:
                attempts.append(repaired_fallback_attempt)
            return _case_result(candidate, case, attempts, fallback_used=True)

    return _case_result(candidate, case, attempts, fallback_used=False)


def _maybe_run_repair_attempt(
    backend_name: str | None,
    model_id: str,
    case: TraceCase,
    context: BackendContext,
    backend_cache: Dict[Tuple[str, str], Any],
    *,
    backend_options: Dict[str, Any],
    cost_model: Dict[str, Any],
    prior_attempt: AttemptResult,
) -> Optional[AttemptResult]:
    validation = prior_attempt.validation
    if backend_name not in {"local_hf", "warp_bitnet"}:
        return None
    if prior_attempt.status != "ok" or validation is None or validation.passed:
        return None
    if case.validator != "json_fields" or "json_parse" in validation.failed_checks:
        return None

    repair_attempt = _run_repair_attempt(
        backend_name,
        model_id,
        case,
        context,
        backend_cache,
        backend_options=backend_options,
        cost_model=cost_model,
        prior_attempt=prior_attempt,
    )
    repair_validation = repair_attempt.validation
    if repair_attempt.status == "ok" and repair_validation is not None and repair_validation.passed:
        return repair_attempt
    return None


def _run_repair_attempt(
    backend_name: str,
    model_id: str,
    case: TraceCase,
    context: BackendContext,
    backend_cache: Dict[Tuple[str, str], Any],
    *,
    backend_options: Dict[str, Any],
    cost_model: Dict[str, Any],
    prior_attempt: AttemptResult,
) -> AttemptResult:
    try:
        backend = _get_backend(backend_name, model_id, backend_options, backend_cache)
        validation = prior_attempt.validation or ValidationResult(passed=False)
        response: BackendResponse = backend.repair(case, prior_attempt.output, validation, context)
        repair_validation = validate_output(case, response.text)
        cost_usd = _estimate_cost(response, cost_model)
        return AttemptResult(
            backend=backend_name,
            model=model_id,
            status="ok",
            output=response.text,
            elapsed_s=response.elapsed_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            remote_calls=response.remote_calls,
            cost_usd=cost_usd,
            validation=repair_validation,
        )
    except BackendUnavailable as exc:
        return AttemptResult(backend=backend_name, model=model_id, status="skipped", error=str(exc))
    except Exception as exc:
        return AttemptResult(backend=backend_name, model=model_id, status="error", error=str(exc))


def _run_attempt(
    backend_name: str,
    model_id: str,
    case: TraceCase,
    context: BackendContext,
    backend_cache: Dict[Tuple[str, str], Any],
    *,
    backend_options: Dict[str, Any],
    cost_model: Dict[str, Any],
) -> AttemptResult:
    try:
        backend = _get_backend(backend_name, model_id, backend_options, backend_cache)
        response: BackendResponse = backend.generate(case, context)
        validation = validate_output(case, response.text)
        cost_usd = _estimate_cost(response, cost_model)
        return AttemptResult(
            backend=backend_name,
            model=model_id,
            status="ok",
            output=response.text,
            elapsed_s=response.elapsed_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            remote_calls=response.remote_calls,
            cost_usd=cost_usd,
            validation=validation,
        )
    except BackendUnavailable as exc:
        return AttemptResult(backend=backend_name, model=model_id, status="skipped", error=str(exc))
    except Exception as exc:
        return AttemptResult(backend=backend_name, model=model_id, status="error", error=str(exc))


def _get_backend(
    backend_name: str,
    model_id: str,
    backend_options: Dict[str, Any],
    backend_cache: Dict[Tuple[str, str], Any],
):
    options_key = json.dumps(backend_options, sort_keys=True)
    key = (backend_name, model_id, options_key)
    if key in backend_cache:
        return backend_cache[key]
    if backend_name == "deterministic":
        backend = DeterministicBackend()
    elif backend_name == "deterministic_bad":
        backend = DeterministicBackend(broken=True)
    elif backend_name == "local_hf":
        backend = LocalHFBackend(model_id)
    elif backend_name == "warp_bitnet":
        backend = WarpBitNetBackend(model_id, model_dir=str(backend_options.get("model_dir", "") or ""))
    elif backend_name == "api_openai":
        backend = OpenAIBackend(model_id, base_url=str(backend_options.get("base_url", "") or ""))
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    backend_cache[key] = backend
    return backend


def _estimate_cost(response: BackendResponse, cost_model: Dict[str, Any]) -> float:
    if not cost_model:
        return response.cost_usd

    kind = str(cost_model.get("kind", "") or "").strip().lower()
    if kind == "openai_tokens":
        input_rate = float(cost_model.get("input_per_million", 0.0) or 0.0)
        output_rate = float(cost_model.get("output_per_million", 0.0) or 0.0)
        return (response.input_tokens * input_rate + response.output_tokens * output_rate) / 1_000_000.0
    if kind == "local_time":
        usd_per_hour = float(cost_model.get("usd_per_hour", 0.0) or 0.0)
        fixed_cost = float(cost_model.get("fixed_cost_per_request", 0.0) or 0.0)
        multiplier = float(cost_model.get("multiplier", 1.0) or 1.0)
        return fixed_cost + ((response.elapsed_s * multiplier) / 3600.0) * usd_per_hour
    if kind == "per_request":
        usd_per_request = float(cost_model.get("usd_per_request", 0.0) or 0.0)
        request_count = max(1, int(response.remote_calls or 0))
        return usd_per_request * request_count
    if kind == "passthrough":
        return response.cost_usd
    return response.cost_usd


def _case_result(candidate: CandidateSpec, case: TraceCase, attempts: List[AttemptResult], *, fallback_used: bool) -> CaseResult:
    final_attempt = attempts[-1]
    validation = final_attempt.validation or ValidationResult(
        passed=False,
        failed_checks=[final_attempt.status],
        failure_kind=final_attempt.status,
        notes=final_attempt.error,
    )
    if final_attempt.status == "skipped":
        status = "skipped"
    elif final_attempt.status == "error":
        status = "error"
    elif validation.passed:
        status = "passed"
    else:
        status = "failed"
    return CaseResult(
        case_id=case.case_id,
        task_type=case.task_type,
        candidate=candidate.name,
        strategy=candidate.strategy,
        status=status,
        passed=validation.passed,
        elapsed_s=sum(attempt.elapsed_s for attempt in attempts),
        remote_calls=sum(attempt.remote_calls for attempt in attempts),
        cost_usd=sum(attempt.cost_usd for attempt in attempts),
        output=final_attempt.output,
        validation=validation,
        attempts=attempts,
        fallback_used=fallback_used,
        error=final_attempt.error,
    )


def _aggregate_results(cases: List[TraceCase], candidates: List[CandidateSpec], results: List[CaseResult]) -> Dict[str, Any]:
    candidate_rows = {}
    for candidate in candidates:
        subset = [result for result in results if result.candidate == candidate.name]
        evaluated = [result for result in subset if result.status != "skipped"]
        passed = [result for result in subset if result.passed]
        case_count = len(subset)
        evaluated_count = len(evaluated)
        candidate_rows[candidate.name] = {
            "case_count": case_count,
            "evaluated_count": evaluated_count,
            "passed_count": len(passed),
            "failed_count": len([result for result in subset if result.status == "failed"]),
            "error_count": len([result for result in subset if result.status == "error"]),
            "skipped_count": len([result for result in subset if result.status == "skipped"]),
            "pass_rate": len(passed) / max(case_count, 1),
            "evaluated_pass_rate": len(passed) / max(evaluated_count, 1),
            "completion_rate": evaluated_count / max(case_count, 1),
            "remote_call_rate": sum(result.remote_calls for result in subset) / max(case_count, 1),
            "fallback_rate": len([result for result in subset if result.fallback_used]) / max(case_count, 1),
            "avg_latency_s": sum(result.elapsed_s for result in subset) / max(case_count, 1),
            "cost_usd": sum(result.cost_usd for result in subset),
        }

    return {
        "case_count": len(cases),
        "candidate_count": len(candidates),
        "result_count": len(results),
        "candidate_summary": candidate_rows,
    }


def _compile_policy(report: Dict[str, Any]) -> Dict[str, Any]:
    results = report["results"]
    candidates = {candidate["name"]: candidate for candidate in report["candidates"]}
    task_types = sorted({result["task_type"] for result in results})
    routes = {}
    for task_type in task_types:
        candidate_scores = []
        for candidate_index, candidate_name in enumerate(candidates):
            subset = [result for result in results if result["candidate"] == candidate_name and result["task_type"] == task_type]
            evaluated = [result for result in subset if result["status"] != "skipped"]
            if not evaluated:
                continue
            pass_rate = sum(1 for result in evaluated if result["passed"]) / len(evaluated)
            remote_rate = sum(result["remote_calls"] for result in subset) / max(len(subset), 1)
            fallback_rate = sum(1 for result in subset if result["fallback_used"]) / max(len(subset), 1)
            latency = sum(result["elapsed_s"] for result in subset) / max(len(subset), 1)
            candidate_scores.append((pass_rate, -remote_rate, -fallback_rate, -latency, -candidate_index, candidate_name))
        if not candidate_scores:
            continue
        candidate_scores.sort(reverse=True)
        selected_name = candidate_scores[0][5]
        selected = candidates[selected_name]
        route = {
            "candidate": selected_name,
            "default_backend": selected["primary_backend"],
            "default_model": selected["primary_model"],
            "escalate_on": ["validation_failed", "json_parse", "missing_fields", "field_mismatch"],
        }
        if selected.get("fallback_backend"):
            route["fallback_backend"] = selected["fallback_backend"]
            route["fallback_model"] = selected.get("fallback_model", "")
        routes[task_type] = route
    return {"version": 1, "generated_at": report["created_at"], "routes": routes}


def _write_artifacts(report: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "scorecard.json", report)
    _write_json(out_dir / "manifest.json", report["manifest"])
    (out_dir / "policy.yaml").write_text(yaml.safe_dump(report["policy"], sort_keys=False), encoding="utf-8")
    (out_dir / "scorecard.md").write_text(_render_markdown(report), encoding="utf-8")
    with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for result in report["results"]:
            if not result["passed"]:
                handle.write(json.dumps(result, sort_keys=True) + "\n")


def _persist_evidence(report: Dict[str, Any], evidence_db: str):
    try:
        from cortex_core.agent_cloud import PersistentAgentCloud
    except Exception:
        return

    cloud = PersistentAgentCloud(hidden_dim=64, device="cpu", shared_manifold_capacity=4096, shared_store_path=evidence_db)
    for result in report["results"]:
        validation = result["validation"]
        failure_kind = validation.get("failure_kind") or "passed"
        text = (
            f"Scorecard {report['run_id']} case {result['case_id']} candidate {result['candidate']} "
            f"status {result['status']} task_type {result['task_type']} failure {failure_kind}."
        )
        keywords = [
            "scorecard",
            result["candidate"],
            result["task_type"],
            result["status"],
            failure_kind,
        ] + list(validation.get("missing_fields") or []) + list((validation.get("mismatches") or {}).keys())
        cloud.remember_shared_text(
            text=text,
            node_type="scorecard_result",
            source="scorecard",
            score=1.0 if result["passed"] else 0.7,
            metadata={
                "node_id": f"scorecard:{report['run_id']}:{result['candidate']}:{result['case_id']}",
                "run_id": report["run_id"],
                "case_id": result["case_id"],
                "candidate": result["candidate"],
                "task_type": result["task_type"],
                "status": result["status"],
                "passed": result["passed"],
                "failure_kind": failure_kind,
                "keywords": keywords,
                "entity_refs": keywords,
            },
            replace_existing=True,
            refresh_hot_state=False,
        )


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# Warp Cortex Scorecard {report['run_id']}",
        "",
        f"Created: {report['created_at']}",
        f"Suite: {report['suite']}",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Pass Rate | Remote Calls / Task | Fallback Rate | Avg Latency (s) | Cost USD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["aggregate"]["candidate_summary"].items():
        lines.append(
            f"| {name} | {summary['pass_rate']:.2f} | {summary['remote_call_rate']:.2f} | "
            f"{summary['fallback_rate']:.2f} | {summary['avg_latency_s']:.3f} | {summary['cost_usd']:.6f} |"
        )

    failures = [result for result in report["results"] if not result["passed"]]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failures recorded.")
    else:
        for result in failures[:25]:
            validation = result["validation"]
            lines.append(
                f"- {result['candidate']} / {result['case_id']}: {result['status']} "
                f"({validation.get('failure_kind') or result.get('error') or 'failed'})"
            )

    lines.extend(["", "## Policy Preview", "", "```yaml", yaml.safe_dump(report["policy"], sort_keys=False).strip(), "```", ""])
    return "\n".join(lines)


def _build_manifest(*, local_model: str, api_model: str) -> Dict[str, Any]:
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git_sha": _git_sha(),
        "models": {"local": local_model, "api": api_model},
        "packages": {},
    }
    for package_name in ("torch", "transformers", "yaml"):
        try:
            module = __import__(package_name)
            manifest["packages"][package_name] = getattr(module, "__version__", "unknown")
        except Exception:
            manifest["packages"][package_name] = "unavailable"
    return manifest


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _write_json(path: Path, payload: Dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _candidate_to_dict(candidate: CandidateSpec) -> Dict[str, Any]:
    return {
        "name": candidate.name,
        "strategy": candidate.strategy,
        "primary_backend": candidate.primary_backend,
        "primary_model": candidate.primary_model,
        "fallback_backend": candidate.fallback_backend,
        "fallback_model": candidate.fallback_model,
        "primary_backend_options": candidate.primary_backend_options,
        "fallback_backend_options": candidate.fallback_backend_options,
        "primary_cost_model": candidate.primary_cost_model,
        "fallback_cost_model": candidate.fallback_cost_model,
        "metadata": candidate.metadata,
    }


def _case_to_dict(case: TraceCase) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "task_type": case.task_type,
        "validator": case.validator,
        "expected_keys": sorted(case.expected.keys()),
        "metadata": case.metadata,
    }