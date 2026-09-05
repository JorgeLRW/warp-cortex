from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TraceCase:
    """One scored workload item."""

    case_id: str
    task_type: str
    prompt: str
    expected: Dict[str, Any]
    validator: str = "json_fields"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSpec:
    """A backend/config candidate to evaluate."""

    name: str
    strategy: str
    primary_backend: str
    primary_model: str = ""
    fallback_backend: Optional[str] = None
    fallback_model: str = ""
    primary_backend_options: Dict[str, Any] = field(default_factory=dict)
    fallback_backend_options: Dict[str, Any] = field(default_factory=dict)
    primary_cost_model: Dict[str, Any] = field(default_factory=dict)
    fallback_cost_model: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScorecardConfig:
    """Runtime configuration for one scorecard run."""

    suite: str = "builtin"
    out_dir: str = "local_artifacts/scorecards/latest"
    max_tokens: int = 96
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    device: str = "auto"
    hf_home: str = ""
    offline: bool = True
    evidence_db: str = ""
    limit: int = 0


@dataclass
class ValidationResult:
    passed: bool
    parsed: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    mismatches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_checks: List[str] = field(default_factory=list)
    failure_kind: str = ""
    notes: str = ""


@dataclass
class BackendResponse:
    text: str
    elapsed_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    remote_calls: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptResult:
    backend: str
    model: str
    status: str
    output: str = ""
    elapsed_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    remote_calls: int = 0
    cost_usd: float = 0.0
    error: str = ""
    validation: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "output": self.output,
            "elapsed_s": self.elapsed_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "remote_calls": self.remote_calls,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }
        if self.validation is not None:
            payload["validation"] = validation_to_dict(self.validation)
        return payload


@dataclass
class CaseResult:
    case_id: str
    task_type: str
    candidate: str
    strategy: str
    status: str
    passed: bool
    elapsed_s: float
    remote_calls: int
    cost_usd: float
    output: str
    validation: ValidationResult
    attempts: List[AttemptResult] = field(default_factory=list)
    fallback_used: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task_type": self.task_type,
            "candidate": self.candidate,
            "strategy": self.strategy,
            "status": self.status,
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "remote_calls": self.remote_calls,
            "cost_usd": self.cost_usd,
            "output": self.output,
            "validation": validation_to_dict(self.validation),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


def validation_to_dict(result: ValidationResult) -> Dict[str, Any]:
    return {
        "passed": result.passed,
        "parsed": result.parsed,
        "missing_fields": result.missing_fields,
        "mismatches": result.mismatches,
        "failed_checks": result.failed_checks,
        "failure_kind": result.failure_kind,
        "notes": result.notes,
    }