from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import TraceCase


def builtin_trace_cases() -> List[TraceCase]:
    return [
        TraceCase(
            case_id="support_refund_json_001",
            task_type="support_json",
            prompt=(
                "Return ONLY a JSON object. Support note: customer refund R-1042 "
                "was approved for 1299 cents. Required keys: refund_id, amount_cents."
            ),
            expected={"refund_id": "R-1042", "amount_cents": 1299},
            metadata={"suite": "coding-smoke", "failure_neighborhood": "json_field_extraction"},
        ),
        TraceCase(
            case_id="payment_retry_json_001",
            task_type="support_json",
            prompt=(
                "Return ONLY a JSON object. Checkout ticket PX-17 uses "
                "retry_header=X-Payment-Retry-Key and replay seal field replay_token_px17. "
                "Required keys: retry_header, seal."
            ),
            expected={"retry_header": "X-Payment-Retry-Key", "seal": "replay_token_px17"},
            metadata={"suite": "coding-smoke", "failure_neighborhood": "payment_retry_fields"},
        ),
        TraceCase(
            case_id="recall_location_json_001",
            task_type="recall_json",
            prompt=(
                "Return ONLY a JSON object. Jenny left the red boots in locker 14. "
                "Required keys: color, where."
            ),
            expected={"color": "red", "where": "locker 14"},
            metadata={"suite": "recall-smoke", "failure_neighborhood": "entity_binding"},
        ),
    ]


def load_trace_cases(path: str) -> List[TraceCase]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if trace_path.suffix.lower() == ".jsonl":
        rows = []
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    else:
        with trace_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        rows = loaded.get("cases", loaded) if isinstance(loaded, dict) else loaded

    if not isinstance(rows, list):
        raise ValueError("Trace file must contain a JSON array or an object with a 'cases' array")

    return [_case_from_mapping(row) for row in rows]


def limit_cases(cases: Iterable[TraceCase], limit: int) -> List[TraceCase]:
    selected = list(cases)
    if limit > 0:
        return selected[:limit]
    return selected


def _case_from_mapping(row: Dict[str, Any]) -> TraceCase:
    case_id = str(row.get("id") or row.get("case_id") or "").strip()
    if not case_id:
        raise ValueError(f"Trace case is missing id/case_id: {row}")
    expected = row.get("expected") or {}
    if not isinstance(expected, dict):
        raise ValueError(f"Trace case expected value must be a mapping: {case_id}")
    return TraceCase(
        case_id=case_id,
        task_type=str(row.get("task_type") or "default"),
        prompt=str(row.get("prompt") or ""),
        expected=expected,
        validator=str(row.get("validator") or "json_fields"),
        metadata=dict(row.get("metadata") or {}),
    )