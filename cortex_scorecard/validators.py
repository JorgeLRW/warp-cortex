from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

from .schema import TraceCase, ValidationResult


def validate_output(case: TraceCase, output: str) -> ValidationResult:
    if case.validator != "json_fields":
        return ValidationResult(
            passed=False,
            failed_checks=["unsupported_validator"],
            failure_kind="validator_error",
            notes=f"Unsupported validator: {case.validator}",
        )
    return validate_json_fields(output, case.expected, field_aliases=_field_aliases(case))


def validate_json_fields(output: str, expected: Dict[str, Any], *, field_aliases: Optional[Dict[str, Any]] = None) -> ValidationResult:
    parsed = extract_json_object(output)
    if parsed is None:
        return ValidationResult(
            passed=False,
            failed_checks=["json_parse"],
            failure_kind="format_failure",
            notes="No JSON object could be parsed from the model output.",
        )

    missing_fields = []
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key, expected_value in expected.items():
        if key not in parsed:
            missing_fields.append(key)
            continue
        actual_value = parsed[key]
        aliases = _aliases_for_field(field_aliases, key)
        if not values_match(actual_value, expected_value, aliases=aliases):
            mismatches[key] = {"expected": expected_value, "actual": actual_value}

    failed_checks = []
    if missing_fields:
        failed_checks.append("missing_fields")
    if mismatches:
        failed_checks.append("field_mismatch")

    failure_kind = ""
    if missing_fields:
        failure_kind = "missing_field"
    elif mismatches:
        failure_kind = "wrong_value"

    return ValidationResult(
        passed=not failed_checks,
        parsed=parsed,
        missing_fields=missing_fields,
        mismatches=mismatches,
        failed_checks=failed_checks,
        failure_kind=failure_kind,
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def values_match(actual: Any, expected: Any, *, aliases: Optional[list[str]] = None) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        actual_number = _as_float(actual)
        expected_number = _as_float(expected)
        if actual_number is None or expected_number is None:
            return False
        return math.isclose(actual_number, expected_number, rel_tol=1e-9, abs_tol=1e-9)

    if isinstance(expected, bool):
        return actual is expected

    normalized_actual = str(actual).strip().lower()
    normalized_expected = str(expected).strip().lower()
    if normalized_actual == normalized_expected:
        return True
    normalized_aliases = {str(alias).strip().lower() for alias in aliases or [] if str(alias).strip()}
    return normalized_actual in normalized_aliases


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field_aliases(case: TraceCase) -> Dict[str, Any]:
    metadata = case.metadata or {}
    aliases = metadata.get("field_aliases") or {}
    if not isinstance(aliases, dict):
        return {}
    return aliases


def _aliases_for_field(field_aliases: Optional[Dict[str, Any]], key: str) -> list[str]:
    if not isinstance(field_aliases, dict):
        return []
    aliases = field_aliases.get(key) or []
    if isinstance(aliases, list):
        return [str(alias) for alias in aliases]
    return [str(aliases)]