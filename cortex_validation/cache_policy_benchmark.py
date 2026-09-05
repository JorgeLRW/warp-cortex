"""Controlled comparison harness for Python cache policies.

The harness measures the policy operation plus an optional decode callback. It
cannot infer answer quality, so callers provide ``evaluate_fn`` for task
accuracy and ``equivalent_fn`` for output equivalence.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional

from cortex_core.cache_control import (
    CachePolicy,
    CachePolicyMeasurement,
    PythonKVCacheAdapter,
)


def run_cache_policy_benchmark(
    cache: Any,
    *,
    adapter: Optional[PythonKVCacheAdapter] = None,
    synapse: Any = None,
    query_states: Any = None,
    keep_tokens: int = 64,
    keep_ratio: Optional[float] = None,
    decode_fn: Optional[Callable[[Any], Any]] = None,
    evaluate_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    equivalent_fn: Optional[Callable[[Any, Any], bool]] = None,
) -> List[CachePolicyMeasurement]:
    """Run none, truncate, and landmark trials over independent cache copies.

    ``decode_fn`` should run the same prompt/task continuation for each cache.
    ``evaluate_fn`` should return task-specific metrics such as exact-match or
    verifier pass rate. ``equivalent_fn`` receives ``(baseline, candidate)``.
    """
    adapter = adapter or PythonKVCacheAdapter()
    baseline_output = None
    measurements = []
    for policy in (CachePolicy.NONE, CachePolicy.TRUNCATE, CachePolicy.LANDMARK):
        trial_cache = copy.deepcopy(cache)
        sequence_before = adapter.sequence_length(trial_cache)
        bytes_before = adapter.cache_bytes(trial_cache)
        started = time.perf_counter()
        controlled_cache = adapter.apply_policy(
            trial_cache,
            policy,
            synapse=synapse,
            query_states=query_states,
            keep_tokens=keep_tokens,
            keep_ratio=keep_ratio,
        )
        output = decode_fn(controlled_cache) if decode_fn is not None else None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if policy is CachePolicy.NONE:
            baseline_output = output
        output_equivalent = None
        if equivalent_fn is not None and decode_fn is not None:
            output_equivalent = bool(equivalent_fn(baseline_output, output))
        evaluation = evaluate_fn(output) if evaluate_fn is not None else None
        measurements.append(CachePolicyMeasurement(
            policy=policy.value,
            elapsed_ms=elapsed_ms,
            bytes_before=bytes_before,
            bytes_after=adapter.cache_bytes(controlled_cache),
            sequence_before=sequence_before,
            sequence_after=adapter.sequence_length(controlled_cache),
            output_equivalent=output_equivalent,
            evaluation=evaluation,
        ))
    return measurements
