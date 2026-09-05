"""Tests for the explicit inference cache-control boundary."""

from __future__ import annotations

import pytest
import torch

from cortex_core.cache_control import PythonKVCacheAdapter
from cortex_validation.cache_policy_benchmark import run_cache_policy_benchmark


class RecordingSynapse:
    def __init__(self):
        self.calls = []

    def update_kv_landmarks(self, cache, **kwargs):
        self.calls.append((cache, kwargs))

    def get_landmarks(self):
        return ((
            torch.zeros(1, 2, 4, 8),
            torch.zeros(1, 2, 4, 8),
        ),)


def _cache(sequence_length=12):
    return tuple(
        (
            torch.zeros(1, 2, sequence_length, 8),
            torch.zeros(1, 2, sequence_length, 8),
        )
        for _ in range(2)
    )


def test_python_adapter_reports_non_native_capabilities():
    adapter = PythonKVCacheAdapter()
    snapshot = adapter.snapshot(_cache())

    assert adapter.sequence_length(_cache()) == 12
    assert snapshot.backend == "huggingface-python-kv"
    assert snapshot.layer_count == 2
    assert snapshot.capabilities.native_paged is False
    assert snapshot.capabilities.in_place_mutation is False
    assert snapshot.as_dict()["capabilities"]["topology_compaction"] is True


def test_python_adapter_delegates_compaction_with_ratio():
    adapter = PythonKVCacheAdapter()
    synapse = RecordingSynapse()
    cache = _cache()

    compacted = adapter.compact(cache, synapse, keep_ratio=0.25)

    assert compacted[0][0].shape[2] == 4
    assert synapse.calls[0][0] is cache
    assert synapse.calls[0][1] == {"query_states": None, "keep_ratio": 0.25}


def test_policy_benchmark_reports_all_comparison_dimensions():
    adapter = PythonKVCacheAdapter()
    measurements = run_cache_policy_benchmark(
        _cache(12),
        adapter=adapter,
        synapse=RecordingSynapse(),
        keep_tokens=4,
        decode_fn=adapter.sequence_length,
        evaluate_fn=lambda output: {"sequence": output},
        equivalent_fn=lambda baseline, candidate: baseline == candidate,
    )

    assert [item.policy for item in measurements] == ["none", "truncate", "landmark"]
    assert measurements[0].sequence_after == 12
    assert measurements[1].sequence_after == 4
    assert measurements[2].sequence_after == 4
    assert measurements[0].output_equivalent is True
    assert measurements[1].output_equivalent is False
    assert all(item.bytes_before > 0 for item in measurements)
    assert all(item.evaluation is not None for item in measurements)


def test_python_adapter_does_not_claim_native_paged_control():
    with pytest.raises(NotImplementedError, match="native paged-KV control"):
        PythonKVCacheAdapter().require_native_paged()
