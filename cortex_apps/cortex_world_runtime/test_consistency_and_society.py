"""
Unit & Invariant Tests for Causal Consistency, 2PC Atomicity, and Event Wake Routing.
"""

from __future__ import annotations

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.causal_consistency import EpochWatermarkSubstrate
from cortex_apps.cortex_world_runtime.heterogeneous_agent_society import (
    instantiate_society,
)
from cortex_apps.cortex_world_runtime.fast_world_substrate import FastWorldSubstrate


def test_causal_consistency_cut():
    substrate = EpochWatermarkSubstrate(num_regions=4)
    substrate.populate(num_entities=100)

    # Cause in Region 0
    ok1, _, ev1 = substrate.commit_multi_shard_atomic(
        "agent_a", {0: 1}, {0: [("ent_000000", {"status": "DESTROYED"})]}
    )
    assert ok1 is True

    # Effect in Region 1
    ok2, _, ev2 = substrate.commit_multi_shard_atomic(
        "agent_b", {1: 1}, {1: [("ent_000001", {"status": "BLOCKED"})]},
        causal_cause_event=ev1,
    )
    assert ok2 is True

    substrate.advance_epoch()
    snap = substrate.acquire_causally_consistent_snapshot()

    # In a causally consistent cut, both cause and effect must be visible together
    st0 = snap.get_state("ent_000000")
    st1 = snap.get_state("ent_000001")
    assert st0["status"] == "DESTROYED"
    assert st1["status"] == "BLOCKED"


def test_multi_shard_2pc_all_or_nothing():
    substrate = EpochWatermarkSubstrate(num_regions=4)
    substrate.populate(num_entities=100)

    # Scenario: Version conflict on shard 2 causes abort on multi-shard transaction
    # Shard 0: version 1, Shard 1: version 1, Shard 2: version 2 (simulate stale expectation 1)
    substrate.regional_versions[2] = 2

    ok, msg, ev = substrate.commit_multi_shard_atomic(
        "tx_agent",
        expected_regional_versions={0: 1, 1: 1, 2: 1},  # stale version 1 on shard 2
        deltas_by_shard={
            0: [("ent_000000", {"resource_units": 999})],
            1: [("ent_000001", {"resource_units": 999})],
            2: [("ent_000002", {"resource_units": 999})],
        }
    )

    assert ok is False
    assert "version mismatch" in msg

    # Verify zero partial state modifications (Atomicity Invariant)
    assert "resource_units" not in substrate.entities["ent_000000"]
    assert "resource_units" not in substrate.entities["ent_000001"]
    assert "resource_units" not in substrate.entities["ent_000002"]
    assert substrate.entities["ent_000000"]["status"] == "NORMAL"


def test_heterogeneous_society_event_wake():
    substrate = FastWorldSubstrate(num_clusters=8)
    substrate.populate_synthetic_world(num_entities=1000, edges_per_entity=4)
    snapshot = substrate.current_snapshot()

    agents = instantiate_society(substrate, total_agents=128)
    assert len(agents) == 128

    # Target event on node ent_000010
    event_target = "ent_000010"
    impacted = set(snapshot.bfs(event_target, max_depth=2, max_nodes=10))
    impacted.add(event_target)

    woken = [a for a in agents if a.focus_entity_id in impacted]
    # Woken agents should be a small fraction of the total 128
    assert len(woken) < len(agents)
