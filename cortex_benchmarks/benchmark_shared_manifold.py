"""Shared-manifold internal benchmark and demo harness.

Runs deterministic scenarios through the real prompt-context and runtime-refresh
paths using a probe engine with stubbed tokenizer/model. This isolates the
shared-memory behavior from full-model quality variance.

Also includes a small executable coding-repair slice that converts recalled
shared memory into concrete code edits and measures pass or fail outcomes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.async_delegate import _execute_code
from cortex_core.agent_cloud import PersistentAgentCloud
from cortex_engine import CortexEngine


@dataclass
class SharedManifoldScenario:
    name: str
    writer_agent: str
    writer_role: str
    writer_profile: str
    reader_agent: str
    reader_role: str
    reader_profile: str
    memories: List[str]
    task: str
    recent_text: str
    expected_terms: List[str]


@dataclass
class SharedManifoldRepairStep:
    name: str
    trigger_terms: List[str]
    old: str
    new: str


@dataclass
class SharedManifoldCodingTask:
    name: str
    writer_agent: str
    writer_role: str
    writer_profile: str
    reader_agent: str
    reader_role: str
    reader_profile: str
    memories: List[str]
    task: str
    recent_text: str
    guidance_terms: List[str]
    starter_code: str
    repair_steps: List[SharedManifoldRepairStep]
    tests: str
    acceptance_criteria: List[str] = field(default_factory=list)


@dataclass
class SharedManifoldRecallTask:
    name: str
    writer_agent: str
    writer_role: str
    writer_profile: str
    reader_agent: str
    reader_role: str
    reader_profile: str
    memories: List[str]
    question: str
    expected_terms: List[str]
    answer_format: str = ""
    expected_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class SharedManifoldWriterSession:
    agent_id: str
    role: str
    profile: str
    memories: List[str]


@dataclass
class SharedManifoldNecessityTask:
    name: str
    writer_sessions: List[SharedManifoldWriterSession]
    reader_agent: str
    reader_role: str
    reader_profile: str
    question: str
    expected_terms: List[str]
    answer_format: str = ""
    expected_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class SharedManifoldTopologyMemory:
    text: str
    region: str
    keywords: List[str] = field(default_factory=list)
    entity_refs: List[str] = field(default_factory=list)
    is_bridge: bool = False
    source: str = "topology_benchmark"
    node_type: str = "topology_memory"


@dataclass
class SharedManifoldTopologyTask:
    name: str
    writer_agent: str
    writer_role: str
    writer_profile: str
    reader_agent: str
    reader_role: str
    reader_profile: str
    memories: List[SharedManifoldTopologyMemory]
    query_text: str
    recent_text: str
    top_k: int
    target_region: str
    expected_texts: List[str]
    expected_component_count: int
    expected_active_region_size: int
    question: str = ""
    expected_terms: List[str] = field(default_factory=list)
    answer_format: str = ""
    expected_fields: Dict[str, str] = field(default_factory=dict)
    forbidden_regions: List[str] = field(default_factory=list)
    expected_bridge_texts: List[str] = field(default_factory=list)


@dataclass
class SharedManifoldEnergyReuseTask:
    name: str
    target_task: SharedManifoldCodingTask
    distractor_tasks: List[SharedManifoldCodingTask]
    primer_query: str
    followup_query: str
    primer_repeats: int = 2

    @property
    def task_board_tasks(self) -> List[SharedManifoldCodingTask]:
        return [self.target_task, *self.distractor_tasks]

    @property
    def expected_task_id(self) -> str:
        return self.target_task.name

    @property
    def reader_agent(self) -> str:
        return self.target_task.reader_agent

    @property
    def expected_patch_names(self) -> List[str]:
        return [step.name for step in self.target_task.repair_steps]


class DummyTokenBatch:
    def __init__(self, text: str):
        self.input_ids = torch.tensor([[len(text) % 17 + 1]], dtype=torch.long)


class DummyTokenizer:
    def __init__(self):
        self.calls: List[str] = []

    def __call__(self, text: str, return_tensors: str = "pt"):
        self.calls.append(text)
        return DummyTokenBatch(text)


class DummyOutput:
    def __init__(self, past_key_values):
        self.past_key_values = past_key_values


class DummyModel:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, input_ids, past_key_values=None, output_hidden_states: bool = False):
        self.calls.append({
            "input_ids": input_ids.clone(),
            "past_key_values": past_key_values,
        })
        return DummyOutput({"memory_steps": len(self.calls)})


def default_scenarios() -> List[SharedManifoldScenario]:
    return [
        SharedManifoldScenario(
            name="payment_retry",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks checkout reliability decisions.",
            reader_agent="coder",
            reader_role="coder",
            reader_profile="Implements checkout changes safely.",
            memories=[
                "Use idempotency keys on payment retries to avoid duplicate captures.",
                "Emit retry telemetry so duplicate payment attempts can be debugged quickly.",
            ],
            task="Implement payment retry safety in checkout.",
            recent_text="Need to stop duplicate capture when a flaky network retries payment.",
            expected_terms=["idempotency", "duplicate capture", "retry telemetry"],
        ),
        SharedManifoldScenario(
            name="schema_backfill",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks migration safety constraints.",
            reader_agent="reviewer",
            reader_role="reviewer",
            reader_profile="Reviews rollout safety.",
            memories=[
                "Backfills must keep old index names unique until cutover completes.",
                "Dual writes should remain enabled during the migration backfill window.",
            ],
            task="Review the migration rollout for index backfill safety.",
            recent_text="The rollout plan changes an index while the backfill is still running.",
            expected_terms=["old index names", "dual writes", "backfill"],
        ),
        SharedManifoldScenario(
            name="token_rotation",
            writer_agent="security",
            writer_role="security",
            writer_profile="Tracks auth and secret rotation incidents.",
            reader_agent="operator",
            reader_role="operator",
            reader_profile="Operates production rollout workflows.",
            memories=[
                "During token rotation, keep both old and new keys valid for one deployment window.",
                "Log the active key version with every auth failure for rollback triage.",
            ],
            task="Prepare the production token rotation runbook.",
            recent_text="Auth failures increased after rotating credentials during deployment.",
            expected_terms=["old and new keys", "key version", "rotation"],
        ),
    ]


def default_coding_tasks() -> List[SharedManifoldCodingTask]:
    return [
        SharedManifoldCodingTask(
            name="payment_retry_repair",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks checkout reliability decisions.",
            reader_agent="coder",
            reader_role="coder",
            reader_profile="Repairs checkout helpers safely.",
            memories=[
                "Keep idempotency keys stable across payment retries so duplicate captures collapse safely.",
                "Emit retry telemetry with the attempt count so flaky payment loops can be debugged.",
            ],
            task="Repair the checkout retry helper so retries are safe and observable.",
            recent_text="A flaky checkout path is issuing duplicate captures after retries.",
            guidance_terms=["idempotency keys", "duplicate captures", "retry telemetry", "attempt count"],
            starter_code=textwrap.dedent(
                """
                def build_retry_request(charge_id, attempt, idempotency_key, telemetry):
                    headers = {"X-Charge-Id": charge_id}
                    if attempt > 1:
                        headers["X-Retry-Attempt"] = str(attempt)
                    payload = {"headers": headers, "telemetry": dict(telemetry)}
                    return payload
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="add_idempotency_key",
                    trigger_terms=["idempotency keys", "duplicate captures"],
                    old='    headers = {"X-Charge-Id": charge_id}\n',
                    new='    headers = {"X-Charge-Id": charge_id, "Idempotency-Key": idempotency_key}\n',
                ),
                SharedManifoldRepairStep(
                    name="record_retry_telemetry",
                    trigger_terms=["retry telemetry", "attempt count"],
                    old='    payload = {"headers": headers, "telemetry": dict(telemetry)}\n',
                    new='    telemetry_payload = dict(telemetry)\n    telemetry_payload["retry_count"] = attempt\n    payload = {"headers": headers, "telemetry": telemetry_payload}\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                result = build_retry_request("ch_123", 2, "idem-42", {"source": "checkout"})
                assert result["headers"]["Idempotency-Key"] == "idem-42"
                assert result["headers"]["X-Retry-Attempt"] == "2"
                assert result["telemetry"]["retry_count"] == 2
                assert result["telemetry"]["source"] == "checkout"
                print("PASS")
                """
            ).strip(),
        ),
        SharedManifoldCodingTask(
            name="schema_backfill_repair",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks migration safety constraints.",
            reader_agent="reviewer",
            reader_role="reviewer",
            reader_profile="Repairs rollout helpers before migration cutover.",
            memories=[
                "Backfills must keep old index names unique until cutover completes.",
                "Dual writes should remain enabled during the migration backfill window.",
            ],
            task="Repair the migration cutover helper so the backfill window stays safe.",
            recent_text="The rollout flips reads to the new index before the backfill is finished.",
            guidance_terms=["old index names", "cutover", "dual writes", "backfill window"],
            starter_code=textwrap.dedent(
                """
                def build_cutover_plan(backfill_done, old_index, new_index):
                    plan = {"read_index": new_index, "write_index": new_index, "dual_write": False}
                    if not backfill_done:
                        plan["legacy_index"] = new_index
                    else:
                        plan["legacy_index"] = old_index
                    return plan
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="keep_dual_writes_during_backfill",
                    trigger_terms=["dual writes", "backfill window"],
                    old='    plan = {"read_index": new_index, "write_index": new_index, "dual_write": False}\n',
                    new='    plan = {"read_index": old_index if not backfill_done else new_index, "write_index": new_index, "dual_write": not backfill_done}\n',
                ),
                SharedManifoldRepairStep(
                    name="preserve_old_index_name",
                    trigger_terms=["old index names", "cutover"],
                    old='        plan["legacy_index"] = new_index\n',
                    new='        plan["legacy_index"] = old_index\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                inflight = build_cutover_plan(False, "orders_v1", "orders_v2")
                assert inflight["read_index"] == "orders_v1"
                assert inflight["write_index"] == "orders_v2"
                assert inflight["dual_write"] is True
                assert inflight["legacy_index"] == "orders_v1"

                complete = build_cutover_plan(True, "orders_v1", "orders_v2")
                assert complete["read_index"] == "orders_v2"
                assert complete["write_index"] == "orders_v2"
                assert complete["dual_write"] is False
                assert complete["legacy_index"] == "orders_v1"
                print("PASS")
                """
            ).strip(),
        ),
        SharedManifoldCodingTask(
            name="token_rotation_repair",
            writer_agent="security",
            writer_role="security",
            writer_profile="Tracks auth and secret rotation incidents.",
            reader_agent="operator",
            reader_role="operator",
            reader_profile="Repairs auth rollout helpers safely.",
            memories=[
                "During token rotation, keep both old and new keys valid for one deployment window.",
                "Log the active key version with every auth failure for rollback triage.",
            ],
            task="Repair the token rotation helpers so the rollout keeps compatibility and debuggability.",
            recent_text="Auth failures spiked right after rotating credentials during deployment.",
            guidance_terms=["old and new keys", "deployment window", "key version", "auth failure"],
            starter_code=textwrap.dedent(
                """
                def active_key_versions(old_key_version, new_key_version, deployment_window_open):
                    return [new_key_version]

                def build_auth_failure_event(active_key_version, reason):
                    return {"reason": reason}
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="keep_old_and_new_keys_during_window",
                    trigger_terms=["old and new keys", "deployment window"],
                    old='    return [new_key_version]\n',
                    new='    return [old_key_version, new_key_version] if deployment_window_open else [new_key_version]\n',
                ),
                SharedManifoldRepairStep(
                    name="log_key_version_on_failure",
                    trigger_terms=["key version", "auth failure"],
                    old='    return {"reason": reason}\n',
                    new='    return {"reason": reason, "key_version": active_key_version}\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                assert active_key_versions("k1", "k2", True) == ["k1", "k2"]
                assert active_key_versions("k1", "k2", False) == ["k2"]

                event = build_auth_failure_event("k2", "denied")
                assert event["reason"] == "denied"
                assert event["key_version"] == "k2"
                print("PASS")
                """
            ).strip(),
        ),
    ]


def default_real_coding_tasks() -> List[SharedManifoldCodingTask]:
    return [
        SharedManifoldCodingTask(
            name="retry_replay_token_simple",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks checkout reliability decisions.",
            reader_agent="coder",
            reader_role="coder",
            reader_profile="Repairs checkout helpers safely.",
            memories=[
                'Store the replay token under the "Replay-Safety-Token" header.',
            ],
            task="Repair the retry header helper so retries stay replay-safe.",
            recent_text="A flaky payment path is creating duplicate charges after retries.",
            guidance_terms=["Replay-Safety-Token"],
            starter_code=textwrap.dedent(
                """
                def build_retry_headers(charge_id, replay_token):
                    return {"X-Charge-Id": charge_id}
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="add_replay_safety_token_header",
                    trigger_terms=["Replay-Safety-Token"],
                    old='    return {"X-Charge-Id": charge_id}\n',
                    new='    return {"X-Charge-Id": charge_id, "Replay-Safety-Token": replay_token}\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                result = build_retry_headers("ch_123", "rp-42")
                assert result["X-Charge-Id"] == "ch_123"
                assert result["Replay-Safety-Token"] == "rp-42"
                print("PASS")
                """
            ).strip(),
            acceptance_criteria=[
                'Preserve the exact signature `def build_retry_headers(charge_id, replay_token):`.',
                'Keep the existing charge id header unchanged.',
                'Add the replay-safety header described in shared context.',
            ],
        ),
        SharedManifoldCodingTask(
            name="rotation_key_field",
            writer_agent="security",
            writer_role="security",
            writer_profile="Tracks auth and secret rotation incidents.",
            reader_agent="operator",
            reader_role="operator",
            reader_profile="Repairs auth rollout helpers safely.",
            memories=[
                'During rotation incidents, store the active key version under the "rotation_key_version" field.',
            ],
            task="Repair the auth failure event helper so rotation incidents are debuggable.",
            recent_text="Operators cannot tell which key version caused a denied auth event.",
            guidance_terms=["rotation_key_version"],
            starter_code=textwrap.dedent(
                """
                def build_auth_failure_event(active_key_version, reason):
                    return {"reason": reason}
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="add_rotation_key_version_field",
                    trigger_terms=["rotation_key_version"],
                    old='    return {"reason": reason}\n',
                    new='    return {"reason": reason, "rotation_key_version": active_key_version}\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                event = build_auth_failure_event("k2", "denied")
                assert event["reason"] == "denied"
                assert event["rotation_key_version"] == "k2"
                print("PASS")
                """
            ).strip(),
            acceptance_criteria=[
                'Preserve the exact signature `def build_auth_failure_event(active_key_version, reason):`.',
                'Keep the original reason field unchanged.',
                'Add the exact key-version field described in shared context.',
            ],
        ),
        SharedManifoldCodingTask(
            name="session_trace_field",
            writer_agent="ops",
            writer_role="ops",
            writer_profile="Tracks incident audit requirements.",
            reader_agent="maintainer",
            reader_role="maintainer",
            reader_profile="Repairs audit helpers safely.",
            memories=[
                'Store the session id under the "session_trace_id" field for audit replay.',
            ],
            task="Repair the session audit event helper so incident replay is possible.",
            recent_text="Support cannot correlate the action with the original session.",
            guidance_terms=["session_trace_id"],
            starter_code=textwrap.dedent(
                """
                def build_session_audit_event(session_id, action):
                    return {"action": action}
                """
            ).strip(),
            repair_steps=[
                SharedManifoldRepairStep(
                    name="add_session_trace_id_field",
                    trigger_terms=["session_trace_id"],
                    old='    return {"action": action}\n',
                    new='    return {"action": action, "session_trace_id": session_id}\n',
                ),
            ],
            tests=textwrap.dedent(
                """
                event = build_session_audit_event("sess-9", "approve")
                assert event["action"] == "approve"
                assert event["session_trace_id"] == "sess-9"
                print("PASS")
                """
            ).strip(),
            acceptance_criteria=[
                'Preserve the exact signature `def build_session_audit_event(session_id, action):`.',
                'Keep the original action field unchanged.',
                'Add the audit replay field described in shared context.',
            ],
        ),
    ]


def default_real_recall_tasks() -> List[SharedManifoldRecallTask]:
    return [
        SharedManifoldRecallTask(
            name="jenny_boots_red",
            writer_agent="scribe",
            writer_role="scribe",
            writer_profile="Records short factual notes exactly.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers short factual questions from transferred shared context.",
            memories=[
                "Jenny's boots are red.",
            ],
            question="What color were Jenny's boots?",
            expected_terms=["red"],
        ),
        SharedManifoldRecallTask(
            name="jenny_boots_locker",
            writer_agent="scribe",
            writer_role="scribe",
            writer_profile="Records short factual notes exactly.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers short factual questions from transferred shared context.",
            memories=[
                "Jenny's boots are red.",
                "Jenny left her boots beside locker 14 after the rain.",
            ],
            question="What color were Jenny's boots and where were they left?",
            expected_terms=["red", "locker 14"],
            answer_format="color=VALUE; where=PLACE",
            expected_fields={"color": "red", "where": "locker 14"},
        ),
        SharedManifoldRecallTask(
            name="cedar_compass_chain",
            writer_agent="scribe",
            writer_role="scribe",
            writer_profile="Records short factual notes exactly.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers short factual questions from transferred shared context.",
            memories=[
                "Eli handed Nora the bronze compass before the archive audit.",
                "Nora hid the bronze compass inside the cedar drawer.",
            ],
            question="Where did Nora hide the bronze compass Eli handed her?",
            expected_terms=["cedar drawer"],
            answer_format="where=PLACE",
            expected_fields={"where": "cedar drawer"},
        ),
        SharedManifoldRecallTask(
            name="silver_keycard_chain",
            writer_agent="scribe",
            writer_role="scribe",
            writer_profile="Records short factual notes exactly.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers short factual questions from transferred shared context.",
            memories=[
                "Priya took the silver keycard at breakfast.",
                "Before lunch, Priya gave the silver keycard to Omar.",
                "Omar locked the silver keycard in locker 12.",
            ],
            question="Who had the silver keycard right before it was locked away, and where is it now?",
            expected_terms=["omar", "locker 12"],
            answer_format="who=NAME; where=PLACE",
            expected_fields={"who": "omar", "where": "locker 12"},
        ),
        SharedManifoldRecallTask(
            name="color_distractor_mix",
            writer_agent="scribe",
            writer_role="scribe",
            writer_profile="Records short factual notes exactly.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers short factual questions from transferred shared context.",
            memories=[
                "Marta's coat is green.",
                "Jenny's boots are red.",
                "Theo's scarf is blue.",
                "Nina's umbrella is yellow.",
            ],
            question="What color were Jenny's boots, not Marta's coat or Theo's scarf?",
            expected_terms=["red"],
        ),
    ]


def default_real_necessity_tasks() -> List[SharedManifoldNecessityTask]:
    return [
        SharedManifoldNecessityTask(
            name="vx17_badge_locker",
            writer_sessions=[
                SharedManifoldWriterSession(
                    agent_id="intake_a",
                    role="intake",
                    profile="Records intake ticket metadata exactly.",
                    memories=[
                        "Vault ticket VX-17 uses the badge color teal.",
                    ],
                ),
                SharedManifoldWriterSession(
                    agent_id="intake_b",
                    role="locker",
                    profile="Records locker placement updates exactly.",
                    memories=[
                        "Vault ticket VX-17 was sealed in locker 42.",
                    ],
                ),
            ],
            reader_agent="auditor",
            reader_role="auditor",
            reader_profile="Answers only from transferred cross-session shared context.",
            question="For vault ticket VX-17, what badge color does it use and where was it sealed?",
            expected_terms=["teal", "locker 42"],
            answer_format="color=VALUE; where=PLACE",
            expected_fields={"color": "teal", "where": "locker 42"},
        ),
        SharedManifoldNecessityTask(
            name="rq91_parking_chain",
            writer_sessions=[
                SharedManifoldWriterSession(
                    agent_id="handoff_a",
                    role="handoff",
                    profile="Tracks courier possession changes exactly.",
                    memories=[
                        "Courier RQ-91 started with Mina.",
                    ],
                ),
                SharedManifoldWriterSession(
                    agent_id="handoff_b",
                    role="handoff",
                    profile="Tracks courier possession changes exactly.",
                    memories=[
                        "Before the tunnel check, Mina handed courier RQ-91 to Jules.",
                    ],
                ),
                SharedManifoldWriterSession(
                    agent_id="handoff_c",
                    role="parking",
                    profile="Tracks final parking locations exactly.",
                    memories=[
                        "Jules parked courier RQ-91 in bay 6.",
                        "Courier RQ-73 stayed with Oren in bay 3.",
                    ],
                ),
            ],
            reader_agent="dispatcher",
            reader_role="dispatcher",
            reader_profile="Answers routing questions only from shared cross-session context.",
            question="Who had courier RQ-91 right before it was parked, and where is it now?",
            expected_terms=["jules", "bay 6"],
            answer_format="who=NAME; where=PLACE",
            expected_fields={"who": "jules", "where": "bay 6"},
        ),
        SharedManifoldNecessityTask(
            name="cedar88_drawer",
            writer_sessions=[
                SharedManifoldWriterSession(
                    agent_id="ledger_a",
                    role="ledger",
                    profile="Records specimen ownership exactly.",
                    memories=[
                        "Specimen Cedar-88 belongs to Nora.",
                    ],
                ),
                SharedManifoldWriterSession(
                    agent_id="ledger_b",
                    role="ledger",
                    profile="Records specimen storage moves exactly.",
                    memories=[
                        "Nora hid specimen Cedar-88 inside drawer cedar-3.",
                        "Specimen Maple-20 stayed on tray 9.",
                    ],
                ),
            ],
            reader_agent="archivist",
            reader_role="archivist",
            reader_profile="Answers archive questions only from shared cross-session context.",
            question="Where did Nora hide specimen Cedar-88?",
            expected_terms=["drawer cedar-3"],
            answer_format="where=PLACE",
            expected_fields={"where": "drawer cedar-3"},
        ),
    ]


def default_topology_tasks() -> List[SharedManifoldTopologyTask]:
    return [
        SharedManifoldTopologyTask(
            name="payment_region_isolation",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks checkout reliability decisions.",
            reader_agent="coder",
            reader_role="coder",
            reader_profile="Implements changes safely.",
            memories=[
                SharedManifoldTopologyMemory(
                    text="Payment retries must reuse idempotency key PX-17.",
                    region="payments",
                    keywords=["payment", "retry", "idempotency"],
                    entity_refs=["payment_flow", "px17"],
                ),
                SharedManifoldTopologyMemory(
                    text="PX-17 replay seal keeps the same charge lineage in checkout.",
                    region="payments",
                    keywords=["checkout", "replay", "charge", "lineage"],
                    entity_refs=["payment_flow", "px17"],
                ),
                SharedManifoldTopologyMemory(
                    text="Retry the cargo handoff when duplicate bridge manifests appear.",
                    region="shipping",
                    keywords=["retry", "cargo", "duplicate", "handoff"],
                    entity_refs=["cargo_flow", "bridge_scan"],
                ),
                SharedManifoldTopologyMemory(
                    text="Bridge handoff retries merge duplicate manifests for cargo.",
                    region="shipping",
                    keywords=["duplicate", "bridge", "cargo", "handoff"],
                    entity_refs=["cargo_flow", "bridge_scan"],
                ),
            ],
            query_text="Implement payment retry idempotency with PX-17 so duplicate charges collapse safely.",
            recent_text="Checkout retries are replaying the same payment path.",
            top_k=2,
            target_region="payments",
            expected_texts=[
                "Payment retries must reuse idempotency key PX-17.",
                "PX-17 replay seal keeps the same charge lineage in checkout.",
            ],
            expected_component_count=2,
            expected_active_region_size=2,
            forbidden_regions=["shipping"],
        ),
        SharedManifoldTopologyTask(
            name="bridge_recall_vs_flat_leakage",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks handoff continuity across regions.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Needs the correct handoff chain.",
            memories=[
                SharedManifoldTopologyMemory(
                    text="Alpha vault stores the original field notes.",
                    region="alpha",
                    keywords=["alpha", "vault", "field"],
                    entity_refs=["alpha", "vault"],
                ),
                SharedManifoldTopologyMemory(
                    text="Relay PX-9 findings across the beta route seam.",
                    region="alpha_beta",
                    keywords=["alpha", "beta", "route", "seam"],
                    entity_refs=["alpha", "beta", "handoff"],
                    is_bridge=True,
                ),
                SharedManifoldTopologyMemory(
                    text="Beta checkpoint packet closes the route.",
                    region="beta",
                    keywords=["beta", "route", "checkpoint", "packet"],
                    entity_refs=["beta", "handoff"],
                ),
                SharedManifoldTopologyMemory(
                    text="Beta checkpoint archive packet failed in cold storage.",
                    region="archive",
                    keywords=["archive", "storage", "checksum"],
                    entity_refs=["archive", "checksum"],
                ),
            ],
            query_text="Need the beta route checkpoint packet details.",
            recent_text="Reader only knows the final beta checkpoint request.",
            top_k=2,
            target_region="beta",
            expected_texts=[
                "Beta checkpoint packet closes the route.",
                "Relay PX-9 findings across the beta route seam.",
            ],
            expected_component_count=2,
            expected_active_region_size=3,
            forbidden_regions=["archive"],
            expected_bridge_texts=["Relay PX-9 findings across the beta route seam."],
        ),
    ]


def default_real_topology_tasks() -> List[SharedManifoldTopologyTask]:
    return [
        SharedManifoldTopologyTask(
            name="real_payment_retry_fields",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks checkout reliability decisions.",
            reader_agent="coder",
            reader_role="coder",
            reader_profile="Answers exact field questions from shared context.",
            memories=[
                SharedManifoldTopologyMemory(
                    text="Checkout ticket PX-17 uses retry_header=X-Payment-Retry-Key.",
                    region="payments",
                    keywords=["checkout", "ticket", "px17", "retry_header"],
                    entity_refs=["checkout", "px17", "payment_retry"],
                ),
                SharedManifoldTopologyMemory(
                    text="Checkout ticket PX-17 uses seal=replay_token_px17.",
                    region="payments",
                    keywords=["px17", "replay", "seal", "field"],
                    entity_refs=["checkout", "px17", "payment_retry"],
                ),
                SharedManifoldTopologyMemory(
                    text="Cargo ticket PX-17 uses retry_header=X-Bridge-Retry-Key.",
                    region="shipping",
                    keywords=["cargo", "ticket", "px17", "retry_header"],
                    entity_refs=["cargo", "px17", "bridge_retry"],
                ),
                SharedManifoldTopologyMemory(
                    text="Cargo manifest field for PX-17 is bridge_manifest_px17.",
                    region="shipping",
                    keywords=["cargo", "manifest", "px17", "field"],
                    entity_refs=["cargo", "px17", "bridge_retry"],
                ),
            ],
            query_text="For checkout ticket PX-17, what retry_header and seal should be used?",
            recent_text="Checkout retries are replaying the same payment path.",
            top_k=2,
            target_region="payments",
            expected_texts=[
                "Checkout ticket PX-17 uses retry_header=X-Payment-Retry-Key.",
                "Checkout ticket PX-17 uses seal=replay_token_px17.",
            ],
            expected_component_count=2,
            expected_active_region_size=2,
            question="For checkout ticket PX-17, what retry_header and seal should be used?",
            expected_terms=["x-payment-retry-key", "replay_token_px17"],
            answer_format="retry_header=VALUE; seal=VALUE",
            expected_fields={"retry_header": "X-Payment-Retry-Key", "seal": "replay_token_px17"},
            forbidden_regions=["shipping"],
        ),
        SharedManifoldTopologyTask(
            name="real_bridge_route_chain",
            writer_agent="planner",
            writer_role="planner",
            writer_profile="Tracks bridge handoff continuity.",
            reader_agent="reader",
            reader_role="reader",
            reader_profile="Answers exact route-chain questions from shared context.",
            memories=[
                SharedManifoldTopologyMemory(
                    text="Packet PX-9 started in the alpha vault.",
                    region="alpha",
                    keywords=["packet", "px9", "alpha", "vault"],
                    entity_refs=["packet_chain", "alpha_route"],
                ),
                SharedManifoldTopologyMemory(
                    text="Packet PX-9 crosses seam_route=beta-seam before the final checkpoint.",
                    region="alpha_beta",
                    keywords=["packet", "px9", "crosses", "seam_route"],
                    entity_refs=["packet_chain", "alpha_route", "beta_route"],
                    is_bridge=True,
                ),
                SharedManifoldTopologyMemory(
                    text="Packet PX-9 finishes at final_location=cedar-checkpoint.",
                    region="beta",
                    keywords=["packet", "px9", "finishes", "final_location"],
                    entity_refs=["beta_route", "cedar_checkpoint"],
                ),
                SharedManifoldTopologyMemory(
                    text="Archive mirror PX-9 stored final_location=cold-archive.",
                    region="archive",
                    keywords=["archive", "mirror", "px9", "final_location"],
                    entity_refs=["archive_mirror", "cold_archive"],
                ),
            ],
            query_text="For packet PX-9, what seam_route does it cross and what final_location does it finish at?",
            recent_text="Reader only knows the final packet PX-9 checkpoint request.",
            top_k=2,
            target_region="beta",
            expected_texts=[
                "Packet PX-9 crosses seam_route=beta-seam before the final checkpoint.",
                "Packet PX-9 finishes at final_location=cedar-checkpoint.",
            ],
            expected_component_count=2,
            expected_active_region_size=3,
            question="For packet PX-9, what seam_route does it cross and what final_location does it finish at?",
            expected_terms=["beta-seam", "cedar-checkpoint"],
            answer_format="seam_route=VALUE; final_location=VALUE",
            expected_fields={"seam_route": "beta-seam", "final_location": "cedar-checkpoint"},
            forbidden_regions=["archive"],
            expected_bridge_texts=["Packet PX-9 crosses seam_route=beta-seam before the final checkpoint."],
        ),
    ]


def _make_real_energy_reuse_board_task(
    *,
    task_name: str,
    helper_name: str,
    focus_word: str,
    field_name: str,
) -> SharedManifoldCodingTask:
    signature = f"def build_{helper_name}_event(route_value, attempt):"
    return SharedManifoldCodingTask(
        name=task_name,
        writer_agent="planner",
        writer_role="planner",
        writer_profile="Tracks repair directives that must survive repeated handoffs.",
        reader_agent="coder",
        reader_role="coder",
        reader_profile="Repairs helpers safely from shared task-board context.",
        memories=[f'Store the exact value under the "{field_name}" field.'],
        task=f"Repair the {helper_name} helper so the missing {focus_word} field is restored.",
        recent_text=f"The {helper_name} helper dropped the {focus_word} metadata during retries.",
        guidance_terms=[field_name],
        starter_code=textwrap.dedent(
            f"""
            def build_{helper_name}_event(route_value, attempt):
                return {{"attempt": attempt}}
            """
        ).strip(),
        repair_steps=[
            SharedManifoldRepairStep(
                name=f"add_{field_name}",
                trigger_terms=[field_name],
                old='    return {"attempt": attempt}\n',
                new=f'    return {{"attempt": attempt, "{field_name}": route_value}}\n',
            )
        ],
        tests=textwrap.dedent(
            f"""
            event = build_{helper_name}_event("value-7", 2)
            assert event["attempt"] == 2
            assert event["{field_name}"] == "value-7"
            print("PASS")
            """
        ).strip(),
        acceptance_criteria=[
            f"Preserve the exact signature `{signature}`.",
            "Keep the existing attempt field unchanged.",
            f"Add the exact `{field_name}` field from shared context.",
        ],
    )


def _build_real_energy_reuse_case(
    *,
    name: str,
    helper_name: str,
    target_word: str,
    distractor_word: str,
    extra_words: List[str],
) -> SharedManifoldEnergyReuseTask:
    target_field = f"{target_word}_{helper_name}_field"
    target_task = _make_real_energy_reuse_board_task(
        task_name=f"{helper_name}_target",
        helper_name=helper_name,
        focus_word=target_word,
        field_name=target_field,
    )
    distractor_tasks = [
        _make_real_energy_reuse_board_task(
            task_name=f"{helper_name}_{focus_word}",
            helper_name=helper_name,
            focus_word=focus_word,
            field_name=f"{focus_word}_{helper_name}_field",
        )
        for focus_word in [distractor_word, *extra_words]
    ]
    return SharedManifoldEnergyReuseTask(
        name=name,
        target_task=target_task,
        distractor_tasks=distractor_tasks,
        primer_query=f"Repair the helper so {target_field} is restored.",
        followup_query=f"Repair the helper so the {distractor_word} {target_word} field is restored again.",
        primer_repeats=2,
    )


def default_real_energy_reuse_tasks() -> List[SharedManifoldEnergyReuseTask]:
    return [
        _build_real_energy_reuse_case(
            name="gateway_route_reuse",
            helper_name="gateway",
            target_word="route",
            distractor_word="checkpoint",
            extra_words=["metadata", "retry"],
        ),
        _build_real_energy_reuse_case(
            name="handoff_trail_reuse",
            helper_name="handoff",
            target_word="trail",
            distractor_word="marker",
            extra_words=["audit", "token"],
        ),
        _build_real_energy_reuse_case(
            name="ledger_seal_reuse",
            helper_name="ledger",
            target_word="seal",
            distractor_word="checkpoint",
            extra_words=["audit", "token"],
        ),
    ]


def build_probe_engine(enable_shared_manifold: bool = True, hidden_dim: int = 32) -> CortexEngine:
    engine = object.__new__(CortexEngine)
    engine.device = "cpu"
    engine.tokenizer = DummyTokenizer()
    engine.model = DummyModel()
    engine.agent_cloud = PersistentAgentCloud(
        hidden_dim=hidden_dim,
        device="cpu",
        shared_manifold_capacity=16,
    )
    engine.shared_manifold_enabled = enable_shared_manifold
    engine.shared_manifold_energy_feedback_enabled = False
    engine.agent_cloud.shared_energy_feedback_enabled = False
    engine.shared_manifold_trace = []
    engine.shared_manifold_prompt_hits = 0
    engine.shared_manifold_prompt_misses = 0
    engine.shared_manifold_runtime_refreshes = 0
    engine.shared_manifold_nodes_consumed = 0
    engine.shared_manifold_refresh_interval = 4
    engine.shared_manifold_refresh_top_k = 2
    return engine


def _seed_probe_engine(engine: CortexEngine, item: Any):
    engine.agent_cloud.ensure_agent(
        item.writer_agent,
        role=item.writer_role,
        profile=item.writer_profile,
    )
    engine.agent_cloud.ensure_agent(
        item.reader_agent,
        role=item.reader_role,
        profile=item.reader_profile,
    )

    for index, memory in enumerate(item.memories):
        engine.agent_cloud.remember_text(
            agent_id=item.writer_agent,
            text=memory,
            role=item.writer_role,
            source="scenario",
            metadata={"sequence_index": index},
        )


def _collect_shared_state(
    engine: CortexEngine,
    *,
    task: str,
    recent_text: str,
    reader_agent: str,
) -> Dict[str, Any]:
    prompt_context = engine._build_shared_manifold_context(task)
    used_texts: set[str] = set()
    _, refresh_count = engine._maybe_refresh_shared_manifold(
        base_prompt=task,
        recent_text=recent_text,
        used_texts=used_texts,
        past_key_values=None,
        agent_id=reader_agent,
    )
    metrics = engine.get_shared_manifold_metrics()
    trace = engine.get_shared_manifold_trace()
    shared_calls = [text for text in engine.tokenizer.calls if "[Shared:" in text]
    return {
        "prompt_context": prompt_context,
        "refresh_count": refresh_count,
        "metrics": metrics,
        "trace": trace,
        "shared_calls": shared_calls,
    }


def _seed_topology_probe_engine(engine: CortexEngine, task: SharedManifoldTopologyTask):
    engine.agent_cloud.ensure_agent(
        task.writer_agent,
        role=task.writer_role,
        profile=task.writer_profile,
    )
    engine.agent_cloud.ensure_agent(
        task.reader_agent,
        role=task.reader_role,
        profile=task.reader_profile,
    )

    for index, memory in enumerate(task.memories):
        engine.agent_cloud.remember_shared_text(
            text=memory.text,
            score=1.0,
            source=memory.source,
            node_type=memory.node_type,
            agent_id=task.writer_agent,
            metadata={
                "sequence_index": index,
                "keywords": list(memory.keywords),
                "entity_refs": list(memory.entity_refs),
                "benchmark_region": memory.region,
                "benchmark_is_bridge": bool(memory.is_bridge),
            },
        )


def _query_flat_shared_manifold(
    cloud: PersistentAgentCloud,
    *,
    query_text: str,
    top_k: int,
    agent_id: Optional[str] = None,
) -> List[Any]:
    with cloud._shared_lock:
        nodes = [
            node
            for node in cloud._shared_nodes
            if node.node_type != 'projection_summary'
        ]
    if not nodes:
        return []

    query_embedding = cloud.encode_text(query_text)
    matrix = torch.stack([cloud._prepare_embedding(node.embedding) for node in nodes], dim=0)
    sims = torch.matmul(matrix, query_embedding)
    centrality = cloud._shared_centrality(matrix)
    confidence = torch.tensor([node.score for node in nodes], dtype=sims.dtype)
    lexical = torch.tensor(
        [cloud._token_overlap(query_text, node.text) for node in nodes],
        dtype=sims.dtype,
    )
    same_agent = torch.tensor(
        [0.05 if agent_id and node.agent_id == agent_id else 0.0 for node in nodes],
        dtype=sims.dtype,
    )
    semantic_enabled = cloud.tokenizer is not None and cloud.embed_layer is not None and cloud._proj is not None
    if semantic_enabled:
        combined = sims + 0.35 * lexical + 0.05 * confidence + 0.05 * centrality + same_agent
    else:
        combined = 0.70 * lexical + 0.10 * confidence + 0.05 * centrality + same_agent

    selected: List[Any] = []
    for idx in torch.argsort(combined, descending=True).tolist():
        if float(combined[idx].item()) < 0.10:
            continue
        selected.append(nodes[int(idx)])
        if len(selected) >= top_k:
            break
    return selected


def _evaluate_topology_nodes(task: SharedManifoldTopologyTask, nodes: List[Any]) -> Dict[str, Any]:
    selected_texts = [node.text for node in nodes]
    selected_regions = [str((node.metadata or {}).get("benchmark_region", "")) for node in nodes]
    leakage_texts = [
        node.text
        for node in nodes
        if str((node.metadata or {}).get("benchmark_region", "")) in task.forbidden_regions
    ]
    bridge_hits = [
        node.text
        for node in nodes
        if bool((node.metadata or {}).get("benchmark_is_bridge", False))
    ]
    matched_expected = [text for text in task.expected_texts if text in selected_texts]
    missing_expected = [text for text in task.expected_texts if text not in selected_texts]
    matched_bridge = [text for text in task.expected_bridge_texts if text in selected_texts]

    return {
        "selected_texts": selected_texts,
        "selected_regions": selected_regions,
        "matched_expected": matched_expected,
        "missing_expected": missing_expected,
        "matched_bridge": matched_bridge,
        "bridge_hits": bridge_hits,
        "bridge_recall": float(len(matched_bridge) / max(len(task.expected_bridge_texts), 1)) if task.expected_bridge_texts else 1.0,
        "leakage_texts": leakage_texts,
        "leakage_count": len(leakage_texts),
        "leakage_rate": float(len(leakage_texts) / max(len(nodes), 1)) if nodes else 0.0,
        "expected_recall": float(len(matched_expected) / max(len(task.expected_texts), 1)),
        "passed": not missing_expected and not leakage_texts,
    }


def run_topology_task(task: SharedManifoldTopologyTask) -> Dict[str, Any]:
    engine = build_probe_engine(enable_shared_manifold=True)
    _seed_topology_probe_engine(engine, task)

    topology_nodes, topology_view, active_component = engine.agent_cloud._select_shared_nodes(
        query_text=task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    flat_nodes = _query_flat_shared_manifold(
        engine.agent_cloud,
        query_text=task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    topology_context = engine.agent_cloud.build_shared_context(
        task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    stats = engine.agent_cloud.shared_manifold_stats()

    topology_eval = _evaluate_topology_nodes(task, topology_nodes)
    flat_eval = _evaluate_topology_nodes(task, flat_nodes)
    topology_eval["active_region_size"] = len(active_component)
    topology_eval["region_count"] = len(topology_view.components)
    topology_eval["component_count_match"] = int(stats.get("component_count", 0)) == task.expected_component_count
    topology_eval["active_region_match"] = len(active_component) == task.expected_active_region_size

    return {
        "name": task.name,
        "query_text": task.query_text,
        "top_k": task.top_k,
        "bridge_expected_count": len(task.expected_bridge_texts),
        "shared_manifold_stats": stats,
        "prompt_context": topology_context,
        "topology": topology_eval,
        "flat": flat_eval,
    }


def _summarize_topology(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    bridge_items = [item for item in results if int(item.get("bridge_expected_count", 0)) > 0]
    bridge_denominator = max(len(bridge_items), 1)
    return {
        "task_count": len(results),
        "component_accuracy_rate": sum(int(item["topology"]["component_count_match"]) for item in results) / count,
        "active_region_accuracy_rate": sum(int(item["topology"]["active_region_match"]) for item in results) / count,
        "topology_expected_recall_rate": sum(item["topology"]["expected_recall"] for item in results) / count,
        "flat_expected_recall_rate": sum(item["flat"]["expected_recall"] for item in results) / count,
        "topology_bridge_recall_rate": sum(item["topology"]["bridge_recall"] for item in bridge_items) / bridge_denominator,
        "flat_bridge_recall_rate": sum(item["flat"]["bridge_recall"] for item in bridge_items) / bridge_denominator,
        "topology_leakage_rate": sum(item["topology"]["leakage_rate"] for item in results) / count,
        "flat_leakage_rate": sum(item["flat"]["leakage_rate"] for item in results) / count,
        "topology_win_rate": sum(
            int(
                item["topology"]["expected_recall"] >= item["flat"]["expected_recall"]
                and item["topology"]["bridge_recall"] >= item["flat"]["bridge_recall"]
                and item["topology"]["leakage_rate"] <= item["flat"]["leakage_rate"]
            )
            for item in results
        ) / count,
    }


def compare_topology_slice(
    tasks: Optional[Iterable[SharedManifoldTopologyTask]] = None,
) -> Dict[str, Any]:
    task_list = list(tasks or default_topology_tasks())
    results = [run_topology_task(task) for task in task_list]
    return {
        "aggregate": _summarize_topology(results),
        "tasks": results,
    }


def run_topology_demo(name: str = "payment_region_isolation") -> Dict[str, Any]:
    task_map = {task.name: task for task in default_topology_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown topology task: {name}")
    result = run_topology_task(task_map[name])
    result["topology_task"] = name
    return result


def _lower_matches(text: str, expected_terms: Iterable[str]) -> List[str]:
    lowered = text.lower()
    return [term for term in expected_terms if term.lower() in lowered]


def _contains_terms(text: str, expected_terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in expected_terms)


def _parse_answer_fields(text: str, expected_keys: Optional[Iterable[str]] = None) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for chunk in re.split(r"[;\n]+", text):
        stripped_chunk = chunk.strip()
        if not stripped_chunk:
            continue
        if "=" in stripped_chunk:
            key, value = stripped_chunk.split("=", 1)
            normalized_key = key.strip().lower()
            normalized_value = value.strip().strip(" .")
            if normalized_key and normalized_value:
                parsed[normalized_key] = normalized_value
            continue
        for key in expected_keys or []:
            lowered_key = key.lower()
            lowered_chunk = stripped_chunk.lower()
            if not lowered_chunk.startswith(lowered_key):
                continue
            value = stripped_chunk[len(key):].lstrip(" =:").strip().strip(" .")
            if value:
                parsed[lowered_key] = value
            break
    return parsed


def _normalize_field_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip(" .").lower())


def _evaluate_recall_answer(task: SharedManifoldRecallTask, answer: str) -> Dict[str, Any]:
    parsed_fields = _parse_answer_fields(answer, task.expected_fields.keys())
    matched_terms = _lower_matches(answer, task.expected_terms)
    missing_fields = [
        key
        for key, expected_value in task.expected_fields.items()
        if _normalize_field_value(parsed_fields.get(key, "")) != _normalize_field_value(expected_value)
    ]
    if task.expected_fields:
        passed = not missing_fields
    else:
        passed = _contains_terms(answer, task.expected_terms)
    return {
        "matched_terms": matched_terms,
        "parsed_fields": parsed_fields,
        "missing_fields": missing_fields,
        "passed": passed,
    }


def _augment_recall_context_with_focus(shared_context: str, prompt_nodes: List[Any]) -> str:
    if not shared_context.startswith("[Shared Manifold]"):
        return shared_context

    def node_text(node: Any) -> str:
        if isinstance(node, str):
            return node.strip()
        return str(getattr(node, "text", "")).strip()

    def node_label(node: Any) -> str:
        if isinstance(node, str):
            return "recall memory"
        return str(getattr(node, "node_type", "recall_memory")).replace("_", " ")

    focus_texts: List[str] = []
    seen: set[str] = set()
    for node in prompt_nodes:
        normalized = node_text(node)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        focus_texts.append(normalized)

    lines = shared_context.splitlines()
    header = lines[:2]
    rendered_nodes: List[str] = []
    for node in prompt_nodes:
        text = node_text(node)
        if not text:
            continue
        label = node_label(node)
        rendered_nodes.append(f"- [{label}] {text}")

    if len(focus_texts) <= 1:
        return "\n".join(header + rendered_nodes)

    return "\n".join(
        header
        + ["- [recall focus] Relevant facts, highest priority first: " + " | ".join(focus_texts)]
        + rendered_nodes
    )


def _synthesize_candidate_code(task: SharedManifoldCodingTask, guidance_text: str):
    code = task.starter_code
    applied_repairs: List[str] = []
    for repair_step in task.repair_steps:
        if not _contains_terms(guidance_text, repair_step.trigger_terms):
            continue

        old_text = repair_step.old
        new_text = repair_step.new
        if old_text in code:
            code = code.replace(old_text, new_text, 1)
            applied_repairs.append(repair_step.name)
            continue

        old_text = repair_step.old.rstrip("\n")
        if old_text and old_text in code:
            code = code.replace(old_text, repair_step.new.rstrip("\n"), 1)
            applied_repairs.append(repair_step.name)
    return code, applied_repairs


def _evaluate_candidate_code(candidate_code: str, tests: str) -> Dict[str, Any]:
    script = f"{candidate_code.rstrip()}\n\n{tests.rstrip()}\n"
    result = _execute_code(script, timeout=10.0)
    return {
        "passed": result.success,
        "output": result.output,
        "error": result.error,
    }


def _extract_exact_signature(code: str) -> str:
    match = re.search(r"^def\s+[^\n]+", code, flags=re.MULTILINE)
    return match.group(0).strip() if match else ""


def _extract_python_code(raw_output: str) -> str:
    if not raw_output.strip():
        return ""

    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw_output, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\s*\(|class\s+\w+\s*[:(])", raw_output)
    if start:
        candidate = raw_output[start.start():].strip()
        candidate = candidate.split("```", 1)[0].strip()
        return candidate

    return raw_output.strip()


def build_real_engine(
    *,
    enable_shared_manifold: bool = True,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    shared_store_path: Optional[str] = None,
    shared_store_cache_key: str = "default",
) -> CortexEngine:
    engine = CortexEngine(model_id=model_id, device=device)
    if shared_store_path is not None or shared_store_cache_key != getattr(engine.agent_cloud, "shared_store_cache_key", "default"):
        hidden_dim = int(getattr(engine.model.config, "hidden_size", 896))
        prior_cloud = engine.agent_cloud
        engine.agent_cloud = PersistentAgentCloud(
            hidden_dim=hidden_dim,
            tokenizer=engine.tokenizer,
            embed_layer=engine.model.get_input_embeddings(),
            device=engine.device,
            max_episodes_per_agent=getattr(prior_cloud, "max_episodes_per_agent", 128),
            shared_manifold_capacity=getattr(prior_cloud, "shared_manifold_capacity", 256),
            shared_hot_capacity=getattr(prior_cloud, "shared_hot_capacity", 8),
            adapter_rank=getattr(prior_cloud, "adapter_rank", 8),
            synapse_ttl_seconds=getattr(prior_cloud, "synapse_ttl_seconds", 3600.0),
            shared_store_path=shared_store_path,
            shared_store_cache_key=shared_store_cache_key,
            shared_energy_feedback_enabled=bool(enable_energy_feedback),
        )
    engine.shared_manifold_energy_feedback_enabled = bool(enable_energy_feedback)
    engine.agent_cloud.shared_energy_feedback_enabled = bool(enable_energy_feedback)
    engine.set_shared_manifold_enabled(enable_shared_manifold)
    return engine


def _reset_real_engine_state(engine: CortexEngine, enable_shared_manifold: bool):
    hidden_dim = int(getattr(engine.model.config, "hidden_size", 896))
    prior_cloud = engine.agent_cloud
    energy_feedback_enabled = bool(getattr(engine, "shared_manifold_energy_feedback_enabled", False))
    engine.agent_cloud = PersistentAgentCloud(
        hidden_dim=hidden_dim,
        tokenizer=engine.tokenizer,
        embed_layer=engine.model.get_input_embeddings(),
        device=engine.device,
        max_episodes_per_agent=getattr(prior_cloud, "max_episodes_per_agent", 128),
        shared_manifold_capacity=getattr(prior_cloud, "shared_manifold_capacity", 256),
        shared_hot_capacity=getattr(prior_cloud, "shared_hot_capacity", 8),
        adapter_rank=getattr(prior_cloud, "adapter_rank", 8),
        synapse_ttl_seconds=getattr(prior_cloud, "synapse_ttl_seconds", 3600.0),
        shared_store_path=getattr(prior_cloud, "shared_store_path", None),
        shared_store_cache_key=getattr(prior_cloud, "shared_store_cache_key", "default"),
        shared_energy_feedback_enabled=energy_feedback_enabled,
    )
    engine.shared_manifold_energy_feedback_enabled = energy_feedback_enabled
    engine.agent_cloud.shared_energy_feedback_enabled = energy_feedback_enabled
    engine.reset_shared_manifold_trace()
    engine.set_shared_manifold_enabled(enable_shared_manifold)


def _seed_real_engine(engine: CortexEngine, item: Any):
    engine.register_persistent_agent(item.writer_agent, profile=item.writer_profile, role=item.writer_role)
    engine.register_persistent_agent(item.reader_agent, profile=item.reader_profile, role=item.reader_role)
    if isinstance(item, SharedManifoldCodingTask):
        _publish_coding_task_board(engine, item)
        return
    node_type = "coding_memory" if isinstance(item, SharedManifoldCodingTask) else "recall_memory"
    for index, memory in enumerate(item.memories):
        engine.remember_shared_event(
            text=memory,
            source="scenario",
            node_type=node_type,
            metadata={"sequence_index": index},
        )


def _seed_real_topology_engine(engine: CortexEngine, task: SharedManifoldTopologyTask):
    engine.register_persistent_agent(task.writer_agent, profile=task.writer_profile, role=task.writer_role)
    engine.register_persistent_agent(task.reader_agent, profile=task.reader_profile, role=task.reader_role)
    for index, memory in enumerate(task.memories):
        engine.remember_shared_event(
            text=memory.text,
            source=memory.source,
            node_type=memory.node_type,
            metadata={
                "sequence_index": index,
                "keywords": list(memory.keywords),
                "entity_refs": list(memory.entity_refs),
                "benchmark_region": memory.region,
                "benchmark_is_bridge": bool(memory.is_bridge),
            },
        )


def _seed_energy_reuse_engine(engine: CortexEngine, task: SharedManifoldEnergyReuseTask):
    for board_task in task.task_board_tasks:
        engine.register_persistent_agent(
            board_task.writer_agent,
            profile=board_task.writer_profile,
            role=board_task.writer_role,
        )
        engine.register_persistent_agent(
            board_task.reader_agent,
            profile=board_task.reader_profile,
            role=board_task.reader_role,
        )
        _publish_coding_task_board(engine, board_task)


def _publish_coding_task_board(engine: CortexEngine, task: SharedManifoldCodingTask):
    signature = _extract_exact_signature(task.starter_code)
    engine.agent_cloud.publish_task_spec(
        task_id=task.name,
        summary=task.task,
        recent_text=task.recent_text,
        signature=signature,
        acceptance_criteria=task.acceptance_criteria,
        source="scenario",
        agent_id=task.writer_agent,
    )
    for index, memory in enumerate(task.memories):
        engine.agent_cloud.publish_task_note(
            task_id=task.name,
            note_text=memory,
            sequence_index=10 + index,
            source="scenario",
            agent_id=task.writer_agent,
        )
    for index, repair_step in enumerate(task.repair_steps):
        engine.agent_cloud.publish_task_patch(
            task_id=task.name,
            patch_name=repair_step.name,
            old_text=repair_step.old,
            new_text=repair_step.new,
            trigger_terms=repair_step.trigger_terms,
            sequence_index=100 + index,
            source="scenario",
            agent_id=task.writer_agent,
        )


def _topology_recall_task(task: SharedManifoldTopologyTask) -> SharedManifoldRecallTask:
    question = task.question or task.query_text
    expected_terms = list(task.expected_terms)
    if not expected_terms:
        expected_terms = [text.split("=", 1)[-1] for text in task.expected_texts]
    return SharedManifoldRecallTask(
        name=task.name,
        writer_agent=task.writer_agent,
        writer_role=task.writer_role,
        writer_profile=task.writer_profile,
        reader_agent=task.reader_agent,
        reader_role=task.reader_role,
        reader_profile=task.reader_profile,
        memories=[memory.text for memory in task.memories],
        question=question,
        expected_terms=expected_terms,
        answer_format=task.answer_format,
        expected_fields=dict(task.expected_fields),
    )


def _build_flat_topology_context(nodes: List[Any]) -> str:
    if not nodes:
        return ""
    lines = [
        "[Shared Manifold]",
        f"[Topology: flat_baseline=1, selected_nodes={len(nodes)}]",
    ]
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        label = getattr(node, "node_type", "memory").replace("_", " ")
        region = metadata.get("benchmark_region")
        if region:
            lines.append(f"- [{label} region={region}] {node.text}")
        else:
            lines.append(f"- [{label}] {node.text}")
    return "\n".join(lines)


def _normalize_context_signature(context: str) -> tuple[str, ...]:
    signature_lines: List[str] = []
    for raw_line in str(context or '').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('[Shared Manifold]') or line.startswith('[Shared Projection]'):
            continue
        if line.startswith('[Projection:') or line.startswith('[Topology:'):
            continue
        if line.startswith('- [projection summary]'):
            continue
        signature_lines.append(line)
    return tuple(sorted(signature_lines))


def _parse_task_board_context(context: str) -> Dict[str, List[str]]:
    task_ids: List[str] = []
    patch_names: List[str] = []
    for raw_line in str(context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        task_match = re.match(r"\[Task:\s*([^\]]+)\]", line)
        if task_match:
            task_ids.append(task_match.group(1).strip())
            continue
        if line.startswith("patch="):
            patch_name = line.split("=", 1)[1].strip()
            if patch_name:
                patch_names.append(patch_name)
    return {
        "task_ids": task_ids,
        "patch_names": patch_names,
    }


def _build_real_coding_prompt(engine: CortexEngine, task: SharedManifoldCodingTask) -> tuple[str, str, List[str]]:
    prompt_nodes = engine.agent_cloud.query_shared_manifold(query_text=task.task, top_k=1, agent_id=task.reader_agent)
    board_nodes = engine.agent_cloud.query_task_board(query_text=task.task, top_k=1, agent_id=task.reader_agent)
    prompt_node_texts = [node.text for node in board_nodes] if board_nodes else [node.text for node in prompt_nodes]
    shared_context = engine._build_shared_manifold_context(task.task, top_k=1) if prompt_nodes else ""

    system = (
        "You read a shared task board and choose deterministic patch operations. "
        "Do not write code. Copy patch ids verbatim from the task board. "
        "Return exactly one line in the format apply=PATCH_ID[,PATCH_ID...]. "
        "If no patch is needed, return apply=none."
    )
    if shared_context:
        patch_choices = ", ".join(step.name for step in task.repair_steps)
        if patch_choices:
            system += f" Valid patch ids for this task: {patch_choices}."
        system += "\nTask board:\n" + shared_context

    user = textwrap.dedent(
        f"""
        Choose the smallest correct patch set for this task.
        Task: {task.task}
        Recent context: {task.recent_text}

        Buggy code:
        ```python
        {task.starter_code}
        ```

        Return only the one-line apply=... decision.
        Never echo the function signature.
        Never return patch=... .
        """
    ).strip()

    prompt = engine.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, shared_context, prompt_node_texts


def _normalize_patch_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _parse_selected_patch_names(raw_output: str, repair_steps: List[SharedManifoldRepairStep]) -> List[str]:
    available_names = [step.name for step in repair_steps]
    normalized_to_name = {_normalize_patch_key(name): name for name in available_names}
    selected: List[str] = []

    def add_candidate(candidate: str):
        normalized = _normalize_patch_key(candidate)
        matched_name = normalized_to_name.get(normalized)
        if matched_name and matched_name not in selected:
            selected.append(matched_name)

    apply_match = re.search(r"(?:apply|patch)\s*=\s*([^\n\r`]+)", raw_output, flags=re.IGNORECASE)
    if apply_match:
        candidate_text = apply_match.group(1).strip()
        if _normalize_patch_key(candidate_text) == "none":
            return []
        for token in re.split(r"[,;]+", candidate_text):
            add_candidate(token)

    lowered_output = raw_output.lower()
    for name in available_names:
        if name in lowered_output or _normalize_patch_key(name) in _normalize_patch_key(raw_output):
            add_candidate(name)

    if selected:
        return selected

    for step in repair_steps:
        if step.trigger_terms and _contains_terms(raw_output, step.trigger_terms):
            add_candidate(step.name)
    return selected


def _apply_selected_repairs(task: SharedManifoldCodingTask, selected_patch_names: List[str]) -> tuple[str, List[str]]:
    code = task.starter_code
    applied_repairs: List[str] = []
    selected_set = set(selected_patch_names)
    for repair_step in task.repair_steps:
        if repair_step.name not in selected_set:
            continue

        old_text = repair_step.old
        new_text = repair_step.new
        if old_text in code:
            code = code.replace(old_text, new_text, 1)
            applied_repairs.append(repair_step.name)
            continue

        old_text = repair_step.old.rstrip("\n")
        if old_text and old_text in code:
            code = code.replace(old_text, repair_step.new.rstrip("\n"), 1)
            applied_repairs.append(repair_step.name)
    return code, applied_repairs


def _compose_real_recall_prompt(
    engine: CortexEngine,
    task: SharedManifoldRecallTask,
    shared_context: str,
) -> str:
    system_parts = [
        "Answer the user's question using only the shared context. "
        "Shared-context bullet points may be ordered by relevance rather than time, so reconstruct the chronology mentally before answering. "
        "When the same object changes state over time, use the latest relevant state rather than the first mention. "
        "Track transfers and final locations before answering. "
        "If one person gives an object to another and the second person later stores, hides, parks, or locks it, "
        "the correct holder is the most recent person before that final storage event, not the original owner. "
    ]
    if task.answer_format:
        system_parts.append(
            "A schema is provided for this task, so always answer with that schema on a single line. "
            "If a field is truly missing, write key=unknown for that field instead of replying with bare unknown. "
            "Narrative sentences count as direct evidence even when they are not written as literal key=value facts. "
            "Keep the answer to one short line."
        )
    else:
        system_parts.append(
            "If the shared context is empty or does not contain the answer, reply with exactly 'unknown'. "
            "Keep the answer to one short line."
        )
    system = "".join(system_parts)
    if shared_context:
        system += "\nShared context:\n" + shared_context

    format_line = "Return only the shortest direct answer."
    placeholder_line = ""
    structure_line = ""
    example_line = ""
    completion_line = ""
    if task.answer_format:
        format_keys = list(task.expected_fields.keys())
        if not format_keys:
            format_keys = [chunk.split("=", 1)[0].strip().lower() for chunk in task.answer_format.split(";") if "=" in chunk]
        format_line = f"Return one line using this schema with the real values filled in: {task.answer_format}"
        placeholder_line = "Replace every VALUE placeholder with the exact value from shared context. Never output the words VALUE, PLACE, or NAME."
        completion_line = "Fill every listed key before stopping. Never answer with only the first key."
        if format_keys:
            structure_line = (
                f"Use exactly these keys in this order and preserve every '=' sign: {'; '.join(format_keys)}. "
                "Do not invent extra keys or copy schema fields that are not listed above."
            )
            sample_values = {
                "who": "Ben",
                "where": "drawer 3",
                "color": "blue",
            }
            example_parts = [f"{key}={sample_values.get(key, 'sample')}" for key in format_keys]
            example_line = (
                "Example shape only: " + "; ".join(example_parts) + ". "
                "Do not copy those example values; replace them with the real values from shared context."
            )

    extraction_line = ""
    if task.answer_format:
        extraction_line = (
            "If the shared context contains literal key=value facts or explicit field names, copy those exact right-hand-side values into the schema. "
            "Do not answer unknown when the exact values appear literally in shared context."
        )
        extraction_line += (
            " Field values may be code-like identifiers with uppercase letters, underscores, or hyphens; copy them character-for-character exactly as written."
        )

    user = textwrap.dedent(
        f"""
        Question: {task.question}
        {format_line}
        {"Use only the bare field values after '=' with no extra words, articles, or explanations." if task.answer_format else ''}
        {placeholder_line}
        {completion_line}
        {structure_line}
        {example_line}
        {extraction_line}
        {"For transfer questions, resolve the chain mentally as previous holder -> transfer -> final holder -> final location, then fill the schema with only the final answers." if task.answer_format else ''}
        Read all shared-context lines before answering and resolve the final state mentally first.
        """
    ).strip()

    prompt = engine.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def _build_real_recall_prompt(
    engine: CortexEngine,
    task: SharedManifoldRecallTask,
    *,
    top_k_override: Optional[int] = None,
    shared_context_override: Optional[str] = None,
    prompt_node_texts_override: Optional[List[str]] = None,
) -> tuple[str, str, List[str]]:
    if shared_context_override is not None:
        prompt_node_texts = list(prompt_node_texts_override or [])
        shared_context = shared_context_override
        if task.answer_format and prompt_node_texts and shared_context.startswith("[Shared Manifold]"):
            shared_context = _augment_recall_context_with_focus(shared_context, prompt_node_texts)
        prompt = _compose_real_recall_prompt(engine, task, shared_context)
        return prompt, shared_context, prompt_node_texts

    top_k = top_k_override if top_k_override is not None else max(1, min(3, len(task.memories)))
    prompt_nodes = engine.agent_cloud.query_shared_manifold(query_text=task.question, top_k=top_k)
    prompt_node_texts = [node.text for node in prompt_nodes]
    if task.answer_format and prompt_nodes:
        engine.agent_cloud.resolve_shared_projection(
            query_text=task.question,
            top_k=top_k,
            materialize_missing=True,
            projection_kind="benchmark_recall_prompt",
        )
    shared_context = engine._build_shared_manifold_context(task.question, top_k=top_k) if prompt_nodes else ""
    if task.answer_format and prompt_nodes and shared_context.startswith("[Shared Manifold]"):
        shared_context = _augment_recall_context_with_focus(shared_context, prompt_nodes)
    prompt = _compose_real_recall_prompt(engine, task, shared_context)
    return prompt, shared_context, prompt_node_texts


def run_real_coding_task(
    engine: CortexEngine,
    *,
    enable_shared_manifold: bool,
    task: SharedManifoldCodingTask,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    _reset_real_engine_state(engine, enable_shared_manifold)
    _seed_real_engine(engine, task)
    return _run_real_coding_task_with_current_state(engine, task=task, max_tokens=max_tokens)


def _run_real_coding_task_with_current_state(
    engine: CortexEngine,
    *,
    task: SharedManifoldCodingTask,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    engine.shared_manifold_refresh_interval = 4
    engine.shared_manifold_refresh_top_k = 1
    prompt, prompt_context, prompt_node_texts = _build_real_coding_prompt(engine, task)
    if engine.shared_manifold_enabled:
        engine.agent_cloud.claim_task(
            task_id=task.name,
            agent_id=task.reader_agent,
            status="claimed",
            source="benchmark_runtime",
        )

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        raw_output = engine.generate_text(
            prompt,
            max_tokens=max_tokens,
            stream=False,
            enrich_prompt=False,
            query_text=task.task,
            initial_used_texts=set(prompt_node_texts),
            seed_used_shared_texts=False,
            shared_query_top_k=1,
        )
    runtime_log = capture.getvalue()

    selected_patches = _parse_selected_patch_names(raw_output, task.repair_steps)
    candidate_code, applied_repairs = _apply_selected_repairs(task, selected_patches)
    evaluation = _evaluate_candidate_code(candidate_code, task.tests)
    if engine.shared_manifold_enabled:
        engine.agent_cloud.publish_task_result(
            task_id=task.name,
            agent_id=task.reader_agent,
            status="passed" if evaluation["passed"] else "failed",
            selected_patches=selected_patches,
            result_text=(
                f"apply={','.join(selected_patches) if selected_patches else 'none'}; "
                f"passed={str(evaluation['passed']).lower()}"
            ),
            source="benchmark_runtime",
        )
    metrics = engine.get_shared_manifold_metrics()
    trace = engine.get_shared_manifold_trace()
    matched_terms = _lower_matches(prompt_context, task.guidance_terms)

    return {
        "name": task.name,
        "enabled": bool(engine.shared_manifold_enabled),
        "prompt_context": prompt_context,
        "matched_terms": matched_terms,
        "prompt_hit": bool(prompt_context.strip()),
        "refresh_hit": metrics["runtime_refreshes"] > 0,
        "refresh_nodes": metrics["nodes_consumed"],
        "runtime_refreshes": metrics["runtime_refreshes"],
        "raw_output": raw_output,
        "selected_patches": selected_patches,
        "applied_repairs": applied_repairs,
        "candidate_code": candidate_code,
        "passed": evaluation["passed"],
        "output": evaluation["output"],
        "error": evaluation["error"],
        "metrics": metrics,
        "trace": trace,
        "runtime_log": runtime_log,
    }


def _run_real_recall_task_with_current_state(
    engine: CortexEngine,
    *,
    task: SharedManifoldRecallTask,
    max_tokens: int = 48,
    top_k_override: Optional[int] = None,
    shared_context_override: Optional[str] = None,
    prompt_node_texts_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    engine.shared_manifold_refresh_interval = (max_tokens + 1) if task.answer_format else 2
    engine.shared_manifold_refresh_top_k = 1
    prompt, prompt_context, prompt_node_texts = _build_real_recall_prompt(
        engine,
        task,
        top_k_override=top_k_override,
        shared_context_override=shared_context_override,
        prompt_node_texts_override=prompt_node_texts_override,
    )

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        raw_output = engine.generate_text(
            prompt,
            max_tokens=max_tokens,
            stream=False,
            enrich_prompt=False,
            query_text=task.question,
            initial_used_texts=set(prompt_node_texts),
            seed_used_shared_texts=False,
            shared_query_top_k=1,
        )
    runtime_log = capture.getvalue()

    answer = raw_output.strip()
    metrics = engine.get_shared_manifold_metrics()
    trace = engine.get_shared_manifold_trace()
    evaluation = _evaluate_recall_answer(task, answer)

    return {
        "name": task.name,
        "question": task.question,
        "expected_terms": list(task.expected_terms),
        "expected_fields": dict(task.expected_fields),
        "enabled": bool(engine.shared_manifold_enabled),
        "prompt_context": prompt_context,
        "prompt_hit": bool(prompt_context.strip()),
        "matched_terms": evaluation["matched_terms"],
        "parsed_fields": evaluation["parsed_fields"],
        "missing_fields": evaluation["missing_fields"],
        "passed": evaluation["passed"],
        "answer": answer,
        "raw_output": raw_output,
        "refresh_hit": metrics["runtime_refreshes"] > 0,
        "refresh_nodes": metrics["nodes_consumed"],
        "runtime_refreshes": metrics["runtime_refreshes"],
        "metrics": metrics,
        "trace": trace,
        "runtime_log": runtime_log,
    }


def _summarize_real_topology(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    bridge_items = [item for item in results if int(item.get("bridge_expected_count", 0)) > 0]
    bridge_denominator = max(len(bridge_items), 1)
    return {
        "task_count": len(results),
        "component_accuracy_rate": sum(int(item["topology_retrieval"]["component_count_match"]) for item in results) / count,
        "active_region_accuracy_rate": sum(int(item["topology_retrieval"]["active_region_match"]) for item in results) / count,
        "topology_expected_recall_rate": sum(item["topology_retrieval"]["expected_recall"] for item in results) / count,
        "flat_expected_recall_rate": sum(item["flat_retrieval"]["expected_recall"] for item in results) / count,
        "topology_bridge_recall_rate": sum(item["topology_retrieval"]["bridge_recall"] for item in bridge_items) / bridge_denominator,
        "flat_bridge_recall_rate": sum(item["flat_retrieval"]["bridge_recall"] for item in bridge_items) / bridge_denominator,
        "topology_leakage_rate": sum(item["topology_retrieval"]["leakage_rate"] for item in results) / count,
        "flat_leakage_rate": sum(item["flat_retrieval"]["leakage_rate"] for item in results) / count,
        "topology_answer_rate": sum(int(item["topology_reader"]["passed"]) for item in results) / count,
        "flat_answer_rate": sum(int(item["flat_reader"]["passed"]) for item in results) / count,
        "topology_prompt_hit_rate": sum(int(item["topology_reader"]["prompt_hit"]) for item in results) / count,
        "flat_prompt_hit_rate": sum(int(item["flat_reader"]["prompt_hit"]) for item in results) / count,
        "topology_win_rate": sum(
            int(
                item["topology_retrieval"]["expected_recall"] >= item["flat_retrieval"]["expected_recall"]
                and item["topology_retrieval"]["bridge_recall"] >= item["flat_retrieval"]["bridge_recall"]
                and item["topology_retrieval"]["leakage_rate"] <= item["flat_retrieval"]["leakage_rate"]
                and item["topology_reader"]["passed"] >= item["flat_reader"]["passed"]
            )
            for item in results
        ) / count,
    }


def run_real_topology_task(
    engine: CortexEngine,
    *,
    task: SharedManifoldTopologyTask,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    _reset_real_engine_state(engine, True)
    _seed_real_topology_engine(engine, task)
    recall_task = _topology_recall_task(task)

    topology_nodes, topology_view, active_component = engine.agent_cloud._select_shared_nodes(
        query_text=task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    topology_context = engine.agent_cloud.build_shared_context(
        task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    flat_nodes = _query_flat_shared_manifold(
        engine.agent_cloud,
        query_text=task.query_text,
        top_k=task.top_k,
        agent_id=task.reader_agent,
    )
    flat_context = _build_flat_topology_context(flat_nodes)
    stats = engine.agent_cloud.shared_manifold_stats()

    topology_retrieval = _evaluate_topology_nodes(task, topology_nodes)
    topology_retrieval["active_region_size"] = len(active_component)
    topology_retrieval["region_count"] = len(topology_view.components)
    topology_retrieval["component_count_match"] = int(stats.get("component_count", 0)) == task.expected_component_count
    topology_retrieval["active_region_match"] = len(active_component) == task.expected_active_region_size
    flat_retrieval = _evaluate_topology_nodes(task, flat_nodes)

    topology_reader = _run_real_recall_task_with_current_state(
        engine,
        task=recall_task,
        max_tokens=max_tokens,
        shared_context_override=topology_context,
        prompt_node_texts_override=[node.text for node in topology_nodes],
    )
    flat_reader = _run_real_recall_task_with_current_state(
        engine,
        task=recall_task,
        max_tokens=max_tokens,
        shared_context_override=flat_context,
        prompt_node_texts_override=[node.text for node in flat_nodes],
    )

    return {
        "name": task.name,
        "query_text": task.query_text,
        "question": recall_task.question,
        "bridge_expected_count": len(task.expected_bridge_texts),
        "shared_manifold_stats": stats,
        "topology_context": topology_context,
        "flat_context": flat_context,
        "topology_retrieval": topology_retrieval,
        "flat_retrieval": flat_retrieval,
        "topology_reader": topology_reader,
        "flat_reader": flat_reader,
    }


def compare_real_topology_slice(
    tasks: Optional[Iterable[SharedManifoldTopologyTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_topology_tasks())
    engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    results = [
        run_real_topology_task(
            engine,
            task=task,
            max_tokens=max_tokens,
        )
        for task in task_list
    ]
    return {
        "aggregate": _summarize_real_topology(results),
        "tasks": results,
        "model_id": getattr(engine, "model", None).name_or_path if getattr(engine, "model", None) is not None else None,
        "device": str(getattr(engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def run_real_topology_demo(
    name: str = "real_payment_retry_fields",
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    task_map = {task.name: task for task in default_real_topology_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown real topology task: {name}")

    engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    result = run_real_topology_task(engine, task=task_map[name], max_tokens=max_tokens)
    result["topology_task"] = name
    return result


def _summarize_real_energy_reuse(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "primer_target_hit_rate": sum(item["primer_target_hit_rate"] for item in results) / count,
        "followup_prompt_hit_rate": sum(int(item["followup_prompt_hit"]) for item in results) / count,
        "followup_target_hit_rate": sum(int(item["followup_target_hit"]) for item in results) / count,
        "followup_patch_hit_rate": sum(int(item["followup_patch_hit"]) for item in results) / count,
        "distractor_capture_rate": sum(
            int(item["followup_selected_task_id"] in item["distractor_task_ids"])
            for item in results
        ) / count,
        "avg_energy_peak": sum(float(item["shared_manifold_stats"].get("energy_peak", 0.0)) for item in results) / count,
        "avg_energy_abs_total": sum(float(item["shared_manifold_stats"].get("energy_abs_total", 0.0)) for item in results) / count,
    }


def _run_real_energy_reuse_task_with_current_state(
    engine: CortexEngine,
    *,
    task: SharedManifoldEnergyReuseTask,
) -> Dict[str, Any]:
    primer_contexts: List[str] = []
    primer_selected_task_ids: List[str] = []
    primer_repeats = max(1, int(task.primer_repeats))
    for _ in range(primer_repeats):
        primer_context = engine._build_shared_manifold_context(task.primer_query, top_k=1)
        primer_contexts.append(primer_context)
        primer_parsed = _parse_task_board_context(primer_context)
        primer_selected_task_ids.append(primer_parsed["task_ids"][0] if primer_parsed["task_ids"] else "")

    followup_context = engine._build_shared_manifold_context(task.followup_query, top_k=1)
    followup_parsed = _parse_task_board_context(followup_context)
    followup_selected_task_id = followup_parsed["task_ids"][0] if followup_parsed["task_ids"] else ""
    selected_patch_names = list(followup_parsed["patch_names"])
    expected_patch_names = set(task.expected_patch_names)
    metrics = engine.get_shared_manifold_metrics()
    trace = engine.get_shared_manifold_trace()
    shared_manifold_stats = engine.agent_cloud.shared_manifold_stats()

    return {
        "name": task.name,
        "enabled": bool(engine.shared_manifold_enabled),
        "energy_feedback_enabled": bool(getattr(engine, "shared_manifold_energy_feedback_enabled", False)),
        "expected_task_id": task.expected_task_id,
        "expected_patch_names": list(task.expected_patch_names),
        "distractor_task_ids": [board_task.name for board_task in task.distractor_tasks],
        "primer_query": task.primer_query,
        "followup_query": task.followup_query,
        "primer_repeats": primer_repeats,
        "primer_contexts": primer_contexts,
        "primer_selected_task_ids": primer_selected_task_ids,
        "primer_target_hit_rate": sum(
            int(task_id == task.expected_task_id)
            for task_id in primer_selected_task_ids
        ) / max(len(primer_selected_task_ids), 1),
        "followup_context": followup_context,
        "followup_prompt_hit": bool(followup_context.strip()),
        "followup_selected_task_id": followup_selected_task_id,
        "selected_patch_names": selected_patch_names,
        "followup_target_hit": followup_selected_task_id == task.expected_task_id,
        "followup_patch_hit": bool(expected_patch_names) and expected_patch_names.issubset(set(selected_patch_names)),
        "metrics": metrics,
        "shared_manifold_stats": shared_manifold_stats,
        "trace": trace,
    }


def run_real_energy_reuse_task(
    engine: CortexEngine,
    *,
    task: SharedManifoldEnergyReuseTask,
) -> Dict[str, Any]:
    _reset_real_engine_state(engine, True)
    _seed_energy_reuse_engine(engine, task)
    return _run_real_energy_reuse_task_with_current_state(engine, task=task)


def compare_real_energy_reuse_slice(
    tasks: Optional[Iterable[SharedManifoldEnergyReuseTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_energy_reuse_tasks())
    engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    results = [
        run_real_energy_reuse_task(
            engine,
            task=task,
        )
        for task in task_list
    ]
    return {
        "aggregate": _summarize_real_energy_reuse(results),
        "tasks": results,
        "model_id": getattr(engine, "model", None).name_or_path if getattr(engine, "model", None) is not None else None,
        "device": str(getattr(engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def _summarize_real_handoff(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "context_match_rate": sum(int(item["context_match"]) for item in results) / count,
        "output_match_rate": sum(int(item["output_match"]) for item in results) / count,
        "fresh_prompt_hit_rate": sum(int(item["fresh_reader"]["prompt_hit"]) for item in results) / count,
        "loaded_prompt_hit_rate": sum(int(item["loaded_reader"]["prompt_hit"]) for item in results) / count,
        "fresh_pass_rate": sum(int(item["fresh_reader"]["passed"]) for item in results) / count,
        "loaded_pass_rate": sum(int(item["loaded_reader"]["passed"]) for item in results) / count,
    }


def _summarize_real_recall_handoff(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "context_match_rate": sum(int(item["context_match"]) for item in results) / count,
        "writer_answer_rate": sum(int(item["writer"]["passed"]) for item in results) / count,
        "fresh_prompt_hit_rate": sum(int(item["fresh_reader"]["prompt_hit"]) for item in results) / count,
        "loaded_prompt_hit_rate": sum(int(item["loaded_reader"]["prompt_hit"]) for item in results) / count,
        "fresh_answer_rate": sum(int(item["fresh_reader"]["passed"]) for item in results) / count,
        "loaded_answer_rate": sum(int(item["loaded_reader"]["passed"]) for item in results) / count,
    }


def _flatten_necessity_memories(task: SharedManifoldNecessityTask) -> List[str]:
    memories: List[str] = []
    for session in task.writer_sessions:
        memories.extend(session.memories)
    return memories


def _necessity_recall_task(task: SharedManifoldNecessityTask) -> SharedManifoldRecallTask:
    return SharedManifoldRecallTask(
        name=task.name,
        writer_agent=task.writer_sessions[0].agent_id if task.writer_sessions else "writer",
        writer_role=task.writer_sessions[0].role if task.writer_sessions else "writer",
        writer_profile=task.writer_sessions[0].profile if task.writer_sessions else "",
        reader_agent=task.reader_agent,
        reader_role=task.reader_role,
        reader_profile=task.reader_profile,
        memories=_flatten_necessity_memories(task),
        question=task.question,
        expected_terms=list(task.expected_terms),
        answer_format=task.answer_format,
        expected_fields=dict(task.expected_fields),
    )


def _build_flat_necessity_context(task: SharedManifoldNecessityTask) -> str:
    memory_count = len(_flatten_necessity_memories(task))
    lines = [
        "[Shared Manifold]",
        f"[Topology: density=0.00, spread=0.00, coverage=1.00, regions=1, active_region={memory_count}, bridges=0]",
    ]
    for session in task.writer_sessions:
        label = session.role.replace("_", " ")
        for memory in session.memories:
            lines.append(f"- [{label} from {session.agent_id}] {memory}")
    return "\n".join(lines)


def _seed_real_writer_session(
    engine: CortexEngine,
    session: SharedManifoldWriterSession,
    *,
    sequence_index_start: int,
) -> int:
    engine.register_persistent_agent(session.agent_id, profile=session.profile, role=session.role)
    for offset, memory in enumerate(session.memories):
        engine.remember_shared_event(
            text=memory,
            source="necessity_session",
            node_type="necessity_memory",
            metadata={
                "sequence_index": sequence_index_start + offset,
                "writer_session": session.agent_id,
            },
        )
    return sequence_index_start + len(session.memories)


def _prepare_real_reader(engine: CortexEngine, task: SharedManifoldNecessityTask):
    engine.register_persistent_agent(task.reader_agent, profile=task.reader_profile, role=task.reader_role)


def _summarize_real_necessity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "isolated_prompt_hit_rate": sum(int(item["isolated_reader"]["prompt_hit"]) for item in results) / count,
        "manifold_prompt_hit_rate": sum(int(item["manifold_reader"]["prompt_hit"]) for item in results) / count,
        "isolated_answer_rate": sum(int(item["isolated_reader"]["passed"]) for item in results) / count,
        "manifold_answer_rate": sum(int(item["manifold_reader"]["passed"]) for item in results) / count,
        "oracle_answer_rate": sum(int(item["oracle_reader"]["passed"]) for item in results) / count,
        "necessity_win_rate": sum(
            int(item["manifold_reader"]["passed"] and not item["isolated_reader"]["passed"])
            for item in results
        ) / count,
    }


def run_real_necessity_task(
    writer_engine: CortexEngine,
    reader_engine: CortexEngine,
    *,
    task: SharedManifoldNecessityTask,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    recall_task = _necessity_recall_task(task)
    flat_context = _build_flat_necessity_context(task)
    total_memories = len(recall_task.memories)
    top_k = max(1, min(6, total_memories))

    with tempfile.TemporaryDirectory(prefix="warp_cortex_necessity_") as tmpdir:
        db_path = os.path.join(tmpdir, "shared_manifold.sqlite")
        writer_engine.agent_cloud.shared_store_path = db_path
        reader_engine.agent_cloud.shared_store_path = db_path

        sequence_index = 0
        for session in task.writer_sessions:
            _reset_real_engine_state(writer_engine, True)
            sequence_index = _seed_real_writer_session(
                writer_engine,
                session,
                sequence_index_start=sequence_index,
            )

        store_stats = writer_engine.agent_cloud.shared_manifold_stats()
        store_stats.pop("shared_store_path", None)

        _reset_real_engine_state(reader_engine, False)
        _prepare_real_reader(reader_engine, task)
        isolated_reader = _run_real_recall_task_with_current_state(
            reader_engine,
            task=recall_task,
            max_tokens=max_tokens,
            top_k_override=top_k,
        )

        _reset_real_engine_state(reader_engine, True)
        _prepare_real_reader(reader_engine, task)
        manifold_reader = _run_real_recall_task_with_current_state(
            reader_engine,
            task=recall_task,
            max_tokens=max_tokens,
            top_k_override=top_k,
        )

        _reset_real_engine_state(reader_engine, False)
        _prepare_real_reader(reader_engine, task)
        oracle_reader = _run_real_recall_task_with_current_state(
            reader_engine,
            task=recall_task,
            max_tokens=max_tokens,
            top_k_override=top_k,
            shared_context_override=flat_context,
            prompt_node_texts_override=list(recall_task.memories),
        )

    return {
        "name": task.name,
        "writer_session_count": len(task.writer_sessions),
        "memory_count": len(recall_task.memories),
        "question": task.question,
        "expected_fields": dict(task.expected_fields),
        "flat_context": flat_context,
        "shared_store_stats": store_stats,
        "isolated_reader": isolated_reader,
        "manifold_reader": manifold_reader,
        "oracle_reader": oracle_reader,
        "necessity_win": bool(manifold_reader["passed"] and not isolated_reader["passed"]),
        "oracle_supported": bool(oracle_reader["passed"]),
    }


def compare_real_necessity_slice(
    tasks: Optional[Iterable[SharedManifoldNecessityTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_necessity_tasks())
    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
        shared_store_path="",
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
        shared_store_path="",
    )
    results = [
        run_real_necessity_task(
            writer_engine,
            reader_engine,
            task=task,
            max_tokens=max_tokens,
        )
        for task in task_list
    ]
    return {
        "aggregate": _summarize_real_necessity(results),
        "tasks": results,
        "model_id": getattr(writer_engine, "model", None).name_or_path if getattr(writer_engine, "model", None) is not None else None,
        "device": str(getattr(writer_engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def run_real_necessity_demo(
    name: str = "vx17_badge_locker",
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    if name == "payment_retry":
        name = "vx17_badge_locker"
    task_map = {task.name: task for task in default_real_necessity_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown real necessity task: {name}")

    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
        shared_store_path="",
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
        shared_store_path="",
    )
    result = run_real_necessity_task(writer_engine, reader_engine, task=task_map[name], max_tokens=max_tokens)
    result["necessity_task"] = name
    return result


def run_real_handoff_task(
    writer_engine: CortexEngine,
    reader_engine: CortexEngine,
    *,
    task: SharedManifoldCodingTask,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    _reset_real_engine_state(writer_engine, True)
    _seed_real_engine(writer_engine, task)
    writer_result = _run_real_coding_task_with_current_state(writer_engine, task=task, max_tokens=max_tokens)

    with tempfile.TemporaryDirectory(prefix="warp_cortex_handoff_") as tmpdir:
        snapshot_path = os.path.join(tmpdir, "agent_cloud.pt")
        writer_engine.save_agent_population(snapshot_path)

        _reset_real_engine_state(reader_engine, True)
        fresh_reader = _run_real_coding_task_with_current_state(reader_engine, task=task, max_tokens=max_tokens)

        _reset_real_engine_state(reader_engine, True)
        load_stats = reader_engine.load_agent_population(snapshot_path)
        reader_engine.reset_shared_manifold_trace()
        loaded_reader = _run_real_coding_task_with_current_state(reader_engine, task=task, max_tokens=max_tokens)

    return {
        "name": task.name,
        "writer": writer_result,
        "fresh_reader": fresh_reader,
        "loaded_reader": loaded_reader,
        "snapshot_load": load_stats,
        "context_match": (
            _normalize_context_signature(writer_result["prompt_context"])
            == _normalize_context_signature(loaded_reader["prompt_context"])
        ),
        "output_match": writer_result["candidate_code"] == loaded_reader["candidate_code"],
    }


def run_real_recall_handoff_task(
    writer_engine: CortexEngine,
    reader_engine: CortexEngine,
    *,
    task: SharedManifoldRecallTask,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    _reset_real_engine_state(writer_engine, True)
    _seed_real_engine(writer_engine, task)
    writer_result = _run_real_recall_task_with_current_state(writer_engine, task=task, max_tokens=max_tokens)

    with tempfile.TemporaryDirectory(prefix="warp_cortex_recall_handoff_") as tmpdir:
        snapshot_path = os.path.join(tmpdir, "agent_cloud.pt")
        writer_engine.save_agent_population(snapshot_path)

        _reset_real_engine_state(reader_engine, True)
        fresh_reader = _run_real_recall_task_with_current_state(reader_engine, task=task, max_tokens=max_tokens)

        _reset_real_engine_state(reader_engine, True)
        load_stats = reader_engine.load_agent_population(snapshot_path)
        reader_engine.reset_shared_manifold_trace()
        loaded_reader = _run_real_recall_task_with_current_state(reader_engine, task=task, max_tokens=max_tokens)

    return {
        "name": task.name,
        "writer": writer_result,
        "fresh_reader": fresh_reader,
        "loaded_reader": loaded_reader,
        "snapshot_load": load_stats,
        "context_match": (
            _normalize_context_signature(writer_result["prompt_context"])
            == _normalize_context_signature(loaded_reader["prompt_context"])
        ),
    }


def compare_real_handoff_slice(
    tasks: Optional[Iterable[SharedManifoldCodingTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_coding_tasks())
    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    results = [
        run_real_handoff_task(
            writer_engine,
            reader_engine,
            task=task,
            max_tokens=max_tokens,
        )
        for task in task_list
    ]
    return {
        "aggregate": _summarize_real_handoff(results),
        "tasks": results,
        "model_id": getattr(writer_engine, "model", None).name_or_path if getattr(writer_engine, "model", None) is not None else None,
        "device": str(getattr(writer_engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def compare_real_recall_handoff_slice(
    tasks: Optional[Iterable[SharedManifoldRecallTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_recall_tasks())
    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    results = [
        run_real_recall_handoff_task(
            writer_engine,
            reader_engine,
            task=task,
            max_tokens=max_tokens,
        )
        for task in task_list
    ]
    return {
        "aggregate": _summarize_real_recall_handoff(results),
        "tasks": results,
        "model_id": getattr(writer_engine, "model", None).name_or_path if getattr(writer_engine, "model", None) is not None else None,
        "device": str(getattr(writer_engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def run_real_handoff_demo(
    name: str = "auth_failure_event_real",
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    task_map = {task.name: task for task in default_real_coding_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown real handoff task: {name}")

    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    result = run_real_handoff_task(writer_engine, reader_engine, task=task_map[name], max_tokens=max_tokens)
    result["coding_task"] = name
    return result


def run_real_recall_handoff_demo(
    name: str = "jenny_boots_red",
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 48,
) -> Dict[str, Any]:
    task_map = {task.name: task for task in default_real_recall_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown real recall handoff task: {name}")

    writer_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    reader_engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    result = run_real_recall_handoff_task(writer_engine, reader_engine, task=task_map[name], max_tokens=max_tokens)
    result["recall_task"] = name
    return result


def _summarize_real_coding(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "pass_rate": sum(int(item["passed"]) for item in results) / count,
        "prompt_hit_rate": sum(int(item["prompt_hit"]) for item in results) / count,
        "refresh_hit_rate": sum(int(item["refresh_hit"]) for item in results) / count,
        "code_extract_rate": sum(int(bool(item["candidate_code"])) for item in results) / count,
        "avg_output_chars": sum(len(item["raw_output"]) for item in results) / count,
        "total_runtime_refreshes": sum(item["runtime_refreshes"] for item in results),
        "total_nodes_consumed": sum(item["metrics"]["nodes_consumed"] for item in results),
    }


def run_probe_scenario(enable_shared_manifold: bool, scenario: SharedManifoldScenario) -> Dict[str, Any]:
    engine = build_probe_engine(enable_shared_manifold=enable_shared_manifold)
    _seed_probe_engine(engine, scenario)
    shared_state = _collect_shared_state(
        engine,
        task=scenario.task,
        recent_text=scenario.recent_text,
        reader_agent=scenario.reader_agent,
    )
    prompt_context = shared_state["prompt_context"]
    refresh_count = shared_state["refresh_count"]
    metrics = shared_state["metrics"]
    trace = shared_state["trace"]
    shared_calls = shared_state["shared_calls"]
    matched_terms = _lower_matches(prompt_context, scenario.expected_terms)

    return {
        "name": scenario.name,
        "enabled": enable_shared_manifold,
        "prompt_context": prompt_context,
        "matched_terms": matched_terms,
        "prompt_hit": bool(matched_terms),
        "refresh_nodes": refresh_count,
        "runtime_refreshes": metrics["runtime_refreshes"],
        "refresh_hit": metrics["runtime_refreshes"] > 0 and bool(shared_calls),
        "shared_calls": shared_calls,
        "metrics": metrics,
        "trace": trace,
    }


def run_coding_task(enable_shared_manifold: bool, task: SharedManifoldCodingTask) -> Dict[str, Any]:
    engine = build_probe_engine(enable_shared_manifold=enable_shared_manifold)
    _seed_probe_engine(engine, task)
    shared_state = _collect_shared_state(
        engine,
        task=task.task,
        recent_text=task.recent_text,
        reader_agent=task.reader_agent,
    )

    guidance_text = "\n".join(
        part for part in [shared_state["prompt_context"], *shared_state["shared_calls"]] if part
    )
    matched_terms = _lower_matches(guidance_text, task.guidance_terms)
    candidate_code, applied_repairs = _synthesize_candidate_code(task, guidance_text)
    evaluation = _evaluate_candidate_code(candidate_code, task.tests)

    return {
        "name": task.name,
        "enabled": enable_shared_manifold,
        "prompt_context": shared_state["prompt_context"],
        "matched_terms": matched_terms,
        "prompt_hit": bool(shared_state["prompt_context"].strip()),
        "refresh_nodes": shared_state["refresh_count"],
        "runtime_refreshes": shared_state["metrics"]["runtime_refreshes"],
        "refresh_hit": shared_state["metrics"]["runtime_refreshes"] > 0 and bool(shared_state["shared_calls"]),
        "shared_calls": shared_state["shared_calls"],
        "applied_repairs": applied_repairs,
        "candidate_code": candidate_code,
        "passed": evaluation["passed"],
        "output": evaluation["output"],
        "error": evaluation["error"],
        "metrics": shared_state["metrics"],
        "trace": shared_state["trace"],
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "scenario_count": len(results),
        "prompt_hit_rate": sum(int(item["prompt_hit"]) for item in results) / count,
        "refresh_hit_rate": sum(int(item["refresh_hit"]) for item in results) / count,
        "avg_term_matches": sum(len(item["matched_terms"]) for item in results) / count,
        "total_runtime_refreshes": sum(item["runtime_refreshes"] for item in results),
        "total_nodes_consumed": sum(item["metrics"]["nodes_consumed"] for item in results),
    }


def _summarize_coding(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = max(len(results), 1)
    return {
        "task_count": len(results),
        "pass_rate": sum(int(item["passed"]) for item in results) / count,
        "prompt_hit_rate": sum(int(item["prompt_hit"]) for item in results) / count,
        "refresh_hit_rate": sum(int(item["refresh_hit"]) for item in results) / count,
        "avg_repairs_applied": sum(len(item["applied_repairs"]) for item in results) / count,
        "total_runtime_refreshes": sum(item["runtime_refreshes"] for item in results),
        "total_nodes_consumed": sum(item["metrics"]["nodes_consumed"] for item in results),
    }


def run_probe(enable_shared_manifold: bool, scenarios: Optional[Iterable[SharedManifoldScenario]] = None) -> Dict[str, Any]:
    scenario_list = list(scenarios or default_scenarios())
    results = [run_probe_scenario(enable_shared_manifold, scenario) for scenario in scenario_list]
    return {
        "enabled": enable_shared_manifold,
        "aggregate": _summarize(results),
        "scenarios": results,
    }


def compare_pipeline(scenarios: Optional[Iterable[SharedManifoldScenario]] = None) -> Dict[str, Any]:
    scenario_list = list(scenarios or default_scenarios())
    return {
        "enabled": run_probe(True, scenario_list),
        "disabled": run_probe(False, scenario_list),
    }


def run_coding_probe(
    enable_shared_manifold: bool,
    tasks: Optional[Iterable[SharedManifoldCodingTask]] = None,
) -> Dict[str, Any]:
    task_list = list(tasks or default_coding_tasks())
    results = [run_coding_task(enable_shared_manifold, task) for task in task_list]
    return {
        "enabled": enable_shared_manifold,
        "aggregate": _summarize_coding(results),
        "tasks": results,
    }


def compare_coding_slice(tasks: Optional[Iterable[SharedManifoldCodingTask]] = None) -> Dict[str, Any]:
    task_list = list(tasks or default_coding_tasks())
    return {
        "enabled": run_coding_probe(True, task_list),
        "disabled": run_coding_probe(False, task_list),
    }


def run_real_coding_probe(
    enable_shared_manifold: bool,
    tasks: Optional[Iterable[SharedManifoldCodingTask]] = None,
    *,
    engine: Optional[CortexEngine] = None,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_coding_tasks())
    own_engine = engine is None
    runtime_engine = engine or build_real_engine(
        enable_shared_manifold=enable_shared_manifold,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    runtime_engine.shared_manifold_energy_feedback_enabled = bool(enable_energy_feedback)
    runtime_engine.agent_cloud.shared_energy_feedback_enabled = bool(enable_energy_feedback)
    results = [
        run_real_coding_task(
            runtime_engine,
            enable_shared_manifold=enable_shared_manifold,
            task=task,
            max_tokens=max_tokens,
        )
        for task in task_list
    ]
    report = {
        "enabled": enable_shared_manifold,
        "aggregate": _summarize_real_coding(results),
        "tasks": results,
        "model_id": getattr(runtime_engine, "model", None).name_or_path if getattr(runtime_engine, "model", None) is not None else None,
        "device": str(getattr(runtime_engine, "device", device or "cpu")),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }
    if own_engine:
        del runtime_engine
    return report


def compare_real_coding_slice(
    tasks: Optional[Iterable[SharedManifoldCodingTask]] = None,
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    task_list = list(tasks or default_real_coding_tasks())
    engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    return {
        "enabled": run_real_coding_probe(
            True,
            task_list,
            engine=engine,
            enable_energy_feedback=enable_energy_feedback,
            max_tokens=max_tokens,
        ),
        "disabled": run_real_coding_probe(
            False,
            task_list,
            engine=engine,
            enable_energy_feedback=enable_energy_feedback,
            max_tokens=max_tokens,
        ),
        "energy_feedback_enabled": bool(enable_energy_feedback),
    }


def run_demo_scenario(name: str = "payment_retry") -> Dict[str, Any]:
    scenario_map = {scenario.name: scenario for scenario in default_scenarios()}
    if name not in scenario_map:
        raise ValueError(f"Unknown scenario: {name}")

    scenario = scenario_map[name]
    result = run_probe_scenario(True, scenario)
    timeline = [
        f"1. {scenario.writer_agent} writes {len(scenario.memories)} shared memory node(s).",
        f"2. {scenario.reader_agent} asks: {scenario.task}",
        f"3. Prompt-time shared recall hit: {result['prompt_hit']} ({', '.join(result['matched_terms']) or 'none'})",
        f"4. Runtime refresh count: {result['runtime_refreshes']}",
        f"5. Trace events: {len(result['trace'])}",
    ]
    result["timeline"] = timeline
    result["scenario"] = scenario.name
    return result


def run_coding_demo(name: str = "payment_retry_repair") -> Dict[str, Any]:
    task_map = {task.name: task for task in default_coding_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown coding task: {name}")

    task = task_map[name]
    result = run_coding_task(True, task)
    timeline = [
        f"1. {task.writer_agent} writes {len(task.memories)} shared coding memory node(s).",
        f"2. {task.reader_agent} repairs: {task.task}",
        f"3. Applied repairs: {', '.join(result['applied_repairs']) or 'none'}",
        f"4. Prompt-time shared recall hit: {result['prompt_hit']} ({', '.join(result['matched_terms']) or 'none'})",
        f"5. Runtime refresh count: {result['runtime_refreshes']}",
        f"6. Task passed: {result['passed']}",
    ]
    result["timeline"] = timeline
    result["coding_task"] = task.name
    return result


def run_real_coding_demo(
    name: str = "retry_headers_real",
    *,
    enable_energy_feedback: bool = False,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    max_tokens: int = 160,
) -> Dict[str, Any]:
    task_map = {task.name: task for task in default_real_coding_tasks()}
    if name not in task_map:
        raise ValueError(f"Unknown real coding task: {name}")

    engine = build_real_engine(
        enable_shared_manifold=True,
        enable_energy_feedback=enable_energy_feedback,
        model_id=model_id,
        device=device,
    )
    task = task_map[name]
    result = run_real_coding_task(engine, enable_shared_manifold=True, task=task, max_tokens=max_tokens)
    timeline = [
        f"1. {task.writer_agent} writes {len(task.memories)} shared coding memory node(s).",
        f"2. {task.reader_agent} repairs: {task.task}",
        f"3. Prompt-time shared recall hit: {result['prompt_hit']} ({', '.join(result['matched_terms']) or 'none'})",
        f"4. Runtime refresh count: {result['runtime_refreshes']}",
        f"5. Code extracted: {bool(result['candidate_code'])}",
        f"6. Task passed: {result['passed']}",
    ]
    result["timeline"] = timeline
    result["coding_task"] = task.name
    return result


def _print_compare(report: Dict[str, Any]):
    enabled = report["enabled"]["aggregate"]
    disabled = report["disabled"]["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX SHARED MANIFOLD PIPELINE")
    print("=" * 64)
    print("Enabled:")
    print(f"  prompt_hit_rate     = {enabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {enabled['refresh_hit_rate']:.2f}")
    print(f"  avg_term_matches    = {enabled['avg_term_matches']:.2f}")
    print(f"  runtime_refreshes   = {enabled['total_runtime_refreshes']}")
    print("Disabled:")
    print(f"  prompt_hit_rate     = {disabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {disabled['refresh_hit_rate']:.2f}")
    print(f"  avg_term_matches    = {disabled['avg_term_matches']:.2f}")
    print(f"  runtime_refreshes   = {disabled['total_runtime_refreshes']}")


def _print_coding_compare(report: Dict[str, Any]):
    enabled = report["enabled"]["aggregate"]
    disabled = report["disabled"]["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX SHARED MANIFOLD CODING SLICE")
    print("=" * 64)
    print("Enabled:")
    print(f"  pass_rate           = {enabled['pass_rate']:.2f}")
    print(f"  prompt_hit_rate     = {enabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {enabled['refresh_hit_rate']:.2f}")
    print(f"  avg_repairs_applied = {enabled['avg_repairs_applied']:.2f}")
    print(f"  runtime_refreshes   = {enabled['total_runtime_refreshes']}")
    print("Disabled:")
    print(f"  pass_rate           = {disabled['pass_rate']:.2f}")
    print(f"  prompt_hit_rate     = {disabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {disabled['refresh_hit_rate']:.2f}")
    print(f"  avg_repairs_applied = {disabled['avg_repairs_applied']:.2f}")
    print(f"  runtime_refreshes   = {disabled['total_runtime_refreshes']}")


def _print_real_coding_compare(report: Dict[str, Any]):
    enabled = report["enabled"]["aggregate"]
    disabled = report["disabled"]["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX REAL SHARED MANIFOLD CODING")
    print("=" * 64)
    print(f"Model: {report['enabled'].get('model_id')}  Device: {report['enabled'].get('device')}")
    print("Enabled:")
    print(f"  pass_rate           = {enabled['pass_rate']:.2f}")
    print(f"  prompt_hit_rate     = {enabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {enabled['refresh_hit_rate']:.2f}")
    print(f"  code_extract_rate   = {enabled['code_extract_rate']:.2f}")
    print(f"  avg_output_chars    = {enabled['avg_output_chars']:.1f}")
    print(f"  runtime_refreshes   = {enabled['total_runtime_refreshes']}")
    print("Disabled:")
    print(f"  pass_rate           = {disabled['pass_rate']:.2f}")
    print(f"  prompt_hit_rate     = {disabled['prompt_hit_rate']:.2f}")
    print(f"  refresh_hit_rate    = {disabled['refresh_hit_rate']:.2f}")
    print(f"  code_extract_rate   = {disabled['code_extract_rate']:.2f}")
    print(f"  avg_output_chars    = {disabled['avg_output_chars']:.1f}")
    print(f"  runtime_refreshes   = {disabled['total_runtime_refreshes']}")


def _print_demo(report: Dict[str, Any]):
    print("=" * 64)
    print(f"  SHARED MANIFOLD DEMO  |  scenario={report['scenario']}")
    print("=" * 64)
    for line in report["timeline"]:
        print(line)
    print("\nPrompt Context:")
    print(report["prompt_context"] or "<none>")
    print("\nRuntime Shared Calls:")
    if report["shared_calls"]:
        for call in report["shared_calls"]:
            print(call)
    else:
        print("<none>")
    print("\nTelemetry:")
    print(json.dumps(report["metrics"], indent=2))


def _print_coding_demo(report: Dict[str, Any]):
    print("=" * 64)
    print(f"  SHARED MANIFOLD CODING DEMO  |  task={report['coding_task']}")
    print("=" * 64)
    for line in report["timeline"]:
        print(line)
    print("\nCandidate Code:\n")
    print(report["candidate_code"])
    print("\nExecution:")
    if report["passed"]:
        print(report["output"] or "PASS")
    else:
        print(report["error"] or report["output"] or "FAILED")
    print("\nTelemetry:")
    print(json.dumps(report["metrics"], indent=2))


def _print_real_coding_demo(report: Dict[str, Any]):
    print("=" * 64)
    print(f"  REAL SHARED MANIFOLD CODING DEMO  |  task={report['coding_task']}")
    print("=" * 64)
    for line in report["timeline"]:
        print(line)
    print("\nRaw Output:\n")
    print(report["raw_output"] or "<none>")
    print("\nCandidate Code:\n")
    print(report["candidate_code"] or "<none>")
    print("\nExecution:")
    if report["passed"]:
        print(report["output"] or "PASS")
    else:
        print(report["error"] or report["output"] or "FAILED")
    print("\nTelemetry:")
    print(json.dumps(report["metrics"], indent=2))


def _print_real_handoff_compare(report: Dict[str, Any]):
    aggregate = report["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX REAL TWO-INSTANCE HANDOFF")
    print("=" * 64)
    print(f"Model: {report.get('model_id')}  Device: {report.get('device')}")
    print(f"  context_match_rate  = {aggregate['context_match_rate']:.2f}")
    print(f"  output_match_rate   = {aggregate['output_match_rate']:.2f}")
    print(f"  fresh_prompt_hits   = {aggregate['fresh_prompt_hit_rate']:.2f}")
    print(f"  loaded_prompt_hits  = {aggregate['loaded_prompt_hit_rate']:.2f}")
    print(f"  fresh_pass_rate     = {aggregate['fresh_pass_rate']:.2f}")
    print(f"  loaded_pass_rate    = {aggregate['loaded_pass_rate']:.2f}")


def _print_real_handoff_demo(report: Dict[str, Any]):
    writer = report["writer"]
    fresh_reader = report["fresh_reader"]
    loaded_reader = report["loaded_reader"]
    print("=" * 64)
    print(f"  REAL TWO-INSTANCE HANDOFF DEMO  |  task={report['coding_task']}")
    print("=" * 64)
    print(f"Snapshot load stats: {json.dumps(report['snapshot_load'])}")
    print(f"Context match: {report['context_match']}")
    print(f"Output match: {report['output_match']}")
    print("\nWriter Context:\n")
    print(writer["prompt_context"] or "<none>")
    print("\nFresh Reader Context:\n")
    print(fresh_reader["prompt_context"] or "<none>")
    print("\nLoaded Reader Context:\n")
    print(loaded_reader["prompt_context"] or "<none>")
    print("\nLoaded Reader Candidate Code:\n")
    print(loaded_reader["candidate_code"] or "<none>")
    print("\nLoaded Reader Execution:")
    if loaded_reader["passed"]:
        print(loaded_reader["output"] or "PASS")
    else:
        print(loaded_reader["error"] or loaded_reader["output"] or "FAILED")


def _print_real_recall_handoff_compare(report: Dict[str, Any]):
    aggregate = report["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX REAL RECALL HANDOFF")
    print("=" * 64)
    print(f"Model: {report.get('model_id')}  Device: {report.get('device')}")
    print(f"  context_match_rate  = {aggregate['context_match_rate']:.2f}")
    print(f"  writer_answer_rate  = {aggregate['writer_answer_rate']:.2f}")
    print(f"  fresh_prompt_hits   = {aggregate['fresh_prompt_hit_rate']:.2f}")
    print(f"  loaded_prompt_hits  = {aggregate['loaded_prompt_hit_rate']:.2f}")
    print(f"  fresh_answer_rate   = {aggregate['fresh_answer_rate']:.2f}")
    print(f"  loaded_answer_rate  = {aggregate['loaded_answer_rate']:.2f}")


def _print_real_recall_handoff_demo(report: Dict[str, Any]):
    writer = report["writer"]
    fresh_reader = report["fresh_reader"]
    loaded_reader = report["loaded_reader"]
    print("=" * 64)
    print(f"  REAL RECALL HANDOFF DEMO  |  task={report['recall_task']}")
    print("=" * 64)
    print(f"Question: {loaded_reader['question']}")
    print(f"Expected terms: {', '.join(loaded_reader['expected_terms'])}")
    if loaded_reader["expected_fields"]:
        print(f"Expected fields: {json.dumps(loaded_reader['expected_fields'], sort_keys=True)}")
    print(f"Snapshot load stats: {json.dumps(report['snapshot_load'])}")
    print(f"Context match: {report['context_match']}")
    print("\nWriter Context:\n")
    print(writer["prompt_context"] or "<none>")
    print("\nWriter Answer:\n")
    print(writer["answer"] or "<none>")
    print("\nFresh Reader Context:\n")
    print(fresh_reader["prompt_context"] or "<none>")
    print("\nFresh Reader Answer:\n")
    print(fresh_reader["answer"] or "<none>")
    print(f"Fresh Reader Passed: {fresh_reader['passed']}")
    print("\nLoaded Reader Context:\n")
    print(loaded_reader["prompt_context"] or "<none>")
    print("\nLoaded Reader Answer:\n")
    print(loaded_reader["answer"] or "<none>")
    print(f"Loaded Reader Passed: {loaded_reader['passed']}")


def _print_real_necessity_compare(report: Dict[str, Any]):
    aggregate = report["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX REAL MULTI-SESSION NECESSITY")
    print("=" * 64)
    print(f"Model: {report.get('model_id')}  Device: {report.get('device')}")
    print(f"  isolated_prompt_hits = {aggregate['isolated_prompt_hit_rate']:.2f}")
    print(f"  manifold_prompt_hits = {aggregate['manifold_prompt_hit_rate']:.2f}")
    print(f"  isolated_answer_rate = {aggregate['isolated_answer_rate']:.2f}")
    print(f"  manifold_answer_rate = {aggregate['manifold_answer_rate']:.2f}")
    print(f"  oracle_answer_rate   = {aggregate['oracle_answer_rate']:.2f}")
    print(f"  necessity_win_rate   = {aggregate['necessity_win_rate']:.2f}")


def _print_real_necessity_demo(report: Dict[str, Any]):
    isolated = report["isolated_reader"]
    manifold = report["manifold_reader"]
    oracle = report["oracle_reader"]
    print("=" * 64)
    print(f"  REAL MULTI-SESSION NECESSITY DEMO  |  task={report['necessity_task']}")
    print("=" * 64)
    print(f"Writer sessions: {report['writer_session_count']}  Memory nodes: {report['memory_count']}")
    print(f"Question: {report['question']}")
    if report["expected_fields"]:
        print(f"Expected fields: {json.dumps(report['expected_fields'], sort_keys=True)}")
    print(f"Shared store stats: {json.dumps(report['shared_store_stats'], sort_keys=True)}")
    print(f"Necessity win: {report['necessity_win']}")
    print(f"Oracle supported: {report['oracle_supported']}")
    print("\nFlat Oracle Context:\n")
    print(report["flat_context"] or "<none>")
    print("\nIsolated Reader Answer:\n")
    print(isolated["answer"] or "<none>")
    print(f"Isolated Reader Passed: {isolated['passed']}")
    print("\nManifold Reader Context:\n")
    print(manifold["prompt_context"] or "<none>")
    print("\nManifold Reader Answer:\n")
    print(manifold["answer"] or "<none>")
    print(f"Manifold Reader Passed: {manifold['passed']}")
    print("\nFlat Oracle Answer:\n")
    print(oracle["answer"] or "<none>")
    print(f"Flat Oracle Passed: {oracle['passed']}")


def _print_topology_compare(report: Dict[str, Any]):
    aggregate = report["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX TOPOLOGY RETRIEVAL SLICE")
    print("=" * 64)
    print(f"  component_accuracy_rate      = {aggregate['component_accuracy_rate']:.2f}")
    print(f"  active_region_accuracy_rate  = {aggregate['active_region_accuracy_rate']:.2f}")
    print(f"  topology_expected_recall     = {aggregate['topology_expected_recall_rate']:.2f}")
    print(f"  flat_expected_recall         = {aggregate['flat_expected_recall_rate']:.2f}")
    print(f"  topology_bridge_recall       = {aggregate['topology_bridge_recall_rate']:.2f}")
    print(f"  flat_bridge_recall           = {aggregate['flat_bridge_recall_rate']:.2f}")
    print(f"  topology_leakage_rate        = {aggregate['topology_leakage_rate']:.2f}")
    print(f"  flat_leakage_rate            = {aggregate['flat_leakage_rate']:.2f}")
    print(f"  topology_win_rate            = {aggregate['topology_win_rate']:.2f}")


def _print_topology_demo(report: Dict[str, Any]):
    topology = report["topology"]
    flat = report["flat"]
    print("=" * 64)
    print(f"  TOPOLOGY RETRIEVAL DEMO  |  task={report['topology_task']}")
    print("=" * 64)
    print(f"Query: {report['query_text']}")
    print(f"Shared stats: {json.dumps(report['shared_manifold_stats'], sort_keys=True)}")
    print("\nPrompt Context:\n")
    print(report["prompt_context"] or "<none>")
    print("\nTopology Selection:\n")
    print(json.dumps(topology, indent=2, sort_keys=True))
    print("\nFlat Baseline Selection:\n")
    print(json.dumps(flat, indent=2, sort_keys=True))


def _print_real_topology_compare(report: Dict[str, Any]):
    aggregate = report["aggregate"]
    print("=" * 64)
    print("  WARP CORTEX REAL TOPOLOGY RETRIEVAL")
    print("=" * 64)
    print(f"Model: {report.get('model_id')}  Device: {report.get('device')}")
    print(f"  component_accuracy_rate      = {aggregate['component_accuracy_rate']:.2f}")
    print(f"  active_region_accuracy_rate  = {aggregate['active_region_accuracy_rate']:.2f}")
    print(f"  topology_expected_recall     = {aggregate['topology_expected_recall_rate']:.2f}")
    print(f"  flat_expected_recall         = {aggregate['flat_expected_recall_rate']:.2f}")
    print(f"  topology_bridge_recall       = {aggregate['topology_bridge_recall_rate']:.2f}")
    print(f"  flat_bridge_recall           = {aggregate['flat_bridge_recall_rate']:.2f}")
    print(f"  topology_leakage_rate        = {aggregate['topology_leakage_rate']:.2f}")
    print(f"  flat_leakage_rate            = {aggregate['flat_leakage_rate']:.2f}")
    print(f"  topology_answer_rate         = {aggregate['topology_answer_rate']:.2f}")
    print(f"  flat_answer_rate             = {aggregate['flat_answer_rate']:.2f}")
    print(f"  topology_win_rate            = {aggregate['topology_win_rate']:.2f}")


def _print_real_topology_demo(report: Dict[str, Any]):
    print("=" * 64)
    print(f"  REAL TOPOLOGY RETRIEVAL DEMO  |  task={report['topology_task']}")
    print("=" * 64)
    print(f"Question: {report['question']}")
    print(f"Shared stats: {json.dumps(report['shared_manifold_stats'], sort_keys=True)}")
    print("\nTopology Context:\n")
    print(report["topology_context"] or "<none>")
    print("\nTopology Reader Answer:\n")
    print(report["topology_reader"]["answer"] or "<none>")
    print(f"Topology Reader Passed: {report['topology_reader']['passed']}")
    print("\nFlat Context:\n")
    print(report["flat_context"] or "<none>")
    print("\nFlat Reader Answer:\n")
    print(report["flat_reader"]["answer"] or "<none>")
    print(f"Flat Reader Passed: {report['flat_reader']['passed']}")
    print("\nTopology Retrieval:\n")
    print(json.dumps(report["topology_retrieval"], indent=2, sort_keys=True))
    print("\nFlat Retrieval:\n")
    print(json.dumps(report["flat_retrieval"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Warp Cortex shared-manifold benchmark and demo")
    parser.add_argument(
        "--mode",
        choices=[
            "compare",
            "on",
            "off",
            "demo",
            "coding-compare",
            "coding-on",
            "coding-off",
            "coding-demo",
            "real-coding-compare",
            "real-coding-on",
            "real-coding-off",
            "real-coding-demo",
            "topology-compare",
            "topology-demo",
            "real-topology-compare",
            "real-topology-demo",
            "real-handoff-compare",
            "real-handoff-demo",
            "real-recall-handoff-compare",
            "real-recall-handoff-demo",
            "real-necessity-compare",
            "real-necessity-demo",
        ],
        default="compare",
    )
    parser.add_argument("--scenario", default="payment_retry")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.mode == "compare":
        report = compare_pipeline()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_compare(report)
        return

    if args.mode == "demo":
        report = run_demo_scenario(args.scenario)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_demo(report)
        return

    if args.mode == "coding-compare":
        report = compare_coding_slice()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_coding_compare(report)
        return

    if args.mode == "coding-demo":
        report = run_coding_demo(args.scenario)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_coding_demo(report)
        return

    if args.mode == "real-coding-compare":
        report = compare_real_coding_slice(
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_coding_compare(report)
        return

    if args.mode == "real-coding-demo":
        report = run_real_coding_demo(
            args.scenario,
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_coding_demo(report)
        return

    if args.mode == "topology-compare":
        report = compare_topology_slice()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_topology_compare(report)
        return

    if args.mode == "topology-demo":
        report = run_topology_demo(args.scenario)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_topology_demo(report)
        return

    if args.mode == "real-topology-compare":
        report = compare_real_topology_slice(
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_topology_compare(report)
        return

    if args.mode == "real-topology-demo":
        report = run_real_topology_demo(
            args.scenario,
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_topology_demo(report)
        return

    if args.mode == "real-handoff-compare":
        report = compare_real_handoff_slice(
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_handoff_compare(report)
        return

    if args.mode == "real-handoff-demo":
        report = run_real_handoff_demo(
            args.scenario,
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_handoff_demo(report)
        return

    if args.mode == "real-recall-handoff-compare":
        report = compare_real_recall_handoff_slice(
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_recall_handoff_compare(report)
        return

    if args.mode == "real-recall-handoff-demo":
        report = run_real_recall_handoff_demo(
            args.scenario,
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_recall_handoff_demo(report)
        return

    if args.mode == "real-necessity-compare":
        report = compare_real_necessity_slice(
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_necessity_compare(report)
        return

    if args.mode == "real-necessity-demo":
        report = run_real_necessity_demo(
            args.scenario,
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_real_necessity_demo(report)
        return

    if args.mode in {"coding-on", "coding-off"}:
        report = run_coding_probe(args.mode == "coding-on")
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            label = "enabled" if args.mode == "coding-on" else "disabled"
            print(json.dumps({label: report["aggregate"]}, indent=2))
        return

    if args.mode in {"real-coding-on", "real-coding-off"}:
        report = run_real_coding_probe(
            args.mode == "real-coding-on",
            model_id=args.model_id,
            device=args.device,
            max_tokens=args.max_tokens,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            label = "enabled" if args.mode == "real-coding-on" else "disabled"
            print(json.dumps({label: report["aggregate"]}, indent=2))
        return

    report = run_probe(args.mode == "on")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        label = "enabled" if args.mode == "on" else "disabled"
        print(json.dumps({label: report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()