"""
Agent Council: Multi-Agent Research Decision Workflows.
======================================================
Defines autonomous specialized agent workflows that observe events,
request context from memory substrates, and make operational research commitments.

Tracks:
  - LLM Calls & Tokens packed
  - Decisions (COMMIT vs HALT)
  - Stale / Incorrect Commits (critical safety metric)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cortex_apps.research_agent_system.memory_baselines import RetrievalResult


@dataclass
class AgentDecision:
    agent_id: str
    action: str  # "COMMIT", "HALT", "DEFER"
    reasoning: str
    is_correct: bool
    context_tokens: int
    is_stale_commit: bool = False


@dataclass
class AgentWorkloadMetrics:
    agent_invocations: int = 0
    total_tokens_consumed: int = 0
    correct_decisions: int = 0
    incorrect_decisions: int = 0
    stale_commits: int = 0
    halt_decisions: int = 0
    commit_decisions: int = 0


class ExecutiveScaleUpAgent:
    """
    High-stakes executive decision agent.
    Evaluates whether to commit $250k capital for Bioreactor Pilot Run Alpha
    or $180k for In Vivo PK assays based on the assembled context bundle.
    """

    def __init__(self, agent_id: str = "agent_executive"):
        self.agent_id = agent_id

    def evaluate_scaleup_request(
        self,
        retrieval: RetrievalResult,
        ground_truth_status: str,  # "NOMINAL", "TAINTED_UNRESOLVED", or "REMEDIATED"
    ) -> AgentDecision:
        """
        Frozen downstream decision procedure for scale-up commitment.
        Reads visible retrieved items up to token budget.
        """
        has_sop = False
        has_dataset_42 = False
        dataset_42_source = None  # "MS-4" or "MS-2"
        dataset_42_tainted = False
        ms4_tainted = False
        ms2_tainted = False
        remediation_verified = False

        for it in retrieval.items:
            t = it.title.lower()
            c = it.content.lower()
            tag = it.state_tag

            if "scale-up commitment sop" in t or "bioreactor pilot run" in t:
                has_sop = True

            if it.doc_id == "doc_ds_data42" or "dataset 42: tandem" in t or "dataset 42 proteomics" in t or ("dataset 42" in t and "spectra" in t):
                has_dataset_42 = True
                if "ms-4" in c or "quadrupole ms-4" in c:
                    dataset_42_source = "MS-4"
                elif "ms-2" in c or "quadrupole ms-2" in c:
                    dataset_42_source = "MS-2"
                if tag in ("TAINTED", "SUSPECT", "DRIFT"):
                    dataset_42_tainted = True

            if "ms-4" in t or "ms-4" in c or it.doc_id == "doc_inst_ms4":
                if tag in ("TAINTED", "SUSPECT", "DRIFT") or any(k in c for k in ("ion transmission efficiency dropped", "calibration boundary breached", "drift observed", "critical alert")):
                    ms4_tainted = True

            if "ms-2" in t or "ms-2" in c or it.doc_id == "doc_inst_ms2":
                if tag in ("TAINTED", "SUSPECT", "DRIFT"):
                    ms2_tainted = True

            if any(k in c for k in ("maintenance resolved", "recalibration certificate", "recalibration verified", "calibration restored")):
                if tag in ("VALID", "ACTIVE", ""):
                    remediation_verified = True

            # In periodic summary scratchpad:
            if it.doc_id == "summary_scratchpad" and "active alerts detected" in c and any(k in c for k in ("mass spectrometer", "ms-4", "mass spec")):
                ms4_tainted = True

        # Decision rule:
        # If no SOP present -> cannot authorize (HALT)
        if not has_sop:
            decision = "HALT"
            reason = "Missing authorization SOP in retrieved context."
        elif not has_dataset_42:
            # Cannot verify whether pilot run depends on tainted MS-4 or nominal MS-2
            if ms4_tainted and not remediation_verified:
                decision = "HALT"
                reason = "Unverified provenance: Active MS-4 anomaly detected, cannot verify if pilot dataset is affected."
            else:
                decision = "COMMIT"
                reason = "SOP verified, no visible anomaly."
        else:
            # Verified empirical provenance from Dataset 42 in context
            if dataset_42_source == "MS-4":
                if (ms4_tainted or dataset_42_tainted) and not remediation_verified:
                    decision = "HALT"
                    reason = "Critical: Dataset 42 acquired from tainted Quadrupole MS-4 on pilot causal path."
                else:
                    decision = "COMMIT"
                    reason = "Dataset 42 verified from MS-4; MS-4 nominal/remediated."
            elif dataset_42_source == "MS-2":
                if ms2_tainted and not remediation_verified:
                    decision = "HALT"
                    reason = "Critical: Dataset 42 acquired from tainted Quadrupole MS-2."
                else:
                    decision = "COMMIT"
                    reason = "Verified nominal: Dataset 42 acquired from nominal Quadrupole MS-2. Unaffected by ambient MS-4 drift."
            else:
                if ms4_tainted and not remediation_verified:
                    decision = "HALT"
                    reason = "Unverified provenance: Dataset 42 present but source instrument unconfirmed with active MS-4 drift."
                else:
                    decision = "COMMIT"
                    reason = "Dataset 42 present, no anomaly link."

        # Ground truth correctness:
        # If world is TAINTED_UNRESOLVED -> ground truth is HALT. A COMMIT is a STALE/INCORRECT commit!
        # If world is NOMINAL or REMEDIATED -> ground truth is COMMIT.
        if ground_truth_status == "TAINTED_UNRESOLVED":
            is_correct = (decision == "HALT")
            is_stale_commit = (decision == "COMMIT")
        else:
            is_correct = (decision == "COMMIT")
            is_stale_commit = False

        return AgentDecision(
            agent_id=self.agent_id,
            action=decision,
            reasoning=reason,
            is_correct=is_correct,
            context_tokens=retrieval.total_tokens,
            is_stale_commit=is_stale_commit,
        )


class DataIntegrityMonitorAgent:
    """
    Monitors instrument calibration telemetry and flags contaminated datasets.
    """

    def __init__(self, agent_id: str = "agent_data_monitor"):
        self.agent_id = agent_id

    def evaluate_sensor_telemetry(
        self,
        retrieval: RetrievalResult,
        ground_truth_has_drift: bool,
    ) -> AgentDecision:
        drift_detected = False
        for it in retrieval.items:
            c = it.content.lower()
            if (
                any(k in c for k in ("drift observed", "tolerance exceeded", "calibration boundary breached", "efficiency dropped"))
                or it.state_tag in ("TAINTED", "SUSPECT", "DRIFT")
            ):
                if "ms-4" in c or "quadrupole" in c or it.doc_id == "doc_inst_ms4":
                    drift_detected = True
                    break

        action = "FLAG_ANOMALY" if drift_detected else "NORMAL"
        is_correct = (drift_detected == ground_truth_has_drift)

        return AgentDecision(
            agent_id=self.agent_id,
            action=action,
            reasoning="Telemetry evaluation complete.",
            is_correct=is_correct,
            context_tokens=retrieval.total_tokens,
            is_stale_commit=False,
        )
