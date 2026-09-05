"""
Developer Agents for Cortex Dev Runtime.
=========================================
The three focused development agents:
  1. ImplementationAgent: Context retrieval & patch generation.
  2. VerificationAgent: Invariant checking & targeted test execution.
  3. ReviewerAgent: Causal explanation audit & merge decision.

CRITICAL INVARIANT: The agents are 100% IDENTICAL across both
Architecture A and Architecture B. Only the underlying substrate changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cortex_apps.cortex_dev_runtime.dev_runtime_api import (
    DevContextSubstrate,
    DevEvent,
    FileNode,
    PatchDiff,
    VerificationReport,
)


@dataclass
class AgentTaskOutcome:
    task_id: str
    success: bool
    version: int
    verification: VerificationReport
    explanation: Dict[str, Any]
    agent_steps: int
    reconstructed_tokens: int


class ImplementationAgent:
    """Agent responsible for assembling task context and proposing code patches."""

    def __init__(self, substrate: DevContextSubstrate):
        self.substrate = substrate

    def prepare_context(self, task_description: str, budget: int = 2000) -> List[FileNode]:
        return self.substrate.context(task_description, token_budget=budget)

    def apply_patch(self, patch: PatchDiff) -> int:
        return self.substrate.apply_patch(patch)


class VerificationAgent:
    """Agent responsible for validating patch safety and impacted tests."""

    def __init__(self, substrate: DevContextSubstrate):
        self.substrate = substrate

    def verify_patch(self, patch: PatchDiff) -> VerificationReport:
        return self.substrate.verify(patch)


class ReviewerAgent:
    """Agent responsible for reviewing verification reports and causal audit trails."""

    def __init__(self, substrate: DevContextSubstrate):
        self.substrate = substrate

    def audit_and_decide(self, patch: PatchDiff, report: VerificationReport) -> Tuple[bool, Dict[str, Any]]:
        explanation = self.substrate.explain(patch.patch_id)
        # Reviewer policy: permit must be True, zero syntax errors, zero test failures
        approved = report.permit and (len(report.syntax_errors) == 0) and (len(report.failed_tests) == 0)
        return approved, explanation


class DevAgentCoordinator:
    """Coordinates the 3 agents through a unified task cycle."""

    def __init__(self, substrate: DevContextSubstrate):
        self.substrate = substrate
        self.impl_agent = ImplementationAgent(substrate)
        self.veri_agent = VerificationAgent(substrate)
        self.rev_agent = ReviewerAgent(substrate)

    def execute_task(
        self,
        task_id: str,
        task_description: str,
        patch: PatchDiff,
    ) -> AgentTaskOutcome:
        # Step 1: Implementation agent retrieves context
        context_files = self.impl_agent.prepare_context(task_description)

        # Step 2: Verification agent tests patch before permanent commit
        veri_report = self.veri_agent.verify_patch(patch)

        # Step 3: Reviewer agent decides whether to commit
        approved, explanation = self.rev_agent.audit_and_decide(patch, veri_report)

        if approved:
            v_after = self.impl_agent.apply_patch(patch)
        else:
            v_after = getattr(self.substrate, "version", 1)

        metrics = self.substrate.get_metrics()
        return AgentTaskOutcome(
            task_id=task_id,
            success=approved,
            version=v_after,
            verification=veri_report,
            explanation=explanation,
            agent_steps=3,
            reconstructed_tokens=metrics.tokens_reconstructed,
        )
