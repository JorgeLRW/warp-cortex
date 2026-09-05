"""
Skill Registry and Skill Experience Ledger.
============================================
Implements:
  - SkillDefinition: Typed procedural knowledge.
  - SkillInvocationEvent: Rich execution record saved directly into H_v.
  - SkillRegistry: Versioned catalog of available skills K.
  - SkillSelector: 4-way skill scorer integrating Z, G, S, and H.
    Supports 3 evaluation modes:
      1. STATIC: Pure description/aspect matching (Hermes baseline).
      2. PRIVATE_MEMORY: Uses only the current agent's private history.
      3. SHARED_CORTEX_LEDGER: Learns cross-agent from all past invocations in H_v.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class SkillSelectionMode(str, Enum):
    STATIC = "STATIC"
    PRIVATE_MEMORY = "PRIVATE_MEMORY"
    SHARED_CORTEX_LEDGER = "SHARED_CORTEX_LEDGER"


@dataclass
class SkillDefinition:
    skill_id: str
    version: str
    name: str
    description: str
    aspect_tags: List[str] = field(default_factory=list)
    prerequisites: Dict[str, Any] = field(default_factory=dict)
    expected_effects: List[str] = field(default_factory=list)
    cost_estimate_ms: float = 10.0
    permissions: List[str] = field(default_factory=list)
    execute: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None

    @property
    def full_id(self) -> str:
        return f"{self.skill_id}@{self.version}"


@dataclass
class SkillInvocationEvent:
    invocation_id: str
    skill_id: str
    skill_version: str
    agent_id: str
    world_version: int
    task_query: str
    inputs: Dict[str, Any]
    success: bool
    outcome_summary: str
    latency_ms: float
    token_cost: int
    error_type: Optional[str] = None
    discovered_constraints: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    # Portable-project scope: SHARED_CORTEX_LEDGER selection is scoped to one
    # project world by default. Cross-project learning must be explicitly
    # allowed by the caller, never default. Defaults preserve legacy behavior.
    project_id: str = "default"


class SkillRegistry:
    """
    The Versioned Skill Catalog K.
    """

    def __init__(self):
        self._skills: Dict[str, Dict[str, SkillDefinition]] = {}  # skill_id -> version -> Skill

    def register(self, skill: SkillDefinition) -> None:
        if skill.skill_id not in self._skills:
            self._skills[skill.skill_id] = {}
        self._skills[skill.skill_id][skill.version] = skill

    def get(self, skill_id: str, version: Optional[str] = None) -> Optional[SkillDefinition]:
        if skill_id not in self._skills:
            return None
        versions = self._skills[skill_id]
        if not versions:
            return None
        if version and version in versions:
            return versions[version]
        # Return latest registered version
        latest_ver = sorted(list(versions.keys()))[-1]
        return versions[latest_ver]

    def list_skills(self, include_all_versions: bool = True) -> List[SkillDefinition]:
        out: List[SkillDefinition] = []
        for sid, versions in self._skills.items():
            if include_all_versions:
                for v in versions.values():
                    out.append(v)
            else:
                latest = sorted(list(versions.keys()))[-1]
                out.append(versions[latest])
        return out


class SkillSelector:
    """
    Scores and selects skills using:
      SkillScore(k | q, U_v) = f(Semantic(Z), Prerequisites(G), State(S), History(H))
    """

    def __init__(
        self,
        registry: SkillRegistry,
        mode: SkillSelectionMode = SkillSelectionMode.SHARED_CORTEX_LEDGER,
    ):
        self.registry = registry
        self.mode = mode
        # Private per-agent history cache (used only in PRIVATE_MEMORY mode)
        self.private_agent_history: Dict[str, List[SkillInvocationEvent]] = {}

    def record_invocation(
        self,
        event: SkillInvocationEvent,
        shared_history_buffer: Optional[List[SkillInvocationEvent]] = None,
    ) -> None:
        """Records an execution outcome into private and/or shared history."""
        # Always update private history
        if event.agent_id not in self.private_agent_history:
            self.private_agent_history[event.agent_id] = []
        self.private_agent_history[event.agent_id].append(event)

        # Update shared history ledger if provided
        if shared_history_buffer is not None:
            shared_history_buffer.append(event)

    def select_skill(
        self,
        query: str,
        world_snapshot: Any,
        agent_id: str,
        shared_history: Optional[List[SkillInvocationEvent]] = None,
        top_k: int = 3,
        project_scope: Optional[str] = None,
    ) -> List[Tuple[SkillDefinition, float, str]]:
        """
        Ranks applicable skills for task query `q` under world snapshot U_v.
        Returns list of (SkillDefinition, score, explanation).
        project_scope filters the history pool to one project world when set.
        """
        all_skills = self.registry.list_skills(include_all_versions=True)
        scored: List[Tuple[SkillDefinition, float, str]] = []

        q_terms = set(re.findall(r"\w+", query.lower()))

        # Determine history view based on selection mode
        if self.mode == SkillSelectionMode.STATIC:
            history_pool: List[SkillInvocationEvent] = []
        elif self.mode == SkillSelectionMode.PRIVATE_MEMORY:
            history_pool = self.private_agent_history.get(agent_id, [])
        elif self.mode == SkillSelectionMode.SHARED_CORTEX_LEDGER:
            history_pool = shared_history if shared_history is not None else []
        else:
            history_pool = []

        if project_scope is not None:
            history_pool = [ev for ev in history_pool
                            if getattr(ev, "project_id", "default") == project_scope]

        for skill in all_skills:
            # 1. Semantic Score (Z)
            desc_terms = set(re.findall(r"\w+", (skill.name + " " + skill.description + " " + " ".join(skill.aspect_tags)).lower()))
            q_prefixes = {w[:4] for w in q_terms if len(w) >= 3}
            desc_prefixes = {w[:4] for w in desc_terms if len(w) >= 3}
            prefix_overlap = len(q_prefixes.intersection(desc_prefixes))
            exact_overlap = len(q_terms.intersection(desc_terms))
            overlap = exact_overlap * 2 + prefix_overlap
            s_semantic = (overlap / max(1, len(q_terms) * 2)) * 5.0

            # Filter out completely irrelevant skills (no semantic overlap with query)
            if overlap == 0:
                continue

            # 2. Prerequisite Check (S & G)
            s_prereq = 1.0
            prereq_violations: List[str] = []
            if hasattr(world_snapshot, "state"):
                for req_key, req_val in skill.prerequisites.items():
                    curr_val = world_snapshot.state.get(req_key)
                    if curr_val != req_val:
                        s_prereq -= 3.0
                        prereq_violations.append(f"{req_key}!={req_val}")

            # 3. Historical Experience Score (H)
            s_history = 0.0
            history_notes = []

            # Version-specific invocations
            ver_invocations = [
                ev for ev in history_pool
                if ev.skill_id == skill.skill_id and ev.skill_version == skill.version
            ]
            if ver_invocations:
                wins = sum(1 for ev in ver_invocations if ev.success)
                losses = len(ver_invocations) - wins
                win_rate = wins / len(ver_invocations)
                s_history += (win_rate - 0.5) * 4.0
                history_notes.append(f"{wins}W/{losses}L for {skill.version}")

            # Family-level discovered constraint matching
            family_failures = [
                ev for ev in history_pool
                if ev.skill_id == skill.skill_id and not ev.success and ev.discovered_constraints
            ]
            for ev in family_failures:
                for disc_k, disc_v in ev.discovered_constraints.items():
                    # Check if this skill addresses the constraint in its aspect tags or description
                    skill_text = (skill.name + " " + skill.description + " " + " ".join(skill.aspect_tags)).lower()
                    if disc_k.lower().replace("_", "") in skill_text.replace("_", "") or any(tag.lower() in disc_k.lower() for tag in skill.aspect_tags):
                        s_history += 3.0
                        history_notes.append(f"Addresses learned constraint {disc_k}")
                    elif skill.version == ev.skill_version:
                        s_history -= 3.0
                        history_notes.append(f"Suffers unaddressed constraint {disc_k}")

            total_score = s_semantic + s_prereq + s_history
            explanation = (
                f"Z={s_semantic:.2f}, S_prereq={s_prereq:.1f}"
                + (f" ({', '.join(prereq_violations)})" if prereq_violations else "")
                + f", H={s_history:.2f}"
                + (f" [{', '.join(history_notes)}]" if history_notes else "")
            )
            scored.append((skill, total_score, explanation))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
