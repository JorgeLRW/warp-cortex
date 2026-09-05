"""
Cortex Memory Manager: Auto-Compaction & Persistent Skills
==========================================================
Inspired by Claude Code's PreCompact hook, auto-compaction (at 75% context
capacity), and the SKILL.md persistent tool pattern.

Two subsystems:
1. AutoCompactor — monitors KV cache utilization and triggers topological
   landmarking when capacity crosses a threshold, preventing OOM.
2. PersistentSkill — lets agents carry "tools" (CLI commands, domain rules,
   project conventions) that survive session resets, loaded from JSON files.
"""

import torch
import json
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any


# ======================================================================
# 1. Auto-Compaction
# ======================================================================

@dataclass
class CompactionEvent:
    """Record of a compaction that occurred."""
    timestamp: float
    seq_len_before: int
    seq_len_after: int
    capacity_pct: float
    trigger: str  # "threshold" | "manual" | "pre_injection"


class AutoCompactor:
    """
    Monitors KV-cache utilization and triggers topological landmarking
    before the context window overflows.

    Lifecycle hooks (inspired by Claude Code's PreCompact/PostCompact):
      - pre_compact_hooks:  run BEFORE compaction (e.g., save critical state)
      - post_compact_hooks: run AFTER compaction  (e.g., log, notify agents)
    """

    def __init__(
        self,
        max_seq_len: int = 32768,
        compact_threshold: float = 0.75,
        target_utilization: float = 0.30,
        landmark_k: int = 64,
    ):
        """
        Args:
            max_seq_len: Model's maximum context window.
            compact_threshold: Trigger compaction when utilization >= this (0-1).
            target_utilization: After compaction, aim for this utilization.
            landmark_k: Number of landmarks to keep during compaction.
        """
        self.max_seq_len = max_seq_len
        self.compact_threshold = compact_threshold
        self.target_utilization = target_utilization
        self.landmark_k = landmark_k

        # Hook registries
        self.pre_compact_hooks: List[Callable[[float], None]] = []
        self.post_compact_hooks: List[Callable[[CompactionEvent], None]] = []

        # Telemetry
        self.history: List[CompactionEvent] = []
        self._lock = threading.Lock()

    @property
    def compact_at(self) -> int:
        """Sequence length that triggers compaction."""
        return int(self.max_seq_len * self.compact_threshold)

    def utilization(self, past_key_values) -> float:
        """Current KV-cache utilization as a fraction of max_seq_len."""
        if past_key_values is None:
            return 0.0
        # past_key_values: tuple of (key, value) per layer
        # key shape: [Batch, Heads, Seq, Dim]
        try:
            if hasattr(past_key_values, 'get_seq_length'):
                seq_len = past_key_values.get_seq_length()
            else:
                seq_len = past_key_values[0][0].shape[2]
            return seq_len / self.max_seq_len
        except (IndexError, AttributeError):
            return 0.0

    def should_compact(self, past_key_values) -> bool:
        """Check if compaction should be triggered."""
        return self.utilization(past_key_values) >= self.compact_threshold

    def compact(self, past_key_values, synapse, query_states=None) -> Any:
        """
        Run the full compaction pipeline:
        1. Fire pre-compact hooks
        2. Trigger topological landmarking via the synapse
        3. Return the compressed KV cache (landmarks)
        4. Fire post-compact hooks

        Args:
            past_key_values: The current full KV cache.
            synapse: The TopologicalSynapse instance.
            query_states: Optional query for attention-based selection.

        Returns:
            The compacted past_key_values (landmark-only cache).
        """
        with self._lock:
            util_before = self.utilization(past_key_values)
            seq_before = self._get_seq_len(past_key_values)

            # 1. Pre-compact hooks
            for hook in self.pre_compact_hooks:
                hook(util_before)

            print(f"[Compactor] Triggering at {util_before:.0%} utilization "
                  f"({seq_before}/{self.max_seq_len} tokens)")

            # 2. Run topological landmarking
            synapse.update_kv_landmarks(
                past_key_values,
                query_states=query_states,
                keep_ratio=self.landmark_k / max(seq_before, 1),
            )
            compacted = synapse.get_landmarks()

            seq_after = self._get_seq_len(compacted) if compacted else 0
            util_after = seq_after / self.max_seq_len

            # 3. Record event
            event = CompactionEvent(
                timestamp=time.time(),
                seq_len_before=seq_before,
                seq_len_after=seq_after,
                capacity_pct=util_before,
                trigger="threshold",
            )
            self.history.append(event)

            print(f"[Compactor] Compacted: {seq_before} → {seq_after} tokens "
                  f"({util_after:.0%} utilization)")

            # 4. Post-compact hooks
            for hook in self.post_compact_hooks:
                hook(event)

            return compacted

    def check_and_compact(self, past_key_values, synapse, query_states=None):
        """
        Convenience: only compact if threshold is crossed.
        Returns original KV cache if no compaction needed.
        """
        if self.should_compact(past_key_values):
            return self.compact(past_key_values, synapse, query_states)
        return past_key_values

    def adaptive_k(self, past_key_values) -> int:
        """
        Dynamically adjust landmark count based on current utilization.
        Low utilization → fewer landmarks (save compute).
        High utilization → more landmarks (preserve context).
        """
        util = self.utilization(past_key_values)
        k_min = 16
        k_max = self.landmark_k * 2
        return int(k_min + (k_max - k_min) * util)

    def _get_seq_len(self, kv) -> int:
        if kv is None:
            return 0
        try:
            if hasattr(kv, 'get_seq_length'):
                return kv.get_seq_length()
            return kv[0][0].shape[2]
        except (IndexError, AttributeError):
            return 0


# ======================================================================
# 2. Persistent Skills
# ======================================================================

@dataclass
class Skill:
    """
    A persistent "tool" that an agent can invoke.
    Inspired by SKILL.md — survives session resets, loaded from disk.
    """
    name: str
    description: str
    trigger_patterns: List[str] = field(default_factory=list)  # regex patterns
    system_prompt: str = ""    # Injected when skill activates
    commands: List[str] = field(default_factory=list)  # CLI commands this skill can run
    rules: List[str] = field(default_factory=list)     # Constraints / domain rules
    enabled: bool = True


class SkillRegistry:
    """
    Loads and manages Persistent Skills from JSON agent skill files.
    Skills give sub-agents domain-specific capabilities without
    retraining or fine-tuning.

    Usage:
        registry = SkillRegistry("cortex_resources/agent_skills")
        skill = registry.match("write a unit test for the parser")
        # skill.system_prompt → "[System: Follow TDD. Run pytest after each change...]"
    """

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills: Dict[str, Skill] = {}
        self._skills_dir = skills_dir
        if skills_dir and os.path.isdir(skills_dir):
            self._load_from_dir(skills_dir)

    def register(self, skill: Skill):
        """Register a skill programmatically."""
        self.skills[skill.name] = skill

    def match(self, text: str) -> Optional[Skill]:
        """
        Find the best matching skill for a given text.
        Checks trigger patterns (regex) against the input.
        """
        import re
        text_lower = text.lower()
        for skill in self.skills.values():
            if not skill.enabled:
                continue
            for pattern in skill.trigger_patterns:
                if re.search(pattern, text_lower):
                    return skill
        return None

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        return [f"{s.name}: {s.description}" for s in self.skills.values() if s.enabled]

    def save(self, filepath: Optional[str] = None):
        """Persist all skills to a JSON file."""
        filepath = filepath or os.path.join(self._skills_dir or ".", "skills.json")
        data = {}
        for name, skill in self.skills.items():
            data[name] = {
                "description": skill.description,
                "trigger_patterns": skill.trigger_patterns,
                "system_prompt": skill.system_prompt,
                "commands": skill.commands,
                "rules": skill.rules,
                "enabled": skill.enabled,
            }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Skills] Saved {len(data)} skills to {filepath}")

    def _load_from_dir(self, skills_dir: str):
        """Load skills from all JSON files in a directory."""
        for fname in os.listdir(skills_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(skills_dir, fname)
                self._load_file(fpath)

    def _load_file(self, filepath: str):
        """Load skills from a single JSON file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            for name, sdata in data.items():
                skill = Skill(
                    name=name,
                    description=sdata.get("description", ""),
                    trigger_patterns=sdata.get("trigger_patterns", []),
                    system_prompt=sdata.get("system_prompt", ""),
                    commands=sdata.get("commands", []),
                    rules=sdata.get("rules", []),
                    enabled=sdata.get("enabled", True),
                )
                self.skills[name] = skill
            print(f"[Skills] Loaded {len(data)} skills from {filepath}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Skills] Failed to load {filepath}: {e}")


# ======================================================================
# 3. Context Manager (wraps synapse + compaction + skills)
# ======================================================================

class ContextManager:
    """
    Unified context management layer that combines:
    - Auto-compaction (monitors KV cache, triggers landmarking)
    - Skill injection (augments agent prompts with relevant skills)
    - Capacity telemetry

    Sits between CortexEngine and the TopologicalSynapse.
    """

    def __init__(
        self,
        synapse,
        compactor: AutoCompactor,
        skill_registry: Optional[SkillRegistry] = None,
        shared_context_getter: Optional[Callable[[str], str]] = None,
    ):
        self.synapse = synapse
        self.compactor = compactor
        self.skills = skill_registry or SkillRegistry()
        self.shared_context_getter = shared_context_getter

    def step(self, past_key_values, query_states=None):
        """
        Called every generation step by the engine.
        Returns (possibly compacted) past_key_values.
        """
        return self.compactor.check_and_compact(
            past_key_values, self.synapse, query_states
        )

    def enrich_prompt(self, prompt: str) -> str:
        """
        If a persistent skill matches the prompt, prepend its system prompt.
        """
        enriched = prompt
        skill = self.skills.match(prompt)
        if skill:
            print(f"[Context] Activated skill: {skill.name}")
            enriched = f"{skill.system_prompt}\n{enriched}"

        if self.shared_context_getter is not None:
            shared_context = self.shared_context_getter(prompt)
            if shared_context:
                print("[Context] Activated shared manifold context")
                enriched = f"{shared_context}\n{enriched}"

        return enriched

    def get_stats(self) -> Dict[str, Any]:
        return {
            "compaction_count": len(self.compactor.history),
            "skills_loaded": len(self.skills.skills),
            "last_compaction": self.compactor.history[-1] if self.compactor.history else None,
        }
