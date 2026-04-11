"""
Cortex Orchestrator: Parallel Agent Teams with Specialized Roles
================================================================
Inspired by Claude Code's Task tool dispatch and Agent Teams (Research Preview).

Instead of spawning generic worker threads, we dispatch specialized sub-agents
(Researcher, Reviewer, Coder, Verifier) that work in parallel with dependency
resolution and result aggregation.

This sits on top of the Singleton Weight Sharing—every agent still shares the
same model weights and uses O(k) landmark context via the Topological Synapse.
"""

import torch
import threading
import time
import uuid
from contextlib import nullcontext
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future


class AgentRole(Enum):
    """Specialized roles a sub-agent can assume."""
    RESEARCHER = "researcher"   # Search/verify facts
    REVIEWER   = "reviewer"     # Critique & check logic
    CODER      = "coder"        # Write & verify code
    VERIFIER   = "verifier"     # Validate outputs against constraints
    ARCHITECT  = "architect"    # Plan complex multi-step reasoning


# Role → system prompt prefix injected before the agent's task
ROLE_PROMPTS = {
    AgentRole.RESEARCHER: "[System: You are a research sub-process. Find and verify factual information. Be precise and cite evidence.]",
    AgentRole.REVIEWER:   "[System: You are a review sub-process. Critically evaluate the logic, identify flaws, and suggest corrections.]",
    AgentRole.CODER:      "[System: You are a coding sub-process. Write correct, minimal code. Test edge cases mentally.]",
    AgentRole.VERIFIER:   "[System: You are a verification sub-process. Check the result against the original constraints. Report pass/fail.]",
    AgentRole.ARCHITECT:  "[System: You are a planning sub-process. Break the problem into sub-tasks and define the execution order.]",
}


@dataclass
class SubAgentTask:
    """A unit of work to be dispatched to a sub-agent."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: AgentRole = AgentRole.RESEARCHER
    description: str = ""
    priority: int = 1          # Higher = more urgent (0=background, 1=normal, 2=urgent)
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    max_tokens: int = 30
    # Populated after execution
    result: Optional[str] = None
    result_vector: Optional[torch.Tensor] = None
    status: str = "pending"    # pending → running → completed | failed | rejected


@dataclass
class TeamPlan:
    """
    An Agent Team: a coordinated group of sub-agents solving a complex problem.
    The Architect role decomposes the problem, then parallel agents execute.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    tasks: List[SubAgentTask] = field(default_factory=list)
    final_result: Optional[str] = None
    status: str = "planning"   # planning → executing → aggregating → done


class CortexOrchestrator:
    """
    Manages parallel dispatch of specialized sub-agents.

    Key differences from the basic CortexRouter:
    - Typed roles with domain-specific system prompts
    - Dependency resolution (task B waits for task A)
    - Result aggregation across a team
    - Priority scheduling via thread pool
    """

    def __init__(self, engine, max_workers: int = 8):
        """
        Args:
            engine: Reference to CortexEngine (provides model, tokenizer, synapse).
            max_workers: Max concurrent sub-agent threads.
        """
        self.engine = engine
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cortex-agent")
        self.tasks: Dict[str, SubAgentTask] = {}
        self.futures: Dict[str, Future] = {}
        self.results_lock = threading.Lock()
        self._on_thought_injected: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Single-agent dispatch
    # ------------------------------------------------------------------

    def dispatch(self, task: SubAgentTask) -> str:
        """
        Submit a single sub-agent task. Returns the task ID.
        The agent runs asynchronously in the thread pool.
        """
        self.tasks[task.id] = task
        future = self.pool.submit(self._execute_task, task)
        self.futures[task.id] = future
        print(f"[Orchestrator] Dispatched {task.role.value} agent ({task.id}): {task.description[:60]}")
        return task.id

    def dispatch_from_trigger(self, trigger_text: str, role: Optional[AgentRole] = None) -> str:
        """
        Convenience: create a SubAgentTask from a router trigger string.
        Auto-infers role if not provided.
        """
        if role is None:
            role = self._infer_role(trigger_text)
        task = SubAgentTask(role=role, description=trigger_text)
        return self.dispatch(task)

    # ------------------------------------------------------------------
    # Team dispatch (multi-agent coordination)
    # ------------------------------------------------------------------

    def dispatch_team(self, goal: str, tasks: List[SubAgentTask]) -> TeamPlan:
        """
        Dispatch a coordinated team of sub-agents with dependency resolution.
        Tasks without dependencies start immediately; others wait.

        Returns a TeamPlan that can be polled for completion.
        """
        plan = TeamPlan(goal=goal, tasks=tasks, status="executing")

        for t in tasks:
            self.tasks[t.id] = t

        # Sort by dependency depth (topological order)
        ready, blocked = self._partition_tasks(tasks)

        # Start dependency-free tasks immediately
        for t in ready:
            future = self.pool.submit(self._execute_task, t)
            self.futures[t.id] = future

        # Start a coordinator thread that unblocks dependent tasks
        coordinator = threading.Thread(
            target=self._coordinate_team, args=(plan, blocked), daemon=True
        )
        coordinator.start()

        print(f"[Orchestrator] Team '{goal}' dispatched: {len(ready)} immediate, {len(blocked)} blocked")
        return plan

    def create_review_chain(self, description: str) -> TeamPlan:
        """
        Common pattern: Researcher → Reviewer → Verifier chain.
        Each stage depends on the previous.
        """
        t1 = SubAgentTask(role=AgentRole.RESEARCHER, description=f"Research: {description}", priority=2)
        t2 = SubAgentTask(role=AgentRole.REVIEWER, description=f"Review the research findings", depends_on=[t1.id])
        t3 = SubAgentTask(role=AgentRole.VERIFIER, description=f"Verify the reviewed answer", depends_on=[t2.id])
        return self.dispatch_team(goal=description, tasks=[t1, t2, t3])

    # ------------------------------------------------------------------
    # Status & results
    # ------------------------------------------------------------------

    def get_task_result(self, task_id: str) -> Optional[str]:
        with self.results_lock:
            task = self.tasks.get(task_id)
            if task and task.status == "completed":
                return task.result
        return None

    def get_team_status(self, plan: TeamPlan) -> Dict[str, str]:
        return {t.id: t.status for t in plan.tasks}

    def wait_for_task(self, task_id: str, timeout: float = 30.0) -> Optional[str]:
        """Block until a task completes or timeout."""
        future = self.futures.get(task_id)
        if future:
            future.result(timeout=timeout)
        return self.get_task_result(task_id)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute_task(self, task: SubAgentTask):
        """Run a single sub-agent task using the shared engine."""
        task.status = "running"

        # Wait for dependencies
        for dep_id in task.depends_on:
            dep_future = self.futures.get(dep_id)
            if dep_future:
                dep_future.result(timeout=60.0)
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status != "completed":
                task.status = "failed"
                task.result = f"Dependency {dep_id} failed"
                return

        # Build the prompt with role-specific system prefix
        role_prompt = ROLE_PROMPTS.get(task.role, "")

        # Gather upstream results for dependent tasks
        upstream_context = ""
        for dep_id in task.depends_on:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.result:
                upstream_context += f" [Upstream ({dep_task.role.value}): {dep_task.result}]"

        full_prompt = f"{role_prompt} Task: {task.description}.{upstream_context} Analysis: "

        try:
            thought_text, thought_vector = self._run_side_agent(full_prompt, task.max_tokens)
            task.result = thought_text
            task.result_vector = thought_vector
            task.status = "completed"
            print(f"[Orchestrator] {task.role.value} ({task.id}) completed: {thought_text[:80]}")

            # Inject into synapse
            self.engine.synapse.push_thought(
                f"[{task.role.value.title()}: {thought_text}]", thought_vector
            )

        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            print(f"[Orchestrator] {task.role.value} ({task.id}) FAILED: {e}")

    def _run_side_agent(self, prompt_text: str, max_tokens: int = 30):
        """
        Execute a side-agent forward pass using shared weights + landmark context.
        Returns (text, hidden_state_vector).
        """
        from transformers.cache_utils import DynamicCache

        tokenizer = self.engine.tokenizer
        model = self.engine.model
        device = self.engine.device

        think_prompt = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        landmarks = self.engine.synapse.get_landmarks()

        past_key_values = None
        if landmarks is not None:
            past_key_values = DynamicCache()
            setattr(past_key_values, "key_cache", [k for k, _ in landmarks])
            setattr(past_key_values, "value_cache", [v for _, v in landmarks])

        curr_input = think_prompt
        generated_tokens = []
        outputs = None

        stream_ctx = (
            torch.cuda.stream(self.engine.side_stream)
            if getattr(self.engine, "side_stream", None) is not None
            else nullcontext()
        )
        with stream_ctx:
            for _ in range(max_tokens):
                kwargs = {"input_ids": curr_input, "output_hidden_states": True}
                if past_key_values is not None:
                    seq_len = past_key_values.get_seq_length()
                    position_ids = torch.arange(
                        seq_len, seq_len + curr_input.shape[1], device=device
                    ).unsqueeze(0)
                    kwargs["past_key_values"] = past_key_values
                    kwargs["position_ids"] = position_ids

                outputs = model(**kwargs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                curr_input = next_token
                past_key_values = outputs.past_key_values
                generated_tokens.append(next_token.item())

        thought_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        thought_vector = None
        if outputs is not None:
            thought_vector = outputs.hidden_states[-1][:, -1, :].detach()

        return thought_text, thought_vector

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_role(self, trigger_text: str) -> AgentRole:
        """Auto-detect the best role for a task description."""
        text = trigger_text.lower()
        if any(kw in text for kw in ("search", "find", "research", "look up", "verify fact")):
            return AgentRole.RESEARCHER
        if any(kw in text for kw in ("review", "critique", "check logic", "evaluate")):
            return AgentRole.REVIEWER
        if any(kw in text for kw in ("code", "script", "implement", "write function")):
            return AgentRole.CODER
        if any(kw in text for kw in ("verify", "validate", "confirm", "test")):
            return AgentRole.VERIFIER
        if any(kw in text for kw in ("plan", "decompose", "architect", "break down")):
            return AgentRole.ARCHITECT
        return AgentRole.RESEARCHER  # default fallback

    def _partition_tasks(self, tasks: List[SubAgentTask]):
        """Split tasks into immediately-ready and blocked-by-dependencies."""
        task_ids = {t.id for t in tasks}
        ready, blocked = [], []
        for t in tasks:
            if not t.depends_on or not any(d in task_ids for d in t.depends_on):
                ready.append(t)
            else:
                blocked.append(t)
        return ready, blocked

    def _coordinate_team(self, plan: TeamPlan, blocked: List[SubAgentTask]):
        """Background coordinator: polls dependencies and unblocks tasks."""
        remaining = list(blocked)
        while remaining:
            newly_ready = []
            for t in remaining:
                deps_met = all(
                    self.tasks.get(d) and self.tasks[d].status in ("completed", "failed")
                    for d in t.depends_on
                )
                if deps_met:
                    newly_ready.append(t)
                    future = self.pool.submit(self._execute_task, t)
                    self.futures[t.id] = future

            for t in newly_ready:
                remaining.remove(t)

            if remaining:
                time.sleep(0.1)

        # Wait for all team tasks to finish
        for t in plan.tasks:
            f = self.futures.get(t.id)
            if f:
                f.result(timeout=120.0)

        # Aggregate results
        results = [t.result for t in plan.tasks if t.result and t.status == "completed"]
        plan.final_result = " | ".join(results) if results else "Team produced no results."
        plan.status = "done"
        print(f"[Orchestrator] Team '{plan.goal}' DONE: {plan.final_result[:120]}")

    def shutdown(self):
        self.pool.shutdown(wait=False)
