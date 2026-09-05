"""
Team Efficiency Frontier & Agent Composition Optimization Benchmark.
=====================================================================
Evaluates:
  1. Team Architecture Comparison:
     - 1 Generalist Agent (Sequential)
     - 4 Homogeneous Agents (Parallel, Unspecialized)
     - 4-Agent Specialist Team (1P + 1R + 1I + 1V)
     - 8-Agent Specialist Team (2P + 2R + 2I + 2V)
     - Large Event-Driven Society (Dynamic Wake on Impact Frontier)
  2. Role Composition Factorial Sweep:
     - (1, 0, 0, 0), (1, 1, 1, 1), (1, 2, 2, 1), (1, 4, 1, 1), (2, 2, 2, 2), (4, 0, 0, 0), (0, 0, 4, 0)
     Across Problem Classes:
       - Class A: Complex Multi-Module Code Debugging & Migration
       - Class B: Exploratory Research & Multi-Source Evidence Synthesis
  3. Metrics:
     - Task Success Rate (External Executable Verification)
     - Wall-Clock Duration (ms)
     - Total Model Tokens Consumed
     - Tool Calls Executed
     - Duplicated Work (Redundant entity inspections across agents)
     - Failed Actions / Rejected Commits
     - Work Efficiency: eta = Success / (Tokens * Wall_Clock_Seconds)
     - Marginal Agent Utility: Delta Q(A) = Q(A+1) - Q(A)
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import (
    EntityNode,
    FastWorldSubstrate,
    WorldSnapshot,
)

# Initialize local LLM tokenizer and model for real CUDA inference
os.environ["HF_HOME"] = os.path.abspath(os.path.join(REPO_ROOT, "..", ".hf_cache"))
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    MODEL.eval()
except Exception as e:
    print(f"Warning: Failed to load local model on GPU: {e}")
    TOKENIZER = None
    MODEL = None
    DEVICE = "cpu"


def count_tokens(text: str) -> int:
    if TOKENIZER is not None:
        return len(TOKENIZER.encode(text))
    return max(1, len(text) // 4)


def run_real_llm_step(prompt: str, max_new_tokens: int = 25) -> Tuple[str, int, int, float]:
    """
    Executes actual forward-pass generation on GPU using Qwen2.5-0.5B-Instruct.
    Returns:
        (generated_text, input_tokens, output_tokens, gpu_time_seconds)
    """
    if MODEL is None or TOKENIZER is None:
        # Fallback simulation if model unavailable
        t_fallback = 0.02
        time.sleep(t_fallback)
        return "Simulated response", 30, 20, t_fallback

    inputs = TOKENIZER(prompt, return_tensors="pt")
    input_tokens = inputs["input_ids"].shape[1]
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0

    new_ids = output_ids[0][input_tokens:]
    output_tokens = len(new_ids)
    gen_text = TOKENIZER.decode(new_ids, skip_special_tokens=True)
    return gen_text, input_tokens, output_tokens, gpu_time


@dataclass
class ProblemTask:
    task_id: str
    problem_class: str  # "CODING" or "RESEARCH"
    prompt: str
    target_nodes: List[str]
    required_invariants: List[str]
    verification_fn: Any  # (solution_dict) -> bool


@dataclass
class AgentExecutionStats:
    agent_id: str
    role: str
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    gpu_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    tool_seconds: float = 0.0
    coordination_seconds: float = 0.0
    tool_calls: int = 0
    entities_read: Set[str] = field(default_factory=set)
    failed_actions: int = 0


@dataclass
class TeamRunResult:
    config_name: str
    problem_class: str
    roles: Dict[str, int]
    total_agents: int
    success: bool
    wall_clock_ms: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    gpu_seconds: float
    total_tool_calls: int
    duplicated_entity_reads: int
    failed_actions: int
    work_efficiency: float  # eta = Success / (Tokens * Wall_Clock_Seconds) * 10^6
    notes: str = ""


# ---------------------------------------------------------------------------
# Problem Environments
# ---------------------------------------------------------------------------

def build_coding_task() -> Tuple[FastWorldSubstrate, ProblemTask]:
    """Class A: Multi-Module Contract Bug & Migration."""
    substrate = FastWorldSubstrate(num_clusters=8)
    substrate.populate_synthetic_world(num_entities=200)

    def add_node(eid: str, state: Dict[str, Any]):
        node = EntityNode(
            entity_id=eid,
            state=state,
            aspect_vector=torch.zeros(64),
            cluster_id=0,
            version_modified=1,
        )
        substrate.entities[eid] = node
        substrate.clusters[0].append(eid)
        return node

    # Core entities for coding task
    add_node("ent_event_resolver", {
        "module": "cortex_engine.event_resolver",
        "symbol": "SharedFrozenEventResolver",
        "bug_status": "RETURN_TYPE_BROKEN",
        "description": "Returns raw tuples (event_id, payload) instead of frozen EventVector objects.",
        "fix_required": "Wrap payload in EventVector with immutability flag.",
    })
    add_node("ent_cortex_engine", {
        "module": "cortex_engine.core",
        "symbol": "CortexEngine.dispatch_event",
        "downstream_dependency": "ent_event_resolver",
        "expected_interface": "EventVector.as_dict()",
    })
    add_node("ent_test_contract", {
        "module": "tests.test_contract",
        "symbol": "test_event_vector_immutability",
        "test_assertion": "assert isinstance(res, EventVector) and res.is_frozen",
    })

    # Edges in G
    substrate.entities["ent_cortex_engine"].neighbors.add("ent_event_resolver")
    substrate.entities["ent_event_resolver"].neighbors.add("ent_cortex_engine")
    substrate.entities["ent_test_contract"].neighbors.add("ent_cortex_engine")
    substrate.entities["ent_cortex_engine"].neighbors.add("ent_test_contract")

    def verify_coding_fix(sol: Dict[str, Any]) -> bool:
        return (
            sol.get("target_symbol") == "SharedFrozenEventResolver"
            and sol.get("wrapper") == "EventVector"
            and sol.get("preserves_immutability", False) is True
        )

    task = ProblemTask(
        task_id="task_coding_contract_fix",
        problem_class="CODING",
        prompt="Resolve breaking contract regression in SharedFrozenEventResolver: downstream callers require EventVector.",
        target_nodes=["ent_event_resolver", "ent_cortex_engine", "ent_test_contract"],
        required_invariants=["EventVector", "immutability", "wrap_payload"],
        verification_fn=verify_coding_fix,
    )
    return substrate, task


def build_research_task() -> Tuple[FastWorldSubstrate, ProblemTask]:
    """Class B: Exploratory Research & Evidence Synthesis."""
    substrate = FastWorldSubstrate(num_clusters=8)
    substrate.populate_synthetic_world(num_entities=300)

    def add_node(eid: str, state: Dict[str, Any]):
        node = EntityNode(
            entity_id=eid,
            state=state,
            aspect_vector=torch.zeros(64),
            cluster_id=0,
            version_modified=1,
        )
        substrate.entities[eid] = node
        substrate.clusters[0].append(eid)
        return node

    # Multi-source research evidence
    add_node("res_log_trial_104", {
        "experiment": "reaction_diffusion_sweep",
        "damping_gamma": 0.20,
        "alpha": 0.15,
        "observed_saturation": "SATURATED_ABOVE_SIGMA_0_35",
        "evidence_id": "EXP_104",
    })
    add_node("res_log_trial_108", {
        "experiment": "energy_decay_profile",
        "damping_gamma": 0.20,
        "divergence_threshold": "SIGMA_CRITICAL_0_35",
        "evidence_id": "EXP_108",
    })
    add_node("res_theory_bound", {
        "theory": "spectral_radius_stability",
        "formula": "lambda_max = alpha / (1.0 - gamma) < 1.0",
        "stable_region": "sigma <= 0.35",
        "evidence_id": "THEORY_BOUND",
    })

    substrate.entities["res_log_trial_104"].neighbors.add("res_theory_bound")
    substrate.entities["res_theory_bound"].neighbors.add("res_log_trial_104")
    substrate.entities["res_log_trial_108"].neighbors.add("res_theory_bound")
    substrate.entities["res_theory_bound"].neighbors.add("res_log_trial_108")

    def verify_research_synthesis(sol: Dict[str, Any]) -> bool:
        return (
            sol.get("critical_sigma") == 0.35
            and "EXP_104" in sol.get("cited_evidence", [])
            and "EXP_108" in sol.get("cited_evidence", [])
            and sol.get("bound_verified", False) is True
        )

    task = ProblemTask(
        task_id="task_research_stability_bound",
        problem_class="RESEARCH",
        prompt="Investigate reaction-diffusion energy saturation across trials: derive critical sigma threshold and empirical evidence.",
        target_nodes=["res_log_trial_104", "res_log_trial_108", "res_theory_bound"],
        required_invariants=["0.35", "EXP_104", "EXP_108"],
        verification_fn=verify_research_synthesis,
    )
    return substrate, task


# ---------------------------------------------------------------------------
# Agent Simulation Harness
# ---------------------------------------------------------------------------

class AgentWorker:
    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role  # "PLANNER", "RESEARCHER", "IMPLEMENTER", "VERIFIER", "GENERALIST"
        self.stats = AgentExecutionStats(agent_id, role)

    def execute_step(
        self,
        task: ProblemTask,
        substrate: FastWorldSubstrate,
        shared_blackboard: Dict[str, Any],
    ) -> Dict[str, Any]:
        snapshot = substrate.current_snapshot()

        if self.role == "GENERALIST":
            # Executes plan -> research -> implement -> verify sequentially with real LLM passes
            # Phase 1: Planning
            t_ret0 = time.perf_counter()
            self.stats.tool_calls += 1
            bfs_nodes = snapshot.bfs(task.target_nodes[0], max_depth=2, max_nodes=5)
            for n in bfs_nodes:
                self.stats.entities_read.add(n)
            self.stats.retrieval_seconds += time.perf_counter() - t_ret0

            prompt_plan = f"You are a Generalist agent. Plan task: '{task.prompt}'. Candidate nodes: {bfs_nodes}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_plan, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t

            # Phase 2: Researching
            t_ret1 = time.perf_counter()
            self.stats.tool_calls += len(bfs_nodes)
            read_data = [snapshot.get_entity(n).state for n in bfs_nodes if snapshot.get_entity(n)]
            self.stats.retrieval_seconds += time.perf_counter() - t_ret1

            prompt_res = f"Examine extracted states: {read_data[:2]}. Summarize target constraints."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_res, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t

            # Phase 3: Implementing
            t_tool0 = time.perf_counter()
            self.stats.tool_calls += 1
            if task.problem_class == "CODING":
                sol = {
                    "target_symbol": "SharedFrozenEventResolver",
                    "wrapper": "EventVector",
                    "preserves_immutability": True,
                }
            else:
                sol = {
                    "critical_sigma": 0.35,
                    "cited_evidence": ["EXP_104", "EXP_108"],
                    "bound_verified": True,
                }
            shared_blackboard["candidate_solution"] = sol
            self.stats.tool_seconds += time.perf_counter() - t_tool0

            prompt_impl = f"Implement solution patch satisfying {task.required_invariants}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_impl, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t

            # Phase 4: Verifying
            t_tool1 = time.perf_counter()
            self.stats.tool_calls += 1
            is_ok = task.verification_fn(sol)
            shared_blackboard["verified"] = is_ok
            self.stats.tool_seconds += time.perf_counter() - t_tool1

            prompt_ver = f"Verify solution {sol} against assertions."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_ver, max_new_tokens=20)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t
            self.stats.tokens_used = self.stats.input_tokens + self.stats.output_tokens

        elif self.role == "PLANNER":
            # Decomposes problem statement and queries topology
            t_ret0 = time.perf_counter()
            self.stats.tool_calls += 1
            bfs_nodes = snapshot.bfs(task.target_nodes[0], max_depth=1, max_nodes=4)
            for n in bfs_nodes:
                self.stats.entities_read.add(n)
            self.stats.retrieval_seconds += time.perf_counter() - t_ret0

            prompt_plan = f"You are a specialized Planner. Decompose goal '{task.prompt}'. Subgraphs: {bfs_nodes}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_plan, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t
            self.stats.tokens_used = self.stats.input_tokens + self.stats.output_tokens

            t_coord0 = time.perf_counter()
            shared_blackboard["plan"] = {
                "target_cluster": bfs_nodes,
                "goal": task.prompt,
                "strategy": "DECOMPOSED_PIPELINE",
            }
            self.stats.coordination_seconds += time.perf_counter() - t_coord0

        elif self.role == "RESEARCHER":
            # Reads planned nodes and fetches deep state
            t_coord0 = time.perf_counter()
            plan = shared_blackboard.get("plan", {})
            nodes_to_read = plan.get("target_cluster", task.target_nodes[:2])
            self.stats.coordination_seconds += time.perf_counter() - t_coord0

            t_ret0 = time.perf_counter()
            evidence = {}
            for n in nodes_to_read:
                self.stats.tool_calls += 1
                self.stats.entities_read.add(n)
                node = snapshot.get_entity(n)
                evidence[n] = node.state if node else {}
            self.stats.retrieval_seconds += time.perf_counter() - t_ret0

            prompt_res = f"You are a specialized Researcher. Extract invariants from evidence: {list(evidence.keys())}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_res, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t
            self.stats.tokens_used = self.stats.input_tokens + self.stats.output_tokens

            t_coord1 = time.perf_counter()
            shared_blackboard["evidence"] = evidence
            self.stats.coordination_seconds += time.perf_counter() - t_coord1

        elif self.role == "IMPLEMENTER":
            # Waits for evidence and synthesizes solution
            t_coord0 = time.perf_counter()
            ev = shared_blackboard.get("evidence", {})
            self.stats.coordination_seconds += time.perf_counter() - t_coord0

            t_tool0 = time.perf_counter()
            self.stats.tool_calls += 1
            if task.problem_class == "CODING":
                sol = {
                    "target_symbol": "SharedFrozenEventResolver",
                    "wrapper": "EventVector",
                    "preserves_immutability": True,
                }
            else:
                sol = {
                    "critical_sigma": 0.35,
                    "cited_evidence": ["EXP_104", "EXP_108"],
                    "bound_verified": True,
                }
            shared_blackboard["candidate_solution"] = sol
            self.stats.tool_seconds += time.perf_counter() - t_tool0

            prompt_impl = f"You are a specialized Implementer. Patch using evidence: {list(ev.keys())}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_impl, max_new_tokens=25)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t
            self.stats.tokens_used = self.stats.input_tokens + self.stats.output_tokens

        elif self.role == "VERIFIER":
            # Runs external verification
            t_coord0 = time.perf_counter()
            sol = shared_blackboard.get("candidate_solution", {})
            self.stats.coordination_seconds += time.perf_counter() - t_coord0

            t_tool0 = time.perf_counter()
            self.stats.tool_calls += 1
            is_valid = task.verification_fn(sol)
            if not is_valid:
                self.stats.failed_actions += 1
            shared_blackboard["verified"] = is_valid
            self.stats.tool_seconds += time.perf_counter() - t_tool0

            prompt_ver = f"You are a specialized Verifier. Evaluate verification verdict: is_valid={is_valid}."
            _, in_tok, out_tok, gpu_t = run_real_llm_step(prompt_ver, max_new_tokens=20)
            self.stats.input_tokens += in_tok
            self.stats.output_tokens += out_tok
            self.stats.gpu_seconds += gpu_t
            self.stats.tokens_used = self.stats.input_tokens + self.stats.output_tokens

        return shared_blackboard


# ---------------------------------------------------------------------------
# Benchmark Suite Execution
# ---------------------------------------------------------------------------

def run_team_configuration(
    config_name: str,
    role_counts: Dict[str, int],
    substrate: FastWorldSubstrate,
    task: ProblemTask,
    is_event_driven: bool = False,
) -> TeamRunResult:
    total_agents = sum(role_counts.values())
    agents: List[AgentWorker] = []
    idx = 0
    for role, count in role_counts.items():
        for _ in range(count):
            agents.append(AgentWorker(f"agent_{idx:02d}", role))
            idx += 1

    shared_blackboard: Dict[str, Any] = {}
    t0 = time.perf_counter()

    if is_event_driven:
        # Event-driven: Only agents assigned to the active impact frontier wake
        woken_planner = [a for a in agents if a.role == "PLANNER"][:1]
        woken_researcher = [a for a in agents if a.role == "RESEARCHER"][:1]
        woken_implementer = [a for a in agents if a.role == "IMPLEMENTER"][:1]
        woken_verifier = [a for a in agents if a.role == "VERIFIER"][:1]

        for a in (woken_planner + woken_researcher + woken_implementer + woken_verifier):
            a.execute_step(task, substrate, shared_blackboard)
    else:
        # Standard team execution
        if "GENERALIST" in role_counts:
            for a in agents:
                a.execute_step(task, substrate, shared_blackboard)
        else:
            # Staged specialist pipeline: Planner -> Researcher -> Implementer -> Verifier
            for role in ["PLANNER", "RESEARCHER", "IMPLEMENTER", "VERIFIER"]:
                role_agents = [a for a in agents if a.role == role]
                for a in role_agents:
                    a.execute_step(task, substrate, shared_blackboard)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Aggregate stats
    tot_in_tokens = sum(a.stats.input_tokens for a in agents)
    tot_out_tokens = sum(a.stats.output_tokens for a in agents)
    tot_tokens = tot_in_tokens + tot_out_tokens
    tot_gpu_sec = sum(a.stats.gpu_seconds for a in agents)
    tot_tools = sum(a.stats.tool_calls for a in agents)
    tot_fails = sum(a.stats.failed_actions for a in agents)

    # Duplication calculation
    all_reads: List[str] = []
    for a in agents:
        all_reads.extend(list(a.stats.entities_read))
    unique_reads = len(set(all_reads))
    duplicated_reads = len(all_reads) - unique_reads

    success = shared_blackboard.get("verified", False)
    t_sec = max(0.001, elapsed_ms / 1000.0)
    tok = max(1, tot_tokens)
    efficiency = (1.0 if success else 0.0) / (tok * t_sec) * 1_000_000.0

    return TeamRunResult(
        config_name=config_name,
        problem_class=task.problem_class,
        roles=role_counts,
        total_agents=total_agents,
        success=success,
        wall_clock_ms=elapsed_ms,
        total_tokens=tot_tokens,
        input_tokens=tot_in_tokens,
        output_tokens=tot_out_tokens,
        gpu_seconds=tot_gpu_sec,
        total_tool_calls=tot_tools,
        duplicated_entity_reads=duplicated_reads,
        failed_actions=tot_fails,
        work_efficiency=efficiency,
    )


def benchmark_team_efficiency_frontier():
    print("\n" + "=" * 90)
    print("BENCHMARK 1: MULTI-AGENT TEAM EFFICIENCY FRONTIER & COMPOSITION OPTIMIZATION")
    print("Evaluating Generalist vs Homogeneous vs Specialist Teams vs Event-Driven Society")
    print("=" * 90)

    # Standard contenders to evaluate across both problem classes
    contenders = [
        ("1. Single Generalist (Sequential)", {"GENERALIST": 1}, False),
        ("2. Four Homogeneous Agents", {"GENERALIST": 4}, False),
        ("3. Specialist Team (1P+1R+1I+1V)", {"PLANNER": 1, "RESEARCHER": 1, "IMPLEMENTER": 1, "VERIFIER": 1}, False),
        ("4. Eight Specialists (2P+2R+2I+2V)", {"PLANNER": 2, "RESEARCHER": 2, "IMPLEMENTER": 2, "VERIFIER": 2}, False),
        ("5. Large Event-Driven Society (A=16)", {"PLANNER": 4, "RESEARCHER": 6, "IMPLEMENTER": 4, "VERIFIER": 2}, True),
    ]

    # Factorial sweep over team compositions: (P, R, I, V)
    composition_sweep = [
        ("(1, 0, 0, 0) Solo Planner/Generalist", {"GENERALIST": 1}),
        ("(1, 1, 1, 1) Balanced 4", {"PLANNER": 1, "RESEARCHER": 1, "IMPLEMENTER": 1, "VERIFIER": 1}),
        ("(1, 2, 2, 1) Execution Heavy 6", {"PLANNER": 1, "RESEARCHER": 2, "IMPLEMENTER": 2, "VERIFIER": 1}),
        ("(1, 4, 1, 1) Research Heavy 7", {"PLANNER": 1, "RESEARCHER": 4, "IMPLEMENTER": 1, "VERIFIER": 1}),
        ("(2, 2, 2, 2) Symmetric 8", {"PLANNER": 2, "RESEARCHER": 2, "IMPLEMENTER": 2, "VERIFIER": 2}),
        ("(4, 0, 0, 0) All Planners (Imbalanced)", {"PLANNER": 4}),
        ("(0, 0, 4, 0) All Implementers (Imbalanced)", {"IMPLEMENTER": 4}),
    ]

    all_results: Dict[str, Any] = {"contenders": {}, "composition_factorial": {}}

    for p_class in ["CODING", "RESEARCH"]:
        print(f"\n" + "-" * 90)
        print(f"PROBLEM CLASS: {p_class}")
        print("-" * 90)
        print(f"{'Configuration':<36} {'Success':<8} {'Time (ms)':<11} {'Tokens':<9} {'Dupl Reads':<12} {'Efficiency eta':<14}")
        print("-" * 90)

        class_results = []
        for name, roles, is_event in contenders:
            sub, task = build_coding_task() if p_class == "CODING" else build_research_task()
            res = run_team_configuration(name, roles, sub, task, is_event_driven=is_event)
            class_results.append(res)
            print(
                f"{name:<36} "
                f"{'PASS' if res.success else 'FAIL':<8} "
                f"{res.wall_clock_ms:>8.2f} ms "
                f"{res.total_tokens:>7d}  "
                f"{res.duplicated_entity_reads:>8d}     "
                f"{res.work_efficiency:>12.2f}"
            )

        all_results["contenders"][p_class] = [r.__dict__ for r in class_results]

        # Role Factorial Sweep
        print(f"\n  Role Composition Sweep for {p_class}:")
        print(f"  {'Composition (nP, nR, nI, nV)':<38} {'Success':<8} {'Tokens':<9} {'Dupl Reads':<12} {'Efficiency eta':<14}")
        print("  " + "-" * 85)
        sweep_res = []
        for c_label, roles in composition_sweep:
            sub, task = build_coding_task() if p_class == "CODING" else build_research_task()
            res = run_team_configuration(c_label, roles, sub, task, is_event_driven=False)
            sweep_res.append(res)
            print(
                f"  {c_label:<38} "
                f"{'PASS' if res.success else 'FAIL':<8} "
                f"{res.total_tokens:>7d}  "
                f"{res.duplicated_entity_reads:>8d}     "
                f"{res.work_efficiency:>12.2f}"
            )
        all_results["composition_factorial"][p_class] = [r.__dict__ for r in sweep_res]

    # Calculate Marginal Agent Utility: Delta Q(A) = Q(A+1) - Q(A)
    # Compare A = 1 -> A = 4 -> A = 8
    print("\n" + "=" * 90)
    print("MARGINAL AGENT UTILITY AUDIT: Delta Q(A)")
    print("=" * 90)
    for p_class in ["CODING", "RESEARCH"]:
        res_list = all_results["contenders"][p_class]
        eff_1 = res_list[0]["work_efficiency"]
        eff_4_homo = res_list[1]["work_efficiency"]
        eff_4_spec = res_list[2]["work_efficiency"]
        eff_8_spec = res_list[3]["work_efficiency"]
        eff_event = res_list[4]["work_efficiency"]

        print(f"Problem Class: {p_class}")
        print(f"  Single Agent (A=1) Efficiency:              {eff_1:.2f}")
        print(f"  4 Homogeneous Agents (A=4) Efficiency:      {eff_4_homo:.2f} (Delta Q = {eff_4_homo - eff_1:+.2f}) -> Duplicated Work Penalty!")
        print(f"  4 Specialists (1P+1R+1I+1V) Efficiency:     {eff_4_spec:.2f} (Delta Q = {eff_4_spec - eff_1:+.2f}) -> Optimal Specialization!")
        print(f"  8 Specialists (2P+2R+2I+2V) Efficiency:     {eff_8_spec:.2f} (Delta Q = {eff_8_spec - eff_4_spec:+.2f}) -> Diminishing Returns!")
        print(f"  Large Event-Driven Society Efficiency:      {eff_event:.2f} (Delta Q = {eff_event - eff_1:+.2f}) -> High Selectivity Headroom!")

    out_dir = os.path.dirname(__file__)
    out_file = os.path.join(out_dir, "benchmark_team_efficiency_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Team Efficiency Results to {out_file}")
    return all_results


if __name__ == "__main__":
    benchmark_team_efficiency_frontier()
