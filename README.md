# Warp Cortex

> **STATUS — architecture frozen, evaluation honest.** The world-runtime research line (`cortex_apps/cortex_world_runtime/`) converged to a negative result that is now the headline: a matched boring unified store ties Cortex on memory/retrieval (`D_Cortex ~= D_U0`) and matched Modular C ties it on reasoning (`Q` ties when information is matched) (`cortex_apps/cortex_world_runtime/test_boring_store_kill.py:14-122`, `boring_unified_store.py:12-247`). Cortex is therefore **not** claimed as a novel memory primitive, superior retrieval algorithm, intelligence amplifier, or compute shortcut. Its surviving value is a **standardized persistent world model**: shared `S,G,Z,H,K` with deterministic context services over one consistency domain. Everything below this banner is either the portable product path (boring, explicit) or older research narrative kept for context and marked as such.

## What this repo actually is

**One portable directory per project** that many apps/agents/skills can reuse without each maintaining a separate notion of "what is this project." That is the product. The rest is research instrumentation.

```
<project>/.cortex/                          # copy = move world (zip-safe)
├── cortex.sqlite    # CANONICAL S,G,Z,H,K  # machines read this (WAL, fsync, hash chain)
├── manifest.json    # format cortex-world-v1, encoder_id, budgets, history policy
├── entities/*.md    # human/app mirror     # NOT canonical, NOT lossless recovery
└── skills/<id>/SKILL.md                    # agentskills-v1 mirrors
```

Authority is explicit: sqlite is canonical; Markdown is a mirror (`cortex_apps/cortex_world_runtime/cortex_world/store.py:1-14`, `store.py:248-283`). Budgets in the manifest are **operational, not retrieval guarantees** — fixed-budget recall is known to degrade with world size (`retrieval_law_results.json`).

Stable public surface (deliberately tiny, `cortex_apps/cortex_world_runtime/cortex_world/`):

```bash
python -m cortex_apps.cortex_world_runtime.cortex_world.cli open <project_dir>
python -m cortex_apps.cortex_world_runtime.cortex_world.cli ingest <project_dir> <file>
python -m cortex_apps.cortex_world_runtime.cortex_world.cli recall <project_dir> <query...>
python -m cortex_apps.cortex_world_runtime.cortex_world.cli bfs <project_dir> <entity_id>
python -m cortex_apps.cortex_world_runtime.cortex_world.cli select-skill <project_dir> <query...>
python -m cortex_apps.cortex_world_runtime.cortex_world.cli record-invocation <project_dir> <skill> <ver> <0|1>
```

Internally: `S` = current state, `G` = typed structure (`depends_on, blocks, supports, refutes, mentions, derived_from` in `cortex_world/graph.py:8-12`), `Z` = semantic index, `H` = bounded provenance/history (hot cap + snapshots in `store.py:270-307`), `K` = versioned procedural skills. `recall()` returns `RecallResult{ hits=[RecallHit{entity, score, edge_path, event_seq, provenance}], snapshot_version, candidate_budget, candidates_examined }` (`recall.py:14-27`) so callers see degradation instead of thin context. `cascade_invalidate()` propagates through `depends_on/blocks/derived_from` only; `mentions` never invalidates (`graph.py:14-15`).

Skills are project-scoped: `SHARED_CORTEX_LEDGER` selection filters to one project world by default (`skill_registry.py:155-167`); cross-project learning is never default and must be explicitly allowed. `SKILL.md` remains the portable human/agent representation; sqlite holds the indexed runtime representation.

Research machinery — `sharded_world_substrate`, society sims, scorecard cost router, benchmark-only multi-agent harnesses — stays in the research tree and is **not** part of the portable path.

## Current evidence (frozen — supersedes older claims)

| Property | Result | Artifact |
|---|---|---|
| Reasoning vs matched baselines | **tie** (`Q` equal when information is matched) | `test_boring_store_kill.py` |
| Memory/retrieval vs boring `U_0` | **tie** (`D_Cortex ~= D_U0` measured: `u0/cortex=0.988`) | `boring_unified_store.py` |
| 4-store duplication overhead | **~+10% in-process lower bound** (`+9.5%` measured; old `+27.2%` retired as unsourced) | `test_boring_store_kill.py:99-122` |
| Small-world retrieval (2k entities, true query-only retrieval) | **10/20 = 10/20 = 10/20**, provenance recall **0.70 vs ≤0.20** | `unseen_synthesis_results.json` + `unseen_synthesis_suite.py:15-42` (GPU: `Qwen2.5-0.5B`) |
| Bounded-retrieval recall scaling | **fails**: `0.45@10k → 0.00@1M` at `B=400`; ranking dead for `N≥10k`; 16× budget only `0.05` at 1M | `retrieval_law_results.json` (`test_boring_store_kill.py:175-340`) |
| Storage / latency scaling | **good** (linear bytes, bounded ms/tokens) | same |
| Determinism | **fixed** (sorted traversal; cross-`PYTHONHASHSEED` parity tested) | `fast_world_substrate.py:88`, `sharded_world_substrate.py:60`, `test_boring_store_kill.py:44-78` |
| Dirty-world durability (P2, 1M mutations, 4 policies × 2 seeds) | **canonical state holds** (`~0` mismatch); unmaintained `SSR→1.0`, `dangling 13.3/edge`, `resurr 0.84`; `rebuild` zeroes structure; `incremental` least memory (742 MB); `checkpoint` caps log at 500 but sampled provenance `→0.00`; recall `≈0` at 1M all policies | `dirty_world_results.json`, `dirty_world.py:1-30` |

The old "decisive missing benchmark" framing — API-backed `api_single` vs `hybrid_repair` cost comparison — is retired with the claim that Cortex amplifies model quality. What matters now is the degradation law and the durability battery. Open directions in order: **P2 durability → P4 history bounds → P5 skill-at-scale → P7 service composability → P6 natural history → P8 multi-agent workloads**.

## What "council" actually means here

There are **two separate** multi-agent mechanisms, neither is a deliberating swarm that beats single-agent reasoning, and the filename is legacy.

**1. `AdaptiveGenerator` council — `cortex_core/adaptive_engine.py:53-56`, `cortex_core/adaptive_engine.py:407-516`:** a research mode (`DelegationMode.COUNCIL`, `council_size` at `adaptive_engine.py:230,244`) that spawns `N` threads (`ThreadPoolExecutor` at `adaptive_engine.py:461`) each doing an independent greedy decode with a different temperature (`0.3 + 0.2*i` at `adaptive_engine.py:417-419`), no shared context, no ensembling, no inter-agent communication. Final answer by `####` majority vote (`_council_vote` at `adaptive_engine.py:492-516`), fallback `max(..., key=len)`. Thread-unsafe on CUDA (HF `generate` + shared model object) and never integrated with `CortexEngine`. Not used in the official GSM8K benchmark — `cortex_benchmarks/benchmark_cortex_gsm8k.py:283-286` shims `council → orchestrated`.

**2. `CortexOrchestrator` team — `cortex_core/cortex_orchestrator.py:63-166`, `cortex_core/cortex_orchestrator.py:93,129,158`:** a real orchestrator with `TeamPlan`, typed `AgentRole`s (`RESEARCHER, REVIEWER, CODER, VERIFIER, ARCHITECT` at `cortex_orchestrator.py:26-32`), role prompts (`cortex_orchestrator.py:36-42`), `depends_on` dependencies, topological `ready/blocked` partition and `_coordinate_team` background thread, `ThreadPoolExecutor(max_workers=8)` and synapse/landmark sharing (`cortex_orchestrator.py:278-281`). This is what `CortexEngine` actually owns (`cortex_engine.py:556`, `cortex_engine.py:1302-1322`). On the current 20-task synthesis benchmark: `Q_single = Q_specialists = 50%` — adding specialists did **not** improve reasoning (`cortex_apps/cortex_world_runtime/unseen_synthesis_suite.py` head comment).

**3. `cortex_scripts/council_live.py` — legacy name.** File header at `council_live.py:18-20` states the default is now orchestrator-style sparse delegation (`[DELEGATE:...]` → `AsyncDelegationManager`) rather than always-on council voting. The benchmark in that file compares Single vs Orchestrated delegation, not council.

Measured policy: **one agent by default; multiple agents only when there is genuinely parallel/specialized work.**

## What else has which status

- **Entropy router / delegation gate — `cortex_core/entropy_router.py`, `cortex_core/delegation_gate.py`, `cortex_core/adaptive_engine.py:258-303`:** real code (per-layer attention entropy + logit entropy, Welford online, z-score thresholds; optional `LinearDelegationGate` on frozen hidden states). Predictive controller — can trigger before a draft finishes, but executable validation is the hard controller. Not a retrieval or reasoning amplifier.

- **Reaction manifold — `cortex_core/reaction_harness.py:1-60`:** a **CPU simulation** on the unit sphere `S^{D-1}` (radial Gaussian kernel `I_i(e)=max exp(-d^2/2σ²)·u`, Laplacian diffusion `h^{t+1}=(1-γ)h^t+αW@h^t`). "0 GPU FLOPs on dormant agents" is tautologically true because dormant agents are simply not executed; it is not a profiled GPU saving (`cortex_validation/test_reaction_manifold.py:97-113` asserts `is_triggered()` only). Keep the mechanism, drop the FLOPs framing.

- **Epistemic manifold — `cortex_core/epistemic_manifold.py`:** research prototype for signed constraints (`depends_on, supports, refutes, blocks`). Not part of the portable store.

- **Scorecard/cost router — `cortex_scorecard/`:** local-vs-API evaluation harness that compiles `policy.yaml`. Real code, separate from the world model; not evidence for Cortex-specific intelligence.

- **KV-cache compression — `cortex_core/turbo_quant.py`:** TurboQuant-style (Hadamard + 2/3/4-bit) as an efficiency layer, not a core novelty claim. Entry points and `cortex_benchmarks/` that mention it are real but optional.

## Installation

```bash
pip install -e .
pip install -e .[benchmarks,api]
```

Entry points (`setup.py:31-37`): `warp-cortex-live` (`cortex_scripts/council_live.py:main` — legacy name), `warp-cortex-gsm8k` (`cortex_benchmarks/benchmark_cortex_gsm8k:main`), `warp-cortex-manifold` (`cortex_benchmarks/benchmark_shared_manifold:main`), `warp-cortex-scorecard` (`cortex_scorecard/cli:main`).

Config (`config/settings.yaml`, `config/settings.local.yaml` override, `${ENV_VAR}` supported): see `cortex_core/settings.py`. User-facing entry points that read it: `cortex_scripts/council_live.py`, `cortex_benchmarks/benchmark_cortex_gsm8k.py`, `cortex_engine.py`.

## Project layout

- `cortex_core/` — runtime internals (`adaptive_engine`, `entropy_router`, `turbo_quant`, `async_delegate`, `cortex_router`; plus `synapse`, `agent_cloud`, `reaction_harness`, `epistemic_manifold` — see status above). [frozen except portable fixes]
- `cortex_apps/cortex_world_runtime/` — frozen world-model research: `fast_world_substrate.py` (production substrate, now deterministic), `dirty_world.py` (P2 battery), `unseen_synthesis_suite.py` (controlled retrieval benchmark), `boring_unified_store.py` + `test_boring_store_kill.py` (U_0 kills, retrieval law), `cortex_world/` (portable per-project product path). [frozen / product]
- `cortex_core/cortex_world_runtime/cortex_world/` — `store.py`, `recall.py`, `graph.py`, `skills.py`, `ingest.py`, `cli.py`, `__init__.py` — boring, tested (`test_cortex_world.py: 9/9`). [product]
- `cortex_scripts/`, `cortex_benchmarks/`, `cortex_validation/`, `cortex_resources/` — CLIs / benchmarks / regression checks / skill fixtures. [research]
- `cortex_engine.py` — legacy programmatic engine surface (synapse + router + orchestrator + compression). [legacy, not portable path]
- `docs/CORTEX_OS.md` — longer architecture notes (predates frozen thesis; read after this README).

`research/`, `paper/`, `local_artifacts/` are gitignored; `warp_cortex/.gitignore` also excludes `None/`, `.hf_cache/`, `*.pt/*.safetensors`, `.pytest_cache/`.

## Shared-manifold validation (historical slices, Qwen0.5B-CPU — not flagship)

Kept for context; not the current claim:

| Slice | Aggregate result |
|---|---|
| Real coding compare (3 tasks) | `pass_rate=0.67` vs `0.00`; `prompt_hit_rate=1.00` vs `0.00` |
| Targeted energy reuse (3 tasks) | off: `followup_target_hit_rate=0.00`, `distractor_capture_rate=1.00`; on: `1.00`/`0.00` |
| Coding handoff (3 tasks) | `context_match_rate=1.00`, `output_match_rate=1.00` |
| Recall handoff (5 tasks) | `fresh_answer_rate=0.00`; `loaded_answer_rate` `0.60-0.80` (brittle on small model) |
| Necessity ablation | `manifold_prompt_hit_rate=1.00` vs `0.00`, `necessity_win_rate=1.00` |
| Topology retrieval (2 tasks) | `topology_expected_recall_rate=1.00` vs `flat 0.75`, `topology_answer_rate=1.00` vs `0.00` |

See `cortex_benchmarks/benchmark_shared_manifold.py --mode topology-compare` etc. These are mechanism demos, not scale results.

## What not to claim

No faster end-to-end agent latency on the current LLM-bound benchmark (`T_Cortex = T_C ≈ 1510 ms`); no novel retrieval algorithm (cleanly killed by `U_0`); no inherent multi-agent intelligence gain; no profiled FLOPs saving. The thesis is:

> **same reasoning, less duplicated contextual state, fewer cross-service joins, one consistency domain — and bounded retrieval that currently does not scale in recall.**

## Entry points (quick)

```bash
# portable world
python -m cortex_apps.cortex_world_runtime.cortex_world.cli open /path/to/project
python -m cortex_apps.cortex_world_runtime.cortex_world.cli recall /path/to/project "query"

# legacy engine / benchmarks
python cortex_scripts/council_live.py "How many r's are in strawberry?"
python cortex_benchmarks/benchmark_cortex_gsm8k.py --n 20 --modes single,orchestrated
python cortex_validation/tests.py
python -m pytest cortex_apps/cortex_world_runtime/test_cortex_world.py cortex_apps/cortex_world_runtime/test_boring_store_kill.py
```

## Docs

`docs/CORTEX_OS.md` elaborates the manifold view. Read the frozen thesis (top of this file) first; the OS doc predates it.
