# Warp Cortex: Continuous Semantic Manifold & Inference Control Plane

> **STATUS (architecture frozen).** The world-runtime research line
> (`cortex_apps/cortex_world_runtime/`) converged to a negative result that
> is now the headline: a matched boring unified store ties Cortex on
> memory/retrieval (`D_Cortex ~= D_U0`), and matched Modular C ties it on
> reasoning (`Q` ties when information is matched). Cortex is therefore
> **not** claimed as a novel memory primitive, superior retrieval algorithm,
> intelligence amplifier, or compute shortcut. Its surviving value is a
> **standardized persistent world model**: shared state, structure,
> semantics, provenance, and procedural experience with deterministic
> context services over one consistency domain. Small-world retrieval works
> (10/20 vs 7/20 baselines, 0.70 vs <=0.20 provenance recall); bounded
> retrieval recall collapses with scale (0.45@10k → 0.00@1M at fixed
> budget). Open next: dirty-world durability (P2), history bounds (P4).
> Everything below this banner is the older narrative and is kept for
> context, not as current claims.

Warp Cortex is a **continuous topological cognition engine and inference control plane**. 

Instead of treating AI interactions as static text prompts passed to disconnected models, Cortex models systems as an **elastic geometric manifold** where knowledge, hypotheses, character psychology, and user actions interact as continuous dynamical fields.

It operates on two complementary levels:
1. **The Continuous Semantic & Epistemic Manifold (`cortex_core/`)**:
   * **Reaction Manifold (`reaction_harness.py`)**: A reaction-based dynamical field for game worlds and interactive NPCs. Player actions inject localized energy impulses; normalized graph Laplacian heat diffusion awakens only relevant characters, spending **0 GPU FLOPs** on dormant agents.
   * **Epistemic Manifold (`epistemic_manifold.py`)**: A causal research project manifold. Hypotheses, axioms, and empirical assays are bound by signed topological constraints (`depends_on`, `supports`, `refutes`, `blocks`). Disproving a keystone claim automatically collapses dependent research branches, while contradiction tension surfaces the active research frontier.
2. **The Inference Reliability & Cost Control Plane (`cortex_scorecard/`)**:
   * Evaluates local open-source models vs. frontier APIs, validates outputs with executable checks, and compiles reproducible routing policies (`policy.yaml`) that cut enterprise cloud bills by up to 85% with verified accuracy.

The full architectural blueprint is documented in [docs/CORTEX_OS.md](docs/CORTEX_OS.md).

The practical goal is not to beat a frontier API on every single task. The goal is to recover a meaningful share of frontier coding quality while making remote calls infrequent, measurable, and optional.

## Escalation Policy

Warp Cortex should be understood as a hybrid controller, not as an entropy demo with product language wrapped around it.

1. Executable validation is the hard controller: if a draft fails visible checks, the system has evidence that escalation is justified.
2. Entropy signals and hidden-state gates are the predictive controller: they can ask for extra reasoning earlier, before a weak draft finishes or before a failure is fully visible.
3. Explicit delegation remains available as a compatibility path when the model is prompted to request help directly.

The strongest product story is therefore not "entropy replaces testing." It is: entropy predicts trouble, validation confirms it, and remote repair is spent only where the evidence says it matters.

## Current Evidence

Two things are already established.

1. The Warp Cortex runtime no longer degrades the underlying local model when delegation is disabled.
2. The decisive missing benchmark is the API-backed comparison between `api_single` and `hybrid_repair`, measured by pass rate, remote call rate, and estimated cost.

The measurement harness for that table already exists under `research/coding_hybrid_eval/`, and the local parity check between `local_hf` and `single` is already in place. What is still missing is the credentialed API run itself, not the benchmark machinery.

That missing table is the one that turns the architecture into a product claim.

| Mode | Pass Rate | Remote Call Rate | Estimated Cost per 100 Tasks |
|---|---|---|---|
| `api_single` | ? | 1.00 | $X |
| `hybrid_repair` | ? | < 1.00 | $Y |

If `hybrid_repair` preserves most of `api_single`'s quality while making meaningfully fewer remote calls, Warp Cortex has a real commercial story rather than just an interesting local runtime.

## Installation

For local development:

```bash
pip install -e .
pip install -e .[benchmarks,api]
```

Available console entry points include `warp-cortex-live`, `warp-cortex-gsm8k`,
`warp-cortex-manifold`, and `warp-cortex-scorecard`.

Run the first Cortex OS scorecard surface with a deterministic smoke candidate:

```bash
warp-cortex-scorecard run --candidate deterministic --out-dir local_artifacts/scorecards/smoke
```

Run local/API/hybrid candidates when credentials and cached models are available:

```bash
warp-cortex-scorecard run \
    --candidate local \
    --candidate api \
    --candidate hybrid \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --api-model gpt-4o-mini \
    --hf-home D:\hf_cache\huggingface-home \
    --out-dir local_artifacts/scorecards/local_api
```

Each run writes `scorecard.json`, `scorecard.md`, `failures.jsonl`,
`policy.yaml`, `manifest.json`, and `evidence.sqlite`.

## Adaptive Generation Engine

The `AdaptiveGenerator` provides token-by-token generation with three delegation modes:

### System A — AWARE (default)

The model is told about delegation syntax in its system prompt. It chooses when and what to delegate by emitting structured blocks:

```text
[DELEGATE:math] 17 * 23 [/DELEGATE]
[DELEGATE:code] print(sum(range(10))) [/DELEGATE]
[DELEGATE:llm] Focused sub-question [/DELEGATE]
[DELEGATE:search] capital of Mongolia [/DELEGATE]
[DELEGATE:custom:expert_name] instructions | payload [/DELEGATE]
```

Worker results are fed back into the next model turn. There is no fixed council fan-out and no majority vote.

### System B — SILENT

The model has no idea delegation exists. The `EntropyRouter` monitors attention entropy at every token. When the model's internal uncertainty spikes beyond its running baseline (adaptive z-score thresholds), the engine:

1. Pauses generation
2. Auto-classifies the needed expert from the hidden state
3. Dispatches to the same worker pool
4. Injects the result back into the KV cache

The model never knows delegation happened. This uncertainty-triggered path is the core novel routing mode in Warp Cortex.

An optional learned gate can sit on top of the frozen last-layer hidden state as a scalar projection `nn.Linear(d_model, 1)`. In code, the gate is trained online on detached hidden states and can veto entropy spikes without mutating the backbone weights or invalidating KV caches.

In the product framing, this silent path is best understood as an early-warning signal. It can request extra reasoning before a bad draft finishes, while executable validation remains the hard correctness check.

### System C — COUNCIL (opt-in)

N independent agents (same model, different temperatures) answer the same question in parallel. Final answer selected by majority vote on `####` lines. No ensembling, no shared context — each agent reasons independently.

```python
engine = AdaptiveGenerator(
    model, tokenizer,
    mode=DelegationMode.COUNCIL,
    council_size=5,               # 5 agents
)
```

### Per-Expert Backends

Each worker kind can use a different backend / model. The `BitNetBackend` adapter wraps `warp_bitnet`'s `BitNetGenerator` (string-based API) to match the chat-style interface:

```python
from cortex_core.async_delegate import BitNetBackend
from warp_bitnet.research.generate import BitNetGenerator

bitnet = BitNetGenerator.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

engine = AdaptiveGenerator(
    model, tokenizer,
    mode=DelegationMode.AWARE,
    expert_backends={
        "math": BitNetBackend(bitnet),   # 1.58-bit for math workers
        "llm": hf_backend,               # FP16 for general sub-thinking
    },
)
```

Any expert kind not in `expert_backends` falls back to the default `llm_backend`.

### Usage

```python
from cortex_core.adaptive_engine import AdaptiveGenerator, DelegationMode

engine = AdaptiveGenerator(
    model, tokenizer,
    mode=DelegationMode.AWARE,    # or SILENT, COUNCIL
    turbo_quant_bits=4,           # KV cache compression: 2, 3, or 4
    turbo_quant_enabled=True,     # disable with False
    use_learned_gate=True,        # optional hidden-state gate for SILENT mode
    council_size=3,               # agents for COUNCIL mode
)
result = engine.generate("What is the area of a 5x7 rectangle?")
```

## Central Settings

Runtime defaults now live in `config/settings.yaml`.

- Put committed, shared defaults in `config/settings.yaml`
- Put machine-specific overrides or real secrets in `config/settings.local.yaml`
- Use `${ENV_VAR}` in YAML when you want to pull from the environment instead of hardcoding

Example:

```yaml
backends:
    default: api
    api:
        base_url: https://api.openai.com/v1
        model: gpt-4o-mini
        key: ${OPENAI_API_KEY}

adaptive_engine:
    mode: council
    council_size: 5
    turbo_quant:
        bits: 3
    learned_gate:
        enabled: true
        threshold: 0.55
        warmup_steps: 64
```

The main user-facing entrypoints now read from this config layer:

- `cortex_scripts/council_live.py`
- `cortex_benchmarks/benchmark_cortex_gsm8k.py`
- `cortex_engine.py`

## Project Layout

Warp Cortex is organized into canonical folders instead of treating every top-level script as primary:

- `cortex_core/` — core runtime modules and reusable internals
- `cortex_scripts/` — interactive / CLI runners
- `cortex_benchmarks/` — official evaluation benchmarks
- `cortex_validation/` — repo-only tests and regression checks
- `cortex_resources/` — runtime data such as persistent agent skill definitions

The repo root is now intentionally thin. The canonical implementation lives in the folders above, and only `cortex_engine.py` remains at the top level as the public programmatic engine surface.

Local-only folders such as `research/`, `paper/`, and `local_artifacts/` are intentionally gitignored so the public repo stays concise.

## Integrated KV Cache Compression

Warp Cortex can integrate TurboQuant-style KV cache compression as an efficiency layer. That compression path is useful infrastructure, but it is not the core novelty claim of the project.

Both modes use TurboQuant for KV cache compression during generation:

- **Stage 1 (PolarQuant):** Hadamard rotation + symmetric uniform quantization at 2, 3, or 4 bits
- **Stage 2 (QJL):** 1-bit Johnson-Lindenstrauss residual correction for unbiased attention scores

Effective compression: 4-bit → 3.2×, 3-bit → 4×, 2-bit → 5.3× vs FP16. Applied periodically (default every 64 steps) to the growing KV cache.

## Workers

| Worker | Kind | Implementation |
|--------|------|----------------|
| Math | `math` | Safe eval with restricted builtins |
| Code | `code` | Subprocess sandbox with timeout |
| LLM | `llm` | Sub-model query (needs backend) |
| Search | `search` | DuckDuckGo Instant Answer API (no API key) |
| Custom | `custom:name` | User-defined expert profiles |

## Entropy-Guided Delegation

The `EntropyRouter` is the main control layer in Warp Cortex. It computes per-layer attention entropy and logit entropy at every generation step, maintains running statistics with Welford's online algorithm, and fires delegation signals based on z-score deviations rather than hardcoded entropy thresholds. Configurable via `spread_z_threshold` and `logit_z_threshold`.

When enabled, a learned scalar gate runs on the same last-layer hidden states already produced by the model. This keeps the backbone frozen while letting the runtime learn a cheap delegate or do-not-delegate boundary on top of the hidden state stream.

## Memory Model

Warp Cortex uses the same low-level substrate to keep the worker path lightweight:

1. Singleton weight sharing: load the main weights once.
2. Topological synapse: keep compact landmark context instead of cloning the full prompt state per worker.
3. Optional injection and validation layers: worker outputs can be encoded back into the shared memory path.

That compression path is not arbitrary. The broader WarpOS-era attention-geometry work, including the mean-irrelevance view of keys and the Q-K / low-effective-rank story, is the reason landmarking is plausible in the first place: if the query-relevant signal lives in a smaller, deviation-dominated subspace, the runtime does not need to preserve the whole prompt uniformly. The current manifold implementation does not yet run the full query-weighted spectral projector from that research line; it uses a pragmatic salience + topology + coverage heuristic instead. But the connection is real: projection summaries, bridge-aware retrieval, and persistent shared nodes are the runtime expression of that theory.

## Shared-Manifold Validation

The Persistent Shared Manifold now has five validated proof-of-mechanism slices on the local Qwen/Qwen2.5-0.5B-Instruct CPU path. The current consolidated evaluation shows a clear split: coding transfer, necessity, topology, and the targeted energy-reuse slice all expose the intended mechanism behavior, while recall handoff remains the main small-model weak point.

| Slice | Aggregate result |
|---|---|
| Real coding compare (3 tasks) | manifold-enabled `pass_rate=0.67` vs disabled `pass_rate=0.00`; enabled `prompt_hit_rate=1.00` vs disabled `prompt_hit_rate=0.00` |
| Targeted energy reuse (3 tasks, ablation) | energy off: `followup_target_hit_rate=0.00`, `distractor_capture_rate=1.00`; energy on: `followup_target_hit_rate=1.00`, `distractor_capture_rate=0.00`, `followup_patch_hit_rate=1.00`, `avg_energy_peak=0.18` |
| Coding handoff (3 tasks) | `context_match_rate=1.00`, `output_match_rate=1.00`, `fresh_pass_rate=0.00`, `loaded_pass_rate=0.67` |
| Recall handoff (5 tasks) | `fresh_answer_rate=0.00`; recent reruns place `loaded_answer_rate` in the `0.60-0.80` band instead of a stable `1.00`, so this remains the main brittle slice on the small local model |
| Necessity ablation (3 tasks) | `isolated_prompt_hit_rate=0.00`, `manifold_prompt_hit_rate=1.00`, `isolated_answer_rate=0.00`, `manifold_answer_rate=1.00`, `oracle_answer_rate=1.00`, `necessity_win_rate=1.00` |
| Real topology retrieval (2 tasks) | `component_accuracy_rate=1.00`, `active_region_accuracy_rate=1.00`, `topology_expected_recall_rate=1.00`, `flat_expected_recall_rate=0.75`, `topology_answer_rate=1.00`, `flat_answer_rate=0.00`, `topology_win_rate=1.00` |

The energy-reuse slice is intentionally narrower than the others. It seeds several nearly identical task-board neighborhoods, primes one neighborhood twice, and then asks a blended follow-up query. Its purpose is not breadth; it is to demonstrate that prompt-time manifold energy can measurably bias subsequent task-board selection inside the same live engine session.

The topology slice is specifically about retrieval structure rather than prompt inheritance. It checks that active-component, bridge-aware selection over persistent shared nodes materially outperforms a flat lexical baseline on real field-extraction and bridge-chain tasks.

Useful benchmark entry points:

```bash
# Deterministic topology isolate
python cortex_benchmarks/benchmark_shared_manifold.py --mode topology-compare

# Real local-model topology slice
python cortex_benchmarks/benchmark_shared_manifold.py --mode real-topology-compare --device cpu --max-tokens 48

# Full consolidated evaluation with energy ablation
python cortex_benchmarks/full_shared_manifold_evaluation.py --device cpu --real-repeats 1 --include-energy-ablation --energy-ablation-repeats 1

# Unified validations
python cortex_validation/tests.py
```

## Entry Points

Commands below assume you are in the `warp_cortex/` repo root.

```bash
# Live orchestrated reasoning runner
python cortex_scripts/council_live.py "How many r's are in strawberry?"

# GSM8K benchmark
python cortex_benchmarks/benchmark_cortex_gsm8k.py --n 20 --modes single,orchestrated

# Tests
python cortex_validation/tests.py
```

Local-only experiments, paper drafts, and generated artifacts stay outside the public repo surface.

## Layout

1. `cortex_core/`: internal runtime modules (adaptive engine, entropy router, TurboQuant, async delegation, cortex router).
2. `cortex_scripts/`: user-facing CLIs.
3. `cortex_benchmarks/`: official quality benchmarks.
4. `cortex_validation/`: repo-only regression and validation scripts.
5. `cortex_resources/`: persistent runtime agent skill definitions.
6. `docs/`: architecture notes.
