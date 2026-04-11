# Warp Cortex

Warp Cortex is a single-model orchestration runtime with inline entropy monitoring and adaptive delegation. The main model answers directly by default and only uses workers when needed — either explicitly (System A), via entropy-detected confusion (System B), or through parallel council deliberation (System C).

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

The model never knows delegation happened.

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

## TurboQuant KV Cache Compression

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

## Entropy Router

The `EntropyRouter` computes per-layer attention entropy and logit entropy at every generation step. It uses Welford's online algorithm for running statistics and fires delegation signals based on z-score deviations — no hardcoded entropy thresholds. Configurable via `spread_z_threshold` and `logit_z_threshold`.

## Memory Model

Warp Cortex uses the same low-level substrate:

1. Singleton weight sharing: load the main weights once.
2. Topological synapse: keep compact landmark context instead of cloning the full prompt state per worker.
3. Optional injection and validation layers: worker outputs can be encoded back into the shared memory path.

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
