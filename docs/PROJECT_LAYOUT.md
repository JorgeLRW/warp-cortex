# Warp Cortex Project Layout

This repo is intentionally organized around a small public runtime surface.

## Canonical Folders

### `cortex_core/`

Core runtime modules.

- `adaptive_engine.py` — AWARE / SILENT / COUNCIL generation engine
- `async_delegate.py` — worker dispatch, BitNet adapter, search worker
- `entropy_router.py` — inline uncertainty monitoring
- `synapse.py`, `turbo_quant.py`, `stream_inject.py` — memory and injection primitives

### `cortex_scripts/`

Primary interactive runners.

- `council_live.py` — orchestrated reasoning CLI

### `cortex_benchmarks/`

Official evaluation benchmarks.

- `benchmark_cortex_gsm8k.py` — quality benchmark

### `cortex_validation/`

Validation and regression checks.

- `tests.py` — unified test suite
- `test_upgrades.py`, `test_automation.py`, `test_council.py` — targeted checks

### `cortex_resources/`

Runtime-owned data files.

- `agent_skills/default_skills.json` — persistent skill definitions loaded by the engine

## Config Files

### `config/settings.yaml`

Committed shared defaults.

### `config/settings.local.yaml`

Untracked local overrides and secrets.

Use this for:

- API keys
- alternate model defaults
- machine-specific Hugging Face cache paths
- preferred benchmark or demo defaults

## Root Policy

The repository root should stay small.

- Keep `cortex_engine.py` as the main top-level engine module
- Keep setup, docs, config, and package folders
- If a legacy import must survive, keep it as a thin compatibility shim only
- Put runtime-owned data under `cortex_resources/`
- Put generated outputs under `local_artifacts/`, not inside source directories
- Put CLIs, benchmarks, and tests under their functional folders instead of root wrappers

## Local-Only Paths

These are intentionally kept out of the public repo surface and should stay gitignored locally.

- `research/` — experiments, probes, and exploratory scripts
- `paper/` — draft paper material
- `local_artifacts/` — generated outputs, caches, and machine-specific artifacts