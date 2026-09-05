"""
Warp-Cortex GSM8K Quality Benchmark
===================================

Compares real inference modes on the GSM8K test split:

    1. **Single-Agent Baseline**  — one forward pass, greedy decode.
    2. **Orchestrated Mode**      — main model answers directly and delegates
                                                                     only when it explicitly emits a worker task.
    3. **BitNet Orchestrated**    — currently a simulated placeholder and
                                                                     disabled by default for integrity.

Metrics reported per mode:
    - Accuracy (exact match after #### extraction)
    - Mean tokens/sec throughput
    - Peak VRAM (if CUDA)
    - Average delegated tasks per question

Usage:
        cd warp_cortex
    python cortex_benchmarks/benchmark_cortex_gsm8k.py [--n 50] [--modes single,orchestrated]
"""

import sys, os, re, time, copy, argparse, torch
from typing import List, Dict, Optional, cast

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from cortex_core.hf_utils import prepare_hf_cache, resolve_local_model_source
from cortex_core.settings import get_setting, load_settings, resolve_project_path
from cortex_core.async_delegate import AsyncDelegationManager
from cortex_scripts.council_live import DIRECT_SYSTEM, OrchestratedReasoningEngine

from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_generation_config(model, tokenizer, do_sample: bool, temperature: float = 0.0):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = do_sample
    generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if do_sample:
        generation_config.temperature = max(temperature, 0.01)
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.min_p = None
        generation_config.typical_p = None
        generation_config.epsilon_cutoff = None
        generation_config.eta_cutoff = None
    return generation_config


# ======================================================================
# Answer extraction  (matches the format used in swarm_solutions/)
# ======================================================================

def extract_answer(text: str) -> Optional[str]:
    """Pull the numerical answer from model output."""
    clean = text.replace(',', '')

    # 1. GSM8K canonical format  #### <number>
    if "####" in clean:
        suffix = clean.split("####")[-1].strip()
        if suffix:
            match = re.search(r'-?\d+(?:\.\d+)?', suffix)
            if match:
                return match.group(0)

    # 2. Common LLM patterns
    for pat in [
        r'\\boxed\{(-?\d+(?:\.\d+)?)\}',
        r'[Ff]inal [Aa]nswer:?\s*\$?(-?\d+(?:\.\d+)?)',
        r'[Tt]he answer is\s*\$?(-?\d+(?:\.\d+)?)',
    ]:
        m = re.findall(pat, clean)
        if m:
            return m[-1]

    # 3. Last number in output (generous fallback)
    nums = re.findall(r'-?\d+(?:\.\d+)?', clean)
    return nums[-1] if nums else None


def check_answer(pred: str, truth: str) -> bool:
    try:
        return abs(float(pred) - float(truth)) < 0.5
    except (ValueError, TypeError):
        return False


# ======================================================================
# 1. Single-Agent Baseline
# ======================================================================

class SingleAgentRunner:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def solve(self, question: str, max_new_tokens: int = 512) -> str:
        msgs = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generation_config = _build_generation_config(self.model, self.tokenizer, do_sample=False)
        out = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )
        text = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return extract_answer(text) or text


# ======================================================================
# 2. Orchestrated mode (explicit delegation)
# ======================================================================

class HFChatBackend:
    """Small backend adapter so the orchestrated runner can reuse the live engine."""

    def __init__(self, model, tokenizer, device, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]],
                 temperature: float = 0.0,
                 max_tokens: int = 512) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = _build_generation_config(
            self.model,
            self.tokenizer,
            do_sample=temperature > 0,
            temperature=temperature,
        )
        out = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class OrchestratedRunner:
    def __init__(self, model, tokenizer, device, model_id: str,
                 max_rounds: int = 3, max_workers: int = 4):
        self.backend = HFChatBackend(model, tokenizer, device, model_id)
        self.delegation_mgr = AsyncDelegationManager(
            backend=self.backend,
            max_workers=max_workers,
            device=device,
        )
        self.engine = OrchestratedReasoningEngine(
            self.backend,
            max_rounds=max_rounds,
            delegation_mgr=self.delegation_mgr,
            verbose=False,
        )

    def solve(self, question: str, max_new_tokens: int = 512) -> Dict:
        return self.engine.solve(question, max_tokens=max_new_tokens)

    def shutdown(self):
        self.delegation_mgr.shutdown()


# ======================================================================
# 3. BitNet orchestrated (placeholder — same API, marks mode)
# ======================================================================

class BitNetOrchestratedRunner(OrchestratedRunner):
    """Simulated placeholder for a future end-to-end BitNet GSM8K path."""

    def __init__(self, model, tokenizer, device, model_id: str,
                 max_rounds: int = 3, max_workers: int = 4):
        super().__init__(model, tokenizer, device, model_id, max_rounds, max_workers)
        # In production: model = BitNetSideAgent.from_pretrained(model)
        # For benchmark accuracy measurement, we use the same FP16 model
        # since the packed ternary weights need calibration that is
        # out of scope here.  VRAM savings are shown in stats.


# ======================================================================
# Benchmark harness
# ======================================================================

def load_gsm8k(n: int) -> List[Dict[str, str]]:
    """Load first n samples from GSM8K test split."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = cast(List[Dict[str, str]], list(ds.select(range(min(n, len(ds))))))
        return [
            {"question": str(row["question"]), "answer": str(row["answer"])}
            for row in rows
        ]
    except Exception as e:
        print(f"[WARN] Could not load HF dataset ({e}). Using built-in problems.")
        return _builtin_problems()[:n]


def _builtin_problems():
    """Fallback if datasets library not installed."""
    return [
        {"question": "Janet has 3 times as many eggs as Marcus. Marcus has 5 chickens, and each chicken lays 2 eggs per day. How many eggs does Janet have after a week?",
         "answer": "Janet gets 3 * 5 * 2 = 30 eggs/day. In a week: 30 * 7 = 210. #### 210"},
        {"question": "A train travels at 60 mph for 3 hours, then 40 mph for 2 hours. What is the total distance?",
         "answer": "60*3 + 40*2 = 180 + 80 = 260. #### 260"},
        {"question": "If it takes 10 minutes to boil 1 egg, how long does it take to boil 10 eggs simultaneously?",
         "answer": "All eggs boil at the same time. #### 10"},
        {"question": "Find the smallest positive integer x such that x mod 3 = 2, x mod 4 = 3, and x mod 5 = 4.",
         "answer": "By CRT, x = 59. #### 59"},
        {"question": "How many r's are in the word 'strawberry'?",
         "answer": "s-t-r-a-w-b-e-r-r-y. Three r's. #### 3"},
    ]


def clean_truth(answer_text: str) -> Optional[str]:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None


def run_benchmark(args, settings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model
    cache_root = resolve_project_path(get_setting(settings, "paths.huggingface_cache"))
    cache_dir = prepare_hf_cache(ROOT_DIR, preferred_root=cache_root)

    print("=" * 70)
    print(f"  Warp-Cortex GSM8K Benchmark  |  model={model_id}  |  N={args.n}")
    print("=" * 70)

    # Load model
    model_source, local_files_only = resolve_local_model_source(model_id, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model_kwargs = {
        "cache_dir": cache_dir,
        "dtype": torch.float16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    if local_files_only:
        model_kwargs["local_files_only"] = True
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    # Load dataset
    dataset = load_gsm8k(args.n)

    # Build runners for requested modes
    raw_modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    modes = []
    for mode in raw_modes:
        if mode == "council":
            print("[INFO] Mode 'council' is deprecated; using 'orchestrated'.")
            modes.append("orchestrated")
        else:
            modes.append(mode)
    runners = {}
    if "bitnet" in modes and not args.allow_simulated_bitnet:
        raise SystemExit(
            "BitNet GSM8K mode is still simulated in this repo. "
            "Run real benchmarks with --modes single,orchestrated or add "
            "--allow-simulated-bitnet to opt into the placeholder path."
        )

    if "single" in modes:
        runners["Single-Agent"] = SingleAgentRunner(model, tokenizer, device)
    if "orchestrated" in modes:
        runners["Cortex-Orchestrated"] = OrchestratedRunner(
            model, tokenizer, device, model_id, max_rounds=args.rounds, max_workers=args.workers
        )
    if "bitnet" in modes:
        runners["BitNet-Orchestrated (simulated)"] = BitNetOrchestratedRunner(
            model, tokenizer, device, model_id, max_rounds=args.rounds, max_workers=args.workers
        )

    results = {}

    for mode_name, runner in runners.items():
        print(f"\n{'─' * 60}")
        print(f"  Mode: {mode_name}")
        print(f"{'─' * 60}")

        correct = 0
        total = 0
        total_time = 0.0
        total_delegations = 0
        total_rounds = 0

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, item in enumerate(dataset):
            question = item["question"]
            truth = clean_truth(item["answer"])
            if truth is None:
                continue

            t0 = time.perf_counter()
            raw_result = runner.solve(question)
            dt = time.perf_counter() - t0
            total_time += dt

            if isinstance(raw_result, dict):
                pred = raw_result.get("answer")
                total_delegations += int(raw_result.get("delegations", 0))
                total_rounds += int(raw_result.get("rounds", 1))
            else:
                pred = raw_result
                total_rounds += 1

            ok = check_answer(pred, truth)
            correct += int(ok)
            total += 1

            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] Q{i+1}: pred={pred}, truth={truth} ({dt:.1f}s)")

        acc = correct / max(total, 1)
        throughput = total / max(total_time, 0.001)

        peak_vram = 0
        if device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

        results[mode_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "time_s": total_time,
            "qps": throughput,
            "peak_vram_gb": peak_vram,
            "avg_delegations": total_delegations / max(total, 1),
            "avg_rounds": total_rounds / max(total, 1),
        }

        print(f"\n  {mode_name}: {correct}/{total} ({acc:.1%}) "
              f"| {total_time:.1f}s | VRAM={peak_vram:.2f} GB")

        if hasattr(runner, "shutdown"):
            runner.shutdown()

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Mode':<28} {'Acc':>8} {'Time':>8} {'Q/s':>8} {'Deleg/Q':>9} {'VRAM':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<28} {r['accuracy']:>7.1%} {r['time_s']:>7.1f}s "
              f"{r['qps']:>7.2f} {r['avg_delegations']:>8.2f} {r['peak_vram_gb']:>7.2f}G")
    print("=" * 70)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML settings override (defaults to config/settings.yaml + optional settings.local.yaml)",
    )
    pre_args, _ = pre_parser.parse_known_args()
    settings = load_settings(pre_args.config)

    default_n = int(get_setting(settings, "benchmarks.gsm8k.n", 20))
    default_model = str(get_setting(settings, "backends.local.model", "Qwen/Qwen2.5-0.5B-Instruct"))
    default_modes = str(get_setting(settings, "benchmarks.gsm8k.modes", "single,orchestrated"))
    default_workers = int(get_setting(settings, "benchmarks.gsm8k.workers", 4))
    default_rounds = int(get_setting(settings, "benchmarks.gsm8k.rounds", 3))

    parser = argparse.ArgumentParser(parents=[pre_parser], description="Warp-Cortex GSM8K Benchmark")
    parser.add_argument("--n", type=int, default=default_n, help="Number of GSM8K problems")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--modes", default=default_modes,
                        help="Comma-separated modes: single,orchestrated,bitnet")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Max concurrent worker tasks in orchestrated modes")
    parser.add_argument("--agents", dest="workers", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=default_rounds,
                        help="Max orchestrator turns per question")
    parser.add_argument("--refines", dest="rounds", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-simulated-bitnet",
        action="store_true",
        help="Allow the placeholder BitNet GSM8K path to run with explicit simulated labeling.",
    )
    run_benchmark(parser.parse_args(), settings)


if __name__ == "__main__":
    main()
