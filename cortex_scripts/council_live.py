"""
Warp-Cortex Orchestrated Reasoning
==================================

The main model answers directly by default. When it decides a narrow
subtask should be delegated, it emits an explicit delegation block,
Warp Cortex runs that worker task, and the result is fed back into the
next model turn.

Architecture:
    main model -> optional [DELEGATE:*] -> focused worker -> worker result
    -> main model resumes with the new evidence

This keeps the default path cheap and simple while preserving the
ability to offload calculations, code execution, or focused subqueries.

Legacy note:
    The public filename remains council_live.py for compatibility, but
    the default behavior is now orchestrator-style sparse delegation
    rather than always-on council voting or universal verification.

Usage:
    # Local model (RTX 4090):
    python cortex_scripts/council_live.py "How many r's in strawberry?"

    # Enable explicit worker delegation:
    python cortex_scripts/council_live.py --async-delegate "What is 17*23?"

    # OpenAI API:
    python cortex_scripts/council_live.py --api openai --api-model gpt-4o-mini "What is 17*23?"

    # GSM8K benchmark:
    python cortex_scripts/council_live.py --bench 10
"""

import sys, os, re, time, copy, argparse, torch
from typing import Any, List, Dict, Optional, Tuple, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from cortex_core.hf_utils import prepare_hf_cache
from cortex_core.settings import get_setting, load_settings, resolve_project_path

# ── ANSI ─────────────────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_RED    = "\033[31m"
C_GREEN  = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE   = "\033[34m"
C_CYAN   = "\033[36m"
C_MAG    = "\033[35m"


# ======================================================================
# Backend abstraction — local model or API
# ======================================================================

class LocalBackend:
    """HuggingFace model on local GPU."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 device: str = "cuda",
                 cache_root: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"{C_DIM}Loading {model_id}...{C_RESET}")
        cache_dir = prepare_hf_cache(ROOT_DIR, preferred_root=cache_root)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        runtime_device = (
            device if str(device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        model_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "dtype": torch.float16 if runtime_device.startswith("cuda") else torch.float32,
        }
        if runtime_device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not runtime_device.startswith("cuda"):
            self.model.to(runtime_device)
        self.model.eval()
        self.device = runtime_device
        self.model_id = model_id

    @property
    def embed_layer(self):
        """Expose model's token embedding layer for ClaimEncoder."""
        return self.model.model.embed_tokens

    @property
    def hidden_dim(self):
        """Model's hidden dimension."""
        return self.model.config.hidden_size

    def _build_generation_config(self, do_sample: bool, temperature: float):
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = do_sample
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
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

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]],
                 temperature: float = 0.3,
                 max_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        do_sample = temperature > 0
        generation_config = self._build_generation_config(do_sample, temperature)
        out = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, messages_list: List[List[Dict[str, str]]],
                       temperature: float = 0.0,
                       max_tokens: int = 64) -> List[str]:
        """Batch generation — used for parallel worker dispatch."""
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(self.device)
        do_sample = temperature > 0
        generation_config = self._build_generation_config(do_sample, temperature)
        out = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )
        results = []
        for ids in out:
            text = self.tokenizer.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append(text)
        return results


class APIBackend:
    """OpenAI-compatible API (works with OpenAI, Anthropic, vLLM, Ollama)."""

    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError:
            raise RuntimeError("pip install openai  (needed for API backend)")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(base_url=base_url, api_key=self.api_key)
        self.model = model
        self.model_id = f"api:{model}"
        print(f"{C_DIM}Using API: {base_url} / {model}{C_RESET}")

    def generate(self, messages: List[Dict[str, str]],
                 temperature: float = 0.3,
                 max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=cast(Any, messages),
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def generate_batch(self, messages_list: List[List[Dict[str, str]]],
                       temperature: float = 0.0,
                       max_tokens: int = 64) -> List[str]:
        results: List[str] = [""] * len(messages_list)
        with ThreadPoolExecutor(max_workers=min(8, len(messages_list))) as pool:
            futs = {
                pool.submit(self.generate, m, temperature, max_tokens): i
                for i, m in enumerate(messages_list)
            }
            for f in as_completed(futs):
                results[futs[f]] = f.result()
        return results


# ======================================================================
# Answer extraction
# ======================================================================

def extract_answer(text: str) -> Optional[str]:
    clean = text.replace(",", "")
    if "####" in clean:
        suffix = clean.split("####")[-1].strip()
        if suffix:
            match = re.search(r'-?\d+(?:\.\d+)?', suffix)
            if match:
                return match.group(0)
    for pat in [
        r'\\boxed\{(-?\d+(?:\.\d+)?)\}',
        r'[Ff]inal [Aa]nswer:?\s*\$?(-?\d+(?:\.\d+)?)',
        r'[Tt]he answer is\s*\$?(-?\d+(?:\.\d+)?)',
    ]:
        m = re.findall(pat, clean)
        if m: return m[-1]
    nums = re.findall(r'-?\d+(?:\.\d+)?', clean)
    return nums[-1] if nums else None


def answers_match(a: str, b: str) -> bool:
    try: return abs(float(a) - float(b)) < 0.5
    except (ValueError, TypeError): return False


# ======================================================================
# Claim Extraction — the main model decomposes its own reasoning
# ======================================================================

# Map display operators to Python ops
_OP_MAP = {'+': '+', '-': '-', '*': '*', '/': '/', '×': '*', '÷': '/'}


def _strip_latex(text: str) -> str:
    """Strip LaTeX formatting to expose bare arithmetic."""
    s = text
    # Remove \[ ... \] and \( ... \) delimiters
    s = re.sub(r'\\\[|\\\]|\\\(|\\\)', ' ', s)
    # Remove \text{...} (including leading/trailing spaces inside braces)
    s = re.sub(r'\\text\s*\{[^}]*\}', ' ', s)
    # Remove \frac{a}{b} → (a)/(b)
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    # Remove \times → *, \div → /, \cdot → *
    s = s.replace('\\times', '*').replace('\\div', '/').replace('\\cdot', '*')
    # Remove other LaTeX commands (\left, \right, etc.)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# Chain pattern: matches "A op B op C ... = result" (2+ operands)
_CHAIN_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)'                        # first operand
    r'((?:\s*[+\-*/×÷]\s*\d+(?:\.\d+)?)+)'   # one or more " op operand"
    r'\s*=\s*'
    r'(-?\d+(?:\.\d+)?)'                       # result
)

# Simple binary: "A op B = result"
_BINARY_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)'
)


def extract_claims(reasoning: str) -> List[Dict]:
    """
    Pull verifiable arithmetic claims from the model's own reasoning.
    Handles plain text AND LaTeX-formatted math (\\text{}, \\[...\\], etc.).
    Returns list of {"expression": "48/3", "claimed": "16", "type": "arithmetic"}
    """
    claims = []
    seen = set()

    # Work on both original text and LaTeX-stripped version
    variants = [reasoning, _strip_latex(reasoning)]

    for text in variants:
        # 1) Try chain pattern first (catches "A + B + C = D")
        for m in _CHAIN_PATTERN.finditer(text):
            first_op = m.group(1)
            chain = m.group(2)
            result = m.group(3)

            # Reconstruct expression: replace display ops with Python ops
            expr = first_op + chain
            for display_op, py_op in _OP_MAP.items():
                expr = expr.replace(display_op, f' {py_op} ')
            expr = re.sub(r'\s+', ' ', expr).strip()

            key = f"{expr}={result}"
            if key not in seen:
                seen.add(key)
                claims.append({
                    "expression": expr,
                    "claimed": result,
                    "type": "arithmetic",
                    "raw": m.group(0),
                })

        # 2) Also grab simple binary patterns (may overlap, dedup by key)
        for m in _BINARY_PATTERN.finditer(text):
            a, op, b, result = m.groups()
            op_py = _OP_MAP.get(op, op)
            expr = f"{a} {op_py} {b}"

            key = f"{expr}={result}"
            if key not in seen:
                seen.add(key)
                claims.append({
                    "expression": expr,
                    "claimed": result,
                    "type": "arithmetic",
                    "raw": m.group(0),
                })

    return claims


_SAFE_EXPR = re.compile(r'^[\d\s+\-*/().]+$')

def verify_claim_locally(claim: Dict) -> Dict:
    """
    Mechanically verify a single claim. No LLM needed for arithmetic.
    Returns the claim dict with 'verified' (bool) and 'actual' (str) added.
    """
    if claim["type"] == "arithmetic":
        expr = claim["expression"]
        if not _SAFE_EXPR.match(expr):
            claim["actual"] = "UNSAFE"
            claim["verified"] = False
            return claim
        try:
            actual = eval(expr)  # safe: validated by regex above
            claimed = float(claim["claimed"])
            passed = abs(actual - claimed) < 0.5
            claim["actual"] = str(actual)
            claim["verified"] = passed
        except Exception:
            claim["actual"] = "ERROR"
            claim["verified"] = False
    else:
        claim["verified"] = True  # can't check non-arithmetic locally
        claim["actual"] = claim["claimed"]
    return claim


# ======================================================================
# Orchestrated reasoning prompts
# ======================================================================

DIRECT_SYSTEM = (
    "You are a careful math and reasoning solver. Think step by step, solve "
    "the problem directly, and end with #### followed by ONLY the final "
    "number."
)

ORCHESTRATION_APPENDIX = (
    "If a narrow subtask truly needs external help, you may emit one or more "
    "delegation blocks instead of guessing.\n\n"
    "Delegation syntax:\n"
    "  [DELEGATE:math] expression [/DELEGATE]\n"
    "  [DELEGATE:code] python_code_here [/DELEGATE]\n"
    "  [DELEGATE:llm] focused question [/DELEGATE]\n"
    "  [DELEGATE:custom:expert_name] instructions | payload [/DELEGATE]\n\n"
    "Important rules:\n"
    "- Delegate only the subtask, not the whole problem.\n"
    "- Do not invent worker outputs. If you delegate, wait for results.\n"
    "- After worker results arrive, continue solving from that evidence.\n"
    "- If no worker is needed, solve the problem directly as usual."
)

ORCHESTRATOR_SYSTEM = f"{DIRECT_SYSTEM}\n\n{ORCHESTRATION_APPENDIX}"

REVIEW_SYSTEM = (
    "You are a careful problem solver. A lightweight arithmetic review found "
    "specific mismatches in your previous attempt. Fix only the incorrect "
    "steps, keep the valid reasoning, and end with #### followed by ONLY the "
    "final number."
)

# Worker prompt — ultra-short, mechanical verification
WORKER_SYSTEM = (
    "You are a verification worker. You receive a claim to check. "
    "Compute the result. Reply with ONLY: PASS <result> or FAIL <correct_result>"
)


_DELEGATE_BLOCK_PATTERN = re.compile(
    r'\[DELEGATE:\w+(?::\w+)?\]\s*.*?\s*\[/DELEGATE\]',
    re.DOTALL,
)


def strip_delegation_blocks(text: str) -> str:
    cleaned = _DELEGATE_BLOCK_PATTERN.sub('', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def format_delegation_results(results: List[Any]) -> str:
    lines = [
        "Worker results:",
        "Use these results if helpful. If you need more help, emit new [DELEGATE:...] blocks.",
        "Otherwise finish the solution and end with #### followed by ONLY the final number.",
        "",
    ]
    for result in results:
        status = "SUCCESS" if result.success else "FAIL"
        lines.append(f"- task={result.task_id} kind={result.expert_kind} status={status}")
        lines.append(f"  payload: {result.payload}")
        if result.success:
            lines.append(f"  output: {result.output or '<empty>'}")
        else:
            lines.append(f"  error: {result.error or '<unknown>'}")
    return "\n".join(lines)


def extract_answer_from_worker_results(results: List[Any]) -> Optional[str]:
    for result in reversed(results):
        if result.success and result.output:
            worker_answer = extract_answer(result.output)
            if worker_answer is not None:
                return worker_answer
    return None


def build_review_feedback(question: str, failed_claims: List[Dict]) -> str:
    lines = [
        question,
        "",
        "Arithmetic review mismatches:",
    ]
    for claim in failed_claims:
        lines.append(
            f"- You wrote '{claim['raw']}', but {claim['expression']} = {claim['actual']} "
            f"(not {claim['claimed']})."
        )
    lines.append("")
    lines.append("Fix the incorrect steps and continue to the final answer.")
    return "\n".join(lines)


class OrchestratedReasoningEngine:
    """
    Main-model-first reasoning with explicit, sparse delegation.

    Default behavior:
    - The main model answers in one shot.

    Optional behavior:
    - If the model emits [DELEGATE:...] blocks, focused workers execute
      those subtasks and their results are fed back into the next turn.
    - If verify_claims is enabled, explicit arithmetic claims can be
      reviewed after each turn and only failing claims trigger another turn.
    """

    def __init__(self, backend, max_rounds: Optional[int] = None,
                 max_refine: Optional[int] = None, verify_claims: bool = False,
                 use_llm_workers: bool = False, stream_injector=None,
                 delegation_mgr=None, verbose: bool = True):
        self.backend = backend
        resolved_rounds = max_rounds if max_rounds is not None else max_refine
        self.max_rounds = max(1, int(resolved_rounds if resolved_rounds is not None else 3))
        self.verify_claims = verify_claims
        self.use_llm_workers = use_llm_workers
        self.stream_injector = stream_injector  # StreamInjector from stream_inject.py
        self.delegation_mgr = delegation_mgr    # AsyncDelegationManager from async_delegate.py
        self.verbose = verbose
        self.trace: List[Dict] = []

    def _emit(self, text: str = ""):
        if self.verbose:
            print(text)

    def _display_reasoning(self, label: str, text: str):
        display_source = text.strip() or "[no visible reasoning emitted]"
        lines = display_source.split("\n")
        display = "\n".join(f"    {line}" for line in lines[:18])
        if len(lines) > 18:
            display += f"\n    {C_DIM}... ({len(lines) - 18} more lines){C_RESET}"
        self._emit(f"\n{label}")
        self._emit(f"{C_CYAN}{display}{C_RESET}")

    def _review_claims(self, reasoning: str, round_index: int) -> Tuple[int, List[Dict]]:
        claims = extract_claims(reasoning)
        if not claims:
            self._emit(f"  {C_DIM}No explicit arithmetic claims to review.{C_RESET}")
            return 0, []

        self._emit(f"\n  {C_DIM}Reviewing {len(claims)} explicit arithmetic claim(s):{C_RESET}")
        failed_claims = []
        for claim in claims:
            if not self.use_llm_workers:
                claim = verify_claim_locally(claim)
            else:
                claim = self._verify_via_worker(claim)

            status_icon = f"{C_GREEN}✓{C_RESET}" if claim["verified"] else f"{C_RED}✗{C_RESET}"
            expr_display = f"{claim['expression']} = {claim['claimed']}"
            if claim["verified"]:
                self._emit(f"    {status_icon} {expr_display}")
            else:
                self._emit(
                    f"    {status_icon} {expr_display}  "
                    f"{C_RED}(actual: {claim['actual']}){C_RESET}"
                )
                failed_claims.append(claim)

            if self.stream_injector is not None:
                from cortex_core.stream_inject import VerifiedClaim

                self.stream_injector.inject_verified_claim(
                    VerifiedClaim(
                        expression=claim["expression"],
                        claimed=claim["claimed"],
                        actual=claim.get("actual", claim["claimed"]),
                        verified=claim["verified"],
                    )
                )

            self.trace.append({
                "round": round_index,
                "phase": "review",
                "claim": claim["expression"],
                "claimed": claim["claimed"],
                "actual": claim.get("actual", claim["claimed"]),
                "passed": claim["verified"],
            })

        return len(claims), failed_claims

    def solve(self, question: str, max_tokens: int = 400) -> Dict:
        self.trace = []
        t0 = time.perf_counter()

        self._emit(f"\n{'='*70}")
        self._emit(f"{C_BOLD}QUESTION:{C_RESET} {question}")
        self._emit(f"{'='*70}")

        final_answer = None
        final_text = ""
        total_claims_checked = 0
        total_rounds = 0
        total_delegations = 0
        delegation_results = []
        seen_requests = set()
        repeated_request_seen = False

        delegation_prompt_enabled = False
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": question},
        ]

        for round_index in range(self.max_rounds):
            total_rounds = round_index + 1
            if round_index == 0:
                self._emit(
                    f"\n{C_CYAN}{C_BOLD}  [Main Model]{C_RESET} {C_DIM}Solving...{C_RESET}"
                )
            else:
                self._emit(
                    f"\n{C_YELLOW}{C_BOLD}  [Main Model]{C_RESET} "
                    f"{C_DIM}Continuing (round {round_index + 1}/{self.max_rounds})...{C_RESET}"
                )

            raw_output = self.backend.generate(messages, temperature=0.0, max_tokens=max_tokens)
            clean_output = strip_delegation_blocks(raw_output)
            visible_output = clean_output if clean_output else raw_output.strip()
            final_text = visible_output
            answer = extract_answer(raw_output) or extract_answer(clean_output)
            if answer:
                final_answer = answer

            self._display_reasoning("", visible_output)
            if answer:
                self._emit(f"  {C_CYAN}  -> Answer: {C_BOLD}{answer}{C_RESET}")

            self.trace.append({
                "round": round_index,
                "phase": "model",
                "answer": answer,
                "delegation_requested": "[DELEGATE:" in raw_output,
                "output_len": len(raw_output),
            })

            requests = []
            had_delegation_markup = False
            if self.delegation_mgr is not None:
                from cortex_core.async_delegate import detect_delegation_requests

                parsed_requests = detect_delegation_requests(raw_output)
                had_delegation_markup = bool(parsed_requests)
                for request in parsed_requests:
                    signature = (
                        request.expert_kind.strip().lower(),
                        request.instructions.strip(),
                        request.payload.strip(),
                    )
                    if signature in seen_requests:
                        repeated_request_seen = True
                        continue
                    seen_requests.add(signature)
                    requests.append(request)

            if requests:
                if round_index == self.max_rounds - 1:
                    self._emit(
                        f"\n  {C_YELLOW}Delegation requested at the round limit — "
                        f"returning the best available answer.{C_RESET}"
                    )
                    break

                if not delegation_prompt_enabled:
                    messages[0] = {"role": "system", "content": ORCHESTRATOR_SYSTEM}
                    delegation_prompt_enabled = True

                task_ids = self.delegation_mgr.dispatch_batch(requests)
                total_delegations += len(task_ids)
                self._emit(
                    f"\n  {C_MAG}{C_BOLD}[Dispatching {len(task_ids)} delegated task(s)]{C_RESET}"
                )
                for request in requests:
                    self._emit(
                        f"    {C_DIM}{request.expert_kind}: {request.payload[:80]}{C_RESET}"
                    )

                self.delegation_mgr.wait_all(timeout=30.0)
                round_results = self.delegation_mgr.poll_results()
                delegation_results.extend(round_results)

                self.trace.append({
                    "round": round_index,
                    "phase": "delegate",
                    "task_ids": task_ids,
                })

                self._emit(f"\n  {C_MAG}{C_BOLD}[Worker results]{C_RESET}")
                for result in round_results:
                    icon = f"{C_GREEN}✓{C_RESET}" if result.success else f"{C_RED}✗{C_RESET}"
                    preview = result.output[:80] if result.success else result.error[:80]
                    self._emit(f"    {icon} [{result.expert_kind}] {preview}")

                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": format_delegation_results(round_results)})
                continue

            if had_delegation_markup and repeated_request_seen and round_index < self.max_rounds - 1:
                worker_answer = extract_answer_from_worker_results(delegation_results)
                if not clean_output and worker_answer is not None:
                    final_answer = worker_answer
                    final_text = worker_answer
                    self.trace.append({
                        "round": round_index,
                        "phase": "worker-finalize",
                        "answer": worker_answer,
                    })
                    break

                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "Those delegation requests were already completed earlier. "
                        "Reuse the existing worker results already in the conversation and "
                        "continue to the final answer. End with #### followed by ONLY the "
                        "final number."
                    ),
                })
                continue

            failed_claims: List[Dict] = []
            if self.verify_claims:
                checked_count, failed_claims = self._review_claims(clean_output, round_index)
                total_claims_checked += checked_count
                if failed_claims and round_index < self.max_rounds - 1:
                    if self.delegation_mgr is not None and not delegation_prompt_enabled:
                        messages[0] = {"role": "system", "content": ORCHESTRATOR_SYSTEM}
                        delegation_prompt_enabled = True
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({
                        "role": "user",
                        "content": build_review_feedback(question, failed_claims),
                    })
                    continue

            if answer is None and self.delegation_mgr is not None and round_index < self.max_rounds - 1:
                if not delegation_prompt_enabled:
                    messages[0] = {"role": "system", "content": ORCHESTRATOR_SYSTEM}
                    delegation_prompt_enabled = True
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "If a narrow subtask needs external help, you may now use "
                        "[DELEGATE:...] blocks. Otherwise continue directly to the "
                        "final answer and end with #### followed by ONLY the final number."
                    ),
                })
                continue

            if answer is not None or not failed_claims:
                break

        elapsed = time.perf_counter() - t0

        status = "DELEGATED" if total_delegations else "DIRECT"
        if self.verify_claims and total_claims_checked:
            status = f"{status}+REVIEWED"
        if final_answer is None:
            status = "BEST-EFFORT"

        self._emit(f"\n{'='*70}")
        self._emit(
            f"{C_BOLD}FINAL ANSWER: {final_answer}  "
            f"({status}, rounds={total_rounds}, claims_checked={total_claims_checked}, "
            f"delegations={total_delegations}, time={elapsed:.1f}s){C_RESET}"
        )
        self._emit(f"{'='*70}\n")

        return {
            "answer": final_answer,
            "status": status,
            "reasoning": final_text,
            "rounds": total_rounds,
            "refines": max(total_rounds - 1, 0),
            "claims_checked": total_claims_checked,
            "delegations": total_delegations,
            "delegation_results": delegation_results,
            "elapsed": elapsed,
            "trace": self.trace,
        }

    def _verify_via_worker(self, claim: Dict) -> Dict:
        """Use an LLM worker (BitNet-lite) to verify a single claim."""
        prompt = f"Compute: {claim['expression']}"
        out = self.backend.generate(
            [{"role": "system", "content": WORKER_SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=30,
        )
        # Parse PASS/FAIL from worker
        out_clean = out.strip().upper()
        if "PASS" in out_clean:
            claim["verified"] = True
            nums = re.findall(r'-?\d+(?:\.\d+)?', out_clean)
            claim["actual"] = nums[0] if nums else claim["claimed"]
        elif "FAIL" in out_clean:
            claim["verified"] = False
            nums = re.findall(r'-?\d+(?:\.\d+)?', out_clean)
            claim["actual"] = nums[0] if nums else "?"
        else:
            # Worker didn't follow format — try to extract a number
            nums = re.findall(r'-?\d+(?:\.\d+)?', out_clean)
            if nums:
                try:
                    actual = float(nums[-1])
                    claimed = float(claim["claimed"])
                    claim["verified"] = abs(actual - claimed) < 0.5
                    claim["actual"] = nums[-1]
                except ValueError:
                    claim["verified"] = True  # can't parse — assume ok
                    claim["actual"] = claim["claimed"]
            else:
                claim["verified"] = True
                claim["actual"] = claim["claimed"]
        return claim


System2Engine = OrchestratedReasoningEngine


# ======================================================================
# Benchmark: single-agent vs orchestrated delegation
# ======================================================================

def _builtin_problems():
    return [
        {"question": "Janet has 3 times as many marbles as Tim. Tim has 12 marbles. How many marbles does Janet have?",
         "answer": "3 * 12 = 36. #### 36"},
        {"question": "A store sells apples for $2 each. If you buy 5 or more, you get a $1 discount per apple. How much do 7 apples cost?",
         "answer": "7 * (2-1) = 7. #### 7"},
        {"question": "A train travels at 60 mph for 3 hours, then 40 mph for 2 hours. What is the total distance?",
         "answer": "60*3 + 40*2 = 180 + 80 = 260. #### 260"},
        {"question": "Find the smallest positive integer x such that x mod 3 = 2, x mod 4 = 3, and x mod 5 = 4.",
         "answer": "By CRT, x = 59. #### 59"},
        {"question": "How many r's are in the word 'strawberry'?",
         "answer": "s-t-r-a-w-b-e-r-r-y. Three r's. #### 3"},
        {"question": "If 4 workers can build a wall in 6 days, how many days will it take 3 workers to build the same wall?",
         "answer": "4*6 = 24 worker-days. 24/3 = 8. #### 8"},
        {"question": "Sarah has 48 cookies. She gives 1/3 to her brother and 1/4 of the remaining to her friend. How many does she have left?",
         "answer": "Gives brother 48/3=16, left 32. Gives friend 32/4=8, left 24. #### 24"},
        {"question": "A rectangle's length is 3 times its width. If the perimeter is 64 cm, what is the area?",
         "answer": "2(3w+w)=64, 8w=64, w=8, l=24. Area=8*24=192. #### 192"},
        {"question": "Tom is twice as old as Jerry. In 5 years, Tom will be 1.5 times as old as Jerry. How old is Jerry now?",
         "answer": "T=2J. T+5=1.5(J+5). 2J+5=1.5J+7.5. 0.5J=2.5. J=5. #### 5"},
        {"question": "A car uses 8 liters of fuel per 100 km. How many liters are needed for a 350 km trip?",
         "answer": "8 * 350/100 = 28. #### 28"},
    ]


def load_gsm8k(n: int) -> List[Dict[str, str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = cast(List[Dict[str, str]], list(ds.select(range(min(n, len(ds))))))
        return [
            {"question": str(row["question"]), "answer": str(row["answer"])}
            for row in rows
        ]
    except Exception:
        return _builtin_problems()[:n]


def run_benchmark(backend, n: int = 10, max_rounds: int = 3, stream_injector=None):
    """Single-agent baseline vs explicit-delegation orchestration."""
    dataset = load_gsm8k(n)

    from cortex_core.async_delegate import AsyncDelegationManager

    print(f"\n{'='*70}")
    print(f"{C_BOLD}  GSM8K BENCHMARK  |  model={backend.model_id}  |  N={len(dataset)}")
    print(f"{'='*70}")

    # ── Single agent baseline ────────────────────────────────────────
    print(f"\n{C_BOLD}>>> SINGLE AGENT (System 1){C_RESET}")
    single_correct = 0
    single_time = 0.0
    for i, item in enumerate(dataset):
        q = item["question"]
        truth = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else None
        if not truth: continue

        t0 = time.perf_counter()
        out = backend.generate(
            [{"role": "system", "content": DIRECT_SYSTEM},
             {"role": "user", "content": q}],
            temperature=0.0, max_tokens=400,
        )
        dt = time.perf_counter() - t0
        single_time += dt
        pred = extract_answer(out)
        ok = answers_match(pred, truth) if pred else False
        single_correct += int(ok)
        status = f"{C_GREEN}PASS{C_RESET}" if ok else f"{C_RED}FAIL{C_RESET}"
        print(f"  [{status}] Q{i+1}: pred={pred}, truth={truth} ({dt:.1f}s)")

    single_acc = single_correct / max(len(dataset), 1)
    print(f"\n  System 1: {single_correct}/{len(dataset)} ({single_acc:.0%}) in {single_time:.1f}s")

    # ── Orchestrated delegation ──────────────────────────────────────
    print(f"\n{C_BOLD}>>> ORCHESTRATED MODE (explicit delegation, max {max_rounds} rounds){C_RESET}")
    delegation_mgr = AsyncDelegationManager(
        stream_injector=stream_injector,
        backend=backend,
        max_workers=4,
        device=getattr(backend, 'device', 'cpu'),
    )
    engine = OrchestratedReasoningEngine(
        backend,
        max_rounds=max_rounds,
        stream_injector=stream_injector,
        delegation_mgr=delegation_mgr,
        verbose=False,
    )
    orch_correct = 0
    orch_time = 0.0
    orch_rounds_total = 0
    orch_delegations_total = 0

    for i, item in enumerate(dataset):
        q = item["question"]
        truth = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else None
        if not truth: continue

        result = engine.solve(q)
        orch_time += result["elapsed"]
        orch_rounds_total += result["rounds"]
        orch_delegations_total += result["delegations"]

        ok = answers_match(result["answer"], truth) if result["answer"] else False
        orch_correct += int(ok)
        status = f"{C_GREEN}PASS{C_RESET}" if ok else f"{C_RED}FAIL{C_RESET}"
        print(
            f"  [{status}] Q{i+1}: orch={result['answer']}, truth={truth} "
            f"({result['status']}, rounds={result['rounds']}, "
            f"delegations={result['delegations']}, {result['elapsed']:.1f}s)"
        )

    delegation_mgr.shutdown()

    n_total = max(len(dataset), 1)
    orch_acc = orch_correct / n_total
    delta = orch_acc - single_acc
    sign = "+" if delta >= 0 else ""

    print(f"\n{'='*70}")
    print(f"{C_BOLD}  RESULTS{C_RESET}")
    print(f"{'─'*70}")
    print(f"  {'Mode':<28} {'Accuracy':>10} {'Time':>10} {'Deleg/Q':>10}")
    print(f"  {'─'*58}")
    print(f"  {'System 1 (single)':<28} {single_acc:>9.0%} {single_time:>9.1f}s {'0':>10}")
    print(f"  {'Orchestrated delegation':<28} {orch_acc:>9.0%} {orch_time:>9.1f}s "
          f"{orch_delegations_total/n_total:>9.1f}")
    print(f"  {'─'*58}")
    print(f"  {'Improvement':<28} {sign}{delta:>8.0%}")
    print(f"  {'Avg rounds/question':<28} {orch_rounds_total/n_total:>9.1f}")
    print(f"{'='*70}\n")

    return {
        "single_acc": single_acc,
        "s2_acc": orch_acc,
        "delta": delta,
        "single_time": single_time,
        "s2_time": orch_time,
    }


# ======================================================================
# Stream Injection factory
# ======================================================================

def build_stream_injector(backend):
    """
    Build a StreamInjector from a LocalBackend.
    Returns None if backend doesn't expose model internals (e.g. API).
    """
    if not isinstance(backend, LocalBackend):
        print(f"{C_YELLOW}Stream injection requires local model — disabled for API backend.{C_RESET}")
        return None

    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.stream_inject import ClaimEncoder, StreamInjector

    dim = backend.hidden_dim
    device = backend.device

    synapse = TopologicalSynapse(dim=dim, max_injections=128, device=device)
    claim_encoder = ClaimEncoder(
        dim=dim,
        tokenizer=backend.tokenizer,
        embed_layer=backend.embed_layer,
        device=device,
    )

    # Try to get a CUDA stream pool if available
    stream_pool = None
    try:
        from cortex_engine import CUDAStreamPool
        stream_pool = CUDAStreamPool()
    except Exception:
        pass

    injector = StreamInjector(
        synapse=synapse,
        claim_encoder=claim_encoder,
        stream_pool=stream_pool,
        device=device,
    )
    print(f"{C_DIM}Stream injection enabled: dim={dim}, synapse landmarks=128{C_RESET}")
    return injector


# ======================================================================
# CLI
# ======================================================================

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML settings override (defaults to config/settings.yaml + optional settings.local.yaml)",
    )
    pre_args, _ = pre_parser.parse_known_args()
    settings = load_settings(pre_args.config)

    default_backend = str(get_setting(settings, "backends.default", "local")).lower()
    default_model = str(get_setting(settings, "backends.local.model", "Qwen/Qwen2.5-0.5B-Instruct"))
    default_device = str(get_setting(settings, "runtime.device", "cuda"))
    default_api_base = str(get_setting(settings, "backends.api.base_url", "https://api.openai.com/v1"))
    default_api_model = str(get_setting(settings, "backends.api.model", "gpt-4o-mini"))
    default_api_key = str(get_setting(settings, "backends.api.key", ""))
    default_rounds = int(get_setting(settings, "orchestrator.max_rounds", 3))
    default_workers = int(get_setting(settings, "orchestrator.worker_count", 4))
    default_verify_claims = bool(get_setting(settings, "orchestrator.verify_claims", False))
    default_llm_workers = bool(get_setting(settings, "orchestrator.llm_workers", False))
    default_stream_inject = bool(get_setting(settings, "orchestrator.stream_inject", False))
    default_async_delegate = bool(get_setting(settings, "orchestrator.async_delegate", False))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Warp-Cortex orchestrated reasoning runner"
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to solve (omit for benchmark mode)")
    parser.add_argument("--bench", type=int, default=0,
                        help="Run GSM8K benchmark with N problems")
    parser.add_argument("--backend", choices=["local", "api"], default=default_backend,
                        help="Execution backend. Defaults to settings file.")
    parser.add_argument("--model", default=default_model,
                        help="Local HF model (ignored if --api is set)")
    parser.add_argument("--device", default=default_device,
                        help="Device for local backend (default from settings)")
    parser.add_argument("--api", default=None,
                        help="API base URL ('openai', 'http://localhost:11434/v1', etc.)")
    parser.add_argument("--api-key", default=default_api_key,
                        help="API key (or set in settings / OPENAI_API_KEY)")
    parser.add_argument("--api-model", default=default_api_model,
                        help="Model name for API backend")
    parser.add_argument("--rounds", type=int, default=default_rounds,
                        help="Max model turns for delegation/review loops")
    parser.add_argument("--refines", dest="rounds", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="Max concurrent worker tasks for async delegation")
    parser.add_argument("--verify-claims", dest="verify_claims", action="store_true",
                        help="Enable optional arithmetic claim review after each turn")
    parser.add_argument("--no-verify-claims", dest="verify_claims", action="store_false",
                        help="Disable optional arithmetic claim review")
    parser.add_argument("--llm-workers", dest="llm_workers", action="store_true",
                        help="Use LLM workers instead of local eval during optional claim review")
    parser.add_argument("--no-llm-workers", dest="llm_workers", action="store_false",
                        help="Disable LLM workers during optional claim review")
    parser.add_argument("--stream-inject", dest="stream_inject", action="store_true",
                        help="Enable embedding-level stream injection into SynapseBuffer")
    parser.add_argument("--no-stream-inject", dest="stream_inject", action="store_false",
                        help="Disable embedding-level stream injection")
    parser.add_argument("--async-delegate", dest="async_delegate", action="store_true",
                        help="Enable model-driven worker delegation on single-question runs")
    parser.add_argument("--no-async-delegate", dest="async_delegate", action="store_false",
                        help="Disable model-driven worker delegation")
    parser.set_defaults(
        verify_claims=default_verify_claims,
        llm_workers=default_llm_workers,
        stream_inject=default_stream_inject,
        async_delegate=default_async_delegate,
    )
    args = parser.parse_args()

    if args.api is not None:
        args.backend = "api"

    cache_root = resolve_project_path(get_setting(settings, "paths.huggingface_cache"))

    # Build backend
    if args.backend == "api":
        base_url = args.api or default_api_base
        if base_url.lower() == "openai":
            base_url = "https://api.openai.com/v1"
        elif base_url.lower() == "anthropic":
            base_url = "https://api.anthropic.com/v1"
        backend = APIBackend(base_url=base_url, api_key=args.api_key, model=args.api_model)
    else:
        backend = LocalBackend(model_id=args.model, device=args.device, cache_root=cache_root)

    # Build stream injector if requested
    injector = None
    if args.stream_inject:
        injector = build_stream_injector(backend)

    # Build async delegation manager if requested
    delegation_mgr = None
    if args.async_delegate:
        from cortex_core.async_delegate import AsyncDelegationManager
        delegation_mgr = AsyncDelegationManager(
            stream_injector=injector,
            backend=backend,
            max_workers=args.workers,
            device=getattr(backend, 'device', 'cpu'),
        )
        print(f"{C_DIM}Async delegation enabled: {args.workers} workers, "
              f"injection={'ON' if injector else 'text-only'}{C_RESET}")

    if args.bench > 0:
        run_benchmark(backend, n=args.bench, max_rounds=args.rounds,
                      stream_injector=injector)
    elif args.question:
        engine = OrchestratedReasoningEngine(
            backend, max_rounds=args.rounds,
            verify_claims=args.verify_claims,
            use_llm_workers=args.llm_workers,
            stream_injector=injector,
            delegation_mgr=delegation_mgr,
        )
        result = engine.solve(args.question)
        if delegation_mgr is not None:
            delegation_mgr.shutdown()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
