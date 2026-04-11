"""
Adaptive Generation Engine — Dual-mode delegation.

Two modes of delegation that share the same worker infrastructure:

  Mode A — AWARE (default):
      The model is told about [DELEGATE:...] syntax in its system prompt.
      It chooses when and what to delegate. The engine scans its output
      for delegation tags and dispatches to AsyncDelegationManager.
      Results are fed back as continuation text.

  Mode B — SILENT:
      The model has no idea delegation exists. The EntropyRouter monitors
      every token's attention entropy inline. When the model's internal
      uncertainty spikes beyond its running baseline, the engine pauses
      generation, auto-classifies the needed expert from the hidden state,
      and dispatches to the same worker pool. The result is injected back
      into the generation context.

Both modes use:
  - Token-by-token generation with KV cache (output_attentions=True)
  - EntropyRouter for live entropy monitoring (diagnostics in A, trigger in B)
  - AsyncDelegationManager for actual worker dispatch
  - CortexRouter for intent classification (semantic + regex)
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn.functional as F

from .entropy_router import EntropyRouter, EntropySignal
from .async_delegate import (
    AsyncDelegationManager,
    DelegationRequest,
    DelegationResult,
    detect_delegation_requests,
)
from .cortex_router import CortexRouter
from .turbo_quant import TurboQuantCache


# ── Configuration ────────────────────────────────────────────────────

class DelegationMode(Enum):
    AWARE = "aware"      # Model knows about delegation, emits [DELEGATE:...] tags
    SILENT = "silent"    # Model is unaware, entropy-based auto-dispatch
    COUNCIL = "council"  # Multiple agents deliberate in parallel, best-of-N


AWARE_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step.\n\n"
    "If a subtask genuinely needs external help, you may emit a delegation "
    "block instead of guessing:\n"
    "  [DELEGATE:math] expression [/DELEGATE]\n"
    "  [DELEGATE:code] python_code [/DELEGATE]\n"
    "  [DELEGATE:llm] focused question [/DELEGATE]\n"
    "  [DELEGATE:search] factual query to look up [/DELEGATE]\n\n"
    "Rules:\n"
    "- Only delegate narrow subtasks, not the whole problem.\n"
    "- Do not invent worker outputs. If you delegate, STOP and wait.\n"
    "- After receiving worker results, continue solving.\n"
    "- If no worker is needed, solve directly."
)

SILENT_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step."
)

COUNCIL_SYSTEM_PROMPT = (
    "You are one of several independent reasoning agents working on the same "
    "problem. Think step by step. Show your work clearly and end with your "
    "final answer on the last line prefixed with ####."
)


@dataclass
class GenerationResult:
    """Full result of an adaptive generation run."""
    text: str
    tokens: List[str]
    delegation_events: List[Dict]      # Each: {step, token, signal, mode, ...}
    worker_results: List[DelegationResult]
    entropy_signals: List[EntropySignal]
    elapsed: float
    mode: DelegationMode
    total_tokens: int
    delegation_count: int
    kv_compression_ratio: float = 1.0  # TurboQuant compression ratio (1.0 = no compression)
    council_responses: Optional[List[str]] = None  # All council agent responses (COUNCIL mode)


# ── Intent mapper for entropy-based auto-dispatch ────────────────────

# Maps CortexRouter intent IDs → async_delegate expert kinds
_INTENT_TO_EXPERT = {
    "code": "code",
    "check": "math",       # fact-check → math evaluator
    "search": "search",    # search → web search worker
    "summarise": "llm",
    "delegate": "llm",     # generic delegation → LLM sub-thinker
}


def _classify_delegation_kind(
    hidden_state: Optional[torch.Tensor],
    partial_text: str,
    cortex_router: Optional[CortexRouter],
) -> str:
    """
    Determine what kind of expert to call when entropy fires.

    Strategy:
    1. If CortexRouter is bootstrapped, classify from hidden state
    2. Fallback: heuristic scan of partial text
    3. Default: "llm" (sub-thinker handles anything)
    """
    # Semantic classification
    if hidden_state is not None and cortex_router is not None:
        task, confidence = cortex_router.classify_hidden(hidden_state)
        if task is not None and confidence > 0.4:
            for intent_id, expert in _INTENT_TO_EXPERT.items():
                if intent_id in (task or "").lower():
                    return expert

    # Heuristic: look at recent text
    recent = partial_text[-300:].lower()
    if any(kw in recent for kw in ["calculate", "compute", "=", "sum", "product",
                                     "multiply", "divide", "add", "subtract"]):
        return "math"
    if any(kw in recent for kw in ["```", "def ", "import ", "print(", "class "]):
        return "code"

    return "llm"


def _build_delegation_payload(
    kind: str,
    partial_text: str,
    question: str,
) -> Optional[str]:
    """
    Build the payload string sent to the worker.
    Varies by expert kind.
    """
    recent = partial_text[-500:]

    if kind == "math":
        # Try to extract the expression the model was working on
        import re
        # Look for unfinished arithmetic (must be a clean expression)
        expressions = re.findall(r'[\d]+(?:\.\d+)?\s*[+\-*/×÷]\s*[\d]+(?:\.\d+)?(?:\s*[+\-*/×÷]\s*[\d]+(?:\.\d+)?)*',
                                 recent)
        if expressions:
            return expressions[-1].strip()
        # No clean expression found — escalate to LLM worker instead of
        # sending prose to the math evaluator (which will reject it)
        return None  # caller checks for None → falls back to LLM

    if kind == "code":
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', recent, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        return recent[-300:]

    # LLM: send the question + partial reasoning for the sub-thinker to help
    return (
        f"The following question is being solved:\n{question}\n\n"
        f"Current partial reasoning:\n{recent}\n\n"
        f"The solver appears stuck. Provide a helpful hint or verify "
        f"the reasoning so far. Be concise."
    )


# ── Main Engine ──────────────────────────────────────────────────────

class AdaptiveGenerator:
    """
    Token-by-token generation with dual-mode delegation.

    Usage:
        engine = AdaptiveGenerator(model, tokenizer, mode=DelegationMode.AWARE)
        result = engine.generate("What is the area of a 5x7 rectangle?")

        # Switch to silent mode:
        engine = AdaptiveGenerator(model, tokenizer, mode=DelegationMode.SILENT)
        result = engine.generate("12 coins puzzle...")
    """

    def __init__(
        self,
        model,
        tokenizer,
        mode: DelegationMode = DelegationMode.AWARE,
        # Entropy router config
        spread_z_threshold: float = 2.0,
        logit_z_threshold: float = 2.5,
        compound_mode: bool = False,
        warmup_steps: int = 8,
        # KV cache compression (TurboQuant)
        turbo_quant_bits: int = 4,           # 2, 3, or 4 — lower = more compression
        turbo_quant_enabled: bool = True,    # False = no KV compression
        turbo_quant_interval: int = 64,      # compress every N steps
        # Delegation manager
        delegation_mgr: Optional[AsyncDelegationManager] = None,
        # Semantic router (for silent mode intent classification)
        cortex_router: Optional[CortexRouter] = None,
        # LLM backend for LLM-type workers (default for all expert kinds)
        llm_backend: Optional[Any] = None,
        # Per-expert backend overrides: {"math": BitNetBackend, "llm": hf_backend}
        expert_backends: Optional[Dict[str, Any]] = None,
        # Council mode config
        council_size: int = 3,               # Number of agents in COUNCIL mode
        # Generation defaults
        max_tokens: int = 300,
        temperature: float = 0.0,
        max_delegations: int = 5,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_delegations = max_delegations
        self.council_size = council_size
        self.verbose = verbose

        self.device = device or str(next(model.parameters()).device)

        # TurboQuant KV cache compression
        self.turbo_quant_bits = turbo_quant_bits
        self.turbo_quant_enabled = turbo_quant_enabled
        self.turbo_quant_interval = turbo_quant_interval

        # Entropy router — always active (diagnostics in aware, trigger in silent)
        self.entropy_router = EntropyRouter(
            spread_z_threshold=spread_z_threshold,
            logit_z_threshold=logit_z_threshold,
            compound_mode=compound_mode,
            warmup_steps=warmup_steps,
        )

        # Delegation manager — shared worker pool for both modes
        if delegation_mgr is not None:
            self.delegation_mgr = delegation_mgr
        else:
            self.delegation_mgr = AsyncDelegationManager(
                backend=llm_backend,
                expert_backends=expert_backends or {},
                max_workers=max(4, council_size),
                device=self.device,
            )

        # Semantic router for intent classification (silent mode)
        self.cortex_router = cortex_router

        self._eos_id = tokenizer.eos_token_id

    # ── Public API ───────────────────────────────────────────────

    def generate(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate a response with adaptive delegation.

        The system prompt defaults based on mode:
          - AWARE: includes delegation syntax instructions
          - SILENT: plain reasoning prompt (model has no idea)
        """
        self.entropy_router.reset()
        max_tok = max_tokens or self.max_tokens

        if system_prompt is None:
            if self.mode == DelegationMode.AWARE:
                system_prompt = AWARE_SYSTEM_PROMPT
            elif self.mode == DelegationMode.COUNCIL:
                system_prompt = COUNCIL_SYSTEM_PROMPT
            else:
                system_prompt = SILENT_SYSTEM_PROMPT

        if self.mode == DelegationMode.AWARE:
            return self._generate_aware(question, system_prompt, max_tok)
        elif self.mode == DelegationMode.COUNCIL:
            return self._generate_council(question, system_prompt, max_tok)
        else:
            return self._generate_silent(question, system_prompt, max_tok)

    # ── Mode A: AWARE — model emits [DELEGATE:...] tags ─────────

    def _generate_aware(self, question: str, system_prompt: str,
                        max_tokens: int) -> GenerationResult:
        t0 = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        all_tokens: List[str] = []
        all_events: List[Dict] = []
        all_worker_results: List[DelegationResult] = []
        all_signals: List[EntropySignal] = []
        delegation_count = 0
        total_generated = 0

        # Outer loop: generate → detect tags → dispatch → inject results → continue
        for turn in range(self.max_delegations + 1):
            tokens, events, signals = self._token_loop(
                messages, max_tokens - total_generated,
            )

            all_tokens.extend(tokens)
            all_events.extend(events)
            all_signals.extend(signals)
            total_generated += len(tokens)

            # Check for delegation tags in the generated text
            generated_text = "".join(tokens)
            requests = detect_delegation_requests(generated_text)

            if not requests or delegation_count >= self.max_delegations:
                break

            # Dispatch all requests and wait for results
            delegation_count += len(requests)
            results = self._dispatch_and_wait(requests)
            all_worker_results.extend(results)

            # Feed results back as assistant + system continuation
            result_text = self._format_worker_results(results)
            messages.append({"role": "assistant", "content": generated_text})
            messages.append({"role": "system", "content": result_text})

            self._log(f"\n  [AWARE] {len(requests)} delegation(s) dispatched, "
                      f"results injected — continuing generation")

        full_text = "".join(all_tokens)
        # Strip any remaining delegation blocks from final text
        full_text = _strip_delegate_blocks(full_text)

        return GenerationResult(
            text=full_text,
            tokens=all_tokens,
            delegation_events=all_events,
            worker_results=all_worker_results,
            entropy_signals=all_signals,
            elapsed=time.time() - t0,
            mode=DelegationMode.AWARE,
            total_tokens=total_generated,
            delegation_count=delegation_count,
        )

    # ── Mode C: COUNCIL — parallel agents, majority vote ─────────

    def _generate_council(self, question: str, system_prompt: str,
                          max_tokens: int) -> GenerationResult:
        """
        Run N independent agents on the same question in parallel.
        Each agent generates with a slightly different temperature.
        Final answer chosen by majority vote over #### lines.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        t0 = time.time()
        temperatures = [
            0.3 + 0.2 * i for i in range(self.council_size)
        ]  # e.g. [0.3, 0.5, 0.7] for council_size=3

        def _run_agent(temp: float) -> str:
            """Single council agent — token-by-token generation."""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

            past_kv = None
            current_ids = input_ids
            gen_ids = []

            for step in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=current_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                past_kv = outputs.past_key_values
                logits = outputs.logits[0, -1, :]

                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits.argmax().unsqueeze(0)

                gen_ids.append(next_token.item())
                current_ids = next_token.unsqueeze(0)

                if next_token.item() == self._eos_id:
                    break

            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Run agents in parallel
        self._log(f"  [COUNCIL] Spawning {self.council_size} agents "
                  f"(temps={[round(t, 1) for t in temperatures]})")

        responses: List[str] = []
        with ThreadPoolExecutor(max_workers=self.council_size) as pool:
            futures = {pool.submit(_run_agent, t): i
                       for i, t in enumerate(temperatures)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    resp = future.result()
                    responses.append(resp)
                    self._log(f"  [COUNCIL] Agent {idx} finished "
                              f"({len(resp)} chars)")
                except Exception as e:
                    self._log(f"  [COUNCIL] Agent {idx} failed: {e}")
                    responses.append("")

        # Majority vote on #### final answer lines
        best = self._council_vote(responses)

        return GenerationResult(
            text=best,
            tokens=[],  # council doesn't track per-token
            delegation_events=[],
            worker_results=[],
            entropy_signals=[],
            elapsed=time.time() - t0,
            mode=DelegationMode.COUNCIL,
            total_tokens=sum(len(r.split()) for r in responses),
            delegation_count=0,
            council_responses=responses,
        )

    @staticmethod
    def _council_vote(responses: List[str]) -> str:
        """
        Pick the best response by majority-voting on the #### final answer.
        If no majority, return the longest response (most reasoning shown).
        """
        import re
        from collections import Counter

        answers = {}  # response_idx → extracted answer
        for i, resp in enumerate(responses):
            # Look for #### <answer> pattern (GSM8K-style)
            match = re.search(r'####\s*(.+)', resp)
            if match:
                answers[i] = match.group(1).strip().lower()

        if answers:
            counts = Counter(answers.values())
            winner_answer, winner_count = counts.most_common(1)[0]
            # Return the full response that produced the winning answer
            for idx, ans in answers.items():
                if ans == winner_answer:
                    return responses[idx]

        # No #### answers found — return longest response
        return max(responses, key=len) if responses else ""

    # ── TurboQuant KV cache helpers ──────────────────────────────

    def _maybe_compress_kv(self, past_kv, step: int):
        """
        Periodically compress the KV cache with TurboQuant.
        Returns (past_kv, compression_ratio).
        """
        if (not self.turbo_quant_enabled
                or past_kv is None
                or step == 0
                or step % self.turbo_quant_interval != 0):
            return past_kv, 1.0

        # Convert HF DynamicCache → list-of-tuples for TurboQuant
        kv_tuples = self._extract_kv_tuples(past_kv)
        if not kv_tuples:
            return past_kv, 1.0

        orig_bytes = sum(
            k.nelement() * k.element_size() + v.nelement() * v.element_size()
            for k, v in kv_tuples
        )

        tq = TurboQuantCache(
            bits=self.turbo_quant_bits,
            device=self.device,
        )
        tq.compress(kv_tuples)
        ratio = tq.compression_ratio(orig_bytes)

        # Decompress back into the format the model expects
        decompressed = tq.decompress()
        past_kv = self._rebuild_kv_cache(decompressed, past_kv)

        self._log(
            f"  [TurboQuant] KV compressed {ratio:.1f}× "
            f"({self.turbo_quant_bits}-bit + QJL) at step {step}",
            dim=True,
        )
        return past_kv, ratio

    @staticmethod
    def _extract_kv_tuples(past_kv):
        """Extract list of (K, V) tensor tuples from HF cache object."""
        # DynamicCache: supports len() and __getitem__ → (K, V) tuples
        if hasattr(past_kv, '__getitem__') and hasattr(past_kv, '__len__'):
            try:
                n = len(past_kv)
                if n > 0:
                    first = past_kv[0]
                    if isinstance(first, (tuple, list)) and len(first) == 2:
                        return [past_kv[i] for i in range(n)]
            except (IndexError, TypeError):
                pass
        # Legacy tuple-of-tuples format
        if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
            if isinstance(past_kv[0], (tuple, list)) and len(past_kv[0]) == 2:
                return [(k, v) for k, v in past_kv]
        return []

    @staticmethod
    def _rebuild_kv_cache(decompressed, original_cache):
        """
        Build a new DynamicCache from decompressed (K, V) tuples.
        Uses from_legacy_cache if available (HF DynamicCache),
        otherwise returns a plain tuple.
        """
        from transformers.cache_utils import DynamicCache
        if isinstance(original_cache, DynamicCache):
            return DynamicCache.from_legacy_cache(tuple(decompressed))
        return tuple(decompressed)

    # ── Mode B: SILENT — entropy-triggered auto-dispatch ─────────

    def _generate_silent(self, question: str, system_prompt: str,
                         max_tokens: int) -> GenerationResult:
        t0 = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        all_tokens: List[str] = []
        all_events: List[Dict] = []
        all_worker_results: List[DelegationResult] = []
        all_signals: List[EntropySignal] = []
        delegation_count = 0
        last_compression_ratio = 1.0

        past_kv = None
        current_ids = input_ids  # first step: full prefill
        generated_token_ids = []  # track generated IDs for final decode

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )

            past_kv = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            signal = self.entropy_router.step(outputs.attentions, logits)
            all_signals.append(signal)

            # Next token
            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax().unsqueeze(0)

            token_str = self.tokenizer.decode(next_token, skip_special_tokens=False)
            all_tokens.append(token_str)
            generated_token_ids.append(next_token.item())

            # Next step: only the new token (KV cache handles history)
            current_ids = next_token.unsqueeze(0)

            # ── TurboQuant periodic compression ──
            past_kv, ratio = self._maybe_compress_kv(past_kv, step)
            if ratio > 1.0:
                last_compression_ratio = ratio

            # ── Entropy-based delegation check ──
            if (signal.should_delegate
                    and delegation_count < self.max_delegations):
                delegation_count += 1
                partial_text = "".join(all_tokens)

                # Classify intent from hidden state
                hidden_state = outputs.hidden_states[-1][0, -1, :]
                kind = _classify_delegation_kind(
                    hidden_state, partial_text, self.cortex_router,
                )

                payload = _build_delegation_payload(kind, partial_text, question)

                # If payload is None (e.g. math with no clean expression),
                # fall back to LLM worker which can handle prose context
                if payload is None:
                    kind = "llm"
                    payload = _build_delegation_payload(kind, partial_text, question)
                assert payload is not None

                event = {
                    "step": step,
                    "token": token_str,
                    "signal": signal,
                    "mode": "silent",
                    "expert_kind": kind,
                    "confidence": signal.confidence,
                }
                all_events.append(event)

                self._log(
                    f"  [SILENT] t={step} entropy spike "
                    f"(spread_z={signal.spread_z_score:+.2f}, "
                    f"logit_z={signal.logit_z_score:+.2f}) "
                    f"→ auto-dispatch to '{kind}'"
                )

                # Dispatch and wait
                request = DelegationRequest(
                    task_id=f"entropy_{step}",
                    expert_kind=kind,
                    payload=payload,
                )
                results = self._dispatch_and_wait([request])
                all_worker_results.extend(results)

                # Inject result into the generation context
                if results and results[0].success:
                    injection = f" [{results[0].output}]"
                    inject_ids = self.tokenizer(
                        injection, return_tensors="pt",
                    ).input_ids.to(self.device)
                    # Feed injection through model to update KV cache
                    with torch.no_grad():
                        inject_out = self.model(
                            input_ids=inject_ids,
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    past_kv = inject_out.past_key_values
                    for tid in inject_ids[0].tolist():
                        generated_token_ids.append(tid)
                    self._log(f"  [SILENT] Injected: {injection[:80]}")

            # Logging (sparse)
            if self.verbose and (step < 10 or signal.should_delegate or step % 30 == 0):
                marker = " ◆DELEGATE" if signal.should_delegate else ""
                self._log(
                    f"  t={step:>3} '{token_str}' "
                    f"spread_z={signal.spread_z_score:+.2f} "
                    f"logit_z={signal.logit_z_score:+.2f}"
                    f"{marker}",
                    dim=not signal.should_delegate,
                )

            if next_token.item() == self._eos_id:
                break

        full_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        return GenerationResult(
            text=full_text,
            tokens=all_tokens,
            delegation_events=all_events,
            worker_results=all_worker_results,
            entropy_signals=all_signals,
            elapsed=time.time() - t0,
            mode=DelegationMode.SILENT,
            total_tokens=len(all_tokens),
            delegation_count=delegation_count,
            kv_compression_ratio=last_compression_ratio,
        )

    # ── Shared generation loop (for aware mode) ─────────────────

    def _token_loop(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> Tuple[List[str], List[Dict], List[EntropySignal]]:
        """
        Token-by-token generation with KV caching and TurboQuant compression.
        Returns (tokens, events, signals).
        Stops at EOS, max_tokens, or if a [DELEGATE:...] block is
        detected in the accumulated text (aware mode only).
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        tokens: List[str] = []
        events: List[Dict] = []
        signals: List[EntropySignal] = []

        past_kv = None
        current_ids = input_ids  # first step: full prefill

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )

            past_kv = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            signal = self.entropy_router.step(outputs.attentions, logits)
            signals.append(signal)

            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax().unsqueeze(0)

            token_str = self.tokenizer.decode(next_token, skip_special_tokens=False)
            tokens.append(token_str)

            # Next step will only feed the new token
            current_ids = next_token.unsqueeze(0)

            # TurboQuant periodic compression
            past_kv, _ = self._maybe_compress_kv(past_kv, step)

            # Log entropy (diagnostic even in aware mode)
            if signal.should_delegate:
                events.append({
                    "step": step, "token": token_str,
                    "signal": signal, "mode": "aware_entropy_diagnostic",
                    "confidence": signal.confidence,
                })

            if self.verbose and (step < 10 or signal.should_delegate or step % 30 == 0):
                marker = " ◆" if signal.should_delegate else ""
                self._log(
                    f"  t={step:>3} '{token_str}' "
                    f"spread_z={signal.spread_z_score:+.2f} "
                    f"logit_z={signal.logit_z_score:+.2f}"
                    f"{marker}",
                    dim=not signal.should_delegate,
                )

            if next_token.item() == self._eos_id:
                break

            # In aware mode, check if we've accumulated a complete delegation block
            if self.mode == DelegationMode.AWARE and "[/DELEGATE]" in "".join(tokens):
                break

        return tokens, events, signals

    # ── Worker dispatch ──────────────────────────────────────────

    def _dispatch_and_wait(
        self,
        requests: List[DelegationRequest],
        timeout: float = 15.0,
    ) -> List[DelegationResult]:
        """Dispatch requests to AsyncDelegationManager and wait for results."""
        task_ids = self.delegation_mgr.dispatch_batch(requests)
        self.delegation_mgr.wait_all(timeout=timeout)

        results = []
        with self.delegation_mgr._lock:
            for result in self.delegation_mgr._results:
                if result.task_id in task_ids:
                    results.append(result)
        return results

    def _format_worker_results(self, results: List[DelegationResult]) -> str:
        """Format worker results for injection back into conversation."""
        lines = ["Worker results:"]
        for r in results:
            status = "SUCCESS" if r.success else "FAIL"
            lines.append(f"[{r.expert_kind}] {status}: {r.output or r.error}")
        lines.append("\nContinue solving using these results.")
        return "\n".join(lines)

    # ── Logging ──────────────────────────────────────────────────

    def _log(self, msg: str, dim: bool = False):
        if self.verbose:
            if dim:
                print(f"\033[2m{msg}\033[0m")
            else:
                print(msg)


# ── Utility ──────────────────────────────────────────────────────────

import re

_DELEGATE_BLOCK_RE = re.compile(
    r'\[DELEGATE:\w+(?::\w+)?\]\s*.*?\s*\[/DELEGATE\]',
    re.DOTALL,
)


def _strip_delegate_blocks(text: str) -> str:
    """Remove [DELEGATE:...] blocks from final output text."""
    cleaned = _DELEGATE_BLOCK_RE.sub('', text)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()
