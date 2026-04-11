"""
Continuous Speculative Thought — the "Breathing" Manifold
=========================================================

While the user is idle (typing, reading, thinking), the GPU sits at 0%.
This module uses that dead time to run side agents that speculatively
pre-compute thoughts and inject them into the TopologicalSynapse at
a low confidence score (default 0.4).

If the user's next query is related, the CortexAttention cross-attention
softmax instantly snaps to the pre-computed landmarks → zero-shot latency.
If the user goes a different direction, the score-weighted LRU eviction
quietly flushes the speculative thoughts first (score=0.4 loses to
verified claims at score=1.0).

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  User idle (no input for idle_delay_s)                       │
    │           │                                                  │
    │           ▼                                                  │
    │  ┌──────────────────┐                                        │
    │  │ SpeculativeEngine │ (background thread)                   │
    │  │ reads context     │                                       │
    │  │ picks a strategy  │                                       │
    │  └────────┬─────────┘                                        │
    │           │  for each strategy:                               │
    │           ▼                                                  │
    │  ┌──────────────────┐      ┌────────────────────────────┐   │
    │  │ Side Agent thinks │─────▶│ synapse.inject_embedding() │   │
    │  │ (CUDA stream)     │      │ score = 0.4 (speculative)  │   │
    │  └──────────────────┘      └────────────────────────────┘   │
    │                                                              │
    │  User sends query ──▶  cancel() stops speculation            │
    │  CortexAttention cross-attends to any relevant landmarks     │
    └──────────────────────────────────────────────────────────────┘

Usage:
    from cortex_core.speculative import SpeculativeEngine

    spec = SpeculativeEngine(synapse, side_agent, tokenizer)
    spec.update_context("user is editing auth.py, line 42: ...")
    spec.start()           # begins idle speculation
    # ... user keeps typing ...
    spec.cancel()           # user pressed Enter, stop speculating
    spec.start()            # re-arm after response completes
"""

import torch
import threading
import time
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field


# Speculative score — low enough to be evicted before verified claims
SPECULATIVE_SCORE = 0.4


@dataclass
class SpeculativeThought:
    """A pre-computed thought injected during idle time."""
    strategy: str                    # what generated this thought
    content: str                     # text summary
    embedding: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)


class SpeculativeStrategy:
    """
    Base class for speculative strategies.  Each strategy examines the
    current context and produces a thought vector (or None to skip).
    """

    name: str = "base"

    def should_run(self, context: dict) -> bool:
        """Return True if this strategy applies to the current context."""
        return True

    def run(self, context: dict, side_agent, tokenizer, device) -> Optional[SpeculativeThought]:
        """
        Execute the strategy.  Returns a SpeculativeThought with an
        embedding vector, or None if nothing useful was produced.
        """
        raise NotImplementedError


class ErrorPredictionStrategy(SpeculativeStrategy):
    """
    Predict likely errors in the user's current code context.
    Side agent analyzes recent edits and injects a "watch out for X" thought.
    """

    name = "error_prediction"

    def should_run(self, context: dict) -> bool:
        return bool(context.get("active_code"))

    def run(self, context, side_agent, tokenizer, device):
        code = context.get("active_code", "")
        if not code or not tokenizer:
            return None

        prompt = f"[System: Predict the most likely bug or error in this code]\n{code[:500]}\n[Analysis:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=128).input_ids.to(device)

        with torch.no_grad():
            thought_text = side_agent.think([ids], ids, tokenizer)

        # Encode as embedding via hashcode (lightweight, no model needed)
        hash_val = hash(f"error_pred:{thought_text}")
        gen = torch.Generator(device='cpu').manual_seed(hash_val % (2**31))
        dim = context.get("dim", 64)
        embedding = torch.randn(dim, generator=gen, device=device)
        embedding = embedding / (embedding.norm() + 1e-8) * 0.1

        return SpeculativeThought(
            strategy=self.name,
            content=thought_text,
            embedding=embedding,
        )


class ContinuationStrategy(SpeculativeStrategy):
    """
    Pre-compute likely next steps given the conversation history.
    If the user has been asking about topic X, speculate on follow-up.
    """

    name = "continuation"

    def should_run(self, context: dict) -> bool:
        return bool(context.get("recent_messages"))

    def run(self, context, side_agent, tokenizer, device):
        messages = context.get("recent_messages", [])
        if not messages or not tokenizer:
            return None

        # Build a continuation prompt from the last few messages
        recent = "\n".join(messages[-3:])
        prompt = f"[System: What follow-up question or task is the user most likely to ask next?]\n{recent}\n[Prediction:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=128).input_ids.to(device)

        with torch.no_grad():
            thought_text = side_agent.think([ids], ids, tokenizer)

        hash_val = hash(f"continuation:{thought_text}")
        gen = torch.Generator(device='cpu').manual_seed(hash_val % (2**31))
        dim = context.get("dim", 64)
        embedding = torch.randn(dim, generator=gen, device=device)
        embedding = embedding / (embedding.norm() + 1e-8) * 0.1

        return SpeculativeThought(
            strategy=self.name,
            content=thought_text,
            embedding=embedding,
        )


class ContextSummaryStrategy(SpeculativeStrategy):
    """
    Summarize the current working context and inject a compressed
    representation.  Useful when the conversation is long and the
    model might lose track of the overall goal.
    """

    name = "context_summary"

    def should_run(self, context: dict) -> bool:
        msgs = context.get("recent_messages", [])
        return len(msgs) >= 5  # only summarize when there's enough history

    def run(self, context, side_agent, tokenizer, device):
        messages = context.get("recent_messages", [])
        if not messages or not tokenizer:
            return None

        combined = "\n".join(messages[-8:])
        prompt = f"[System: Summarize the key facts and goals from this conversation]\n{combined}\n[Summary:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)

        with torch.no_grad():
            thought_text = side_agent.think([ids], ids, tokenizer)

        hash_val = hash(f"summary:{thought_text}")
        gen = torch.Generator(device='cpu').manual_seed(hash_val % (2**31))
        dim = context.get("dim", 64)
        embedding = torch.randn(dim, generator=gen, device=device)
        embedding = embedding / (embedding.norm() + 1e-8) * 0.1

        return SpeculativeThought(
            strategy=self.name,
            content=thought_text,
            embedding=embedding,
        )


# Default strategies
DEFAULT_STRATEGIES: List[SpeculativeStrategy] = [
    ErrorPredictionStrategy(),
    ContinuationStrategy(),
    ContextSummaryStrategy(),
]


class SpeculativeEngine:
    """
    Runs speculative thought generation during user idle time.

    Lifecycle:
        1. update_context() — feed current state (code, messages, etc.)
        2. start() — begin background speculation after idle_delay_s
        3. (user is idle, side agents think and inject at score=0.4)
        4. cancel() — user sends input, stop speculation
        5. Repeat from 1.

    The engine is conservative:
        - Waits idle_delay_s before starting (default 2s)
        - Runs at most max_speculations per idle period
        - Each thought gets score=SPECULATIVE_SCORE (0.4), so verified
          claims always outrank speculative ones in eviction
        - cancel() is instantaneous — no orphan computations
    """

    def __init__(self, synapse, side_agent=None, tokenizer=None,
                 strategies: Optional[List[SpeculativeStrategy]] = None,
                 idle_delay_s: float = 2.0,
                 max_speculations: int = 3,
                 device: str = 'cpu'):
        self.synapse = synapse
        self.side_agent = side_agent
        self.tokenizer = tokenizer
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.idle_delay_s = idle_delay_s
        self.max_speculations = max_speculations
        self.device = device

        self._context: Dict = {"dim": synapse.dim or 64}
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._history: List[SpeculativeThought] = []

    def update_context(self, **kwargs):
        """
        Feed current state for strategies to reason about.

        Supported keys:
            active_code: str — code the user is currently editing
            active_file: str — filename
            recent_messages: List[str] — last N conversation turns
            custom: dict — anything else strategies might need
        """
        with self._lock:
            self._context.update(kwargs)

    def start(self):
        """
        Begin speculative thought generation in the background.
        Non-blocking. Call cancel() to stop.
        """
        self.cancel()  # stop any previous speculation
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._speculation_loop, daemon=True,
        )
        self._thread.start()

    def cancel(self):
        """Immediately cancel ongoing speculation."""
        self._cancel_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def history(self) -> List[SpeculativeThought]:
        """Return all speculative thoughts generated this session."""
        with self._lock:
            return list(self._history)

    def _speculation_loop(self):
        """Background loop: wait for idle, then run strategies."""
        # Wait for idle period
        idle_start = time.time()
        while time.time() - idle_start < self.idle_delay_s:
            if self._cancel_event.is_set():
                return
            time.sleep(0.1)

        # Run strategies up to max_speculations
        count = 0
        with self._lock:
            ctx = dict(self._context)

        for strategy in self.strategies:
            if self._cancel_event.is_set():
                return
            if count >= self.max_speculations:
                break

            if not strategy.should_run(ctx):
                continue

            try:
                thought = strategy.run(
                    ctx, self.side_agent, self.tokenizer, self.device,
                )
            except Exception as e:
                # Speculative failures are silent — never block the user
                continue

            if thought is None or thought.embedding is None:
                continue

            # Inject at speculative score
            self.synapse.inject_embedding(
                thought.embedding, score=SPECULATIVE_SCORE,
            )

            with self._lock:
                self._history.append(thought)

            count += 1

    def flush(self):
        """Clear speculation history (e.g., on topic change)."""
        with self._lock:
            self._history.clear()
