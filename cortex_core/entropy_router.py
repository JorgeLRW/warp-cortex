"""
EntropyRouter — Inline attention-entropy monitor for adaptive delegation.

Hooks into every forward pass, computes per-layer entropy from attention
weights that already exist in GPU memory, and emits delegation signals
when the model's internal uncertainty exceeds its own running baseline.

Zero extra forward passes. Microseconds of overhead per step.

Usage:
    router = EntropyRouter()
    # During token-by-token generation:
    signal = router.step(outputs.attentions, outputs.logits[:, -1, :])
    if signal.should_delegate:
        # model is confused — trigger delegation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class EntropySignal:
    """Result from a single step of entropy monitoring."""
    step: int

    # Per-layer stats (last-token attention only)
    layer_head_spread: list[float]       # max-min entropy across heads, per layer
    layer_norm_entropy: list[float]      # mean normalized entropy, per layer

    # Logit entropy
    logit_entropy: float                 # absolute Shannon entropy of output dist
    logit_norm_entropy: float            # normalized to [0, 1]

    # Aggregate signals
    max_head_spread: float               # highest head spread across all layers
    max_spread_layer: int                # which layer has the most head disagreement
    mean_head_spread: float              # average spread across all layers

    # Deviation from running baseline
    spread_z_score: float                # how many std above the running mean
    logit_z_score: float                 # logit entropy z-score

    # Decision
    should_delegate: bool                # compound threshold crossed
    confidence: float                    # 0-1, how sure we are about delegation

    @property
    def difficulty_estimate(self) -> str:
        """Human-readable difficulty bucket."""
        if self.confidence < 0.2:
            return "trivial"
        elif self.confidence < 0.4:
            return "easy"
        elif self.confidence < 0.6:
            return "medium"
        elif self.confidence < 0.8:
            return "hard"
        else:
            return "very_hard"


@dataclass
class _RunningStats:
    """Welford's online algorithm for mean/variance."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return max(math.sqrt(self.variance), 1e-8)

    def z_score(self, value: float) -> float:
        if self.count < 5:
            return 0.0  # not enough data yet
        return (value - self.mean) / self.std


# ── Main router ──────────────────────────────────────────────────────

class EntropyRouter:
    """
    Inline attention-entropy monitor.

    Attaches to the forward pass outputs (attentions + logits) that
    already exist after model(..., output_attentions=True). Tracks
    running statistics and fires delegation signals when the model's
    own uncertainty deviates from its baseline.

    Args:
        spread_z_threshold: How many std above mean head-spread triggers
            delegation. Default 2.0 (adaptive, no hardcoded entropy value).
        logit_z_threshold: Z-score threshold for logit entropy spikes.
        compound_mode: If True, requires BOTH spread and logit signals.
            If False, either signal alone can trigger.
        warmup_steps: Number of steps before the router starts firing.
            During warmup it only collects baseline stats.
        ema_alpha: Exponential moving average decay for recent-window stats.
            0 = use all history equally, 0.1 = weight recent 10x more.
    """

    def __init__(
        self,
        spread_z_threshold: float = 2.0,
        logit_z_threshold: float = 2.0,
        compound_mode: bool = False,
        warmup_steps: int = 10,
        ema_alpha: float = 0.0,
    ):
        self.spread_z_threshold = spread_z_threshold
        self.logit_z_threshold = logit_z_threshold
        self.compound_mode = compound_mode
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha

        # Running stats — one per tracked metric
        self._spread_stats = _RunningStats()
        self._logit_stats = _RunningStats()

        # EMA (optional fast-adapting baseline)
        self._ema_spread: Optional[float] = None
        self._ema_logit: Optional[float] = None

        self._step_count = 0
        self._history: list[EntropySignal] = []

    def reset(self):
        """Clear all state (e.g., between conversations)."""
        self._spread_stats = _RunningStats()
        self._logit_stats = _RunningStats()
        self._ema_spread = None
        self._ema_logit = None
        self._step_count = 0
        self._history.clear()

    # ── Core: single step ────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> EntropySignal:
        """
        Analyze one generation step.

        Args:
            attentions: Tuple of (batch, heads, q_len, k_len) per layer.
                        From model(..., output_attentions=True).attentions
            logits: (batch, vocab_size) or (vocab_size,) — next-token logits.

        Returns:
            EntropySignal with all metrics and delegation decision.
        """
        self._step_count += 1

        # ── Attention entropy ──
        layer_head_spread = []
        layer_norm_entropy = []

        if attentions is not None:
            for attn in attentions:
                spread, norm_ent = self._layer_entropy(attn)
                layer_head_spread.append(spread)
                layer_norm_entropy.append(norm_ent)

        max_spread = max(layer_head_spread) if layer_head_spread else 0.0
        max_spread_layer = (
            layer_head_spread.index(max_spread) if layer_head_spread else -1
        )
        mean_spread = (
            sum(layer_head_spread) / len(layer_head_spread)
            if layer_head_spread else 0.0
        )

        # ── Logit entropy ──
        logit_ent = 0.0
        logit_norm = 0.0
        if logits is not None:
            logit_ent, logit_norm = self._logit_entropy(logits)

        # ── Update running stats ──
        if attentions is not None and len(layer_head_spread) > 0:
            self._spread_stats.update(max_spread)
        if logits is not None:
            self._logit_stats.update(logit_ent)

        if self.ema_alpha > 0:
            if attentions is not None and len(layer_head_spread) > 0:
                if self._ema_spread is None:
                    self._ema_spread = max_spread
                else:
                    self._ema_spread = self.ema_alpha * max_spread + (1 - self.ema_alpha) * self._ema_spread
            if logits is not None:
                if self._ema_logit is None:
                    self._ema_logit = logit_ent
                else:
                    self._ema_logit = self.ema_alpha * logit_ent + (1 - self.ema_alpha) * self._ema_logit

        # ── Z-scores (deviation from baseline) ──
        spread_z = self._spread_stats.z_score(max_spread) if (attentions is not None and len(layer_head_spread) > 0) else 0.0
        logit_z = self._logit_stats.z_score(logit_ent) if logits is not None else 0.0

        # ── Delegation decision ──
        in_warmup = self._step_count <= self.warmup_steps
        spread_fires = spread_z > self.spread_z_threshold
        logit_fires = logit_z > self.logit_z_threshold

        if in_warmup:
            should_delegate = False
        elif attentions is None:
            # Attention-free mode: preserve FlashAttention / SDPA, route purely on logit uncertainty
            should_delegate = logit_fires
        elif self.compound_mode:
            should_delegate = spread_fires and logit_fires
        else:
            should_delegate = spread_fires or logit_fires

        # Confidence: how far above threshold (clamped 0-1)
        if attentions is None:
            max_z = logit_z
            threshold = self.logit_z_threshold
        else:
            max_z = max(spread_z, logit_z)
            threshold = max(self.spread_z_threshold, self.logit_z_threshold)
        confidence = min(max(max_z / (threshold * 2), 0.0), 1.0)

        signal = EntropySignal(
            step=self._step_count,
            layer_head_spread=layer_head_spread,
            layer_norm_entropy=layer_norm_entropy,
            logit_entropy=logit_ent,
            logit_norm_entropy=logit_norm,
            max_head_spread=max_spread,
            max_spread_layer=max_spread_layer,
            mean_head_spread=mean_spread,
            spread_z_score=spread_z,
            logit_z_score=logit_z,
            should_delegate=should_delegate,
            confidence=confidence,
        )

        self._history.append(signal)
        return signal

    # ── Batch convenience ────────────────────────────────────────

    def check_prefill(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> EntropySignal:
        """
        Check the prefill pass (before generation begins).
        Same as step() but doesn't count toward the running baseline —
        useful for an early "should we even try direct?" check.
        """
        # Temporarily save state
        saved = (
            self._step_count,
            _RunningStats(
                self._spread_stats.count,
                self._spread_stats.mean,
                self._spread_stats.m2,
            ),
            _RunningStats(
                self._logit_stats.count,
                self._logit_stats.mean,
                self._logit_stats.m2,
            ),
        )

        signal = self.step(attentions, logits)

        # Restore state (don't pollute baseline with prefill)
        self._step_count = saved[0]
        self._spread_stats = saved[1]
        self._logit_stats = saved[2]
        self._history.pop()

        return signal

    # ── Diagnostics ──────────────────────────────────────────────

    @property
    def baseline(self) -> dict:
        """Current running baseline stats."""
        return {
            "steps": self._spread_stats.count,
            "spread_mean": self._spread_stats.mean,
            "spread_std": self._spread_stats.std,
            "logit_mean": self._logit_stats.mean,
            "logit_std": self._logit_stats.std,
            "ema_spread": self._ema_spread,
            "ema_logit": self._ema_logit,
        }

    @property
    def delegation_rate(self) -> float:
        """Fraction of steps that triggered delegation."""
        if not self._history:
            return 0.0
        return sum(1 for s in self._history if s.should_delegate) / len(self._history)

    def summary(self) -> str:
        """One-line summary of router state."""
        b = self.baseline
        return (
            f"EntropyRouter: {b['steps']} steps, "
            f"spread μ={b['spread_mean']:.3f} σ={b['spread_std']:.3f}, "
            f"logit μ={b['logit_mean']:.3f} σ={b['logit_std']:.3f}, "
            f"delegation rate={self.delegation_rate:.1%}"
        )

    # ── Internal math ────────────────────────────────────────────

    @staticmethod
    def _layer_entropy(attn: torch.Tensor) -> Tuple[float, float]:
        """
        Compute head spread and normalized entropy for one layer.

        Args:
            attn: (batch, heads, q_len, k_len)

        Returns:
            (head_spread, mean_norm_entropy)
        """
        a = attn[0].float()  # (heads, q_len, k_len) — drop batch
        last_tok = a[:, -1, :]  # (heads, k_len) — last generated token

        # Clamp and renormalize
        last_tok = last_tok.clamp(min=1e-8)
        last_tok = last_tok / last_tok.sum(dim=-1, keepdim=True)

        # Per-head entropy
        head_ent = -(last_tok * last_tok.log2()).sum(dim=-1)  # (heads,)
        head_ent = torch.nan_to_num(head_ent, nan=0.0)

        # Head spread (disagreement)
        spread = (head_ent.max() - head_ent.min()).item()

        # Normalized mean entropy
        k_len = attn.shape[-1]
        max_ent = math.log2(k_len) if k_len > 1 else 1.0
        norm_ent = head_ent.mean().item() / max(max_ent, 1e-10)

        return spread, norm_ent

    @staticmethod
    def _logit_entropy(logits: torch.Tensor) -> Tuple[float, float]:
        """
        Entropy of the output distribution.

        Args:
            logits: (batch, vocab) or (vocab,)

        Returns:
            (absolute_entropy, normalized_entropy)
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits.float(), dim=-1)
        ent = -(probs * (probs + 1e-10).log2()).sum(dim=-1).item()
        max_ent = math.log2(probs.shape[-1])
        return ent, ent / max(max_ent, 1e-10)
