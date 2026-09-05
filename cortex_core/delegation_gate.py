from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateDecision:
    """Scalar decision emitted by the learned delegation gate."""

    logit: float
    probability: float
    threshold: float
    ready: bool
    should_delegate: bool


class LinearDelegationGate(nn.Module):
    """
    Minimal learned gate over frozen backbone hidden states.

    The gate is intentionally tiny: a single linear projection
    ``nn.Linear(d_model, 1)`` trained on detached hidden states so the
    main model weights remain frozen and KV caches stay valid.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        threshold: float = 0.5,
        lr: float = 1e-3,
        warmup_steps: int = 64,
        normalize_input: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.threshold = threshold
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.normalize_input = normalize_input
        self._device = device
        self._trained_steps = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.proj: Optional[nn.Linear] = None

        if input_dim is not None:
            self._build(input_dim, device=device)

    @property
    def trained_steps(self) -> int:
        return self._trained_steps

    @property
    def ready(self) -> bool:
        return self._trained_steps >= self.warmup_steps

    def _build(self, input_dim: int, device: Optional[str] = None):
        if self.proj is not None:
            return
        layer = nn.Linear(input_dim, 1)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        target_device = device or self._device
        if target_device is not None:
            layer = layer.to(target_device)
        self.proj = layer
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _prepare_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.dim() != 1:
            hidden_state = hidden_state.reshape(-1)
        hidden_state = hidden_state.detach().float()
        if self.normalize_input:
            hidden_state = F.normalize(hidden_state, dim=0)
        if self.proj is None:
            self._build(hidden_state.shape[0], device=str(hidden_state.device))
        assert self.proj is not None
        return hidden_state.to(self.proj.weight.device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        vector = self._prepare_hidden(hidden_state)
        assert self.proj is not None
        return self.proj(vector.unsqueeze(0)).squeeze(0).squeeze(-1)

    @torch.no_grad()
    def decide(self, hidden_state: torch.Tensor) -> GateDecision:
        logit = float(self.forward(hidden_state).item())
        probability = float(torch.sigmoid(torch.tensor(logit)).item())
        ready = self.ready
        return GateDecision(
            logit=logit,
            probability=probability,
            threshold=self.threshold,
            ready=ready,
            should_delegate=ready and probability >= self.threshold,
        )

    def partial_fit(self, hidden_state: torch.Tensor, target: float) -> float:
        vector = self._prepare_hidden(hidden_state)
        assert self.proj is not None
        assert self._optimizer is not None

        self.train()
        logits = self.proj(vector.unsqueeze(0)).squeeze(0).squeeze(-1)
        labels = torch.tensor([target], device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits.unsqueeze(0), labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.eval()
        self._trained_steps += 1
        return float(loss.item())

    def save(self, path: str):
        if self.proj is None:
            raise RuntimeError("Cannot save an uninitialized gate")
        payload = {
            "state_dict": self.state_dict(),
            "threshold": self.threshold,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "normalize_input": self.normalize_input,
            "trained_steps": self._trained_steps,
            "input_dim": self.proj.in_features,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    def load(self, path: str, map_location: Optional[str] = None):
        payload = torch.load(path, map_location=map_location or self._device or "cpu")
        input_dim = int(payload["input_dim"])
        self.threshold = float(payload.get("threshold", self.threshold))
        self.lr = float(payload.get("lr", self.lr))
        self.warmup_steps = int(payload.get("warmup_steps", self.warmup_steps))
        self.normalize_input = bool(payload.get("normalize_input", self.normalize_input))
        self._build(input_dim, device=map_location or self._device)
        self.load_state_dict(payload["state_dict"])
        self._trained_steps = int(payload.get("trained_steps", 0))
        self.eval()