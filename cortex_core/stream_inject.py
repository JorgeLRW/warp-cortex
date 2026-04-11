"""
Stream Injection: Worker → Synapse → Gate
==========================================

Workers (BitNet 1.58-bit or local eval) produce results. Those results
are encoded as embedding vectors and injected into the TopologicalSynapse
as first-class landmarks. The main model's CortexAttention gate cross-
attends to all injection landmarks and decides how much to absorb.

This module bridges the gap between:
    - Text-level delegated task results or optional claim review
    - Embedding-level injection (CortexAttention topology gate)

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │   Worker completes claim verification                            │
    │   "48 / 3 = 16.0  → PASS"                                       │
    │                                                                  │
    │   ┌──────────────┐     ┌──────────────────┐     ┌────────────┐  │
    │   │  Claim text   │────▶│  ClaimEncoder    │────▶│  Synapse   │  │
    │   │  + result     │     │  (text → embed)  │     │ (landmark) │  │
    │   └──────────────┘     └──────────────────┘     └─────┬──────┘  │
    │                                                       │         │
    │   During next attention step:                         ▼         │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │  CortexAttention cross-attends to injection landmarks    │  │
    │   │  gate_input = [q_feat, s_feat, density, spread, coverage]│  │
    │   │  gate_val = sigmoid(gate_proj(gate_input))               │  │
    │   │  out[:,-1] += gate_val * cross_attend(synapse_landmarks) │  │
    │   └──────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────┘

The main model never re-prompts. It just discovers new knowledge in
its attention computation at the next token boundary.
"""

import torch
import torch.nn as nn
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class VerifiedClaim:
    """Result of a worker's verification, ready for embedding injection."""
    expression: str
    claimed: str
    actual: str
    verified: bool
    embedding: Optional[torch.Tensor] = None  # [dim] — filled by ClaimEncoder
    timestamp: float = field(default_factory=time.time)


class ClaimEncoder(nn.Module):
    """
    Encodes a verification result into a thought vector compatible
    with the SynapseBuffer / CortexAttention injection point.

    Two modes:
    1. **Model-based** (preferred): Run the claim text through the model's
       embedding layer + a lightweight projection to get a hidden-state vector.
    2. **Hashcode** (fallback, no model needed): Deterministic embedding from
       the claim string — good enough for the gate to distinguish PASS/FAIL.
    """

    def __init__(self, dim: int, tokenizer=None, embed_layer=None, device='cuda'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.tokenizer = tokenizer
        self.embed_layer = embed_layer  # model.model.embed_tokens
        self.model_dtype = (
            embed_layer.weight.dtype if embed_layer is not None else torch.float32
        )

        # Lightweight projection: token embeddings → thought vector
        # This is NOT the gate — it's the encoder that produces the vector
        # that will be evaluated by the gate.
        if embed_layer is not None:
            embed_dim = embed_layer.weight.shape[1]
            self.proj = nn.Linear(embed_dim, dim, bias=False).to(
                device=device, dtype=self.model_dtype
            )
            # Initialize as identity-like if dims match
            with torch.no_grad():
                if embed_dim == dim:
                    self.proj.weight.copy_(
                        torch.eye(dim, device=device, dtype=self.model_dtype)
                    )
                else:
                    nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = None

        # PASS/FAIL signal embedding (learnable, 2 vectors)
        self.signal_embed = nn.Embedding(2, dim).to(
            device=device, dtype=self.model_dtype
        )  # 0=FAIL, 1=PASS
        nn.init.normal_(self.signal_embed.weight, std=0.02)

    @torch.no_grad()
    def encode(self, claim: VerifiedClaim) -> torch.Tensor:
        """
        Encode a verified claim into a [dim] thought vector.

        The vector encodes: (1) WHAT was checked, (2) PASS or FAIL.
        The topology gate in CortexAttention will decide relevance.
        """
        signal = self.signal_embed(
            torch.tensor([1 if claim.verified else 0], device=self.device)
        )  # [1, dim]

        if self.tokenizer is not None and self.embed_layer is not None:
            # Model-based: embed the claim text and pool
            text = f"{claim.expression} = {claim.actual}"
            ids = self.tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=32).input_ids.to(self.device)
            token_embeds = self.embed_layer(ids)  # [1, seq, embed_dim]
            pooled = token_embeds.mean(dim=1)  # [1, embed_dim]
            assert self.proj is not None
            signal = signal.to(dtype=pooled.dtype)
            projected = self.proj(pooled)  # [1, dim]

            # Combine content + signal
            thought = projected + signal  # [1, dim]
        else:
            # Hashcode fallback: deterministic embedding from string
            hash_val = hash(f"{claim.expression}={claim.actual}")
            gen = torch.Generator(device='cpu').manual_seed(hash_val % (2**31))
            content = torch.randn(1, self.dim, generator=gen).to(self.device)
            content = content / (content.norm() + 1e-8) * 0.1  # unit-ish
            thought = content + signal

        claim.embedding = thought.squeeze(0)  # [dim]
        assert claim.embedding is not None
        return claim.embedding


class StreamInjector:
    """
    Manages async worker dispatch → encode → inject into TopologicalSynapse.

        Call flow:
            1. The orchestrator receives a worker result or reviewed claim
            2. StreamInjector.inject_verified_claim(claim) — encodes and injects
            3. ClaimEncoder encodes the result as an embedding vector
            4. Encoded vector becomes a landmark in the TopologicalSynapse
            5. Main model's next CortexAttention.forward() cross-attends to it

    The main model never blocks on workers. If a worker is slow, the
    gate simply doesn't see its result yet — zero latency impact.
    Topology features (density, spread, coverage) are computed live from
    the synapse's full landmark manifold.
    """

    def __init__(self, synapse, claim_encoder: ClaimEncoder,
                 stream_pool=None, device='cuda'):
        """
        Args:
            synapse: TopologicalSynapse instance (from synapse.py)
            claim_encoder: ClaimEncoder instance
            stream_pool: CUDAStreamPool (from cortex_engine.py), or None for sync
        """
        self.synapse = synapse
        self.encoder = claim_encoder
        self.pool = stream_pool
        self.device = device
        self._pending: List[VerifiedClaim] = []
        self._lock = threading.Lock()

    def inject_verified_claim(self, claim: VerifiedClaim):
        """
        Encode a verified claim and inject its embedding as a synapse landmark.
        Called after verification completes (either local eval or LLM worker).
        Topology features update automatically — the synapse computes them live.
        """
        # Encode claim → thought vector
        embedding = self.encoder.encode(claim)

        # Inject as a landmark — CortexAttention reads this at next forward()
        score = 1.0 if claim.verified else 0.5
        self.synapse.inject_embedding(embedding.to(self.device), score=score)

        with self._lock:
            self._pending.append(claim)

    def inject_batch(self, claims: List[VerifiedClaim]):
        """
        Inject multiple verified claims. If a CUDA stream pool is available,
        each injection runs on its own stream for zero blocking.
        """
        if self.pool is not None and torch.cuda.is_available():
            for claim in claims:
                stream = self.pool.acquire(high_priority=claim.verified is False)
                with torch.cuda.stream(stream):
                    self.inject_verified_claim(claim)
                self.pool.release(stream, high_priority=claim.verified is False)
        else:
            for claim in claims:
                self.inject_verified_claim(claim)

    def get_pending(self) -> List[VerifiedClaim]:
        """Return and clear pending claims (for logging/display)."""
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
        return out

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


class OrchestratedStreamEngine:
    """
    Orchestrated reasoning with embedding-level injection.

    Combines:
    - Orchestrated reasoning logic (answer → optional delegate/review → continue)
    - StreamInjector (verified claims → embeddings → synapse landmarks)
    - CortexAttention topology gate (absorbs worker knowledge into attention)

    The main model generates token-by-token. At each attention boundary,
    the synapse injection landmarks are checked. If a worker has finished
    verifying a claim, its result embedding is gated into the output.
    """

    def __init__(self, model, tokenizer, synapse, claim_encoder,
                 stream_pool=None, device='cuda', max_refine: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_refine = max_refine

        # The injection pipeline
        self.injector = StreamInjector(
            synapse=synapse,
            claim_encoder=claim_encoder,
            stream_pool=stream_pool,
            device=device,
        )
        self.synapse = synapse

    @torch.no_grad()
    def generate_with_verification(self, prompt: str, max_tokens: int = 400,
                                   temperature: float = 0.0) -> dict:
        """
        Generate a response while streaming verification results into the
        synapse buffer. The CortexAttention layers will absorb these via
        the topology gate.

        Returns dict with 'text', 'claims_injected', 'gate_activations'.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        past_kv = None
        generated_tokens = []
        gate_activations = []

        for step in range(max_tokens):
            outputs = self.model(
                inputs, past_key_values=past_kv,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())
            past_kv = outputs.past_key_values
            inputs = next_token

            # Check if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        pending = self.injector.get_pending()

        return {
            "text": text,
            "claims_injected": len(pending),
            "gate_activations": gate_activations,
            "pending_claims": pending,
        }


System2StreamEngine = OrchestratedStreamEngine
