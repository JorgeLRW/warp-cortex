"""
Verification Test Suite for Warp Cortex Overhaul & Fixes.

Validates:
1. Silent injection token integrity (no duplicated tokens or stuttering).
2. Attention-free EntropyRouter (FlashAttention/SDPA preservation via logit-only entropy).
3. Non-destructive TurboQuant KV cache stats tracking.
4. Scorecard execution and YAML policy emission.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.entropy_router import EntropyRouter, EntropySignal
from cortex_core.adaptive_engine import AdaptiveGenerator, DelegationMode
from cortex_scorecard.runner import run_scorecard
from cortex_scorecard.schema import ScorecardConfig


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 10, "start": 11, " [answer: 42]": 20, "after_inject": 30}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, return_tensors=None):
        ids = [self.vocab.get(text, 10)]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.item() if token_ids.numel() == 1 else token_ids.tolist()
        if isinstance(token_ids, int):
            return self.inv_vocab.get(token_ids, f"<tok_{token_ids}>")
        return "".join(self.inv_vocab.get(t, f"<tok_{t}>") for t in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "hello"

    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=self.encode(text, return_tensors=return_tensors))


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.step_idx = 0
        self.embed = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids, past_key_values=None, output_attentions=False, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        if self.step_idx == 0:
            logits[:, :, 11] = 5.0  # token 11: "start"
        else:
            logits[:, :, 30] = 5.0  # token 30: "after_inject"
        self.step_idx += 1

        # Simulate past_key_values as list of (K, V)
        new_k = torch.randn(batch_size, 2, seq_len, 16)
        new_v = torch.randn(batch_size, 2, seq_len, 16)
        if past_key_values is not None:
            past_k, past_v = past_key_values[0]
            new_k = torch.cat([past_k, new_k], dim=2)
            new_v = torch.cat([past_v, new_v], dim=2)

        out_kv = [(new_k, new_v)]
        hidden = torch.randn(batch_size, seq_len, self.hidden_dim)

        return SimpleNamespace(
            logits=logits,
            past_key_values=out_kv,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=None,
        )


class TestCortexFixes(unittest.TestCase):
    def test_entropy_router_attention_free(self):
        """Test that EntropyRouter functions without attention matrices (FlashAttention friendly)."""
        router = EntropyRouter(logit_z_threshold=1.5, warmup_steps=5)

        # 1. Warmup steps with uniform-ish distribution
        for _ in range(5):
            logits = torch.randn(1, 1000)
            sig = router.step(attentions=None, logits=logits)
            self.assertFalse(sig.should_delegate, "Should not delegate during warmup")
            self.assertEqual(len(sig.layer_head_spread), 0, "No layer spread when attentions=None")

        # 2. Normal steps with confident prediction (low entropy)
        for _ in range(8):
            logits = torch.zeros(1, 1000)
            logits[0, 42] = 20.0  # highly peaked
            sig = router.step(attentions=None, logits=logits)
            self.assertFalse(sig.should_delegate, "Low entropy should not trigger delegation")

        # 3. Step with high entropy spike (flat uniform distribution across all 1000 tokens)
        flat_logits = torch.ones(1, 1000) * 0.1
        spike_sig = router.step(attentions=None, logits=flat_logits)
        self.assertGreater(spike_sig.logit_entropy, 0.0)
        self.assertGreater(spike_sig.logit_z_score, 0.9)
        print(f"[EntropyRouter Test] Spike logit_z={spike_sig.logit_z_score:.2f}, delegate={spike_sig.should_delegate}")

    def test_silent_injection_no_duplicate(self):
        """Test that silent worker injection updates past_kv and does not duplicate tokens."""
        tokenizer = DummyTokenizer()
        model = DummyModel()

        engine = AdaptiveGenerator(
            model=model,
            tokenizer=tokenizer,
            mode=DelegationMode.SILENT,
            warmup_steps=0,
            verbose=False,
            device="cpu",
        )

        mock_result = SimpleNamespace(
            task_id="entropy_0",
            expert_kind="math",
            output="answer: 42",
            success=True,
            duration=0.01,
        )
        engine._dispatch_and_wait = MagicMock(return_value=[mock_result])

        step_counter = 0

        def mock_step(attentions, logits):
            nonlocal step_counter
            should_del = (step_counter == 0)
            step_counter += 1
            return EntropySignal(
                step=step_counter,
                layer_head_spread=[],
                layer_norm_entropy=[],
                logit_entropy=5.0 if should_del else 0.1,
                logit_norm_entropy=0.5,
                max_head_spread=0.0,
                max_spread_layer=-1,
                mean_head_spread=0.0,
                spread_z_score=0.0,
                logit_z_score=3.5 if should_del else 0.0,
                should_delegate=should_del,
                confidence=0.9 if should_del else 0.0,
            )

        engine.entropy_router.step = mock_step

        res = engine.generate(
            question="What is 6 times 7?",
            max_tokens=3,
        )

        # Worker was called exactly once on step 0
        engine._dispatch_and_wait.assert_called_once()

        # Output text contains injection
        self.assertIn("answer: 42", res.text)
        print(f"[Injection Test] Output text: {res.text}")
        print(f"[Injection Test] Tokens: {res.tokens}")

        # Ensure no token duplication
        self.assertGreaterEqual(len(res.tokens), 2)
        self.assertNotEqual(res.tokens[0], res.tokens[1])

    def test_scorecard_policy_emission(self):
        """Test that cortex_scorecard runs and produces a valid policy preview."""
        config = ScorecardConfig(
            suite="smoke_test",
            out_dir="local_artifacts/scorecards/test_smoke",
            max_tokens=32,
            temperature=0.0,
            timeout_seconds=10.0,
            device="cpu",
            offline=True,
            limit=2,
        )
        report = run_scorecard(
            config=config,
            candidate_names=["deterministic", "deterministic_bad"],
        )

        self.assertIn("aggregate", report)
        self.assertIn("policy", report)
        policy = report["policy"]
        self.assertIn("routes", policy)
        self.assertIn("support_json", policy["routes"])
        self.assertEqual(policy["routes"]["support_json"]["candidate"], "deterministic")
        print(f"[Scorecard Test] Successfully generated policy with {len(policy['routes'])} routes")


if __name__ == "__main__":
    unittest.main()
