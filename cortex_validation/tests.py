"""
Unified test suite for warp-cortex.

Consolidates: _validate_all.py + test_stream_inject.py + test_async_delegate.py

Run:  python cortex_validation/tests.py
"""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# Section 1: Core Upgrades (from _validate_all.py)
# ======================================================================

def test_turbo_quant():
    """TurboQuant KV Cache Compression."""
    from cortex_core.turbo_quant import TurboQuantCache, hadamard_rotate, hadamard_unrotate

    x = torch.randn(4, 64)
    x_back = hadamard_unrotate(hadamard_rotate(x))
    err = (x - x_back).abs().max().item()
    assert err < 1e-4, f'Hadamard roundtrip failed: {err}'

    B, H, S, D = 1, 8, 256, 64
    kv = [(torch.randn(B, H, S, D), torch.randn(B, H, S, D)) for _ in range(2)]
    orig_bytes = sum(k.nelement() * 2 + v.nelement() * 2 for k, v in kv)
    ratio = 0.0

    for bits in [4, 3]:
        tq = TurboQuantCache(bits=bits, device='cpu')
        tq.compress(kv)
        ratio = tq.compression_ratio(orig_bytes)
        recon = tq.decompress()
        mse = sum((k1 - k2).pow(2).mean().item()
                   for (k1, _), (k2, _) in zip(kv, recon)) / len(kv)
    assert ratio > 1, 'Compression must exceed 1x'
    print('[PASS] test_turbo_quant')


def test_adaptive_k_and_eviction():
    """Adaptive k + LRU Eviction on unified TopologicalSynapse."""
    from cortex_core.synapse import TopologicalSynapse

    syn = TopologicalSynapse(
        dim=64, max_landmarks=128, device='cpu',
        adaptive_k=True, k_min=8, k_max=64, ttl_seconds=0.5,
    )

    attn_focused = torch.zeros(1, 4, 100, 100)
    attn_focused[:, :, :, 5] = 100.0
    k_focused = syn.compute_adaptive_k(attn_focused)
    assert k_focused == 8, f'Expected k=8, got {k_focused}'

    attn_diffuse = torch.ones(1, 4, 100, 100)
    k_diffuse = syn.compute_adaptive_k(attn_diffuse)
    assert k_diffuse > 50, f'Expected k>50, got {k_diffuse}'

    keys = torch.randn(1, 100, 64)
    values = torch.randn(1, 100, 64)
    syn.update_landmarks(keys, values, attn_diffuse)
    count_before = syn.count
    assert count_before > 0
    time.sleep(0.7)
    evicted = syn.evict_stale()
    assert syn.count == 0, f'Expected 0 after TTL, got {syn.count}'
    print(f'[PASS] test_adaptive_k_and_eviction (evicted {evicted})')


def test_adaptive_validation_gate():
    """Adaptive Validation Gate."""
    from cortex_engine import AdaptiveValidationGate

    gate = AdaptiveValidationGate(initial_threshold=0.4, target_accept_rate=0.5)
    for _ in range(20):
        gate.should_accept(0.9)
    assert gate.threshold > 0.4

    gate2 = AdaptiveValidationGate(initial_threshold=0.5, target_accept_rate=0.5)
    for _ in range(20):
        gate2.should_accept(0.1)
    assert gate2.threshold < 0.5
    print('[PASS] test_adaptive_validation_gate')


def test_learnable_injection_gate():
    """CortexAttention topology-induced gate with synapse."""
    from cortex_core.cortex_attention import CortexAttention
    from cortex_core.synapse import TopologicalSynapse

    dim = 256
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    attn = CortexAttention(dim=dim, num_heads=8)

    x = torch.randn(1, 5, dim)
    out1 = attn(x, synapse)

    # Inject a landmark
    synapse.inject_embedding(torch.randn(dim))
    out2 = attn(x, synapse)

    diff = (out1 - out2).abs().sum().item()
    assert diff > 0, 'Gate should produce different output'
    n_params = sum(p.numel() for p in attn.gate_proj.parameters())
    assert n_params > 0
    print(f'[PASS] test_learnable_injection_gate (diff={diff:.4f}, {n_params} gate params)')


def test_semantic_router():
    """Semantic Router (MLP classifier head)."""
    from cortex_core.cortex_router import CortexRouter, _IntentClassifierHead

    head = _IntentClassifierHead(input_dim=64, num_intents=5, hidden=32)
    dummy_h = torch.randn(1, 64)
    logits = head(dummy_h)
    assert logits.shape == (1, 5)

    router = CortexRouter(confidence_threshold=0.5)
    tag_result = router.check_for_triggers('[TASK: verify the math]')
    assert tag_result == 'verify the math', f'Got: {tag_result}'

    task, conf = router.classify_hidden(dummy_h.squeeze(0))
    assert task is None  # pre-bootstrap
    print('[PASS] test_semantic_router')


def test_bitnet_side_agent():
    """BitNet Side Agent."""
    from cortex_engine import BitNetSideAgent, _BITNET_AVAILABLE

    assert _BITNET_AVAILABLE, 'warp-bitnet should be importable in this workspace'

    agent = BitNetSideAgent(hidden_size=128, num_layers=2, device='cpu')
    assert not agent._fallback, 'BitNetSideAgent should use real BitNet path, not FP16 fallback'
    x = torch.randn(1, 3, 128)
    out = agent(x)
    assert out.shape == (1, 3, 128)
    assert torch.isfinite(out).all(), 'BitNetSideAgent output should stay finite'
    thought = agent.think([torch.randn(1, 10, 128)], None)
    assert isinstance(thought, str) and len(thought) > 0
    assert 'nan' not in thought.lower()
    print(f'[PASS] test_bitnet_side_agent')


def test_bitnet_cuda_kernel_path():
    """BitNet CUDA path matches dense ternary reference."""
    if not torch.cuda.is_available():
        print('[SKIP] test_bitnet_cuda_kernel_path — no CUDA')
        return

    from warp_bitnet.kernel.bit_linear import BitLinear
    from warp_bitnet.kernel.packer import unpack_ternary_weights

    layer = BitLinear(16, 16, bias=False).to('cuda')
    dense_weight = torch.randint(-1, 2, (16, 16), device='cpu', dtype=torch.int8).float()
    scale = torch.tensor([0.5], dtype=torch.float16)
    layer.load_from_dense_weights(dense_weight, scale=scale)
    layer = layer.to('cuda')

    x = torch.randn(2, 16, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        y = layer(x)

    unpacked = unpack_ternary_weights(
        layer.packed_weight,
        (layer.out_features, layer.in_features),
    ).to(device='cuda', dtype=torch.float16)
    y_ref = torch.nn.functional.linear(x, unpacked * layer.weight_scale.to('cuda'))

    max_err = (y - y_ref).abs().max().item()
    assert max_err < 1e-3, f'BitNet CUDA kernel mismatch: max_err={max_err}'
    print(f'[PASS] test_bitnet_cuda_kernel_path (max_err={max_err:.6f})')


def test_bitnet_side_agent_cuda_forward():
    """BitNetSideAgent runs end-to-end on CUDA without dtype mismatch."""
    if not torch.cuda.is_available():
        print('[SKIP] test_bitnet_side_agent_cuda_forward — no CUDA')
        return

    from cortex_engine import BitNetSideAgent

    agent = BitNetSideAgent(hidden_size=128, num_layers=1, device='cuda')
    assert not agent._fallback
    x = torch.randn(1, 2, 128, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        out = agent(x)
    assert out.shape == (1, 2, 128)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), 'CUDA BitNetSideAgent output should stay finite'
    print('[PASS] test_bitnet_side_agent_cuda_forward')


def test_distributed_synapse():
    """Distributed Synapse (single-GPU mode, inherits from unified synapse)."""
    from cortex_core.distributed_synapse import DistributedSynapse

    dsyn = DistributedSynapse(
        dim=64, max_landmarks=128, world_size=1, rank=0, device='cpu',
    )
    keys = torch.randn(1, 100, 64)
    values = torch.randn(1, 100, 64)
    attn_scores = torch.ones(1, 4, 100, 100)
    dsyn.update_landmarks(keys, values, attn_scores)
    assert dsyn.count > 0
    k_out, v_out = dsyn.get_context()
    assert k_out.shape[0] == dsyn.count
    dsyn.sync()  # no-op in single-GPU mode

    # Also test injection on distributed synapse (inherited)
    dsyn.inject_embedding(torch.randn(64))
    assert dsyn.injection_count == 1

    # Eviction
    dsyn2 = DistributedSynapse(
        dim=64, max_landmarks=64, world_size=1, rank=0,
        device='cpu', adaptive_k=True, k_min=4, k_max=32, ttl_seconds=0.3,
    )
    dsyn2.update_landmarks(keys, values, attn_scores)
    before = dsyn2.count
    time.sleep(0.5)
    ev = dsyn2.evict_stale()
    assert dsyn2.count == 0, f'Expected 0, got {dsyn2.count}'
    print(f'[PASS] test_distributed_synapse (evicted {ev})')


def test_cuda_stream_pool():
    """CUDA Stream Pool."""
    if not torch.cuda.is_available():
        print('[SKIP] test_cuda_stream_pool — no CUDA')
        return
    from cortex_engine import CUDAStreamPool

    pool = CUDAStreamPool(pool_size=8, device='cuda')
    n = pool.available()
    s1 = pool.acquire(True)
    s2 = pool.acquire(False)
    assert pool.available() == n - 2
    pool.release(s1, True)
    pool.release(s2, False)
    assert pool.available() == n
    print(f'[PASS] test_cuda_stream_pool')


def test_gsm8k_benchmark_utils():
    """GSM8K Benchmark Script (import + utility check)."""
    from cortex_benchmarks.benchmark_cortex_gsm8k import (
        _builtin_problems,
        check_answer,
        extract_answer,
    )

    assert extract_answer('The answer is #### 42') == '42'
    assert extract_answer('Final answer: 100') == '100'
    assert extract_answer('\\boxed{256}') == '256'
    assert check_answer('42', '42.0')
    assert not check_answer('41', '42.5')
    problems = _builtin_problems()
    assert len(problems) >= 5
    print(f'[PASS] test_gsm8k_benchmark_utils')


# ======================================================================
# Section 2: Stream Injection (from test_stream_inject.py)
# ======================================================================

def test_claim_encoder_hashcode():
    """ClaimEncoder with hashcode fallback (no model needed)."""
    from cortex_core.stream_inject import VerifiedClaim, ClaimEncoder

    dim = 64
    encoder = ClaimEncoder(dim=dim, tokenizer=None, embed_layer=None, device='cpu')

    claim_pass = VerifiedClaim(
        expression="48 / 3", claimed="16", actual="16.0", verified=True,
    )
    vec_pass = encoder.encode(claim_pass)
    assert vec_pass.shape == (dim,)
    assert claim_pass.embedding is not None

    claim_fail = VerifiedClaim(
        expression="48 / 3", claimed="15", actual="16.0", verified=False,
    )
    vec_fail = encoder.encode(claim_fail)
    assert vec_fail.shape == (dim,)

    cos_sim = torch.nn.functional.cosine_similarity(
        vec_pass.unsqueeze(0), vec_fail.unsqueeze(0),
    )
    assert cos_sim.item() < 0.99, f"PASS/FAIL too similar: {cos_sim.item():.4f}"
    print("[PASS] test_claim_encoder_hashcode")


def test_claim_encoder_model_dtype_alignment():
    """Model-based ClaimEncoder should align projection dtype with embed layer dtype."""
    from cortex_core.stream_inject import VerifiedClaim, ClaimEncoder

    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=32):
            class TokenBatch:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

            return TokenBatch()

    dim = 8
    embed_layer = torch.nn.Embedding(32, dim, dtype=torch.float64)
    tokenizer = DummyTokenizer()
    encoder = ClaimEncoder(
        dim=dim,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        device='cpu',
    )

    claim = VerifiedClaim(
        expression="16 + 7", claimed="23", actual="23", verified=True,
    )
    vec = encoder.encode(claim)
    assert vec.shape == (dim,)
    assert vec.dtype == torch.float64
    assert torch.isfinite(vec).all()
    print("[PASS] test_claim_encoder_model_dtype_alignment")


def test_synapse_inject_read():
    """TopologicalSynapse inject → read cycle (replaces old SynapseBuffer test)."""
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=8, device='cpu')

    # Empty read
    inj_k, inj_v = synapse.get_injection_context()
    assert inj_k is None

    # Inject and read
    vec = torch.randn(dim)
    synapse.inject_embedding(vec)
    inj_k, inj_v = synapse.get_injection_context()
    assert inj_k is not None
    assert inj_k.shape == (1, dim)
    assert torch.allclose(inj_k[0], vec)

    print("[PASS] test_synapse_inject_read")


def test_stream_injector_pipeline():
    """Full pipeline: claim → encode → inject as landmark."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.stream_inject import VerifiedClaim, ClaimEncoder, StreamInjector

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    encoder = ClaimEncoder(dim=dim, device='cpu')
    injector = StreamInjector(synapse=synapse, claim_encoder=encoder, device='cpu')

    claim = VerifiedClaim(
        expression="7 * 8", claimed="56", actual="56", verified=True,
    )
    injector.inject_verified_claim(claim)

    # Synapse should have 1 injection landmark
    inj_k, inj_v = synapse.get_injection_context()
    assert inj_k is not None
    assert inj_k.shape == (1, dim)

    # Pending should have 1 claim
    pending = injector.get_pending()
    assert len(pending) == 1
    assert pending[0].expression == "7 * 8"
    print("[PASS] test_stream_injector_pipeline")


def test_topo_features_on_inject():
    """Topology features update when claims are injected."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.stream_inject import VerifiedClaim, ClaimEncoder, StreamInjector

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    encoder = ClaimEncoder(dim=dim, device='cpu')
    injector = StreamInjector(synapse=synapse, claim_encoder=encoder, device='cpu')

    # Before injection: topo features trivial
    d, s, c = synapse.topo_features()
    assert c == 0.0

    # Inject 3 claims (need >=2 for meaningful topo)
    for i in range(3):
        claim = VerifiedClaim(
            expression=f"{i} + 1", claimed=str(i + 1),
            actual=str(i + 1), verified=True,
        )
        injector.inject_verified_claim(claim)

    d, s, c = synapse.topo_features()
    # Coverage = 3 / (128 + 128) for total capacity
    expected_coverage = 3 / (128 + 128)
    assert c > 0, f"Coverage should be > 0, got {c}"
    assert abs(c - expected_coverage) < 0.01, f"Unexpected coverage: {c}"
    print("[PASS] test_topo_features_on_inject")


def test_cortex_attention_gate_absorbs():
    """CortexAttention cross-attends to injection landmarks."""
    from cortex_core.cortex_attention import CortexAttention
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    num_heads = 4
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    attn = CortexAttention(dim=dim, num_heads=num_heads)
    attn.eval()

    x = torch.randn(1, 5, dim)

    # Without injection
    out_clean = attn(x, synapse)

    # Inject a strong signal
    synapse.inject_embedding(torch.randn(dim) * 10.0)

    # With injection — output should differ at last token
    out_inject = attn(x, synapse)

    diff = (out_inject[:, -1, :] - out_clean[:, -1, :]).abs().max().item()
    assert diff > 1e-6, f"Gate should modify output, but diff={diff}"

    # Earlier tokens should be unchanged
    diff_early = (out_inject[:, :-1, :] - out_clean[:, :-1, :]).abs().max().item()
    assert diff_early < 1e-5, f"Earlier tokens should be unchanged, diff={diff_early}"
    print("[PASS] test_cortex_attention_gate_absorbs")


def test_latex_claim_extraction():
    """extract_claims handles LaTeX-formatted math."""
    from cortex_scripts.council_live import extract_claims

    claims = extract_claims("48 / 3 = 16")
    assert len(claims) >= 1
    assert any(c["claimed"] == "16" for c in claims)

    claims = extract_claims("60 + 40 + 20 = 120")
    assert len(claims) >= 1
    assert any(c["claimed"] == "120" for c in claims)

    latex = r"\[16 \text{ eggs} + 3 \text{ eggs} + 4 \text{ eggs} = 23\]"
    claims = extract_claims(latex)
    assert len(claims) >= 1, f"Should extract claims from LaTeX, got {claims}"
    assert any(c["claimed"] == "23" for c in claims)

    latex2 = r"3 \times 12 = 36"
    claims = extract_claims(latex2)
    assert len(claims) >= 1
    assert any(c["claimed"] == "36" for c in claims)
    print("[PASS] test_latex_claim_extraction")


def test_batch_inject():
    """inject_batch processes multiple claims."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.stream_inject import VerifiedClaim, ClaimEncoder, StreamInjector

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    encoder = ClaimEncoder(dim=dim, device='cpu')
    injector = StreamInjector(synapse=synapse, claim_encoder=encoder, device='cpu')

    claims = [
        VerifiedClaim(expression="2 + 3", claimed="5", actual="5", verified=True),
        VerifiedClaim(expression="10 / 2", claimed="5", actual="5.0", verified=True),
        VerifiedClaim(expression="7 * 3", claimed="20", actual="21", verified=False),
    ]
    injector.inject_batch(claims)

    pending = injector.get_pending()
    assert len(pending) == 3

    # Synapse should have 3 injection landmarks
    assert synapse.injection_count == 3
    print("[PASS] test_batch_inject")


# ======================================================================
# Section 3: Async Delegation (from test_async_delegate.py)
# ======================================================================

def test_detect_explicit_delegation():
    """Parse [DELEGATE:...] tags from model output."""
    from cortex_core.async_delegate import detect_delegation_requests

    text = """
    Let me think step by step.
    First, I'll compute 2^10:
    [DELEGATE:code] print(2**10) [/DELEGATE]
    And simplify this:
    [DELEGATE:math] 3 * 7 + 1 [/DELEGATE]
    """
    reqs = detect_delegation_requests(text)
    assert len(reqs) == 2
    assert reqs[0].expert_kind == "code"
    assert "print(2**10)" in reqs[0].payload
    assert reqs[1].expert_kind == "math"
    print("[PASS] test_detect_explicit_delegation")


def test_detect_custom_expert():
    """Parse custom expert with instructions."""
    from cortex_core.async_delegate import detect_delegation_requests

    text = "[DELEGATE:custom:optimizer] Minimize for x | x**2 - 4*x + 3 [/DELEGATE]"
    reqs = detect_delegation_requests(text)
    assert len(reqs) == 1
    assert reqs[0].expert_kind == "optimizer"
    assert reqs[0].instructions == "Minimize for x"
    print("[PASS] test_detect_custom_expert")


def test_no_false_positives():
    """Normal text shouldn't trigger delegation."""
    from cortex_core.async_delegate import detect_delegation_requests

    text = "The answer is 42. Let me verify: 6 * 7 = 42. #### 42"
    reqs = detect_delegation_requests(text)
    assert len(reqs) == 0
    print("[PASS] test_no_false_positives")


def test_orchestrated_engine_direct_by_default():
    """The orchestrated runner should stay direct when no worker is requested."""
    from cortex_scripts.council_live import OrchestratedReasoningEngine

    class StubBackend:
        model_id = "stub"

        def __init__(self):
            self.calls = 0

        def generate(self, messages, temperature=0.0, max_tokens=512):
            self.calls += 1
            return "Work it out directly. #### 42"

    backend = StubBackend()
    engine = OrchestratedReasoningEngine(backend, max_rounds=3, verbose=False)
    result = engine.solve("What is 40 + 2?")

    assert result["answer"] == "42"
    assert result["delegations"] == 0
    assert result["rounds"] == 1
    assert backend.calls == 1
    print("[PASS] test_orchestrated_engine_direct_by_default")


def test_orchestrated_engine_extends_direct_prompt_and_respects_max_tokens():
    """Delegation mode should stay direct on the first turn and propagate max token budgets."""
    from cortex_core.async_delegate import AsyncDelegationManager
    from cortex_scripts.council_live import DIRECT_SYSTEM, OrchestratedReasoningEngine

    class StubBackend:
        model_id = "stub"

        def __init__(self):
            self.calls = []

        def generate(self, messages, temperature=0.0, max_tokens=512):
            self.calls.append({"messages": messages, "max_tokens": max_tokens})
            return "Work it out directly. #### 42"

    backend = StubBackend()
    manager = AsyncDelegationManager(stream_injector=None, backend=backend, max_workers=1)
    engine = OrchestratedReasoningEngine(
        backend,
        max_rounds=3,
        delegation_mgr=manager,
        verbose=False,
    )

    try:
        result = engine.solve("What is 40 + 2?", max_tokens=321)
    finally:
        manager.shutdown()

    assert result["answer"] == "42"
    assert len(backend.calls) == 1
    assert backend.calls[0]["max_tokens"] == 321
    system_prompt = backend.calls[0]["messages"][0]["content"]
    assert system_prompt == DIRECT_SYSTEM
    print("[PASS] test_orchestrated_engine_extends_direct_prompt_and_respects_max_tokens")


def test_orchestrated_engine_delegates_explicitly():
    """The orchestrated runner should only use workers when delegation markup appears."""
    from cortex_core.async_delegate import AsyncDelegationManager
    from cortex_scripts.council_live import OrchestratedReasoningEngine

    class StubBackend:
        model_id = "stub"

        def __init__(self):
            self.calls = 0

        def generate(self, messages, temperature=0.0, max_tokens=512):
            self.calls += 1
            if self.calls == 1:
                return (
                    "I should check the arithmetic. "
                    "[DELEGATE:math] 7 * 8 [/DELEGATE]"
                )
            return "The worker confirmed the result. #### 56"

    backend = StubBackend()
    manager = AsyncDelegationManager(stream_injector=None, backend=backend, max_workers=2)
    engine = OrchestratedReasoningEngine(
        backend,
        max_rounds=3,
        delegation_mgr=manager,
        verbose=False,
    )

    try:
        result = engine.solve("What is 7 times 8?")
    finally:
        manager.shutdown()

    assert result["answer"] == "56"
    assert result["delegations"] == 1
    assert result["rounds"] == 2
    assert backend.calls == 2
    print("[PASS] test_orchestrated_engine_delegates_explicitly")


def test_orchestrated_engine_uses_worker_result_on_repeat():
    """Repeated identical delegation should finalize from the worker result instead of looping."""
    from cortex_core.async_delegate import AsyncDelegationManager
    from cortex_scripts.council_live import OrchestratedReasoningEngine

    class StubBackend:
        model_id = "stub"

        def __init__(self):
            self.calls = 0

        def generate(self, messages, temperature=0.0, max_tokens=512):
            self.calls += 1
            return "[DELEGATE:math] 7 * 8 = 56 [/DELEGATE]"

    backend = StubBackend()
    manager = AsyncDelegationManager(stream_injector=None, backend=backend, max_workers=2)
    engine = OrchestratedReasoningEngine(
        backend,
        max_rounds=3,
        delegation_mgr=manager,
        verbose=False,
    )

    try:
        result = engine.solve("Use a worker to compute 7 times 8.")
    finally:
        manager.shutdown()

    assert result["answer"] == "56"
    assert result["delegations"] == 1
    assert result["rounds"] == 2
    assert backend.calls == 2
    print("[PASS] test_orchestrated_engine_uses_worker_result_on_repeat")


def test_math_executor():
    """Math expression evaluation."""
    from cortex_core.async_delegate import _evaluate_math

    result = _evaluate_math("3 * 7 + 1")
    assert result.success
    assert result.output == "22"

    result = _evaluate_math("2 ** 10")
    assert result.success
    assert result.output == "1024"

    result = _evaluate_math("17 * 23 = 391")
    assert result.success
    assert result.output == "391"

    result = _evaluate_math("import os")
    assert not result.success
    print("[PASS] test_math_executor")


def test_code_executor():
    """Code execution in subprocess sandbox."""
    from cortex_core.async_delegate import _execute_code

    result = _execute_code("print(2 ** 10)", timeout=10.0)
    assert result.success, f"Code exec failed: {result.error}"
    assert result.output.strip() == "1024"
    print("[PASS] test_code_executor")


def test_async_manager_lifecycle():
    """Manager dispatches tasks and collects results."""
    from cortex_core.async_delegate import AsyncDelegationManager, DelegationRequest

    mgr = AsyncDelegationManager(stream_injector=None, backend=None, max_workers=2)
    req = DelegationRequest(task_id="", expert_kind="math", payload="100 / 4")
    tid = mgr.dispatch(req)
    assert tid is not None
    mgr.wait_all(timeout=10.0)
    results = mgr.poll_results()
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "25.0"
    mgr.shutdown()
    print("[PASS] test_async_manager_lifecycle")


def test_scan_and_dispatch():
    """End-to-end: scan text → dispatch → wait → results."""
    from cortex_core.async_delegate import AsyncDelegationManager, scan_and_dispatch

    mgr = AsyncDelegationManager(stream_injector=None, backend=None, max_workers=2)
    text = "Let me check: [DELEGATE:math] 7 * 8 [/DELEGATE] while I continue thinking."
    task_ids = scan_and_dispatch(text, mgr)
    assert len(task_ids) == 1
    mgr.wait_all(timeout=10.0)
    results = mgr.poll_results()
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "56"
    mgr.shutdown()
    print("[PASS] test_scan_and_dispatch")


def test_concurrent_dispatch():
    """Multiple tasks dispatched concurrently."""
    from cortex_core.async_delegate import AsyncDelegationManager, scan_and_dispatch

    mgr = AsyncDelegationManager(stream_injector=None, backend=None, max_workers=4)
    text = """
    [DELEGATE:math] 2 + 3 [/DELEGATE]
    [DELEGATE:math] 10 * 10 [/DELEGATE]
    [DELEGATE:code] print(sum(range(10))) [/DELEGATE]
    """
    task_ids = scan_and_dispatch(text, mgr)
    assert len(task_ids) == 3
    mgr.wait_all(timeout=15.0)
    results = mgr.poll_results()
    assert len(results) == 3
    successes = sum(1 for r in results if r.success)
    assert successes == 3
    mgr.shutdown()
    print("[PASS] test_concurrent_dispatch")


def test_custom_expert_registration():
    """Register and dispatch to a custom expert."""
    from cortex_core.async_delegate import (
        AsyncDelegationManager, DelegationRequest, ExpertProfile, ExpertKind,
    )

    mgr = AsyncDelegationManager(stream_injector=None, backend=None, max_workers=2)
    mgr.register_expert("fast_calc", ExpertProfile(
        kind=ExpertKind.MATH_SIMPLIFY, name="fast_calc", timeout=5.0,
    ))
    req = DelegationRequest(task_id="", expert_kind="fast_calc", payload="2 ** 20")
    mgr.dispatch(req)
    mgr.wait_all(timeout=10.0)
    results = mgr.poll_results()
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "1048576"
    mgr.shutdown()
    print("[PASS] test_custom_expert_registration")


def test_stream_injection_from_delegation():
    """Delegation results get injected into synapse when injector present."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.stream_inject import ClaimEncoder, StreamInjector
    from cortex_core.async_delegate import AsyncDelegationManager, DelegationRequest

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    encoder = ClaimEncoder(dim=dim, device='cpu')
    injector = StreamInjector(synapse=synapse, claim_encoder=encoder, device='cpu')

    mgr = AsyncDelegationManager(
        stream_injector=injector, backend=None, max_workers=2, device='cpu',
    )

    req = DelegationRequest(task_id="", expert_kind="math", payload="5 * 5")
    mgr.dispatch(req)
    mgr.wait_all(timeout=10.0)

    # The result should have been injected as a landmark
    assert synapse.injection_count > 0, \
        "Delegation result should be injected into synapse"
    inj_k, inj_v = synapse.get_injection_context()
    assert inj_k is not None
    assert inj_k.shape[1] == dim

    mgr.shutdown()
    print("[PASS] test_stream_injection_from_delegation")


# ======================================================================
# Section 4: Score-Weighted LRU Eviction
# ======================================================================

def test_score_weighted_eviction():
    """Low-score injections evicted before high-score ones when full."""
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=4, device='cpu')

    # Fill with 4 injections at different scores
    vecs = [torch.randn(dim) for _ in range(4)]
    scores = [1.0, 0.4, 0.8, 0.5]
    for v, s in zip(vecs, scores):
        synapse.inject_embedding(v, score=s)
    assert synapse.injection_count == 4

    # Inject a 5th — should evict the lowest score (0.4, index 1)
    new_vec = torch.randn(dim)
    synapse.inject_embedding(new_vec, score=0.9)
    assert synapse.injection_count == 4  # still 4 (buffer is full)

    # The 0.4-score vector should have been evicted
    remaining_scores = synapse.injection_scores[:4].tolist()
    assert 0.4 not in [round(s, 1) for s in remaining_scores], \
        f"Score 0.4 should be evicted, but got {remaining_scores}"
    assert 0.9 in [round(s, 1) for s in remaining_scores], \
        f"New score 0.9 should be present, got {remaining_scores}"
    print("[PASS] test_score_weighted_eviction")


def test_high_score_resists_eviction():
    """Verified truths (1.0) stubbornly resist eviction."""
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=3, device='cpu')

    # Insert: truth (1.0), speculation (0.4), speculation (0.4)
    truth = torch.ones(dim)  # recognizable
    synapse.inject_embedding(truth, score=1.0)
    synapse.inject_embedding(torch.randn(dim), score=0.4)
    synapse.inject_embedding(torch.randn(dim), score=0.4)

    # Force 5 more injections at score 0.5 — truth should survive
    for _ in range(5):
        synapse.inject_embedding(torch.randn(dim), score=0.5)

    # truth (1.0) should still be in the buffer
    keys = synapse.injection_keys[:synapse.injection_count]
    found_truth = any(torch.allclose(keys[i], truth.to(keys.device))
                      for i in range(synapse.injection_count))
    assert found_truth, "1.0-score landmark should resist eviction"
    print("[PASS] test_high_score_resists_eviction")


# ======================================================================
# Section 5: Speculative Thought Engine
# ======================================================================

def test_speculative_engine_lifecycle():
    """SpeculativeEngine starts, runs, and cancels cleanly."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.speculative import SpeculativeEngine, SpeculativeStrategy, SpeculativeThought

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')

    # Custom strategy that always produces a thought (no model needed)
    class AlwaysThinkStrategy(SpeculativeStrategy):
        name = "always_think"
        def should_run(self, context):
            return True
        def run(self, context, side_agent, tokenizer, device):
            embedding = torch.randn(context.get("dim", 64))
            return SpeculativeThought(
                strategy=self.name,
                content="speculative thought",
                embedding=embedding,
            )

    spec = SpeculativeEngine(
        synapse, strategies=[AlwaysThinkStrategy()],
        idle_delay_s=0.1,  # short delay for test
        max_speculations=2,
        device='cpu',
    )
    spec.start()
    time.sleep(1.0)  # let it run
    spec.cancel()

    # Should have injected up to max_speculations thoughts
    assert synapse.injection_count > 0, "Should have injected speculative thoughts"
    assert synapse.injection_count <= 2, f"Max 2 speculations, got {synapse.injection_count}"

    # Scores should be 0.4 (speculative)
    for i in range(synapse.injection_count):
        assert abs(synapse.injection_scores[i].item() - 0.4) < 1e-6, \
            f"Speculative score should be 0.4, got {synapse.injection_scores[i].item()}"

    assert len(spec.history) == synapse.injection_count
    print("[PASS] test_speculative_engine_lifecycle")


def test_speculative_cancel_is_immediate():
    """cancel() stops speculation even if idle delay hasn't passed."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.speculative import SpeculativeEngine

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')

    spec = SpeculativeEngine(
        synapse, idle_delay_s=10.0,  # very long delay
        device='cpu',
    )
    spec.start()
    assert spec.is_active
    spec.cancel()
    assert not spec.is_active

    # Nothing should have been injected (cancelled before idle delay)
    assert synapse.injection_count == 0
    print("[PASS] test_speculative_cancel_is_immediate")


def test_speculative_evicted_before_verified():
    """Speculative thoughts (0.4) evicted before verified claims (1.0)."""
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=3, device='cpu')

    # Inject 2 speculative (0.4) + 1 verified (1.0)
    synapse.inject_embedding(torch.randn(dim), score=0.4)
    synapse.inject_embedding(torch.randn(dim), score=0.4)
    verified = torch.ones(dim) * 0.5  # recognizable
    synapse.inject_embedding(verified, score=1.0)

    # Push a new verified claim — should evict a speculative one
    synapse.inject_embedding(torch.randn(dim), score=1.0)

    # At least one 0.4 should have been evicted
    scores = synapse.injection_scores[:synapse.injection_count].tolist()
    n_speculative = sum(1 for s in scores if abs(s - 0.4) < 0.01)
    assert n_speculative <= 1, f"Expected at most 1 speculative, got {n_speculative}"
    print("[PASS] test_speculative_evicted_before_verified")


# ======================================================================
# Section 6: Red Team Agent
# ======================================================================

def test_red_team_sql_injection():
    """Red Team catches SQL injection."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    code = '''
def get_user(name):
    db.execute(f"SELECT * FROM users WHERE name = '{name}'")
'''
    critiques = red.review(code)
    assert len(critiques) >= 1
    assert any(c.category == "security" for c in critiques)
    assert any("SQL" in c.finding or "sql" in c.finding.lower() for c in critiques)

    # Should have been injected into synapse
    assert synapse.injection_count >= 1
    print("[PASS] test_red_team_sql_injection")


def test_red_team_resource_leak():
    """Red Team catches file handle leak."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    code = 'f = open("data.txt", "r")\ndata = f.read()'
    critiques = red.review(code)
    assert len(critiques) >= 1
    assert any(c.category == "bug" for c in critiques)
    assert any("resource" in c.finding.lower() or "with" in c.finding.lower()
               for c in critiques)
    print("[PASS] test_red_team_resource_leak")


def test_red_team_no_false_positives():
    """Clean code should produce no critiques."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    clean_code = '''
def add(a: int, b: int) -> int:
    return a + b
'''
    critiques = red.review(clean_code)
    assert len(critiques) == 0, f"Clean code got critiques: {[c.finding for c in critiques]}"
    print("[PASS] test_red_team_no_false_positives")


def test_red_team_code_blocks():
    """Red Team extracts and reviews code blocks from model output."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    text = '''Here's a function to query users:
```python
def find_user(uid):
    db.execute(f"SELECT * FROM users WHERE id = {uid}")
    return db.fetchone()
```
This should work.'''

    critiques = red.review_code_blocks(text)
    assert len(critiques) >= 1
    assert any(c.category == "security" for c in critiques)
    print("[PASS] test_red_team_code_blocks")


def test_red_team_async():
    """Non-blocking review completes correctly."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    future = red.review_async('f = open("x.txt")\ndata = f.read()')
    critiques = future.result(timeout=5.0)
    assert len(critiques) >= 1

    red.shutdown()
    print("[PASS] test_red_team_async")


def test_red_team_severity_scoring():
    """Critical bugs get higher injection scores."""
    from cortex_core.synapse import TopologicalSynapse
    from cortex_core.red_team import RedTeamAgent, RED_TEAM_SCORE

    dim = 64
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    red = RedTeamAgent(synapse, device='cpu')

    # SQL injection = severity 1.0, resource leak = severity 0.6
    code = '''
db.execute(f"DELETE FROM users WHERE name = '{name}'")
f = open("log.txt")
'''
    critiques = red.review(code)
    severities = sorted([c.severity for c in critiques], reverse=True)
    assert len(severities) >= 2
    assert severities[0] > severities[1], "SQL injection should be more severe"

    # Check injection scores in synapse
    scores = synapse.injection_scores[:synapse.injection_count].tolist()
    assert max(scores) > min(scores), "Higher severity → higher injection score"
    print("[PASS] test_red_team_severity_scoring")

def main() -> int:
    print("=" * 60)
    print("  WARP-CORTEX UNIFIED TEST SUITE")
    print("=" * 60)

    all_tests = [
        # Section 1: Core upgrades
        test_turbo_quant,
        test_adaptive_k_and_eviction,
        test_adaptive_validation_gate,
        test_learnable_injection_gate,
        test_semantic_router,
        test_bitnet_side_agent,
        test_bitnet_cuda_kernel_path,
        test_bitnet_side_agent_cuda_forward,
        test_distributed_synapse,
        test_cuda_stream_pool,
        test_gsm8k_benchmark_utils,
        # Section 2: Stream injection
        test_claim_encoder_hashcode,
        test_claim_encoder_model_dtype_alignment,
        test_synapse_inject_read,
        test_stream_injector_pipeline,
        test_topo_features_on_inject,
        test_cortex_attention_gate_absorbs,
        test_latex_claim_extraction,
        test_batch_inject,
        # Section 3: Async delegation
        test_detect_explicit_delegation,
        test_detect_custom_expert,
        test_no_false_positives,
        test_orchestrated_engine_direct_by_default,
        test_orchestrated_engine_extends_direct_prompt_and_respects_max_tokens,
        test_orchestrated_engine_delegates_explicitly,
        test_orchestrated_engine_uses_worker_result_on_repeat,
        test_math_executor,
        test_code_executor,
        test_async_manager_lifecycle,
        test_scan_and_dispatch,
        test_concurrent_dispatch,
        test_custom_expert_registration,
        test_stream_injection_from_delegation,
        # Section 4: Score-weighted LRU
        test_score_weighted_eviction,
        test_high_score_resists_eviction,
        # Section 5: Speculative thought
        test_speculative_engine_lifecycle,
        test_speculative_cancel_is_immediate,
        test_speculative_evicted_before_verified,
        # Section 6: Red team
        test_red_team_sql_injection,
        test_red_team_resource_leak,
        test_red_team_no_false_positives,
        test_red_team_code_blocks,
        test_red_team_async,
        test_red_team_severity_scoring,
    ]

    passed = 0
    failed = 0
    skipped = 0
    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                import traceback
                print(f"[FAIL] {test.__name__}: {e}")
                traceback.print_exc()
                failed += 1

    total = passed + failed + skipped
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} tests passed"
          f"{f', {skipped} skipped' if skipped else ''}"
          f"{f', {failed} FAILED' if failed else ''}")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
