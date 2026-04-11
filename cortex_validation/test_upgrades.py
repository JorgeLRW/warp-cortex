import os
import sys
import time
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.hf_utils import prepare_hf_cache

print('=' * 60)
print('TEST 1: Semantic Router')
print('=' * 60)
from cortex_core.cortex_router import CortexRouter, _IntentClassifierHead

# Test 1a: Legacy regex still works
router = CortexRouter()
assert router.check_for_triggers('[TASK: verify the proof]') == 'verify the proof'
assert router.check_for_triggers('[DELEGATE: summarise this]') == 'summarise this'
assert router.check_for_triggers('[SEARCH]') == 'Perform a search to verify this information.'
print('[PASS] Legacy regex triggers intact')

# Test 1b: Classifier head architecture
head = _IntentClassifierHead(input_dim=896, num_intents=5)
x = torch.randn(4, 896)
logits = head(x)
assert logits.shape == (4, 5), f'Expected (4,5), got {logits.shape}'
print(f'[PASS] Classifier head: {sum(p.numel() for p in head.parameters())} params')

# Test 1c: Bootstrap with a real model (Qwen 0.5B)
from transformers import AutoModelForCausalLM, AutoTokenizer
cache_dir = prepare_hf_cache(ROOT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    cache_dir=cache_dir,
    dtype=torch.float16,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', cache_dir=cache_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

router2 = CortexRouter()
router2.bootstrap(model, tokenizer, device=device)
assert router2._bootstrapped
assert router2._head is not None
print('[PASS] Bootstrap completed on Qwen2.5-0.5B')

# Test 1d: Classify hidden states
with torch.no_grad():
    ids = tokenizer('Write a Python script to sort a list', return_tensors='pt').input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    h = out.hidden_states[-1].mean(dim=1).squeeze(0)
    task, conf = router2.classify_hidden(h)
    print(f'[PASS] Semantic classify: task="{task}" conf={conf:.2f}')

# Test 1e: Hybrid check_for_triggers with hidden state
result = router2.check_for_triggers('Write a Python script to sort a list', hidden_state=h)
print(f'[PASS] Hybrid trigger: "{result}"')

print()
print('=' * 60)
print('TEST 2: BitNet Side Agent')
print('=' * 60)
from cortex_engine import BitNetSideAgent, _BITNET_AVAILABLE
print(f'[INFO] warp-bitnet available: {_BITNET_AVAILABLE}')

# Test 2a: Scratch init (ternary MLP stack)
agent = BitNetSideAgent(device='cpu', hidden_size=128, num_layers=2)
x = torch.randn(1, 5, 128)
out = agent.forward(x)
assert out.shape == (1, 5, 128), f'Expected (1,5,128), got {out.shape}'
print(f'[PASS] BitNet forward: {out.shape}')

# Test 2b: Think with landmark input
landmarks = [(torch.randn(1, 4, 10, 128), torch.randn(1, 4, 10, 128))]
thought = agent.think(landmarks, None)
assert 'BitNet' in thought or 'bitnet' in thought.lower(), f'Unexpected thought: {thought}'
print(f'[PASS] BitNet think: "{thought}"')

# Test 2c: Think with tensor landmarks
tensor_landmarks = torch.randn(10, 128)
thought2 = agent.think(tensor_landmarks, None)
print(f'[PASS] BitNet tensor think: "{thought2}"')

print()
print('=' * 60)
print('TEST 3: Distributed Synapse')
print('=' * 60)
from cortex_core.distributed_synapse import DistributedSynapse

# Test 3a: Single-rank usage (no dist init needed)
dsyn = DistributedSynapse(dim=64, max_landmarks=32, world_size=1, rank=0, device='cpu')
keys = torch.randn(1, 100, 64)
values = torch.randn(1, 100, 64)
attn = torch.ones(1, 4, 100, 100)
dsyn.update_landmarks(keys, values, attn)
assert dsyn.count > 0
print(f'[PASS] DistSynapse update: count={dsyn.count}')

# Test 3b: Adaptive k
attn_focused = torch.zeros(1, 4, 100, 100)
attn_focused[:, :, :, 5] = 100.0
k = dsyn.compute_adaptive_k(attn_focused)
print(f'[PASS] DistSynapse adaptive k (focused): {k}')

# Test 3c: Eviction
dsyn2 = DistributedSynapse(dim=64, max_landmarks=32, world_size=1, rank=0, device='cpu', ttl_seconds=0.3)
dsyn2.update_landmarks(keys, values, attn)
count_before = dsyn2.count
time.sleep(0.5)
evicted = dsyn2.evict_stale()
print(f'[PASS] DistSynapse eviction: {count_before} -> {dsyn2.count} (evicted {evicted})')
assert dsyn2.count == 0

# Test 3d: sync() on single GPU (should be no-op)
dsyn.sync()
print('[PASS] DistSynapse sync (single-GPU no-op)')

# Test 3e: get_context
k_out, v_out = dsyn.get_context()
assert k_out.shape[0] == dsyn.count
print(f'[PASS] DistSynapse get_context: {k_out.shape}')

print()
print('=' * 60)
print('TEST 4: GSM8K Benchmark Module')
print('=' * 60)
from cortex_benchmarks.benchmark_cortex_gsm8k import (
    _builtin_problems,
    check_answer,
    extract_answer,
)

assert extract_answer('The answer is #### 42') == '42'
assert extract_answer('So we get \\boxed{59}') == '59'
assert extract_answer('Final answer: 210') == '210'
assert check_answer('42', '42')
assert check_answer('42.0', '42')
assert not check_answer('41', '42')
probs = _builtin_problems()
assert len(probs) >= 5
print(f'[PASS] Answer extraction + {len(probs)} built-in problems')

print()
print('=' * 60)
print('ALL TESTS PASSED')
print('=' * 60)
