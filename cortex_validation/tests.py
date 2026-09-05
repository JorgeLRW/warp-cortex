"""
Unified test suite for warp-cortex.

Consolidates: _validate_all.py + test_stream_inject.py + test_async_delegate.py

Run:  python cortex_validation/tests.py
"""
import sys, os, time, tempfile, torch
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
        restored = TurboQuantCache.from_state(tq.export_state(), device='cpu')
        assert restored.num_layers() == tq.num_layers()
        assert restored.memory_bytes() == tq.memory_bytes()
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


def test_linear_delegation_gate():
    """Linear hidden-state gate stays tiny and trainable without backbone grads."""
    from cortex_core.delegation_gate import LinearDelegationGate

    gate = LinearDelegationGate(threshold=0.5, lr=1e-2, warmup_steps=4, device='cpu')
    positive = torch.ones(32)
    negative = -torch.ones(32)

    for _ in range(6):
        gate.partial_fit(positive, 1.0)
        gate.partial_fit(negative, 0.0)

    pos = gate.decide(positive)
    neg = gate.decide(negative)
    assert gate.ready, 'gate should be ready after warmup updates'
    assert pos.should_delegate, 'positive sample should cross threshold'
    assert not neg.should_delegate, 'negative sample should stay below threshold'
    print(f'[PASS] test_linear_delegation_gate (pos={pos.probability:.2f}, neg={neg.probability:.2f})')


def test_low_rank_memory_adapter():
    """Low-rank memory adapter learns a reusable detached-state recall direction."""
    from cortex_core.agent_cloud import LowRankMemoryAdapter

    adapter = LowRankMemoryAdapter(input_dim=16, rank=4, lr=5e-2, warmup_steps=4, device='cpu')
    hidden = torch.tensor([1.0, 0.5] + [0.0] * 14)
    target = torch.tensor([0.0, 1.0] + [0.0] * 14)

    baseline = adapter.predict(hidden)
    baseline_sim = torch.nn.functional.cosine_similarity(
        baseline.unsqueeze(0), target.unsqueeze(0)
    ).item()

    for _ in range(16):
        adapter.partial_fit(hidden, target)

    pred = adapter.predict(hidden)
    pred_sim = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(0), target.unsqueeze(0)
    ).item()
    assert adapter.ready, 'adapter should be ready after warmup updates'
    assert pred_sim > baseline_sim + 0.25, f'expected improved similarity, got {baseline_sim:.3f} -> {pred_sim:.3f}'
    print(f'[PASS] test_low_rank_memory_adapter (sim={baseline_sim:.2f}->{pred_sim:.2f})')


def test_persistent_agent_cloud_isolates_memory():
    """Persistent agent cloud keeps identity memory isolated across agents."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu')
    cloud.ensure_agent('npc_blacksmith', role='npc', profile='Gruff blacksmith who tracks forged weapons.')
    cloud.ensure_agent('npc_healer', role='npc', profile='Quiet healer who tracks herbs and injuries.')

    smith_query = cloud.encode_text('forged iron sword for the ranger')
    healer_query = cloud.encode_text('mixed herbs for an injured traveler')

    cloud.remember_text(
        agent_id='npc_blacksmith',
        text='Forged an iron sword for the ranger.',
        hidden_state=smith_query,
        role='npc',
    )
    cloud.remember_text(
        agent_id='npc_healer',
        text='Prepared a bitter herb tonic for an injured traveler.',
        hidden_state=healer_query,
        role='npc',
    )

    smith_prompt = cloud.compose_prompt(
        'npc_blacksmith',
        task='Offer the ranger a status update about their weapon.',
        role_prompt='[System: Stay in character.]',
    )
    healer_prompt = cloud.compose_prompt(
        'npc_healer',
        task='Offer the traveler a status update about their treatment.',
        role_prompt='[System: Stay in character.]',
    )

    assert 'iron sword' in smith_prompt.lower(), smith_prompt
    assert 'herb tonic' not in smith_prompt.lower(), smith_prompt
    assert 'herb tonic' in healer_prompt.lower(), healer_prompt
    assert 'iron sword' not in healer_prompt.lower(), healer_prompt

    stats = cloud.population_stats()
    assert stats['agent_count'] == 2
    assert stats['total_episodes'] == 2
    print('[PASS] test_persistent_agent_cloud_isolates_memory')


def test_shared_manifold_recall_across_agents():
    """Shared manifold should let one agent benefit from another agent's bounded memory trace."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=8)
    cloud.ensure_agent('planner', role='planner', profile='Tracks architecture decisions.')
    cloud.ensure_agent('coder', role='coder', profile='Implements changes safely.')

    cloud.remember_text(
        agent_id='planner',
        text='Use idempotency keys on payment retries to avoid duplicate captures.',
        role='planner',
        source='design',
    )
    cloud.remember_text(
        agent_id='planner',
        text='Migration notes: old index names must stay unique during backfills.',
        role='planner',
        source='design',
    )

    prompt = cloud.compose_prompt(
        'coder',
        task='Implement payment retry idempotency in the checkout client.',
        role_prompt='[System: Write minimal safe code.]',
    )

    assert '[Shared Manifold]' in prompt, prompt
    assert 'idempotency keys' in prompt.lower(), prompt
    assert 'duplicate captures' in prompt.lower(), prompt
    print('[PASS] test_shared_manifold_recall_across_agents')


def test_shared_manifold_sqlite_store_sync():
    """Independent cloud instances should converge on one shared manifold store."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'shared_manifold.sqlite')
        writer = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8, shared_store_path=db_path)
        reader = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8, shared_store_path=db_path)

        writer.remember_shared_text(text='Alpha route state is green.', node_type='fact', source='writer_a')
        reader.remember_shared_text(text='Beta route state is amber.', node_type='fact', source='writer_b')

        writer_hits = writer.query_shared_manifold(query_text='alpha beta route state', top_k=4)
        reader_hits = reader.query_shared_manifold(query_text='alpha beta route state', top_k=4)

        writer_texts = {node.text for node in writer_hits}
        reader_texts = {node.text for node in reader_hits}
        assert 'Alpha route state is green.' in writer_texts, writer_texts
        assert 'Beta route state is amber.' in writer_texts, writer_texts
        assert writer_texts == reader_texts, (writer_texts, reader_texts)
    print('[PASS] test_shared_manifold_sqlite_store_sync')


def test_shared_manifold_hot_cache_materializes_turbo_kv():
    """The shared hot cache should persist a TurboQuant-compressed KV summary across sessions."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'shared_hot.sqlite')
        cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8, shared_store_path=db_path)
        cloud.remember_shared_text(text='Summarize the active route memory.', node_type='fact', source='writer')

        kv = [
            (torch.randn(1, 2, 64, 8), torch.randn(1, 2, 64, 8)),
            (torch.randn(1, 2, 64, 8), torch.randn(1, 2, 64, 8)),
        ]
        hot_state = cloud.materialize_shared_hot_cache(kv_landmarks=tuple(kv), turbo_bits=4, turbo_device='cpu')

        assert hot_state['kv_stats']['layer_count'] == 2, hot_state
        assert hot_state['kv_stats']['compressed_bytes'] > 0, hot_state
        assert hot_state['kv_stats']['compression_ratio'] > 1.0, hot_state
        assert '[Shared Hot Cache]' in hot_state['summary_text'], hot_state['summary_text']
        assert 'regions=1' in hot_state['summary_text'], hot_state['summary_text']
        assert 'largest_region=1' in hot_state['summary_text'], hot_state['summary_text']
        assert hot_state['hot_projection_id'], hot_state
        assert hot_state['hot_projection_node_id'], hot_state

        restored = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8, shared_store_path=db_path)
        restored_state = restored.get_shared_hot_state()
        restored_cache = restored.get_shared_hot_turbo_cache(device='cpu')
        restored_projection = restored._find_shared_node(projection_id=restored_state['hot_projection_id'])
        restored_projection_cache = restored.get_projection_residue(restored_state['hot_projection_id'], device='cpu')

        assert restored_state['kv_stats']['layer_count'] == 2, restored_state
        assert restored_cache is not None
        assert restored_projection is not None, restored_state
        assert restored_projection.node_type == 'projection_summary', restored_projection
        assert restored_projection_cache is not None
        assert restored_cache.memory_bytes() == hot_state['kv_stats']['compressed_bytes'], restored_state
    print('[PASS] test_shared_manifold_hot_cache_materializes_turbo_kv')


def test_context_manager_injects_shared_manifold_context():
    """Main prompts should be able to pull bounded shared-manifold context before decoding."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_core.cortex_memory import AutoCompactor, ContextManager
    from cortex_core.synapse import TopologicalSynapse

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=8)
    cloud.remember_shared_text(
        text='Use idempotency keys when retrying payment capture requests.',
        node_type='fact',
        source='shared_memory',
    )

    ctx = ContextManager(
        TopologicalSynapse(dim=32, device='cpu'),
        AutoCompactor(max_seq_len=128),
        shared_context_getter=cloud.build_shared_context,
    )
    enriched = ctx.enrich_prompt('Add payment retry idempotency to checkout.')

    assert '[Shared Manifold]' in enriched, enriched
    assert 'idempotency keys' in enriched.lower(), enriched
    assert 'regions=1' in enriched, enriched
    assert 'active_region=1' in enriched, enriched
    print('[PASS] test_context_manager_injects_shared_manifold_context')


def test_shared_manifold_plans_non_redundant_refresh():
    """Shared-manifold refresh planning should return fresh nodes and stop once they are consumed."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=8)
    cloud.remember_shared_text(
        text='Payment retries must use idempotency keys to avoid duplicate capture.',
        node_type='fact',
        source='shared_memory',
    )
    cloud.remember_shared_text(
        text='Expose retry count in telemetry to debug duplicate payment attempts.',
        node_type='fact',
        source='shared_memory',
    )

    refresh_text, fresh_nodes = cloud.plan_shared_injection(
        query_text='Implement payment retry idempotency and retry telemetry.',
        used_texts=set(),
        top_k=2,
    )
    assert '[Shared Recall]' in refresh_text, refresh_text
    assert len(fresh_nodes) >= 1, fresh_nodes

    used = {node.text for node in fresh_nodes}
    refresh_text_2, fresh_nodes_2 = cloud.plan_shared_injection(
        query_text='Implement payment retry idempotency and retry telemetry.',
        used_texts=used,
        top_k=2,
    )
    assert refresh_text_2 == '', refresh_text_2
    assert fresh_nodes_2 == [], fresh_nodes_2
    print('[PASS] test_shared_manifold_plans_non_redundant_refresh')


def test_shared_manifold_task_board_stays_compact_for_prompting():
    """Task-board context should surface task specs and patch ops without leaking ephemeral claim/result chatter."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=16)
    cloud.publish_task_spec(
        task_id='retry_replay_token_simple',
        summary='Repair the retry header helper so retries stay replay-safe.',
        recent_text='A flaky payment path is creating duplicate charges after retries.',
        signature='def build_retry_headers(charge_id, replay_token):',
        acceptance_criteria=['Keep the existing charge id header unchanged.'],
        agent_id='planner',
        source='test',
    )
    cloud.publish_task_note(
        task_id='retry_replay_token_simple',
        note_text='Store the replay token under the "Replay-Safety-Token" header.',
        sequence_index=10,
        agent_id='planner',
        source='test',
    )
    cloud.publish_task_patch(
        task_id='retry_replay_token_simple',
        patch_name='add_replay_safety_token_header',
        old_text='return {"X-Charge-Id": charge_id}',
        new_text='return {"X-Charge-Id": charge_id, "Replay-Safety-Token": replay_token}',
        trigger_terms=['Replay-Safety-Token'],
        sequence_index=100,
        agent_id='planner',
        source='test',
    )
    cloud.claim_task(task_id='retry_replay_token_simple', agent_id='coder', source='test')
    cloud.publish_task_result(
        task_id='retry_replay_token_simple',
        agent_id='coder',
        status='passed',
        selected_patches=['add_replay_safety_token_header'],
        result_text='apply=add_replay_safety_token_header; passed=true',
        source='test',
    )

    context = cloud.build_task_board_context(
        'Repair the retry header helper so retries stay replay-safe.',
        top_k=1,
        agent_id='coder',
    )
    refresh_text, fresh_nodes = cloud.plan_shared_injection(
        query_text='Repair the retry header helper so retries stay replay-safe.',
        used_texts=set(),
        top_k=1,
        agent_id='coder',
    )

    assert '[Task Board]' in context, context
    assert '[Task: retry_replay_token_simple]' in context, context
    assert 'patch=add_replay_safety_token_header' in context, context
    assert 'note=Store the replay token under the "Replay-Safety-Token" header.' in context, context
    assert 'claim' not in context.lower(), context
    assert 'passed=true' not in context.lower(), context
    assert '[Task Board Recall]' in refresh_text, refresh_text
    assert len(fresh_nodes) == 3, fresh_nodes
    print('[PASS] test_shared_manifold_task_board_stays_compact_for_prompting')


def test_shared_manifold_structural_edges_connect_related_nodes():
    """Explicit node relations should join semantically distant nodes into one local manifold region."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=12)
    artifact = cloud.remember_shared_text(
        text='Release artifact shard A is published at bundle path alpha_release_artifact.',
        node_type='artifact',
        source='builder',
        metadata={'keywords': ['release', 'artifact', 'bundle']},
        refresh_hot_state=False,
    )
    checklist = cloud.remember_shared_text(
        text='Operations launch checklist must confirm the final publish bundle before rollout.',
        node_type='task',
        source='planner',
        metadata={
            'keywords': ['operations', 'launch', 'checklist'],
            'depends_on': [artifact.node_id],
        },
        refresh_hot_state=False,
    )
    cloud.remember_shared_text(
        text='Greenhouse irrigation notes stay in drawer 9 beside the seed ledger.',
        node_type='fact',
        source='scribe',
        metadata={'keywords': ['greenhouse', 'irrigation', 'drawer']},
        refresh_hot_state=False,
    )

    stats = cloud.shared_manifold_stats()
    view = cloud._build_shared_topology_view(list(cloud._shared_nodes))
    hits = cloud.query_shared_manifold(query_text='Need the operations launch checklist for rollout.', top_k=2)
    texts = {node.text for node in hits}

    assert stats['component_count'] == 2, stats
    assert stats['structural_edge_count'] >= 1, stats
    assert 'depends_on' in view.edge_types.get((0, 1), []), view.edge_types
    assert checklist.text in texts, texts
    assert artifact.text in texts, texts
    print('[PASS] test_shared_manifold_structural_edges_connect_related_nodes')


def test_shared_manifold_energy_deformation_biases_retrieval():
    """Positive manifold energy should locally deform retrieval toward the energized region."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=12)
    artifact = cloud.remember_shared_text(
        text='Release artifact shard A is published at bundle path alpha_release_artifact.',
        node_type='artifact',
        source='builder',
        metadata={'keywords': ['release', 'artifact', 'bundle']},
        refresh_hot_state=False,
    )
    checklist = cloud.remember_shared_text(
        text='Operations launch checklist must confirm the final publish bundle before rollout.',
        node_type='task',
        source='planner',
        metadata={
            'keywords': ['operations', 'launch', 'checklist'],
            'depends_on': [artifact.node_id],
        },
        refresh_hot_state=False,
    )
    cloud.remember_shared_text(
        text='Greenhouse irrigation notes stay in drawer 9 beside the seed ledger.',
        node_type='fact',
        source='scribe',
        metadata={'keywords': ['greenhouse', 'irrigation', 'drawer']},
        refresh_hot_state=False,
    )

    before = cloud.query_shared_manifold(query_text='Need the operations launch checklist for rollout.', top_k=1)
    report = cloud.deform_manifold(artifact.node_id, delta=4.0, max_depth=1, edge_decay=0.85)
    after = cloud.query_shared_manifold(query_text='Need the operations launch checklist for rollout.', top_k=1)
    artifact_node = cloud._find_shared_node(node_id=artifact.node_id)
    checklist_node = cloud._find_shared_node(node_id=checklist.node_id)
    stats = cloud.shared_manifold_stats()

    assert before and before[0].node_id == checklist.node_id, before
    assert report['affected_node_count'] >= 2, report
    assert after and after[0].node_id == artifact.node_id, after
    assert artifact_node is not None and checklist_node is not None
    assert cloud._node_energy(artifact_node) > cloud._node_energy(checklist_node) > 0.0
    assert stats['energized_node_count'] >= 2, stats
    assert stats['energy_peak'] > 0.0, stats
    print('[PASS] test_shared_manifold_energy_deformation_biases_retrieval')


def test_shared_manifold_maintenance_decays_energy():
    """Background-style maintenance should decay manifold energy instead of letting it only accumulate."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8)
    node = cloud.remember_shared_text(
        text='North tower lantern registry stays on shelf seven.',
        node_type='fact',
        source='scribe',
        refresh_hot_state=False,
    )
    cloud.deform_manifold(node.node_id, delta=2.0, max_depth=0, refresh_hot_state=False)
    seeded = cloud._find_shared_node(node_id=node.node_id)
    assert seeded is not None
    assert abs(cloud._node_energy(seeded) - 2.0) < 1e-6, seeded.metadata

    report = cloud.run_manifold_maintenance(
        energy_decay=0.5,
        energy_floor=0.01,
        refresh_hot_state=False,
    )
    decayed = cloud._find_shared_node(node_id=node.node_id)

    assert decayed is not None
    assert abs(cloud._node_energy(decayed) - 1.0) < 1e-6, decayed.metadata
    assert report['updated_nodes'] == 1, report
    assert abs(report['energy_abs_before'] - 2.0) < 1e-6, report
    assert abs(report['energy_abs_after'] - 1.0) < 1e-6, report
    print('[PASS] test_shared_manifold_maintenance_decays_energy')


def test_shared_manifold_energy_snapshot_roundtrip():
    """Node energy should survive a cloud snapshot roundtrip because it is part of manifold state."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8)
    ledger = cloud.remember_shared_text(
        text='Ledger alpha stays under the west archive seal.',
        node_type='fact',
        source='scribe',
        refresh_hot_state=False,
    )
    handoff = cloud.remember_shared_text(
        text='Archive handoff depends on ledger alpha before seal verification.',
        node_type='task',
        source='planner',
        metadata={'supports': [ledger.node_id]},
        refresh_hot_state=False,
    )
    cloud.deform_manifold(handoff.node_id, delta=1.5, max_depth=1, edge_decay=0.8, refresh_hot_state=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'energy_cloud.pt')
        cloud.save(snapshot_path)

        restored = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8)
        restored.load(snapshot_path)

    restored_ledger = restored._find_shared_node(node_id=ledger.node_id)
    restored_handoff = restored._find_shared_node(node_id=handoff.node_id)
    stats = restored.shared_manifold_stats()

    assert restored_ledger is not None and restored_handoff is not None
    assert restored._node_energy(restored_handoff) > 0.0
    assert restored._node_energy(restored_ledger) > 0.0
    assert stats['energized_node_count'] >= 2, stats
    print('[PASS] test_shared_manifold_energy_snapshot_roundtrip')


def test_shared_manifold_task_result_feedback_energizes_task_board():
    """Task-board results should reinforce the task region when automatic energy feedback is enabled."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(
        hidden_dim=16,
        device='cpu',
        shared_manifold_capacity=12,
        shared_energy_feedback_enabled=True,
    )
    spec = cloud.publish_task_spec(
        task_id='checkout_retry',
        summary='Repair checkout retries so duplicate captures collapse safely.',
        recent_text='Duplicate captures appear after flaky payment retries.',
        signature='build_retry_request',
        acceptance_criteria=['Keep the idempotency key stable across attempts.'],
    )
    note = cloud.publish_task_note(
        task_id='checkout_retry',
        note_text='Reuse the same retry key and attach attempt telemetry.',
        sequence_index=1,
    )
    patch = cloud.publish_task_patch(
        task_id='checkout_retry',
        patch_name='persist_retry_key',
        old_text='headers = {"X-Charge-Id": charge_id}',
        new_text='headers = {"X-Charge-Id": charge_id, "Idempotency-Key": idempotency_key}',
        trigger_terms=['duplicate capture', 'retry telemetry'],
        sequence_index=2,
    )
    result = cloud.publish_task_result(
        task_id='checkout_retry',
        agent_id='coder',
        result_text='Applied the stable retry key patch and preserved telemetry.',
        status='passed',
        selected_patches=['persist_retry_key'],
        score=1.0,
    )

    spec_node = cloud._find_shared_node(node_id=spec.node_id)
    note_node = cloud._find_shared_node(node_id=note.node_id)
    patch_node = cloud._find_shared_node(node_id=patch.node_id)
    result_node = cloud._find_shared_node(node_id=result.node_id)
    stats = cloud.shared_manifold_stats()

    assert spec_node is not None and note_node is not None and patch_node is not None and result_node is not None
    assert cloud._node_energy(result_node) > 0.0, result_node.metadata
    assert cloud._node_energy(patch_node) > 0.0, patch_node.metadata
    assert cloud._node_energy(note_node) > 0.0, note_node.metadata
    assert cloud._node_energy(spec_node) > 0.0, spec_node.metadata
    assert stats['energized_node_count'] >= 4, stats
    print('[PASS] test_shared_manifold_task_result_feedback_energizes_task_board')


def test_shared_manifold_projection_landmark_carries_kv_residue():
    """Materialized projections should become literal manifold nodes and optionally carry compressed KV residue."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=12)
    header = cloud.remember_shared_text(
        text='Payment retries use retry_header=X-Payment-Retry-Key.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['payment_retry', 'retry_header']},
    )
    seal = cloud.remember_shared_text(
        text='Payment replay seal field is replay_token_px17.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['payment_retry', 'replay_token_px17']},
    )
    kv = ((torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),)

    projection = cloud.materialize_projection(
        query_text='Which retry header and replay seal field belong to payment retries?',
        top_k=2,
        kv_landmarks=kv,
        turbo_bits=4,
        turbo_device='cpu',
    )
    projection_node = cloud._find_shared_node(node_id=projection['node_id'])
    projection_cache = cloud.get_projection_residue(projection['projection_id'], device='cpu')
    stats = cloud.shared_manifold_stats()
    view = cloud._build_shared_topology_view(list(cloud._shared_nodes))
    projection_index = next(index for index, node in enumerate(cloud._shared_nodes) if node.node_id == projection['node_id'])

    assert projection['projection_id'], projection
    assert projection_node is not None, projection
    assert projection_node.node_type == 'projection_summary', projection_node
    assert set(projection_node.metadata['projection_node_ids']) == {header.node_id, seal.node_id}, projection_node.metadata
    assert projection_cache is not None
    assert projection_cache.memory_bytes() > 0
    assert stats['projection_node_count'] >= 1, stats
    assert stats['projection_residue_count'] >= 1, stats
    assert 'projection_member' in view.edge_types.get((0, projection_index), []) or 'projection_member' in view.edge_types.get((1, projection_index), []), view.edge_types
    print('[PASS] test_shared_manifold_projection_landmark_carries_kv_residue')


def test_shared_manifold_projection_snapshot_roundtrip():
    """Projection landmarks and their KV residues should survive a cloud snapshot roundtrip."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Bridge checkpoint token is cedar-bridge-17.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['bridge_token']},
    )
    cloud.remember_shared_text(
        text='Bridge route handoff lands at cedar-checkpoint.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['bridge_route']},
    )
    kv = ((torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),)
    projection = cloud.materialize_projection(
        query_text='Summarize the bridge route handoff neighborhood.',
        top_k=2,
        kv_landmarks=kv,
        turbo_bits=4,
        turbo_device='cpu',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'projection_cloud.pt')
        cloud.save(snapshot_path)

        restored = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=12)
        restored.load(snapshot_path)

    projection_node = restored._find_shared_node(node_id=projection['node_id'])
    projection_cache = restored.get_projection_residue(projection['projection_id'], device='cpu')
    stats = restored.shared_manifold_stats()

    assert projection_node is not None, projection
    assert projection_node.node_id == projection['node_id'], projection_node
    assert projection_cache is not None
    assert projection_cache.memory_bytes() > 0
    assert stats['projection_node_count'] >= 1, stats
    assert stats['projection_residue_count'] >= 1, stats
    print('[PASS] test_shared_manifold_projection_snapshot_roundtrip')


def test_shared_manifold_prefers_projection_context_for_refresh():
    """Projection summaries should become the preferred compact recall surface once they exist."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Rotate the session signing key version when auth sessions are reissued.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['session', 'signing', 'key', 'version'], 'entity_refs': ['auth_session', 'signing_key_version']},
    )
    cloud.remember_shared_text(
        text='Attach the rotation key version to the auth session payload before persisting it.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['session', 'rotation', 'key', 'version'], 'entity_refs': ['auth_session', 'rotation_key_version']},
    )
    kv = ((torch.randn(1, 2, 24, 8), torch.randn(1, 2, 24, 8)),)
    projection = cloud.materialize_projection(
        query_text='Need the auth session rotation key version details.',
        top_k=2,
        projection_kind='runtime_decode',
        kv_landmarks=kv,
        turbo_bits=4,
        turbo_device='cpu',
    )

    context = cloud.build_shared_context('Need the auth session rotation key version details.', top_k=2)
    refresh_text, fresh_nodes = cloud.plan_shared_injection(
        query_text='Need the auth session rotation key version details.',
        used_texts=set(),
        top_k=2,
    )

    assert '[Shared Projection]' in context, context
    assert projection['summary_text'] in context, context
    assert '[Shared Projection Recall]' in refresh_text, refresh_text
    assert any(node.node_type == 'projection_summary' for node in fresh_nodes), fresh_nodes
    print('[PASS] test_shared_manifold_prefers_projection_context_for_refresh')


def test_shared_manifold_regions_stay_local_to_query():
    """Region-aware manifold retrieval should stay within the active component when enough local context exists."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Payment retries need idempotency keys in checkout.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['payment', 'retry', 'idempotency'], 'entity_refs': ['payment', 'checkout']},
    )
    cloud.remember_shared_text(
        text='Duplicate payment capture happens without retry guards.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['payment', 'capture', 'retry'], 'entity_refs': ['payment', 'capture']},
    )
    cloud.remember_shared_text(
        text='The caravan crossed the east bridge at dusk.',
        node_type='fact',
        source='scout',
        metadata={'keywords': ['caravan', 'bridge', 'dusk'], 'entity_refs': ['caravan', 'east_bridge']},
    )
    cloud.remember_shared_text(
        text='The caravan turned south toward the old road.',
        node_type='fact',
        source='scout',
        metadata={'keywords': ['caravan', 'south', 'road'], 'entity_refs': ['caravan', 'old_road']},
    )

    stats = cloud.shared_manifold_stats()
    assert stats['component_count'] == 2, stats
    assert stats['largest_component_size'] == 2, stats

    hits = cloud.query_shared_manifold(
        query_text='Implement payment retry idempotency to avoid duplicate capture.',
        top_k=2,
    )
    texts = [node.text.lower() for node in hits]
    assert len(texts) == 2, texts
    assert all('payment' in text for text in texts), texts
    assert all('caravan' not in text for text in texts), texts

    context = cloud.build_shared_context(
        'Implement payment retry idempotency to avoid duplicate capture.',
        top_k=2,
    )
    assert 'regions=2' in context, context
    assert 'active_region=2' in context, context
    print('[PASS] test_shared_manifold_regions_stay_local_to_query')


def test_shared_manifold_preserves_bridge_nodes_for_recall():
    """Bridge nodes should be retained so recall can cross adjacent regions without flattening the manifold."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=32, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Alpha vault stores the original field notes.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['alpha', 'vault'], 'entity_refs': ['alpha', 'vault']},
    )
    cloud.remember_shared_text(
        text='Relay the alpha findings into beta route handoff.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['alpha', 'beta'], 'entity_refs': ['alpha', 'beta']},
    )
    cloud.remember_shared_text(
        text='Beta route depends on the final handoff checkpoint.',
        node_type='fact',
        source='planner',
        metadata={'keywords': ['beta', 'route'], 'entity_refs': ['beta', 'route']},
    )

    stats = cloud.shared_manifold_stats()
    assert stats['component_count'] == 1, stats
    assert stats['largest_component_size'] == 3, stats
    assert stats['bridge_count'] >= 1, stats

    hits = cloud.query_shared_manifold(query_text='Need the beta route handoff details.', top_k=2)
    texts = {node.text for node in hits}
    assert 'Beta route depends on the final handoff checkpoint.' in texts, texts
    assert 'Relay the alpha findings into beta route handoff.' in texts, texts

    context = cloud.build_shared_context('Need the beta route handoff details.', top_k=2)
    assert 'bridges=1' in context, context
    print('[PASS] test_shared_manifold_preserves_bridge_nodes_for_recall')


def test_engine_refreshes_shared_manifold_into_kv():
    """CortexEngine should be able to inject fresh shared-manifold recall into live decode memory."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    class DummyTokenBatch:
        def __init__(self, text: str):
            self.input_ids = torch.tensor([[len(text) % 17 + 1]], dtype=torch.long)

    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text: str, return_tensors='pt'):
            self.calls.append(text)
            return DummyTokenBatch(text)

    class DummyOutput:
        def __init__(self, past_key_values):
            self.past_key_values = past_key_values

    class DummyModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, past_key_values=None, output_hidden_states=False):
            self.calls.append({
                'input_ids': input_ids.clone(),
                'past_key_values': past_key_values,
            })
            return DummyOutput({'memory_steps': len(self.calls)})

    engine = object.__new__(CortexEngine)
    engine.device = 'cpu'
    engine.tokenizer = DummyTokenizer()
    engine.model = DummyModel()
    engine.shared_manifold_enabled = True
    engine.shared_manifold_trace = []
    engine.shared_manifold_prompt_hits = 0
    engine.shared_manifold_prompt_misses = 0
    engine.shared_manifold_runtime_refreshes = 0
    engine.shared_manifold_nodes_consumed = 0
    engine.shared_manifold_refresh_top_k = 2
    engine.agent_cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8)
    engine.agent_cloud.remember_shared_text(
        text='Retry payment capture with idempotency keys.',
        node_type='fact',
        source='shared_memory',
    )

    used = set()
    past_key_values, refresh_count = engine._maybe_refresh_shared_manifold(
        base_prompt='Implement payment retry safety.',
        recent_text='Need to stop duplicate capture on retries.',
        used_texts=used,
        past_key_values=None,
    )
    assert refresh_count == 1, refresh_count
    assert past_key_values == {'memory_steps': 1}, past_key_values
    assert any('[Shared:' in text for text in engine.tokenizer.calls), engine.tokenizer.calls
    metrics = engine.get_shared_manifold_metrics()
    assert metrics['runtime_refreshes'] == 1, metrics
    assert metrics['nodes_consumed'] == 1, metrics
    assert metrics['trace_length'] == 1, metrics

    past_key_values_2, refresh_count_2 = engine._maybe_refresh_shared_manifold(
        base_prompt='Implement payment retry safety.',
        recent_text='Need to stop duplicate capture on retries.',
        used_texts=used,
        past_key_values=past_key_values,
    )
    assert refresh_count_2 == 0, refresh_count_2
    assert past_key_values_2 == past_key_values, (past_key_values_2, past_key_values)
    assert len(engine.model.calls) == 1, engine.model.calls
    print('[PASS] test_engine_refreshes_shared_manifold_into_kv')


def test_engine_prompt_context_feedback_energizes_nodes():
    """Prompt-context reads should lightly energize the recalled manifold region when feedback is enabled."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    engine = object.__new__(CortexEngine)
    engine.shared_manifold_enabled = True
    engine.shared_manifold_energy_feedback_enabled = True
    engine.shared_manifold_trace = []
    engine.shared_manifold_prompt_hits = 0
    engine.shared_manifold_prompt_misses = 0
    engine.shared_manifold_runtime_refreshes = 0
    engine.shared_manifold_nodes_consumed = 0
    engine.agent_cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=8)
    engine.agent_cloud.shared_energy_prompt_delta = 0.35

    node = engine.agent_cloud.remember_shared_text(
        text='Retry payment capture with idempotency keys and attempt telemetry.',
        node_type='fact',
        source='planner',
    )

    context = CortexEngine._build_shared_manifold_context(
        engine,
        'Need idempotency keys for payment retries.',
        top_k=1,
    )
    recalled = engine.agent_cloud._find_shared_node(node_id=node.node_id)

    assert '[Shared Manifold]' in context, context
    assert recalled is not None
    assert engine.agent_cloud._node_energy(recalled) > 0.0, recalled.metadata
    assert engine.shared_manifold_trace and engine.shared_manifold_trace[0]['stage'] == 'prompt_context', engine.shared_manifold_trace
    assert node.node_id in engine.shared_manifold_trace[0].get('node_ids', []), engine.shared_manifold_trace
    print('[PASS] test_engine_prompt_context_feedback_energizes_nodes')


def test_engine_uses_shared_hot_cache_for_workers():
    """Side-agent landmark resolution should fall back to the persisted TurboQuant hot cache."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    class DummySynapse:
        def get_landmarks(self):
            return None

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'worker_hot.sqlite')
        writer = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_store_path=db_path)
        kv = ((torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),)
        writer.materialize_shared_hot_cache(kv_landmarks=kv, turbo_bits=4, turbo_device='cpu')

        reader = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_store_path=db_path)
        engine = object.__new__(CortexEngine)
        engine.device = 'cpu'
        engine.synapse = DummySynapse()
        engine.agent_cloud = reader
        engine.prefer_hot_cache_for_workers = True
        engine._turbo_cache = None

        landmarks = CortexEngine._resolve_worker_landmarks(engine)
        assert landmarks is not None, landmarks
        assert len(landmarks) == 1, landmarks
        assert tuple(landmarks[0][0].shape) == tuple(kv[0][0].shape), (landmarks[0][0].shape, kv[0][0].shape)
    print('[PASS] test_engine_uses_shared_hot_cache_for_workers')


def test_engine_seeds_projection_residue_from_hot_cache():
    """Main decode should be able to materialize and seed a query-specific projection residue from the shared hot cache."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Auth sessions must carry the signing key version field.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['auth_session', 'signing_key_version']},
    )
    cloud.remember_shared_text(
        text='Rotate session signing keys during reissue to preserve replay safety.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['auth_session', 'replay_safety']},
    )
    kv = ((torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8)),)
    cloud.materialize_shared_hot_cache(kv_landmarks=kv, turbo_bits=4, turbo_device='cpu')

    engine = object.__new__(CortexEngine)
    engine.device = 'cpu'
    engine.agent_cloud = cloud
    engine.shared_manifold_enabled = True
    engine.shared_manifold_projection_top_k = 4
    engine.shared_manifold_trace = []
    engine.shared_manifold_nodes_consumed = 0
    engine.turbo_quant_bits = 4

    used = set()
    past_key_values, seeded_count = CortexEngine._seed_shared_projection_cache(
        engine,
        query_text='Need the auth session signing key version details.',
        used_texts=used,
        past_key_values=None,
    )
    projection = cloud.resolve_shared_projection(
        query_text='Need the auth session signing key version details.',
        top_k=2,
        require_residue=True,
    )

    assert seeded_count >= 1, seeded_count
    assert past_key_values is not None, past_key_values
    assert tuple(past_key_values.key_cache[0].shape) == tuple(kv[0][0].shape), (past_key_values.key_cache[0].shape, kv[0][0].shape)
    assert projection is not None and projection['has_residue'], projection
    assert any(event['stage'] == 'projection_seed' for event in engine.shared_manifold_trace), engine.shared_manifold_trace
    print('[PASS] test_engine_seeds_projection_residue_from_hot_cache')


def test_engine_projection_seed_feedback_energizes_projection_summary():
    """Projection-seed reuse should energize the projection summary node when runtime feedback is enabled."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_manifold_capacity=12)
    cloud.remember_shared_text(
        text='Auth sessions must carry the signing key version field.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['auth_session', 'signing_key_version']},
    )
    cloud.remember_shared_text(
        text='Rotate session signing keys during reissue to preserve replay safety.',
        node_type='fact',
        source='planner',
        metadata={'entity_refs': ['auth_session', 'replay_safety']},
    )
    kv = ((torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8)),)
    cloud.materialize_shared_hot_cache(kv_landmarks=kv, turbo_bits=4, turbo_device='cpu')
    cloud.shared_energy_projection_delta = 0.4

    engine = object.__new__(CortexEngine)
    engine.device = 'cpu'
    engine.agent_cloud = cloud
    engine.shared_manifold_enabled = True
    engine.shared_manifold_energy_feedback_enabled = True
    engine.shared_manifold_projection_top_k = 4
    engine.shared_manifold_trace = []
    engine.shared_manifold_nodes_consumed = 0
    engine.turbo_quant_bits = 4

    used = set()
    _, seeded_count = CortexEngine._seed_shared_projection_cache(
        engine,
        query_text='Need the auth session signing key version details.',
        used_texts=used,
        past_key_values=None,
    )
    projection = cloud.resolve_shared_projection(
        query_text='Need the auth session signing key version details.',
        top_k=2,
        require_residue=True,
    )
    projection_node = cloud._find_shared_node(projection_id=projection['projection_id']) if projection is not None else None

    assert seeded_count >= 1, seeded_count
    assert projection is not None and projection_node is not None, projection
    assert cloud._node_energy(projection_node) > 0.0, projection_node.metadata
    assert any(event['stage'] == 'projection_seed' for event in engine.shared_manifold_trace), engine.shared_manifold_trace
    print('[PASS] test_engine_projection_seed_feedback_energizes_projection_summary')


def test_engine_prefers_projection_residue_for_workers():
    """Worker fallback should prefer a query-matched projection residue over the generic shared hot cache."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    class DummySynapse:
        def get_landmarks(self):
            return None

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'worker_projection.sqlite')
        writer = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_store_path=db_path)
        writer.remember_shared_text(
            text='Auth sessions persist a signing key version field.',
            node_type='fact',
            source='planner',
            metadata={'entity_refs': ['auth_session', 'signing_key_version']},
        )
        writer.remember_shared_text(
            text='Session rotation must advance the signing key version before handoff.',
            node_type='fact',
            source='planner',
            metadata={'entity_refs': ['auth_session', 'rotation']},
        )

        hot_kv = ((torch.randn(1, 2, 16, 8), torch.randn(1, 2, 16, 8)),)
        writer.materialize_shared_hot_cache(kv_landmarks=hot_kv, turbo_bits=4, turbo_device='cpu')

        projection_kv = ((torch.randn(1, 2, 28, 8), torch.randn(1, 2, 28, 8)),)
        writer.materialize_projection(
            query_text='Need the auth session signing key version details.',
            top_k=2,
            projection_kind='worker_context',
            kv_landmarks=projection_kv,
            turbo_bits=4,
            turbo_device='cpu',
        )

        reader = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_store_path=db_path)
        engine = object.__new__(CortexEngine)
        engine.device = 'cpu'
        engine.synapse = DummySynapse()
        engine.agent_cloud = reader
        engine.prefer_hot_cache_for_workers = True
        engine.shared_manifold_enabled = True
        engine.shared_manifold_projection_top_k = 4
        engine.turbo_quant_bits = 4
        engine._turbo_cache = None

        landmarks = CortexEngine._resolve_worker_landmarks(
            engine,
            query_text='Need the auth session signing key version details.',
        )
        assert landmarks is not None, landmarks
        assert tuple(landmarks[0][0].shape) == tuple(projection_kv[0][0].shape), (landmarks[0][0].shape, projection_kv[0][0].shape)
    print('[PASS] test_engine_prefers_projection_residue_for_workers')


def test_engine_memory_accounting_reports_hot_cache():
    """Engine memory accounting should include shared hot-cache KV stats and live TurboQuant bytes."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_engine import CortexEngine

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'accounting_hot.sqlite')
        cloud = PersistentAgentCloud(hidden_dim=16, device='cpu', shared_store_path=db_path)
        kv = ((torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),)
        cloud.materialize_shared_hot_cache(kv_landmarks=kv, turbo_bits=4, turbo_device='cpu')

        engine = object.__new__(CortexEngine)
        engine.model = TinyModel()
        engine.agent_cloud = cloud
        engine.turbo_quant_bits = 4
        engine._turbo_cache = cloud.get_shared_hot_turbo_cache(device='cpu')

        accounting = CortexEngine.get_memory_accounting(engine)
        assert accounting['model_total_bytes'] > 0, accounting
        assert accounting['shared_hot_kv']['compressed_bytes'] > 0, accounting
        assert accounting['live_turbo_cache_bytes'] == accounting['shared_hot_kv']['compressed_bytes'], accounting
    print('[PASS] test_engine_memory_accounting_reports_hot_cache')


def test_shared_manifold_benchmark_pipeline():
    """Internal benchmark pipeline should show a clear manifold-on recall advantage on deterministic scenarios."""
    from cortex_benchmarks.benchmark_shared_manifold import compare_pipeline

    report = compare_pipeline()
    enabled = report['enabled']['aggregate']
    disabled = report['disabled']['aggregate']

    assert enabled['prompt_hit_rate'] > disabled['prompt_hit_rate'], report
    assert enabled['refresh_hit_rate'] > disabled['refresh_hit_rate'], report
    print('[PASS] test_shared_manifold_benchmark_pipeline')


def test_shared_manifold_coding_slice_pipeline():
    """Executable coding slice should show a clear manifold-on pass-rate advantage."""
    from cortex_benchmarks.benchmark_shared_manifold import compare_coding_slice

    report = compare_coding_slice()
    enabled = report['enabled']['aggregate']
    disabled = report['disabled']['aggregate']

    assert enabled['pass_rate'] > disabled['pass_rate'], report
    assert enabled['avg_repairs_applied'] > disabled['avg_repairs_applied'], report
    print('[PASS] test_shared_manifold_coding_slice_pipeline')


def test_shared_manifold_topology_slice_pipeline():
    """Topology slice should beat the flat global baseline on leakage and bridge recall."""
    from cortex_benchmarks.benchmark_shared_manifold import compare_topology_slice

    report = compare_topology_slice()
    aggregate = report['aggregate']

    assert aggregate['component_accuracy_rate'] == 1.0, report
    assert aggregate['active_region_accuracy_rate'] == 1.0, report
    assert aggregate['topology_expected_recall_rate'] >= aggregate['flat_expected_recall_rate'], report
    assert aggregate['topology_bridge_recall_rate'] > aggregate['flat_bridge_recall_rate'], report
    assert aggregate['topology_leakage_rate'] < aggregate['flat_leakage_rate'], report
    print('[PASS] test_shared_manifold_topology_slice_pipeline')


def test_real_coding_slice_threads_energy_feedback_flag():
    """Real coding compare should thread the energy-feedback toggle into engine construction and per-side probes."""
    import cortex_benchmarks.benchmark_shared_manifold as bench

    build_calls = []
    probe_calls = []

    class FakeEngine:
        def __init__(self):
            self.model = None
            self.device = 'cpu'

    original_build = bench.build_real_engine
    original_probe = bench.run_real_coding_probe
    try:
        def fake_build_real_engine(*, enable_shared_manifold=True, enable_energy_feedback=False, model_id=None, device=None, shared_store_path=None, shared_store_cache_key='default'):
            build_calls.append({
                'enable_shared_manifold': enable_shared_manifold,
                'enable_energy_feedback': enable_energy_feedback,
                'device': device,
            })
            return FakeEngine()

        def fake_run_real_coding_probe(enable_shared_manifold, tasks=None, *, engine=None, enable_energy_feedback=False, model_id=None, device=None, max_tokens=160):
            probe_calls.append({
                'enable_shared_manifold': enable_shared_manifold,
                'enable_energy_feedback': enable_energy_feedback,
                'engine': engine,
                'task_count': len(list(tasks or [])),
            })
            return {
                'enabled': enable_shared_manifold,
                'aggregate': {'task_count': len(list(tasks or [])), 'pass_rate': 1.0 if enable_shared_manifold else 0.0},
                'tasks': [],
                'model_id': None,
                'device': 'cpu',
                'energy_feedback_enabled': enable_energy_feedback,
            }

        bench.build_real_engine = fake_build_real_engine
        bench.run_real_coding_probe = fake_run_real_coding_probe

        report = bench.compare_real_coding_slice(
            tasks=[object()],
            device='cpu',
            enable_energy_feedback=True,
        )
    finally:
        bench.build_real_engine = original_build
        bench.run_real_coding_probe = original_probe

    assert len(build_calls) == 1, build_calls
    assert build_calls[0]['enable_shared_manifold'] is True, build_calls
    assert build_calls[0]['enable_energy_feedback'] is True, build_calls
    assert len(probe_calls) == 2, probe_calls
    assert all(call['enable_energy_feedback'] is True for call in probe_calls), probe_calls
    assert {call['enable_shared_manifold'] for call in probe_calls} == {True, False}, probe_calls
    assert report['energy_feedback_enabled'] is True, report
    print('[PASS] test_real_coding_slice_threads_energy_feedback_flag')


def test_real_energy_reuse_slice_threads_energy_feedback_flag():
    """Targeted real energy-reuse compare should thread the energy-feedback toggle into engine construction and task runs."""
    import cortex_benchmarks.benchmark_shared_manifold as bench

    build_calls = []
    task_calls = []

    class FakeTask:
        def __init__(self, name):
            self.name = name

    class FakeEngine:
        def __init__(self, *, enable_energy_feedback):
            self.model = None
            self.device = 'cpu'
            self.shared_manifold_enabled = True
            self.shared_manifold_energy_feedback_enabled = enable_energy_feedback

    original_build = bench.build_real_engine
    original_run = bench.run_real_energy_reuse_task
    try:
        def fake_build_real_engine(*, enable_shared_manifold=True, enable_energy_feedback=False, model_id=None, device=None, shared_store_path=None, shared_store_cache_key='default'):
            build_calls.append({
                'enable_shared_manifold': enable_shared_manifold,
                'enable_energy_feedback': enable_energy_feedback,
                'device': device,
            })
            return FakeEngine(enable_energy_feedback=enable_energy_feedback)

        def fake_run_real_energy_reuse_task(engine, *, task):
            task_calls.append({
                'task_name': task.name,
                'energy_feedback_enabled': getattr(engine, 'shared_manifold_energy_feedback_enabled', None),
            })
            energy_peak = 0.6 if getattr(engine, 'shared_manifold_energy_feedback_enabled', False) else 0.0
            return {
                'name': task.name,
                'primer_target_hit_rate': 1.0,
                'followup_prompt_hit': True,
                'followup_target_hit': True,
                'followup_patch_hit': True,
                'followup_selected_task_id': 'target_task',
                'distractor_task_ids': ['distractor_task'],
                'shared_manifold_stats': {'energy_peak': energy_peak, 'energy_abs_total': energy_peak},
            }

        bench.build_real_engine = fake_build_real_engine
        bench.run_real_energy_reuse_task = fake_run_real_energy_reuse_task

        report = bench.compare_real_energy_reuse_slice(
            tasks=[FakeTask('energy_task')],
            device='cpu',
            enable_energy_feedback=True,
        )
    finally:
        bench.build_real_engine = original_build
        bench.run_real_energy_reuse_task = original_run

    assert len(build_calls) == 1, build_calls
    assert build_calls[0]['enable_shared_manifold'] is True, build_calls
    assert build_calls[0]['enable_energy_feedback'] is True, build_calls
    assert len(task_calls) == 1, task_calls
    assert task_calls[0]['energy_feedback_enabled'] is True, task_calls
    assert report['energy_feedback_enabled'] is True, report
    assert report['aggregate']['followup_target_hit_rate'] == 1.0, report
    print('[PASS] test_real_energy_reuse_slice_threads_energy_feedback_flag')


def test_real_energy_reuse_summary_tracks_target_hits():
    """Targeted energy-reuse aggregation should expose target-hit, distractor-capture, and energy totals."""
    from cortex_benchmarks.benchmark_shared_manifold import _summarize_real_energy_reuse

    aggregate = _summarize_real_energy_reuse([
        {
            'primer_target_hit_rate': 1.0,
            'followup_prompt_hit': True,
            'followup_target_hit': True,
            'followup_patch_hit': True,
            'followup_selected_task_id': 'target_a',
            'distractor_task_ids': ['distractor_a'],
            'shared_manifold_stats': {'energy_peak': 0.5, 'energy_abs_total': 1.2},
        },
        {
            'primer_target_hit_rate': 0.5,
            'followup_prompt_hit': True,
            'followup_target_hit': False,
            'followup_patch_hit': False,
            'followup_selected_task_id': 'distractor_b',
            'distractor_task_ids': ['distractor_b'],
            'shared_manifold_stats': {'energy_peak': 0.0, 'energy_abs_total': 0.0},
        },
    ])

    assert aggregate['task_count'] == 2, aggregate
    assert aggregate['primer_target_hit_rate'] == 0.75, aggregate
    assert aggregate['followup_target_hit_rate'] == 0.5, aggregate
    assert aggregate['followup_patch_hit_rate'] == 0.5, aggregate
    assert aggregate['distractor_capture_rate'] == 0.5, aggregate
    assert aggregate['avg_energy_peak'] == 0.25, aggregate
    assert aggregate['avg_energy_abs_total'] == 0.6, aggregate
    print('[PASS] test_real_energy_reuse_summary_tracks_target_hits')


def test_full_eval_energy_ablation_delta_summary():
    """Full evaluation energy ablation helper should report numeric deltas for latest off-vs-on aggregates."""
    from cortex_benchmarks.full_shared_manifold_evaluation import _run_energy_ablation

    def runner_off():
        return {
            'enabled': {'aggregate': {'pass_rate': 0.4, 'prompt_hit_rate': 1.0}},
            'disabled': {'aggregate': {'pass_rate': 0.0, 'prompt_hit_rate': 0.0}},
        }

    def runner_on():
        return {
            'enabled': {'aggregate': {'pass_rate': 0.7, 'prompt_hit_rate': 1.0}},
            'disabled': {'aggregate': {'pass_rate': 0.0, 'prompt_hit_rate': 0.0}},
        }

    payload = _run_energy_ablation(
        'coding_compare',
        1,
        runner_off,
        runner_on,
        lambda report: report,
    )

    assert payload['latest_off']['enabled']['pass_rate'] == 0.4, payload
    assert payload['latest_on']['enabled']['pass_rate'] == 0.7, payload
    assert payload['delta_latest']['enabled']['pass_rate'] == 0.3, payload
    assert payload['delta_latest']['disabled']['pass_rate'] == 0.0, payload
    print('[PASS] test_full_eval_energy_ablation_delta_summary')


def test_shared_manifold_real_coding_slice_smoke():
    """Opt-in smoke test for the real local-model coding slice."""
    if os.environ.get('WARP_CORTEX_RUN_REAL_BENCHMARKS') != '1':
        raise RuntimeError('SKIP real shared-manifold benchmark smoke test disabled')

    from cortex_benchmarks.benchmark_shared_manifold import compare_real_coding_slice, default_real_coding_tasks

    device = os.environ.get('WARP_CORTEX_REAL_BENCH_DEVICE') or 'cpu'
    report = compare_real_coding_slice(
        tasks=default_real_coding_tasks()[:1],
        device=device,
        max_tokens=96,
    )
    enabled = report['enabled']
    disabled = report['disabled']
    task_enabled = enabled['tasks'][0]
    task_disabled = disabled['tasks'][0]

    assert enabled['aggregate']['task_count'] == 1, report
    assert disabled['aggregate']['task_count'] == 1, report
    assert task_enabled['name'] == 'retry_replay_token_simple', task_enabled
    assert task_enabled['prompt_hit'], task_enabled
    assert task_enabled['passed'], task_enabled
    assert task_enabled['selected_patches'] == ['add_replay_safety_token_header'], task_enabled
    assert 'Replay-Safety-Token' in task_enabled['candidate_code'], task_enabled
    assert not task_disabled['prompt_hit'], task_disabled
    assert not task_disabled['passed'], task_disabled
    print('[PASS] test_shared_manifold_real_coding_slice_smoke')


def test_shared_manifold_real_handoff_smoke():
    """Opt-in smoke test for the real two-instance handoff path."""
    if os.environ.get('WARP_CORTEX_RUN_REAL_BENCHMARKS') != '1':
        raise RuntimeError('SKIP real shared-manifold handoff smoke test disabled')

    from cortex_benchmarks.benchmark_shared_manifold import compare_real_handoff_slice, default_real_coding_tasks

    device = os.environ.get('WARP_CORTEX_REAL_BENCH_DEVICE') or 'cpu'
    report = compare_real_handoff_slice(
        tasks=default_real_coding_tasks()[:1],
        device=device,
        max_tokens=64,
    )
    aggregate = report['aggregate']
    task = report['tasks'][0]

    assert aggregate['task_count'] == 1, report
    assert aggregate['fresh_prompt_hit_rate'] == 0.0, report
    assert aggregate['loaded_prompt_hit_rate'] == 1.0, report
    assert aggregate['fresh_pass_rate'] == 0.0, report
    assert aggregate['loaded_pass_rate'] == 1.0, report
    assert task['context_match'], task
    assert task['output_match'], task
    assert not task['fresh_reader']['passed'], task
    assert task['loaded_reader']['passed'], task
    assert task['loaded_reader']['selected_patches'] == ['add_replay_safety_token_header'], task
    assert 'Replay-Safety-Token' in task['loaded_reader']['candidate_code'], task
    assert 'snapshot_load' in task, task
    print('[PASS] test_shared_manifold_real_handoff_smoke')


def test_shared_manifold_real_recall_handoff_smoke():
    """Opt-in smoke test for the real two-instance recall handoff path."""
    if os.environ.get('WARP_CORTEX_RUN_REAL_BENCHMARKS') != '1':
        raise RuntimeError('SKIP real shared-manifold recall handoff smoke test disabled')

    from cortex_benchmarks.benchmark_shared_manifold import compare_real_recall_handoff_slice, default_real_recall_tasks

    device = os.environ.get('WARP_CORTEX_REAL_BENCH_DEVICE') or 'cpu'
    task_names = {'jenny_boots_locker', 'cedar_compass_chain'}
    tasks = [task for task in default_real_recall_tasks() if task.name in task_names]
    report = compare_real_recall_handoff_slice(
        tasks=tasks,
        device=device,
        max_tokens=48,
    )
    aggregate = report['aggregate']
    task_map = {task['name']: task for task in report['tasks']}

    assert aggregate['task_count'] == 2, report
    assert aggregate['fresh_prompt_hit_rate'] == 0.0, report
    assert aggregate['loaded_prompt_hit_rate'] == 1.0, report
    assert aggregate['loaded_answer_rate'] == 1.0, report
    jenny = task_map['jenny_boots_locker']
    cedar = task_map['cedar_compass_chain']
    assert jenny['context_match'], jenny
    assert cedar['context_match'], cedar
    assert jenny['loaded_reader']['passed'], jenny
    assert cedar['loaded_reader']['passed'], cedar
    assert jenny['loaded_reader']['parsed_fields'] == {'color': 'red', 'where': 'locker 14'}, jenny
    assert cedar['loaded_reader']['parsed_fields'] == {'where': 'cedar drawer'}, cedar
    assert cedar['fresh_reader']['parsed_fields'] in ({}, {'where': 'unknown'}), cedar
    assert 'snapshot_load' in jenny, jenny
    assert 'snapshot_load' in cedar, cedar
    print('[PASS] test_shared_manifold_real_recall_handoff_smoke')


def test_real_recall_prompt_keeps_single_field_schema_local():
    """Single-field recall prompts should not leak extra schema keys or canned example values."""
    from cortex_benchmarks.benchmark_shared_manifold import _compose_real_recall_prompt, default_real_recall_tasks

    class _EchoTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "\n".join(message["content"] for message in messages)

    engine = type("PromptOnlyEngine", (), {"tokenizer": _EchoTokenizer()})()
    cedar = next(task for task in default_real_recall_tasks() if task.name == 'cedar_compass_chain')
    prompt = _compose_real_recall_prompt(
        engine,
        cedar,
        "[Shared Manifold]\n- [recall memory] Eli handed Nora the bronze compass before the archive audit.\n- [recall memory] Nora hid the bronze compass inside the cedar drawer.",
    )

    lowered = prompt.lower()
    assert 'who=' not in lowered, prompt
    assert 'locker 7' not in lowered, prompt
    assert 'where=place' in lowered, prompt


def test_hf_cache_snapshot_resolution():
    """Cached Hugging Face repos should resolve to a concrete snapshot path."""
    from cortex_core.hf_utils import resolve_local_model_source

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_id = 'abc123'
        repo_dir = os.path.join(tmpdir, 'hub', 'models--Qwen--Qwen2.5-0.5B-Instruct')
        snapshot_dir = os.path.join(repo_dir, 'snapshots', snapshot_id)
        refs_dir = os.path.join(repo_dir, 'refs')
        os.makedirs(snapshot_dir, exist_ok=True)
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, 'main'), 'w', encoding='utf-8') as handle:
            handle.write(snapshot_id)

        source, local_files_only = resolve_local_model_source('Qwen/Qwen2.5-0.5B-Instruct', tmpdir)
        assert local_files_only, source
        assert os.path.normpath(source) == os.path.normpath(snapshot_dir), source
    print('[PASS] test_hf_cache_snapshot_resolution')

def test_scorecard_runner_writes_artifacts_and_evidence():
    """The Cortex OS scorecard runner should emit the promised artifact set."""
    import json
    import os
    import tempfile
    from cortex_core.shared_manifold_store import SQLiteSharedManifoldStore
    from cortex_scorecard.runner import run_scorecard
    from cortex_scorecard.schema import ScorecardConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, 'scorecard')
        report = run_scorecard(
            config=ScorecardConfig(out_dir=out_dir, limit=2),
            candidate_names=['deterministic'],
        )

        assert report['aggregate']['candidate_summary']['deterministic']['pass_rate'] == 1.0, report
        for name in ('scorecard.json', 'scorecard.md', 'failures.jsonl', 'policy.yaml', 'manifest.json', 'evidence.sqlite'):
            assert os.path.exists(os.path.join(out_dir, name)), name

        with open(os.path.join(out_dir, 'failures.jsonl'), 'r', encoding='utf-8') as handle:
            assert handle.read() == ''
        with open(os.path.join(out_dir, 'scorecard.json'), 'r', encoding='utf-8') as handle:
            saved = json.load(handle)
        assert saved['policy']['routes']['support_json']['candidate'] == 'deterministic', saved['policy']

        store = SQLiteSharedManifoldStore(os.path.join(out_dir, 'evidence.sqlite'))
        nodes = store.list_nodes()
        assert len(nodes) == 2, nodes
        assert all(node['node_type'] == 'scorecard_result' for node in nodes), nodes
    print('[PASS] test_scorecard_runner_writes_artifacts_and_evidence')


def test_scorecard_hybrid_demo_repairs_failed_primary():
    """Hybrid scorecards should record fallback use after primary validation failure."""
    from cortex_scorecard.runner import run_scorecard
    from cortex_scorecard.schema import ScorecardConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_scorecard(
            config=ScorecardConfig(out_dir=os.path.join(tmpdir, 'scorecard'), limit=1),
            candidate_names=['hybrid_demo'],
        )

    summary = report['aggregate']['candidate_summary']['hybrid_demo']
    result = report['results'][0]
    assert summary['pass_rate'] == 1.0, report
    assert summary['fallback_rate'] == 1.0, report
    assert result['fallback_used'], result
    assert result['attempts'][0]['validation']['failed_checks'] == ['missing_fields'], result
    assert result['attempts'][1]['validation']['passed'], result
    print('[PASS] test_scorecard_hybrid_demo_repairs_failed_primary')


def test_shared_manifold_real_necessity_smoke():
    """Opt-in smoke test for the real multi-session necessity slice."""
    if os.environ.get('WARP_CORTEX_RUN_REAL_BENCHMARKS') != '1':
        raise RuntimeError('SKIP real shared-manifold necessity smoke test disabled')

    from cortex_benchmarks.benchmark_shared_manifold import compare_real_necessity_slice, default_real_necessity_tasks

    device = os.environ.get('WARP_CORTEX_REAL_BENCH_DEVICE') or 'cpu'
    report = compare_real_necessity_slice(
        tasks=default_real_necessity_tasks()[:1],
        device=device,
        max_tokens=48,
    )
    aggregate = report['aggregate']
    task = report['tasks'][0]

    assert aggregate['task_count'] == 1, report
    assert aggregate['isolated_prompt_hit_rate'] == 0.0, report
    assert aggregate['manifold_prompt_hit_rate'] == 1.0, report
    assert aggregate['isolated_answer_rate'] == 0.0, report
    assert aggregate['manifold_answer_rate'] == 1.0, report
    assert aggregate['oracle_answer_rate'] == 1.0, report
    assert aggregate['necessity_win_rate'] == 1.0, report
    assert task['name'] == 'vx17_badge_locker', task
    assert task['shared_store_stats']['node_count'] == 2, task
    assert not task['isolated_reader']['passed'], task
    assert task['manifold_reader']['passed'], task
    assert task['oracle_reader']['passed'], task
    assert task['manifold_reader']['parsed_fields'] == {'color': 'teal', 'where': 'locker 42'}, task
    assert task['isolated_reader']['missing_fields'] == ['color', 'where'], task
    print('[PASS] test_shared_manifold_real_necessity_smoke')


def test_shared_manifold_real_topology_smoke():
    """Opt-in smoke test for the full real topology-aware shared-manifold slice."""
    if os.environ.get('WARP_CORTEX_RUN_REAL_BENCHMARKS') != '1':
        raise RuntimeError('SKIP real shared-manifold topology smoke test disabled')

    from cortex_benchmarks.benchmark_shared_manifold import compare_real_topology_slice, default_real_topology_tasks

    device = os.environ.get('WARP_CORTEX_REAL_BENCH_DEVICE') or 'cpu'
    report = compare_real_topology_slice(
        tasks=default_real_topology_tasks(),
        device=device,
        max_tokens=48,
    )
    aggregate = report['aggregate']
    task_map = {task['name']: task for task in report['tasks']}

    assert aggregate['task_count'] == 2, report
    assert aggregate['component_accuracy_rate'] == 1.0, report
    assert aggregate['active_region_accuracy_rate'] == 1.0, report
    assert aggregate['topology_expected_recall_rate'] == 1.0, report
    assert aggregate['flat_expected_recall_rate'] == 0.75, report
    assert aggregate['topology_bridge_recall_rate'] == 1.0, report
    assert aggregate['flat_bridge_recall_rate'] == 1.0, report
    assert aggregate['topology_leakage_rate'] == 0.0, report
    assert aggregate['flat_leakage_rate'] == 0.25, report
    assert aggregate['topology_prompt_hit_rate'] == 1.0, report
    assert aggregate['flat_prompt_hit_rate'] == 1.0, report
    assert aggregate['topology_answer_rate'] == 1.0, report
    assert aggregate['flat_answer_rate'] == 0.0, report
    assert aggregate['topology_win_rate'] == 1.0, report

    payment = task_map['real_payment_retry_fields']
    bridge = task_map['real_bridge_route_chain']

    assert payment['shared_manifold_stats']['component_count'] == 2, payment
    assert payment['topology_retrieval']['active_region_size'] == 2, payment
    assert payment['topology_retrieval']['matched_expected'] == [
        'Checkout ticket PX-17 uses retry_header=X-Payment-Retry-Key.',
        'PX-17 replay seal field is replay_token_px17.',
    ], payment
    assert payment['flat_retrieval']['expected_recall'] < payment['topology_retrieval']['expected_recall'], payment
    assert payment['flat_retrieval']['leakage_count'] >= payment['topology_retrieval']['leakage_count'], payment
    assert payment['topology_reader']['passed'], payment
    assert not payment['flat_reader']['passed'], payment
    assert payment['topology_reader']['parsed_fields'] == {
        'retry_header': 'X-Payment-Retry-Key',
        'seal': 'replay_token_px17',
    }, payment

    assert bridge['shared_manifold_stats']['component_count'] == 2, bridge
    assert bridge['topology_retrieval']['active_region_size'] == 3, bridge
    assert bridge['topology_retrieval']['matched_bridge'] == [
        'Packet PX-9 crosses seam_route=beta-seam before the final checkpoint.',
    ], bridge
    assert bridge['topology_reader']['passed'], bridge
    assert not bridge['flat_reader']['passed'], bridge
    assert bridge['topology_reader']['parsed_fields'] == {
        'seam_route': 'beta-seam',
        'final_location': 'cedar-checkpoint',
    }, bridge
    print('[PASS] test_shared_manifold_real_topology_smoke')


def test_orchestrator_persists_agent_results():
    """Orchestrator tasks should compose persistent agent context and save results back to that agent."""
    from cortex_core.agent_cloud import PersistentAgentCloud
    from cortex_core.cortex_orchestrator import CortexOrchestrator, SubAgentTask, AgentRole
    from cortex_core.synapse import TopologicalSynapse

    class DummyEngine:
        def __init__(self):
            self.device = 'cpu'
            self.synapse = TopologicalSynapse(dim=16, device='cpu')
            self.agent_cloud = PersistentAgentCloud(hidden_dim=16, device='cpu')

    engine = DummyEngine()
    engine.agent_cloud.ensure_agent(
        'npc_guard',
        role='npc',
        profile='A city guard who tracks suspicious visitors.',
    )

    orchestrator = CortexOrchestrator(engine, max_workers=1)
    captured = {}

    def fake_run_side_agent(prompt_text: str, max_tokens: int = 30):
        captured['prompt'] = prompt_text
        return 'Observed the suspicious traveler at the north gate.', torch.ones(1, 16)

    orchestrator._run_side_agent = fake_run_side_agent

    task = SubAgentTask(
        agent_id='npc_guard',
        role=AgentRole.RESEARCHER,
        description='Report the latest suspicious visitor.',
        max_tokens=8,
    )
    task_id = orchestrator.dispatch(task)
    result = orchestrator.wait_for_task(task_id, timeout=5.0)

    assert result is not None and 'north gate' in result.lower(), result
    assert '[Persistent Agent: npc_guard]' in captured['prompt'], captured.get('prompt', '')
    assert 'city guard' in captured['prompt'].lower(), captured['prompt']

    state = engine.agent_cloud.get_agent('npc_guard')
    assert state is not None
    assert len(state.episodes) == 1
    assert 'north gate' in state.episodes[-1].text.lower(), state.episodes[-1].text
    assert state.adapter.trained_steps == 1
    print('[PASS] test_orchestrator_persists_agent_results')


def test_persistent_agent_cloud_roundtrip_snapshot():
    """Persistent agent cloud should survive a disk roundtrip with memory and adapter state intact."""
    from cortex_core.agent_cloud import PersistentAgentCloud

    cloud = PersistentAgentCloud(hidden_dim=16, device='cpu')
    cloud.ensure_agent(
        'npc_scout',
        role='npc',
        profile='A field scout who tracks caravan movement across the frontier.',
    )

    query = cloud.encode_text('Track the caravan near the east bridge at dusk.')
    cloud.remember_text(
        agent_id='npc_scout',
        text='Spotted the caravan crossing the east bridge before dusk.',
        hidden_state=query,
        role='npc',
    )
    cloud.store_task_result(
        agent_id='npc_scout',
        task_text='Where did the caravan go after the bridge?',
        result_text='[Researcher] The caravan turned south toward the old road.',
        result_vector=torch.linspace(0.0, 1.0, steps=16),
        role='npc',
    )
    kv = [
        (torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),
        (torch.randn(1, 2, 32, 8), torch.randn(1, 2, 32, 8)),
    ]
    cloud.materialize_shared_hot_cache(kv_landmarks=tuple(kv), turbo_bits=4, turbo_device='cpu')

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = os.path.join(tmpdir, 'agent_cloud.pt')
        cloud.save(snapshot_path)

        restored = PersistentAgentCloud(hidden_dim=16, device='cpu')
        result = restored.load(snapshot_path)

    assert result['loaded_agents'] == 1, result
    assert result['total_agents'] == 1, result
    assert result['shared_nodes'] >= 2, result

    restored_state = restored.get_agent('npc_scout')
    assert restored_state is not None
    assert restored_state.profile == 'A field scout who tracks caravan movement across the frontier.'
    assert len(restored_state.episodes) == 2
    assert restored_state.adapter.trained_steps == 2
    assert restored_state.synapse.injection_count == 2

    prompt = restored.compose_prompt(
        'npc_scout',
        task='Summarize the caravan movement after the east bridge and southern turn.',
    )
    manifold_stats = restored.shared_manifold_stats()
    assert manifold_stats['node_count'] >= 2, manifold_stats
    assert manifold_stats['kv_layer_count'] == 2, manifold_stats
    assert manifold_stats['kv_compressed_bytes'] > 0, manifold_stats
    assert '[Shared Manifold]' in prompt, prompt
    assert 'east bridge' in prompt.lower(), prompt
    assert 'turned south' in prompt.lower(), prompt
    print('[PASS] test_persistent_agent_cloud_roundtrip_snapshot')


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


def test_synapse_topology_stats_capture_bridge_structure():
    """Synapse topology stats should expose components, articulation bridges, and isolated landmarks."""
    from cortex_core.synapse import TopologicalSynapse

    synapse = TopologicalSynapse(dim=4, max_injections=8, device='cpu')
    synapse.inject_embedding(torch.tensor([1.0, 0.0, 0.0, 0.0]), score=1.0)
    synapse.inject_embedding(torch.tensor([1.0, 1.0, 0.0, 0.0]), score=1.0)
    synapse.inject_embedding(torch.tensor([0.0, 1.0, 0.0, 0.0]), score=1.0)
    synapse.inject_embedding(torch.tensor([0.0, 0.0, 1.0, 0.0]), score=1.0)

    stats = synapse.topology_stats()
    features = synapse.topology_feature_vector()

    assert stats['node_count'] == 4.0, stats
    assert stats['component_count'] == 2.0, stats
    assert stats['largest_component_size'] == 3.0, stats
    assert stats['bridge_count'] == 1.0, stats
    assert stats['isolated_count'] == 1.0, stats
    assert abs(stats['largest_component_ratio'] - 0.75) < 1e-6, stats
    assert len(features) == 7, features
    print('[PASS] test_synapse_topology_stats_capture_bridge_structure')


def test_cortex_attention_gate_absorbs():
    """CortexAttention cross-attends to injection landmarks."""
    from cortex_core.cortex_attention import CortexAttention
    from cortex_core.synapse import TopologicalSynapse

    dim = 64
    num_heads = 4
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device='cpu')
    attn = CortexAttention(dim=dim, num_heads=num_heads)
    attn.eval()
    assert attn.gate_proj[0].in_features == 2 * (dim // num_heads) + attn.N_TOPO_FEATURES

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
        test_linear_delegation_gate,
        test_low_rank_memory_adapter,
        test_persistent_agent_cloud_isolates_memory,
        test_shared_manifold_recall_across_agents,
        test_shared_manifold_sqlite_store_sync,
        test_shared_manifold_hot_cache_materializes_turbo_kv,
        test_context_manager_injects_shared_manifold_context,
        test_shared_manifold_plans_non_redundant_refresh,
        test_shared_manifold_task_board_stays_compact_for_prompting,
        test_shared_manifold_structural_edges_connect_related_nodes,
        test_shared_manifold_energy_deformation_biases_retrieval,
        test_shared_manifold_maintenance_decays_energy,
        test_shared_manifold_energy_snapshot_roundtrip,
        test_shared_manifold_task_result_feedback_energizes_task_board,
        test_shared_manifold_projection_landmark_carries_kv_residue,
        test_shared_manifold_projection_snapshot_roundtrip,
        test_shared_manifold_prefers_projection_context_for_refresh,
        test_shared_manifold_regions_stay_local_to_query,
        test_shared_manifold_preserves_bridge_nodes_for_recall,
        test_engine_refreshes_shared_manifold_into_kv,
        test_engine_prompt_context_feedback_energizes_nodes,
        test_engine_uses_shared_hot_cache_for_workers,
        test_engine_seeds_projection_residue_from_hot_cache,
        test_engine_projection_seed_feedback_energizes_projection_summary,
        test_engine_prefers_projection_residue_for_workers,
        test_engine_memory_accounting_reports_hot_cache,
        test_shared_manifold_benchmark_pipeline,
        test_shared_manifold_coding_slice_pipeline,
        test_shared_manifold_topology_slice_pipeline,
        test_real_coding_slice_threads_energy_feedback_flag,
        test_real_energy_reuse_slice_threads_energy_feedback_flag,
        test_real_energy_reuse_summary_tracks_target_hits,
        test_full_eval_energy_ablation_delta_summary,
        test_shared_manifold_real_coding_slice_smoke,
        test_shared_manifold_real_handoff_smoke,
        test_shared_manifold_real_recall_handoff_smoke,
        test_hf_cache_snapshot_resolution,
        test_scorecard_runner_writes_artifacts_and_evidence,
        test_scorecard_hybrid_demo_repairs_failed_primary,
        test_shared_manifold_real_necessity_smoke,
        test_shared_manifold_real_topology_smoke,
        test_orchestrator_persists_agent_results,
        test_persistent_agent_cloud_roundtrip_snapshot,
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
        test_synapse_topology_stats_capture_bridge_structure,
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
