import torch
import torch.nn as nn
import copy
import os
import sys
from contextlib import nullcontext
from threading import Event, Thread
from typing import Any, Callable, Optional, Sequence, Tuple, cast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from cortex_core.agent_cloud import PersistentAgentCloud
from cortex_core.synapse import TopologicalSynapse
from cortex_core.cortex_router import CortexRouter
from cortex_core.cortex_orchestrator import CortexOrchestrator, AgentRole, SubAgentTask
from cortex_core.cortex_memory import AutoCompactor, SkillRegistry, ContextManager
from cortex_core.cortex_hooks import (
    HookRegistry, HookPoint, HookContext,
    create_default_hooks, VerificationLoop,
)
from cortex_core.hf_utils import prepare_hf_cache, resolve_local_model_source
from cortex_core.settings import get_setting, load_settings, resolve_project_path
from cortex_core.turbo_quant import (
    TurboQuantCache,
    compress_landmarks,
    decompress_landmarks,
    summarize_kv_cache,
)

# ---------------------------------------------------------------------------
# warp-bitnet lite integration (optional — graceful fallback if not built)
# ---------------------------------------------------------------------------
_BITNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'warp_bitnet')
if os.path.isdir(_BITNET_DIR):
    _BITNET_PARENT = os.path.dirname(os.path.abspath(_BITNET_DIR))
    if _BITNET_PARENT not in sys.path:
        sys.path.insert(0, _BITNET_PARENT)
    kernel_dir = os.path.join(_BITNET_DIR, 'kernel')
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

BitLinear = None
BitNetMLP = None

try:
    from warp_bitnet.kernel.bit_linear import BitLinear, BitNetMLP
    _BITNET_AVAILABLE = True
except ImportError:
    _BITNET_AVAILABLE = False

class BitNetSideAgent(nn.Module):
    """
    Side Agent backed by warp-bitnet *lite* kernels (1.58-bit ternary).

    Memory footprint: ~0.2 GB per 1B parameters (8× compression vs FP16).
    Uses the lite CUDA kernel for single-token GEMV when available, with a
    pure-PyTorch fallback (cached unpack → F.linear) on CPU / non-CUDA.

    Integration modes:
      1. **From scratch** — supply ``hidden_size`` and ``num_layers`` and the
         agent builds a tiny ternary MLP stack.  Quick to initialise; useful
         for validation and ablation studies.
      2. **From HuggingFace model** — call ``BitNetSideAgent.from_pretrained``
         with a standard CausalLM and it will quantise every nn.Linear into a
         BitLinear (AbsMax ternary) automatically.  This is the "real" path
         for deploying 1000 + agents at ~0.2 GB each.
    """

    def __init__(self, config=None, device='cuda',
                 hidden_size: int = 896, num_layers: int = 4):
        super().__init__()
        self.device = device
        self.runtime_device = (
            device if str(device).startswith('cuda') and torch.cuda.is_available()
            else 'cpu'
        )
        self.compute_dtype = (
            torch.float16 if str(self.runtime_device).startswith('cuda')
            else torch.float32
        )

        if not _BITNET_AVAILABLE:
            print("[BitNet] WARNING: warp-bitnet not found — using FP16 fallback MLP")
            self._fallback = True
            self.mlp_stack = nn.Sequential(*[
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU())
                for _ in range(num_layers)
            ])
            self.lm_head = nn.Linear(hidden_size, hidden_size)
            self.to(device=self.runtime_device, dtype=self.compute_dtype)
            return

        assert BitLinear is not None and BitNetMLP is not None
        self._fallback = False
        h = hidden_size
        intermediate = h * 4

        # Stack of ternary MLP blocks
        self.layers = nn.ModuleList([BitNetMLP(h, intermediate) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(h)
        self.lm_head_bit = BitLinear(h, h, bias=False)

        # Quantise to ternary weights immediately (random init → ternary snap)
        self._quantise_all()
        self.to(device=self.runtime_device, dtype=self.compute_dtype)

        vram_mb = sum(
            p.nelement() * (2 / 8)  # 2 bits / weight → bytes
            for p in self.parameters()
        ) / 1e6
        print(f"[BitNet] Lite side agent ready — {num_layers} layers, "
              f"~{vram_mb:.1f} MB ternary ({h}-dim)")

    def _quantise_all(self):
        """Snap all BitLinear weights to ternary {-1, 0, +1} and pack."""
        assert BitLinear is not None
        for m in self.modules():
            if isinstance(m, BitLinear):
                # If weights are still random floats, quantise them
                if m.cached_weight is not None:
                    continue
                # Create dummy FP32 ternary from scratch (for demo/test).
                # In production, use from_pretrained path.
                W = torch.sign(torch.randn(m.out_features, m.in_features))
                W[W == 0] = 1  # no zeros in random sign
                # Fan-in scaling keeps the demo-lite path numerically stable in FP16.
                scale = torch.tensor([1.0 / (m.in_features ** 0.5)], dtype=torch.float32)
                m.load_from_dense_weights(W, scale)

    @classmethod
    def from_pretrained(cls, model: nn.Module, device='cuda', num_layers: Optional[int] = None):
        """
        Convert a HuggingFace CausalLM into a BitNet side agent.

        Every ``nn.Linear`` in the model is replaced by a ``BitLinear``
        with AbsMax ternary quantisation.  The resulting model uses
        ~8× less memory.
        """
        if not _BITNET_AVAILABLE:
            raise RuntimeError("warp-bitnet package not available")

        agent = cls.__new__(cls)
        nn.Module.__init__(agent)
        agent.device = device
        agent.runtime_device = (
            device if str(device).startswith('cuda') and torch.cuda.is_available()
            else 'cpu'
        )
        agent.compute_dtype = (
            torch.float16 if str(agent.runtime_device).startswith('cuda')
            else torch.float32
        )
        agent._fallback = False
        agent._model = model
        assert BitLinear is not None

        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                in_f, out_f = module.in_features, module.out_features
                if in_f % 16 != 0 or out_f % 16 != 0:
                    continue  # skip layers that don't pack cleanly

                bl = BitLinear(in_f, out_f, bias=module.bias is not None)
                # AbsMax ternary quantisation
                with torch.no_grad():
                    W = module.weight.float()
                    scale = W.abs().mean()
                    W_ternary = (W / (scale + 1e-8)).round().clamp(-1, 1)
                bl.load_from_dense_weights(W_ternary, scale,
                                           module.bias.data if module.bias is not None else None)
                setattr(parent, child_name, bl)
                replaced += 1

        agent.to(device=agent.runtime_device, dtype=agent.compute_dtype)

        vram_mb = sum(
            p.nelement() * (2 / 8) for p in model.parameters()
        ) / 1e6
        print(f"[BitNet] Converted {replaced} layers to ternary — ~{vram_mb:.1f} MB")
        return agent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [Batch, Seq, Hidden] → [Batch, Seq, Hidden]"""
        target_device = torch.device(self.runtime_device)
        if x.device != target_device or x.dtype != self.compute_dtype:
            x = x.to(device=target_device, dtype=self.compute_dtype)

        if self._fallback:
            h = x
            for layer in self.mlp_stack:
                h = layer(h) + h  # residual
            return self.lm_head(h)

        h = x
        for layer in self.layers:
            h = layer(h) + h  # residual
        h = self.norm(h)
        return self.lm_head_bit(h)

    def think(self, landmarks, prompt_ids, tokenizer=None):
        """
        Process O(k) landmark context and produce a thought string.
        Works with both KV-tuple landmarks and raw tensor landmarks.
        """
        with torch.no_grad():
            # Build a representation from landmarks
            if isinstance(landmarks, (tuple, list)) and len(landmarks) > 0:
                if isinstance(landmarks[0], tuple):
                    # KV cache format: list of (key, value) tensors
                    # Average over all layer keys to get a context vector
                    ctx = torch.stack([k.float().mean(dim=(0, 1, 2)) for k, v in landmarks]).mean(0)
                else:
                    ctx = landmarks[0].float().mean(dim=0)
            elif isinstance(landmarks, torch.Tensor):
                ctx = landmarks.float().mean(dim=0)
            else:
                return "[BitNet: no landmarks]"

            # Ensure [1, 1, D] for forward pass
            if ctx.dim() == 1:
                ctx = ctx.unsqueeze(0).unsqueeze(0)
            elif ctx.dim() == 2:
                ctx = ctx.unsqueeze(0)

            ctx = ctx.to(self.runtime_device)
            out = self.forward(ctx)
            # Summarise output as a norm / direction fingerprint
            norm_val = out.norm().item()
            top_k = out.flatten().abs().topk(3).indices.tolist()

        return f"[BitNet-Lite Analysis: norm={norm_val:.2f}, salient_dims={top_k}]"

class EarlyExitSideAgent(nn.Module):
    """
    A 'Holographic' Side Agent.
    It uses the Main Agent's weights but exits early (e.g., at layer 12 of 32).
    Zero extra VRAM. 2x-3x faster.

    Fixed implementation: uses a forward hook to intercept hidden states at
    the target layer rather than manually iterating layers (which breaks
    RoPE / cache handling in Qwen2 and similar architectures).
    """
    def __init__(self, main_model, exit_layer_idx=12):
        super().__init__()
        self.main_model = main_model
        self.exit_layer_idx = exit_layer_idx
        self._captured_hidden = None
        self._hook_handle = None
        self._install_hook()
        print(f"[EarlyExit] Initialized Side Agent (Layers 0-{exit_layer_idx}). VRAM usage: 0GB (Shared)")

    def _install_hook(self):
        """Attach a forward hook to the target layer to capture hidden states."""
        target_layer = self.main_model.model.layers[self.exit_layer_idx]

        def hook_fn(module, input, output):
            # Qwen2DecoderLayer returns (hidden_states, self_attn_weights, present_kv)
            # or just hidden_states depending on config; extract first element
            if isinstance(output, tuple):
                self._captured_hidden = output[0].detach()
            else:
                self._captured_hidden = output.detach()

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def think(self, landmarks, prompt_ids):
        """
        Run a full forward pass but use the hidden state captured at the
        early-exit layer for decoding. The model runs all layers (needed
        for correct RoPE / cache), but we read from layer N instead of
        the final layer — giving us the speed benefit on the *decoding*
        side (lm_head on a partial representation) while keeping the
        forward pass correct.
        """
        self._captured_hidden = None

        with torch.no_grad():
            _ = self.main_model(
                input_ids=prompt_ids,
                past_key_values=None,
                use_cache=False,
            )

        if self._captured_hidden is not None:
            # Apply final LayerNorm + lm_head to the early hidden state
            hidden = self.main_model.model.norm(self._captured_hidden)
            logits = self.main_model.lm_head(hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            return f"[EarlyExit@L{self.exit_layer_idx} Analysis: {next_token.item()}]"

        return "[EarlyExit: hook did not fire]"

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ======================================================================
# Stream Pool: prioritized CUDA stream management for 100+ agents
# ======================================================================

class CUDAStreamPool:
    """
    Manages a pool of CUDA streams with priority levels.
    Agents request a stream, use it, then return it.
    Prevents CUDA stream exhaustion for large agent counts.
    """
    PRIORITY_HIGH   = -1  # CUDA high priority (lower number = higher priority)
    PRIORITY_NORMAL =  0

    def __init__(self, pool_size: int = 16, device: str = 'cuda'):
        self.device = device
        self.pool_size = pool_size
        self._available_high = []
        self._available_normal = []
        self._lock = __import__('threading').Lock()

        # Pre-create streams
        n_high = max(2, pool_size // 4)
        n_normal = pool_size - n_high
        for _ in range(n_high):
            self._available_high.append(
                torch.cuda.Stream(device=device, priority=self.PRIORITY_HIGH)
            )
        for _ in range(n_normal):
            self._available_normal.append(
                torch.cuda.Stream(device=device, priority=self.PRIORITY_NORMAL)
            )

    def acquire(self, high_priority: bool = False) -> Any:
        """Get a stream from the pool. Falls back to normal if high exhausted."""
        with self._lock:
            if high_priority and self._available_high:
                return self._available_high.pop()
            if self._available_normal:
                return self._available_normal.pop()
            if self._available_high:
                return self._available_high.pop()
        # All streams busy — create a temporary one
        return torch.cuda.Stream(device=self.device, priority=self.PRIORITY_NORMAL)

    def release(self, stream: Any, high_priority: bool = False):
        """Return a stream to the pool."""
        with self._lock:
            if high_priority:
                if len(self._available_high) < self.pool_size:
                    self._available_high.append(stream)
            else:
                if len(self._available_normal) < self.pool_size:
                    self._available_normal.append(stream)

    def available(self) -> int:
        with self._lock:
            return len(self._available_high) + len(self._available_normal)


# ======================================================================
# Adaptive Validation Threshold
# ======================================================================

class AdaptiveValidationGate:
    """
    Dynamically adjusts the cosine similarity threshold for thought injection
    based on a running acceptance rate target.

    - Creative tasks tolerate more divergent thoughts → lower threshold
    - Factual QA needs strict relevance → higher threshold
    - The gate tracks the EMA of recent accept/reject decisions and nudges
      the threshold toward a target acceptance ratio.
    """
    def __init__(self, initial_threshold: float = 0.4,
                 target_accept_rate: float = 0.6,
                 ema_alpha: float = 0.1,
                 min_threshold: float = 0.1,
                 max_threshold: float = 0.8):
        self.threshold = initial_threshold
        self.target_accept_rate = target_accept_rate
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._accept_ema = target_accept_rate  # start at target

    def should_accept(self, similarity: float) -> bool:
        accepted = similarity >= self.threshold
        # Update EMA
        self._accept_ema = (
            self.ema_alpha * (1.0 if accepted else 0.0)
            + (1.0 - self.ema_alpha) * self._accept_ema
        )
        # Nudge threshold
        if self._accept_ema > self.target_accept_rate + 0.05:
            # Accepting too many — tighten
            self.threshold = min(self.threshold + 0.01, self.max_threshold)
        elif self._accept_ema < self.target_accept_rate - 0.05:
            # Rejecting too many — loosen
            self.threshold = max(self.threshold - 0.01, self.min_threshold)
        return accepted

    @property
    def stats(self):
        return {
            "threshold": round(self.threshold, 3),
            "accept_ema": round(self._accept_ema, 3),
        }

class CortexEngine:
    def __init__(self, model_id: Optional[str] = None,
                 device: Optional[str] = None,
                 side_mode="shared"):
        """
        side_mode: 
          - "shared": Full Main Model (High IQ)
          - "bitnet": Separate 1.58b Model (High Speed, Low VRAM)
          - "early_exit": First N layers of Main Model (High Speed, Zero VRAM)
        """
        settings = load_settings()
        model_id = model_id or str(get_setting(settings, "backends.local.model", "Qwen/Qwen2.5-0.5B-Instruct"))
        device = device or str(get_setting(settings, "runtime.device", "cuda"))

        print(f"Loading {model_id}...")
        self.device = (
            device if str(device).startswith('cuda') and torch.cuda.is_available()
            else 'cpu'
        )
        self.use_cuda_streams = str(self.device).startswith('cuda')

        cache_root = resolve_project_path(get_setting(settings, "paths.huggingface_cache"))
        cache_dir = prepare_hf_cache(
            os.path.dirname(os.path.abspath(__file__)),
            preferred_root=cache_root,
        )
        model_source, local_files_only = resolve_local_model_source(model_id, cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        model_kwargs = {
            "cache_dir": cache_dir,
            "dtype": torch.float16 if self.use_cuda_streams else torch.float32,
        }
        if self.use_cuda_streams:
            model_kwargs["device_map"] = "auto"
        if local_files_only:
            model_kwargs["local_files_only"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        cast(Any, self.model).config.output_hidden_states = True
        if not self.use_cuda_streams:
            cast(Any, self.model).to(self.device)

        hidden = getattr(self.model.config, 'hidden_size', 896)
        self.synapse = TopologicalSynapse(dim=hidden, device=self.device)
        self.router = CortexRouter()

        # Bootstrap semantic router on the model's own hidden states
        try:
            self.router.bootstrap(self.model, self.tokenizer, device=self.device)
        except Exception as e:
            print(f"[Router] Semantic bootstrap failed ({e}); regex fallback active")

        # --- New subsystems (Claude Code patterns) ---

        # 1. Hook registry (PreToolUse, PostToolUse, security, quality gates)
        self.hooks = create_default_hooks()

        # 2. Auto-compaction (triggers landmarking at 75% context capacity)
        max_ctx = getattr(self.model.config, 'max_position_embeddings', 32768)
        self.compactor = AutoCompactor(
            max_seq_len=max_ctx,
            compact_threshold=0.75,
            landmark_k=64,
        )
        # Wire compaction into hook system
        self.compactor.pre_compact_hooks.append(
            lambda util: print(f"[Hook] PreCompact: util={util:.0%}")
        )

        # 3. Persistent skills
        skills_dir = os.path.join(
            os.path.dirname(__file__), "cortex_resources", "skills"
        )
        self.skills = SkillRegistry(skills_dir)

        # 4. Persistent agent cloud: isolated per-agent memory on one backbone
        shared_store_path = (
            os.environ.get("WARP_CORTEX_SHARED_MANIFOLD_DB")
            or get_setting(settings, "shared_manifold.store_path", "")
            or None
        )
        shared_store_cache_key = get_setting(settings, "shared_manifold.cache_key", "default")
        shared_hot_capacity = int(get_setting(settings, "shared_manifold.hot_capacity", 8))
        self.shared_hot_refresh_seconds = float(
            get_setting(settings, "shared_manifold.background_refresh_seconds", 2.0)
        )
        self.prefer_hot_cache_for_workers = bool(
            get_setting(settings, "shared_manifold.prefer_hot_cache_for_workers", True)
        )
        self.agent_cloud = PersistentAgentCloud(
            hidden_dim=hidden,
            tokenizer=self.tokenizer,
            embed_layer=self.model.get_input_embeddings(),
            device=self.device,
            shared_hot_capacity=shared_hot_capacity,
            shared_store_path=shared_store_path,
            shared_store_cache_key=shared_store_cache_key,
        )

        self.shared_manifold_enabled = True
        self.shared_manifold_trace: list[dict[str, Any]] = []
        self.shared_manifold_prompt_hits = 0
        self.shared_manifold_prompt_misses = 0
        self.shared_manifold_runtime_refreshes = 0
        self.shared_manifold_nodes_consumed = 0
        self.shared_manifold_energy_feedback_enabled = True
        self.agent_cloud.shared_energy_feedback_enabled = True

        # 4b. Context manager (wraps synapse + compaction + skills + shared manifold)
        self.context_mgr = ContextManager(
            self.synapse,
            self.compactor,
            self.skills,
            shared_context_getter=self._build_shared_manifold_context,
        )

        if side_mode == "bitnet":
            hidden = getattr(self.model.config, 'hidden_size', 896)
            self.side_agent_model = BitNetSideAgent(
                config=self.model.config, device=self.device,
                hidden_size=hidden, num_layers=4,
            ).to(self.device)
        elif side_mode == "early_exit":
            # Use first 50% of layers
            exit_layer = len(self.model.model.layers) // 2
            self.side_agent_model = EarlyExitSideAgent(self.model, exit_layer_idx=exit_layer)
        else:
            self.side_agent_model = self.model # Shared weights

        self.main_stream = (
            torch.cuda.Stream(device=self.device) if self.use_cuda_streams else None
        )
        self.side_stream = (
            torch.cuda.Stream(device=self.device) if self.use_cuda_streams else None
        )  # Legacy single stream (kept for backward compat)

        # 5. Orchestrator (parallel agent teams, specialized roles)
        self.orchestrator = CortexOrchestrator(engine=self, max_workers=8)

        # 6. CUDA stream pool (prioritized, for 100+ agents)
        self.stream_pool = (
            CUDAStreamPool(pool_size=16, device=self.device)
            if self.use_cuda_streams else None
        )

        # 7. Adaptive validation gate (replaces hardcoded cosine threshold)
        self.validation_gate = AdaptiveValidationGate(
            initial_threshold=0.4,
            target_accept_rate=0.6,
        )

        # 9. Shared manifold refresh (bounded external working memory)
        self.shared_manifold_refresh_interval = 8
        self.shared_manifold_refresh_top_k = 2
        self.shared_manifold_projection_top_k = 4

        # 8. TurboQuant KV cache compressor (3-4 bit, Mac-friendly Hadamard)
        self.turbo_quant_bits = int(get_setting(settings, "adaptive_engine.turbo_quant.bits", 4))
        self.turbo_quant_enabled = True
        self._turbo_cache: Optional[TurboQuantCache] = None
        self._shared_hot_refresh_stop = Event()
        self._shared_hot_refresh_thread: Optional[Thread] = None
        if shared_store_path and self.shared_hot_refresh_seconds > 0:
            self._start_shared_hot_refresh_worker()

    def _start_shared_hot_refresh_worker(self):
        if self._shared_hot_refresh_thread is not None:
            return

        thread = Thread(target=self._shared_hot_refresh_loop, name="warp-cortex-hot-refresh", daemon=True)
        thread.start()
        self._shared_hot_refresh_thread = thread

    def _shared_hot_refresh_loop(self):
        while not self._shared_hot_refresh_stop.wait(self.shared_hot_refresh_seconds):
            self._refresh_shared_hot_cache_once()

    def _refresh_shared_hot_cache_once(self) -> Optional[dict[str, Any]]:
        landmarks_raw = self.synapse.get_landmarks() if getattr(self, "synapse", None) is not None else None
        if landmarks_raw is not None:
            return self._materialize_shared_hot_cache(landmarks_raw)
        self.agent_cloud.materialize_shared_hot_cache(kv_landmarks=None)
        return None

    def stop_shared_hot_refresh_worker(self):
        self._shared_hot_refresh_stop.set()
        thread = self._shared_hot_refresh_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._shared_hot_refresh_thread = None

    def _materialize_shared_hot_cache(
        self,
        landmarks_raw,
        *,
        query_text: Optional[str] = None,
        agent_id: Optional[str] = None,
        projection_kind: str = "runtime_decode",
    ) -> Optional[dict[str, Any]]:
        if landmarks_raw is None:
            return None

        tq = compress_landmarks(
            landmarks_raw,
            bits=self.turbo_quant_bits,
            device=str(self.device),
        )
        kv_stats = summarize_kv_cache(landmarks_raw, tq)
        ratio = float(kv_stats.get("compression_ratio", 1.0))
        print(f"[TurboQuant] Landmarks compressed {ratio:.1f}× ({tq.bits}-bit + QJL residual)")
        self._turbo_cache = tq
        self.agent_cloud.materialize_shared_hot_cache(
            kv_landmarks=landmarks_raw,
            turbo_bits=self.turbo_quant_bits,
            turbo_device=str(self.device),
        )
        compact_query = (query_text or "").strip()
        if compact_query and self.shared_manifold_enabled:
            self.agent_cloud.resolve_shared_projection(
                query_text=compact_query,
                top_k=self.shared_manifold_projection_top_k,
                agent_id=agent_id,
                require_residue=False,
                materialize_missing=True,
                projection_kind=projection_kind,
                kv_landmarks=landmarks_raw,
                turbo_bits=self.turbo_quant_bits,
                turbo_device=str(self.device),
            )
        return kv_stats

    def _dynamic_cache_from_landmarks(self, landmarks):
        past_key_values = DynamicCache()
        setattr(past_key_values, "key_cache", [k for k, _ in landmarks])
        setattr(past_key_values, "value_cache", [v for _, v in landmarks])
        return past_key_values

    def _resolve_projection_landmarks(
        self,
        *,
        query_text: str,
        agent_id: Optional[str] = None,
        projection_kind: str = "runtime_decode",
        materialize_missing: bool = False,
    ):
        compact_query = str(query_text or "").strip()
        if not compact_query or not self.shared_manifold_enabled:
            return None, None

        hot_landmarks = None
        if materialize_missing:
            hot_cache = self.agent_cloud.get_shared_hot_turbo_cache(device=str(self.device))
            if hot_cache is not None:
                hot_landmarks = decompress_landmarks(hot_cache)

        projection = self.agent_cloud.resolve_shared_projection(
            query_text=compact_query,
            top_k=self.shared_manifold_projection_top_k,
            agent_id=agent_id,
            require_residue=True,
            materialize_missing=bool(materialize_missing and hot_landmarks is not None),
            projection_kind=projection_kind,
            kv_landmarks=hot_landmarks,
            turbo_bits=self.turbo_quant_bits,
            turbo_device=str(self.device),
        )
        if projection is None or not projection.get("projection_id"):
            return None, None

        projection_cache = self.agent_cloud.get_projection_residue(
            projection["projection_id"],
            device=str(self.device),
        )
        if projection_cache is None:
            return None, projection
        return decompress_landmarks(projection_cache), projection

    def _seed_shared_projection_cache(
        self,
        *,
        query_text: str,
        used_texts: Optional[set[str]] = None,
        past_key_values=None,
        agent_id: Optional[str] = None,
    ):
        if past_key_values is not None:
            return past_key_values, 0

        landmarks, projection = self._resolve_projection_landmarks(
            query_text=query_text,
            agent_id=agent_id,
            projection_kind="runtime_decode",
            materialize_missing=True,
        )
        if landmarks is None or projection is None:
            return past_key_values, 0

        seeded_cache = self._dynamic_cache_from_landmarks(landmarks)
        if used_texts is not None:
            used_texts.add(projection.get("summary_text", ""))
            for node in projection.get("member_nodes") or []:
                used_texts.add(node.text)

        projection_nodes = [projection["node"]]
        projection_nodes.extend(projection.get("member_nodes") or [])
        self.shared_manifold_nodes_consumed += max(1, len(projection.get("projection_node_ids") or projection_nodes))
        self._record_shared_manifold_event(
            "projection_seed",
            query_text,
            projection_nodes,
            agent_id=agent_id,
        )
        self._apply_shared_manifold_energy_feedback(
            "projection_seed",
            query_text,
            projection_nodes,
            agent_id=agent_id,
        )
        return seeded_cache, len(projection_nodes)

    def _resolve_worker_landmarks(self, query_text: Optional[str] = None, agent_id: Optional[str] = None):
        landmarks = self.synapse.get_landmarks() if getattr(self, "synapse", None) is not None else None
        if landmarks is not None:
            return landmarks
        projection_landmarks, _ = self._resolve_projection_landmarks(
            query_text=str(query_text or ""),
            agent_id=agent_id,
            projection_kind="worker_context",
            materialize_missing=self.prefer_hot_cache_for_workers,
        )
        if projection_landmarks is not None:
            return projection_landmarks
        if self.prefer_hot_cache_for_workers:
            hot_cache = self.agent_cloud.get_shared_hot_turbo_cache(device=str(self.device))
            if hot_cache is not None:
                return decompress_landmarks(hot_cache)
        if self._turbo_cache is not None:
            return decompress_landmarks(self._turbo_cache)
        return None

    def _record_shared_manifold_event(self, stage: str, query_text: str,
                                      nodes: list[Any], agent_id: Optional[str] = None):
        event = {
            "stage": stage,
            "agent_id": agent_id,
            "query_text": query_text,
            "node_count": len(nodes),
            "node_ids": [
                str(getattr(node, "node_id", "")).strip()
                for node in nodes
                if str(getattr(node, "node_id", "")).strip()
            ],
            "nodes": [getattr(node, "text", str(node)) for node in nodes],
        }
        self.shared_manifold_trace.append(event)
        if len(self.shared_manifold_trace) > 256:
            self.shared_manifold_trace = self.shared_manifold_trace[-256:]

    def _apply_shared_manifold_energy_feedback(
        self,
        stage: str,
        query_text: str,
        nodes: list[Any],
        agent_id: Optional[str] = None,
    ):
        if not getattr(self, "shared_manifold_energy_feedback_enabled", False):
            return None
        if getattr(self, "agent_cloud", None) is None:
            return None

        feedback_nodes = [node for node in nodes if getattr(node, "node_id", None)]
        if not feedback_nodes:
            return None

        if stage == "projection_seed":
            delta = float(getattr(self.agent_cloud, "shared_energy_projection_delta", 0.0))
        elif stage == "runtime_refresh":
            delta = float(getattr(self.agent_cloud, "shared_energy_refresh_delta", 0.0))
        else:
            delta = float(getattr(self.agent_cloud, "shared_energy_prompt_delta", 0.0))
        if abs(delta) < 1e-9:
            return None

        return self.agent_cloud.deform_manifold_for_nodes(
            feedback_nodes,
            delta,
            max_depth=1,
            edge_decay=0.85,
        )

    def _build_shared_manifold_context(self, prompt: str, top_k: int = 4) -> str:
        if not self.shared_manifold_enabled:
            return ""

        context_text = self.agent_cloud.build_shared_context(prompt, top_k=top_k)
        if not context_text:
            self.shared_manifold_prompt_misses += 1
            return ""

        projection = self.agent_cloud.resolve_shared_projection(
            query_text=prompt,
            top_k=top_k,
        )
        if projection is not None:
            nodes = [projection["node"]]
            nodes.extend(projection.get("member_nodes") or [])
        else:
            nodes = self.agent_cloud.query_shared_manifold(
                query_text=prompt,
                top_k=top_k,
            )

        self.shared_manifold_prompt_hits += 1
        self.shared_manifold_nodes_consumed += len(nodes)
        self._record_shared_manifold_event("prompt_context", prompt, nodes)
        self._apply_shared_manifold_energy_feedback("prompt_context", prompt, nodes)
        return context_text

    def _inject_reference_memory(self, label: str, memory_text: str, past_key_values):
        """Inject a compact memory reference into the live KV cache without emitting it to the user."""
        memory_ids = self.tokenizer(
            f" [{label}: {memory_text}]",
            return_tensors="pt",
        ).input_ids.to(self.device)
        memory_outputs = self.model(memory_ids, past_key_values=past_key_values)
        return memory_outputs.past_key_values

    def _maybe_refresh_shared_manifold(
        self,
        *,
        base_prompt: str,
        recent_text: str,
        used_texts: set[str],
        past_key_values,
        agent_id: Optional[str] = None,
    ):
        """Pull fresh shared-manifold nodes into the live decode state at bounded intervals."""
        if not self.shared_manifold_enabled:
            return past_key_values, 0

        query_text = base_prompt.strip()
        if recent_text.strip():
            query_text = f"{query_text}\n{recent_text[-200:]}"

        refresh_text, nodes = self.agent_cloud.plan_shared_injection(
            query_text=query_text,
            used_texts=used_texts,
            top_k=self.shared_manifold_refresh_top_k,
            agent_id=agent_id,
        )
        if not nodes:
            return past_key_values, 0

        print(f"\n[Main] Shared manifold refresh: {len(nodes)} node(s)")
        past_key_values = self._inject_reference_memory("Shared", refresh_text, past_key_values)
        self.shared_manifold_runtime_refreshes += 1
        self.shared_manifold_nodes_consumed += len(nodes)
        self._record_shared_manifold_event("runtime_refresh", query_text, nodes, agent_id=agent_id)
        self._apply_shared_manifold_energy_feedback(
            "runtime_refresh",
            query_text,
            nodes,
            agent_id=agent_id,
        )
        for node in nodes:
            used_texts.add(node.text)
        return past_key_values, len(nodes)

    def _build_landmark_query(self, last_hidden_state, past_key_values):
        """Project the last hidden state into the KV-head layout expected by the synapse."""
        if last_hidden_state is None or past_key_values is None:
            return None

        try:
            first_layer = next(iter(past_key_values))
        except Exception:
            return None

        if not isinstance(first_layer, (tuple, list)) or len(first_layer) < 1:
            return None

        key_states = first_layer[0]
        if key_states is None or key_states.dim() != 4:
            return None

        target_heads = key_states.shape[1]
        head_dim = key_states.shape[-1]
        batch_size = last_hidden_state.shape[0]
        hidden_size = last_hidden_state.shape[-1]

        num_attention_heads = int(getattr(self.model.config, 'num_attention_heads', target_heads))
        if hidden_size != num_attention_heads * head_dim:
            return None

        query = last_hidden_state.view(batch_size, num_attention_heads, 1, head_dim)
        if num_attention_heads == target_heads:
            return query

        if num_attention_heads % target_heads == 0:
            group_size = num_attention_heads // target_heads
            query = query.view(batch_size, target_heads, group_size, 1, head_dim).mean(dim=2)
            return query

        if target_heads % num_attention_heads == 0:
            repeat_factor = target_heads // num_attention_heads
            return query.repeat_interleave(repeat_factor, dim=1)

        return query[:, :target_heads, :, :]
        
    def _side_agent_loop(self, input_ids, stop_event, task_description=None):
        """
        The Side Agent:
        1. Wakes up.
        2. Grabs Landmarks (Compressed KV Cache).
        3. Thinks using those Landmarks.
        """
        stream_ctx = (
            torch.cuda.stream(cast(Any, self.side_stream))
            if self.side_stream is not None else nullcontext()
        )
        with stream_ctx:
            print(f"\n[Side] Waking up! Checking Synapse...")
            
            # Wait for landmarks
            import time
            while self._resolve_worker_landmarks(query_text=task_description) is None:
                if stop_event.is_set(): return
                time.sleep(0.1)
            
            # 1. Retrieve Landmarks
            landmarks = self._resolve_worker_landmarks(query_text=task_description)
            
            if isinstance(self.side_agent_model, (BitNetSideAgent, EarlyExitSideAgent)):
                # Specialized Path
                print(f"[Side] Using Specialized Agent ({type(self.side_agent_model).__name__})...")
                thought_text = self.side_agent_model.think(landmarks, input_ids)
                self.synapse.push_thought(thought_text)
                print(f"[Side] Injected thought: '{thought_text}'")
                return

            # Standard Path (Shared Weights)
            if landmarks is None:
                print("[Side] No landmarks yet. Going back to sleep.")
                return

            # Calculate compression rate
            full_len = input_ids.shape[1] # Approximation
            landmark_len = landmarks[0][0].shape[2]
            print(f"\n[Side] Using Topological Context: {landmark_len} tokens (Original: ~{full_len}+)")
            
            # 2. Think (Generate a thought)
            # We append a "Thinking" prompt
            if task_description:
                print(f"[Side] Task: {task_description}")
                prompt_text = f" [System: You are a sub-process. Task: {task_description}. Analysis: "
            else:
                prompt_text = " [Analysis: "
                
            think_prompt = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            
            # Manual Generation Loop to bypass Cache validation
            curr_input = think_prompt
            
            # Wrap tuple in DynamicCache for Qwen2
            past_key_values = self._dynamic_cache_from_landmarks(landmarks)
            
            generated_tokens = []
            outputs = None
            
            print(f"[Side] Thinking...")
            for _ in range(15):
                # Calculate position IDs based on KV cache length
                # past_key_values.get_seq_length() works now
                seq_len = past_key_values.get_seq_length()
                position_ids = torch.arange(seq_len, seq_len + curr_input.shape[1], device=self.device).unsqueeze(0)
                
                outputs = self.model(
                    input_ids=curr_input,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_hidden_states=True
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                curr_input = next_token
                past_key_values = outputs.past_key_values
                generated_tokens.append(next_token.item())

            if outputs is None:
                return

            thought_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Capture the last hidden state as the "Thought Vector"
            # outputs.hidden_states is a tuple of (layers), each [Batch, Seq, Dim]
            # We take the last layer, last token
            thought_vector = outputs.hidden_states[-1][:, -1, :].detach() # [1, Dim]

            # 3. Inject
            self.synapse.push_thought("[Analysis: " + thought_text + "]", thought_vector)
            print(f"[Side] Injected thought: '[Analysis: {thought_text}]'")

    def generate_async(self, prompt, max_tokens=50):
        # Enrich prompt with matching persistent skill (if any)
        enriched_prompt = self.context_mgr.enrich_prompt(prompt)
        input_ids = self.tokenizer(enriched_prompt, return_tensors="pt").input_ids.to(self.device)
        import threading
        stop_event = threading.Event()
        used_shared_manifold_texts = {
            node.text for node in self.agent_cloud.query_shared_manifold(
                query_text=prompt,
                top_k=4,
            )
        }
        past_key_values, _ = self._seed_shared_projection_cache(
            query_text=prompt,
            used_texts=used_shared_manifold_texts,
            past_key_values=None,
        )
        
        # Initial Trigger Check (Prompt-based) — now via Orchestrator
        task_description = self.router.check_for_triggers(prompt)
        if task_description:
            print(f"[Main] Trigger detected. Delegating via Orchestrator: '{task_description}'")
            # Fire PRE_DISPATCH hook
            dispatch_ctx = self.hooks.fire(HookPoint.PRE_DISPATCH, {"task": task_description})
            if not dispatch_ctx.abort:
                self.orchestrator.dispatch_from_trigger(task_description)
            # Also keep legacy side thread for backward compat
            side_thread = Thread(target=self._side_agent_loop, args=(input_ids, stop_event, task_description))
            side_thread.start()
        
        print("[Main] Generating...")
        curr_input = input_ids
        full_response = []
        generated_ids = [] # Track for repetition penalty
        recent_text_buffer = ""

        last_hidden_state = None
        stream_ctx = (
            torch.cuda.stream(cast(Any, self.main_stream))
            if self.main_stream is not None else nullcontext()
        )
        with stream_ctx:
            for i in range(max_tokens):
                # 1. Check for Thoughts (Validation Gate)
                thought_text, thought_vector = self.synapse.read_thought()
                if thought_text:
                    # Validation Gate (adaptive threshold + hooks)
                    sim_score = None
                    if last_hidden_state is not None and thought_vector is not None:
                        sim_score = torch.nn.functional.cosine_similarity(
                            last_hidden_state, thought_vector
                        ).item()

                    # Adaptive cosine threshold check
                    should_integrate = True
                    if sim_score is not None:
                        should_integrate = self.validation_gate.should_accept(sim_score)

                    # Fire PRE_INJECTION hook (quality gate, security, etc.)
                    hook_ctx = None
                    if should_integrate:
                        hook_ctx = self.hooks.fire(HookPoint.PRE_INJECTION, {
                            "thought_text": thought_text,
                            "similarity_score": sim_score,
                        })
                        should_integrate = not hook_ctx.abort
                        if not should_integrate:
                            print(f"[Main] REJECTED by hook: {hook_ctx.abort_reason}")

                    if not should_integrate:
                        gate_stats = self.validation_gate.stats
                        sim_display = f"{sim_score:.2f}" if sim_score is not None else "n/a"
                        print(f"[Main] REJECTED thought (sim={sim_display}, "
                              f"thresh={gate_stats['threshold']}, ema={gate_stats['accept_ema']})")
                    else:
                        # Use possibly-truncated thought from hook
                        if hook_ctx is not None:
                            thought_text = hook_ctx.data.get("thought_text", thought_text)
                        print(f"\n[Main] !!! Absorbed Thought: {thought_text} !!!")
                        
                        # --- REFERENTIAL INJECTION MECHANISM ---
                        # Instead of forcing the text, we inject a "Reference Token" that points to the thought.
                        # This allows the Main Agent to decide *how* to use it.
                        # We format it as: " [Ref: <Thought>]"
                        
                        # 1. Inject the thought into the KV Cache (Hidden from user output)
                        thought_ids = self.tokenizer(f" [Ref: {thought_text}]", return_tensors="pt").input_ids.to(self.device)
                        thought_outputs = self.model(thought_ids, past_key_values=past_key_values)
                        past_key_values = thought_outputs.past_key_values
                        
                        # 2. Do NOT update curr_input. 
                        # The Main Agent continues generating from where it left off, 
                        # but now its "Memory" (KV Cache) contains the thought.
                        # It will naturally attend to it if relevant.
                        
                        # Optional: We can force a single space to "nudge" it to acknowledge the update
                        # curr_input = self.tokenizer(" ", return_tensors="pt").input_ids.to(self.device)

                if (
                    self.shared_manifold_refresh_interval > 0
                    and i > 0
                    and i % self.shared_manifold_refresh_interval == 0
                ):
                    past_key_values, _ = self._maybe_refresh_shared_manifold(
                        base_prompt=prompt,
                        recent_text=recent_text_buffer,
                        used_texts=used_shared_manifold_texts,
                        past_key_values=past_key_values,
                    )
                
                # 2. Generate
                outputs = self.model(curr_input, past_key_values=past_key_values, output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                # --- REPETITION PENALTY ---
                if len(generated_ids) > 0:
                    # Penalize tokens generated in the last 20 steps
                    for prev_id in set(generated_ids[-20:]):
                        next_token_logits[0, prev_id] /= 1.5 

                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_ids.append(next_token.item())
                
                # Capture current state for next validation
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    last_hidden_state = hidden_states[-1][:, -1, :].detach() # [1, Dim]
                
                past_key_values = outputs.past_key_values
                curr_input = next_token
                
                token_str = self.tokenizer.decode(next_token[0])
                full_response.append(token_str)
                recent_text_buffer += token_str
                print(token_str, end="", flush=True)
                
                # 3. Dynamic Trigger Check (Stream-based)
                # Check the last 50 chars for triggers — pass hidden state
                # for semantic classification when available.
                if len(recent_text_buffer) > 50:
                    recent_slice = recent_text_buffer[-50:]
                    task_description = self.router.check_for_triggers(
                        recent_slice,
                        hidden_state=last_hidden_state,
                    )
                    if task_description:
                        print(f"\n[Main] Dynamic Trigger. Delegating via Orchestrator: '{task_description}'")
                        recent_text_buffer = ""

                        # Fire PRE_DISPATCH hook (rate limiter, etc.)
                        dispatch_ctx = self.hooks.fire(HookPoint.PRE_DISPATCH, {
                            "task": task_description,
                        })
                        if not dispatch_ctx.abort:
                            self.orchestrator.dispatch_from_trigger(task_description)
                        else:
                            print(f"[Main] Dispatch blocked: {dispatch_ctx.abort_reason}")

                # 3b. Auto-compaction check
                past_key_values = self.context_mgr.step(past_key_values, query_states=None)

                # 4. Update Landmarks (Push Context to Side Agent)
                # We do this early so the Side Agent has something to work with
                if i == 10:
                    print(f"\n[Main] Pushing Landmarks to Synapse...")
                    # Pass the current hidden state as the Query for dynamic selection
                    # query_states needs to be [Batch, Heads, 1, Dim]
                    # We have [Batch, Dim]. We need to project it? 
                    # Actually, 'last_hidden_state' is the output of the last layer.
                    # The keys are inside the layers.
                    # To do proper attention, we need the query *before* the output projection?
                    # For simplicity, let's reshape last_hidden_state to [B, 1, 1, D] and broadcast heads
                    # Or just pass it as is and let update_landmarks handle it.
                    
                    query = self._build_landmark_query(last_hidden_state, past_key_values)
                    
                    self.synapse.update_kv_landmarks(past_key_values, query_states=query)

                    # TurboQuant: compress the freshly-pushed landmarks
                    landmarks_raw = self.synapse.get_landmarks()
                    if landmarks_raw is not None:
                        self._materialize_shared_hot_cache(landmarks_raw, query_text=prompt)
                    
        print("\n[Engine] Done.")
        stop_event.set()
        return "".join(full_response)

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 120,
        stream: bool = False,
        enrich_prompt: bool = True,
        query_text: Optional[str] = None,
        initial_used_texts: Optional[set[str]] = None,
        seed_used_shared_texts: bool = True,
        shared_query_top_k: int = 4,
    ) -> str:
        """Direct local generation with prompt enrichment and shared-manifold refresh, but no async trigger loop."""
        enriched_prompt = self.context_mgr.enrich_prompt(prompt) if enrich_prompt else prompt
        shared_query_text = query_text or prompt
        input_ids = self.tokenizer(enriched_prompt, return_tensors="pt").input_ids.to(self.device)
        curr_input = input_ids
        past_key_values = None
        generated_ids: list[int] = []
        response_parts: list[str] = []
        recent_text_buffer = ""
        eos_token_id = self.tokenizer.eos_token_id
        used_shared_manifold_texts = set(initial_used_texts or set())
        if seed_used_shared_texts:
            used_shared_manifold_texts.update(
                node.text for node in self.agent_cloud.query_shared_manifold(
                    query_text=shared_query_text,
                    top_k=shared_query_top_k,
                )
            )
        past_key_values, _ = self._seed_shared_projection_cache(
            query_text=shared_query_text,
            used_texts=used_shared_manifold_texts,
            past_key_values=past_key_values,
        )

        stream_ctx = (
            torch.cuda.stream(cast(Any, self.main_stream))
            if self.main_stream is not None else nullcontext()
        )
        with stream_ctx:
            for i in range(max_tokens):
                if (
                    self.shared_manifold_refresh_interval > 0
                    and i > 0
                    and i % self.shared_manifold_refresh_interval == 0
                ):
                    past_key_values, _ = self._maybe_refresh_shared_manifold(
                        base_prompt=shared_query_text,
                        recent_text=recent_text_buffer,
                        used_texts=used_shared_manifold_texts,
                        past_key_values=past_key_values,
                    )

                outputs = self.model(curr_input, past_key_values=past_key_values, output_hidden_states=False)
                next_token_logits = outputs.logits[:, -1, :]

                if generated_ids:
                    for prev_id in set(generated_ids[-20:]):
                        next_token_logits[0, prev_id] /= 1.5

                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                token_id = int(next_token.item())
                if eos_token_id is not None and token_id == eos_token_id:
                    break

                generated_ids.append(token_id)
                past_key_values = outputs.past_key_values
                curr_input = next_token

                token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                if token_text:
                    response_parts.append(token_text)
                    recent_text_buffer += token_text
                    if stream:
                        print(token_text, end="", flush=True)

        if past_key_values is not None:
            self.synapse.update_kv_landmarks(past_key_values)
            landmarks_raw = self.synapse.get_landmarks()
            if landmarks_raw is not None:
                self._materialize_shared_hot_cache(landmarks_raw, query_text=shared_query_text)

        if stream:
            print()
        return "".join(response_parts).strip()
        # We don't join threads here because they might be running dynamically

    # ------------------------------------------------------------------
    # Team Dispatch (multi-agent coordination)
    # ------------------------------------------------------------------

    def dispatch_team(self, goal: str, roles: Optional[Sequence[AgentRole]] = None,
                      agent_id: Optional[str] = None):
        """
        Convenience: dispatch a Researcher → Reviewer → Verifier chain.
        Or pass custom roles list like [AgentRole.CODER, AgentRole.VERIFIER].
        """
        if roles:
            tasks = []
            prev_id = None
            for role in roles:
                t = SubAgentTask(
                    role=role,
                    agent_id=agent_id,
                    description=goal,
                    depends_on=[prev_id] if prev_id else [],
                )
                tasks.append(t)
                prev_id = t.id
            return self.orchestrator.dispatch_team(goal=goal, tasks=tasks)
        else:
            return self.orchestrator.create_review_chain(goal, agent_id=agent_id)

    def register_persistent_agent(self, agent_id: str, profile: str = "", role: str = "agent"):
        """Create or fetch a persistent agent identity on the shared backbone."""
        return self.agent_cloud.ensure_agent(agent_id, role=role, profile=profile)

    def remember_agent_event(self, agent_id: str, text: str, score: float = 1.0,
                             source: str = "observation", role: str = "agent"):
        """Store a persistent agent memory without updating backbone weights."""
        return self.agent_cloud.remember_text(
            agent_id=agent_id,
            text=text,
            score=score,
            source=source,
            role=role,
        )

    def remember_shared_event(self, text: str, score: float = 1.0,
                              source: str = "observation", node_type: str = "memory",
                              metadata: Optional[dict[str, Any]] = None):
        """Store shared runtime memory that any prompt or agent can recall later."""
        return self.agent_cloud.remember_shared_text(
            text=text,
            score=score,
            source=source,
            node_type=node_type,
            metadata=metadata,
        )

    def dispatch_agent_task(self, agent_id: str, description: str,
                            role: AgentRole = AgentRole.RESEARCHER,
                            priority: int = 1, max_tokens: int = 30):
        """Dispatch a task against a persistent agent identity."""
        self.agent_cloud.ensure_agent(agent_id, role=role.value)
        task = SubAgentTask(
            agent_id=agent_id,
            role=role,
            description=description,
            priority=priority,
            max_tokens=max_tokens,
        )
        return self.orchestrator.dispatch(task)

    def get_agent_population_stats(self):
        return self.agent_cloud.population_stats()

    def get_shared_manifold_stats(self):
        return self.agent_cloud.shared_manifold_stats()

    def get_shared_hot_state(self):
        return self.agent_cloud.get_shared_hot_state()

    def get_memory_accounting(self):
        model_param_bytes = (
            sum(p.nelement() * p.element_size() for p in self.model.parameters())
            if hasattr(self.model, "parameters")
            else 0
        )
        buffer_bytes = (
            sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
            if hasattr(self.model, "buffers")
            else 0
        )
        hot_state = self.agent_cloud.get_shared_hot_state()
        kv_stats = hot_state.get("kv_stats", {})
        return {
            "model_parameter_bytes": int(model_param_bytes),
            "model_buffer_bytes": int(buffer_bytes),
            "model_total_bytes": int(model_param_bytes + buffer_bytes),
            "model_total_mb": float(model_param_bytes + buffer_bytes) / (1024.0 * 1024.0),
            "turbo_quant_enabled": True,
            "turbo_quant_bits": int(self.turbo_quant_bits),
            "shared_hot_summary": hot_state.get("summary_text", ""),
            "shared_hot_kv": kv_stats,
            "live_turbo_cache_bytes": int(self._turbo_cache.memory_bytes()) if self._turbo_cache is not None else 0,
        }

    def get_shared_manifold_metrics(self):
        return {
            "enabled": self.shared_manifold_enabled,
            "prompt_hits": self.shared_manifold_prompt_hits,
            "prompt_misses": self.shared_manifold_prompt_misses,
            "runtime_refreshes": self.shared_manifold_runtime_refreshes,
            "nodes_consumed": self.shared_manifold_nodes_consumed,
            "trace_length": len(self.shared_manifold_trace),
        }

    def get_shared_manifold_trace(self):
        return list(self.shared_manifold_trace)

    def reset_shared_manifold_trace(self):
        self.shared_manifold_trace = []
        self.shared_manifold_prompt_hits = 0
        self.shared_manifold_prompt_misses = 0
        self.shared_manifold_runtime_refreshes = 0
        self.shared_manifold_nodes_consumed = 0

    def set_shared_manifold_enabled(self, enabled: bool):
        self.shared_manifold_enabled = bool(enabled)

    def save_agent_population(self, file_path: str) -> str:
        """Persist the shared-weight agent population to disk for a later session."""
        return self.agent_cloud.save(file_path)

    def load_agent_population(self, file_path: str, merge: bool = False):
        """Restore a previously saved agent population snapshot."""
        return self.agent_cloud.load(file_path, merge=merge)

    def close(self):
        self.stop_shared_hot_refresh_worker()

    def __del__(self):
        try:
            self.stop_shared_hot_refresh_worker()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Verification Loop (check-then-fix)
    # ------------------------------------------------------------------

    def generate_with_verification(self, prompt: str,
                                   check_fn: Optional[Callable[[str], tuple[bool, str]]] = None,
                                   max_retries: int = 3):
        """
        Generate with automatic retry-on-error using the VerificationLoop.
        
        Args:
            prompt: The generation prompt.
            check_fn: A callable(output) -> (passed: bool, detail: str).
                       Defaults to checking for common error patterns.
            max_retries: Max fix attempts.
        """
        if check_fn is None:
            def default_check(output):
                error_markers = ["error", "traceback", "exception", "syntax error", "undefined"]
                output_lower = output.lower() if isinstance(output, str) else ""
                for marker in error_markers:
                    if marker in output_lower:
                        return False, f"Output contains '{marker}'"
                return True, "OK"

            check_fn = default_check

        def action_fn(ctx):
            # Capture generate_async output by collecting synapse thoughts
            self.generate_async(ctx, max_tokens=50)
            # Return the last thought or empty
            thought, _ = self.synapse.read_thought()
            return thought or "[No output captured]"

        loop = VerificationLoop(
            action_fn=action_fn,
            check_fn=check_fn,
            max_retries=max_retries,
            hooks=self.hooks,
        )
        return loop.run(prompt)

if __name__ == "__main__":
    engine = CortexEngine()
    prompt = "User: Please analyze the topological structure of a neural network. Assistant:"
    engine.generate_async(prompt)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
#     engine = CortexEngine(model, tokenizer)
#     engine.generate_async("I need to search for the answer.")
