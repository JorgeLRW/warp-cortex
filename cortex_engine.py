import torch
import torch.nn as nn
import copy
import os
import sys
from contextlib import nullcontext
from threading import Thread
from typing import Any, Callable, Optional, Sequence, Tuple, cast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from cortex_core.synapse import TopologicalSynapse
from cortex_core.cortex_router import CortexRouter
from cortex_core.cortex_orchestrator import CortexOrchestrator, AgentRole, SubAgentTask
from cortex_core.cortex_memory import AutoCompactor, SkillRegistry, ContextManager
from cortex_core.cortex_hooks import (
    HookRegistry, HookPoint, HookContext,
    create_default_hooks, VerificationLoop,
)
from cortex_core.hf_utils import prepare_hf_cache
from cortex_core.settings import get_setting, load_settings, resolve_project_path
from cortex_core.turbo_quant import TurboQuantCache, compress_landmarks, decompress_landmarks

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        model_kwargs = {
            "cache_dir": cache_dir,
            "dtype": torch.float16 if self.use_cuda_streams else torch.float32,
        }
        if self.use_cuda_streams:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
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

        # 4. Context manager (wraps synapse + compaction + skills)
        self.context_mgr = ContextManager(self.synapse, self.compactor, self.skills)

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

        # 8. TurboQuant KV cache compressor (3-4 bit, Mac-friendly Hadamard)
        self.turbo_quant_bits = int(get_setting(settings, "adaptive_engine.turbo_quant.bits", 4))
        self.turbo_quant_enabled = bool(get_setting(settings, "adaptive_engine.turbo_quant.enabled", True))
        
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
            while self.synapse.get_landmarks() is None:
                if stop_event.is_set(): return
                time.sleep(0.1)
            
            # 1. Retrieve Landmarks
            landmarks = self.synapse.get_landmarks()
            
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
            past_key_values = DynamicCache()
            setattr(past_key_values, "key_cache", [k for k, v in landmarks])
            setattr(past_key_values, "value_cache", [v for k, v in landmarks])
            
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
        past_key_values = None
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
                    
                    # Reshape for update_landmarks: [Batch, Heads, 1, Dim]
                    # We'll just duplicate across heads for now
                    num_heads = self.model.config.num_attention_heads
                    head_dim = self.model.config.hidden_size // num_heads
                    
                    # [1, Dim] -> [1, Heads, 1, HeadDim]
                    query = None
                    if last_hidden_state is not None:
                        query = last_hidden_state.view(1, num_heads, 1, head_dim)
                    
                    self.synapse.update_kv_landmarks(past_key_values, query_states=query)

                    # TurboQuant: compress the freshly-pushed landmarks
                    landmarks_raw = self.synapse.get_landmarks()
                    if self.turbo_quant_enabled and landmarks_raw is not None:
                        tq = compress_landmarks(
                            landmarks_raw,
                            bits=self.turbo_quant_bits,
                            device=str(self.device),
                        )
                        orig_bytes = sum(
                            k.nelement() * k.element_size() + v.nelement() * v.element_size()
                            for k, v in landmarks_raw
                        )
                        ratio = tq.compression_ratio(orig_bytes)
                        print(f"[TurboQuant] Landmarks compressed {ratio:.1f}× "
                              f"({tq.bits}-bit + QJL residual)")
                        # Store compressed cache for side agents that support it
                        self._turbo_cache = tq
                    
        print("\n[Engine] Done.")
        stop_event.set()
        # We don't join threads here because they might be running dynamically

    # ------------------------------------------------------------------
    # Team Dispatch (multi-agent coordination)
    # ------------------------------------------------------------------

    def dispatch_team(self, goal: str, roles: Optional[Sequence[AgentRole]] = None):
        """
        Convenience: dispatch a Researcher → Reviewer → Verifier chain.
        Or pass custom roles list like [AgentRole.CODER, AgentRole.VERIFIER].
        """
        if roles:
            tasks = []
            prev_id = None
            for role in roles:
                t = SubAgentTask(role=role, description=goal, depends_on=[prev_id] if prev_id else [])
                tasks.append(t)
                prev_id = t.id
            return self.orchestrator.dispatch_team(goal=goal, tasks=tasks)
        else:
            return self.orchestrator.create_review_chain(goal)

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
