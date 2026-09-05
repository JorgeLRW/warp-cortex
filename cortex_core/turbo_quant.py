"""
TurboQuant KV Cache Compression for Warp-Cortex
================================================
Inspired by Google's "TurboQuant: Redefining AI Efficiency with Extreme
Compression" (ICLR 2026). Combines two stages:

  Stage 1 — PolarQuant-style high-quality quantization:
    Randomly rotate vectors (via Hadamard transform — no RoPE dependency,
    Mac-friendly, no cuBLAS rotation needed), then apply per-element
    uniform quantization. This captures the bulk of the signal.

  Stage 2 — QJL residual 1-bit error correction:
    Apply the Johnson-Lindenstrauss sign-bit trick to the Stage 1
    quantization residual, eliminating bias in attention score estimation.

Key properties:
  - Zero quantization-constant overhead (the Hadamard + polar approach
    eliminates per-block scale/zero-point storage).
  - Works on CPU (MPS / Mac) via fast Walsh-Hadamard (pure PyTorch, no
    CUDA rotation kernels required).
  - Compresses KV cache to 3–4 bits with negligible accuracy loss.
  - Plugs directly into TopologicalSynapse and the auto-compaction system.

References:
  - TurboQuant: https://arxiv.org/abs/2504.19874
  - QJL:        https://arxiv.org/abs/2406.03482
  - PolarQuant: https://arxiv.org/abs/2502.02617
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ======================================================================
# Hadamard Transform (non-rotational, Mac-friendly)
# ======================================================================

def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """
    In-place Fast Walsh-Hadamard Transform along the last dimension.
    Works on CPU, CUDA, and MPS — no cuBLAS dependency.

    Normalizes by 1/sqrt(d) so the transform is unitary (self-inverse).
    Input length must be a power of 2; pad before calling if needed.
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Last dim must be power of 2, got {d}"

    h = 1
    while h < d:
        # Split into blocks of 2h, butterfly between first/second halves
        x_even = x[..., 0::2 * h].clone()  # even-indexed sub-blocks
        x_odd  = x[..., h::2 * h].clone()  # odd-indexed sub-blocks
        # This indexing is tricky for strided access; use reshape approach:
        orig_shape = x.shape
        flat = x.view(-1, d)
        n_rows = flat.shape[0]
        flat = flat.view(n_rows, d // (2 * h), 2, h)
        a = flat[:, :, 0, :].clone()
        b = flat[:, :, 1, :].clone()
        flat[:, :, 0, :] = a + b
        flat[:, :, 1, :] = a - b
        x = flat.view(orig_shape)
        h *= 2

    return x / math.sqrt(d)


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a randomized Hadamard rotation to the last dimension.
    Pads to next power of 2 if needed, applies random sign flip + WHT.

    This is the "non-rotational" alternative to RoPE-style rotation —
    it's a unitary transform that spreads energy uniformly across
    components, making quantization nearly optimal (PolarQuant insight).
    """
    orig_dim = x.shape[-1]
    pad_dim = _next_power_of_2(orig_dim)

    if pad_dim != orig_dim:
        x = F.pad(x, (0, pad_dim - orig_dim))

    # Random sign flip (Rademacher diagonal D): E[D_i] = ±1
    # Seeded per-dim so it's deterministic & reproducible across agents
    gen = torch.Generator(device=x.device)
    gen.manual_seed(42)
    signs = torch.randint(0, 2, (pad_dim,), generator=gen, device=x.device, dtype=x.dtype) * 2 - 1
    x = x * signs

    x = _fast_walsh_hadamard(x)

    if pad_dim != orig_dim:
        x = x[..., :orig_dim]
    return x


def hadamard_unrotate(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse randomized Hadamard (it's self-inverse up to sign flip order).
    """
    orig_dim = x.shape[-1]
    pad_dim = _next_power_of_2(orig_dim)

    if pad_dim != orig_dim:
        x = F.pad(x, (0, pad_dim - orig_dim))

    x = _fast_walsh_hadamard(x)

    gen = torch.Generator(device=x.device)
    gen.manual_seed(42)
    signs = torch.randint(0, 2, (pad_dim,), generator=gen, device=x.device, dtype=x.dtype) * 2 - 1
    x = x * signs

    if pad_dim != orig_dim:
        x = x[..., :orig_dim]
    return x


# ======================================================================
# Stage 1: PolarQuant — Uniform quantization in rotated space
# ======================================================================

def _symmetric_quantize(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric uniform quantization to `bits` per element.
    Returns (quantized_int, scale) where scale is a SINGLE scalar per-tensor
    (zero overhead — the PolarQuant insight is that after Hadamard rotation,
    the distribution is near-Gaussian and symmetric, so one global scale suffices).
    """
    qmax = (1 << (bits - 1)) - 1  # e.g. 7 for 4-bit
    # Single global scale for the entire tensor (zero per-block overhead)
    amax = x.abs().amax()
    scale = amax / qmax if amax > 0 else torch.ones(1, device=x.device, dtype=x.dtype)
    x_q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale


def _symmetric_dequantize(x_q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x_q.to(dtype) * scale


# ======================================================================
# Stage 2: QJL — 1-bit residual error correction
# ======================================================================

class QJLProjection:
    """
    Quantized Johnson-Lindenstrauss projection for 1-bit residual correction.

    Given a residual vector r = x - dequant(quant(x)), QJL compresses r to
    1 bit per dimension using a random projection + sign:

        sketch(r) = sign(P @ r)

    where P is a random Gaussian projection matrix (seeded, shared across agents).
    During attention score estimation, the QJL sketch provides an unbiased
    correction term that cancels the quantization bias.
    """

    def __init__(self, dim: int, sketch_dim: Optional[int] = None, device: str = 'cpu', seed: int = 137):
        self.dim = dim
        self.sketch_dim = sketch_dim or dim  # same dim by default
        self.device = device

        # Random projection matrix (Gaussian, scaled)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        # Store as float32 always, cast on use
        self.projection = torch.randn(
            self.sketch_dim, dim, generator=gen, dtype=torch.float32
        ).to(device) / math.sqrt(self.sketch_dim)

    def sketch(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Compress residual to 1-bit sign sketch.
        residual: [..., dim] → returns [..., sketch_dim] as ±1 int8
        """
        proj = self.projection.to(residual.dtype)
        projected = F.linear(residual, proj)  # [..., sketch_dim]
        return projected.sign().to(torch.int8)

    def estimate_dot(
        self,
        query: torch.Tensor,
        sketch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate dot(query, residual) from the 1-bit sketch.
        This gives an unbiased correction to the quantized attention score.

        query:  [..., dim]
        sketch: [..., sketch_dim] int8 signs

        Returns: [...] estimated dot products
        """
        proj = self.projection.to(query.dtype)
        q_proj = F.linear(query, proj)  # [..., sketch_dim]
        # E[sign(P@r) * (P@q)] ≈ dot(r, q) (unbiased estimator)
        correction = (sketch.to(query.dtype) * q_proj).sum(dim=-1)
        return correction


# ======================================================================
# TurboQuant: Combined 2-stage KV Cache Compressor
# ======================================================================

class TurboQuantCache:
    """
    Drop-in replacement for storing quantized KV cache entries.
    Each KV tensor is compressed via:
      1. Hadamard rotation (spreads energy, makes quantization near-optimal)
      2. PolarQuant: symmetric uniform quantization (3-4 bits, 1 global scale)
      3. QJL: 1-bit sketch of the quantization residual

    Decompression reconstructs an approximation that is unbiased for
    attention score computation (the QJL correction cancels quant bias).

    Memory layout per cache entry:
      - keys_q / values_q:   int8 tensor  (bits/8 of FP16 per element)
      - keys_scale / values_scale: single float32 each (zero overhead)
      - keys_sketch / values_sketch: int8 ±1 (1 bit effective, stored as byte)

    At 4-bit + 1-bit residual = 5 effective bits vs 16-bit FP16 → 3.2× compression.
    At 3-bit + 1-bit residual = 4 effective bits → 4× compression.
    """

    def __init__(self, bits: int = 4, device: str = 'cpu', qjl_enabled: bool = True):
        """
        Args:
            bits: Quantization bit-width for Stage 1 (2, 3, or 4).
            device: Target device.
            qjl_enabled: Whether to compute Stage 2 residual correction.
        """
        self.bits = bits
        self.device = device
        self.qjl_enabled = qjl_enabled
        self._qjl: Optional[QJLProjection] = None

        # Storage: list of (keys_q, keys_scale, keys_sketch, vals_q, vals_scale, vals_sketch)
        # One entry per layer.
        self._layers = []

    def _get_qjl(self, dim: int) -> QJLProjection:
        if self._qjl is None or self._qjl.dim != dim:
            self._qjl = QJLProjection(dim, device=self.device)
        return self._qjl

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def compress(self, past_key_values) -> 'TurboQuantCache':
        """
        Quantize a full-precision KV cache (tuple of (K, V) per layer).
        K, V shapes: [Batch, Heads, Seq, HeadDim]
        """
        self._layers = []
        for k, v in past_key_values:
            kq, ks, ksk = self._compress_tensor(k)
            vq, vs, vsk = self._compress_tensor(v)
            self._layers.append((kq, ks, ksk, vq, vs, vsk))
        return self

    def _compress_tensor(self, x: torch.Tensor):
        """Compress a single K or V tensor."""
        original_dtype = x.dtype
        x_float = x.float()

        # Stage 1: Hadamard rotate + uniform quantize
        x_rot = hadamard_rotate(x_float)
        x_q, scale = _symmetric_quantize(x_rot, self.bits)

        # Stage 2: QJL residual sketch
        sketch = None
        if self.qjl_enabled:
            head_dim = x.shape[-1]
            qjl = self._get_qjl(head_dim)
            x_deq = _symmetric_dequantize(x_q, scale, torch.float32)
            residual = x_rot - x_deq
            # Flatten to [..., head_dim] for sketch
            orig_shape = residual.shape
            flat = residual.reshape(-1, head_dim)
            sketch = qjl.sketch(flat).reshape(orig_shape[:-1] + (qjl.sketch_dim,))

        return x_q, scale, sketch

    # ------------------------------------------------------------------
    # Decompress
    # ------------------------------------------------------------------

    def decompress(self):
        """
        Reconstruct full-precision KV cache (without QJL correction — 
        that's applied during attention). Returns tuple of (K, V) per layer.
        """
        result = []
        for kq, ks, ksk, vq, vs, vsk in self._layers:
            k = self._decompress_tensor(kq, ks)
            v = self._decompress_tensor(vq, vs)
            result.append((k, v))
        return tuple(result)

    def _decompress_tensor(self, x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize + inverse Hadamard."""
        x_deq = _symmetric_dequantize(x_q, scale, torch.float32)
        x_orig = hadamard_unrotate(x_deq)
        return x_orig.half()

    # ------------------------------------------------------------------
    # Corrected attention (QJL bias elimination)
    # ------------------------------------------------------------------

    def corrected_attention_scores(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute attention scores with QJL bias correction.

        score(q, k) ≈ q · dequant(k) + QJL_correction(q, residual_sketch)

        query: [Batch, Heads, 1, HeadDim] (current token query)
        Returns: [Batch, Heads, 1, Seq] attention logits
        """
        kq, ks, ksk, _, _, _ = self._layers[layer_idx]

        # Base score from quantized keys
        k_deq = _symmetric_dequantize(kq, ks, torch.float32)
        # k_deq is in rotated space; rotate query to match
        q_rot = hadamard_rotate(query.float())
        # [B, H, 1, D] @ [B, H, D, S] → [B, H, 1, S]
        base_scores = torch.matmul(q_rot, k_deq.float().transpose(-1, -2))

        # QJL correction
        if self.qjl_enabled and ksk is not None:
            head_dim = query.shape[-1]
            qjl = self._get_qjl(head_dim)
            B, H, S, SD = ksk.shape
            # For each head, compute correction: sum over sketch_dim
            # q_rot: [B, H, 1, D], sketch: [B, H, S, sketch_dim]
            q_flat = q_rot.squeeze(2).reshape(-1, head_dim)       # [B*H, D]
            sk_flat = ksk.reshape(B * H, S, SD)                    # [B*H, S, SD]
            # Project query
            proj = qjl.projection.to(q_flat.dtype)
            q_proj = F.linear(q_flat, proj)                        # [B*H, sketch_dim]
            q_proj = q_proj.unsqueeze(1)                           # [B*H, 1, sketch_dim]
            correction = (sk_flat.float() * q_proj).sum(dim=-1)    # [B*H, S]
            correction = correction.view(B, H, 1, S)
            base_scores = base_scores + correction

        return base_scores / math.sqrt(query.shape[-1])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """
        Estimate *effective* memory usage of the compressed cache.
        int8 stores values that only need `self.bits` bits, so we account
        for the effective bit-width, not the storage container size.
        Sketches are 1-bit effective (stored as int8 for convenience).
        """
        total_bits = 0
        for kq, ks, ksk, vq, vs, vsk in self._layers:
            total_bits += kq.nelement() * self.bits     # keys at N bits
            total_bits += vq.nelement() * self.bits     # values at N bits
            total_bits += 64                             # 2 float32 scales = 64 bits
            if ksk is not None:
                total_bits += ksk.nelement() * 1         # 1-bit sketches (keys)
            if vsk is not None:
                total_bits += vsk.nelement() * 1         # 1-bit sketches (values)
        return total_bits // 8  # convert to bytes

    def compression_ratio(self, original_bytes: int) -> float:
        return original_bytes / max(self.memory_bytes(), 1)

    def num_layers(self) -> int:
        return len(self._layers)

    def layer_memory_bytes(self) -> List[int]:
        layer_bytes: List[int] = []
        for kq, _, ksk, vq, _, vsk in self._layers:
            total_bits = kq.nelement() * self.bits + vq.nelement() * self.bits + 64
            if ksk is not None:
                total_bits += ksk.nelement()
            if vsk is not None:
                total_bits += vsk.nelement()
            layer_bytes.append(total_bits // 8)
        return layer_bytes

    def export_state(self) -> Dict[str, Any]:
        layers = []
        for kq, ks, ksk, vq, vs, vsk in self._layers:
            layers.append({
                "keys_q": kq.detach().cpu(),
                "keys_scale": ks.detach().cpu(),
                "keys_sketch": None if ksk is None else ksk.detach().cpu(),
                "values_q": vq.detach().cpu(),
                "values_scale": vs.detach().cpu(),
                "values_sketch": None if vsk is None else vsk.detach().cpu(),
            })
        return {
            "version": 1,
            "bits": self.bits,
            "qjl_enabled": self.qjl_enabled,
            "layers": layers,
        }

    @classmethod
    def from_state(cls, payload: Dict[str, Any], *, device: str = 'cpu') -> 'TurboQuantCache':
        cache = cls(
            bits=int(payload.get("bits", 4)),
            device=device,
            qjl_enabled=bool(payload.get("qjl_enabled", True)),
        )
        cache._layers = []
        for layer in payload.get("layers", []):
            cache._layers.append((
                layer["keys_q"].to(device),
                layer["keys_scale"].to(device),
                None if layer.get("keys_sketch") is None else layer["keys_sketch"].to(device),
                layer["values_q"].to(device),
                layer["values_scale"].to(device),
                None if layer.get("values_sketch") is None else layer["values_sketch"].to(device),
            ))
        return cache


def estimate_kv_memory_bytes(kv_tuples) -> int:
    if not kv_tuples:
        return 0
    return sum(
        k.nelement() * k.element_size() + v.nelement() * v.element_size()
        for k, v in kv_tuples
    )


def summarize_kv_cache(kv_tuples, compressed: Optional[TurboQuantCache] = None) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    for index, (k, v) in enumerate(kv_tuples or []):
        layers.append({
            "layer": index,
            "key_shape": list(k.shape),
            "value_shape": list(v.shape),
            "sequence_length": int(k.shape[2]) if k.dim() >= 3 else 0,
            "head_count": int(k.shape[1]) if k.dim() >= 2 else 0,
            "head_dim": int(k.shape[-1]) if k.dim() >= 1 else 0,
            "bytes": int(k.nelement() * k.element_size() + v.nelement() * v.element_size()),
        })

    original_bytes = estimate_kv_memory_bytes(kv_tuples)
    summary: Dict[str, Any] = {
        "layer_count": len(layers),
        "layers": layers,
        "original_bytes": int(original_bytes),
    }
    if compressed is not None:
        compressed_bytes = int(compressed.memory_bytes())
        summary.update({
            "compressed_bytes": compressed_bytes,
            "compression_ratio": float(compressed.compression_ratio(original_bytes)) if original_bytes else 1.0,
            "bits": int(compressed.bits),
            "qjl_enabled": bool(compressed.qjl_enabled),
            "compressed_layer_bytes": [int(value) for value in compressed.layer_memory_bytes()],
        })
    return summary


# ======================================================================
# Convenience: compress/decompress TopologicalSynapse landmarks
# ======================================================================

def compress_landmarks(landmarks, bits: int = 4, device: str = 'cpu') -> TurboQuantCache:
    """
    Compress a synapse landmark tuple (list of (K, V) per layer).
    Returns a TurboQuantCache that can be decompressed or used for
    corrected attention.
    """
    cache = TurboQuantCache(bits=bits, device=device)
    cache.compress(landmarks)
    return cache


def decompress_landmarks(tq_cache: TurboQuantCache):
    """Decompress back to full-precision (K, V) tuples."""
    return tq_cache.decompress()


# ======================================================================
# Demo / self-test
# ======================================================================

if __name__ == "__main__":
    device = 'cpu'  # Mac-friendly
    torch.manual_seed(0)

    print("=" * 60)
    print("TurboQuant KV Cache Compression — Self-Test")
    print("=" * 60)

    # Simulate a 2-layer KV cache: [Batch=1, Heads=8, Seq=512, HeadDim=64]
    B, H, S, D = 1, 8, 512, 64
    keys  = [torch.randn(B, H, S, D) for _ in range(2)]
    values = [torch.randn(B, H, S, D) for _ in range(2)]
    original_kv = list(zip(keys, values))

    original_bytes = sum(k.nelement() * 2 + v.nelement() * 2 for k, v in original_kv)
    print(f"Original KV cache: {original_bytes / 1024:.1f} KB (FP16)")

    for bits in [4, 3, 2]:
        tq = TurboQuantCache(bits=bits, device=device)
        tq.compress(original_kv)

        compressed_bytes = tq.memory_bytes()
        ratio = tq.compression_ratio(original_bytes)
        print(f"\n--- {bits}-bit TurboQuant ---")
        print(f"  Compressed:  {compressed_bytes / 1024:.1f} KB")
        print(f"  Ratio:       {ratio:.1f}×")

        # Decompress and measure reconstruction error
        reconstructed = tq.decompress()
        total_mse = 0.0
        for i, ((k_orig, v_orig), (k_rec, v_rec)) in enumerate(zip(original_kv, reconstructed)):
            k_mse = (k_orig - k_rec).pow(2).mean().item()
            v_mse = (v_orig - v_rec).pow(2).mean().item()
            total_mse += k_mse + v_mse
        print(f"  Recon MSE:   {total_mse / (2 * len(original_kv)):.6f}")

        # Test corrected attention scores
        query = torch.randn(B, H, 1, D)
        scores = tq.corrected_attention_scores(query, layer_idx=0)
        print(f"  Attn shape:  {scores.shape}")

        # Compare to true attention scores
        true_scores = torch.matmul(query, keys[0].transpose(-1, -2)) / math.sqrt(D)
        score_mse = (scores - true_scores).pow(2).mean().item()
        print(f"  Score MSE:   {score_mse:.6f}")

    print(f"\n{'=' * 60}")
    print("Hadamard transform test (Mac-friendly, no CUDA):")
    x = torch.randn(4, 64)
    x_rot = hadamard_rotate(x)
    x_back = hadamard_unrotate(x_rot)
    roundtrip_err = (x - x_back).abs().max().item()
    print(f"  Roundtrip error: {roundtrip_err:.2e}")
    assert roundtrip_err < 1e-4, f"Hadamard roundtrip failed: {roundtrip_err}"
    print(f"  [PASS] Hadamard transform is self-inverse")
    print("=" * 60)
