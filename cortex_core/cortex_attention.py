import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, cast

from .synapse import TopologicalSynapse


class CortexAttention(nn.Module):
    """
    Standard Self-Attention augmented with Topological Synapse Cross-Attention.

    Reads from the TopologicalSynapse's injection landmarks (not a ring
    buffer).  Cross-attends to ALL current injection landmarks.  The
    topology-induced gate scales the overall synapse contribution based
    on both query-landmark relevance AND the geometric structure of the
    landmark manifold (density, spread, coverage).

    When injection_count == 0, this is pure self-attention (zero overhead).
    When injection_count >= 1, the last token cross-attends to all N
    injection landmarks and the gate decides how much to absorb.
    """

    # Number of topology features appended to the gate input
    N_TOPO_FEATURES = 3

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Standard Self-Attention Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # Synapse Cross-Attention Projections
        self.synapse_k_proj = nn.Linear(dim, dim)
        self.synapse_v_proj = nn.Linear(dim, dim)

        # Topology-induced learnable gate
        # Input: [q_feat, s_feat, topo_density, topo_spread, topo_coverage]
        gate_in = 2 * self.head_dim + self.N_TOPO_FEATURES
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_in, self.head_dim),
            nn.SiLU(),
            nn.Linear(self.head_dim, 1),
        )
        # Initialize conservatively so sigmoid(gate)≈0.12 at start
        gate_in_proj = cast(nn.Linear, self.gate_proj[0])
        gate_out_proj = cast(nn.Linear, self.gate_proj[2])
        nn.init.zeros_(gate_in_proj.weight)
        nn.init.zeros_(gate_out_proj.weight)
        nn.init.constant_(gate_out_proj.bias, -2.0)

    def forward(self, x, synapse: Optional[TopologicalSynapse] = None):
        """
        x: [B, L, D]
        synapse: TopologicalSynapse instance (or None for pure self-attention)
        """
        B, L, D = x.shape

        # 1. Standard Self-Attention
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 2. Synapse Cross-Attention (reads injection landmarks from synapse)
        if synapse is not None:
            inj_k, inj_v = synapse.get_injection_context()

            if inj_k is not None and inj_v is not None and inj_k.shape[0] > 0:
                N = inj_k.shape[0]

                # Project injection landmarks → [1, N, D] → [1, H, N, head_dim]
                s_k = self.synapse_k_proj(inj_k.unsqueeze(0))
                s_k = s_k.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)
                s_v = self.synapse_v_proj(inj_v.unsqueeze(0))
                s_v = s_v.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)
                # Expand for batch
                s_k = s_k.expand(B, -1, -1, -1)
                s_v = s_v.expand(B, -1, -1, -1)

                # Cross-attend: last token queries ALL injection landmarks
                q_last = q[:, :, -1:, :]  # [B, H, 1, head_dim]
                syn_scores = torch.matmul(
                    q_last, s_k.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)  # [B, H, 1, N]
                syn_attn = F.softmax(syn_scores, dim=-1)
                synapse_out = torch.matmul(syn_attn, s_v)  # [B, H, 1, head_dim]

                # Topology-induced gate
                density, spread, coverage = synapse.topo_features()
                topo = torch.tensor(
                    [density, spread, coverage],
                    device=x.device, dtype=x.dtype,
                ).unsqueeze(0).expand(B, -1)  # [B, 3]

                q_feat = q_last[:, 0, 0, :]          # [B, head_dim]
                s_feat = synapse_out[:, 0, 0, :]      # [B, head_dim] (attention-weighted)
                gate_input = torch.cat([q_feat, s_feat, topo], dim=-1)
                gate_val = torch.sigmoid(self.gate_proj(gate_input))  # [B, 1]
                gate_val = gate_val.unsqueeze(1).unsqueeze(1)         # [B, 1, 1, 1]

                # Add gated synapse output to the last token only
                out[:, :, -1:, :] = out[:, :, -1:, :] + (gate_val * synapse_out)

                print(f"[Cortex] Injected from {N} landmark(s). "
                      f"Gate: {gate_val.mean().item():.4f}")

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


# Demo
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 256
    synapse = TopologicalSynapse(dim=dim, max_injections=128, device=device)
    model = CortexAttention(dim=dim, num_heads=8).to(device)
    x = torch.randn(1, 10, dim, device=device)

    print("--- Step 1: Normal Generation (no injections) ---")
    output1 = model(x, synapse)

    print("\n--- Step 2: Inject embedding as landmark ---")
    synapse.inject_embedding(torch.randn(dim, device=device))

    print("--- Step 3: Generation with injection ---")
    output2 = model(x, synapse)

    print("\n[Success] Cortex Attention + Synapse validated.")
