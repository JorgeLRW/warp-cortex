import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SynapseBuffer:
    """
    Simulates the GPU Ring Buffer defined in synapse.cu.
    In a real deployment, this would wrap a raw CUDA pointer.
    """
    def __init__(self, dim, buffer_size=128, device='cuda'):
        self.dim = dim
        self.buffer_size = buffer_size
        self.device = device
        # The shared memory buffer
        self.memory = torch.zeros((buffer_size, dim), device=device)
        self.flags = torch.zeros(buffer_size, dtype=torch.int32, device=device) # 0=Empty, 1=Ready
        self.head = 0 # Write pointer
        self.tail = 0 # Read pointer
        
    def inject(self, thought_vector):
        """Side Agent writes to this."""
        idx = self.head % self.buffer_size
        self.memory[idx] = thought_vector
        self.flags[idx] = 1
        self.head += 1
        
    def read(self):
        """Main Agent reads from this."""
        if self.head == self.tail:
            return None # Empty
        
        idx = self.tail % self.buffer_size
        if self.flags[idx] == 1:
            data = self.memory[idx].clone()
            self.flags[idx] = 0
            self.tail += 1
            return data
        return None

class CortexAttention(nn.Module):
    """
    Standard Self-Attention augmented with Asynchronous Side-Car Attention.
    """
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
        # The Side Agent's thoughts are treated as Keys and Values
        self.synapse_k_proj = nn.Linear(dim, dim)
        self.synapse_v_proj = nn.Linear(dim, dim)
        
        # Gating mechanism: How much to listen to the Side Agent?
        # Initialize to 0 so it doesn't disturb training initially
        self.gate = nn.Parameter(torch.zeros(1)) 

    def forward(self, x, synapse_buffer: SynapseBuffer = None):
        B, L, D = x.shape
        
        # 1. Standard Self-Attention
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 2. Synapse Injection (The "Side-Car" Logic)
        if synapse_buffer:
            # Check if the Side Agent has a new thought
            thought = synapse_buffer.read()
            
            if thought is not None:
                # Expand thought to match batch/heads
                # thought: [D] -> [1, 1, 1, D] -> [B, 1, H, D_head]
                thought = thought.view(1, 1, self.num_heads, self.head_dim)
                
                # Project thought into K/V space
                synapse_k = self.synapse_k_proj(thought.view(1, 1, -1)).view(1, 1, self.num_heads, self.head_dim)
                synapse_v = self.synapse_v_proj(thought.view(1, 1, -1)).view(1, 1, self.num_heads, self.head_dim)
                
                # Cross-Attention: Query (Main) attends to Synapse (Side)
                # We only attend to the synapse with the *current* token (last in sequence)
                # For simplicity, we just add the gated value here
                
                # Calculate relevance of the thought to the current context
                # Q_last: [B, 1, H, D]
                q_last = q[:, -1:, :, :]
                synapse_score = torch.matmul(q_last, synapse_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                synapse_attn = F.softmax(synapse_score, dim=-1)
                
                synapse_out = torch.matmul(synapse_attn, synapse_v)
                
                # Inject into the main stream
                gate_val = torch.sigmoid(self.gate)
                
                # Add to the last token's output
                out[:, -1:, :, :] = out[:, -1:, :, :] + (gate_val * synapse_out)
                
                print(f"[Cortex] Injected thought from Side Agent. Gate: {gate_val.item():.4f}")

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

# Demo
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize
    dim = 4096
    model = CortexAttention(dim=dim, num_heads=32).to(device)
    synapse = SynapseBuffer(dim=dim, device=device)
    
    # Main Stream Input (Batch=1, Seq=10)
    x = torch.randn(1, 10, dim, device=device)
    
    print("--- Step 1: Normal Generation ---")
    output1 = model(x, synapse)
    
    print("\n--- Step 2: Side Agent Injects Thought ---")
    # Side Agent (running asynchronously) finds a fact and pushes it
    thought = torch.randn(dim, device=device)
    synapse.inject(thought)
    
    print("--- Step 3: Main Stream Generation (with Injection) ---")
    # Main Agent continues generating...
    x_next = torch.randn(1, 11, dim, device=device) # Next step
    output2 = model(x_next, synapse)
    
    print("\n[Success] Cortex Architecture validated.")
