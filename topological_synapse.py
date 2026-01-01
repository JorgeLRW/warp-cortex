import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalSynapse:
    """
    A shared memory structure that holds 'Landmarks' - key context states selected by the Main Agent.
    This allows the Side Agent to operate in O(k) complexity where k is the number of landmarks,
    instead of O(L) where L is the full context length.
    """
    def __init__(self, dim, max_landmarks=64, device='cuda'):
        self.dim = dim
        self.max_landmarks = max_landmarks
        self.device = device
        
        # The Landmark Memory (Key-Value pairs)
        # These are the "pinned" thoughts that the Side Agent can see.
        self.landmark_keys = torch.zeros(max_landmarks, dim, device=device)
        self.landmark_values = torch.zeros(max_landmarks, dim, device=device)
        self.landmark_scores = torch.zeros(max_landmarks, device=device) # Importance score
        self.count = 0
        
    def update_landmarks(self, keys, values, attention_scores):
        """
        Main Agent calls this.
        Identifies high-importance tokens and promotes them to Landmarks.
        
        keys, values: [Batch, Seq, Dim] - The Main Agent's current KV cache
        attention_scores: [Batch, Heads, Seq, Seq] - The Main Agent's attention map
        """
        # 1. Calculate "Global Importance" of each token in the current window
        # Sum attention scores received by each token (Column sum of attention matrix)
        # "How much did other tokens attend to this token?"
        # shape: [Seq]
        token_importance = attention_scores.sum(dim=(0, 1, 2)) 
        
        # 2. Select Top-K candidates
        # We only look at the new tokens or re-evaluate the window
        k = min(self.max_landmarks, token_importance.shape[0])
        top_scores, top_indices = torch.topk(token_importance, k)
        
        # 3. Update the Synapse (Simplified: Just overwrite for now)
        # In a real system, we would merge/evict based on score
        self.landmark_keys[:k] = keys[0, top_indices, :]
        self.landmark_values[:k] = values[0, top_indices, :]
        self.landmark_scores[:k] = top_scores
        self.count = k
        
        # print(f"[Synapse] Updated {k} landmarks. Top score: {top_scores[0]:.4f}")

    def get_context(self):
        """
        Side Agent calls this.
        Returns the compressed O(k) context.
        """
        return self.landmark_keys[:self.count], self.landmark_values[:self.count]

class TopologicalSideAgent(nn.Module):
    """
    A Side Agent that attends ONLY to the Topological Synapse.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
    def forward(self, x, synapse: TopologicalSynapse):
        """
        x: [Batch, 1, Dim] - The Side Agent's current thought
        synapse: The shared landmark memory
        """
        B, _, D = x.shape
        
        # 1. Get Landmarks (O(k))
        # k_mem, v_mem: [k, Dim]
        k_mem, v_mem = synapse.get_context()
        
        if k_mem.shape[0] == 0:
            return x # No context yet
            
        # 2. Attend to Landmarks
        # Expand for batch/heads
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        k = k_mem.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)
        v = v_mem.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2).expand(B, -1, -1, -1)
        
        # Attention: Q(Side) * K(Landmarks)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v) # [B, Heads, 1, HeadDim]
        context = context.transpose(1, 2).reshape(B, 1, D)
        
        return self.o_proj(context)

# Simulation
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 128
    seq_len = 1000 # Main context length
    
    synapse = TopologicalSynapse(dim, max_landmarks=10, device=device)
    side_agent = TopologicalSideAgent(dim, num_heads=4).to(device)
    
    print(f"--- Simulation: O(k) Topological Attention ---")
    print(f"Main Context Length: {seq_len}")
    print(f"Synapse Capacity: {synapse.max_landmarks} Landmarks")
    
    # 1. Main Agent runs (Simulated)
    # It has a huge context, but only a few tokens are "important"
    main_keys = torch.randn(1, seq_len, dim, device=device)
    main_values = torch.randn(1, seq_len, dim, device=device)
    
    # Simulate attention scores: Token 50 and Token 900 are very important
    attn_scores = torch.rand(1, 4, seq_len, seq_len, device=device) * 0.1
    attn_scores[:, :, :, 50] += 10.0 # Landmark 1
    attn_scores[:, :, :, 900] += 5.0 # Landmark 2
    
    # 2. Main Agent updates Synapse
    synapse.update_landmarks(main_keys, main_values, attn_scores)
    
    # 3. Side Agent runs (O(k))
    side_input = torch.randn(1, 1, dim, device=device)
    output = side_agent(side_input, synapse)
    
    print(f"Side Agent Output Shape: {output.shape}")
    print(f"Success: Side Agent attended to {synapse.count} landmarks instead of {seq_len} tokens.")
