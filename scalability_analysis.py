import torch

def calculate_scalability(gpu_vram_gb=24, main_model_size_b=0.5, side_model_size_b=1.0, side_quant_bits=1.58):
    """
    Calculates how many Side Agents can fit alongside a Main Agent.
    """
    print(f"--- Scalability Analysis (GPU: {gpu_vram_gb} GB) ---")
    
    # 1. System Overhead
    system_overhead = 2.0 # GB (CUDA context, display, etc.)
    available_vram = gpu_vram_gb - system_overhead
    print(f"Available VRAM: {available_vram:.2f} GB")
    
    # 2. Main Agent (Assumed FP16 for quality)
    # 2 bytes per param
    main_vram = main_model_size_b * 2 
    # KV Cache for Main (Assumed 4k context, standard attention)
    # 2 * 2 * Layers * Dim * SeqLen
    # Approx 1GB for 70B, less for 0.5B. Let's be conservative.
    main_kv = 1.0 if main_model_size_b > 10 else 0.2
    
    total_main = main_vram + main_kv
    print(f"Main Agent ({main_model_size_b}B FP16): {total_main:.2f} GB")
    
    remaining_vram = available_vram - total_main
    if remaining_vram <= 0:
        print("Error: Main Agent does not fit!")
        return
        
    print(f"Remaining for Side Agents: {remaining_vram:.2f} GB")
    
    # 3. Side Agent (BitNet)
    # bits per param / 8 = bytes per param
    side_weight_vram = side_model_size_b * (side_quant_bits / 8)
    
    # Topological KV Cache (O(k))
    # k=64 tokens. 
    # CRITICAL: This is SHARED memory. All agents read from the same Synapse.
    # The only unique memory is the "Thought" generation (e.g. 10 tokens).
    # 10 tokens * Dim * Layers * 2 bytes ~= negligible.
    side_kv_vram = 0.001 # Effectively zero per agent
    
    total_side = side_weight_vram + side_kv_vram
    print(f"Side Agent ({side_model_size_b}B BitNet {side_quant_bits}b): {total_side:.4f} GB")
    
    # 4. Calculation
    max_agents = int(remaining_vram / total_side)
    
    print(f"\n>>> THEORETICAL LIMIT: {max_agents} Concurrent Side Agents <<<")
    print("(Shared Weights + Shared Topological Cache = Massive Scalability)")

if __name__ == "__main__":
    # Scenario A: Consumer GPU, Small Main
    calculate_scalability(gpu_vram_gb=24, main_model_size_b=0.5, side_model_size_b=1.0)
    
    print("\n" + "="*30 + "\n")
    
    # Scenario B: H100, Large Main
    calculate_scalability(gpu_vram_gb=80, main_model_size_b=70, side_model_size_b=1.0)
    
    print("\n" + "="*30 + "\n")
    
    # Scenario C: Consumer GPU, Large Main (Quantized)
    # 70B @ 4bit = 35GB. Won't fit on 24GB.
    # Let's try 8B Main.
    calculate_scalability(gpu_vram_gb=24, main_model_size_b=8, side_model_size_b=1.0)
