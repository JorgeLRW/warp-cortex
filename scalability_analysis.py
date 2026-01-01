import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_theoretical_scalability(gpu_vram_gb=24, main_model_size_b=0.5, side_model_size_b=1.0, side_quant_bits=1.58):
    """
    Calculates how many Side Agents can fit alongside a Main Agent.
    """
    print(f"\n=== Theoretical Scalability Analysis (GPU: {gpu_vram_gb} GB) ===")
    
    # 1. System Overhead
    system_overhead = 2.0 # GB (CUDA context, display, etc.)
    available_vram = gpu_vram_gb - system_overhead
    
    # 2. Main Agent (Assumed FP16 for quality)
    # 2 bytes per param
    main_vram = main_model_size_b * 2 
    # KV Cache for Main (Assumed 4k context, standard attention)
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
    side_kv_vram = 0.001 # Effectively zero per agent
    
    total_side = side_weight_vram + side_kv_vram
    print(f"Side Agent ({side_model_size_b}B BitNet {side_quant_bits}b): {total_side:.4f} GB")
    
    # 4. Calculation
    max_agents = int(remaining_vram / total_side)
    
    print(f">>> THEORETICAL LIMIT: {max_agents} Concurrent Side Agents <<<")
    print("(Shared Weights + Shared Topological Cache = Massive Scalability)")

def run_hardware_benchmark(num_agents=5):
    """
    Runs a real hardware stress test using the Cortex Engine.
    """
    print(f"\n=== Hardware Benchmark: {num_agents} Concurrent Agents ===")
    try:
        from cortex_engine import CortexEngine
    except ImportError:
        print("Error: cortex_engine.py not found. Skipping hardware test.")
        return

    # Initialize Engine (Shared Mode)
    print("Initializing Engine...")
    engine = CortexEngine(side_mode="shared")
    
    # Populate Mock Synapse
    print("Populating Synapse...")
    dummy_kv = []
    for _ in range(engine.model.config.num_hidden_layers):
        k = torch.randn(1, engine.model.config.num_key_value_heads, 10, 
                       engine.model.config.hidden_size // engine.model.config.num_attention_heads, 
                       device=engine.device, dtype=torch.float16)
        v = k.clone()
        dummy_kv.append((k, v))
    engine.synapse.update_landmarks(tuple(dummy_kv))

    # Benchmark
    stop_event = threading.Event()
    threads = []
    
    print("Starting Side Agents...")
    start_time = time.time()
    
    for i in range(num_agents):
        # Dummy input for side agent
        input_ids = torch.randint(0, 1000, (1, 10), device=engine.device)
        t = threading.Thread(target=engine._side_agent_loop, args=(input_ids, stop_event))
        t.start()
        threads.append(t)
        
    # Let them run for 5 seconds
    time.sleep(5)
    stop_event.set()
    
    for t in threads:
        t.join()
        
    end_time = time.time()
    print(f"Benchmark Complete. Duration: {end_time - start_time:.2f}s")
    print(f"Total Thoughts Generated: {len(engine.synapse.thought_memory)}")
    print(f"Thoughts per Second: {len(engine.synapse.thought_memory) / (end_time - start_time):.2f}")

if __name__ == "__main__":
    # 1. Run Math
    calculate_theoretical_scalability(gpu_vram_gb=24, main_model_size_b=0.5, side_model_size_b=1.0)
    
    # 2. Run Code (Optional)
    print("\nRun hardware benchmark? (y/n)")
    # For automation purposes, we default to running if arguments provided or just run it.
    # We'll just run a small test.
    if torch.cuda.is_available():
        run_hardware_benchmark(num_agents=5)
    else:
        print("No GPU detected. Skipping hardware benchmark.")
