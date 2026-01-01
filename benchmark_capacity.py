import torch
import time
import threading
import psutil
import os
from cortex_engine import CortexEngine

def get_vram_usage():
    """Returns VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def benchmark_capacity(agent_counts=[10, 50, 100]):
    print("================================================================")
    print("WARP CORTEX: CAPACITY BENCHMARK")
    print("================================================================")
    
    # 1. Initialize Engine (Shared Weights Mode)
    print("\n[1] Initializing Engine (Shared Weights)...")
    initial_vram = get_vram_usage()
    engine = CortexEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct", side_mode="shared")
    base_vram = get_vram_usage()
    print(f"Base VRAM (Model Loaded): {base_vram:.2f} GB")
    
    # 2. Prime the Synapse (Generate Landmarks)
    print("\n[2] Priming Synapse with Landmarks...")
    prompt = "The quick brown fox jumps over the lazy dog. " * 10 # ~100 tokens
    input_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
    
    # Run one forward pass to populate KV cache and landmarks
    with torch.no_grad():
        outputs = engine.model(input_ids, use_cache=True)
        engine.synapse.update_landmarks(outputs.past_key_values)
    
    print(f"Synapse Primed. Landmarks available.")
    
    # 3. Benchmark Loop
    results = []
    
    for count in agent_counts:
        print(f"\n----------------------------------------------------------------")
        print(f"Testing Capacity: {count} Concurrent Agents")
        print(f"----------------------------------------------------------------")
        
        threads = []
        stop_event = threading.Event()
        
        # Define a worker that just runs inference using the shared model
        def agent_worker(agent_id):
            # Each agent gets its own CUDA stream
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                # Simulate reading landmarks (Zero Copy)
                landmarks = engine.synapse.get_landmarks()
                if landmarks is None: return
                
                # Simulate Inference Loop
                # We don't need to generate real text, just allocate the activation memory
                # by running a forward pass
                
                # Create a dummy input for the agent
                dummy_input = torch.tensor([[101]], device=engine.device) # Start token
                
                # We use the shared model
                # In a real scenario, we'd use past_key_values from landmarks
                # Here we just run a forward pass to consume VRAM for activations
                with torch.no_grad():
                    _ = engine.model(dummy_input)
                    
                # Keep the memory alive for a bit
                time.sleep(0.5)

        # Spawn Agents
        start_vram = get_vram_usage()
        
        for i in range(count):
            t = threading.Thread(target=agent_worker, args=(i,))
            threads.append(t)
            t.start()
            
        # Wait for all to be "running" (simulated by sleep)
        time.sleep(0.2) 
        
        peak_vram = get_vram_usage()
        vram_delta = peak_vram - base_vram
        
        print(f"Agents: {count}")
        print(f"Peak VRAM: {peak_vram:.2f} GB")
        print(f"Delta VRAM: {vram_delta:.4f} GB")
        print(f"VRAM per Agent: {(vram_delta / count) * 1000:.2f} MB")
        
        results.append({
            "agents": count,
            "vram": peak_vram,
            "delta": vram_delta
        })
        
        # Cleanup
        for t in threads:
            t.join()
            
        torch.cuda.empty_cache()
        
    print("\n================================================================")
    print("FINAL REPORT")
    print("================================================================")
    print(f"{'Agents':<10} | {'Total VRAM (GB)':<15} | {'Delta (GB)':<15}")
    print("-" * 45)
    for res in results:
        print(f"{res['agents']:<10} | {res['vram']:<15.2f} | {res['delta']:<15.4f}")
    print("================================================================")

if __name__ == "__main__":
    benchmark_capacity()
