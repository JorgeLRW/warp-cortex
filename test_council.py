import torch
import time
from cortex_engine import CortexEngine
from threading import Thread

def test_council_scalability():
    print("--- Testing Council Scalability (Shared Model) ---")
    
    # Initialize Engine with Early Exit (Zero VRAM overhead)
    engine = CortexEngine(side_mode="early_exit")
    
    # Mock Landmarks for the test
    # We need to manually populate the synapse because we aren't running the main loop
    print("Populating Synapse with Mock Landmarks...")
    # Create dummy KV cache: 1 layer, tuple(k, v)
    # Shape: [1, 2, 10, 64] (Batch, Heads, Seq, Dim)
    dummy_k = torch.randn(1, 2, 10, 896, device=engine.device, dtype=torch.float16) # 896 is Qwen 0.5B dim
    dummy_v = torch.randn(1, 2, 10, 896, device=engine.device, dtype=torch.float16)
    # Qwen has 24 layers. We need a tuple of 24 tuples.
    dummy_kv = tuple([(dummy_k, dummy_v) for _ in range(24)])
    
    engine.synapse.update_landmarks(dummy_kv)
    
    # Spawn 5 Side Agents
    print("\n>>> Spawning 5 Concurrent Side Agents <<<")
    threads = []
    stop_event = torch.cuda.Event() # Dummy event
    
    # We need a threading event for the loop
    import threading
    stop_thread_event = threading.Event()
    
    start_time = time.time()
    
    for i in range(5):
        # Each agent gets a different prompt
        prompt = f"Agent {i} Analysis"
        input_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
        
        t = Thread(target=engine._side_agent_loop, args=(input_ids, stop_thread_event))
        threads.append(t)
        t.start()
        
    # Wait for them
    for t in threads:
        t.join()
        
    end_time = time.time()
    print(f"\nAll 5 Agents finished in {end_time - start_time:.4f}s")
    print(f"Thoughts in Synapse: {len(engine.synapse.thought_memory)}")
    print(f"Synapse Content: {engine.synapse.thought_memory}")

if __name__ == "__main__":
    test_council_scalability()
