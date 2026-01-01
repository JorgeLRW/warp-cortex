import torch
import time
import threading
from cortex_engine import CortexEngine
from cortex_router import CortexRouter

def test_router_logic():
    print("\n=== Testing Cortex Router Logic ===")
    router = CortexRouter()
    
    # Test cases
    test_cases = [
        ("I need to [SEARCH] for the answer.", "Perform a search to verify this information."),
        ("Can you write a python script for this?", "Write a script to solve this."),
        ("[TASK: Verify the quantum physics]", "Verify the quantum physics"),
        ("[DELEGATE: Optimize this loop]", "Optimize this loop"),
        ("Hello, how are you?", None)
    ]
    
    for text, expected_task in test_cases:
        task = router.check_for_triggers(text)
        status = "PASS" if task == expected_task else f"FAIL (Expected '{expected_task}', got '{task}')"
        print(f"Input: '{text}' -> Task: '{task}' [{status}]")

def test_engine_automation():
    print("\n=== Testing Engine Automation (Live) ===")
    if not torch.cuda.is_available():
        print("No GPU detected. Skipping engine test.")
        return

    # Initialize Engine
    print("Initializing Cortex Engine...")
    engine = CortexEngine(side_mode="shared")
    
    # Test 1: Initial Trigger (Explicit Task)
    print("\n>>> Test 1: Prompt-based Trigger (Explicit Task) <<<")
    prompt = "User: [TASK: Analyze the sentiment of this text] I am feeling great today."
    print(f"Prompt: {prompt}")
    
    # We expect the engine to print "[Main] Trigger detected. Delegating Task: 'Analyze the sentiment...'"
    engine.generate_async(prompt, max_tokens=60)
    
    print("\n>>> Test Complete <<<")

if __name__ == "__main__":
    test_router_logic()
    test_engine_automation()
