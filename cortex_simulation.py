import torch
import threading
import time
import queue

# Mocking the components for the simulation
class MockLLM:
    def __init__(self, name, speed_ms=100):
        self.name = name
        self.speed_ms = speed_ms / 1000.0
        self.vocab = {0: "...", 1: "checking", 2: "the", 3: "data", 4: "[SEARCH]", 5: "found:", 6: "42"}
        
    def generate_step(self, context):
        time.sleep(self.speed_ms)
        # Simple mock logic
        if "meaning" in context and "[SEARCH]" not in context:
            return 4 # [SEARCH]
        return 0 # ...

class SynapseBuffer:
    def __init__(self):
        self.buffer = queue.Queue()
        self.latest_thought = None
        
    def push(self, content):
        self.latest_thought = content
        self.buffer.put(content)
        
    def peek(self):
        return self.latest_thought

class CortexRouter:
    """
    The Brain's Traffic Controller.
    It monitors the Main Agent's output stream and dispatches Side Agents.
    """
    def __init__(self, synapse):
        self.synapse = synapse
        self.side_agent_active = False
        
    def monitor(self, token_id, context):
        """
        Checks if the Main Agent is asking for help.
        """
        if token_id == 4: # [SEARCH] token
            print(f"[{self.__class__.__name__}] Detected [SEARCH] trigger!")
            self.dispatch_side_agent(context)
            return True
        return False
    
    def dispatch_side_agent(self, context):
        """
        Spins up the Side Agent in a background thread.
        """
        self.side_agent_active = True
        thread = threading.Thread(target=self._side_agent_task, args=(context,))
        thread.start()
        
    def _side_agent_task(self, context):
        print(f"   [Side Agent] Received context: '{context}'")
        print(f"   [Side Agent] Thinking/Searching...")
        time.sleep(2.0) # Simulate work (search, code execution)
        result = "The answer is 42"
        print(f"   [Side Agent] Injection result into Synapse: '{result}'")
        self.synapse.push(result)
        self.side_agent_active = False

def run_simulation():
    synapse = SynapseBuffer()
    router = CortexRouter(synapse)
    main_agent = MockLLM("Main", speed_ms=500)
    
    context = "User: What is the meaning of life? Assistant: I am "
    print(f"Start: {context}")
    
    for i in range(10):
        # 1. Main Agent generates a token
        token_id = main_agent.generate_step(context)
        token_str = main_agent.vocab.get(token_id, "?")
        
        print(f"[Main] Generated: {token_str}")
        context += " " + token_str
        
        # 2. Router checks the token
        if router.monitor(token_id, context):
            # If triggered, we might modify behavior, but here we just let it flow
            pass
            
        # 3. Main Agent checks Synapse (The "Gated Attention" part)
        thought = synapse.peek()
        if thought:
            print(f"[Main] !!! ATTENDING TO SYNAPSE !!! Found: '{thought}'")
            # In a real model, this would alter the hidden state immediately
            # Here we just append it to context to show it worked
            context += f" ({thought})"
            # Clear synapse for this demo (or keep reading if stream)
            synapse.latest_thought = None 
            
    print(f"\nFinal Transcript: {context}")

if __name__ == "__main__":
    run_simulation()
