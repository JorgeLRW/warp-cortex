import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

class CortexScaler:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = device
        self.model.eval()
        
        # Create a dummy landmark cache (just random noise for load testing)
        # Shape: [Batch=1, Heads, Seq=50, Dim]
        self.dummy_landmarks = []
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_key_value_heads
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        
        for _ in range(num_layers):
            k = torch.randn(1, num_heads, 50, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(1, num_heads, 50, head_dim, device=device, dtype=torch.float16)
            self.dummy_landmarks.append((k, v))
            
    def run_side_agent(self, agent_id, stop_event, stream):
        """
        Simulates a side agent thinking continuously.
        """
        with torch.cuda.stream(stream):
            # Create local cache wrapper
            past_key_values = DynamicCache()
            past_key_values.key_cache = [k for k, v in self.dummy_landmarks]
            past_key_values.value_cache = [v for k, v in self.dummy_landmarks]
            
            curr_input = torch.randint(0, 1000, (1, 1), device=self.device)
            
            while not stop_event.is_set():
                # Run a forward pass (Thinking)
                seq_len = past_key_values.get_seq_length()
                position_ids = torch.arange(seq_len, seq_len + 1, device=self.device).unsqueeze(0)
                
                _ = self.model(
                    input_ids=curr_input,
                    past_key_values=past_key_values,
                    position_ids=position_ids
                )
                # We don't care about output, just load
                
    def benchmark(self, num_agents):
        print(f"\n--- Benchmarking with {num_agents} Side Agents ---")
        
        stop_event = threading.Event()
        threads = []
        streams = []
        
        # Spin up Side Agents
        for i in range(num_agents):
            stream = torch.cuda.Stream()
            streams.append(stream)
            t = threading.Thread(target=self.run_side_agent, args=(i, stop_event, stream))
            t.start()
            threads.append(t)
            
        # Run Main Agent (Measurement)
        # Main agent runs on default stream
        start_time = time.time()
        num_tokens = 50
        
        curr_input = torch.randint(0, 1000, (1, 1), device=self.device)
        past_key_values = None # Start fresh
        
        for _ in range(num_tokens):
            outputs = self.model(curr_input, past_key_values=past_key_values)
            curr_input = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            past_key_values = outputs.past_key_values
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Cleanup
        stop_event.set()
        for t in threads:
            t.join()
            
        duration = end_time - start_time
        tps = num_tokens / duration
        print(f"Main Agent Speed: {tps:.2f} tokens/sec")
        return tps

if __name__ == "__main__":
    scaler = CortexScaler()
    
    baseline = scaler.benchmark(0)
    scaler.benchmark(1)
    scaler.benchmark(5)
    scaler.benchmark(10)
    scaler.benchmark(20)
