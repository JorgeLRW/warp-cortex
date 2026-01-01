# Warp Cortex: The Asynchronous Reasoning Engine

**Warp Cortex** is a "Brain" architecture that enables a single LLM to perform massive parallel reasoning without duplicating memory usage. It transforms a standard LLM into a **Council of Agents**.

## 🧠 The Architecture

Unlike standard "Multi-Agent" frameworks that spawn multiple processes (and blow up VRAM), Warp Cortex uses **Hardware-Level Concurrency** within a single model instance.

### 1. The "River & Stream" Topology
*   **The River (Main Agent)**: The primary generation stream. It maintains the user persona and conversation flow.
*   **The Stream (Side Agents)**: Asynchronous threads of thought that branch off, perform deep reasoning, and merge back.

### 2. The Prism (Weight Sharing)
We use a **Singleton Model Pattern**.
*   **One Model, Many Minds**: We load the model weights *once* into VRAM.
*   **Zero-Copy Execution**: 100 Side Agents can run in parallel using the *exact same* weight pointers.

### 3. The Topological Synapse (O(k) Attention)
Side Agents do not read the Main Agent's full context (which is slow).
*   **Landmarks**: The Main Agent identifies key context states.
*   **Shared Memory**: These landmarks are stored in a shared GPU buffer (The Synapse).
*   **O(k) Speed**: Side Agents attend only to these $k$ landmarks (e.g., 64 tokens), making their execution effectively instant.

### 4. The Validation Gate (Quality Control)
To prevent hallucinations, the Main Agent performs a **Cosine Similarity Check** ($O(1)$) on the Side Agent's thought vector.
*   **Comparison**: `CosineSim(Main_Hidden_State, Side_Thought_Vector)`
*   **Threshold**: If similarity < 0.5, the thought is rejected as irrelevant.
*   **Result**: Only high-quality, contextually relevant thoughts enter the stream.

### 5. Referential Injection (Non-Intrusive Memory)
Instead of pasting text into the output (which disrupts flow), we use **KV Cache Injection**.
*   **Mechanism**: We run a forward pass on `[Ref: <Thought>]` to update the `past_key_values`.
*   **Effect**: The Main Agent "remembers" the thought but continues its own sentence structure naturally.
*   **Zero-Latency**: The user sees no interruption, just a smarter model.

---

## 🚦 The Cortex Router (Dynamic Delegation)

Instead of hardcoded "Roles" (like Critic or Coder), the system uses **Dynamic Task Delegation**.
The Side Agent is a generic "Worker Thread" that receives a specific task description from the Router.

### How it works
1.  **Trigger**: The Main Agent (or User) outputs a trigger like `[TASK: Check the math]` or `[DELEGATE: Write a script]`.
2.  **Extraction**: The Router extracts the task description: `"Check the math"`.
3.  **Delegation**: A Side Agent is spawned with the system prompt: *"You are a sub-process. Task: Check the math."*

### Triggers

| Trigger Type | Example | Action |
| :--- | :--- | :--- |
| **Explicit** | `[TASK: Verify this logic]` | Spawns agent with task "Verify this logic" |
| **Explicit** | `[DELEGATE: Search for X]` | Spawns agent with task "Search for X" |
| **Implicit** | `[SEARCH]`, `check facts` | Spawns agent with task "Perform a search..." |
| **Implicit** | `[CODE]`, `write script` | Spawns agent with task "Write and verify code..." |

**Dynamic Spawning**:
The Router monitors the Main Agent's output stream in real-time. If the Main Agent says "I need to `[SEARCH]`...", a generic worker is spawned immediately to handle that specific request.

---

## 🚀 Scalability (Consumer GPU)

On a single RTX 3090/4090 (24GB VRAM):

| Component | VRAM Usage | Count |
| :--- | :--- | :--- |
| **Main Agent (0.5B FP16)** | 1.2 GB | 1 |
| **Side Agent (Shared)** | **0.0 GB** | **100+** |
| **Side Agent (BitNet)** | 0.2 GB | ~100 |

**Result**: You can run a **Council of 100 Agents** in real-time on a single GPU.

---

## 🛠️ Usage

### 1. Initialize the Engine
```python
from cortex_engine import CortexEngine

# Shared Weights Mode (Maximum Scalability)
engine = CortexEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct", side_mode="shared")
```

### 2. Run Asynchronous Generation
```python
# The Router will automatically detect the intent and spawn the Coder Agent
prompt = "User: Write a python script to calculate the fibonacci sequence."
engine.generate_async(prompt)
```

### 3. Dynamic Triggers
```python
# The Main Agent triggers the Researcher mid-generation
prompt = "User: Who is the CEO of Warp Corp?"
engine.generate_async(prompt)
# Output: "I am not sure, let me [SEARCH]..." -> (Researcher Spawns)
```

---

## 📂 File Structure

*   `cortex_engine.py`: The core engine implementing the River/Stream logic.
*   `cortex_router.py`: The regex-based traffic controller and agent registry.
*   `ARCHITECTURE.md`: Low-level documentation of the memory topology.
*   `scalability_analysis.py`: VRAM math proving the 100-agent capacity.
*   `test_council.py`: Validation script spawning 5 concurrent agents.

