# WARP CORTEX: Low-Level Architecture Reference

This document details the internal mechanics of the Warp Cortex engine and how it supports main-model-first orchestration with on-demand worker concurrency on a single GPU.

## Core Components

1.  **The River & Stream Topology** (Main/Side Streams)
2.  **The Prism** (Singleton Weight Sharing)
3.  **The Topological Synapse** (Shared Memory)
4.  **The Cortex Router** (Dynamic Delegation)
5.  **The Validation Gate** (Quality Control)
6.  **Referential Injection** (Non-Intrusive Memory Update)

---

## 1. The Prism (Singleton Weight Sharing)

Instead of loading $N$ models, we load **ONE** model and share it across all worker threads using pointer-level view slicing.

### Memory Complexity
$$M_{total} = \text{Mem}(W) + \sum_{i=1}^{N} (K_i + V_i) \approx \text{Mem}(W) + N \cdot \text{Mem}(\text{Synapse})$$

Where $\text{Mem}(\text{Synapse}) \ll \text{Mem}(H)$, effectively reducing the memory growth from $O(N \cdot L)$ to $O(N \cdot k)$ where $k$ is the number of landmark tokens.

### The Mechanism
We use a **Singleton Model Pattern**. The physical weights (FP16/INT8) are loaded into GPU Global Memory once.

*   **Main Agent**: Full depth, full precision. High-quality generation and persona maintenance.
*   **Worker Threads**: Share the exact same weight pointers. Zero additional weight memory.
*   **VRAM Usage**: Constant $O(1)$ for weights, regardless of agent count.

**Result**: On a 24GB GPU, you can keep 100+ worker threads available simultaneously.

---

## 2. The Topological Synapse (Shared Memory)

Standard Multi-Agent systems copy the full context window ($L$ tokens) for each agent.
*   **Standard Cost**: $O(N \times L)$. This explodes VRAM.

Warp Cortex uses a **Topological Synapse** to compress context to high-centrality "Landmarks".
*   **Cortex Cost**: $O(N \times k)$, where $k \ll L$.

### Landmark Selection Policy

Given the Main Agent's query state $Q_t$ at timestep $t$:

1.  **Attention Score Summation**: Compute $A_i = \sum_{h=1}^{H} \text{softmax}(Q_t K_i^T / \sqrt{d_k})$ $\forall h \in \{1, \dots, H\}$, where $d_k$ is the dimension of key vectors.
2.  **Top-$k$ Selection**: Select the top $k$ tokens (e.g., $k=64$) with highest $A_i$ values.
3.  **Storage**: Store these $k$ tokens in a shared ring buffer accessible to all Side Agents.
4.  **Access**: Side Agents attend *only* to the Synapse, not the full history.

**Memory Savings**: Reduces per-agent cost from ~1GB (32k context) to ~10MB (64 landmarks).

---

## 3. The Cortex Router (Explicit Delegation)

Warp Cortex is no longer designed around a fixed council or always-on worker fan-out. The main model remains in control and explicitly emits delegation blocks only when a narrow subtask should be offloaded.

### How It Works

1.  **Direct Path First**: The main model attempts to solve the problem itself.
2.  **Explicit Trigger**: If it wants help, it emits a structured block such as `[DELEGATE:math] ... [/DELEGATE]`.
3.  **Just-in-Time Worker**: The runtime dispatches only that requested worker task.
4.  **Result Injection**: The worker result is returned to the next model turn as focused evidence.

**Example**:
- Main-model output: `"I should check the arithmetic. [DELEGATE:math] 17 * 23 [/DELEGATE]"`
- Runtime action: Execute the math worker and feed `391` back into the next turn.

---

## 4. The Validation Gate (Quality Control)

To prevent "hallucination cascades" where poor reasoning infects the main stream, we implement a **geometric quality control check**.

### Cosine Similarity Validation

Let $h_t^{(L)}$ represent the latent representation of the $t$-th token at the final layer $L$.

Before a Side Agent's thought $T_{side}$ is merged, we extract its last-token hidden state and calculate:

$$\text{Score} = \frac{h_{main}^{(L)} \cdot T_{side}}{\|h_{main}^{(L)}\| \|T_{side}\|}$$

If $\text{Score} < \theta$ (hyperparameter, typically 0.5), the thought is **rejected**.

**Result**: Only contextually relevant thoughts enter the stream, filtering out low-quality or off-topic contributions.

---

## 5. Referential Injection (Non-Intrusive Memory Update)

Traditional injection involves pasting text into the context, which disrupts the Main Agent's generation flow.

### KV Cache Injection

**Mechanism**:
1.  The engine runs a forward pass on the thought vector $T_{side}$ marked as a "Reference".
2.  The resulting keys and values are appended to the Main Agent's `past_key_values`.
3.  **Positional Integrity**: We utilize Rotary Position Embeddings (RoPE), assigning injected thoughts a virtual positional index that marks them as auxiliary context.
4.  **Result**: The Main Agent "remembers" the thought but continues generating its original sentence structure seamlessly.

**Benefit**: Zero disruption to the output stream. The user sees clean, coherent text while the model benefits from asynchronous reasoning.

---

## 6. River & Stream (Async Execution)

We utilize **CUDA Streams** to achieve hardware-level parallelism. Python threads dispatch kernels; the GPU scheduler executes them concurrently.

### Execution Flow

1.  **Cycle 0 (The River)**:
    *   Main Agent begins solving directly.
    *   **Action**: Pushes important landmarks to the synapse.
    *   *Stream: `cuda.Stream(priority=High)`*

2.  **Cycle 1 (Optional Worker Dispatch)**:
    *   The main model emits an explicit delegation block only if a narrow subtask should be offloaded.
    *   **Action**: A focused worker reads the compact context it needs and executes the requested task.
    *   *Stream: `cuda.Stream(priority=Medium)`*

3.  **Cycle 2 (Resume)**:
    *   The worker result is fed back into the next model turn.
    *   **Action**: The main model resumes from new evidence rather than from a full re-vote or council merge.
    *   *Stream: `cuda.Stream(priority=High)`*

---

## 7. Scalability Math

Why can we fit 100 worker tasks on a 24GB GPU?

**Empirical Benchmark Results** (Qwen2.5-0.5B-Instruct):

| Agent Count | Total VRAM | Delta VRAM | VRAM per Agent |
| :--- | :--- | :--- | :--- |
| Baseline (1) | 0.93 GB | --- | --- |
| 10 | 1.05 GB | 0.12 GB | 12 MB |
| 50 | 1.44 GB | 0.52 GB | 10 MB |
| 100 | **2.22 GB** | **1.29 GB** | **13 MB** |

**Total Cost per Worker Slot**: ~13 MB

$$\text{Capacity (24 GB)} = \frac{24 - 0.93}{0.013} \approx 1,775 \text{ Agents}$$

**Practical Limit**: ~1,000 agents before compute latency becomes the bottleneck.

*Note: We are Compute Bound, not Memory Bound.*
