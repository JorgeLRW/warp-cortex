# WARP CORTEX: Low-Level Architecture Reference

This document details the internal mechanics of the Warp Cortex engine and how it supports main-model-first orchestration with entropy-triggered, on-demand worker concurrency on a single GPU.

## Core Components

1.  **The River & Stream Topology** (Main/Side Streams)
2.  **The Prism** (Singleton Weight Sharing)
3.  **The Topological Synapse** (Shared Memory)
4.  **Entropy-Guided Delegation** (EntropyRouter + Cortex Router)
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

**Result**: On an RTX 4090 Laptop GPU, you can keep 100+ worker threads available simultaneously.

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

This runtime policy is heuristic, but it is theory-guided rather than arbitrary. The broader attention-geometry work around WarpOS argues that attention is driven by deviations from the key mean and that useful score structure often concentrates in a low-dimensional, query-conditioned subspace. That means the landmark budget should preserve salient, non-redundant deviations and bridge points rather than average background context. The current synapse does not implement the full spectral projector from that research line; it uses top-$k$ salience plus lightweight diversity pressure as a practical approximation.

**Memory Savings**: Reduces per-agent cost from ~1GB (32k context) to ~10MB (64 landmarks).

---

## 3. Entropy-Guided Delegation

Warp Cortex is not designed around a fixed council or always-on worker fan-out. The main model stays on the direct path by default, and the runtime escalates only when there is evidence that more reasoning is warranted.

### Silent Trigger Path

The primary control mechanism is the `EntropyRouter`, which monitors attention disagreement and logit entropy on every decoding step. Let $s_t$ be the head-spread signal and $\ell_t$ the logit-entropy signal at step $t$.

$$z^{spread}_t = \frac{s_t - \mu_s}{\sigma_s}, \qquad z^{logit}_t = \frac{\ell_t - \mu_\ell}{\sigma_\ell}$$

After warmup, the runtime delegates when one of these z-scores rises above its configured threshold. This means the trigger is relative to the model's own recent baseline, not a hardcoded universal entropy cutoff.

An optional learned gate can sit on top of the frozen last-token hidden state:

$$g_t = \sigma(w^\top h_t + b)$$

In the runtime, delegation can require both the entropy trigger and gate approval. Because the gate is trained only on detached hidden states, the main model weights and existing KV caches remain valid.

### Dispatch Flow

1.  **Direct Path First**: The main model attempts to solve the problem itself.
2.  **Entropy Spike**: The router detects internal uncertainty from the current forward pass.
3.  **Expert Selection**: The runtime maps the local hidden state and partial text to a worker kind such as `math`, `code`, `search`, or `llm`.
4.  **Just-in-Time Worker**: Only the needed worker task is launched.
5.  **Result Injection**: The worker result is returned to the next model turn as focused evidence.

### Explicit Compatibility Path

Warp Cortex can also run in an explicit mode where the model emits a structured block such as `[DELEGATE:math] ... [/DELEGATE]`. That path is useful for controlled prompting, but the distinctive routing idea in Warp Cortex is the silent, uncertainty-triggered path.

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

## 5a. Cache Ownership Boundary

The runtime's cache-aware path is explicitly a Python-level Hugging Face
adapter (`cortex_core/cache_control.py`). It can inspect tuple or
`DynamicCache` objects, return landmark-based replacements, and report
content-free cache telemetry. Auto-compaction uses this adapter rather than
assuming one concrete cache representation.

This is not native paged-KV control. Cortex does not currently own vLLM or
SGLang block tables, mutate backend pages in place, perform scheduler-level
preemption, or merge arbitrary KV branches with copy-on-write semantics. A
backend-specific implementation must advertise those capabilities explicitly
before the architecture can claim them.

---

## 6. River & Stream (Async Execution)

We utilize **CUDA Streams** to achieve hardware-level parallelism. Python threads dispatch kernels; the GPU scheduler executes them concurrently.

### Execution Flow

1.  **Cycle 0 (The River)**:
    *   Main Agent begins solving directly.
    *   **Action**: Pushes important landmarks to the synapse.
    *   *Stream: `cuda.Stream(priority=High)`*

2.  **Cycle 1 (Optional Worker Dispatch)**:
    *   The runtime detects an entropy spike or receives an explicit delegation request.
    *   **Action**: A focused worker reads the compact context it needs and executes the requested task.
    *   *Stream: `cuda.Stream(priority=Medium)`*

3.  **Cycle 2 (Resume)**:
    *   The worker result is fed back into the next model turn.
    *   **Action**: The main model resumes from new evidence rather than from a full re-vote or council merge.
    *   *Stream: `cuda.Stream(priority=High)`*

---

## 7. Scalability Math

Why can we fit 100 worker tasks on a laptop RTX 4090?

**Empirical Benchmark Results** (Qwen2.5-0.5B-Instruct):

| Agent Count | Total VRAM | Delta VRAM | VRAM per Agent |
| :--- | :--- | :--- | :--- |
| Baseline (1) | 0.93 GB | --- | --- |
| 10 | 1.05 GB | 0.12 GB | 12 MB |
| 50 | 1.44 GB | 0.52 GB | 10 MB |
| 100 | **2.22 GB** | **1.29 GB** | **13 MB** |

**Total Cost per Worker Slot**: ~13 MB

$$\text{Capacity (16 GB)} = \frac{16 - 0.93}{0.013} \approx 1,159 \text{ Agents}$$

**Practical Limit**: Hundreds of agents before compute latency becomes the bottleneck.

*Note: We are Compute Bound, not Memory Bound.*
