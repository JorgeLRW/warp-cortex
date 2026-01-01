# WARP CORTEX: Low-Level Architecture Reference

This document details the internal mechanics of the Warp Cortex engine and how it achieves "Council of Agents" scalability on a single GPU.

## Core Components

1.  **The River** (Main Stream)
2.  **The Stream** (Side Stream)
3.  **The Synapse** (Shared Memory)
4.  **The Prism** (Weight Sharing)

---

## 1. The Prism (Weight Sharing)

Instead of loading $N$ models, we load **ONE** model and project it through different "lenses".
*   **VRAM Usage**: Constant $O(1)$, regardless of agent count.

### The Mechanism
We use a **Singleton Model Pattern**. The physical weights (FP16/INT8) are loaded into GPU Global Memory at a specific address (e.g., `0x7f...`).

*   **Lens A: The Main Agent**
    *   **View**: Full Depth, Full Precision.
    *   **Role**: High-quality generation, persona maintenance.
*   **Lens B: The Early Exit Agent**
    *   **View**: First 50% of layers (e.g., Layers 0-12).
    *   **Mechanism**: Uses the *same* memory pointers but terminates the forward pass early.
    *   **Benefit**: 2x Speed, 0x Extra VRAM.
*   **Lens C: The BitNet Agent**
    *   **View**: If available, a separate 1.58-bit quantized model.
    *   **Cost**: ~0.2GB per instance (if not shared).

---

## 2. The Topological Synapse (Shared Memory)

Standard Multi-Agent systems copy the context window ($N$ tokens) for each agent.
*   **Standard Cost**: $O(N \times Agents)$. This explodes VRAM.

Warp Cortex uses a **Topological Synapse**.
*   **Cortex Cost**: $O(k)$, where $k$ is the number of 'Landmarks'.

### How it Works
1.  **Main Context**: Resides in the Main Stream's KV Cache (e.g., 32k tokens).
2.  **Compression**: The Main Agent identifies high-centrality tokens (**Landmarks**) using attention scores.
3.  **The Synapse**: These landmarks ($k \approx 64$) are stored in a shared buffer.
4.  **Access**: Side Agents read *only* from this Synapse.
    *   **Size**: $64 \times HiddenDim \times Layers \times 2 bytes \approx 10MB$ (Negligible).

---

## 3. River & Stream (Async Execution)

We utilize **CUDA Streams** to achieve hardware-level parallelism. Python threads dispatch kernels; the GPU scheduler executes them concurrently.

### Execution Flow

1.  **Cycle 0 (The River)**:
    *   Main Agent generates: "Let me think..."
    *   **Action**: Pushes Landmarks to Synapse.
    *   *Stream: `cuda.Stream(priority=High)`*

2.  **Cycle 1 (The Streams)**:
    *   Side Agents wake up.
    *   **Action**: Zero-Copy Read from Synapse.
    *   **Compute**: $Thought = Model_{forward}(Synapse)$
    *   *Stream: `cuda.Stream(priority=Medium)`*

3.  **Cycle 2 (Merge)**:
    *   Main Agent absorbs thoughts.
    *   **Mechanism**: Gated Cross-Attention integrates the injected vectors.
    *   *Stream: `cuda.Stream(priority=High)`*

---

## 4. Scalability Math

Why can we fit 100 Agents on a 24GB GPU?

**GPU Capacity**: 24 GB

| Item | Cost | Running Total |
| :--- | :--- | :--- |
| **Main Model (0.5B FP16)** | **-1.2 GB** | 22.8 GB |
| **System Overhead** | **-2.0 GB** | 20.8 GB |
| **Side Agent Weights** | **0.0 GB** | 20.8 GB (Shared) |
| **Side Agent KV Cache** | **~0.001 GB** | 20.8 GB (Shared Synapse) |
| **CUDA Context** | **~0.05 GB** | 20.8 GB |

**Total Cost per Side Agent**: ~0.05 GB

$$ Capacity = \frac{20.8 GB}{0.05 GB} \approx 416 \text{ Agents} $$

*Note: We are Compute Bound, not Memory Bound.*
