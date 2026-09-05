# Cortex OS: Continuous Semantic & Epistemic Manifold

Warp Cortex is a **continuous topological cognition engine and inference control plane**. 

Instead of treating AI interactions as static, episodic text prompts passed to disconnected models, Cortex models systems as an **elastic geometric manifold** where knowledge, hypotheses, character psychology, and player/user actions interact as continuous dynamical fields.

---

## 1. The Core Paradigm: Semantic Manifolds vs. Flat Context

| Capability | Standard LLM Architecture (RAG / LangChain) | Cortex OS Continuous Manifold |
| :--- | :--- | :--- |
| **Representation** | **Discrete & Flat**: Text strings, JSON payloads, chunked vector database rows. | **Continuous & Topological**: Nodes embedded on Riemannian manifold $\mathcal{S}^{D-1}$ bound by signed constraints. |
| **Interaction** | **Episodic Ping**: Player/user action sends a full prompt to every agent. Compute scales $O(N)$ with agent count. | **Impulse Diffusion**: User action injects a localized energy perturbation $\delta(\mathbf{x})$. Energy ripples via graph Laplacian. Compute is $O(1)$ for dormant agents. |
| **Dependencies** | **Amnesiac & Implicit**: If page 3 contradicts page 78, the model hallucinates or misses the contradiction. | **Causal Tension**: Links (`depends_on`, `supports`, `refutes`, `blocks`) enforce structural consistency. Falsifying a premise automatically cascades down dependent claims. |
| **Keystones** | **Blind**: Cannot identify which claim is the structural linchpin of a research paper or world state. | **Articulation Points**: Tarjan bridge analysis mathematically isolates keystone hypotheses that govern the entire topology. |

---

## 2. The Two Primary Manifold Applications

```text
                     ┌──────────────────────────────────────────────┐
                     │          CORTEX CONTINUOUS MANIFOLD          │
                     └──────────────────────────────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
   THE REACTION MANIFOLD                                          THE EPISTEMIC MANIFOLD
   (Game Worlds & Interactive NPCs)                               (Scientific Research & Discovery)
   - Entities reside at semantic coordinates.                     - Hypotheses, axioms, and assays.
   - Player actions inject continuous energy impulses.            - Signed constraints (depends_on, refutes).
   - Normalized heat diffusion activates nearby agents.           - Empirical results cascade collapses.
   - Zero GPU waste on quiescent NPCs (E < theta).                - Articulation analysis surfaces keystones.
```

### Application A: Dynamic Game Worlds & AI Characters (`reaction_harness.py`)
In games or simulated environments with hundreds of autonomous NPCs:
1. **Semantic Coordinates**: Guard Captain, Tavern Barkeep, and Court Scholar occupy distinct coordinates on the manifold.
2. **Impulse Injection**: When a player draws a sword or smashes a chair, an impulse is injected into the manifold at the "combat / tavern brawl" coordinate.
3. **Heat Diffusion**: Energy diffuses via normalized graph Laplacian:
   $$E_{t+1} = (1 - \gamma) E_t + \alpha \cdot \left( D^{-1/2} A D^{-1/2} \right) E_t$$
4. **Selective Activation**: The Guard and Barkeep cross their activation thresholds ($\theta$) and generate immediate reactions. The Scholar receives 0 energy and remains dormant, wasting **zero GPU compute**.

### Application B: Research Projects & Truth-Seeking (`epistemic_manifold.py`)
In complex research projects with interconnected hypotheses and empirical assays:
1. **Hypothesis Trees**: Hypotheses depend on foundational axioms (`depends_on`) or compete with alternative pathways (`refutes`).
2. **Cascade Collapse**: When an experiment disproves a keystone hypothesis, the negative impulse automatically cascades down the dependency tree, collapsing dependent downstream claims while keeping foundational axioms intact.
3. **Epistemic Strain & Research Frontier**: When empirical data conflicts with theoretical claims, the manifold generates a localized **shear strain tensor**. The system automatically directs autonomous research agents to the highest-strain boundaries to resolve the contradiction.

---

## 3. The Inference Control Plane (`cortex_scorecard`)

Complementing the continuous semantic manifold, `cortex_scorecard` provides empirical verification and cost optimization for production inference:

* **Scorecard Evaluation**: Evaluates local open-source models vs. frontier APIs across structured test traces.
* **Cost & Accuracy Tracking**: Measures exact pass rate, fallback rate, latency, and USD cost per task.
* **Automated Policy Compiler**: Generates reproducible `policy.yaml` routing configurations that save up to 85% of enterprise inference costs with guaranteed verification guardrails.

---

## 4. Summary Architecture

```text
customer traces + manifold topology + candidate models
    -> continuous topological tension & scorecard evaluation
    -> pass/fail, latency, cost, and epistemic strain
    -> persistent SQLite manifold store (shared across agents)
    -> recommended serving & reaction policy
```