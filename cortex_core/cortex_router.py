import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ======================================================================
# Intent definitions — semantic classes the router recognises.
# Each intent has an id, a human-readable label, a description template
# returned as the task string, and a set of *seed sentences* used to
# bootstrap the lightweight classifier.
# ======================================================================

_INTENTS = [
    {
        "id": "search",
        "label": "Search / Verify",
        "task": "Perform a search to verify this information.",
        "seeds": [
            "We should look this up.",
            "Can you verify the claim?",
            "Search online for a source.",
            "Double-check this fact.",
            "I want to confirm this is true.",
        ],
    },
    {
        "id": "code",
        "label": "Write Code",
        "task": "Write and verify code for this problem.",
        "seeds": [
            "Write a Python script to solve this.",
            "Implement this as a function.",
            "Can you code this up?",
            "Write the algorithm in code.",
            "Create a program that does this.",
        ],
    },
    {
        "id": "check",
        "label": "Fact-Check / Verify",
        "task": "Double check the logic and facts.",
        "seeds": [
            "Check the math here.",
            "Verify the logical steps.",
            "Is this reasoning correct?",
            "Re-examine the argument.",
            "Audit this proof for errors.",
        ],
    },
    {
        "id": "summarise",
        "label": "Summarise",
        "task": "Summarise the preceding information.",
        "seeds": [
            "Can you summarise this?",
            "Give me a TL;DR.",
            "Provide a brief overview.",
            "Condense the key points.",
            "What's the short version?",
        ],
    },
    {
        "id": "delegate",
        "label": "General Delegation",
        "task": None,  # task text comes from the text itself
        "seeds": [
            "I need you to handle this sub-task.",
            "Delegate this to a side agent.",
            "Offload this analysis to another worker.",
            "Have an assistant process this part.",
            "Spin up a worker to do this.",
        ],
    },
]


class _IntentClassifierHead(nn.Module):
    """
    Lightweight MLP that maps a single hidden-state vector to intent logits.
    Designed to run on the *same* hidden states already computed by the
    backbone model — zero extra forward passes.
    """
    def __init__(self, input_dim: int, num_intents: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_intents),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., input_dim]  →  [..., num_intents]"""
        return self.proj(h)


class CortexRouter:
    """
    Semantic + Regex Hybrid Router.

    Primary path — **Semantic**:
        A small MLP classifier head on top of the backbone model's hidden
        states.  The head is bootstrapped from seed sentences using the
        backbone's own encoder (no external embedding model needed).

    Fallback path — **Regex** (kept for explicit tags):
        [TASK: ...] and [DELEGATE: ...] are still honoured directly.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.intents = _INTENTS
        self.num_intents = len(_INTENTS)

        # Explicit-tag regex (always checked first — near-zero cost)
        self._explicit_patterns = [
            r"\[TASK:\s*(.*?)\]",
            r"\[DELEGATE:\s*(.*?)\]",
        ]

        # ---------- classifier head ----------
        # Initialised lazily in `bootstrap()` once we know the hidden dim.
        self._head: Optional[_IntentClassifierHead] = None
        self._bootstrapped = False
        self._hidden_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Bootstrap: teach the classifier from seed sentences
    # ------------------------------------------------------------------

    @torch.no_grad()
    def bootstrap(self, model, tokenizer, device='cuda', epochs: int = 80, lr: float = 3e-3):
        """
        One-time setup: encode every seed sentence through the backbone,
        then train the MLP head to map those hidden states to their intent
        class.  Runs in < 1 second on GPU.
        """
        if self._bootstrapped:
            return

        # 1. Collect (hidden_state, label) pairs from seed sentences
        embeddings: List[torch.Tensor] = []
        labels: List[int] = []

        for intent_idx, intent in enumerate(self.intents):
            for seed in intent["seeds"]:
                ids = tokenizer(seed, return_tensors="pt").input_ids.to(device)
                out = model(ids, output_hidden_states=True)
                # Mean-pool last hidden layer over sequence
                h = out.hidden_states[-1].mean(dim=1).squeeze(0)  # [D]
                embeddings.append(h)
                labels.append(intent_idx)

        X = torch.stack(embeddings).float()               # [N, D] — ensure float32
        y = torch.tensor(labels, device=device)            # [N]
        self._hidden_dim = X.shape[-1]

        # 2. Train a tiny MLP on these pairs (need grad enabled)
        head = _IntentClassifierHead(self._hidden_dim, self.num_intents).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=lr)

        head.train()
        with torch.enable_grad():
            for _ in range(epochs):
                logits = head(X)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        # Quick sanity: training accuracy
        with torch.no_grad():
            acc = (head(X).argmax(-1) == y).float().mean().item()
        print(f"[Router] Bootstrapped semantic classifier "
              f"({self._hidden_dim}->{self.num_intents}) acc={acc:.0%}")

        self._head = head
        self._bootstrapped = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify_hidden(self, hidden_state: torch.Tensor) -> Tuple[Optional[str], float]:
        """
        Classify a hidden-state vector into an intent.

        Args:
            hidden_state: [D] or [1, D] — last hidden state from backbone.

        Returns:
            (task_description | None, confidence)
        """
        if self._head is None:
            return None, 0.0

        h = hidden_state.detach().float()
        if h.dim() == 2:
            h = h.squeeze(0)   # [D]

        logits = self._head(h.unsqueeze(0))  # [1, num_intents]
        probs = F.softmax(logits, dim=-1).squeeze(0)
        conf, idx = probs.max(dim=0)

        if conf.item() < self.confidence_threshold:
            return None, conf.item()

        intent = self.intents[int(idx.item())]
        return intent["task"], conf.item()

    def check_for_triggers(self, text_stream: str,
                           hidden_state: Optional[torch.Tensor] = None) -> Optional[str]:
        """
        Hybrid trigger detection.

        1. Explicit regex tags ([TASK: ...], [DELEGATE: ...]) — always honoured.
        2. Semantic classification on model hidden states (if available).
        3. Legacy regex fallback for backward compat (if no hidden state).

        Returns: task_description (str) or None
        """
        # ---- 1. Explicit tags (still useful for programmatic control) ----
        for pattern in self._explicit_patterns:
            match = re.search(pattern, text_stream, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # ---- 2. Semantic path ----
        if hidden_state is not None and self._head is not None:
            task, conf = self.classify_hidden(hidden_state)
            if task is not None:
                return task

        # ---- 3. Legacy regex fallback (no hidden state available) ----
        _legacy_triggers = {
            r"\[SEARCH\]": "Perform a search to verify this information.",
            r"\[CODE\]": "Write and verify code for this problem.",
            r"\[CHECK\]": "Double check the logic and facts.",
            r"write.*script": "Write a script to solve this.",
            r"check.*facts": "Verify these facts.",
        }
        for pattern, task_desc in _legacy_triggers.items():
            if re.search(pattern, text_stream, re.IGNORECASE):
                return task_desc

        return None

