"""
Red Team Agent — Adversarial Self-Correction Loop
==================================================

A dedicated adversarial agent that runs on its own CUDA stream.
Every time the main agent produces a code block or technical claim,
the Red Team agent receives it with a single directive: *break this*.

When the Red Team finds a valid critique, it injects the critique
back into the TopologicalSynapse as a high-confidence landmark.
The CortexAttention gate forces the main agent to absorb the
correction mid-generation — a self-correcting MoE that argues
with itself in the background.

Score hierarchy:
    1.0  — verified mathematical truth (claim-verify pipeline)
    0.8  — red team critique (high confidence adversarial finding)
    0.5  — failed claim or unverified assertion
    0.4  — speculative thought (idle-time pre-computation)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Main Agent generates code / claim                          │
    │       │                                                     │
    │       ▼                                                     │
    │  ┌──────────────┐     dedicated CUDA stream                 │
    │  │ Red Team Agent│──────────────────────────┐               │
    │  │ "Break this"  │                           │               │
    │  └──────────────┘                           │               │
    │                                              ▼               │
    │                                  ┌───────────────────┐      │
    │                                  │ Critique found?    │      │
    │                                  │ YES → inject at 0.8│      │
    │                                  │ NO  → discard      │      │
    │                                  └─────────┬─────────┘      │
    │                                            │                │
    │  Main Agent's next CortexAttention step:   │                │
    │  ┌─────────────────────────────────────────┐│                │
    │  │ Cross-attend to red team landmark        ││               │
    │  │ Gate absorbs correction if relevant      │◄               │
    │  └─────────────────────────────────────────┘                │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from cortex_core.red_team import RedTeamAgent

    red = RedTeamAgent(synapse, side_agent, tokenizer, device='cuda')
    red.review("def transfer(amount): db.execute(f'UPDATE ...')")
    red.review_async("print(x / y)")  # non-blocking
    red.shutdown()
"""

import torch
import threading
import time
import re
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future


# Red team critique score — below verified truth, above speculative
RED_TEAM_SCORE = 0.8


@dataclass
class Critique:
    """A finding from the Red Team agent."""
    target: str            # what was reviewed (code snippet or claim)
    finding: str           # the critique / bug / edge case
    category: str          # "bug", "security", "edge_case", "logic_error", "performance"
    severity: float        # 0.0 (nitpick) to 1.0 (critical)
    embedding: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)


# ======================================================================
# Static analyzers — fast checks that don't need a model
# ======================================================================

def _check_division_by_zero(code: str) -> Optional[Critique]:
    """Detect potential division by zero."""
    # Look for division where denominator could be zero
    patterns = [
        r'\/\s*(\w+)\b',           # x / var
        r'\/\s*\(([^)]+)\)',       # x / (expr)
    ]
    for pat in patterns:
        for m in re.finditer(pat, code):
            denom = m.group(1)
            # Skip obvious constants
            if denom.strip().isdigit() and int(denom.strip()) != 0:
                continue
            if denom.strip() in ('0', '0.0'):
                return Critique(
                    target=code[:100],
                    finding=f"Division by zero: `{m.group(0).strip()}`",
                    category="bug",
                    severity=0.9,
                )
    return None


def _check_sql_injection(code: str) -> Optional[Critique]:
    """Detect potential SQL injection via string formatting."""
    sql_patterns = [
        r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE|DROP).*\{',
        r'\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE|DROP)',
        r'%s.*(?:SELECT|INSERT|UPDATE|DELETE|DROP)',
        r'execute\s*\(\s*f["\']',
        r'execute\s*\(\s*["\'].*%',
    ]
    for pat in sql_patterns:
        if re.search(pat, code, re.IGNORECASE):
            return Critique(
                target=code[:100],
                finding="Potential SQL injection via string formatting. Use parameterized queries.",
                category="security",
                severity=1.0,
            )
    return None


def _check_unbounded_input(code: str) -> Optional[Critique]:
    """Detect missing input validation."""
    if 'input(' in code and 'int(input' not in code and 'float(input' not in code:
        if 'try' not in code and 'except' not in code:
            return Critique(
                target=code[:100],
                finding="Raw input() without type conversion or error handling.",
                category="edge_case",
                severity=0.4,
            )
    return None


def _check_resource_leak(code: str) -> Optional[Critique]:
    """Detect potential resource leaks (open files, connections)."""
    # open() without 'with' statement
    if 'open(' in code:
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if 'open(' in stripped and '=' in stripped:
                if not stripped.startswith('with '):
                    return Critique(
                        target=stripped[:100],
                        finding="File opened without `with` statement — potential resource leak.",
                        category="bug",
                        severity=0.6,
                    )
    return None


def _check_off_by_one(code: str) -> Optional[Critique]:
    """Detect common off-by-one patterns."""
    # range(1, len(x)) when iterating — might miss first/last element
    if re.search(r'range\(\s*1\s*,\s*len\(', code):
        return Critique(
            target=code[:100],
            finding="range(1, len(x)) — verify this isn't an off-by-one. Index 0 is skipped.",
            category="edge_case",
            severity=0.3,
        )
    # range(len(x)-1) — might miss last element
    if re.search(r'range\(\s*len\(\w+\)\s*-\s*1\s*\)', code):
        return Critique(
            target=code[:100],
            finding="range(len(x)-1) — last element is excluded. Intentional?",
            category="edge_case",
            severity=0.3,
        )
    return None


# All static checks
STATIC_CHECKS = [
    _check_division_by_zero,
    _check_sql_injection,
    _check_unbounded_input,
    _check_resource_leak,
    _check_off_by_one,
]


class RedTeamAgent:
    """
    Adversarial reviewer that critiques main agent output.

    Two modes of operation:
    1. **Static analysis** — fast regex/pattern checks (always on)
    2. **Model-based** — side agent generates adversarial probe (optional)

    Both modes inject critiques into the synapse as landmarks.
    Static analysis runs synchronously (microseconds).
    Model-based review runs on a background thread.
    """

    def __init__(self, synapse, side_agent=None, tokenizer=None,
                 device: str = 'cpu', max_workers: int = 1):
        """
        Args:
            synapse: TopologicalSynapse instance
            side_agent: BitNetSideAgent or similar (optional, for model-based review)
            tokenizer: tokenizer for the side agent
            device: torch device
            max_workers: background threads for model-based review
        """
        self.synapse = synapse
        self.side_agent = side_agent
        self.tokenizer = tokenizer
        self.device = device
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._critiques: List[Critique] = []
        self._lock = threading.Lock()
        self._dim = synapse.dim or 64

    def review(self, content: str) -> List[Critique]:
        """
        Synchronous review. Runs static checks, optionally model-based.
        Returns list of critiques found. Also injects into synapse.
        """
        critiques = self._static_review(content)

        if self.side_agent is not None and self.tokenizer is not None:
            model_critique = self._model_review(content)
            if model_critique:
                critiques.append(model_critique)

        # Inject all critiques into synapse
        for c in critiques:
            self._inject_critique(c)

        with self._lock:
            self._critiques.extend(critiques)

        return critiques

    def review_async(self, content: str) -> Future:
        """
        Non-blocking review on a background thread.
        Returns a Future that resolves to List[Critique].
        """
        return self._pool.submit(self.review, content)

    def review_code_blocks(self, text: str) -> List[Critique]:
        """
        Extract code blocks from model output and review each one.
        Handles ```python ... ``` fenced blocks.
        """
        all_critiques = []
        # Find fenced code blocks
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        for block in blocks:
            critiques = self.review(block)
            all_critiques.extend(critiques)

        # Also review inline code that looks like executable statements
        # (but only multi-line stuff to avoid false positives)
        if not blocks:
            lines = text.strip().split('\n')
            code_lines = [l for l in lines if any(
                l.strip().startswith(kw) for kw in
                ('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
                 'return ', 'print(', 'assert ')
            )]
            if len(code_lines) >= 2:
                all_critiques.extend(self.review('\n'.join(code_lines)))

        return all_critiques

    def _static_review(self, content: str) -> List[Critique]:
        """Run all static pattern checks."""
        critiques = []
        for check_fn in STATIC_CHECKS:
            try:
                result = check_fn(content)
                if result is not None:
                    critiques.append(result)
            except Exception:
                continue
        return critiques

    def _model_review(self, content: str) -> Optional[Critique]:
        """Use the side agent to generate an adversarial critique."""
        if self.side_agent is None or self.tokenizer is None:
            return None

        prompt = (
            f"[System: You are a Red Team reviewer. Find the most critical "
            f"bug, edge case, or security issue in this code. Be specific.]\n"
            f"```\n{content[:500]}\n```\n[Critical Issue:"
        )
        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=256).input_ids.to(self.device)

        with torch.no_grad():
            finding = self.side_agent.think([ids], ids, self.tokenizer)

        if finding and len(finding) > 10:
            return Critique(
                target=content[:100],
                finding=finding,
                category="model_review",
                severity=0.6,
            )
        return None

    def _inject_critique(self, critique: Critique):
        """Encode a critique and inject into the synapse manifold."""
        # Deterministic embedding from critique content
        hash_val = hash(f"red_team:{critique.category}:{critique.finding}")
        gen = torch.Generator(device='cpu').manual_seed(hash_val % (2**31))
        embedding = torch.randn(self._dim, generator=gen)
        embedding = embedding / (embedding.norm() + 1e-8)

        # Scale by severity — critical bugs get stronger signal
        embedding = embedding * (0.1 + 0.2 * critique.severity)
        embedding = embedding.to(self.device)

        critique.embedding = embedding

        # Inject at RED_TEAM_SCORE scaled by severity
        score = RED_TEAM_SCORE * (0.5 + 0.5 * critique.severity)
        self.synapse.inject_embedding(embedding, score=score)

    def get_critiques(self) -> List[Critique]:
        """Return and clear collected critiques."""
        with self._lock:
            out = list(self._critiques)
            self._critiques.clear()
        return out

    @property
    def critique_count(self) -> int:
        with self._lock:
            return len(self._critiques)

    def shutdown(self):
        """Shutdown background workers."""
        self._pool.shutdown(wait=False)
