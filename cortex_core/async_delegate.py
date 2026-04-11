"""
Async Delegation: Model-Driven Task Dispatch
=============================================

The main model decides — mid-generation — to spawn a task for a worker.
It keeps generating. The worker runs on a background thread, and when
it finishes, the result is encoded as an embedding and injected into
the SynapseBuffer. The topology gate absorbs it at the next attention
boundary.

This is fundamentally different from the claim-verify pipeline:
  - Claim-verify: SYSTEM extracts claims after generation, dispatches checks.
  - Async delegate: MODEL requests a task during/between generation steps.

Both coexist. The model can also define ad-hoc expert profiles:
  "Have the code expert run this snippet and tell me the output"
  "Have the math expert simplify this expression"

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Main Model generating token-by-token                          │
    │  ...                                                           │
    │  "Let me have a worker execute this code"                      │
    │       │                                                        │
    │       ▼  DelegationRequest detected                            │
    │  ┌──────────────┐                                              │
    │  │ TaskRouter    │── picks or creates an expert profile         │
    │  │ (regex/embed) │                                              │
    │  └──────┬───────┘                                              │
    │         ▼                                                      │
    │  ┌──────────────┐     background thread / CUDA stream          │
    │  │ AsyncWorker   │──────────────────────────────────┐          │
    │  │ (exec/LLM/…) │                                   │          │
    │  └──────────────┘                                   │          │
    │                                                     ▼          │
    │  Main model keeps generating...              ┌───────────┐     │
    │  ...                                         │ Encode     │     │
    │  ...                                         │ result →   │     │
    │  At next attention step:                     │ SynapseBuffer   │
    │  ┌──────────────────────────────────────┐    └───────────┘     │
    │  │ CortexAttention reads buffer          │         │           │
    │  │ topology gate absorbs worker result   │◄────────┘           │
    │  └──────────────────────────────────────┘                      │
    └─────────────────────────────────────────────────────────────────┘

The main model never blocks. If the worker is slow, the model just
doesn't see the result yet — it keeps generating with what it has.
When the result lands in the buffer, the gate decides absorption.
"""

import re
import time
import json
import threading
import subprocess
import textwrap
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future


# ======================================================================
# Expert Profiles — the model can pick from built-ins or define ad-hoc
# ======================================================================

class ExpertKind(Enum):
    CODE_EXEC = "code_exec"       # Execute Python code, return stdout
    MATH_SIMPLIFY = "math_simplify"  # Simplify/evaluate a math expression
    SEARCH = "search"             # (Placeholder) Search for information
    LLM_QUERY = "llm_query"      # Ask a sub-model a focused question
    CUSTOM = "custom"             # Model-defined ad-hoc expert


@dataclass
class ExpertProfile:
    """
    Defines what an expert can do. The main model can pick from
    built-in profiles or dynamically create a CUSTOM one with
    specific instructions.
    """
    kind: ExpertKind
    name: str
    system_prompt: str = ""       # Instructions for LLM-based experts
    timeout: float = 30.0         # Max seconds before giving up
    priority: bool = False        # Use high-priority CUDA stream


# Built-in expert profiles
BUILTIN_EXPERTS: Dict[str, ExpertProfile] = {
    "code": ExpertProfile(
        kind=ExpertKind.CODE_EXEC,
        name="code_executor",
        system_prompt="Execute the given Python code and return the output.",
        timeout=10.0,
    ),
    "math": ExpertProfile(
        kind=ExpertKind.MATH_SIMPLIFY,
        name="math_evaluator",
        system_prompt="Evaluate or simplify the given mathematical expression.",
        timeout=5.0,
    ),
    "search": ExpertProfile(
        kind=ExpertKind.SEARCH,
        name="searcher",
        system_prompt="Search for the requested information.",
        timeout=15.0,
    ),
    "llm": ExpertProfile(
        kind=ExpertKind.LLM_QUERY,
        name="sub_thinker",
        system_prompt="Answer the following question concisely and accurately.",
        timeout=30.0,
    ),
}


# ======================================================================
# Backend Adapters — wrap different model APIs into a uniform interface
# ======================================================================

class BitNetBackend:
    """
    Adapter wrapping warp_bitnet's BitNetGenerator to match the
    backend.generate(messages, temperature, max_tokens) interface.

    Usage:
        from warp_bitnet.research.generate import BitNetGenerator
        gen = BitNetGenerator.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
        backend = BitNetBackend(gen)
    """

    def __init__(self, generator, chat_template: str = "simple"):
        """
        Args:
            generator: BitNetGenerator instance (from warp_bitnet)
            chat_template: How to format chat messages into a plain prompt.
                           "simple" = concatenate with role prefixes.
        """
        self.generator = generator
        self.chat_template = chat_template

    def generate(self, messages, temperature: float = 0.3,
                 max_tokens: int = 200) -> str:
        prompt = self._format_messages(messages)
        return self.generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _format_messages(messages) -> str:
        """Convert chat-style messages to a plain string prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


# ======================================================================
# Delegation Request — what the model asks for
# ======================================================================

@dataclass
class DelegationRequest:
    """A task the main model wants a worker to execute."""
    task_id: str                      # Unique ID for tracking
    expert_kind: str                  # "code", "math", "llm", or custom name
    payload: str                      # The actual task content (code, expression, question)
    instructions: str = ""            # Optional: ad-hoc instructions from the model
    priority: bool = False            # High-priority?
    timestamp: float = field(default_factory=time.time)


@dataclass
class DelegationResult:
    """What the worker returns after completing the task."""
    task_id: str
    expert_kind: str
    payload: str                      # Original task
    output: str                       # Worker's result (stdout, answer, etc.)
    success: bool                     # Did it complete without error?
    elapsed: float = 0.0             # Seconds taken
    error: str = ""                  # Error message if failed
    timestamp: float = field(default_factory=time.time)


# ======================================================================
# Task Detection — scan model output for delegation triggers
# ======================================================================

# Patterns the model can use to request delegation:
#   [DELEGATE:code] print(2**10) [/DELEGATE]
#   [DELEGATE:math] simplify 3x + 2x [/DELEGATE]
#   [DELEGATE:llm] What is the capital of France? [/DELEGATE]
#   [DELEGATE:custom:my_expert] <instructions> | <payload> [/DELEGATE]
_DELEGATE_PATTERN = re.compile(
    r'\[DELEGATE:(\w+)(?::(\w+))?\]\s*(.*?)\s*\[/DELEGATE\]',
    re.DOTALL
)

# Lighter pattern: model says "let me run..." or "execute:" etc.
_IMPLICIT_CODE_PATTERN = re.compile(
    r'```python\n(.*?)```',
    re.DOTALL
)


def detect_delegation_requests(text: str) -> List[DelegationRequest]:
    """
    Scan text for delegation triggers. Returns list of DelegationRequests.
    Supports both explicit [DELEGATE:...] tags and implicit code blocks.
    """
    requests = []
    seen_ids = set()

    # 1. Explicit delegation tags
    for m in _DELEGATE_PATTERN.finditer(text):
        kind = m.group(1).lower()
        custom_name = m.group(2)
        payload = m.group(3).strip()

        task_id = f"d_{len(requests)}_{hash(payload) % 10000:04d}"
        if task_id in seen_ids:
            continue
        seen_ids.add(task_id)

        instructions = ""
        if kind == "custom" and "|" in payload:
            # Custom format: instructions | actual_payload
            parts = payload.split("|", 1)
            instructions = parts[0].strip()
            payload = parts[1].strip()

        requests.append(DelegationRequest(
            task_id=task_id,
            expert_kind=custom_name or kind,
            payload=payload,
            instructions=instructions,
            priority=kind == "code",  # code exec is high-priority by default
        ))

    return requests


# ======================================================================
# Worker Executors — one per expert kind
# ======================================================================

_SAFE_BUILTINS = {"abs": abs, "min": min, "max": max, "round": round,
                  "int": int, "float": float, "len": len, "range": range,
                  "sum": sum, "pow": pow, "sorted": sorted, "enumerate": enumerate,
                  "zip": zip, "map": map, "filter": filter, "list": list,
                  "tuple": tuple, "set": set, "dict": dict, "str": str,
                  "bool": bool, "print": print, "True": True, "False": False,
                  "None": None}


def _execute_code(payload: str, timeout: float = 10.0) -> DelegationResult:
    """Execute Python code in a subprocess sandbox. Returns stdout."""
    import tempfile
    t0 = time.time()
    try:
        # Write to temp file and run in subprocess for isolation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(payload)
            tmp_path = f.name

        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True, text=True,
            timeout=timeout,
        )
        import os
        os.unlink(tmp_path)

        output = result.stdout.strip()
        if result.returncode != 0:
            return DelegationResult(
                task_id="", expert_kind="code",
                payload=payload, output=result.stderr.strip(),
                success=False, elapsed=time.time() - t0,
                error=f"Exit code {result.returncode}",
            )
        return DelegationResult(
            task_id="", expert_kind="code",
            payload=payload, output=output,
            success=True, elapsed=time.time() - t0,
        )
    except subprocess.TimeoutExpired:
        return DelegationResult(
            task_id="", expert_kind="code",
            payload=payload, output="",
            success=False, elapsed=timeout,
            error=f"Timeout after {timeout}s",
        )
    except Exception as e:
        return DelegationResult(
            task_id="", expert_kind="code",
            payload=payload, output="",
            success=False, elapsed=time.time() - t0,
            error=str(e),
        )


_SAFE_MATH_EXPR = re.compile(r'^[\d\s+\-*/().%,eE]+$')


def _evaluate_math(payload: str, timeout: float = 5.0) -> DelegationResult:
    """Evaluate a math expression safely."""
    t0 = time.time()
    expr = payload.strip()
    rhs_expr = None
    if "=" in expr and "==" not in expr:
        lhs_expr, rhs_candidate = expr.split("=", 1)
        lhs_expr = lhs_expr.strip()
        rhs_candidate = rhs_candidate.strip()
        if lhs_expr and _SAFE_MATH_EXPR.match(lhs_expr):
            expr = lhs_expr
            if rhs_candidate and _SAFE_MATH_EXPR.match(rhs_candidate):
                rhs_expr = rhs_candidate

    if not _SAFE_MATH_EXPR.match(expr):
        return DelegationResult(
            task_id="", expert_kind="math",
            payload=payload, output="",
            success=False, elapsed=time.time() - t0,
            error="Unsafe expression",
        )
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if rhs_expr is not None:
            try:
                rhs_value = eval(rhs_expr, {"__builtins__": {}}, {})
                if abs(float(result) - float(rhs_value)) < 1e-9:
                    output = str(rhs_value)
                else:
                    output = str(result)
            except Exception:
                output = str(result)
        else:
            output = str(result)
        return DelegationResult(
            task_id="", expert_kind="math",
            payload=payload, output=output,
            success=True, elapsed=time.time() - t0,
        )
    except Exception as e:
        return DelegationResult(
            task_id="", expert_kind="math",
            payload=payload, output="",
            success=False, elapsed=time.time() - t0,
            error=str(e),
        )


def _query_llm(payload: str, backend, system_prompt: str = "",
               timeout: float = 30.0) -> DelegationResult:
    """Ask a sub-model (via backend) a focused question."""
    t0 = time.time()
    try:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": payload})
        output = backend.generate(msgs, temperature=0.0, max_tokens=200)
        return DelegationResult(
            task_id="", expert_kind="llm",
            payload=payload, output=output.strip(),
            success=True, elapsed=time.time() - t0,
        )
    except Exception as e:
        return DelegationResult(
            task_id="", expert_kind="llm",
            payload=payload, output="",
            success=False, elapsed=time.time() - t0,
            error=str(e),
        )


def _web_search(query: str, max_results: int = 3,
                timeout: float = 10.0) -> DelegationResult:
    """
    Search the web using DuckDuckGo Instant Answer API.
    No API key needed, stdlib only (urllib).

    Returns a DelegationResult with summarized search snippets.
    """
    t0 = time.time()
    try:
        # DuckDuckGo Instant Answer JSON API (public, no key required)
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "WarpCortex/1.0 (search-worker)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        snippets = []

        # Abstract (Wikipedia-style summary)
        if data.get("AbstractText"):
            source = data.get("AbstractSource", "")
            snippets.append(f"[{source}] {data['AbstractText']}")

        # Answer (direct computation / fact)
        if data.get("Answer"):
            snippets.append(f"Answer: {data['Answer']}")

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(snippets) >= max_results:
                break
            if isinstance(topic, dict) and "Text" in topic:
                snippets.append(topic["Text"])
            # Handle sub-groups (Topics with .Topics list)
            elif isinstance(topic, dict) and "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(snippets) >= max_results:
                        break
                    if isinstance(sub, dict) and "Text" in sub:
                        snippets.append(sub["Text"])

        if not snippets:
            return DelegationResult(
                task_id="", expert_kind="search",
                payload=query, output="",
                success=False, elapsed=time.time() - t0,
                error="No results found for query",
            )

        output = "\n".join(f"- {s}" for s in snippets[:max_results])
        return DelegationResult(
            task_id="", expert_kind="search",
            payload=query, output=output,
            success=True, elapsed=time.time() - t0,
        )
    except urllib.error.URLError as e:
        return DelegationResult(
            task_id="", expert_kind="search",
            payload=query, output="",
            success=False, elapsed=time.time() - t0,
            error=f"Network error: {e.reason}",
        )
    except Exception as e:
        return DelegationResult(
            task_id="", expert_kind="search",
            payload=query, output="",
            success=False, elapsed=time.time() - t0,
            error=str(e),
        )


# ======================================================================
# Async Delegation Manager
# ======================================================================

class AsyncDelegationManager:
    """
    Manages the lifecycle of delegated tasks:
    1. Receives DelegationRequests from the main model
    2. Dispatches to appropriate worker (code exec, math eval, LLM query)
    3. When worker finishes, encodes result → SynapseBuffer
    4. Main model absorbs via topology gate at next attention step

    The main model never blocks. All workers run on background threads.
    Results stream in asynchronously.
    """

    def __init__(self, stream_injector=None, backend=None,
                 max_workers: int = 4, device: str = 'cpu',
                 expert_backends: Optional[Dict[str, Any]] = None):
        """
        Args:
            stream_injector: StreamInjector instance (for embedding injection)
            backend: Default LLM backend for LLM workers
            max_workers: Max concurrent background tasks
            device: torch device
            expert_backends: Per-expert backend overrides.
                             Maps expert kind → backend instance.
                             e.g. {"math": BitNetBackend(gen), "llm": hf_backend}
                             Falls back to `backend` if a kind has no override.
        """
        self.stream_injector = stream_injector
        self.backend = backend
        self.expert_backends = expert_backends or {}
        self.device = device
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._active: Dict[str, Future] = {}
        self._results: List[DelegationResult] = []
        self._custom_experts: Dict[str, ExpertProfile] = {}
        self._lock = threading.Lock()
        self._task_counter = 0

    def register_expert(self, name: str, profile: ExpertProfile):
        """
        Register a custom expert. The model can say:
        [DELEGATE:custom:my_expert] instructions | payload [/DELEGATE]
        and this expert will handle it.
        """
        self._custom_experts[name] = profile

    def dispatch(self, request: DelegationRequest) -> str:
        """
        Dispatch a delegation request to a background worker.
        Returns the task_id for tracking. Non-blocking.
        """
        with self._lock:
            self._task_counter += 1
            if not request.task_id:
                request.task_id = f"task_{self._task_counter}"

        future = self._pool.submit(self._execute_and_inject, request)
        with self._lock:
            self._active[request.task_id] = future

        return request.task_id

    def dispatch_batch(self, requests: List[DelegationRequest]) -> List[str]:
        """Dispatch multiple requests. Returns list of task_ids."""
        return [self.dispatch(r) for r in requests]

    def _execute_and_inject(self, request: DelegationRequest) -> DelegationResult:
        """
        Execute the task, encode the result, inject into synapse buffer.
        Runs on a background thread.
        """
        # Resolve expert profile
        kind = request.expert_kind.lower()
        profile = (self._custom_experts.get(kind)
                   or BUILTIN_EXPERTS.get(kind)
                   or BUILTIN_EXPERTS.get("llm"))  # fallback to LLM
        assert profile is not None

        # Execute based on kind
        if profile.kind == ExpertKind.CODE_EXEC:
            result = _execute_code(request.payload, timeout=profile.timeout)
        elif profile.kind == ExpertKind.MATH_SIMPLIFY:
            result = _evaluate_math(request.payload, timeout=profile.timeout)
        elif profile.kind in (ExpertKind.LLM_QUERY, ExpertKind.CUSTOM):
            system = request.instructions or profile.system_prompt
            # Per-expert backend override → default backend → error
            active_backend = self.expert_backends.get(kind, self.backend)
            if active_backend is not None:
                result = _query_llm(request.payload, active_backend,
                                    system_prompt=system, timeout=profile.timeout)
            else:
                result = DelegationResult(
                    task_id=request.task_id, expert_kind=kind,
                    payload=request.payload, output="",
                    success=False, error="No backend available for LLM delegation",
                )
        elif profile.kind == ExpertKind.SEARCH:
            result = _web_search(request.payload, timeout=profile.timeout)
        else:
            result = DelegationResult(
                task_id=request.task_id, expert_kind=kind,
                payload=request.payload, output="",
                success=False, error=f"Unknown expert kind: {kind}",
            )

        result.task_id = request.task_id
        result.expert_kind = kind

        # Inject into synapse buffer (embedding-level) if available
        if self.stream_injector is not None:
            self._inject_result(result)

        # Store result
        with self._lock:
            self._results.append(result)
            self._active.pop(request.task_id, None)

        return result

    def _inject_result(self, result: DelegationResult):
        """Encode a DelegationResult as a VerifiedClaim and inject."""
        from .stream_inject import VerifiedClaim
        assert self.stream_injector is not None

        # Package as a VerifiedClaim (reusing the existing encoder pipeline)
        # expression = the task payload (what was asked)
        # actual = the output (what the worker returned)
        # verified = success flag
        claim = VerifiedClaim(
            expression=textwrap.shorten(result.payload, width=100, placeholder="..."),
            claimed=textwrap.shorten(result.payload, width=50, placeholder="..."),
            actual=textwrap.shorten(result.output, width=200, placeholder="...") if result.output else result.error,
            verified=result.success,
        )
        self.stream_injector.inject_verified_claim(claim)

    def poll_results(self) -> List[DelegationResult]:
        """Return and clear completed results (for display/logging)."""
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def completed_results(self) -> List[DelegationResult]:
        with self._lock:
            return list(self._results)

    def wait_all(self, timeout: float = 60.0):
        """Block until all active tasks complete (for testing/cleanup)."""
        with self._lock:
            futures = list(self._active.values())
        for f in futures:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def shutdown(self):
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=False)


# ======================================================================
# Convenience: scan text + dispatch in one call
# ======================================================================

def scan_and_dispatch(text: str, manager: AsyncDelegationManager) -> List[str]:
    """
    Scan model output for delegation triggers and dispatch them all.
    Returns list of task_ids. Non-blocking.
    """
    requests = detect_delegation_requests(text)
    if not requests:
        return []
    return manager.dispatch_batch(requests)
