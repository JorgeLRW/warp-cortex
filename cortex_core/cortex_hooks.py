"""
Cortex Hooks: Execution Interception & Verification Loops
==========================================================
Inspired by Claude Code's PreToolUse / PostToolUse hooks and the
"check-then-fix" verification pattern.

Three subsystems:
1. Hook Registry — attach pre/post hooks to any engine action
   (generation, injection, tool use, compaction).
2. Security Gate — intercepts agent actions before execution;
   enforces allow/deny rules per agent role.
3. Verification Loop — automatic retry-on-error pattern:
   run → check → fix → re-run until success or max retries.
"""

import time
import traceback
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any


# ======================================================================
# 1. Hook System
# ======================================================================

class HookPoint(Enum):
    """Points in the engine lifecycle where hooks can fire."""
    PRE_GENERATION    = "pre_generation"     # Before each token generation step
    POST_GENERATION   = "post_generation"    # After each token generation step
    PRE_INJECTION     = "pre_injection"      # Before a thought is injected into KV cache
    POST_INJECTION    = "post_injection"     # After a thought is injected
    PRE_TOOL_USE      = "pre_tool_use"       # Before a sub-agent executes a tool/command
    POST_TOOL_USE     = "post_tool_use"      # After a sub-agent finishes a tool/command
    PRE_COMPACTION    = "pre_compaction"      # Before context compaction
    POST_COMPACTION   = "post_compaction"     # After context compaction
    PRE_DISPATCH      = "pre_dispatch"        # Before an agent is dispatched
    POST_DISPATCH     = "post_dispatch"       # After an agent completes


@dataclass
class HookContext:
    """
    Context passed to every hook function.
    Hooks can read/modify `data` and set `abort=True` to cancel the action.
    """
    hook_point: HookPoint
    data: Dict[str, Any] = field(default_factory=dict)
    abort: bool = False
    abort_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """
    Central registry for lifecycle hooks.

    Usage:
        hooks = HookRegistry()
        hooks.register(HookPoint.PRE_INJECTION, my_security_check)

        ctx = hooks.fire(HookPoint.PRE_INJECTION, {"thought": "...", "source": "researcher"})
        if ctx.abort:
            print(f"Blocked: {ctx.abort_reason}")
    """

    def __init__(self):
        self._hooks: Dict[HookPoint, List[Callable[[HookContext], None]]] = {
            hp: [] for hp in HookPoint
        }

    def register(self, hook_point: HookPoint, fn: Callable[[HookContext], None]):
        """Attach a hook function to a lifecycle point."""
        self._hooks[hook_point].append(fn)

    def unregister(self, hook_point: HookPoint, fn: Callable):
        """Remove a previously registered hook."""
        self._hooks[hook_point] = [h for h in self._hooks[hook_point] if h is not fn]

    def fire(self, hook_point: HookPoint, data: Optional[Dict[str, Any]] = None) -> HookContext:
        """
        Fire all hooks for a given point. Returns the (possibly modified) context.
        If any hook sets ctx.abort = True, subsequent hooks still run but the
        caller should check ctx.abort and skip the action.
        """
        ctx = HookContext(hook_point=hook_point, data=data or {})
        for fn in self._hooks[hook_point]:
            try:
                fn(ctx)
            except Exception as e:
                print(f"[Hooks] Error in {hook_point.value} hook: {e}")
                traceback.print_exc()
        return ctx

    def list_hooks(self) -> Dict[str, int]:
        return {hp.value: len(fns) for hp, fns in self._hooks.items() if fns}


# ======================================================================
# 2. Built-in Security Hooks
# ======================================================================

# Patterns that should be blocked from agent execution
_DANGEROUS_PATTERNS = [
    "rm -rf",
    "rmdir /s",
    "format c:",
    "drop table",
    "drop database",
    "; rm ",
    "| rm ",
    "sudo rm",
    "del /f /s /q",
    "shutdown",
    "mkfs",
]


def security_pre_tool_hook(ctx: HookContext):
    """
    PreToolUse hook: blocks dangerous commands before execution.
    Expects ctx.data to contain 'command' or 'action' key.
    """
    command = ctx.data.get("command", "") or ctx.data.get("action", "")
    command_lower = command.lower()

    for pattern in _DANGEROUS_PATTERNS:
        if pattern in command_lower:
            ctx.abort = True
            ctx.abort_reason = f"Security: blocked dangerous pattern '{pattern}' in command"
            print(f"[Security] BLOCKED: {ctx.abort_reason}")
            return

    # Check for command injection via shell metacharacters in user-influenced data
    user_input = ctx.data.get("user_input", "")
    if user_input and any(c in user_input for c in (";", "|", "`", "$(")):
        ctx.abort = True
        ctx.abort_reason = "Security: potential command injection in user input"
        print(f"[Security] BLOCKED: {ctx.abort_reason}")


def injection_quality_hook(ctx: HookContext):
    """
    PreInjection hook: validates thought quality before KV cache injection.
    Expects ctx.data to contain 'thought_text' and optionally 'similarity_score'.
    """
    thought = ctx.data.get("thought_text", "")
    sim_score = ctx.data.get("similarity_score")

    # Reject empty thoughts
    if not thought or len(thought.strip()) < 3:
        ctx.abort = True
        ctx.abort_reason = "Injection blocked: empty or trivial thought"
        return

    # Reject if similarity score is below threshold
    if sim_score is not None and sim_score < 0.3:
        ctx.abort = True
        ctx.abort_reason = f"Injection blocked: similarity {sim_score:.2f} below 0.3 threshold"
        return

    # Reject excessively long thoughts (prevents KV cache bloat)
    if len(thought) > 2000:
        ctx.data["thought_text"] = thought[:2000]
        print("[Hooks] Truncated oversized thought to 2000 chars")


def dispatch_rate_limit_hook(ctx: HookContext, _state={"last_dispatch": 0, "count": 0}):
    """
    PreDispatch hook: rate-limits agent spawning to prevent runaway cascades.
    Max 10 dispatches per second.
    """
    now = time.time()
    if now - _state["last_dispatch"] < 1.0:
        _state["count"] += 1
        if _state["count"] > 10:
            ctx.abort = True
            ctx.abort_reason = "Rate limit: too many agent dispatches per second"
            print(f"[Hooks] THROTTLED: {ctx.abort_reason}")
            return
    else:
        _state["count"] = 0
        _state["last_dispatch"] = now


# ======================================================================
# 3. Verification Loop
# ======================================================================

@dataclass
class VerificationResult:
    """Outcome of a verification loop iteration."""
    iteration: int
    passed: bool
    output: Any = None
    error: Optional[str] = None


class VerificationLoop:
    """
    Implements the "check-then-fix" pattern from Claude Code:
    1. Run the action
    2. Check the result for errors
    3. If errors, feed them back and retry
    4. Repeat until success or max retries

    Usage:
        loop = VerificationLoop(
            action_fn=lambda ctx: agent.generate(ctx),
            check_fn=lambda result: ("error" not in result, result),
            fix_fn=lambda result, error: f"Fix this error: {error}. Previous: {result}",
            max_retries=3,
        )
        final = loop.run(initial_context="Write a Python function...")
    """

    def __init__(
        self,
        action_fn: Callable[[str], Any],
        check_fn: Callable[[Any], tuple],  # Returns (passed: bool, detail: Any)
        fix_fn: Optional[Callable[[Any, str], str]] = None,
        max_retries: int = 3,
        hooks: Optional[HookRegistry] = None,
    ):
        self.action_fn = action_fn
        self.check_fn = check_fn
        self.fix_fn = fix_fn or self._default_fix
        self.max_retries = max_retries
        self.hooks = hooks
        self.history: List[VerificationResult] = []

    def run(self, initial_context: str) -> VerificationResult:
        """Execute the verify loop. Returns the final result."""
        context = initial_context

        for i in range(self.max_retries + 1):
            # 1. Run action
            print(f"[Verify] Attempt {i + 1}/{self.max_retries + 1}")
            try:
                output = self.action_fn(context)
            except Exception as e:
                result = VerificationResult(iteration=i, passed=False, error=str(e))
                self.history.append(result)
                context = self.fix_fn(context, str(e))
                continue

            # 2. Check
            passed, detail = self.check_fn(output)
            result = VerificationResult(
                iteration=i, passed=passed, output=output,
                error=None if passed else str(detail)
            )
            self.history.append(result)

            if passed:
                print(f"[Verify] PASSED on attempt {i + 1}")
                return result

            # 3. Fix context for retry
            error_msg = str(detail)
            print(f"[Verify] FAILED: {error_msg[:100]}. Retrying...")
            context = self.fix_fn(output, error_msg)

        # Exhausted retries
        print(f"[Verify] EXHAUSTED {self.max_retries + 1} attempts")
        return self.history[-1]

    @staticmethod
    def _default_fix(previous_output: Any, error: str) -> str:
        """Default fix strategy: append error feedback to context."""
        return (
            f"Previous attempt produced an error: {error}\n"
            f"Previous output: {str(previous_output)[:500]}\n"
            f"Please fix the error and try again."
        )


# ======================================================================
# 4. Convenience: wire up default hooks
# ======================================================================

def create_default_hooks() -> HookRegistry:
    """
    Create a HookRegistry pre-loaded with security, quality, and rate-limit hooks.
    """
    registry = HookRegistry()
    registry.register(HookPoint.PRE_TOOL_USE, security_pre_tool_hook)
    registry.register(HookPoint.PRE_INJECTION, injection_quality_hook)
    registry.register(HookPoint.PRE_DISPATCH, dispatch_rate_limit_hook)
    return registry
