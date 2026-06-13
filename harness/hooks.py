"""Hooks — the extensibility seams (§5).

The harness owns *when* hooks fire; agents register *what* runs. Every hook is
wrapped in try/except so a failing hook can never crash the run: hooks are
**fail-open** by design. That is exactly why correctness (verification, §6)
is NOT a hook — see verification.py.

Observability is the canonical hook: a fail-open lifecycle hook set you make
"mandatory" by registering it by default, not by welding it into the loop.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from .types import GATING_HOOKS, HookContext, HookPoint

logger = logging.getLogger("harness.hooks")

Hook = Callable[[HookContext], Optional[dict]]


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[HookPoint, List[Hook]] = {}

    def register(self, point: HookPoint, hook: Hook) -> None:
        self._hooks.setdefault(point, []).append(hook)

    def run(self, point: HookPoint, hctx: HookContext) -> Optional[dict]:
        """Run every hook at `point`. For gating points the first non-None
        return wins; later hooks still run (for observability) but cannot
        override the decision."""
        gate: Optional[dict] = None
        for hook in self._hooks.get(point, []):
            try:
                result = hook(hctx)
                if point in GATING_HOOKS and result and gate is None:
                    gate = result
            except Exception as e:  # fail-open: telemetry/extension bugs never break the run
                logger.warning("hook at %s raised: %s", point.value, e)
        return gate if point in GATING_HOOKS else None


# --- Built-in observability hook set ---------------------------------------

def observability_hooks() -> Dict[HookPoint, List[Hook]]:
    """A default-on lifecycle hook set that logs the run. This is how you make
    observability 'mandatory' — register it everywhere — without making it a
    non-skippable control-flow step."""

    def on_pre_round(h: HookContext):
        logger.info("[round %d] start", h.round_number + 1)

    def on_post_llm(h: HookContext):
        resp = h.llm_response or {}
        calls = resp.get("tool_calls", [])
        if calls:
            logger.info("[round %d] model wants tools: %s",
                        h.round_number + 1, [c.name for c in calls])
        else:
            logger.info("[round %d] model produced a final answer", h.round_number + 1)

    def on_post_tool(h: HookContext):
        tr = h.tool_result
        if tr:
            status = "ok" if tr.success else "FAIL"
            logger.info("    tool %s -> %s (%d chars)",
                        tr.tool_name, status, len(tr.content or ""))

    def on_error(h: HookContext):
        logger.error("[round %d] loop error: %s", h.round_number + 1, h.error)

    return {
        HookPoint.PRE_ROUND: [on_pre_round],
        HookPoint.POST_LLM_CALL: [on_post_llm],
        HookPoint.POST_TOOL_EXEC: [on_post_tool],
        HookPoint.ON_ERROR: [on_error],
    }


# --- Permission gate (§7) --------------------------------------------------

def read_only_permission_gate(h: HookContext) -> Optional[dict]:
    """A PRE_TOOL_EXEC gate answering 'am I allowed to run *this* call?'.

    The starter is read-only, so this allows everything. The seam is the point:
    return ``{"block": True, "reason": "..."}`` to deny a specific call — e.g.
    require human confirmation before a mutating tool. Capability ('may I?',
    answered by which tools are registered) and authorization ('am I allowed?',
    answered here) are different layers.
    """
    # Example of how you'd gate a mutating tool:
    # if h.tool_call and h.tool_call.name == "delete_everything":
    #     return {"block": True, "reason": "destructive; needs confirmation"}
    return None
