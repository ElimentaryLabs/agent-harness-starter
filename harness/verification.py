"""Verification — guaranteed, not pluggable (§6).

Verification is NOT a hook. Hooks are optional and fail-open; verification is a
guaranteed step the loop always runs after tool execution. It replaces the
model's *belief* that a tool worked with a deterministic *check* — a
postcondition / quality gate.

A rule returns None if the result is acceptable, or a string reason if it
isn't; the loop marks the ToolResult verified/failed accordingly so the model
sees the verdict next round.

Note: this verdict is *advisory*, not a hard block. A failed postcondition is
prefixed onto the model-visible content but the result still flows back as
`is_error=False` — the model decides what to do with it. A stricter harness
would fail the round outright; here the check is deterministic and the model
stays in charge of the response.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from .types import ToolResult

logger = logging.getLogger("harness.verification")

Rule = Callable[[ToolResult], Optional[str]]


def non_empty_success(tr: ToolResult) -> Optional[str]:
    """A successful tool must actually return content. Catches the classic
    'model assumes the call worked' failure mode."""
    if tr.success and not (tr.content or "").strip():
        return "tool reported success but returned no content"
    return None


class VerificationPipeline:
    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self.rules = rules if rules is not None else [non_empty_success]

    def verify(self, results: List[ToolResult]) -> None:
        for tr in results:
            reason = None
            for rule in self.rules:
                reason = rule(tr)
                if reason:
                    break
            if reason:
                tr.verified = False
                tr.verification_reason = reason
                logger.warning("verification failed for %s: %s", tr.tool_name, reason)
            else:
                tr.verified = True
