"""Tool registry + executor (§4).

Holds the agent's capability surface and runs tool calls. Progressive
disclosure would live here too (load tools on demand); this starter keeps the
full set registered up front for clarity, but the seam is the registry.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from .types import ExecutionContext, ToolCall, ToolDef, ToolResult

logger = logging.getLogger("harness.registry")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> List[dict]:
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def execute(self, call: ToolCall, ctx: ExecutionContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.id, call.name, success=False,
                              error=f"unknown tool '{call.name}'")
        try:
            out = tool.handler(call.arguments, ctx)
            # Handlers return {"content": str, "success": bool?}.
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                success=bool(out.get("success", True)),
                content=str(out.get("content", "")),
                error=out.get("error"),
            )
        except Exception as e:  # a tool crash is a failed result, not a dead loop
            logger.exception("tool %s raised", call.name)
            return ToolResult(call.id, call.name, success=False, error=str(e))
