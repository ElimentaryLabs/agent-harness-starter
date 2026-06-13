"""The execution loop — the spine (§1).

One pass = one round. The model reasons; the harness manages turns, tools,
verification, context, hooks, and termination. The step numbers below match the
13-step diagram in the article. Notice how small the LLM call is relative to
everything around it: you program the loop, not the model.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .context import ContextManager
from .hooks import HookManager
from .registry import ToolRegistry
from .types import (
    ExecutionContext,
    HookContext,
    HookPoint,
    LoopResult,
    StopReason,
    ToolResult,
)
from .verification import VerificationPipeline

logger = logging.getLogger("harness.loop")


class AgentLoop:
    def __init__(
        self,
        llm,
        registry: ToolRegistry,
        context_manager: Optional[ContextManager] = None,
        verification: Optional[VerificationPipeline] = None,
        hooks: Optional[HookManager] = None,
        max_rounds: int = 12,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.cm = context_manager or ContextManager()
        self.verification = verification or VerificationPipeline()
        self.hooks = hooks or HookManager()
        self.max_rounds = max_rounds

    def run(self, system: str, ctx: ExecutionContext) -> LoopResult:
        messages = [{"role": "user", "content": ctx.user_query}]
        result = LoopResult()
        round_num = 0
        try:
            for round_num in range(self.max_rounds):
                # ① PRE_ROUND
                self.hooks.run(HookPoint.PRE_ROUND, HookContext(HookPoint.PRE_ROUND, round_num, ctx, messages))

                # ② CONTEXT — compact if over budget
                messages = self.cm.maybe_compact(messages)

                # ③ PRE_LLM_CALL
                self.hooks.run(HookPoint.PRE_LLM_CALL, HookContext(HookPoint.PRE_LLM_CALL, round_num, ctx, messages))

                # ④ LLM CALL — the one non-deterministic step
                resp = self.llm.complete(system, messages, self.registry.schemas())

                # ⑤ POST_LLM_CALL
                self.hooks.run(HookPoint.POST_LLM_CALL,
                               HookContext(HookPoint.POST_LLM_CALL, round_num, ctx, messages, llm_response=resp))

                tool_calls = resp["tool_calls"]

                # No tool calls => the model is done acting. END_TURN.
                if not tool_calls:
                    result.content = resp["text"]
                    result.stop_reason = StopReason.END_TURN
                    break

                # ⑥–⑧ per tool: permission gate, execute, observe
                tool_results: List[ToolResult] = []
                for call in tool_calls:
                    gate = self.hooks.run(
                        HookPoint.PRE_TOOL_EXEC,
                        HookContext(HookPoint.PRE_TOOL_EXEC, round_num, ctx, messages, tool_call=call),
                    )
                    if gate and gate.get("block"):
                        tr = ToolResult(call.id, call.name, success=False,
                                        error=f"blocked by permission gate: {gate.get('reason', '')}")
                    else:
                        tr = self.registry.execute(call, ctx)
                    tool_results.append(tr)
                    self.hooks.run(HookPoint.POST_TOOL_EXEC,
                                   HookContext(HookPoint.POST_TOOL_EXEC, round_num, ctx, messages, tool_result=tr))

                # ⑨ VERIFY — guaranteed, not a hook
                self.verification.verify(tool_results)

                # ⑩ APPEND — assistant turn + tool results enter history
                tool_result_blocks = [
                    {"type": "tool_result", "tool_use_id": tr.tool_call_id, "content": tr.to_model_content(),
                     "is_error": not tr.success}
                    for tr in tool_results
                ]
                messages = self.cm.append_round(messages, resp["content_blocks"], tool_result_blocks)

                # ⑪ POST_ROUND — standing guidance; may set the terminate flag
                self.hooks.run(HookPoint.POST_ROUND, HookContext(HookPoint.POST_ROUND, round_num, ctx, messages))
                if ctx.get("_terminate"):
                    result.content = resp["text"]
                    result.stop_reason = StopReason.TERMINATED
                    break
            else:
                result.stop_reason = StopReason.MAX_ROUNDS
                logger.warning("hit max rounds (%d)", self.max_rounds)

        except Exception as e:  # ⑫ ON_ERROR
            result.stop_reason = StopReason.ERROR
            result.error = str(e)
            logger.exception("loop error")
            self.hooks.run(HookPoint.ON_ERROR,
                           HookContext(HookPoint.ON_ERROR, round_num, ctx, messages, error=e))
        finally:  # ⑬ ON_LOOP_END — always runs
            result.rounds = round_num + 1
            ctx.set("_messages", messages)  # exposed for checkpointing
            self.hooks.run(HookPoint.ON_LOOP_END, HookContext(HookPoint.ON_LOOP_END, round_num, ctx, messages))

        return result
