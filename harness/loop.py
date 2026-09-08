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

    def run(
        self,
        system: str,
        ctx: ExecutionContext,
        initial_messages: Optional[List[dict]] = None,
    ) -> LoopResult:
        # Seed from a checkpoint if given (§8 resume), else start fresh.
        messages = initial_messages if initial_messages else [{"role": "user", "content": ctx.user_query}]
        result = LoopResult()
        round_num = -1  # so a zero-round run reports rounds=0, not 1
        try:
            for round_num in range(self.max_rounds):
                # ① PRE_ROUND
                self.hooks.run(HookPoint.PRE_ROUND, HookContext(HookPoint.PRE_ROUND, round_num, ctx, messages))

                # ② CONTEXT — compact if over budget
                messages = self.cm.maybe_compact(messages)
                # If the tail alone still busts the budget, compaction can't help.
                # Stop cleanly instead of sailing on to an API ERROR next call.
                if self.cm.over_budget(messages):
                    result.stop_reason = StopReason.TOKEN_BUDGET
                    logger.warning("token budget exceeded after compaction")
                    break

                # ③ PRE_LLM_CALL
                self.hooks.run(HookPoint.PRE_LLM_CALL, HookContext(HookPoint.PRE_LLM_CALL, round_num, ctx, messages))

                # ④ LLM CALL — the one non-deterministic step
                resp = self.llm.complete(system, messages, self.registry.schemas())

                # ⑤ POST_LLM_CALL
                self.hooks.run(HookPoint.POST_LLM_CALL,
                               HookContext(HookPoint.POST_LLM_CALL, round_num, ctx, messages, llm_response=resp))

                # Terminal responses (model done, or cut off at max_tokens) end here.
                terminal = self._terminal_stop(resp)
                if terminal is not None:
                    result.content = resp["text"]
                    result.stop_reason = terminal
                    break

                # ⑥–⑨ execute tools (permission gate, execute, observe) then VERIFY
                tool_results = self._execute_tools(resp["tool_calls"], round_num, ctx, messages)
                self.verification.verify(tool_results)  # guaranteed, not a hook

                # ⑩ APPEND — assistant turn + tool results enter history
                messages = self.cm.append_round(messages, resp["content_blocks"], _result_blocks(tool_results))

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

    @staticmethod
    def _terminal_stop(resp: dict) -> Optional[StopReason]:
        """A stop reason if this response ends the loop, else None."""
        # Cut off at max_tokens: a trailing tool_use may be truncated, so
        # appending it would send malformed input next round. Stop instead.
        # ponytail: known ceiling. A fuller harness would retry with a higher
        # max_tokens; here we surface it and stop.
        if resp["stop_reason"] == "max_tokens":
            logger.warning("model stopped at max_tokens — output may be truncated")
            return StopReason.MAX_TOKENS
        # No tool calls => the model is done acting.
        if not resp["tool_calls"]:
            return StopReason.END_TURN
        return None

    def _execute_tools(self, tool_calls, round_num, ctx, messages) -> List[ToolResult]:
        """⑥–⑧ per tool: permission gate, execute, observe."""
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
        return tool_results


def _result_blocks(tool_results: List[ToolResult]) -> List[dict]:
    return [
        {"type": "tool_result", "tool_use_id": tr.tool_call_id, "content": tr.to_model_content(),
         "is_error": not tr.success}
        for tr in tool_results
    ]
