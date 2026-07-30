"""Tests for the two subtlest pieces — compaction and verification — plus the
loop edge cases touched by the review fixes. All run with no API key.
"""
from harness import AgentLoop, ContextManager, ExecutionContext, VerificationPipeline
from harness.types import StopReason, ToolCall, ToolResult


# --- compaction (§3) -------------------------------------------------------

def _history():
    """A history whose head holds a full tool_result and the tail is recent."""
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "X" * 5000}]},
        {"role": "assistant", "content": [{"type": "text", "text": "recent-1"}]},
        {"role": "user", "content": [{"type": "text", "text": "recent-2"}]},
    ]


def test_compaction_stubs_old_results_keeps_tool_use_and_tail():
    cm = ContextManager(token_budget=100, keep_recent=2)
    msgs = cm.maybe_compact(_history())

    # Old tool_result stubbed...
    old_result = msgs[2]["content"][0]
    assert old_result["type"] == "tool_result"
    assert "compacted" in old_result["content"]
    # ...but the action record (tool_use) survives...
    assert msgs[1]["content"][0]["type"] == "tool_use"
    # ...and the recent tail is untouched.
    assert msgs[-1]["content"][0]["text"] == "recent-2"
    assert msgs[-2]["content"][0]["text"] == "recent-1"


def test_compaction_is_idempotent():
    cm = ContextManager(token_budget=100, keep_recent=2)
    once = cm.maybe_compact(_history())
    tokens_once = cm.estimate_tokens(once)
    twice = cm.maybe_compact(once)
    assert cm.estimate_tokens(twice) == tokens_once  # no further shrink, no growth


def test_under_budget_is_left_alone():
    cm = ContextManager(token_budget=10_000, keep_recent=2)
    msgs = _history()
    before = cm.estimate_tokens(msgs)
    cm.maybe_compact(msgs)
    assert cm.estimate_tokens(msgs) == before


# --- verification (§6) -----------------------------------------------------

def test_verification_flags_empty_success():
    tr = ToolResult(tool_call_id="t1", tool_name="read", success=True, content="")
    VerificationPipeline().verify([tr])
    assert tr.verified is False
    assert tr.verification_reason
    # advisory: the failed verdict is prefixed onto the model-visible content
    assert "verification failed" in tr.to_model_content()


def test_verification_passes_nonempty_success():
    tr = ToolResult(tool_call_id="t1", tool_name="read", success=True, content="real content")
    VerificationPipeline().verify([tr])
    assert tr.verified is True
    assert tr.to_model_content() == "real content"


# --- loop edges touched by the fixes ---------------------------------------

class _NeverCalled:
    def complete(self, *a, **k):
        raise AssertionError("llm should not be called")


def test_zero_rounds_reports_zero():
    from harness import ToolRegistry
    loop = AgentLoop(_NeverCalled(), ToolRegistry(), max_rounds=0)
    result = loop.run("sys", ExecutionContext(session_id="t", user_query="q"))
    assert result.stop_reason == StopReason.MAX_ROUNDS
    assert result.rounds == 0


def test_token_budget_stop_before_llm_call():
    from harness import ToolRegistry
    cm = ContextManager(token_budget=1, keep_recent=4)  # any real history busts it
    loop = AgentLoop(_NeverCalled(), ToolRegistry(), context_manager=cm, max_rounds=5)
    result = loop.run("sys", ExecutionContext(session_id="t", user_query="a long question here"))
    assert result.stop_reason == StopReason.TOKEN_BUDGET


class _RecordingLLM:
    def __init__(self):
        self.seen = None

    def complete(self, system, messages, tools, max_tokens=8000):
        self.seen = messages
        return {"text": "ok", "tool_calls": [], "content_blocks": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn"}


def test_resume_seeds_prior_messages():
    from harness import ToolRegistry
    llm = _RecordingLLM()
    loop = AgentLoop(llm, ToolRegistry(), max_rounds=5)
    seed = [{"role": "user", "content": "earlier"},
            {"role": "assistant", "content": [{"type": "text", "text": "prior turn"}]}]
    loop.run("sys", ExecutionContext(session_id="t", user_query="ignored"), initial_messages=seed)
    assert llm.seen == seed  # loop resumed from the checkpoint, not a fresh user turn


def test_max_tokens_stop_does_not_append_truncated_tool_use():
    from harness import ToolRegistry

    class _Truncated:
        def complete(self, *a, **k):
            return {"text": "partial", "stop_reason": "max_tokens",
                    "tool_calls": [ToolCall(id="t1", name="x", arguments={})],
                    "content_blocks": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]}

    loop = AgentLoop(_Truncated(), ToolRegistry(), max_rounds=5)
    result = loop.run("sys", ExecutionContext(session_id="t", user_query="q"))
    assert result.stop_reason == StopReason.MAX_TOKENS
    assert result.content == "partial"
