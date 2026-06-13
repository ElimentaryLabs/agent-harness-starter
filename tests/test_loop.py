"""Loop test with a fake LLM — runs with no API key.

Proves the spine works: the loop calls a tool, verifies, appends, then ends the
turn when the model stops asking for tools.
"""
from harness import AgentLoop, ExecutionContext, ToolDef, ToolRegistry
from harness.types import StopReason, ToolCall


class FakeLLM:
    """Scripts two rounds: round 1 calls `echo`, round 2 answers."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, messages, tools, max_tokens=8000):
        self.calls += 1
        if self.calls == 1:
            return {
                "text": "",
                "tool_calls": [ToolCall(id="t1", name="echo", arguments={"text": "hi"})],
                "content_blocks": [{"type": "tool_use", "id": "t1", "name": "echo", "input": {"text": "hi"}}],
                "stop_reason": "tool_use",
            }
        return {
            "text": "done: hi",
            "tool_calls": [],
            "content_blocks": [{"type": "text", "text": "done: hi"}],
            "stop_reason": "end_turn",
        }


def _echo(args, ctx):
    return {"content": f"echo: {args['text']}"}


def test_loop_runs_tool_then_answers():
    registry = ToolRegistry()
    registry.register(ToolDef(
        name="echo",
        description="echo the input back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=_echo,
    ))
    loop = AgentLoop(FakeLLM(), registry, max_rounds=5)
    ctx = ExecutionContext(session_id="test", user_query="say hi")

    result = loop.run("you are a test agent", ctx)

    assert result.stop_reason == StopReason.END_TURN
    assert "done" in result.content
    assert result.rounds == 2


def test_unknown_tool_is_a_failed_result_not_a_crash():
    registry = ToolRegistry()  # no tools registered
    ctx = ExecutionContext(session_id="t", user_query="x")
    call = ToolCall(id="t1", name="nope", arguments={})
    tr = registry.execute(call, ctx)
    assert tr.success is False and "unknown tool" in (tr.error or "")
