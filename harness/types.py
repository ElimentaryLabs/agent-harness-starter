"""Core types shared across the harness — the shared vocabulary.

Read these alongside the article: a ToolDef is a *spec* (§4), HookPoint names
the *seams* (§5), and ToolResult is what VERIFY (§6) gates on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# --- Tools -----------------------------------------------------------------

@dataclass
class ToolDef:
    """A tool the model may call.

    `handler(args, ctx) -> dict` runs the side-effecting work. `mutating`
    marks tools that change the world — the permission gate (§7) treats those
    differently from read-only tools.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], "ExecutionContext"], Dict[str, Any]]
    mutating: bool = False

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    """A tool call parsed out of the model's response."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing one tool call."""
    tool_call_id: str
    tool_name: str
    success: bool
    content: str = ""           # text returned to the model next round
    error: Optional[str] = None
    # set by the verification pipeline (§6)
    verified: Optional[bool] = None
    verification_reason: Optional[str] = None

    def to_model_content(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if self.verified is False:
            return f"[verification failed: {self.verification_reason}]\n{self.content}"
        return self.content or "(tool executed successfully)"


# --- Loop control ----------------------------------------------------------

class StopReason(str, Enum):
    END_TURN = "end_turn"          # model answered with no tool call
    MAX_ROUNDS = "max_rounds"      # hit the turn budget
    TOKEN_BUDGET = "token_budget"  # context exhausted
    TERMINATED = "terminated"      # a POST_ROUND hook set the terminate flag
    ERROR = "error"


@dataclass
class LoopResult:
    content: str = ""
    stop_reason: StopReason = StopReason.END_TURN
    rounds: int = 0
    error: Optional[str] = None


# --- Hooks (§5) ------------------------------------------------------------

class HookPoint(str, Enum):
    """The named seams. The harness owns *when* these fire; agents plug in
    *what* runs. Observability lives here as a fail-open hook set (§5)."""
    PRE_ROUND = "pre_round"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_EXEC = "pre_tool_exec"      # gating: may return {"block": True, ...}
    POST_TOOL_EXEC = "post_tool_exec"
    POST_ROUND = "post_round"
    ON_ERROR = "on_error"
    ON_LOOP_END = "on_loop_end"


# Hook points whose return value can block/alter the action.
GATING_HOOKS = frozenset({HookPoint.PRE_TOOL_EXEC})


@dataclass
class HookContext:
    """Everything a hook might need. Fields are populated per hook point."""
    hook_point: HookPoint
    round_number: int
    ctx: "ExecutionContext"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    llm_response: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None


# --- Execution context -----------------------------------------------------

@dataclass
class ExecutionContext:
    """Shared state threaded through one run. Domain agents stash their own
    state in `state` (e.g. the PDF agent keeps its page cache here)."""
    session_id: str
    user_query: str = ""
    state: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
