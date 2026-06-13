"""A minimal, readable agent harness — the loop that turns a language model
into a system you can trust.

Pair this code with the companion article. Each module maps to a section:
  loop.py          §1  the execution loop (the spine)
  llm.py           §2  prompt assembly + the one non-deterministic call
  context.py       §3  context management & compaction
  registry.py      §4  skills & tools (the capability surface)
  hooks.py         §5  hooks — the extensibility seams (incl. observability)
  verification.py  §6  verification — guaranteed, not pluggable
  hooks.py         §7  permissions — the read_only_permission_gate
  checkpoint.py    §8  persistence & recovery
"""
from .context import ContextManager
from .hooks import (
    HookManager,
    observability_hooks,
    read_only_permission_gate,
)
from .checkpoint import checkpoint_hooks, load_checkpoint
from .llm import LLMClient
from .loop import AgentLoop
from .registry import ToolRegistry
from .types import (
    ExecutionContext,
    HookPoint,
    LoopResult,
    StopReason,
    ToolDef,
    ToolResult,
)
from .verification import VerificationPipeline

__all__ = [
    "AgentLoop",
    "ToolRegistry",
    "ContextManager",
    "VerificationPipeline",
    "HookManager",
    "observability_hooks",
    "read_only_permission_gate",
    "checkpoint_hooks",
    "load_checkpoint",
    "LLMClient",
    "ExecutionContext",
    "ToolDef",
    "ToolResult",
    "HookPoint",
    "LoopResult",
    "StopReason",
]
