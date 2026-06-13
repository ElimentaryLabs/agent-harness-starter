"""Assemble the PDF Q&A agent from the harness pieces.

This is the whole point of the starter: the harness is generic; an *agent* is
just a system prompt + a set of tools + the hooks you choose to plug in. Swap
the tools and prompt and you have a different agent on the same loop.
"""
from __future__ import annotations

import uuid

from harness import (
    AgentLoop,
    ContextManager,
    ExecutionContext,
    HookManager,
    LLMClient,
    LoopResult,
    ToolRegistry,
    VerificationPipeline,
    checkpoint_hooks,
    observability_hooks,
    read_only_permission_gate,
)
from harness.types import HookPoint

from .prompt import SYSTEM_PROMPT
from .tools import PdfDocument, pdf_tools


def answer_question(pdf_path: str, question: str, *, max_rounds: int = 12) -> LoopResult:
    """Run the agent end to end: load the PDF, wire the harness, ask the question."""
    llm = LLMClient()

    # §4 capability surface
    registry = ToolRegistry()
    for tool in pdf_tools():
        registry.register(tool)

    # §5/§7/§8 hooks — observability (default-on), the permission gate, checkpointing
    session_id = uuid.uuid4().hex[:12]
    hooks = HookManager()
    for point, fns in observability_hooks().items():
        for fn in fns:
            hooks.register(point, fn)
    hooks.register(HookPoint.PRE_TOOL_EXEC, read_only_permission_gate)
    for point, fns in checkpoint_hooks(session_id).items():
        for fn in fns:
            hooks.register(point, fn)

    # Shared run state — the PDF document and the LLM client the read_page tool uses.
    ctx = ExecutionContext(session_id=session_id, user_query=question)
    ctx.set("pdf", PdfDocument(pdf_path))
    ctx.set("llm", llm)

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        context_manager=ContextManager(),
        verification=VerificationPipeline(),  # §6 guaranteed
        hooks=hooks,
        max_rounds=max_rounds,
    )
    return loop.run(SYSTEM_PROMPT, ctx)
