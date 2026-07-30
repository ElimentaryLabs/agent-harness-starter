"""Assemble the PDF Q&A agent from the harness pieces.

This is the whole point of the starter: the harness is generic; an *agent* is
just a system prompt + a set of tools + the hooks you choose to plug in. Swap
the tools and prompt and you have a different agent on the same loop.
"""
from __future__ import annotations

import os
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
    load_checkpoint,
    observability_hooks,
    read_only_permission_gate,
)
from harness.types import HookPoint

from .prompt import SYSTEM_PROMPT
from .tools import PdfDocument, pdf_tools


def _resume_messages(session_id: str | None) -> list | None:
    """Saved messages for a session, if it has a checkpoint (§8)."""
    if not session_id:
        return None
    ckpt = load_checkpoint(session_id)
    return ckpt["messages"] if ckpt else None


def _register_all(hooks: HookManager, hookmap: dict) -> None:
    for point, fns in hookmap.items():
        for fn in fns:
            hooks.register(point, fn)


def _build_hooks(session_id: str) -> HookManager:
    """Wire the fail-open hook sets: observability, permission gate, checkpointing."""
    hooks = HookManager()
    _register_all(hooks, observability_hooks())
    hooks.register(HookPoint.PRE_TOOL_EXEC, read_only_permission_gate)
    _register_all(hooks, checkpoint_hooks(session_id))
    return hooks


def answer_question(
    pdf_path: str,
    question: str,
    *,
    max_rounds: int | None = None,
    session_id: str | None = None,
) -> LoopResult:
    """Run the agent end to end: load the PDF, wire the harness, ask the question.

    Pass an existing `session_id` to resume from its checkpoint (§8) — the loop
    picks up the saved messages instead of starting over.
    """
    # Config seams: env overrides, documented in .env.example (§1).
    max_rounds = max_rounds if max_rounds is not None else int(os.getenv("HARNESS_MAX_ROUNDS", "12"))
    token_budget = int(os.getenv("HARNESS_TOKEN_BUDGET", "120000"))

    llm = LLMClient()

    # §4 capability surface
    registry = ToolRegistry()
    for tool in pdf_tools():
        registry.register(tool)

    # §8 resume: if a known session_id has a checkpoint, seed the loop with it.
    resume_messages = _resume_messages(session_id)

    # §5/§7/§8 hooks — observability (default-on), the permission gate, checkpointing
    session_id = session_id or uuid.uuid4().hex[:12]
    hooks = _build_hooks(session_id)

    # Shared run state — the PDF document and the LLM client the read_page tool uses.
    ctx = ExecutionContext(session_id=session_id, user_query=question)
    ctx.set("pdf", PdfDocument(pdf_path))
    ctx.set("llm", llm)

    loop = AgentLoop(
        llm=llm,
        registry=registry,
        context_manager=ContextManager(token_budget=token_budget),
        verification=VerificationPipeline(),  # §6 guaranteed
        hooks=hooks,
        max_rounds=max_rounds,
    )
    return loop.run(SYSTEM_PROMPT, ctx, initial_messages=resume_messages)
