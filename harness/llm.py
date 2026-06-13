"""Anthropic integration — the one non-deterministic step (§ the loop).

Wraps the Messages API for: (1) the reasoning loop's tool-use calls, and
(2) per-page PDF extraction using Claude's native PDF parsing. Everything the
SDK gives us — typed errors, prompt caching, adaptive thinking — is used here
so the rest of the harness can stay simple.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

from .types import ToolCall

logger = logging.getLogger("harness.llm")

# Defaults follow the Anthropic guidance: Opus for reasoning; a cheaper model
# for the high-volume per-page extraction sub-task (the "cheaper model for the
# sub-task" pattern — like Claude Code's Explore subagents on Haiku).
DEFAULT_MODEL = os.getenv("HARNESS_MODEL", "claude-opus-4-8")
DEFAULT_PARSE_MODEL = os.getenv("HARNESS_PARSE_MODEL", "claude-haiku-4-5")


def _supports_adaptive_thinking(model: str) -> bool:
    return model.startswith("claude-opus-4-") or model == "claude-sonnet-4-6"


class LLMClient:
    def __init__(self, model: Optional[str] = None, parse_model: Optional[str] = None):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or `export ANTHROPIC_API_KEY=sk-ant-...`."
            )
        self.client = anthropic.Anthropic()         # reads ANTHROPIC_API_KEY
        self.model = model or DEFAULT_MODEL
        self.parse_model = parse_model or DEFAULT_PARSE_MODEL
        self._thinking = (
            {"type": "adaptive"} if os.getenv("HARNESS_THINKING", "on") != "off"
            and _supports_adaptive_thinking(self.model) else None
        )

    # -- the reasoning call -------------------------------------------------

    def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[dict],
        max_tokens: int = 8000,
    ) -> Dict[str, Any]:
        """One model call. Returns parsed pieces plus the raw content blocks
        (which the loop appends verbatim so thinking + tool_use survive)."""
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            # System is stable across rounds -> prompt-cache it (tools render
            # before system, so this one breakpoint caches tools + system).
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
        )
        if self._thinking:
            kwargs["thinking"] = self._thinking

        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API error %s: %s", getattr(e, "status_code", "?"), e)
            raise

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        content_blocks: List[Dict[str, Any]] = []

        for block in resp.content:
            content_blocks.append(_serialize_block(block))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        u = resp.usage
        logger.info(
            "llm round: in=%s out=%s cache_read=%s tool_calls=%d stop=%s",
            u.input_tokens, u.output_tokens,
            getattr(u, "cache_read_input_tokens", 0), len(tool_calls), resp.stop_reason,
        )
        return {
            "text": "".join(text_parts),
            "tool_calls": tool_calls,
            "content_blocks": content_blocks,
            "stop_reason": resp.stop_reason,
        }

    # -- native PDF page extraction ----------------------------------------

    def parse_pdf_page(self, page_pdf_bytes: bytes) -> str:
        """Send a single-page PDF to Claude and get clean markdown back. This is
        where Anthropic's native PDF parsing is used — as a tool, not by dumping
        the whole document into the agent's context."""
        b64 = base64.standard_b64encode(page_pdf_bytes).decode("utf-8")
        try:
            resp = self.client.messages.create(
                model=self.parse_model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "document",
                         "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                        {"type": "text", "text":
                            "Transcribe this page to clean Markdown. Preserve headings, lists, and "
                            "tables (as Markdown tables). Output only the page content — no preamble."},
                    ],
                }],
            )
        except anthropic.APIStatusError as e:
            raise RuntimeError(
                f"PDF parse failed on model '{self.parse_model}' ({e}). "
                "If this model lacks PDF support, set HARNESS_PARSE_MODEL to one that has it "
                "(e.g. claude-sonnet-4-6)."
            ) from e
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def _serialize_block(block: Any) -> Dict[str, Any]:
    """Convert an SDK content block to a plain dict we can re-send, estimate,
    compact, and checkpoint. Thinking blocks keep their signature so adaptive
    thinking survives across tool rounds."""
    t = block.type
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fallback: best-effort dump.
    return block.model_dump() if hasattr(block, "model_dump") else {"type": t}
