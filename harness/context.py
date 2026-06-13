"""Context management & compaction (§3).

Tool results enter history *full* in the round they occur (the model needs them
to act). When the running estimate crosses the budget, we compact: old
tool_result blocks are squeezed to a stub, while the identity prompt (held
separately as `system`), the live thread (last messages), and the action record
(tool_use blocks) are preserved.

The agent's tools cache their real output elsewhere (e.g. the PDF page cache) and
the stub tells the model how to re-fetch — so compaction is lossless in
practice: shrink the tokens, keep the meaning recoverable.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("harness.context")

_CHARS_PER_TOKEN = 4


class ContextManager:
    def __init__(self, token_budget: int = 120_000, keep_recent: int = 4) -> None:
        self.token_budget = token_budget
        self.keep_recent = keep_recent  # messages at the tail left untouched

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            c = m.get("content", "")
            total += len(c) if isinstance(c, str) else len(json.dumps(c, default=str))
        return total // _CHARS_PER_TOKEN

    def append_round(
        self,
        messages: List[Dict[str, Any]],
        assistant_content: List[Dict[str, Any]],
        tool_result_blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if tool_result_blocks:
            messages.append({"role": "user", "content": tool_result_blocks})
        return messages

    def maybe_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.estimate_tokens(messages) <= self.token_budget:
            return messages
        if len(messages) <= self.keep_recent:
            return messages

        head, tail = messages[:-self.keep_recent], messages[-self.keep_recent:]
        before = self.estimate_tokens(messages)
        for m in head:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                # Keep tool_use blocks (the action record); stub old tool_results.
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block["content"] = "[older result compacted — re-call the tool to re-fetch]"
        after = self.estimate_tokens(messages)
        logger.info("compacted context: ~%d -> ~%d tokens", before, after)
        return messages
