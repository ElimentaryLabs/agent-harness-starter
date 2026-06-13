"""Persistence & recovery (§8).

The model is stateless — it remembers nothing between calls except what we feed
it. So we make the checkpoint first-class: snapshot the messages each round and
on loop end. A crash costs you a round, not the whole climb. The checkpoint
*is* the state.

This is a fail-open hook set (persistence is operability, not correctness), so
it's registered like any other hook — see hooks.py for why that's the right
home and verification.py for the one thing that is *not* a hook.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

from .types import HookContext, HookPoint

logger = logging.getLogger("harness.checkpoint")


def checkpoint_hooks(session_id: str, directory: str = ".harness_checkpoints") -> Dict[HookPoint, List]:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{session_id}.json")

    def _save(h: HookContext) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"session_id": session_id, "round": h.round_number + 1,
                           "messages": h.messages}, f, default=str, indent=2)
        except Exception as e:  # fail-open
            logger.warning("checkpoint save failed: %s", e)

    return {HookPoint.POST_ROUND: [_save], HookPoint.ON_LOOP_END: [_save]}


def load_checkpoint(session_id: str, directory: str = ".harness_checkpoints") -> dict | None:
    path = os.path.join(directory, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
