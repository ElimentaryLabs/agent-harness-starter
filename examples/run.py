"""CLI: ask a question about a PDF.

    python -m examples.run path/to/document.pdf "What is the refund policy?"
    python -m examples.run --session <id> path/to/document.pdf "..."   # resume (§8)

Reads ANTHROPIC_API_KEY from the environment (or a local .env file).
"""
from __future__ import annotations

import logging
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()  # load .env if present

from agents.pdf_qa import answer_question  # noqa: E402  (after load_dotenv)


def main() -> int:
    args = sys.argv[1:]
    session_id = None
    if len(args) >= 2 and args[0] == "--session":
        session_id, args = args[1], args[2:]
    if len(args) < 2:
        print(__doc__)
        return 2

    pdf_path, question = args[0], " ".join(args[1:])
    # Own the id here so we can print it — the user needs it to resume (§8).
    session_id = session_id or uuid.uuid4().hex[:12]

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet the noisy HTTP client; keep the harness breadcrumbs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    print(f"\n📄  {pdf_path}\n❓  {question}\n🔖  session {session_id}\n" + "─" * 60)
    result = answer_question(pdf_path, question, session_id=session_id)
    print("─" * 60)
    print(f"\n💡  Answer (stop={result.stop_reason.value}, rounds={result.rounds}):\n")
    print(result.content or result.error or "(no answer)")
    print(f"\n↺  resume: python -m examples.run --session {session_id} {pdf_path} \"...\"")
    return 0 if not result.error else 1


if __name__ == "__main__":
    raise SystemExit(main())
