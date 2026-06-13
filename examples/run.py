"""CLI: ask a question about a PDF.

    python -m examples.run path/to/document.pdf "What is the refund policy?"

Reads ANTHROPIC_API_KEY from the environment (or a local .env file).
"""
from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # load .env if present

from agents.pdf_qa import answer_question  # noqa: E402  (after load_dotenv)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    pdf_path, question = sys.argv[1], " ".join(sys.argv[2:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    # Quiet the noisy HTTP client; keep the harness breadcrumbs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    print(f"\n📄  {pdf_path}\n❓  {question}\n" + "─" * 60)
    result = answer_question(pdf_path, question)
    print("─" * 60)
    print(f"\n💡  Answer (stop={result.stop_reason.value}, rounds={result.rounds}):\n")
    print(result.content or result.error or "(no answer)")
    print()
    return 0 if not result.error else 1


if __name__ == "__main__":
    raise SystemExit(main())
