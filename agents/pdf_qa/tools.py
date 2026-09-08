"""Tools for the PDF Q&A agent (§4 — the agent's capability surface).

The PDF is never dumped into the model's context. Instead the agent navigates
it just-in-time: a cheap `list_pages`/`search_pdf` to orient, then `read_page`
to pull a single page — parsed natively by Claude — only when needed. Each
extracted page is cached, so a re-read (or a compacted history) is a cheap
re-fetch from the source of truth, not a lost result.
"""
from __future__ import annotations

import io
from typing import Any, Dict

from pypdf import PdfReader, PdfWriter

from harness import ExecutionContext, ToolDef


class PdfDocument:
    """Loads a PDF and lazily caches Claude-extracted Markdown per page."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.reader = PdfReader(path)
        self.num_pages = len(self.reader.pages)
        self._markdown_cache: Dict[int, str] = {}
        self._raw_text_cache: Dict[int, str] = {}

    def _raw_text(self, idx: int) -> str:
        if idx not in self._raw_text_cache:
            try:
                self._raw_text_cache[idx] = (self.reader.pages[idx].extract_text() or "").strip()
            except Exception:
                self._raw_text_cache[idx] = ""
        return self._raw_text_cache[idx]

    def page_pdf_bytes(self, idx: int) -> bytes:
        writer = PdfWriter()
        writer.add_page(self.reader.pages[idx])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def markdown(self, idx: int, llm) -> tuple[str, bool]:
        """Return (markdown, was_cached). Misses call Claude's native PDF parse."""
        if idx in self._markdown_cache:
            return self._markdown_cache[idx], True
        md = llm.parse_pdf_page(self.page_pdf_bytes(idx))
        self._markdown_cache[idx] = md
        return md, False


# --- tool handlers ---------------------------------------------------------

def _list_pages(args: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    doc: PdfDocument = ctx.get("pdf")
    lines = [f"The document has {doc.num_pages} page(s). Cheap text previews:"]
    for i in range(doc.num_pages):
        preview = " ".join(doc._raw_text(i).split())[:120]
        lines.append(f"  p{i + 1}: {preview or '(no extractable text — likely scanned/visual)'}")
    return {"content": "\n".join(lines)}


def _read_page(args: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    doc: PdfDocument = ctx.get("pdf")
    llm = ctx.get("llm")
    page = int(args.get("page", 0))
    if page < 1 or page > doc.num_pages:
        return {"success": False, "error": f"page {page} out of range (1..{doc.num_pages})"}
    md, cached = doc.markdown(page - 1, llm)
    header = f"## Page {page} of {doc.num_pages}" + (" (cached)" if cached else "")
    note = "\n\n_(call read_page again any time to re-fetch this page)_"
    return {"content": f"{header}\n\n{md}{note}"}


def _search_pdf(args: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    doc: PdfDocument = ctx.get("pdf")
    query = str(args.get("query", "")).strip().lower()
    if not query:
        return {"success": False, "error": "query is required"}
    hits = []
    for i in range(doc.num_pages):
        text = doc._raw_text(i)
        pos = text.lower().find(query)
        if pos != -1:
            start = max(0, pos - 60)
            snippet = " ".join(text[start:pos + 120].split())
            hits.append(f"  p{i + 1}: …{snippet}…")
    if not hits:
        return {"content": f"No raw-text matches for '{query}'. It may be on a visual/scanned "
                           f"page — use read_page to parse pages natively."}
    return {"content": f"Matches for '{query}':\n" + "\n".join(hits)}


def pdf_tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="list_pages",
            description="List the document's pages with a cheap one-line text preview of each. "
                        "Call this first to orient yourself before reading pages.",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=_list_pages,
        ),
        ToolDef(
            name="search_pdf",
            description="Find which pages mention a phrase (fast raw-text search). Call this to "
                        "locate where a topic is discussed before reading full pages. Returns page "
                        "numbers and snippets; misses on scanned/visual pages.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Phrase to find"}},
                "required": ["query"],
            },
            handler=_search_pdf,
        ),
        ToolDef(
            name="read_page",
            description="Read one page's full content as clean Markdown, parsed natively by Claude "
                        "(handles tables, figures, and scanned pages). Call this when you need the "
                        "actual content of a page to answer the question. Read pages one at a time, "
                        "only the ones you need.",
            input_schema={
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "1-based page number"}},
                "required": ["page"],
            },
            handler=_read_page,
        ),
    ]
