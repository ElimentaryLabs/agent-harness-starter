# agent-harness-starter

A tiny, readable **agent harness** in Python — the loop that turns a language
model into a system you can trust — with a working **PDF Q&A agent** built on
top. Bring your own Anthropic API key and go.

> The model is the climber. The harness is everything that keeps a long climb
> from ending in a fall.

A raw LLM left alone on a long task *digresses* — forgets the goal, assumes a
tool worked, loops forever. The harness is the deterministic frame around the
non-deterministic model: it manages turns, tools, verification, context,
permissions, observability, and recovery. **You don't program the model. You
program the loop around it.**

This repo is the companion code to the article *"The Harness — For Engineers,
By an Engineer."* Each module maps to a section of it.

---

## Quickstart

```bash
git clone <your-fork-url> agent-harness-starter
cd agent-harness-starter

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then put your key in .env
# or: export ANTHROPIC_API_KEY=sk-ant-...

python -m examples.run path/to/document.pdf "What is the refund policy?"
```

Get a key at <https://console.anthropic.com/settings/keys>.

You'll see the harness narrate each round (which tools the model reached for,
results, the final answer):

```
[round 1] start
[round 1] model wants tools: ['list_pages']
    tool list_pages -> ok (412 chars)
[round 2] model wants tools: ['read_page']
    tool read_page -> ok (1843 chars)
[round 3] model produced a final answer
💡  Answer (stop=end_turn, rounds=3):
The refund policy allows returns within 30 days… (p. 4)
```

No key handy? The harness itself is covered by a fake-LLM test that needs no
network:

```bash
pip install pytest && pytest -q
```

---

## How the PDF agent works

The PDF is **never dumped into the model's context**. The agent navigates it
just-in-time, the way a person would:

1. `list_pages` — a cheap overview (page count + text previews).
2. `search_pdf` — find which pages mention a topic.
3. `read_page` — pull one page's full content, **parsed natively by Claude**
   (handles tables, figures, and scanned pages). Read only the pages needed.

Each extracted page is cached, so a re-read — or a page that got squeezed out of
a long history during compaction — is a cheap re-fetch from the source of truth,
not a lost result. This is the "lightweight identifier + re-fetch" pattern: keep
references in context, not megabytes of data.

The agent finishes by simply **stopping** — when the model answers without
calling a tool, the turn ends. No special "done" signal.

---

## How the harness maps to the article

| Module | Section | What it does |
|---|---|---|
| `harness/loop.py` | §1 | the execution loop — the 13-step spine |
| `harness/llm.py` | §2 | prompt assembly (cached system) + the one non-deterministic call |
| `harness/context.py` | §3 | context management & compaction |
| `harness/registry.py` | §4 | the tool/capability surface |
| `harness/hooks.py` | §5 | hooks — the extensibility seams (+ default-on observability) |
| `harness/verification.py` | §6 | verification — **guaranteed, not a hook** |
| `harness/hooks.py` | §7 | permissions — the `read_only_permission_gate` |
| `harness/checkpoint.py` | §8 | persistence & recovery |

Two distinctions worth internalizing from the code:

- **Hooks are fail-open** (wrapped in try/except — a telemetry or extension bug
  can never crash a run). That's why **verification is not a hook**: correctness
  must be guaranteed, so it's a built-in loop step, not an optional plugin.
- **Observability is the canonical hook** — a lifecycle hook set you make
  "mandatory" by registering it by default, not by welding it into the loop.

---

## Build your own agent

The harness is generic. An agent is just **a system prompt + a set of tools +
the hooks you choose**. To make a new one, copy `agents/pdf_qa/` and swap the
tools and prompt:

```python
from harness import (
    AgentLoop, ToolRegistry, ToolDef, ExecutionContext,
    HookManager, observability_hooks,
)

def get_weather(args, ctx):
    return {"content": f"72°F and sunny in {args['city']}"}

registry = ToolRegistry()
registry.register(ToolDef(
    name="get_weather",
    description="Get current weather for a city. Call this when the user asks about weather.",
    input_schema={"type": "object",
                  "properties": {"city": {"type": "string"}}, "required": ["city"]},
    handler=get_weather,
))

hooks = HookManager()
for point, fns in observability_hooks().items():
    for fn in fns:
        hooks.register(point, fn)

from harness import LLMClient
loop = AgentLoop(LLMClient(), registry, hooks=hooks)
result = loop.run("You are a helpful weather assistant.",
                  ExecutionContext(session_id="demo", user_query="Weather in Paris?"))
print(result.content)
```

Tool descriptions are part of the prompt — be **prescriptive about *when* to
call** a tool, not just what it does.

---

## Configuration

Environment variables (all optional except the key):

| Var | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **required** |
| `HARNESS_MODEL` | `claude-opus-4-8` | the reasoning agent's model |
| `HARNESS_PARSE_MODEL` | `claude-haiku-4-5` | per-page PDF extraction (a cheaper model for the bulk sub-task) |
| `HARNESS_THINKING` | `on` | set `off` to disable adaptive thinking |

If your chosen parse model lacks PDF support, set `HARNESS_PARSE_MODEL` to one
that has it (e.g. `claude-sonnet-4-6`).

---

## Project layout

```
harness/            the reusable harness (domain-agnostic)
  loop.py           §1  the execution loop
  llm.py            §2  Anthropic integration (tool use, caching, native PDF)
  context.py        §3  compaction
  registry.py       §4  tools
  hooks.py          §5/§7  hooks, observability, permission gate
  verification.py   §6  guaranteed checks
  checkpoint.py     §8  persistence
  types.py          shared vocabulary
agents/pdf_qa/      a sample agent: prompt + tools + assembly
examples/run.py     CLI entry point
tests/test_loop.py  fake-LLM test (no API key)
```

---

## What this is (and isn't)

It's a **teaching scaffold** — small enough to read in one sitting, faithful to
the patterns that make agents reliable. It is intentionally synchronous and
dependency-light. For production you'd add streaming, richer retrieval,
rate-limit handling beyond the SDK's defaults, and durable storage. The shapes,
though, are the same ones you'd keep.

MIT licensed — fork it, gut it, ship your own.
