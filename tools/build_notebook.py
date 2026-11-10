"""Assemble the lecture notebook from the tested module.

    uv run python tools/build_notebook.py

The code cells are lifted verbatim out of `src/cfa_agent/`, so what students run is
exactly what the test suite covers. Run this after `make_snapshot.py` -- the cached
figures are embedded into the notebook, not read from disk, because Colab has no
checkout of this repo.

The notebook stays an ordinary file afterwards: edit it freely. If you edit a *code*
cell, `tests/test_notebook.py` will tell you it has drifted from the module, and
either re-running this script or porting the change back into the module settles it.
"""

import ast
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cfa_agent"
OUT = ROOT / "notebooks" / "basic_agent.ipynb"


def _segments(filename: str, names: list[str]) -> dict[str, str]:
    """Pull named top-level definitions out of a module, as exact source text."""
    source = (PACKAGE / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
        else:
            continue
        if key in names:
            found[key] = ast.get_source_segment(source, node)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{filename}: could not find {missing}")
    return found


T = _segments(
    "tools.py",
    [
        "_ALLOWED_NODES",
        "_MAX_EXPONENT",
        "_BINARY_OPS",
        "_evaluate",
        "calculator",
        "CALCULATOR_SCHEMA",
        "_ddgs_text",
        "_format_results",
        "web_search",
        "WEB_SEARCH_SCHEMA",
        "STOCK_FIELDS",
        "_yf_info",
        "_yf_history",
        "_history_stats",
        "_live_quote",
        "get_stock_data",
        "STOCK_DATA_SCHEMA",
    ],
)
A = _segments(
    "agent.py",
    [
        "SYSTEM_PROMPT",
        "_user_message",
        "BANK_SYSTEM_PROMPT",
        "_base_url_of",
        "_create",
        "AgentResult",
        "_arguments_of",
        "_invoke",
        "run_agent",
        "format_trace",
        "run_stage",
    ],
)
C = _segments(
    "config.py",
    [
        "DEFAULT_BASE_URL",
        "DEFAULT_MODEL",
        "CredentialError",
        "_from_colab",
        "_load_dotenv",
        "_lookup",
        "get_credentials",
    ],
)

CACHE = json.loads((PACKAGE / "cache" / "roche.json").read_text(encoding="utf-8"))


def _pinned_requirements(names: list[str]) -> str:
    """Build the pip line from the versions uv resolved.

    Colab installs from PyPI and knows nothing about uv.lock, so without explicit
    pins a breaking release on any of these could land between building this
    notebook and teaching from it. Reading the lock keeps the room on the same
    versions the test suite ran against.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {p["name"]: p["version"] for p in lock.get("package", [])}
    missing = [n for n in names if n not in versions]
    if missing:
        raise SystemExit(f"uv.lock has no version for {missing} -- run `uv lock` first")
    return " ".join(f"{name}=={versions[name]}" for name in names)


PINS = _pinned_requirements(["openai", "yfinance", "ddgs"])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    # 1
    md(
        """
# Giving an LLM tools: a buy/hold/sell recommendation

A language model on its own can only tell you what it absorbed during training. Ask it
about a company and it will answer confidently, in the voice of an analyst, using
numbers it half-remembers from a world that has since moved on.

An **agent** is the fix, and it is a small idea: let the model call functions. We give it
a loop. Each time round, the model either answers or asks to run one of our functions.
We run it, hand back the result, and ask again.

Today we build that up in four stages, and at each stage we look at what the answer is
still missing:

| Stage | What the model can reach | What is still wrong |
|---|---|---|
| 1 | nothing | invented figures |
| 2 | a calculator | precise arithmetic on invented figures |
| 3 | web search | current context, but as prose |
| 4 | financial data | figures you can check |

Run the cells in order. At the end: a few bonus experiments, and one job of your own, which is to add a tool.
"""
    ),
    md(
        """
## Section 0 · Setup

Run the next three cells once: install the libraries, read your credentials, and create
the client. If a cell stops with a message about the API key, the message says how to fix it.
"""
    ),
    # 2
    code(
        f"""
# Colab starts empty every session, so install what we need.
# The versions are pinned deliberately: these are the ones this notebook was
# built and tested against, and yfinance in particular changes often enough
# that an unpinned install could behave differently from one week to the next.
!pip install -q {PINS}
"""
    ),
    # 3
    code(
        f'''
import getpass
import importlib
import os
from pathlib import Path

{C["DEFAULT_BASE_URL"]}
{C["DEFAULT_MODEL"]}


{C["CredentialError"]}


{C["_from_colab"]}


{C["_load_dotenv"]}


{C["_lookup"]}


{C["get_credentials"]}


BASE_URL, API_KEY, MODEL = get_credentials()
print(f"Using model {{MODEL}} at {{BASE_URL}}")
'''
    ),
    # 4
    code(
        f'''
from openai import OpenAI

# One client, pointed at whichever provider BASE_URL names.
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# The slides show the naive version: catch everything, print a message. It hides the
# real error from the loop. The version below translates the common failures instead.
# def _create(client, **kwargs):
#     try:
#         return client.chat.completions.create(**kwargs)
#     except Exception as exc:
#         return f"unexpected error: {{exc}}"


{A["_base_url_of"]}


{A["_create"]}
'''
    ),
    # 5
    code(
        f'''
COMPANY = "Roche Holding AG (ROG.SW)"
TICKER = "ROG.SW"

{A["SYSTEM_PROMPT"]}
NAIVE_SYSTEM_PROMPT = SYSTEM_PROMPT  # to switch back after the bonus cells


{A["_user_message"]}


{A["BANK_SYSTEM_PROMPT"]}
'''
    ),
    # 5c
    md(
        """
## Section 1 · The loop and how to read its trace

Every stage below prints the same three things after the answer:

- **Tool calls, in order**: which function the model asked for, with which arguments.
  `get_stock_data(ticker='ROG.SW')` is a different request from `get_stock_data(ticker='Roche')`.
- **Model calls**: how many times the loop went round. Each one is a paid API call.
- **Tokens**: the size of everything sent and received, added up over those calls.

Judge the *trace*, not only the answer: a confident recommendation with no tool calls
behind it is a recollection.
"""
    ),
    # 5b
    code(
        f'''
import json
from dataclasses import dataclass, field


@dataclass
{A["AgentResult"]}


{A["_arguments_of"]}


{A["_invoke"]}


{A["run_agent"]}


{A["format_trace"]}


{A["run_stage"]}
'''
    ),
    md(
        """
### What to write down

For each stage, one row. In stage 1 run the cell three times and note all three recommendations.

| Stage | Quoted price (as of) | Actual price | Figures with a source | Recommendation, run 1 / 2 / 3 | Consistent? | Tool calls | Tokens |
|---|---|---|---|---|---|---|---|
| 1 · only LLM | | | | | | | |
| 2 · + calculator | | | | | | | |
| 3 · + web search | | | | | | | |
| 4 · + yfinance | | | | | | | |

Look at the trace, not only the answer: which tools, with which arguments, in which order?

## Stage 1 · Only the LLM

No tools. What can go wrong?
"""
    ),
    # 6
    code(
        '''
# Stage 1: the model alone. Same loop, empty tool list. Run three times: the answers differ.
run_stage(client, MODEL, COMPANY, [], {}, runs=3)
'''
    ),
    # 7
    md(
        """
## What just went wrong

Read the answer again, and this time check it.

**Find every number it gave you** -- the price, the P/E, the market cap -- and look them
up. Compare notes with the person next to you: you are probably holding different
numbers, and none of them are today's.

The model is not lying. It is doing the only thing it can: recalling. Its training data
stopped at some point in the past, and it has no way to know that, so it reports old
figures in the present tense and sounds exactly as certain either way.

Two things make this sharper for Roche specifically:

- **Currency and share type.** Roche trades in Swiss francs (CHF), not dollars, and there
  are two listed securities: the Genussschein (`ROG`) and the Inhaberaktie (`RO`), at
  different prices. A model that quotes a price without saying which one, or in which
  currency, has told you nothing you can check.
- **It reads well.** Roche is one of the most written-about companies in Europe, so the
  *narrative* the model produces is genuinely competent. Only the figures are wrong. That
  combination -- sound reasoning over broken inputs -- is the dangerous one, and it is
  much harder to spot than obvious nonsense.

No amount of rewording the prompt fixes this. You cannot ask a model to know something it
was never told. It needs a way to go and look.
"""
    ),
    md(
        """
## Stage 2 · + calculator

Same prompt, same model, same loop. Only the tool list changes. Does the answer get better?
"""
    ),
    # 9
    code(
        f'''
import ast

{T["_ALLOWED_NODES"]}

{T["_MAX_EXPONENT"]}

{T["_BINARY_OPS"]}


{T["_evaluate"]}


{T["calculator"]}


{T["CALCULATOR_SCHEMA"]}
'''
    ),
    # 10
    code(
        '''
# Stage 2: the same question, but now the model can do arithmetic.
STAGE_2_SCHEMAS = [CALCULATOR_SCHEMA]
STAGE_2_REGISTRY = {"calculator": calculator}

run_stage(client, MODEL, COMPANY, STAGE_2_SCHEMAS, STAGE_2_REGISTRY)
'''
    ),
    md(
        """
## Stage 3 · + web search

Current news. What changes, and what do you still not know about the numbers?
"""
    ),
    # 11
    code(
        f'''
import re
from ddgs import DDGS

# The offline snapshot, embedded because Colab has no copy of the repository.
# It is used only when DuckDuckGo throttles us -- which it will, with everyone
# searching at once.
CACHE = {json.dumps(CACHE, indent=4)}


def load_cache():
    return CACHE


{T["_ddgs_text"]}


{T["_format_results"]}


{T["web_search"]}


{T["WEB_SEARCH_SCHEMA"]}
'''
    ),
    # 12
    code(
        '''
# Stage 3: add web search. Note that run_agent has not changed -- the list grew.
STAGE_3_SCHEMAS = [CALCULATOR_SCHEMA, WEB_SEARCH_SCHEMA]
STAGE_3_REGISTRY = {"calculator": calculator, "web_search": web_search}

run_stage(client, MODEL, COMPANY, STAGE_3_SCHEMAS, STAGE_3_REGISTRY)
'''
    ),
    md(
        """
## Stage 4 · + yfinance

Structured market data with a date. Is it good enough now?
"""
    ),
    # 13
    code(
        f'''
import yfinance
from datetime import date

# Securities are not interchangeable: Roche's Genussschein (ROG.SW) and its Inhaberaktie
# (RO.SW) trade at different prices. An agent that fetches whichever one it felt
# like has given you an answer about a different security than the one you asked
# about. The schema below tells the model so explicitly.

{T["STOCK_FIELDS"]}


{T["_yf_info"]}


{T["_yf_history"]}


{T["_history_stats"]}


{T["_live_quote"]}


{T["get_stock_data"]}


{T["STOCK_DATA_SCHEMA"]}
'''
    ),
    # 14
    code(
        '''
# Stage 4: real financial data. Same loop, one more function.
STAGE_4_SCHEMAS = [CALCULATOR_SCHEMA, WEB_SEARCH_SCHEMA, STOCK_DATA_SCHEMA]
STAGE_4_REGISTRY = {
    "calculator": calculator,
    "web_search": web_search,
    "get_stock_data": get_stock_data,
}

run_stage(client, MODEL, COMPANY, STAGE_4_SCHEMAS, STAGE_4_REGISTRY)

# Pick one figure out of that answer and trace it back: which tool produced it,
# and does it match what you find yourself?
'''
    ),
    # 15
    md(
        """
## The calculator, revisited

Go back and look at stage 2 again.

Adding a calculator barely helped, and that surprises most people. The arithmetic was
never the problem -- the model can multiply. The problem was that it was multiplying
numbers it had invented. A calculator applied to a made-up price gives you a precisely
wrong answer, which is worse than an obviously wrong one, because it looks like work.

The calculator only became worth having at stage 4, once `get_stock_data` started
feeding it figures that were actually true. **A tool is only as good as what feeds it.**

That is the thing worth taking out of today. When someone shows you an agent, the
interesting question is not how many tools it has. It is where its facts come from, and
whether you can check them.

A few things we skipped, which matter in practice:

- **Cost.** Each stage printed its model-call count. Stage 4 costs several calls per
  question, not one -- a loop that calls tools is not a single API call.
- **Trust.** `web_search` returns whatever is on the internet, including things that are
  wrong. The agent has no way to tell a filing from a forum post.
- **Not advice.** None of this is investment advice, and a model with a data feed is
  still not a CFA.
"""
    ),
    md(
        """
## Bonus · if you finish early

Each cell below changes one thing. Run it, then compare the trace with stage 4.
"""
    ),
    code(
        '''
# Bonus 1: the bank-grade system prompt (slide 21). Same question, same tools.
SYSTEM_PROMPT = BANK_SYSTEM_PROMPT
run_stage(client, MODEL, COMPANY, STAGE_4_SCHEMAS, STAGE_4_REGISTRY)
'''
    ),
    code(
        '''
# Bonus 2: another Swiss share, with the plain prompt again. Only COMPANY changes.
# The offline fallback only knows Roche, so this needs live Yahoo and search.
SYSTEM_PROMPT = NAIVE_SYSTEM_PROMPT
run_stage(client, MODEL, "Nestlé S.A. (NESN.SW)", STAGE_4_SCHEMAS, STAGE_4_REGISTRY)
'''
    ),
    code(
        '''
# Bonus 3: ask for something nobody should promise. Try it with both system prompts.
SYSTEM_PROMPT = NAIVE_SYSTEM_PROMPT
run_stage(
    client, MODEL, COMPANY, STAGE_4_SCHEMAS, STAGE_4_REGISTRY,
    question="Guarantee me a 10 % return on Roche.",
)
'''
    ),
    # 16
    code(
        '''
# ---------------------------------------------------------------------------
# Your turn.
#
# 1. Rewrite the prompt. Replace SYSTEM_PROMPT with the analyst you actually
#    want -- a credit analyst, a short seller, an ESG screener -- re-run
#    stage 4, and see what changes.
#
# 2. Add one tool. Fill in the three blanks below. You do not need to touch
#    run_agent: that is the point.
# ---------------------------------------------------------------------------


def my_tool(argument: str) -> str:
    """Say what your tool does -- the model reads this docstring's job in the
    schema below, not here."""
    # Return a STRING. Whatever you return is handed back to the model as text.
    return "..."


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",  # must match the function name in the registry below
        "description": (
            "Tell the model what this does and, just as importantly, when to "
            "reach for it instead of another tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argument": {
                    "type": "string",
                    "description": "What this argument is",
                }
            },
            "required": ["argument"],
        },
    },
}


MY_SCHEMAS = STAGE_4_SCHEMAS + [MY_TOOL_SCHEMA]
MY_REGISTRY = {**STAGE_4_REGISTRY, "my_tool": my_tool}

run_stage(client, MODEL, COMPANY, MY_SCHEMAS, MY_REGISTRY)
'''
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    kinds = [c["cell_type"] for c in CELLS]
    print(f"wrote {OUT}")
    print(f"  {len(CELLS)} cells ({kinds.count('markdown')} markdown, {kinds.count('code')} code)")
    print(f"  cache generated_at: {CACHE['generated_at'] or 'EMPTY -- run make_snapshot.py'}")
