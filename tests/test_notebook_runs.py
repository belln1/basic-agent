"""Execute the notebook's own code in a namespace, with a scripted model.

This is Task 8's offline half: it proves the cells students run actually execute, that
both offline fallbacks hold when the network fails, and that the cell-16 template adds
a working tool without anyone touching `run_agent`. The live-key half (stages 1-4
against a real model) cannot be done here and is listed in the pre-lecture checklist.
"""

import json
from pathlib import Path

from fake_client import FakeClient, content, tool_call

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "cfa_agent_lecture.ipynb"

# Cells that need a real model or a real client are driven by hand instead.
SKIP_MARKERS = ("OpenAI(base_url", "run_agent(client", "ask_without_tools(client")


def notebook_namespace(monkeypatch):
    """Run every definition cell and hand back the resulting namespace."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-used")
    monkeypatch.setenv("BASE_URL", "https://api.example.invalid/v1")
    monkeypatch.setenv("MODEL", "test-model")

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        body = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        if any(marker in body for marker in SKIP_MARKERS):
            continue
        exec(compile(body, "<notebook>", "exec"), namespace)
    return namespace


def test_definition_cells_execute(monkeypatch):
    ns = notebook_namespace(monkeypatch)
    for name in ["calculator", "web_search", "get_fundamentals", "run_agent", "SYSTEM_PROMPT"]:
        assert name in ns


def test_search_fallback_holds_when_duckduckgo_fails(monkeypatch):
    # Task 8 Step 4
    ns = notebook_namespace(monkeypatch)

    def boom(query, max_results):
        raise RuntimeError("202 Ratelimit")

    ns["_ddgs_text"] = boom
    out = ns["web_search"]("GOOGL earnings")
    assert out.strip() != "" and not out.startswith("Error")


def test_fundamentals_fallback_holds_when_yahoo_fails(monkeypatch):
    # Task 8 Step 4
    ns = notebook_namespace(monkeypatch)

    def boom(ticker):
        raise RuntimeError("Yahoo 404")

    ns["_yf_info"] = boom
    out = ns["get_fundamentals"]("GOOGL")
    assert "GOOGL" in out and not out.startswith("Error")


def test_added_tool_is_called_without_touching_run_agent(monkeypatch):
    # Task 8 Step 5 -- follows the cell-16 template exactly.
    ns = notebook_namespace(monkeypatch)

    def dividend_yield(ticker: str) -> str:
        return f"{ticker} dividend yield: 0.42%"

    my_schema = {
        "type": "function",
        "function": {
            "name": "dividend_yield",
            "description": "Look up a dividend yield.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Ticker"}},
                "required": ["ticker"],
            },
        },
    }

    schemas = [ns["CALCULATOR_SCHEMA"], my_schema]
    registry = {"calculator": ns["calculator"], "dividend_yield": dividend_yield}

    client = FakeClient(
        [tool_call("dividend_yield", {"ticker": "GOOGL"}), content("Hold, on that yield.")]
    )
    result = ns["run_agent"](client, "test-model", "Alphabet Inc. Class A", schemas, registry)

    assert result.tool_calls == ["dividend_yield"]
    assert result.answer == "Hold, on that yield."
    assert client.messages_seen[-1][-1]["content"] == "GOOGL dividend yield: 0.42%"
