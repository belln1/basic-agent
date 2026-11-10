"""Execute the notebook the way a student does, with a scripted model.

This is Task 8's offline half. Rather than skipping the cells that would reach the
network, it substitutes a fake `openai` module, so **every** cell runs exactly as
written and in the order the room will run them.

That ordering is the point. A name defined one cell too late is invisible to a
harness that execs all the definitions up front, and it is the first thing a
student hits -- which is exactly how `_create` shipped broken once.

The live-key half -- stages 1 to 4 against a real model -- cannot be done here and
is listed in the pre-lecture checklist.
"""

import json
import sys
import types
from pathlib import Path

from fake_client import FakeClient, content, raises, tool_call

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "basic_agent.ipynb"


def _code_cells():
    """Every code cell, in order, with Jupyter magics stripped (they are not Python)."""
    for number, cell in enumerate(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        body = "\n".join(
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        yield number, body


def notebook_namespace(monkeypatch):
    """Run every cell in order against a scripted provider; return the namespace."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-used")
    monkeypatch.setenv("BASE_URL", "https://api.example.invalid/v1")
    monkeypatch.setenv("MODEL", "test-model")

    # Stand in for the provider, so the stage cells genuinely execute without a key.
    scripted = FakeClient([content("Hold.")] * 40)
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: scripted
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    namespace: dict = {}
    for number, body in _code_cells():
        try:
            exec(compile(body, f"<cell {number}>", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"cell {number} failed: {type(exc).__name__}: {exc}") from exc
    return namespace


def test_every_cell_runs_in_order(monkeypatch):
    # The regression that matters: a forward reference between cells.
    notebook_namespace(monkeypatch)


def test_expected_names_are_defined(monkeypatch):
    ns = notebook_namespace(monkeypatch)
    for name in [
        "CredentialError",
        "get_credentials",
        "_create",
        "run_agent",
        "ask_without_tools",
        "calculator",
        "web_search",
        "get_fundamentals",
        "SYSTEM_PROMPT",
    ]:
        assert name in ns, f"{name} is never defined in the notebook"


def test_search_fallback_holds_when_duckduckgo_fails(monkeypatch):
    # Task 8 Step 4
    ns = notebook_namespace(monkeypatch)

    def boom(query, max_results):
        raise RuntimeError("202 Ratelimit")

    ns["_ddgs_text"] = boom
    out = ns["web_search"]("Roche Q3 sales")
    assert out.strip() != "" and not out.startswith("Error")


def test_fundamentals_fallback_holds_when_yahoo_fails(monkeypatch):
    # Task 8 Step 4
    ns = notebook_namespace(monkeypatch)

    def boom(ticker):
        raise RuntimeError("Yahoo 404")

    ns["_yf_info"] = boom
    out = ns["get_fundamentals"]("ROG.SW")
    assert "ROG.SW" in out and not out.startswith("Error")


def test_rejected_key_is_readable_from_the_notebook(monkeypatch):
    ns = notebook_namespace(monkeypatch)

    class Unauthorized(Exception):
        status_code = 401

    client = FakeClient([raises(Unauthorized("401 Incorrect API key provided"))])
    try:
        ns["run_agent"](client, "test-model", "Alphabet", [], {})
    except ns["CredentialError"] as exc:
        assert "API key" in str(exc) and "Runtime" in str(exc)
    else:
        raise AssertionError("a rejected key did not surface as CredentialError")


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


def test_unreachable_base_url_is_readable_from_the_notebook(monkeypatch):
    """A dropped helper (`_base_url_of`) once turned this into a NameError."""
    ns = notebook_namespace(monkeypatch)

    class APIConnectionError(Exception):
        pass

    client = FakeClient([raises(APIConnectionError("Connection error."))])
    try:
        ns["run_agent"](client, "test-model", "Roche", [], {})
    except ns["CredentialError"] as exc:
        assert "BASE_URL" in str(exc)
    else:
        raise AssertionError("an unreachable provider did not surface as CredentialError")
