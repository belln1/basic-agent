"""The tools the agent can call.

Each tool is a plain function that takes simple arguments and returns a **string**.
That return type is not an accident: whatever a tool returns is handed straight back
to the model as the text of a message, so a tool that raises would end the
conversation instead of informing it. Every path here returns a string, errors
included.

Next to each function is its schema -- the description the model reads when deciding
whether to call it. The schemas are written out by hand rather than generated,
because writing one is the exercise.
"""

import ast
import json
from pathlib import Path

_CACHE_PATH = Path(__file__).parent / "cache" / "roche.json"


def load_cache() -> dict:
    """The offline snapshot of Roche data, used when a live source is unavailable."""
    return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))


# Arithmetic only. Anything outside this list is rejected rather than evaluated,
# which is why this tool cannot be talked into running `__import__('os')`.
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.UnaryOp,
    ast.BinOp,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)

# `2 ** 10000000` is perfectly good arithmetic and would hang the notebook.
_MAX_EXPONENT = 1000

_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _evaluate(node):
    """Walk the validated tree and compute its value."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and right > _MAX_EXPONENT:
            raise ValueError(f"exponent above {_MAX_EXPONENT} is too large to compute")
        return _BINARY_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "1234.5 * 0.21"."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"Error: {expression!r} is not a valid arithmetic expression."

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return (
                f"Error: {expression!r} contains something that is not arithmetic. "
                "Pass numbers and the operators + - * / // % ** only."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return f"Error: {node.value!r} is not a number."
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return f"Error: {node.value!r} is not a number."

    try:
        return str(_evaluate(tree))
    except ZeroDivisionError:
        return "Error: cannot divide by zero."
    except OverflowError:
        return "Error: that result is too large to compute."
    except ValueError as exc:
        return f"Error: {exc}."


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    """The live DuckDuckGo call. Separated out so tests can stand in for the network."""
    from ddgs import DDGS

    return list(DDGS().text(query, max_results=max_results))


def _format_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('body', '')}\n   {r.get('href', '')}")
    return "\n".join(lines)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results as text."""
    max_results = max(1, min(int(max_results or 5), 10))
    try:
        results = _ddgs_text(query, max_results)
    except Exception:
        results = []
    if results:
        return _format_results(results)

    # DuckDuckGo throttles when a whole classroom searches at once. That is an
    # expected outcome here, not an error: fall back to the cached snapshot and
    # say nothing, so a throttled student gets the same lesson as everyone else.
    #
    # But the snapshot is about ONE company. Serving its headlines for a query
    # about some other company would be inventing news, so the fallback only
    # applies when the query is actually about the company we cached.
    cache = load_cache()
    aliases = cache.get("aliases") or [cache.get("ticker", "")]
    if not any(alias and alias.lower() in query.lower() for alias in aliases):
        return (
            f"Error: the live search failed and there is no offline snapshot for "
            f"{query!r}. The cached snapshot only covers {cache.get('company') or cache.get('ticker')}."
        )
    cached = cache.get("search") or []
    if not cached:
        return (
            "Error: the live search failed and the offline snapshot holds no results. "
            "Ask whoever prepared this notebook to regenerate it."
        )
    return _format_results(cached[:max_results])


# The figures worth putting in front of the model. Everything yfinance returns is
# far more than it needs, and a wall of 150 keys buries the eight that matter.
FUNDAMENTAL_FIELDS = [
    "currentPrice",
    "trailingPE",
    "forwardPE",
    "marketCap",
    "returnOnEquity",
    "profitMargins",
    "totalRevenue",
    "freeCashflow",
]


def _yf_info(ticker: str) -> dict:
    """The live Yahoo call. Separated out so tests can stand in for the network."""
    import yfinance

    return yfinance.Ticker(ticker).info


def get_fundamentals(ticker: str) -> str:
    """Look up the headline financials for a ticker, e.g. "ROG.SW"."""
    try:
        info = _yf_info(ticker)
    except Exception:
        # yfinance reads an undocumented endpoint that breaks without warning. The
        # cached snapshot keeps the lecture running; the student sees no difference.
        #
        # The snapshot covers ONE ticker, though. Filling an AAPL heading with the
        # cached Roche figures would attribute real numbers to the wrong security --
        # and ROG.SW vs RO.SW is the very share-class trap this lecture teaches. So the
        # fallback applies only when the ticker asked for is the one we cached.
        cache = load_cache()
        if ticker.strip().upper() != str(cache.get("ticker", "")).strip().upper():
            return (
                f"Error: the live lookup for {ticker} failed and there is no offline "
                f"snapshot for it (the snapshot covers {cache.get('ticker')} only). "
                f"Do not substitute figures from another ticker."
            )
        info = cache.get("fundamentals") or {}
    lines = [f"{ticker} fundamentals:"]
    for field in FUNDAMENTAL_FIELDS:
        if info.get(field) is not None:
            lines.append(f"{field}: {info[field]}")
    if len(lines) == 1:
        # Yahoo failed and the snapshot was never generated. Say so rather than
        # hand back a bare heading: a silent empty result makes stage 4 look like
        # stage 1, and nobody in the room would know why.
        lines.append(
            "no data available -- the live lookup failed and the offline snapshot is "
            "empty. Ask whoever prepared this notebook to regenerate it."
        )
    return "\n".join(lines)


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate one arithmetic expression and return the result. "
            "Use this for any ratio, growth rate, or multiple you need to compute "
            "rather than doing the arithmetic yourself. Pass numbers and the "
            "operators + - * / // % ** only -- no variable names, no function calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic to evaluate, e.g. '(180 - 150) / 150 * 100'",
                }
            },
            "required": ["expression"],
        },
    },
}

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and return the top results as text. Use this for current "
            "news, recent events, and analyst commentary -- anything that happened "
            "after your training data ends. Results are prose, not figures: use "
            "get_fundamentals when you need numbers you can compute with."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'Roche Q3 sales analyst reaction'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return. Defaults to 5.",
                },
            },
            "required": ["query"],
        },
    },
}

FUNDAMENTALS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_fundamentals",
        "description": (
            "Look up current headline financials for a stock ticker: price, trailing "
            "and forward P/E, market cap, return on equity, profit margin, revenue "
            "and free cash flow. Use this whenever you need a figure you intend to "
            "reason or compute with, instead of recalling one. "
            "Pass the ticker exactly as given to you and do not substitute a related "
            "one: ROG.SW (Roche Genussschein) and RO.SW (Roche Inhaberaktie) are "
            "different securities of the same company, with different prices, and "
            "they are not interchangeable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The exact ticker symbol, e.g. 'ROG.SW'",
                }
            },
            "required": ["ticker"],
        },
    },
}

# Declared in the order the lecture introduces them. Stage 2 uses the first entry,
# stage 3 the first two, stage 4 all three -- adding a capability means appending
# one function and one schema here, and nothing else changes.
TOOL_SCHEMAS = [
    CALCULATOR_SCHEMA,
    WEB_SEARCH_SCHEMA,
    FUNDAMENTALS_SCHEMA,
]

TOOL_REGISTRY = {
    "calculator": calculator,
    "web_search": web_search,
    "get_fundamentals": get_fundamentals,
}
