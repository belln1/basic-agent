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
from datetime import date
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
# far more than it needs, and a wall of 150 keys buries the few that matter.
STOCK_FIELDS = [
    "currency",
    "currentPrice",
    "trailingPE",
    "forwardPE",
    "marketCap",
]


def _yf_info(ticker: str) -> dict:
    """The live Yahoo call. Separated out so tests can stand in for the network."""
    import yfinance

    return yfinance.Ticker(ticker).info


def _yf_history(ticker: str) -> list[float]:
    """One year of daily closing prices, oldest first. Separate for the same reason."""
    import yfinance

    return [float(c) for c in yfinance.Ticker(ticker).history(period="1y")["Close"].tolist()]


def _history_stats(closes: list[float]) -> dict:
    """1-year return and annualised volatility, computed here rather than by the model."""
    if len(closes) < 3 or closes[0] <= 0:
        return {}
    daily = [b / a - 1 for a, b in zip(closes, closes[1:])]
    mean = sum(daily) / len(daily)
    variance = sum((r - mean) ** 2 for r in daily) / (len(daily) - 1)
    return {
        "oneYearReturnPct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "annualisedVolatilityPct": round(variance**0.5 * 252**0.5 * 100, 2),
    }


def _live_quote(ticker: str) -> dict:
    """Fetch the figures for one ticker. Raises if Yahoo has no price for it."""
    info = _yf_info(ticker)
    price = info.get("currentPrice")
    if price is None:
        raise ValueError(f"no price found for {ticker!r}")
    quote = {f: info[f] for f in STOCK_FIELDS if info.get(f) is not None}
    # Yahoo reports dividendYield as a fraction in some versions and a percentage in
    # others. The annual dividend per share divided by the price has no such ambiguity.
    if info.get("dividendRate") is not None and price:
        quote["dividendYieldPct"] = round(info["dividendRate"] / price * 100, 2)
    try:
        quote.update(_history_stats(_yf_history(ticker)))
    except Exception:
        pass  # a quote without history is still worth returning
    return quote


def get_stock_data(ticker: str) -> str:
    """Look up price, P/E, dividend yield, 1-year return and volatility, e.g. "ROG.SW"."""
    try:
        quote = _live_quote(ticker)
        as_of = date.today().isoformat()
    except Exception:
        # yfinance reads an undocumented endpoint that breaks without warning. The
        # cached snapshot keeps the lecture running; the student sees no difference.
        #
        # The snapshot covers ONE ticker, though. Filling an NESN.SW heading with the
        # cached Roche figures would attribute real numbers to the wrong security --
        # and ROG.SW vs RO.SW is the very share-class trap this lecture teaches. So the
        # fallback applies only when the ticker asked for is the one we cached.
        cache = load_cache()
        if ticker.strip().upper() != str(cache.get("ticker", "")).strip().upper():
            return (
                f"Error: the live lookup for {ticker!r} failed and there is no offline "
                f"snapshot for it (the snapshot covers {cache.get('ticker')} only). "
                f"Check the ticker includes its exchange, e.g. 'ROG.SW'. "
                f"Do not substitute figures from another ticker."
            )
        quote = cache.get("fundamentals") or {}
        as_of = cache.get("generated_at") or "unknown"
    lines = [f"{ticker} stock data (as of {as_of}):"]
    for field, value in quote.items():
        lines.append(f"{field}: {value}")
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
            "get_stock_data when you need numbers you can compute with."
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

STOCK_DATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_stock_data",
        "description": (
            "Current price, P/E, dividend yield and 1-year return for a stock ticker. "
            "Use for any market figure. Also returns annualised volatility, the "
            "currency and the date of the data. Pass the ticker exactly as given "
            "to you and do not substitute a related one: ROG.SW (Roche "
            "Genussschein) and RO.SW (Roche Inhaberaktie) are different securities "
            "with different prices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock symbol, optional with exchange",
                    "examples": ["ROG.SW", "AAPL"],
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
    STOCK_DATA_SCHEMA,
]

TOOL_REGISTRY = {
    "calculator": calculator,
    "web_search": web_search,
    "get_stock_data": get_stock_data,
}
