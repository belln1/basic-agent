import pytest

from cfa_agent import tools
from cfa_agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_stock_data

INFO = {
    "currentPrice": 250.0,
    "currency": "CHF",
    "trailingPE": 18.5,
    "forwardPE": 16.0,
    "marketCap": 200_000_000_000,
    "dividendRate": 9.6,
}
CLOSES = [200.0, 220.0, 210.0, 250.0]


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: dict(INFO))
    monkeypatch.setattr("cfa_agent.tools._yf_history", lambda t: list(CLOSES))


def _down(monkeypatch):
    def boom(t):
        raise RuntimeError("Yahoo 404")

    monkeypatch.setattr("cfa_agent.tools._yf_info", boom)


def _cache(monkeypatch, ticker="ROG.SW", fundamentals=None, generated_at="2026-11-09"):
    monkeypatch.setattr(
        "cfa_agent.tools.load_cache",
        lambda: {
            "ticker": ticker,
            "generated_at": generated_at,
            "fundamentals": fundamentals if fundamentals is not None else {"currentPrice": 279.84},
            "search": [],
        },
    )


def test_live_output_has_the_figures_slide_24_promises(live):
    out = get_stock_data("ROG.SW")
    assert out.startswith("ROG.SW stock data")
    assert "as of" in out and "CHF" in out
    assert "currentPrice: 250.0" in out
    assert "trailingPE: 18.5" in out
    # 9.6 / 250 = 3.84 %: computed in code from rate and price, so the unit is unambiguous
    assert "dividendYieldPct: 3.84" in out
    # (250 - 200) / 200 = 25 %
    assert "oneYearReturnPct: 25.0" in out
    assert "annualisedVolatilityPct" in out


def test_history_failure_still_returns_the_quote(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: dict(INFO))

    def boom(t):
        raise RuntimeError("no history")

    monkeypatch.setattr("cfa_agent.tools._yf_history", boom)
    out = get_stock_data("ROG.SW")
    assert "currentPrice: 250.0" in out and "oneYearReturnPct" not in out


def test_unknown_ticker_is_a_readable_error_not_an_exception(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: {"trailingPegRatio": None})
    monkeypatch.setattr("cfa_agent.tools._yf_history", lambda t: [])
    _cache(monkeypatch)
    out = get_stock_data("Roche")
    assert out.startswith("Error") and "ROG.SW" in out


def test_failure_falls_back_to_cache_for_the_cached_ticker(monkeypatch):
    _down(monkeypatch)
    _cache(monkeypatch)
    out = get_stock_data("rog.sw")  # case-insensitive
    assert "279.84" in out and "2026-11-09" in out


def test_fallback_refuses_to_relabel_another_ticker(monkeypatch):
    _down(monkeypatch)
    _cache(monkeypatch)
    out = get_stock_data("NESN.SW")
    assert "279.84" not in out
    assert "NESN.SW" in out and "no offline snapshot" in out


def test_fallback_refuses_to_relabel_the_other_roche_security(monkeypatch):
    """RO.SW is the Inhaberaktie; serving Genussschein figures under it is the trap."""
    _down(monkeypatch)
    _cache(monkeypatch)
    assert "279.84" not in get_stock_data("RO.SW")


def test_empty_cache_says_so_instead_of_returning_a_bare_header(monkeypatch):
    _down(monkeypatch)
    _cache(monkeypatch, fundamentals={})
    out = get_stock_data("ROG.SW")
    # addressed to a Colab student, who has no copy of tools/
    assert "regenerate" in out and "make_snapshot" not in out


def test_registry_and_schemas_agree():
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOL_REGISTRY)
    assert "get_stock_data" in TOOL_REGISTRY


def test_schema_matches_slide_24():
    schema = tools.STOCK_DATA_SCHEMA["function"]
    assert schema["description"].startswith(
        "Current price, P/E, dividend yield and 1-year return for a stock ticker. "
        "Use for any market figure."
    )
    ticker = schema["parameters"]["properties"]["ticker"]
    assert ticker["examples"] == ["ROG.SW", "AAPL"]
    assert schema["parameters"]["required"] == ["ticker"]
