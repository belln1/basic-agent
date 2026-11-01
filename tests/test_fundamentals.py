from cfa_agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, get_fundamentals

FIELDS = [
    "currentPrice",
    "trailingPE",
    "forwardPE",
    "marketCap",
    "returnOnEquity",
    "profitMargins",
    "totalRevenue",
    "freeCashflow",
]


def test_formats_live_fundamentals(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: {f: 1 for f in FIELDS})
    out = get_fundamentals("GOOGL")
    assert all(f in out for f in FIELDS)


def test_failure_falls_back_to_cache(monkeypatch):
    def boom(t):
        raise RuntimeError("Yahoo 404")

    monkeypatch.setattr("cfa_agent.tools._yf_info", boom)
    assert "GOOGL" in get_fundamentals("GOOGL")


def test_share_class_is_not_silently_swapped(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: {"currentPrice": 1, "symbol": t})
    assert "GOOG\n" not in get_fundamentals("GOOGL")


def test_registry_and_schemas_agree():
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOL_REGISTRY)
