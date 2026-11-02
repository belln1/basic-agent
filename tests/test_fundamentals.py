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


def test_live_lookup_reports_the_ticker_it_was_asked_for(monkeypatch):
    # Replaces a tautological assertion: get_fundamentals echoes its own argument,
    # so "GOOG\n" could never have appeared regardless of the implementation. The
    # share class risk is real but lives in the *fallback* path, tested below.
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: {"currentPrice": 1, "symbol": t})
    assert get_fundamentals("GOOGL").startswith("GOOGL fundamentals:")


def test_registry_and_schemas_agree():
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOL_REGISTRY)


def test_empty_cache_says_so_instead_of_returning_a_bare_header(monkeypatch):
    # Final review: the search fallback fails loudly, this one failed silently.
    def boom(t):
        raise RuntimeError("Yahoo 404")

    monkeypatch.setattr("cfa_agent.tools._yf_info", boom)
    monkeypatch.setattr(
        "cfa_agent.tools.load_cache",
        lambda: {"ticker": "GOOGL", "fundamentals": {}, "search": []},
    )
    out = get_fundamentals("GOOGL")
    # addressed to a Colab student, who has no copy of tools/
    assert "regenerate" in out and "make_snapshot" not in out


def test_fallback_refuses_to_relabel_another_ticker(monkeypatch):
    """Review: the snapshot is GOOGL-only; it must never be served as AAPL."""
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(
        "cfa_agent.tools.load_cache",
        lambda: {"ticker": "GOOGL", "fundamentals": {"currentPrice": 279.84}, "search": []},
    )
    out = get_fundamentals("AAPL")
    assert "279.84" not in out
    assert "AAPL" in out and "no offline snapshot" in out


def test_fallback_refuses_to_relabel_the_other_share_class(monkeypatch):
    """GOOG is Class C. Serving Class A figures under it is the trap the lecture teaches."""
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(
        "cfa_agent.tools.load_cache",
        lambda: {"ticker": "GOOGL", "fundamentals": {"currentPrice": 279.84}, "search": []},
    )
    assert "279.84" not in get_fundamentals("GOOG")


def test_fallback_still_silent_for_the_cached_ticker(monkeypatch):
    """Spec requires the GOOGL path to be invisible to a student."""
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(
        "cfa_agent.tools.load_cache",
        lambda: {"ticker": "GOOGL", "fundamentals": {"currentPrice": 279.84}, "search": []},
    )
    out = get_fundamentals("googl")  # case-insensitive
    assert "279.84" in out and "snapshot" not in out
