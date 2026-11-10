from cfa_agent.tools import load_cache, web_search


def test_formats_live_results(monkeypatch):
    monkeypatch.setattr(
        "cfa_agent.tools._ddgs_text",
        lambda q, n: [
            {"title": "Alphabet beats estimates", "body": "Revenue up", "href": "https://x.test/1"},
        ],
    )
    out = web_search("GOOGL earnings")
    assert "Alphabet beats estimates" in out and "https://x.test/1" in out


def test_throttle_falls_back_to_cache(monkeypatch):
    def boom(q, n):
        raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr("cfa_agent.tools._ddgs_text", boom)
    out = web_search("Roche Q3 sales")
    assert out.strip() != "" and "Error" not in out


def test_empty_results_fall_back_to_cache(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._ddgs_text", lambda q, n: [])
    assert web_search("Roche Q3 sales").strip() != ""


def test_cache_ships_with_search_results():
    assert load_cache()["search"]


CACHE = {
    "ticker": "GOOGL",
    "company": "Alphabet",
    "aliases": ["GOOGL", "GOOG", "Alphabet", "Google"],
    "fundamentals": {},
    "search": [{"title": "Alphabet beats estimates", "body": "x", "href": "https://x/1"}],
}


def test_fallback_refuses_unrelated_queries(monkeypatch):
    """Review: GOOGL headlines must not be returned for a query about Apple."""
    monkeypatch.setattr("cfa_agent.tools._ddgs_text", lambda q, n: [])
    monkeypatch.setattr("cfa_agent.tools.load_cache", lambda: CACHE)
    out = web_search("Apple Q4 earnings")
    # the cached headline must not appear; naming the cached company is fine
    assert "Alphabet beats estimates" not in out
    assert "no offline snapshot" in out


def test_fallback_serves_queries_about_the_cached_company(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._ddgs_text", lambda q, n: [])
    monkeypatch.setattr("cfa_agent.tools.load_cache", lambda: CACHE)
    out = web_search("alphabet quarterly results")
    assert "Alphabet beats estimates" in out and "snapshot" not in out


def test_empty_cached_search_says_so_instead_of_returning_nothing(monkeypatch):
    """Review: get_fundamentals got this guard; its sibling silently returned ''."""
    monkeypatch.setattr("cfa_agent.tools._ddgs_text", lambda q, n: [])
    monkeypatch.setattr("cfa_agent.tools.load_cache", lambda: {**CACHE, "search": []})
    out = web_search("Alphabet earnings")
    assert out.strip() != "" and "no results" in out


def test_zero_max_results_is_clamped(monkeypatch):
    """max_results is model-supplied; 0 must not produce an empty tool result."""
    monkeypatch.setattr(
        "cfa_agent.tools._ddgs_text", lambda q, n: [{"title": "t", "body": "b", "href": "h"}] * n
    )
    monkeypatch.setattr("cfa_agent.tools.load_cache", lambda: CACHE)
    assert web_search("Alphabet", max_results=0).strip() != ""
