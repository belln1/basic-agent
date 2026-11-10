"""The committed snapshot must be about Roche, and must not answer for anyone else."""

from cfa_agent import tools


def test_committed_snapshot_is_for_roche():
    cache = tools.load_cache()
    assert cache["ticker"] == "ROG.SW"
    assert "Roche" in cache["aliases"]


def test_other_swiss_ticker_does_not_get_roche_figures(monkeypatch):
    def boom(ticker):
        raise RuntimeError("Yahoo down")

    monkeypatch.setattr(tools, "_yf_info", boom)
    out = tools.get_fundamentals("NESN.SW")
    assert out.startswith("Error") and "ROG.SW" in out


def test_search_about_another_company_is_not_answered_with_roche_news(monkeypatch):
    def boom(query, max_results):
        raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr(tools, "_ddgs_text", boom)
    assert tools.web_search("Nestlé Q3 sales").startswith("Error")
