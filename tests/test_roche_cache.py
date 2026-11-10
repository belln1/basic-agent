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
    out = tools.get_stock_data("NESN.SW")
    assert out.startswith("Error") and "ROG.SW" in out


def test_search_about_another_company_is_not_answered_with_roche_news(monkeypatch):
    def boom(query, max_results):
        raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr(tools, "_ddgs_text", boom)
    assert tools.web_search("Nestlé Q3 sales").startswith("Error")


def test_alias_must_match_a_whole_word_not_a_substring(monkeypatch):
    """'ROG' sits inside 'programme' and 'progress'; that is not a query about Roche."""

    def boom(query, max_results):
        raise RuntimeError("202 Ratelimit")

    monkeypatch.setattr(tools, "_ddgs_text", boom)
    assert tools.web_search("Nestle cost savings programme progress").startswith("Error")
    assert not tools.web_search("ROG.SW sales").startswith("Error")
