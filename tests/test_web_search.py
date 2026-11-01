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
    out = web_search("GOOGL earnings")
    assert out.strip() != "" and "Error" not in out


def test_empty_results_fall_back_to_cache(monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._ddgs_text", lambda q, n: [])
    assert web_search("GOOGL earnings").strip() != ""


def test_cache_ships_with_search_results():
    assert load_cache()["search"]
