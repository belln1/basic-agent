import json

from make_snapshot import build_snapshot


def test_writes_all_sections(tmp_path, monkeypatch):
    monkeypatch.setattr("cfa_agent.tools._yf_info", lambda t: {"currentPrice": 190.0})
    monkeypatch.setattr(
        "cfa_agent.tools._ddgs_text", lambda q, n: [{"title": "t", "body": "b", "href": "h"}]
    )
    build_snapshot(tmp_path / "googl.json")
    data = json.loads((tmp_path / "googl.json").read_text())
    assert data["ticker"] == "GOOGL"
    assert data["fundamentals"]["currentPrice"] == 190.0
    assert data["search"] and data["generated_at"]
