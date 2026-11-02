"""Regenerate the offline GOOGL snapshot.

Run this shortly before the lecture:

    uv run python tools/make_snapshot.py

It fetches the figures and news the notebook falls back to when Yahoo is down or
DuckDuckGo throttles a roomful of students at once. A snapshot more than a few days
old ships stale numbers into that fallback, so regenerate rather than trusting
whatever is committed.
"""

import json
from datetime import date
from pathlib import Path

from cfa_agent import tools

TICKER = "GOOGL"
COMPANY = "Alphabet"
# Used to decide whether a cached result may answer a given query at all.
ALIASES = ["GOOGL", "GOOG", "Alphabet", "Google"]

# Enough coverage for the model to situate a recommendation, without being a corpus.
QUERIES = [
    f"{TICKER} Alphabet latest quarterly earnings results",
    f"{TICKER} Alphabet analyst price target buy or sell",
    f"{TICKER} Alphabet recent news",
]

CACHE_PATH = Path(__file__).resolve().parents[1] / "src" / "cfa_agent" / "cache" / "googl.json"


def build_snapshot(path: Path) -> None:
    """Fetch live data for GOOGL and write it to `path` in the cache's shape."""
    info = tools._yf_info(TICKER)
    fundamentals = {
        field: info[field] for field in tools.FUNDAMENTAL_FIELDS if info.get(field) is not None
    }

    search: list[dict] = []
    for query in QUERIES:
        for result in tools._ddgs_text(query, 3):
            search.append(
                {
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "href": result.get("href", ""),
                }
            )

    if not fundamentals or not search:
        # A snapshot with an empty section is worse than none: it is the silent
        # path that degrades stage 3 or 4 without anyone noticing on the day.
        raise SystemExit(
            f"refusing to write a partial snapshot: "
            f"{len(fundamentals)} fundamentals, {len(search)} search results. "
            f"Check the network and run again."
        )

    snapshot = {
        "generated_at": date.today().isoformat(),
        "ticker": TICKER,
        "company": COMPANY,
        "aliases": ALIASES,
        "fundamentals": fundamentals,
        "search": search,
    }
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    # Printed so whoever runs this can sanity-check the numbers before committing.
    print(f"wrote {path}")
    print(f"  generated_at: {snapshot['generated_at']}")
    print(f"  currentPrice: {fundamentals.get('currentPrice', 'MISSING')}")
    print(f"  fundamentals: {len(fundamentals)} fields")
    print(f"  search: {len(search)} results")


if __name__ == "__main__":
    build_snapshot(CACHE_PATH)
