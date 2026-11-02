# basic-agent

Teaching material for a one-day lecture, **Agentic AI in Banking & Finance**: a Colab
notebook that builds a tool-using LLM agent in four stages, where each stage fails in a
way that motivates the next.

**Open in Colab:**
[notebooks/cfa_agent_lecture.ipynb](https://colab.research.google.com/github/belln1/basic-agent/blob/main/notebooks/cfa_agent_lecture.ipynb)

Students need one secret, `OPENAI_API_KEY`, added via the key icon in Colab's left
sidebar. `BASE_URL` and `MODEL` are optional secrets that override the defaults
(`https://api.openai.com/v1`, `gpt-4o-mini`).

## The notebook is generated — do not hand-edit its code cells

`notebooks/cfa_agent_lecture.ipynb` is a build artifact. Its code cells are lifted
verbatim out of `src/cfa_agent/` by `tools/build_notebook.py`, so what students run is
exactly what the test suite covers.

Change behaviour in `src/cfa_agent/`, then rebuild. `tests/test_notebook.py` fails if the
committed notebook does not match what the builder would produce, so drift cannot ship
quietly. Markdown cells live in the builder itself.

## Refreshing the offline data before a lecture

`src/cfa_agent/cache/googl.json` is the snapshot the notebook falls back to when Yahoo is
down or DuckDuckGo throttles a roomful of students at once. **What is committed here is a
deliberate placeholder** — titles say `PLACEHOLDER`, URLs point at `example.invalid` — so
that a skipped refresh fails visibly instead of quietly feeding students invented figures.

Run both commands, in this order, a day or two before teaching:

```bash
uv run python tools/make_snapshot.py    # fetch live figures + news; refuses to write a partial snapshot
uv run python tools/build_notebook.py   # embed them into the notebook and refresh the pinned versions
uv run --extra dev pytest               # must be green before committing
```

Both are required. Colab has no checkout of this repo, so the snapshot is embedded in the
notebook as a literal — regenerating it without rebuilding changes nothing a student sees.
Check the `currentPrice` **and** `search: N results` lines that `make_snapshot.py` prints
before committing.

## Development

```bash
uv run --extra dev pytest    # the whole suite
```

Dependency versions are pinned in `uv.lock`, and `tools/build_notebook.py` reads them from
there to pin Colab's `pip install` line — so the room installs the versions the suite ran
against. Locally, a `.env` file with `BASE_URL` / `MODEL` / `OPENAI_API_KEY` is picked up
automatically.

MIT licensed. Nothing here is investment advice.
