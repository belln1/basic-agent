import ast
import json
import re
from pathlib import Path

from cfa_agent.agent import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "basic_agent.ipynb"
PACKAGE = ROOT / "src" / "cfa_agent"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def all_source(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _strip_magics(source: str) -> str:
    """Drop Jupyter shell/magic lines (`!pip ...`, `%cd ...`) -- they are not Python."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
    )


def notebook_source():
    nb = notebook()
    return "\n".join(
        _strip_magics("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"
    )


def module_source():
    return "\n".join(
        (PACKAGE / name).read_text(encoding="utf-8")
        for name in ("config.py", "tools.py", "agent.py")
    )


def ast_of(name, source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    raise AssertionError(f"{name} is not defined in this source")


def test_starts_with_a_markdown_introduction():
    assert notebook()["cells"][0]["cell_type"] == "markdown"


def test_stage_markers_present():
    src = all_source(notebook())
    for stage in ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]:
        assert stage in src


def test_inlined_functions_match_module():
    # AST equality, not text equality: formatting may differ, logic may not
    for name in [
        "get_credentials",
        "_from_colab",
        "_load_dotenv",
        "_lookup",
        "calculator",
        "web_search",
        "get_stock_data",
        "_create",
        "run_agent",
        "format_trace",
        "run_stage",
    ]:
        assert ast_of(name, notebook_source()) == ast_of(name, module_source())


def test_notebook_imports_nothing_project_local():
    assert "cfa_agent" not in notebook_source()


def test_system_prompt_verbatim_in_notebook():
    assert SYSTEM_PROMPT in notebook_source()


def test_no_framework_imports():
    src = notebook_source().lower()
    for banned in ["langchain", "llama_index", "agents"]:
        assert not re.search(rf"^\s*(import|from)\s+{banned}\b", src, re.MULTILINE)


def test_notebook_is_valid_python():
    # Every code cell must parse; a broken cell is a dead lecture.
    for cell in notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        ast.parse(_strip_magics("".join(cell["source"])))


def test_notebook_matches_what_the_builder_would_write():
    """The whole file, not a list of names.

    This is the guard that catches the likeliest operator mistake in the pre-lecture
    refresh: regenerating the snapshot and forgetting to rebuild the notebook, which
    leaves students reading stale embedded data while every other test stays green.
    It also subsumes drift in the definitions the name-based AST check never covered
    (schemas, _evaluate, _invoke, the embedded CACHE).
    """
    import build_notebook

    expected = json.dumps(build_notebook.NOTEBOOK, indent=1) + "\n"
    assert NOTEBOOK.read_text(encoding="utf-8") == expected, (
        "notebooks/basic_agent.ipynb is out of sync with its source. "
        "Run: uv run python tools/build_notebook.py"
    )


def test_company_names_the_ticker_the_model_must_use():
    """FUNDAMENTALS_SCHEMA tells the model to pass the ticker 'exactly as given to
    you' -- so the ticker has to actually be given to it. COMPANY is the only
    company string that reaches the prompt."""
    src = notebook_source()
    assert 'COMPANY = "Roche Holding AG (ROG.SW)"' in src


def test_stage_cells_surface_the_round_limit():
    """A capped run prints a truncated answer; without this the student has no idea
    why. Observed live: a search-happy model used all its rounds on stage 3. The
    note lives in run_stage, so every stage cell must go through it."""
    src = notebook_source()
    assert src.count("if result.stopped_early:") == 1
    assert "hit the round limit" in ast_of("run_stage", src) or "hit the round limit" in src
    assert "run_agent(client, MODEL" not in src  # no stage cell bypasses run_stage
    assert src.count("run_stage(") >= 5  # stages 1-4 and the exercise, at least


def test_stage_one_runs_three_times_with_no_tools():
    """Slide 52: stage 1 is run three times to show the recommendation can change."""
    assert "run_stage(client, MODEL, COMPANY, [], {}, runs=3)" in notebook_source()


def test_section_headings_match_the_lab_slide():
    src = all_source(notebook())
    for heading in [
        "## Section 0",
        "## Section 1",
        "## Stage 1",
        "## Stage 2",
        "## Stage 3",
        "## Stage 4",
        "## Bonus",
    ]:
        assert heading in src, heading


def test_fill_in_table_has_the_columns_of_slide_52():
    src = all_source(notebook())
    for column in ["Quoted price", "Actual price", "Figures with a source", "Recommendation", "Tool calls", "Tokens"]:
        assert column in src, column


def test_bonus_cells_cover_the_three_slide_53_tasks():
    src = notebook_source()
    assert "SYSTEM_PROMPT = BANK_SYSTEM_PROMPT" in src
    assert "NESN.SW" in src
    assert 'question="Guarantee me a 10 % return on Roche."' in src
