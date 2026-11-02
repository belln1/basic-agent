import pytest
from fake_client import FakeClient, content, raises, tool_call

from cfa_agent.agent import SYSTEM_PROMPT, ask_without_tools, run_agent
from cfa_agent.config import CredentialError
from cfa_agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS


def test_no_tools_single_call():
    client = FakeClient([content("Hold.")])
    r = run_agent(client, "m", "Alphabet", [], {})
    assert r.answer == "Hold." and r.model_calls == 1 and r.tool_calls == []


def test_one_tool_round_trip():
    client = FakeClient([tool_call("calculator", {"expression": "2+2"}), content("Buy.")])
    r = run_agent(client, "m", "Alphabet", TOOL_SCHEMAS, TOOL_REGISTRY)
    assert r.answer == "Buy." and r.model_calls == 2 and r.tool_calls == ["calculator"]


def test_tool_result_is_appended_with_matching_id():
    client = FakeClient([tool_call("calculator", {"expression": "2+2"}), content("Buy.")])
    run_agent(client, "m", "Alphabet", TOOL_SCHEMAS, TOOL_REGISTRY)
    tool_msg = client.messages_seen[-1][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == client.issued_ids[0]
    assert tool_msg["content"] == "4"


def test_unknown_tool_is_reported_not_raised():
    # Review Focus #1
    client = FakeClient([tool_call("nope", {}), content("Recovered.")])
    r = run_agent(client, "m", "Alphabet", TOOL_SCHEMAS, TOOL_REGISTRY)
    assert r.answer == "Recovered."
    assert "nope" in client.messages_seen[-1][-1]["content"]


def test_tool_exception_becomes_a_tool_result():
    # Review Focus #2
    def angry(**_):
        raise ValueError("boom")

    client = FakeClient([tool_call("angry", {}), content("Noted.")])
    r = run_agent(client, "m", "Alphabet", TOOL_SCHEMAS, {"angry": angry})
    assert r.answer == "Noted." and "boom" in client.messages_seen[-1][-1]["content"]


def test_iteration_cap_stops_and_flags():
    # Review Focus #3
    client = FakeClient([tool_call("calculator", {"expression": "1+1"})] * 10)
    r = run_agent(client, "m", "Alphabet", TOOL_SCHEMAS, TOOL_REGISTRY, max_rounds=3)
    assert r.stopped_early and r.model_calls == 3 and r.answer != ""


def test_system_prompt_is_verbatim():
    assert SYSTEM_PROMPT == (
        "You are a CFA expert and have to give a buy/hold/sell recommendation about a company."
    )


def test_user_message_asks_for_figures():
    client = FakeClient([content("Hold.")])
    run_agent(client, "m", "Alphabet", [], {})
    user = client.messages_seen[0][1]["content"]
    assert "Alphabet" in user and "figures" in user.lower()


def test_rejected_key_gives_a_readable_message_not_a_traceback():
    # Final review: Review Focus #5 was tested but never wired into the call path.
    class Unauthorized(Exception):
        status_code = 401

    client = FakeClient([raises(Unauthorized("401 Incorrect API key provided"))])
    with pytest.raises(CredentialError) as excinfo:
        run_agent(client, "m", "Alphabet", [], {})
    assert "API key" in str(excinfo.value) and "Runtime" in str(excinfo.value)


def test_rejected_key_is_readable_in_stage_one_too():
    class Unauthorized(Exception):
        status_code = 401

    client = FakeClient([raises(Unauthorized("401 Incorrect API key provided"))])
    with pytest.raises(CredentialError):
        ask_without_tools(client, "m", "Alphabet")


def test_retired_model_gives_a_readable_message():
    class NotFound(Exception):
        status_code = 404

    client = FakeClient([raises(NotFound("404 model not found"))])
    with pytest.raises(CredentialError) as excinfo:
        run_agent(client, "gpt-retired", "Alphabet", [], {})
    assert "gpt-retired" in str(excinfo.value) and "MODEL" in str(excinfo.value)


def test_rate_limit_gives_a_readable_message():
    class TooMany(Exception):
        status_code = 429

    client = FakeClient([raises(TooMany("429 rate limit"))])
    with pytest.raises(CredentialError) as excinfo:
        run_agent(client, "m", "Alphabet", [], {})
    assert "rate-limiting" in str(excinfo.value)
