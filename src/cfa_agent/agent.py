"""The agent loop.

This is the whole idea of an agent, and it is shorter than most people expect. The
model is called in a loop. Each time, it either answers -- in which case we are done
-- or it asks to call a tool, in which case we run the tool, hand the result back,
and call it again.

Nothing else changes between stage 2 and stage 4 of the lecture. Adding a capability
means adding a function to the registry, never editing this file.
"""

import json
from dataclasses import dataclass, field

from cfa_agent.config import CredentialError

# Slide 51's experiment prompt is SYSTEM_PROMPT plus _user_message, word for word. It
# asks for nothing about sources or figures: stage 1 has to be free to go wrong.
SYSTEM_PROMPT = "You are a stock analyst."


def _user_message(company: str) -> str:
    """What we actually ask for. Kept apart from SYSTEM_PROMPT, which is fixed."""
    return f"Give a buy/hold/sell recommendation for {company} with a short justification."


# Slide 21, word for word: the system prompt as a job description. The lab bonus swaps
# it in for SYSTEM_PROMPT. It forbids the very recommendation the user message asks
# for, which is the point of slide 55.
BANK_SYSTEM_PROMPT = """ROLE
You are a research assistant at a private bank. You prepare
stock briefings for client advisors, not for clients.

GOAL
Answer the advisor's question in a one-page briefing.

TOOLS
Use web_search for news, get_stock_data for prices and
dividends, calculator for every number you compute.

RULES
- Never calculate in your head. Always use the calculator.
- Cite the source and date of every figure.
- If data is missing or older than 7 days, say so.
- Do not give a buy/sell recommendation.

OUTPUT
Summary (3 sentences) · Key figures (table) · Risks · Sources"""


@dataclass
class AgentResult:
    answer: str
    model_calls: int
    tool_calls: list[str]
    stopped_early: bool
    # One entry per tool call: {"round", "tool", "arguments", "result"}.
    trace: list[dict] = field(default_factory=list)
    total_tokens: int = 0


def _base_url_of(client) -> str | None:
    """Best-effort read of the client's configured base URL, for error messages."""
    return str(getattr(client, "base_url", "") or "") or None


def _create(client, **kwargs):
    """Call the model, turning a rejected key into something a student can act on.

    The provider raises a 401 as a long traceback naming headers and request ids.
    A finance student reading that has no idea the fix is "re-paste the key", so
    we translate it once, here, where every model call passes through.
    """
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 401 or "authentication" in type(exc).__name__.lower():
            raise CredentialError.unauthorized() from exc
        if status == 404:
            raise CredentialError.unknown_model(kwargs.get("model")) from exc
        if status == 429:
            raise CredentialError.rate_limited() from exc
        name = type(exc).__name__.lower()
        if "connection" in name or "timeout" in name:
            raise CredentialError.unreachable(kwargs.get("base_url") or _base_url_of(client)) from exc
        raise


def ask_without_tools(client, model: str, company: str) -> str:
    """Stage 1: one call, no tools, no loop. Whatever comes back is all we get."""
    response = _create(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(company)},
        ],
    )
    return response.choices[0].message.content


def _invoke(call, registry: dict) -> str:
    """Run one tool call and return its result as text. Never raises.

    A tool that raised would end the conversation instead of informing it, so every
    failure here becomes a message the model can read and react to -- which is what
    lets a student's half-finished tool fail without killing the cell.
    """
    name = call.function.name
    function = registry.get(name)
    if function is None:
        return f"Error: unknown tool {name}"
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: could not read the arguments for {name}: {exc}"
    try:
        return str(function(**arguments))
    except Exception as exc:
        return f"Error: {exc}"


def _arguments_of(call):
    """The arguments the model sent, as a dict -- or the raw text if it is not JSON."""
    raw = call.function.arguments or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def format_trace(result: AgentResult) -> str:
    """One line per tool call, in order: `1. get_stock_data(ticker='ROG.SW')`."""
    lines = []
    for number, step in enumerate(result.trace, start=1):
        args = step["arguments"]
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items()) if isinstance(args, dict) else args
        lines.append(f"{number}. {step['tool']}({shown})")
    return "\n".join(lines)


def run_agent(
    client,
    model: str,
    company: str,
    schemas: list[dict],
    registry: dict,
    max_rounds: int = 8,
    question: str | None = None,
) -> AgentResult:
    """Stages 2-4: call the model, run any tool it asks for, repeat until it answers."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question or _user_message(company)},
    ]
    model_calls = 0
    total_tokens = 0
    called: list[str] = []
    trace: list[dict] = []

    for _ in range(max_rounds):
        # Leave the key out entirely when there are no tools: some providers reject
        # "tools": null.
        extra = {"tools": schemas} if schemas else {}
        response = _create(client, model=model, messages=messages, **extra)
        model_calls += 1
        total_tokens += getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return AgentResult(
                message.content, model_calls, called, False, trace, total_tokens
            )

        for call in message.tool_calls:
            called.append(call.function.name)
            result = _invoke(call, registry)
            trace.append(
                {
                    "round": model_calls,
                    "tool": call.function.name,
                    "arguments": _arguments_of(call),
                    "result": result,
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    # The model kept asking for tools and never settled. Stop rather than spend the
    # shared key on a loop that is not converging. The default of 8 (slide 37) is a
    # deliberate cap; in one live run a search-happy model used 6 rounds on stage 3 alone and was
    # cut off mid-answer.
    last = next(
        (m.content for m in reversed(messages) if getattr(m, "content", None)),
        None,
    )
    return AgentResult(
        last or f"Stopped after {max_rounds} rounds without a final answer.",
        model_calls,
        called,
        True,
        trace,
        total_tokens,
    )


def run_stage(
    client,
    model: str,
    company: str,
    schemas: list[dict],
    registry: dict,
    runs: int = 1,
    question: str | None = None,
):
    """Run one stage `runs` times, each from a clean start, and print answer and trace."""
    results = []
    for number in range(1, runs + 1):
        if runs > 1:
            print(f"--- Run {number} of {runs} ---")
        result = run_agent(client, model, company, schemas, registry, question=question)
        print(result.answer)
        print("\nTool calls, in order:")
        print(format_trace(result) or "(none)")
        print(f"\n--- {result.model_calls} model calls, {result.total_tokens} tokens")
        if result.stopped_early:
            print("--- NOTE: hit the round limit, so the answer above is incomplete.")
        results.append(result)
    return results
