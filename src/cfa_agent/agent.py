"""The agent loop.

This is the whole idea of an agent, and it is shorter than most people expect. The
model is called in a loop. Each time, it either answers -- in which case we are done
-- or it asks to call a tool, in which case we run the tool, hand the result back,
and call it again.

Nothing else changes between stage 2 and stage 4 of the lecture. Adding a capability
means adding a function to the registry, never editing this file.
"""

import json
from dataclasses import dataclass

from cfa_agent.config import CredentialError

# The lecturer's prompt, word for word. Do not edit it: the exercise is for students
# to rewrite it themselves and see what changes.
SYSTEM_PROMPT = (
    "You are a CFA expert and have to give a buy/hold/sell recommendation about a company."
)


def _user_message(company: str) -> str:
    """What we actually ask for. Kept apart from SYSTEM_PROMPT, which is fixed."""
    return (
        f"Give your buy/hold/sell recommendation for {company}. "
        "Back it with specific, current figures rather than ones you recall, "
        "and list the figures you relied on so they can be checked."
    )


@dataclass
class AgentResult:
    answer: str
    model_calls: int
    tool_calls: list[str]
    stopped_early: bool


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


def run_agent(
    client,
    model: str,
    company: str,
    schemas: list[dict],
    registry: dict,
    max_rounds: int = 6,
) -> AgentResult:
    """Stages 2-4: call the model, run any tool it asks for, repeat until it answers."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(company)},
    ]
    model_calls = 0
    called: list[str] = []

    for _ in range(max_rounds):
        response = _create(
            client,
            model=model,
            messages=messages,
            tools=schemas or None,
        )
        model_calls += 1
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return AgentResult(message.content, model_calls, called, stopped_early=False)

        for call in message.tool_calls:
            called.append(call.function.name)
            result = _invoke(call, registry)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    # The model kept asking for tools and never settled. Stop rather than spend the
    # shared key on a loop that is not converging.
    last = next(
        (m.content for m in reversed(messages) if getattr(m, "content", None)),
        None,
    )
    return AgentResult(
        last or f"Stopped after {max_rounds} rounds without a final answer.",
        model_calls,
        called,
        stopped_early=True,
    )
