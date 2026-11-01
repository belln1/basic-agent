"""A scripted stand-in for the OpenAI client.

Build one with a list of canned responses and it hands them out in order, recording
every `messages` payload it was given. That record is how the tests check what the
loop actually sent back to the model.
"""

import json


def content(text: str):
    """Script a plain text response, with no tool calls."""
    return ("content", text)


def tool_call(name: str, arguments: dict):
    """Script a response in which the model asks for one tool."""
    return ("tools", [(name, arguments)])


def raises(exc: Exception):
    """Script the provider rejecting the call."""
    return ("raise", exc)


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, json.dumps(arguments))


class _FakeMessage:
    def __init__(self, text: str | None, tool_calls: list | None):
        self.role = "assistant"
        self.content = text
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage):
        self.choices = [_FakeChoice(message)]


class _Completions:
    def __init__(self, client: "FakeClient"):
        self._client = client

    def create(self, *, model=None, messages=None, tools=None, **_):
        return self._client._next(model=model, messages=messages, tools=tools)


class _Chat:
    def __init__(self, client: "FakeClient"):
        self.completions = _Completions(client)


class FakeClient:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self._served = 0
        self.messages_seen: list[list] = []
        self.tools_seen: list = []
        self.issued_ids: list[str] = []
        self.chat = _Chat(self)

    def _next(self, *, model, messages, tools):
        if self._served >= len(self._responses):
            raise AssertionError(
                f"FakeClient was called {self._served + 1} times but only "
                f"{len(self._responses)} responses were scripted"
            )
        # Snapshot, not alias: the loop keeps mutating the same list.
        self.messages_seen.append(list(messages))
        self.tools_seen.append(tools)
        kind, payload = self._responses[self._served]
        self._served += 1

        if kind == "raise":
            raise payload

        if kind == "content":
            return _FakeResponse(_FakeMessage(payload, None))

        calls = []
        for name, arguments in payload:
            call_id = f"call_{len(self.issued_ids)}"
            self.issued_ids.append(call_id)
            calls.append(_FakeToolCall(call_id, name, arguments))
        return _FakeResponse(_FakeMessage(None, calls))
