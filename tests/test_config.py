import sys
import types

from cfa_agent.config import DEFAULT_MODEL, CredentialError, get_credentials


def test_reads_from_env(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    assert get_credentials() == ("https://api.example.com/v1", "sk-test", "gpt-4o-mini")


def test_model_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MODEL", raising=False)
    assert get_credentials()[2] == DEFAULT_MODEL


def test_colab_secret_wins_over_env(monkeypatch):
    fake = types.ModuleType("google.colab.userdata")
    fake.get = lambda name: {"OPENAI_API_KEY": "sk-from-colab"}[name]
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert get_credentials()[1] == "sk-from-colab"


def test_prompts_when_nothing_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda *_: "sk-typed")
    assert get_credentials()[1] == "sk-typed"


def test_unauthorized_error_is_readable():
    # Review Focus #5
    msg = str(CredentialError.unauthorized())
    assert "API key" in msg and "Runtime" in msg
