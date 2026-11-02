import sys
import types

from cfa_agent.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    CredentialError,
    get_credentials,
)


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


def fake_colab(monkeypatch, secrets: dict):
    """Install a stand-in for google.colab.userdata holding the given secrets."""
    userdata = types.ModuleType("google.colab.userdata")
    userdata.get = lambda name: secrets[name]  # raises KeyError when absent, as Colab does
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)


def test_colab_secret_wins_over_env(monkeypatch):
    fake_colab(monkeypatch, {"OPENAI_API_KEY": "sk-from-colab"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert get_credentials()[1] == "sk-from-colab"


def test_base_url_and_model_come_from_colab_secrets(monkeypatch):
    fake_colab(monkeypatch, {
        "BASE_URL": "https://gateway.colab/v1",
        "MODEL": "model-from-colab",
        "OPENAI_API_KEY": "sk-from-colab",
    })
    monkeypatch.setenv("BASE_URL", "https://gateway.env/v1")
    monkeypatch.setenv("MODEL", "model-from-env")
    base_url, _, model = get_credentials()
    assert base_url == "https://gateway.colab/v1"
    assert model == "model-from-colab"


def test_dotenv_supplies_values_when_colab_has_none(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "BASE_URL=https://gateway.dotenv/v1",
                'MODEL="model-from-dotenv"',
                "# a comment",
                "OPENAI_API_KEY=sk-from-dotenv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("BASE_URL", "MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    base_url, api_key, model = get_credentials()
    assert base_url == "https://gateway.dotenv/v1"
    assert model == "model-from-dotenv"
    assert api_key == "sk-from-dotenv"


def test_colab_secret_beats_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MODEL=model-from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("MODEL", raising=False)
    fake_colab(monkeypatch, {"MODEL": "model-from-colab", "OPENAI_API_KEY": "sk"})
    assert get_credentials()[2] == "model-from-colab"


def test_defaults_apply_when_nothing_supplies_them(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("BASE_URL", "MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    base_url, _, model = get_credentials()
    assert base_url == DEFAULT_BASE_URL and model == DEFAULT_MODEL


def test_prompts_when_nothing_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("getpass.getpass", lambda *_: "sk-typed")
    assert get_credentials()[1] == "sk-typed"


def test_unauthorized_error_is_readable():
    # Review Focus #5
    msg = str(CredentialError.unauthorized())
    assert "API key" in msg and "Runtime" in msg
