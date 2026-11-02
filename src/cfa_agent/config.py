"""Where the three configuration values come from.

Students run this on a fresh Colab runtime with a key the lecturer shares, so the
lookup tries the places a value could plausibly live, in order of least typing:

    Colab secret  ->  environment variable  ->  ask

Only the API key is ever asked for. A missing base URL means "plain OpenAI", and a
missing model name means the default below -- neither is worth interrupting a
lecture to type.
"""

import getpass
import importlib
import os
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class CredentialError(Exception):
    """Raised when credentials are present but the provider rejects them."""

    @classmethod
    def unauthorized(cls) -> "CredentialError":
        return cls(
            "The API key was rejected. Check that you pasted the whole key, then "
            "re-enter it: open the key icon in the left sidebar to fix the Colab "
            "secret, or use Runtime -> Restart session and run the cells again."
        )

    @classmethod
    def unknown_model(cls, model: str | None = None) -> "CredentialError":
        return cls(
            f"The provider does not recognise the model {model!r}. Model names get "
            "retired: set a MODEL secret (key icon in the left sidebar) to one your "
            "account has, then use Runtime -> Restart session."
        )

    @classmethod
    def unreachable(cls, base_url: str | None = None) -> "CredentialError":
        return cls(
            f"Could not reach the model provider at {base_url!r}. Check the BASE_URL "
            "secret. Note that an address on a private network -- anything like "
            "192.168.x.x, 10.x.x.x or localhost -- is not reachable from Colab, "
            "because this notebook runs on a Google server rather than on your machine."
        )

    @classmethod
    def rate_limited(cls) -> "CredentialError":
        return cls(
            "The provider is rate-limiting this key -- usually because everyone in "
            "the room is calling it at once. Wait a few seconds and run the cell again."
        )


def _from_colab(name: str) -> str | None:
    """Read a Colab secret, or None when not on Colab / not set / not granted."""
    try:
        userdata = importlib.import_module("google.colab.userdata")
        value = userdata.get(name)
    except Exception:
        return None
    return value or None


def _load_dotenv(path: str = ".env") -> None:
    """Copy `KEY=value` lines from a .env file into the environment, if one exists.

    Colab has no .env file, so this does nothing there -- it is what makes the same
    notebook run locally, where a .env is the usual place to keep these. Values
    already in the environment win, so an explicit export still overrides the file.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _lookup(name: str) -> str | None:
    """Colab secret first, then the environment (which includes any .env file)."""
    return _from_colab(name) or os.getenv(name) or None


def get_credentials() -> tuple[str, str, str]:
    """Return (base_url, api_key, model), asking for the key only if nothing has it.

    Each value is looked for in the same order: a Colab secret, then the
    environment (including a .env file if one is present), then the default
    above. Only the API key has no default, so only the API key can prompt.
    """
    _load_dotenv()
    base_url = _lookup("BASE_URL") or DEFAULT_BASE_URL
    model = _lookup("MODEL") or DEFAULT_MODEL
    api_key = _lookup("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Paste the API key for this session: ").strip()
    return base_url, api_key, model


def make_client(base_url: str, api_key: str):
    """Build the OpenAI client. Imported lazily so tests need no provider package."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)
