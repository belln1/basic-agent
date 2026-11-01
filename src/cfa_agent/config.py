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


def _from_colab(name: str) -> str | None:
    """Read a Colab secret, or None when not on Colab / not set / not granted."""
    try:
        userdata = importlib.import_module("google.colab.userdata")
        value = userdata.get(name)
    except Exception:
        return None
    return value or None


def _lookup(name: str) -> str | None:
    """Colab secret first, then the environment."""
    return _from_colab(name) or os.getenv(name) or None


def get_credentials() -> tuple[str, str, str]:
    """Return (base_url, api_key, model), asking for the key only if nothing has it."""
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
