"""API key rotation for Gemini — round-robin across multiple keys.

Usage:
    from utils.key_rotator import get_gemini_client

    client = get_gemini_client()  # rotates automatically
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------------------------
# Load all keys once
# ---------------------------------------------------------------------------

def _load_keys() -> list[str]:
    """Load all GEMINI_API_KEY_N from .env, skip placeholders."""
    # Find and load .env from repo root
    repo_root = Path(__file__).parent.parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    else:
        load_dotenv()

    keys = []
    for i in range(1, 10):
        key = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if key and not key.startswith("PASTE_"):
            keys.append(key)

    # Fallback: also check plain GEMINI_API_KEY
    fallback = os.getenv("GEMINI_API_KEY", "")
    if fallback and not fallback.startswith("PASTE_") and fallback not in keys:
        keys.insert(0, fallback)

    if not keys:
        raise RuntimeError(
            "No GEMINI_API_KEY_N found in .env. "
            "Add at least one key: GEMINI_API_KEY_1=your_key"
        )

    return keys


_KEYS: list[str] = []
_current_index: int = 0
_call_count: int = 0


def _ensure_keys() -> None:
    """Load keys on first use."""
    global _KEYS
    if not _KEYS:
        _KEYS = _load_keys()
        print(f"[KEYS] Loaded {len(_KEYS)} API keys for rotation")


def get_gemini_client() -> genai.Client:
    """Get a Gemini client using the next API key in rotation."""
    global _current_index, _call_count
    _ensure_keys()

    _call_count += 1
    key = _KEYS[_current_index % len(_KEYS)]

    # Rotate to next key for the next call
    _current_index = (_current_index + 1) % len(_KEYS)

    return genai.Client(api_key=key)


def get_gemini_client_with_retry(
    max_retries: int = 5,
    base_delay: int = 10,
) -> genai.Client:
    """Try each key, rotating on 429 errors.

    This is a helper — the actual retry happens in the caller.
    This just picks the next key.
    """
    return get_gemini_client()


def rotate_on_error() -> None:
    """Force rotation to next key (call after a 429 error)."""
    global _current_index
    _ensure_keys()
    _current_index = (_current_index + 1) % len(_KEYS)
