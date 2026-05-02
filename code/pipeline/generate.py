"""Generation layer -- one Gemini API call to produce a grounded response.

Entry point: generate_response(issue, domain, chunks) -> str | None

Makes exactly one API call with temperature=0, model=gemini-2.5-flash.
Returns the response string or None if CORPUS_GAP or API error.

Loads the system prompt template from prompts/generator.txt.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Load prompt template (once at module import)
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    """Load generator prompt template from prompts/generator.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "generator.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Generator prompt not found at {prompt_path}"
        )
    return prompt_path.read_text(encoding="utf-8")


PROMPT_TEMPLATE = _load_prompt_template()


# ---------------------------------------------------------------------------
# Chunk formatting
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks for the prompt."""
    if not chunks:
        return "(No documentation available)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section", "Unknown")
        url = chunk.get("url", chunk.get("filepath", ""))
        text = chunk.get("text", "")
        parts.append(
            f"[{i}] Section: {section} | Source: {url}\n{text}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(issue: str, domain: str, chunks: list[dict]) -> str:
    """Build the complete prompt by filling template placeholders."""
    formatted_chunks = _format_chunks(chunks)

    return PROMPT_TEMPLATE.format(
        domain=domain,
        chunks=formatted_chunks,
        issue=issue,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_response(
    issue: str,
    domain: str,
    chunks: list[dict],
) -> Optional[str]:
    """Generate a support response using Gemini API with corpus context.

    Args:
        issue: The sanitised issue text.
        domain: The classified domain (hackerrank, claude, visa).
        chunks: Retrieved corpus chunks with text, section, url keys.

    Returns:
        Response string if successful, None if CORPUS_GAP detected
        or any error occurs (caller should escalate).
    """
    try:
        import time
        from utils.key_rotator import get_gemini_client, rotate_on_error

        # Build the full prompt
        full_prompt = _build_prompt(issue, domain, chunks)

        max_retries = 8
        base_delay = 5

        for attempt in range(max_retries):
            try:
                client = get_gemini_client()
                response = client.models.generate_content(
                    model=MODEL,
                    contents=full_prompt,
                    config={"temperature": 0},
                )

                response_text = response.text.strip()

                # Check for CORPUS_GAP signal
                if "CORPUS_GAP" in response_text:
                    return None

                return response_text

            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                    rotate_on_error()
                    delay = base_delay * (attempt + 1)
                    print(f"  [RATE] Gen key rotated, retry {attempt + 1}/{max_retries} in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        # All retries exhausted
        print("  [WARN] Generator rate limit: all keys exhausted")
        return None

    except (ConnectionError, TimeoutError) as e:
        print(f"  [WARN] Generator API error: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Generator error: {type(e).__name__}: {e}")
        return None
