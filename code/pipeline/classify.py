"""Classification layer — one Gemini API call per ticket.

Entry point: classify_ticket(issue, subject, company, sanitised_issue)
             -> ClassifierResult

Makes exactly one API call with temperature=0, model=gemini-2.5-flash.
Returns structured JSON parsed into ClassifierResult.

This module loads the system prompt from prompts/classifier.txt.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from models import ClassifierResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_REQUEST_TYPES = frozenset([
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
])

VALID_DOMAINS = frozenset(["hackerrank", "claude", "visa", "unknown"])

VALID_SEVERITIES = frozenset(["low", "medium", "high", "critical"])

REQUIRED_JSON_KEYS = [
    "domain", "request_type", "product_area", "severity",
    "escalate", "escalation_reason", "confidence", "search_query",
    "justification", "company_coherence", "cross_domain_mismatch",
]

MODEL = "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Load system prompt from file (once at module import)
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    """Load classifier system prompt from prompts/classifier.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "classifier.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Classifier prompt not found at {prompt_path}"
        )
    return prompt_path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load_system_prompt()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str, context: str) -> dict:
    """Parse JSON from API response. Raises ValueError on failure."""
    cleaned = text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parse failure in {context}: {e}\n"
            f"Raw text: {text[:300]}"
        )


def _validate_keys(data: dict) -> None:
    """Raise ValueError if any required key is missing."""
    missing = [k for k in REQUIRED_JSON_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Missing required JSON keys: {missing}"
        )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _build_user_message(
    issue: str,
    subject: str,
    company: str,
    sanitised_issue: str,
) -> str:
    """Build the user message for the classifier API call."""
    parts = [
        f"Company: {company}",
        f"Subject: {subject}",
        f"Issue: {sanitised_issue}",
    ]
    return "\n".join(parts)


def _call_classifier_api(user_message: str) -> str:
    """Make one Gemini API call and return the raw response text.

    Uses temperature=0 and model=gemini-2.5-flash-lite. Rotates through
    multiple API keys on rate limit errors.
    """
    import time
    from utils.key_rotator import get_gemini_client, rotate_on_error

    # Prepend system prompt to user message
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_message

    max_retries = 8
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model=MODEL,
                contents=full_prompt,
                config={"temperature": 0},
            )
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "resource_exhausted" in err_str or "quota" in err_str:
                rotate_on_error()
                delay = base_delay * (attempt + 1)
                print(f"  [RATE] Key rotated, retry {attempt + 1}/{max_retries} in {delay}s...")
                time.sleep(delay)
            else:
                raise  # Non-rate-limit error, let caller handle

    # All retries exhausted
    raise RuntimeError("Rate limit: all keys and retries exhausted")


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------

def _build_result(data: dict) -> ClassifierResult:
    """Build ClassifierResult from validated JSON dict with enum guards."""
    # Enum guard: request_type
    request_type = str(data.get("request_type", "invalid")).lower()
    if request_type not in VALID_REQUEST_TYPES:
        request_type = "invalid"

    # Enum guard: domain
    domain = str(data.get("domain", "unknown")).lower()
    if domain not in VALID_DOMAINS:
        domain = "unknown"

    # Enum guard: severity
    severity = str(data.get("severity", "medium")).lower()
    if severity not in VALID_SEVERITIES:
        severity = "medium"

    # Parse numeric fields safely
    try:
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    try:
        company_coherence = float(data.get("company_coherence", 0.0))
        company_coherence = max(0.0, min(1.0, company_coherence))
    except (TypeError, ValueError):
        company_coherence = 0.0

    # Parse boolean fields safely
    escalate = bool(data.get("escalate", False))
    cross_domain_mismatch = bool(data.get("cross_domain_mismatch", False))

    # Low confidence auto-escalation
    if confidence < 0.72 and not escalate:
        escalate = True
        data["escalation_reason"] = (
            f"low confidence ({confidence:.2f}) -- routed to human"
        )

    # Cross-domain mismatch auto-escalation
    if cross_domain_mismatch and not escalate:
        escalate = True
        data["escalation_reason"] = (
            "company field contradicts issue content"
        )

    # Keep product_area as classified -- do not clear on escalation
    product_area = str(data.get("product_area", ""))

    return ClassifierResult(
        domain=domain,
        request_type=request_type,
        product_area=product_area,
        severity=severity,
        escalate=escalate,
        escalation_reason=str(data.get("escalation_reason", "")),
        confidence=confidence,
        search_query=str(data.get("search_query", "")),
        justification=str(data.get("justification", "")),
        company_coherence=company_coherence,
        cross_domain_mismatch=cross_domain_mismatch,
    )


def _build_error_fallback(reason: str) -> ClassifierResult:
    """Build a safe fallback ClassifierResult on any error."""
    return ClassifierResult(
        domain="unknown",
        request_type="invalid",
        product_area="",
        severity="medium",
        escalate=True,
        escalation_reason=reason,
        confidence=0.0,
        search_query="",
        justification=f"Classification failed: {reason}",
        company_coherence=0.0,
        cross_domain_mismatch=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_ticket(
    issue: str,
    subject: str,
    company: str,
    sanitised_issue: str,
) -> ClassifierResult:
    """Classify a support ticket via one Gemini API call.

    Args:
        issue: Original raw issue text.
        subject: Ticket subject line.
        company: Company field from input CSV.
        sanitised_issue: PII-scrubbed issue from ingest.

    Returns:
        ClassifierResult with classification, escalation decision,
        and retrieval query.
    """
    try:
        # Build and send API call
        user_message = _build_user_message(
            issue, subject, company, sanitised_issue,
        )
        raw_response = _call_classifier_api(user_message)

        # Parse JSON
        data = _safe_parse_json(raw_response, "classifier")

        # Validate required keys
        _validate_keys(data)

        # Build result with enum guards
        return _build_result(data)

    except RuntimeError as e:
        # API key missing
        return _build_error_fallback(
            f"classifier error -- {e}"
        )
    except (ConnectionError, TimeoutError) as e:
        # API call failed (network, rate limit, etc.)
        return _build_error_fallback(
            f"classifier API error -- routed to human ({type(e).__name__})"
        )
    except ValueError as e:
        # JSON parse or key validation failed
        return _build_error_fallback(
            f"classifier error -- routed to human ({e})"
        )
    except Exception as e:
        # Catch-all for unexpected errors
        return _build_error_fallback(
            f"classifier error -- routed to human ({type(e).__name__}: {e})"
        )
