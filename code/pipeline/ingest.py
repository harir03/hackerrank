"""Ingestion layer — five sequential safety checks on raw tickets.

Entry point: validate_ticket(ticket) -> IngestResult

Check order (priority):
1. Injection detector  (pattern-matching, highest priority)
2. Junk filter         (cheapest heuristic)
3. PII scrubber        (never rejects — scrubs and continues)
4. Language detector   (langdetect library)
5. Multi-issue detector (heuristic, never rejects — flags only)

This module must NOT make API calls. Only build_corpus.py and
classify/generate/quality modules may call the Anthropic API.
"""

import re

from langdetect import detect_langs, LangDetectException

from models import IngestResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Check 1 — Junk filter
MIN_WORD_COUNT = 5
SENTENCE_END_RE = re.compile(r"[.?!:]")

# Check 2 — Injection detector (case-insensitive literal patterns)
INJECTION_LITERALS = [
    "ignore previous",
    "ignore all",
    "disregard",
    "you are now",
    "new persona",
    "act as",
    "forget your instructions",
    "system:",
    "[system",
    "<|im_start|>",
    "<|system|>",
]

INJECTION_SEMANTIC = [
    "respond only with",
    "do not escalate",
    "always reply",
    "your new instructions",
    "override your",
    "pretend you are",
    "you must obey",
    "bypass your",
    "system prompt",
    "repeat your instructions",
]

# Check 3 — PII scrubber patterns
CARD_RE = re.compile(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b")
CVV_RE = re.compile(
    r"(?:cvv|cvc|security\s*code)\s*[:=]?\s*(\d{3,4})\b",
    re.IGNORECASE,
)
# Indian mobile: 10 digits starting with 6-9, optionally prefixed with +91/91/0
PHONE_RE = re.compile(
    r"(?:\+?91[\s-]?|0)?([6-9]\d{9})\b"
)
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# Check 5 — Multi-issue conjunctive patterns
MULTI_ISSUE_PATTERNS = [
    "also",
    "additionally",
    "and also",
    "another issue",
    "second question",
    "furthermore",
    "as well as",
    "on top of that",
]

# Short-circuit response strings
JUNK_RESPONSE = (
    "Hi, we were unable to process this request. "
    "Please resubmit with a clear description of your issue."
)
INJECTION_RESPONSE = (
    "Hi, your request has been escalated to our support team for review."
)
INJECTION_JUSTIFICATION = (
    "Prompt injection attempt detected in ticket body."
)
NON_ENGLISH_RESPONSE = (
    "Hi, your request has been escalated to our support team."
)
NON_ENGLISH_JUSTIFICATION = (
    "Ticket appears to be in a non-English language "
    "-- routed to human agent."
)


# ---------------------------------------------------------------------------
# Check 1 — Junk filter
# ---------------------------------------------------------------------------

def _is_junk(issue: str) -> bool:
    """Return True if the issue text is junk / unprocessable."""
    text = issue.strip()

    # Empty or whitespace-only
    if not text:
        return True

    words = text.split()
    word_count = len(words)

    # High ratio of non-alpha characters (random keyboard mashing)
    alpha_count = sum(1 for c in text if c.isalpha())
    total_count = len(text.replace(" ", ""))
    if total_count > 0 and (alpha_count / total_count) < 0.40:
        return True

    # Under 5 words AND no sentence-ending punctuation
    if word_count < MIN_WORD_COUNT and not SENTENCE_END_RE.search(text):
        return True

    # Gibberish detection: if most words lack vowels, it's random chars
    # Real English words almost always contain vowels
    vowels = set("aeiouAEIOU")
    no_vowel_count = sum(
        1 for w in words
        if len(w) > 1 and not any(c in vowels for c in w)
    )
    if word_count > 0 and (no_vowel_count / word_count) > 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# Check 2 — Injection detector
# ---------------------------------------------------------------------------

def _is_injection(issue: str) -> bool:
    """Return True if the issue text contains prompt injection patterns."""
    text_lower = issue.lower()

    for pattern in INJECTION_LITERALS:
        if pattern.lower() in text_lower:
            return True

    for pattern in INJECTION_SEMANTIC:
        if pattern.lower() in text_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Check 3 — PII scrubber
# ---------------------------------------------------------------------------

def _scrub_pii(issue: str) -> str:
    """Replace PII patterns with redaction tokens. Never rejects."""
    text = issue

    # Credit card numbers (16 digits, possibly spaced/dashed)
    text = CARD_RE.sub("[CARD_NUMBER]", text)

    # CVV / CVC / security code
    text = CVV_RE.sub(
        lambda m: m.group(0).replace(m.group(1), "[CVV]"),
        text,
    )

    # Phone numbers (Indian mobile)
    text = PHONE_RE.sub("[PHONE]", text)

    # Email addresses
    text = EMAIL_RE.sub("[EMAIL]", text)

    return text


# ---------------------------------------------------------------------------
# Check 4 — Language detector
# ---------------------------------------------------------------------------

def _is_non_english(text: str) -> bool:
    """Return True if the text is confidently non-English."""
    # Very short text is unreliable for language detection — benefit of doubt
    if len(text.split()) < 4:
        return False

    try:
        results = detect_langs(text)
        if not results:
            return False

        top = results[0]
        # If the top language is not English AND confidence > 0.85
        if top.lang != "en" and top.prob > 0.85:
            return True

        return False
    except LangDetectException:
        # Too short to detect, or other issue — do not reject
        return False


# ---------------------------------------------------------------------------
# Check 5 — Multi-issue detector
# ---------------------------------------------------------------------------

def _has_multiple_issues(text: str) -> bool:
    """Return True if the text contains 2+ conjunctive patterns."""
    text_lower = text.lower()
    hit_count = 0

    for pattern in MULTI_ISSUE_PATTERNS:
        if pattern in text_lower:
            hit_count += 1
        if hit_count >= 2:
            return True

    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_ticket(issue: str, subject: str = "", company: str = "") -> IngestResult:
    """Run five sequential safety checks on a raw ticket.

    Args:
        issue: Raw issue text from the ticket.
        subject: Raw subject line.
        company: Company name from the ticket.

    Returns:
        IngestResult with is_valid=True if ticket passes all checks,
        or is_valid=False with reject fields populated for short-circuit.
    """
    # Check 1 — Injection detector (highest priority)
    if _is_injection(issue):
        return IngestResult(
            is_valid=False,
            reject_reason="injection",
            sanitised_issue=issue,
            status="escalated",
            request_type="invalid",
            response=INJECTION_RESPONSE,
            justification=INJECTION_JUSTIFICATION,
        )

    # Check 2 — Junk filter
    if _is_junk(issue):
        return IngestResult(
            is_valid=False,
            reject_reason="junk",
            sanitised_issue=issue,
            status="replied",
            request_type="invalid",
            response=JUNK_RESPONSE,
            justification="Ticket body is empty, too short, or "
                          "contains no meaningful content.",
        )

    # Check 3 — PII scrubber (never rejects)
    sanitised = _scrub_pii(issue)

    # Check 4 — Language detector (uses sanitised text)
    if _is_non_english(sanitised):
        return IngestResult(
            is_valid=False,
            reject_reason="non_english",
            sanitised_issue=sanitised,
            status="escalated",
            request_type="invalid",
            response=NON_ENGLISH_RESPONSE,
            justification=NON_ENGLISH_JUSTIFICATION,
        )

    # Check 5 — Multi-issue detector (never rejects)
    multi = _has_multiple_issues(sanitised)
    secondary_note = ""
    if multi:
        secondary_note = (
            "Multiple requests detected -- primary request addressed only"
        )

    # All checks passed
    return IngestResult(
        is_valid=True,
        sanitised_issue=sanitised,
        has_multiple_issues=multi,
        secondary_issue_note=secondary_note,
    )
