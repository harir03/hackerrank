"""Data contracts for the Multi-Domain Support Triage Agent.

Ticket is the single source of truth for a ticket's state as it moves
through the pipeline. Every pipeline function takes a Ticket and returns
a modified Ticket. Never pass raw dicts between pipeline stages.

OutputRow defines the exact CSV output schema.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Ticket:
    """Internal representation of a support ticket through the pipeline."""

    # Identifiers
    ticket_id: str               # Zero-padded row index: "0001"
    raw_issue: str               # Original, unmodified
    raw_subject: str
    company: str                 # HackerRank | Claude | Visa | None
    scrubbed_issue: str          # After PII scrubber
    corpus_version: str          # From metadata.json

    # Populated by ingest.py
    is_junk: bool = False
    is_injection: bool = False
    is_non_english: bool = False
    has_multiple_requests: bool = False
    primary_issue: str = ""      # If multi-request, the primary one
    secondary_issue: str = ""    # If multi-request, the secondary one

    # Populated by classify.py
    inferred_domain: str = ""
    request_type: str = ""       # One of four enum values
    product_area: str = ""
    confidence: float = 0.0
    escalate: bool = False
    escalation_reason: str = ""
    search_query: str = ""
    severity: str = ""           # low | medium | high | critical
    company_mismatch: bool = False
    cross_domain_ambiguous: bool = False

    # Populated by retrieve.py
    retrieved_chunks: Optional[list] = field(default_factory=list)
    coverage_score: float = 0.0

    # Populated by generate.py
    generated_response: str = ""
    corpus_gap: bool = False

    # Populated by quality.py
    quality_passed: bool = False
    quality_failure_reason: str = ""

    # Final output
    final_status: str = ""
    final_response: str = ""
    final_justification: str = ""
    gate_triggered: str = ""     # Which of the 5 gates caused escalation


@dataclass
class OutputRow:
    """Exact CSV output schema — five columns, no extras."""

    status: str          # Exactly "replied" or "escalated"
    product_area: str    # Title case phrase
    response: str        # Always populated
    justification: str   # Always populated
    request_type: str    # Exactly one of four values


@dataclass
class IngestResult:
    """Return type for pipeline/ingest.py validate_ticket().

    If is_valid is False, the ticket was rejected by an ingest check
    and the status/request_type/response/justification fields are set
    for direct output — no further pipeline stages should run.
    """

    is_valid: bool
    reject_reason: str = ""              # "", "junk", "injection", "non_english"
    sanitised_issue: str = ""            # PII-scrubbed version of issue
    has_multiple_issues: bool = False
    secondary_issue_note: str = ""       # "" if single issue

    # Only set when is_valid=False (short-circuit output)
    status: str = ""
    request_type: str = ""
    response: str = ""
    justification: str = ""


@dataclass
class ClassifierResult:
    """Return type for pipeline/classify.py classify_ticket().

    Contains the full classification output from the Claude API call.
    product_area is always snake_case. If escalated, product_area is "".
    """

    domain: str = ""                  # "hackerrank", "claude", "visa", "unknown"
    request_type: str = ""            # product_issue, feature_request, bug, invalid
    product_area: str = ""            # snake_case, "" if escalated
    severity: str = ""                # low, medium, high, critical
    escalate: bool = False
    escalation_reason: str = ""       # "" if escalate=False
    confidence: float = 0.0           # 0.0 to 1.0
    search_query: str = ""            # optimised retrieval query for Phase 5
    justification: str = ""           # internal-facing reasoning, 1-2 sentences
    company_coherence: float = 0.0    # 0.0 to 1.0
    cross_domain_mismatch: bool = False

