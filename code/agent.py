"""CLI entry point for the Multi-Domain Support Triage Agent.

Usage:
    python main.py
    python main.py --input ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv

All three modes share the same entry point. Business logic lives in
pipeline/ modules — this file handles CLI parsing, orchestration,
and the progress loop only.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from models import OutputRow
from pipeline.classify import classify_ticket
from pipeline.generate import generate_response
from pipeline.ingest import validate_ticket
from pipeline.retrieve import compute_coverage_score, retrieve_chunks
from utils.csv_io import read_input_csv, write_output_csv


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

ESCALATED_RESPONSE = (
    "Hi, your request has been escalated to our support team and will be "
    "reviewed as a priority. Please expect a response shortly."
)

FEATURE_REQUEST_RESPONSE = (
    "Hi, thank you for your feedback. We've noted your suggestion and "
    "it will be reviewed by our product team."
)

INVALID_RESPONSE = (
    "Hi, we're sorry but this request falls outside the scope of our "
    "support for HackerRank, Claude, and Visa. Please contact the "
    "relevant provider directly."
)

LOW_COVERAGE_RESPONSE = (
    "Hi, your request has been escalated to our support team as we "
    "could not find relevant documentation to address your query."
)

CORPUS_GAP_RESPONSE = (
    "Hi, your request has been escalated to our support team as our "
    "documentation does not cover this specific query."
)

# Coverage threshold
COVERAGE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent",
    )
    # Default paths relative to code/ directory
    repo_root = Path(__file__).parent.parent
    default_input = str(repo_root / "support_tickets" / "support_tickets.csv")
    default_output = str(repo_root / "support_tickets" / "output.csv")

    parser.add_argument(
        "--input",
        type=str,
        default=default_input,
        help="Path to the input support tickets CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path to write the output CSV.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After processing, run evaluate.py to compare against expected.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip already-processed rows).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def process_tickets(input_path: str, output_path: str, resume: bool = False) -> None:
    """Read input CSV, process each ticket through full pipeline, write output.

    Pipeline stages:
    Phase 3: ingest (validate_ticket)
    Phase 4: classify (classify_ticket)
    Phase 5: retrieve (retrieve_chunks) + generate (generate_response)
    """
    # Read and validate input
    rows = read_input_csv(input_path)
    print(f"Read {len(rows)} tickets from {input_path}")

    # Process each ticket
    output_rows: list[OutputRow] = []
    stats = {
        "junk": 0, "injection": 0, "non_english": 0,
        "classified": 0, "escalated_cls": 0,
        "feature_request": 0, "invalid_oos": 0,
        "retrieved": 0, "low_coverage": 0, "corpus_gap": 0,
        "replied": 0,
    }

    for idx, row in enumerate(rows):
        # Rate limit: small delay between tickets for API quotas
        if idx > 0:
            import time
            time.sleep(2)

        ticket_id = f"{idx + 1:04d}"
        issue = str(row.get("issue", ""))
        subject = str(row.get("subject", ""))
        company = str(row.get("company", "")) or ""

        # ---------------------------------------------------------------
        # Stage 1: Ingest validation
        # ---------------------------------------------------------------
        ingest_result = validate_ticket(issue, subject, company)

        if not ingest_result.is_valid:
            stats[ingest_result.reject_reason] = (
                stats.get(ingest_result.reject_reason, 0) + 1
            )
            output_row = OutputRow(
                status=ingest_result.status,
                product_area="",
                response=ingest_result.response,
                justification=ingest_result.justification,
                request_type=ingest_result.request_type,
            )
            output_rows.append(output_row)

            tag = f"[{ingest_result.reject_reason.upper()}]"
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{ingest_result.request_type:15s} | "
                f"{ingest_result.status:10s} <- {tag}"
            )
            continue

        # ---------------------------------------------------------------
        # Stage 2: Classification
        # ---------------------------------------------------------------
        stats["classified"] += 1
        cls = classify_ticket(
            issue=issue,
            subject=subject,
            company=company,
            sanitised_issue=ingest_result.sanitised_issue,
        )

        # Stage 2a: Classifier escalation
        if cls.escalate:
            stats["escalated_cls"] += 1
            output_row = OutputRow(
                status="escalated",
                product_area=cls.product_area,
                response=ESCALATED_RESPONSE,
                justification=cls.justification,
                request_type=cls.request_type,
            )
            output_rows.append(output_row)
            esc_tag = "ESC-CLS"
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{cls.domain:12s} | {cls.request_type:15s} | "
                f"sev={cls.severity:8s} | conf={cls.confidence:.2f} | "
                f"{esc_tag}"
            )
            continue

        # Stage 2b: Feature request — reply immediately
        if cls.request_type == "feature_request":
            stats["feature_request"] += 1
            stats["replied"] += 1
            output_row = OutputRow(
                status="replied",
                product_area=cls.product_area,
                response=FEATURE_REQUEST_RESPONSE,
                justification=cls.justification,
                request_type="feature_request",
            )
            output_rows.append(output_row)
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{cls.domain:12s} | feature_request   | "
                f"sev={cls.severity:8s} | replied"
            )
            continue

        # Stage 2c: Invalid / out-of-scope — reply immediately
        if cls.request_type == "invalid":
            stats["invalid_oos"] += 1
            stats["replied"] += 1
            output_row = OutputRow(
                status="replied",
                product_area=cls.product_area,
                response=INVALID_RESPONSE,
                justification=cls.justification,
                request_type="invalid",
            )
            output_rows.append(output_row)
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{cls.domain:12s} | invalid           | "
                f"sev={cls.severity:8s} | replied"
            )
            continue

        # ---------------------------------------------------------------
        # Stage 3: Retrieval
        # ---------------------------------------------------------------
        stats["retrieved"] += 1
        chunks = retrieve_chunks(
            cls.search_query, cls.domain, top_k=5,
        )
        coverage = compute_coverage_score(chunks)

        # Stage 3a: Low coverage — escalate
        if coverage < COVERAGE_THRESHOLD:
            stats["low_coverage"] += 1
            justification = (
                f"{cls.justification} Escalated: corpus coverage "
                f"too low (score={coverage:.2f})."
            )
            output_row = OutputRow(
                status="escalated",
                product_area=cls.product_area,
                response=LOW_COVERAGE_RESPONSE,
                justification=justification,
                request_type=cls.request_type,
            )
            output_rows.append(output_row)
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{cls.domain:12s} | {cls.request_type:15s} | "
                f"sev={cls.severity:8s} | "
                f"chunks={len(chunks)} | cov={coverage:.2f} | "
                f"ESC-COV"
            )
            continue

        # ---------------------------------------------------------------
        # Stage 4: Generation
        # ---------------------------------------------------------------
        response_text = generate_response(
            issue=ingest_result.sanitised_issue,
            domain=cls.domain,
            chunks=chunks,
        )

        # Stage 4a: CORPUS_GAP or generation error — escalate
        if response_text is None:
            stats["corpus_gap"] += 1
            justification = (
                f"{cls.justification} Escalated: CORPUS_GAP or "
                f"generation error."
            )
            output_row = OutputRow(
                status="escalated",
                product_area=cls.product_area,
                response=CORPUS_GAP_RESPONSE,
                justification=justification,
                request_type=cls.request_type,
            )
            output_rows.append(output_row)
            print(
                f"[{ticket_id}] {company or 'Unknown':12s} | "
                f"{cls.domain:12s} | {cls.request_type:15s} | "
                f"sev={cls.severity:8s} | "
                f"chunks={len(chunks)} | cov={coverage:.2f} | "
                f"ESC-GAP"
            )
            continue

        # Stage 4b: Successful generation — reply
        stats["replied"] += 1
        justification = (
            f"{cls.justification} Corpus coverage: {coverage:.2f}. "
            f"Retrieved {len(chunks)} chunks from {cls.domain} corpus."
        )
        output_row = OutputRow(
            status="replied",
            product_area=cls.product_area,
            response=response_text,
            justification=justification,
            request_type=cls.request_type,
        )
        output_rows.append(output_row)
        print(
            f"[{ticket_id}] {company or 'Unknown':12s} | "
            f"{cls.domain:12s} | {cls.request_type:15s} | "
            f"sev={cls.severity:8s} | "
            f"chunks={len(chunks)} | cov={coverage:.2f} | "
            f"replied"
        )

    # Write and validate output
    write_output_csv(output_path, output_rows)

    # Summary
    print(f"\nProcessed {len(output_rows)} rows -> {output_path}")
    print(f"  Ingest rejected:   "
          f"{stats['junk'] + stats['injection'] + stats['non_english']}")
    print(f"    Junk:            {stats['junk']}")
    print(f"    Injection:       {stats['injection']}")
    print(f"    Non-English:     {stats['non_english']}")
    print(f"  Classified:        {stats['classified']}")
    print(f"    Escalated (cls): {stats['escalated_cls']}")
    print(f"    Feature request: {stats['feature_request']}")
    print(f"    Invalid/OOS:     {stats['invalid_oos']}")
    print(f"  Retrieved:         {stats['retrieved']}")
    print(f"    Low coverage:    {stats['low_coverage']}")
    print(f"    CORPUS_GAP:      {stats['corpus_gap']}")
    print(f"    Replied:         {stats['replied']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    # Load .env from repo root or code/ directory
    repo_root = Path(__file__).parent.parent
    env_path = repo_root / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    args = parse_args()
    process_tickets(args.input, args.output, resume=args.resume)

    if args.evaluate:
        print("\n--- Running evaluation ---")
        eval_script = Path(__file__).parent / "evaluate.py"
        subprocess.run(
            [sys.executable, str(eval_script), args.output, args.input],
            check=False,
        )


if __name__ == "__main__":
    main()
