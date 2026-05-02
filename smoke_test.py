"""Smoke test: process first 3 rows only."""
import sys
import os
sys.path.insert(0, "code")

from dotenv import load_dotenv
load_dotenv(".env")

from pipeline.retrieve import _ensure_loaded
from pipeline.ingest import validate_ticket
from pipeline.classify import classify_ticket

_ensure_loaded()

from utils.csv_io import read_input_csv
rows = read_input_csv("support_tickets/support_tickets.csv")
print(f"Read {len(rows)} tickets\n")

for idx, row in enumerate(rows[:3]):
    issue = str(row.get("issue", ""))
    subject = str(row.get("subject", ""))
    company = str(row.get("company", ""))

    # Ingest
    ingest = validate_ticket(issue, subject, company)
    if not ingest.is_valid:
        print(f"[{idx+1:04d}] {company:12s} | INGEST REJECTED: {ingest.reject_reason}")
        continue

    # Classify
    cls = classify_ticket(issue, subject, company, ingest.sanitised_issue)
    print(
        f"[{idx+1:04d}] {company:12s} | "
        f"{cls.domain:12s} | {cls.request_type:15s} | "
        f"sev={cls.severity:8s} | conf={cls.confidence:.2f} | "
        f"esc={cls.escalate}"
    )

print("\nSmoke test complete.")
