"""CSV I/O helpers for the Multi-Domain Support Triage Agent.

Two public functions:
- read_input_csv:  reads and validates input CSV
- write_output_csv: validates and writes output CSV

All terminal output goes through utils/terminal_display.py — this
module only raises exceptions on validation failure.
"""

from pathlib import Path

import pandas as pd

from models import OutputRow


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_INPUT_COLUMNS = {"issue", "subject", "company"}

VALID_STATUSES = frozenset(["replied", "escalated"])

VALID_REQUEST_TYPES = frozenset([
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
])

OUTPUT_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
]


# ---------------------------------------------------------------------------
# read_input_csv
# ---------------------------------------------------------------------------

def read_input_csv(path: str | Path) -> list[dict]:
    """Read input CSV and validate required columns exist.

    Columns are matched case-insensitively and normalised to lowercase
    in the returned dicts.

    Args:
        path: Path to the input CSV file.

    Returns:
        List of row dicts with lowercase keys.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If any required column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalise column names to lowercase for matching
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    actual_columns = set(df.columns)
    missing = REQUIRED_INPUT_COLUMNS - actual_columns
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {sorted(actual_columns)}"
        )

    # Fill NaN with empty string so downstream code always sees strings
    df = df.fillna("")

    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# write_output_csv
# ---------------------------------------------------------------------------

def write_output_csv(path: str | Path, rows: list[OutputRow]) -> None:
    """Validate every row and write output CSV.

    Before validation, applies safety guards:
    - Empty/None response → escalation template
    - None product_area → ""
    - Empty/None justification → default text
    - Short replied response → terminal warning

    Args:
        path: Destination path for the output CSV.
        rows: List of OutputRow dataclass instances.

    Raises:
        ValueError: If any row has an invalid status or request_type,
                    including the failing row index in the message.
    """
    # --- Safety guards (fix before validation) ---
    for idx, row in enumerate(rows):
        # Guard 1: empty/None response → escalation template
        if not row.response:
            row.response = (
                "Hi, your request has been escalated to our support team."
            )

        # Guard 2: None product_area → ""
        if row.product_area is None:
            row.product_area = ""

        # Guard 3: empty/None justification → default
        if not row.justification:
            row.justification = "Processed by automated triage system."

        # Guard 4: short replied response → warning
        if row.status == "replied":
            word_count = len(row.response.split())
            if word_count < 10:
                print(
                    f"  [WARNING] Row {idx}: replied response is "
                    f"very short ({word_count} words)"
                )

    # --- Validation ---
    for idx, row in enumerate(rows):
        if row.status not in VALID_STATUSES:
            raise ValueError(
                f"Row {idx}: invalid status '{row.status}'. "
                f"Must be one of {sorted(VALID_STATUSES)}."
            )

        if row.request_type not in VALID_REQUEST_TYPES:
            raise ValueError(
                f"Row {idx}: invalid request_type '{row.request_type}'. "
                f"Must be one of {sorted(VALID_REQUEST_TYPES)}."
            )

        if not row.response:
            raise ValueError(
                f"Row {idx}: response must not be empty."
            )

        if not row.justification:
            raise ValueError(
                f"Row {idx}: justification must not be empty."
            )

    # Build DataFrame from validated rows
    data = [
        {
            "status": row.status,
            "product_area": row.product_area,
            "response": row.response,
            "justification": row.justification,
            "request_type": row.request_type,
        }
        for row in rows
    ]

    df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")

    # Post-write verification
    written = pd.read_csv(path)
    if len(written) != len(rows):
        raise ValueError(
            f"Post-write verification failed: wrote {len(rows)} rows "
            f"but file contains {len(written)} rows."
        )

