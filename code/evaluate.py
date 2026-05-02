"""Evaluation script — compare predicted output against sample expected values.

Usage:
    python evaluate.py predicted.csv expected.csv

Or via agent.py:
    python agent.py --evaluate --input sample_support_tickets.csv --output results.csv
"""

import math
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Column mappings: sample CSV → our output CSV
# ---------------------------------------------------------------------------

EXPECTED_COL_MAP = {
    "Status": "status",
    "Request Type": "request_type",
    "Product Area": "product_area",
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm(val: object) -> str:
    """Normalise a value for comparison: strip, lowercase, handle NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip().lower()


def _is_nan(val: object) -> bool:
    """Return True if value is NaN, None, or empty string."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(predicted_path: str, expected_path: str) -> dict:
    """Compare predicted output CSV against expected sample CSV.

    Args:
        predicted_path: Path to our output CSV.
        expected_path: Path to the sample CSV with expected values.

    Returns:
        Dict with accuracy metrics.
    """
    pred_df = pd.read_csv(predicted_path)
    exp_df = pd.read_csv(expected_path)

    n = min(len(pred_df), len(exp_df))
    if len(pred_df) != len(exp_df):
        print(f"[WARNING] Row count mismatch: predicted={len(pred_df)}, "
              f"expected={len(exp_df)}. Comparing first {n} rows.")

    # Track per-field accuracy
    status_correct = 0
    status_wrong: list[str] = []

    rtype_correct = 0
    rtype_wrong: list[str] = []

    parea_correct = 0
    parea_wrong: list[str] = []
    parea_na = 0  # NaN in expected → not counted

    per_row: list[str] = []

    for i in range(n):
        # --- Status ---
        pred_status = _norm(pred_df.loc[i, "status"])
        exp_status = _norm(exp_df.loc[i, "Status"])
        status_ok = (pred_status == exp_status)
        if status_ok:
            status_correct += 1
            status_tag = "[OK]"
        else:
            status_wrong.append(
                f"Row {i}: got={pred_status}, exp={exp_status}"
            )
            status_tag = f"[FAIL](got={pred_status}, exp={exp_status})"

        # --- Request Type ---
        pred_rtype = _norm(pred_df.loc[i, "request_type"])
        exp_rtype = _norm(exp_df.loc[i, "Request Type"])
        rtype_ok = (pred_rtype == exp_rtype)
        if rtype_ok:
            rtype_correct += 1
            rtype_tag = "[OK]"
        else:
            rtype_wrong.append(
                f"Row {i}: got={pred_rtype}, exp={exp_rtype}"
            )
            rtype_tag = f"[FAIL](got={pred_rtype}, exp={exp_rtype})"

        # --- Product Area ---
        exp_parea_raw = exp_df.loc[i, "Product Area"]
        if _is_nan(exp_parea_raw):
            parea_na += 1
            parea_tag = "N/A"
        else:
            pred_parea = _norm(pred_df.loc[i, "product_area"])
            exp_parea = _norm(exp_parea_raw)
            parea_ok = (pred_parea == exp_parea)
            if parea_ok:
                parea_correct += 1
                parea_tag = "[OK]"
            else:
                parea_wrong.append(
                    f"Row {i}: got={pred_parea}, exp={exp_parea}"
                )
                parea_tag = f"[FAIL](got={pred_parea}, exp={exp_parea})"

        per_row.append(
            f"  Row {i}: status={status_tag} "
            f"request_type={rtype_tag} "
            f"product_area={parea_tag}"
        )

    # Denominators
    parea_denom = n - parea_na  # Only count rows with non-NaN expected
    total_fields = n + n + parea_denom  # status + rtype + (non-NaN parea)
    total_correct = status_correct + rtype_correct + parea_correct

    # Print report
    print()
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total rows: {n}")
    print()

    # Status
    pct_s = (status_correct / n * 100) if n > 0 else 0
    print(f"  STATUS accuracy:")
    print(f"    Correct: {status_correct}/{n} ({pct_s:.1f}%)")
    if status_wrong:
        print(f"    Wrong rows:")
        for w in status_wrong:
            print(f"      {w}")
    print()

    # Request Type
    pct_r = (rtype_correct / n * 100) if n > 0 else 0
    print(f"  REQUEST_TYPE accuracy:")
    print(f"    Correct: {rtype_correct}/{n} ({pct_r:.1f}%)")
    if rtype_wrong:
        print(f"    Wrong rows:")
        for w in rtype_wrong:
            print(f"      {w}")
    print()

    # Product Area
    pct_p = (parea_correct / parea_denom * 100) if parea_denom > 0 else 0
    print(f"  PRODUCT_AREA accuracy:")
    print(f"    Correct: {parea_correct}/{parea_denom} ({pct_p:.1f}%)"
          f"  ({parea_na} NaN rows excluded)")
    if parea_wrong:
        print(f"    Wrong rows:")
        for w in parea_wrong:
            print(f"      {w}")
    print()

    # Per-row breakdown
    print(f"  PER-ROW BREAKDOWN:")
    for line in per_row:
        print(line)
    print()

    # Overall
    pct_all = (total_correct / total_fields * 100) if total_fields > 0 else 0
    print(f"  Overall score: {total_correct}/{total_fields} fields "
          f"correct ({pct_all:.1f}%)")
    print("=" * 60)
    print()

    return {
        "n": n,
        "status_correct": status_correct,
        "rtype_correct": rtype_correct,
        "parea_correct": parea_correct,
        "parea_denom": parea_denom,
        "total_correct": total_correct,
        "total_fields": total_fields,
        "pct_overall": pct_all,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py predicted.csv expected.csv")
        sys.exit(1)

    predicted = sys.argv[1]
    expected = sys.argv[2]

    if not Path(predicted).exists():
        print(f"Error: predicted file not found: {predicted}")
        sys.exit(1)
    if not Path(expected).exists():
        print(f"Error: expected file not found: {expected}")
        sys.exit(1)

    run_evaluation(predicted, expected)
