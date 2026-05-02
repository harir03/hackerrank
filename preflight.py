"""Pre-flight checklist — run from orch/ root."""
import sys
sys.path.insert(0, "code")

# Check 1: corpus loads
from pipeline.retrieve import _ensure_loaded
_ensure_loaded()
print("[OK] Corpus loads from data/")

# Check 2: prompts exist
from pathlib import Path
assert Path("code/prompts/classifier.txt").exists()
assert Path("code/prompts/generator.txt").exists()
print("[OK] Both prompt files exist")

# Check 3: env var readable
import os
from dotenv import load_dotenv
load_dotenv(".env")
key = os.getenv("ANTHROPIC_API_KEY", "")
if key.startswith("sk-ant"):
    print("[OK] ANTHROPIC_API_KEY loaded")
else:
    print("[WARN] ANTHROPIC_API_KEY not found or wrong format")

# Check 4: output directory writable
import pandas as pd
pd.DataFrame({"test": [1]}).to_csv("support_tickets/output.csv", index=False)
print("[OK] support_tickets/output.csv is writable")

# Check 5: input file has 29 rows
from utils.csv_io import read_input_csv
rows = read_input_csv("support_tickets/support_tickets.csv")
assert len(rows) == 29, f"Expected 29 rows, got {len(rows)}"
print(f"[OK] Input has {len(rows)} rows")

print()
print("All pre-flight checks passed. Ready for Phase D.")
