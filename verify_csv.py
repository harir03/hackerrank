"""Verify CSV column handling."""
import sys
sys.path.insert(0, "code")
from utils.csv_io import read_input_csv

rows = read_input_csv("support_tickets/support_tickets.csv")
print(f"Rows: {len(rows)}")
print(f"Keys: {list(rows[0].keys())}")
print(f"First issue: {rows[0]['issue'][:80]}")
print(f"First company: {rows[0]['company']}")
