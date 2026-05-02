"""Diagnose: classify one ticket and show raw API response."""
import sys
import os
sys.path.insert(0, "code")
from dotenv import load_dotenv
load_dotenv(".env")

from pipeline.classify import SYSTEM_PROMPT, _build_user_message
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

msg = _build_user_message(
    "I lost access to my Claude team workspace after our IT admin removed my seat.",
    "Claude access lost",
    "Claude",
    "I lost access to my Claude team workspace after our IT admin removed my seat.",
)

full_prompt = SYSTEM_PROMPT + "\n\n" + msg

print("=== SENDING TO GEMINI ===")
print(f"Prompt length: {len(full_prompt)} chars")
print()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=full_prompt,
    config={"temperature": 0},
)

raw = response.text
print("=== RAW RESPONSE ===")
print(repr(raw[:500]))
print()
print("=== FORMATTED ===")
print(raw[:500])
print()

# Try parsing
import json
cleaned = raw.strip().strip("```json").strip("```").strip()
print("=== AFTER CLEANING ===")
print(repr(cleaned[:300]))
print()

try:
    data = json.loads(cleaned)
    print("=== PARSED OK ===")
    for k, v in data.items():
        print(f"  {k}: {v}")
except json.JSONDecodeError as e:
    print(f"=== JSON PARSE FAILED ===")
    print(f"Error: {e}")
    print(f"Cleaned text: {cleaned[:200]}")
