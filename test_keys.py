"""Test each key individually."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")

for i in range(1, 6):
    key = os.getenv(f"GEMINI_API_KEY_{i}", "")
    if not key or key.startswith("PASTE_"):
        print(f"  Key {i}: SKIPPED (placeholder)")
        continue
    try:
        c = genai.Client(api_key=key)
        r = c.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="Say OK",
            config={"temperature": 0},
        )
        print(f"  Key {i}: OK ({key[:12]}...)")
    except Exception as e:
        if "429" in str(e):
            print(f"  Key {i}: EXHAUSTED ({key[:12]}...)")
        else:
            print(f"  Key {i}: ERROR - {type(e).__name__}")
