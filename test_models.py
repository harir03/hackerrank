"""Test which models have quota remaining."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models_to_try = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

for model_name in models_to_try:
    try:
        r = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: OK",
            config={"temperature": 0},
        )
        print(f"  [OK] {model_name}: {r.text.strip()}")
    except Exception as e:
        err = str(e)
        if "limit:" in err:
            # Extract limit info
            import re
            limit_match = re.search(r'limit: (\d+)', err)
            limit = limit_match.group(1) if limit_match else "?"
            print(f"  [429] {model_name}: limit={limit}")
        else:
            print(f"  [ERR] {model_name}: {type(e).__name__}")
