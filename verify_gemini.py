"""Quick Gemini API verification using google.genai SDK."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: OK",
    config={"temperature": 0},
)
print("Response:", response.text)
