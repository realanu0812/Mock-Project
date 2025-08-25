# phase3/utils/gemini_client.py
from typing import Optional
import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

print(f"[Gemini] Loaded API key: {GEMINI_API_KEY[:6]}... (length: {len(GEMINI_API_KEY)})")


def chat_completion(prompt: str, system: Optional[str] = None) -> str:
    """
    Call Gemini API and return the response text.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    if system:
        # Gemini does not have a system prompt, but you can prepend it to the user prompt
        data["contents"][0]["parts"][0]["text"] = f"System: {system}\nUser: {prompt}"
    params = {"key": GEMINI_API_KEY}
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        # Extract the generated text
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[Gemini] Error details: {e}\nResponse: {getattr(e, 'response', None)}")
        return f"⚠️ (Gemini API error) {e}"
