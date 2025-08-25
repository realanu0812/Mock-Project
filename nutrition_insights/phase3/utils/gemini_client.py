# phase3/utils/gemini_client.py
from typing import Optional
import os
import requests

# --- robust .env loading ---
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional; if missing, rely on real environment vars
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"

def chat_completion(prompt: str, system: Optional[str] = None, timeout: int = 60) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ (Gemini) GEMINI_API_KEY is not set"

    # --- Load and filter protein-related evidence from the filtered corpus (RAG-style) ---
    import json
    import re
    # Use filtered corpus for better relevance
    evidence_path = os.path.join(os.path.dirname(__file__), "../../data/corpus_filtered.jsonl")
    evidence = []
    try:
        with open(evidence_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    evidence.append(item)
                except Exception:
                    continue
    except Exception:
        evidence = []

    # --- RAG-style retrieval: match protein + user query keywords ---
    query_keywords = set(re.findall(r"\w+", prompt.lower()))
    protein_evidence = []
    reddit_evidence = []
    for item in evidence:
        text = item.get("text", "")
        source_type = item.get("source_type", "")
        # Match for protein, whey, and timing keywords
        timing_keywords = {"night", "sleep", "timing", "evening", "pre-sleep"}
        is_protein = "protein" in text.lower()
        is_whey = "whey" in text.lower()
        is_timing = any(kw in text.lower() for kw in timing_keywords)
        # Score by overlap with query keywords and explicit matching for whey/timing
        score = 0
        if len(query_keywords) > 1:
            score = sum(1 for kw in query_keywords if kw in text.lower())
        else:
            score = 1 if is_protein or is_whey else 0
        # Boost score if whey and timing are both present
        if is_whey and is_timing:
            score += 2
        if is_protein or is_whey:
            if source_type.startswith("reddit") or source_type.startswith("blog"):
                reddit_evidence.append((score, text))
            else:
                protein_evidence.append((score, text))

    # Sort by score (most relevant first)
    protein_evidence.sort(reverse=True, key=lambda x: x[0])
    reddit_evidence.sort(reverse=True, key=lambda x: x[0])
    top_evidence = [t for s, t in protein_evidence if s > 0]
    top_reddit = [t for s, t in reddit_evidence if s > 0]

    # Fallback: if no direct match, show general protein evidence
    if not top_evidence:
        # Try to find any protein evidence mentioning timing, sleep, or night
        timing_keywords = {"night", "sleep", "timing", "evening", "pre-sleep"}
        timing_evidence = [t for s, t in protein_evidence if any(kw in t.lower() for kw in timing_keywords)]
        if timing_evidence:
            top_evidence = timing_evidence[:6]
        else:
            top_evidence = [t for s, t in protein_evidence[:6]]

    # Fallback for reddit/blog evidence
    if not top_reddit:
        timing_keywords = {"night", "sleep", "timing", "evening", "pre-sleep"}
        timing_reddit = [t for s, t in reddit_evidence if any(kw in t.lower() for kw in timing_keywords)]
        if timing_reddit:
            top_reddit = timing_reddit[:6]
        else:
            top_reddit = [t for s, t in reddit_evidence[:6]]

    # If still no evidence, fallback
    if not top_evidence and not top_reddit:
        return "⚠️ (Gemini) No protein-related evidence found in the filtered data corpus. Please check your query or try again later."

    # Build context for Gemini prompt
    context_parts = []
    if top_evidence:
        context_parts.append("Journal/Verified Evidence:\n" + "\n---\n".join(top_evidence[:8]))
    if top_reddit:
        context_parts.append("Community Buzz (Reddit/Blogs):\n" + "\n---\n".join(top_reddit[:4]))
    context_snippets = "\n\n".join(context_parts)
    user_text = f"Context (protein evidence from data corpus):\n{context_snippets}\n\nUser question: {prompt}"
    if system:
        user_text = f"System: {system}\n{user_text}"

    # If context is too short, fallback to a user-friendly message
    if len(context_snippets.strip()) < 40:
        # Show raw evidence for transparency
        return f"⚠️ (Gemini) No sufficient protein-related evidence for a summary. Here are raw protein evidence snippets:\n\n{context_snippets}"
    body = {"contents": [{"role": "user", "parts": [{"text": user_text}]}]}

    try:
        resp = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Gemini sometimes returns out-of-scope or irrelevant answers; filter them
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if "out of scope" in result.lower() or "not related" in result.lower():
            return "⚠️ (Gemini) No relevant protein-related summary could be generated. Please check your context."
        return result
    except Exception as e:
        return f"⚠️ (Gemini API error) {e}"