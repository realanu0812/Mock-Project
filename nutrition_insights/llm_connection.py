# rag/llm_connection.py
from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any, Optional

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

class LLMError(RuntimeError):
    pass

def _post_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Ollama /api/chat with stream=False so we get a single JSON object back.
    This avoids 'Extra data' JSON errors that happen if you try to json.loads() a streamed NDJSON body.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        raise LLMError(f"Ollama not reachable at {url}: {e}")
    if r.status_code != 200:
        # Often an HTML or plain-text error comes back; include a short preview for debugging.
        preview = r.text[:200].replace("\n", " ")
        raise LLMError(f"Ollama error {r.status_code}: {preview}")
    try:
        data = r.json()
    except Exception as e:
        raise LLMError(f"Failed to parse Ollama JSON response: {e}. Preview: {r.text[:200]}")
    return data

def chat_ollama(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Synchronously get a completion from Ollama /api/chat.
    Returns the assistant message string.
    """
    model = model or DEFAULT_MODEL
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,            # critical to avoid NDJSON parsing issues
        "options": {
            "temperature": temperature,
        },
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens

    data = _post_chat(payload)
    # Ollama returns {"message":{"role":"assistant","content":"..."}, ...}
    try:
        return data["message"]["content"]
    except KeyError:
        # Some models respond with "done": true and a list of messages; try to salvage
        if isinstance(data.get("messages"), list) and data["messages"]:
            for m in reversed(data["messages"]):
                if m.get("role") == "assistant" and m.get("content"):
                    return m["content"]
        raise LLMError(f"Malformed Ollama response: {json.dumps(data)[:300]}")

def ensure_model_available(model: Optional[str] = None) -> None:
    """
    Quick check that Ollama is up and that the model exists.
    """
    model = model or DEFAULT_MODEL
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        tags = r.json().get("models", [])
    except Exception as e:
        raise LLMError(f"Cannot list models from Ollama at {OLLAMA_URL}: {e}")
    names = {m.get("name") for m in tags if isinstance(m, dict)}
    if model not in names:
        raise LLMError(
            f"Model '{model}' not found on Ollama. "
            f"Run:  ollama pull {model}"
        )

def get_chat_fn(model: Optional[str] = None, temperature: float = 0.2):
    """
    Adapter used by query_cli.py — returns a callable(system, prompt)->str
    """
    m = model or DEFAULT_MODEL
    ensure_model_available(m)
    def _chat(system_prompt: str, prompt: str) -> str:
        return chat_ollama(system_prompt, prompt, model=m, temperature=temperature)
    return _chat