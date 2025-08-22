# phase3/utils/ollama_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import requests


@dataclass(frozen=True)
class ModelConfig:
    """Minimal model config for Ollama completions."""
    model: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    temperature: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
    max_tokens: Optional[int] = None  # None = let server decide
    timeout: int = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class OllamaError(RuntimeError):
    pass


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        # /api/generate returns a JSON object (non-stream mode)
        return r.json()
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ollama request failed: {e}") from e
    except ValueError as e:
        # JSON parse error
        raise OllamaError(f"Ollama invalid JSON response: {e}") from e


def chat_completion(
    prompt: str,
    cfg: Optional[ModelConfig] = None,
    system: Optional[str] = None,
) -> str:
    """
    Call Ollama /api/generate (non-streaming) and return the 'response' text.
    Falls back to a simple deterministic stub if Ollama is not reachable.
    """
    cfg = cfg or ModelConfig()
    url = cfg.host.rstrip("/") + "/api/generate"

    payload = {
        "model": cfg.model,
        "prompt": _build_prompt(prompt, system),
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }
    if cfg.max_tokens is not None:
        payload["options"]["num_predict"] = int(cfg.max_tokens)

    try:
        data = _post_json(url, payload, timeout=cfg.timeout)
        txt = data.get("response", "")
        return txt.strip()
    except OllamaError:
        # Graceful fallback so your dashboard doesn't crash if Ollama is down.
        return _fallback_response(prompt).strip()


def _build_prompt(user_prompt: str, system_msg: Optional[str]) -> str:
    if not system_msg:
        return user_prompt
    # Simple "system + user" composition for /api/generate
    return f"System:\n{system_msg}\n\nUser:\n{user_prompt}\nAssistant:"


def _fallback_response(prompt: str) -> str:
    # Deterministic stub: summarize minimally if model is unavailable.
    # Keeps the UI working (with a clear note).
    head = "⚠️ (Ollama unavailable) Heuristic summary:"
    # Pick first ~3 lines or bullets from user prompt as a naive "summary".
    lines = [ln.strip("-• ").strip() for ln in prompt.splitlines() if ln.strip()]
    if not lines:
        return f"{head} No content."
    lines = lines[:3]
    bullets = "\n".join(f"- {ln}" for ln in lines)
    return f"{head}\n{bullets}"