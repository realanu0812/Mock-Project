# phase3/services/query_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import math

import pandas as pd

from utils.common import (
    tokenize,
    normalize_text,
    looks_protein_related,
    ensure_utc,
)

# ----------------------------------
# Public API
# ----------------------------------

def is_in_scope(query: str) -> bool:
    """
    Guardrail: only allow nutrition/protein-like questions through.
    """
    if not isinstance(query, str) or not query.strip():
        return False
    return looks_protein_related(query)


@dataclass(frozen=True)
class Snippet:
    title: str
    source: str
    date: Optional[str]
    url: Optional[str]
    excerpt: str
    score: float


def build_context_snippets(
    df: pd.DataFrame,
    query: str,
    *,
    topn: int = 8,
    per_source_cap: int = 4,
    min_chars: int = 220,
    max_chars: int = 600,
    recency_boost: float = 0.15,  # small boost for newer items
) -> List[Snippet]:
    """
    Rank and format context snippets for the LLM.

    Expects df with columns: ['text','title','source','date','url'].
    - Scores by token overlap with query (normalized) + optional recency boost.
    - Trims to readable excerpts (min_chars..max_chars).
    - Applies a per-source cap to keep variety.
    """
    if df is None or not len(df):
        return []

    q_toks = set(tokenize(query))
    if not q_toks:
        return []

    # Ensure expected columns
    text = _safe_series(df, "text")
    title = _safe_series(df, "title")
    source = _safe_series(df, "source", fill="unknown")
    url = _safe_series(df, "url")
    date_col = _coerce_utc_series(_safe_series(df, "date"))

    # Precompute normalized text + score
    norm_text = text.map(normalize_text)
    scores = _overlap_score(norm_text, q_toks)

    if recency_boost and len(date_col):
        # Convert to numeric age (days), then boost newer (smaller age)
        ages = _age_days(date_col)
        # Normalize to 0..1 inverted (newer ≈1, older ≈0)
        inv = 1.0 - _minmax(ages)
        scores = scores + (recency_boost * inv)

    # Build preliminary table
    tmp = pd.DataFrame(
        {
            "score": scores,
            "title": title,
            "source": source.str.lower().fillna("unknown"),
            "url": url,
            "date": date_col,
            "text": text,
        }
    ).sort_values("score", ascending=False)

    tmp = tmp[tmp["score"] > 0]

    # Keep top-N with per-source cap for diversity
    out_rows: List[Snippet] = []
    counts: dict[str, int] = {}

    for _, row in tmp.iterrows():
        src = (row.get("source") or "unknown").lower()
        if counts.get(src, 0) >= per_source_cap:
            continue

        snippet_text = _make_excerpt(row.get("text") or "", q_toks, min_chars, max_chars)
        if not snippet_text:
            continue

        dt = row.get("date")
        dt_str = None
        if pd.notna(dt):
            try:
                dt_str = ensure_utc(dt).strftime("%Y-%m-%d")
            except Exception:
                dt_str = None

        # Fallback: use first 8 words of text if title is missing
        raw_title = (row.get("title") or "").strip()
        if not raw_title:
            raw_title = " ".join((row.get("text") or "").strip().split()[:8])
        out_rows.append(
            Snippet(
                title=raw_title or "(untitled)",
                source=src,
                date=dt_str,
                url=(row.get("url") or None),
                excerpt=snippet_text,
                score=float(row.get("score") or 0.0),
            )
        )
        counts[src] = counts.get(src, 0) + 1
        if len(out_rows) >= topn:
            break

    return out_rows


# ----------------------------------
# Helpers (internal)
# ----------------------------------

def _safe_series(df: pd.DataFrame, col: str, *, fill: Optional[str] = None) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)
    if fill is not None:
        return s.fillna(fill)
    return s


def _coerce_utc_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(s)), utc=True)


def _age_days(s: pd.Series) -> pd.Series:
    """Age in days (float) from now (UTC). NaT -> large age (penalize)."""
    now = pd.Timestamp.utcnow()
    delta = now - s
    days = delta.dt.total_seconds() / 86400.0
    days = days.fillna(days.max() if len(days) else 0.0)
    # negative (future dates) -> clamp to 0
    return days.clip(lower=0.0)


def _minmax(s: pd.Series) -> pd.Series:
    if s is None or not len(s):
        return pd.Series([0.0])
    lo = float(s.min())
    hi = float(s.max())
    if hi <= lo:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def _overlap_score(norm_text: pd.Series, q_toks: set[str]) -> pd.Series:
    def score_one(t: str) -> float:
        if not isinstance(t, str) or not t:
            return 0.0
        toks = set(tokenize(t))
        if not toks:
            return 0.0
        overlap = len(toks.intersection(q_toks))
        # Normalize by query size (so longer queries don't dominate)
        return overlap / max(1, len(q_toks))

    return norm_text.map(score_one).astype(float)


def _make_excerpt(
    raw: str,
    q_toks: set[str],
    min_chars: int,
    max_chars: int,
) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""

    text = " ".join(raw.split())  # squash whitespace
    if len(text) <= max_chars:
        return text

    # Try to find segment around first matched token
    lower = text.lower()
    hit_pos = -1
    for tok in q_toks:
        p = lower.find(tok)
        if p >= 0:
            hit_pos = p
            break

    if hit_pos < 0:
        # no direct hit, just take head
        return text[:max_chars].rstrip()

    half = max(min_chars // 2, 80)
    start = max(0, hit_pos - half)
    end = min(len(text), start + max_chars)
    snip = text[start:end].strip()

    # Add ellipses if clipped
    if start > 0:
        snip = "…" + snip
    if end < len(text):
        snip = snip + "…"
    return snip