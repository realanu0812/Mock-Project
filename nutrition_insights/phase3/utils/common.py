
# phase3/utils/common.py
from __future__ import annotations

# ---------------------------
# Export protein keywords for components
# ---------------------------
def protein_keywords() -> set[str]:
    """Return set of protein-related keywords."""
    return set(PROTEIN_WORDS)

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import html
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd


# ---------------------------
# App metadata
# ---------------------------
APP_TITLE: str = "ProteinScope"
APP_VERSION: str = "v0.1.0"


# ---------------------------
# Time helpers
# ---------------------------
def utc_now() -> datetime:
    """Timezone-aware 'now' in UTC."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | pd.Timestamp | None) -> Optional[datetime]:
    """Coerce a datetime/Timestamp to timezone-aware UTC (keeps None)."""
    if dt is None:
        return None
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def coerce_utc_scalar(x) -> Optional[datetime]:
    """Best-effort convert a scalar to UTC-aware datetime (or None)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return ensure_utc(pd.to_datetime(x, utc=True))
    except Exception:
        return None


def coerce_utc_series(s: pd.Series) -> pd.Series:
    """Convert a Series to UTC-aware datetimes; invalids → NaT[UTC]."""
    try:
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(s)), utc=True)


# ---------------------------
# Paths & data dirs
# ---------------------------
def data_dir_default(candidates: Sequence[Path]) -> Optional[Path]:
    """
    Return the first existing path from candidates (ignores blanks).
    Honors env var NI_DATA_DIR if present and valid.
    """
    env = os.environ.get("NI_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if p.exists():
            return p
    return None


# ---------------------------
# Text helpers
# ---------------------------
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)?")  # words incl. simple hyphen/' cases
_TAG = re.compile(r"<[^>]+>")

def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    if not isinstance(text, str):
        return ""
    return html.unescape(_TAG.sub(" ", text))

def normalize_text(text: str) -> str:
    """Lowercase, strip HTML, squash whitespace."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", strip_html(text).lower()).strip()

def tokenize(text: str) -> list[str]:
    """Simple alnum tokenizer suitable for keyword counting."""
    if not isinstance(text, str):
        return []
    return _TOKEN.findall(normalize_text(text))


# ---------------------------
# Pretty/format helpers
# ---------------------------
def pretty_int(n: int | float | None) -> str:
    """Human friendly integers: 123_456 → '123.5k', None → '—'."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "—"
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n < 1_000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n/1_000:.1f}k"
    if n < 1_000_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n/1_000_000_000:.1f}B"


# ---------------------------
# Light data utilities many components expect
# ---------------------------
@dataclass(frozen=True)
class SourceCounts:
    reddit: int = 0
    journals: int = 0
    blogs: int = 0
    total: int = 0


def dataset_counts(df: pd.DataFrame) -> SourceCounts:
    """
    Compute simple per-source counts. Expects a 'source' column with values like:
    'reddit' | 'journals' | 'blogs' (case-insensitive).
    """
    if df is None or not len(df):
        return SourceCounts()

    s = df.get("source")
    if s is None:
        return SourceCounts(total=len(df))

    s = s.astype(str).str.lower()
    return SourceCounts(
        reddit=int((s == "reddit").sum()),
        journals=int((s == "journals").sum()),
        blogs=int((s == "blogs").sum()),
        total=len(df),
    )


# ---------------------------
# Small guards used by components
# ---------------------------
PROTEIN_WORDS = {
    "protein", "whey", "casein", "leucine", "bcaa", "eaa",
    "collagen", "amino", "pea", "soy", "isolate", "concentrate",
    "hydrolysate", "shake", "powder", "bar",
    "egg", "animal", "plant", "animal protein", "plant protein", "amino acids"
}

def looks_protein_related(text: str) -> bool:
    """Quick heuristic; components/services may use a stricter router."""
    return bool(set(tokenize(text)).intersection(PROTEIN_WORDS))