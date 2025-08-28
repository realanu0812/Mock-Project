# phase3/utils/data.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List
import json

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# Import the helpers you already have in utils/common.py
from utils.common import (
    utc_now,
    coerce_utc_series,
    dataset_counts,
)

# ---------------------------------------------------------------------
# Global hashable utility for dict/list/set fields
# ---------------------------------------------------------------------
import json
def force_hashable(val):
    """
    Recursively convert dict/list/set values to JSON strings for hash/caching safety.
    Use on any record or DataFrame before caching or writing.
    """
    if isinstance(val, dict):
        return {k: force_hashable(v) for k, v in val.items()}
    elif isinstance(val, (list, set)):
        return json.dumps([force_hashable(x) for x in val], ensure_ascii=False)
    else:
        return val

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _stringify_unhashables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/dict/set cells into JSON strings so Streamlit can hash the DF for caching.
    Run this after column normalization.
    """
    if df is None or df.empty:
        return df
    for col in df.columns:
        ser = df[col]
        if ser.dtype == "object":
            # Only transform unhashable container types
            if ser.map(lambda x: isinstance(x, (list, dict, set))).any():
                df[col] = ser.map(
                    lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
                    if isinstance(x, (list, dict, set)) else x
                )
    return df


def _coalesce_cols(df: pd.DataFrame, cols: List[str], default: Optional[object] = None) -> pd.Series:
    """
    Row-wise coalesce across the first existing columns in `cols`.
    Returns the first non-null value per row. If none, uses `default`.
    """
    existing = [c for c in cols if c in df.columns]
    if not existing:
        # Return a Series indexed like df with the default scalar value
        return pd.Series(default, index=df.index)
    # Back-fill across columns so each row's first non-null "moves" left
    s = df[existing].bfill(axis=1).iloc[:, 0]
    s = s.infer_objects(copy=False)
    if default is not None:
        s = s.fillna(default)
    return s


# ---------------------------------------------------------------------
# File discovery & reading
# ---------------------------------------------------------------------
def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def _read_any_json(path: Path) -> pd.DataFrame:
    """
    Read a JSON/JSONL file into a DataFrame.
    Tries JSON Lines (orient=records) first; falls back to normal JSON (list/dict) or pandas auto.
    """
    if not path or not path.exists():
        return pd.DataFrame()

    # Try JSON Lines quickly
    try:
        with path.open("r", encoding="utf-8") as fh:
            head = []
            for _ in range(5):
                try:
                    head.append(next(fh))
                except StopIteration:
                    break
        if any(x.strip().startswith("{") for x in head):
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                # Not jsonl; fall through to other strategies
                pass
    except Exception:
        pass

    # Try as normal JSON (list or dict)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # common shapes: {"records":[...]}, {"data":[...]}, {"items":[...]}
            for key in ("records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])  # single object
        elif isinstance(data, list):
            return pd.DataFrame(data)
    except Exception:
        pass

    # Last resort: let pandas guess
    try:
        return pd.read_json(path, lines=False)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------
# Normalization to tidy schema: source, kind, text, date, url
# ---------------------------------------------------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the heterogeneous combined dataset into:
        ['source', 'kind', 'text', 'date', 'url']
    """
    out = (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).copy()

    # --- PATCH: stringify unhashables in all columns before column selection ---
    out = _stringify_unhashables(out)
    # --- END PATCH ---

    # source (string)
    out["source"] = (
        _coalesce_cols(out, ["source"], default="unknown")
        .astype(str)
        .str.strip()
    )

    # kind/source_type (string)
    out["kind"] = (
        _coalesce_cols(out, ["source_type"], default="")
        .astype(str)
        .str.strip()
    )

    # text/content (string). Your merged file already puts everything under 'combined_text'.
    out["text"] = (
        _coalesce_cols(out, ["combined_text", "text", "content", "body", "abstract", "selftext", "description"], default="")
        .astype(str)
        .str.strip()
    )

    # url/permalink (string)
    out["url"] = (
        _coalesce_cols(out, ["url", "permalink", "link"], default="")
        .astype(str)
        .str.strip()
    )

    # date → UTC-aware datetime
    raw_date = _coalesce_cols(out, ["published_at", "date", "created_utc"], default=pd.NaT)
    out["date"] = coerce_utc_series(raw_date)

    # Keep only the columns the dashboard expects (ensure they exist)
    cols = ["source", "kind", "text", "date", "url"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


# ---------------------------------------------------------------------
# Public load helpers
# ---------------------------------------------------------------------
def load_data(data_dir: Path | str) -> pd.DataFrame:
    """
    Load the unified dataset. Priority order:
      1) combined.jsonl / combined.json
      2) individual sources: reddit.json(l), journals.json(l), blogs.json(l)
    Returns a normalized DataFrame with columns: ['source','kind','text','date','url'].
    """
    data_dir = Path(data_dir)

    # PATCH: Only use combined.json(l) as dashboard source
    combined = _first_existing([
        data_dir / "corpus_filtered.jsonl",
    ])
    if combined:
        df = _read_any_json(combined)
        return _ensure_columns(df)

    dfs: List[pd.DataFrame] = []
    for name in ("reddit", "journals", "blogs"):
        p = _first_existing([data_dir / f"{name}.jsonl", data_dir / f"{name}.json"])
        if p:
            dfs.append(_read_any_json(p))

    if not dfs:
        return _ensure_columns(pd.DataFrame())

    df = pd.concat(dfs, ignore_index=True)
    return _ensure_columns(df)


def load_meta(data_dir: Path | str) -> dict:
    """
    Load meta information if present. Tries:
      - corpus_meta.json
      - index_info.json
      - merge_state.json
    Returns a small dict with counts and timestamps even if files are missing.
    """
    data_dir = Path(data_dir)
    meta: dict = {}

    for fn in ("corpus_meta.json", "index_info.json", "merge_state.json"):
        p = data_dir / fn
        if p.exists():
            try:
                meta[fn] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                meta[fn] = {}

    # Lightweight roll-up
    df = load_data(data_dir)
    counts = dataset_counts(df)
    meta["rollup"] = {
        "total": counts.total,
        "reddit": counts.reddit,
        "journals": counts.journals,
        "blogs": counts.blogs,
        "last_loaded_utc": utc_now().isoformat(),
    }
    # last document date if available
    if "date" in df.columns and len(df):
        last_dt = pd.to_datetime(df["date"], utc=True, errors="coerce").max()
        if pd.notna(last_dt):
            meta["rollup"]["max_doc_date_utc"] = str(last_dt.to_pydatetime())
    return meta


# ---------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------
def filter_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Return rows newer than now-`days` (UTC). Keeps rows with NaT out.
    Uses tz-aware comparisons to avoid tz_localize errors.
    """
    if df is None or not len(df) or "date" not in df.columns:
        return df if df is not None else pd.DataFrame()
    cutoff = pd.Timestamp(utc_now()).tz_convert("UTC") - pd.Timedelta(days=days)
    dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df[dt >= cutoff].copy()