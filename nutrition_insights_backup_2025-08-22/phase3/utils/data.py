# phase3/utils/data.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import json

import pandas as pd
from typing import List

def _coalesce_cols(df: pd.DataFrame, cols: List[str], default: Optional[object] = None) -> pd.Series:
    """
    Row-wise coalesce across first existing columns in `cols`.
    Returns the first non-null value per row. If none, uses `default`.
    """
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(default, index=df.index)
    # bfill across columns, then take first column (now each row is the first non-na)
    s = df[existing].bfill(axis=1).iloc[:, 0]
    if default is not None:
        s = s.fillna(default)
    return s

from utils.common import (
    utc_now,
    coerce_utc_series,
    dataset_counts,
)


# ---------------------------
# File discovery
# ---------------------------
def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def _read_any_json(path: Path) -> pd.DataFrame:
    """
    Read a JSON/JSONL file into a DataFrame.
    Tries jsonlines (orient=records) first; falls back to pandas reader.
    """
    if not path or not path.exists():
        return pd.DataFrame()

    # Try JSON Lines quickly
    try:
        # Heuristic: if any line starts with "{" -> assume jsonl of objects.
        with path.open("r", encoding="utf-8") as fh:
            head = [next(fh) for _ in range(5)]
        if any(x.strip().startswith("{") for x in head):
            # If it’s a single big array, pandas will also handle; otherwise we try line-wise.
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                pass
    except StopIteration:
        # empty file
        return pd.DataFrame()
    except Exception:
        pass

    # Try as normal JSON (list of dicts)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # common shapes: {"records":[...]}, {"data":[...]}
            for key in ("records", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            # else: single dict -> one row
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
    except Exception:
        pass

    # Last resort: let pandas guess
    try:
        return pd.read_json(path, lines=False)
    except Exception:
        return pd.DataFrame()


# ---------------------------
# Data loading & normalization
# ---------------------------
def _pick_text_col(df: pd.DataFrame) -> pd.Series:
    """Choose the best available text/content column."""
    for col in ("text", "content", "body", "abstract", "selftext", "description"):
        if col in df.columns:
            return df[col]
    # build from parts if nothing found
    title = df.get("title", "")
    summary = df.get("summary", "")
    return (title.astype(str) + " " + summary.astype(str)).str.strip()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the heterogenous combined dataset into a tidy frame with
    columns: source, kind, text, date, url.
    """
    out = df.copy()

    # source
    out["source"] = (
        _coalesce_cols(out, ["source", "src", "channel"], default="unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # kind / type
    out["kind"] = (
        _coalesce_cols(out, ["kind", "type"], default="").astype(str).str.strip().str.lower()
    )

    # text-like content (order gives precedence)
    out["text"] = (
        _coalesce_cols(
            out,
            ["text", "title", "selftext", "abstract", "content", "summary", "body"],
            default="",
        )
        .astype(str)
        .str.strip()
    )

    # url/permalink
    out["url"] = (
        _coalesce_cols(out, ["url", "link", "permalink"], default="")
        .astype(str)
        .str.strip()
    )

    # date (several possible fields), then coerce to UTC
    raw_date = _coalesce_cols(
        out,
        ["date", "created_utc", "published", "pub_date", "time", "created", "timestamp"],
        default=pd.NaT,
    )
    out["date"] = coerce_utc_series(raw_date)

    # keep only the columns the dashboard expects
    cols = ["source", "kind", "text", "date", "url"]
    # make sure they exist even if empty
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def load_data(data_dir: Path) -> pd.DataFrame:
    """
    Load the unified dataset. Priority order:
      1) combined.json (or .jsonl)
      2) individual sources: reddit.json, journals.json, blogs.json
    """
    data_dir = Path(data_dir)

    combined = _first_existing([
        data_dir / "combined.jsonl",
        data_dir / "combined.json",
    ])
    if combined:
        df = _read_any_json(combined)
        return _ensure_columns(df)

    dfs = []
    for name in ("reddit", "journals", "blogs"):
        p = _first_existing([data_dir / f"{name}.jsonl", data_dir / f"{name}.json"])
        if p:
            dfs.append(_read_any_json(p))
    if not dfs:
        return _ensure_columns(pd.DataFrame())
    return _ensure_columns(pd.concat(dfs, ignore_index=True))


def load_meta(data_dir: Path) -> dict:
    """
    Load meta information if present. Tries:
      - corpus_meta.json
      - index_info.json
      - merge_state.json
    Returns a small dict with counts and timestamps even if files are missing.
    """
    data_dir = Path(data_dir)
    meta = {}

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


# ---------------------------
# Filtering helpers
# ---------------------------
def filter_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return rows newer than now-`days` (UTC). Keeps rows with NaT out."""
    if df is None or not len(df) or "date" not in df.columns:
        return df if df is not None else pd.DataFrame()
    cutoff = pd.Timestamp(utc_now()).tz_convert("UTC") - pd.Timedelta(days=days)
    dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df[dt >= cutoff].copy()