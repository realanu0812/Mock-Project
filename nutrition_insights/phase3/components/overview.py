# phase3/components/overview.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from utils.common import pretty_int, SourceCounts, ensure_utc


# Helper to infer SourceCounts from a DataFrame, for backward compatibility
def _infer_counts(df: pd.DataFrame) -> SourceCounts:
    try:
        total = int(len(df))
    except Exception:
        total = 0
    src = (
        df.get("source")
        .astype(str)
        .str.strip()
        .str.lower()
        if isinstance(df, pd.DataFrame) and "source" in df.columns
        else pd.Series([], dtype=str)
    )
    # normalize common labels
    reddit = int((src == "reddit").sum())
    journals = int((src.isin(["pubmed", "journals", "journal"])).sum())
    blogs = int((src == "blogs").sum())
    return SourceCounts(total=total, reddit=reddit, journals=journals, blogs=blogs)

def _card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid #eee; border-radius:12px; padding:14px; text-align:center;
            background:rgba(0,0,0,0.02);">
            <div style="font-size:0.9rem; color:#666;">{label}</div>
            <div style="font-size:1.6rem; font-weight:700; margin-top:4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _last_updated(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return "—"
    lu = meta.get("last_updated") or meta.get("last_run")
    if not lu:
        return "—"
    try:
        dt = ensure_utc(pd.to_datetime(lu, utc=True))
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(lu)

def render(df: pd.DataFrame, counts: SourceCounts, meta: Optional[Dict[str, Any]]) -> None:
    st.subheader("Data Overview")

    # --- Refresh Button ---
    MERGE_PATH = Path(__file__).parent.parent.parent / "merge_scrapper.py"
    last_run = None
    if meta and (meta.get("last_updated") or meta.get("last_run")):
        try:
            last_run = pd.to_datetime(meta.get("last_updated") or meta.get("last_run"), utc=True)
        except Exception:
            last_run = None
    refresh_threshold = timedelta(hours=12)
    now = datetime.utcnow()
    needs_refresh = not last_run or (now - last_run.to_pydatetime()) > refresh_threshold
    if st.button("🔄 Refresh Data", help="Run scrappers if last update > 12h ago"):
        if needs_refresh:
            st.info("Running data scrappers (merge_scrapper.py)...")
            result = subprocess.run(["python", str(MERGE_PATH)], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Data refreshed! Please manually rerun the app or reload the page to see updated metrics.")
            else:
                st.error(f"Scrapper failed: {result.stderr}")
        else:
            st.info("Data is fresh (updated < 12h ago). No need to re-scrape.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: _card("Total Records", pretty_int(counts.total))
    with c2: _card("Reddit",        pretty_int(counts.reddit))
    with c3: _card("Journals",      pretty_int(counts.journals))
    with c4: _card("Blogs",         pretty_int(counts.blogs))

    st.caption(f"Last updated: {_last_updated(meta)}")

    with st.expander("Peek at sample rows", expanded=False):
        if df is not None and len(df):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No rows to show yet. Please check your data source or filters.")


def fix_sources(records):
    for r in records:
        url = str(r.get("url", "")).lower()
        text = str(r.get("text", "")).lower()
        # Infer source from url or text
        if "reddit.com" in url:
            r["source"] = "reddit"
        elif any(x in url for x in ["pubmed", "journal"]):
            r["source"] = "journals"
        elif any(x in url for x in ["news.google.com", "healthline", "bodybuilding.com", "barbend"]):
            r["source"] = "blogs"
        else:
            r["source"] = r.get("source", "unknown")
    return records


# Back-compat wrapper expected by app.py:
# computes SourceCounts from df and delegates to render(...).
def render_overview(df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Back-compat wrapper expected by app.py:
    computes SourceCounts from df and delegates to render(...).
    """
    # Fix sources before computing counts
    if isinstance(df, pd.DataFrame):
        df = pd.DataFrame(fix_sources(df.to_dict(orient="records")))
    counts = _infer_counts(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    render(df, counts, meta)
