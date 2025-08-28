# phase3/components/overview.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Optional, Dict
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
    src_type = (
        df.get("source_type")
        .astype(str)
        .str.strip()
        .str.lower()
        if isinstance(df, pd.DataFrame) and "source_type" in df.columns
        else pd.Series([], dtype=str)
    )
    # Use both source and source_type for robust detection
    reddit = int(((src == "reddit") & (src_type == "reddit_post")).sum())
    journals = int(((src == "journals") & (src_type == "journal_article")).sum())
    blogs = int(((src == "blogs") & (src_type == "blog_article")).sum())
    return SourceCounts(total=total, reddit=reddit, journals=journals, blogs=blogs)

def _card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(255,255,255,0.25);
            border-radius:16px;
            padding:18px;
            text-align:center;
            background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            box-shadow:0 8px 22px rgba(0,0,0,0.06);
            backdrop-filter: blur(6px);
        ">
            <div style="font-size:0.85rem; color:#7a7a7a; letter-spacing:.02em;">{label}</div>
            <div style="font-size:1.8rem; font-weight:800; margin-top:6px; line-height:1;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _last_updated(meta: Optional[Dict[str, str]]) -> str:
    if not meta:
        return "—"
    lu = meta.get("last_updated") or meta.get("last_run")
    if not lu:
        return "—"
    try:
        dt = ensure_utc(pd.to_datetime(lu, utc=True))
        # Human-ish display
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(lu)

def render(df: pd.DataFrame, counts: SourceCounts, meta: Optional[Dict[str, str]]) -> None:
    # Ensure all columns are hashable before caching
    if df is not None and not df.empty:
        import json
        def force_hashable(val):
            if isinstance(val, (dict, list, set)):
                return json.dumps(val, ensure_ascii=False)
            return val
        for col in df.columns:
            df[col] = df[col].apply(force_hashable)
    st.subheader("Data Overview")
    st.markdown('<hr style="border:none;height:1px;background:linear-gradient(90deg, rgba(0,0,0,0), rgba(0,0,0,0.15), rgba(0,0,0,0));">', unsafe_allow_html=True)
    

    # --- Refresh Button ---
    # Run full pipeline: merge scrapper, filter, RAG index
    MERGE_SCRAPPER = Path(__file__).parent.parent.parent / "merge_scrapper.py"
    FILTER_CORPUS = Path(__file__).parent.parent.parent / "rag" / "filter_corpus.py"
    RAG_INDEX = Path(__file__).parent.parent.parent / "rag" / "build_index.py"
    last_run = None
    if meta and (meta.get("last_updated") or meta.get("last_run")):
        try:
            last_run = pd.to_datetime(meta.get("last_updated") or meta.get("last_run"), utc=True)
        except Exception:
            last_run = None
    refresh_threshold = timedelta(hours=12)
    now = datetime.utcnow()
    needs_refresh = not last_run or (now - last_run.to_pydatetime()) > refresh_threshold
    if st.button("🔄 Refresh Data", help="To keep the data up-to-date"):
        try:
            with st.spinner("Running merge scrapper..."):
                subprocess.run(["python", str(MERGE_SCRAPPER)], capture_output=True, text=True, check=True)
            with st.spinner("Filtering corpus..."):
                subprocess.run(["python", str(FILTER_CORPUS)], capture_output=True, text=True, check=True)
            with st.spinner("Building RAG index..."):
                subprocess.run(["python", str(RAG_INDEX)], capture_output=True, text=True, check=True)
            st.toast("Data pipeline refreshed!", icon="✅")
            st.rerun()
        except subprocess.CalledProcessError:
            st.toast("Refresh failed", icon="❌")

    st.markdown(
        """
        <div style="padding:10px 12px; border-radius:18px;
                    background:linear-gradient(180deg, rgba(0,0,0,0.04), rgba(0,0,0,0.02));">
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: _card("Total Records", pretty_int(counts.total))
    with c2: _card("Reddit",        pretty_int(counts.reddit))
    with c3: _card("Journals",      pretty_int(counts.journals))
    with c4: _card("Blogs",         pretty_int(counts.blogs))
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Last updated · {_last_updated(meta)}")


def fix_sources(records):
    # Use explicit source/source_type if present and valid
    for r in records:
        src = str(r.get("source", "")).strip().lower()
        src_type = str(r.get("source_type", "")).strip().lower()
        url = str(r.get("url", "")).lower()
        # Robust blog detection (expanded and guaranteed)
        if (
            src_type == "blog_article"
            or src in ["blogs", "blog", "blogger", "blog_article", ""]
            or "blog" in url
            or (src_type == "" and "blog" in (r.get("title", "") + r.get("text", "")).lower())
        ):
            r["source"] = "blogs"
            r["source_type"] = "blog_article"
            continue
        if src_type == "reddit_post" or (src == "reddit" and src_type == "reddit_post"):
            r["source"] = "reddit"
            r["source_type"] = "reddit_post"
            continue
        if src_type == "journal_article" or (src == "journals" and src_type == "journal_article"):
            r["source"] = "journals"
            r["source_type"] = "journal_article"
            continue
        # Fallback: infer from url/text if missing
        if "reddit.com" in url:
            r["source"] = "reddit"
            r["source_type"] = "reddit_post"
        elif any(x in url for x in ["pubmed", "journal"]):
            r["source"] = "journals"
            r["source_type"] = "journal_article"
        else:
            r["source"] = "unknown"
            r["source_type"] = "unknown"
    return records


# Back-compat wrapper expected by app.py:
# computes SourceCounts from df and delegates to render(...).
def render_overview(df: pd.DataFrame, meta: Optional[Dict[str, str]] = None) -> None:
    """
    Back-compat wrapper expected by app.py:
    computes SourceCounts from df and delegates to render(...).
    """
    # Fix sources before computing counts
    if isinstance(df, pd.DataFrame):
        df = pd.DataFrame(fix_sources(df.to_dict(orient="records")))
    counts = _infer_counts(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    render(df, counts, meta)
