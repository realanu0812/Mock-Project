# phase3/components/trending.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from collections import Counter
from typing import Iterable, List, Dict
import json

from utils.common import tokenize
from utils.charts import bar

TEXT_COLS = ("title", "text", "summary", "content")

def _gather_text(df: pd.DataFrame) -> Iterable[str]:
    for col in TEXT_COLS:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            for v in s:
                if v:
                    yield v

def _keyword_counts(df: pd.DataFrame, keywords: List[str]) -> Dict[str, int]:
    """Count exact keyword matches using our tokenizer (lowercase, simple word tokens)."""
    if df is None or df.empty or not keywords:
        return {}
    kw = [k.strip().lower() for k in keywords if k and isinstance(k, str)]
    kw_set = set(kw)
    counts = Counter()
    for txt in _gather_text(df):
        toks = tokenize(txt)
        # only count tokens that are in the keyword set
        for t in toks:
            if t in kw_set:
                counts[t] += 1
    return dict(counts)

def render(df: pd.DataFrame, keywords: List[str], source_filter: str, meta=None, top_n: int = 15) -> None:
    # Ensure all columns are hashable before caching
    if df is not None and not df.empty:
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict, set)) else x)
    """
    Trending Topics:
      - Keyword frequency bar chart (protein-only keywords)
      - Top-N table
    """
    st.subheader("Trending Topics")

    if df is None or df.empty:
        st.info("No trending topic data available. Please check your filters or data source.")
        return

    # Optional local source filter (app may already filter; this keeps it robust)
    if source_filter and source_filter != "All" and "source" in df.columns:
        df = df[df["source"].str.lower() == source_filter.lower()]

    counts = _keyword_counts(df, keywords)
    if not counts:
        st.warning("No trending keywords found for the current filters or keyword list.")
        return

    # Build Top-N dataframe
    freq_df = (
        pd.DataFrame(
            sorted(counts.items(), key=lambda x: x[1], reverse=True),
            columns=["keyword", "count"],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    # Chart + table side-by-side
    c1, c2 = st.columns([3, 2])

    with c1:
        bar(
            data=freq_df,
            x="keyword",
            y="count",
            title="Keyword Frequency (Top)",
            height=360,
        )

    with c2:
        st.markdown("**Top Keywords**")
        st.dataframe(
            freq_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Keywords are matched on tokenized text from titles/summaries/content. "
        "Only protein-related terms supplied in the config are counted."
    )

def render_trending():
    # Load filtered corpus for trending
    data_path = "../data/corpus_filtered.jsonl"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        df = pd.DataFrame(records)
    except Exception as e:
        st.error(f"❌ Failed to load filtered corpus: {e}")
        return