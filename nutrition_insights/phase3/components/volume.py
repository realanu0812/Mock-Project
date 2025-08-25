# phase3/components/volume.py
from __future__ import annotations

import itertools
import pandas as pd
import streamlit as st

from utils.charts import time_series, heatmap, bubble
from utils.common import tokenize


def _filter(df: pd.DataFrame, source_filter: str, window_days: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    # window
    if "date" in d.columns:
        try:
            cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=window_days)
        except TypeError:
            # already tz-aware
            cutoff = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=window_days)
        d = d[d["date"] >= cutoff]

    # source
    if source_filter and source_filter != "All" and "source" in d.columns:
        d = d[d["source"].str.lower() == source_filter.lower()]

    # canon text
    txt = d.get("text")
    if txt is None:
        txt = d.get("abstract")
    if txt is None:
        d["__text__"] = ""
    else:
        d["__text__"] = txt.fillna("").astype(str)

    return d.reset_index(drop=True)


def _explode_tokens(d: pd.DataFrame, max_tokens_per_doc: int = 40) -> pd.DataFrame:
    if d.empty:
        return d
    rows = []
    for _, r in d.iterrows():
        toks = tokenize(r["__text__"])
        # keep it light to avoid skew from very long threads
        toks = toks[:max_tokens_per_doc]
        rows.append({"date": r.get("date"), "tokens": toks})
    return pd.DataFrame(rows)


def _cooccurrence(df_tokens: pd.DataFrame, min_pair_count: int = 3) -> pd.DataFrame:
    """Build co-occurrence counts per doc (unordered pairs)."""
    if df_tokens.empty:
        return pd.DataFrame(columns=["a", "b", "count"])

    from collections import Counter
    ctr = Counter()
    for toks in df_tokens["tokens"]:
        uniq = sorted(set(toks))
        for a, b in itertools.combinations(uniq, 2):
            ctr[(a, b)] += 1

    data = [{"a": k[0], "b": k[1], "count": v} for k, v in ctr.items() if v >= min_pair_count]
    return pd.DataFrame(data).sort_values("count", ascending=False).reset_index(drop=True)


def render(df: pd.DataFrame, source_filter: str, window_days: int) -> None:
    st.subheader("Monthly Post Volume by Source")

    d = _filter(df, source_filter, window_days)
    # Remove any rows with year >= 2026 or after today (typo/invalid future data)
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        today = pd.Timestamp.utcnow().tz_convert("UTC")
        d = d[(d["date"].dt.year < 2026) & (d["date"] <= today)]
    if d.empty:
        st.info("No data in the selected window or source.")
        return

    # --- Monthly post volume for each source ---
    if "date" in d.columns and "source" in d.columns:
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        d = d.dropna(subset=["date", "source"])
        d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
        sources = ["reddit", "journals", "blogs"]
        fig = None
        import plotly.express as px
        plot_df = d[d["source"].isin(sources)].groupby(["month", "source"]).size().reset_index(name="count")
        fig = px.line(plot_df, x="month", y="count", color="source", markers=True,
                     title="Monthly Post Volume by Source", template="plotly_white")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No timestamp/source column available to plot monthly volume.")

    # --- Word Cloud (protein keywords only) ---
    from utils.wordcloud_utils import render_wordcloud
    from utils.config_loader import protein_keywords
    pkws = set([k.lower() for k in protein_keywords()])
    # Only keep tokens that are protein keywords
    def filter_text(text):
        import re
        words = re.findall(r"\w+", str(text).lower())
        return " ".join([w for w in words if w in pkws])
    d["_protein_text_"] = d["__text__"].apply(filter_text)
    st.markdown("#### Protein Keyword Word Cloud")
    render_wordcloud(d, text_col="_protein_text_", title="Protein Keyword Word Cloud")