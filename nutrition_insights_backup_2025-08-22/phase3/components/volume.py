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
    st.subheader("Volume & Relationships")

    d = _filter(df, source_filter, window_days)
    if d.empty:
        st.info("No data in the selected window or source.")
        return

    # --- Volume over time (daily counts)
    if "date" in d.columns and not d["date"].isna().all():
        tdf = (
            d[["date"]]
            .assign(date=lambda x: pd.to_datetime(x["date"], utc=True).dt.tz_convert("UTC").dt.date)
            .groupby("date")
            .size()
            .rename("count")
            .reset_index()
        )
        fig_ts = time_series(tdf, x="date", y="count", title="Mentions per day")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.caption("No timestamp column available to plot volume.")

    # --- Co-occurrence heatmap & bubble
    tok_df = _explode_tokens(d)
    pair_df = _cooccurrence(tok_df, min_pair_count=2)
    from utils.common import PROTEIN_WORDS
    pair_df = pair_df[
        pair_df["a"].isin(PROTEIN_WORDS) & pair_df["b"].isin(PROTEIN_WORDS)
    ].reset_index(drop=True)

    tabs = st.tabs(["Heatmap (Top 20 pairs)", "Bubble (Top 50 pairs)"])
    with tabs[0]:
        if pair_df.empty:
            # Fallback: show bar chart of protein term frequencies
            term_counts = {k: 0 for k in PROTEIN_WORDS}
            for text in d["__text__"].dropna():
                for k in PROTEIN_WORDS:
                    if k in text.lower():
                        term_counts[k] += 1
            filtered_counts = {k: v for k, v in term_counts.items() if v > 0}
            if filtered_counts:
                st.subheader("Protein Term Frequency")
                freq_df = pd.DataFrame(list(filtered_counts.items()), columns=["Term", "Count"])
                st.bar_chart(freq_df.set_index("Term"))
            else:
                st.caption("No protein-related terms found in the data.")
        else:
            top = pair_df.head(20)
            pivot = top.pivot(index="a", columns="b", values="count").fillna(0)
            fig_hm = heatmap(pivot, x_labels=pivot.columns, y_labels=pivot.index, title="Keyword co-occurrence")
            st.plotly_chart(fig_hm, use_container_width=True)

    with tabs[1]:
        if pair_df.empty:
            st.caption("Not enough repeated token pairs to build a bubble chart.")
        else:
            top = pair_df.head(50).copy()
            top["pair"] = top["a"] + " + " + top["b"]
            fig_b = bubble(top, x="pair", y="count", size="count", title="Co-occurrence bubbles")
            st.plotly_chart(fig_b, use_container_width=True)