# phase3/components/insights.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from textwrap import shorten
from typing import List


from services.summarizer import summarize_bullets


@st.cache_data(show_spinner=False)

def _pick_journal_rows(df: pd.DataFrame, max_items: int = 40) -> pd.DataFrame:
    """Keep only journals; prefer recent and with abstracts."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "source" in d.columns:
        d = d[d["source"].str.lower() == "journals"]
    if d.empty:
        return d
    # Light quality heuristic
    if "abstract" in d.columns:
        has_abs = d["abstract"].fillna("").str.len()
    else:
        has_abs = pd.Series([0] * len(d))
    if "title" in d.columns:
        has_ttl = d["title"].fillna("").str.len()
    else:
        has_ttl = pd.Series([0] * len(d))
    d = d.assign(_q=(has_abs > 120).astype(int) + (has_ttl > 20).astype(int))
    if "date" in d.columns:
        d = d.sort_values(["_q", "date"], ascending=[False, False])
    else:
        d = d.sort_values(["_q"], ascending=False)
    return d.head(max_items).reset_index(drop=True)


# NEW: Buzz row picker for Reddit/Blogs
@st.cache_data(show_spinner=False)
def _pick_buzz_rows(df: pd.DataFrame, max_items: int = 40) -> pd.DataFrame:
    """Keep only Reddit/Blog records; prefer recent and with content/title."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    # Accept both source and source_type
    buzz_types = ["blog_article", "reddit_post", "reddit_comment"]
    buzz_sources = ["blogs", "reddit"]
    if "source_type" in d.columns:
        d = d[d["source_type"].str.lower().isin(buzz_types)]
    elif "source" in d.columns:
        d = d[d["source"].str.lower().isin(buzz_sources)]
    if d.empty:
        return d
    # Quality: prefer longer title or combined_text/content
    if "combined_text" in d.columns:
        has_txt = d["combined_text"].fillna("").str.len()
    elif "content" in d.columns:
        has_txt = d["content"].fillna("").str.len()
    else:
        has_txt = pd.Series([0] * len(d))
    if "title" in d.columns:
        has_ttl = d["title"].fillna("").str.len()
    else:
        has_ttl = pd.Series([0] * len(d))
    d = d.assign(_q=(has_txt > 80).astype(int) + (has_ttl > 10).astype(int))
    if "date" in d.columns:
        d = d.sort_values(["_q", "date"], ascending=[False, False])
    else:
        d = d.sort_values(["_q"], ascending=False)
    return d.head(max_items).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _compose_context(rows: pd.DataFrame, max_chars: int = 8000) -> str:
    """Compact context for LLM: title + (journal) + year + 1–2 key lines."""
    parts: List[str] = []
    for _, r in rows.iterrows():
        title = str(r.get("title", "") or "").strip()
        src = str(r.get("journal", r.get("venue", "Journal")) or "").strip()
        year = ""
        if pd.notna(r.get("date")):
            try:
                year = pd.to_datetime(r["date"]).year
            except Exception:
                year = ""
        head = " - ".join([x for x in [title, src, str(year) if year else ""] if x])
        abstract = str(r.get("abstract", "") or "").strip()
        # Keep it tight — first ~2 sentences/lines
        snippet = shorten(abstract.replace("\n", " "), width=420, placeholder=" …")
        parts.append(f"- {head}\n  {snippet}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "JOURNAL EVIDENCE (curated):\n" + "\n".join(parts)


def render(df: pd.DataFrame, source_filter: str, model: str | None) -> None:
    """
    Verified Findings (Journals):
      • 5 summarized, business-relevant bullets (LLM)
      • Fallback list if LLM unavailable
    """
    st.subheader("Verified Findings (Journals)")

    if df is None or df.empty:
        st.info("No data available.")
        return

    # Respect the global source filter if user forces non-journal — but
    # this section is journals-only, so show a gentle note.
    if source_filter and source_filter != "All" and source_filter.lower() != "journals":
        st.info("Switch the source filter to **Journals** to see verified findings.")
        return

    rows = _pick_journal_rows(df, max_items=40)
    if rows.empty:
        st.warning("No recent journal items found.")
        return

    ctx = _compose_context(rows)

    prompt = (
        "You are summarizing peer-reviewed nutrition studies for product teams.\n"
        "From the JOURNAL EVIDENCE below, extract **exactly five** concise, "
        "actionable findings relevant to the protein market (formulation, claims, "
        "dosing, safety, consumer outcomes). Each bullet must:\n"
        "• start with a bold 3–6 word headline\n"
        "• be one sentence, plain language\n"
        "• avoid hype; reflect evidence level (e.g., 'small RCT', 'systematic review')\n"
        "• include an indicative direction or number if stated (e.g., '~25% higher MPS')\n"
        "Output as '- ' bullets only.\n\n"
        f"{ctx}\n"
    )

    bullets = summarize_bullets([prompt], n=5)

    if bullets and not all(b.strip().startswith("You are summarizing") or b.strip().startswith("From the JOURNAL EVIDENCE") for b in bullets):
        for b in bullets:
            with st.container(border=True):
                st.markdown(b)
        with st.expander("Show sources used"):
            # compact table of titles & links for transparency
            cols = [c for c in ["title", "url", "date"] if c in rows.columns]
            if cols:
                show = rows[cols].rename({"title": "Title", "url": "Link", "date": "Date"}, axis=1)
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.caption("No source columns available to display.")
    else:
        # Fallback: show top 5 titles as a minimal usable output
        st.warning("LLM summary unavailable — showing top journal items instead.")
        top = rows.head(5)
        for _, r in top.iterrows():
            with st.container(border=True):
                t = str(r.get("title", "Untitled"))
                link = str(r.get("url", "")) or None
                j = str(r.get("journal", r.get("venue", "")) or "")
                dt = ""
                if pd.notna(r.get("date")):
                    try:
                        dt = pd.to_datetime(r["date"]).date().isoformat()
                    except Exception:
                        dt = ""
                meta = " • ".join([x for x in [j, dt] if x])
                if link:
                    st.markdown(f"**[{t}]({link})**")
                else:
                    st.markdown(f"**{t}**")
                if meta:
                    st.caption(meta)