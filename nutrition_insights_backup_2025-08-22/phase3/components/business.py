# phase3/components/business.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from textwrap import shorten

from services.summarizer import summarize_bullets
from utils.common import looks_protein_related


def _slice_for_business(df: pd.DataFrame, source_filter: str, max_items: int = 120) -> pd.DataFrame:
    """Mix journals + community to capture both evidence and market voice."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if source_filter and source_filter != "All":
        sf = source_filter.lower()
        d = d[d["source"].str.lower() == sf]
    # prefer more recent & with decent text length
    if "text" in d.columns:
        textlen = d["text"].fillna("").str.len()
    elif "abstract" in d.columns:
        textlen = d["abstract"].fillna("").str.len()
    else:
        textlen = pd.Series([0] * len(d))
    d = d.assign(_q=(textlen > 160).astype(int))
    if "date" in d.columns:
        d = d.sort_values(["_q", "date"], ascending=[False, False])
    else:
        d = d.sort_values("_q", ascending=False)
    return d.head(max_items).reset_index(drop=True)


def _compose_market_context(rows: pd.DataFrame, max_chars: int = 9000) -> str:
    """
    Compact multiline context: Title/handle + 1–2 salient lines from body.
    Works for journals, blogs, reddit alike.
    """
    lines = []
    for _, r in rows.iterrows():
        # Use available columns for title and body
        title = str(r.get("title") or r.get("kind") or r.get("subreddit") or "Item").strip()
        src = str(r.get("source", "") or "").strip().capitalize()
        body = str(r.get("abstract") or r.get("summary") or r.get("text") or "").replace("\n", " ").strip()
        body = shorten(body, width=420, placeholder=" …")
        meta = []
        if pd.notna(r.get("date")):
            try:
                meta.append(pd.to_datetime(r["date"]).date().isoformat())
            except Exception:
                pass
        if src:
            meta.append(src)
        meta_str = " • ".join(meta)
        head = f"- {title}" + (f" ({meta_str})" if meta_str else "")
        lines.append(f"{head}\n  {body}")
        if sum(len(x) for x in lines) > max_chars:
            break
    return "MARKET CONTEXT (mixed sources):\n" + "\n".join(lines)


def _panel(
    title: str,
    instruction: str,
    ctx: str,
    model: str | None,
    bullets: int = 5,
    fallback_hint: str | None = None,
) -> list[str]:
    """
    Ask the LLM for strictly N bullets. On failure, return a light fallback.
    """
    sys_prompt = (
        "You are a pragmatic product strategist for the protein market. "
        "Use only the provided context; do not invent facts. "
        "Each bullet: one sentence, plain language, actionable; "
        "start with a bold 3–6 word headline."
    )
    user_prompt = (
        f"{instruction}\n"
        f"Output exactly {bullets} bullets prefixed with '- '.\n\n{ctx}"
    )

    raw_output = summarize_bullets(f"{sys_prompt}\n\n{user_prompt}", cfg=model, n=bullets)
    import streamlit as st
    st.write("Business Highlights: Raw LLM output", raw_output)
    if raw_output and len(raw_output) >= 1:
        return raw_output[:bullets]

    # Fallback: very simple heuristics (still useful; clearly labeled)
    if fallback_hint:
        return [f"- **(Heuristic)** {fallback_hint}"] + ["- *(awaiting LLM summary)*"] * (bullets - 1)
    return ["- *(awaiting LLM summary)*"] * bullets


def render(df: pd.DataFrame, source_filter: str, model: str | None) -> None:
    """
    Business-Relevant Highlights:
      • 💡 Investment Opportunities (5)
      • 🛠 Product Improvement Areas (5)
      • 👥 Customer Preferences & Market Shifts (5)
    """
    st.subheader("Business-Relevant Highlights")

    if df is None or df.empty:
        st.info("No data available.")
        return

    rows = _slice_for_business(df, source_filter)
    st.write("Business Highlights: DataFrame shape", rows.shape)
    st.write("Business Highlights: DataFrame columns", list(rows.columns))
    if rows.empty:
        st.warning("No items found for the current filter/window.")
        return

    # Strictly filter for protein-related items before building context
    protein_rows = rows[rows["text"].apply(lambda x: looks_protein_related(str(x)) if pd.notna(x) else False)]
    if protein_rows.empty:
        st.warning("No protein-related items found for the current filter/window.")
        return

    ctx = _compose_market_context(protein_rows)
    st.write("Business Highlights: LLM context preview", ctx[:1000])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 💡 Investment Opportunities")
        bullets = _panel(
            "Investment Opportunities",
            instruction=(
                "From MARKET CONTEXT, extract opportunities where deploying capital, "
                "R&D, partnerships, or go-to-market could plausibly yield returns. "
                "Tie to signals like demand, evidence strength, or whitespace."
            ),
            ctx=ctx,
            model=model,
            bullets=5,
            fallback_hint="Emerging interest in high-protein ready-to-drink (RTD) formats and convenient on-the-go SKUs.",
        )
        for b in bullets:
            st.markdown(b)

    with col2:
        st.markdown("### 🛠 Product Improvement Areas")
        bullets = _panel(
            "Product Improvement Areas",
            instruction=(
                "From MARKET CONTEXT, list concrete improvements to protein products "
                "(formulation, taste/mixability, digestibility, allergens, sustainability, packaging). "
                "Prefer items that are repeatedly mentioned or evidence-backed."
            ),
            ctx=ctx,
            model=model,
            bullets=5,
            fallback_hint="Improve mixability and flavor profiles for whey blends while keeping clean labels.",
        )
        for b in bullets:
            st.markdown(b)

    with col3:
        st.markdown("### 👥 Customer Preferences & Market Shifts")
        bullets = _panel(
            "Customer Preferences & Market Shifts",
            instruction=(
                "From MARKET CONTEXT, extract consumer preferences and shifts "
                "(taste, claims, certifications, plant vs whey, price sensitivity, formats). "
                "Be specific and avoid generic trends."
            ),
            ctx=ctx,
            model=model,
            bullets=5,
            fallback_hint="Growing interest in low-sugar, high-protein snacks with clear macro labeling.",
        )
        for b in bullets:
            st.markdown(b)

    with st.expander("Show mixed-source items used"):
        cols = [c for c in ["title", "source", "url", "date"] if c in rows.columns]
        if cols:
            show = rows[cols].rename({"title": "Title", "source": "Source", "url": "Link", "date": "Date"}, axis=1)
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.caption("No source columns available to display.")