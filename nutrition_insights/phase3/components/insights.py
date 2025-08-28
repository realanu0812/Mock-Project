# phase3/components/insights.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple
import json

from utils.common import tokenize, protein_keywords
from utils.charts import bar, heatmap


# ---- helpers ----

def _only_protein_terms(tokens: Iterable[str]) -> List[str]:
    kw = set(k.lower() for k in protein_keywords())
    return [t for t in tokens if t in kw]

def _extract_tokens(series: pd.Series) -> List[List[str]]:
    out: List[List[str]] = []
    for txt in series.fillna("").astype(str):
        out.append(tokenize(txt))
    return out

def _top_pairs(token_lists: List[List[str]], top_k_terms: List[str]) -> List[Tuple[str, str]]:
    """Build unordered co-occurrence pairs within a row (no self-pairs)."""
    top_set = set(top_k_terms)
    pairs: Counter = Counter()
    for toks in token_lists:
        toks = [t for t in toks if t in top_set]
        if len(toks) < 2:
            continue
        uniq = sorted(set(toks))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pairs[(uniq[i], uniq[j])] += 1
    return pairs.most_common(200)


# ---- 3. Community Buzz ----

def render_buzz(df: pd.DataFrame, meta=None) -> None:
    """
    Premium UI: Community Buzz summarized by Gemini, cartoonic cloud-style card, off-white, playful fonts/colors, references in dropdown
    """
    if df is None or df.empty:
        st.info("No data available yet.")
        return
    # Normalize sources for robust filtering
    from components.overview import fix_sources
    df = pd.DataFrame(fix_sources(df.to_dict(orient="records")))

    mask = df["source"].str.lower().isin(["reddit", "blogs"])
    sdf = df[mask].copy()
    if not len(sdf):
        st.info("No Reddit/Blogs content available.")
        return

    # Only keep protein-related posts
    from utils.common import looks_protein_related
    def safe_concat(*args):
        return " ".join([str(x) if not isinstance(x, float) else "" for x in args])
    protein_mask = sdf.apply(lambda r: looks_protein_related(safe_concat(r.get("title", ""), r.get("summary", ""), r.get("text", ""))), axis=1)
    protein_sdf = sdf[protein_mask].copy()
    if protein_sdf.empty:
        st.warning("⚠️ (Gemini) No protein-related evidence found in the data corpus. Please check your query or try again later.")
        return

    # Prepare context for Gemini: use top buzz posts with evidence
    buzz_evidence = []
    for _, r in protein_sdf.iterrows():
        def safe_strip(val):
            if isinstance(val, str):
                return val.strip()
            return ""
        title = safe_strip(r.get("title") or r.get("headline"))
        text = safe_strip(r.get("summary") or r.get("text"))
        url = safe_strip(r.get("url") or r.get("permalink"))
        if not title and not text:
            continue
        buzz_evidence.append(f"- **{title}**: {text} (ref: {url})")
    buzz_context = "\n".join(buzz_evidence)

    from utils.gemini_client import chat_completion
    prompt = (
        "Summarize the latest community buzz from Reddit and Blogs on protein-related topics. "
        "Use the following posts as evidence. List each finding as a bullet. "
        "At the end, provide references (with title and link) as a numbered list, separated by a line 'REFERENCES:'.\n\n"
        f"{buzz_context}"
    )
    summary = chat_completion(prompt)
    # Split summary into findings and references
    findings, references = summary, ""
    if "REFERENCES:" in summary:
        findings, references = summary.split("REFERENCES:", 1)
    st.markdown("""
        <div style=\"background:linear-gradient(135deg,#f7f7f7 0%,#f3f3e7 100%);border-radius:48px 48px 48px 48px/60px 60px 60px 60px;padding:28px 32px;margin-bottom:18px;box-shadow:0 8px 32px rgba(120,60,60,0.10);border:4px dashed #e0e0e0;position:relative;overflow:hidden;\">
            <div style=\"position:absolute;top:-32px;left:24px;width:120px;height:60px;background:rgba(255,255,255,0.8);border-radius:60px 60px 60px 60px/60px 60px 60px 60px;box-shadow:0 2px 12px rgba(120,60,60,0.08);z-index:1;\"></div>
            <div style=\"position:absolute;bottom:-32px;right:24px;width:120px;height:60px;background:rgba(255,255,255,0.8);border-radius:60px 60px 60px 60px/60px 60px 60px 60px;box-shadow:0 2px 12px rgba(120,60,60,0.08);z-index:1;\"></div>
            <h3 style=\"margin-top:0;margin-bottom:10px;font-size:1.35rem;font-weight:900;color:#6a38b6;font-family:'Comic Sans MS', 'Comic Sans', cursive;z-index:2;position:relative;text-shadow:1px 2px 8px #e0e0e0;\">Community Buzz (Reddit & Blogs)</h3>
            <div style=\"font-size:1.08rem;color:#333;line-height:1.7;z-index:2;position:relative;font-family:'Comic Sans MS', 'Comic Sans', cursive;letter-spacing:0.01em;\">\n{}\n</div>
        </div>
    """.format(findings.strip()), unsafe_allow_html=True)
    if references.strip():
        with st.expander("Show References"):
            st.markdown(references.strip(), unsafe_allow_html=True)


# ---- 4. Graphs (no monthly volume) ----

def render(df: pd.DataFrame, meta=None) -> None:
    """
    Premium UI: Verified Findings from Journals with references (modern black card, references in dropdown)
    """
    if df is None or df.empty:
        st.info("No data available yet.")
        return
    # Normalize sources for robust filtering
    from components.overview import fix_sources
    df = pd.DataFrame(fix_sources(df.to_dict(orient="records")))

    mask = df["source"].str.lower() == "journals" if "source" in df.columns else None
    jdf = df[mask].copy() if mask is not None else pd.DataFrame()
    if jdf.empty:
        st.info("No journal findings available.")
        return

    # Prepare context for Gemini: journal findings with evidence
    journal_evidence = []
    for _, r in jdf.iterrows():
        title = (r.get("title") or r.get("headline") or "").strip()
        text = (r.get("summary") or r.get("text") or "").strip()
        url = (r.get("url") or r.get("permalink") or "").strip()
        if not title and not text:
            continue
        journal_evidence.append(f"- **{title}**: {text} (ref: {url})")
    journal_context = "\n".join(journal_evidence)

    from utils.gemini_client import chat_completion
    prompt = (
        "Summarize the latest verified findings from journals on protein-related topics. "
        "Use the following findings as evidence. List each finding as a bullet. "
        "At the end, provide references (with title and link) as a numbered list, separated by a line 'REFERENCES:'.\n\n"
        f"{journal_context}"
    )
    summary = chat_completion(prompt)
    findings, references = summary, ""
    if "REFERENCES:" in summary:
        findings, references = summary.split("REFERENCES:", 1)
    st.markdown("""
        <div style=\"background:linear-gradient(90deg,#18181b 0%,#232326 100%);border-radius:18px;padding:28px 32px;margin-bottom:18px;box-shadow:0 4px 24px rgba(0,0,0,0.18);color:#fff;\">
            <h3 style=\"margin-top:0;margin-bottom:12px;font-size:1.5rem;font-weight:800;color:#fff;\">Verified Findings (Journals)</h3>
            <div style=\"font-size:1.08rem;color:#e5e5e5;line-height:1.7;\">\n{}\n</div>
        </div>
    """.format(findings.strip()), unsafe_allow_html=True)
    if references.strip():
        with st.expander("Show References"):
            st.markdown(references.strip(), unsafe_allow_html=True)

def render_insights():
    # Load combined.json directly
    data_path = "../data/combined.json"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
    except Exception as e:
        st.error(f"❌ Failed to load combined.json: {e}")
        return