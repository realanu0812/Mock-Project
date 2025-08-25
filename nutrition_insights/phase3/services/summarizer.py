# phase3/services/summarizer.py
from __future__ import annotations

from typing import List, Optional
import textwrap
import pandas as pd

from utils.gemini_client import chat_completion

def get_gemini_response(prompt: str, system: str = None, timeout: int = 60) -> str:
    return chat_completion(prompt, system=system, timeout=timeout)


def _join_texts(texts: List[str], cap: int = 4000) -> str:
    """Join texts with separators and cap length to avoid huge prompts."""
    parts: List[str] = []
    total = 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        if total + len(t) > cap:
            break
        parts.append(t)
        total += len(t)
    return "\n\n---\n\n".join(parts)


def _to_bullets(raw: str, n: int) -> List[str]:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    # keep only lines that look like bullets; else fallback: take first n lines
    bullets = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            bullets.append(ln.lstrip("-•* ").strip())
    if not bullets:
        bullets = lines
    # de-dup + trim to n
    seen = set()
    out: List[str] = []
    for b in bullets:
        if b in seen:
            continue
        seen.add(b)
        out.append(b)
        if len(out) >= n:
            break
    return out


def summarize_bullets(texts: List[str], cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Generic bullet summarizer for any list of strings.
    """
    corpus = _join_texts(texts)
    prompt = textwrap.dedent(f"""
        You are a concise analyst. Read the snippets and return exactly {n} short bullet points.
        - Only include concrete facts or clear claims.
        - Avoid hedging and filler.
        - No introductions or conclusions.
        - Output as plain bullets starting with '- '.

        Snippets:
        {corpus}
    """).strip()
    resp = chat_completion(prompt, cfg)
    return _to_bullets(resp, n)


def summarize_verified_findings(df: pd.DataFrame, cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Summarize journal (verified) findings.
    Expects df with columns: 'source' == 'journals', 'title', 'summary' or 'abstract'.
    """
    if df is None or df.empty:
        return []
    mask = df["source"].astype(str).str.lower().eq("journals")
    sub = df.loc[mask, ["title", "summary", "abstract"]].fillna("")
    texts = [f"{r.title}\n{r.summary or r.abstract}" for r in sub.itertuples(index=False)]
    return summarize_bullets(texts, cfg, n=n)


def summarize_buzz(df: pd.DataFrame, cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Summarize community buzz from reddit + blogs.
    Expects df with 'source' in {'reddit','blogs'} and 'title'/'text' columns.
    """
    if df is None or df.empty:
        return []
    s = df["source"].astype(str).str.lower()
    sub = df.loc[s.isin({"reddit", "blogs"})].copy()
    cols = [c for c in ["title", "text", "summary"] if c in sub.columns]
    sub = sub[cols].fillna("")
    texts = ["\n".join([x for x in r if x]) for r in sub.itertuples(index=False)]
    return summarize_bullets(texts, cfg, n=n)


def summarize_investments(df: pd.DataFrame, cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Business: short 'where to invest' bullets grounded on signals in df.
    """
    texts = _collect_business_texts(df)
    prompt = textwrap.dedent(f"""
        From the data snippets, produce {n} succinct bullets under the theme 'Investment Opportunities'.
        - Focus on areas a protein-nutrition company could invest in (R&D, supply chain, channels, SKUs).
        - Use market/consumer signals implied by the snippets.
        - No fluff; each bullet should be concrete and useful.

        Snippets:
        {_join_texts(texts)}
    """).strip()
    resp = chat_completion(prompt, cfg)
    return _to_bullets(resp, n)


def summarize_improvements(df: pd.DataFrame, cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Business: product improvement areas from signals in df.
    """
    texts = _collect_business_texts(df)
    prompt = textwrap.dedent(f"""
        From the data snippets, produce {n} succinct bullets under 'Product Improvement Areas'.
        - Focus on formulation, flavor, mixability, digestibility, packaging, sustainability, and safety.
        - Grounded in observed issues or trends in the snippets.

        Snippets:
        {_join_texts(texts)}
    """).strip()
    resp = chat_completion(prompt, cfg)
    return _to_bullets(resp, n)


def summarize_preferences(df: pd.DataFrame, cfg: Optional[object] = None, n: int = 5) -> List[str]:
    """
    Business: customer preferences & market shifts.
    """
    texts = _collect_business_texts(df)
    prompt = textwrap.dedent(f"""
        From the data snippets, produce {n} succinct bullets under 'Customer Preferences & Market Shifts'.
        - Emphasize flavors, formats, clean labels, plant vs whey, pricing sensitivity, and use-cases.
        - Avoid generic claims; reflect what the snippets imply.

        Snippets:
        {_join_texts(texts)}
    """).strip()
    resp = chat_completion(prompt, cfg)
    return _to_bullets(resp, n)


def _collect_business_texts(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    cols = [c for c in ["title", "text", "summary", "abstract"] if c in df.columns]
    sub = df[cols].fillna("")
    return ["\n".join([x for x in r if x]) for r in sub.itertuples(index=False)]