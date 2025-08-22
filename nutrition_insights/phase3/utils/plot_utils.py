# phase3/utils/plot_utils.py
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable, Sequence, Tuple, Optional, List, Dict, Set

import numpy as np
import pandas as pd

from utils.common import tokenize, PROTEIN_WORDS


# ---------------------------
# Keyword utilities
# ---------------------------
def keyword_counts(
    df: pd.DataFrame,
    text_col: str = "text",
    whitelist: Optional[Iterable[str]] = None,
    top_n: int = 30,
    min_len: int = 2,
) -> pd.DataFrame:
    """
    Count keyword frequency from a text column.
    - Applies `tokenize`.
    - Optional `whitelist` (e.g., protein keywords). If None, uses PROTEIN_WORDS.
    - Filters very short tokens (len < min_len).
    Returns: DataFrame[keyword, count]
    """
    if df is None or text_col not in df.columns or not len(df):
        return pd.DataFrame(columns=["keyword", "count"])

    allowed: Set[str] = set(whitelist) if whitelist is not None else set(PROTEIN_WORDS)

    ctr: Counter[str] = Counter()
    for t in df[text_col].astype(str).tolist():
        toks = [w for w in tokenize(t) if len(w) >= min_len]
        if allowed:
            toks = [w for w in toks if w in allowed]
        ctr.update(toks)

    items = sorted(ctr.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        items = items[:top_n]
    return pd.DataFrame(items, columns=["keyword", "count"])


def explode_tokens(
    df: pd.DataFrame,
    text_col: str = "text",
    whitelist: Optional[Iterable[str]] = None,
    min_len: int = 2,
) -> pd.DataFrame:
    """
    Turn each row into multiple rows of tokens (1 token per row).
    Keeps original columns; adds 'token'.
    """
    if df is None or text_col not in df.columns or not len(df):
        return pd.DataFrame(columns=list(df.columns) + ["token"] if df is not None else ["token"])

    allowed: Set[str] = set(whitelist) if whitelist is not None else set(PROTEIN_WORDS)

    rows = []
    for _, row in df.iterrows():
        toks = [w for w in tokenize(str(row[text_col])) if len(w) >= min_len]
        if allowed:
            toks = [w for w in toks if w in allowed]
        for tok in toks:
            rec = dict(row)
            rec["token"] = tok
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------
# Co-occurrence utilities
# ---------------------------
def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def cooccurrence_counts_from_texts(
    texts: Iterable[str],
    whitelist: Optional[Iterable[str]] = None,
    min_len: int = 2,
) -> Dict[Tuple[str, str], int]:
    """
    Compute pairwise co-occurrence counts from an iterable of raw texts.
    """
    allowed: Set[str] = set(whitelist) if whitelist is not None else set(PROTEIN_WORDS)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for t in texts:
        toks = [w for w in tokenize(str(t)) if len(w) >= min_len]
        if allowed:
            toks = [w for w in toks if w in allowed]
        uniq = sorted(set(toks))
        for a, b in combinations(uniq, 2):
            counts[_pair_key(a, b)] += 1
    return counts


def cooccurrence_matrix(
    texts: Iterable[str],
    whitelist: Optional[Iterable[str]] = None,
    top_k_terms: int = 25,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a symmetric co-occurrence matrix (DataFrame) and the ordered label list.
    - Selects top_k_terms by marginal frequency (co-occurrence degree).
    Returns: (matrix_df, labels)
      where matrix_df is shape [k x k] with zeros on diagonal.
    """
    pair_counts = cooccurrence_counts_from_texts(texts, whitelist=whitelist)
    if not pair_counts:
        return pd.DataFrame(), []

    # Build degree (node strength)
    deg: Counter[str] = Counter()
    for (a, b), c in pair_counts.items():
        deg[a] += c
        deg[b] += c

    labels = [w for w, _ in deg.most_common(top_k_terms)]
    idx = {w: i for i, w in enumerate(labels)}
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)

    for (a, b), c in pair_counts.items():
        ia = idx.get(a); ib = idx.get(b)
        if ia is None or ib is None:
            continue
        mat[ia, ib] += c
        mat[ib, ia] += c

    df = pd.DataFrame(mat, index=labels, columns=labels)
    np.fill_diagonal(df.values, 0)
    return df, labels


def bubble_df_from_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a co-occurrence matrix to long-form suitable for a bubble chart.
    Returns columns: x, y, size (and optionally 'pair' for hover).
    """
    if mat is None or mat.empty:
        return pd.DataFrame(columns=["x", "y", "size", "pair"])

    m = mat.copy()
    # Upper triangle (without diagonal) to avoid duplicates
    xs, ys, sizes, pairs = [], [], [], []
    labels = list(m.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = int(m.iat[i, j])
            if v <= 0:
                continue
            xs.append(labels[i])
            ys.append(labels[j])
            sizes.append(v)
            pairs.append(f"{labels[i]} + {labels[j]}")
    return pd.DataFrame({"x": xs, "y": ys, "size": sizes, "pair": pairs})


# ---------------------------
# Time/keyword helpers
# ---------------------------
def keyword_trend_over_time(
    df: pd.DataFrame,
    date_col: str,
    text_col: str,
    keywords: Sequence[str],
    freq: str = "W",
) -> pd.DataFrame:
    """
    Count mentions of given keywords over time.
    Returns: DataFrame[date, keyword, count]
    """
    if df is None or not len(df) or date_col not in df.columns or text_col not in df.columns:
        return pd.DataFrame(columns=["date", "keyword", "count"])

    t = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
    tmp = df.loc[t.index, [date_col, text_col]].copy()
    tmp[date_col] = t

    rows = []
    kws = list(keywords)
    for _, r in tmp.iterrows():
        toks = set(tokenize(str(r[text_col])))
        for k in kws:
            if k in toks:
                rows.append({date_col: r[date_col], "keyword": k, "count": 1})

    if not rows:
        return pd.DataFrame(columns=["date", "keyword", "count"])

    out = pd.DataFrame(rows)
    out = (
        out.groupby([pd.Grouper(key=date_col, freq=freq), "keyword"])["count"]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    return out