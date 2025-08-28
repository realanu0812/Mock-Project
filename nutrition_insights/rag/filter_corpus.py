"""
Protein relevance filtering pipeline for nutrition_insights.

Reads data/combined.json (list of records), computes a protein-related
relevance score for each record (keyword/phrase based with simple boosts),
and writes the filtered, scored set to data/filtered_corpus.jsonl (one JSON per line).

Usage:
    python -m nutrition_insights.rag.filter_corpus \
        --min-score 1.0 \
        --top 0

Notes:
- No heavy deps (pandas, sentence-transformers) are required.
- If you later add embedding/TF‑IDF scoring, you can plug it into `compute_relevance`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Tuple, Any

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "combined.json"
OUTPUT_FILE = DATA_DIR / "corpus_filtered.jsonl"

# --------------------------------------------------------------------------------------
# Keyword & phrase weights (tweak as needed)
# Word entries are matched with word-boundary regexes; phrases are matched literally.
# --------------------------------------------------------------------------------------
KEYWORD_WEIGHTS: Dict[str, float] = {
    # Core terms
    "protein": 1.0,
    "whey": 1.2,
    "casein": 1.1,
    "plant": 0.6,
    "muscle": 0.8,
    "hypertrophy": 1.2,
    "strength": 0.8,
    "satiety": 0.9,
    "digestibility": 1.0,
    "bioavailability": 1.0,
    "thermogenesis": 1.1,
    "leucine": 1.0,
    "bcaa": 0.9,
    "eaa": 0.9,
    "amino": 0.7,
    "diaas": 1.3,
    "pdcaas": 1.2,
    "timing": 0.7,
    "sleep": 0.6,
    "night": 0.6,
    "dose": 0.7,
    "safety": 0.7,
    "quality": 0.6,
    "recovery": 0.7,
    "performance": 0.6,
    "lean": 0.5,
    "anabolic": 0.9,
    "mps": 1.1,
    "muscle protein synthesis": 1.3,
    "weight loss": 0.8,
    "appetite": 0.6,

    # Product forms
    "pea": 0.6,
    "soy": 0.6,
    "rice": 0.5,
    "hemp": 0.6,
    "collagen": 0.7,
    "isolate": 0.7,
    "concentrate": 0.5,
    "hydrolysate": 0.8,
    "micellar": 0.7,

    # Use cases
    "pre-workout": 0.6,
    "post-workout": 0.8,
    "peri-workout": 0.6,
}

# Precompile regex patterns for single-word terms (word boundaries)
WORD_PATTERNS: Dict[str, re.Pattern] = {
    k: re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE)
    for k, w in KEYWORD_WEIGHTS.items()
    if " " not in k  # single word
}

# For phrases (contain spaces), use simple case-insensitive find
PHRASES: Dict[str, float] = {k: w for k, w in KEYWORD_WEIGHTS.items() if " " in k}

# --------------------------------------------------------------------------------------
# Simple boosts
# --------------------------------------------------------------------------------------
TITLE_BOOST = 1.2           # if keyword occurs in title
SOURCE_BOOSTS = {
    "pubmed": 1.3,          # journals / higher quality
    "journal": 1.25,
    "reddit": 1.0,          # community
    "blog": 1.05,           # semi‑formal
    "news": 1.0,
}

MIN_LEN_FOR_FULL_SCORE = 200  # length normalization pivot (characters)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(_as_str(t) for t in x)
    if isinstance(x, dict):
        # join values
        return " ".join(_as_str(v) for v in x.values())
    return str(x)

def extract_source(rec: Dict[str, Any]) -> str:
    for key in ("source", "src", "channel", "origin"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return v.lower()
    # Try to infer from URL if present
    url = _as_str(rec.get("url"))
    if "ncbi.nlm.nih.gov" in url or "pubmed" in url:
        return "pubmed"
    if "reddit.com" in url:
        return "reddit"
    return ""

def count_hits(text: str) -> Tuple[float, Counter]:
    """
    Count weighted hits in text (single words + phrases).
    Returns (weighted_score, Counter_of_hit_terms)
    """
    if not text:
        return 0.0, Counter()

    score = 0.0
    hits = Counter()
    # Single-word matches using regex word boundaries
    for term, pat in WORD_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            weight = KEYWORD_WEIGHTS[term]
            count = len(matches)
            score += weight * count
            hits[term] += count

    # Phrase matches (case-insensitive)
    low = text.lower()
    for phrase, weight in PHRASES.items():
        # Count non-overlapping occurrences
        count = low.count(phrase.lower())
        if count:
            score += weight * count
            hits[phrase] += count

    return score, hits

def compute_relevance(rec: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Compute a heuristic relevance score for a record.
    """
    title = _as_str(rec.get("title"))
    body = _as_str(rec.get("text"))
    combined = " ".join([title, body]).strip()
    base_score, hits = count_hits(combined)

    # Title boost if hits also occur in title
    if title:
        title_score, _ = count_hits(title)
        if title_score > 0:
            base_score *= TITLE_BOOST

    # Source-based quality boost
    src = extract_source(rec)
    for k, b in SOURCE_BOOSTS.items():
        if k in src:
            base_score *= b
            break

    # Length normalization (mild boost for longer, informative text)
    ln = 1.0 + math.log1p(max(0, len(combined) - MIN_LEN_FOR_FULL_SCORE)) / 6.0 if len(combined) > MIN_LEN_FOR_FULL_SCORE else 1.0
    final = base_score * ln

    return final, sorted(list(hits.keys()))

def is_protein_related(score: float, min_score: float) -> bool:
    return score >= min_score

def record_key(rec: Dict[str, Any]) -> str:
    """
    Create a stable key for deduping (normalized title+text).
    """
    title = _as_str(rec.get("title")).strip().lower()
    text = _as_str(rec.get("text")).strip().lower()
    blob = f"{title}||{text}"
    return md5(blob.encode("utf-8")).hexdigest()

def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def run(min_score: float, top: int, input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE) -> None:
    if not input_file.exists():
        print(f"[filter_corpus] Input not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    records = load_records(input_file)
    scored: List[Tuple[float, Dict[str, Any], List[str]]] = []

    seen = set()
    for rec in records:
        k = record_key(rec)
        if k in seen:
            continue
        seen.add(k)

        score, hits = compute_relevance(rec)
        if is_protein_related(score, min_score):
            rec_out = dict(rec)
            rec_out["relevance_score"] = round(score, 4)
            if hits:
                rec_out["relevance_hits"] = hits
            scored.append((score, rec_out, hits))

    # Sort by score desc
    scored.sort(key=lambda t: t[0], reverse=True)

    # Limit top-N if requested
    if top and top > 0:
        scored = scored[:top]

    out = [r for _, r, _ in scored]
    save_jsonl(output_file, out)

    # Summary
    total = len(records)
    kept = len(out)
    print(f"[filter_corpus] {kept}/{total} kept (min_score={min_score}, top={top or 'ALL'}) → {output_file}")

def cli(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Filter combined.json to protein-related records with relevance scoring.")
    p.add_argument("--min-score", type=float, default=1.0, help="Minimum relevance score to keep a record. Default: 1.0")
    p.add_argument("--top", type=int, default=0, help="If >0, keep only top-N records by score. Default: 0 (all).")
    p.add_argument("--input", type=str, default=str(INPUT_FILE), help="Path to combined.json")
    p.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Path to write filtered_corpus.jsonl")
    args = p.parse_args(argv)

    run(
        min_score=args.min_score,
        top=args.top,
        input_file=Path(args.input),
        output_file=Path(args.output),
    )

if __name__ == "__main__":
    cli()