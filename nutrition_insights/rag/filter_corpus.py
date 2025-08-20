# nutrition_insights/rag/filter_corpus.py
from __future__ import annotations
import json
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import numpy as np

# Optional: semantic upgrade if available. Falls back to TF-IDF if not.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Lightweight & always-available (install scikit-learn if you want TF-IDF fallback)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False

ROOT = Path(__file__).resolve().parents[1]        # .../nutrition_insights
DATA_DIR = ROOT / "data"
INPUT_FILES = {
    "blogs": DATA_DIR / "blogs.json",
    "reddit": DATA_DIR / "reddit.json",
    "journals": DATA_DIR / "journals.json",
}
OUT_JSONL = DATA_DIR / "corpus_filtered.jsonl"
META_FILE = DATA_DIR / "corpus_meta.json"

# -----------------------------------
# Protein taxonomy / keywords
# -----------------------------------
PROTEIN_SEED_QUERIES = [
    "whey protein benefits muscle recovery leucine EAA BCAA",
    "plant protein soy pea rice hemp PDCAAS DIAAS amino acid profile",
    "casein slow digestion night protein satiety",
    "collagen protein limitations low EAA not complete",
    "protein powder quality purity heavy metals third-party testing",
    "high-protein diet athletes strength hypertrophy MPS",
    "protein timing per-meal distribution leucine threshold",
    "protein for weight loss satiety thermogenesis",
    "bioavailability digestibility DIAAS PDCAAS",
]

SUBTOPIC_TAGS = {
    "whey": r"\bwhey\b",
    "casein": r"\bcasein\b",
    "plant": r"\bplant protein\b|\bpea protein\b|\bsoy protein\b|\brice protein\b|\bhemp protein\b",
    "collagen": r"\bcollagen\b",
    "timing": r"\bprotein timing\b|\bpost[- ]workout\b|\bper[- ]meal\b|\bleucine threshold\b",
    "quality": r"\bPDCAAS\b|\bDIAAS\b|\bbioavailability\b|\bdigestibility\b|\bthird[- ]party\b",
    "athletes": r"\bathlete\b|\bhypertrophy\b|\bstrength\b|\bMPS\b|\bmuscle protein synthesis\b",
    "weight_loss": r"\bweight loss\b|\bsatiety\b|\bthermogenesis\b",
    "safety": r"\bheavy metals\b|\bcontaminant\b|\bsafety\b",
}

LANG_RE = re.compile(r"[a-zA-Z]")  # super-simple language gate (English-ish)

# -----------------------------------
# Helpers
# -----------------------------------
def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(val):
    if not val:
        return None
    try:
        # ISO or RFC-ish
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            # unix timestamp?
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None

def load_list(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def write_jsonl(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def textify(rec: dict) -> str:
    parts = []
    for k in ("title", "headline", "combined_text", "content", "selftext", "summary", "abstract"):
        v = rec.get(k)
        if v and isinstance(v, str):
            parts.append(v.strip())
    txt = "\n\n".join([p for p in parts if p])
    return txt

def is_englishish(txt: str) -> bool:
    return bool(txt and LANG_RE.search(txt))

def short_clean(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tag_subtopics(text: str) -> list[str]:
    t = text.lower()
    tags = []
    for k, pattern in SUBTOPIC_TAGS.items():
        if re.search(pattern, t):
            tags.append(k)
    return tags

def source_trust_weight(source_type: str, is_verified: bool) -> float:
    # Baseline trust by source class, with an extra boost if flagged verified
    base = {"journal": 1.0, "blog_article": 0.6, "reddit_post": 0.45}.get(source_type, 0.5)
    return base + (0.15 if is_verified else 0.0)

def recency_weight(dt: datetime | None, half_life_days=90) -> float:
    if not dt:
        return 0.7  # undated: neutral-ish
    days = (now_utc() - dt).days
    return math.exp(-days / half_life_days)

def engagement_weight(rec: dict) -> float:
    # Reddit: upvotes & comments; Blogs/Journals: nothing/1.0
    if rec.get("source_type") == "reddit_post":
        ups = float(rec.get("ups") or rec.get("score") or 0)
        num_comments = float(rec.get("num_comments") or 0)
        # lightweight normalization
        return min(1.0, 0.2 + math.log1p(ups + 0.5 * num_comments) / 8.0)
    return 1.0

# -----------------------------------
# Vectorizers (semantic or TF-IDF fallback)
# -----------------------------------
class RelevanceScorer:
    def __init__(self, seeds: list[str], use_embeddings=True):
        self.seeds = seeds
        self.use_embeddings = use_embeddings and _HAS_ST
        self._model = None
        self._seed_vec = None
        self._tfidf = None
        self._seed_tfidf = None

        if self.use_embeddings:
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._seed_vec = self._model.encode(self.seeds, normalize_embeddings=True).mean(axis=0)
        else:
            if not _HAS_SK:
                raise RuntimeError("No sentence-transformers or scikit-learn available for relevance scoring.")
            self._tfidf = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
            self._seed_tfidf = self._tfidf.fit_transform(self.seeds).mean(axis=0)

    def score(self, texts: list[str]) -> np.ndarray:
        if self.use_embeddings:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            seed = self._seed_vec
            return (vecs @ seed).astype(float)  # cosine via normalized dot
        else:
            X = self._tfidf.transform(texts)
            sims = cosine_similarity(X, self._seed_tfidf)
            return np.asarray(sims).ravel().astype(float)

# -----------------------------------
# Pipeline
# -----------------------------------
def run(
    min_len=200,
    window_days=180,
    protein_min_relevance=0.25,
    near_dupe_thresh=0.92,
    keep_only_verified=False,
):
    # Load
    blogs = load_list(INPUT_FILES["blogs"])
    reddit = load_list(INPUT_FILES["reddit"])
    journals = load_list(INPUT_FILES["journals"])

    raw = []
    for rec in blogs + reddit + journals:
        # normalize source_type if absent
        st = rec.get("source_type") or rec.get("source") or ""
        st = {"blog": "blog_article", "reddit": "reddit_post", "journal": "journal"}.get(st, st or "blog_article")
        rec["source_type"] = st
        raw.append(rec)

    # Standardize/augment
    prepared = []
    cutoff = now_utc() - timedelta(days=window_days)
    for r in raw:
        url = short_clean(r.get("url") or "")
        if not url:
            continue

        text = textify(r)
        text = short_clean(text)
        if len(text) < min_len or not is_englishish(text):
            continue

        # date
        published = r.get("published_at") or r.get("published") or r.get("created_utc") or r.get("date")
        dt = parse_dt(published)
        if dt and dt < cutoff:
            continue

        # verified flag defaulting by source
        if "is_verified" not in r:
            if r["source_type"] == "journal":
                r["is_verified"] = True
            else:
                r["is_verified"] = False

        prepared.append({
            "title": short_clean(r.get("title") or r.get("headline") or ""),
            "url": url,
            "source_type": r["source_type"],
            "is_verified": bool(r.get("is_verified")),
            "published_at": dt.isoformat() if dt else None,
            "raw_published": published,
            "text": text,
            "ups": r.get("ups") or r.get("score"),
            "num_comments": r.get("num_comments"),
        })

    if keep_only_verified:
        prepared = [p for p in prepared if p["is_verified"]]

    if not prepared:
        write_jsonl(OUT_JSONL, [])
        write_json(META_FILE, {"total": 0, "using_embeddings": False})
        print("No items passed the hard filters.")
        return

    # Relevance scoring
    try:
        scorer = RelevanceScorer(PROTEIN_SEED_QUERIES, use_embeddings=True)
        using_embeddings = True
    except Exception:
        scorer = RelevanceScorer(PROTEIN_SEED_QUERIES, use_embeddings=False)
        using_embeddings = False

    texts = [p["text"] for p in prepared]
    rel = scorer.score(texts)  # 0..1-ish
    for p, s in zip(prepared, rel):
        p["relevance"] = float(s)

    # Filter by minimum relevance
    prepared = [p for p in prepared if p["relevance"] >= protein_min_relevance]
    if not prepared:
        write_jsonl(OUT_JSONL, [])
        write_json(META_FILE, {"total": 0, "using_embeddings": using_embeddings})
        print("No items passed the relevance filter.")
        return

    # Tagging
    for p in prepared:
        p["tags"] = tag_subtopics(p["text"])

    # Quality score
    for p in prepared:
        w_src = source_trust_weight(p["source_type"], p["is_verified"])
        w_rec = recency_weight(parse_dt(p["published_at"]) if p["published_at"] else None)
        w_eng = engagement_weight(p)
        # blend; tune weights as needed
        p["quality_score"] = float(
            0.50 * p["relevance"] +
            0.25 * w_rec +
            0.20 * w_src +
            0.05 * w_eng
        )

    # Near-duplicate removal (by cosine on TF-IDF to avoid extra downloads)
    kept = []
    if _HAS_SK and len(prepared) > 1:
        vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2))
        X = vec.fit_transform([p["text"] for p in prepared])
        # Greedy keep-best algorithm
        order = np.argsort([-p["quality_score"] for p in prepared])
        taken = np.zeros(len(prepared), dtype=bool)
        for i in order:
            if taken[i]:
                continue
            kept.append(prepared[i])
            sims = cosine_similarity(X[i], X).ravel()
            dupes = np.where(sims >= near_dupe_thresh)[0]
            taken[dupes] = True
    else:
        kept = prepared

    # Sort for downstream readability
    kept.sort(key=lambda x: (-x["quality_score"], x["source_type"] != "journal"))

    # Emit JSONL
    rows = []
    for p in kept:
        rows.append({
            "url": p["url"],
            "title": p["title"],
            "source_type": p["source_type"],
            "is_verified": p["is_verified"],
            "published_at": p["published_at"],
            "quality_score": round(p["quality_score"], 4),
            "relevance": round(p["relevance"], 4),
            "tags": p["tags"],
            "text": p["text"],
        })

    write_jsonl(OUT_JSONL, rows)
    write_json(META_FILE, {
        "total": len(rows),
        "using_embeddings": using_embeddings,
        "min_len": min_len,
        "window_days": window_days,
        "protein_min_relevance": protein_min_relevance,
        "near_dupe_thresh": near_dupe_thresh,
        "verified_only": keep_only_verified,
    })

    print(f"✅ Filtered corpus → {OUT_JSONL}  (n={len(rows)})")
    print(f"   Meta → {META_FILE}")
    print(f"   Embeddings used: {using_embeddings}")

def cli():
    ap = argparse.ArgumentParser(description="Filter scraped items into an RAG-ready protein corpus")
    ap.add_argument("--min-len", type=int, default=200, help="Minimum characters of combined text")
    ap.add_argument("--window-days", type=int, default=180, help="Recency window")
    ap.add_argument("--protein-min-relevance", type=float, default=0.25, help="Semantic relevance threshold (0..1)")
    ap.add_argument("--near-dupe-thresh", type=float, default=0.92, help="Cosine similarity threshold to drop near-duplicates")
    ap.add_argument("--verified-only", action="store_true", help="Keep only is_verified=True (journals)")
    args = ap.parse_args()

    run(
        min_len=args.min_len,
        window_days=args.window_days,
        protein_min_relevance=args.protein_min_relevance,
        near_dupe_thresh=args.near_dupe_thresh,
        keep_only_verified=args.verified_only,
    )

if __name__ == "__main__":
    cli()