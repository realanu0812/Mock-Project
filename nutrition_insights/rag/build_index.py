# nutrition_insights/rag/build_index.py
from __future__ import annotations
import json, argparse, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ---- paths
ROOT = Path(__file__).resolve().parents[1]   # .../nutrition_insights
DATA = ROOT / "data"
IN_JSONL = DATA / "corpus_filtered.jsonl"
OUT_INDEX = DATA / "faiss.index"
OUT_META = DATA / "index_meta.jsonl"
OUT_INFO = DATA / "index_info.json"

# ---- try FAISS
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---- embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_jsonl(p: Path):
    if not p.exists(): return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    def _make_hashable(val):
        if isinstance(val, (list, dict, set)):
            return json.dumps(val, ensure_ascii=False)
        return val
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            # Ensure all values in dict are hashable
            r_hashable = {k: _make_hashable(v) for k, v in r.items()}
            f.write(json.dumps(r_hashable, ensure_ascii=False) + "\n")

def chunk_text(text: str, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = text or ""
    if len(text) <= size: return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        if i + size >= len(text): break
        i += size - overlap
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index from filtered corpus")
    ap.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize vectors (recommended for IP)")
    ap.add_argument("--dims", type=int, default=None, help="Force dims (usually auto)")
    ap.add_argument("--verified-only", action="store_true", help="Index only is_verified=True (journals)")
    ap.add_argument("--min-quality", type=float, default=0.0, help="Drop rows below this quality_score")
    ap.add_argument("--max-docs", type=int, default=None, help="For quick dev, limit number of docs")
    args = ap.parse_args()

    rows = load_jsonl(IN_JSONL)
    if args.verified_only:
        rows = [r for r in rows if r.get("is_verified")]
    if args.min_quality > 0:
        rows = [r for r in rows if float(r.get("quality_score", 0.0)) >= args.min_quality]
    if args.max_docs:
        rows = rows[:args.max_docs]

    if not rows:
        print("No rows to index. Run filter_corpus.py first.")
        return

    # Prepare chunks & meta
    meta = []
    texts = []
    MIN_CHUNK_LEN = 50
    for ridx, r in enumerate(rows):
        # Use combined_text if available, else fallback to text
        text_to_chunk = r.get("combined_text") or r.get("text") or ""
        chunks = chunk_text(text_to_chunk)
        for cidx, chunk in enumerate(chunks):
            if len(chunk.strip()) < MIN_CHUNK_LEN:
                continue
            meta.append({
                "rid": ridx,
                "chunk_id": cidx,
                "url": r.get("url"),
                "title": r.get("title"),
                "source": r.get("source"),
                "source_type": r.get("source_type"),
                "is_verified": r.get("is_verified", False),
                "published_at": r.get("published_at"),
                "quality_score": r.get("quality_score"),
                "relevance": r.get("relevance"),
                "tags": r.get("tags", []),
            })
            texts.append(chunk)

    print(f"Prepared {len(texts)} chunks from {len(rows)} docs")

    # Embeddings
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=args.normalize)
    emb = np.asarray(emb).astype("float32")
    dim = emb.shape[1]

    if not _HAS_FAISS:
        print("FAISS not installed. Install with: pip install faiss-cpu")
        return

    # Build FAISS (Inner Product + normalized vectors recommended)
    index = faiss.IndexFlatIP(dim)
    if not args.normalize:
        # if not normalized, convert to L2 + Map2IP if desired. But we recommend normalize.
        pass
    index.add(emb)

    # Save
    DATA.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_INDEX))
    write_jsonl(OUT_META, meta)
    with OUT_INFO.open("w", encoding="utf-8") as f:
        json.dump({
            "built_at": now_iso(),
            "model": args.model,
            "normalized": args.normalize,
            "dim": dim,
            "num_vectors": int(index.ntotal),
            "source": str(IN_JSONL),
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved FAISS index → {OUT_INDEX}")
    print(f"   Meta → {OUT_META}")
    print(f"   Info → {OUT_INFO}")

if __name__ == "__main__":
    main()