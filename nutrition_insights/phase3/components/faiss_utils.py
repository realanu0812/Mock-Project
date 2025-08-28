import faiss
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FAISS_INDEX = DATA_DIR / "faiss.index"
INDEX_META = DATA_DIR / "index_meta.jsonl"
INDEX_INFO = DATA_DIR / "index_info.json"


def load_faiss_index():
    if not FAISS_INDEX.exists() or not INDEX_META.exists():
        raise RuntimeError("Missing FAISS files; run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX))
    meta = []
    for line in INDEX_META.read_text(encoding="utf-8").splitlines():
        if line.strip():
            meta.append(json.loads(line))
    return index, meta

def embed_query(q: str, dim: int = None) -> np.ndarray:
    # Use the same embedding as in build_index (MiniLM or fallback hash)
    import hashlib
    if dim is None:
        # Try to get from index_info
        if INDEX_INFO.exists():
            with open(INDEX_INFO) as f:
                info = json.load(f)
                dim = info.get("dim", 384)
        else:
            dim = 384
    h = hashlib.sha256(q.encode("utf-8")).digest()
    vals = [((h[i % len(h)] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    v = np.array(vals, dtype="float32")
    n = np.linalg.norm(v) + 1e-9
    return (v / n).reshape(1, -1)

def faiss_topk(query: str, k: int = 8):
    index, meta = load_faiss_index()
    qvec = embed_query(query, dim=index.d)
    D, I = index.search(qvec.astype("float32"), k)
    hits = []
    for rank, (d, i) in enumerate(zip(D[0], I[0])):
        if i < 0 or i >= len(meta):
            continue
        item = meta[i].copy()
        item["_rank"] = rank
        item["_score"] = float(d)
        hits.append(item)
    return hits
