import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from .config import (
    COMMUNITY_STORE, VERIFIED_STORE,
    COMMUNITY_INDEX, VERIFIED_INDEX,
    EMBEDDING_MODEL
)

def _embed_texts(model, texts):
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embs.astype("float32")

def _build_index(embs: np.ndarray, out_path):
    index = faiss.IndexFlatIP(embs.shape[1])  # cosine (vectors normalized)
    if len(embs): index.add(embs)
    faiss.write_index(index, str(out_path))

def run():
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Community
    comm = pd.read_parquet(COMMUNITY_STORE)
    comm_embs = _embed_texts(model, comm["text"].tolist()) if not comm.empty else np.zeros((0,384), dtype="float32")
    _build_index(comm_embs, COMMUNITY_INDEX)

    # Verified
    ver = pd.read_parquet(VERIFIED_STORE)
    ver_embs = _embed_texts(model, ver["text"].tolist()) if not ver.empty else np.zeros((0,384), dtype="float32")
    _build_index(ver_embs, VERIFIED_INDEX)

    print(f"[index] community: {len(comm)} chunks -> {COMMUNITY_INDEX}")
    print(f"[index] verified : {len(ver)} chunks -> {VERIFIED_INDEX}")

if __name__ == "__main__":
    run()