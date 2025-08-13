import faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .config import (
    COMMUNITY_STORE, VERIFIED_STORE, COMMUNITY_INDEX, VERIFIED_INDEX,
    EMBEDDING_MODEL, TOPK_COMMUNITY, TOPK_VERIFIED,
    OPENAI_API_KEY, OPENAI_MODEL
)

def _load(ix_path, store_path):
    index = faiss.read_index(str(ix_path))
    store = pd.read_parquet(store_path)
    return index, store

def _embed_q(model, q):
    v = model.encode([q], normalize_embeddings=True)[0].astype("float32")
    return v.reshape(1, -1)

def _search(index, store, qvec, k):
    if index.ntotal == 0 or store.empty:
        return []
    D, I = index.search(qvec, k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1: continue
        row = store.iloc[int(idx)]
        hits.append({**row.to_dict(), "score": float(dist)})
    return hits

def _format_cites(hits):
    return "\n".join(f"- {h['title']} ({h['url']})" for h in hits)

def ask(question: str):
    comm_ix, comm_store = _load(COMMUNITY_INDEX, COMMUNITY_STORE)
    ver_ix,  ver_store  = _load(VERIFIED_INDEX,  VERIFIED_STORE)
    model = SentenceTransformer(EMBEDDING_MODEL)

    qvec = _embed_q(model, question)
    comm_hits = _search(comm_ix, comm_store, qvec, TOPK_COMMUNITY)
    ver_hits  = _search(ver_ix,  ver_store,  qvec, TOPK_VERIFIED)

    comm_ctx = "\n\n---\n\n".join(h["text"] for h in comm_hits)
    ver_ctx  = "\n\n---\n\n".join(h["text"] for h in ver_hits)

    system = "You are a careful nutrition analyst. Separate community buzz from verified evidence. Be concise and actionable."
    user = f"""Question: {question}

# Community Buzz (unverified)
Use only the following snippets:
{comm_ctx}

# Verified Evidence (PubMed)
Use only the following snippets:
{ver_ctx}

Then:
1) List 'Community Buzz' trends (do not claim truth).
2) List 'Verified Evidence' that supports/refutes.
3) Give a 'Verdict' (what's supported; what's unverified).
4) Add 'Citations' with the URLs below.

Citations (community):
{_format_cites(comm_hits)}

Citations (verified):
{_format_cites(ver_hits)}
"""

    if not OPENAI_API_KEY:
        # Retrieval-only fallback
        print("[No OPENAI_API_KEY] Showing retrieved snippets.\n")
        print(user[:4000])
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are the top protein trends this week?"
    ask(q)