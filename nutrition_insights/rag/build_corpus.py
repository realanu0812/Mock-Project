import json, pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from .config import (
    COMBINED_JSON, COMMUNITY_STORE, VERIFIED_STORE,
    CHUNK_WORDS, CHUNK_OVERLAP, MIN_CHARS_PER_DOC
)
from .utils import clean_text, word_chunks

def _load() -> List[Dict[str, Any]]:
    with open(COMBINED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "combined.json must be a list"
    return data

def _norm(rec: Dict[str, Any]) -> Dict[str, Any]:
    ct = rec.get("combined_text") or rec.get("text") or rec.get("summary") or ""
    title = rec.get("title") or ""
    url = rec.get("url") or ""
    stype = rec.get("source_type") or ""
    is_verified = bool(rec.get("is_verified", False))
    return {
        "title": clean_text(title),
        "url": url.strip(),
        "source_type": stype,
        "is_verified": is_verified,
        "combined_text": clean_text(ct),
    }

def run():
    raw = _load()
    rows = [_norm(r) for r in raw]
    rows = [r for r in rows if r["url"] and r["combined_text"] and len(r["combined_text"]) >= MIN_CHARS_PER_DOC]
    df = pd.DataFrame(rows)

    # Split
    df_comm = df[~df["is_verified"]].copy()
    df_ver  = df[df["is_verified"]].copy()

    def explode_chunks(sub: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, row in sub.iterrows():
            chunks = word_chunks(row["combined_text"], CHUNK_WORDS, CHUNK_OVERLAP)
            for i, ch in enumerate(chunks):
                out.append({
                    "url": row["url"],
                    "title": row["title"],
                    "source_type": row["source_type"],
                    "is_verified": bool(row["is_verified"]),
                    "chunk_id": f"{row['url']}#{i}",
                    "text": ch
                })
        return pd.DataFrame(out)

    comm_store = explode_chunks(df_comm)
    ver_store  = explode_chunks(df_ver)

    comm_store.to_parquet(COMMUNITY_STORE, index=False)
    ver_store.to_parquet(VERIFIED_STORE, index=False)

    print(f"[corpus] community chunks: {len(comm_store)} -> {COMMUNITY_STORE}")
    print(f"[corpus] verified  chunks: {len(ver_store)} -> {VERIFIED_STORE}")

if __name__ == "__main__":
    run()