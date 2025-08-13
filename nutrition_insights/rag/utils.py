import re
from typing import List

def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def word_chunks(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = clean_text(text).split()
    if not words: return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i+chunk_words])
        if piece: chunks.append(piece)
        if i + chunk_words >= len(words): break
    return chunks