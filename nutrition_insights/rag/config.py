from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

COMBINED_JSON = DATA_DIR / "combined.json"

# Corpus outputs
COMMUNITY_STORE = ARTIFACTS_DIR / "community_store.parquet"
VERIFIED_STORE  = ARTIFACTS_DIR / "verified_store.parquet"

# FAISS indexes
COMMUNITY_INDEX = ARTIFACTS_DIR / "faiss_community.index"
VERIFIED_INDEX  = ARTIFACTS_DIR / "faiss_verified.index"

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast & decent

# Chunking
CHUNK_WORDS = 280
CHUNK_OVERLAP = 60
MIN_CHARS_PER_DOC = 400  # drop tiny docs

# Retrieval
TOPK_COMMUNITY = 8
TOPK_VERIFIED  = 6

# Optional LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")