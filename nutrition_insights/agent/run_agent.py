import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# --- Project paths ---
PKG_DIR = Path(__file__).resolve().parents[1]           # .../nutrition_insights
PROJECT_ROOT = PKG_DIR.parent                            # repo root
SCRAPER = PKG_DIR / "merge_scrapper.py"

# RAG imports (local package)
sys.path.insert(0, str(PROJECT_ROOT))  # ensure `nutrition_insights` package is importable
from nutrition_insights.rag import build_corpus as rag_build_corpus
from nutrition_insights.rag import build_index as rag_build_index
from nutrition_insights.rag.config import (
    DATA_DIR, ARTIFACTS_DIR, COMBINED_JSON,
    COMMUNITY_STORE, VERIFIED_STORE,
    COMMUNITY_INDEX, VERIFIED_INDEX,
    EMBEDDING_MODEL, TOPK_COMMUNITY, TOPK_VERIFIED,
    OPENAI_API_KEY, OPENAI_MODEL
)

# Retrieval imports
import faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import requests

# --- LLM config (OpenAI or Ollama) ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def llm_chat(system: str, user: str) -> str:
    """Use OpenAI if OPENAI_API_KEY is set; otherwise use Ollama; else return context."""
    if OPENAI_API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return r.choices[0].message.content
    # Ollama
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "options": {"temperature": 0.2},
            "stream": False
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content") or data.get("response","")
    except Exception as e:
        return f"[No LLM configured] {e}\n\n" + user[:4000]

# --- Utilities ---
def mtime(p: Path) -> float:
    try: return p.stat().st_mtime
    except FileNotFoundError: return 0.0

def run(cmd: list[str], cwd: Path):
    print(f"\n>>> {' '.join(map(str, cmd))}  (cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=str(cwd))

def ensure_scrape(merge_only: bool):
    """Optionally run merge_scrapper.py to create/refresh data/*.json and data/combined.json."""
    args = ["--merge-only"] if merge_only else []
    run([sys.executable, str(SCRAPER), *args], cwd=PKG_DIR)

def needs_rebuild(force: bool) -> bool:
    """Check if we should rebuild corpus/index based on timestamps or --force-rebuild."""
    if force: return True
    # If artifacts missing → rebuild
    if not COMMUNITY_STORE.exists() or not VERIFIED_STORE.exists():
        return True
    if not COMMUNITY_INDEX.exists() or not VERIFIED_INDEX.exists():
        return True
    # If combined.json is newer than chunk stores → rebuild
    latest_input = mtime(COMBINED_JSON)
    stores_old = min(mtime(COMMUNITY_STORE), mtime(VERIFIED_STORE))
    if latest_input > stores_old:
        return True
    # If stores newer than indexes → rebuild indexes
    indexes_old = min(mtime(COMMUNITY_INDEX), mtime(VERIFIED_INDEX))
    if max(mtime(COMMUNITY_STORE), mtime(VERIFIED_STORE)) > indexes_old:
        return True
    return False

def rebuild_all(force: bool):
    """Build corpus and indexes as needed."""
    if force or not COMMUNITY_STORE.exists() or not VERIFIED_STORE.exists() or mtime(COMBINED_JSON) > min(mtime(COMMUNITY_STORE), mtime(VERIFIED_STORE)):
        rag_build_corpus.run()
    if force or not COMMUNITY_INDEX.exists() or not VERIFIED_INDEX.exists() or max(mtime(COMMUNITY_STORE), mtime(VERIFIED_STORE)) > min(mtime(COMMUNITY_INDEX), mtime(VERIFIED_INDEX)):
        rag_build_index.run()

# --- Retrieval tools ---
_model = None
_comm_ix = None
_ver_ix  = None
_comm_store = None
_ver_store  = None

def _lazy_load_indices():
    global _model, _comm_ix, _ver_ix, _comm_store, _ver_store
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _comm_ix is None:
        _comm_ix = faiss.read_index(str(COMMUNITY_INDEX))
        _comm_store = pd.read_parquet(COMMUNITY_STORE)
    if _ver_ix is None:
        _ver_ix = faiss.read_index(str(VERIFIED_INDEX))
        _ver_store = pd.read_parquet(VERIFIED_STORE)

def _embed_q(q: str) -> np.ndarray:
    v = _model.encode([q], normalize_embeddings=True)[0].astype("float32")
    return v.reshape(1, -1)

def _search(index, store, qvec, k):
    if index.ntotal == 0 or store.empty: return []
    D, I = index.search(qvec, k)
    hits=[]
    for dist, idx in zip(D[0], I[0]):
        if idx == -1: continue
        row = store.iloc[int(idx)]
        hits.append({**row.to_dict(), "score": float(dist)})
    return hits

def search_vector(query: str, store: str = "both", k_comm=TOPK_COMMUNITY, k_ver=TOPK_VERIFIED):
    _lazy_load_indices()
    qv = _embed_q(query)
    out = {}
    if store in ("both","community"):
        out["community"] = _search(_comm_ix, _comm_store, qv, k_comm)
    if store in ("both","verified"):
        out["verified"]  = _search(_ver_ix,  _ver_store,  qv, k_ver)
    return out

# Simple heuristic fact-check stub (keep it lightweight)
def fact_check(claim: str, verified_hits: List[Dict[str,Any]]):
    joined = " ".join(h.get("text","") for h in verified_hits).lower()
    needles = ["randomized", "meta-analysis", "hypertrophy", "muscle protein synthesis", "mps",
               "leucine", "pdcaas", "diaas", "strength", "trial", "cohort"]
    supported = any(n in joined for n in needles)
    return {"label": "supported" if supported else "inconclusive",
            "why": "Heuristic keyword check over verified snippets. Upgrade with LLM for deeper checks."}

def rank_insights(items: List[Dict[str,Any]]):
    # Prefer verified, then higher similarity score
    return sorted(items, key=lambda x: (not x.get("is_verified", False), -x.get("score", 0.0)))

# --- Agent planning and synthesis ---
SYSTEM = """You are a nutrition analyst agent. Plan tool calls as JSON, then act.
Be concise, separate 'Community Buzz' (unverified) from 'Verified Evidence' (PubMed). Cite URLs.
"""

PLAN_PROMPT = """Question: "{q}"

Available tools:
- search_vector(query, store="community|verified|both")
- fact_check(claim, verified_hits)
- rank_insights(items)

Return a JSON list of steps, e.g.:
[
  {{"tool":"search_vector","input":{{"query":"{q}","store":"both"}}}},
  {{"tool":"fact_check","input":{{"claim":"Summarize whether whey outperforms plant protein for hypertrophy"}}}},
  {{"tool":"rank_insights","input":{{"items":"$LAST_RESULT"}}}}
]
"""

def run_agent(question: str) -> str:
    # 1) Get a plan
    plan_txt = llm_chat(SYSTEM, PLAN_PROMPT.format(q=question)).strip()
    try:
        plan = json.loads(plan_txt)
        if not isinstance(plan, list): raise ValueError
    except Exception:
        plan = [{"tool":"search_vector","input":{"query":question,"store":"both"}}]

    # 2) Execute up to 6 steps
    last = None
    for step in plan[:6]:
        tool = step.get("tool")
        inp  = step.get("input", {})
        if isinstance(inp, dict) and inp.get("items") == "$LAST_RESULT":
            inp["items"] = last

        if tool == "search_vector":
            query = inp.get("query", question)
            store = inp.get("store","both")
            last = search_vector(query, store)
        elif tool == "fact_check":
            ver_hits = []
            if isinstance(last, dict) and "verified" in last:
                ver_hits = last["verified"]
            last = fact_check(inp.get("claim", question), ver_hits)
        elif tool == "rank_insights":
            items = inp.get("items") or []
            last = rank_insights(items if isinstance(items, list) else [])
        else:
            last = {"error": f"unknown tool {tool}"}

    # 3) Build citations & synthesis prompt from latest search result if present
    community, verified = [], []
    if isinstance(last, dict) and "community" in last:
        community = last.get("community", [])
        verified  = last.get("verified", [])
    else:
        # try to find last search in plan execution by running a fresh search as fallback
        res = search_vector(question, "both")
        community, verified = res.get("community", []), res.get("verified", [])

    comm_cites = "\n".join(f"- {h.get('title','')[:90]} ({h.get('url','')})" for h in community)
    ver_cites  = "\n".join(f"- {h.get('title','')[:90]} ({h.get('url','')})" for h in verified)

    synthesis = f"""Question: {question}

Community sources:
{comm_cites}

Verified sources:
{ver_cites}

Write:
1) Community Buzz — 3-5 bullets (do NOT assert truth).
2) Verified Evidence — 3-5 bullets using verified PubMed-derived snippets.
3) Verdict — what's supported vs still unverified; practical takeaway.
4) Citations — 6-10 URLs (mix of community + verified) as bullets.
"""
    return llm_chat(SYSTEM, synthesis)

# --- Orchestrator ---
def main():
    ap = argparse.ArgumentParser(description="Agentic RAG: (optional) scrape/merge -> rebuild if needed -> agent answer")
    ap.add_argument("question", nargs="*", help="Your query for the agent")
    ap.add_argument("--scrape", action="store_true", help="Run merge_scrapper.py (re-scrape or merge) before answering")
    ap.add_argument("--merge-only", action="store_true", help="Run merge_scrapper.py --merge-only")
    ap.add_argument("--force-rebuild", action="store_true", help="Force rebuild corpus & indexes")
    args = ap.parse_args()

    question = " ".join(args.question) or "Top protein trends this month"

    # 1) Optional scrape/merge
    if args.scrape or args.merge_only:
        ensure_scrape(merge_only=args.merge_only)

    # 2) Rebuild if needed (or forced)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    if not COMBINED_JSON.exists():
        print(f"[ERROR] Missing {COMBINED_JSON}. Run merge_scrapper first (or use --scrape).")
        sys.exit(1)

    if needs_rebuild(args.force_rebuild):
        rebuild_all(force=args.force_rebuild)
    else:
        print("[build] Artifacts up to date — skipping rebuild.")

    # 3) Run agent
    out = run_agent(question)
    print("\n" + out + "\n")

if __name__ == "__main__":
    main()