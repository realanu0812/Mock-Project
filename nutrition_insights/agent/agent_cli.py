import os, json, time
from pathlib import Path
from typing import Any, Dict, List
from nutrition_insights.rag.qa_cli import _format_cites  # reuse helper if you like
from nutrition_insights.rag.config import TOPK_COMMUNITY, TOPK_VERIFIED
from nutrition_insights.rag.qa_cli import SentenceTransformer, faiss, pd, np
from nutrition_insights.rag.config import (
    COMMUNITY_STORE, VERIFIED_STORE, COMMUNITY_INDEX, VERIFIED_INDEX,
    EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_MODEL
)
import requests

# ---- LLMs (OpenAI or Ollama) ----
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def llm_chat(system: str, user: str) -> str:
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
    payload = {"model": OLLAMA_MODEL, "messages": [
        {"role":"system","content":system},{"role":"user","content":user}
    ], "options":{"temperature":0.2}, "stream": False}
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content") or data.get("response","")

# ---- Tools (reuse your FAISS + scrapers) ----
def _load(ix_path, store_path):
    index = faiss.read_index(str(ix_path))
    store = pd.read_parquet(store_path)
    return index, store

_model = SentenceTransformer(EMBEDDING_MODEL)
_comm_ix, _comm_store = _load(COMMUNITY_INDEX, COMMUNITY_STORE)
_ver_ix,  _ver_store  = _load(VERIFIED_INDEX,  VERIFIED_STORE)

def _embed_q(q): 
    v = _model.encode([q], normalize_embeddings=True)[0].astype("float32")
    import numpy as np
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
    qv = _embed_q(query)
    out = {}
    if store in ("both","community"):
        out["community"] = _search(_comm_ix, _comm_store, qv, k_comm)
    if store in ("both","verified"):
        out["verified"]  = _search(_ver_ix,  _ver_store,  qv, k_ver)
    return out

# Simple “fact-check”: supported if there’s a high-sim verified chunk containing key terms
def fact_check(claim: str, verified_hits: List[Dict[str,Any]]):
    text = " ".join(h["text"] for h in verified_hits)
    claim_l = claim.lower()
    supported = sum(kw in text.lower() for kw in ["randomized","meta-analysis","mps","hypertrophy","leucine","diaux","pdcaas","diaas","strength"]) >= 1
    return {"label": "supported" if supported else "mixed", "why": "Heuristic check; refine with LLM if needed."}

def rank_insights(items: List[Dict[str,Any]]):
    # toy ranking: favor verified first, then score, then recency if you add timestamps later
    return sorted(items, key=lambda x: (not x.get("is_verified", False), -x.get("score",0.0)))

# ---- Planner + loop ----
SYSTEM = """You are a nutrition analyst agent. Plan tool calls as JSON, then act. Tools: search_vector, fact_check, rank_insights.
Be concise, verify claims with verified evidence when possible. Output final answer with sections:
Community Buzz, Verified Evidence, Verdict, Next Actions.
"""

PLAN_PROMPT = """Question: "{q}"

Available tools:
- search_vector(query, store="community|verified|both")
- fact_check(claim, verified_hits)
- rank_insights(items)

Return a JSON list of steps. Examples:
[{{"tool":"search_vector","input":{{"query":"{q}","store":"both"}}}},
 {{"tool":"fact_check","input":{{"claim":"whey > plant for MPS"}}}},
 {{"tool":"rank_insights","input":{{"items":"$LAST_RESULT"}}}}]
"""

def run_agent(question: str):
    # 1) Plan
    plan_txt = llm_chat(SYSTEM, PLAN_PROMPT.format(q=question))
    try:
        plan = json.loads(plan_txt)
    except Exception:
        # fallback plan
        plan = [{"tool":"search_vector","input":{"query":question,"store":"both"}}]
    state = {"question": question, "trace": [], "last": None}

    # 2) Act
    for step in plan[:6]:  # cap steps
        tool = step.get("tool")
        inp  = step.get("input", {})
        if isinstance(inp, dict) and inp.get("items") == "$LAST_RESULT":
            inp["items"] = state["last"]
        if tool == "search_vector":
            res = search_vector(inp.get("query", question), inp.get("store","both"))
        elif tool == "fact_check":
            # needs verified hits from last search
            ver_hits = []
            if isinstance(state["last"], dict) and "verified" in state["last"]:
                ver_hits = state["last"]["verified"]
            res = fact_check(inp.get("claim", question), ver_hits)
        elif tool == "rank_insights":
            res = rank_insights(inp.get("items") or [])
        else:
            res = {"error": f"unknown tool {tool}"}
        state["trace"].append({"step": step, "result_preview": str(res)[:400]})
        state["last"] = res

    # 3) Compose final answer
    # Build small context from latest results
    comm = []
    ver  = []
    if isinstance(state["last"], dict) and "community" in state["last"]:
        comm = state["last"]["community"]
        ver  = state["last"]["verified"]
    elif isinstance(state["last"], list) and state["trace"]:
        # previous search is in trace
        for t in reversed(state["trace"]):
            if isinstance(t.get("result_preview"), str) and "community" in t["result_preview"]:
                # ignore preview parsing; just break loop; in real code store full objects
                break

    community_cites = "\n".join(f"- {h.get('title','')[:80]} ({h.get('url','')})" for h in (comm or []))
    verified_cites  = "\n".join(f"- {h.get('title','')[:80]} ({h.get('url','')})" for h in (ver  or []))

    synthesis_prompt = f"""Question: {question}

Community snippets:
{community_cites}

Verified snippets:
{verified_cites}

Write:
1) Community Buzz: 3-5 bullets (do NOT claim truth).
2) Verified Evidence: 3-5 bullets from verified hits only.
3) Verdict: what's supported vs still unverified; practical takeaway.
4) Citations (URLs only, short list)."""

    answer = llm_chat(SYSTEM, synthesis_prompt)
    print(answer)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Top protein trends this week"
    run_agent(q)