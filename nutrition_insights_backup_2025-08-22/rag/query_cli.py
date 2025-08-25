# nutrition_insights/rag/query_cli.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# ---------------- Path setup so imports work no matter how we run this ----------------
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]          # .../nutrition_insights
REPO_ROOT = PKG_ROOT.parent         # repo root

# Ensure packages exist for imports like nutrition_insights.*
for d in (PKG_ROOT, PKG_ROOT / "rag"):
    initf = d / "__init__.py"
    if not initf.exists():
        try:
            initf.write_text("", encoding="utf-8")
        except Exception:
            pass

# Make repo root & package root importable
for p in (str(REPO_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------- Robust import of llm_connection ----------------
class LLMError(Exception):
    pass

def _import_llm():
    # 1) preferred: nutrition_insights.rag.llm_connection
    try:
        from nutrition_insights.rag.llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass
    # 2) common: nutrition_insights/llm_connection.py
    try:
        from nutrition_insights.llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass
    # 3) same-directory fallback
    try:
        from llm_connection import get_chat_fn, LLMError as _LE  # type: ignore
        return get_chat_fn, _LE
    except Exception:
        pass

    def _stub(*_, **__):
        raise LLMError(
            "Could not import llm_connection. Place it at either:\n"
            " - nutrition_insights/llm_connection.py, or\n"
            " - nutrition_insights/rag/llm_connection.py\n"
            "and ensure __init__.py files exist."
        )
    return _stub, LLMError

get_chat_fn, LLMError = _import_llm()

# ---------------- Paths & data ----------------
DATA = PKG_ROOT / "data"
FAISS_INDEX = DATA / "faiss.index"
INDEX_META  = DATA / "index_meta.jsonl"
INDEX_INFO  = DATA / "index_info.json"
FILTERED    = DATA / "corpus_filtered.jsonl"

# ---------------- Retrieval (FAISS) ----------------
import numpy as np
try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("FAISS is required. Install with: pip install faiss-cpu") from e

def load_index():
    if not FAISS_INDEX.exists() or not INDEX_META.exists():
        raise SystemExit("Missing FAISS files; run build_index.py first.")
    index = faiss.read_index(str(FAISS_INDEX))
    meta = [json.loads(line) for line in INDEX_META.read_text(encoding="utf-8").splitlines() if line.strip()]
    return index, meta

# NOTE: build_index uses an embedding function; at query time we mirror dim with a stable hash.
def embed_query(q: str, dim: int = 768) -> np.ndarray:
    import hashlib
    h = hashlib.sha256(q.encode("utf-8")).digest()
    vals = [((h[i % len(h)] / 255.0) * 2.0 - 1.0) for i in range(dim)]
    v = np.array(vals, dtype="float32")
    n = np.linalg.norm(v) + 1e-9
    return (v / n).reshape(1, -1)

def topk(index, meta, qvec, k: int):
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

# ---------------- RAG assembly & guardrails ----------------
SYSTEM = (
    "You are ProteinScope, a nutrition domain assistant. Answer only about dietary protein, "
    "protein supplements, amino acids, protein timing, dose, quality (DIAAS/PDCAAS), safety, "
    "performance, recovery, satiety, body composition, clinical use (e.g., sarcopenia), and related topics. "
    "If a question is out of scope (e.g., fashion, stocks), say it's out of scope."
)

def format_context(chunks: list[dict], ctx_per_source: int = 10, verified_boost: float = 0.1):
    # Prefer verified items by nudging their score
    scored = []
    for c in chunks:
        s = c.get("_score", 0.0)
        if c.get("is_verified"):
            s += verified_boost
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [c for _, c in scored[:ctx_per_source]]
    lines = []
    for i, c in enumerate(chosen, 1):
        src = c.get("source", "unknown")
        url = c.get("url") or c.get("link") or ""
        title = c.get("title") or c.get("id") or src
        body = c.get("text") or c.get("content") or ""
        when = c.get("published") or c.get("date") or ""
        lines.append(f"[{i}] {title} | {when} | {url}\n{body}\n")
    return "\n\n".join(lines), chosen

# —— Generous, opt‑in scope guard
def is_in_scope(question: str, top_hits: list[dict], scope_thresh: float, debug: bool = False) -> bool:
    """
    Decide if the question is about protein/nutrition. We allow:
    - Obvious protein terms (hard allow),
    - Timing/nutrition intent with protein terms,
    - Fallback via retrieved hits (esp. verified) with decent scores.
    """
    q = " ".join(question.strip().split()).lower()

    base_terms = {"protein", "whey", "casein", "eaa", "bcaa", "amino acid", "leucine",
                  "plant protein", "milk protein", "soy", "pea", "collagen"}
    timing_terms = {
        "timing", "at night", "before bed", "pre bed", "bedtime", "nighttime",
        "morning", "pre workout", "post workout", "after workout",
        "preworkout", "postworkout", "overnight", "sleep", "recovery",
        "muscle protein synthesis", "mps"
    }
    nutrition_terms = {
        "dose", "dosage", "scoop", "grams", "digestibility", "pdcaas", "diaas",
        "satiety", "hypertrophy", "strength", "performance", "body composition",
        "weight loss", "sarcopenia", "bioavailability", "absorption", "safety", "kidney"
    }

    # 1) Hard allow if base protein terms exist
    if any(t in q for t in base_terms):
        if debug: print("[SCOPE] Hard-allow: base term matched.")
        return True

    # 2) Allow intent + base terms
    if (any(t in q for t in timing_terms) and any(t in q for t in base_terms)) or \
       (any(t in q for t in nutrition_terms) and any(t in q for t in base_terms)):
        if debug: print("[SCOPE] Allow: timing/nutrition + base term matched.")
        return True

    # 3) Fallback: retrieved hits (especially verified) with decent scores
    for h in top_hits or []:
        s = float(h.get("_score", 0.0) or 0.0)
        is_verified = bool(h.get("is_verified", False))
        if is_verified and s >= max(0.05, scope_thresh * 0.5):
            if debug: print(f"[SCOPE] Allow via verified hit: score={s:.3f}")
            return True
        if s >= scope_thresh:
            if debug: print(f"[SCOPE] Allow via hit: score={s:.3f}")
            return True

    if debug:
        print("[SCOPE] Reject: no terms matched and no sufficiently strong hits.")
    return False

def main():
    ap = argparse.ArgumentParser(description="Query the FAISS index with guardrails + Ollama")
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=10)
    ap.add_argument("--verified-boost", type=float, default=0.10)
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    ap.add_argument("--temp", type=float, default=0.2)

    # Scope options (opt‑in)
    ap.add_argument("--strict", action="store_true",
                    help="If set, reject queries outside protein scope.")
    ap.add_argument("--scope-thresh", type=float, default=0.35,
                    help="Similarity/score threshold used only when --strict is on.")
    ap.add_argument("--debug-scope", action="store_true",
                    help="Print scope diagnostics when --strict is on.")

    # Minimal evidence gates
    ap.add_argument("--min-support", type=int, default=1,
                    help="Minimum retrieved hits required to attempt an answer.")
    ap.add_argument("--min-topscore", type=float, default=0.0,
                    help="If >0, require the top hit score to exceed this to answer.")

    args = ap.parse_args()

    # Normalize question early and use consistently
    question = " ".join(args.question.strip().split())

    # Load index & retrieve first (scope can use hits as a signal)
    index, meta = load_index()
    qvec = embed_query(question, dim=index.d)
    hits = topk(index, meta, qvec, args.k)

    # Optional strict scope guard
    if args.strict:
        if not is_in_scope(question, hits, args.scope_thresh, debug=args.debug_scope):
            print("This question looks outside the protein/nutrition scope. "
                  "Please ask about dietary protein, supplements, amino acids, "
                  "timing, dose, quality, safety, or performance.")
            sys.exit(0)

    # Basic support checks
    if len(hits) < args.min_support:
        print("Not enough supporting documents in the index to answer confidently.")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h.get('title','')[:80]} | {h.get('url','')}")
        sys.exit(0)
    if args.min_topscore > 0 and (hits[0].get("_score", 0.0) or 0.0) < args.min_topscore:
        print(f"Top hit score {hits[0].get('_score', 0.0):.3f} is below --min-topscore={args.min_topscore}.")
        sys.exit(0)

    # Prepare context for LLM
    ctx_text, chosen = format_context(hits, args.ctx, args.verified_boost)

    # IMPORTANT: Do NOT include SYSTEM text *inside* the user prompt (we pass it separately).
    user_prompt = (
        f"User question: {question}\n\n"
        f"Use the following context snippets (may include community + verified). "
        f"When making claims, lean on verified studies. If evidence is weak or mixed, say so explicitly.\n\n"
        f"{ctx_text}\n\n"
        f"Answer in 6–10 sentences. End with 3–6 bullet citations using the URLs from the snippets."
    )

    chat = get_chat_fn(model=args.model, temperature=args.temp)
    try:
        # FIX: call signature is (system, prompt)
        answer = chat(SYSTEM, user_prompt).strip()
        print(answer)
    except LLMError as e:
        print("[No LLM configured]", str(e))
        # Even without an LLM, print the sources to help debugging
        verified = [c for c in chosen if c.get("is_verified")]
        community = [c for c in chosen if not c.get("is_verified")]
        if verified:
            print("\nVerified Sources:")
            for i, c in enumerate(verified, 1):
                print(f"[{i}] {c.get('title','')[:80]} | {c.get('published','')} | {c.get('url','')}")
        if community:
            print("\nCommunity Sources:")
            for i, c in enumerate(community, 1):
                print(f"[{i}] {c.get('title','')[:80]} | {c.get('published','')} | {c.get('url','')}")
        print("\n\nCitations:")
        for c in verified + community:
            u = c.get("url") or c.get("link") or ""
            if u:
                print(u)
        sys.exit(0)

if __name__ == "__main__":
    main()