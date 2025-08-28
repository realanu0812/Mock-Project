# phase3/components/chatbot.py
from __future__ import annotations

import streamlit as st
import pandas as pd

from services.query_router import is_in_scope, build_context_snippets
from utils.gemini_client import chat_completion

def get_gemini_response(prompt: str, system: str = None, timeout: int = 60) -> str:
    return chat_completion(prompt, system=system, timeout=timeout)


OUT_OF_SCOPE = (
    "Out of scope — this question is not related to protein or the available data."
)


def _init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[dict(role, content)]


def _render_history():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


def _system_prompt() -> str:
    return (
        "You are ProteinScope Assistant. Answer strictly about dietary protein: "
        "types (whey/casein/plant), dosage, timing, safety, performance, consumer trends. "
        "Ground answers in the provided snippets. "
        "If you lack support or the query is off-topic, say: "
        f"\"{OUT_OF_SCOPE}\""
    )


def _format_context(snippets: list[dict]) -> str:
    # Each snippet: Snippet object with attributes
    blocks = []
    for i, s in enumerate(snippets, 1):
        src = getattr(s, "source", "unknown")
        dt = getattr(s, "date", "")
        title = getattr(s, "title", "") or ""
        text = getattr(s, "excerpt", "") or ""
        blocks.append(f"[{i}] ({src} | {dt}) {title}\n{text}")
    return "\n\n".join(blocks)


def render(df: pd.DataFrame, source_filter: str, window_days: int) -> None:
    st.subheader("Chatbot")
    _init_state()
    _render_history()

    q = st.chat_input("Ask about protein (timing, dose, safety, trends)...")
    if not q:
        st.caption("Tip: try “Best time for whey?”, “Collagen trending complaints?”, “BCAA vs EAA?”.")
        return

    # Guardrail
    if not is_in_scope(q):
        with st.chat_message("assistant"):
            st.write(OUT_OF_SCOPE)
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": OUT_OF_SCOPE})
        return

    # Efficient DataFrame filtering with Streamlit caching
    @st.cache_data(show_spinner=False)
    def filter_df(df, source_filter, window_days):
        # Ignore all filtering; return the full DataFrame (date filtering is handled at scraping)
        return df


    @st.cache_data(show_spinner=False)
    def cached_snippets(df_filtered, q):
        # Lower strictness: allow partial matches, more snippets, and lower min_chars
        try:
            return build_context_snippets(df_filtered, q, topn=15, per_source_cap=6, min_chars=40, max_chars=900, recency_boost=0.0)
        except Exception as e:
            st.warning(f"Context build failed: {e}")
            return []

    import time
    start_snip = time.time()
    df_filtered = filter_df(df, source_filter, window_days)
    snippets = cached_snippets(df_filtered, q)
    snip_time = time.time() - start_snip
    if snip_time > 2:
        st.warning(f"Context building took {snip_time:.2f}s. Consider optimizing index or filtering.")

    # If no snippets found, fallback to FAISS vector search
    used_faiss = False
    if not snippets:
        try:
            from components.faiss_utils import faiss_topk
            faiss_hits = faiss_topk(q, k=8)
            if faiss_hits:
                used_faiss = True
                # Convert FAISS hits to snippet-like dicts
                snippets = []
                for h in faiss_hits:
                    snippets.append(type('Snippet', (), h))
        except Exception as e:
            st.warning(f"FAISS fallback failed: {e}")

    # Pre-extract attributes for formatting (faster than getattr in loop)
    context_blocks = []
    for i, s in enumerate(snippets, 1):
        src = getattr(s, "source", "unknown")
        dt = getattr(s, "date", "")
        title = getattr(s, "title", "") or ""
        excerpt = getattr(s, "excerpt", getattr(s, "text", "")) or ""
        context_blocks.append(f"[{i}] ({src} | {dt}) {title}\n{excerpt}")
    context = "\n\n".join(context_blocks)

    # Compose prompt
    user_prompt = (
        "Question:\n"
        f"{q}\n\n"
        "Context snippets (use these; cite inline like [1], [2] where helpful):\n"
        f"{context}\n\n"
        "Answer concisely for a business/product audience. If insufficient context or off-topic, reply exactly with:\n"
        f"{OUT_OF_SCOPE}"
    )

    # Call LLM with increased timeout for slow responses
    with st.spinner("Thinking..."):
        reply = chat_completion(
            prompt=user_prompt,
            system=_system_prompt(),
            timeout=180,  # Increased timeout for Gemini API
        )

    # Show and persist
    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.chat_history.append({"role": "assistant", "content": reply or OUT_OF_SCOPE})

    with st.chat_message("assistant"):
        st.markdown(reply or OUT_OF_SCOPE)

    # Optional: show which snippets were used
    with st.expander("Context used"):
        if snippets:
            for i, s in enumerate(snippets, 1):
                st.markdown(f"**[{i}] {getattr(s, 'title', '(no title)')}** — {getattr(s, 'source', '?')} | {getattr(s, 'date', '')}")
                excerpt = getattr(s, "excerpt", getattr(s, "text", ""))
                st.caption((excerpt[:400]) + ("..." if len(excerpt) > 400 else ""))
        else:
            st.caption("No matching snippets.")
    if used_faiss:
        st.info("Used semantic search (FAISS) fallback for context.")