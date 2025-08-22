# phase3/components/chatbot.py
from __future__ import annotations

import streamlit as st
import pandas as pd

from services.query_router import is_in_scope, build_context_snippets
from utils.ollama_client import chat_completion


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

    # Filter dataframe by source_filter and window_days before building context snippets
    df_filtered = df.copy()
    if source_filter and source_filter != "All" and "source" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["source"].str.lower() == source_filter.lower()]
    if window_days and "date" in df_filtered.columns:
        try:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
            df_filtered = df_filtered[pd.to_datetime(df_filtered["date"], errors="coerce") >= cutoff]
        except Exception:
            pass
    snippets = build_context_snippets(df_filtered, q, topn=8)
    context = _format_context(snippets)

    # Compose prompt
    user_prompt = (
        "Question:\n"
        f"{q}\n\n"
        "Context snippets (use these; cite inline like [1], [2] where helpful):\n"
        f"{context}\n\n"
        "Answer concisely for a business/product audience. If insufficient context or off-topic, reply exactly with:\n"
        f"{OUT_OF_SCOPE}"
    )

    # Call LLM
    with st.spinner("Thinking..."):
        reply = chat_completion(
            prompt=user_prompt,
            system=_system_prompt(),
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
                st.caption((getattr(s, "excerpt", "")[:400]) + ("..." if len(getattr(s, "excerpt", "")) > 400 else ""))
        else:
            st.caption("No matching snippets.")