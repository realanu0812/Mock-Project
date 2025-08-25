# phase3/components/header.py
from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def render_header(title: str, version: str) -> None:
    """
    Render the dashboard header used by app.py:
      header.render_header(APP_TITLE, APP_VERSION)
    """
    # Top title row
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(f"# {title}")
        st.caption(f"Version {version}")
    with cols[1]:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9rem; color:var(--text-color, #666);'>"
            f"Last updated (UTC): {_utc_now_str()}</div>",
            unsafe_allow_html=True,
        )

    # Subtle divider
    st.markdown(
        "<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(0,0,0,0.12),transparent);'/>",
        unsafe_allow_html=True,
    )