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
    # Premium centered header (black & white, larger, minimal)
    st.markdown(
        """
        <div style='width:100%;text-align:center;margin-top:10px;margin-bottom:8px;'>
            <span style='font-family:Montserrat,Segoe UI,sans-serif;font-weight:900;font-size:2.8rem;letter-spacing:0.04em;color:#18181b;'>ProteinScope</span><br>
            <span style='font-family:Montserrat,Segoe UI,sans-serif;font-weight:600;font-size:1.25rem;color:#232326;opacity:0.85;'>v0.1.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='text-align:center; font-size:1.05rem; color:#666; margin-bottom:2px;'>Last updated (UTC): {_utc_now_str()}</div>",
        unsafe_allow_html=True,
    )
    # Subtle divider
    st.markdown(
        "<hr style='border:none;height:2px;background:linear-gradient(90deg,rgba(0,0,0,0.10),rgba(0,0,0,0.18),rgba(0,0,0,0.10));margin-bottom:0;'/>",
        unsafe_allow_html=True,
    )