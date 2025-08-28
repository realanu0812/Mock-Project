# phase3/app.py
from __future__ import annotations

# --- Make local packages importable when running from phase3/ ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent  # .../nutrition_insights/phase3
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Std libs / third-party ---
import json
import pandas as pd
import streamlit as st

# --- Local imports (relative to phase3/) ---
from utils.common import (
    APP_TITLE, APP_VERSION, utc_now, data_dir_default
)
from utils.data import load_data, load_meta
from components import header, overview, insights, trending, business, chatbot, export


# =============================
# Helpers
# =============================
def find_data_dir() -> Path | None:
    """Best-effort discovery of the data folder."""
    candidates = [
        ROOT.parent / "data",                         # nutrition_insights/data
        ROOT.parent.parent / "data",                  # repo/data
        Path.cwd() / "nutrition_insights" / "data",   # CWD/nutrition_insights/data
        Path.cwd() / "data",                          # CWD/data
    ]
    return data_dir_default(candidates)  # returns first that exists (or None)

def read_styles() -> str:
    css_path = ROOT / "assets" / "styles.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""

def _force_hashable(val):
    """Make nested lists/dicts cache-hashable for Streamlit."""
    if isinstance(val, dict):
        return {k: _force_hashable(v) for k, v in val.items()}
    if isinstance(val, (list, set, tuple)):
        # serialize lists/sets/tuples deterministically
        return json.dumps([_force_hashable(x) for x in list(val)], ensure_ascii=False)
    return val

def _force_hashable_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    for c in df.columns:
        # Ensure all columns are string-lowered for robust source detection
        df[c] = df[c].apply(_force_hashable)
        if c in ["source", "source_type"]:
            df[c] = df[c].astype(str).str.lower()
    return df


# =============================
# Streamlit Config & Styles
# =============================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css = read_styles()
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# Header (support both render_header and render)
try:
    header.render_header(APP_TITLE, APP_VERSION)
except AttributeError:
    header.render(APP_TITLE, APP_VERSION)


# =============================
# Sidebar Navigation
# =============================
DATA_DIR = find_data_dir()
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4em; }
        .sidebar-nav-radio .stRadio label { font-size: 1.02rem; font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    SECTION = st.radio(
        "Go to",
        ["Research", "Business Strategies", "ChatBot", "Email/Export"],
        index=0,
        key="sidebar_nav_radio",
    )


# =============================
# Data Loading (cached)
# =============================

def _load_df(data_dir: Path | None) -> pd.DataFrame | None:
    # Always load from filtered corpus
    if data_dir is None:
        return None
    filtered_path = data_dir / "corpus_filtered.jsonl"
    if not filtered_path.exists():
        return None
    with open(filtered_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)
    return _force_hashable_df(df)


def _load_meta(data_dir: Path | None) -> dict:
    if data_dir is None:
        return {}
    return load_meta(data_dir)

df = _load_df(DATA_DIR)
meta = _load_meta(DATA_DIR)
last_updated = meta.get("last_updated") or utc_now()


# =============================
# Safe runner
# =============================
def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"Section failed: {e}")
        st.exception(e)


# =============================
# Routing
# =============================
if SECTION == "Research":
    st.markdown("## Research")

    # 1) Overview
    # Use overview.render_overview for correct signature
    _safe(overview.render_overview, df, meta)

    # 2) Verified Findings (Journals)
    st.markdown("### Verified Findings (Journals)")
    _safe(insights.render, df=df, meta=meta)

    # 3) Community Buzz (Reddit + Blogs)
    st.markdown("### Community Buzz")
    _safe(insights.render_buzz, df=df, meta=meta)


elif SECTION == "Business Strategies":
    st.markdown("## Business Strategies")
    _safe(business.render, source_filter="All", model=None)

elif SECTION == "ChatBot":
    st.markdown("## ChatBot")
    _safe(chatbot.render, df, source_filter="All", window_days=30)

elif SECTION == "Email/Export":
    st.markdown("## Email / Export")
    # support either export.render or export.render_export
    if hasattr(export, "render"):
        _safe(export.render, df=df, source_filter="All", window_days=30)
    else:
        _safe(getattr(export, "render_export"), df=df, source_filter="All", window_days=30)


# =============================
# Footer
# =============================
st.markdown("""
    <style>
        .proteinscope-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100vw;
            z-index: 1000;
            background: linear-gradient(90deg,#18181b 0%,#232326 100%);
            padding: 7px 0 14px 0;
            margin: 0;
            border-top: 1px solid #222;
        }
        .proteinscope-footer-content {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            width: calc(100vw - 48px);
            max-width: 1600px;
            margin: 0 auto;
        }
        .proteinscope-footer-content span {
            font-size: 0.82rem;
            color: #bdbdbd;
            font-family: Montserrat,Arial,sans-serif;
            font-weight: 400;
            opacity: 0.7;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        @media (max-width: 900px) {
            .proteinscope-footer-content {
                flex-direction: column;
                align-items: flex-start;
                gap: 2px;
            }
            .proteinscope-footer-content span {
                font-size: 0.78rem;
                width: 100%;
            }
        }
    </style>
    <div class='proteinscope-footer'>
        <div class='proteinscope-footer-content'>
            <span>© ProteinScope • {APP_VERSION}</span>
            <span>Last updated (IST): {last_updated_str}</span>
            <span>Sources: Reddit, PubMed, curated blogs • Not medical or financial advice.</span>
        </div>
    </div>
""", unsafe_allow_html=True)