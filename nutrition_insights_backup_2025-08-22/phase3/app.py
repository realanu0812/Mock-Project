# phase3/app.py
from __future__ import annotations
import sys, pathlib
from pathlib import Path

# Make 'phase3' folder importable as top-level (utils/, services/, components/)
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent          # .../nutrition_insights/phase3
PKG_ROOT = HERE                                  # keep everything relative to phase3
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# --- Imports (use local packages, no absolute 'nutrition_insights.' required) ---
import streamlit as st

from utils.common import (
    APP_TITLE, APP_VERSION, utc_now, data_dir_default, pretty_int
)
from utils.data import load_data, load_meta  # assumes your existing utils/data.py
from components import header, overview, trending, insights, business, volume, chatbot, export
from utils.config_loader import protein_keywords

import os
SAFE_BOOT = os.getenv("PHASE3_SAFE_BOOT", "0") == "1"

st.sidebar.info(
    f"🔧 Safe Boot: {'ON' if SAFE_BOOT else 'OFF'}\n"
    f"NI_DATA_DIR: {os.getenv('NI_DATA_DIR', '(not set)')}"
)
# -----------------------------
# Config & lightweight helpers
# -----------------------------
def find_data_dir() -> Path | None:
    """Best-effort discovery of the data folder."""
    candidates = [
        HERE.parent / "data",                         # nutrition_insights/data
        HERE.parent.parent / "data",                  # repo/data
        Path.cwd() / "nutrition_insights" / "data",   # CWD/nutrition_insights/data
        Path.cwd() / "data",                          # CWD/data
    ]
    return data_dir_default(candidates)  # returns first existing or None

def read_styles() -> str:
    css_path = HERE / "assets" / "styles.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS (optional)
_css = read_styles()
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# Header
try:
    header.render_header(APP_TITLE, APP_VERSION)
except AttributeError:
    # Fallback if the component standardized on `render(...)`
    header.render(APP_TITLE, APP_VERSION)

# -----------------------------
# Sidebar navigation & inputs
# -----------------------------
st.sidebar.header("Navigation")
SECTION = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Trending Topics",
        "Insights (Journals)",
        "Business Highlights",
        "Volume & Relationships",
        "Chatbot",
        "Export",
    ],
    index=0,
)

st.sidebar.header("Data")
DATA_DIR = find_data_dir()
if DATA_DIR is None:
    st.sidebar.error("Could not locate a data directory. Expected 'nutrition_insights/data' or 'data'.")
else:
    st.sidebar.success(f"Data dir: {DATA_DIR}")

# -----------------------------
# Load data/meta once per run
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_df(data_dir: Path | None):
    if data_dir is None:
        return None
    return load_data(data_dir)

@st.cache_data(show_spinner=False)
def _load_meta(data_dir: Path | None):
    if data_dir is None:
        return {}
    return load_meta(data_dir)

df = _load_df(DATA_DIR)
meta = _load_meta(DATA_DIR)
last_updated = meta.get("last_updated") or utc_now()

# Overview metrics strip (always visible)
with st.container():
    try:
        overview.render_overview(df, meta)
    except AttributeError:
        overview.render(df, meta)

st.markdown("---")

# -----------------------------
# Route to sections (robust)
# -----------------------------
def _safe(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except Exception as e:  # show but don't break the app
        st.error(f"Section failed: {e}")
        st.exception(e)

if SECTION == "Overview":
    # Overview already shows KPI; add any extra overviews here if needed.
    pass
elif SECTION == "Trending Topics":
    _safe(trending.render, df, keywords=protein_keywords(), source_filter="All")
elif SECTION == "Insights (Journals)":
    _safe(insights.render, df, source_filter="All", model=None)
elif SECTION == "Business Highlights":
    _safe(business.render, df, source_filter="All", model=None)
elif SECTION == "Volume & Relationships":
    _safe(volume.render, df, source_filter="All", window_days=30)
elif SECTION == "Chatbot":
    _safe(chatbot.render, df, source_filter="All", window_days=30)
elif SECTION == "Export":
    _safe(export.render, df, source_filter="All", window_days=30)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
c1, c2, c3 = st.columns([2, 2, 6])
with c1:
    st.caption(f"© ProteinScope • {APP_VERSION}")
with c2:
    st.caption(f"Last updated (UTC): {last_updated}")
with c3:
    st.caption("Sources: Reddit, PubMed, curated blogs • Not medical or financial advice.")