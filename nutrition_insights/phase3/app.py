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

# -----------------------------
# Load data/meta once per run (move up so sidebar can use)
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

def format_last_refreshed(ts):
    import datetime
    if not ts:
        return "—"
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.datetime.utcfromtimestamp(ts)
        elif isinstance(ts, str):
            dt = datetime.datetime.fromisoformat(ts)
        elif isinstance(ts, datetime.datetime):
            dt = ts
        else:
            return str(ts)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)

# -----------------------------
# Sidebar navigation & inputs
# -----------------------------

# --- Custom Sidebar Layout ---
DATA_DIR = find_data_dir()
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5em; }
    .sidebar-section { margin-bottom: 1.2em; }
    .sidebar-nav-radio .stRadio label { font-size: 1.08rem; font-weight: 500; }
    .sidebar-data-box { background: #23323c; color: #fff; border-radius: 10px; padding: 1em 1em 0.5em 1em; margin-bottom: 1em; }
    .sidebar-refreshed { font-size: 0.95em; color: #6fa3ef; margin-top: 0.5em; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    SECTION = st.radio(
        "Go to",
        ["Research", "Business Strategies", "ChatBot", "Email/Export"],
        index=0,
        key="sidebar_nav_radio"
    )

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

import datetime
refresh_clicked = False
if SECTION == "Research":
    st.markdown("---")
else:
    st.markdown("---")

def format_last_refreshed(ts):
    if not ts:
        return "—"
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.datetime.utcfromtimestamp(ts)
        elif isinstance(ts, str):
            dt = datetime.datetime.fromisoformat(ts)
        elif isinstance(ts, datetime.datetime):
            dt = ts
        else:
            return str(ts)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


# -----------------------------
# Route to sections (robust)
# -----------------------------
def _safe(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except Exception as e:
        st.error(f"Section failed: {e}")
        st.exception(e)

if SECTION == "Research":
    # Research: overview, insights, graphs, community buzz
    # Data Overview only once
    overview.render_overview(df, meta)
    # Verified findings vs Community Buzz side by side (only show once)
    c1, c2 = st.columns([2, 2])
    with c1:
        st.subheader("Verified Findings (Journals)")
        from components.insights import _pick_journal_rows, _compose_context, summarize_bullets
        import pandas as pd
        journal_df = None
        today = pd.Timestamp("2025-08-22", tz="UTC")
        if df is not None:
            if "source" in df.columns:
                journal_df = df[df["source"].fillna("").astype(str).str.lower().str.strip() == "journals"]
            elif "source_type" in df.columns:
                journal_df = df[df["source_type"].fillna("").astype(str).str.lower().str.strip() == "journal_article"]
        # Remove future-dated journals
        if journal_df is not None and "date" in journal_df.columns:
            journal_df = journal_df.copy()
            journal_df["date_parsed"] = pd.to_datetime(journal_df["date"], errors="coerce")
            journal_df = journal_df[journal_df["date_parsed"] <= today]
        # Filter for protein-related keywords
        keywords = protein_keywords() if callable(protein_keywords) else protein_keywords
        def row_has_protein(row):
            text_fields = [str(row.get(f, "")).lower() for f in ["abstract", "summary", "content", "text", "title"]]
            combined = " ".join(text_fields)
            return any(kw.lower() in combined for kw in keywords)
        if journal_df is not None:
            journal_df = journal_df[journal_df.apply(row_has_protein, axis=1)]
        journal_count = len(journal_df) if journal_df is not None else 0
        st.caption(f"Journal records found (protein-related): {journal_count}")
        if journal_df is not None and not journal_df.empty:
            rows = _pick_journal_rows(journal_df, max_items=12)
            st.caption(f"Rows used for summary: {len(rows)}")
            if not rows.empty:
                # Fast fetch: summarize directly from combined.json (skip row iteration)
                from utils.gemini_client import chat_completion
                def get_gemini_response(prompt: str, system: str = None, timeout: int = 60) -> str:
                    return chat_completion(prompt, system=system, timeout=timeout)
                import json
                combined_path = DATA_DIR / "combined.json"
                try:
                    with open(combined_path, "r", encoding="utf-8") as f:
                        combined_data = json.load(f)
                except Exception as e:
                    combined_data = []
                # Filter for protein-related journal entries only
                keywords = protein_keywords() if callable(protein_keywords) else protein_keywords
                def is_protein_related(entry):
                    text_fields = [str(entry.get(f, "")).lower() for f in ["abstract", "summary", "content", "text", "title"]]
                    combined = " ".join(text_fields)
                    return any(kw.lower() in combined for kw in keywords)
                journal_entries = [e for e in combined_data if (str(e.get("source_type", "")).lower().strip() == "journal_article" or str(e.get("source", "")).lower().strip() == "journals") and is_protein_related(e)]
                # Remove future-dated journals
                from pandas import to_datetime
                today = to_datetime("2025-08-22", utc=True)
                filtered_entries = []
                for e in journal_entries:
                    try:
                        dt = to_datetime(e.get("date", None), errors="coerce")
                        if dt is not None and dt <= today:
                            filtered_entries.append(e)
                    except Exception:
                        pass
                # Use up to 12 entries for context, fallback if empty
                context_texts = []
                for r in filtered_entries[:12]:
                    for field in ["abstract", "summary", "content", "text", "title"]:
                        v = r.get(field, None)
                        if v:
                            context_texts.append(str(v))
                            break
                # Fallback: if no valid context, use top 12 protein-related journal entries from combined.json
                if not context_texts:
                    for r in journal_entries[:12]:
                        for field in ["abstract", "summary", "content", "text", "title"]:
                            v = r.get(field, None)
                            if v:
                                context_texts.append(str(v))
                                break
                context_str = "\n".join(context_texts)
                prompt = (
                    "Summarize the following protein-related journal evidence into 3-5 concise, actionable findings for product teams. Each bullet should be a short, plain-language sentence, not just a keyword.\n\n" + context_str
                )
                if not context_texts or len(context_str.strip()) < 40:
                    st.info("Not enough protein-related evidence for summarization. Showing raw evidence below.")
                    for i, txt in enumerate(context_texts):
                        st.markdown(f"**Evidence {i+1}:** {txt}")
                    summary_bullets = []
                else:
                    try:
                        summary_text = chat_completion(prompt)
                        import re
                        bullet_match = re.search(r"(^|\n)([-•]\s+)", summary_text)
                        if bullet_match:
                            start_idx = bullet_match.start(2)
                            summary_text = summary_text[start_idx:]
                        summary_bullets = [
                            b.lstrip("-• ").strip()
                            for b in summary_text.splitlines()
                            if b.strip().startswith(('-', '•')) and not b.lower().startswith("⚠️ (gemini api error)") and not b.lower().startswith("⚠️ (gemini)")
                        ]
                        if not summary_bullets:
                            summary_text_clean = summary_text.replace(prompt, "")
                            summary_bullets = re.split(r'(?<=[.!?]) +', summary_text_clean.strip())
                            summary_bullets = [b.strip() for b in summary_bullets if b]
                        # If Gemini returns a fallback message, show raw evidence
                        if any("no relevant protein-related summary" in b.lower() for b in summary_bullets):
                            st.info("Gemini could not generate a relevant summary. Showing raw evidence below.")
                            for i, txt in enumerate(context_texts):
                                st.markdown(f"**Evidence {i+1}:** {txt}")
                            summary_bullets = []
                    except Exception as e:
                        summary_bullets = ["Summary unavailable due to error."]
                st.markdown("**Short Summaries (Protein-related Journal Evidence):**")
                for b in summary_bullets:
                    st.markdown(f"- {b}")
                # Show sources used with improved title fetching
                with st.expander("Show sources used"):
                    import pandas as pd
                    def get_best_title(r):
                        title = str(r.get("title", "")).strip()
                        if not title:
                            # Try to fetch from combined.json if available
                            url = str(r.get("url", "")).strip()
                            found = None
                            for entry in combined_data:
                                if str(entry.get("url", "")).strip() == url:
                                    found = entry
                                    break
                            if found:
                                title = str(found.get("title", "")).strip()
                        if not title:
                            title = str(r.get("summary", "")).strip()
                        if not title:
                            title = str(r.get("headline", "")).strip()
                        if not title:
                            title = str(r.get("content", "")).strip()
                        if not title:
                            title = "No Title"
                        return title
                    show = pd.DataFrame({
                        "#": [f"[{i+1}]" for i in range(len(rows))],
                        "Title": [get_best_title(r) for _, r in rows.iterrows()],
                        "Journal": [str(r.get("journal", r.get("venue", "Journal"))) for _, r in rows.iterrows()],
                        "Date Published": [str(r.get("date", "")) for _, r in rows.iterrows()],
                        "Link": [str(r.get("url", "")) for _, r in rows.iterrows()],
                    })
                    st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info("No suitable journal rows found.")
        else:
            st.info("No journal findings available.")
    with c2:
        st.subheader("Community Buzz (Latest Unverified Sources)")
        from components.insights import _pick_buzz_rows, _compose_context, summarize_bullets
        buzz_df = None
        # Directly load and filter the raw data corpus for Reddit/Blog records
        import json
        combined_path = DATA_DIR / "combined.json"
        buzz_types = ["blog_article", "reddit_post", "reddit_comment"]
        buzz_sources = ["blogs", "reddit"]
        try:
            with open(combined_path, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
        except Exception as e:
            combined_data = []
        # Filter for Reddit/Blog records
        buzz_entries = [e for e in combined_data if (str(e.get("source_type", "")).lower().strip() in buzz_types or str(e.get("source", "")).lower().strip() in buzz_sources)]
        buzz_count = len(buzz_entries)
        st.caption(f"Reddit/Blogs records found: {buzz_count}")
        if buzz_entries:
            import pandas as pd
            buzz_df = pd.DataFrame(buzz_entries)
            rows = _pick_buzz_rows(buzz_df, max_items=12)
            st.caption(f"Rows used for summary: {len(rows)}")
            if not rows.empty:
                context = _compose_context(rows)
                prompt = (
                    "You are summarizing community buzz from Reddit and Blogs for product teams.\n"
                    "From the BUZZ EVIDENCE below, extract **exactly five** concise, actionable findings relevant to the protein market (trends, consumer opinions, product feedback, safety, outcomes). Each bullet must be a short, plain-language sentence, not just a keyword.\n\n"
                    f"{context}\n"
                )
                bullets = summarize_bullets([prompt], n=5)
                fallback_needed = (
                    not bullets or
                    all(
                        b.strip().lower().startswith("i cannot provide") or
                        b.strip().lower().startswith("can i help you") or
                        b.strip().startswith("you are summarizing") or
                        b.strip().startswith("from the buzz evidence")
                        for b in bullets
                    )
                )
                st.markdown("**Actionable Findings:**")
                if not fallback_needed:
                    for b in bullets:
                        st.markdown(f"- {b}")
                    with st.expander("Show sources used"):
                        cols = [c for c in ["title", "url", "date"] if c in rows.columns]
                        if cols:
                            show = rows[cols].rename({"title": "Title", "url": "Link", "date": "Date"}, axis=1)
                            st.dataframe(show, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No source columns available to display.")
                else:
                    st.info("Buzz summary unavailable — showing top items instead.")
                    top = rows.head(5)
                    for _, r in top.iterrows():
                        t = str(r.get("title", "Untitled"))
                        link = str(r.get("url", "")) or None
                        dt = ""
                        if pd.notna(r.get("date")):
                            try:
                                dt = pd.to_datetime(r["date"]).date().isoformat()
                            except Exception:
                                dt = ""
                        meta = " • ".join([x for x in [dt] if x])
                        if link:
                            st.markdown(f"**[{t}]({link})**")
                        else:
                            st.markdown(f"**{t}**")
                        if meta:
                            st.caption(meta)
            else:
                st.info("No suitable buzz rows found. If you expect data here, check your sources or refresh.")
        else:
            st.info("No Reddit/Blogs records found in the data. If you expect data here, check your sources or refresh.")
    st.markdown("---")
    # Graphs (monthly volume by source with navigation)
    st.subheader("Monthly Post Volume by Source")
    source_options = ["journals", "blogs", "reddit"]
    selected_source = st.radio("Select Source", source_options, horizontal=True)
    with st.spinner(f"Loading monthly volume for {selected_source.title()}..."):
        _safe(volume.render, df, source_filter=selected_source, window_days=365)

elif SECTION == "Business Strategies":
    # Business: practical strategies, brand comparison
    st.subheader("Business Strategies")
    _safe(business.render, df, source_filter="All", model=None)
    st.markdown("---")
    st.subheader("Brand Comparison")
    # TODO: Add brand comparison logic here

elif SECTION == "ChatBot":
    st.subheader("ChatBot")
    _safe(chatbot.render, df, source_filter="All", window_days=30)

elif SECTION == "Email/Export":
    st.subheader("Email / Export Functionality")
    _safe(export.render, df, source_filter="All", window_days=30)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
c1, c2, c3 = st.columns([2, 2, 6])
with c1:
    st.caption(f"© ProteinScope • {APP_VERSION}")
with c2:
    import pytz, datetime
    try:
        dt = datetime.datetime.fromisoformat(str(last_updated))
        dt_ist = dt.astimezone(pytz.timezone("Asia/Kolkata"))
        last_updated_str = dt_ist.strftime("%d-%b-%Y %I:%M %p IST")
    except Exception:
        last_updated_str = str(last_updated)
    st.caption(f"Last updated (IST): {last_updated_str}")
with c3:
    st.caption("Sources: Reddit, PubMed, curated blogs • Not medical or financial advice.")