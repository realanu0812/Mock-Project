# phase3/components/business.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from textwrap import shorten
import json
import re
from collections import Counter, defaultdict

from utils.gemini_client import chat_completion
from utils.common import looks_protein_related


# ----------------- Helper Functions -----------------

def looks_booming(text: str) -> bool:
    """Detect 'booming' signals in text."""
    if not isinstance(text, str):
        return False
    booming_keywords = [
        "growth", "trend", "surge", "rising", "increasing", "hot", "booming",
        "popular", "demand", "momentum", "expanding", "spike", "uptick",
        "interest", "buzz", "adoption", "exploding", "taking off", "mainstream",
        "skyrocketing", "rapid", "accelerating"
    ]
    return any(word in text.lower() for word in booming_keywords)


def _slice_for_business(df: pd.DataFrame, source_filter: str, max_items: int = 120) -> pd.DataFrame:
    """Filter for recent, long-text entries."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if source_filter and source_filter != "All":
        sf = source_filter.lower()
        d = d[d["source"].str.lower() == sf]

    if "text" in d.columns:
        textlen = d["text"].fillna("").str.len()
    elif "abstract" in d.columns:
        textlen = d["abstract"].fillna("").str.len()
    else:
        textlen = pd.Series([0] * len(d))

    d = d.assign(_q=(textlen > 160).astype(int))
    if "date" in d.columns:
        d = d.sort_values(["_q", "date"], ascending=[False, False])
    else:
        d = d.sort_values("_q", ascending=False)

    return d.head(max_items).reset_index(drop=True)


def _compose_market_context(rows: pd.DataFrame, max_chars: int = 9000) -> str:
    """Compact context builder for Gemini."""
    lines = []
    for _, r in rows.iterrows():
        title = str(r.get("title") or r.get("kind") or r.get("subreddit") or "Item").strip()
        src = str(r.get("source", "") or "").strip().capitalize()
        body = str(r.get("abstract") or r.get("summary") or r.get("text") or "").replace("\n", " ").strip()
        body = shorten(body, width=420, placeholder=" …")

        meta = []
        if pd.notna(r.get("date")):
            try:
                meta.append(pd.to_datetime(r["date"]).date().isoformat())
            except Exception:
                pass
        if src:
            meta.append(src)

        meta_str = " • ".join(meta)
        head = f"- {title}" + (f" ({meta_str})" if meta_str else "")
        lines.append(f"{head}\n  {body}")

        if sum(len(x) for x in lines) > max_chars:
            break

    return "MARKET CONTEXT (mixed sources):\n" + "\n".join(lines)


# ---------- TEXT & KEYWORD HELPERS ----------

_WORD_BOUNDARY = r"(?i)(?<![A-Za-z0-9]){kw}(?![A-Za-z0-9])"

def _safe_iter_text(series):
    for x in series.fillna(""):
        if isinstance(x, str):
            yield x
        else:
            yield str(x)

def _count_keywords(text_series: pd.Series, keywords: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(keywords, 0)
    for txt in _safe_iter_text(text_series):
        for kw in keywords:
            if re.search(_WORD_BOUNDARY.format(kw=re.escape(kw)), txt):
                counts[kw] += 1
    return counts

def _count_keyword_groups(text_series: pd.Series, groups: dict[str, list[str]]) -> dict[str, int]:
    """Counts at group level (a hit for any word in the group counts once per document)."""
    group_counts = dict.fromkeys(groups.keys(), 0)
    for txt in _safe_iter_text(text_series):
        for gname, kws in groups.items():
            for kw in kws:
                if re.search(_WORD_BOUNDARY.format(kw=re.escape(kw)), txt):
                    group_counts[gname] += 1
                    break  # avoid double counting a group within the same doc
    return group_counts

def _series_bar(series: pd.Series, title: str, sort_desc: bool = True, top_n: int | None = None):
    s = series.dropna()
    if sort_desc:
        s = s.sort_values(ascending=True)  # horizontal chart sorts bottom->top
    if top_n:
        s = s.tail(top_n)
    st.bar_chart(s, height=320, use_container_width=True)
    st.caption(title)


# ----------------- Main Render Function -----------------

import streamlit as st


# Only cache on refresh_key and simple args, never on DataFrame or objects
@st.cache_data(show_spinner=False)
def cached_analysis(refresh_key: int = 0, source_filter: str = "All", model: str | None = None):
    """
    Cached business analysis. Invalidate only when refresh_key changes.
    """
    data_path = "../data/combined.json"
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            combined_data = json.load(f)
    except Exception as e:
        return None, f"❌ Failed to load combined.json: {e}"
    return combined_data, None

def render(source_filter: str = None, model: str | None = None, refresh_key: int = 0) -> None:
    """
    Business Dashboard: Reads combined.json, analyzes with Gemini,
    shows attractive insights + visualizations.
    """
    # Only pass refresh_key and simple args to cached_analysis
    combined_data, err = cached_analysis(refresh_key, source_filter or "All", model)
    if err:
        st.error(err)
        return

    df = pd.DataFrame(combined_data)

    # Optional source filter
    if source_filter and source_filter != "All" and "source_type" in df.columns:
        df = df[df["source_type"].str.lower() == source_filter.lower()]

    # --- Helper: Source label ---
    def label_source_type(stype):
        stype = str(stype or "").lower()
        if "reddit" in stype:
            return "Reddit"
        if "blog" in stype:
            return "Blog"
        if "pubmed" in stype or "journal" in stype:
            return "Journal"
        return stype.capitalize() if stype else "Other"

    # --- Visualization 1: Source distribution ---

    # --- Visualization 2: Mentions over time ---
    # --- Visualization 3: Top brands ---
    st.markdown("### 🏷️ Top Mentioned Brands")
    brand_keywords = [
        # Global leaders
        "MyProtein", "Optimum Nutrition", "Dymatize", "MuscleBlaze", "MuscleTech",
        "GNC", "Ultimate Nutrition", "Labrada", "BigMuscles", "BSN",
        "Cellucor", "Isopure", "Evlution Nutrition", "Kaged Muscle",
        "Rule One", "Ronnie Coleman Signature Series",

        # Premium / Specialty
        "Transparent Labs", "Legion Athletics", "Bulk Powders", "Naked Nutrition",
        "Orgain", "Vega", "Garden of Life", "PlantFusion", "Sunwarrior",
        "Vital Proteins", "PEScience", "Ghost Lifestyle", "Alani Nu",
        "Redcon1", "JYM Supplement Science", "Xtend", "RSP Nutrition",

        # India / Asia focused
        "HealthKart", "Ultimate Nutrition India", "Proburst", "Sinew Nutrition",
        "MuscleXP", "Fast&Up", "Wellcore", "BigFlex", "AS-IT-IS Nutrition",

        # Retail / Online exclusive
        "Bodybuilding.com", "Muscle & Strength", "BarBend", "True Nutrition",
        "Promix Nutrition", "BPI Sports", "Met-Rx", "Six Star Pro Nutrition"
    ]

    brand_counts = {b: 0 for b in brand_keywords}
    for text in df.get("combined_text", []):
        if isinstance(text, str):
            for brand in brand_keywords:
                if re.search(re.escape(brand), text, re.IGNORECASE):
                    brand_counts[brand] += 1
    brand_counts = {b: c for b, c in brand_counts.items() if c > 0}
    if brand_counts:
        st.bar_chart(pd.Series(brand_counts, name="Mentions"))
    else:
        st.info("No major brand mentions found in this dataset.")

    st.divider()

    # =========================
    # 2) CONSUMER PREFERENCE TRENDS
    # =========================
    st.markdown("## 🌱 Consumer Preference Trends")

    attr_groups = {
        "Protein Type • Whey/Isolate": ["whey", "isolate", "wpi", "concentrate", "hydrolysate"],
        "Protein Type • Casein": ["casein", "micellar"],
        "Protein Type • Plant": ["plant-based", "plant based", "vegan", "pea protein", "soy protein", "rice protein"],
        "Protein Type • Collagen": ["collagen", "collagen peptides"],
        "Format • RTD / Drinks": ["rtd", "ready to drink", "protein shake", "bottled shake"],
        "Format • Bars/Snacks": ["protein bar", "snack bar", "energy bar"],
        "Attributes • Low Sugar/Carb": ["low sugar", "no sugar", "sugar-free", "low carb", "keto"],
        "Attributes • Lactose/Gluten": ["lactose-free", "lactose free", "gluten-free", "gluten free"],
        "Attributes • Clean/Organic": ["clean label", "organic", "non-gmo", "grass-fed", "natural"],
        "Taste • Flavors": ["chocolate", "vanilla", "strawberry", "coffee", "cookie", "unflavored", "salted caramel"],
        "Convenience • Mix/Sachets": ["mixability", "scoop", "single-serve", "sachet", "sticks", "on-the-go"]
    }

    attr_counts = _count_keyword_groups(df.get("combined_text", pd.Series([], dtype=object)), attr_groups)
    attr_series = pd.Series(attr_counts, dtype="int64")

    if attr_series.sum() > 0:
        _series_bar(attr_series, "Mentions by Consumer Preference Group", sort_desc=True)
    else:
        st.info("No clear consumer preference keywords detected in this dataset.")

    st.divider()


    # --- Prepare Gemini context ---
    entries, refs = [], []
    for _, row in df.iterrows():
        stype = label_source_type(row.get("source_type", ""))
        title = row.get("title", "")
        url = row.get("url", "")
        text = row.get("combined_text", "")
        if isinstance(text, str) and len(text) > 50:
            entries.append(f"[{stype}] {title}\n{text}")
            refs.append(f"- **{stype}** | [{title}]({url})")

    max_entries = 120
    context = "\n\n".join(entries[:max_entries])
    references = "\n".join(refs[:max_entries])

    gemini_prompt = (
        "Analyze the following recent protein market news, research, and discussions. "
        "Summarize into **3 categories**: (1) 📈 Trends, (2) 💡 Opportunities, (3) ⚠ Risks. "
        "For Trends, focus on emerging patterns in consumer preferences, product innovations, and market dynamics. And don't give references inline."
        "For Opportunities, focus on practical strategies a protein-nutrition company can apply to improve product quality, customer satisfaction, and revenue. "
        "Include actionable suggestions for R&D, formulation, supply chain, marketing, and new SKUs. "
        "Use short, business-focused bullet points. "
        "Do NOT put references inline. After last of these I will give references.\n\n"
        + context
    )
    gemini_analysis = chat_completion(gemini_prompt)

    # --- Premium Matte Black UI ---
    st.markdown("""
        <div style='background:linear-gradient(90deg,#18181b 0%,#232326 100%);border-radius:22px;padding:32px 36px;margin-bottom:24px;box-shadow:0 8px 32px rgba(0,0,0,0.18);color:#fff;'>
            <h2 style='margin-top:0;margin-bottom:18px;font-size:2.1rem;font-weight:900;color:#fff;font-family:Montserrat,Arial,sans-serif;letter-spacing:0.01em;'>Business Strategies</h2>
            <h3 style='margin-top:0;margin-bottom:10px;font-size:1.35rem;font-weight:800;color:#e5e5e5;font-family:Montserrat,Arial,sans-serif;'>📈 Market Trends</h3>
            <div style='font-size:1.08rem;color:#e5e5e5;line-height:1.7;font-family:Montserrat,Arial,sans-serif;'>
                {trends}
            </div>
            <h3 style='margin-top:28px;margin-bottom:10px;font-size:1.35rem;font-weight:800;color:#e5e5e5;font-family:Montserrat,Arial,sans-serif;'>💡 Business Opportunities</h3>
            <div style='font-size:1.08rem;color:#b3ffb3;line-height:1.7;font-family:Montserrat,Arial,sans-serif;'>
                {opps}
            </div>
            <h3 style='margin-top:28px;margin-bottom:10px;font-size:1.35rem;font-weight:800;color:#e5e5e5;font-family:Montserrat,Arial,sans-serif;'>⚠ Risks & Challenges</h3>
            <div style='font-size:1.08rem;color:#ffb3b3;line-height:1.7;font-family:Montserrat,Arial,sans-serif;'>
                {risks}
            </div>
        </div>
    """.format(
        trends=gemini_analysis.split("💡 Opportunities")[0].replace("📈 Trends", "").strip(),
        opps=gemini_analysis.split("💡 Opportunities")[1].split("⚠ Risks")[0].strip() if "💡 Opportunities" in gemini_analysis else "",
        risks=gemini_analysis.split("⚠ Risks")[1].strip() if "⚠ Risks" in gemini_analysis else ""
    ), unsafe_allow_html=True)
    if references.strip():
        with st.expander("📚 Sources"):
            st.markdown(references, unsafe_allow_html=True)