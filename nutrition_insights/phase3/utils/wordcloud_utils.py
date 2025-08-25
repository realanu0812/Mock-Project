import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

def render_wordcloud(df: pd.DataFrame, text_col: str = "text", title: str = "Word Cloud"):
    if df is None or df.empty or text_col not in df.columns:
        st.info("No data available for word cloud.")
        return
    # Get protein keywords from config
    try:
        from utils.config_loader import protein_keywords
        pkws = set([k.lower() for k in protein_keywords()])
    except Exception:
        pkws = set()
    # Collect all words, but only keep those matching protein keywords
    import re
    all_words = []
    for text in df[text_col].dropna().astype(str):
        words = re.findall(r"\w+", text.lower())
        all_words.extend([w for w in words if w in pkws])
    # If too few, fallback to top 50 frequent words
    if len(all_words) < 10:
        from collections import Counter
        word_counts = Counter()
        for text in df[text_col].dropna().astype(str):
            words = re.findall(r"\w+", text.lower())
            word_counts.update(words)
        top_words = [w for w, _ in word_counts.most_common(50)]
        all_words.extend([w for w in top_words if w not in all_words])
    text = " ".join(all_words)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    st.caption(title)
