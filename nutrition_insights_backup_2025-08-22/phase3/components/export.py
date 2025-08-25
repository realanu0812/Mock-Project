# phase3/components/export.py
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from utils.common import pretty_int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


def _filter_by_source(df: pd.DataFrame, source_filter: str) -> pd.DataFrame:
    if source_filter == "All":
        return df
    s = str(source_filter).lower()
    return df[df["source"].astype(str).str.lower() == s].copy()


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the essentials if present
    cols = [c for c in ["date", "source", "title", "url", "summary", "text"] if c in df.columns]
    return df[cols].copy() if cols else df.copy()


def _make_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _make_pdf(df: pd.DataFrame, note: str) -> bytes | None:
    """
    Create a very small PDF with a header and top N rows (title + source + date).
    Returns None if reportlab is not available.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except Exception:
        return None

    top = df.copy()
    if "date" in top.columns:
        top = top.sort_values("date", ascending=False)
    show = top.head(25)  # concise

    def safe(v): return "" if pd.isna(v) else str(v)
    data = [["Title", "Source", "Date"]]
    for _, r in show.iterrows():
        data.append([safe(r.get("title")), safe(r.get("source")), safe(r.get("date"))])

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="ProteinScope Export")

    styles = getSampleStyleSheet()
    elems = [
        Paragraph("<b>ProteinScope — Export</b>", styles["Title"]),
        Paragraph(note, styles["Normal"]),
        Spacer(1, 10),
    ]
    tbl = Table(data, colWidths=[280, 90, 120])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elems.append(tbl)
    doc.build(elems)
    return buf.getvalue()


def render(df: pd.DataFrame, source_filter: str, window_days: int) -> None:
    st.subheader("Download / Export")

    fdf = _filter_by_source(df, source_filter)
    fdf = _select_columns(fdf)
    n = len(fdf)

    st.caption(
        f"Preparing export for **{source_filter}** over last **{window_days}** day(s). "
        f"Rows: **{pretty_int(n)}**"
    )

    # CSV
    csv_bytes = _make_csv(fdf)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"proteinscope_{source_filter.lower()}_{_utc_stamp()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # PDF (optional)
    with st.expander("Optional: Export PDF summary"):
        st.write(
            "Creates a concise PDF (top 25 by recency: title, source, date). "
            "If it fails, install `reportlab`."
        )
        if st.button("Generate PDF"):
            note = (
                f"Scope: {source_filter} | Window: {window_days} day(s) | "
                f"Rows available: {n} | Generated: {_utc_stamp()}"
            )
            pdf_bytes = _make_pdf(fdf, note)
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"proteinscope_{source_filter.lower()}_{_utc_stamp()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning("PDF disabled — install `reportlab` to enable (e.g., `pip install reportlab`).")