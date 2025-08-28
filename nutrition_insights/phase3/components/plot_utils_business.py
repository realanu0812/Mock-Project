

# Ensure all imports are at the top and not shadowed in any function
import streamlit as st
import plotly.express as px
import pandas as pd

def horizontal_bar_chart(data: dict, title: str = "", x_label: str = "Mentions", y_label: str = "", height: int = 350, color: str = "#6a38b6"):
    if not data:
        st.info("No data to display.")
        return
    ser = pd.Series(data)
    ser = ser.sort_values(ascending=True)
    fig = px.bar(
        ser,
        orientation="h",
        labels={"value": x_label, "index": y_label},
        height=height,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        plot_bgcolor="#18181b",
        paper_bgcolor="#18181b",
        font_color="#fff",
        margin=dict(l=60, r=30, t=40, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title=dict(text=title, font=dict(size=18, color="#fff")),
    )
    st.plotly_chart(fig, use_container_width=True)

def horizontal_bar_chart_series(series: pd.Series, title: str = "", x_label: str = "Mentions", y_label: str = "", height: int = 350, color: str = "#6a38b6"):
    if series.empty:
        st.info("No data to display.")
        return
    ser = series.sort_values(ascending=True)
    fig = px.bar(
        ser,
        orientation="h",
        labels={"value": x_label, "index": y_label},
        height=height,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        plot_bgcolor="#18181b",
        paper_bgcolor="#18181b",
        font_color="#fff",
        margin=dict(l=60, r=30, t=40, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title=dict(text=title, font=dict(size=18, color="#fff")),
    )
    st.plotly_chart(fig, use_container_width=True)
