# phase3/utils/charts.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---- Common Plotly defaults ---------------------------------------------------
_TEMPLATE = "plotly_white"


# ---- Basic charts -------------------------------------------------------------
def bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    orientation: str = "v",
    title: Optional[str] = None,
    height: int = 380,
):
    """Simple bar chart."""
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        template=_TEMPLATE,
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    return fig


def line(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    height: int = 380,
):
    """Simple line chart (works for time series and grouped lines)."""
    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        template=_TEMPLATE,
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None, showgrid=True)
    return fig


def area(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    stack: bool = True,
    title: Optional[str] = None,
    height: int = 380,
):
    """Area chart (stacked by default if color provided)."""
    fig = px.area(
        data,
        x=x,
        y=y,
        color=color,
        groupnorm=None,
        title=title,
        template=_TEMPLATE,
    )
    if not stack:
        # unstack by setting stackgroup None via traces
        for tr in fig.data:
            tr.update(stackgroup=None)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None, showgrid=True)
    return fig


def bubble(
    data: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: Optional[str] = None,
    hover_data: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    height: int = 420,
):
    """Bubble chart for co-occurrences / volumes."""
    fig = px.scatter(
        data,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_data=hover_data,
        size_max=40,
        template=_TEMPLATE,
        title=title,
    )
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None, showgrid=True)
    return fig


def heatmap(
    z: np.ndarray | pd.DataFrame,
    x_labels: Iterable[str],
    y_labels: Iterable[str],
    title: Optional[str] = None,
    height: int = 420,
):
    """
    Heatmap for keyword co-occurrence matrices, etc.
    z can be a 2D numpy array or DataFrame.
    """
    z_vals = z.values if isinstance(z, pd.DataFrame) else np.asarray(z)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=list(x_labels),
            y=list(y_labels),
            coloraxis="coloraxis",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        coloraxis=dict(colorscale="Blues"),
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---- Time utilities for charts ------------------------------------------------
def time_count(
    df: pd.DataFrame,
    date_col: str = "date",
    freq: str = "D",
    group_col: Optional[str] = None,
    min_count: int = 0,
) -> pd.DataFrame:
    """
    Aggregate counts over time for volume charts.
    - freq: 'D', 'W', 'M', etc.
    - group_col: optional source/category to split lines.
    """
    if df is None or not len(df) or date_col not in df.columns:
        return pd.DataFrame(columns=[date_col, "count"] + ([group_col] if group_col else []))

    t = pd.to_datetime(df[date_col], utc=True, errors="coerce").dropna()
    tmp = df.loc[t.index].copy()
    tmp[date_col] = t

    if group_col:
        g = (
            tmp.groupby([pd.Grouper(key=date_col, freq=freq), tmp[group_col]])
            .size()
            .reset_index(name="count")
        )
        if min_count > 0:
            g = g[g["count"] >= min_count]
        return g
    else:
        g = tmp.groupby(pd.Grouper(key=date_col, freq=freq)).size().reset_index(name="count")
        if min_count > 0:
            g = g[g["count"] >= min_count]
        return g
# --- compatibility alias ---
def time_series(df, x, y, title: str = "", height: int = 300):
    """Alias for line() to keep older components working."""
    return line(df, x=x, y=y, title=title, height=height)