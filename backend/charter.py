"""Turn a pandas query result (DataFrame / Series / scalar) into a themed
Plotly figure, auto-picking a chart type from the shape of the data.

Two public entry points:
- `build_chart(result, question)` -> JSON str, used by main.py's SSE stream.
- `generate_chart(df, chart_type, title, x_col, y_col)` -> dict (`fig.to_dict()`),
  for callers that want the figure as a plain object (e.g. a REST response).
`get_chart_type_label(df)` exposes the auto-selection's pick as a display string.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Theme — Bloomberg-terminal-style dark theme. Every chart goes through
# _apply_theme() before being handed back, so this is the single source of
# truth for colors/fonts across all chart types.
# ---------------------------------------------------------------------------
BG = "#0F1923"
PAPER = "#111E2A"
GRID = "rgba(31, 78, 121, 0.3)"  # #1F4E79 @ 30% opacity
PRIMARY = "#2E86AB"   # cyan
SECONDARY = "#E07B39"  # amber
TEXT = "#E0E0E0"
FONT_FAMILY = "JetBrains Mono, Consolas, monospace"

MAX_DONUT_SLICES = 8
MAX_PSEUDO_CATEGORY_VALUES = 20
PCT_KEYWORDS = ("pct", "percent", "proportion", "share", "%", "ratio")

# Explicit chart-type words/phrases in a question override the auto-picked type
# (when the data can actually support that type) — checked most-specific first.
QUESTION_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("heatmap", (r"heat ?map", r"correlation matrix")),
    ("grouped_bar", (r"grouped bar", r"clustered bar")),
    ("donut", (r"\bdonut\b", r"\bpie\b")),
    ("box", (r"box[\s-]?plot", r"box and whisker")),
    ("histogram", (r"\bhistogram\b",)),
    ("scatter", (r"\bscatter\b",)),
    ("line", (r"\bline chart\b", r"\bline graph\b", r"\btrend\b", r"\bover time\b")),
    ("bar", (r"\bbar chart\b", r"\bbar graph\b")),
]

LABELS = {
    "bar": "Bar Chart",
    "line": "Line Chart",
    "donut": "Donut Chart",
    "histogram": "Histogram",
    "scatter": "Scatter Plot",
    "box": "Box Plot",
    "heatmap": "Heatmap",
    "grouped_bar": "Grouped Bar Chart",
    "empty": "No Data",
}

TYPE_ALIASES = {
    "pie": "donut", "pie chart": "donut", "donut chart": "donut",
    "bar chart": "bar", "barchart": "bar",
    "line chart": "line", "linechart": "line", "trend": "line",
    "hist": "histogram",
    "scatter plot": "scatter", "scatterplot": "scatter",
    "box plot": "box", "boxplot": "box",
    "heat map": "heatmap", "correlation": "heatmap", "correlation matrix": "heatmap",
    "grouped bar": "grouped_bar", "grouped bar chart": "grouped_bar", "group bar": "grouped_bar",
}


# ---------------------------------------------------------------------------
# Color helpers — cyan-to-amber gradient, generated on demand for however
# many series/slices/groups a chart needs.
# ---------------------------------------------------------------------------
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def _gradient_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [PRIMARY]
    start, end = _hex_to_rgb(PRIMARY), _hex_to_rgb(SECONDARY)
    return [
        _rgb_to_hex(tuple(start[i] + (end[i] - start[i]) * (step / (n - 1)) for i in range(3)))
        for step in range(n)
    ]


# ---------------------------------------------------------------------------
# Data shape helpers
# ---------------------------------------------------------------------------
def _to_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        # A default RangeIndex (e.g. from `df['col']` or `df.assign(...)['col']`)
        # carries no information — exposing it as a column would masquerade a
        # single-column result (histogram-shaped) as two numeric columns
        # (scatter-shaped). Only promote the index when it's actually meaningful
        # (a groupby key, a date index, etc).
        if isinstance(result.index, pd.RangeIndex) and result.index.name is None:
            return pd.DataFrame({result.name or "value": result.values})
        out = result.reset_index()
        if out.shape[1] == 2:
            out.columns = [result.index.name or "index", result.name or "value"]
        return out
    return pd.DataFrame({"result": [result]})


def _usable_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Numeric / categorical / datetime column names, skipping all-null columns."""
    numeric_cols, category_cols, datetime_cols = [], [], []
    for col in df.columns:
        series = df[col]
        if series.isna().all():
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
        else:
            category_cols.append(col)
    return numeric_cols, category_cols, datetime_cols


def _looks_like_percentage(col_name: str, series: pd.Series) -> bool:
    if any(k in str(col_name).lower() for k in PCT_KEYWORDS):
        return True
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.size < 2:  # a single value trivially "sums" to itself — not a meaningful signal
        return False
    lo, hi, total = numeric.min(), numeric.max(), numeric.sum()
    if lo >= 0 and hi <= 100 and 95 <= total <= 105:
        return True
    return bool(lo >= 0 and hi <= 1 and 0.95 <= total <= 1.05)


def _is_pseudo_category(series: pd.Series, n_rows: int) -> bool:
    """A numeric column that's actually a grouping key (e.g. JobLevel 1-5, one
    row per level after a groupby) rather than a continuous measurement."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    is_whole = pd.api.types.is_integer_dtype(series) or (non_null % 1 == 0).all()
    if not is_whole:
        return False
    nunique = non_null.nunique()
    return nunique <= MAX_PSEUDO_CATEGORY_VALUES and nunique == n_rows


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------
def _auto_select_chart_type(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"

    numeric_cols, category_cols, datetime_cols = _usable_columns(df)

    if not numeric_cols and not category_cols and not datetime_cols:
        return "empty"
    if datetime_cols:
        return "line"
    if len(numeric_cols) >= 5:
        return "heatmap"
    if len(category_cols) >= 2 and numeric_cols:
        return "grouped_bar"
    if len(category_cols) == 1 and not numeric_cols:
        return "donut"  # single category column -> chart the value_counts
    if len(category_cols) == 1 and len(numeric_cols) == 1:
        cat_col, val_col = category_cols[0], numeric_cols[0]
        if df[cat_col].duplicated().any():
            return "box"  # repeated categories -> multiple observations per group
        if _looks_like_percentage(val_col, df[val_col]) and df[cat_col].nunique() <= MAX_DONUT_SLICES:
            return "donut"
        return "bar"
    if len(numeric_cols) == 1 and not category_cols:
        return "histogram"
    if len(numeric_cols) == 2 and not category_cols:
        a, b = numeric_cols
        if _is_pseudo_category(df[a], len(df)) != _is_pseudo_category(df[b], len(df)):
            return "bar"  # one column is really a grouping key, not a measurement
        return "scatter"
    if len(numeric_cols) >= 2:
        return "scatter"
    return "bar"


def get_chart_type_label(df: pd.DataFrame) -> str:
    """Auto-selected chart type as a display string, for the frontend to show."""
    chart_type = _auto_select_chart_type(_to_dataframe(df))
    return LABELS.get(chart_type, chart_type.replace("_", " ").title())


def _normalize_type(chart_type: str) -> str:
    key = " ".join(str(chart_type).strip().lower().replace("-", " ").replace("_", " ").split())
    return TYPE_ALIASES.get(key, key.replace(" ", "_"))


def _question_chart_hint(question: str) -> str | None:
    """An explicit chart-type word/phrase the user typed (e.g. "pie chart"), if any."""
    if not question:
        return None
    q = question.lower()
    for chart_type, patterns in QUESTION_TYPE_HINTS:
        if any(re.search(p, q) for p in patterns):
            return chart_type
    return None


def _can_render_type(chart_type: str, numeric_cols: list[str], category_cols: list[str]) -> bool:
    """Whether the data actually has the columns a requested chart type needs."""
    if chart_type == "donut":
        return bool(category_cols)
    if chart_type in ("box", "histogram"):
        return bool(numeric_cols)
    if chart_type == "heatmap":
        return len(numeric_cols) >= 2
    if chart_type == "scatter":
        return len(numeric_cols) >= 2 or (numeric_cols and category_cols)
    return True  # bar / line / grouped_bar all degrade gracefully


# ---------------------------------------------------------------------------
# Chart builders — every builder takes the same signature so the dispatch
# table below can call any of them uniformly, and returns an untheme'd Figure.
# ---------------------------------------------------------------------------
def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=15, color=TEXT))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _bar_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    if x_col:
        cat = x_col
    elif category_cols:
        cat = category_cols[0]
    elif datetime_cols:
        cat = datetime_cols[0]
    elif len(numeric_cols) >= 2:
        # No real category column — use whichever numeric column looks most
        # like a grouping key (fewest unique values) as the axis.
        cat = min(numeric_cols, key=lambda c: df[c].nunique())
    else:
        cat = df.columns[0]

    value_cols = [y_col] if y_col else [c for c in numeric_cols if c != cat]

    fig = go.Figure()
    if not value_cols:
        counts = df[cat].value_counts()
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=PRIMARY))
        fig.update_layout(xaxis_title=cat, yaxis_title="count")
        return fig

    colors = _gradient_colors(len(value_cols))
    for col, color in zip(value_cols, colors):
        fig.add_trace(go.Bar(x=df[cat], y=df[col], name=str(col), marker_color=color))
    fig.update_layout(barmode="group", xaxis_title=cat, yaxis_title=value_cols[0] if len(value_cols) == 1 else None)
    return fig


def _grouped_bar_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    cat = x_col or category_cols[0]
    remaining = [c for c in category_cols if c != cat]
    sub = remaining[0] if remaining else None
    val = y_col or (numeric_cols[0] if numeric_cols else None)

    if sub is None or val is None:
        return _bar_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col)

    pivot = df.pivot_table(index=cat, columns=sub, values=val, aggfunc="sum")
    colors = _gradient_colors(len(pivot.columns))
    fig = go.Figure()
    for color, col in zip(colors, pivot.columns):
        fig.add_trace(go.Bar(x=pivot.index, y=pivot[col], name=str(col), marker_color=color))
    fig.update_layout(barmode="group", xaxis_title=cat, yaxis_title=val)
    return fig


def _line_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    x = x_col or (datetime_cols[0] if datetime_cols else df.columns[0])
    ordered = df.sort_values(x) if pd.api.types.is_datetime64_any_dtype(df[x]) else df
    value_cols = [y_col] if y_col else numeric_cols

    fig = go.Figure()
    if not value_cols:
        counts = ordered[x].value_counts().sort_index()
        fig.add_trace(go.Scatter(x=counts.index, y=counts.values, mode="lines+markers", line=dict(color=PRIMARY)))
        fig.update_layout(xaxis_title=x, yaxis_title="count")
        return fig

    colors = _gradient_colors(len(value_cols))
    for col, color in zip(value_cols, colors):
        fig.add_trace(go.Scatter(x=ordered[x], y=ordered[col], mode="lines+markers", name=str(col), line=dict(color=color)))
    fig.update_layout(xaxis_title=x, yaxis_title=value_cols[0] if len(value_cols) == 1 else None)
    return fig


def _donut_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    cat = x_col or (category_cols[0] if category_cols else df.columns[0])
    val_col = y_col or (numeric_cols[0] if numeric_cols else None)
    if val_col:
        labels, values = df[cat], df[val_col]
    else:
        counts = df[cat].value_counts()
        labels, values = counts.index, counts.values

    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values), hole=0.55,
        marker=dict(colors=_gradient_colors(len(labels)), line=dict(color=BG, width=1)),
        textfont=dict(color=TEXT),
    ))
    return fig


def _histogram_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    col = x_col or y_col or (numeric_cols[0] if numeric_cols else df.columns[0])
    fig = go.Figure(go.Histogram(x=df[col], marker_color=PRIMARY))
    fig.update_layout(xaxis_title=col, yaxis_title="count")
    return fig


def _scatter_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    x = x_col or (numeric_cols[0] if numeric_cols else df.columns[0])
    y = y_col or (numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else df.columns[-1]))

    fig = go.Figure()
    if category_cols and not x_col and not y_col:
        groups = df[category_cols[0]].dropna().unique()
        colors = _gradient_colors(len(groups))
        for color, group in zip(colors, groups):
            subset = df[df[category_cols[0]] == group]
            fig.add_trace(go.Scatter(x=subset[x], y=subset[y], mode="markers", name=str(group), marker=dict(color=color)))
    else:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers", marker=dict(color=PRIMARY)))
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


def _box_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    cat = x_col or (category_cols[0] if category_cols else None)
    val = y_col or (numeric_cols[0] if numeric_cols else None)

    fig = go.Figure()
    if not val:
        return fig
    if cat:
        groups = df[cat].dropna().unique()
        colors = _gradient_colors(len(groups))
        for color, group in zip(colors, groups):
            fig.add_trace(go.Box(y=df.loc[df[cat] == group, val], name=str(group), marker_color=color, line=dict(color=color)))
        fig.update_layout(xaxis_title=cat, yaxis_title=val)
    else:
        fig.add_trace(go.Box(y=df[val], name=str(val), marker_color=PRIMARY, line=dict(color=PRIMARY)))
        fig.update_layout(yaxis_title=val)
    return fig


def _heatmap_chart(df, numeric_cols, category_cols, datetime_cols, x_col, y_col) -> go.Figure:
    cols = numeric_cols or df.select_dtypes("number").columns.tolist()
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        colorscale=[[0, SECONDARY], [0.5, PAPER], [1, PRIMARY]],
        zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(color=TEXT),
        colorbar=dict(title="corr", tickfont=dict(color=TEXT), outlinewidth=0),
    ))
    return fig


_BUILDERS = {
    "bar": _bar_chart,
    "grouped_bar": _grouped_bar_chart,
    "line": _line_chart,
    "donut": _donut_chart,
    "histogram": _histogram_chart,
    "scatter": _scatter_chart,
    "box": _box_chart,
    "heatmap": _heatmap_chart,
}


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------
def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    layout_kwargs = dict(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family=FONT_FAMILY, color=TEXT, size=12),
        colorway=_gradient_colors(8),
        margin=dict(l=48, r=24, t=48 if title else 24, b=44),
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top", bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT)),
        transition=dict(duration=0),
        hovermode="closest",
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=TEXT, size=15), x=0.02, xanchor="left")
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showline=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showline=False, linecolor=GRID, ticks="")
    return fig


def _build_figure(df: pd.DataFrame, chart_type: str, x_col: str | None, y_col: str | None) -> go.Figure:
    if df is None or df.empty:
        return _empty_chart("No data available")

    numeric_cols, category_cols, datetime_cols = _usable_columns(df)
    if not numeric_cols and not category_cols and not datetime_cols:
        return _empty_chart("All columns are empty")

    resolved = _normalize_type(chart_type) if chart_type and chart_type != "auto" else _auto_select_chart_type(df)
    builder = _BUILDERS.get(resolved, _bar_chart)
    return builder(df, numeric_cols, category_cols, datetime_cols, x_col, y_col)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_chart(
    df: pd.DataFrame, chart_type: str = "auto", title: str = "",
    x_col: str = None, y_col: str = None, question: str = "",
) -> dict:
    """Build a themed Plotly figure and return it as a dict (`fig.to_dict()`),
    ready to send to the React frontend for rendering with Plotly.js.

    `question` is optional: when `chart_type` is left as "auto", an explicit
    chart-type word/phrase in the question (e.g. "pie chart") overrides the
    auto-picked type, provided the data can actually support it.
    """
    df = _to_dataframe(df)
    resolved_type = chart_type

    if (not chart_type or chart_type == "auto") and question and not df.empty:
        hint = _question_chart_hint(question)
        if hint:
            numeric_cols, category_cols, _ = _usable_columns(df)
            if _can_render_type(hint, numeric_cols, category_cols):
                resolved_type = hint

    fig = _build_figure(df, resolved_type, x_col, y_col)
    fig = _apply_theme(fig, title)
    # `fig.to_dict()` leaves non-numeric arrays (e.g. string category labels)
    # as raw numpy ndarrays, which stdlib `json.dumps` can't serialize — only
    # numeric traces get Plotly's safe binary-array treatment. Round-tripping
    # through Plotly's own encoder guarantees a fully plain-JSON-safe dict.
    return json.loads(pio.to_json(fig))


def build_chart(result: Any, question: str = "") -> str:
    """Returns a Plotly figure serialized as JSON (ready to embed in an SSE payload)."""
    return json.dumps(generate_chart(result, question=question))
