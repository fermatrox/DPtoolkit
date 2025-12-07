"""Visualization generator for comparing original vs protected datasets.

This module provides interactive visualizations using Plotly for comparing
original and protected datasets:

- Overlay histograms for numeric columns
- Correlation heatmaps with difference highlighting
- Distribution comparison charts
- Category frequency bar charts
- Summary dashboard generation

All visualizations are designed for use in Streamlit applications.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Constants
# =============================================================================

# Default color scheme
COLORS = {
    "original": "#1f77b4",  # Blue
    "protected": "#ff7f0e",  # Orange
    "positive_diff": "#2ca02c",  # Green
    "negative_diff": "#d62728",  # Red
    "neutral": "#7f7f7f",  # Gray
}

# Default chart dimensions
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 500

# Default number of histogram bins
DEFAULT_BINS = 50


# =============================================================================
# Enums
# =============================================================================


class ChartType(Enum):
    """Types of charts available."""

    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BAR = "bar"
    BOX = "box"
    SCATTER = "scatter"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChartConfig:
    """Configuration for chart generation.

    Attributes:
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Chart title.
        show_legend: Whether to show legend.
        template: Plotly template name.
        colors: Custom color dictionary.
    """

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    title: Optional[str] = None
    show_legend: bool = True
    template: str = "plotly_white"
    colors: Optional[Dict[str, str]] = None

    def get_colors(self) -> Dict[str, str]:
        """Get color dictionary with defaults."""
        if self.colors:
            return {**COLORS, **self.colors}
        return COLORS


# =============================================================================
# Histogram Visualizations
# =============================================================================


def create_histogram_overlay(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    bins: int = DEFAULT_BINS,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create overlay histogram comparing original and protected distributions.

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name for the column (uses series name if None).
        bins: Number of histogram bins.
        config: Chart configuration.

    Returns:
        Plotly Figure with overlay histogram.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Value"

    # Remove nulls
    orig_valid = original.dropna().astype(float)
    prot_valid = protected.dropna().astype(float)

    # Calculate common bin edges
    all_values = np.concatenate([orig_valid.values, prot_valid.values])
    if len(all_values) == 0:
        # Empty data - return empty figure
        fig = go.Figure()
        fig.update_layout(
            title=config.title or f"Distribution: {name}",
            template=config.template,
        )
        return fig

    bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    fig = go.Figure()

    # Original histogram
    fig.add_trace(
        go.Histogram(
            x=orig_valid,
            xbins=dict(
                start=bin_edges[0],
                end=bin_edges[-1],
                size=(bin_edges[-1] - bin_edges[0]) / bins,
            ),
            name="Original",
            marker_color=colors["original"],
            opacity=0.6,
            histnorm="probability density",
        )
    )

    # Protected histogram
    fig.add_trace(
        go.Histogram(
            x=prot_valid,
            xbins=dict(
                start=bin_edges[0],
                end=bin_edges[-1],
                size=(bin_edges[-1] - bin_edges[0]) / bins,
            ),
            name="Protected",
            marker_color=colors["protected"],
            opacity=0.6,
            histnorm="probability density",
        )
    )

    fig.update_layout(
        title=config.title or f"Distribution Comparison: {name}",
        xaxis_title=name,
        yaxis_title="Density",
        barmode="overlay",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_histogram_sidebyside(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    bins: int = DEFAULT_BINS,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create side-by-side histogram comparison.

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name for the column.
        bins: Number of histogram bins.
        config: Chart configuration.

    Returns:
        Plotly Figure with side-by-side histograms.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Value"

    orig_valid = original.dropna().astype(float)
    prot_valid = protected.dropna().astype(float)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Original", "Protected"),
        shared_yaxes=True,
    )

    # Common bin range
    all_values = np.concatenate([orig_valid.values, prot_valid.values])
    if len(all_values) == 0:
        return fig

    bin_range: Tuple[float, float] = (float(np.min(all_values)), float(np.max(all_values)))

    fig.add_trace(
        go.Histogram(
            x=orig_valid,
            nbinsx=bins,
            marker_color=colors["original"],
            name="Original",
            xbins=dict(start=bin_range[0], end=bin_range[1]),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=prot_valid,
            nbinsx=bins,
            marker_color=colors["protected"],
            name="Protected",
            xbins=dict(start=bin_range[0], end=bin_range[1]),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=config.title or f"Distribution Comparison: {name}",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
    )

    return fig


# =============================================================================
# Correlation Heatmaps
# =============================================================================


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create correlation heatmap for a dataset.

    Args:
        df: DataFrame to analyze.
        columns: Columns to include (uses all numeric if None).
        method: Correlation method (pearson, spearman, kendall).
        config: Chart configuration.

    Returns:
        Plotly Figure with correlation heatmap.
    """
    config = config or ChartConfig()

    if columns is None:
        columns = list(df.select_dtypes(include=["number"]).columns)

    if len(columns) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    corr = df[columns].corr(method=method)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=config.title or f"Correlation Matrix ({method.capitalize()})",
        width=config.width,
        height=config.height,
        template=config.template,
        xaxis=dict(tickangle=45),
    )

    return fig


def create_correlation_diff_heatmap(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create heatmap showing correlation differences.

    Args:
        original: Original DataFrame.
        protected: Protected DataFrame.
        columns: Columns to include.
        method: Correlation method.
        config: Chart configuration.

    Returns:
        Plotly Figure with correlation difference heatmap.
    """
    config = config or ChartConfig()

    if columns is None:
        columns = list(original.select_dtypes(include=["number"]).columns)

    if len(columns) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    orig_corr = original[columns].corr(method=method)
    prot_corr = protected[columns].corr(method=method)
    diff = prot_corr - orig_corr

    fig = go.Figure(
        data=go.Heatmap(
            z=diff.values,
            x=diff.columns.tolist(),
            y=diff.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(diff.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                "<b>%{x}</b> vs <b>%{y}</b><br>"
                "Difference: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=config.title or "Correlation Change (Protected - Original)",
        width=config.width,
        height=config.height,
        template=config.template,
        xaxis=dict(tickangle=45),
    )

    return fig


def create_correlation_comparison(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create side-by-side correlation heatmaps with difference.

    Args:
        original: Original DataFrame.
        protected: Protected DataFrame.
        columns: Columns to include.
        method: Correlation method.
        config: Chart configuration.

    Returns:
        Plotly Figure with three heatmaps.
    """
    config = config or ChartConfig()

    if columns is None:
        columns = list(original.select_dtypes(include=["number"]).columns)

    if len(columns) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    orig_corr = original[columns].corr(method=method)
    prot_corr = protected[columns].corr(method=method)
    diff = prot_corr - orig_corr

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Original", "Protected", "Difference"),
        horizontal_spacing=0.1,
    )

    # Original correlation
    fig.add_trace(
        go.Heatmap(
            z=orig_corr.values,
            x=orig_corr.columns.tolist(),
            y=orig_corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.round(orig_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
        ),
        row=1,
        col=1,
    )

    # Protected correlation
    fig.add_trace(
        go.Heatmap(
            z=prot_corr.values,
            x=prot_corr.columns.tolist(),
            y=prot_corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.round(prot_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
        ),
        row=1,
        col=2,
    )

    # Difference
    fig.add_trace(
        go.Heatmap(
            z=diff.values,
            x=diff.columns.tolist(),
            y=diff.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            text=np.round(diff.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(title="Corr", x=1.02),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=config.title or "Correlation Comparison",
        width=config.width + 200,
        height=config.height,
        template=config.template,
    )

    return fig


# =============================================================================
# Category Visualizations
# =============================================================================


def create_category_bar_chart(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    top_n: int = 15,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create grouped bar chart comparing category frequencies.

    Args:
        original: Original categorical column.
        protected: Protected categorical column.
        column_name: Name for the column.
        top_n: Number of top categories to show.
        config: Chart configuration.

    Returns:
        Plotly Figure with grouped bar chart.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Category"

    # Get frequency counts
    orig_counts = original.value_counts(normalize=True).head(top_n)
    prot_counts = protected.value_counts(normalize=True)

    # Get union of categories
    categories = list(orig_counts.index)
    for cat in prot_counts.head(top_n).index:
        if cat not in categories:
            categories.append(cat)
    categories = categories[:top_n]

    orig_values = [float(orig_counts.get(cat, 0)) for cat in categories]
    prot_values = [float(prot_counts.get(cat, 0)) for cat in categories]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Original",
            x=categories,
            y=orig_values,
            marker_color=colors["original"],
        )
    )

    fig.add_trace(
        go.Bar(
            name="Protected",
            x=categories,
            y=prot_values,
            marker_color=colors["protected"],
        )
    )

    fig.update_layout(
        title=config.title or f"Category Distribution: {name}",
        xaxis_title=name,
        yaxis_title="Frequency",
        barmode="group",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
        xaxis=dict(tickangle=45),
    )

    return fig


def create_category_drift_chart(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    top_n: int = 15,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create bar chart showing category frequency changes.

    Args:
        original: Original categorical column.
        protected: Protected categorical column.
        column_name: Name for the column.
        top_n: Number of categories to show.
        config: Chart configuration.

    Returns:
        Plotly Figure showing frequency changes.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Category"

    orig_counts = original.value_counts(normalize=True)
    prot_counts = protected.value_counts(normalize=True)

    # Calculate differences
    all_cats = set(orig_counts.index) | set(prot_counts.index)
    changes = []
    for cat in all_cats:
        orig_freq = orig_counts.get(cat, 0)
        prot_freq = prot_counts.get(cat, 0)
        diff = prot_freq - orig_freq
        changes.append((cat, diff))

    # Sort by absolute change
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    changes = changes[:top_n]

    categories = [c[0] for c in changes]
    diffs = [c[1] for c in changes]

    # Color by direction of change
    bar_colors = [
        colors["positive_diff"] if d > 0 else colors["negative_diff"] for d in diffs
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=diffs,
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Change: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color=colors["neutral"])

    fig.update_layout(
        title=config.title or f"Category Frequency Changes: {name}",
        xaxis_title=name,
        yaxis_title="Frequency Change",
        width=config.width,
        height=config.height,
        showlegend=False,
        template=config.template,
        xaxis=dict(tickangle=45),
    )

    return fig


# =============================================================================
# Box Plot Visualizations
# =============================================================================


def create_box_comparison(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create side-by-side box plots.

    Args:
        original: Original numeric column.
        protected: Protected numeric column.
        column_name: Name for the column.
        config: Chart configuration.

    Returns:
        Plotly Figure with box plots.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Value"

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=original.dropna(),
            name="Original",
            marker_color=colors["original"],
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Box(
            y=protected.dropna(),
            name="Protected",
            marker_color=colors["protected"],
            boxmean=True,
        )
    )

    fig.update_layout(
        title=config.title or f"Distribution Comparison: {name}",
        yaxis_title=name,
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
    )

    return fig


def create_multi_box_comparison(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create box plots for multiple columns.

    Args:
        original: Original DataFrame.
        protected: Protected DataFrame.
        columns: Columns to compare.
        config: Chart configuration.

    Returns:
        Plotly Figure with multiple box plots.
    """
    config = config or ChartConfig()
    colors = config.get_colors()

    if columns is None:
        columns = list(original.select_dtypes(include=["number"]).columns)

    fig = go.Figure()

    for col in columns:
        fig.add_trace(
            go.Box(
                y=original[col].dropna(),
                name=f"{col} (Orig)",
                marker_color=colors["original"],
                boxmean=True,
            )
        )
        fig.add_trace(
            go.Box(
                y=protected[col].dropna(),
                name=f"{col} (Prot)",
                marker_color=colors["protected"],
                boxmean=True,
            )
        )

    fig.update_layout(
        title=config.title or "Multi-Column Distribution Comparison",
        yaxis_title="Value",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
    )

    return fig


# =============================================================================
# Summary Visualizations
# =============================================================================


def create_divergence_summary_chart(
    divergences: Dict[str, float],
    column_name: Optional[str] = None,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create bar chart of divergence metrics.

    Args:
        divergences: Dictionary of metric name to value.
        column_name: Column name for title.
        config: Chart configuration.

    Returns:
        Plotly Figure with divergence summary.
    """
    config = config or ChartConfig()
    colors = config.get_colors()

    metrics = list(divergences.keys())
    values = list(divergences.values())

    # Filter out None values
    valid = [(m, v) for m, v in zip(metrics, values) if v is not None]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(
            text="No divergence metrics available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    unzipped = list(zip(*valid))
    filtered_metrics: List[str] = list(unzipped[0])
    filtered_values: List[float] = list(unzipped[1])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=filtered_metrics,
            y=filtered_values,
            marker_color=colors["original"],
            hovertemplate="<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>",
        )
    )

    title = config.title or "Divergence Metrics"
    if column_name:
        title = f"{title}: {column_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Value",
        width=config.width,
        height=config.height,
        showlegend=False,
        template=config.template,
    )

    return fig


def create_column_comparison_summary(
    mae_values: Dict[str, float],
    rmse_values: Optional[Dict[str, float]] = None,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create grouped bar chart summarizing column-level errors.

    Args:
        mae_values: Dictionary of column name to MAE.
        rmse_values: Optional dictionary of column name to RMSE.
        config: Chart configuration.

    Returns:
        Plotly Figure with error summary.
    """
    config = config or ChartConfig()
    colors = config.get_colors()

    columns = list(mae_values.keys())
    mae_list = [mae_values[c] for c in columns]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="MAE",
            x=columns,
            y=mae_list,
            marker_color=colors["original"],
        )
    )

    if rmse_values:
        rmse_list = [rmse_values.get(c, 0) for c in columns]
        fig.add_trace(
            go.Bar(
                name="RMSE",
                x=columns,
                y=rmse_list,
                marker_color=colors["protected"],
            )
        )

    fig.update_layout(
        title=config.title or "Column-Level Error Summary",
        xaxis_title="Column",
        yaxis_title="Error",
        barmode="group",
        width=config.width,
        height=config.height,
        showlegend=True,
        template=config.template,
        xaxis=dict(tickangle=45),
    )

    return fig


# =============================================================================
# Scatter Plots
# =============================================================================


def create_scatter_comparison(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    x_col: str,
    y_col: str,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create scatter plots comparing original and protected.

    Args:
        original: Original DataFrame.
        protected: Protected DataFrame.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        config: Chart configuration.

    Returns:
        Plotly Figure with scatter plots.
    """
    config = config or ChartConfig()
    colors = config.get_colors()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Original", "Protected"),
    )

    fig.add_trace(
        go.Scatter(
            x=original[x_col],
            y=original[y_col],
            mode="markers",
            marker=dict(color=colors["original"], opacity=0.5),
            name="Original",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=protected[x_col],
            y=protected[y_col],
            mode="markers",
            marker=dict(color=colors["protected"], opacity=0.5),
            name="Protected",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=config.title or f"Scatter: {x_col} vs {y_col}",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
    )

    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)

    return fig


def create_qq_plot(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """Create Q-Q plot comparing distributions.

    Args:
        original: Original numeric column.
        protected: Protected numeric column.
        column_name: Name for the column.
        config: Chart configuration.

    Returns:
        Plotly Figure with Q-Q plot.
    """
    config = config or ChartConfig()
    colors = config.get_colors()
    name = column_name or original.name or "Value"

    orig_sorted = np.sort(original.dropna().values)
    prot_sorted = np.sort(protected.dropna().values)

    # Match lengths using quantiles
    n = min(len(orig_sorted), len(prot_sorted), 1000)
    quantiles = np.linspace(0, 1, n)

    orig_quantiles = np.quantile(orig_sorted, quantiles)
    prot_quantiles = np.quantile(prot_sorted, quantiles)

    fig = go.Figure()

    # Q-Q points
    fig.add_trace(
        go.Scatter(
            x=orig_quantiles,
            y=prot_quantiles,
            mode="markers",
            marker=dict(color=colors["original"], opacity=0.5),
            name="Q-Q Points",
        )
    )

    # Reference line (y = x)
    min_val = min(orig_quantiles.min(), prot_quantiles.min())
    max_val = max(orig_quantiles.max(), prot_quantiles.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=colors["neutral"], dash="dash"),
            name="y = x",
        )
    )

    fig.update_layout(
        title=config.title or f"Q-Q Plot: {name}",
        xaxis_title="Original Quantiles",
        yaxis_title="Protected Quantiles",
        width=config.width,
        height=config.height,
        showlegend=config.show_legend,
        template=config.template,
    )

    return fig


# =============================================================================
# Dashboard Generator
# =============================================================================


class ComparisonDashboard:
    """Generator for comparison dashboards.

    Creates a collection of related visualizations for comparing
    original and protected datasets.

    Example:
        >>> dashboard = ComparisonDashboard(original_df, protected_df)
        >>> fig = dashboard.create_numeric_column_dashboard("age")
    """

    def __init__(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        config: Optional[ChartConfig] = None,
    ) -> None:
        """Initialize dashboard generator.

        Args:
            original: Original DataFrame.
            protected: Protected DataFrame.
            config: Default chart configuration.
        """
        self._original = original
        self._protected = protected
        self._config = config or ChartConfig()

    def create_numeric_column_dashboard(
        self,
        column: str,
    ) -> go.Figure:
        """Create dashboard for a numeric column.

        Includes histogram overlay, box plots, and Q-Q plot.

        Args:
            column: Column name.

        Returns:
            Plotly Figure with dashboard layout.
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution Overlay",
                "Box Plot Comparison",
                "Q-Q Plot",
                "Statistics",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        orig = self._original[column].dropna()
        prot = self._protected[column].dropna()
        colors = self._config.get_colors()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=orig,
                name="Original",
                marker_color=colors["original"],
                opacity=0.6,
                histnorm="probability density",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=prot,
                name="Protected",
                marker_color=colors["protected"],
                opacity=0.6,
                histnorm="probability density",
            ),
            row=1,
            col=1,
        )

        # Box plots
        fig.add_trace(
            go.Box(y=orig, name="Original", marker_color=colors["original"]),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Box(y=prot, name="Protected", marker_color=colors["protected"]),
            row=1,
            col=2,
        )

        # Q-Q plot
        n = min(len(orig), len(prot), 500)
        quantiles = np.linspace(0, 1, n)
        orig_q = np.quantile(orig, quantiles)
        prot_q = np.quantile(prot, quantiles)

        fig.add_trace(
            go.Scatter(
                x=orig_q,
                y=prot_q,
                mode="markers",
                marker=dict(color=colors["original"], opacity=0.5),
                name="Q-Q",
            ),
            row=2,
            col=1,
        )

        # Reference line
        min_v, max_v = min(orig_q.min(), prot_q.min()), max(orig_q.max(), prot_q.max())
        fig.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                line=dict(color=colors["neutral"], dash="dash"),
                name="y=x",
            ),
            row=2,
            col=1,
        )

        # Statistics table
        stats = [
            ["Metric", "Original", "Protected", "Diff"],
            ["Mean", f"{orig.mean():.2f}", f"{prot.mean():.2f}",
             f"{prot.mean() - orig.mean():.2f}"],
            ["Std", f"{orig.std():.2f}", f"{prot.std():.2f}",
             f"{prot.std() - orig.std():.2f}"],
            ["Median", f"{orig.median():.2f}", f"{prot.median():.2f}",
             f"{prot.median() - orig.median():.2f}"],
            ["Min", f"{orig.min():.2f}", f"{prot.min():.2f}",
             f"{prot.min() - orig.min():.2f}"],
            ["Max", f"{orig.max():.2f}", f"{prot.max():.2f}",
             f"{prot.max() - orig.max():.2f}"],
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=stats[0], fill_color="paleturquoise"),
                cells=dict(
                    values=list(zip(*stats[1:])),
                    fill_color="lavender",
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Column Dashboard: {column}",
            width=self._config.width + 200,
            height=self._config.height + 200,
            showlegend=True,
            template=self._config.template,
            barmode="overlay",
        )

        return fig

    def create_categorical_column_dashboard(
        self,
        column: str,
        top_n: int = 10,
    ) -> go.Figure:
        """Create dashboard for a categorical column.

        Args:
            column: Column name.
            top_n: Number of top categories.

        Returns:
            Plotly Figure with dashboard layout.
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Category Frequencies", "Frequency Changes"),
        )

        orig = self._original[column].dropna().astype(str)
        prot = self._protected[column].dropna().astype(str)
        colors = self._config.get_colors()

        orig_counts = orig.value_counts(normalize=True).head(top_n)
        prot_counts = prot.value_counts(normalize=True)

        categories = list(orig_counts.index)

        # Grouped bar chart
        fig.add_trace(
            go.Bar(
                name="Original",
                x=categories,
                y=[float(orig_counts.get(c, 0)) for c in categories],
                marker_color=colors["original"],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Protected",
                x=categories,
                y=[float(prot_counts.get(c, 0)) for c in categories],
                marker_color=colors["protected"],
            ),
            row=1,
            col=1,
        )

        # Difference chart
        diffs = [float(prot_counts.get(c, 0) - orig_counts.get(c, 0)) for c in categories]
        bar_colors = [
            colors["positive_diff"] if d > 0 else colors["negative_diff"]
            for d in diffs
        ]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=diffs,
                marker_color=bar_colors,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title=f"Category Dashboard: {column}",
            width=self._config.width + 200,
            height=self._config.height,
            showlegend=True,
            template=self._config.template,
            barmode="group",
        )

        return fig

    def create_overview_dashboard(
        self,
        numeric_columns: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create overview dashboard with correlation comparison.

        Args:
            numeric_columns: Numeric columns to include.

        Returns:
            Plotly Figure with overview.
        """
        if numeric_columns is None:
            numeric_columns = list(
                self._original.select_dtypes(include=["number"]).columns
            )

        return create_correlation_comparison(
            self._original,
            self._protected,
            columns=numeric_columns,
            config=ChartConfig(
                width=self._config.width + 400,
                height=self._config.height,
                title="Correlation Overview",
            ),
        )
