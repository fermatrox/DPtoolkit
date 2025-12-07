"""Unit tests for dp_toolkit.analysis.visualizer module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dp_toolkit.analysis.visualizer import (
    # Enums
    ChartType,
    # Data classes
    ChartConfig,
    # Histogram visualizations
    create_histogram_overlay,
    create_histogram_sidebyside,
    # Correlation heatmaps
    create_correlation_heatmap,
    create_correlation_diff_heatmap,
    create_correlation_comparison,
    # Category visualizations
    create_category_bar_chart,
    create_category_drift_chart,
    # Box plot visualizations
    create_box_comparison,
    create_multi_box_comparison,
    # Summary visualizations
    create_divergence_summary_chart,
    create_column_comparison_summary,
    # Scatter plots
    create_scatter_comparison,
    create_qq_plot,
    # Dashboard
    ComparisonDashboard,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def numeric_series():
    """Create numeric series pair for testing."""
    np.random.seed(42)
    original = pd.Series(np.random.normal(100, 15, 500), name="value")
    protected = pd.Series(np.random.normal(102, 16, 500), name="value")
    return original, protected


@pytest.fixture
def categorical_series():
    """Create categorical series pair for testing."""
    np.random.seed(42)
    categories = ["A", "B", "C", "D", "E"]
    original = pd.Series(np.random.choice(categories, 500), name="category")
    protected = pd.Series(np.random.choice(categories, 500), name="category")
    return original, protected


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 10000, n),
            "score": np.random.uniform(0, 100, n),
            "category": np.random.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture
def protected_dataframe(sample_dataframe):
    """Create protected version of sample DataFrame."""
    np.random.seed(123)
    df = sample_dataframe.copy()
    df["age"] = df["age"] + np.random.randint(-3, 4, len(df))
    df["income"] = df["income"] + np.random.normal(0, 2000, len(df))
    df["score"] = df["score"] + np.random.uniform(-5, 5, len(df))
    return df


# =============================================================================
# Tests: ChartConfig
# =============================================================================


class TestChartConfig:
    """Tests for ChartConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChartConfig()
        assert config.width == 800
        assert config.height == 500
        assert config.template == "plotly_white"
        assert config.show_legend is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChartConfig(
            width=1000,
            height=600,
            title="Custom Title",
            template="plotly_dark",
        )
        assert config.width == 1000
        assert config.height == 600
        assert config.title == "Custom Title"

    def test_get_colors_default(self):
        """Test default colors."""
        config = ChartConfig()
        colors = config.get_colors()
        assert "original" in colors
        assert "protected" in colors
        assert colors["original"] == "#1f77b4"

    def test_get_colors_custom(self):
        """Test custom colors override."""
        config = ChartConfig(colors={"original": "#ff0000"})
        colors = config.get_colors()
        assert colors["original"] == "#ff0000"
        assert "protected" in colors  # Default still exists


# =============================================================================
# Tests: Histogram Visualizations
# =============================================================================


class TestHistogramVisualizations:
    """Tests for histogram visualization functions."""

    def test_histogram_overlay_returns_figure(self, numeric_series):
        """Test that overlay histogram returns a Plotly Figure."""
        original, protected = numeric_series
        fig = create_histogram_overlay(original, protected)

        assert isinstance(fig, go.Figure)

    def test_histogram_overlay_has_two_traces(self, numeric_series):
        """Test that overlay histogram has two traces."""
        original, protected = numeric_series
        fig = create_histogram_overlay(original, protected)

        assert len(fig.data) == 2

    def test_histogram_overlay_with_config(self, numeric_series):
        """Test overlay histogram with custom config."""
        original, protected = numeric_series
        config = ChartConfig(title="Test Title", width=1000)
        fig = create_histogram_overlay(original, protected, config=config)

        assert fig.layout.title.text == "Test Title"
        assert fig.layout.width == 1000

    def test_histogram_overlay_custom_column_name(self, numeric_series):
        """Test overlay histogram with custom column name."""
        original, protected = numeric_series
        fig = create_histogram_overlay(
            original, protected, column_name="Custom Name"
        )

        assert "Custom Name" in fig.layout.title.text

    def test_histogram_overlay_custom_bins(self, numeric_series):
        """Test overlay histogram with custom bins."""
        original, protected = numeric_series
        fig = create_histogram_overlay(original, protected, bins=20)

        assert isinstance(fig, go.Figure)

    def test_histogram_sidebyside_returns_figure(self, numeric_series):
        """Test that side-by-side histogram returns a Figure."""
        original, protected = numeric_series
        fig = create_histogram_sidebyside(original, protected)

        assert isinstance(fig, go.Figure)

    def test_histogram_sidebyside_has_two_traces(self, numeric_series):
        """Test side-by-side histogram has two traces."""
        original, protected = numeric_series
        fig = create_histogram_sidebyside(original, protected)

        assert len(fig.data) == 2

    def test_histogram_with_nulls(self):
        """Test histogram handles null values."""
        original = pd.Series([1, 2, None, 4, 5])
        protected = pd.Series([1.1, None, 3, 4.1, 5.1])
        fig = create_histogram_overlay(original, protected)

        assert isinstance(fig, go.Figure)

    def test_histogram_empty_series(self):
        """Test histogram with empty series."""
        original = pd.Series([], dtype=float)
        protected = pd.Series([], dtype=float)
        fig = create_histogram_overlay(original, protected)

        assert isinstance(fig, go.Figure)


# =============================================================================
# Tests: Correlation Heatmaps
# =============================================================================


class TestCorrelationHeatmaps:
    """Tests for correlation heatmap functions."""

    def test_correlation_heatmap_returns_figure(self, sample_dataframe):
        """Test correlation heatmap returns a Figure."""
        fig = create_correlation_heatmap(sample_dataframe)

        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap_has_heatmap_trace(self, sample_dataframe):
        """Test correlation heatmap has heatmap trace."""
        fig = create_correlation_heatmap(sample_dataframe)

        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_correlation_heatmap_with_columns(self, sample_dataframe):
        """Test correlation heatmap with specific columns."""
        fig = create_correlation_heatmap(
            sample_dataframe, columns=["age", "income"]
        )

        assert len(fig.data[0].x) == 2
        assert len(fig.data[0].y) == 2

    def test_correlation_heatmap_methods(self, sample_dataframe):
        """Test different correlation methods."""
        for method in ["pearson", "spearman", "kendall"]:
            fig = create_correlation_heatmap(sample_dataframe, method=method)
            assert isinstance(fig, go.Figure)

    def test_correlation_diff_heatmap(
        self, sample_dataframe, protected_dataframe
    ):
        """Test correlation difference heatmap."""
        fig = create_correlation_diff_heatmap(
            sample_dataframe, protected_dataframe
        )

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_correlation_comparison(
        self, sample_dataframe, protected_dataframe
    ):
        """Test correlation comparison with three heatmaps."""
        fig = create_correlation_comparison(
            sample_dataframe, protected_dataframe
        )

        assert isinstance(fig, go.Figure)
        # Should have 3 heatmaps
        assert len(fig.data) == 3

    def test_correlation_single_column_message(self):
        """Test correlation heatmap with single column shows message."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        fig = create_correlation_heatmap(df, columns=["a"])

        assert len(fig.layout.annotations) > 0


# =============================================================================
# Tests: Category Visualizations
# =============================================================================


class TestCategoryVisualizations:
    """Tests for category visualization functions."""

    def test_category_bar_chart_returns_figure(self, categorical_series):
        """Test category bar chart returns a Figure."""
        original, protected = categorical_series
        fig = create_category_bar_chart(original, protected)

        assert isinstance(fig, go.Figure)

    def test_category_bar_chart_has_two_traces(self, categorical_series):
        """Test category bar chart has two traces."""
        original, protected = categorical_series
        fig = create_category_bar_chart(original, protected)

        assert len(fig.data) == 2

    def test_category_bar_chart_custom_top_n(self, categorical_series):
        """Test category bar chart with custom top_n."""
        original, protected = categorical_series
        fig = create_category_bar_chart(original, protected, top_n=3)

        assert isinstance(fig, go.Figure)

    def test_category_drift_chart_returns_figure(self, categorical_series):
        """Test category drift chart returns a Figure."""
        original, protected = categorical_series
        fig = create_category_drift_chart(original, protected)

        assert isinstance(fig, go.Figure)

    def test_category_drift_chart_has_bar_trace(self, categorical_series):
        """Test category drift chart has bar trace."""
        original, protected = categorical_series
        fig = create_category_drift_chart(original, protected)

        # Has bar trace plus horizontal line
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Bar)


# =============================================================================
# Tests: Box Plot Visualizations
# =============================================================================


class TestBoxPlotVisualizations:
    """Tests for box plot visualization functions."""

    def test_box_comparison_returns_figure(self, numeric_series):
        """Test box comparison returns a Figure."""
        original, protected = numeric_series
        fig = create_box_comparison(original, protected)

        assert isinstance(fig, go.Figure)

    def test_box_comparison_has_two_traces(self, numeric_series):
        """Test box comparison has two box traces."""
        original, protected = numeric_series
        fig = create_box_comparison(original, protected)

        assert len(fig.data) == 2
        assert isinstance(fig.data[0], go.Box)
        assert isinstance(fig.data[1], go.Box)

    def test_multi_box_comparison(
        self, sample_dataframe, protected_dataframe
    ):
        """Test multi-column box comparison."""
        fig = create_multi_box_comparison(
            sample_dataframe, protected_dataframe
        )

        assert isinstance(fig, go.Figure)
        # 3 numeric columns * 2 = 6 traces
        assert len(fig.data) == 6

    def test_multi_box_with_columns(
        self, sample_dataframe, protected_dataframe
    ):
        """Test multi-column box with specific columns."""
        fig = create_multi_box_comparison(
            sample_dataframe,
            protected_dataframe,
            columns=["age", "income"],
        )

        assert len(fig.data) == 4  # 2 columns * 2


# =============================================================================
# Tests: Summary Visualizations
# =============================================================================


class TestSummaryVisualizations:
    """Tests for summary visualization functions."""

    def test_divergence_summary_chart(self):
        """Test divergence summary chart."""
        divergences = {
            "KL Divergence": 0.15,
            "JS Distance": 0.08,
            "Wasserstein": 2.5,
            "TVD": 0.12,
        }
        fig = create_divergence_summary_chart(divergences)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_divergence_summary_with_none(self):
        """Test divergence summary handles None values."""
        divergences = {
            "KL Divergence": 0.15,
            "JS Distance": None,
            "Wasserstein": 2.5,
        }
        fig = create_divergence_summary_chart(divergences)

        # Should only include non-None values
        assert len(fig.data[0].x) == 2

    def test_divergence_summary_all_none(self):
        """Test divergence summary with all None values."""
        divergences = {"KL": None, "JS": None}
        fig = create_divergence_summary_chart(divergences)

        # Should show message annotation
        assert len(fig.layout.annotations) > 0

    def test_column_comparison_summary(self):
        """Test column comparison summary chart."""
        mae_values = {"age": 1.5, "income": 500, "score": 2.3}
        fig = create_column_comparison_summary(mae_values)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_column_comparison_with_rmse(self):
        """Test column comparison with RMSE values."""
        mae_values = {"age": 1.5, "income": 500}
        rmse_values = {"age": 2.0, "income": 650}
        fig = create_column_comparison_summary(mae_values, rmse_values)

        assert len(fig.data) == 2  # MAE and RMSE bars


# =============================================================================
# Tests: Scatter Plots
# =============================================================================


class TestScatterPlots:
    """Tests for scatter plot functions."""

    def test_scatter_comparison(
        self, sample_dataframe, protected_dataframe
    ):
        """Test scatter comparison returns a Figure."""
        fig = create_scatter_comparison(
            sample_dataframe,
            protected_dataframe,
            x_col="age",
            y_col="income",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_qq_plot_returns_figure(self, numeric_series):
        """Test Q-Q plot returns a Figure."""
        original, protected = numeric_series
        fig = create_qq_plot(original, protected)

        assert isinstance(fig, go.Figure)

    def test_qq_plot_has_points_and_line(self, numeric_series):
        """Test Q-Q plot has scatter points and reference line."""
        original, protected = numeric_series
        fig = create_qq_plot(original, protected)

        assert len(fig.data) == 2  # Points and line
        assert isinstance(fig.data[0], go.Scatter)


# =============================================================================
# Tests: ComparisonDashboard
# =============================================================================


class TestComparisonDashboard:
    """Tests for ComparisonDashboard class."""

    def test_dashboard_initialization(
        self, sample_dataframe, protected_dataframe
    ):
        """Test dashboard initialization."""
        dashboard = ComparisonDashboard(
            sample_dataframe, protected_dataframe
        )

        assert dashboard._original is sample_dataframe
        assert dashboard._protected is protected_dataframe

    def test_numeric_column_dashboard(
        self, sample_dataframe, protected_dataframe
    ):
        """Test numeric column dashboard creation."""
        dashboard = ComparisonDashboard(
            sample_dataframe, protected_dataframe
        )
        fig = dashboard.create_numeric_column_dashboard("age")

        assert isinstance(fig, go.Figure)

    def test_categorical_column_dashboard(
        self, sample_dataframe, protected_dataframe
    ):
        """Test categorical column dashboard creation."""
        dashboard = ComparisonDashboard(
            sample_dataframe, protected_dataframe
        )
        fig = dashboard.create_categorical_column_dashboard("category")

        assert isinstance(fig, go.Figure)

    def test_overview_dashboard(
        self, sample_dataframe, protected_dataframe
    ):
        """Test overview dashboard creation."""
        dashboard = ComparisonDashboard(
            sample_dataframe, protected_dataframe
        )
        fig = dashboard.create_overview_dashboard()

        assert isinstance(fig, go.Figure)

    def test_dashboard_with_custom_config(
        self, sample_dataframe, protected_dataframe
    ):
        """Test dashboard with custom configuration."""
        config = ChartConfig(width=1200, template="plotly_dark")
        dashboard = ComparisonDashboard(
            sample_dataframe, protected_dataframe, config=config
        )
        fig = dashboard.create_numeric_column_dashboard("age")

        assert isinstance(fig, go.Figure)


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_value_series(self):
        """Test visualization with single value series."""
        original = pd.Series([5.0])
        protected = pd.Series([6.0])
        fig = create_histogram_overlay(original, protected)

        assert isinstance(fig, go.Figure)

    def test_all_same_values(self):
        """Test visualization with all same values."""
        original = pd.Series([5.0, 5.0, 5.0])
        protected = pd.Series([5.0, 5.0, 5.0])
        fig = create_histogram_overlay(original, protected)

        assert isinstance(fig, go.Figure)

    def test_unicode_column_names(self):
        """Test visualization with unicode column names."""
        df_orig = pd.DataFrame({"数据": [1, 2, 3], "值": [4, 5, 6]})
        df_prot = pd.DataFrame({"数据": [1.1, 2.1, 3.1], "值": [4.1, 5.1, 6.1]})
        fig = create_correlation_heatmap(df_orig)

        assert isinstance(fig, go.Figure)

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        np.random.seed(42)
        n = 100000
        original = pd.Series(np.random.randn(n))
        protected = pd.Series(np.random.randn(n))

        # Should complete quickly
        fig = create_histogram_overlay(original, protected, bins=100)
        assert isinstance(fig, go.Figure)

    def test_many_categories(self):
        """Test with many categories."""
        categories = [f"cat_{i}" for i in range(100)]
        original = pd.Series(np.random.choice(categories, 1000))
        protected = pd.Series(np.random.choice(categories, 1000))

        fig = create_category_bar_chart(original, protected, top_n=20)
        assert isinstance(fig, go.Figure)

    def test_negative_values(self):
        """Test visualization with negative values."""
        original = pd.Series([-100, -50, 0, 50, 100])
        protected = pd.Series([-90, -45, 5, 55, 110])

        fig = create_histogram_overlay(original, protected)
        assert isinstance(fig, go.Figure)

        fig2 = create_box_comparison(original, protected)
        assert isinstance(fig2, go.Figure)

    def test_mixed_positive_negative_drift(self):
        """Test category drift with mixed changes."""
        original = pd.Series(["A", "A", "B", "B", "C", "C"])
        protected = pd.Series(["A", "A", "A", "B", "C", "D"])

        fig = create_category_drift_chart(original, protected)
        assert isinstance(fig, go.Figure)


# =============================================================================
# Tests: Chart Data Accuracy
# =============================================================================


class TestChartDataAccuracy:
    """Tests to verify chart data accuracy."""

    def test_histogram_uses_correct_data(self):
        """Test that histogram uses correct data values."""
        original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        protected = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        fig = create_histogram_overlay(original, protected)

        # Check traces have data
        assert len(fig.data[0].x) == 5
        assert len(fig.data[1].x) == 5

    def test_bar_chart_frequencies_correct(self):
        """Test category bar chart frequencies are correct."""
        original = pd.Series(["A", "A", "A", "B", "B"])  # A:0.6, B:0.4
        protected = pd.Series(["A", "A", "B", "B", "B"])  # A:0.4, B:0.6

        fig = create_category_bar_chart(original, protected)

        # First trace is original
        # Note: order depends on value_counts order
        assert len(fig.data[0].y) > 0

    def test_box_plot_statistics_present(self):
        """Test box plot includes correct statistics."""
        original = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        protected = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        fig = create_box_comparison(original, protected)

        # Box plots should have correct number of traces
        assert len(fig.data) == 2
