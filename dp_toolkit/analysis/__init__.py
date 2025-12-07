"""Statistical comparison, divergence metrics, and visualization."""

from dp_toolkit.analysis.comparator import (
    # Enums
    ComparisonType,
    # Data classes - Numeric
    NumericDivergence,
    NumericComparison,
    # Data classes - Categorical
    CategoricalDivergence,
    CategoricalComparison,
    # Data classes - Date
    DateDivergence,
    DateComparison,
    # Data classes - Correlation
    CorrelationPreservation,
    # Data classes - Dataset
    DatasetComparison,
    # Comparator classes
    ColumnComparator,
    DatasetComparator,
    # Convenience functions
    compare_numeric_column,
    compare_categorical_column,
    compare_date_column,
    compare_datasets,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
)
from dp_toolkit.analysis.divergence import (
    # Enums
    DivergenceType,
    # Data classes
    DivergenceResult,
    CategoryDriftResult,
    NumericDistributionComparison,
    # Core divergence functions
    kl_divergence,
    js_distance,
    js_divergence,
    wasserstein_distance,
    total_variation_distance,
    hellinger_distance,
    # Entropy functions
    entropy,
    cross_entropy,
    # Utility functions
    series_to_distribution,
    numeric_to_histogram,
    # Analysis functions
    analyze_category_drift,
    compare_numeric_distributions,
    # Convenience functions
    calculate_kl_divergence,
    calculate_js_distance,
    calculate_wasserstein_distance,
    calculate_tvd,
    calculate_all_divergences,
)
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

__all__ = [
    # Comparator - Enums
    "ComparisonType",
    # Comparator - Data classes - Numeric
    "NumericDivergence",
    "NumericComparison",
    # Comparator - Data classes - Categorical
    "CategoricalDivergence",
    "CategoricalComparison",
    # Comparator - Data classes - Date
    "DateDivergence",
    "DateComparison",
    # Comparator - Data classes - Correlation
    "CorrelationPreservation",
    # Comparator - Data classes - Dataset
    "DatasetComparison",
    # Comparator - classes
    "ColumnComparator",
    "DatasetComparator",
    # Comparator - Convenience functions
    "compare_numeric_column",
    "compare_categorical_column",
    "compare_date_column",
    "compare_datasets",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    # Divergence - Enums
    "DivergenceType",
    # Divergence - Data classes
    "DivergenceResult",
    "CategoryDriftResult",
    "NumericDistributionComparison",
    # Divergence - Core functions
    "kl_divergence",
    "js_distance",
    "js_divergence",
    "wasserstein_distance",
    "total_variation_distance",
    "hellinger_distance",
    # Divergence - Entropy functions
    "entropy",
    "cross_entropy",
    # Divergence - Utility functions
    "series_to_distribution",
    "numeric_to_histogram",
    # Divergence - Analysis functions
    "analyze_category_drift",
    "compare_numeric_distributions",
    # Divergence - Convenience functions
    "calculate_kl_divergence",
    "calculate_js_distance",
    "calculate_wasserstein_distance",
    "calculate_tvd",
    "calculate_all_divergences",
    # Visualizer - Enums
    "ChartType",
    # Visualizer - Data classes
    "ChartConfig",
    # Visualizer - Histogram visualizations
    "create_histogram_overlay",
    "create_histogram_sidebyside",
    # Visualizer - Correlation heatmaps
    "create_correlation_heatmap",
    "create_correlation_diff_heatmap",
    "create_correlation_comparison",
    # Visualizer - Category visualizations
    "create_category_bar_chart",
    "create_category_drift_chart",
    # Visualizer - Box plot visualizations
    "create_box_comparison",
    "create_multi_box_comparison",
    # Visualizer - Summary visualizations
    "create_divergence_summary_chart",
    "create_column_comparison_summary",
    # Visualizer - Scatter plots
    "create_scatter_comparison",
    "create_qq_plot",
    # Visualizer - Dashboard
    "ComparisonDashboard",
]
