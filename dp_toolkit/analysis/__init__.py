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

__all__ = [
    # Enums
    "ComparisonType",
    # Data classes - Numeric
    "NumericDivergence",
    "NumericComparison",
    # Data classes - Categorical
    "CategoricalDivergence",
    "CategoricalComparison",
    # Data classes - Date
    "DateDivergence",
    "DateComparison",
    # Data classes - Correlation
    "CorrelationPreservation",
    # Data classes - Dataset
    "DatasetComparison",
    # Comparator classes
    "ColumnComparator",
    "DatasetComparator",
    # Convenience functions
    "compare_numeric_column",
    "compare_categorical_column",
    "compare_date_column",
    "compare_datasets",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
]
