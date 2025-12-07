"""Column sensitivity classification and epsilon/mechanism advisor."""

from dp_toolkit.recommendations.classifier import (
    # Enums
    SensitivityLevel,
    # Data classes
    ClassificationResult,
    DatasetClassification,
    # Classes
    ColumnClassifier,
    ContentAnalyzer,
    # Convenience functions
    classify_column,
    classify_columns,
    classify_dataset,
    get_sensitivity_for_column,
    get_sensitive_columns,
    # Pattern lists (for customization)
    HIGH_SENSITIVITY_PATTERNS,
    MEDIUM_SENSITIVITY_PATTERNS,
    LOW_SENSITIVITY_PATTERNS,
)

__all__ = [
    # Enums
    "SensitivityLevel",
    # Data classes
    "ClassificationResult",
    "DatasetClassification",
    # Classes
    "ColumnClassifier",
    "ContentAnalyzer",
    # Convenience functions
    "classify_column",
    "classify_columns",
    "classify_dataset",
    "get_sensitivity_for_column",
    "get_sensitive_columns",
    # Pattern lists
    "HIGH_SENSITIVITY_PATTERNS",
    "MEDIUM_SENSITIVITY_PATTERNS",
    "LOW_SENSITIVITY_PATTERNS",
]
