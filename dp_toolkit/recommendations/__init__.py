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

from dp_toolkit.recommendations.advisor import (
    # Enums
    RecommendedMechanism,
    DataType,
    UtilityLevel,
    # Data classes
    EpsilonRecommendation,
    MechanismRecommendation,
    ColumnRecommendation,
    DatasetRecommendation,
    # Classes
    EpsilonAdvisor,
    MechanismAdvisor,
    RecommendationAdvisor,
    # Convenience functions
    recommend_epsilon,
    recommend_mechanism,
    recommend_for_column,
    recommend_for_dataset,
    get_epsilon_for_column,
    get_mechanism_for_series,
    validate_epsilon,
    # Constants
    EPSILON_RANGES,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
)

__all__ = [
    # Classifier - Enums
    "SensitivityLevel",
    # Classifier - Data classes
    "ClassificationResult",
    "DatasetClassification",
    # Classifier - Classes
    "ColumnClassifier",
    "ContentAnalyzer",
    # Classifier - Convenience functions
    "classify_column",
    "classify_columns",
    "classify_dataset",
    "get_sensitivity_for_column",
    "get_sensitive_columns",
    # Classifier - Pattern lists
    "HIGH_SENSITIVITY_PATTERNS",
    "MEDIUM_SENSITIVITY_PATTERNS",
    "LOW_SENSITIVITY_PATTERNS",
    # Advisor - Enums
    "RecommendedMechanism",
    "DataType",
    "UtilityLevel",
    # Advisor - Data classes
    "EpsilonRecommendation",
    "MechanismRecommendation",
    "ColumnRecommendation",
    "DatasetRecommendation",
    # Advisor - Classes
    "EpsilonAdvisor",
    "MechanismAdvisor",
    "RecommendationAdvisor",
    # Advisor - Convenience functions
    "recommend_epsilon",
    "recommend_mechanism",
    "recommend_for_column",
    "recommend_for_dataset",
    "get_epsilon_for_column",
    "get_mechanism_for_series",
    "validate_epsilon",
    # Advisor - Constants
    "EPSILON_RANGES",
    "DEFAULT_EPSILON",
    "DEFAULT_DELTA",
]
