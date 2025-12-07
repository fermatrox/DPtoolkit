"""Epsilon and mechanism recommendation advisor.

This module provides intelligent recommendations for differential privacy
parameters based on column sensitivity levels and data types:

- Epsilon values by sensitivity classification (High, Medium, Low)
- DP mechanism by data type (Laplace, Gaussian, Exponential)
- Explanatory text for each recommendation

Epsilon ranges (from PRD):
- High sensitivity: ε = 0.1–0.5 (stronger privacy)
- Medium sensitivity: ε = 0.5–2.0 (balanced)
- Low sensitivity: ε = 2.0–5.0 (more utility)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dp_toolkit.recommendations.classifier import (
    SensitivityLevel,
    ClassificationResult,
    ColumnClassifier,
)


# =============================================================================
# Enums
# =============================================================================


class RecommendedMechanism(Enum):
    """Recommended DP mechanism types."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


class DataType(Enum):
    """Column data types for mechanism selection."""

    NUMERIC_BOUNDED = "numeric_bounded"
    NUMERIC_UNBOUNDED = "numeric_unbounded"
    CATEGORICAL = "categorical"
    DATE = "date"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


class UtilityLevel(Enum):
    """Expected utility preservation level."""

    HIGH = "high"  # Low noise, good utility
    MEDIUM = "medium"  # Balanced
    LOW = "low"  # High noise, lower utility

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


# =============================================================================
# Constants - Epsilon Ranges
# =============================================================================


# Epsilon ranges by sensitivity level (from PRD)
EPSILON_RANGES: Dict[SensitivityLevel, Tuple[float, float]] = {
    SensitivityLevel.HIGH: (0.1, 0.5),
    SensitivityLevel.MEDIUM: (0.5, 2.0),
    SensitivityLevel.LOW: (2.0, 5.0),
    SensitivityLevel.UNKNOWN: (0.5, 2.0),  # Default to medium
}

# Default epsilon values (midpoint of each range)
DEFAULT_EPSILON: Dict[SensitivityLevel, float] = {
    SensitivityLevel.HIGH: 0.3,
    SensitivityLevel.MEDIUM: 1.0,
    SensitivityLevel.LOW: 3.0,
    SensitivityLevel.UNKNOWN: 1.0,
}

# Delta for Gaussian mechanism (standard value)
DEFAULT_DELTA = 1e-5


# =============================================================================
# Explanation Templates
# =============================================================================


SENSITIVITY_EXPLANATIONS: Dict[SensitivityLevel, str] = {
    SensitivityLevel.HIGH: (
        "This column contains highly sensitive information (e.g., direct identifiers, "
        "protected health information). A lower epsilon (ε = 0.1–0.5) provides stronger "
        "privacy protection by adding more noise, which is recommended for data that "
        "could directly identify individuals."
    ),
    SensitivityLevel.MEDIUM: (
        "This column contains moderately sensitive information (e.g., quasi-identifiers, "
        "demographic data). A medium epsilon (ε = 0.5–2.0) balances privacy protection "
        "with data utility, suitable for data that could contribute to re-identification "
        "when combined with other attributes."
    ),
    SensitivityLevel.LOW: (
        "This column contains less sensitive information (e.g., operational data, "
        "aggregated metrics). A higher epsilon (ε = 2.0–5.0) preserves more utility "
        "while still providing formal privacy guarantees, appropriate for data with "
        "lower re-identification risk."
    ),
    SensitivityLevel.UNKNOWN: (
        "The sensitivity of this column could not be determined automatically. "
        "A medium epsilon (ε = 0.5–2.0) is recommended as a balanced default. "
        "Consider manually reviewing this column to assign an appropriate sensitivity level."
    ),
}

MECHANISM_EXPLANATIONS: Dict[RecommendedMechanism, str] = {
    RecommendedMechanism.LAPLACE: (
        "The Laplace mechanism is recommended for bounded numeric data. It provides "
        "pure ε-differential privacy by adding noise from the Laplace distribution. "
        "This mechanism is efficient and provides strong privacy guarantees when the "
        "data range is known."
    ),
    RecommendedMechanism.GAUSSIAN: (
        "The Gaussian mechanism is recommended for unbounded numeric data. It provides "
        "(ε,δ)-differential privacy by adding noise from the Gaussian distribution. "
        "This mechanism is useful when data bounds are unknown or when tighter "
        "composition bounds are needed across multiple queries."
    ),
    RecommendedMechanism.EXPONENTIAL: (
        "The Exponential mechanism is recommended for categorical data. It selects "
        "categories with probability proportional to a utility score, providing "
        "ε-differential privacy while preserving the general distribution shape "
        "of categorical values."
    ),
}

DATA_TYPE_EXPLANATIONS: Dict[DataType, str] = {
    DataType.NUMERIC_BOUNDED: (
        "This numeric column has identifiable bounds (min/max values). "
        "The Laplace mechanism can efficiently add calibrated noise based on "
        "the sensitivity derived from these bounds."
    ),
    DataType.NUMERIC_UNBOUNDED: (
        "This numeric column does not have clear bounds or has extreme outliers. "
        "The Gaussian mechanism is more appropriate as it doesn't require strict "
        "bounds and provides (ε,δ)-differential privacy."
    ),
    DataType.CATEGORICAL: (
        "This column contains categorical (non-numeric) values. "
        "The Exponential mechanism preserves the distribution shape while "
        "providing differential privacy guarantees."
    ),
    DataType.DATE: (
        "This column contains date/time values. Dates are converted to numeric "
        "epoch values, then the Laplace mechanism is applied with appropriate "
        "bounds based on the date range."
    ),
    DataType.UNKNOWN: (
        "The data type of this column could not be determined. "
        "Please review and manually specify the appropriate mechanism."
    ),
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EpsilonRecommendation:
    """Epsilon recommendation for a column.

    Attributes:
        sensitivity: Sensitivity level of the column.
        epsilon: Recommended epsilon value.
        epsilon_range: Valid range (min, max) for this sensitivity level.
        explanation: Human-readable explanation of the recommendation.
        confidence: Confidence in the recommendation (0.0 to 1.0).
    """

    sensitivity: SensitivityLevel
    epsilon: float
    epsilon_range: Tuple[float, float]
    explanation: str
    confidence: float = 0.8

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sensitivity": self.sensitivity.value,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_range[0],
            "epsilon_max": self.epsilon_range[1],
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


@dataclass
class MechanismRecommendation:
    """Mechanism recommendation for a column.

    Attributes:
        data_type: Detected data type of the column.
        mechanism: Recommended DP mechanism.
        delta: Delta parameter (for Gaussian only).
        explanation: Human-readable explanation.
        confidence: Confidence in the recommendation (0.0 to 1.0).
    """

    data_type: DataType
    mechanism: RecommendedMechanism
    delta: Optional[float] = None
    explanation: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "data_type": self.data_type.value,
            "mechanism": self.mechanism.value,
            "delta": self.delta,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


@dataclass
class ColumnRecommendation:
    """Complete recommendation for a single column.

    Attributes:
        column_name: Name of the column.
        epsilon_recommendation: Epsilon recommendation.
        mechanism_recommendation: Mechanism recommendation.
        classification: Original sensitivity classification.
        utility_level: Expected utility preservation.
        overall_explanation: Combined explanation text.
        is_override: Whether recommendations were manually overridden.
    """

    column_name: str
    epsilon_recommendation: EpsilonRecommendation
    mechanism_recommendation: MechanismRecommendation
    classification: Optional[ClassificationResult] = None
    utility_level: UtilityLevel = UtilityLevel.MEDIUM
    overall_explanation: str = ""
    is_override: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "column_name": self.column_name,
            "epsilon": self.epsilon_recommendation.to_dict(),
            "mechanism": self.mechanism_recommendation.to_dict(),
            "classification": (
                self.classification.to_dict() if self.classification else None
            ),
            "utility_level": self.utility_level.value,
            "overall_explanation": self.overall_explanation,
            "is_override": self.is_override,
        }


@dataclass
class DatasetRecommendation:
    """Recommendations for an entire dataset.

    Attributes:
        column_recommendations: Dictionary of column recommendations.
        total_epsilon: Sum of all column epsilons (sequential composition).
        global_epsilon_suggestion: Suggested global epsilon if using uniform.
    """

    column_recommendations: Dict[str, ColumnRecommendation] = field(
        default_factory=dict
    )

    @property
    def total_epsilon(self) -> float:
        """Calculate total epsilon (sequential composition)."""
        return sum(
            rec.epsilon_recommendation.epsilon
            for rec in self.column_recommendations.values()
        )

    @property
    def global_epsilon_suggestion(self) -> float:
        """Suggest a global epsilon based on average sensitivity."""
        if not self.column_recommendations:
            return DEFAULT_EPSILON[SensitivityLevel.MEDIUM]

        epsilons = [
            rec.epsilon_recommendation.epsilon
            for rec in self.column_recommendations.values()
        ]
        return sum(epsilons) / len(epsilons)

    @property
    def high_sensitivity_columns(self) -> List[str]:
        """Get columns with high sensitivity."""
        return [
            name
            for name, rec in self.column_recommendations.items()
            if rec.epsilon_recommendation.sensitivity == SensitivityLevel.HIGH
        ]

    @property
    def columns_by_mechanism(self) -> Dict[RecommendedMechanism, List[str]]:
        """Group columns by recommended mechanism."""
        result: Dict[RecommendedMechanism, List[str]] = {
            RecommendedMechanism.LAPLACE: [],
            RecommendedMechanism.GAUSSIAN: [],
            RecommendedMechanism.EXPONENTIAL: [],
        }
        for name, rec in self.column_recommendations.items():
            mechanism = rec.mechanism_recommendation.mechanism
            result[mechanism].append(name)
        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "columns": {
                name: rec.to_dict()
                for name, rec in self.column_recommendations.items()
            },
            "summary": {
                "total_epsilon": self.total_epsilon,
                "global_epsilon_suggestion": self.global_epsilon_suggestion,
                "high_sensitivity_columns": self.high_sensitivity_columns,
                "columns_by_mechanism": {
                    k.value: v for k, v in self.columns_by_mechanism.items()
                },
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for display."""
        rows = []
        for name, rec in self.column_recommendations.items():
            rows.append({
                "Column": name,
                "Sensitivity": rec.epsilon_recommendation.sensitivity.value.upper(),
                "Epsilon": f"{rec.epsilon_recommendation.epsilon:.2f}",
                "Mechanism": rec.mechanism_recommendation.mechanism.value.title(),
                "Data Type": rec.mechanism_recommendation.data_type.value.replace(
                    "_", " "
                ).title(),
                "Utility": rec.utility_level.value.upper(),
            })
        return pd.DataFrame(rows)


# =============================================================================
# Epsilon Advisor
# =============================================================================


class EpsilonAdvisor:
    """Recommends epsilon values based on column sensitivity.

    This advisor maps sensitivity levels to appropriate epsilon ranges
    as specified in the PRD:
    - High sensitivity: ε = 0.1–0.5
    - Medium sensitivity: ε = 0.5–2.0
    - Low sensitivity: ε = 2.0–5.0

    Example:
        >>> advisor = EpsilonAdvisor()
        >>> rec = advisor.recommend_epsilon(SensitivityLevel.HIGH)
        >>> print(f"Epsilon: {rec.epsilon}")
        Epsilon: 0.3
    """

    def __init__(
        self,
        custom_ranges: Optional[Dict[SensitivityLevel, Tuple[float, float]]] = None,
        custom_defaults: Optional[Dict[SensitivityLevel, float]] = None,
    ):
        """Initialize the advisor.

        Args:
            custom_ranges: Override default epsilon ranges.
            custom_defaults: Override default epsilon values.
        """
        self._ranges = dict(EPSILON_RANGES)
        self._defaults = dict(DEFAULT_EPSILON)

        if custom_ranges:
            self._ranges.update(custom_ranges)
        if custom_defaults:
            self._defaults.update(custom_defaults)

    def recommend_epsilon(
        self,
        sensitivity: SensitivityLevel,
        prefer_privacy: bool = False,
        prefer_utility: bool = False,
    ) -> EpsilonRecommendation:
        """Recommend an epsilon value for a sensitivity level.

        Args:
            sensitivity: The sensitivity level of the column.
            prefer_privacy: If True, use lower end of range.
            prefer_utility: If True, use higher end of range.

        Returns:
            EpsilonRecommendation with recommended value and explanation.
        """
        eps_range = self._ranges.get(
            sensitivity, self._ranges[SensitivityLevel.MEDIUM]
        )
        default_eps = self._defaults.get(
            sensitivity, self._defaults[SensitivityLevel.MEDIUM]
        )

        # Adjust based on preference
        if prefer_privacy:
            epsilon = eps_range[0]  # Lower = more privacy
            confidence = 0.9
        elif prefer_utility:
            epsilon = eps_range[1]  # Higher = more utility
            confidence = 0.9
        else:
            epsilon = default_eps
            confidence = 0.8

        explanation = SENSITIVITY_EXPLANATIONS.get(
            sensitivity, SENSITIVITY_EXPLANATIONS[SensitivityLevel.UNKNOWN]
        )

        return EpsilonRecommendation(
            sensitivity=sensitivity,
            epsilon=epsilon,
            epsilon_range=eps_range,
            explanation=explanation,
            confidence=confidence,
        )

    def get_epsilon_range(
        self, sensitivity: SensitivityLevel
    ) -> Tuple[float, float]:
        """Get the epsilon range for a sensitivity level."""
        return self._ranges.get(
            sensitivity, self._ranges[SensitivityLevel.MEDIUM]
        )

    def validate_epsilon(
        self,
        epsilon: float,
        sensitivity: SensitivityLevel,
    ) -> Tuple[bool, str]:
        """Validate an epsilon value against the recommended range.

        Args:
            epsilon: The epsilon value to validate.
            sensitivity: The sensitivity level.

        Returns:
            Tuple of (is_valid, message).
        """
        eps_range = self.get_epsilon_range(sensitivity)

        if epsilon < 0.01:
            return False, "Epsilon must be at least 0.01"
        if epsilon > 10.0:
            return False, "Epsilon should not exceed 10.0"

        if epsilon < eps_range[0]:
            return True, (
                f"Warning: Epsilon {epsilon} is below the recommended range "
                f"({eps_range[0]}–{eps_range[1]}) for {sensitivity.value} sensitivity. "
                "This will add more noise than typical for this sensitivity level."
            )
        if epsilon > eps_range[1]:
            return True, (
                f"Warning: Epsilon {epsilon} is above the recommended range "
                f"({eps_range[0]}–{eps_range[1]}) for {sensitivity.value} sensitivity. "
                "This may not provide adequate privacy protection."
            )

        return True, "Epsilon is within the recommended range."


# =============================================================================
# Mechanism Advisor
# =============================================================================


class MechanismAdvisor:
    """Recommends DP mechanisms based on column data type.

    Mechanism recommendations (from PRD):
    - Laplace: Bounded numeric data
    - Gaussian: Unbounded numeric data
    - Exponential: Categorical data

    Example:
        >>> advisor = MechanismAdvisor()
        >>> rec = advisor.recommend_mechanism(DataType.CATEGORICAL)
        >>> print(f"Mechanism: {rec.mechanism}")
        Mechanism: RecommendedMechanism.EXPONENTIAL
    """

    def __init__(self, default_delta: float = DEFAULT_DELTA):
        """Initialize the advisor.

        Args:
            default_delta: Default delta for Gaussian mechanism.
        """
        self._default_delta = default_delta

    def detect_data_type(
        self,
        series: pd.Series,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> DataType:
        """Detect the data type of a column.

        Args:
            series: The column data.
            bounds: Optional user-specified bounds.

        Returns:
            Detected DataType.
        """
        # Check if all null
        if series.isna().all():
            return DataType.UNKNOWN

        # Check for boolean first (before numeric, since bool can be treated as numeric)
        if pd.api.types.is_bool_dtype(series):
            return DataType.CATEGORICAL

        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATE

        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            # If bounds provided, it's bounded
            if bounds is not None:
                return DataType.NUMERIC_BOUNDED

            # Check for natural bounds
            non_null = series.dropna()
            if len(non_null) == 0:
                return DataType.UNKNOWN

            # Check for extreme outliers (suggests unbounded)
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            outliers = ((non_null < lower_bound) | (non_null > upper_bound)).sum()
            outlier_ratio = outliers / len(non_null) if len(non_null) > 0 else 0

            # If many outliers, treat as unbounded
            if outlier_ratio > 0.05:
                return DataType.NUMERIC_UNBOUNDED

            # Otherwise, treat as bounded
            return DataType.NUMERIC_BOUNDED

        # Check for categorical/string
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        ):
            return DataType.CATEGORICAL

        return DataType.UNKNOWN

    def recommend_mechanism(
        self,
        data_type: DataType,
        include_delta: bool = True,
    ) -> MechanismRecommendation:
        """Recommend a mechanism for a data type.

        Args:
            data_type: The data type of the column.
            include_delta: Whether to include delta for Gaussian.

        Returns:
            MechanismRecommendation with mechanism and explanation.
        """
        mechanism_map: Dict[DataType, RecommendedMechanism] = {
            DataType.NUMERIC_BOUNDED: RecommendedMechanism.LAPLACE,
            DataType.NUMERIC_UNBOUNDED: RecommendedMechanism.GAUSSIAN,
            DataType.CATEGORICAL: RecommendedMechanism.EXPONENTIAL,
            DataType.DATE: RecommendedMechanism.LAPLACE,
            DataType.UNKNOWN: RecommendedMechanism.LAPLACE,  # Default
        }

        mechanism = mechanism_map.get(data_type, RecommendedMechanism.LAPLACE)

        # Add delta for Gaussian
        delta = None
        if mechanism == RecommendedMechanism.GAUSSIAN and include_delta:
            delta = self._default_delta

        # Build explanation
        type_explanation = DATA_TYPE_EXPLANATIONS.get(
            data_type, DATA_TYPE_EXPLANATIONS[DataType.UNKNOWN]
        )
        mechanism_explanation = MECHANISM_EXPLANATIONS.get(
            mechanism, MECHANISM_EXPLANATIONS[RecommendedMechanism.LAPLACE]
        )
        full_explanation = f"{type_explanation}\n\n{mechanism_explanation}"

        return MechanismRecommendation(
            data_type=data_type,
            mechanism=mechanism,
            delta=delta,
            explanation=full_explanation,
            confidence=0.85 if data_type != DataType.UNKNOWN else 0.5,
        )

    def recommend_for_series(
        self,
        series: pd.Series,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> MechanismRecommendation:
        """Recommend a mechanism for a pandas Series.

        Args:
            series: The column data.
            bounds: Optional user-specified bounds.

        Returns:
            MechanismRecommendation.
        """
        data_type = self.detect_data_type(series, bounds)
        return self.recommend_mechanism(data_type)


# =============================================================================
# Combined Recommendation Advisor
# =============================================================================


class RecommendationAdvisor:
    """Combined advisor for epsilon and mechanism recommendations.

    This class integrates sensitivity classification, epsilon recommendation,
    and mechanism recommendation into a single interface.

    Example:
        >>> advisor = RecommendationAdvisor()
        >>> df = pd.DataFrame({"ssn": ["123-45-6789"], "age": [25]})
        >>> recommendations = advisor.recommend_for_dataset(df)
        >>> print(recommendations.high_sensitivity_columns)
        ['ssn']
    """

    def __init__(
        self,
        classifier: Optional[ColumnClassifier] = None,
        epsilon_advisor: Optional[EpsilonAdvisor] = None,
        mechanism_advisor: Optional[MechanismAdvisor] = None,
    ):
        """Initialize the combined advisor.

        Args:
            classifier: Custom column classifier.
            epsilon_advisor: Custom epsilon advisor.
            mechanism_advisor: Custom mechanism advisor.
        """
        self._classifier = classifier or ColumnClassifier()
        self._epsilon_advisor = epsilon_advisor or EpsilonAdvisor()
        self._mechanism_advisor = mechanism_advisor or MechanismAdvisor()
        self._overrides: Dict[str, ColumnRecommendation] = {}

    def add_override(
        self,
        column_name: str,
        epsilon: Optional[float] = None,
        mechanism: Optional[RecommendedMechanism] = None,
        sensitivity: Optional[SensitivityLevel] = None,
    ) -> None:
        """Add a manual override for a column.

        Args:
            column_name: Name of the column.
            epsilon: Override epsilon value.
            mechanism: Override mechanism.
            sensitivity: Override sensitivity level.
        """
        key = column_name.lower()

        # Get or create base recommendation
        if key in self._overrides:
            base = self._overrides[key]
        else:
            # Create default
            sens = sensitivity or SensitivityLevel.MEDIUM
            eps_rec = self._epsilon_advisor.recommend_epsilon(sens)
            mech_rec = MechanismRecommendation(
                data_type=DataType.UNKNOWN,
                mechanism=mechanism or RecommendedMechanism.LAPLACE,
                explanation="Manual override",
            )
            base = ColumnRecommendation(
                column_name=column_name,
                epsilon_recommendation=eps_rec,
                mechanism_recommendation=mech_rec,
                is_override=True,
            )

        # Apply overrides
        if epsilon is not None:
            base.epsilon_recommendation = EpsilonRecommendation(
                sensitivity=base.epsilon_recommendation.sensitivity,
                epsilon=epsilon,
                epsilon_range=base.epsilon_recommendation.epsilon_range,
                explanation="Manually specified epsilon value.",
                confidence=1.0,
            )
        if mechanism is not None:
            base.mechanism_recommendation = MechanismRecommendation(
                data_type=base.mechanism_recommendation.data_type,
                mechanism=mechanism,
                delta=(
                    DEFAULT_DELTA
                    if mechanism == RecommendedMechanism.GAUSSIAN
                    else None
                ),
                explanation="Manually specified mechanism.",
                confidence=1.0,
            )
        if sensitivity is not None:
            eps_rec = self._epsilon_advisor.recommend_epsilon(sensitivity)
            base.epsilon_recommendation = eps_rec
            base.classification = ClassificationResult(
                column_name=column_name,
                sensitivity=sensitivity,
                confidence=1.0,
                is_override=True,
                override_reason="Manual sensitivity override",
            )

        base.is_override = True
        self._overrides[key] = base

    def remove_override(self, column_name: str) -> bool:
        """Remove an override for a column."""
        key = column_name.lower()
        if key in self._overrides:
            del self._overrides[key]
            return True
        return False

    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()

    def _determine_utility_level(
        self,
        epsilon: float,
        sensitivity: SensitivityLevel,
    ) -> UtilityLevel:
        """Determine expected utility level based on epsilon and sensitivity."""
        eps_range = self._epsilon_advisor.get_epsilon_range(sensitivity)
        range_size = eps_range[1] - eps_range[0]

        if range_size == 0:
            return UtilityLevel.MEDIUM

        # Normalize epsilon within the range
        normalized = (epsilon - eps_range[0]) / range_size

        if normalized < 0.33:
            return UtilityLevel.LOW  # Lower epsilon = more noise = lower utility
        elif normalized > 0.66:
            return UtilityLevel.HIGH  # Higher epsilon = less noise = higher utility
        else:
            return UtilityLevel.MEDIUM

    def _build_overall_explanation(
        self,
        column_name: str,
        classification: ClassificationResult,
        eps_rec: EpsilonRecommendation,
        mech_rec: MechanismRecommendation,
    ) -> str:
        """Build a combined explanation for a column."""
        parts = [
            f"**Column: {column_name}**\n",
            f"**Sensitivity:** {classification.sensitivity.value.upper()}",
        ]

        if classification.pattern_description:
            parts.append(f" (detected as: {classification.pattern_description})")

        parts.append(f"\n**Recommended Epsilon:** {eps_rec.epsilon:.2f}")
        parts.append(f" (range: {eps_rec.epsilon_range[0]}–{eps_rec.epsilon_range[1]})")
        parts.append(f"\n**Recommended Mechanism:** {mech_rec.mechanism.value.title()}")

        if mech_rec.delta:
            parts.append(f" (δ = {mech_rec.delta})")

        return "".join(parts)

    def recommend_for_column(
        self,
        column_name: str,
        series: Optional[pd.Series] = None,
        bounds: Optional[Tuple[float, float]] = None,
        prefer_privacy: bool = False,
        prefer_utility: bool = False,
    ) -> ColumnRecommendation:
        """Generate recommendations for a single column.

        Args:
            column_name: Name of the column.
            series: Optional column data for mechanism detection.
            bounds: Optional bounds for numeric data.
            prefer_privacy: Prefer lower epsilon values.
            prefer_utility: Prefer higher epsilon values.

        Returns:
            ColumnRecommendation with epsilon and mechanism.
        """
        # Check for override
        key = column_name.lower()
        if key in self._overrides:
            return self._overrides[key]

        # Classify column
        classification = self._classifier.classify_column(column_name)

        # Get epsilon recommendation
        eps_rec = self._epsilon_advisor.recommend_epsilon(
            classification.sensitivity,
            prefer_privacy=prefer_privacy,
            prefer_utility=prefer_utility,
        )

        # Get mechanism recommendation
        if series is not None:
            mech_rec = self._mechanism_advisor.recommend_for_series(series, bounds)
        else:
            # Default based on common patterns
            mech_rec = MechanismRecommendation(
                data_type=DataType.UNKNOWN,
                mechanism=RecommendedMechanism.LAPLACE,
                explanation=(
                    "Default mechanism. Provide column data for more accurate "
                    "mechanism recommendation."
                ),
                confidence=0.5,
            )

        # Determine utility level
        utility = self._determine_utility_level(
            eps_rec.epsilon, classification.sensitivity
        )

        # Build explanation
        explanation = self._build_overall_explanation(
            column_name, classification, eps_rec, mech_rec
        )

        return ColumnRecommendation(
            column_name=column_name,
            epsilon_recommendation=eps_rec,
            mechanism_recommendation=mech_rec,
            classification=classification,
            utility_level=utility,
            overall_explanation=explanation,
        )

    def recommend_for_dataset(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        prefer_privacy: bool = False,
        prefer_utility: bool = False,
    ) -> DatasetRecommendation:
        """Generate recommendations for an entire dataset.

        Args:
            df: The DataFrame to analyze.
            columns: Optional list of columns to analyze.
            bounds: Optional bounds dictionary by column name.
            prefer_privacy: Prefer lower epsilon values.
            prefer_utility: Prefer higher epsilon values.

        Returns:
            DatasetRecommendation with all column recommendations.
        """
        cols = columns if columns is not None else list(df.columns)
        bounds = bounds or {}

        recommendations: Dict[str, ColumnRecommendation] = {}

        for col in cols:
            if col in df.columns:
                series = df[col]
                col_bounds = bounds.get(col)
            else:
                series = None
                col_bounds = None

            recommendations[col] = self.recommend_for_column(
                column_name=col,
                series=series,
                bounds=col_bounds,
                prefer_privacy=prefer_privacy,
                prefer_utility=prefer_utility,
            )

        return DatasetRecommendation(column_recommendations=recommendations)


# =============================================================================
# Convenience Functions
# =============================================================================


def recommend_epsilon(
    sensitivity: SensitivityLevel,
    prefer_privacy: bool = False,
    prefer_utility: bool = False,
) -> EpsilonRecommendation:
    """Recommend an epsilon value for a sensitivity level.

    Args:
        sensitivity: The sensitivity level.
        prefer_privacy: Prefer lower epsilon.
        prefer_utility: Prefer higher epsilon.

    Returns:
        EpsilonRecommendation.

    Example:
        >>> rec = recommend_epsilon(SensitivityLevel.HIGH)
        >>> print(f"Epsilon: {rec.epsilon}")
        Epsilon: 0.3
    """
    advisor = EpsilonAdvisor()
    return advisor.recommend_epsilon(sensitivity, prefer_privacy, prefer_utility)


def recommend_mechanism(
    data_type: DataType,
) -> MechanismRecommendation:
    """Recommend a mechanism for a data type.

    Args:
        data_type: The data type.

    Returns:
        MechanismRecommendation.

    Example:
        >>> rec = recommend_mechanism(DataType.CATEGORICAL)
        >>> print(f"Mechanism: {rec.mechanism}")
        Mechanism: RecommendedMechanism.EXPONENTIAL
    """
    advisor = MechanismAdvisor()
    return advisor.recommend_mechanism(data_type)


def recommend_for_column(
    column_name: str,
    series: Optional[pd.Series] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> ColumnRecommendation:
    """Generate recommendations for a single column.

    Args:
        column_name: Name of the column.
        series: Optional column data.
        bounds: Optional bounds for numeric data.

    Returns:
        ColumnRecommendation.

    Example:
        >>> rec = recommend_for_column("patient_ssn")
        >>> print(f"Epsilon: {rec.epsilon_recommendation.epsilon}")
        Epsilon: 0.3
    """
    advisor = RecommendationAdvisor()
    return advisor.recommend_for_column(column_name, series, bounds)


def recommend_for_dataset(
    df: pd.DataFrame,
    prefer_privacy: bool = False,
    prefer_utility: bool = False,
) -> DatasetRecommendation:
    """Generate recommendations for an entire dataset.

    Args:
        df: The DataFrame to analyze.
        prefer_privacy: Prefer lower epsilon values.
        prefer_utility: Prefer higher epsilon values.

    Returns:
        DatasetRecommendation.

    Example:
        >>> df = pd.DataFrame({"ssn": ["123-45-6789"], "age": [25]})
        >>> recs = recommend_for_dataset(df)
        >>> print(recs.high_sensitivity_columns)
        ['ssn']
    """
    advisor = RecommendationAdvisor()
    return advisor.recommend_for_dataset(
        df, prefer_privacy=prefer_privacy, prefer_utility=prefer_utility
    )


def get_epsilon_for_column(column_name: str) -> float:
    """Quick utility to get recommended epsilon for a column name.

    Args:
        column_name: Name of the column.

    Returns:
        Recommended epsilon value.

    Example:
        >>> epsilon = get_epsilon_for_column("patient_ssn")
        >>> print(f"Epsilon: {epsilon}")
        Epsilon: 0.3
    """
    rec = recommend_for_column(column_name)
    return rec.epsilon_recommendation.epsilon


def get_mechanism_for_series(series: pd.Series) -> RecommendedMechanism:
    """Quick utility to get recommended mechanism for a Series.

    Args:
        series: The column data.

    Returns:
        Recommended mechanism.

    Example:
        >>> s = pd.Series(["A", "B", "C"])
        >>> mech = get_mechanism_for_series(s)
        >>> print(f"Mechanism: {mech}")
        Mechanism: RecommendedMechanism.EXPONENTIAL
    """
    advisor = MechanismAdvisor()
    rec = advisor.recommend_for_series(series)
    return rec.mechanism


def validate_epsilon(
    epsilon: float,
    sensitivity: SensitivityLevel,
) -> Tuple[bool, str]:
    """Validate an epsilon value against the recommended range.

    Args:
        epsilon: The epsilon value.
        sensitivity: The sensitivity level.

    Returns:
        Tuple of (is_valid, message).

    Example:
        >>> valid, msg = validate_epsilon(0.1, SensitivityLevel.LOW)
        >>> print(f"Valid: {valid}, Message: {msg}")
    """
    advisor = EpsilonAdvisor()
    return advisor.validate_epsilon(epsilon, sensitivity)
