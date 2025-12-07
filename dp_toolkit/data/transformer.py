"""Column transformer for differential privacy.

This module provides DP transformation for individual columns, applying
the appropriate mechanism based on column type:
- Numeric (bounded): Laplace mechanism
- Numeric (unbounded): Gaussian mechanism
- Categorical: Exponential mechanism
- Date: Epoch conversion + Laplace mechanism

Key features:
- Null preservation (nulls pass through unchanged)
- Type preservation (integers rounded after noise)
- Date handling via epoch conversion
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dp_toolkit.core.mechanisms import (
    DELTA_MIN,
    DELTA_MAX,
    EPSILON_MIN,
    EPSILON_MAX,
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    PrivacyUsage,
)


# =============================================================================
# Enums and Constants
# =============================================================================


class MechanismType(Enum):
    """Type of DP mechanism to apply."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


class TransformColumnType(Enum):
    """Type of column for transformation purposes."""

    NUMERIC_BOUNDED = "numeric_bounded"
    NUMERIC_UNBOUNDED = "numeric_unbounded"
    CATEGORICAL = "categorical"
    DATE = "date"


# Default delta for Gaussian mechanism if not specified
DEFAULT_DELTA = 1e-6


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ColumnConfig:
    """Configuration for transforming a single column.

    Attributes:
        epsilon: Privacy parameter for this column.
        mechanism: Type of mechanism to use (auto-detected if None).
        lower: Lower bound for bounded numeric data.
        upper: Upper bound for bounded numeric data.
        delta: Delta parameter for Gaussian mechanism.
        sensitivity: User-specified sensitivity (for Gaussian).
        preserve_type: Whether to preserve integer types (default True).
    """

    epsilon: float = 1.0
    mechanism: Optional[MechanismType] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    delta: Optional[float] = None
    sensitivity: Optional[float] = None
    preserve_type: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.epsilon < EPSILON_MIN or self.epsilon > EPSILON_MAX:
            raise ValueError(
                f"Epsilon must be in [{EPSILON_MIN}, {EPSILON_MAX}], "
                f"got {self.epsilon}"
            )
        if self.delta is not None:
            if self.delta < DELTA_MIN or self.delta > DELTA_MAX:
                raise ValueError(
                    f"Delta must be in [{DELTA_MIN}, {DELTA_MAX}], " f"got {self.delta}"
                )


@dataclass
class TransformResult:
    """Result of transforming a column.

    Attributes:
        data: Transformed data as pandas Series.
        original_dtype: Original data type.
        column_type: Detected column type.
        mechanism_type: Mechanism used for transformation.
        privacy_usage: Privacy budget consumed.
        null_count: Number of null values preserved.
        bounds: Tuple of (lower, upper) bounds used (if applicable).
        metadata: Additional metadata about the transformation.
    """

    data: pd.Series
    original_dtype: np.dtype
    column_type: TransformColumnType
    mechanism_type: MechanismType
    privacy_usage: PrivacyUsage
    null_count: int
    bounds: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Column Transformer
# =============================================================================


class ColumnTransformer:
    """Transform individual columns with differential privacy.

    This class applies the appropriate DP mechanism based on column type:
    - Bounded numeric: Laplace mechanism
    - Unbounded numeric: Gaussian mechanism
    - Categorical: Exponential mechanism
    - Date: Epoch conversion + Laplace mechanism

    Example:
        >>> transformer = ColumnTransformer()
        >>> result = transformer.transform_numeric(
        ...     series=df['age'],
        ...     epsilon=1.0,
        ...     lower=0,
        ...     upper=120
        ... )
        >>> df['age_protected'] = result.data
    """

    def __init__(
        self,
        default_epsilon: float = 1.0,
        default_delta: float = DEFAULT_DELTA,
    ) -> None:
        """Initialize the transformer.

        Args:
            default_epsilon: Default epsilon for transformations.
            default_delta: Default delta for Gaussian mechanism.
        """
        if default_epsilon < EPSILON_MIN or default_epsilon > EPSILON_MAX:
            raise ValueError(
                f"Default epsilon must be in [{EPSILON_MIN}, {EPSILON_MAX}]"
            )
        if default_delta < DELTA_MIN or default_delta > DELTA_MAX:
            raise ValueError(f"Default delta must be in [{DELTA_MIN}, {DELTA_MAX}]")

        self._default_epsilon = default_epsilon
        self._default_delta = default_delta

    @property
    def default_epsilon(self) -> float:
        """Get default epsilon."""
        return self._default_epsilon

    @property
    def default_delta(self) -> float:
        """Get default delta."""
        return self._default_delta

    # -------------------------------------------------------------------------
    # Type Detection
    # -------------------------------------------------------------------------

    def detect_column_type(
        self,
        series: pd.Series,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> TransformColumnType:
        """Detect the column type for transformation.

        Args:
            series: Column data.
            bounds: Optional bounds for numeric data.

        Returns:
            Detected TransformColumnType.
        """
        # Check for datetime first
        if pd.api.types.is_datetime64_any_dtype(series):
            return TransformColumnType.DATE

        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            # If bounds provided, use bounded mechanism
            if bounds is not None:
                return TransformColumnType.NUMERIC_BOUNDED
            # Default to unbounded for numeric
            return TransformColumnType.NUMERIC_UNBOUNDED

        # Default to categorical for object/string types
        return TransformColumnType.CATEGORICAL

    def is_integer_type(self, series: pd.Series) -> bool:
        """Check if series is integer type.

        Args:
            series: Column data.

        Returns:
            True if integer type (including nullable integers).
        """
        return bool(pd.api.types.is_integer_dtype(series))

    # -------------------------------------------------------------------------
    # Numeric Transformation
    # -------------------------------------------------------------------------

    def transform_numeric_bounded(
        self,
        series: pd.Series,
        epsilon: float,
        lower: float,
        upper: float,
        preserve_type: bool = True,
    ) -> TransformResult:
        """Transform numeric column with Laplace mechanism.

        Applies Laplace noise for ε-differential privacy on bounded data.

        Args:
            series: Numeric column data.
            epsilon: Privacy parameter.
            lower: Lower bound of data.
            upper: Upper bound of data.
            preserve_type: If True, round to integers for integer columns.

        Returns:
            TransformResult with protected data.

        Raises:
            TypeError: If series is not numeric.
            ValueError: If bounds are invalid.
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Series must be numeric, got {series.dtype}")

        original_dtype = series.dtype
        is_integer = self.is_integer_type(series)
        null_mask = series.isna()
        null_count = int(null_mask.sum())

        # Create mechanism
        mechanism = LaplaceMechanism(lower=lower, upper=upper, epsilon=epsilon)

        # Get non-null values
        non_null_values = series.dropna().values

        if len(non_null_values) > 0:
            # Apply mechanism with clamping
            noisy_values = mechanism.release_array_clamped(non_null_values)

            # Round for integer types
            if preserve_type and is_integer:
                noisy_values = np.round(noisy_values).astype(np.int64)

            # Reconstruct series with nulls preserved
            result_data = pd.Series(index=series.index, dtype=float)
            result_data.loc[~null_mask] = noisy_values
            result_data.loc[null_mask] = np.nan
        else:
            # All nulls, return as-is
            result_data = series.copy()

        # Convert to appropriate type
        if preserve_type and is_integer and null_count == 0:
            result_data = result_data.astype(np.int64)
        elif preserve_type and is_integer:
            # Use nullable integer for nulls
            result_data = result_data.astype("Int64")

        return TransformResult(
            data=result_data,
            original_dtype=original_dtype,
            column_type=TransformColumnType.NUMERIC_BOUNDED,
            mechanism_type=MechanismType.LAPLACE,
            privacy_usage=mechanism.get_privacy_usage(),
            null_count=null_count,
            bounds=(lower, upper),
            metadata={
                "sensitivity": mechanism.sensitivity,
                "scale": mechanism.scale,
            },
        )

    def transform_numeric_unbounded(
        self,
        series: pd.Series,
        epsilon: float,
        sensitivity: float,
        delta: Optional[float] = None,
        preserve_type: bool = True,
    ) -> TransformResult:
        """Transform numeric column with Gaussian mechanism.

        Applies Gaussian noise for (ε,δ)-differential privacy on unbounded data.

        Args:
            series: Numeric column data.
            epsilon: Privacy parameter.
            sensitivity: L2 sensitivity of the data.
            delta: Approximate DP parameter (uses default if None).
            preserve_type: If True, round to integers for integer columns.

        Returns:
            TransformResult with protected data.

        Raises:
            TypeError: If series is not numeric.
            ValueError: If sensitivity is not positive.
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Series must be numeric, got {series.dtype}")

        if delta is None:
            delta = self._default_delta

        original_dtype = series.dtype
        is_integer = self.is_integer_type(series)
        null_mask = series.isna()
        null_count = int(null_mask.sum())

        # Create mechanism
        mechanism = GaussianMechanism(
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta,
        )

        # Get non-null values
        non_null_values = series.dropna().values

        if len(non_null_values) > 0:
            # Apply mechanism
            noisy_values = mechanism.release_array(non_null_values)

            # Round for integer types
            if preserve_type and is_integer:
                noisy_values = np.round(noisy_values).astype(np.int64)

            # Reconstruct series with nulls preserved
            result_data = pd.Series(index=series.index, dtype=float)
            result_data.loc[~null_mask] = noisy_values
            result_data.loc[null_mask] = np.nan
        else:
            result_data = series.copy()

        # Convert to appropriate type
        if preserve_type and is_integer and null_count == 0:
            result_data = result_data.astype(np.int64)
        elif preserve_type and is_integer:
            result_data = result_data.astype("Int64")

        return TransformResult(
            data=result_data,
            original_dtype=original_dtype,
            column_type=TransformColumnType.NUMERIC_UNBOUNDED,
            mechanism_type=MechanismType.GAUSSIAN,
            privacy_usage=mechanism.get_privacy_usage(),
            null_count=null_count,
            bounds=None,
            metadata={
                "sensitivity": mechanism.sensitivity,
                "scale": mechanism.scale,
            },
        )

    # -------------------------------------------------------------------------
    # Categorical Transformation
    # -------------------------------------------------------------------------

    def transform_categorical(
        self,
        series: pd.Series,
        epsilon: float,
        sensitivity: float = 1.0,
    ) -> TransformResult:
        """Transform categorical column with exponential mechanism.

        For each value, samples from the distribution of existing values
        with probability proportional to exp(ε * count / (2 * sensitivity)).

        This preserves the overall distribution shape while providing
        per-record differential privacy.

        Args:
            series: Categorical column data.
            epsilon: Privacy parameter.
            sensitivity: Sensitivity of the utility function (default 1.0).

        Returns:
            TransformResult with protected data.
        """
        original_dtype = series.dtype
        null_mask = series.isna()
        null_count = int(null_mask.sum())

        # Get value counts for utility scores
        non_null = series.dropna()

        if len(non_null) == 0:
            # All nulls
            return TransformResult(
                data=series.copy(),
                original_dtype=original_dtype,
                column_type=TransformColumnType.CATEGORICAL,
                mechanism_type=MechanismType.EXPONENTIAL,
                privacy_usage=PrivacyUsage(epsilon=epsilon),
                null_count=null_count,
            )

        # Get unique categories and their counts
        value_counts = non_null.value_counts()
        categories: List[Any] = list(value_counts.index)
        counts = value_counts.values.astype(float)

        # Need at least 2 categories for exponential mechanism
        if len(categories) < 2:
            # Only one category, return as-is (no randomness needed)
            return TransformResult(
                data=series.copy(),
                original_dtype=original_dtype,
                column_type=TransformColumnType.CATEGORICAL,
                mechanism_type=MechanismType.EXPONENTIAL,
                privacy_usage=PrivacyUsage(epsilon=epsilon),
                null_count=null_count,
                metadata={"single_category": True},
            )

        # Create mechanism
        mechanism = ExponentialMechanism(
            categories=categories,
            epsilon=epsilon,
            sensitivity=sensitivity,
        )

        # Sample new values for each non-null entry
        n_samples = len(non_null)
        sampled_values = mechanism.sample(counts, n_samples)

        # Reconstruct series with nulls preserved
        result_data = pd.Series(index=series.index, dtype=object)
        result_data.loc[~null_mask] = sampled_values
        result_data.loc[null_mask] = None

        # Try to preserve original dtype if categorical
        if pd.api.types.is_categorical_dtype(original_dtype):
            result_data = result_data.astype("category")

        return TransformResult(
            data=result_data,
            original_dtype=original_dtype,
            column_type=TransformColumnType.CATEGORICAL,
            mechanism_type=MechanismType.EXPONENTIAL,
            privacy_usage=mechanism.get_privacy_usage(),
            null_count=null_count,
            metadata={
                "n_categories": len(categories),
                "categories": categories,
            },
        )

    # -------------------------------------------------------------------------
    # Date Transformation
    # -------------------------------------------------------------------------

    def transform_date(
        self,
        series: pd.Series,
        epsilon: float,
        lower: Optional[pd.Timestamp] = None,
        upper: Optional[pd.Timestamp] = None,
        unit: str = "D",
    ) -> TransformResult:
        """Transform date column via epoch conversion.

        Converts dates to epoch (days or seconds), applies Laplace noise,
        and converts back to datetime.

        Args:
            series: Datetime column data.
            epsilon: Privacy parameter.
            lower: Lower bound date (uses min if None).
            upper: Upper bound date (uses max if None).
            unit: Time unit for noise - 'D' for days, 's' for seconds.

        Returns:
            TransformResult with protected data.

        Raises:
            TypeError: If series is not datetime.
            ValueError: If unit is invalid.
        """
        if not pd.api.types.is_datetime64_any_dtype(series):
            raise TypeError(f"Series must be datetime, got {series.dtype}")

        if unit not in ("D", "s"):
            raise ValueError(f"Unit must be 'D' or 's', got {unit}")

        original_dtype = series.dtype
        null_mask = series.isna()
        null_count = int(null_mask.sum())

        non_null = series.dropna()

        if len(non_null) == 0:
            return TransformResult(
                data=series.copy(),
                original_dtype=original_dtype,
                column_type=TransformColumnType.DATE,
                mechanism_type=MechanismType.LAPLACE,
                privacy_usage=PrivacyUsage(epsilon=epsilon),
                null_count=null_count,
            )

        # Determine bounds
        if lower is None:
            lower = non_null.min()
        if upper is None:
            upper = non_null.max()

        # Ensure bounds are Timestamps
        lower = pd.Timestamp(lower)
        upper = pd.Timestamp(upper)

        # Reference point for epoch conversion
        epoch = pd.Timestamp("1970-01-01")

        # Convert to numeric (days or seconds since epoch)
        if unit == "D":
            numeric_values = (non_null - epoch).dt.days.values.astype(float)
            lower_num = (lower - epoch).days
            upper_num = (upper - epoch).days
        else:  # seconds
            numeric_values = (non_null - epoch).dt.total_seconds().values
            lower_num = (lower - epoch).total_seconds()
            upper_num = (upper - epoch).total_seconds()

        # Apply Laplace mechanism
        mechanism = LaplaceMechanism(
            lower=lower_num,
            upper=upper_num,
            epsilon=epsilon,
        )
        noisy_values = mechanism.release_array_clamped(numeric_values)

        # Round to whole units for cleaner dates
        noisy_values = np.round(noisy_values)

        # Convert back to datetime
        if unit == "D":
            noisy_dates = pd.to_datetime(
                epoch + pd.to_timedelta(noisy_values, unit="D")
            )
        else:
            noisy_dates = pd.to_datetime(
                epoch + pd.to_timedelta(noisy_values, unit="s")
            )

        # Reconstruct series with nulls preserved
        result_data = pd.Series(index=series.index, dtype="datetime64[ns]")
        result_data.loc[~null_mask] = noisy_dates.values
        result_data.loc[null_mask] = pd.NaT

        return TransformResult(
            data=result_data,
            original_dtype=original_dtype,
            column_type=TransformColumnType.DATE,
            mechanism_type=MechanismType.LAPLACE,
            privacy_usage=mechanism.get_privacy_usage(),
            null_count=null_count,
            bounds=(lower, upper),
            metadata={
                "unit": unit,
                "sensitivity": mechanism.sensitivity,
                "scale": mechanism.scale,
            },
        )

    # -------------------------------------------------------------------------
    # Generic Transform
    # -------------------------------------------------------------------------

    def transform(
        self,
        series: pd.Series,
        config: Optional[ColumnConfig] = None,
        epsilon: Optional[float] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        sensitivity: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> TransformResult:
        """Transform a column with automatic mechanism selection.

        This is the main entry point that automatically detects the column
        type and applies the appropriate mechanism.

        Args:
            series: Column data to transform.
            config: Optional ColumnConfig with all settings.
            epsilon: Privacy parameter (overrides config).
            lower: Lower bound for bounded numeric (overrides config).
            upper: Upper bound for bounded numeric (overrides config).
            sensitivity: Sensitivity for unbounded (overrides config).
            delta: Delta for Gaussian (overrides config).

        Returns:
            TransformResult with protected data.
        """
        # Merge config with explicit parameters
        if config is None:
            config = ColumnConfig(epsilon=epsilon or self._default_epsilon)

        eps = epsilon or config.epsilon
        lb = lower if lower is not None else config.lower
        ub = upper if upper is not None else config.upper
        sens = sensitivity if sensitivity is not None else config.sensitivity
        dlt = delta if delta is not None else config.delta

        # Detect column type
        bounds = (lb, ub) if lb is not None and ub is not None else None
        col_type = self.detect_column_type(series, bounds)

        # Apply appropriate mechanism
        if col_type == TransformColumnType.DATE:
            return self.transform_date(
                series=series,
                epsilon=eps,
                lower=pd.Timestamp(lb) if lb else None,
                upper=pd.Timestamp(ub) if ub else None,
            )

        elif col_type == TransformColumnType.NUMERIC_BOUNDED:
            if lb is None or ub is None:
                raise ValueError("Bounds (lower, upper) required for bounded numeric")
            return self.transform_numeric_bounded(
                series=series,
                epsilon=eps,
                lower=lb,
                upper=ub,
                preserve_type=config.preserve_type,
            )

        elif col_type == TransformColumnType.NUMERIC_UNBOUNDED:
            if sens is None:
                # Estimate sensitivity from data range
                non_null = series.dropna()
                if len(non_null) > 0:
                    sens = float(non_null.max() - non_null.min())
                else:
                    sens = 1.0
            return self.transform_numeric_unbounded(
                series=series,
                epsilon=eps,
                sensitivity=sens,
                delta=dlt,
                preserve_type=config.preserve_type,
            )

        else:  # CATEGORICAL
            return self.transform_categorical(
                series=series,
                epsilon=eps,
                sensitivity=sens or 1.0,
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def transform_numeric(
    series: pd.Series,
    epsilon: float,
    lower: float,
    upper: float,
    preserve_type: bool = True,
) -> pd.Series:
    """Transform numeric column with Laplace mechanism.

    Convenience function for quick transformations.

    Args:
        series: Numeric column data.
        epsilon: Privacy parameter.
        lower: Lower bound of data.
        upper: Upper bound of data.
        preserve_type: If True, round to integers for integer columns.

    Returns:
        Transformed pandas Series.
    """
    transformer = ColumnTransformer()
    result = transformer.transform_numeric_bounded(
        series=series,
        epsilon=epsilon,
        lower=lower,
        upper=upper,
        preserve_type=preserve_type,
    )
    return result.data


def transform_categorical(
    series: pd.Series,
    epsilon: float,
    sensitivity: float = 1.0,
) -> pd.Series:
    """Transform categorical column with exponential mechanism.

    Convenience function for quick transformations.

    Args:
        series: Categorical column data.
        epsilon: Privacy parameter.
        sensitivity: Sensitivity of the utility function.

    Returns:
        Transformed pandas Series.
    """
    transformer = ColumnTransformer()
    result = transformer.transform_categorical(
        series=series,
        epsilon=epsilon,
        sensitivity=sensitivity,
    )
    return result.data


def transform_date(
    series: pd.Series,
    epsilon: float,
    lower: Optional[pd.Timestamp] = None,
    upper: Optional[pd.Timestamp] = None,
    unit: str = "D",
) -> pd.Series:
    """Transform date column via epoch conversion.

    Convenience function for quick transformations.

    Args:
        series: Datetime column data.
        epsilon: Privacy parameter.
        lower: Lower bound date.
        upper: Upper bound date.
        unit: Time unit for noise - 'D' for days, 's' for seconds.

    Returns:
        Transformed pandas Series.
    """
    transformer = ColumnTransformer()
    result = transformer.transform_date(
        series=series,
        epsilon=epsilon,
        lower=lower,
        upper=upper,
        unit=unit,
    )
    return result.data


# =============================================================================
# Dataset Transformer - Protection Modes
# =============================================================================


class ProtectionMode(Enum):
    """Mode for how a column should be handled during transformation.

    Attributes:
        PROTECT: Apply differential privacy transformation.
        PASSTHROUGH: Keep column unchanged (no DP applied).
        EXCLUDE: Remove column from output entirely.
    """

    PROTECT = "protect"
    PASSTHROUGH = "passthrough"
    EXCLUDE = "exclude"


@dataclass
class DatasetColumnConfig:
    """Configuration for a single column in dataset transformation.

    Attributes:
        mode: How to handle this column (protect, passthrough, exclude).
        epsilon: Privacy parameter (only used if mode is PROTECT).
        lower: Lower bound for bounded numeric data.
        upper: Upper bound for bounded numeric data.
        delta: Delta parameter for Gaussian mechanism.
        sensitivity: User-specified sensitivity.
        preserve_type: Whether to preserve integer types.
    """

    mode: ProtectionMode = ProtectionMode.PROTECT
    epsilon: Optional[float] = None  # None means use global epsilon
    lower: Optional[float] = None
    upper: Optional[float] = None
    delta: Optional[float] = None
    sensitivity: Optional[float] = None
    preserve_type: bool = True


@dataclass
class DatasetConfig:
    """Configuration for transforming an entire dataset.

    Attributes:
        global_epsilon: Default epsilon for all protected columns.
        global_delta: Default delta for Gaussian mechanisms.
        column_configs: Per-column configuration overrides.
        default_mode: Default protection mode for unconfigured columns.
    """

    global_epsilon: float = 1.0
    global_delta: float = DEFAULT_DELTA
    column_configs: Dict[str, DatasetColumnConfig] = field(default_factory=dict)
    default_mode: ProtectionMode = ProtectionMode.PROTECT

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.global_epsilon < EPSILON_MIN or self.global_epsilon > EPSILON_MAX:
            raise ValueError(
                f"Global epsilon must be in [{EPSILON_MIN}, {EPSILON_MAX}], "
                f"got {self.global_epsilon}"
            )
        if self.global_delta < DELTA_MIN or self.global_delta > DELTA_MAX:
            raise ValueError(
                f"Global delta must be in [{DELTA_MIN}, {DELTA_MAX}], "
                f"got {self.global_delta}"
            )

    def get_column_config(self, column: str) -> DatasetColumnConfig:
        """Get configuration for a specific column.

        Args:
            column: Column name.

        Returns:
            Column configuration (from column_configs or default).
        """
        if column in self.column_configs:
            return self.column_configs[column]
        return DatasetColumnConfig(mode=self.default_mode)

    def get_epsilon(self, column: str) -> float:
        """Get epsilon for a specific column.

        Args:
            column: Column name.

        Returns:
            Epsilon value (column-specific or global).
        """
        config = self.get_column_config(column)
        if config.epsilon is not None:
            return config.epsilon
        return self.global_epsilon

    def set_column_mode(
        self,
        column: str,
        mode: ProtectionMode,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Set the protection mode for a column.

        Args:
            column: Column name.
            mode: Protection mode.
            epsilon: Optional column-specific epsilon.
            **kwargs: Additional column config parameters.
        """
        self.column_configs[column] = DatasetColumnConfig(
            mode=mode,
            epsilon=epsilon,
            **kwargs,
        )

    def protect_columns(
        self, columns: List[str], epsilon: Optional[float] = None
    ) -> None:
        """Set multiple columns to PROTECT mode.

        Args:
            columns: List of column names.
            epsilon: Optional epsilon for all these columns.
        """
        for col in columns:
            self.set_column_mode(col, ProtectionMode.PROTECT, epsilon=epsilon)

    def passthrough_columns(self, columns: List[str]) -> None:
        """Set multiple columns to PASSTHROUGH mode.

        Args:
            columns: List of column names.
        """
        for col in columns:
            self.set_column_mode(col, ProtectionMode.PASSTHROUGH)

    def exclude_columns(self, columns: List[str]) -> None:
        """Set multiple columns to EXCLUDE mode.

        Args:
            columns: List of column names.
        """
        for col in columns:
            self.set_column_mode(col, ProtectionMode.EXCLUDE)


@dataclass
class ColumnTransformSummary:
    """Summary of transformation for a single column.

    Attributes:
        column_name: Name of the column.
        mode: Protection mode applied.
        mechanism_type: DP mechanism used (if protected).
        epsilon: Epsilon used (if protected).
        delta: Delta used (if Gaussian).
        null_count: Number of null values.
        error: Error message if transformation failed.
    """

    column_name: str
    mode: ProtectionMode
    mechanism_type: Optional[MechanismType] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    null_count: int = 0
    error: Optional[str] = None


@dataclass
class DatasetTransformResult:
    """Result of transforming an entire dataset.

    Attributes:
        data: Transformed DataFrame.
        column_summaries: Per-column transformation summaries.
        total_epsilon: Total privacy budget consumed.
        total_delta: Total delta consumed.
        protected_columns: List of columns that were protected.
        passthrough_columns: List of columns passed through unchanged.
        excluded_columns: List of columns excluded from output.
        row_count: Number of rows in the dataset.
    """

    data: pd.DataFrame
    column_summaries: Dict[str, ColumnTransformSummary]
    total_epsilon: float
    total_delta: Optional[float]
    protected_columns: List[str]
    passthrough_columns: List[str]
    excluded_columns: List[str]
    row_count: int

    @property
    def column_count(self) -> int:
        """Get number of columns in output."""
        return len(self.data.columns)

    @property
    def protection_rate(self) -> float:
        """Get fraction of output columns that were protected."""
        if self.column_count == 0:
            return 0.0
        return len(self.protected_columns) / self.column_count


# Type alias for progress callback
ProgressCallback = Optional[Any]  # Callable[[str, int, int], None]


class DatasetTransformer:
    """Transform entire datasets with differential privacy.

    This class processes all columns in a DataFrame according to a
    configuration that specifies how each column should be handled:
    - PROTECT: Apply DP transformation
    - PASSTHROUGH: Keep unchanged
    - EXCLUDE: Remove from output

    Example:
        >>> config = DatasetConfig(global_epsilon=1.0)
        >>> config.passthrough_columns(['id', 'timestamp'])
        >>> config.exclude_columns(['ssn', 'email'])
        >>> transformer = DatasetTransformer()
        >>> result = transformer.transform(df, config)
        >>> protected_df = result.data

    Attributes:
        column_transformer: Internal ColumnTransformer instance.
    """

    def __init__(self) -> None:
        """Initialize the dataset transformer."""
        self._column_transformer = ColumnTransformer()

    @property
    def column_transformer(self) -> ColumnTransformer:
        """Get the internal column transformer."""
        return self._column_transformer

    def transform(
        self,
        df: pd.DataFrame,
        config: Optional[DatasetConfig] = None,
        progress_callback: ProgressCallback = None,
    ) -> DatasetTransformResult:
        """Transform a dataset with differential privacy.

        Args:
            df: Input DataFrame to transform.
            config: Dataset configuration. If None, uses defaults.
            progress_callback: Optional callback for progress updates.
                Called as callback(column_name, current_index, total_columns).

        Returns:
            DatasetTransformResult with transformed data and metadata.
        """
        if config is None:
            config = DatasetConfig()

        # Categorize columns by mode
        protected_cols: List[str] = []
        passthrough_cols: List[str] = []
        excluded_cols: List[str] = []

        for col in df.columns:
            col_config = config.get_column_config(col)
            if col_config.mode == ProtectionMode.PROTECT:
                protected_cols.append(col)
            elif col_config.mode == ProtectionMode.PASSTHROUGH:
                passthrough_cols.append(col)
            else:  # EXCLUDE
                excluded_cols.append(col)

        # Build output DataFrame
        output_columns = protected_cols + passthrough_cols
        total_columns = len(output_columns)
        result_data: Dict[str, pd.Series] = {}
        column_summaries: Dict[str, ColumnTransformSummary] = {}
        total_epsilon = 0.0
        total_delta: Optional[float] = None

        # Process each column
        for idx, col in enumerate(output_columns):
            # Progress callback
            if progress_callback is not None:
                progress_callback(col, idx, total_columns)

            col_config = config.get_column_config(col)

            if col_config.mode == ProtectionMode.PASSTHROUGH:
                # Passthrough: copy unchanged
                result_data[col] = df[col].copy()
                column_summaries[col] = ColumnTransformSummary(
                    column_name=col,
                    mode=ProtectionMode.PASSTHROUGH,
                    null_count=int(df[col].isna().sum()),
                )
            else:
                # Protect: apply DP transformation
                epsilon = config.get_epsilon(col)
                try:
                    transform_result = self._transform_column(
                        series=df[col],
                        epsilon=epsilon,
                        col_config=col_config,
                        global_delta=config.global_delta,
                    )
                    result_data[col] = transform_result.data
                    column_summaries[col] = ColumnTransformSummary(
                        column_name=col,
                        mode=ProtectionMode.PROTECT,
                        mechanism_type=transform_result.mechanism_type,
                        epsilon=transform_result.privacy_usage.epsilon,
                        delta=transform_result.privacy_usage.delta,
                        null_count=transform_result.null_count,
                    )
                    # Accumulate privacy budget
                    total_epsilon += transform_result.privacy_usage.epsilon
                    if transform_result.privacy_usage.delta is not None:
                        if total_delta is None:
                            total_delta = 0.0
                        total_delta += transform_result.privacy_usage.delta
                except Exception as e:
                    # On error, passthrough with error noted
                    result_data[col] = df[col].copy()
                    column_summaries[col] = ColumnTransformSummary(
                        column_name=col,
                        mode=ProtectionMode.PROTECT,
                        null_count=int(df[col].isna().sum()),
                        error=str(e),
                    )

        # Add summaries for excluded columns
        for col in excluded_cols:
            column_summaries[col] = ColumnTransformSummary(
                column_name=col,
                mode=ProtectionMode.EXCLUDE,
            )

        # Final progress callback
        if progress_callback is not None:
            progress_callback("complete", total_columns, total_columns)

        # Build output DataFrame preserving column order
        output_df = pd.DataFrame(result_data)

        return DatasetTransformResult(
            data=output_df,
            column_summaries=column_summaries,
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            protected_columns=protected_cols,
            passthrough_columns=passthrough_cols,
            excluded_columns=excluded_cols,
            row_count=len(df),
        )

    def _transform_column(
        self,
        series: pd.Series,
        epsilon: float,
        col_config: DatasetColumnConfig,
        global_delta: float,
    ) -> TransformResult:
        """Transform a single column with DP.

        Args:
            series: Column data.
            epsilon: Epsilon to use.
            col_config: Column configuration.
            global_delta: Global delta for Gaussian.

        Returns:
            TransformResult from column transformation.
        """
        # Build ColumnConfig from DatasetColumnConfig
        column_config = ColumnConfig(
            epsilon=epsilon,
            lower=col_config.lower,
            upper=col_config.upper,
            delta=col_config.delta or global_delta,
            sensitivity=col_config.sensitivity,
            preserve_type=col_config.preserve_type,
        )

        return self._column_transformer.transform(
            series=series,
            config=column_config,
        )

    def transform_with_budget(
        self,
        df: pd.DataFrame,
        total_epsilon: float,
        columns_to_protect: Optional[List[str]] = None,
        columns_to_exclude: Optional[List[str]] = None,
        progress_callback: ProgressCallback = None,
    ) -> DatasetTransformResult:
        """Transform dataset with automatic epsilon allocation.

        Automatically divides the total epsilon budget equally among
        all protected columns.

        Args:
            df: Input DataFrame.
            total_epsilon: Total privacy budget to spend.
            columns_to_protect: Columns to protect (None = all except excluded).
            columns_to_exclude: Columns to exclude from output.
            progress_callback: Optional progress callback.

        Returns:
            DatasetTransformResult with transformed data.
        """
        columns_to_exclude = columns_to_exclude or []

        if columns_to_protect is None:
            # Protect all columns except excluded
            columns_to_protect = [c for c in df.columns if c not in columns_to_exclude]

        # Calculate per-column epsilon
        n_protected = len(columns_to_protect)
        if n_protected == 0:
            per_column_epsilon = total_epsilon
        else:
            per_column_epsilon = total_epsilon / n_protected

        # Clamp to valid range
        per_column_epsilon = max(EPSILON_MIN, min(EPSILON_MAX, per_column_epsilon))

        # Build config
        config = DatasetConfig(global_epsilon=per_column_epsilon)
        config.protect_columns(columns_to_protect)
        config.exclude_columns(columns_to_exclude)

        # Set non-protected, non-excluded columns to passthrough
        for col in df.columns:
            if col not in columns_to_protect and col not in columns_to_exclude:
                config.set_column_mode(col, ProtectionMode.PASSTHROUGH)

        return self.transform(df, config, progress_callback)


# =============================================================================
# Dataset Convenience Functions
# =============================================================================


def transform_dataset(
    df: pd.DataFrame,
    epsilon: float = 1.0,
    columns_to_protect: Optional[List[str]] = None,
    columns_to_exclude: Optional[List[str]] = None,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Transform a dataset with differential privacy.

    Convenience function for quick dataset transformations.

    Args:
        df: Input DataFrame.
        epsilon: Global privacy parameter.
        columns_to_protect: Columns to protect (None = all except excluded).
        columns_to_exclude: Columns to exclude from output.
        progress_callback: Optional progress callback.

    Returns:
        Transformed DataFrame.
    """
    transformer = DatasetTransformer()
    result = transformer.transform_with_budget(
        df=df,
        total_epsilon=epsilon,
        columns_to_protect=columns_to_protect,
        columns_to_exclude=columns_to_exclude,
        progress_callback=progress_callback,
    )
    return result.data
