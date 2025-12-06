"""Statistical profiling module for DPtoolkit.

Provides comprehensive statistical analysis for dataset columns,
supporting numeric, categorical, and date types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class ProfileType(Enum):
    """Type of column profile."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"


@dataclass
class NumericProfile:
    """Statistical profile for a numeric column.

    Attributes:
        count: Number of non-null values.
        null_count: Number of null/missing values.
        mean: Arithmetic mean.
        std: Standard deviation.
        min: Minimum value.
        max: Maximum value.
        median: Median (50th percentile).
        q1: First quartile (25th percentile).
        q3: Third quartile (75th percentile).
        iqr: Interquartile range (Q3 - Q1).
        p5: 5th percentile.
        p10: 10th percentile.
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        skewness: Measure of distribution asymmetry.
        kurtosis: Measure of distribution tailedness.
        outlier_count: Number of outliers detected (IQR method).
        outlier_bounds: Tuple of (lower_bound, upper_bound) for outliers.
    """

    count: int
    null_count: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    median: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    iqr: Optional[float]
    p5: Optional[float]
    p10: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    outlier_count: int
    outlier_bounds: tuple[Optional[float], Optional[float]] = field(
        default=(None, None)
    )


class NumericProfiler:
    """Calculate comprehensive statistics for numeric columns.

    Handles null values correctly and provides outlier detection
    using the IQR method.
    """

    # IQR multiplier for outlier detection (1.5 is standard)
    IQR_MULTIPLIER = 1.5

    def __init__(self, iqr_multiplier: float = 1.5) -> None:
        """Initialize the profiler.

        Args:
            iqr_multiplier: Multiplier for IQR in outlier detection.
                Default is 1.5 (standard). Use 3.0 for extreme outliers only.
        """
        self.iqr_multiplier = iqr_multiplier

    def profile(self, series: pd.Series) -> NumericProfile:
        """Calculate comprehensive statistics for a numeric series.

        Args:
            series: A pandas Series containing numeric data.

        Returns:
            NumericProfile with all calculated statistics.

        Raises:
            TypeError: If series is not numeric.
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Series must be numeric, got {series.dtype}")

        # Get counts
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        count = len(non_null)

        # Handle empty or all-null case
        if count == 0:
            return NumericProfile(
                count=0,
                null_count=null_count,
                mean=None,
                std=None,
                min=None,
                max=None,
                median=None,
                q1=None,
                q3=None,
                iqr=None,
                p5=None,
                p10=None,
                p90=None,
                p95=None,
                p99=None,
                skewness=None,
                kurtosis=None,
                outlier_count=0,
                outlier_bounds=(None, None),
            )

        # Basic statistics
        mean_val = float(non_null.mean())
        std_val = float(non_null.std()) if count > 1 else 0.0
        min_val = float(non_null.min())
        max_val = float(non_null.max())

        # Percentiles
        percentiles = non_null.quantile(
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        p5 = float(percentiles[0.05])
        p10 = float(percentiles[0.10])
        q1 = float(percentiles[0.25])
        median_val = float(percentiles[0.50])
        q3 = float(percentiles[0.75])
        p90 = float(percentiles[0.90])
        p95 = float(percentiles[0.95])
        p99 = float(percentiles[0.99])

        # IQR
        iqr_val = q3 - q1

        # Skewness and kurtosis (need at least 3 values for meaningful result)
        # Also skip if all values are the same (std == 0)
        if count >= 3 and std_val > 0:
            skewness_val = float(stats.skew(non_null, nan_policy="omit"))
            kurtosis_val = float(stats.kurtosis(non_null, nan_policy="omit"))
        else:
            skewness_val = None
            kurtosis_val = None

        # Outlier detection using IQR method
        lower_bound = q1 - self.iqr_multiplier * iqr_val
        upper_bound = q3 + self.iqr_multiplier * iqr_val
        outlier_count = int(((non_null < lower_bound) | (non_null > upper_bound)).sum())

        return NumericProfile(
            count=count,
            null_count=null_count,
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            median=median_val,
            q1=q1,
            q3=q3,
            iqr=iqr_val,
            p5=p5,
            p10=p10,
            p90=p90,
            p95=p95,
            p99=p99,
            skewness=skewness_val,
            kurtosis=kurtosis_val,
            outlier_count=outlier_count,
            outlier_bounds=(lower_bound, upper_bound),
        )

    def profile_dataframe(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> dict[str, NumericProfile]:
        """Profile multiple numeric columns in a DataFrame.

        Args:
            df: The DataFrame to profile.
            columns: List of column names to profile. If None, profiles
                all numeric columns.

        Returns:
            Dictionary mapping column names to their NumericProfile.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        profiles = {}
        for col in columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    profiles[col] = self.profile(df[col])

        return profiles

    def get_outliers(
        self, series: pd.Series, profile: Optional[NumericProfile] = None
    ) -> pd.Series:
        """Get the outlier values from a series.

        Args:
            series: The numeric series to check.
            profile: Pre-computed profile. If None, will compute.

        Returns:
            Series containing only the outlier values.
        """
        if profile is None:
            profile = self.profile(series)

        if profile.outlier_bounds[0] is None:
            return pd.Series([], dtype=series.dtype)

        lower, upper = profile.outlier_bounds
        non_null = series.dropna()
        return non_null[(non_null < lower) | (non_null > upper)]

    def get_outlier_indices(
        self, series: pd.Series, profile: Optional[NumericProfile] = None
    ) -> pd.Index:
        """Get the indices of outlier values.

        Args:
            series: The numeric series to check.
            profile: Pre-computed profile. If None, will compute.

        Returns:
            Index of outlier positions.
        """
        outliers = self.get_outliers(series, profile)
        return outliers.index


def profile_numeric(series: pd.Series) -> NumericProfile:
    """Convenience function to profile a numeric series.

    Args:
        series: A pandas Series containing numeric data.

    Returns:
        NumericProfile with all calculated statistics.
    """
    profiler = NumericProfiler()
    return profiler.profile(series)


def profile_numeric_columns(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> dict[str, NumericProfile]:
    """Convenience function to profile numeric columns in a DataFrame.

    Args:
        df: The DataFrame to profile.
        columns: List of column names to profile. If None, profiles
            all numeric columns.

    Returns:
        Dictionary mapping column names to their NumericProfile.
    """
    profiler = NumericProfiler()
    return profiler.profile_dataframe(df, columns)


# =============================================================================
# Categorical Profiling
# =============================================================================


@dataclass
class CategoricalProfile:
    """Statistical profile for a categorical column.

    Attributes:
        count: Number of non-null values.
        null_count: Number of null/missing values.
        cardinality: Number of unique values.
        mode: Most frequent value.
        mode_count: Count of the mode value.
        mode_frequency: Frequency of mode (mode_count / count).
        top_values: Top N most frequent values with counts.
        entropy: Shannon entropy of the distribution (bits).
        is_unique: True if all values are unique (potential identifier).
    """

    count: int
    null_count: int
    cardinality: int
    mode: Optional[str]
    mode_count: int
    mode_frequency: Optional[float]
    top_values: list[tuple[str, int]] = field(default_factory=list)
    entropy: Optional[float] = None
    is_unique: bool = False


class CategoricalProfiler:
    """Calculate comprehensive statistics for categorical columns.

    Handles null values correctly and provides frequency distribution
    and entropy calculations.
    """

    # Default number of top values to include
    DEFAULT_TOP_N = 10

    def __init__(self, top_n: int = 10) -> None:
        """Initialize the profiler.

        Args:
            top_n: Number of top frequent values to include in profile.
        """
        self.top_n = top_n

    def profile(self, series: pd.Series) -> CategoricalProfile:
        """Calculate comprehensive statistics for a categorical series.

        Args:
            series: A pandas Series containing categorical data.

        Returns:
            CategoricalProfile with all calculated statistics.
        """
        # Get counts
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        count = len(non_null)

        # Handle empty or all-null case
        if count == 0:
            return CategoricalProfile(
                count=0,
                null_count=null_count,
                cardinality=0,
                mode=None,
                mode_count=0,
                mode_frequency=None,
                top_values=[],
                entropy=None,
                is_unique=False,
            )

        # Cardinality (number of unique values)
        value_counts = non_null.value_counts()
        cardinality = len(value_counts)

        # Mode (most frequent value)
        mode_val = str(value_counts.index[0])
        mode_count = int(value_counts.iloc[0])
        mode_frequency = mode_count / count

        # Top N values
        top_values = [
            (str(val), int(cnt))
            for val, cnt in value_counts.head(self.top_n).items()
        ]

        # Shannon entropy (in bits)
        probabilities = value_counts.values / count
        entropy_val = float(stats.entropy(probabilities, base=2))

        # Check if all values are unique (potential identifier column)
        is_unique = cardinality == count

        return CategoricalProfile(
            count=count,
            null_count=null_count,
            cardinality=cardinality,
            mode=mode_val,
            mode_count=mode_count,
            mode_frequency=mode_frequency,
            top_values=top_values,
            entropy=entropy_val,
            is_unique=is_unique,
        )

    def profile_dataframe(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> dict[str, CategoricalProfile]:
        """Profile multiple categorical columns in a DataFrame.

        Args:
            df: The DataFrame to profile.
            columns: List of column names to profile. If None, profiles
                all object/string/category columns.

        Returns:
            Dictionary mapping column names to their CategoricalProfile.
        """
        if columns is None:
            columns = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

        profiles = {}
        for col in columns:
            if col in df.columns:
                profiles[col] = self.profile(df[col])

        return profiles

    def get_frequency_distribution(
        self, series: pd.Series, normalize: bool = False
    ) -> pd.Series:
        """Get the full frequency distribution of a series.

        Args:
            series: The categorical series.
            normalize: If True, return proportions instead of counts.

        Returns:
            Series with value counts (or proportions).
        """
        non_null = series.dropna()
        return non_null.value_counts(normalize=normalize)


def profile_categorical(series: pd.Series) -> CategoricalProfile:
    """Convenience function to profile a categorical series.

    Args:
        series: A pandas Series containing categorical data.

    Returns:
        CategoricalProfile with all calculated statistics.
    """
    profiler = CategoricalProfiler()
    return profiler.profile(series)


def profile_categorical_columns(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> dict[str, CategoricalProfile]:
    """Convenience function to profile categorical columns in a DataFrame.

    Args:
        df: The DataFrame to profile.
        columns: List of column names to profile. If None, profiles
            all object/string/category columns.

    Returns:
        Dictionary mapping column names to their CategoricalProfile.
    """
    profiler = CategoricalProfiler()
    return profiler.profile_dataframe(df, columns)


# =============================================================================
# Date Profiling
# =============================================================================


@dataclass
class DateProfile:
    """Statistical profile for a date/datetime column.

    Attributes:
        count: Number of non-null values.
        null_count: Number of null/missing values.
        min_date: Earliest date.
        max_date: Latest date.
        range_days: Range in days between min and max.
        cardinality: Number of unique dates.
        mode: Most frequent date.
        mode_count: Count of the mode date.
        weekday_distribution: Counts per day of week (Mon=0, Sun=6).
        month_distribution: Counts per month (Jan=1, Dec=12).
        year_distribution: Counts per year.
    """

    count: int
    null_count: int
    min_date: Optional[datetime]
    max_date: Optional[datetime]
    range_days: Optional[int]
    cardinality: int
    mode: Optional[datetime]
    mode_count: int
    weekday_distribution: dict[int, int] = field(default_factory=dict)
    month_distribution: dict[int, int] = field(default_factory=dict)
    year_distribution: dict[int, int] = field(default_factory=dict)


class DateProfiler:
    """Calculate comprehensive statistics for date/datetime columns.

    Handles null values correctly and provides temporal distribution
    analysis.
    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        pass

    def profile(self, series: pd.Series) -> DateProfile:
        """Calculate comprehensive statistics for a datetime series.

        Args:
            series: A pandas Series containing datetime data.

        Returns:
            DateProfile with all calculated statistics.

        Raises:
            TypeError: If series cannot be converted to datetime.
        """
        # Try to convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors="coerce")
            except Exception as e:
                raise TypeError(f"Cannot convert series to datetime: {e}")

        # Get counts
        null_count = int(series.isna().sum())
        non_null = series.dropna()
        count = len(non_null)

        # Handle empty or all-null case
        if count == 0:
            return DateProfile(
                count=0,
                null_count=null_count,
                min_date=None,
                max_date=None,
                range_days=None,
                cardinality=0,
                mode=None,
                mode_count=0,
                weekday_distribution={},
                month_distribution={},
                year_distribution={},
            )

        # Min/max
        min_date = non_null.min().to_pydatetime()
        max_date = non_null.max().to_pydatetime()
        range_days = (max_date - min_date).days

        # Cardinality
        value_counts = non_null.value_counts()
        cardinality = len(value_counts)

        # Mode
        mode_val = value_counts.index[0].to_pydatetime()
        mode_count = int(value_counts.iloc[0])

        # Weekday distribution (Monday=0, Sunday=6)
        weekday_counts = non_null.dt.dayofweek.value_counts().sort_index()
        weekday_distribution = {int(k): int(v) for k, v in weekday_counts.items()}

        # Month distribution (January=1, December=12)
        month_counts = non_null.dt.month.value_counts().sort_index()
        month_distribution = {int(k): int(v) for k, v in month_counts.items()}

        # Year distribution
        year_counts = non_null.dt.year.value_counts().sort_index()
        year_distribution = {int(k): int(v) for k, v in year_counts.items()}

        return DateProfile(
            count=count,
            null_count=null_count,
            min_date=min_date,
            max_date=max_date,
            range_days=range_days,
            cardinality=cardinality,
            mode=mode_val,
            mode_count=mode_count,
            weekday_distribution=weekday_distribution,
            month_distribution=month_distribution,
            year_distribution=year_distribution,
        )

    def profile_dataframe(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> dict[str, DateProfile]:
        """Profile multiple datetime columns in a DataFrame.

        Args:
            df: The DataFrame to profile.
            columns: List of column names to profile. If None, profiles
                all datetime columns.

        Returns:
            Dictionary mapping column names to their DateProfile.
        """
        if columns is None:
            columns = df.select_dtypes(
                include=["datetime64", "datetime64[ns]"]
            ).columns.tolist()

        profiles = {}
        for col in columns:
            if col in df.columns:
                profiles[col] = self.profile(df[col])

        return profiles


def profile_date(series: pd.Series) -> DateProfile:
    """Convenience function to profile a datetime series.

    Args:
        series: A pandas Series containing datetime data.

    Returns:
        DateProfile with all calculated statistics.
    """
    profiler = DateProfiler()
    return profiler.profile(series)


def profile_date_columns(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> dict[str, DateProfile]:
    """Convenience function to profile datetime columns in a DataFrame.

    Args:
        df: The DataFrame to profile.
        columns: List of column names to profile. If None, profiles
            all datetime columns.

    Returns:
        Dictionary mapping column names to their DateProfile.
    """
    profiler = DateProfiler()
    return profiler.profile_dataframe(df, columns)


# =============================================================================
# Unified Column Profile
# =============================================================================


@dataclass
class ColumnProfile:
    """Unified profile for any column type.

    This is a wrapper that holds the appropriate profile type
    based on the column's data type.

    Attributes:
        column_name: Name of the column.
        profile_type: Type of profile (numeric, categorical, date).
        profile: The actual profile (NumericProfile, CategoricalProfile, or DateProfile).
    """

    column_name: str
    profile_type: ProfileType
    profile: Union[NumericProfile, CategoricalProfile, DateProfile]

    @property
    def count(self) -> int:
        """Get the non-null count from the underlying profile."""
        return self.profile.count

    @property
    def null_count(self) -> int:
        """Get the null count from the underlying profile."""
        return self.profile.null_count

    @property
    def is_numeric(self) -> bool:
        """Check if this is a numeric profile."""
        return self.profile_type == ProfileType.NUMERIC

    @property
    def is_categorical(self) -> bool:
        """Check if this is a categorical profile."""
        return self.profile_type == ProfileType.CATEGORICAL

    @property
    def is_date(self) -> bool:
        """Check if this is a date profile."""
        return self.profile_type == ProfileType.DATE


class ColumnProfiler:
    """Profile any column type automatically.

    Detects the appropriate profiler based on column dtype and
    returns a unified ColumnProfile.
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        top_n: int = 10,
    ) -> None:
        """Initialize the profiler.

        Args:
            iqr_multiplier: IQR multiplier for numeric outlier detection.
            top_n: Number of top values for categorical profiles.
        """
        self.numeric_profiler = NumericProfiler(iqr_multiplier=iqr_multiplier)
        self.categorical_profiler = CategoricalProfiler(top_n=top_n)
        self.date_profiler = DateProfiler()

    def profile(
        self, series: pd.Series, column_name: Optional[str] = None
    ) -> ColumnProfile:
        """Profile a column automatically based on its dtype.

        Args:
            series: The pandas Series to profile.
            column_name: Name for the column. If None, uses series.name.

        Returns:
            ColumnProfile with the appropriate profile type.
        """
        name = column_name or series.name or "unnamed"

        # Check for datetime first
        if pd.api.types.is_datetime64_any_dtype(series):
            date_profile = self.date_profiler.profile(series)
            return ColumnProfile(
                column_name=str(name),
                profile_type=ProfileType.DATE,
                profile=date_profile,
            )

        # Check for numeric (but not boolean)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            numeric_profile = self.numeric_profiler.profile(series)
            return ColumnProfile(
                column_name=str(name),
                profile_type=ProfileType.NUMERIC,
                profile=numeric_profile,
            )

        # Everything else is categorical
        cat_profile = self.categorical_profiler.profile(series)
        return ColumnProfile(
            column_name=str(name),
            profile_type=ProfileType.CATEGORICAL,
            profile=cat_profile,
        )

    def profile_dataframe(
        self, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> dict[str, ColumnProfile]:
        """Profile multiple columns in a DataFrame.

        Args:
            df: The DataFrame to profile.
            columns: List of column names to profile. If None, profiles
                all columns.

        Returns:
            Dictionary mapping column names to their ColumnProfile.
        """
        if columns is None:
            columns = df.columns.tolist()

        profiles = {}
        for col in columns:
            if col in df.columns:
                profiles[col] = self.profile(df[col], column_name=col)

        return profiles


def profile_column(
    series: pd.Series, column_name: Optional[str] = None
) -> ColumnProfile:
    """Convenience function to profile any column.

    Args:
        series: A pandas Series to profile.
        column_name: Name for the column. If None, uses series.name.

    Returns:
        ColumnProfile with the appropriate profile type.
    """
    profiler = ColumnProfiler()
    return profiler.profile(series, column_name=column_name)


def profile_columns(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> dict[str, ColumnProfile]:
    """Convenience function to profile columns in a DataFrame.

    Args:
        df: The DataFrame to profile.
        columns: List of column names to profile. If None, profiles
            all columns.

    Returns:
        Dictionary mapping column names to their ColumnProfile.
    """
    profiler = ColumnProfiler()
    return profiler.profile_dataframe(df, columns)
