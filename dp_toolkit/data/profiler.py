"""Statistical profiling module for DPtoolkit.

Provides comprehensive statistical analysis for dataset columns,
supporting numeric, categorical, and date types.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


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
