"""Unit tests for the numeric profiler module."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dp_toolkit.data.profiler import (
    NumericProfile,
    NumericProfiler,
    profile_numeric,
    profile_numeric_columns,
)


class TestNumericProfilerBasicStats:
    """Tests for basic statistical calculations."""

    def test_count(self):
        """Test count of non-null values."""
        series = pd.Series([1, 2, 3, 4, 5])
        profile = profile_numeric(series)

        assert profile.count == 5
        assert profile.null_count == 0

    def test_mean(self):
        """Test mean calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        profile = profile_numeric(series)

        assert profile.mean == 3.0

    def test_std(self):
        """Test standard deviation calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        profile = profile_numeric(series)

        # pandas uses ddof=1 by default (sample std)
        expected_std = series.std()
        assert abs(profile.std - expected_std) < 1e-10

    def test_min_max(self):
        """Test min and max calculation."""
        series = pd.Series([10, 5, 20, 15, 3])
        profile = profile_numeric(series)

        assert profile.min == 3
        assert profile.max == 20

    def test_median(self):
        """Test median calculation."""
        series = pd.Series([1, 2, 3, 4, 5])
        profile = profile_numeric(series)

        assert profile.median == 3.0

    def test_median_even_count(self):
        """Test median with even number of values."""
        series = pd.Series([1, 2, 3, 4])
        profile = profile_numeric(series)

        assert profile.median == 2.5


class TestNumericProfilerPercentiles:
    """Tests for percentile calculations."""

    def test_quartiles(self):
        """Test Q1, Q3, and IQR calculation."""
        # Use a dataset where quartiles are clear
        series = pd.Series(range(1, 101))  # 1 to 100
        profile = profile_numeric(series)

        # For 1-100, Q1 ~ 25.75, Q3 ~ 75.25 (linear interpolation)
        assert 25 <= profile.q1 <= 26
        assert 75 <= profile.q3 <= 76
        assert profile.iqr == profile.q3 - profile.q1

    def test_percentile_accuracy(self):
        """Test that percentiles match pandas calculations."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(1000))
        profile = profile_numeric(series)

        # Compare with pandas quantile
        expected = series.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        assert abs(profile.p5 - expected[0.05]) < 1e-10
        assert abs(profile.p10 - expected[0.10]) < 1e-10
        assert abs(profile.q1 - expected[0.25]) < 1e-10
        assert abs(profile.median - expected[0.50]) < 1e-10
        assert abs(profile.q3 - expected[0.75]) < 1e-10
        assert abs(profile.p90 - expected[0.90]) < 1e-10
        assert abs(profile.p95 - expected[0.95]) < 1e-10
        assert abs(profile.p99 - expected[0.99]) < 1e-10


class TestNumericProfilerSkewnessKurtosis:
    """Tests for skewness and kurtosis calculations."""

    def test_symmetric_distribution_skewness(self):
        """Test that symmetric distribution has near-zero skewness."""
        # Symmetric data
        series = pd.Series([-2, -1, 0, 1, 2])
        profile = profile_numeric(series)

        assert abs(profile.skewness) < 0.1

    def test_right_skewed_distribution(self):
        """Test that right-skewed data has positive skewness."""
        # Right-skewed: many small values, few large values
        series = pd.Series([1, 1, 1, 1, 1, 2, 2, 3, 10, 100])
        profile = profile_numeric(series)

        assert profile.skewness > 0

    def test_left_skewed_distribution(self):
        """Test that left-skewed data has negative skewness."""
        # Left-skewed: few small values, many large values
        series = pd.Series([1, 50, 90, 95, 99, 99, 100, 100, 100, 100])
        profile = profile_numeric(series)

        assert profile.skewness < 0

    def test_kurtosis_normal_distribution(self):
        """Test kurtosis of approximately normal distribution."""
        np.random.seed(42)
        # Large sample from normal distribution
        series = pd.Series(np.random.randn(10000))
        profile = profile_numeric(series)

        # scipy.stats.kurtosis uses Fisher's definition (normal = 0)
        # Should be close to 0 for normal distribution
        assert abs(profile.kurtosis) < 0.2

    def test_skewness_kurtosis_match_scipy(self):
        """Test that skewness/kurtosis match scipy calculations."""
        np.random.seed(42)
        series = pd.Series(np.random.exponential(2, 1000))
        profile = profile_numeric(series)

        expected_skew = stats.skew(series)
        expected_kurt = stats.kurtosis(series)

        assert abs(profile.skewness - expected_skew) < 1e-10
        assert abs(profile.kurtosis - expected_kurt) < 1e-10


class TestNumericProfilerOutliers:
    """Tests for outlier detection."""

    def test_no_outliers(self):
        """Test data with no outliers."""
        series = pd.Series([1, 2, 3, 4, 5])
        profile = profile_numeric(series)

        assert profile.outlier_count == 0

    def test_outliers_detected(self):
        """Test that outliers are correctly detected."""
        # Data with clear outliers
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        profile = profile_numeric(series)

        assert profile.outlier_count >= 1

    def test_outlier_bounds(self):
        """Test that outlier bounds are correctly calculated."""
        series = pd.Series(range(1, 101))
        profile = profile_numeric(series)

        q1, q3, iqr = profile.q1, profile.q3, profile.iqr
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        assert abs(profile.outlier_bounds[0] - expected_lower) < 1e-10
        assert abs(profile.outlier_bounds[1] - expected_upper) < 1e-10

    def test_get_outliers(self):
        """Test retrieving outlier values."""
        series = pd.Series([1, 2, 3, 4, 5, 100, -50])
        profiler = NumericProfiler()
        profile = profiler.profile(series)
        outliers = profiler.get_outliers(series, profile)

        # Both 100 and -50 should be outliers
        assert 100 in outliers.values
        assert -50 in outliers.values

    def test_get_outlier_indices(self):
        """Test retrieving outlier indices."""
        series = pd.Series([1, 2, 3, 4, 5, 100], index=["a", "b", "c", "d", "e", "f"])
        profiler = NumericProfiler()
        indices = profiler.get_outlier_indices(series)

        assert "f" in indices  # 100 is the outlier at index "f"

    def test_custom_iqr_multiplier(self):
        """Test custom IQR multiplier for outlier detection."""
        series = pd.Series([1, 2, 3, 4, 5, 20])

        # With default 1.5 multiplier
        profiler_default = NumericProfiler(iqr_multiplier=1.5)
        profile_default = profiler_default.profile(series)

        # With 3.0 multiplier (more lenient)
        profiler_lenient = NumericProfiler(iqr_multiplier=3.0)
        profile_lenient = profiler_lenient.profile(series)

        # Lenient should detect fewer or equal outliers
        assert profile_lenient.outlier_count <= profile_default.outlier_count


class TestNumericProfilerNullHandling:
    """Tests for null value handling."""

    def test_nulls_excluded_from_count(self):
        """Test that nulls are excluded from count."""
        series = pd.Series([1, 2, None, 4, 5])
        profile = profile_numeric(series)

        assert profile.count == 4
        assert profile.null_count == 1

    def test_nulls_excluded_from_mean(self):
        """Test that nulls are excluded from mean calculation."""
        series = pd.Series([1, 2, None, 4, 5])
        profile = profile_numeric(series)

        # Mean of [1, 2, 4, 5] = 3.0
        assert profile.mean == 3.0

    def test_nulls_excluded_from_percentiles(self):
        """Test that nulls are excluded from percentile calculations."""
        series = pd.Series([1, None, None, None, 5])
        profile = profile_numeric(series)

        # With only [1, 5], median should be 3
        assert profile.median == 3.0

    def test_all_nulls(self):
        """Test handling of all-null series."""
        series = pd.Series([None, None, None], dtype=float)
        profile = profile_numeric(series)

        assert profile.count == 0
        assert profile.null_count == 3
        assert profile.mean is None
        assert profile.std is None
        assert profile.min is None
        assert profile.max is None
        assert profile.outlier_count == 0

    def test_mixed_nan_and_none(self):
        """Test handling of mixed NaN and None values."""
        series = pd.Series([1, np.nan, None, 4, float("nan")])
        profile = profile_numeric(series)

        assert profile.count == 2  # Only 1 and 4
        assert profile.null_count == 3


class TestNumericProfilerEdgeCases:
    """Tests for edge cases."""

    def test_single_value(self):
        """Test profiling a series with a single value."""
        series = pd.Series([42])
        profile = profile_numeric(series)

        assert profile.count == 1
        assert profile.mean == 42
        assert profile.std == 0.0  # std of single value is 0
        assert profile.min == 42
        assert profile.max == 42
        assert profile.median == 42
        assert profile.skewness is None  # Need 3+ values
        assert profile.kurtosis is None

    def test_two_values(self):
        """Test profiling a series with two values."""
        series = pd.Series([10, 20])
        profile = profile_numeric(series)

        assert profile.count == 2
        assert profile.mean == 15
        assert profile.median == 15
        assert profile.skewness is None  # Need 3+ values
        assert profile.kurtosis is None

    def test_all_same_values(self):
        """Test profiling a series where all values are identical."""
        series = pd.Series([5, 5, 5, 5, 5])
        profile = profile_numeric(series)

        assert profile.count == 5
        assert profile.mean == 5
        assert profile.std == 0.0
        assert profile.min == 5
        assert profile.max == 5
        assert profile.iqr == 0.0
        # Skewness/kurtosis are undefined when all values are the same
        assert profile.skewness is None
        assert profile.kurtosis is None

    def test_negative_values(self):
        """Test profiling with negative values."""
        series = pd.Series([-10, -5, 0, 5, 10])
        profile = profile_numeric(series)

        assert profile.mean == 0
        assert profile.min == -10
        assert profile.max == 10

    def test_float_values(self):
        """Test profiling with float values."""
        series = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        profile = profile_numeric(series)

        assert profile.mean == 3.5
        assert profile.min == 1.5
        assert profile.max == 5.5

    def test_integer_series(self):
        """Test profiling integer series."""
        series = pd.Series([1, 2, 3, 4, 5], dtype=int)
        profile = profile_numeric(series)

        assert profile.count == 5
        assert isinstance(profile.mean, float)

    def test_empty_series(self):
        """Test profiling empty series."""
        series = pd.Series([], dtype=float)
        profile = profile_numeric(series)

        assert profile.count == 0
        assert profile.null_count == 0
        assert profile.mean is None


class TestNumericProfilerDataFrame:
    """Tests for DataFrame profiling."""

    def test_profile_dataframe_all_numeric(self):
        """Test profiling all numeric columns in a DataFrame."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        profiles = profile_numeric_columns(df)

        assert len(profiles) == 3
        assert "a" in profiles
        assert "b" in profiles
        assert "c" in profiles
        assert profiles["a"].mean == 3.0
        assert profiles["b"].mean == 30.0

    def test_profile_dataframe_specific_columns(self):
        """Test profiling specific columns only."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        profiles = profile_numeric_columns(df, columns=["a", "c"])

        assert len(profiles) == 2
        assert "a" in profiles
        assert "c" in profiles
        assert "b" not in profiles

    def test_profile_dataframe_mixed_types(self):
        """Test profiling DataFrame with mixed column types."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "string": ["a", "b", "c", "d", "e"],
            "float": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        profiles = profile_numeric_columns(df)

        # Should only profile numeric columns
        assert len(profiles) == 2
        assert "numeric" in profiles
        assert "float" in profiles
        assert "string" not in profiles

    def test_profile_dataframe_skips_non_numeric(self):
        """Test that non-numeric columns in list are skipped."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "string": ["a", "b", "c", "d", "e"],
        })
        # Request both, but string should be skipped
        profiles = profile_numeric_columns(df, columns=["numeric", "string"])

        assert len(profiles) == 1
        assert "numeric" in profiles


class TestNumericProfilerTypeErrors:
    """Tests for type error handling."""

    def test_non_numeric_series_raises(self):
        """Test that non-numeric series raises TypeError."""
        series = pd.Series(["a", "b", "c"])

        with pytest.raises(TypeError, match="must be numeric"):
            profile_numeric(series)

    def test_datetime_series_raises(self):
        """Test that datetime series raises TypeError."""
        series = pd.Series(pd.date_range("2023-01-01", periods=5))

        with pytest.raises(TypeError, match="must be numeric"):
            profile_numeric(series)


class TestNumericProfilerPerformance:
    """Performance tests."""

    def test_performance_100k_rows(self):
        """Test profiling performance with 100K rows."""
        import time

        np.random.seed(42)
        series = pd.Series(np.random.randn(100000))

        start = time.time()
        profile = profile_numeric(series)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
        assert profile.count == 100000

    def test_performance_dataframe_many_columns(self):
        """Test profiling DataFrame with many columns."""
        import time

        np.random.seed(42)
        n_rows = 10000
        n_cols = 50

        df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f"col_{i}" for i in range(n_cols)]
        )

        start = time.time()
        profiles = profile_numeric_columns(df)
        elapsed = time.time() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0
        assert len(profiles) == n_cols


class TestNumericProfilerAccuracy:
    """Tests for statistical accuracy against known datasets."""

    def test_known_dataset_accuracy(self):
        """Test accuracy against a dataset with known statistics."""
        # Simple dataset: 1 to 10
        series = pd.Series(range(1, 11))
        profile = profile_numeric(series)

        # Known values
        assert profile.count == 10
        assert profile.mean == 5.5
        assert profile.min == 1
        assert profile.max == 10
        assert profile.median == 5.5

    def test_standard_normal_accuracy(self):
        """Test accuracy with standard normal distribution."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(100000))
        profile = profile_numeric(series)

        # Standard normal: mean ~ 0, std ~ 1
        assert abs(profile.mean) < 0.02  # Close to 0
        assert abs(profile.std - 1.0) < 0.02  # Close to 1

    def test_uniform_distribution_accuracy(self):
        """Test accuracy with uniform distribution."""
        np.random.seed(42)
        series = pd.Series(np.random.uniform(0, 100, 100000))
        profile = profile_numeric(series)

        # Uniform [0, 100]: mean ~ 50, min ~ 0, max ~ 100
        assert abs(profile.mean - 50) < 1
        assert profile.min < 1
        assert profile.max > 99
