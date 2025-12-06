"""Unit tests for unified column profiler."""

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.profiler import (
    ColumnProfile,
    ColumnProfiler,
    ProfileType,
    NumericProfile,
    CategoricalProfile,
    DateProfile,
    profile_column,
    profile_columns,
)


class TestProfileType:
    """Tests for ProfileType enum."""

    def test_profile_types(self):
        """Test ProfileType enum values."""
        assert ProfileType.NUMERIC.value == "numeric"
        assert ProfileType.CATEGORICAL.value == "categorical"
        assert ProfileType.DATE.value == "date"


class TestColumnProfile:
    """Tests for ColumnProfile dataclass."""

    def test_numeric_profile_wrapper(self):
        """Test wrapping a NumericProfile."""
        numeric_profile = NumericProfile(
            count=100, null_count=5, mean=50.0, std=10.0,
            min=0.0, max=100.0, median=50.0, q1=25.0, q3=75.0,
            iqr=50.0, p5=5.0, p10=10.0, p90=90.0, p95=95.0, p99=99.0,
            skewness=0.0, kurtosis=0.0, outlier_count=2
        )
        profile = ColumnProfile(
            column_name="test_col",
            profile_type=ProfileType.NUMERIC,
            profile=numeric_profile
        )

        assert profile.column_name == "test_col"
        assert profile.is_numeric is True
        assert profile.is_categorical is False
        assert profile.is_date is False
        assert profile.count == 100
        assert profile.null_count == 5

    def test_categorical_profile_wrapper(self):
        """Test wrapping a CategoricalProfile."""
        cat_profile = CategoricalProfile(
            count=100, null_count=10, cardinality=5,
            mode="A", mode_count=40, mode_frequency=0.4
        )
        profile = ColumnProfile(
            column_name="category_col",
            profile_type=ProfileType.CATEGORICAL,
            profile=cat_profile
        )

        assert profile.is_categorical is True
        assert profile.is_numeric is False
        assert profile.count == 100

    def test_date_profile_wrapper(self):
        """Test wrapping a DateProfile."""
        date_profile = DateProfile(
            count=50, null_count=5, min_date=None, max_date=None,
            range_days=None, cardinality=50, mode=None, mode_count=1
        )
        profile = ColumnProfile(
            column_name="date_col",
            profile_type=ProfileType.DATE,
            profile=date_profile
        )

        assert profile.is_date is True
        assert profile.is_numeric is False
        assert profile.count == 50


class TestColumnProfiler:
    """Tests for ColumnProfiler class."""

    def test_auto_detect_numeric(self):
        """Test automatic detection of numeric columns."""
        series = pd.Series([1, 2, 3, 4, 5], name="numbers")
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.profile_type == ProfileType.NUMERIC
        assert profile.column_name == "numbers"
        assert isinstance(profile.profile, NumericProfile)

    def test_auto_detect_float(self):
        """Test automatic detection of float columns."""
        series = pd.Series([1.5, 2.5, 3.5, 4.5], name="floats")
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.profile_type == ProfileType.NUMERIC
        assert isinstance(profile.profile, NumericProfile)

    def test_auto_detect_categorical(self):
        """Test automatic detection of categorical columns."""
        series = pd.Series(["A", "B", "C", "A", "B"], name="categories")
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.profile_type == ProfileType.CATEGORICAL
        assert isinstance(profile.profile, CategoricalProfile)

    def test_auto_detect_date(self):
        """Test automatic detection of date columns."""
        series = pd.Series(
            pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"]),
            name="dates"
        )
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.profile_type == ProfileType.DATE
        assert isinstance(profile.profile, DateProfile)

    def test_boolean_as_categorical(self):
        """Test that boolean columns are treated as categorical."""
        series = pd.Series([True, False, True, False], name="flags")
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.profile_type == ProfileType.CATEGORICAL
        assert isinstance(profile.profile, CategoricalProfile)

    def test_custom_column_name(self):
        """Test specifying custom column name."""
        series = pd.Series([1, 2, 3])
        profiler = ColumnProfiler()
        profile = profiler.profile(series, column_name="custom_name")

        assert profile.column_name == "custom_name"

    def test_unnamed_series(self):
        """Test profiling series without a name."""
        series = pd.Series([1, 2, 3])
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        assert profile.column_name == "unnamed"

    def test_profiler_configuration(self):
        """Test profiler configuration options."""
        profiler = ColumnProfiler(iqr_multiplier=3.0, top_n=5)

        # Verify configuration was passed to sub-profilers
        assert profiler.numeric_profiler.iqr_multiplier == 3.0
        assert profiler.categorical_profiler.top_n == 5


class TestColumnProfilerDataframe:
    """Tests for DataFrame-level column profiling."""

    def test_profile_all_columns(self):
        """Test profiling all columns in a DataFrame."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "C", "B"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01",
                                   "2023-04-01", "2023-05-01"]),
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert len(profiles) == 3
        assert "numeric" in profiles
        assert "category" in profiles
        assert "date" in profiles

        assert profiles["numeric"].profile_type == ProfileType.NUMERIC
        assert profiles["category"].profile_type == ProfileType.CATEGORICAL
        assert profiles["date"].profile_type == ProfileType.DATE

    def test_profile_specific_columns(self):
        """Test profiling specific columns."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "C", "B"],
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df, columns=["numeric"])

        assert len(profiles) == 1
        assert "numeric" in profiles
        assert "category" not in profiles

    def test_profile_nonexistent_column(self):
        """Test that nonexistent columns are skipped."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df, columns=["numeric", "missing"])

        assert len(profiles) == 1
        assert "numeric" in profiles
        assert "missing" not in profiles


class TestUnifiedConvenienceFunctions:
    """Tests for convenience functions."""

    def test_profile_column_function(self):
        """Test profile_column convenience function."""
        series = pd.Series([1, 2, 3, 4, 5], name="test")
        profile = profile_column(series)

        assert isinstance(profile, ColumnProfile)
        assert profile.profile_type == ProfileType.NUMERIC

    def test_profile_column_with_name(self):
        """Test profile_column with custom name."""
        series = pd.Series(["A", "B", "C"])
        profile = profile_column(series, column_name="categories")

        assert profile.column_name == "categories"
        assert profile.profile_type == ProfileType.CATEGORICAL

    def test_profile_columns_function(self):
        """Test profile_columns convenience function."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "category": ["A", "B", "C"],
        })
        profiles = profile_columns(df)

        assert len(profiles) == 2
        assert profiles["numeric"].is_numeric
        assert profiles["category"].is_categorical


class TestUnifiedMixedTypes:
    """Tests for mixed type handling."""

    def test_mixed_dataframe(self):
        """Test profiling DataFrame with all column types."""
        df = pd.DataFrame({
            "integers": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            "strings": ["A", "B", "C", "D", "E"],
            "booleans": [True, False, True, False, True],
            "dates": pd.to_datetime([
                "2023-01-01", "2023-02-01", "2023-03-01",
                "2023-04-01", "2023-05-01"
            ]),
            "categories": pd.Categorical(["X", "Y", "X", "Z", "Y"]),
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert profiles["integers"].profile_type == ProfileType.NUMERIC
        assert profiles["floats"].profile_type == ProfileType.NUMERIC
        assert profiles["strings"].profile_type == ProfileType.CATEGORICAL
        assert profiles["booleans"].profile_type == ProfileType.CATEGORICAL
        assert profiles["dates"].profile_type == ProfileType.DATE
        assert profiles["categories"].profile_type == ProfileType.CATEGORICAL

    def test_nulls_in_mixed_types(self):
        """Test handling nulls in mixed type columns."""
        df = pd.DataFrame({
            "numeric": [1, 2, None, 4, 5],
            "category": ["A", None, "C", None, "E"],
            "date": pd.to_datetime([
                "2023-01-01", pd.NaT, "2023-03-01", pd.NaT, "2023-05-01"
            ]),
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert profiles["numeric"].null_count == 1
        assert profiles["category"].null_count == 2
        assert profiles["date"].null_count == 2


class TestUnifiedEdgeCases:
    """Edge case tests for unified profiling."""

    def test_empty_dataframe(self):
        """Test profiling empty DataFrame."""
        df = pd.DataFrame()
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert len(profiles) == 0

    def test_dataframe_all_empty_columns(self):
        """Test profiling DataFrame with empty columns."""
        df = pd.DataFrame({
            "numeric": pd.Series([], dtype=float),
            "category": pd.Series([], dtype=object),
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert profiles["numeric"].count == 0
        assert profiles["category"].count == 0

    def test_single_row_dataframe(self):
        """Test profiling single-row DataFrame."""
        df = pd.DataFrame({
            "numeric": [42],
            "category": ["X"],
        })
        profiler = ColumnProfiler()
        profiles = profiler.profile_dataframe(df)

        assert profiles["numeric"].count == 1
        assert profiles["category"].count == 1

    def test_profile_object_column_with_mixed_types(self):
        """Test profiling object column with mixed Python types."""
        series = pd.Series([1, "A", 2.5, True, None], name="mixed")
        profiler = ColumnProfiler()
        profile = profiler.profile(series)

        # Object dtype should be treated as categorical
        assert profile.profile_type == ProfileType.CATEGORICAL
