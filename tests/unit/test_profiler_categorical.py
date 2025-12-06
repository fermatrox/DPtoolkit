"""Unit tests for categorical profiler."""

import math

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.profiler import (
    CategoricalProfile,
    CategoricalProfiler,
    profile_categorical,
    profile_categorical_columns,
)


class TestCategoricalProfile:
    """Tests for CategoricalProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a CategoricalProfile."""
        profile = CategoricalProfile(
            count=100,
            null_count=5,
            cardinality=10,
            mode="A",
            mode_count=30,
            mode_frequency=0.3,
            top_values=[("A", 30), ("B", 25)],
            entropy=2.5,
            is_unique=False,
        )
        assert profile.count == 100
        assert profile.null_count == 5
        assert profile.cardinality == 10
        assert profile.mode == "A"
        assert profile.mode_count == 30
        assert profile.mode_frequency == 0.3
        assert len(profile.top_values) == 2
        assert profile.entropy == 2.5
        assert profile.is_unique is False


class TestCategoricalProfiler:
    """Tests for CategoricalProfiler class."""

    def test_basic_profile(self):
        """Test basic categorical profiling."""
        series = pd.Series(["A", "B", "A", "C", "A", "B", "D", "A"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 8
        assert profile.null_count == 0
        assert profile.cardinality == 4
        assert profile.mode == "A"
        assert profile.mode_count == 4
        assert profile.mode_frequency == 0.5

    def test_profile_with_nulls(self):
        """Test profiling with null values."""
        series = pd.Series(["A", "B", None, "A", np.nan, "C"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 4
        assert profile.null_count == 2
        assert profile.cardinality == 3
        assert profile.mode == "A"
        assert profile.mode_count == 2

    def test_empty_series(self):
        """Test profiling an empty series."""
        series = pd.Series([], dtype=object)
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 0
        assert profile.null_count == 0
        assert profile.cardinality == 0
        assert profile.mode is None
        assert profile.entropy is None

    def test_all_null_series(self):
        """Test profiling series with all nulls."""
        series = pd.Series([None, np.nan, None])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 0
        assert profile.null_count == 3
        assert profile.cardinality == 0
        assert profile.mode is None

    def test_single_value(self):
        """Test profiling series with single unique value."""
        series = pd.Series(["X", "X", "X", "X", "X"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 1
        assert profile.mode == "X"
        assert profile.mode_count == 5
        assert profile.mode_frequency == 1.0
        # Entropy should be 0 for single value
        assert profile.entropy == 0.0
        assert profile.is_unique is False

    def test_all_unique_values(self):
        """Test profiling series with all unique values."""
        series = pd.Series(["A", "B", "C", "D", "E"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 5
        assert profile.is_unique is True
        assert profile.mode_frequency == 0.2  # 1/5

    def test_top_values_ordering(self):
        """Test that top values are ordered by frequency."""
        series = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 3 + ["D"] * 1)
        profiler = CategoricalProfiler(top_n=4)
        profile = profiler.profile(series)

        assert len(profile.top_values) == 4
        assert profile.top_values[0] == ("A", 10)
        assert profile.top_values[1] == ("B", 5)
        assert profile.top_values[2] == ("C", 3)
        assert profile.top_values[3] == ("D", 1)

    def test_top_n_limit(self):
        """Test that top_n limits the number of top values."""
        series = pd.Series(["A", "B", "C", "D", "E", "F", "G"])
        profiler = CategoricalProfiler(top_n=3)
        profile = profiler.profile(series)

        assert len(profile.top_values) == 3

    def test_entropy_uniform_distribution(self):
        """Test entropy for uniform distribution."""
        # 4 categories, each appearing once = max entropy = log2(4) = 2
        series = pd.Series(["A", "B", "C", "D"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.entropy is not None
        assert abs(profile.entropy - 2.0) < 0.001

    def test_entropy_skewed_distribution(self):
        """Test entropy for skewed distribution."""
        # Highly skewed = lower entropy
        series = pd.Series(["A"] * 100 + ["B"] * 1)
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.entropy is not None
        # Entropy should be low but > 0
        assert 0 < profile.entropy < 0.2

    def test_high_cardinality(self):
        """Test handling of high cardinality columns."""
        # 1000+ unique values
        values = [f"value_{i}" for i in range(1500)]
        series = pd.Series(values)
        profiler = CategoricalProfiler(top_n=10)
        profile = profiler.profile(series)

        assert profile.count == 1500
        assert profile.cardinality == 1500
        assert profile.is_unique is True
        assert len(profile.top_values) == 10

    def test_numeric_like_strings(self):
        """Test profiling strings that look like numbers."""
        series = pd.Series(["1", "2", "1", "3", "1", "2"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 6
        assert profile.cardinality == 3
        assert profile.mode == "1"

    def test_mixed_types_as_string(self):
        """Test profiling mixed types converted to strings."""
        series = pd.Series([1, "A", 2.5, None, "B"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        # Note: Mixed types in pandas object series
        assert profile.count == 4
        assert profile.null_count == 1

    def test_category_dtype(self):
        """Test profiling pandas Categorical dtype."""
        series = pd.Series(["A", "B", "A", "C"], dtype="category")
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 4
        assert profile.cardinality == 3
        assert profile.mode == "A"

    def test_boolean_as_categorical(self):
        """Test profiling boolean values as categorical."""
        series = pd.Series([True, False, True, True, False])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 2
        # Mode should be True (3 occurrences)
        assert profile.mode == "True"
        assert profile.mode_count == 3


class TestCategoricalProfilerDataframe:
    """Tests for DataFrame-level categorical profiling."""

    def test_profile_dataframe(self):
        """Test profiling multiple categorical columns."""
        df = pd.DataFrame({
            "category1": ["A", "B", "A", "C"],
            "category2": ["X", "Y", "X", "X"],
            "numeric": [1, 2, 3, 4],
        })
        profiler = CategoricalProfiler()
        profiles = profiler.profile_dataframe(df)

        # Should only include object columns by default
        assert "category1" in profiles
        assert "category2" in profiles
        assert "numeric" not in profiles

    def test_profile_specific_columns(self):
        """Test profiling specific columns."""
        df = pd.DataFrame({
            "category1": ["A", "B", "A", "C"],
            "category2": ["X", "Y", "X", "X"],
        })
        profiler = CategoricalProfiler()
        profiles = profiler.profile_dataframe(df, columns=["category1"])

        assert "category1" in profiles
        assert "category2" not in profiles

    def test_frequency_distribution(self):
        """Test getting frequency distribution."""
        series = pd.Series(["A", "B", "A", "C", "A", "B"])
        profiler = CategoricalProfiler()

        dist = profiler.get_frequency_distribution(series)
        assert dist["A"] == 3
        assert dist["B"] == 2
        assert dist["C"] == 1

    def test_frequency_distribution_normalized(self):
        """Test getting normalized frequency distribution."""
        series = pd.Series(["A", "B", "A", "C", "A", "B"])
        profiler = CategoricalProfiler()

        dist = profiler.get_frequency_distribution(series, normalize=True)
        assert dist["A"] == 0.5
        assert abs(dist["B"] - 1 / 3) < 0.001
        assert abs(dist["C"] - 1 / 6) < 0.001


class TestCategoricalConvenienceFunctions:
    """Tests for convenience functions."""

    def test_profile_categorical_function(self):
        """Test profile_categorical convenience function."""
        series = pd.Series(["A", "B", "A", "C"])
        profile = profile_categorical(series)

        assert isinstance(profile, CategoricalProfile)
        assert profile.count == 4
        assert profile.cardinality == 3

    def test_profile_categorical_columns_function(self):
        """Test profile_categorical_columns convenience function."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C"],
            "numeric": [1, 2, 3, 4],
        })
        profiles = profile_categorical_columns(df)

        assert "category" in profiles
        assert "numeric" not in profiles


class TestCategoricalEdgeCases:
    """Edge case tests for categorical profiling."""

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        series = pd.Series(["A", "", "A", "", "B"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 3
        assert profile.mode == "A"  # or "" if tied

    def test_whitespace_values(self):
        """Test handling of whitespace-only values."""
        series = pd.Series(["A", "  ", "A", "\t", "B"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        # Whitespace is distinct from empty string
        assert profile.cardinality >= 3

    def test_unicode_values(self):
        """Test handling of unicode values."""
        series = pd.Series(["α", "β", "α", "γ", "日本語"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 4
        assert profile.mode == "α"

    def test_very_long_strings(self):
        """Test handling of very long string values."""
        long_str = "A" * 10000
        series = pd.Series([long_str, "B", long_str])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        assert profile.cardinality == 2

    def test_special_characters(self):
        """Test handling of special characters."""
        series = pd.Series(["@#$", "A", "!@#", "A", "&*()"])
        profiler = CategoricalProfiler()
        profile = profiler.profile(series)

        assert profile.count == 5
        assert profile.cardinality == 4
        assert profile.mode == "A"
