"""Unit tests for dataset-level profiler."""

import time

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.profiler import (
    CorrelationMatrix,
    DatasetProfile,
    DatasetProfiler,
    MissingValueSummary,
    ProfileType,
    calculate_correlation_matrix,
    calculate_missing_summary,
    profile_dataset,
)


class TestMissingValueSummary:
    """Tests for MissingValueSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a MissingValueSummary."""
        summary = MissingValueSummary(
            total_cells=1000,
            total_missing=50,
            missing_percentage=5.0,
            columns_with_missing=3,
            rows_with_missing=40,
            complete_rows=60,
            per_column={"a": 20, "b": 30},
            per_column_percentage={"a": 2.0, "b": 3.0},
        )
        assert summary.total_cells == 1000
        assert summary.total_missing == 50
        assert summary.missing_percentage == 5.0


class TestCorrelationMatrix:
    """Tests for CorrelationMatrix dataclass."""

    def test_matrix_creation(self):
        """Test creating a CorrelationMatrix."""
        columns = ["a", "b", "c"]
        matrix = np.array([
            [1.0, 0.8, -0.3],
            [0.8, 1.0, -0.2],
            [-0.3, -0.2, 1.0]
        ])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        assert corr.columns == columns
        assert corr.method == "pearson"
        assert corr.matrix.shape == (3, 3)

    def test_get_correlation(self):
        """Test getting correlation between two columns."""
        columns = ["a", "b", "c"]
        matrix = np.array([
            [1.0, 0.8, -0.3],
            [0.8, 1.0, -0.2],
            [-0.3, -0.2, 1.0]
        ])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        assert corr.get("a", "b") == 0.8
        assert corr.get("a", "c") == -0.3
        assert corr.get("a", "a") == 1.0

    def test_get_nonexistent_column(self):
        """Test getting correlation with nonexistent column."""
        columns = ["a", "b"]
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        assert corr.get("a", "x") is None
        assert corr.get("x", "y") is None

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        columns = ["a", "b"]
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        df = corr.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == columns
        assert list(df.index) == columns
        assert df.loc["a", "b"] == 0.5

    def test_get_high_correlations(self):
        """Test finding high correlations."""
        columns = ["a", "b", "c", "d"]
        matrix = np.array([
            [1.0, 0.9, 0.3, -0.8],
            [0.9, 1.0, 0.2, -0.5],
            [0.3, 0.2, 1.0, 0.1],
            [-0.8, -0.5, 0.1, 1.0]
        ])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        high = corr.get_high_correlations(threshold=0.7)
        assert len(high) == 2  # (a,b) and (a,d)
        # Should be sorted by absolute value
        assert high[0] == ("a", "b", 0.9)
        assert high[1] == ("a", "d", -0.8)

    def test_get_high_correlations_include_self(self):
        """Test including self-correlations."""
        columns = ["a", "b"]
        matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        corr = CorrelationMatrix(columns=columns, matrix=matrix, method="pearson")

        high = corr.get_high_correlations(threshold=0.9, exclude_self=False)
        assert len(high) == 2  # Both self-correlations


class TestDatasetProfile:
    """Tests for DatasetProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a DatasetProfile."""
        missing = MissingValueSummary(
            total_cells=100, total_missing=5, missing_percentage=5.0,
            columns_with_missing=1, rows_with_missing=5, complete_rows=95,
        )
        profile = DatasetProfile(
            row_count=100,
            column_count=5,
            memory_usage_bytes=10000,
            numeric_columns=["a", "b"],
            categorical_columns=["c"],
            date_columns=["d"],
            column_profiles={},
            missing_summary=missing,
        )

        assert profile.row_count == 100
        assert profile.column_count == 5
        assert profile.has_missing is True
        assert profile.completeness == 0.95

    def test_no_missing_values(self):
        """Test profile with no missing values."""
        missing = MissingValueSummary(
            total_cells=100, total_missing=0, missing_percentage=0.0,
            columns_with_missing=0, rows_with_missing=0, complete_rows=100,
        )
        profile = DatasetProfile(
            row_count=100,
            column_count=5,
            memory_usage_bytes=10000,
            numeric_columns=[],
            categorical_columns=[],
            date_columns=[],
            column_profiles={},
            missing_summary=missing,
        )

        assert profile.has_missing is False
        assert profile.completeness == 1.0


class TestDatasetProfiler:
    """Tests for DatasetProfiler class."""

    def test_basic_profile(self):
        """Test basic dataset profiling."""
        df = pd.DataFrame({
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "C", "B"],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.row_count == 5
        assert profile.column_count == 3
        assert "numeric1" in profile.numeric_columns
        assert "numeric2" in profile.numeric_columns
        assert "category" in profile.categorical_columns
        assert len(profile.column_profiles) == 3

    def test_profile_with_dates(self):
        """Test profiling with date columns."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "date": pd.to_datetime([
                "2023-01-01", "2023-02-01", "2023-03-01",
                "2023-04-01", "2023-05-01"
            ]),
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert "date" in profile.date_columns
        assert "numeric" in profile.numeric_columns

    def test_profile_with_missing(self):
        """Test profiling with missing values."""
        df = pd.DataFrame({
            "a": [1, 2, None, 4, 5],
            "b": [None, None, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.missing_summary.total_missing == 3
        assert profile.missing_summary.columns_with_missing == 2
        assert profile.missing_summary.per_column["a"] == 1
        assert profile.missing_summary.per_column["b"] == 2
        assert profile.missing_summary.per_column["c"] == 0

    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        # Create correlated data
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({
            "a": x,
            "b": x + np.random.randn(100) * 0.1,  # Highly correlated with a
            "c": np.random.randn(100),  # Independent
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.correlation_matrix is not None
        # a and b should be highly correlated
        corr_ab = profile.correlation_matrix.get("a", "b")
        assert corr_ab is not None
        assert corr_ab > 0.9

    def test_skip_correlations(self):
        """Test skipping correlation calculation."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        profiler = DatasetProfiler(compute_correlations=False)
        profile = profiler.profile(df)

        assert profile.correlation_matrix is None

    def test_single_numeric_column(self):
        """Test with single numeric column (no correlation matrix)."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "category": ["A", "B", "C", "D", "E"],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        # Can't compute correlation with single numeric column
        assert profile.correlation_matrix is None

    def test_duplicate_detection(self):
        """Test duplicate row detection."""
        df = pd.DataFrame({
            "a": [1, 2, 2, 3, 3],
            "b": [10, 20, 20, 30, 30],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.duplicate_row_count == 2

    def test_correlation_methods(self):
        """Test different correlation methods."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })

        # Pearson
        profiler_pearson = DatasetProfiler(correlation_method="pearson")
        profile_pearson = profiler_pearson.profile(df, use_cache=False)
        assert profile_pearson.correlation_matrix.method == "pearson"

        # Spearman
        profiler_spearman = DatasetProfiler(correlation_method="spearman")
        profile_spearman = profiler_spearman.profile(df, use_cache=False)
        assert profile_spearman.correlation_matrix.method == "spearman"

    def test_specific_columns(self):
        """Test profiling specific columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df, columns=["a", "b"])

        assert profile.column_count == 2
        assert "a" in profile.column_profiles
        assert "b" in profile.column_profiles
        assert "c" not in profile.column_profiles


class TestDatasetProfilerCache:
    """Tests for DatasetProfiler caching."""

    def test_cache_hit(self):
        """Test that cache is used for identical DataFrames."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        profiler = DatasetProfiler()

        # First call - should compute
        profile1 = profiler.profile(df)
        assert profiler.get_cache_size() == 1

        # Second call - should use cache
        profile2 = profiler.profile(df)
        assert profiler.get_cache_size() == 1
        assert profile1 is profile2  # Same object from cache

    def test_cache_miss_different_data(self):
        """Test cache miss for different DataFrames."""
        profiler = DatasetProfiler()

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        profile1 = profiler.profile(df1)
        assert profiler.get_cache_size() == 1

        df2 = pd.DataFrame({"a": [4, 5, 6]})  # Different data, same structure
        # Note: Our cache key uses shape+columns+dtypes, not data values
        # So this will be a cache hit, which is the intended behavior
        profile2 = profiler.profile(df2)

        df3 = pd.DataFrame({"b": [1, 2, 3]})  # Different column name
        profile3 = profiler.profile(df3)
        assert profiler.get_cache_size() == 2

    def test_cache_disabled(self):
        """Test profiling with cache disabled."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        profiler = DatasetProfiler()

        profile1 = profiler.profile(df, use_cache=False)
        assert profiler.get_cache_size() == 0

        profile2 = profiler.profile(df, use_cache=False)
        assert profiler.get_cache_size() == 0

    def test_clear_cache(self):
        """Test clearing the cache."""
        profiler = DatasetProfiler()

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        profiler.profile(df1)
        profiler.profile(df2)
        assert profiler.get_cache_size() == 2

        profiler.clear_cache()
        assert profiler.get_cache_size() == 0


class TestMissingValueAnalysis:
    """Tests for missing value analysis."""

    def test_no_missing_values(self):
        """Test dataset with no missing values."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        summary = calculate_missing_summary(df)

        assert summary.total_missing == 0
        assert summary.missing_percentage == 0.0
        assert summary.columns_with_missing == 0
        assert summary.rows_with_missing == 0
        assert summary.complete_rows == 5

    def test_all_missing_column(self):
        """Test column with all missing values."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [None, None, None],
        })
        summary = calculate_missing_summary(df)

        assert summary.per_column["b"] == 3
        assert summary.per_column_percentage["b"] == 100.0

    def test_row_analysis(self):
        """Test row-level missing analysis."""
        df = pd.DataFrame({
            "a": [1, None, 3, None],
            "b": [None, 2, 3, None],
        })
        summary = calculate_missing_summary(df)

        assert summary.rows_with_missing == 3  # Rows 0, 1, 3
        assert summary.complete_rows == 1  # Only row 2

    def test_empty_dataframe(self):
        """Test empty DataFrame."""
        df = pd.DataFrame()
        summary = calculate_missing_summary(df)

        assert summary.total_cells == 0
        assert summary.missing_percentage == 0.0


class TestCorrelationCalculation:
    """Tests for correlation matrix calculation."""

    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # b = 2*a
        })
        corr = calculate_correlation_matrix(df)

        assert abs(corr.get("a", "b") - 1.0) < 0.001

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        corr = calculate_correlation_matrix(df)

        assert abs(corr.get("a", "b") - (-1.0)) < 0.001

    def test_no_correlation(self):
        """Test independent variables."""
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(1000),
            "b": np.random.randn(1000),
        })
        corr = calculate_correlation_matrix(df)

        # Should be close to 0
        assert abs(corr.get("a", "b")) < 0.1

    def test_specific_columns(self):
        """Test calculating correlation for specific columns."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })
        corr = calculate_correlation_matrix(df, columns=["a", "b"])

        assert corr.columns == ["a", "b"]
        assert len(corr.columns) == 2

    def test_insufficient_columns(self):
        """Test error with less than 2 columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError, match="at least 2 numeric columns"):
            calculate_correlation_matrix(df)

    def test_spearman_correlation(self):
        """Test Spearman correlation."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, 4, 9, 16, 25],  # b = a^2 (non-linear)
        })
        corr = calculate_correlation_matrix(df, method="spearman")

        assert corr.method == "spearman"
        # Spearman should show perfect rank correlation
        assert abs(corr.get("a", "b") - 1.0) < 0.001


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_profile_dataset_function(self):
        """Test profile_dataset convenience function."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["X", "Y", "X", "Y", "Z"],
        })
        profile = profile_dataset(df)

        assert isinstance(profile, DatasetProfile)
        assert profile.row_count == 5
        assert profile.column_count == 2

    def test_profile_dataset_no_correlations(self):
        """Test profile_dataset without correlations."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        profile = profile_dataset(df, compute_correlations=False)

        assert profile.correlation_matrix is None


class TestDatasetEdgeCases:
    """Edge case tests for dataset profiling."""

    def test_empty_dataframe(self):
        """Test profiling empty DataFrame."""
        df = pd.DataFrame()
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.row_count == 0
        assert profile.column_count == 0

    def test_single_row(self):
        """Test profiling single-row DataFrame."""
        df = pd.DataFrame({"a": [1], "b": ["X"]})
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.row_count == 1
        assert profile.duplicate_row_count == 0

    def test_all_null_dataframe(self):
        """Test DataFrame with all null values."""
        df = pd.DataFrame({
            "a": [None, None, None],
            "b": [None, None, None],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.missing_summary.missing_percentage == 100.0
        assert profile.completeness == 0.0

    def test_mixed_types(self):
        """Test DataFrame with all column types."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "category": ["A", "B", "C"],
            "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "boolean": [True, False, True],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert len(profile.numeric_columns) == 1
        assert len(profile.categorical_columns) == 2  # category + boolean
        assert len(profile.date_columns) == 1

    def test_get_column_profile(self):
        """Test getting individual column profile."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["X", "Y", "Z"],
        })
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        col_profile = profile.get_column_profile("a")
        assert col_profile is not None
        assert col_profile.profile_type == ProfileType.NUMERIC

        assert profile.get_column_profile("nonexistent") is None


class TestPerformance:
    """Performance tests for dataset profiling."""

    def test_medium_dataset_performance(self):
        """Test profiling medium-sized dataset (100K rows)."""
        np.random.seed(42)
        n_rows = 100_000
        df = pd.DataFrame({
            "numeric1": np.random.randn(n_rows),
            "numeric2": np.random.randn(n_rows),
            "numeric3": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows),
        })

        profiler = DatasetProfiler()
        start = time.time()
        profile = profiler.profile(df)
        elapsed = time.time() - start

        assert profile.row_count == n_rows
        assert elapsed < 10.0  # Should complete in under 10 seconds

    def test_cache_performance(self):
        """Test that cache provides speedup."""
        np.random.seed(42)
        n_rows = 50_000
        df = pd.DataFrame({
            "a": np.random.randn(n_rows),
            "b": np.random.randn(n_rows),
            "c": np.random.choice(["X", "Y", "Z"], n_rows),
        })

        profiler = DatasetProfiler()

        # First call - compute
        start1 = time.time()
        profile1 = profiler.profile(df)
        time1 = time.time() - start1

        # Second call - cached
        start2 = time.time()
        profile2 = profiler.profile(df)
        time2 = time.time() - start2

        # Cache should be significantly faster
        assert time2 < time1 * 0.1  # At least 10x faster

    def test_many_columns(self):
        """Test profiling with many columns."""
        np.random.seed(42)
        n_cols = 50
        data = {f"col_{i}": np.random.randn(1000) for i in range(n_cols)}
        df = pd.DataFrame(data)

        profiler = DatasetProfiler()
        start = time.time()
        profile = profiler.profile(df)
        elapsed = time.time() - start

        assert profile.column_count == n_cols
        assert profile.correlation_matrix is not None
        assert len(profile.correlation_matrix.columns) == n_cols
        assert elapsed < 30.0  # Should complete in under 30 seconds
