"""Unit tests for dp_toolkit.analysis.comparator module."""

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.analysis.comparator import (
    # Enums
    ComparisonType,
    # Data classes
    NumericDivergence,
    NumericComparison,
    CategoricalDivergence,
    CategoricalComparison,
    DateDivergence,
    DateComparison,
    CorrelationPreservation,
    DatasetComparison,
    # Classes
    ColumnComparator,
    DatasetComparator,
    # Convenience functions
    compare_numeric_column,
    compare_categorical_column,
    compare_date_column,
    compare_datasets,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_numeric_data():
    """Simple numeric data for testing."""
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
    protected = pd.Series([1.1, 2.2, 2.9, 4.1, 5.0], name="value")
    return original, protected


@pytest.fixture
def identical_numeric_data():
    """Identical numeric data for testing."""
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
    protected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="value")
    return original, protected


@pytest.fixture
def numeric_with_nulls():
    """Numeric data with null values."""
    original = pd.Series([1.0, 2.0, None, 4.0, 5.0], name="value")
    protected = pd.Series([1.1, None, 3.0, 4.1, 5.0], name="value")
    return original, protected


@pytest.fixture
def simple_categorical_data():
    """Simple categorical data for testing."""
    original = pd.Series(["A", "B", "A", "C", "B"], name="category")
    protected = pd.Series(["A", "A", "A", "C", "B"], name="category")
    return original, protected


@pytest.fixture
def categorical_with_drift():
    """Categorical data with significant drift."""
    original = pd.Series(["A", "A", "A", "B", "B"], name="category")
    protected = pd.Series(["C", "C", "C", "D", "D"], name="category")
    return original, protected


@pytest.fixture
def simple_date_data():
    """Simple date data for testing."""
    original = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), name="date"
    )
    protected = pd.Series(
        pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]), name="date"
    )
    return original, protected


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 10000, n),
            "score": np.random.uniform(0, 100, n),
            "gender": np.random.choice(["M", "F"], n),
            "category": np.random.choice(["A", "B", "C"], n),
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


@pytest.fixture
def protected_dataset(sample_dataset):
    """Protected version of sample dataset with noise."""
    np.random.seed(123)
    df = sample_dataset.copy()
    # Add noise to numeric columns
    df["age"] = df["age"] + np.random.randint(-2, 3, len(df))
    df["income"] = df["income"] + np.random.normal(0, 1000, len(df))
    df["score"] = df["score"] + np.random.uniform(-5, 5, len(df))
    # Keep categorical unchanged for now
    return df


# =============================================================================
# Tests: Numeric Comparison
# =============================================================================


class TestNumericComparison:
    """Tests for numeric column comparison."""

    def test_basic_comparison(self, simple_numeric_data):
        """Test basic numeric comparison."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.column_name == "value"
        assert result.comparison_type == ComparisonType.NUMERIC
        assert result.count == 5
        assert result.null_count == 0

    def test_mae_calculation(self, simple_numeric_data):
        """Test MAE calculation accuracy."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        # MAE = mean(|0.1, 0.2, 0.1, 0.1, 0.0|) = 0.1
        expected_mae = np.mean([0.1, 0.2, 0.1, 0.1, 0.0])
        assert abs(result.divergence.mae - expected_mae) < 0.001

    def test_rmse_calculation(self, simple_numeric_data):
        """Test RMSE calculation accuracy."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        diffs = np.array([0.1, 0.2, -0.1, 0.1, 0.0])
        expected_rmse = np.sqrt(np.mean(diffs**2))
        assert abs(result.divergence.rmse - expected_rmse) < 0.001

    def test_mape_calculation(self, simple_numeric_data):
        """Test MAPE calculation."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        # MAPE should be calculated since no zeros
        assert result.divergence.mape is not None
        assert result.divergence.mape > 0

    def test_mape_with_zeros(self):
        """Test MAPE is None when zeros present."""
        original = pd.Series([0.0, 1.0, 2.0])
        protected = pd.Series([0.1, 1.1, 2.1])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mape is None

    def test_identical_data(self, identical_numeric_data):
        """Test comparison of identical data."""
        original, protected = identical_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mae == 0.0
        assert result.divergence.rmse == 0.0
        assert result.divergence.max_absolute_error == 0.0
        assert result.divergence.mean_difference == 0.0

    def test_null_handling(self, numeric_with_nulls):
        """Test proper handling of null values."""
        original, protected = numeric_with_nulls
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        # Only 3 rows have both non-null values
        assert result.count == 3
        assert result.null_count == 2

    def test_all_null_column(self):
        """Test comparison when all values are null."""
        original = pd.Series([None, None, None])
        protected = pd.Series([None, None, None])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.count == 0
        assert result.null_count == 3
        assert result.divergence.mae == 0.0

    def test_mean_comparison(self, simple_numeric_data):
        """Test mean statistics comparison."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mean_original == 3.0
        assert abs(result.divergence.mean_protected - 3.06) < 0.001

    def test_std_comparison(self, simple_numeric_data):
        """Test standard deviation comparison."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        expected_std_orig = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        assert abs(result.divergence.std_original - expected_std_orig) < 0.001

    def test_percentile_comparison(self, simple_numeric_data):
        """Test percentile comparison."""
        original, protected = simple_numeric_data
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert "p50" in result.percentile_comparison
        assert len(result.percentile_comparison["p50"]) == 3  # (orig, prot, diff)

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        original = pd.Series([1.0, 2.0, 3.0])
        protected = pd.Series([1.0, 2.0])
        comparator = ColumnComparator()

        with pytest.raises(ValueError, match="same length"):
            comparator.compare_numeric(original, protected, "value")

    def test_large_differences(self):
        """Test with large differences."""
        original = pd.Series([100.0, 200.0, 300.0])
        protected = pd.Series([0.0, 0.0, 0.0])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mae == 200.0
        assert result.divergence.max_absolute_error == 300.0


# =============================================================================
# Tests: Categorical Comparison
# =============================================================================


class TestCategoricalComparison:
    """Tests for categorical column comparison."""

    def test_basic_comparison(self, simple_categorical_data):
        """Test basic categorical comparison."""
        original, protected = simple_categorical_data
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.column_name == "category"
        assert result.comparison_type == ComparisonType.CATEGORICAL
        assert result.count == 5

    def test_category_drift(self, simple_categorical_data):
        """Test category drift calculation."""
        original, protected = simple_categorical_data
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        # 1 out of 5 values changed (B -> A at position 1)
        assert result.divergence.category_drift == 0.2

    def test_full_drift(self, categorical_with_drift):
        """Test 100% category drift."""
        original, protected = categorical_with_drift
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.category_drift == 1.0

    def test_no_drift(self):
        """Test zero category drift."""
        original = pd.Series(["A", "B", "C"])
        protected = pd.Series(["A", "B", "C"])
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.category_drift == 0.0

    def test_cardinality_comparison(self, simple_categorical_data):
        """Test cardinality comparison."""
        original, protected = simple_categorical_data
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.cardinality_original == 3  # A, B, C
        assert result.divergence.cardinality_protected == 3  # A, B, C

    def test_mode_comparison(self, simple_categorical_data):
        """Test mode comparison."""
        original, protected = simple_categorical_data
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        # Original mode: A (2) or B (2)
        # Protected mode: A (3)
        assert result.divergence.mode_protected == "A"

    def test_mode_preserved(self):
        """Test mode preservation flag."""
        original = pd.Series(["A", "A", "A", "B"])
        protected = pd.Series(["A", "A", "B", "B"])
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.mode_original == "A"
        assert result.divergence.mode_preserved

    def test_new_categories(self, categorical_with_drift):
        """Test detection of new categories."""
        original, protected = categorical_with_drift
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert set(result.divergence.new_categories) == {"C", "D"}

    def test_missing_categories(self, categorical_with_drift):
        """Test detection of missing categories."""
        original, protected = categorical_with_drift
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert set(result.divergence.missing_categories) == {"A", "B"}

    def test_frequency_comparison(self, simple_categorical_data):
        """Test frequency comparison output."""
        original, protected = simple_categorical_data
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert "A" in result.frequency_comparison
        orig_freq, prot_freq = result.frequency_comparison["A"]
        assert orig_freq == 0.4  # 2/5
        assert prot_freq == 0.6  # 3/5

    def test_null_handling(self):
        """Test categorical comparison with nulls."""
        original = pd.Series(["A", "B", None, "C"])
        protected = pd.Series(["A", None, "B", "C"])
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        # Only 2 rows have both non-null values
        assert result.count == 2
        assert result.null_count == 2

    def test_all_null_column(self):
        """Test all-null categorical column."""
        original = pd.Series([None, None, None])
        protected = pd.Series([None, None, None])
        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.count == 0
        assert result.divergence.category_drift == 0.0


# =============================================================================
# Tests: Date Comparison
# =============================================================================


class TestDateComparison:
    """Tests for date column comparison."""

    def test_basic_comparison(self, simple_date_data):
        """Test basic date comparison."""
        original, protected = simple_date_data
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        assert result.column_name == "date"
        assert result.comparison_type == ComparisonType.DATE
        assert result.count == 3

    def test_mae_days(self, simple_date_data):
        """Test MAE in days calculation."""
        original, protected = simple_date_data
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        # Each date shifted by 1 day
        assert result.divergence.mae_days == 1.0

    def test_rmse_days(self, simple_date_data):
        """Test RMSE in days calculation."""
        original, protected = simple_date_data
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        assert result.divergence.rmse_days == 1.0

    def test_date_range_comparison(self, simple_date_data):
        """Test date range comparison."""
        original, protected = simple_date_data
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        assert result.divergence.range_original_days == 2
        assert result.divergence.range_protected_days == 2
        assert result.divergence.range_difference_days == 0

    def test_identical_dates(self):
        """Test comparison of identical dates."""
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
        comparator = ColumnComparator()
        result = comparator.compare_date(dates, dates.copy(), "date")

        assert result.divergence.mae_days == 0.0
        assert result.divergence.rmse_days == 0.0

    def test_string_date_conversion(self):
        """Test comparison with string dates."""
        original = pd.Series(["2020-01-01", "2020-01-02"])
        protected = pd.Series(["2020-01-02", "2020-01-03"])
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        assert result.divergence.mae_days == 1.0

    def test_null_date_handling(self):
        """Test date comparison with nulls."""
        original = pd.Series(
            pd.to_datetime(["2020-01-01", None, "2020-01-03"]), name="date"
        )
        protected = pd.Series(
            pd.to_datetime([None, "2020-01-02", "2020-01-04"]), name="date"
        )
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        # Only 1 row has both non-null
        assert result.count == 1
        assert result.null_count == 2

    def test_large_date_difference(self):
        """Test large date differences."""
        original = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
        protected = pd.Series(pd.to_datetime(["2021-01-01", "2021-01-02"]))
        comparator = ColumnComparator()
        result = comparator.compare_date(original, protected, "date")

        assert result.divergence.mae_days == 366.0  # 2020 was a leap year


# =============================================================================
# Tests: Correlation Preservation
# =============================================================================


class TestCorrelationPreservation:
    """Tests for correlation preservation analysis."""

    def test_perfect_preservation(self):
        """Test perfect correlation preservation."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        comparator = DatasetComparator()
        result = comparator.compare(df, df.copy(), numeric_columns=["a", "b"])

        assert result.correlation_preservation is not None
        assert result.correlation_preservation.preservation_rate == 1.0
        assert result.correlation_preservation.mae == 0.0

    def test_correlation_difference(self):
        """Test correlation difference calculation."""
        original = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        protected = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})

        comparator = DatasetComparator()
        result = comparator.compare(
            original, protected, numeric_columns=["a", "b"]
        )

        # Original correlation is -1, protected is +1
        assert result.correlation_preservation is not None
        assert result.correlation_preservation.max_absolute_error == 2.0

    def test_correlation_threshold(self):
        """Test correlation preservation threshold."""
        np.random.seed(42)
        original = pd.DataFrame(
            {"a": np.random.randn(100), "b": np.random.randn(100)}
        )
        protected = original.copy()
        protected["a"] = protected["a"] + np.random.randn(100) * 0.05

        comparator = DatasetComparator(correlation_threshold=0.2)
        result = comparator.compare(
            original, protected, numeric_columns=["a", "b"]
        )

        assert result.correlation_preservation is not None
        # With small noise, correlation should be preserved
        assert result.correlation_preservation.preservation_rate > 0.5

    def test_multiple_columns(self, sample_dataset, protected_dataset):
        """Test correlation preservation with multiple columns."""
        comparator = DatasetComparator()
        result = comparator.compare(
            sample_dataset,
            protected_dataset,
            numeric_columns=["age", "income", "score"],
        )

        assert result.correlation_preservation is not None
        # 3 columns = 3 correlation pairs
        assert result.correlation_preservation.total_count == 3

    def test_single_column_no_correlations(self):
        """Test that single column produces no correlations."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        comparator = DatasetComparator()
        result = comparator.compare(df, df.copy(), numeric_columns=["a"])

        assert result.correlation_preservation is None


# =============================================================================
# Tests: Dataset Comparison
# =============================================================================


class TestDatasetComparison:
    """Tests for full dataset comparison."""

    def test_basic_comparison(self, sample_dataset, protected_dataset):
        """Test basic dataset comparison."""
        comparator = DatasetComparator()
        result = comparator.compare(sample_dataset, protected_dataset)

        assert result.row_count == len(sample_dataset)
        assert result.column_count == len(sample_dataset.columns)

    def test_auto_type_detection(self, sample_dataset, protected_dataset):
        """Test automatic column type detection."""
        comparator = DatasetComparator()
        result = comparator.compare(sample_dataset, protected_dataset)

        # Should detect 3 numeric, 2 categorical, 1 date
        assert len(result.numeric_comparisons) == 3
        assert len(result.categorical_comparisons) == 2
        assert len(result.date_comparisons) == 1

    def test_explicit_column_types(self, sample_dataset, protected_dataset):
        """Test explicit column type specification."""
        comparator = DatasetComparator()
        result = comparator.compare(
            sample_dataset,
            protected_dataset,
            numeric_columns=["age", "income"],
            categorical_columns=["gender"],
            date_columns=["date"],
        )

        assert len(result.numeric_comparisons) == 2
        assert len(result.categorical_comparisons) == 1
        assert len(result.date_comparisons) == 1

    def test_overall_metrics(self, sample_dataset, protected_dataset):
        """Test overall metric calculation."""
        comparator = DatasetComparator()
        result = comparator.compare(sample_dataset, protected_dataset)

        assert result.overall_numeric_mae is not None
        assert result.overall_numeric_rmse is not None
        assert result.overall_numeric_mae >= 0
        assert result.overall_numeric_rmse >= 0

    def test_get_comparison_method(self, sample_dataset, protected_dataset):
        """Test get_comparison method."""
        comparator = DatasetComparator()
        result = comparator.compare(sample_dataset, protected_dataset)

        age_comp = result.get_comparison("age")
        assert age_comp is not None
        assert age_comp.column_name == "age"

        nonexistent = result.get_comparison("nonexistent")
        assert nonexistent is None

    def test_to_summary_dict(self, sample_dataset, protected_dataset):
        """Test summary dictionary generation."""
        comparator = DatasetComparator()
        result = comparator.compare(sample_dataset, protected_dataset)

        summary = result.to_summary_dict()
        assert "row_count" in summary
        assert "column_count" in summary
        assert "numeric_columns" in summary
        assert "overall_numeric_mae" in summary

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2]})

        comparator = DatasetComparator()
        with pytest.raises(ValueError, match="same shape"):
            comparator.compare(df1, df2)

    def test_column_mismatch_raises(self):
        """Test that column mismatch raises error."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})

        comparator = DatasetComparator()
        with pytest.raises(ValueError, match="same columns"):
            comparator.compare(df1, df2)

    def test_empty_dataframes(self):
        """Test comparison of empty dataframes."""
        df = pd.DataFrame({"a": [], "b": []})
        comparator = DatasetComparator()
        result = comparator.compare(df, df.copy())

        assert result.row_count == 0

    def test_skip_correlations(self, sample_dataset, protected_dataset):
        """Test skipping correlation analysis."""
        comparator = DatasetComparator()
        result = comparator.compare(
            sample_dataset,
            protected_dataset,
            compare_correlations=False,
        )

        assert result.correlation_preservation is None


# =============================================================================
# Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compare_numeric_column(self, simple_numeric_data):
        """Test compare_numeric_column convenience function."""
        original, protected = simple_numeric_data
        result = compare_numeric_column(original, protected)

        assert result.column_name == "value"
        assert isinstance(result, NumericComparison)

    def test_compare_categorical_column(self, simple_categorical_data):
        """Test compare_categorical_column convenience function."""
        original, protected = simple_categorical_data
        result = compare_categorical_column(original, protected)

        assert result.column_name == "category"
        assert isinstance(result, CategoricalComparison)

    def test_compare_date_column(self, simple_date_data):
        """Test compare_date_column convenience function."""
        original, protected = simple_date_data
        result = compare_date_column(original, protected)

        assert result.column_name == "date"
        assert isinstance(result, DateComparison)

    def test_compare_datasets_function(self, sample_dataset, protected_dataset):
        """Test compare_datasets convenience function."""
        result = compare_datasets(sample_dataset, protected_dataset)

        assert isinstance(result, DatasetComparison)
        assert result.row_count == len(sample_dataset)

    def test_calculate_mae_function(self, simple_numeric_data):
        """Test calculate_mae function."""
        original, protected = simple_numeric_data
        mae = calculate_mae(original, protected)

        expected = np.mean([0.1, 0.2, 0.1, 0.1, 0.0])
        assert abs(mae - expected) < 0.001

    def test_calculate_rmse_function(self, simple_numeric_data):
        """Test calculate_rmse function."""
        original, protected = simple_numeric_data
        rmse = calculate_rmse(original, protected)

        diffs = np.array([0.1, 0.2, -0.1, 0.1, 0.0])
        expected = np.sqrt(np.mean(diffs**2))
        assert abs(rmse - expected) < 0.001

    def test_calculate_mape_function(self, simple_numeric_data):
        """Test calculate_mape function."""
        original, protected = simple_numeric_data
        mape = calculate_mape(original, protected)

        assert mape is not None
        assert mape > 0

    def test_calculate_mape_with_zeros(self):
        """Test calculate_mape returns None with zeros."""
        original = pd.Series([0.0, 1.0, 2.0])
        protected = pd.Series([0.1, 1.1, 2.1])
        mape = calculate_mape(original, protected)

        assert mape is None

    def test_calculate_mae_with_nulls(self):
        """Test calculate_mae handles nulls."""
        original = pd.Series([1.0, None, 3.0])
        protected = pd.Series([1.1, 2.0, None])
        mae = calculate_mae(original, protected)

        # Only first value can be compared (both non-null)
        assert abs(mae - 0.1) < 0.001

    def test_column_name_fallback(self):
        """Test column name fallback when not specified."""
        original = pd.Series([1.0, 2.0, 3.0])  # No name
        protected = pd.Series([1.1, 2.1, 3.1])
        result = compare_numeric_column(original, protected)

        assert result.column_name == "unknown"


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_value(self):
        """Test comparison of single-value series."""
        original = pd.Series([5.0])
        protected = pd.Series([5.5])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mae == 0.5
        assert result.divergence.std_original == 0.0

    def test_all_same_values(self):
        """Test comparison when all values are the same."""
        original = pd.Series([5.0, 5.0, 5.0])
        protected = pd.Series([6.0, 6.0, 6.0])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mae == 1.0
        assert result.divergence.std_original == 0.0
        assert result.divergence.std_protected == 0.0

    def test_negative_values(self):
        """Test comparison with negative values."""
        original = pd.Series([-10.0, -5.0, 0.0, 5.0, 10.0])
        protected = pd.Series([-9.0, -4.0, 1.0, 6.0, 11.0])
        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.divergence.mae == 1.0
        assert result.divergence.mean_original == 0.0
        assert result.divergence.mean_protected == 1.0

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        np.random.seed(42)
        n = 100000
        original = pd.Series(np.random.randn(n))
        protected = pd.Series(np.random.randn(n))

        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        assert result.count == n
        # Should complete quickly (< 1 second)

    def test_high_cardinality_categorical(self):
        """Test high cardinality categorical comparison."""
        n = 10000
        original = pd.Series([f"cat_{i}" for i in range(n)])
        protected = pd.Series([f"cat_{i+1}" for i in range(n)])

        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.cardinality_original == n
        assert result.divergence.category_drift > 0

    def test_unicode_categories(self):
        """Test categorical comparison with unicode."""
        original = pd.Series(["日本語", "中文", "한국어"])
        protected = pd.Series(["日本語", "中文", "English"])

        comparator = ColumnComparator()
        result = comparator.compare_categorical(original, protected, "category")

        assert result.divergence.category_drift == pytest.approx(1 / 3)
        assert "한국어" in result.divergence.missing_categories
        assert "English" in result.divergence.new_categories

    def test_mixed_null_patterns(self):
        """Test various null patterns."""
        original = pd.Series([None, 1.0, 2.0, None, 4.0])
        protected = pd.Series([0.0, None, 2.5, 3.0, None])

        comparator = ColumnComparator()
        result = comparator.compare_numeric(original, protected, "value")

        # Only position 2 has both non-null
        assert result.count == 1
        assert result.null_count == 4
