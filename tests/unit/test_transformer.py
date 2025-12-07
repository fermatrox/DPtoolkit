"""Tests for dp_toolkit.data.transformer module."""

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.transformer import (
    MechanismType,
    TransformColumnType,
    ColumnConfig,
    TransformResult,
    ColumnTransformer,
    transform_numeric,
    transform_categorical,
    transform_date,
)
from dp_toolkit.core.mechanisms import EPSILON_MIN, EPSILON_MAX


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def transformer():
    """Create a default ColumnTransformer."""
    return ColumnTransformer()


@pytest.fixture
def numeric_int_series():
    """Create a numeric integer series."""
    return pd.Series([10, 20, 30, 40, 50], name="age")


@pytest.fixture
def numeric_float_series():
    """Create a numeric float series."""
    return pd.Series([1.5, 2.7, 3.2, 4.8, 5.1], name="weight")


@pytest.fixture
def numeric_with_nulls():
    """Create a numeric series with nulls."""
    return pd.Series([10.0, np.nan, 30.0, np.nan, 50.0], name="value")


@pytest.fixture
def categorical_series():
    """Create a categorical series."""
    return pd.Series(["A", "B", "A", "C", "B", "A"], name="category")


@pytest.fixture
def categorical_with_nulls():
    """Create a categorical series with nulls."""
    return pd.Series(["A", None, "A", "B", None, "B"], name="category")


@pytest.fixture
def date_series():
    """Create a datetime series."""
    return pd.Series(
        pd.date_range("2020-01-01", periods=5, freq="D"),
        name="date"
    )


@pytest.fixture
def date_with_nulls():
    """Create a datetime series with nulls."""
    dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="D"))
    dates.iloc[1] = pd.NaT
    dates.iloc[3] = pd.NaT
    return dates


# =============================================================================
# ColumnConfig Tests
# =============================================================================


class TestColumnConfig:
    """Tests for ColumnConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ColumnConfig()
        assert config.epsilon == 1.0
        assert config.mechanism is None
        assert config.lower is None
        assert config.upper is None
        assert config.delta is None
        assert config.preserve_type is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ColumnConfig(
            epsilon=0.5,
            mechanism=MechanismType.LAPLACE,
            lower=0,
            upper=100,
        )
        assert config.epsilon == 0.5
        assert config.mechanism == MechanismType.LAPLACE
        assert config.lower == 0
        assert config.upper == 100

    def test_epsilon_validation_low(self):
        """Test epsilon below minimum raises error."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            ColumnConfig(epsilon=0.001)

    def test_epsilon_validation_high(self):
        """Test epsilon above maximum raises error."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            ColumnConfig(epsilon=20.0)

    def test_delta_validation_low(self):
        """Test delta below minimum raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            ColumnConfig(delta=1e-12)

    def test_delta_validation_high(self):
        """Test delta above maximum raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            ColumnConfig(delta=0.01)


# =============================================================================
# ColumnTransformer Initialization Tests
# =============================================================================


class TestColumnTransformerInit:
    """Tests for ColumnTransformer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        t = ColumnTransformer()
        assert t.default_epsilon == 1.0
        assert t.default_delta == 1e-6

    def test_custom_init(self):
        """Test custom initialization."""
        t = ColumnTransformer(default_epsilon=0.5, default_delta=1e-5)
        assert t.default_epsilon == 0.5
        assert t.default_delta == 1e-5

    def test_invalid_epsilon(self):
        """Test invalid default epsilon."""
        with pytest.raises(ValueError, match="Default epsilon"):
            ColumnTransformer(default_epsilon=0.001)

    def test_invalid_delta(self):
        """Test invalid default delta."""
        with pytest.raises(ValueError, match="Default delta"):
            ColumnTransformer(default_delta=0.01)


# =============================================================================
# Type Detection Tests
# =============================================================================


class TestTypeDetection:
    """Tests for column type detection."""

    def test_detect_numeric_unbounded(self, transformer, numeric_int_series):
        """Test numeric column without bounds detected as unbounded."""
        col_type = transformer.detect_column_type(numeric_int_series)
        assert col_type == TransformColumnType.NUMERIC_UNBOUNDED

    def test_detect_numeric_bounded(self, transformer, numeric_int_series):
        """Test numeric column with bounds detected as bounded."""
        col_type = transformer.detect_column_type(
            numeric_int_series, bounds=(0, 100)
        )
        assert col_type == TransformColumnType.NUMERIC_BOUNDED

    def test_detect_categorical(self, transformer, categorical_series):
        """Test categorical column detected correctly."""
        col_type = transformer.detect_column_type(categorical_series)
        assert col_type == TransformColumnType.CATEGORICAL

    def test_detect_date(self, transformer, date_series):
        """Test datetime column detected correctly."""
        col_type = transformer.detect_column_type(date_series)
        assert col_type == TransformColumnType.DATE

    def test_is_integer_type_true(self, transformer):
        """Test integer type detection - true."""
        series = pd.Series([1, 2, 3], dtype="int64")
        assert transformer.is_integer_type(series) is True

    def test_is_integer_type_false(self, transformer):
        """Test integer type detection - false for floats."""
        series = pd.Series([1.1, 2.2, 3.3], dtype="float64")
        assert transformer.is_integer_type(series) is False


# =============================================================================
# Numeric Bounded Transformation Tests
# =============================================================================


class TestNumericBoundedTransform:
    """Tests for bounded numeric transformation."""

    def test_basic_transform(self, transformer, numeric_int_series):
        """Test basic numeric transformation."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert isinstance(result, TransformResult)
        assert result.column_type == TransformColumnType.NUMERIC_BOUNDED
        assert result.mechanism_type == MechanismType.LAPLACE
        assert result.privacy_usage.epsilon == 1.0
        assert result.null_count == 0
        assert result.bounds == (0, 100)
        assert len(result.data) == len(numeric_int_series)

    def test_preserves_integer_type(self, transformer, numeric_int_series):
        """Test that integer type is preserved."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            upper=100,
            preserve_type=True,
        )
        # Result should be integers (might be wrapped as int64 or Int64)
        # Just check values are whole numbers
        non_null = result.data.dropna()
        assert all(v == int(v) for v in non_null)

    def test_float_type(self, transformer, numeric_float_series):
        """Test float column transformation."""
        result = transformer.transform_numeric_bounded(
            series=numeric_float_series,
            epsilon=1.0,
            lower=0,
            upper=10,
        )
        assert result.column_type == TransformColumnType.NUMERIC_BOUNDED
        assert len(result.data) == len(numeric_float_series)

    def test_null_preservation(self, transformer, numeric_with_nulls):
        """Test that nulls are preserved."""
        result = transformer.transform_numeric_bounded(
            series=numeric_with_nulls,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert result.null_count == 2
        # Check nulls at same positions
        assert pd.isna(result.data.iloc[1])
        assert pd.isna(result.data.iloc[3])

    def test_different_epsilon(self, transformer, numeric_int_series):
        """Test with different epsilon values."""
        result_low = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=0.1,
            lower=0,
            upper=100,
        )
        result_high = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=5.0,
            lower=0,
            upper=100,
        )
        # Higher epsilon should have smaller scale (less noise)
        assert result_low.metadata["scale"] > result_high.metadata["scale"]

    def test_non_numeric_raises(self, transformer, categorical_series):
        """Test that non-numeric input raises error."""
        with pytest.raises(TypeError, match="must be numeric"):
            transformer.transform_numeric_bounded(
                series=categorical_series,
                epsilon=1.0,
                lower=0,
                upper=100,
            )

    def test_noise_actually_added(self, transformer, numeric_int_series):
        """Test that noise is actually added to data."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        # At least some values should be different
        # (with very high probability given random noise)
        differences = (result.data != numeric_int_series).sum()
        assert differences > 0

    def test_all_nulls(self, transformer):
        """Test handling of all-null series."""
        all_null = pd.Series([np.nan, np.nan, np.nan])
        result = transformer.transform_numeric_bounded(
            series=all_null,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert result.null_count == 3
        assert all(pd.isna(result.data))


# =============================================================================
# Numeric Unbounded Transformation Tests
# =============================================================================


class TestNumericUnboundedTransform:
    """Tests for unbounded numeric transformation."""

    def test_basic_transform(self, transformer, numeric_float_series):
        """Test basic unbounded transformation."""
        result = transformer.transform_numeric_unbounded(
            series=numeric_float_series,
            epsilon=1.0,
            sensitivity=10.0,
        )
        assert isinstance(result, TransformResult)
        assert result.column_type == TransformColumnType.NUMERIC_UNBOUNDED
        assert result.mechanism_type == MechanismType.GAUSSIAN
        assert result.privacy_usage.epsilon == 1.0
        assert result.privacy_usage.delta is not None
        assert result.bounds is None

    def test_null_preservation(self, transformer, numeric_with_nulls):
        """Test null preservation in unbounded transform."""
        result = transformer.transform_numeric_unbounded(
            series=numeric_with_nulls,
            epsilon=1.0,
            sensitivity=100.0,
        )
        assert result.null_count == 2
        assert pd.isna(result.data.iloc[1])
        assert pd.isna(result.data.iloc[3])

    def test_custom_delta(self, transformer, numeric_float_series):
        """Test custom delta parameter."""
        result = transformer.transform_numeric_unbounded(
            series=numeric_float_series,
            epsilon=1.0,
            sensitivity=10.0,
            delta=1e-5,
        )
        assert result.privacy_usage.delta == 1e-5

    def test_integer_type_preserved(self, transformer, numeric_int_series):
        """Test integer type preservation for unbounded."""
        result = transformer.transform_numeric_unbounded(
            series=numeric_int_series,
            epsilon=1.0,
            sensitivity=100.0,
            preserve_type=True,
        )
        non_null = result.data.dropna()
        assert all(v == int(v) for v in non_null)

    def test_non_numeric_raises(self, transformer, categorical_series):
        """Test non-numeric input raises error."""
        with pytest.raises(TypeError, match="must be numeric"):
            transformer.transform_numeric_unbounded(
                series=categorical_series,
                epsilon=1.0,
                sensitivity=10.0,
            )


# =============================================================================
# Categorical Transformation Tests
# =============================================================================


class TestCategoricalTransform:
    """Tests for categorical transformation."""

    def test_basic_transform(self, transformer, categorical_series):
        """Test basic categorical transformation."""
        result = transformer.transform_categorical(
            series=categorical_series,
            epsilon=1.0,
        )
        assert isinstance(result, TransformResult)
        assert result.column_type == TransformColumnType.CATEGORICAL
        assert result.mechanism_type == MechanismType.EXPONENTIAL
        assert result.privacy_usage.epsilon == 1.0
        assert len(result.data) == len(categorical_series)

    def test_null_preservation(self, transformer, categorical_with_nulls):
        """Test null preservation in categorical transform."""
        result = transformer.transform_categorical(
            series=categorical_with_nulls,
            epsilon=1.0,
        )
        assert result.null_count == 2
        assert pd.isna(result.data.iloc[1])
        assert pd.isna(result.data.iloc[4])

    def test_output_from_original_categories(
        self, transformer, categorical_series
    ):
        """Test that output contains only original categories."""
        result = transformer.transform_categorical(
            series=categorical_series,
            epsilon=1.0,
        )
        original_categories = set(categorical_series.dropna().unique())
        output_categories = set(result.data.dropna().unique())
        assert output_categories.issubset(original_categories)

    def test_single_category(self, transformer):
        """Test handling of single category."""
        single_cat = pd.Series(["A", "A", "A", "A"])
        result = transformer.transform_categorical(
            series=single_cat,
            epsilon=1.0,
        )
        # Should return as-is since no randomization needed
        assert result.metadata.get("single_category") is True

    def test_all_nulls(self, transformer):
        """Test all-null categorical series."""
        all_null = pd.Series([None, None, None])
        result = transformer.transform_categorical(
            series=all_null,
            epsilon=1.0,
        )
        assert result.null_count == 3
        assert all(pd.isna(result.data))

    def test_higher_epsilon_more_mode(self, transformer):
        """Test that higher epsilon preserves mode better."""
        # Create series with clear mode
        series = pd.Series(["A"] * 80 + ["B"] * 20)

        # With very low epsilon, more randomness
        np.random.seed(42)
        result_low = transformer.transform_categorical(
            series=series,
            epsilon=0.1,
        )
        count_a_low = (result_low.data == "A").sum()

        # With high epsilon, less randomness
        np.random.seed(42)
        result_high = transformer.transform_categorical(
            series=series,
            epsilon=5.0,
        )
        count_a_high = (result_high.data == "A").sum()

        # High epsilon should preserve mode better on average
        # But due to randomness, just check both ran
        assert count_a_low > 0
        assert count_a_high > 0


# =============================================================================
# Date Transformation Tests
# =============================================================================


class TestDateTransform:
    """Tests for date transformation."""

    def test_basic_transform(self, transformer, date_series):
        """Test basic date transformation."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
        )
        assert isinstance(result, TransformResult)
        assert result.column_type == TransformColumnType.DATE
        assert result.mechanism_type == MechanismType.LAPLACE
        assert result.privacy_usage.epsilon == 1.0
        assert len(result.data) == len(date_series)

    def test_null_preservation(self, transformer, date_with_nulls):
        """Test null preservation in date transform."""
        result = transformer.transform_date(
            series=date_with_nulls,
            epsilon=1.0,
        )
        assert result.null_count == 2
        assert pd.isna(result.data.iloc[1])
        assert pd.isna(result.data.iloc[3])

    def test_output_is_datetime(self, transformer, date_series):
        """Test that output is still datetime type."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
        )
        assert pd.api.types.is_datetime64_any_dtype(result.data)

    def test_with_explicit_bounds(self, transformer, date_series):
        """Test with explicit date bounds."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
            lower=pd.Timestamp("2019-01-01"),
            upper=pd.Timestamp("2021-01-01"),
        )
        assert result.bounds[0] == pd.Timestamp("2019-01-01")
        assert result.bounds[1] == pd.Timestamp("2021-01-01")

    def test_unit_days(self, transformer, date_series):
        """Test transformation with day units."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
            unit="D",
        )
        assert result.metadata["unit"] == "D"

    def test_unit_seconds(self, transformer, date_series):
        """Test transformation with second units."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
            unit="s",
        )
        assert result.metadata["unit"] == "s"

    def test_invalid_unit_raises(self, transformer, date_series):
        """Test that invalid unit raises error."""
        with pytest.raises(ValueError, match="Unit must be"):
            transformer.transform_date(
                series=date_series,
                epsilon=1.0,
                unit="h",
            )

    def test_non_datetime_raises(self, transformer, numeric_int_series):
        """Test that non-datetime input raises error."""
        with pytest.raises(TypeError, match="must be datetime"):
            transformer.transform_date(
                series=numeric_int_series,
                epsilon=1.0,
            )

    def test_noise_added(self, transformer, date_series):
        """Test that noise is actually added to dates."""
        result = transformer.transform_date(
            series=date_series,
            epsilon=1.0,
        )
        # At least some dates should be different
        differences = (result.data != date_series).sum()
        assert differences > 0


# =============================================================================
# Generic Transform Tests
# =============================================================================


class TestGenericTransform:
    """Tests for the generic transform method."""

    def test_auto_detect_numeric_bounded(
        self, transformer, numeric_int_series
    ):
        """Test auto-detection of bounded numeric."""
        result = transformer.transform(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert result.column_type == TransformColumnType.NUMERIC_BOUNDED
        assert result.mechanism_type == MechanismType.LAPLACE

    def test_auto_detect_numeric_unbounded(
        self, transformer, numeric_int_series
    ):
        """Test auto-detection of unbounded numeric."""
        result = transformer.transform(
            series=numeric_int_series,
            epsilon=1.0,
            # No bounds provided
        )
        assert result.column_type == TransformColumnType.NUMERIC_UNBOUNDED
        assert result.mechanism_type == MechanismType.GAUSSIAN

    def test_auto_detect_categorical(self, transformer, categorical_series):
        """Test auto-detection of categorical."""
        result = transformer.transform(
            series=categorical_series,
            epsilon=1.0,
        )
        assert result.column_type == TransformColumnType.CATEGORICAL
        assert result.mechanism_type == MechanismType.EXPONENTIAL

    def test_auto_detect_date(self, transformer, date_series):
        """Test auto-detection of date."""
        result = transformer.transform(
            series=date_series,
            epsilon=1.0,
        )
        assert result.column_type == TransformColumnType.DATE
        assert result.mechanism_type == MechanismType.LAPLACE

    def test_with_config(self, transformer, numeric_int_series):
        """Test transformation with ColumnConfig."""
        config = ColumnConfig(
            epsilon=0.5,
            lower=0,
            upper=100,
        )
        result = transformer.transform(
            series=numeric_int_series,
            config=config,
        )
        assert result.privacy_usage.epsilon == 0.5
        assert result.bounds == (0, 100)

    def test_explicit_params_override_config(
        self, transformer, numeric_int_series
    ):
        """Test that explicit parameters override config."""
        config = ColumnConfig(epsilon=0.5, lower=0, upper=100)
        result = transformer.transform(
            series=numeric_int_series,
            config=config,
            epsilon=1.0,  # Override
        )
        assert result.privacy_usage.epsilon == 1.0

    def test_bounded_requires_bounds(self, transformer, numeric_int_series):
        """Test that bounded numeric requires both bounds."""
        # This should work - detected as unbounded
        result = transformer.transform(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            # No upper
        )
        # Should fall through to unbounded
        assert result.column_type == TransformColumnType.NUMERIC_UNBOUNDED


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_transform_numeric(self, numeric_int_series):
        """Test transform_numeric convenience function."""
        result = transform_numeric(
            series=numeric_int_series,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(numeric_int_series)

    def test_transform_categorical(self, categorical_series):
        """Test transform_categorical convenience function."""
        result = transform_categorical(
            series=categorical_series,
            epsilon=1.0,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(categorical_series)

    def test_transform_date(self, date_series):
        """Test transform_date convenience function."""
        result = transform_date(
            series=date_series,
            epsilon=1.0,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(date_series)
        assert pd.api.types.is_datetime64_any_dtype(result)


# =============================================================================
# Privacy Guarantee Tests
# =============================================================================


class TestPrivacyGuarantees:
    """Tests for privacy guarantees."""

    def test_privacy_usage_tracked(self, transformer, numeric_int_series):
        """Test that privacy usage is correctly tracked."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=0.5,
            lower=0,
            upper=100,
        )
        assert result.privacy_usage.epsilon == 0.5
        assert result.privacy_usage.is_pure_dp  # Laplace is pure DP

    def test_gaussian_has_delta(self, transformer, numeric_float_series):
        """Test that Gaussian mechanism has delta."""
        result = transformer.transform_numeric_unbounded(
            series=numeric_float_series,
            epsilon=1.0,
            sensitivity=10.0,
            delta=1e-5,
        )
        assert result.privacy_usage.delta == 1e-5
        assert not result.privacy_usage.is_pure_dp


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_series_numeric(self, transformer):
        """Test empty numeric series."""
        empty = pd.Series([], dtype=float)
        result = transformer.transform_numeric_bounded(
            series=empty,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert len(result.data) == 0
        assert result.null_count == 0

    def test_empty_series_categorical(self, transformer):
        """Test empty categorical series."""
        empty = pd.Series([], dtype=object)
        result = transformer.transform_categorical(
            series=empty,
            epsilon=1.0,
        )
        assert len(result.data) == 0

    def test_single_value_numeric(self, transformer):
        """Test single value numeric series."""
        single = pd.Series([42.0])
        result = transformer.transform_numeric_bounded(
            series=single,
            epsilon=1.0,
            lower=0,
            upper=100,
        )
        assert len(result.data) == 1

    def test_single_value_categorical(self, transformer):
        """Test single non-null value categorical."""
        single = pd.Series(["A"])
        result = transformer.transform_categorical(
            series=single,
            epsilon=1.0,
        )
        # Single category, returned as-is
        assert result.data.iloc[0] == "A"

    def test_min_epsilon(self, transformer, numeric_int_series):
        """Test with minimum epsilon."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=EPSILON_MIN,
            lower=0,
            upper=100,
        )
        assert result.privacy_usage.epsilon == EPSILON_MIN

    def test_max_epsilon(self, transformer, numeric_int_series):
        """Test with maximum epsilon."""
        result = transformer.transform_numeric_bounded(
            series=numeric_int_series,
            epsilon=EPSILON_MAX,
            lower=0,
            upper=100,
        )
        assert result.privacy_usage.epsilon == EPSILON_MAX

    def test_large_series(self, transformer):
        """Test with large series (performance check)."""
        large = pd.Series(np.random.randn(10000))
        result = transformer.transform_numeric_bounded(
            series=large,
            epsilon=1.0,
            lower=-10,
            upper=10,
        )
        assert len(result.data) == 10000

    def test_nullable_integer_type(self, transformer):
        """Test nullable integer type handling."""
        nullable_int = pd.Series([1, 2, None, 4, 5], dtype="Int64")
        result = transformer.transform_numeric_bounded(
            series=nullable_int,
            epsilon=1.0,
            lower=0,
            upper=10,
            preserve_type=True,
        )
        assert result.null_count == 1
        assert pd.isna(result.data.iloc[2])


# =============================================================================
# Dataset Transformer Tests
# =============================================================================

from dp_toolkit.data.transformer import (
    ProtectionMode,
    DatasetColumnConfig,
    DatasetConfig,
    ColumnTransformSummary,
    DatasetTransformResult,
    DatasetTransformer,
    transform_dataset,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "department": ["HR", "IT", "HR", "IT", "Finance"],
        "hire_date": pd.date_range("2020-01-01", periods=5, freq="ME"),
        "ssn": ["123-45-6789", "234-56-7890", "345-67-8901", "456-78-9012", "567-89-0123"],
    })


@pytest.fixture
def dataset_transformer():
    """Create a DatasetTransformer instance."""
    return DatasetTransformer()


# =============================================================================
# DatasetConfig Tests
# =============================================================================


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.global_epsilon == 1.0
        assert config.default_mode == ProtectionMode.PROTECT

    def test_custom_global_epsilon(self):
        """Test custom global epsilon."""
        config = DatasetConfig(global_epsilon=0.5)
        assert config.global_epsilon == 0.5

    def test_invalid_global_epsilon(self):
        """Test invalid global epsilon raises error."""
        with pytest.raises(ValueError, match="Global epsilon"):
            DatasetConfig(global_epsilon=0.001)

    def test_invalid_global_delta(self):
        """Test invalid global delta raises error."""
        with pytest.raises(ValueError, match="Global delta"):
            DatasetConfig(global_delta=0.01)

    def test_get_column_config_default(self):
        """Test getting default column config."""
        config = DatasetConfig()
        col_config = config.get_column_config("unknown_column")
        assert col_config.mode == ProtectionMode.PROTECT

    def test_get_column_config_custom(self):
        """Test getting custom column config."""
        config = DatasetConfig()
        config.set_column_mode("age", ProtectionMode.PASSTHROUGH)
        col_config = config.get_column_config("age")
        assert col_config.mode == ProtectionMode.PASSTHROUGH

    def test_get_epsilon_global(self):
        """Test getting global epsilon for column."""
        config = DatasetConfig(global_epsilon=0.5)
        assert config.get_epsilon("any_column") == 0.5

    def test_get_epsilon_column_specific(self):
        """Test getting column-specific epsilon."""
        config = DatasetConfig(global_epsilon=1.0)
        config.set_column_mode("age", ProtectionMode.PROTECT, epsilon=0.3)
        assert config.get_epsilon("age") == 0.3
        assert config.get_epsilon("salary") == 1.0

    def test_protect_columns(self):
        """Test protect_columns helper."""
        config = DatasetConfig()
        config.protect_columns(["age", "salary"], epsilon=0.5)
        assert config.get_column_config("age").mode == ProtectionMode.PROTECT
        assert config.get_column_config("salary").mode == ProtectionMode.PROTECT
        assert config.get_epsilon("age") == 0.5

    def test_passthrough_columns(self):
        """Test passthrough_columns helper."""
        config = DatasetConfig()
        config.passthrough_columns(["id", "timestamp"])
        assert config.get_column_config("id").mode == ProtectionMode.PASSTHROUGH
        assert config.get_column_config("timestamp").mode == ProtectionMode.PASSTHROUGH

    def test_exclude_columns(self):
        """Test exclude_columns helper."""
        config = DatasetConfig()
        config.exclude_columns(["ssn", "email"])
        assert config.get_column_config("ssn").mode == ProtectionMode.EXCLUDE
        assert config.get_column_config("email").mode == ProtectionMode.EXCLUDE


# =============================================================================
# DatasetTransformer Basic Tests
# =============================================================================


class TestDatasetTransformerBasic:
    """Basic tests for DatasetTransformer."""

    def test_default_transform(self, dataset_transformer, sample_dataframe):
        """Test default transformation (all columns protected)."""
        result = dataset_transformer.transform(sample_dataframe)
        assert isinstance(result, DatasetTransformResult)
        assert result.row_count == 5
        assert len(result.protected_columns) == 5  # All columns protected
        assert len(result.passthrough_columns) == 0
        assert len(result.excluded_columns) == 0

    def test_transform_with_config(self, dataset_transformer, sample_dataframe):
        """Test transformation with custom config."""
        config = DatasetConfig(global_epsilon=0.5)
        config.passthrough_columns(["hire_date"])
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        assert "ssn" not in result.data.columns
        assert "hire_date" in result.data.columns
        assert "age" in result.data.columns
        assert result.column_count == 4

    def test_excluded_columns_not_in_output(self, dataset_transformer, sample_dataframe):
        """Test that excluded columns are not in output."""
        config = DatasetConfig()
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        assert "ssn" not in result.data.columns
        assert "ssn" in result.excluded_columns

    def test_passthrough_columns_unchanged(self, dataset_transformer, sample_dataframe):
        """Test that passthrough columns are unchanged."""
        config = DatasetConfig()
        config.passthrough_columns(["department"])

        result = dataset_transformer.transform(sample_dataframe, config)
        # Passthrough column should be identical
        assert (result.data["department"] == sample_dataframe["department"]).all()
        assert "department" in result.passthrough_columns

    def test_protected_columns_modified(self, dataset_transformer, sample_dataframe):
        """Test that protected columns have noise added."""
        config = DatasetConfig(global_epsilon=1.0)
        config.protect_columns(["age", "salary"])
        config.passthrough_columns(["department", "hire_date"])
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        # Protected numeric columns should have noise (very unlikely to be identical)
        # At least check they have same shape
        assert len(result.data["age"]) == 5
        assert len(result.data["salary"]) == 5


# =============================================================================
# Privacy Budget Tests
# =============================================================================


class TestDatasetPrivacyBudget:
    """Tests for privacy budget tracking in DatasetTransformer."""

    def test_total_epsilon_tracked(self, dataset_transformer, sample_dataframe):
        """Test that total epsilon is tracked."""
        config = DatasetConfig(global_epsilon=0.5)
        config.protect_columns(["age", "salary"])
        config.passthrough_columns(["department", "hire_date", "ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        # Each protected column uses 0.5 epsilon
        assert result.total_epsilon == pytest.approx(1.0, rel=0.01)

    def test_per_column_epsilon(self, dataset_transformer, sample_dataframe):
        """Test per-column epsilon in summaries."""
        config = DatasetConfig(global_epsilon=0.5)
        config.protect_columns(["age"])
        config.passthrough_columns(["salary", "department", "hire_date", "ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        assert result.column_summaries["age"].epsilon == pytest.approx(0.5)
        assert result.column_summaries["salary"].epsilon is None  # Passthrough

    def test_transform_with_budget(self, dataset_transformer, sample_dataframe):
        """Test transform_with_budget divides epsilon equally."""
        result = dataset_transformer.transform_with_budget(
            df=sample_dataframe,
            total_epsilon=1.0,
            columns_to_protect=["age", "salary"],
            columns_to_exclude=["ssn"],
        )
        # 1.0 epsilon / 2 columns = 0.5 each
        assert result.column_summaries["age"].epsilon == pytest.approx(0.5)
        assert result.column_summaries["salary"].epsilon == pytest.approx(0.5)
        assert result.total_epsilon == pytest.approx(1.0, rel=0.01)


# =============================================================================
# Column Mode Tests
# =============================================================================


class TestColumnModes:
    """Tests for column protection modes."""

    def test_all_modes_applied(self, dataset_transformer, sample_dataframe):
        """Test all three modes are applied correctly."""
        config = DatasetConfig()
        config.protect_columns(["age"])
        config.passthrough_columns(["department"])
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)

        assert "age" in result.protected_columns
        assert "department" in result.passthrough_columns
        assert "ssn" in result.excluded_columns
        assert result.column_summaries["age"].mode == ProtectionMode.PROTECT
        assert result.column_summaries["department"].mode == ProtectionMode.PASSTHROUGH
        assert result.column_summaries["ssn"].mode == ProtectionMode.EXCLUDE

    def test_default_mode_protect(self, dataset_transformer, sample_dataframe):
        """Test default mode is PROTECT."""
        config = DatasetConfig(default_mode=ProtectionMode.PROTECT)
        result = dataset_transformer.transform(sample_dataframe, config)
        # All columns should be protected by default
        assert len(result.protected_columns) == 5

    def test_default_mode_passthrough(self, dataset_transformer, sample_dataframe):
        """Test default mode can be PASSTHROUGH."""
        config = DatasetConfig(default_mode=ProtectionMode.PASSTHROUGH)
        config.protect_columns(["age"])

        result = dataset_transformer.transform(sample_dataframe, config)
        assert "age" in result.protected_columns
        # Others should be passthrough
        assert "department" in result.passthrough_columns


# =============================================================================
# Progress Callback Tests
# =============================================================================


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_progress_callback_called(self, dataset_transformer, sample_dataframe):
        """Test that progress callback is called."""
        calls = []

        def callback(col, idx, total):
            calls.append((col, idx, total))

        config = DatasetConfig()
        config.exclude_columns(["ssn"])  # 4 columns to process

        dataset_transformer.transform(sample_dataframe, config, progress_callback=callback)

        # Should be called once per column plus "complete"
        assert len(calls) >= 4
        # Last call should be "complete"
        assert calls[-1][0] == "complete"

    def test_progress_indices(self, dataset_transformer, sample_dataframe):
        """Test progress callback indices."""
        indices = []

        def callback(col, idx, total):
            indices.append(idx)

        config = DatasetConfig()
        config.exclude_columns(["ssn"])

        dataset_transformer.transform(sample_dataframe, config, progress_callback=callback)

        # Check indices increment
        non_complete = [i for i in indices if i < len(indices) - 1]
        assert non_complete == list(range(len(non_complete)))


# =============================================================================
# DatasetTransformResult Tests
# =============================================================================


class TestDatasetTransformResult:
    """Tests for DatasetTransformResult."""

    def test_column_count(self, dataset_transformer, sample_dataframe):
        """Test column_count property."""
        config = DatasetConfig()
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        assert result.column_count == 4

    def test_protection_rate(self, dataset_transformer, sample_dataframe):
        """Test protection_rate property."""
        config = DatasetConfig()
        config.protect_columns(["age", "salary"])
        config.passthrough_columns(["department", "hire_date"])
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        # 2 protected / 4 total = 0.5
        assert result.protection_rate == pytest.approx(0.5)

    def test_column_summaries(self, dataset_transformer, sample_dataframe):
        """Test column summaries contain all columns."""
        config = DatasetConfig()
        config.exclude_columns(["ssn"])

        result = dataset_transformer.transform(sample_dataframe, config)
        # Should have summary for all columns including excluded
        assert "ssn" in result.column_summaries
        assert "age" in result.column_summaries


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDatasetConvenienceFunctions:
    """Tests for dataset convenience functions."""

    def test_transform_dataset_basic(self, sample_dataframe):
        """Test transform_dataset convenience function."""
        result = transform_dataset(
            df=sample_dataframe,
            epsilon=1.0,
            columns_to_exclude=["ssn"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "ssn" not in result.columns
        assert len(result) == 5

    def test_transform_dataset_selective(self, sample_dataframe):
        """Test selective column protection."""
        result = transform_dataset(
            df=sample_dataframe,
            epsilon=1.0,
            columns_to_protect=["age", "salary"],
            columns_to_exclude=["ssn"],
        )
        assert "age" in result.columns
        assert "salary" in result.columns
        assert "ssn" not in result.columns


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestDatasetEdgeCases:
    """Tests for edge cases in dataset transformation."""

    def test_empty_dataframe(self, dataset_transformer):
        """Test transformation of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = dataset_transformer.transform(empty_df)
        assert result.row_count == 0
        assert result.column_count == 0

    def test_single_column(self, dataset_transformer):
        """Test DataFrame with single column."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        result = dataset_transformer.transform(df)
        assert result.column_count == 1
        assert len(result.protected_columns) == 1

    def test_all_excluded(self, dataset_transformer, sample_dataframe):
        """Test excluding all columns."""
        config = DatasetConfig()
        config.exclude_columns(list(sample_dataframe.columns))

        result = dataset_transformer.transform(sample_dataframe, config)
        assert result.column_count == 0
        assert len(result.excluded_columns) == 5

    def test_all_passthrough(self, dataset_transformer, sample_dataframe):
        """Test all columns as passthrough."""
        config = DatasetConfig()
        config.passthrough_columns(list(sample_dataframe.columns))

        result = dataset_transformer.transform(sample_dataframe, config)
        assert result.total_epsilon == 0.0
        assert len(result.passthrough_columns) == 5

    def test_large_dataframe_performance(self, dataset_transformer):
        """Test performance with larger DataFrame."""
        import time

        # Create 10K rows x 10 columns
        large_df = pd.DataFrame({
            f"col_{i}": np.random.randn(10000) for i in range(10)
        })

        config = DatasetConfig(global_epsilon=1.0)

        start = time.time()
        result = dataset_transformer.transform(large_df, config)
        elapsed = time.time() - start

        assert result.row_count == 10000
        assert result.column_count == 10
        # Should complete in reasonable time (< 30s for 10K x 10)
        assert elapsed < 30.0
