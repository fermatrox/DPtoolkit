"""Unit tests for dp_toolkit.recommendations.advisor module."""

import pytest
import pandas as pd
import numpy as np

from dp_toolkit.recommendations.advisor import (
    # Enums
    RecommendedMechanism,
    DataType,
    UtilityLevel,
    # Data classes
    EpsilonRecommendation,
    MechanismRecommendation,
    ColumnRecommendation,
    DatasetRecommendation,
    # Classes
    EpsilonAdvisor,
    MechanismAdvisor,
    RecommendationAdvisor,
    # Convenience functions
    recommend_epsilon,
    recommend_mechanism,
    recommend_for_column,
    recommend_for_dataset,
    get_epsilon_for_column,
    get_mechanism_for_series,
    validate_epsilon,
    # Constants
    EPSILON_RANGES,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
)
from dp_toolkit.recommendations.classifier import SensitivityLevel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def epsilon_advisor():
    """Create an EpsilonAdvisor."""
    return EpsilonAdvisor()


@pytest.fixture
def mechanism_advisor():
    """Create a MechanismAdvisor."""
    return MechanismAdvisor()


@pytest.fixture
def recommendation_advisor():
    """Create a RecommendationAdvisor."""
    return RecommendationAdvisor()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "patient_ssn": ["123-45-6789", "987-65-4321"],
        "patient_name": ["John Doe", "Jane Smith"],
        "age": [33, 38],
        "income": [50000.0, 75000.0],
        "department": ["Cardiology", "Neurology"],
        "visit_date": pd.to_datetime(["2024-01-15", "2024-02-20"]),
        "record_id": [1001, 1002],
        "visit_count": [5, 3],
    })


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for advisor enums."""

    def test_recommended_mechanism_values(self):
        """Test RecommendedMechanism enum values."""
        assert RecommendedMechanism.LAPLACE.value == "laplace"
        assert RecommendedMechanism.GAUSSIAN.value == "gaussian"
        assert RecommendedMechanism.EXPONENTIAL.value == "exponential"

    def test_data_type_values(self):
        """Test DataType enum values."""
        assert DataType.NUMERIC_BOUNDED.value == "numeric_bounded"
        assert DataType.NUMERIC_UNBOUNDED.value == "numeric_unbounded"
        assert DataType.CATEGORICAL.value == "categorical"
        assert DataType.DATE.value == "date"
        assert DataType.UNKNOWN.value == "unknown"

    def test_utility_level_values(self):
        """Test UtilityLevel enum values."""
        assert UtilityLevel.HIGH.value == "high"
        assert UtilityLevel.MEDIUM.value == "medium"
        assert UtilityLevel.LOW.value == "low"

    def test_enum_str(self):
        """Test enum string representation."""
        assert str(RecommendedMechanism.LAPLACE) == "laplace"
        assert str(DataType.CATEGORICAL) == "categorical"
        assert str(UtilityLevel.HIGH) == "high"


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for advisor constants."""

    def test_epsilon_ranges_from_prd(self):
        """Test that epsilon ranges match PRD specifications."""
        # PRD: High (ε = 0.1–0.5), Medium (ε = 0.5–2.0), Low (ε = 2.0–5.0)
        assert EPSILON_RANGES[SensitivityLevel.HIGH] == (0.1, 0.5)
        assert EPSILON_RANGES[SensitivityLevel.MEDIUM] == (0.5, 2.0)
        assert EPSILON_RANGES[SensitivityLevel.LOW] == (2.0, 5.0)

    def test_default_epsilon_in_range(self):
        """Test that default epsilon values are within their ranges."""
        for sensitivity, default in DEFAULT_EPSILON.items():
            eps_range = EPSILON_RANGES[sensitivity]
            assert eps_range[0] <= default <= eps_range[1], (
                f"Default epsilon {default} not in range {eps_range} "
                f"for {sensitivity}"
            )

    def test_default_delta(self):
        """Test default delta value."""
        assert DEFAULT_DELTA == 1e-5


# =============================================================================
# EpsilonRecommendation Tests
# =============================================================================


class TestEpsilonRecommendation:
    """Tests for EpsilonRecommendation dataclass."""

    def test_basic_recommendation(self):
        """Test creating a basic recommendation."""
        rec = EpsilonRecommendation(
            sensitivity=SensitivityLevel.HIGH,
            epsilon=0.3,
            epsilon_range=(0.1, 0.5),
            explanation="Test explanation",
            confidence=0.9,
        )
        assert rec.sensitivity == SensitivityLevel.HIGH
        assert rec.epsilon == 0.3
        assert rec.epsilon_range == (0.1, 0.5)
        assert rec.confidence == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = EpsilonRecommendation(
            sensitivity=SensitivityLevel.MEDIUM,
            epsilon=1.0,
            epsilon_range=(0.5, 2.0),
            explanation="Test",
        )
        d = rec.to_dict()
        assert d["sensitivity"] == "medium"
        assert d["epsilon"] == 1.0
        assert d["epsilon_min"] == 0.5
        assert d["epsilon_max"] == 2.0


# =============================================================================
# MechanismRecommendation Tests
# =============================================================================


class TestMechanismRecommendation:
    """Tests for MechanismRecommendation dataclass."""

    def test_laplace_recommendation(self):
        """Test Laplace mechanism recommendation."""
        rec = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="Test",
        )
        assert rec.mechanism == RecommendedMechanism.LAPLACE
        assert rec.delta is None

    def test_gaussian_recommendation(self):
        """Test Gaussian mechanism recommendation with delta."""
        rec = MechanismRecommendation(
            data_type=DataType.NUMERIC_UNBOUNDED,
            mechanism=RecommendedMechanism.GAUSSIAN,
            delta=1e-5,
            explanation="Test",
        )
        assert rec.mechanism == RecommendedMechanism.GAUSSIAN
        assert rec.delta == 1e-5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = MechanismRecommendation(
            data_type=DataType.CATEGORICAL,
            mechanism=RecommendedMechanism.EXPONENTIAL,
            explanation="Test",
        )
        d = rec.to_dict()
        assert d["data_type"] == "categorical"
        assert d["mechanism"] == "exponential"


# =============================================================================
# ColumnRecommendation Tests
# =============================================================================


class TestColumnRecommendation:
    """Tests for ColumnRecommendation dataclass."""

    def test_basic_column_recommendation(self):
        """Test creating a column recommendation."""
        eps_rec = EpsilonRecommendation(
            sensitivity=SensitivityLevel.HIGH,
            epsilon=0.3,
            epsilon_range=(0.1, 0.5),
            explanation="High sensitivity",
        )
        mech_rec = MechanismRecommendation(
            data_type=DataType.CATEGORICAL,
            mechanism=RecommendedMechanism.EXPONENTIAL,
            explanation="Categorical data",
        )
        col_rec = ColumnRecommendation(
            column_name="ssn",
            epsilon_recommendation=eps_rec,
            mechanism_recommendation=mech_rec,
        )
        assert col_rec.column_name == "ssn"
        assert col_rec.epsilon_recommendation.epsilon == 0.3
        assert col_rec.mechanism_recommendation.mechanism == RecommendedMechanism.EXPONENTIAL

    def test_to_dict(self):
        """Test conversion to dictionary."""
        eps_rec = EpsilonRecommendation(
            sensitivity=SensitivityLevel.MEDIUM,
            epsilon=1.0,
            epsilon_range=(0.5, 2.0),
            explanation="Test",
        )
        mech_rec = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="Test",
        )
        col_rec = ColumnRecommendation(
            column_name="age",
            epsilon_recommendation=eps_rec,
            mechanism_recommendation=mech_rec,
        )
        d = col_rec.to_dict()
        assert d["column_name"] == "age"
        assert "epsilon" in d
        assert "mechanism" in d


# =============================================================================
# DatasetRecommendation Tests
# =============================================================================


class TestDatasetRecommendation:
    """Tests for DatasetRecommendation dataclass."""

    def test_empty_dataset_recommendation(self):
        """Test empty dataset recommendation."""
        rec = DatasetRecommendation()
        assert len(rec.column_recommendations) == 0
        assert rec.total_epsilon == 0.0

    def test_total_epsilon(self):
        """Test total epsilon calculation."""
        eps1 = EpsilonRecommendation(
            sensitivity=SensitivityLevel.HIGH,
            epsilon=0.3,
            epsilon_range=(0.1, 0.5),
            explanation="",
        )
        eps2 = EpsilonRecommendation(
            sensitivity=SensitivityLevel.MEDIUM,
            epsilon=1.0,
            epsilon_range=(0.5, 2.0),
            explanation="",
        )
        mech = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="",
        )
        rec = DatasetRecommendation(
            column_recommendations={
                "col1": ColumnRecommendation(
                    column_name="col1",
                    epsilon_recommendation=eps1,
                    mechanism_recommendation=mech,
                ),
                "col2": ColumnRecommendation(
                    column_name="col2",
                    epsilon_recommendation=eps2,
                    mechanism_recommendation=mech,
                ),
            }
        )
        assert rec.total_epsilon == pytest.approx(1.3)

    def test_high_sensitivity_columns(self):
        """Test getting high sensitivity columns."""
        eps_high = EpsilonRecommendation(
            sensitivity=SensitivityLevel.HIGH,
            epsilon=0.3,
            epsilon_range=(0.1, 0.5),
            explanation="",
        )
        eps_low = EpsilonRecommendation(
            sensitivity=SensitivityLevel.LOW,
            epsilon=3.0,
            epsilon_range=(2.0, 5.0),
            explanation="",
        )
        mech = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="",
        )
        rec = DatasetRecommendation(
            column_recommendations={
                "ssn": ColumnRecommendation(
                    column_name="ssn",
                    epsilon_recommendation=eps_high,
                    mechanism_recommendation=mech,
                ),
                "record_id": ColumnRecommendation(
                    column_name="record_id",
                    epsilon_recommendation=eps_low,
                    mechanism_recommendation=mech,
                ),
            }
        )
        assert rec.high_sensitivity_columns == ["ssn"]

    def test_columns_by_mechanism(self):
        """Test grouping columns by mechanism."""
        eps = EpsilonRecommendation(
            sensitivity=SensitivityLevel.MEDIUM,
            epsilon=1.0,
            epsilon_range=(0.5, 2.0),
            explanation="",
        )
        mech_laplace = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="",
        )
        mech_exp = MechanismRecommendation(
            data_type=DataType.CATEGORICAL,
            mechanism=RecommendedMechanism.EXPONENTIAL,
            explanation="",
        )
        rec = DatasetRecommendation(
            column_recommendations={
                "age": ColumnRecommendation(
                    column_name="age",
                    epsilon_recommendation=eps,
                    mechanism_recommendation=mech_laplace,
                ),
                "category": ColumnRecommendation(
                    column_name="category",
                    epsilon_recommendation=eps,
                    mechanism_recommendation=mech_exp,
                ),
            }
        )
        by_mech = rec.columns_by_mechanism
        assert "age" in by_mech[RecommendedMechanism.LAPLACE]
        assert "category" in by_mech[RecommendedMechanism.EXPONENTIAL]

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        eps = EpsilonRecommendation(
            sensitivity=SensitivityLevel.MEDIUM,
            epsilon=1.0,
            epsilon_range=(0.5, 2.0),
            explanation="",
        )
        mech = MechanismRecommendation(
            data_type=DataType.NUMERIC_BOUNDED,
            mechanism=RecommendedMechanism.LAPLACE,
            explanation="",
        )
        rec = DatasetRecommendation(
            column_recommendations={
                "age": ColumnRecommendation(
                    column_name="age",
                    epsilon_recommendation=eps,
                    mechanism_recommendation=mech,
                ),
            }
        )
        df = rec.to_dataframe()
        assert len(df) == 1
        assert "Column" in df.columns
        assert "Epsilon" in df.columns
        assert "Mechanism" in df.columns


# =============================================================================
# EpsilonAdvisor Tests
# =============================================================================


class TestEpsilonAdvisor:
    """Tests for EpsilonAdvisor class."""

    def test_recommend_high_sensitivity(self, epsilon_advisor):
        """Test epsilon recommendation for high sensitivity."""
        rec = epsilon_advisor.recommend_epsilon(SensitivityLevel.HIGH)
        assert rec.sensitivity == SensitivityLevel.HIGH
        assert 0.1 <= rec.epsilon <= 0.5
        assert rec.epsilon_range == (0.1, 0.5)

    def test_recommend_medium_sensitivity(self, epsilon_advisor):
        """Test epsilon recommendation for medium sensitivity."""
        rec = epsilon_advisor.recommend_epsilon(SensitivityLevel.MEDIUM)
        assert rec.sensitivity == SensitivityLevel.MEDIUM
        assert 0.5 <= rec.epsilon <= 2.0

    def test_recommend_low_sensitivity(self, epsilon_advisor):
        """Test epsilon recommendation for low sensitivity."""
        rec = epsilon_advisor.recommend_epsilon(SensitivityLevel.LOW)
        assert rec.sensitivity == SensitivityLevel.LOW
        assert 2.0 <= rec.epsilon <= 5.0

    def test_prefer_privacy(self, epsilon_advisor):
        """Test that prefer_privacy uses lower epsilon."""
        rec = epsilon_advisor.recommend_epsilon(
            SensitivityLevel.MEDIUM, prefer_privacy=True
        )
        assert rec.epsilon == 0.5  # Lower end of range

    def test_prefer_utility(self, epsilon_advisor):
        """Test that prefer_utility uses higher epsilon."""
        rec = epsilon_advisor.recommend_epsilon(
            SensitivityLevel.MEDIUM, prefer_utility=True
        )
        assert rec.epsilon == 2.0  # Upper end of range

    def test_explanation_present(self, epsilon_advisor):
        """Test that explanation is present."""
        rec = epsilon_advisor.recommend_epsilon(SensitivityLevel.HIGH)
        assert len(rec.explanation) > 0
        assert "sensitive" in rec.explanation.lower()

    def test_get_epsilon_range(self, epsilon_advisor):
        """Test getting epsilon range."""
        range_high = epsilon_advisor.get_epsilon_range(SensitivityLevel.HIGH)
        assert range_high == (0.1, 0.5)

    def test_validate_epsilon_valid(self, epsilon_advisor):
        """Test validating a valid epsilon."""
        valid, msg = epsilon_advisor.validate_epsilon(0.3, SensitivityLevel.HIGH)
        assert valid
        assert "within" in msg.lower()

    def test_validate_epsilon_below_range(self, epsilon_advisor):
        """Test validating epsilon below range."""
        valid, msg = epsilon_advisor.validate_epsilon(0.05, SensitivityLevel.HIGH)
        assert valid  # Still valid, just warning
        assert "below" in msg.lower() or "warning" in msg.lower()

    def test_validate_epsilon_above_range(self, epsilon_advisor):
        """Test validating epsilon above range."""
        valid, msg = epsilon_advisor.validate_epsilon(1.0, SensitivityLevel.HIGH)
        assert valid  # Still valid, just warning
        assert "above" in msg.lower() or "warning" in msg.lower()

    def test_validate_epsilon_too_small(self, epsilon_advisor):
        """Test validating epsilon that is too small."""
        valid, msg = epsilon_advisor.validate_epsilon(0.001, SensitivityLevel.MEDIUM)
        assert not valid
        assert "at least" in msg.lower()

    def test_validate_epsilon_too_large(self, epsilon_advisor):
        """Test validating epsilon that is too large."""
        valid, msg = epsilon_advisor.validate_epsilon(15.0, SensitivityLevel.LOW)
        assert not valid
        assert "exceed" in msg.lower()

    def test_custom_ranges(self):
        """Test custom epsilon ranges."""
        custom_ranges = {SensitivityLevel.HIGH: (0.05, 0.2)}
        advisor = EpsilonAdvisor(custom_ranges=custom_ranges)
        eps_range = advisor.get_epsilon_range(SensitivityLevel.HIGH)
        assert eps_range == (0.05, 0.2)

    def test_custom_defaults(self):
        """Test custom default epsilon values."""
        custom_defaults = {SensitivityLevel.HIGH: 0.15}
        advisor = EpsilonAdvisor(custom_defaults=custom_defaults)
        rec = advisor.recommend_epsilon(SensitivityLevel.HIGH)
        assert rec.epsilon == 0.15


# =============================================================================
# MechanismAdvisor Tests
# =============================================================================


class TestMechanismAdvisor:
    """Tests for MechanismAdvisor class."""

    def test_detect_numeric_bounded(self, mechanism_advisor):
        """Test detecting bounded numeric data."""
        series = pd.Series([1, 2, 3, 4, 5])
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.NUMERIC_BOUNDED

    def test_detect_numeric_with_bounds(self, mechanism_advisor):
        """Test detecting numeric with explicit bounds."""
        series = pd.Series([1, 2, 3, 100, 200])  # Has outliers
        data_type = mechanism_advisor.detect_data_type(series, bounds=(0, 10))
        assert data_type == DataType.NUMERIC_BOUNDED

    def test_detect_numeric_unbounded(self, mechanism_advisor):
        """Test detecting unbounded numeric data with extreme outliers."""
        # Create data with many extreme outliers (>5% outlier ratio)
        base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 normal values
        outliers = [10000, 20000, 30000]  # 3 extreme outliers = 23% outlier ratio
        series = pd.Series(base + outliers)
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.NUMERIC_UNBOUNDED

    def test_detect_categorical(self, mechanism_advisor):
        """Test detecting categorical data."""
        series = pd.Series(["A", "B", "C", "A", "B"])
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.CATEGORICAL

    def test_detect_date(self, mechanism_advisor):
        """Test detecting date data."""
        series = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.DATE

    def test_detect_boolean(self, mechanism_advisor):
        """Test detecting boolean data (treated as categorical)."""
        series = pd.Series([True, False, True, False])
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.CATEGORICAL

    def test_detect_empty_series(self, mechanism_advisor):
        """Test detecting empty series."""
        series = pd.Series([], dtype=float)
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.UNKNOWN

    def test_recommend_laplace_for_bounded(self, mechanism_advisor):
        """Test Laplace recommendation for bounded numeric."""
        rec = mechanism_advisor.recommend_mechanism(DataType.NUMERIC_BOUNDED)
        assert rec.mechanism == RecommendedMechanism.LAPLACE
        assert rec.delta is None

    def test_recommend_gaussian_for_unbounded(self, mechanism_advisor):
        """Test Gaussian recommendation for unbounded numeric."""
        rec = mechanism_advisor.recommend_mechanism(DataType.NUMERIC_UNBOUNDED)
        assert rec.mechanism == RecommendedMechanism.GAUSSIAN
        assert rec.delta == DEFAULT_DELTA

    def test_recommend_exponential_for_categorical(self, mechanism_advisor):
        """Test Exponential recommendation for categorical."""
        rec = mechanism_advisor.recommend_mechanism(DataType.CATEGORICAL)
        assert rec.mechanism == RecommendedMechanism.EXPONENTIAL
        assert rec.delta is None

    def test_recommend_laplace_for_date(self, mechanism_advisor):
        """Test Laplace recommendation for date."""
        rec = mechanism_advisor.recommend_mechanism(DataType.DATE)
        assert rec.mechanism == RecommendedMechanism.LAPLACE

    def test_recommendation_explanation(self, mechanism_advisor):
        """Test that explanation is present."""
        rec = mechanism_advisor.recommend_mechanism(DataType.CATEGORICAL)
        assert len(rec.explanation) > 0
        assert "exponential" in rec.explanation.lower()

    def test_recommend_for_series(self, mechanism_advisor):
        """Test recommending mechanism for a Series."""
        series = pd.Series(["A", "B", "C"])
        rec = mechanism_advisor.recommend_for_series(series)
        assert rec.mechanism == RecommendedMechanism.EXPONENTIAL

    def test_custom_delta(self):
        """Test custom delta value."""
        advisor = MechanismAdvisor(default_delta=1e-6)
        rec = advisor.recommend_mechanism(DataType.NUMERIC_UNBOUNDED)
        assert rec.delta == 1e-6


# =============================================================================
# RecommendationAdvisor Tests
# =============================================================================


class TestRecommendationAdvisor:
    """Tests for RecommendationAdvisor class."""

    def test_recommend_for_high_sensitivity_column(self, recommendation_advisor):
        """Test recommendation for high sensitivity column."""
        rec = recommendation_advisor.recommend_for_column("patient_ssn")
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.HIGH
        assert rec.epsilon_recommendation.epsilon <= 0.5

    def test_recommend_for_medium_sensitivity_column(self, recommendation_advisor):
        """Test recommendation for medium sensitivity column."""
        rec = recommendation_advisor.recommend_for_column("department")
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.MEDIUM

    def test_recommend_for_low_sensitivity_column(self, recommendation_advisor):
        """Test recommendation for low sensitivity column."""
        rec = recommendation_advisor.recommend_for_column("record_id")
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.LOW
        assert rec.epsilon_recommendation.epsilon >= 2.0

    def test_recommend_with_series(self, recommendation_advisor):
        """Test recommendation with column data."""
        series = pd.Series(["A", "B", "C", "A", "B"])
        rec = recommendation_advisor.recommend_for_column("category", series=series)
        assert rec.mechanism_recommendation.mechanism == RecommendedMechanism.EXPONENTIAL

    def test_recommend_with_bounds(self, recommendation_advisor):
        """Test recommendation with bounds."""
        series = pd.Series([1, 2, 3, 100])  # Has outlier
        rec = recommendation_advisor.recommend_for_column(
            "value", series=series, bounds=(0, 10)
        )
        assert rec.mechanism_recommendation.data_type == DataType.NUMERIC_BOUNDED

    def test_prefer_privacy(self, recommendation_advisor):
        """Test prefer_privacy option."""
        rec = recommendation_advisor.recommend_for_column(
            "department", prefer_privacy=True
        )
        assert rec.epsilon_recommendation.epsilon == 0.5  # Lower end of MEDIUM range

    def test_prefer_utility(self, recommendation_advisor):
        """Test prefer_utility option."""
        rec = recommendation_advisor.recommend_for_column(
            "department", prefer_utility=True
        )
        assert rec.epsilon_recommendation.epsilon == 2.0  # Upper end of MEDIUM range

    def test_recommend_for_dataset(self, recommendation_advisor, sample_df):
        """Test recommendation for entire dataset."""
        recs = recommendation_advisor.recommend_for_dataset(sample_df)
        assert len(recs.column_recommendations) == len(sample_df.columns)
        assert "patient_ssn" in recs.high_sensitivity_columns

    def test_override_epsilon(self, recommendation_advisor):
        """Test overriding epsilon for a column."""
        recommendation_advisor.add_override("custom_col", epsilon=0.5)
        rec = recommendation_advisor.recommend_for_column("custom_col")
        assert rec.epsilon_recommendation.epsilon == 0.5
        assert rec.is_override

    def test_override_mechanism(self, recommendation_advisor):
        """Test overriding mechanism for a column."""
        recommendation_advisor.add_override(
            "custom_col", mechanism=RecommendedMechanism.GAUSSIAN
        )
        rec = recommendation_advisor.recommend_for_column("custom_col")
        assert rec.mechanism_recommendation.mechanism == RecommendedMechanism.GAUSSIAN

    def test_override_sensitivity(self, recommendation_advisor):
        """Test overriding sensitivity for a column."""
        recommendation_advisor.add_override(
            "record_id", sensitivity=SensitivityLevel.HIGH
        )
        rec = recommendation_advisor.recommend_for_column("record_id")
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.HIGH
        assert rec.epsilon_recommendation.epsilon <= 0.5

    def test_remove_override(self, recommendation_advisor):
        """Test removing an override."""
        recommendation_advisor.add_override("col", epsilon=0.1)
        assert recommendation_advisor.remove_override("col")
        # After removing, should use default classification
        rec = recommendation_advisor.recommend_for_column("col")
        assert not rec.is_override

    def test_clear_overrides(self, recommendation_advisor):
        """Test clearing all overrides."""
        recommendation_advisor.add_override("col1", epsilon=0.1)
        recommendation_advisor.add_override("col2", epsilon=0.2)
        recommendation_advisor.clear_overrides()
        rec = recommendation_advisor.recommend_for_column("col1")
        assert not rec.is_override

    def test_utility_level_low(self, recommendation_advisor):
        """Test utility level LOW (high privacy, lower epsilon)."""
        # Using prefer_privacy gets lower end of range = lower utility
        rec = recommendation_advisor.recommend_for_column(
            "patient_ssn", prefer_privacy=True
        )
        assert rec.utility_level == UtilityLevel.LOW

    def test_utility_level_high(self, recommendation_advisor):
        """Test utility level HIGH (lower privacy, higher epsilon)."""
        # Using prefer_utility gets upper end of range = higher utility
        rec = recommendation_advisor.recommend_for_column(
            "patient_ssn", prefer_utility=True
        )
        assert rec.utility_level == UtilityLevel.HIGH

    def test_overall_explanation(self, recommendation_advisor):
        """Test that overall explanation is generated."""
        rec = recommendation_advisor.recommend_for_column("patient_ssn")
        assert len(rec.overall_explanation) > 0
        assert "patient_ssn" in rec.overall_explanation.lower()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_recommend_epsilon(self):
        """Test recommend_epsilon function."""
        rec = recommend_epsilon(SensitivityLevel.HIGH)
        assert rec.sensitivity == SensitivityLevel.HIGH
        assert 0.1 <= rec.epsilon <= 0.5

    def test_recommend_epsilon_prefer_privacy(self):
        """Test recommend_epsilon with prefer_privacy."""
        rec = recommend_epsilon(SensitivityLevel.MEDIUM, prefer_privacy=True)
        assert rec.epsilon == 0.5

    def test_recommend_mechanism(self):
        """Test recommend_mechanism function."""
        rec = recommend_mechanism(DataType.CATEGORICAL)
        assert rec.mechanism == RecommendedMechanism.EXPONENTIAL

    def test_recommend_for_column(self):
        """Test recommend_for_column function."""
        rec = recommend_for_column("patient_ssn")
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.HIGH

    def test_recommend_for_column_with_series(self):
        """Test recommend_for_column with series."""
        series = pd.Series([1, 2, 3, 4, 5])
        rec = recommend_for_column("value", series=series)
        assert rec.mechanism_recommendation.mechanism == RecommendedMechanism.LAPLACE

    def test_recommend_for_dataset(self, sample_df):
        """Test recommend_for_dataset function."""
        recs = recommend_for_dataset(sample_df)
        assert len(recs.column_recommendations) == len(sample_df.columns)

    def test_recommend_for_dataset_prefer_privacy(self, sample_df):
        """Test recommend_for_dataset with prefer_privacy."""
        recs = recommend_for_dataset(sample_df, prefer_privacy=True)
        # All epsilons should be at the lower end of their ranges
        for rec in recs.column_recommendations.values():
            eps_range = rec.epsilon_recommendation.epsilon_range
            assert rec.epsilon_recommendation.epsilon == eps_range[0]

    def test_get_epsilon_for_column(self):
        """Test get_epsilon_for_column function."""
        epsilon = get_epsilon_for_column("patient_ssn")
        assert 0.1 <= epsilon <= 0.5

    def test_get_mechanism_for_series(self):
        """Test get_mechanism_for_series function."""
        series = pd.Series(["A", "B", "C"])
        mech = get_mechanism_for_series(series)
        assert mech == RecommendedMechanism.EXPONENTIAL

    def test_validate_epsilon_valid(self):
        """Test validate_epsilon with valid value."""
        valid, msg = validate_epsilon(0.3, SensitivityLevel.HIGH)
        assert valid
        assert "within" in msg.lower()

    def test_validate_epsilon_warning(self):
        """Test validate_epsilon with out-of-range value."""
        valid, msg = validate_epsilon(1.0, SensitivityLevel.HIGH)
        assert valid
        assert "above" in msg.lower() or "warning" in msg.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the advisor module."""

    def test_full_workflow(self, sample_df):
        """Test complete recommendation workflow."""
        # Get recommendations
        recs = recommend_for_dataset(sample_df)

        # Verify high sensitivity columns are identified
        assert "patient_ssn" in recs.high_sensitivity_columns
        assert "patient_name" in recs.high_sensitivity_columns

        # Verify total epsilon is calculated
        assert recs.total_epsilon > 0

        # Verify mechanisms are appropriate
        by_mech = recs.columns_by_mechanism
        assert "department" in by_mech[RecommendedMechanism.EXPONENTIAL]
        assert "visit_date" in by_mech[RecommendedMechanism.LAPLACE]

        # Verify DataFrame export
        df_summary = recs.to_dataframe()
        assert len(df_summary) == len(sample_df.columns)

    def test_recommendation_matches_prd(self):
        """Test that recommendations match PRD specifications."""
        # PRD: High (ε = 0.1–0.5)
        rec_high = recommend_for_column("patient_ssn")
        assert 0.1 <= rec_high.epsilon_recommendation.epsilon <= 0.5

        # PRD: Medium (ε = 0.5–2.0)
        rec_med = recommend_for_column("department")
        assert 0.5 <= rec_med.epsilon_recommendation.epsilon <= 2.0

        # PRD: Low (ε = 2.0–5.0)
        rec_low = recommend_for_column("record_id")
        assert 2.0 <= rec_low.epsilon_recommendation.epsilon <= 5.0

    def test_mechanism_matches_prd(self):
        """Test that mechanism recommendations match PRD specifications."""
        # PRD: Laplace for bounded numeric
        series_bounded = pd.Series([1, 2, 3, 4, 5])
        mech = get_mechanism_for_series(series_bounded)
        assert mech == RecommendedMechanism.LAPLACE

        # PRD: Exponential for categorical
        series_cat = pd.Series(["A", "B", "C"])
        mech = get_mechanism_for_series(series_cat)
        assert mech == RecommendedMechanism.EXPONENTIAL


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, recommendation_advisor):
        """Test recommendations for empty DataFrame."""
        df = pd.DataFrame()
        recs = recommendation_advisor.recommend_for_dataset(df)
        assert len(recs.column_recommendations) == 0
        assert recs.total_epsilon == 0.0

    def test_single_column_dataframe(self, recommendation_advisor):
        """Test recommendations for single column DataFrame."""
        df = pd.DataFrame({"ssn": ["123-45-6789"]})
        recs = recommendation_advisor.recommend_for_dataset(df)
        assert len(recs.column_recommendations) == 1

    def test_all_null_column(self, mechanism_advisor):
        """Test mechanism detection for all-null column."""
        series = pd.Series([None, None, None])
        data_type = mechanism_advisor.detect_data_type(series)
        assert data_type == DataType.UNKNOWN

    def test_unicode_column_name(self, recommendation_advisor):
        """Test recommendation for unicode column name."""
        rec = recommendation_advisor.recommend_for_column("患者名")  # "Patient name"
        # Should get default medium sensitivity
        assert rec.epsilon_recommendation.sensitivity == SensitivityLevel.MEDIUM

    def test_mixed_types_series(self, mechanism_advisor):
        """Test mechanism detection for mixed types."""
        series = pd.Series([1, "A", 2, "B"])
        data_type = mechanism_advisor.detect_data_type(series)
        # Object dtype should be treated as categorical
        assert data_type == DataType.CATEGORICAL

    def test_large_dataframe_performance(self, recommendation_advisor):
        """Test performance with large DataFrame."""
        import time

        # Create DataFrame with many columns
        data = {f"col_{i}": [1, 2, 3] for i in range(100)}
        df = pd.DataFrame(data)

        start = time.time()
        recs = recommendation_advisor.recommend_for_dataset(df)
        elapsed = time.time() - start

        assert len(recs.column_recommendations) == 100
        assert elapsed < 5.0  # Should complete in under 5 seconds
