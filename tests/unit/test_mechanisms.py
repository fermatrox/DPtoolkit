"""Unit tests for DP mechanisms."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dp_toolkit.core.mechanisms import (
    EPSILON_MAX,
    EPSILON_MIN,
    LaplaceMechanism,
    PrivacyUsage,
    add_laplace_noise,
    add_laplace_noise_array,
    calculate_scale_laplace,
    calculate_sensitivity_bounded,
    create_laplace_mechanism,
    validate_bounds,
    validate_epsilon,
)


class TestValidateEpsilon:
    """Tests for epsilon validation."""

    def test_valid_epsilon(self):
        """Test valid epsilon values."""
        validate_epsilon(0.1)
        validate_epsilon(1.0)
        validate_epsilon(5.0)
        validate_epsilon(EPSILON_MIN)
        validate_epsilon(EPSILON_MAX)

    def test_epsilon_zero(self):
        """Test epsilon = 0 raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_epsilon(0.0)

    def test_epsilon_negative(self):
        """Test negative epsilon raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_epsilon(-1.0)

    def test_epsilon_too_small(self):
        """Test epsilon below minimum raises error."""
        with pytest.raises(ValueError, match="below minimum"):
            validate_epsilon(0.001)

    def test_epsilon_too_large(self):
        """Test epsilon above maximum raises error."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_epsilon(100.0)

    def test_epsilon_wrong_type(self):
        """Test non-numeric epsilon raises error."""
        with pytest.raises(TypeError):
            validate_epsilon("1.0")


class TestValidateBounds:
    """Tests for bounds validation."""

    def test_valid_bounds(self):
        """Test valid bounds."""
        validate_bounds(0.0, 100.0)
        validate_bounds(-50.0, 50.0)
        validate_bounds(0.0, 1.0)

    def test_bounds_equal(self):
        """Test equal bounds raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            validate_bounds(50.0, 50.0)

    def test_bounds_reversed(self):
        """Test reversed bounds raises error."""
        with pytest.raises(ValueError, match="must be less than"):
            validate_bounds(100.0, 0.0)

    def test_bounds_nan(self):
        """Test NaN bounds raises error."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_bounds(float("nan"), 100.0)
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_bounds(0.0, float("nan"))

    def test_bounds_infinite(self):
        """Test infinite bounds raises error."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_bounds(float("-inf"), 100.0)
        with pytest.raises(ValueError, match="must be finite"):
            validate_bounds(0.0, float("inf"))


class TestSensitivityCalculation:
    """Tests for sensitivity calculation."""

    def test_basic_sensitivity(self):
        """Test basic sensitivity calculation."""
        assert calculate_sensitivity_bounded(0.0, 100.0) == 100.0
        assert calculate_sensitivity_bounded(-50.0, 50.0) == 100.0
        assert calculate_sensitivity_bounded(0.0, 1.0) == 1.0

    def test_sensitivity_negative_bounds(self):
        """Test sensitivity with negative bounds."""
        assert calculate_sensitivity_bounded(-100.0, -50.0) == 50.0

    def test_sensitivity_small_range(self):
        """Test sensitivity for small range."""
        sens = calculate_sensitivity_bounded(0.0, 0.001)
        assert abs(sens - 0.001) < 1e-10


class TestScaleCalculation:
    """Tests for Laplace scale calculation."""

    def test_basic_scale(self):
        """Test basic scale calculation."""
        # scale = sensitivity / epsilon
        assert calculate_scale_laplace(100.0, 1.0) == 100.0
        assert calculate_scale_laplace(100.0, 2.0) == 50.0
        assert calculate_scale_laplace(50.0, 0.5) == 100.0

    def test_scale_small_epsilon(self):
        """Test scale with small epsilon (more noise)."""
        scale = calculate_scale_laplace(100.0, 0.1)
        assert scale == 1000.0

    def test_scale_invalid_sensitivity(self):
        """Test scale with invalid sensitivity."""
        with pytest.raises(ValueError, match="must be positive"):
            calculate_scale_laplace(0.0, 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            calculate_scale_laplace(-1.0, 1.0)


class TestPrivacyUsage:
    """Tests for PrivacyUsage dataclass."""

    def test_pure_dp(self):
        """Test pure epsilon-DP usage."""
        usage = PrivacyUsage(epsilon=1.0)
        assert usage.epsilon == 1.0
        assert usage.delta is None
        assert usage.is_pure_dp is True

    def test_approximate_dp(self):
        """Test approximate (ε,δ)-DP usage."""
        usage = PrivacyUsage(epsilon=1.0, delta=1e-5)
        assert usage.epsilon == 1.0
        assert usage.delta == 1e-5
        assert usage.is_pure_dp is False

    def test_zero_delta(self):
        """Test zero delta is considered pure DP."""
        usage = PrivacyUsage(epsilon=1.0, delta=0.0)
        assert usage.is_pure_dp is True


class TestLaplaceMechanism:
    """Tests for LaplaceMechanism class."""

    def test_mechanism_creation(self):
        """Test creating a Laplace mechanism."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        assert mechanism.lower == 0.0
        assert mechanism.upper == 100.0
        assert mechanism.epsilon == 1.0
        assert mechanism.sensitivity == 100.0
        assert mechanism.scale == 100.0

    def test_mechanism_properties(self):
        """Test mechanism property calculations."""
        mechanism = LaplaceMechanism(lower=0.0, upper=50.0, epsilon=0.5)

        assert mechanism.sensitivity == 50.0
        assert mechanism.scale == 100.0  # 50 / 0.5

    def test_release_single_value(self):
        """Test releasing a single value."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        result = mechanism.release(50.0)

        assert isinstance(result, float)
        # Result should be noisy but not extremely far from input
        # With scale=100, 99% of samples are within ~460 of the true value
        assert -500 < result < 600

    def test_release_array(self):
        """Test releasing an array of values."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = mechanism.release_array(values)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_release_array_pandas(self):
        """Test releasing a pandas Series."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        series = pd.Series([10.0, 20.0, 30.0])
        result = mechanism.release_array(series)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_release_array_numpy(self):
        """Test releasing a numpy array."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        arr = np.array([10.0, 20.0, 30.0])
        result = mechanism.release_array(arr)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_clamp_value(self):
        """Test clamping values to bounds."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        assert mechanism.clamp(50.0) == 50.0
        assert mechanism.clamp(-10.0) == 0.0
        assert mechanism.clamp(150.0) == 100.0

    def test_clamp_array(self):
        """Test clamping array to bounds."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        values = [-10.0, 50.0, 150.0]
        result = mechanism.clamp_array(values)

        np.testing.assert_array_equal(result, [0.0, 50.0, 100.0])

    def test_release_clamped(self):
        """Test release with clamping."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        # Value outside bounds should be clamped before noise
        result = mechanism.release_clamped(150.0)
        assert isinstance(result, float)

    def test_release_array_clamped(self):
        """Test array release with clamping."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        values = [-10.0, 50.0, 150.0]
        result = mechanism.release_array_clamped(values)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_privacy_usage(self):
        """Test getting privacy usage."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=0.5)
        usage = mechanism.get_privacy_usage()

        assert usage.epsilon == 0.5
        assert usage.is_pure_dp is True

    def test_invalid_epsilon(self):
        """Test mechanism with invalid epsilon."""
        with pytest.raises(ValueError):
            LaplaceMechanism(lower=0.0, upper=100.0, epsilon=0.0)

    def test_invalid_bounds(self):
        """Test mechanism with invalid bounds."""
        with pytest.raises(ValueError):
            LaplaceMechanism(lower=100.0, upper=0.0, epsilon=1.0)

    def test_release_nan_error(self):
        """Test releasing NaN raises error."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        with pytest.raises(ValueError, match="cannot be NaN"):
            mechanism.release(float("nan"))

    def test_release_array_nan_error(self):
        """Test releasing array with NaN raises error."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        with pytest.raises(ValueError, match="contains NaN"):
            mechanism.release_array([1.0, float("nan"), 3.0])

    def test_release_wrong_type(self):
        """Test releasing wrong type raises error."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        with pytest.raises(TypeError):
            mechanism.release("50")

    def test_repr(self):
        """Test string representation."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        repr_str = repr(mechanism)

        assert "LaplaceMechanism" in repr_str
        assert "lower=0.0" in repr_str
        assert "upper=100.0" in repr_str
        assert "epsilon=1.0" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_laplace_mechanism(self):
        """Test create_laplace_mechanism function."""
        mechanism = create_laplace_mechanism(
            lower=0.0, upper=100.0, epsilon=1.0
        )
        assert isinstance(mechanism, LaplaceMechanism)
        assert mechanism.epsilon == 1.0

    def test_add_laplace_noise(self):
        """Test add_laplace_noise function."""
        result = add_laplace_noise(
            value=50.0, lower=0.0, upper=100.0, epsilon=1.0
        )
        assert isinstance(result, float)

    def test_add_laplace_noise_array(self):
        """Test add_laplace_noise_array function."""
        values = [10.0, 20.0, 30.0]
        result = add_laplace_noise_array(
            values=values, lower=0.0, upper=100.0, epsilon=1.0
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestLaplaceDistribution:
    """Tests verifying Laplace distribution properties."""

    def test_noise_is_laplace_distributed(self):
        """Test that noise follows Laplace distribution (statistical test)."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        # Generate many noisy samples from the same true value
        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]

        # The noise should be Laplace distributed
        noise = np.array(samples) - true_value

        # Perform Kolmogorov-Smirnov test against Laplace distribution
        # scale = sensitivity / epsilon = 100 / 1 = 100
        expected_scale = 100.0
        ks_stat, p_value = stats.kstest(
            noise, stats.laplace(loc=0, scale=expected_scale).cdf
        )

        # p-value should be > 0.01 (not rejecting Laplace hypothesis)
        assert p_value > 0.01, f"KS test failed: p-value = {p_value}"

    def test_noise_mean_near_zero(self):
        """Test that noise has approximately zero mean."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]
        noise = np.array(samples) - true_value

        # Mean should be close to 0
        mean_noise = np.mean(noise)
        assert abs(mean_noise) < 5.0, f"Mean noise = {mean_noise}, expected ~0"

    def test_noise_scale_matches_parameters(self):
        """Test that noise scale matches expected value."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]
        noise = np.array(samples) - true_value

        # For Laplace, mean absolute deviation = scale
        # With scale = 100, MAD should be approximately 100
        mad = np.mean(np.abs(noise))
        expected_scale = 100.0

        # Allow 10% tolerance
        assert abs(mad - expected_scale) / expected_scale < 0.1

    def test_higher_epsilon_less_noise(self):
        """Test that higher epsilon produces less noise."""
        true_value = 50.0
        n_samples = 1000

        # Low privacy (high epsilon, less noise)
        mech_low_privacy = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=5.0)
        samples_low = [mech_low_privacy.release(true_value) for _ in range(n_samples)]
        noise_low = np.abs(np.array(samples_low) - true_value)

        # High privacy (low epsilon, more noise)
        mech_high_privacy = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=0.5)
        samples_high = [mech_high_privacy.release(true_value) for _ in range(n_samples)]
        noise_high = np.abs(np.array(samples_high) - true_value)

        # High privacy should have more noise
        assert np.mean(noise_high) > np.mean(noise_low)


class TestPrivacyGuarantee:
    """Tests for privacy guarantee verification."""

    def test_epsilon_bounds_respected(self):
        """Test that privacy map confirms epsilon bounds."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        # The privacy usage should match the epsilon we specified
        usage = mechanism.get_privacy_usage()
        assert usage.epsilon == 1.0

    def test_sensitivity_bounds_respected(self):
        """Test that sensitivity calculation is correct."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        # Sensitivity should be upper - lower
        assert mechanism.sensitivity == 100.0

        # Different bounds
        mechanism2 = LaplaceMechanism(lower=-50.0, upper=50.0, epsilon=1.0)
        assert mechanism2.sensitivity == 100.0

    def test_independent_noise_per_element(self):
        """Test that each array element gets independent noise."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)

        # Apply to array multiple times
        values = [50.0] * 5
        results = []
        for _ in range(100):
            results.append(mechanism.release_array(values))

        results = np.array(results)

        # Check correlation between elements - should be low
        for i in range(5):
            for j in range(i + 1, 5):
                corr = np.corrcoef(results[:, i], results[:, j])[0, 1]
                assert abs(corr) < 0.2, f"Elements {i},{j} correlated: {corr}"


class TestEdgeCases:
    """Edge case tests for mechanisms."""

    def test_minimum_epsilon(self):
        """Test with minimum epsilon value."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=EPSILON_MIN)
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_maximum_epsilon(self):
        """Test with maximum epsilon value."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=EPSILON_MAX)
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_small_range(self):
        """Test with very small data range."""
        mechanism = LaplaceMechanism(lower=0.0, upper=0.001, epsilon=1.0)
        result = mechanism.release(0.0005)
        assert isinstance(result, float)

    def test_large_range(self):
        """Test with large data range."""
        mechanism = LaplaceMechanism(lower=0.0, upper=1e6, epsilon=1.0)
        result = mechanism.release(500000.0)
        assert isinstance(result, float)

    def test_negative_bounds(self):
        """Test with negative bounds."""
        mechanism = LaplaceMechanism(lower=-100.0, upper=-50.0, epsilon=1.0)
        result = mechanism.release(-75.0)
        assert isinstance(result, float)
        assert mechanism.sensitivity == 50.0

    def test_empty_array(self):
        """Test with empty array."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        result = mechanism.release_array([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_single_element_array(self):
        """Test with single element array."""
        mechanism = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        result = mechanism.release_array([50.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_integer_input(self):
        """Test with integer input values."""
        mechanism = LaplaceMechanism(lower=0, upper=100, epsilon=1)
        result = mechanism.release(50)
        assert isinstance(result, float)

        result_array = mechanism.release_array([10, 20, 30])
        assert isinstance(result_array, np.ndarray)
