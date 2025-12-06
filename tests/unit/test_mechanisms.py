"""Unit tests for DP mechanisms."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dp_toolkit.core.mechanisms import (
    DELTA_MAX,
    DELTA_MIN,
    EPSILON_MAX,
    EPSILON_MIN,
    ExponentialMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyUsage,
    add_gaussian_noise,
    add_gaussian_noise_array,
    add_laplace_noise,
    add_laplace_noise_array,
    calculate_epsilon_from_rho_delta,
    calculate_rho_from_epsilon_delta,
    calculate_scale_exponential,
    calculate_scale_gaussian,
    calculate_scale_laplace,
    calculate_sensitivity_bounded,
    create_exponential_mechanism,
    create_gaussian_mechanism,
    create_laplace_mechanism,
    sample_categories,
    select_category,
    validate_bounds,
    validate_delta,
    validate_epsilon,
    validate_sensitivity,
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

        # Apply to array multiple times (more samples for stable statistics)
        values = [50.0] * 5
        results = []
        for _ in range(500):
            results.append(mechanism.release_array(values))

        results = np.array(results)

        # Check correlation between elements - should be low
        # With 500 samples, spurious correlations should be < 0.1
        for i in range(5):
            for j in range(i + 1, 5):
                corr = np.corrcoef(results[:, i], results[:, j])[0, 1]
                assert abs(corr) < 0.15, f"Elements {i},{j} correlated: {corr}"


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


# =============================================================================
# Gaussian Mechanism Tests
# =============================================================================


class TestValidateDelta:
    """Tests for delta validation."""

    def test_valid_delta(self):
        """Test valid delta values."""
        validate_delta(1e-5)
        validate_delta(1e-6)
        validate_delta(1e-9)
        validate_delta(DELTA_MIN)
        validate_delta(DELTA_MAX)

    def test_delta_zero(self):
        """Test delta = 0 raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_delta(0.0)

    def test_delta_negative(self):
        """Test negative delta raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_delta(-1e-5)

    def test_delta_too_small(self):
        """Test delta below minimum raises error."""
        with pytest.raises(ValueError, match="below minimum"):
            validate_delta(1e-15)

    def test_delta_too_large(self):
        """Test delta above maximum raises error."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_delta(0.1)

    def test_delta_wrong_type(self):
        """Test non-numeric delta raises error."""
        with pytest.raises(TypeError):
            validate_delta("1e-5")


class TestValidateSensitivity:
    """Tests for sensitivity validation."""

    def test_valid_sensitivity(self):
        """Test valid sensitivity values."""
        validate_sensitivity(1.0)
        validate_sensitivity(100.0)
        validate_sensitivity(0.001)

    def test_sensitivity_zero(self):
        """Test sensitivity = 0 raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_sensitivity(0.0)

    def test_sensitivity_negative(self):
        """Test negative sensitivity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_sensitivity(-1.0)

    def test_sensitivity_nan(self):
        """Test NaN sensitivity raises error."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_sensitivity(float("nan"))

    def test_sensitivity_infinite(self):
        """Test infinite sensitivity raises error."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_sensitivity(float("inf"))

    def test_sensitivity_wrong_type(self):
        """Test non-numeric sensitivity raises error."""
        with pytest.raises(TypeError):
            validate_sensitivity("1.0")


class TestGaussianScaleCalculation:
    """Tests for Gaussian scale calculation."""

    def test_basic_scale(self):
        """Test basic scale calculation."""
        sensitivity = 10.0
        epsilon = 1.0
        delta = 1e-5

        scale = calculate_scale_gaussian(sensitivity, epsilon, delta)
        assert scale > 0
        # Scale should be roughly sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        # For these values, approximately 48-50
        assert 40 < scale < 60

    def test_scale_exact_epsilon(self):
        """Test that calculated scale achieves exact target epsilon."""
        sensitivity = 10.0
        epsilon = 1.0
        delta = 1e-5

        scale = calculate_scale_gaussian(sensitivity, epsilon, delta)

        # Verify: rho = (sensitivity/scale)^2 / 2
        rho = (sensitivity / scale) ** 2 / 2
        achieved_epsilon = calculate_epsilon_from_rho_delta(rho, delta)

        # Should achieve exactly the target epsilon
        assert abs(achieved_epsilon - epsilon) < 1e-10

    def test_higher_epsilon_smaller_scale(self):
        """Test that higher epsilon produces smaller scale."""
        sensitivity = 10.0
        delta = 1e-5

        scale_low_eps = calculate_scale_gaussian(sensitivity, 0.5, delta)
        scale_high_eps = calculate_scale_gaussian(sensitivity, 2.0, delta)

        assert scale_low_eps > scale_high_eps

    def test_higher_sensitivity_larger_scale(self):
        """Test that higher sensitivity produces larger scale."""
        epsilon = 1.0
        delta = 1e-5

        scale_low_sens = calculate_scale_gaussian(5.0, epsilon, delta)
        scale_high_sens = calculate_scale_gaussian(20.0, epsilon, delta)

        assert scale_high_sens > scale_low_sens


class TestRhoConversions:
    """Tests for zCDP rho conversions."""

    def test_rho_roundtrip(self):
        """Test rho -> epsilon -> rho roundtrip."""
        epsilon = 1.0
        delta = 1e-5

        rho = calculate_rho_from_epsilon_delta(epsilon, delta)
        recovered_epsilon = calculate_epsilon_from_rho_delta(rho, delta)

        assert abs(recovered_epsilon - epsilon) < 1e-10

    def test_rho_positive(self):
        """Test that rho is always positive."""
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for delta in [1e-9, 1e-7, 1e-5, 1e-4]:
                rho = calculate_rho_from_epsilon_delta(eps, delta)
                assert rho > 0

    def test_higher_epsilon_higher_rho(self):
        """Test that higher epsilon produces higher rho."""
        delta = 1e-5

        rho_low = calculate_rho_from_epsilon_delta(0.5, delta)
        rho_high = calculate_rho_from_epsilon_delta(2.0, delta)

        assert rho_high > rho_low


class TestGaussianMechanism:
    """Tests for Gaussian mechanism."""

    def test_mechanism_creation(self):
        """Test mechanism creation."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        assert mechanism is not None

    def test_mechanism_properties(self):
        """Test mechanism property accessors."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        assert mechanism.sensitivity == 10.0
        assert mechanism.epsilon == 1.0
        assert mechanism.delta == 1e-5
        assert mechanism.scale > 0
        assert mechanism.rho > 0

    def test_release_single_value(self):
        """Test releasing a single value."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        result = mechanism.release(100.0)
        assert isinstance(result, float)

    def test_release_array(self):
        """Test releasing an array of values."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = mechanism.release_array(values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_release_array_pandas(self):
        """Test releasing a pandas Series."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        values = pd.Series([10.0, 20.0, 30.0])
        result = mechanism.release_array(values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_release_array_numpy(self):
        """Test releasing a numpy array."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        values = np.array([10.0, 20.0, 30.0])
        result = mechanism.release_array(values)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_privacy_usage(self):
        """Test privacy usage reporting."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        usage = mechanism.get_privacy_usage()
        assert usage.epsilon == 1.0
        assert usage.delta == 1e-5
        assert not usage.is_pure_dp

    def test_achieved_epsilon(self):
        """Test achieved epsilon calculation."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        achieved = mechanism.get_achieved_epsilon()
        assert abs(achieved - 1.0) < 1e-10

    def test_achieved_epsilon_different_delta(self):
        """Test achieved epsilon with different delta."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        # With smaller delta, achieved epsilon should be higher
        achieved_smaller_delta = mechanism.get_achieved_epsilon(1e-7)
        achieved_larger_delta = mechanism.get_achieved_epsilon(1e-4)

        assert achieved_smaller_delta > achieved_larger_delta

    def test_invalid_epsilon(self):
        """Test invalid epsilon raises error."""
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=10.0, epsilon=0.0, delta=1e-5)

    def test_invalid_delta(self):
        """Test invalid delta raises error."""
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=10.0, epsilon=1.0, delta=0.0)

    def test_invalid_sensitivity(self):
        """Test invalid sensitivity raises error."""
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=0.0, epsilon=1.0, delta=1e-5)

    def test_release_nan_error(self):
        """Test that NaN input raises error."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        with pytest.raises(ValueError, match="cannot be NaN"):
            mechanism.release(float("nan"))

    def test_release_array_nan_error(self):
        """Test that array with NaN raises error."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        with pytest.raises(ValueError, match="contains NaN"):
            mechanism.release_array([1.0, float("nan"), 3.0])

    def test_release_wrong_type(self):
        """Test that wrong type input raises error."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        with pytest.raises(TypeError):
            mechanism.release("not a number")

    def test_repr(self):
        """Test string representation."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        repr_str = repr(mechanism)
        assert "GaussianMechanism" in repr_str
        assert "sensitivity=10.0" in repr_str
        assert "epsilon=1.0" in repr_str
        assert "delta=1e-05" in repr_str


class TestGaussianConvenienceFunctions:
    """Tests for Gaussian convenience functions."""

    def test_create_gaussian_mechanism(self):
        """Test create_gaussian_mechanism function."""
        mechanism = create_gaussian_mechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        assert isinstance(mechanism, GaussianMechanism)
        assert mechanism.epsilon == 1.0

    def test_add_gaussian_noise(self):
        """Test add_gaussian_noise function."""
        result = add_gaussian_noise(
            value=50.0, sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        assert isinstance(result, float)

    def test_add_gaussian_noise_array(self):
        """Test add_gaussian_noise_array function."""
        values = [10.0, 20.0, 30.0]
        result = add_gaussian_noise_array(
            values=values, sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestGaussianDistribution:
    """Tests verifying Gaussian distribution properties."""

    def test_noise_is_gaussian_distributed(self):
        """Test that noise follows Gaussian distribution (statistical test)."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )

        # Generate many noisy samples from the same true value
        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]

        # The noise should be Gaussian distributed
        noise = np.array(samples) - true_value

        # Perform Kolmogorov-Smirnov test against normal distribution
        expected_scale = mechanism.scale
        ks_stat, p_value = stats.kstest(
            noise, stats.norm(loc=0, scale=expected_scale).cdf
        )

        # p-value should be > 0.01 (not rejecting Gaussian hypothesis)
        assert p_value > 0.01, f"KS test failed: p-value = {p_value}"

    def test_noise_mean_near_zero(self):
        """Test that noise has approximately zero mean."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )

        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]
        noise = np.array(samples) - true_value

        # Mean should be close to 0
        mean_noise = np.mean(noise)
        # Allow for statistical variation (3 standard errors)
        se = mechanism.scale / math.sqrt(n_samples)
        assert abs(mean_noise) < 3 * se, f"Mean noise = {mean_noise}, expected ~0"

    def test_noise_std_matches_scale(self):
        """Test that noise standard deviation matches scale."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )

        true_value = 50.0
        n_samples = 10000
        samples = [mechanism.release(true_value) for _ in range(n_samples)]
        noise = np.array(samples) - true_value

        # Standard deviation should match scale
        observed_std = np.std(noise)
        expected_std = mechanism.scale

        # Allow 10% tolerance
        assert abs(observed_std - expected_std) / expected_std < 0.1

    def test_higher_epsilon_less_noise(self):
        """Test that higher epsilon produces less noise."""
        true_value = 50.0
        n_samples = 1000
        sensitivity = 10.0
        delta = 1e-5

        # Low privacy (high epsilon, less noise)
        mech_low_privacy = GaussianMechanism(
            sensitivity=sensitivity, epsilon=5.0, delta=delta
        )
        samples_low = [
            mech_low_privacy.release(true_value) for _ in range(n_samples)
        ]
        noise_low = np.abs(np.array(samples_low) - true_value)

        # High privacy (low epsilon, more noise)
        mech_high_privacy = GaussianMechanism(
            sensitivity=sensitivity, epsilon=0.5, delta=delta
        )
        samples_high = [
            mech_high_privacy.release(true_value) for _ in range(n_samples)
        ]
        noise_high = np.abs(np.array(samples_high) - true_value)

        # High privacy should have more noise
        assert np.mean(noise_high) > np.mean(noise_low)


class TestGaussianPrivacyGuarantee:
    """Tests for Gaussian (ε,δ)-DP privacy guarantee verification."""

    def test_epsilon_delta_bounds_respected(self):
        """Test that privacy usage matches specified parameters."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )

        usage = mechanism.get_privacy_usage()
        assert usage.epsilon == 1.0
        assert usage.delta == 1e-5

    def test_achieved_epsilon_matches_target(self):
        """Test that achieved epsilon matches target epsilon."""
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            mechanism = GaussianMechanism(
                sensitivity=10.0, epsilon=eps, delta=1e-5
            )
            achieved = mechanism.get_achieved_epsilon()
            assert abs(achieved - eps) < 1e-9, f"eps={eps}, achieved={achieved}"

    def test_independent_noise_per_element(self):
        """Test that each array element gets independent noise."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )

        # Apply to array multiple times (more samples for stable statistics)
        values = [50.0] * 5
        results = []
        for _ in range(500):
            results.append(mechanism.release_array(values))

        results = np.array(results)

        # Check correlation between elements - should be low
        # With 500 samples, spurious correlations should be < 0.1
        for i in range(5):
            for j in range(i + 1, 5):
                corr = np.corrcoef(results[:, i], results[:, j])[0, 1]
                assert abs(corr) < 0.15, f"Elements {i},{j} correlated: {corr}"


class TestGaussianVsLaplaceComparison:
    """Tests comparing Gaussian and Laplace mechanisms."""

    def test_gaussian_vs_laplace_noise_level(self):
        """Test noise level comparison for same epsilon.

        Note: Gaussian requires delta > 0, while Laplace is pure DP.
        For the same epsilon, Gaussian typically has higher noise due to delta.
        """
        epsilon = 1.0
        sensitivity = 100.0  # Same as range for Laplace

        # Laplace mechanism
        laplace = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=epsilon)

        # Gaussian mechanism (needs delta)
        gaussian = GaussianMechanism(
            sensitivity=sensitivity, epsilon=epsilon, delta=1e-5
        )

        # Compare scales - Gaussian scale is std dev, Laplace scale is b
        # For Laplace, std = sqrt(2) * b
        laplace_std = math.sqrt(2) * laplace.scale
        gaussian_std = gaussian.scale

        # Gaussian typically has higher std for same epsilon due to delta
        # This is the tradeoff - Gaussian gives approximate DP but needs more noise
        # The ratio depends on delta - smaller delta means more noise
        print(f"Laplace std: {laplace_std:.2f}, Gaussian std: {gaussian_std:.2f}")

    def test_both_mechanisms_produce_similar_utility(self):
        """Test that both mechanisms produce reasonable utility."""
        true_value = 50.0
        n_samples = 1000

        laplace = LaplaceMechanism(lower=0.0, upper=100.0, epsilon=1.0)
        gaussian = GaussianMechanism(
            sensitivity=100.0, epsilon=1.0, delta=1e-5
        )

        laplace_samples = [laplace.release(true_value) for _ in range(n_samples)]
        gaussian_samples = [
            gaussian.release(true_value) for _ in range(n_samples)
        ]

        # Both should have mean close to true value
        laplace_mean = np.mean(laplace_samples)
        gaussian_mean = np.mean(gaussian_samples)

        # Allow for statistical variation (high sensitivity = high variance)
        assert abs(laplace_mean - true_value) < 30.0
        assert abs(gaussian_mean - true_value) < 30.0


class TestGaussianEdgeCases:
    """Edge case tests for Gaussian mechanism."""

    def test_minimum_epsilon(self):
        """Test with minimum epsilon value."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=EPSILON_MIN, delta=1e-5
        )
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_maximum_epsilon(self):
        """Test with maximum epsilon value."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=EPSILON_MAX, delta=1e-5
        )
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_minimum_delta(self):
        """Test with minimum delta value."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=DELTA_MIN
        )
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_maximum_delta(self):
        """Test with maximum delta value."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=DELTA_MAX
        )
        result = mechanism.release(50.0)
        assert isinstance(result, float)

    def test_small_sensitivity(self):
        """Test with very small sensitivity."""
        mechanism = GaussianMechanism(
            sensitivity=0.001, epsilon=1.0, delta=1e-5
        )
        result = mechanism.release(0.0005)
        assert isinstance(result, float)

    def test_large_sensitivity(self):
        """Test with large sensitivity."""
        mechanism = GaussianMechanism(
            sensitivity=1e6, epsilon=1.0, delta=1e-5
        )
        result = mechanism.release(500000.0)
        assert isinstance(result, float)

    def test_empty_array(self):
        """Test with empty array."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        result = mechanism.release_array([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_single_element_array(self):
        """Test with single element array."""
        mechanism = GaussianMechanism(
            sensitivity=10.0, epsilon=1.0, delta=1e-5
        )
        result = mechanism.release_array([50.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_integer_input(self):
        """Test with integer input values."""
        mechanism = GaussianMechanism(
            sensitivity=10, epsilon=1, delta=1e-5
        )
        result = mechanism.release(50)
        assert isinstance(result, float)

        result_array = mechanism.release_array([10, 20, 30])
        assert isinstance(result_array, np.ndarray)


# =============================================================================
# Exponential Mechanism Tests
# =============================================================================


class TestExponentialScaleCalculation:
    """Tests for exponential mechanism scale calculation."""

    def test_basic_scale(self):
        """Test basic scale calculation."""
        # scale = 2 * sensitivity / epsilon
        assert calculate_scale_exponential(1.0, 1.0) == 2.0
        assert calculate_scale_exponential(1.0, 2.0) == 1.0
        assert calculate_scale_exponential(2.0, 1.0) == 4.0

    def test_scale_relationship(self):
        """Test that higher epsilon produces smaller scale."""
        scale_low_eps = calculate_scale_exponential(1.0, 0.5)
        scale_high_eps = calculate_scale_exponential(1.0, 2.0)
        assert scale_low_eps > scale_high_eps


class TestExponentialMechanism:
    """Tests for Exponential mechanism."""

    def test_mechanism_creation(self):
        """Test mechanism creation."""
        categories = ["A", "B", "C", "D"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0, sensitivity=1.0
        )
        assert mechanism is not None
        assert mechanism.n_categories == 4

    def test_mechanism_properties(self):
        """Test mechanism property accessors."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0, sensitivity=2.0
        )
        assert mechanism.categories == ["A", "B", "C"]
        assert mechanism.n_categories == 3
        assert mechanism.epsilon == 1.0
        assert mechanism.sensitivity == 2.0
        assert mechanism.scale == 4.0  # 2 * 2.0 / 1.0

    def test_select_returns_category(self):
        """Test that select returns a valid category."""
        categories = ["A", "B", "C", "D"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = [1.0, 2.0, 3.0, 4.0]
        result = mechanism.select(scores)
        assert result in categories

    def test_select_index_returns_valid_index(self):
        """Test that select_index returns a valid index."""
        categories = ["A", "B", "C", "D"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = [1.0, 2.0, 3.0, 4.0]
        result = mechanism.select_index(scores)
        assert 0 <= result < 4

    def test_sample_returns_correct_count(self):
        """Test that sample returns correct number of results."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = [1.0, 2.0, 3.0]
        result = mechanism.sample(scores, n=10)
        assert len(result) == 10
        for item in result:
            assert item in categories

    def test_sample_indices_returns_correct_count(self):
        """Test that sample_indices returns correct number of results."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = [1.0, 2.0, 3.0]
        result = mechanism.sample_indices(scores, n=10)
        assert len(result) == 10
        for idx in result:
            assert 0 <= idx < 3

    def test_pandas_series_scores(self):
        """Test with pandas Series scores."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = pd.Series([1.0, 2.0, 3.0])
        result = mechanism.select(scores)
        assert result in categories

    def test_numpy_array_scores(self):
        """Test with numpy array scores."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = np.array([1.0, 2.0, 3.0])
        result = mechanism.select(scores)
        assert result in categories

    def test_privacy_usage(self):
        """Test privacy usage reporting."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=1.0
        )
        usage = mechanism.get_privacy_usage()
        assert usage.epsilon == 1.0
        assert usage.delta is None
        assert usage.is_pure_dp

    def test_invalid_too_few_categories(self):
        """Test that too few categories raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            ExponentialMechanism(categories=["A"], epsilon=1.0)

    def test_invalid_epsilon(self):
        """Test invalid epsilon raises error."""
        with pytest.raises(ValueError):
            ExponentialMechanism(
                categories=["A", "B"], epsilon=0.0
            )

    def test_invalid_sensitivity(self):
        """Test invalid sensitivity raises error."""
        with pytest.raises(ValueError):
            ExponentialMechanism(
                categories=["A", "B"], epsilon=1.0, sensitivity=0.0
            )

    def test_wrong_scores_length(self):
        """Test that wrong scores length raises error."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        with pytest.raises(ValueError, match="Expected 3 scores"):
            mechanism.select([1.0, 2.0])  # Only 2 scores

    def test_nan_score_error(self):
        """Test that NaN score raises error."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        with pytest.raises(ValueError, match="is NaN"):
            mechanism.select([1.0, float("nan"), 3.0])

    def test_wrong_type_score(self):
        """Test that wrong type score raises error."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        with pytest.raises(TypeError, match="must be numeric"):
            mechanism.select([1.0, "two", 3.0])

    def test_sample_n_less_than_one(self):
        """Test that n < 1 raises error."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=1.0
        )
        with pytest.raises(ValueError, match="at least 1"):
            mechanism.sample([1.0, 2.0], n=0)

    def test_repr(self):
        """Test string representation."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0, sensitivity=2.0
        )
        repr_str = repr(mechanism)
        assert "ExponentialMechanism" in repr_str
        assert "n_categories=3" in repr_str
        assert "epsilon=1.0" in repr_str
        assert "sensitivity=2.0" in repr_str

    def test_categories_copy(self):
        """Test that categories property returns a copy."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        result = mechanism.categories
        result.append("D")  # Modify the returned list
        assert mechanism.categories == ["A", "B", "C"]  # Original unchanged


class TestExponentialConvenienceFunctions:
    """Tests for Exponential convenience functions."""

    def test_create_exponential_mechanism(self):
        """Test create_exponential_mechanism function."""
        mechanism = create_exponential_mechanism(
            categories=["A", "B", "C"], epsilon=1.0, sensitivity=1.0
        )
        assert isinstance(mechanism, ExponentialMechanism)
        assert mechanism.epsilon == 1.0

    def test_select_category(self):
        """Test select_category function."""
        result = select_category(
            categories=["A", "B", "C"],
            scores=[1.0, 2.0, 3.0],
            epsilon=1.0,
        )
        assert result in ["A", "B", "C"]

    def test_sample_categories(self):
        """Test sample_categories function."""
        result = sample_categories(
            categories=["A", "B", "C"],
            scores=[1.0, 2.0, 3.0],
            n=10,
            epsilon=1.0,
        )
        assert len(result) == 10
        for item in result:
            assert item in ["A", "B", "C"]


class TestExponentialSelectionBias:
    """Tests verifying exponential mechanism selection probabilities."""

    def test_higher_score_more_likely(self):
        """Test that higher scores are selected more often."""
        categories = ["A", "B", "C"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        # C has much higher score
        scores = [1.0, 1.0, 10.0]

        from collections import Counter

        results = Counter(
            mechanism.select(scores) for _ in range(1000)
        )

        # C should be selected most often
        assert results["C"] > results["A"]
        assert results["C"] > results["B"]

    def test_equal_scores_uniform_selection(self):
        """Test that equal scores produce roughly uniform selection."""
        categories = ["A", "B", "C", "D"]
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        scores = [10.0, 10.0, 10.0, 10.0]  # All equal

        from collections import Counter

        results = Counter(
            mechanism.select(scores) for _ in range(2000)
        )

        # All categories should be selected with similar frequency
        # Each should get roughly 500 +/- some variance
        for cat in categories:
            assert 300 < results[cat] < 700

    def test_high_epsilon_more_deterministic(self):
        """Test that higher epsilon makes selection more deterministic."""
        categories = ["A", "B", "C"]
        scores = [1.0, 5.0, 2.0]  # B has highest

        from collections import Counter

        # Low epsilon - more randomness
        mech_low_eps = ExponentialMechanism(
            categories=categories, epsilon=0.1
        )
        results_low = Counter(
            mech_low_eps.select(scores) for _ in range(1000)
        )

        # High epsilon - more deterministic
        mech_high_eps = ExponentialMechanism(
            categories=categories, epsilon=5.0
        )
        results_high = Counter(
            mech_high_eps.select(scores) for _ in range(1000)
        )

        # High epsilon should select B more often
        assert results_high["B"] > results_low["B"]


class TestExponentialPrivacyGuarantee:
    """Tests for exponential mechanism privacy guarantee verification."""

    def test_epsilon_bounds_respected(self):
        """Test that privacy usage matches specified epsilon."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        usage = mechanism.get_privacy_usage()
        assert usage.epsilon == 1.0
        assert usage.is_pure_dp

    def test_various_epsilon_values(self):
        """Test mechanism works with various epsilon values."""
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            mechanism = ExponentialMechanism(
                categories=["A", "B"], epsilon=eps
            )
            assert mechanism.epsilon == eps
            usage = mechanism.get_privacy_usage()
            assert usage.epsilon == eps


class TestExponentialEdgeCases:
    """Edge case tests for Exponential mechanism."""

    def test_minimum_epsilon(self):
        """Test with minimum epsilon value."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=EPSILON_MIN
        )
        result = mechanism.select([1.0, 2.0])
        assert result in ["A", "B"]

    def test_maximum_epsilon(self):
        """Test with maximum epsilon value."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=EPSILON_MAX
        )
        result = mechanism.select([1.0, 2.0])
        assert result in ["A", "B"]

    def test_two_categories(self):
        """Test with minimum number of categories."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=1.0
        )
        result = mechanism.select([1.0, 2.0])
        assert result in ["A", "B"]

    def test_many_categories(self):
        """Test with many categories."""
        categories = [f"cat_{i}" for i in range(100)]
        scores = list(range(100))
        mechanism = ExponentialMechanism(
            categories=categories, epsilon=1.0
        )
        result = mechanism.select(scores)
        assert result in categories

    def test_zero_scores(self):
        """Test with all zero scores."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        result = mechanism.select([0.0, 0.0, 0.0])
        assert result in ["A", "B", "C"]

    def test_negative_scores(self):
        """Test with negative scores."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        result = mechanism.select([-5.0, -2.0, -10.0])
        assert result in ["A", "B", "C"]

    def test_integer_scores(self):
        """Test with integer scores."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        result = mechanism.select([1, 2, 3])
        assert result in ["A", "B", "C"]

    def test_mixed_category_types(self):
        """Test with mixed category types."""
        mechanism = ExponentialMechanism(
            categories=[1, "two", 3.0], epsilon=1.0
        )
        result = mechanism.select([1.0, 2.0, 3.0])
        assert result in [1, "two", 3.0]

    def test_single_sample(self):
        """Test sampling n=1."""
        mechanism = ExponentialMechanism(
            categories=["A", "B", "C"], epsilon=1.0
        )
        result = mechanism.sample([1.0, 2.0, 3.0], n=1)
        assert len(result) == 1
        assert result[0] in ["A", "B", "C"]

    def test_large_sample(self):
        """Test sampling many items."""
        mechanism = ExponentialMechanism(
            categories=["A", "B"], epsilon=1.0
        )
        result = mechanism.sample([1.0, 100.0], n=1000)
        assert len(result) == 1000
