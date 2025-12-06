"""Differential privacy mechanisms for DPtoolkit.

This module provides DP mechanisms wrapped around OpenDP, including:
- Laplace mechanism for bounded numeric data (ε-DP)
- Gaussian mechanism for unbounded numeric data ((ε,δ)-DP)
- Exponential mechanism for categorical data (ε-DP) [future]
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import opendp.prelude as dp
import pandas as pd

# Enable OpenDP contrib features
dp.enable_features("contrib")


# =============================================================================
# Privacy Parameter Bounds
# =============================================================================

# Epsilon bounds (reasonable range for differential privacy)
EPSILON_MIN = 0.01  # Very strong privacy
EPSILON_MAX = 10.0  # Weak privacy but still provides some protection

# Delta bounds for (ε,δ)-DP (Gaussian mechanism)
DELTA_MIN = 1e-10  # Very strong privacy
DELTA_MAX = 1e-3  # Reasonable upper bound (should be << 1/n)


def validate_epsilon(epsilon: float) -> None:
    """Validate epsilon parameter.

    Args:
        epsilon: Privacy parameter (must be positive).

    Raises:
        ValueError: If epsilon is not in valid range.
    """
    if not isinstance(epsilon, (int, float)):
        raise TypeError(f"Epsilon must be a number, got {type(epsilon)}")
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    if epsilon < EPSILON_MIN:
        raise ValueError(
            f"Epsilon {epsilon} is below minimum {EPSILON_MIN}. "
            "Such strong privacy may produce unusable results."
        )
    if epsilon > EPSILON_MAX:
        raise ValueError(
            f"Epsilon {epsilon} exceeds maximum {EPSILON_MAX}. "
            "Consider using a smaller epsilon for meaningful privacy."
        )


def validate_delta(delta: float) -> None:
    """Validate delta parameter for (ε,δ)-DP.

    Args:
        delta: Approximate DP parameter (must be small positive).

    Raises:
        ValueError: If delta is not in valid range.
    """
    if not isinstance(delta, (int, float)):
        raise TypeError(f"Delta must be a number, got {type(delta)}")
    if delta <= 0:
        raise ValueError(f"Delta must be positive, got {delta}")
    if delta < DELTA_MIN:
        raise ValueError(
            f"Delta {delta} is below minimum {DELTA_MIN}. "
            "Such small delta may cause numerical issues."
        )
    if delta > DELTA_MAX:
        raise ValueError(
            f"Delta {delta} exceeds maximum {DELTA_MAX}. "
            "Delta should be much smaller than 1/n for meaningful privacy."
        )


def validate_sensitivity(sensitivity: float) -> None:
    """Validate sensitivity parameter.

    Args:
        sensitivity: Query sensitivity (must be positive).

    Raises:
        ValueError: If sensitivity is not positive.
    """
    if not isinstance(sensitivity, (int, float)):
        raise TypeError(
            f"Sensitivity must be a number, got {type(sensitivity)}"
        )
    if sensitivity <= 0:
        raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
    if np.isnan(sensitivity):
        raise ValueError("Sensitivity cannot be NaN")
    if np.isinf(sensitivity):
        raise ValueError("Sensitivity must be finite")


def validate_bounds(lower: float, upper: float) -> None:
    """Validate bounds for bounded data.

    Args:
        lower: Lower bound.
        upper: Upper bound.

    Raises:
        ValueError: If bounds are invalid.
    """
    if not isinstance(lower, (int, float)) or not isinstance(
        upper, (int, float)
    ):
        raise TypeError("Bounds must be numeric")
    if np.isnan(lower) or np.isnan(upper):
        raise ValueError("Bounds cannot be NaN")
    if np.isinf(lower) or np.isinf(upper):
        raise ValueError("Bounds must be finite")
    if lower >= upper:
        raise ValueError(
            f"Lower bound ({lower}) must be less than upper bound ({upper})"
        )


# =============================================================================
# Sensitivity Calculation
# =============================================================================


def calculate_sensitivity_bounded(lower: float, upper: float) -> float:
    """Calculate sensitivity for bounded numeric data.

    For a single bounded value, the sensitivity is the range (upper - lower)
    because changing one record can change the result by at most this amount.

    Args:
        lower: Lower bound of the data.
        upper: Upper bound of the data.

    Returns:
        Sensitivity value (upper - lower).
    """
    validate_bounds(lower, upper)
    return float(upper - lower)


def calculate_scale_laplace(sensitivity: float, epsilon: float) -> float:
    """Calculate Laplace scale parameter from sensitivity and epsilon.

    For Laplace mechanism: scale = sensitivity / epsilon

    Args:
        sensitivity: Global sensitivity of the query.
        epsilon: Privacy parameter.

    Returns:
        Scale parameter for Laplace distribution.
    """
    if sensitivity <= 0:
        raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
    validate_epsilon(epsilon)
    return sensitivity / epsilon


def calculate_rho_from_epsilon_delta(epsilon: float, delta: float) -> float:
    """Calculate zCDP rho parameter from (ε,δ).

    Solves: ε = ρ + 2*sqrt(ρ*ln(1/δ)) for ρ

    This is a quadratic in sqrt(ρ). Let x = sqrt(ρ), then:
    x² + 2*sqrt(ln(1/δ))*x - ε = 0
    x = -sqrt(ln(1/δ)) + sqrt(ln(1/δ) + ε)
    ρ = x²

    Args:
        epsilon: Privacy parameter.
        delta: Approximate DP parameter.

    Returns:
        zCDP rho parameter.
    """
    ln_inv_delta = math.log(1.0 / delta)
    x = -math.sqrt(ln_inv_delta) + math.sqrt(ln_inv_delta + epsilon)
    return x * x


def calculate_scale_gaussian(
    sensitivity: float, epsilon: float, delta: float
) -> float:
    """Calculate Gaussian scale (std dev) from sensitivity and (ε,δ).

    Uses zCDP-based calculation for exact epsilon targeting:
    1. Convert (ε,δ) to zCDP parameter ρ
    2. Calculate σ = sensitivity / sqrt(2*ρ)

    Args:
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter.
        delta: Approximate DP parameter.

    Returns:
        Scale parameter (standard deviation) for Gaussian distribution.
    """
    validate_sensitivity(sensitivity)
    validate_epsilon(epsilon)
    validate_delta(delta)

    rho = calculate_rho_from_epsilon_delta(epsilon, delta)
    return sensitivity / math.sqrt(2.0 * rho)


def calculate_epsilon_from_rho_delta(rho: float, delta: float) -> float:
    """Calculate epsilon from zCDP rho and delta.

    Uses: ε = ρ + 2*sqrt(ρ*ln(1/δ))

    Args:
        rho: zCDP parameter.
        delta: Approximate DP parameter.

    Returns:
        Epsilon for (ε,δ)-DP.
    """
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


# =============================================================================
# Base Mechanism Class
# =============================================================================


@dataclass
class PrivacyUsage:
    """Privacy budget usage for a mechanism.

    Attributes:
        epsilon: Privacy parameter (for pure ε-DP).
        delta: Approximate DP parameter (for (ε,δ)-DP), None for pure DP.
    """

    epsilon: float
    delta: Optional[float] = None

    @property
    def is_pure_dp(self) -> bool:
        """Check if this is pure ε-DP (no delta)."""
        return self.delta is None or self.delta == 0.0


class DPMechanism(ABC):
    """Abstract base class for differential privacy mechanisms."""

    @abstractmethod
    def release(self, value: float) -> float:
        """Apply the mechanism to release a differentially private value.

        Args:
            value: The true value to protect.

        Returns:
            Noisy value satisfying differential privacy.
        """
        pass

    @abstractmethod
    def release_array(
        self, values: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Apply the mechanism to an array of values.

        Args:
            values: Array of true values to protect.

        Returns:
            Array of noisy values satisfying differential privacy.
        """
        pass

    @abstractmethod
    def get_privacy_usage(self) -> PrivacyUsage:
        """Get the privacy budget used by this mechanism.

        Returns:
            PrivacyUsage with epsilon (and optionally delta).
        """
        pass

    @property
    @abstractmethod
    def sensitivity(self) -> float:
        """Get the sensitivity used by this mechanism."""
        pass

    @property
    @abstractmethod
    def scale(self) -> float:
        """Get the noise scale parameter."""
        pass


# =============================================================================
# Laplace Mechanism
# =============================================================================


class LaplaceMechanism(DPMechanism):
    """Laplace mechanism for ε-differential privacy.

    The Laplace mechanism adds noise drawn from a Laplace distribution
    to achieve ε-differential privacy. It is suitable for numeric queries
    with bounded sensitivity.

    For a query with global sensitivity Δf and privacy parameter ε,
    the mechanism adds noise with scale b = Δf/ε.

    Attributes:
        lower: Lower bound of the data.
        upper: Upper bound of the data.
        epsilon: Privacy parameter.
    """

    def __init__(
        self,
        lower: float,
        upper: float,
        epsilon: float,
    ) -> None:
        """Initialize the Laplace mechanism.

        Args:
            lower: Lower bound of the data range.
            upper: Upper bound of the data range.
            epsilon: Privacy parameter (ε). Must be in [0.01, 10.0].

        Raises:
            ValueError: If parameters are invalid.
        """
        validate_bounds(lower, upper)
        validate_epsilon(epsilon)

        self._lower = float(lower)
        self._upper = float(upper)
        self._epsilon = float(epsilon)
        self._sensitivity = calculate_sensitivity_bounded(lower, upper)
        self._scale = calculate_scale_laplace(self._sensitivity, epsilon)

        # Create OpenDP mechanism for scalar values
        self._scalar_domain = dp.atom_domain(T=float, nan=False)
        self._scalar_metric = dp.absolute_distance(T=float)
        self._scalar_mechanism = dp.m.make_laplace(
            self._scalar_domain,
            self._scalar_metric,
            scale=self._scale,
        )

        # Create OpenDP mechanism for vector values
        self._vector_domain = dp.vector_domain(
            dp.atom_domain(T=float, nan=False)
        )
        self._vector_metric = dp.l1_distance(T=float)
        self._vector_mechanism = dp.m.make_laplace(
            self._vector_domain,
            self._vector_metric,
            scale=self._scale,
        )

    @property
    def lower(self) -> float:
        """Get lower bound."""
        return self._lower

    @property
    def upper(self) -> float:
        """Get upper bound."""
        return self._upper

    @property
    def epsilon(self) -> float:
        """Get epsilon parameter."""
        return self._epsilon

    @property
    def sensitivity(self) -> float:
        """Get the sensitivity (upper - lower)."""
        return self._sensitivity

    @property
    def scale(self) -> float:
        """Get the Laplace scale parameter (sensitivity / epsilon)."""
        return self._scale

    def release(self, value: float) -> float:
        """Apply Laplace mechanism to release a single value.

        Args:
            value: The true value to protect.

        Returns:
            Noisy value with Laplace noise added.

        Raises:
            TypeError: If value is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(value)}")
        if np.isnan(value):
            raise ValueError("Value cannot be NaN")

        return float(self._scalar_mechanism(float(value)))

    def release_array(
        self, values: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Apply Laplace mechanism to an array of values.

        Each value in the array receives independent Laplace noise.

        Args:
            values: Array of true values to protect.

        Returns:
            NumPy array of noisy values.

        Raises:
            ValueError: If array contains NaN values.
        """
        # Convert to numpy array
        if isinstance(values, pd.Series):
            arr = values.to_numpy()
        elif isinstance(values, list):
            arr = np.array(values)
        else:
            arr = np.asarray(values)

        # Check for NaN
        if np.any(np.isnan(arr)):
            raise ValueError("Array contains NaN values")

        # Apply mechanism
        noisy_list = self._vector_mechanism(arr.astype(float).tolist())
        result: np.ndarray = np.array(noisy_list)
        return result

    def get_privacy_usage(self) -> PrivacyUsage:
        """Get the privacy budget used.

        Returns:
            PrivacyUsage with epsilon (pure ε-DP).
        """
        return PrivacyUsage(epsilon=self._epsilon)

    def clamp(self, value: float) -> float:
        """Clamp a value to the mechanism's bounds.

        Args:
            value: Value to clamp.

        Returns:
            Value clamped to [lower, upper].
        """
        return float(np.clip(value, self._lower, self._upper))

    def clamp_array(
        self, values: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Clamp an array of values to the mechanism's bounds.

        Args:
            values: Array of values to clamp.

        Returns:
            NumPy array of clamped values.
        """
        if isinstance(values, pd.Series):
            arr = values.to_numpy()
        elif isinstance(values, list):
            arr = np.array(values)
        else:
            arr = np.asarray(values)

        result: np.ndarray = np.clip(arr, self._lower, self._upper)
        return result

    def release_clamped(self, value: float) -> float:
        """Clamp value to bounds, then apply Laplace mechanism.

        This is the recommended way to release bounded data.

        Args:
            value: The true value (will be clamped to bounds).

        Returns:
            Noisy value with Laplace noise added.
        """
        clamped = self.clamp(value)
        return self.release(clamped)

    def release_array_clamped(
        self, values: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Clamp array to bounds, then apply Laplace mechanism.

        This is the recommended way to release bounded data arrays.

        Args:
            values: Array of true values (will be clamped to bounds).

        Returns:
            NumPy array of noisy values.
        """
        clamped = self.clamp_array(values)
        return self.release_array(clamped)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LaplaceMechanism(lower={self._lower}, upper={self._upper}, "
            f"epsilon={self._epsilon}, scale={self._scale:.4f})"
        )


# =============================================================================
# Gaussian Mechanism
# =============================================================================


class GaussianMechanism(DPMechanism):
    """Gaussian mechanism for (ε,δ)-differential privacy.

    The Gaussian mechanism adds noise drawn from a Gaussian distribution
    to achieve (ε,δ)-differential privacy. It is suitable for numeric queries
    on unbounded data where approximate DP is acceptable.

    Unlike the Laplace mechanism, Gaussian does not require bounded data,
    but does require a user-specified sensitivity.

    For a query with L2 sensitivity Δ₂ and privacy parameters (ε,δ),
    the mechanism adds Gaussian noise with scale σ derived from zCDP.

    Attributes:
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter.
        delta: Approximate DP parameter.
    """

    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Initialize the Gaussian mechanism.

        Args:
            sensitivity: L2 sensitivity of the query.
            epsilon: Privacy parameter (ε). Must be in [0.01, 10.0].
            delta: Approximate DP parameter (δ). Must be in [1e-10, 1e-3].

        Raises:
            ValueError: If parameters are invalid.
        """
        validate_sensitivity(sensitivity)
        validate_epsilon(epsilon)
        validate_delta(delta)

        self._sensitivity = float(sensitivity)
        self._epsilon = float(epsilon)
        self._delta = float(delta)
        self._scale = calculate_scale_gaussian(sensitivity, epsilon, delta)

        # Store zCDP rho for privacy verification
        self._rho = calculate_rho_from_epsilon_delta(epsilon, delta)

        # Create OpenDP mechanism for scalar values
        self._scalar_domain = dp.atom_domain(T=float, nan=False)
        self._scalar_metric = dp.absolute_distance(T=float)
        self._scalar_mechanism = dp.m.make_gaussian(
            self._scalar_domain,
            self._scalar_metric,
            scale=self._scale,
        )

        # Create OpenDP mechanism for vector values
        self._vector_domain = dp.vector_domain(
            dp.atom_domain(T=float, nan=False)
        )
        self._vector_metric = dp.l2_distance(T=float)
        self._vector_mechanism = dp.m.make_gaussian(
            self._vector_domain,
            self._vector_metric,
            scale=self._scale,
        )

    @property
    def epsilon(self) -> float:
        """Get epsilon parameter."""
        return self._epsilon

    @property
    def delta(self) -> float:
        """Get delta parameter."""
        return self._delta

    @property
    def sensitivity(self) -> float:
        """Get the L2 sensitivity."""
        return self._sensitivity

    @property
    def scale(self) -> float:
        """Get the Gaussian scale parameter (standard deviation)."""
        return self._scale

    @property
    def rho(self) -> float:
        """Get the zCDP rho parameter."""
        return self._rho

    def release(self, value: float) -> float:
        """Apply Gaussian mechanism to release a single value.

        Args:
            value: The true value to protect.

        Returns:
            Noisy value with Gaussian noise added.

        Raises:
            TypeError: If value is not numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be numeric, got {type(value)}")
        if np.isnan(value):
            raise ValueError("Value cannot be NaN")

        return float(self._scalar_mechanism(float(value)))

    def release_array(
        self, values: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Apply Gaussian mechanism to an array of values.

        Each value in the array receives independent Gaussian noise.

        Args:
            values: Array of true values to protect.

        Returns:
            NumPy array of noisy values.

        Raises:
            ValueError: If array contains NaN values.
        """
        # Convert to numpy array
        if isinstance(values, pd.Series):
            arr = values.to_numpy()
        elif isinstance(values, list):
            arr = np.array(values)
        else:
            arr = np.asarray(values)

        # Check for NaN
        if np.any(np.isnan(arr)):
            raise ValueError("Array contains NaN values")

        # Apply mechanism
        noisy_list = self._vector_mechanism(arr.astype(float).tolist())
        result: np.ndarray = np.array(noisy_list)
        return result

    def get_privacy_usage(self) -> PrivacyUsage:
        """Get the privacy budget used.

        Returns:
            PrivacyUsage with epsilon and delta ((ε,δ)-DP).
        """
        return PrivacyUsage(epsilon=self._epsilon, delta=self._delta)

    def get_achieved_epsilon(
        self, target_delta: Optional[float] = None
    ) -> float:
        """Calculate achieved epsilon for a given delta.

        This converts the zCDP guarantee to (ε,δ)-DP for any delta.

        Args:
            target_delta: Delta to use for conversion. Defaults to self.delta.

        Returns:
            Achieved epsilon for the given delta.
        """
        if target_delta is None:
            target_delta = self._delta
        validate_delta(target_delta)
        return calculate_epsilon_from_rho_delta(self._rho, target_delta)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GaussianMechanism(sensitivity={self._sensitivity}, "
            f"epsilon={self._epsilon}, delta={self._delta}, "
            f"scale={self._scale:.4f})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_laplace_mechanism(
    lower: float,
    upper: float,
    epsilon: float,
) -> LaplaceMechanism:
    """Create a Laplace mechanism for bounded data.

    Args:
        lower: Lower bound of the data range.
        upper: Upper bound of the data range.
        epsilon: Privacy parameter (ε).

    Returns:
        Configured LaplaceMechanism instance.
    """
    return LaplaceMechanism(lower=lower, upper=upper, epsilon=epsilon)


def add_laplace_noise(
    value: float,
    lower: float,
    upper: float,
    epsilon: float,
) -> float:
    """Add Laplace noise to a single value.

    Convenience function that creates a mechanism and applies it.

    Args:
        value: The true value to protect.
        lower: Lower bound of the data range.
        upper: Upper bound of the data range.
        epsilon: Privacy parameter (ε).

    Returns:
        Noisy value satisfying ε-differential privacy.
    """
    mechanism = LaplaceMechanism(lower=lower, upper=upper, epsilon=epsilon)
    return mechanism.release_clamped(value)


def add_laplace_noise_array(
    values: Union[np.ndarray, pd.Series, list],
    lower: float,
    upper: float,
    epsilon: float,
) -> np.ndarray:
    """Add Laplace noise to an array of values.

    Convenience function that creates a mechanism and applies it.

    Args:
        values: Array of true values to protect.
        lower: Lower bound of the data range.
        upper: Upper bound of the data range.
        epsilon: Privacy parameter (ε).

    Returns:
        NumPy array of noisy values satisfying ε-differential privacy.
    """
    mechanism = LaplaceMechanism(lower=lower, upper=upper, epsilon=epsilon)
    return mechanism.release_array_clamped(values)


def create_gaussian_mechanism(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> GaussianMechanism:
    """Create a Gaussian mechanism for unbounded data.

    Args:
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter (ε).
        delta: Approximate DP parameter (δ).

    Returns:
        Configured GaussianMechanism instance.
    """
    return GaussianMechanism(
        sensitivity=sensitivity, epsilon=epsilon, delta=delta
    )


def add_gaussian_noise(
    value: float,
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Add Gaussian noise to a single value.

    Convenience function that creates a mechanism and applies it.

    Args:
        value: The true value to protect.
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter (ε).
        delta: Approximate DP parameter (δ).

    Returns:
        Noisy value satisfying (ε,δ)-differential privacy.
    """
    mechanism = GaussianMechanism(
        sensitivity=sensitivity, epsilon=epsilon, delta=delta
    )
    return mechanism.release(value)


def add_gaussian_noise_array(
    values: Union[np.ndarray, pd.Series, list],
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> np.ndarray:
    """Add Gaussian noise to an array of values.

    Convenience function that creates a mechanism and applies it.

    Args:
        values: Array of true values to protect.
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy parameter (ε).
        delta: Approximate DP parameter (δ).

    Returns:
        NumPy array of noisy values satisfying (ε,δ)-differential privacy.
    """
    mechanism = GaussianMechanism(
        sensitivity=sensitivity, epsilon=epsilon, delta=delta
    )
    return mechanism.release_array(values)
