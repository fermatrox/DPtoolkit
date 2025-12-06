"""Core DP mechanisms and budget tracking."""

from dp_toolkit.core.mechanisms import (
    # Constants
    EPSILON_MIN,
    EPSILON_MAX,
    DELTA_MIN,
    DELTA_MAX,
    # Validation functions
    validate_epsilon,
    validate_delta,
    validate_sensitivity,
    validate_bounds,
    # Sensitivity / Scale calculation
    calculate_sensitivity_bounded,
    calculate_scale_laplace,
    calculate_scale_gaussian,
    calculate_scale_exponential,
    calculate_rho_from_epsilon_delta,
    calculate_epsilon_from_rho_delta,
    # Classes
    PrivacyUsage,
    DPMechanism,
    LaplaceMechanism,
    GaussianMechanism,
    ExponentialMechanism,
    # Convenience functions - Laplace
    create_laplace_mechanism,
    add_laplace_noise,
    add_laplace_noise_array,
    # Convenience functions - Gaussian
    create_gaussian_mechanism,
    add_gaussian_noise,
    add_gaussian_noise_array,
    # Convenience functions - Exponential
    create_exponential_mechanism,
    select_category,
    sample_categories,
)

from dp_toolkit.core.budget import (
    # Constants
    DEFAULT_DELTA_PRIME,
    # Enums
    CompositionMethod,
    # Data classes
    PrivacyBudget,
    BudgetQuery,
    # Composition functions
    compose_sequential_basic,
    compose_sequential_advanced,
    compose_parallel,
    # Main class
    PrivacyBudgetTracker,
    # Exception
    BudgetExceededError,
    # Convenience functions
    create_budget_tracker,
    calculate_total_budget,
)

__all__ = [
    # Mechanism Constants
    "EPSILON_MIN",
    "EPSILON_MAX",
    "DELTA_MIN",
    "DELTA_MAX",
    # Validation
    "validate_epsilon",
    "validate_delta",
    "validate_sensitivity",
    "validate_bounds",
    # Sensitivity / Scale
    "calculate_sensitivity_bounded",
    "calculate_scale_laplace",
    "calculate_scale_gaussian",
    "calculate_scale_exponential",
    "calculate_rho_from_epsilon_delta",
    "calculate_epsilon_from_rho_delta",
    # Mechanism Classes
    "PrivacyUsage",
    "DPMechanism",
    "LaplaceMechanism",
    "GaussianMechanism",
    "ExponentialMechanism",
    # Convenience functions - Laplace
    "create_laplace_mechanism",
    "add_laplace_noise",
    "add_laplace_noise_array",
    # Convenience functions - Gaussian
    "create_gaussian_mechanism",
    "add_gaussian_noise",
    "add_gaussian_noise_array",
    # Convenience functions - Exponential
    "create_exponential_mechanism",
    "select_category",
    "sample_categories",
    # Budget Constants
    "DEFAULT_DELTA_PRIME",
    # Budget Enums
    "CompositionMethod",
    # Budget Data Classes
    "PrivacyBudget",
    "BudgetQuery",
    # Composition Functions
    "compose_sequential_basic",
    "compose_sequential_advanced",
    "compose_parallel",
    # Budget Tracker
    "PrivacyBudgetTracker",
    "BudgetExceededError",
    # Budget Convenience Functions
    "create_budget_tracker",
    "calculate_total_budget",
]
