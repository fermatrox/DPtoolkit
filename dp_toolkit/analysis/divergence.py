"""Advanced divergence metrics for distribution comparison.

This module provides advanced metrics for comparing probability distributions
between original and protected datasets:

- KL Divergence: Kullback-Leibler divergence with smoothing
- JS Distance: Jensen-Shannon distance (symmetric KL)
- Wasserstein Distance: Earth Mover's Distance
- Total Variation Distance: Maximum difference in probabilities
- Category Drift: Comprehensive categorical distribution shift

These metrics help quantify how well differential privacy protection
preserves the statistical properties of the original data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance as scipy_wasserstein


# =============================================================================
# Constants
# =============================================================================

# Default smoothing parameter for KL divergence (Laplace smoothing)
DEFAULT_SMOOTHING = 1e-10

# Minimum probability for numeric stability
MIN_PROBABILITY = 1e-15


# =============================================================================
# Enums
# =============================================================================


class DivergenceType(Enum):
    """Types of divergence metrics."""

    KL = "kl"  # Kullback-Leibler
    JS = "js"  # Jensen-Shannon
    WASSERSTEIN = "wasserstein"
    TVD = "tvd"  # Total Variation Distance


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DivergenceResult:
    """Result of a divergence calculation.

    Attributes:
        divergence_type: Type of divergence metric.
        value: The divergence/distance value.
        is_symmetric: Whether the metric is symmetric.
        unit: Unit of measurement (bits, nats, or unitless).
        details: Additional metric-specific details.
    """

    divergence_type: DivergenceType
    value: float
    is_symmetric: bool
    unit: str
    details: Dict[str, Any]


@dataclass
class CategoryDriftResult:
    """Comprehensive category drift analysis result.

    Attributes:
        drift_rate: Proportion of values that changed category.
        kl_divergence: KL divergence between distributions.
        js_distance: Jensen-Shannon distance.
        tvd: Total Variation Distance.
        wasserstein: Wasserstein distance (if ordinal).
        entropy_original: Entropy of original distribution.
        entropy_protected: Entropy of protected distribution.
        entropy_difference: Change in entropy.
        top_changes: Top category frequency changes.
    """

    drift_rate: float
    kl_divergence: float
    js_distance: float
    tvd: float
    wasserstein: Optional[float]
    entropy_original: float
    entropy_protected: float
    entropy_difference: float
    top_changes: List[Tuple[str, float, float, float]]


@dataclass
class NumericDistributionComparison:
    """Comparison of numeric distributions.

    Attributes:
        wasserstein: Wasserstein (Earth Mover's) distance.
        ks_statistic: Kolmogorov-Smirnov test statistic.
        ks_pvalue: KS test p-value.
        kl_divergence: KL divergence (via histogram binning).
        js_distance: Jensen-Shannon distance.
        mean_shift: Difference in means (normalized).
        std_ratio: Ratio of standard deviations.
    """

    wasserstein: float
    ks_statistic: float
    ks_pvalue: float
    kl_divergence: Optional[float]
    js_distance: Optional[float]
    mean_shift: float
    std_ratio: float


# =============================================================================
# Core Divergence Functions
# =============================================================================


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    smoothing: float = DEFAULT_SMOOTHING,
    base: str = "e",
) -> float:
    """Calculate Kullback-Leibler divergence D_KL(P || Q).

    KL divergence measures how distribution P diverges from distribution Q.
    Note: KL divergence is asymmetric: D_KL(P||Q) != D_KL(Q||P).

    Args:
        p: Probability distribution P (must sum to 1).
        q: Probability distribution Q (must sum to 1).
        smoothing: Smoothing parameter to avoid log(0). Applied when q has zeros.
        base: Logarithm base - "e" for nats, "2" for bits.

    Returns:
        KL divergence value.

    Raises:
        ValueError: If distributions have different lengths or don't sum to 1.

    Example:
        >>> p = np.array([0.4, 0.3, 0.3])
        >>> q = np.array([0.33, 0.33, 0.34])
        >>> kl = kl_divergence(p, q)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if len(p) != len(q):
        raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")

    # Normalize if needed (handle floating point imprecision)
    p = p / p.sum()
    q = q / q.sum()

    # Apply smoothing to avoid log(0)
    if (q < MIN_PROBABILITY).any():
        q = q + smoothing
        q = q / q.sum()

    # Mask zero probabilities in p (0 * log(0) = 0 by convention)
    mask = p > MIN_PROBABILITY

    # D_KL(P || Q) = sum(p * log(p/q))
    if base == "2":
        kl_value: float = float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    else:
        kl_value = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return kl_value


def js_distance(
    p: np.ndarray,
    q: np.ndarray,
    base: str = "2",
) -> float:
    """Calculate Jensen-Shannon distance.

    JS distance is the square root of Jensen-Shannon divergence.
    It is a symmetric, bounded [0, 1] metric (when using log base 2).

    JS divergence: JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.
        base: Logarithm base - "2" for bits (bounded by 1), "e" for nats.

    Returns:
        Jensen-Shannon distance (square root of divergence).

    Example:
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> js = js_distance(p, q)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Use scipy for numerical stability
    if base == "2":
        return float(jensenshannon(p, q, base=2.0))
    else:
        return float(jensenshannon(p, q, base=np.e))


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: str = "2",
) -> float:
    """Calculate Jensen-Shannon divergence (not distance).

    This is the square of the JS distance.

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.
        base: Logarithm base.

    Returns:
        Jensen-Shannon divergence.
    """
    distance = js_distance(p, q, base=base)
    return distance**2


def wasserstein_distance(
    values1: np.ndarray,
    values2: np.ndarray,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
) -> float:
    """Calculate Wasserstein distance (Earth Mover's Distance).

    Wasserstein distance measures the minimum "work" required to transform
    one distribution into another. Also known as Earth Mover's Distance.

    Args:
        values1: Values from first distribution.
        values2: Values from second distribution.
        weights1: Optional weights for first distribution.
        weights2: Optional weights for second distribution.

    Returns:
        Wasserstein distance.

    Example:
        >>> original = np.array([1, 2, 3, 4, 5])
        >>> protected = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        >>> w = wasserstein_distance(original, protected)
    """
    values1 = np.asarray(values1, dtype=np.float64)
    values2 = np.asarray(values2, dtype=np.float64)

    # Remove NaN values
    values1 = values1[~np.isnan(values1)]
    values2 = values2[~np.isnan(values2)]

    if len(values1) == 0 or len(values2) == 0:
        return 0.0

    if weights1 is not None and weights2 is not None:
        return float(scipy_wasserstein(values1, values2, weights1, weights2))
    else:
        return float(scipy_wasserstein(values1, values2))


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Calculate Total Variation Distance.

    TVD is the maximum difference in probability assigned to any event.
    TVD(P, Q) = 0.5 * sum(|p_i - q_i|)

    Bounded between 0 (identical) and 1 (completely different).

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.

    Returns:
        Total Variation Distance.

    Example:
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> tvd = total_variation_distance(p, q)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return float(0.5 * np.sum(np.abs(p - q)))


def hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """Calculate Hellinger distance.

    Hellinger distance is related to TVD and is bounded [0, 1].
    H(P, Q) = sqrt(0.5 * sum((sqrt(p_i) - sqrt(q_i))^2))

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.

    Returns:
        Hellinger distance.

    Example:
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.9, 0.1])
        >>> h = hellinger_distance(p, q)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


# =============================================================================
# Entropy Functions
# =============================================================================


def entropy(
    p: np.ndarray,
    base: str = "2",
) -> float:
    """Calculate Shannon entropy of a distribution.

    Args:
        p: Probability distribution.
        base: Logarithm base - "2" for bits, "e" for nats.

    Returns:
        Entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()

    # Mask zero probabilities
    mask = p > MIN_PROBABILITY

    if base == "2":
        return float(-np.sum(p[mask] * np.log2(p[mask])))
    else:
        return float(-np.sum(p[mask] * np.log(p[mask])))


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    smoothing: float = DEFAULT_SMOOTHING,
    base: str = "2",
) -> float:
    """Calculate cross-entropy H(P, Q).

    Cross-entropy = Entropy(P) + KL(P || Q)

    Args:
        p: True distribution P.
        q: Predicted distribution Q.
        smoothing: Smoothing for zero probabilities.
        base: Logarithm base.

    Returns:
        Cross-entropy value.
    """
    return entropy(p, base) + kl_divergence(p, q, smoothing, base)


# =============================================================================
# Distribution Conversion Utilities
# =============================================================================


def series_to_distribution(
    series: pd.Series,
    categories: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert a pandas Series to a probability distribution.

    Args:
        series: Categorical series.
        categories: Optional list of all possible categories.

    Returns:
        Tuple of (probability array, category labels).
    """
    # Get value counts
    counts = series.value_counts()

    # Determine all categories
    if categories is None:
        categories = list(counts.index)
    else:
        # Ensure all categories are included
        categories = list(set(categories) | set(counts.index))

    # Build probability distribution
    probs = np.array([counts.get(cat, 0) for cat in categories], dtype=np.float64)
    probs = probs / probs.sum()

    return probs, categories


def numeric_to_histogram(
    values: np.ndarray,
    bins: Union[int, np.ndarray] = 50,
    range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert numeric values to a histogram distribution.

    Args:
        values: Numeric values.
        bins: Number of bins or bin edges.
        range: Optional (min, max) range for binning.

    Returns:
        Tuple of (probability distribution, bin edges).
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.array([]), np.array([])

    hist, bin_edges = np.histogram(values, bins=bins, range=range, density=True)

    # Normalize to probabilities
    bin_widths = np.diff(bin_edges)
    probs = hist * bin_widths

    return probs, bin_edges


# =============================================================================
# Categorical Divergence Analysis
# =============================================================================


def analyze_category_drift(
    original: pd.Series,
    protected: pd.Series,
    include_wasserstein: bool = False,
) -> CategoryDriftResult:
    """Analyze category distribution drift between original and protected.

    Args:
        original: Original categorical series.
        protected: Protected categorical series.
        include_wasserstein: Whether to calculate Wasserstein (assumes ordinal).

    Returns:
        CategoryDriftResult with comprehensive drift metrics.
    """
    # Handle nulls
    mask = original.notna() & protected.notna()
    orig_valid = original[mask].astype(str)
    prot_valid = protected[mask].astype(str)

    if len(orig_valid) == 0:
        return CategoryDriftResult(
            drift_rate=0.0,
            kl_divergence=0.0,
            js_distance=0.0,
            tvd=0.0,
            wasserstein=None,
            entropy_original=0.0,
            entropy_protected=0.0,
            entropy_difference=0.0,
            top_changes=[],
        )

    # Calculate drift rate
    drift_rate = float((orig_valid.values != prot_valid.values).sum() / len(orig_valid))

    # Get all categories
    all_categories = list(set(orig_valid.unique()) | set(prot_valid.unique()))

    # Convert to distributions
    orig_probs, _ = series_to_distribution(orig_valid, all_categories)
    prot_probs, _ = series_to_distribution(prot_valid, all_categories)

    # Calculate divergence metrics
    kl = kl_divergence(orig_probs, prot_probs, base="2")
    js = js_distance(orig_probs, prot_probs, base="2")
    tvd = total_variation_distance(orig_probs, prot_probs)

    # Wasserstein (only meaningful for ordinal categories)
    wass: Optional[float] = None
    if include_wasserstein:
        # Use category indices as ordinal values
        cat_to_idx = {cat: i for i, cat in enumerate(sorted(all_categories))}
        orig_indices = np.array([cat_to_idx[v] for v in orig_valid])
        prot_indices = np.array([cat_to_idx[v] for v in prot_valid])
        wass = wasserstein_distance(orig_indices, prot_indices)

    # Calculate entropies
    entropy_orig = entropy(orig_probs, base="2")
    entropy_prot = entropy(prot_probs, base="2")

    # Top frequency changes
    orig_counts = orig_valid.value_counts(normalize=True)
    prot_counts = prot_valid.value_counts(normalize=True)

    changes = []
    for cat in all_categories:
        orig_freq = float(orig_counts.get(cat, 0.0))
        prot_freq = float(prot_counts.get(cat, 0.0))
        diff = prot_freq - orig_freq
        changes.append((cat, orig_freq, prot_freq, diff))

    # Sort by absolute change
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    top_changes = changes[:10]

    return CategoryDriftResult(
        drift_rate=drift_rate,
        kl_divergence=kl,
        js_distance=js,
        tvd=tvd,
        wasserstein=wass,
        entropy_original=entropy_orig,
        entropy_protected=entropy_prot,
        entropy_difference=entropy_prot - entropy_orig,
        top_changes=top_changes,
    )


# =============================================================================
# Numeric Distribution Comparison
# =============================================================================


def compare_numeric_distributions(
    original: pd.Series,
    protected: pd.Series,
    bins: int = 50,
) -> NumericDistributionComparison:
    """Compare numeric distributions between original and protected.

    Args:
        original: Original numeric series.
        protected: Protected numeric series.
        bins: Number of bins for histogram-based metrics.

    Returns:
        NumericDistributionComparison with all metrics.
    """
    # Handle nulls
    mask = original.notna() & protected.notna()
    orig_valid = original[mask].astype(float).values
    prot_valid = protected[mask].astype(float).values

    if len(orig_valid) == 0:
        return NumericDistributionComparison(
            wasserstein=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            kl_divergence=None,
            js_distance=None,
            mean_shift=0.0,
            std_ratio=1.0,
        )

    # Wasserstein distance
    wass = wasserstein_distance(orig_valid, prot_valid)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(orig_valid, prot_valid)

    # Histogram-based divergences
    kl: Optional[float] = None
    js: Optional[float] = None

    try:
        # Use common bin edges for both distributions
        combined = np.concatenate([orig_valid, prot_valid])
        bin_range: Tuple[float, float] = (float(np.min(combined)), float(np.max(combined)))

        orig_hist, bin_edges = np.histogram(
            orig_valid, bins=bins, range=bin_range, density=False
        )
        prot_hist, _ = np.histogram(
            prot_valid, bins=bins, range=bin_range, density=False
        )

        # Normalize to probabilities
        orig_probs = (orig_hist + 1e-10) / (orig_hist.sum() + 1e-10 * len(orig_hist))
        prot_probs = (prot_hist + 1e-10) / (prot_hist.sum() + 1e-10 * len(prot_hist))

        kl = kl_divergence(orig_probs, prot_probs, base="2")
        js = js_distance(orig_probs, prot_probs, base="2")
    except Exception:
        pass

    # Mean shift (normalized by original std)
    orig_mean = np.mean(orig_valid)
    prot_mean = np.mean(prot_valid)
    orig_std = np.std(orig_valid)
    if orig_std > 0:
        mean_shift = (prot_mean - orig_mean) / orig_std
    else:
        mean_shift = 0.0 if orig_mean == prot_mean else float("inf")

    # Standard deviation ratio
    prot_std = np.std(prot_valid)
    if orig_std > 0:
        std_ratio = prot_std / orig_std
    else:
        std_ratio = 1.0 if prot_std == 0 else float("inf")

    return NumericDistributionComparison(
        wasserstein=float(wass),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        kl_divergence=kl,
        js_distance=js,
        mean_shift=float(mean_shift),
        std_ratio=float(std_ratio),
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_kl_divergence(
    original: pd.Series,
    protected: pd.Series,
    bins: int = 50,
) -> Optional[float]:
    """Calculate KL divergence for a series (convenience function).

    For categorical data, uses category frequencies directly.
    For numeric data, uses histogram binning.

    Args:
        original: Original series.
        protected: Protected series.
        bins: Number of bins for numeric data.

    Returns:
        KL divergence or None if calculation fails.
    """
    # Handle nulls
    mask = original.notna() & protected.notna()

    if mask.sum() == 0:
        return None

    orig_valid = original[mask]
    prot_valid = protected[mask]

    # Check if categorical
    if orig_valid.dtype == "object" or orig_valid.dtype.name == "category":
        # Categorical
        all_cats = list(set(orig_valid.unique()) | set(prot_valid.unique()))
        orig_probs, _ = series_to_distribution(orig_valid, all_cats)
        prot_probs, _ = series_to_distribution(prot_valid, all_cats)
        return kl_divergence(orig_probs, prot_probs)
    else:
        # Numeric
        result = compare_numeric_distributions(orig_valid, prot_valid, bins)
        return result.kl_divergence


def calculate_js_distance(
    original: pd.Series,
    protected: pd.Series,
    bins: int = 50,
) -> Optional[float]:
    """Calculate Jensen-Shannon distance for a series (convenience function).

    Args:
        original: Original series.
        protected: Protected series.
        bins: Number of bins for numeric data.

    Returns:
        JS distance or None if calculation fails.
    """
    mask = original.notna() & protected.notna()

    if mask.sum() == 0:
        return None

    orig_valid = original[mask]
    prot_valid = protected[mask]

    if orig_valid.dtype == "object" or orig_valid.dtype.name == "category":
        all_cats = list(set(orig_valid.unique()) | set(prot_valid.unique()))
        orig_probs, _ = series_to_distribution(orig_valid, all_cats)
        prot_probs, _ = series_to_distribution(prot_valid, all_cats)
        return js_distance(orig_probs, prot_probs)
    else:
        result = compare_numeric_distributions(orig_valid, prot_valid, bins)
        return result.js_distance


def calculate_wasserstein_distance(
    original: pd.Series,
    protected: pd.Series,
) -> float:
    """Calculate Wasserstein distance for a series (convenience function).

    Args:
        original: Original series.
        protected: Protected series.

    Returns:
        Wasserstein distance.
    """
    mask = original.notna() & protected.notna()

    if mask.sum() == 0:
        return 0.0

    orig_valid = original[mask].astype(float).values
    prot_valid = protected[mask].astype(float).values

    return wasserstein_distance(orig_valid, prot_valid)


def calculate_tvd(
    original: pd.Series,
    protected: pd.Series,
    bins: int = 50,
) -> Optional[float]:
    """Calculate Total Variation Distance for a series.

    Args:
        original: Original series.
        protected: Protected series.
        bins: Number of bins for numeric data.

    Returns:
        TVD or None if calculation fails.
    """
    mask = original.notna() & protected.notna()

    if mask.sum() == 0:
        return None

    orig_valid = original[mask]
    prot_valid = protected[mask]

    if orig_valid.dtype == "object" or orig_valid.dtype.name == "category":
        all_cats = list(set(orig_valid.unique()) | set(prot_valid.unique()))
        orig_probs, _ = series_to_distribution(orig_valid, all_cats)
        prot_probs, _ = series_to_distribution(prot_valid, all_cats)
        return total_variation_distance(orig_probs, prot_probs)
    else:
        # For numeric, use histograms
        combined = np.concatenate(
            [orig_valid.astype(float).values, prot_valid.astype(float).values]
        )
        bin_range: Tuple[float, float] = (float(np.min(combined)), float(np.max(combined)))

        orig_hist, _ = np.histogram(
            orig_valid.astype(float).values, bins=bins, range=bin_range, density=False
        )
        prot_hist, _ = np.histogram(
            prot_valid.astype(float).values, bins=bins, range=bin_range, density=False
        )

        orig_probs = orig_hist / orig_hist.sum()
        prot_probs = prot_hist / prot_hist.sum()

        return total_variation_distance(orig_probs, prot_probs)


def calculate_all_divergences(
    original: pd.Series,
    protected: pd.Series,
    bins: int = 50,
) -> Dict[str, Optional[float]]:
    """Calculate all divergence metrics for a series.

    Args:
        original: Original series.
        protected: Protected series.
        bins: Number of bins for numeric data.

    Returns:
        Dictionary with all divergence metrics.
    """
    return {
        "kl_divergence": calculate_kl_divergence(original, protected, bins),
        "js_distance": calculate_js_distance(original, protected, bins),
        "wasserstein": calculate_wasserstein_distance(original, protected),
        "tvd": calculate_tvd(original, protected, bins),
    }
