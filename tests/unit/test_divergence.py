"""Unit tests for dp_toolkit.analysis.divergence module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy
from scipy.stats import wasserstein_distance as scipy_wasserstein

from dp_toolkit.analysis.divergence import (
    # Enums
    DivergenceType,
    # Data classes
    DivergenceResult,
    CategoryDriftResult,
    NumericDistributionComparison,
    # Core functions
    kl_divergence,
    js_distance,
    js_divergence,
    wasserstein_distance,
    total_variation_distance,
    hellinger_distance,
    # Entropy functions
    entropy,
    cross_entropy,
    # Utility functions
    series_to_distribution,
    numeric_to_histogram,
    # Analysis functions
    analyze_category_drift,
    compare_numeric_distributions,
    # Convenience functions
    calculate_kl_divergence,
    calculate_js_distance,
    calculate_wasserstein_distance,
    calculate_tvd,
    calculate_all_divergences,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def uniform_distribution():
    """Uniform probability distribution."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def skewed_distribution():
    """Skewed probability distribution."""
    return np.array([0.7, 0.1, 0.1, 0.1])


@pytest.fixture
def identical_distributions():
    """Two identical distributions."""
    p = np.array([0.3, 0.3, 0.2, 0.2])
    return p, p.copy()


@pytest.fixture
def different_distributions():
    """Two different distributions."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    return p, q


@pytest.fixture
def categorical_series():
    """Categorical pandas series pair."""
    original = pd.Series(["A", "A", "B", "B", "C"])
    protected = pd.Series(["A", "B", "B", "B", "C"])
    return original, protected


@pytest.fixture
def numeric_series():
    """Numeric pandas series pair."""
    np.random.seed(42)
    original = pd.Series(np.random.normal(100, 15, 1000))
    protected = pd.Series(np.random.normal(102, 16, 1000))
    return original, protected


# =============================================================================
# Tests: KL Divergence
# =============================================================================


class TestKLDivergence:
    """Tests for KL divergence calculation."""

    def test_kl_identical_distributions(self, identical_distributions):
        """KL divergence of identical distributions is zero."""
        p, q = identical_distributions
        kl = kl_divergence(p, q)
        assert abs(kl) < 1e-10

    def test_kl_positive(self, different_distributions):
        """KL divergence is always non-negative."""
        p, q = different_distributions
        kl = kl_divergence(p, q)
        assert kl >= 0

    def test_kl_asymmetric(self):
        """KL divergence is asymmetric."""
        # Use truly asymmetric distributions
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != kl_qp

    def test_kl_vs_scipy(self, different_distributions):
        """KL divergence matches scipy implementation."""
        p, q = different_distributions
        # Our implementation with natural log
        our_kl = kl_divergence(p, q, base="e")
        # scipy uses natural log
        scipy_kl = scipy_entropy(p, q)
        assert abs(our_kl - scipy_kl) < 1e-10

    def test_kl_base_2(self, different_distributions):
        """KL divergence in bits (base 2)."""
        p, q = different_distributions
        kl_nats = kl_divergence(p, q, base="e")
        kl_bits = kl_divergence(p, q, base="2")
        # bits = nats / ln(2)
        assert abs(kl_bits - kl_nats / np.log(2)) < 1e-10

    def test_kl_with_zeros_in_q(self):
        """KL divergence handles zeros in Q with smoothing."""
        p = np.array([0.5, 0.3, 0.2, 0.0])
        q = np.array([0.4, 0.3, 0.3, 0.0])  # Zero where p is zero is okay
        kl = kl_divergence(p, q)
        assert np.isfinite(kl)

    def test_kl_smoothing_prevents_infinity(self):
        """Smoothing prevents infinite KL when Q has zero where P is nonzero."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([1.0, 0.0, 0.0])  # Q is zero where P is nonzero
        kl = kl_divergence(p, q, smoothing=1e-10)
        assert np.isfinite(kl)

    def test_kl_length_mismatch_raises(self):
        """KL divergence raises error for different lengths."""
        p = np.array([0.5, 0.5])
        q = np.array([0.33, 0.33, 0.34])
        with pytest.raises(ValueError, match="same length"):
            kl_divergence(p, q)

    def test_kl_extreme_skew(self):
        """KL divergence with extreme distribution skew."""
        p = np.array([0.99, 0.01])
        q = np.array([0.01, 0.99])
        kl = kl_divergence(p, q)
        assert kl > 0
        assert np.isfinite(kl)


# =============================================================================
# Tests: Jensen-Shannon Distance
# =============================================================================


class TestJSDistance:
    """Tests for Jensen-Shannon distance calculation."""

    def test_js_identical_distributions(self, identical_distributions):
        """JS distance of identical distributions is zero."""
        p, q = identical_distributions
        js = js_distance(p, q)
        assert abs(js) < 1e-10

    def test_js_symmetric(self, different_distributions):
        """JS distance is symmetric."""
        p, q = different_distributions
        js_pq = js_distance(p, q)
        js_qp = js_distance(q, p)
        assert abs(js_pq - js_qp) < 1e-10

    def test_js_bounded_zero_one(self, different_distributions):
        """JS distance is bounded between 0 and 1 (with log base 2)."""
        p, q = different_distributions
        js = js_distance(p, q, base="2")
        assert 0 <= js <= 1

    def test_js_vs_scipy(self, different_distributions):
        """JS distance matches scipy implementation."""
        p, q = different_distributions
        our_js = js_distance(p, q, base="2")
        scipy_js = jensenshannon(p, q, base=2.0)
        assert abs(our_js - scipy_js) < 1e-10

    def test_js_maximized_opposite(self):
        """JS distance is maximized for completely opposite distributions."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        js = js_distance(p, q, base="2")
        assert abs(js - 1.0) < 1e-10

    def test_js_divergence_is_square(self, different_distributions):
        """JS divergence is square of JS distance."""
        p, q = different_distributions
        jsd = js_divergence(p, q)
        js = js_distance(p, q)
        assert abs(jsd - js**2) < 1e-10

    def test_js_with_zeros(self):
        """JS distance handles zero probabilities."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.0, 0.5, 0.5])
        js = js_distance(p, q)
        assert np.isfinite(js)
        assert 0 <= js <= 1


# =============================================================================
# Tests: Wasserstein Distance
# =============================================================================


class TestWassersteinDistance:
    """Tests for Wasserstein distance calculation."""

    def test_wasserstein_identical(self):
        """Wasserstein distance of identical values is zero."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = wasserstein_distance(values, values.copy())
        assert abs(w) < 1e-10

    def test_wasserstein_shift(self):
        """Wasserstein distance equals shift for uniform shift."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shifted = original + 10.0
        w = wasserstein_distance(original, shifted)
        assert abs(w - 10.0) < 1e-10

    def test_wasserstein_vs_scipy(self):
        """Wasserstein distance matches scipy implementation."""
        np.random.seed(42)
        values1 = np.random.normal(0, 1, 1000)
        values2 = np.random.normal(1, 1.5, 1000)
        our_w = wasserstein_distance(values1, values2)
        scipy_w = scipy_wasserstein(values1, values2)
        assert abs(our_w - scipy_w) < 1e-10

    def test_wasserstein_non_negative(self):
        """Wasserstein distance is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            v1 = np.random.randn(100)
            v2 = np.random.randn(100)
            w = wasserstein_distance(v1, v2)
            assert w >= 0

    def test_wasserstein_handles_nan(self):
        """Wasserstein distance handles NaN values."""
        values1 = np.array([1.0, 2.0, np.nan, 4.0])
        values2 = np.array([1.5, np.nan, 3.0, 4.5])
        w = wasserstein_distance(values1, values2)
        assert np.isfinite(w)

    def test_wasserstein_empty_returns_zero(self):
        """Wasserstein distance of empty arrays is zero."""
        w = wasserstein_distance(np.array([]), np.array([]))
        assert w == 0.0


# =============================================================================
# Tests: Total Variation Distance
# =============================================================================


class TestTotalVariationDistance:
    """Tests for Total Variation Distance calculation."""

    def test_tvd_identical(self, identical_distributions):
        """TVD of identical distributions is zero."""
        p, q = identical_distributions
        tvd = total_variation_distance(p, q)
        assert abs(tvd) < 1e-10

    def test_tvd_bounded(self, different_distributions):
        """TVD is bounded between 0 and 1."""
        p, q = different_distributions
        tvd = total_variation_distance(p, q)
        assert 0 <= tvd <= 1

    def test_tvd_symmetric(self, different_distributions):
        """TVD is symmetric."""
        p, q = different_distributions
        tvd_pq = total_variation_distance(p, q)
        tvd_qp = total_variation_distance(q, p)
        assert abs(tvd_pq - tvd_qp) < 1e-10

    def test_tvd_maximized_opposite(self):
        """TVD is 1 for completely disjoint distributions."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        tvd = total_variation_distance(p, q)
        assert abs(tvd - 1.0) < 1e-10

    def test_tvd_formula(self):
        """TVD equals 0.5 * sum of absolute differences."""
        p = np.array([0.4, 0.3, 0.3])
        q = np.array([0.2, 0.4, 0.4])
        expected = 0.5 * np.sum(np.abs(p - q))
        tvd = total_variation_distance(p, q)
        assert abs(tvd - expected) < 1e-10


# =============================================================================
# Tests: Hellinger Distance
# =============================================================================


class TestHellingerDistance:
    """Tests for Hellinger distance calculation."""

    def test_hellinger_identical(self, identical_distributions):
        """Hellinger distance of identical distributions is zero."""
        p, q = identical_distributions
        h = hellinger_distance(p, q)
        assert abs(h) < 1e-10

    def test_hellinger_bounded(self, different_distributions):
        """Hellinger distance is bounded between 0 and 1."""
        p, q = different_distributions
        h = hellinger_distance(p, q)
        assert 0 <= h <= 1

    def test_hellinger_symmetric(self, different_distributions):
        """Hellinger distance is symmetric."""
        p, q = different_distributions
        h_pq = hellinger_distance(p, q)
        h_qp = hellinger_distance(q, p)
        assert abs(h_pq - h_qp) < 1e-10

    def test_hellinger_maximized_opposite(self):
        """Hellinger distance is 1 for completely disjoint distributions."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        h = hellinger_distance(p, q)
        assert abs(h - 1.0) < 1e-10

    def test_hellinger_vs_tvd_relation(self, different_distributions):
        """Hellinger and TVD satisfy known inequality."""
        p, q = different_distributions
        h = hellinger_distance(p, q)
        tvd = total_variation_distance(p, q)
        # H^2 <= TVD <= H * sqrt(2)
        assert h**2 <= tvd + 1e-10
        assert tvd <= h * np.sqrt(2) + 1e-10


# =============================================================================
# Tests: Entropy Functions
# =============================================================================


class TestEntropy:
    """Tests for entropy functions."""

    def test_entropy_uniform_maximum(self, uniform_distribution):
        """Uniform distribution has maximum entropy."""
        n = len(uniform_distribution)
        max_entropy = np.log2(n)  # Maximum entropy for n categories
        h = entropy(uniform_distribution, base="2")
        assert abs(h - max_entropy) < 1e-10

    def test_entropy_certain_is_zero(self):
        """Certain distribution has zero entropy."""
        p = np.array([1.0, 0.0, 0.0])
        h = entropy(p)
        assert abs(h) < 1e-10

    def test_entropy_vs_scipy(self, skewed_distribution):
        """Entropy matches scipy implementation."""
        our_h = entropy(skewed_distribution, base="e")
        scipy_h = scipy_entropy(skewed_distribution)
        assert abs(our_h - scipy_h) < 1e-10

    def test_cross_entropy_relation(self, different_distributions):
        """Cross-entropy = entropy + KL divergence."""
        p, q = different_distributions
        # Use same base for all calculations
        h = entropy(p, base="2")
        kl = kl_divergence(p, q, base="2")
        ce = cross_entropy(p, q, base="2")
        assert abs(ce - (h + kl)) < 1e-10


# =============================================================================
# Tests: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_series_to_distribution(self):
        """Convert series to distribution."""
        series = pd.Series(["A", "A", "B", "B", "B", "C"])
        probs, categories = series_to_distribution(series)

        assert len(probs) == 3
        assert abs(probs.sum() - 1.0) < 1e-10
        # B should have highest probability (3/6 = 0.5)
        b_idx = categories.index("B")
        assert abs(probs[b_idx] - 0.5) < 1e-10

    def test_series_to_distribution_with_categories(self):
        """Convert series with predefined categories."""
        series = pd.Series(["A", "B"])
        probs, categories = series_to_distribution(series, categories=["A", "B", "C"])

        assert len(probs) == 3
        c_idx = categories.index("C")
        assert probs[c_idx] == 0.0

    def test_numeric_to_histogram(self):
        """Convert numeric values to histogram."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        probs, edges = numeric_to_histogram(values, bins=5)

        assert len(probs) == 5
        assert len(edges) == 6
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_numeric_to_histogram_empty(self):
        """Handle empty array."""
        probs, edges = numeric_to_histogram(np.array([]))
        assert len(probs) == 0


# =============================================================================
# Tests: Category Drift Analysis
# =============================================================================


class TestCategoryDriftAnalysis:
    """Tests for category drift analysis."""

    def test_analyze_category_drift_basic(self, categorical_series):
        """Basic category drift analysis."""
        original, protected = categorical_series
        result = analyze_category_drift(original, protected)

        assert isinstance(result, CategoryDriftResult)
        assert 0 <= result.drift_rate <= 1
        assert result.kl_divergence >= 0
        assert 0 <= result.js_distance <= 1
        assert 0 <= result.tvd <= 1

    def test_drift_rate_calculation(self, categorical_series):
        """Verify drift rate calculation."""
        original, protected = categorical_series
        result = analyze_category_drift(original, protected)

        # One value changed: A->B at position 1
        expected_drift = 1 / 5
        assert abs(result.drift_rate - expected_drift) < 1e-10

    def test_no_drift(self):
        """Zero drift when series are identical."""
        series = pd.Series(["A", "B", "C"])
        result = analyze_category_drift(series, series.copy())

        assert result.drift_rate == 0.0
        assert result.kl_divergence < 1e-10
        assert result.js_distance < 1e-10

    def test_full_drift(self):
        """Full drift when all values change."""
        original = pd.Series(["A", "A", "A"])
        protected = pd.Series(["B", "B", "B"])
        result = analyze_category_drift(original, protected)

        assert result.drift_rate == 1.0

    def test_entropy_calculated(self, categorical_series):
        """Entropy values are calculated."""
        original, protected = categorical_series
        result = analyze_category_drift(original, protected)

        assert result.entropy_original > 0
        assert result.entropy_protected > 0

    def test_top_changes_populated(self, categorical_series):
        """Top changes list is populated."""
        original, protected = categorical_series
        result = analyze_category_drift(original, protected)

        assert len(result.top_changes) > 0
        # Each entry is (category, orig_freq, prot_freq, diff)
        assert len(result.top_changes[0]) == 4

    def test_with_nulls(self):
        """Handle null values."""
        original = pd.Series(["A", None, "B", "C"])
        protected = pd.Series(["A", "B", None, "C"])
        result = analyze_category_drift(original, protected)

        # Should only compare the 2 valid rows (A and C)
        assert result.drift_rate == 0.0  # A->A and C->C

    def test_wasserstein_optional(self, categorical_series):
        """Wasserstein is optional for categorical."""
        original, protected = categorical_series
        result = analyze_category_drift(original, protected, include_wasserstein=False)
        assert result.wasserstein is None

        result_with = analyze_category_drift(
            original, protected, include_wasserstein=True
        )
        assert result_with.wasserstein is not None


# =============================================================================
# Tests: Numeric Distribution Comparison
# =============================================================================


class TestNumericDistributionComparison:
    """Tests for numeric distribution comparison."""

    def test_compare_numeric_basic(self, numeric_series):
        """Basic numeric distribution comparison."""
        original, protected = numeric_series
        result = compare_numeric_distributions(original, protected)

        assert isinstance(result, NumericDistributionComparison)
        assert result.wasserstein >= 0
        assert 0 <= result.ks_statistic <= 1
        assert 0 <= result.ks_pvalue <= 1

    def test_identical_distributions(self):
        """Comparison of identical distributions."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compare_numeric_distributions(series, series.copy())

        assert result.wasserstein < 1e-10
        assert result.ks_statistic < 1e-10
        assert result.mean_shift == 0.0
        assert abs(result.std_ratio - 1.0) < 1e-10

    def test_mean_shift_normalized(self):
        """Mean shift is normalized by std."""
        original = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])  # mean=20, std=15.81
        std_orig = np.std(original, ddof=0)  # ~15.81
        protected = original + std_orig  # Shift by exactly 1 std
        result = compare_numeric_distributions(original, protected)

        assert abs(result.mean_shift - 1.0) < 0.2  # Allow some tolerance

    def test_std_ratio_calculation(self):
        """Standard deviation ratio calculation."""
        original = pd.Series([1, 2, 3, 4, 5])
        protected = pd.Series([2, 4, 6, 8, 10])  # 2x original
        result = compare_numeric_distributions(original, protected)

        assert abs(result.std_ratio - 2.0) < 1e-10

    def test_ks_test_significant(self):
        """KS test detects significantly different distributions."""
        np.random.seed(42)
        original = pd.Series(np.random.normal(0, 1, 1000))
        protected = pd.Series(np.random.normal(5, 1, 1000))  # Very different
        result = compare_numeric_distributions(original, protected)

        assert result.ks_pvalue < 0.05

    def test_with_nulls(self):
        """Handle null values."""
        original = pd.Series([1.0, 2.0, None, 4.0])
        protected = pd.Series([1.5, None, 3.0, 4.5])
        result = compare_numeric_distributions(original, protected)

        assert np.isfinite(result.wasserstein)


# =============================================================================
# Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_kl_divergence_categorical(self, categorical_series):
        """KL divergence for categorical series."""
        original, protected = categorical_series
        kl = calculate_kl_divergence(original, protected)

        assert kl is not None
        assert kl >= 0

    def test_calculate_kl_divergence_numeric(self, numeric_series):
        """KL divergence for numeric series."""
        original, protected = numeric_series
        kl = calculate_kl_divergence(original, protected)

        assert kl is not None
        assert kl >= 0

    def test_calculate_js_distance_symmetric(self, categorical_series):
        """JS distance is symmetric for series."""
        original, protected = categorical_series
        js_ab = calculate_js_distance(original, protected)
        js_ba = calculate_js_distance(protected, original)

        assert js_ab is not None
        assert abs(js_ab - js_ba) < 1e-10

    def test_calculate_wasserstein_distance(self, numeric_series):
        """Wasserstein distance for numeric series."""
        original, protected = numeric_series
        w = calculate_wasserstein_distance(original, protected)

        assert w >= 0

    def test_calculate_tvd(self, categorical_series):
        """Total variation distance for series."""
        original, protected = categorical_series
        tvd = calculate_tvd(original, protected)

        assert tvd is not None
        assert 0 <= tvd <= 1

    def test_calculate_all_divergences(self, numeric_series):
        """Calculate all divergences at once."""
        original, protected = numeric_series
        results = calculate_all_divergences(original, protected)

        assert "kl_divergence" in results
        assert "js_distance" in results
        assert "wasserstein" in results
        assert "tvd" in results

    def test_all_none_for_empty(self):
        """Handle empty series."""
        original = pd.Series([], dtype=float)
        protected = pd.Series([], dtype=float)

        kl = calculate_kl_divergence(original, protected)
        js = calculate_js_distance(original, protected)
        w = calculate_wasserstein_distance(original, protected)

        assert kl is None
        assert js is None
        assert w == 0.0


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_category(self):
        """Handle single category."""
        original = pd.Series(["A", "A", "A"])
        protected = pd.Series(["A", "A", "A"])
        result = analyze_category_drift(original, protected)

        assert result.drift_rate == 0.0
        assert result.entropy_original == 0.0

    def test_many_categories(self):
        """Handle many categories."""
        n = 1000
        original = pd.Series([f"cat_{i % 100}" for i in range(n)])
        protected = pd.Series([f"cat_{(i + 1) % 100}" for i in range(n)])
        result = analyze_category_drift(original, protected)

        assert result.drift_rate > 0
        assert len(result.top_changes) <= 10

    def test_very_small_probabilities(self):
        """Handle very small probabilities."""
        p = np.array([1e-10, 1.0 - 1e-10])
        q = np.array([1.0 - 1e-10, 1e-10])

        kl = kl_divergence(p, q)
        js = js_distance(p, q)

        assert np.isfinite(kl)
        assert np.isfinite(js)

    def test_all_nulls(self):
        """Handle all-null series."""
        original = pd.Series([None, None, None])
        protected = pd.Series([None, None, None])

        result = analyze_category_drift(original, protected)
        assert result.drift_rate == 0.0

    def test_unicode_categories(self):
        """Handle unicode category names."""
        original = pd.Series(["日本語", "中文", "한국어"])
        protected = pd.Series(["日本語", "日本語", "한국어"])
        result = analyze_category_drift(original, protected)

        assert result.drift_rate == pytest.approx(1 / 3)

    def test_large_numeric_values(self):
        """Handle large numeric values."""
        original = pd.Series([1e10, 2e10, 3e10])
        protected = pd.Series([1.1e10, 2.1e10, 3.1e10])
        result = compare_numeric_distributions(original, protected)

        assert np.isfinite(result.wasserstein)

    def test_negative_values(self):
        """Handle negative numeric values."""
        original = pd.Series([-100, -50, 0, 50, 100])
        protected = pd.Series([-95, -45, 5, 55, 105])
        result = compare_numeric_distributions(original, protected)

        assert np.isfinite(result.wasserstein)
        assert np.isfinite(result.mean_shift)

    def test_single_value(self):
        """Handle single-value series."""
        original = pd.Series([5.0])
        protected = pd.Series([6.0])
        result = compare_numeric_distributions(original, protected)

        assert result.wasserstein == 1.0

    def test_mixed_types_categorical(self):
        """Handle mixed types coerced to string."""
        original = pd.Series([1, "A", 2.5])
        protected = pd.Series(["1", "A", "2.5"])
        # Should work after string coercion
        result = analyze_category_drift(
            original.astype(str), protected.astype(str)
        )
        assert result.drift_rate == 0.0


# =============================================================================
# Tests: Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Tests for mathematical properties of divergences."""

    def test_triangle_inequality_js(self):
        """JS distance satisfies triangle inequality."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.4, 0.3])
        r = np.array([0.2, 0.3, 0.5])

        js_pq = js_distance(p, q)
        js_qr = js_distance(q, r)
        js_pr = js_distance(p, r)

        # Triangle inequality: d(p,r) <= d(p,q) + d(q,r)
        assert js_pr <= js_pq + js_qr + 1e-10

    def test_metric_properties_hellinger(self):
        """Hellinger distance satisfies metric properties."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.4, 0.3])
        r = np.array([0.2, 0.3, 0.5])

        # Non-negativity
        assert hellinger_distance(p, q) >= 0

        # Identity of indiscernibles
        assert hellinger_distance(p, p) < 1e-10

        # Symmetry
        assert abs(hellinger_distance(p, q) - hellinger_distance(q, p)) < 1e-10

        # Triangle inequality
        h_pq = hellinger_distance(p, q)
        h_qr = hellinger_distance(q, r)
        h_pr = hellinger_distance(p, r)
        assert h_pr <= h_pq + h_qr + 1e-10

    def test_kl_chain_rule(self):
        """KL divergence satisfies chain rule for independent distributions."""
        # For independent P(X,Y) = P(X)P(Y), KL decomposes
        px = np.array([0.6, 0.4])
        qx = np.array([0.5, 0.5])

        py = np.array([0.7, 0.3])
        qy = np.array([0.4, 0.6])

        # Joint distribution (outer product for independence)
        pxy = np.outer(px, py).flatten()
        qxy = np.outer(qx, qy).flatten()

        kl_joint = kl_divergence(pxy, qxy)
        kl_x = kl_divergence(px, qx)
        kl_y = kl_divergence(py, qy)

        # For independent: KL(PXY || QXY) = KL(PX || QX) + KL(PY || QY)
        assert abs(kl_joint - (kl_x + kl_y)) < 1e-10
