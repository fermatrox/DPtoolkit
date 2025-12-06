"""Unit tests for privacy budget tracking module."""

import math
import pytest
from datetime import datetime

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


# =============================================================================
# PrivacyBudget Tests
# =============================================================================


class TestPrivacyBudget:
    """Tests for PrivacyBudget dataclass."""

    def test_pure_dp_budget(self):
        """Test creating a pure ε-DP budget."""
        budget = PrivacyBudget(epsilon=1.0)
        assert budget.epsilon == 1.0
        assert budget.delta is None
        assert budget.is_pure_dp()

    def test_approximate_dp_budget(self):
        """Test creating an (ε,δ)-DP budget."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert not budget.is_pure_dp()

    def test_zero_delta_is_pure(self):
        """Test that delta=0 is considered pure DP."""
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        assert budget.is_pure_dp()

    def test_negative_epsilon_raises(self):
        """Test that negative epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon must be non-negative"):
            PrivacyBudget(epsilon=-0.1)

    def test_negative_delta_raises(self):
        """Test that negative delta raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=-0.1)

    def test_delta_greater_than_one_raises(self):
        """Test that delta > 1 raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=1.5)

    def test_budget_addition_pure(self):
        """Test adding two pure DP budgets."""
        b1 = PrivacyBudget(epsilon=0.3)
        b2 = PrivacyBudget(epsilon=0.4)
        result = b1 + b2
        assert result.epsilon == 0.7
        assert result.delta is None

    def test_budget_addition_approximate(self):
        """Test adding two (ε,δ)-DP budgets."""
        b1 = PrivacyBudget(epsilon=0.3, delta=1e-5)
        b2 = PrivacyBudget(epsilon=0.4, delta=2e-5)
        result = b1 + b2
        assert result.epsilon == 0.7
        assert result.delta == pytest.approx(3e-5)

    def test_budget_addition_mixed(self):
        """Test adding pure and approximate budgets."""
        b1 = PrivacyBudget(epsilon=0.3)
        b2 = PrivacyBudget(epsilon=0.4, delta=1e-5)
        result = b1 + b2
        assert result.epsilon == 0.7
        assert result.delta == 1e-5

    def test_budget_comparison_le(self):
        """Test <= comparison of budgets."""
        small = PrivacyBudget(epsilon=0.5)
        large = PrivacyBudget(epsilon=1.0)
        assert small <= large
        assert not large <= small
        assert small <= small

    def test_budget_comparison_with_delta(self):
        """Test comparison with delta values."""
        b1 = PrivacyBudget(epsilon=0.5, delta=1e-5)
        b2 = PrivacyBudget(epsilon=1.0, delta=1e-4)
        assert b1 <= b2
        assert not b2 <= b1

    def test_approximate_cannot_fit_in_pure(self):
        """Test that (ε,δ) budget cannot fit in pure ε budget."""
        approximate = PrivacyBudget(epsilon=0.5, delta=1e-5)
        pure = PrivacyBudget(epsilon=1.0)
        assert not approximate <= pure

    def test_budget_str_pure(self):
        """Test string representation for pure DP."""
        budget = PrivacyBudget(epsilon=1.0)
        assert "ε=1.0" in str(budget)
        assert "δ" not in str(budget)

    def test_budget_str_approximate(self):
        """Test string representation for approximate DP."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert "ε=1.0" in str(budget)
        assert "δ=" in str(budget)

    def test_budget_immutable(self):
        """Test that budget is frozen (immutable)."""
        budget = PrivacyBudget(epsilon=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            budget.epsilon = 2.0


# =============================================================================
# BudgetQuery Tests
# =============================================================================


class TestBudgetQuery:
    """Tests for BudgetQuery dataclass."""

    def test_query_creation(self):
        """Test creating a budget query."""
        query = BudgetQuery(
            query_id="test-123",
            name="test_query",
            epsilon=0.5,
        )
        assert query.query_id == "test-123"
        assert query.name == "test_query"
        assert query.epsilon == 0.5
        assert query.delta is None

    def test_query_with_all_fields(self):
        """Test query with all optional fields."""
        query = BudgetQuery(
            query_id="test-123",
            name="test_query",
            epsilon=0.5,
            delta=1e-5,
            mechanism="laplace",
            column="age",
            metadata={"note": "test"},
        )
        assert query.mechanism == "laplace"
        assert query.column == "age"
        assert query.metadata == {"note": "test"}

    def test_query_budget_property(self):
        """Test that budget property returns correct PrivacyBudget."""
        query = BudgetQuery(
            query_id="test-123",
            name="test_query",
            epsilon=0.5,
            delta=1e-5,
        )
        budget = query.budget
        assert isinstance(budget, PrivacyBudget)
        assert budget.epsilon == 0.5
        assert budget.delta == 1e-5

    def test_query_has_timestamp(self):
        """Test that query gets a timestamp."""
        query = BudgetQuery(
            query_id="test-123",
            name="test_query",
            epsilon=0.5,
        )
        assert isinstance(query.timestamp, datetime)


# =============================================================================
# Composition Function Tests
# =============================================================================


class TestSequentialBasicComposition:
    """Tests for basic sequential composition."""

    def test_empty_list(self):
        """Test composition of empty list."""
        result = compose_sequential_basic([])
        assert result.epsilon == 0.0
        assert result.delta is None

    def test_single_budget(self):
        """Test composition of single budget."""
        budgets = [PrivacyBudget(epsilon=0.5)]
        result = compose_sequential_basic(budgets)
        assert result.epsilon == 0.5
        assert result.delta is None

    def test_multiple_pure_budgets(self):
        """Test composition of multiple pure DP budgets."""
        budgets = [
            PrivacyBudget(epsilon=0.3),
            PrivacyBudget(epsilon=0.3),
            PrivacyBudget(epsilon=0.4),
        ]
        result = compose_sequential_basic(budgets)
        assert result.epsilon == 1.0
        assert result.delta is None

    def test_multiple_approximate_budgets(self):
        """Test composition of multiple (ε,δ)-DP budgets."""
        budgets = [
            PrivacyBudget(epsilon=0.3, delta=1e-6),
            PrivacyBudget(epsilon=0.3, delta=1e-6),
            PrivacyBudget(epsilon=0.4, delta=1e-6),
        ]
        result = compose_sequential_basic(budgets)
        assert result.epsilon == 1.0
        assert result.delta == 3e-6

    def test_mixed_budgets(self):
        """Test composition of mixed pure and approximate budgets."""
        budgets = [
            PrivacyBudget(epsilon=0.3),
            PrivacyBudget(epsilon=0.3, delta=1e-6),
        ]
        result = compose_sequential_basic(budgets)
        assert result.epsilon == 0.6
        assert result.delta == 1e-6


class TestSequentialAdvancedComposition:
    """Tests for advanced sequential composition."""

    def test_empty_list(self):
        """Test composition of empty list."""
        result = compose_sequential_advanced([])
        assert result.epsilon == 0.0
        assert result.delta is None

    def test_single_budget(self):
        """Test composition of single budget returns basic."""
        budgets = [PrivacyBudget(epsilon=0.5)]
        result = compose_sequential_advanced(budgets)
        # For single budget, basic and advanced are same
        assert result.epsilon == 0.5

    def test_many_small_queries_tighter_bound(self):
        """Test that advanced composition gives tighter bound for many queries."""
        # 100 queries with epsilon=0.01 each
        # Basic: 100 * 0.01 = 1.0
        # Advanced should be less
        budgets = [PrivacyBudget(epsilon=0.01) for _ in range(100)]

        basic = compose_sequential_basic(budgets)
        advanced = compose_sequential_advanced(budgets)

        assert basic.epsilon == 1.0
        assert advanced.epsilon < basic.epsilon

    def test_few_large_queries_fallback_to_basic(self):
        """Test that advanced falls back to basic for few large queries."""
        # 2 queries with epsilon=1.0 each
        # Basic: 2.0
        # Advanced might be worse, should fallback
        budgets = [PrivacyBudget(epsilon=1.0) for _ in range(2)]

        basic = compose_sequential_basic(budgets)
        advanced = compose_sequential_advanced(budgets)

        # Should use the better (lower) bound
        assert advanced.epsilon <= basic.epsilon

    def test_custom_delta_prime(self):
        """Test composition with custom delta_prime."""
        # Use many small queries to trigger advanced composition
        budgets = [PrivacyBudget(epsilon=0.01) for _ in range(100)]
        result = compose_sequential_advanced(budgets, delta_prime=1e-6)
        # Advanced composition should give tighter bound and add delta
        assert result.delta is not None
        assert result.epsilon < 1.0  # Basic would be 1.0


class TestParallelComposition:
    """Tests for parallel composition."""

    def test_empty_list(self):
        """Test composition of empty list."""
        result = compose_parallel([])
        assert result.epsilon == 0.0
        assert result.delta is None

    def test_single_budget(self):
        """Test composition of single budget."""
        budgets = [PrivacyBudget(epsilon=0.5)]
        result = compose_parallel(budgets)
        assert result.epsilon == 0.5

    def test_multiple_budgets_takes_max(self):
        """Test that parallel composition takes maximum."""
        budgets = [
            PrivacyBudget(epsilon=0.3),
            PrivacyBudget(epsilon=0.5),
            PrivacyBudget(epsilon=0.4),
        ]
        result = compose_parallel(budgets)
        assert result.epsilon == 0.5

    def test_with_delta_takes_max(self):
        """Test parallel composition with delta takes max of each."""
        budgets = [
            PrivacyBudget(epsilon=0.3, delta=1e-6),
            PrivacyBudget(epsilon=0.5, delta=1e-5),
            PrivacyBudget(epsilon=0.4, delta=1e-7),
        ]
        result = compose_parallel(budgets)
        assert result.epsilon == 0.5
        assert result.delta == 1e-5


# =============================================================================
# PrivacyBudgetTracker Tests
# =============================================================================


class TestPrivacyBudgetTracker:
    """Tests for PrivacyBudgetTracker class."""

    def test_tracker_creation_with_budget(self):
        """Test creating tracker with total budget."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        assert tracker.total_budget.epsilon == 1.0
        assert tracker.query_count == 0

    def test_tracker_creation_unlimited(self):
        """Test creating tracker without budget limit."""
        tracker = PrivacyBudgetTracker()
        assert tracker.total_budget is None
        assert tracker.remaining_budget is None

    def test_record_query(self):
        """Test recording a query."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        query = tracker.record_query("test", epsilon=0.5)
        assert query.name == "test"
        assert query.epsilon == 0.5
        assert tracker.query_count == 1

    def test_record_query_with_details(self):
        """Test recording query with all details."""
        tracker = PrivacyBudgetTracker()
        query = tracker.record_query(
            name="age_noise",
            epsilon=0.5,
            delta=1e-5,
            mechanism="laplace",
            column="age",
            metadata={"bounds": "0-100"},
        )
        assert query.mechanism == "laplace"
        assert query.column == "age"
        assert query.metadata == {"bounds": "0-100"}

    def test_consumed_budget(self):
        """Test that consumed budget is calculated correctly."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.2)

        consumed = tracker.consumed_budget
        assert consumed.epsilon == 0.5

    def test_remaining_budget(self):
        """Test that remaining budget is calculated correctly."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.2)

        remaining = tracker.remaining_budget
        assert remaining.epsilon == 0.5

    def test_budget_exceeded_raises(self):
        """Test that exceeding budget raises error."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.6)

        with pytest.raises(BudgetExceededError):
            tracker.record_query("q2", epsilon=0.5)

    def test_budget_exceeded_skip_check(self):
        """Test recording without budget check."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.6)
        # Should not raise with check_budget=False
        tracker.record_query("q2", epsilon=0.5, check_budget=False)
        assert tracker.is_budget_exceeded

    def test_is_budget_exceeded(self):
        """Test budget exceeded flag."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.5)
        assert not tracker.is_budget_exceeded

        tracker.record_query("q2", epsilon=0.5, check_budget=False)
        assert not tracker.is_budget_exceeded  # Exactly at limit

        tracker.record_query("q3", epsilon=0.1, check_budget=False)
        assert tracker.is_budget_exceeded

    def test_budget_utilization(self):
        """Test budget utilization calculation."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        assert tracker.budget_utilization == 0.0

        tracker.record_query("q1", epsilon=0.5)
        assert tracker.budget_utilization == 0.5

        tracker.record_query("q2", epsilon=0.25)
        assert tracker.budget_utilization == 0.75

    def test_can_afford(self):
        """Test can_afford check."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.5)

        assert tracker.can_afford(epsilon=0.5)
        assert tracker.can_afford(epsilon=0.4)
        assert not tracker.can_afford(epsilon=0.6)

    def test_allocate_budget_equal(self):
        """Test equal budget allocation."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        allocations = tracker.allocate_budget(n_queries=4, equal=True)

        assert len(allocations) == 4
        for alloc in allocations:
            assert alloc.epsilon == 0.25

    def test_allocate_budget_with_consumed(self):
        """Test allocation with some budget already consumed."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.5)
        allocations = tracker.allocate_budget(n_queries=2, equal=True)

        assert len(allocations) == 2
        for alloc in allocations:
            assert alloc.epsilon == 0.25

    def test_allocate_budget_unlimited_raises(self):
        """Test that allocation without budget raises error."""
        tracker = PrivacyBudgetTracker()
        with pytest.raises(ValueError, match="no total budget"):
            tracker.allocate_budget(n_queries=2)

    def test_get_queries_by_column(self):
        """Test filtering queries by column."""
        tracker = PrivacyBudgetTracker()
        tracker.record_query("q1", epsilon=0.3, column="age")
        tracker.record_query("q2", epsilon=0.3, column="salary")
        tracker.record_query("q3", epsilon=0.4, column="age")

        age_queries = tracker.get_queries_by_column("age")
        assert len(age_queries) == 2
        assert all(q.column == "age" for q in age_queries)

    def test_get_queries_by_mechanism(self):
        """Test filtering queries by mechanism."""
        tracker = PrivacyBudgetTracker()
        tracker.record_query("q1", epsilon=0.3, mechanism="laplace")
        tracker.record_query("q2", epsilon=0.3, mechanism="gaussian")
        tracker.record_query("q3", epsilon=0.4, mechanism="laplace")

        laplace_queries = tracker.get_queries_by_mechanism("laplace")
        assert len(laplace_queries) == 2
        assert all(q.mechanism == "laplace" for q in laplace_queries)

    def test_get_column_budgets(self):
        """Test getting budget per column."""
        tracker = PrivacyBudgetTracker()
        tracker.record_query("q1", epsilon=0.3, column="age")
        tracker.record_query("q2", epsilon=0.3, column="salary")
        tracker.record_query("q3", epsilon=0.4, column="age")

        column_budgets = tracker.get_column_budgets()
        assert column_budgets["age"].epsilon == 0.7
        assert column_budgets["salary"].epsilon == 0.3

    def test_reset(self):
        """Test resetting the tracker."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.5)
        assert tracker.query_count == 1

        tracker.reset()
        assert tracker.query_count == 0
        assert tracker.consumed_budget.epsilon == 0.0

    def test_remove_query(self):
        """Test removing a query by ID."""
        tracker = PrivacyBudgetTracker()
        q1 = tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.4)

        assert tracker.query_count == 2
        result = tracker.remove_query(q1.query_id)
        assert result is True
        assert tracker.query_count == 1
        assert tracker.consumed_budget.epsilon == 0.4

    def test_remove_query_not_found(self):
        """Test removing non-existent query."""
        tracker = PrivacyBudgetTracker()
        result = tracker.remove_query("nonexistent")
        assert result is False

    def test_summary(self):
        """Test summary generation."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=0.5)

        summary = tracker.summary()
        assert "total_budget" in summary
        assert "consumed_budget" in summary
        assert "remaining_budget" in summary
        assert "query_count" in summary
        assert summary["query_count"] == 1

    def test_repr(self):
        """Test string representation."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        repr_str = repr(tracker)
        assert "PrivacyBudgetTracker" in repr_str
        assert "total=" in repr_str


class TestTrackerCompositionMethods:
    """Tests for different composition methods in tracker."""

    def test_sequential_basic_composition(self):
        """Test tracker with basic sequential composition."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0),
            composition_method=CompositionMethod.SEQUENTIAL_BASIC,
        )
        tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.3)
        tracker.record_query("q3", epsilon=0.4)

        assert tracker.consumed_budget.epsilon == 1.0

    def test_sequential_advanced_composition(self):
        """Test tracker with advanced sequential composition."""
        # Advanced composition adds delta_prime, so total budget needs delta
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=2.0, delta=1e-5),
            composition_method=CompositionMethod.SEQUENTIAL_ADVANCED,
        )
        # Many small queries
        for _ in range(100):
            tracker.record_query("q", epsilon=0.01)

        # Advanced composition should give tighter bound
        basic_total = 100 * 0.01  # = 1.0
        assert tracker.consumed_budget.epsilon < basic_total

    def test_parallel_composition(self):
        """Test tracker with parallel composition."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0),
            composition_method=CompositionMethod.PARALLEL,
        )
        tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.5)
        tracker.record_query("q3", epsilon=0.4)

        # Parallel takes max
        assert tracker.consumed_budget.epsilon == 0.5

    def test_change_composition_method(self):
        """Test changing composition method."""
        tracker = PrivacyBudgetTracker(
            composition_method=CompositionMethod.SEQUENTIAL_BASIC
        )
        tracker.record_query("q1", epsilon=0.3)
        tracker.record_query("q2", epsilon=0.5)

        # Basic: sum = 0.8
        assert tracker.consumed_budget.epsilon == 0.8

        # Change to parallel
        tracker.composition_method = CompositionMethod.PARALLEL
        # Parallel: max = 0.5
        assert tracker.consumed_budget.epsilon == 0.5


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_budget_tracker_basic(self):
        """Test creating tracker with basic composition."""
        tracker = create_budget_tracker(epsilon=1.0, composition="basic")
        assert tracker.total_budget.epsilon == 1.0
        assert tracker.composition_method == CompositionMethod.SEQUENTIAL_BASIC

    def test_create_budget_tracker_advanced(self):
        """Test creating tracker with advanced composition."""
        tracker = create_budget_tracker(epsilon=1.0, composition="advanced")
        assert tracker.composition_method == CompositionMethod.SEQUENTIAL_ADVANCED

    def test_create_budget_tracker_parallel(self):
        """Test creating tracker with parallel composition."""
        tracker = create_budget_tracker(epsilon=1.0, composition="parallel")
        assert tracker.composition_method == CompositionMethod.PARALLEL

    def test_create_budget_tracker_with_delta(self):
        """Test creating tracker with delta."""
        tracker = create_budget_tracker(epsilon=1.0, delta=1e-5)
        assert tracker.total_budget.delta == 1e-5

    def test_create_budget_tracker_invalid_composition(self):
        """Test error for invalid composition method."""
        with pytest.raises(ValueError, match="Unknown composition"):
            create_budget_tracker(epsilon=1.0, composition="invalid")

    def test_calculate_total_budget_basic(self):
        """Test calculating total budget with basic composition."""
        queries = [(0.3, None), (0.3, None), (0.4, None)]
        result = calculate_total_budget(queries, composition="basic")
        assert result.epsilon == 1.0

    def test_calculate_total_budget_parallel(self):
        """Test calculating total budget with parallel composition."""
        queries = [(0.3, None), (0.5, None), (0.4, None)]
        result = calculate_total_budget(queries, composition="parallel")
        assert result.epsilon == 0.5

    def test_calculate_total_budget_with_delta(self):
        """Test calculating total budget with delta."""
        queries = [(0.3, 1e-6), (0.3, 1e-6), (0.4, 1e-6)]
        result = calculate_total_budget(queries, composition="basic")
        assert result.epsilon == 1.0
        assert result.delta == 3e-6


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_epsilon_budget(self):
        """Test with zero epsilon budget."""
        budget = PrivacyBudget(epsilon=0.0)
        assert budget.epsilon == 0.0

    def test_very_small_epsilon(self):
        """Test with very small epsilon."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=0.001)
        )
        tracker.record_query("q1", epsilon=0.0005)
        assert tracker.remaining_budget.epsilon == pytest.approx(0.0005)

    def test_very_large_number_of_queries(self):
        """Test with many queries."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=100.0)
        )
        for i in range(1000):
            tracker.record_query(f"q{i}", epsilon=0.01)

        assert tracker.query_count == 1000
        assert tracker.consumed_budget.epsilon == pytest.approx(10.0)

    def test_queries_list_is_copy(self):
        """Test that queries property returns a copy."""
        tracker = PrivacyBudgetTracker()
        tracker.record_query("q1", epsilon=0.5)

        queries = tracker.queries
        queries.clear()  # Modify the copy

        assert tracker.query_count == 1  # Original unchanged

    def test_budget_at_exact_limit(self):
        """Test budget exactly at limit."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0)
        )
        tracker.record_query("q1", epsilon=1.0)

        assert not tracker.is_budget_exceeded
        assert tracker.remaining_budget.epsilon == 0.0
        assert not tracker.can_afford(epsilon=0.001)

    def test_default_delta_prime_constant(self):
        """Test that default delta prime constant is set."""
        assert DEFAULT_DELTA_PRIME == 1e-10

    def test_tracker_with_only_delta_queries(self):
        """Test tracker with queries that have delta."""
        tracker = PrivacyBudgetTracker(
            total_budget=PrivacyBudget(epsilon=1.0, delta=1e-4)
        )
        tracker.record_query("q1", epsilon=0.5, delta=1e-5)
        tracker.record_query("q2", epsilon=0.3, delta=2e-5)

        assert tracker.consumed_budget.epsilon == 0.8
        assert tracker.consumed_budget.delta == pytest.approx(3e-5)
        assert tracker.remaining_budget.delta == pytest.approx(1e-4 - 3e-5)
