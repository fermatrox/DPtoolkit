"""Privacy budget tracking and composition.

This module provides tools for tracking privacy budget consumption
across multiple differential privacy queries using composition theorems.

Composition Methods:
- Sequential (Basic): Total epsilon = sum of epsilons
- Sequential (Advanced): Tighter bounds using advanced composition theorem
- Parallel: Total epsilon = max of epsilons (for disjoint data partitions)

References:
- Dwork & Roth (2014): The Algorithmic Foundations of Differential Privacy
- OpenDP: https://docs.opendp.org/
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import uuid


# =============================================================================
# Constants
# =============================================================================

# Default delta for advanced composition when not specified
DEFAULT_DELTA_PRIME = 1e-10


# =============================================================================
# Enums
# =============================================================================


class CompositionMethod(Enum):
    """Methods for composing privacy budgets.

    Attributes:
        SEQUENTIAL_BASIC: Simple sum of epsilons and deltas.
        SEQUENTIAL_ADVANCED: Tighter bounds using advanced composition theorem.
        PARALLEL: Max of epsilons for disjoint data partitions.
    """

    SEQUENTIAL_BASIC = "sequential_basic"
    SEQUENTIAL_ADVANCED = "sequential_advanced"
    PARALLEL = "parallel"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class PrivacyBudget:
    """Represents a privacy budget with epsilon and optional delta.

    Attributes:
        epsilon: Privacy parameter for pure ε-DP or (ε,δ)-DP.
        delta: Approximate DP parameter, None for pure ε-DP.
    """

    epsilon: float
    delta: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate budget parameters."""
        if self.epsilon < 0:
            raise ValueError(f"Epsilon must be non-negative, got {self.epsilon}")
        if self.delta is not None and (self.delta < 0 or self.delta > 1):
            raise ValueError(f"Delta must be in [0, 1], got {self.delta}")

    def __add__(self, other: "PrivacyBudget") -> "PrivacyBudget":
        """Add two budgets (sequential composition)."""
        if not isinstance(other, PrivacyBudget):
            return NotImplemented
        new_epsilon = self.epsilon + other.epsilon
        # Handle delta: if either is None, use the other; else sum
        if self.delta is None and other.delta is None:
            new_delta = None
        elif self.delta is None:
            new_delta = other.delta
        elif other.delta is None:
            new_delta = self.delta
        else:
            new_delta = self.delta + other.delta
        return PrivacyBudget(epsilon=new_epsilon, delta=new_delta)

    def __le__(self, other: "PrivacyBudget") -> bool:
        """Check if this budget is within another budget."""
        if not isinstance(other, PrivacyBudget):
            return NotImplemented
        # Epsilon check
        if self.epsilon > other.epsilon:
            return False
        # Delta check
        if self.delta is not None:
            if other.delta is None:
                return False  # Can't fit (ε,δ) in pure ε-DP
            if self.delta > other.delta:
                return False
        return True

    def __lt__(self, other: "PrivacyBudget") -> bool:
        """Check if this budget is strictly less than another."""
        if not isinstance(other, PrivacyBudget):
            return NotImplemented
        return self <= other and (
            self.epsilon < other.epsilon
            or (
                self.delta is not None
                and other.delta is not None
                and self.delta < other.delta
            )
        )

    def is_pure_dp(self) -> bool:
        """Check if this is pure ε-DP (no delta)."""
        return self.delta is None or self.delta == 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.is_pure_dp():
            return f"ε={self.epsilon:.4f}"
        return f"ε={self.epsilon:.4f}, δ={self.delta:.2e}"


@dataclass
class BudgetQuery:
    """Record of a single privacy budget consumption.

    Attributes:
        query_id: Unique identifier for this query.
        name: Human-readable name for the query.
        epsilon: Epsilon consumed by this query.
        delta: Delta consumed (None for pure ε-DP).
        mechanism: Type of mechanism used (e.g., "laplace", "gaussian").
        column: Column name this query was applied to (if applicable).
        timestamp: When the query was recorded.
        metadata: Additional metadata about the query.
    """

    query_id: str
    name: str
    epsilon: float
    delta: Optional[float] = None
    mechanism: Optional[str] = None
    column: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def budget(self) -> PrivacyBudget:
        """Get the privacy budget for this query."""
        return PrivacyBudget(epsilon=self.epsilon, delta=self.delta)


# =============================================================================
# Composition Functions
# =============================================================================


def compose_sequential_basic(
    budgets: List[PrivacyBudget],
) -> PrivacyBudget:
    """Compose budgets using basic sequential composition.

    Basic composition theorem: If M1, ..., Mk are (εi, δi)-DP mechanisms,
    their sequential composition is (∑εi, ∑δi)-DP.

    Args:
        budgets: List of privacy budgets to compose.

    Returns:
        Composed privacy budget.
    """
    if not budgets:
        return PrivacyBudget(epsilon=0.0, delta=None)

    total_epsilon = sum(b.epsilon for b in budgets)
    deltas = [b.delta for b in budgets if b.delta is not None]

    if deltas:
        total_delta = sum(deltas)
    else:
        total_delta = None

    return PrivacyBudget(epsilon=total_epsilon, delta=total_delta)


def compose_sequential_advanced(
    budgets: List[PrivacyBudget],
    delta_prime: float = DEFAULT_DELTA_PRIME,
) -> PrivacyBudget:
    """Compose budgets using advanced composition theorem.

    Advanced composition theorem provides tighter bounds for multiple
    queries. For k queries each with (ε, δ)-DP:

    Total is (ε', kδ + δ')-DP where:
    ε' = sqrt(2k * ln(1/δ')) * ε + k * ε * (e^ε - 1)

    For heterogeneous queries, uses the maximum epsilon.

    Args:
        budgets: List of privacy budgets to compose.
        delta_prime: Additional delta for the composition (default 1e-10).

    Returns:
        Composed privacy budget with tighter bounds.

    Note:
        Falls back to basic composition if tighter bounds aren't achievable.
    """
    if not budgets:
        return PrivacyBudget(epsilon=0.0, delta=None)

    k = len(budgets)

    # Get maximum epsilon for heterogeneous composition
    max_epsilon = max(b.epsilon for b in budgets)

    # Calculate deltas
    deltas = [b.delta for b in budgets if b.delta is not None]
    sum_delta = sum(deltas) if deltas else 0.0

    # Advanced composition formula
    # ε' = sqrt(2k * ln(1/δ')) * ε + k * ε * (e^ε - 1)
    term1 = math.sqrt(2 * k * math.log(1 / delta_prime)) * max_epsilon
    term2 = k * max_epsilon * (math.exp(max_epsilon) - 1)
    advanced_epsilon = term1 + term2

    # Basic composition epsilon for comparison
    basic_epsilon = sum(b.epsilon for b in budgets)

    # Use the better (lower) of the two bounds
    composed_delta: Optional[float]
    if advanced_epsilon < basic_epsilon:
        composed_epsilon = advanced_epsilon
        composed_delta = sum_delta + delta_prime
    else:
        # Fall back to basic composition
        composed_epsilon = basic_epsilon
        composed_delta = sum_delta if sum_delta > 0 else None

    # Ensure we have delta if we used advanced composition
    if composed_delta == 0:
        composed_delta = delta_prime

    return PrivacyBudget(epsilon=composed_epsilon, delta=composed_delta)


def compose_parallel(
    budgets: List[PrivacyBudget],
) -> PrivacyBudget:
    """Compose budgets using parallel composition.

    Parallel composition theorem: If mechanisms operate on disjoint
    subsets of the data, the total privacy cost is the maximum of
    the individual costs.

    Args:
        budgets: List of privacy budgets to compose (for disjoint data).

    Returns:
        Composed privacy budget (max of inputs).
    """
    if not budgets:
        return PrivacyBudget(epsilon=0.0, delta=None)

    max_epsilon = max(b.epsilon for b in budgets)
    deltas = [b.delta for b in budgets if b.delta is not None]

    if deltas:
        max_delta = max(deltas)
    else:
        max_delta = None

    return PrivacyBudget(epsilon=max_epsilon, delta=max_delta)


# =============================================================================
# Privacy Budget Tracker
# =============================================================================


class PrivacyBudgetTracker:
    """Tracks privacy budget consumption across multiple queries.

    This class maintains a record of all privacy-consuming queries
    and computes the total budget used according to the specified
    composition method.

    Attributes:
        total_budget: Maximum allowed privacy budget.
        composition_method: Method for composing query budgets.
        queries: List of recorded queries.

    Example:
        >>> tracker = PrivacyBudgetTracker(
        ...     total_budget=PrivacyBudget(epsilon=1.0),
        ...     composition_method=CompositionMethod.SEQUENTIAL_BASIC
        ... )
        >>> tracker.record_query("age", epsilon=0.3, mechanism="laplace")
        >>> tracker.record_query("salary", epsilon=0.3, mechanism="laplace")
        >>> tracker.consumed_budget
        PrivacyBudget(epsilon=0.6, delta=None)
        >>> tracker.remaining_budget
        PrivacyBudget(epsilon=0.4, delta=None)
    """

    def __init__(
        self,
        total_budget: Optional[PrivacyBudget] = None,
        composition_method: CompositionMethod = (CompositionMethod.SEQUENTIAL_BASIC),
        delta_prime: float = DEFAULT_DELTA_PRIME,
    ) -> None:
        """Initialize the budget tracker.

        Args:
            total_budget: Maximum allowed privacy budget (None for unlimited).
            composition_method: Method for composing query budgets.
            delta_prime: Delta parameter for advanced composition.
        """
        self._total_budget = total_budget
        self._composition_method = composition_method
        self._delta_prime = delta_prime
        self._queries: List[BudgetQuery] = []

    @property
    def total_budget(self) -> Optional[PrivacyBudget]:
        """Get the total allowed budget."""
        return self._total_budget

    @total_budget.setter
    def total_budget(self, budget: Optional[PrivacyBudget]) -> None:
        """Set the total allowed budget."""
        self._total_budget = budget

    @property
    def composition_method(self) -> CompositionMethod:
        """Get the composition method."""
        return self._composition_method

    @composition_method.setter
    def composition_method(self, method: CompositionMethod) -> None:
        """Set the composition method."""
        self._composition_method = method

    @property
    def queries(self) -> List[BudgetQuery]:
        """Get list of recorded queries (copy)."""
        return list(self._queries)

    @property
    def query_count(self) -> int:
        """Get the number of recorded queries."""
        return len(self._queries)

    def record_query(
        self,
        name: str,
        epsilon: float,
        delta: Optional[float] = None,
        mechanism: Optional[str] = None,
        column: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        check_budget: bool = True,
    ) -> BudgetQuery:
        """Record a privacy-consuming query.

        Args:
            name: Human-readable name for the query.
            epsilon: Epsilon consumed by this query.
            delta: Delta consumed (None for pure ε-DP).
            mechanism: Type of mechanism used.
            column: Column name this query was applied to.
            metadata: Additional metadata about the query.
            check_budget: Whether to check if budget is exceeded.

        Returns:
            The recorded BudgetQuery object.

        Raises:
            BudgetExceededError: If the query would exceed the total budget.
        """
        query = BudgetQuery(
            query_id=str(uuid.uuid4()),
            name=name,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            column=column,
            metadata=metadata or {},
        )

        # Check if this query would exceed budget
        if check_budget and self._total_budget is not None:
            hypothetical_queries = self._queries + [query]
            hypothetical_consumed = self._compute_consumed(hypothetical_queries)
            if not hypothetical_consumed <= self._total_budget:
                raise BudgetExceededError(
                    f"Query '{name}' would exceed budget. "
                    f"Required: {hypothetical_consumed}, "
                    f"Available: {self._total_budget}"
                )

        self._queries.append(query)
        return query

    def _compute_consumed(
        self, queries: Optional[List[BudgetQuery]] = None
    ) -> PrivacyBudget:
        """Compute total consumed budget for given queries."""
        if queries is None:
            queries = self._queries

        if not queries:
            return PrivacyBudget(epsilon=0.0, delta=None)

        budgets = [q.budget for q in queries]

        if self._composition_method == CompositionMethod.SEQUENTIAL_BASIC:
            return compose_sequential_basic(budgets)
        elif self._composition_method == CompositionMethod.SEQUENTIAL_ADVANCED:
            return compose_sequential_advanced(budgets, self._delta_prime)
        elif self._composition_method == CompositionMethod.PARALLEL:
            return compose_parallel(budgets)
        else:
            raise ValueError(f"Unknown composition method: {self._composition_method}")

    @property
    def consumed_budget(self) -> PrivacyBudget:
        """Get the total consumed budget."""
        return self._compute_consumed()

    @property
    def remaining_budget(self) -> Optional[PrivacyBudget]:
        """Get the remaining budget (None if unlimited).

        Returns:
            Remaining budget, or None if no total budget set.
        """
        if self._total_budget is None:
            return None

        consumed = self.consumed_budget

        remaining_epsilon = max(0.0, self._total_budget.epsilon - consumed.epsilon)

        if self._total_budget.delta is not None:
            consumed_delta = consumed.delta or 0.0
            remaining_delta = max(0.0, self._total_budget.delta - consumed_delta)
        else:
            remaining_delta = None

        return PrivacyBudget(epsilon=remaining_epsilon, delta=remaining_delta)

    @property
    def is_budget_exceeded(self) -> bool:
        """Check if the budget has been exceeded."""
        if self._total_budget is None:
            return False
        return not self.consumed_budget <= self._total_budget

    @property
    def budget_utilization(self) -> float:
        """Get budget utilization as a fraction (0.0 to 1.0+).

        Returns:
            Fraction of budget consumed (can exceed 1.0 if over budget).
            Returns 0.0 if no budget is set.
        """
        if self._total_budget is None or self._total_budget.epsilon == 0:
            return 0.0
        return self.consumed_budget.epsilon / self._total_budget.epsilon

    def can_afford(
        self,
        epsilon: float,
        delta: Optional[float] = None,
    ) -> bool:
        """Check if a query with given budget can be afforded.

        Args:
            epsilon: Epsilon for the hypothetical query.
            delta: Delta for the hypothetical query.

        Returns:
            True if the query can be afforded, False otherwise.
        """
        if self._total_budget is None:
            return True

        hypothetical = BudgetQuery(
            query_id="hypothetical",
            name="hypothetical",
            epsilon=epsilon,
            delta=delta,
        )
        hypothetical_queries = self._queries + [hypothetical]
        hypothetical_consumed = self._compute_consumed(hypothetical_queries)
        return hypothetical_consumed <= self._total_budget

    def allocate_budget(
        self,
        n_queries: int,
        equal: bool = True,
    ) -> List[PrivacyBudget]:
        """Allocate remaining budget across future queries.

        Args:
            n_queries: Number of queries to allocate budget for.
            equal: If True, allocate equally; otherwise return remaining.

        Returns:
            List of privacy budgets for each query.

        Raises:
            ValueError: If no budget is set or n_queries < 1.
        """
        if self._total_budget is None:
            raise ValueError("Cannot allocate budget: no total budget set")
        if n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {n_queries}")

        remaining = self.remaining_budget
        if remaining is None or remaining.epsilon <= 0:
            return [PrivacyBudget(epsilon=0.0) for _ in range(n_queries)]

        if equal:
            per_query_epsilon = remaining.epsilon / n_queries
            if remaining.delta is not None:
                per_query_delta = remaining.delta / n_queries
            else:
                per_query_delta = None
            return [
                PrivacyBudget(epsilon=per_query_epsilon, delta=per_query_delta)
                for _ in range(n_queries)
            ]
        else:
            # Return full remaining budget for each (parallel composition)
            return [remaining for _ in range(n_queries)]

    def get_queries_by_column(self, column: str) -> List[BudgetQuery]:
        """Get all queries for a specific column.

        Args:
            column: Column name to filter by.

        Returns:
            List of queries for that column.
        """
        return [q for q in self._queries if q.column == column]

    def get_queries_by_mechanism(self, mechanism: str) -> List[BudgetQuery]:
        """Get all queries using a specific mechanism.

        Args:
            mechanism: Mechanism type to filter by.

        Returns:
            List of queries using that mechanism.
        """
        return [q for q in self._queries if q.mechanism == mechanism]

    def get_column_budgets(self) -> Dict[str, PrivacyBudget]:
        """Get budget consumed per column.

        Returns:
            Dictionary mapping column names to consumed budgets.
        """
        columns: Dict[str, List[BudgetQuery]] = {}
        for query in self._queries:
            if query.column:
                if query.column not in columns:
                    columns[query.column] = []
                columns[query.column].append(query)

        return {
            col: compose_sequential_basic([q.budget for q in queries])
            for col, queries in columns.items()
        }

    def reset(self) -> None:
        """Clear all recorded queries."""
        self._queries.clear()

    def remove_query(self, query_id: str) -> bool:
        """Remove a query by its ID.

        Args:
            query_id: ID of the query to remove.

        Returns:
            True if query was removed, False if not found.
        """
        for i, query in enumerate(self._queries):
            if query.query_id == query_id:
                self._queries.pop(i)
                return True
        return False

    def summary(self) -> Dict[str, object]:
        """Get a summary of budget tracking state.

        Returns:
            Dictionary with budget tracking summary.
        """
        return {
            "total_budget": (
                str(self._total_budget) if self._total_budget else "unlimited"
            ),
            "consumed_budget": str(self.consumed_budget),
            "remaining_budget": (
                str(self.remaining_budget) if self.remaining_budget else "unlimited"
            ),
            "composition_method": self._composition_method.value,
            "query_count": self.query_count,
            "budget_utilization": f"{self.budget_utilization:.1%}",
            "is_exceeded": self.is_budget_exceeded,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PrivacyBudgetTracker("
            f"total={self._total_budget}, "
            f"consumed={self.consumed_budget}, "
            f"queries={self.query_count}, "
            f"method={self._composition_method.value})"
        )


# =============================================================================
# Exceptions
# =============================================================================


class BudgetExceededError(Exception):
    """Raised when a query would exceed the privacy budget."""

    pass


# =============================================================================
# Convenience Functions
# =============================================================================


def create_budget_tracker(
    epsilon: float,
    delta: Optional[float] = None,
    composition: str = "basic",
) -> PrivacyBudgetTracker:
    """Create a privacy budget tracker with specified total budget.

    Args:
        epsilon: Total epsilon budget.
        delta: Total delta budget (None for pure ε-DP).
        composition: Composition method ("basic", "advanced", or "parallel").

    Returns:
        Configured PrivacyBudgetTracker.

    Example:
        >>> tracker = create_budget_tracker(epsilon=1.0)
        >>> tracker.record_query("query1", epsilon=0.5)
        >>> tracker.remaining_budget.epsilon
        0.5
    """
    method_map = {
        "basic": CompositionMethod.SEQUENTIAL_BASIC,
        "advanced": CompositionMethod.SEQUENTIAL_ADVANCED,
        "parallel": CompositionMethod.PARALLEL,
    }
    if composition not in method_map:
        raise ValueError(
            f"Unknown composition method: {composition}. "
            f"Use one of: {list(method_map.keys())}"
        )

    return PrivacyBudgetTracker(
        total_budget=PrivacyBudget(epsilon=epsilon, delta=delta),
        composition_method=method_map[composition],
    )


def calculate_total_budget(
    queries: List[Tuple[float, Optional[float]]],
    composition: str = "basic",
) -> PrivacyBudget:
    """Calculate total budget for a list of (epsilon, delta) pairs.

    Args:
        queries: List of (epsilon, delta) tuples.
        composition: Composition method ("basic", "advanced", or "parallel").

    Returns:
        Total composed privacy budget.

    Example:
        >>> calculate_total_budget([(0.3, None), (0.3, None), (0.4, None)])
        PrivacyBudget(epsilon=1.0, delta=None)
    """
    budgets = [PrivacyBudget(epsilon=e, delta=d) for e, d in queries]

    if composition == "basic":
        return compose_sequential_basic(budgets)
    elif composition == "advanced":
        return compose_sequential_advanced(budgets)
    elif composition == "parallel":
        return compose_parallel(budgets)
    else:
        raise ValueError(f"Unknown composition method: {composition}")
