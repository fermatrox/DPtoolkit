"""Statistical comparator for original vs protected datasets.

This module provides functionality to compare original and protected
datasets, calculating divergence metrics and analyzing how well
statistical properties are preserved after differential privacy
protection.

Key features:
- Per-column comparison (numeric, categorical, date)
- Divergence metrics: MAE, RMSE, MAPE
- Correlation preservation analysis
- Dataset-level comparison summary
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# Enums and Constants
# =============================================================================


class ComparisonType(Enum):
    """Type of column comparison."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"


# =============================================================================
# Data Classes - Numeric Comparison
# =============================================================================


@dataclass
class NumericDivergence:
    """Divergence metrics for numeric column comparison.

    Attributes:
        mae: Mean Absolute Error.
        rmse: Root Mean Square Error.
        mape: Mean Absolute Percentage Error (None if zeros present).
        max_absolute_error: Maximum absolute difference.
        mean_original: Mean of original column.
        mean_protected: Mean of protected column.
        mean_difference: Difference in means.
        mean_relative_error: Relative error in means.
        std_original: Standard deviation of original.
        std_protected: Standard deviation of protected.
        std_difference: Difference in standard deviations.
        std_relative_error: Relative error in std.
        median_original: Median of original.
        median_protected: Median of protected.
        min_original: Min of original.
        min_protected: Min of protected.
        max_original: Max of original.
        max_protected: Max of protected.
    """

    mae: float
    rmse: float
    mape: Optional[float]
    max_absolute_error: float
    mean_original: float
    mean_protected: float
    mean_difference: float
    mean_relative_error: Optional[float]
    std_original: float
    std_protected: float
    std_difference: float
    std_relative_error: Optional[float]
    median_original: float
    median_protected: float
    min_original: float
    min_protected: float
    max_original: float
    max_protected: float


@dataclass
class NumericComparison:
    """Comparison result for a numeric column.

    Attributes:
        column_name: Name of the column.
        comparison_type: Type of comparison (NUMERIC).
        count: Number of non-null values compared.
        null_count: Number of null values.
        divergence: Divergence metrics.
        percentile_comparison: Comparison of key percentiles.
    """

    column_name: str
    comparison_type: ComparisonType
    count: int
    null_count: int
    divergence: NumericDivergence
    percentile_comparison: Dict[str, Tuple[float, float, float]] = field(
        default_factory=dict
    )


# =============================================================================
# Data Classes - Categorical Comparison
# =============================================================================


@dataclass
class CategoricalDivergence:
    """Divergence metrics for categorical column comparison.

    Attributes:
        category_drift: Proportion of values that changed category.
        cardinality_original: Number of unique categories in original.
        cardinality_protected: Number of unique categories in protected.
        cardinality_difference: Difference in cardinality.
        mode_original: Most frequent category in original.
        mode_protected: Most frequent category in protected.
        mode_preserved: Whether the mode is the same.
        frequency_mae: MAE of category frequencies.
        new_categories: Categories in protected but not original.
        missing_categories: Categories in original but not protected.
    """

    category_drift: float
    cardinality_original: int
    cardinality_protected: int
    cardinality_difference: int
    mode_original: Optional[str]
    mode_protected: Optional[str]
    mode_preserved: bool
    frequency_mae: float
    new_categories: List[str]
    missing_categories: List[str]


@dataclass
class CategoricalComparison:
    """Comparison result for a categorical column.

    Attributes:
        column_name: Name of the column.
        comparison_type: Type of comparison (CATEGORICAL).
        count: Number of non-null values compared.
        null_count: Number of null values.
        divergence: Divergence metrics.
        frequency_comparison: Original vs protected frequencies.
    """

    column_name: str
    comparison_type: ComparisonType
    count: int
    null_count: int
    divergence: CategoricalDivergence
    frequency_comparison: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# =============================================================================
# Data Classes - Date Comparison
# =============================================================================


@dataclass
class DateDivergence:
    """Divergence metrics for date column comparison.

    Attributes:
        mae_days: Mean Absolute Error in days.
        rmse_days: Root Mean Square Error in days.
        max_absolute_error_days: Maximum absolute difference in days.
        range_original_days: Date range of original in days.
        range_protected_days: Date range of protected in days.
        range_difference_days: Difference in date ranges.
        min_date_diff_days: Difference in minimum dates.
        max_date_diff_days: Difference in maximum dates.
    """

    mae_days: float
    rmse_days: float
    max_absolute_error_days: float
    range_original_days: int
    range_protected_days: int
    range_difference_days: int
    min_date_diff_days: float
    max_date_diff_days: float


@dataclass
class DateComparison:
    """Comparison result for a date column.

    Attributes:
        column_name: Name of the column.
        comparison_type: Type of comparison (DATE).
        count: Number of non-null values compared.
        null_count: Number of null values.
        divergence: Divergence metrics.
    """

    column_name: str
    comparison_type: ComparisonType
    count: int
    null_count: int
    divergence: DateDivergence


# =============================================================================
# Data Classes - Correlation Comparison
# =============================================================================


@dataclass
class CorrelationPreservation:
    """Metrics for correlation preservation analysis.

    Attributes:
        original_correlations: Dict of column pairs to original correlation.
        protected_correlations: Dict of column pairs to protected correlation.
        correlation_differences: Dict of column pairs to correlation difference.
        mae: Mean Absolute Error of correlations.
        rmse: Root Mean Square Error of correlations.
        max_absolute_error: Maximum correlation difference.
        preserved_count: Number of correlations preserved within threshold.
        total_count: Total number of correlations compared.
        preservation_rate: Rate of preserved correlations.
    """

    original_correlations: Dict[Tuple[str, str], float]
    protected_correlations: Dict[Tuple[str, str], float]
    correlation_differences: Dict[Tuple[str, str], float]
    mae: float
    rmse: float
    max_absolute_error: float
    preserved_count: int
    total_count: int
    preservation_rate: float


# =============================================================================
# Data Classes - Dataset Comparison
# =============================================================================


@dataclass
class DatasetComparison:
    """Full dataset comparison result.

    Attributes:
        row_count: Number of rows compared.
        column_count: Number of columns compared.
        numeric_comparisons: List of numeric column comparisons.
        categorical_comparisons: List of categorical column comparisons.
        date_comparisons: List of date column comparisons.
        correlation_preservation: Correlation preservation analysis.
        overall_numeric_mae: Average MAE across numeric columns.
        overall_numeric_rmse: Average RMSE across numeric columns.
    """

    row_count: int
    column_count: int
    numeric_comparisons: List[NumericComparison]
    categorical_comparisons: List[CategoricalComparison]
    date_comparisons: List[DateComparison]
    correlation_preservation: Optional[CorrelationPreservation]
    overall_numeric_mae: Optional[float]
    overall_numeric_rmse: Optional[float]

    def get_comparison(
        self, column_name: str
    ) -> Optional[Union[NumericComparison, CategoricalComparison, DateComparison]]:
        """Get comparison for a specific column.

        Args:
            column_name: Name of the column.

        Returns:
            Column comparison if found, None otherwise.
        """
        for num_comp in self.numeric_comparisons:
            if num_comp.column_name == column_name:
                return num_comp
        for cat_comp in self.categorical_comparisons:
            if cat_comp.column_name == column_name:
                return cat_comp
        for date_comp in self.date_comparisons:
            if date_comp.column_name == column_name:
                return date_comp
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary.

        Returns:
            Dictionary with comparison summary.
        """
        summary: Dict[str, Any] = {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numeric_columns": len(self.numeric_comparisons),
            "categorical_columns": len(self.categorical_comparisons),
            "date_columns": len(self.date_comparisons),
        }

        if self.overall_numeric_mae is not None:
            summary["overall_numeric_mae"] = self.overall_numeric_mae
        if self.overall_numeric_rmse is not None:
            summary["overall_numeric_rmse"] = self.overall_numeric_rmse
        if self.correlation_preservation is not None:
            summary["correlation_preservation_rate"] = (
                self.correlation_preservation.preservation_rate
            )

        return summary


# =============================================================================
# Column Comparator
# =============================================================================


class ColumnComparator:
    """Compare individual columns between original and protected data.

    This class provides methods to compare numeric, categorical, and date
    columns, calculating appropriate divergence metrics for each type.

    Example:
        >>> comparator = ColumnComparator()
        >>> result = comparator.compare_numeric(
        ...     original=df_orig["age"],
        ...     protected=df_prot["age"],
        ...     column_name="age",
        ... )
        >>> print(f"MAE: {result.divergence.mae}")
    """

    def __init__(self, percentiles: Optional[List[int]] = None) -> None:
        """Initialize the column comparator.

        Args:
            percentiles: Percentiles to compare (default: [5, 25, 50, 75, 95]).
        """
        self._percentiles = percentiles or [5, 25, 50, 75, 95]

    # -------------------------------------------------------------------------
    # Numeric Comparison
    # -------------------------------------------------------------------------

    def compare_numeric(
        self,
        original: pd.Series,
        protected: pd.Series,
        column_name: str,
    ) -> NumericComparison:
        """Compare a numeric column.

        Args:
            original: Original column data.
            protected: Protected column data.
            column_name: Name of the column.

        Returns:
            NumericComparison with divergence metrics.

        Raises:
            ValueError: If series have different lengths.
        """
        if len(original) != len(protected):
            raise ValueError(
                f"Series must have same length: {len(original)} vs {len(protected)}"
            )

        # Handle nulls - compare only where both are non-null
        mask = original.notna() & protected.notna()
        orig_valid = original[mask].astype(float)
        prot_valid = protected[mask].astype(float)

        count = len(orig_valid)
        null_count = len(original) - count

        if count == 0:
            # All nulls - return empty comparison
            return self._empty_numeric_comparison(column_name, null_count)

        # Calculate divergence metrics
        divergence = self._calculate_numeric_divergence(orig_valid, prot_valid)

        # Calculate percentile comparison
        percentile_comparison = {}
        for p in self._percentiles:
            orig_p = float(np.percentile(orig_valid, p))
            prot_p = float(np.percentile(prot_valid, p))
            diff = prot_p - orig_p
            percentile_comparison[f"p{p}"] = (orig_p, prot_p, diff)

        return NumericComparison(
            column_name=column_name,
            comparison_type=ComparisonType.NUMERIC,
            count=count,
            null_count=null_count,
            divergence=divergence,
            percentile_comparison=percentile_comparison,
        )

    def _calculate_numeric_divergence(
        self,
        original: pd.Series,
        protected: pd.Series,
    ) -> NumericDivergence:
        """Calculate numeric divergence metrics.

        Args:
            original: Original values (no nulls).
            protected: Protected values (no nulls).

        Returns:
            NumericDivergence with all metrics.
        """
        diff = protected - original
        abs_diff = np.abs(diff)

        # MAE, RMSE
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs_error = float(np.max(abs_diff))

        # MAPE (only if no zeros in original)
        mape: Optional[float] = None
        if not (original == 0).any():
            mape = float(np.mean(np.abs(diff / original)) * 100)

        # Mean comparison
        mean_orig = float(np.mean(original))
        mean_prot = float(np.mean(protected))
        mean_diff = mean_prot - mean_orig
        mean_rel_error: Optional[float] = None
        if mean_orig != 0:
            mean_rel_error = abs(mean_diff / mean_orig) * 100

        # Std comparison
        std_orig = float(np.std(original, ddof=1)) if len(original) > 1 else 0.0
        std_prot = float(np.std(protected, ddof=1)) if len(protected) > 1 else 0.0
        std_diff = std_prot - std_orig
        std_rel_error: Optional[float] = None
        if std_orig != 0:
            std_rel_error = abs(std_diff / std_orig) * 100

        return NumericDivergence(
            mae=mae,
            rmse=rmse,
            mape=mape,
            max_absolute_error=max_abs_error,
            mean_original=mean_orig,
            mean_protected=mean_prot,
            mean_difference=mean_diff,
            mean_relative_error=mean_rel_error,
            std_original=std_orig,
            std_protected=std_prot,
            std_difference=std_diff,
            std_relative_error=std_rel_error,
            median_original=float(np.median(original)),
            median_protected=float(np.median(protected)),
            min_original=float(np.min(original)),
            min_protected=float(np.min(protected)),
            max_original=float(np.max(original)),
            max_protected=float(np.max(protected)),
        )

    def _empty_numeric_comparison(
        self, column_name: str, null_count: int
    ) -> NumericComparison:
        """Create an empty numeric comparison for all-null columns."""
        return NumericComparison(
            column_name=column_name,
            comparison_type=ComparisonType.NUMERIC,
            count=0,
            null_count=null_count,
            divergence=NumericDivergence(
                mae=0.0,
                rmse=0.0,
                mape=None,
                max_absolute_error=0.0,
                mean_original=0.0,
                mean_protected=0.0,
                mean_difference=0.0,
                mean_relative_error=None,
                std_original=0.0,
                std_protected=0.0,
                std_difference=0.0,
                std_relative_error=None,
                median_original=0.0,
                median_protected=0.0,
                min_original=0.0,
                min_protected=0.0,
                max_original=0.0,
                max_protected=0.0,
            ),
            percentile_comparison={},
        )

    # -------------------------------------------------------------------------
    # Categorical Comparison
    # -------------------------------------------------------------------------

    def compare_categorical(
        self,
        original: pd.Series,
        protected: pd.Series,
        column_name: str,
        top_n: int = 20,
    ) -> CategoricalComparison:
        """Compare a categorical column.

        Args:
            original: Original column data.
            protected: Protected column data.
            column_name: Name of the column.
            top_n: Number of top categories to include in frequency comparison.

        Returns:
            CategoricalComparison with divergence metrics.

        Raises:
            ValueError: If series have different lengths.
        """
        if len(original) != len(protected):
            raise ValueError(
                f"Series must have same length: {len(original)} vs {len(protected)}"
            )

        # Handle nulls
        mask = original.notna() & protected.notna()
        orig_valid = original[mask].astype(str)
        prot_valid = protected[mask].astype(str)

        count = len(orig_valid)
        null_count = len(original) - count

        if count == 0:
            return self._empty_categorical_comparison(column_name, null_count)

        # Calculate divergence metrics
        divergence = self._calculate_categorical_divergence(orig_valid, prot_valid)

        # Frequency comparison for top categories
        orig_counts = orig_valid.value_counts(normalize=True)
        prot_counts = prot_valid.value_counts(normalize=True)

        # Get union of top categories from both
        top_categories = set(orig_counts.head(top_n).index) | set(
            prot_counts.head(top_n).index
        )

        frequency_comparison = {}
        for cat in top_categories:
            orig_freq = float(orig_counts.get(cat, 0.0))
            prot_freq = float(prot_counts.get(cat, 0.0))
            frequency_comparison[cat] = (orig_freq, prot_freq)

        return CategoricalComparison(
            column_name=column_name,
            comparison_type=ComparisonType.CATEGORICAL,
            count=count,
            null_count=null_count,
            divergence=divergence,
            frequency_comparison=frequency_comparison,
        )

    def _calculate_categorical_divergence(
        self,
        original: pd.Series,
        protected: pd.Series,
    ) -> CategoricalDivergence:
        """Calculate categorical divergence metrics.

        Args:
            original: Original values (no nulls, as strings).
            protected: Protected values (no nulls, as strings).

        Returns:
            CategoricalDivergence with all metrics.
        """
        # Category drift - proportion of values that changed
        changed = (original.values != protected.values).sum()
        category_drift = float(changed / len(original))

        # Cardinality
        orig_unique = set(original.unique())
        prot_unique = set(protected.unique())
        cardinality_orig = len(orig_unique)
        cardinality_prot = len(prot_unique)

        # Mode comparison
        mode_orig = original.mode().iloc[0] if len(original.mode()) > 0 else None
        mode_prot = protected.mode().iloc[0] if len(protected.mode()) > 0 else None
        mode_preserved = mode_orig == mode_prot

        # Frequency MAE
        orig_counts = original.value_counts(normalize=True)
        prot_counts = protected.value_counts(normalize=True)

        all_categories = orig_unique | prot_unique
        freq_diffs = []
        for cat in all_categories:
            orig_freq = orig_counts.get(cat, 0.0)
            prot_freq = prot_counts.get(cat, 0.0)
            freq_diffs.append(abs(prot_freq - orig_freq))

        frequency_mae = float(np.mean(freq_diffs)) if freq_diffs else 0.0

        # New and missing categories
        new_categories = list(prot_unique - orig_unique)
        missing_categories = list(orig_unique - prot_unique)

        return CategoricalDivergence(
            category_drift=category_drift,
            cardinality_original=cardinality_orig,
            cardinality_protected=cardinality_prot,
            cardinality_difference=cardinality_prot - cardinality_orig,
            mode_original=mode_orig,
            mode_protected=mode_prot,
            mode_preserved=mode_preserved,
            frequency_mae=frequency_mae,
            new_categories=new_categories,
            missing_categories=missing_categories,
        )

    def _empty_categorical_comparison(
        self, column_name: str, null_count: int
    ) -> CategoricalComparison:
        """Create an empty categorical comparison for all-null columns."""
        return CategoricalComparison(
            column_name=column_name,
            comparison_type=ComparisonType.CATEGORICAL,
            count=0,
            null_count=null_count,
            divergence=CategoricalDivergence(
                category_drift=0.0,
                cardinality_original=0,
                cardinality_protected=0,
                cardinality_difference=0,
                mode_original=None,
                mode_protected=None,
                mode_preserved=True,
                frequency_mae=0.0,
                new_categories=[],
                missing_categories=[],
            ),
            frequency_comparison={},
        )

    # -------------------------------------------------------------------------
    # Date Comparison
    # -------------------------------------------------------------------------

    def compare_date(
        self,
        original: pd.Series,
        protected: pd.Series,
        column_name: str,
    ) -> DateComparison:
        """Compare a date column.

        Args:
            original: Original column data.
            protected: Protected column data.
            column_name: Name of the column.

        Returns:
            DateComparison with divergence metrics.

        Raises:
            ValueError: If series have different lengths.
        """
        if len(original) != len(protected):
            raise ValueError(
                f"Series must have same length: {len(original)} vs {len(protected)}"
            )

        # Convert to datetime if needed
        orig_dt = pd.to_datetime(original, errors="coerce")
        prot_dt = pd.to_datetime(protected, errors="coerce")

        # Handle nulls
        mask = orig_dt.notna() & prot_dt.notna()
        orig_valid = orig_dt[mask]
        prot_valid = prot_dt[mask]

        count = len(orig_valid)
        null_count = len(original) - count

        if count == 0:
            return self._empty_date_comparison(column_name, null_count)

        # Calculate divergence metrics
        divergence = self._calculate_date_divergence(orig_valid, prot_valid)

        return DateComparison(
            column_name=column_name,
            comparison_type=ComparisonType.DATE,
            count=count,
            null_count=null_count,
            divergence=divergence,
        )

    def _calculate_date_divergence(
        self,
        original: pd.Series,
        protected: pd.Series,
    ) -> DateDivergence:
        """Calculate date divergence metrics.

        Args:
            original: Original datetime values (no nulls).
            protected: Protected datetime values (no nulls).

        Returns:
            DateDivergence with all metrics.
        """
        # Convert to days since epoch for comparison
        orig_days = (original - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        prot_days = (protected - pd.Timestamp("1970-01-01")).dt.days.astype(float)

        diff_days = prot_days - orig_days
        abs_diff = np.abs(diff_days)

        # MAE, RMSE in days
        mae_days = float(np.mean(abs_diff))
        rmse_days = float(np.sqrt(np.mean(diff_days**2)))
        max_abs_error = float(np.max(abs_diff))

        # Range comparison
        range_orig = int((original.max() - original.min()).days)
        range_prot = int((protected.max() - protected.min()).days)

        # Min/max differences
        min_diff = float((protected.min() - original.min()).days)
        max_diff = float((protected.max() - original.max()).days)

        return DateDivergence(
            mae_days=mae_days,
            rmse_days=rmse_days,
            max_absolute_error_days=max_abs_error,
            range_original_days=range_orig,
            range_protected_days=range_prot,
            range_difference_days=range_prot - range_orig,
            min_date_diff_days=min_diff,
            max_date_diff_days=max_diff,
        )

    def _empty_date_comparison(
        self, column_name: str, null_count: int
    ) -> DateComparison:
        """Create an empty date comparison for all-null columns."""
        return DateComparison(
            column_name=column_name,
            comparison_type=ComparisonType.DATE,
            count=0,
            null_count=null_count,
            divergence=DateDivergence(
                mae_days=0.0,
                rmse_days=0.0,
                max_absolute_error_days=0.0,
                range_original_days=0,
                range_protected_days=0,
                range_difference_days=0,
                min_date_diff_days=0.0,
                max_date_diff_days=0.0,
            ),
        )


# =============================================================================
# Dataset Comparator
# =============================================================================


class DatasetComparator:
    """Compare entire datasets between original and protected versions.

    This class orchestrates column-level comparisons and provides
    dataset-level analysis including correlation preservation.

    Example:
        >>> comparator = DatasetComparator()
        >>> result = comparator.compare(
        ...     original=df_original,
        ...     protected=df_protected,
        ...     numeric_columns=["age", "income"],
        ...     categorical_columns=["gender"],
        ... )
        >>> print(f"Overall MAE: {result.overall_numeric_mae}")
    """

    def __init__(
        self,
        correlation_threshold: float = 0.1,
        correlation_method: str = "pearson",
    ) -> None:
        """Initialize the dataset comparator.

        Args:
            correlation_threshold: Threshold for considering correlation preserved.
            correlation_method: Method for correlation calculation.
        """
        self._column_comparator = ColumnComparator()
        self._correlation_threshold = correlation_threshold
        self._correlation_method = correlation_method

    def compare(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        compare_correlations: bool = True,
    ) -> DatasetComparison:
        """Compare original and protected datasets.

        Args:
            original: Original DataFrame.
            protected: Protected DataFrame.
            numeric_columns: List of numeric columns to compare.
            categorical_columns: List of categorical columns to compare.
            date_columns: List of date columns to compare.
            compare_correlations: Whether to analyze correlation preservation.

        Returns:
            DatasetComparison with all comparison results.

        Raises:
            ValueError: If DataFrames have different shapes or columns.
        """
        # Validate inputs
        if original.shape != protected.shape:
            raise ValueError(
                f"DataFrames must have same shape: "
                f"{original.shape} vs {protected.shape}"
            )

        if set(original.columns) != set(protected.columns):
            raise ValueError("DataFrames must have the same columns")

        # Auto-detect column types if not specified
        if numeric_columns is None:
            numeric_columns = list(original.select_dtypes(include=["number"]).columns)
        if categorical_columns is None:
            categorical_columns = list(
                original.select_dtypes(include=["object", "category"]).columns
            )
        if date_columns is None:
            date_columns = list(
                original.select_dtypes(include=["datetime64"]).columns
            )

        # Compare numeric columns
        numeric_comparisons: List[NumericComparison] = []
        for col in numeric_columns:
            if col in original.columns:
                num_comp = self._column_comparator.compare_numeric(
                    original=original[col],
                    protected=protected[col],
                    column_name=col,
                )
                numeric_comparisons.append(num_comp)

        # Compare categorical columns
        categorical_comparisons: List[CategoricalComparison] = []
        for col in categorical_columns:
            if col in original.columns:
                cat_comp = self._column_comparator.compare_categorical(
                    original=original[col],
                    protected=protected[col],
                    column_name=col,
                )
                categorical_comparisons.append(cat_comp)

        # Compare date columns
        date_comparisons: List[DateComparison] = []
        for col in date_columns:
            if col in original.columns:
                date_comp = self._column_comparator.compare_date(
                    original=original[col],
                    protected=protected[col],
                    column_name=col,
                )
                date_comparisons.append(date_comp)

        # Calculate overall numeric metrics
        overall_mae: Optional[float] = None
        overall_rmse: Optional[float] = None
        if numeric_comparisons:
            maes = [c.divergence.mae for c in numeric_comparisons if c.count > 0]
            rmses = [c.divergence.rmse for c in numeric_comparisons if c.count > 0]
            if maes:
                overall_mae = float(np.mean(maes))
            if rmses:
                overall_rmse = float(np.mean(rmses))

        # Correlation preservation analysis
        correlation_preservation: Optional[CorrelationPreservation] = None
        if compare_correlations and len(numeric_columns) >= 2:
            correlation_preservation = self._analyze_correlation_preservation(
                original=original,
                protected=protected,
                numeric_columns=numeric_columns,
            )

        return DatasetComparison(
            row_count=len(original),
            column_count=len(original.columns),
            numeric_comparisons=numeric_comparisons,
            categorical_comparisons=categorical_comparisons,
            date_comparisons=date_comparisons,
            correlation_preservation=correlation_preservation,
            overall_numeric_mae=overall_mae,
            overall_numeric_rmse=overall_rmse,
        )

    def _analyze_correlation_preservation(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        numeric_columns: List[str],
    ) -> CorrelationPreservation:
        """Analyze how well correlations are preserved.

        Args:
            original: Original DataFrame.
            protected: Protected DataFrame.
            numeric_columns: Numeric columns to analyze.

        Returns:
            CorrelationPreservation metrics.
        """
        # Calculate correlation matrices
        orig_corr = original[numeric_columns].corr(method=self._correlation_method)
        prot_corr = protected[numeric_columns].corr(method=self._correlation_method)

        # Extract correlation pairs (upper triangle only)
        original_correlations: Dict[Tuple[str, str], float] = {}
        protected_correlations: Dict[Tuple[str, str], float] = {}
        correlation_differences: Dict[Tuple[str, str], float] = {}

        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:  # Upper triangle only
                    pair = (col1, col2)
                    orig_val = orig_corr.loc[col1, col2]
                    prot_val = prot_corr.loc[col1, col2]

                    # Handle NaN correlations
                    if pd.isna(orig_val) or pd.isna(prot_val):
                        continue

                    original_correlations[pair] = float(orig_val)
                    protected_correlations[pair] = float(prot_val)
                    correlation_differences[pair] = float(abs(prot_val - orig_val))

        if not correlation_differences:
            return CorrelationPreservation(
                original_correlations={},
                protected_correlations={},
                correlation_differences={},
                mae=0.0,
                rmse=0.0,
                max_absolute_error=0.0,
                preserved_count=0,
                total_count=0,
                preservation_rate=1.0,
            )

        # Calculate metrics
        diffs = list(correlation_differences.values())
        mae = float(np.mean(diffs))
        rmse = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
        max_error = float(np.max(diffs))

        # Count preserved correlations
        preserved = sum(1 for d in diffs if d <= self._correlation_threshold)
        total = len(diffs)
        preservation_rate = preserved / total if total > 0 else 1.0

        return CorrelationPreservation(
            original_correlations=original_correlations,
            protected_correlations=protected_correlations,
            correlation_differences=correlation_differences,
            mae=mae,
            rmse=rmse,
            max_absolute_error=max_error,
            preserved_count=preserved,
            total_count=total,
            preservation_rate=preservation_rate,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_numeric_column(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
) -> NumericComparison:
    """Compare a numeric column (convenience function).

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name of the column (uses original.name if None).

    Returns:
        NumericComparison with divergence metrics.
    """
    name = column_name or original.name or "unknown"
    comparator = ColumnComparator()
    return comparator.compare_numeric(original, protected, str(name))


def compare_categorical_column(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
) -> CategoricalComparison:
    """Compare a categorical column (convenience function).

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name of the column (uses original.name if None).

    Returns:
        CategoricalComparison with divergence metrics.
    """
    name = column_name or original.name or "unknown"
    comparator = ColumnComparator()
    return comparator.compare_categorical(original, protected, str(name))


def compare_date_column(
    original: pd.Series,
    protected: pd.Series,
    column_name: Optional[str] = None,
) -> DateComparison:
    """Compare a date column (convenience function).

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name of the column (uses original.name if None).

    Returns:
        DateComparison with divergence metrics.
    """
    name = column_name or original.name or "unknown"
    comparator = ColumnComparator()
    return comparator.compare_date(original, protected, str(name))


def compare_datasets(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    date_columns: Optional[List[str]] = None,
) -> DatasetComparison:
    """Compare original and protected datasets (convenience function).

    Args:
        original: Original DataFrame.
        protected: Protected DataFrame.
        numeric_columns: List of numeric columns to compare.
        categorical_columns: List of categorical columns to compare.
        date_columns: List of date columns to compare.

    Returns:
        DatasetComparison with all comparison results.
    """
    comparator = DatasetComparator()
    return comparator.compare(
        original=original,
        protected=protected,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_columns=date_columns,
    )


def calculate_mae(original: pd.Series, protected: pd.Series) -> float:
    """Calculate Mean Absolute Error between two series.

    Args:
        original: Original values.
        protected: Protected values.

    Returns:
        MAE value.
    """
    mask = original.notna() & protected.notna()
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(protected[mask] - original[mask])))


def calculate_rmse(original: pd.Series, protected: pd.Series) -> float:
    """Calculate Root Mean Square Error between two series.

    Args:
        original: Original values.
        protected: Protected values.

    Returns:
        RMSE value.
    """
    mask = original.notna() & protected.notna()
    if not mask.any():
        return 0.0
    diff = protected[mask] - original[mask]
    return float(np.sqrt(np.mean(diff**2)))


def calculate_mape(original: pd.Series, protected: pd.Series) -> Optional[float]:
    """Calculate Mean Absolute Percentage Error between two series.

    Args:
        original: Original values.
        protected: Protected values.

    Returns:
        MAPE value as percentage, or None if zeros present in original.
    """
    mask = original.notna() & protected.notna()
    if not mask.any():
        return None
    orig = original[mask]
    prot = protected[mask]
    if (orig == 0).any():
        return None
    return float(np.mean(np.abs((prot - orig) / orig)) * 100)
