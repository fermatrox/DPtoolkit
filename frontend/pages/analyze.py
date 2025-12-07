"""Analyze page for DPtoolkit.

This page displays statistical comparison and visualizations between
original and protected datasets after applying differential privacy.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import streamlit as st

from dp_toolkit.data.transformer import (
    DatasetTransformer,
    DatasetConfig,
    DatasetColumnConfig,
    ProtectionMode,
)
from dp_toolkit.analysis.comparator import (
    DatasetComparator,
    DatasetComparison,
    NumericComparison,
    CategoricalComparison,
)
from dp_toolkit.analysis.visualizer import (
    create_histogram_overlay,
    create_correlation_comparison,
    create_category_bar_chart,
    create_box_comparison,
)


# =============================================================================
# Session State Helpers
# =============================================================================


def get_comparison_results() -> Optional[DatasetComparison]:
    """Get stored comparison results."""
    return st.session_state.get("comparison_results")


def set_comparison_results(results: DatasetComparison) -> None:
    """Store comparison results."""
    st.session_state.comparison_results = results


def get_protected_df() -> Optional[pd.DataFrame]:
    """Get the protected DataFrame."""
    return st.session_state.get("protected_df")


def set_protected_df(df: pd.DataFrame) -> None:
    """Store the protected DataFrame."""
    st.session_state.protected_df = df


# =============================================================================
# Transformation Functions
# =============================================================================


def build_dataset_config(column_configs: Dict[str, Dict[str, Any]]) -> DatasetConfig:
    """Build DatasetConfig from frontend column configurations."""
    config = DatasetConfig(global_epsilon=1.0)

    for col_name, col_cfg in column_configs.items():
        mode_str = col_cfg.get("mode", "protect")
        epsilon = col_cfg.get("epsilon", 1.0)
        mechanism_str = col_cfg.get("mechanism", "laplace")

        # Map mode
        if mode_str == "protect":
            mode = ProtectionMode.PROTECT
        elif mode_str == "passthrough":
            mode = ProtectionMode.PASSTHROUGH
        else:
            mode = ProtectionMode.EXCLUDE

        # Note: mechanism is auto-selected by transformer based on column type
        # Gaussian mechanism requires delta parameter
        delta = 1e-5 if mechanism_str == "gaussian" else None

        col_config = DatasetColumnConfig(
            mode=mode,
            epsilon=epsilon,
            delta=delta,
        )
        config.column_configs[col_name] = col_config

    return config


def apply_dp_transformation(
    df: pd.DataFrame,
    column_configs: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Apply differential privacy transformation to the dataset."""
    config = build_dataset_config(column_configs)
    transformer = DatasetTransformer()
    result = transformer.transform(df, config)
    return result.data


def run_comparison(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    column_configs: Dict[str, Dict[str, Any]],
) -> DatasetComparison:
    """Run comparison between original and protected datasets."""
    # Identify column types for comparison
    numeric_cols = []
    categorical_cols = []

    for col in protected_df.columns:
        cfg = column_configs.get(col, {})
        if cfg.get("mode") != "protect":
            continue

        col_type = cfg.get("data_type", "unknown")
        if col_type in ["numeric", "numeric_bounded", "numeric_unbounded"]:
            numeric_cols.append(col)
        elif col_type in ["categorical", "unknown"]:
            # Check if actually numeric
            if pd.api.types.is_numeric_dtype(original_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    comparator = DatasetComparator()
    return comparator.compare(
        original=original_df,
        protected=protected_df,
        numeric_columns=numeric_cols if numeric_cols else None,
        categorical_columns=categorical_cols if categorical_cols else None,
    )


# =============================================================================
# Summary Tab
# =============================================================================


def render_summary_tab(
    comparison: DatasetComparison,
    column_configs: Dict[str, Dict[str, Any]],
) -> None:
    """Render the summary tab."""
    st.markdown("### Overall Statistics")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{comparison.row_count:,}")

    with col2:
        st.metric("Columns Analyzed", comparison.column_count)

    with col3:
        # Calculate total epsilon
        total_eps = sum(
            cfg.get("epsilon", 1.0)
            for cfg in column_configs.values()
            if cfg.get("mode") == "protect"
        )
        st.metric("Total ε", f"{total_eps:.2f}")

    with col4:
        if comparison.correlation_preservation:
            rate = comparison.correlation_preservation.preservation_rate * 100
            st.metric("Correlation Preserved", f"{rate:.1f}%")
        else:
            st.metric("Correlation Preserved", "N/A")

    st.markdown("---")

    # Column type breakdown
    st.markdown("### Column Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Numeric Columns", len(comparison.numeric_comparisons))

    with col2:
        st.metric("Categorical Columns", len(comparison.categorical_comparisons))

    with col3:
        st.metric("Date Columns", len(comparison.date_comparisons))

    st.markdown("---")

    # Numeric accuracy metrics
    if comparison.numeric_comparisons:
        st.markdown("### Numeric Column Accuracy")

        col1, col2 = st.columns(2)

        with col1:
            if comparison.overall_numeric_mae is not None:
                st.metric(
                    "Average MAE",
                    f"{comparison.overall_numeric_mae:.4f}",
                    help="Mean Absolute Error across all numeric columns",
                )

        with col2:
            if comparison.overall_numeric_rmse is not None:
                st.metric(
                    "Average RMSE",
                    f"{comparison.overall_numeric_rmse:.4f}",
                    help="Root Mean Square Error across all numeric columns",
                )

        # Per-column summary table
        st.markdown("#### Per-Column Summary")

        summary_data = []
        for num_comp in comparison.numeric_comparisons:
            summary_data.append({
                "Column": num_comp.column_name,
                "MAE": f"{num_comp.divergence.mae:.4f}",
                "RMSE": f"{num_comp.divergence.rmse:.4f}",
                "Mean Diff": f"{num_comp.divergence.mean_difference:.4f}",
                "Std Diff": f"{num_comp.divergence.std_difference:.4f}",
            })

        if summary_data:
            st.dataframe(
                pd.DataFrame(summary_data),
                hide_index=True,
                width="stretch",
            )

    # Categorical accuracy
    if comparison.categorical_comparisons:
        st.markdown("### Categorical Column Accuracy")

        cat_summary = []
        for cat_comp in comparison.categorical_comparisons:
            cat_summary.append({
                "Column": cat_comp.column_name,
                "Categories": cat_comp.divergence.cardinality_original,
                "Freq MAE": f"{cat_comp.divergence.frequency_mae:.4f}",
                "Category Drift": f"{cat_comp.divergence.category_drift:.4f}",
            })

        if cat_summary:
            st.dataframe(
                pd.DataFrame(cat_summary),
                hide_index=True,
                width="stretch",
            )


# =============================================================================
# Per-Column Tab
# =============================================================================


def render_per_column_tab(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    comparison: DatasetComparison,
) -> None:
    """Render the per-column analysis tab."""
    st.markdown("### Per-Column Analysis")

    # Get list of all compared columns
    CompType = Union[NumericComparison, CategoricalComparison]
    all_columns: List[Tuple[str, str, CompType]] = []
    for num_comp in comparison.numeric_comparisons:
        all_columns.append((num_comp.column_name, "numeric", num_comp))
    for cat_comp in comparison.categorical_comparisons:
        all_columns.append((cat_comp.column_name, "categorical", cat_comp))

    if not all_columns:
        st.info("No columns to analyze. Make sure some columns are set to 'Protect'.")
        return

    # Column selector
    column_names = [c[0] for c in all_columns]
    selected_col: Optional[str] = st.selectbox(
        "Select Column",
        options=column_names,
        key="analyze_column_select",
    )

    if selected_col is None:
        return

    # Find the comparison for selected column
    selected_comp = None
    col_type = None
    for name, ctype, comp in all_columns:
        if name == selected_col:
            selected_comp = comp
            col_type = ctype
            break

    if selected_comp is None:
        st.error("Column comparison not found")
        return

    st.markdown("---")

    # Display comparison based on type
    if col_type == "numeric":
        render_numeric_comparison(
            original_df[selected_col],
            protected_df[selected_col],
            cast(NumericComparison, selected_comp),
        )
    elif col_type == "categorical":
        render_categorical_comparison(
            original_df[selected_col],
            protected_df[selected_col],
            cast(CategoricalComparison, selected_comp),
        )


def render_numeric_comparison(
    original: pd.Series,
    protected: pd.Series,
    comparison: NumericComparison,
) -> None:
    """Render comparison for a numeric column."""
    st.markdown(f"#### {comparison.column_name} (Numeric)")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MAE", f"{comparison.divergence.mae:.4f}")

    with col2:
        st.metric("RMSE", f"{comparison.divergence.rmse:.4f}")

    with col3:
        st.metric(
            "Mean",
            f"{comparison.divergence.mean_protected:.2f}",
            delta=f"{comparison.divergence.mean_difference:.2f}",
        )

    with col4:
        st.metric(
            "Std Dev",
            f"{comparison.divergence.std_protected:.2f}",
            delta=f"{comparison.divergence.std_difference:.2f}",
        )

    st.markdown("---")

    # Histogram overlay
    st.markdown("##### Distribution Comparison")

    try:
        fig = create_histogram_overlay(
            original=original,
            protected=protected,
            column_name=comparison.column_name,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"Could not create histogram: {e}")

    # Box plot comparison
    st.markdown("##### Box Plot Comparison")

    try:
        fig = create_box_comparison(
            original=original,
            protected=protected,
            column_name=comparison.column_name,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"Could not create box plot: {e}")

    # Detailed statistics table
    st.markdown("##### Detailed Statistics")

    stats_data = {
        "Statistic": ["Mean", "Std Dev", "Median", "Min", "Max"],
        "Original": [
            f"{comparison.divergence.mean_original:.4f}",
            f"{comparison.divergence.std_original:.4f}",
            f"{comparison.divergence.median_original:.4f}",
            f"{comparison.divergence.min_original:.4f}",
            f"{comparison.divergence.max_original:.4f}",
        ],
        "Protected": [
            f"{comparison.divergence.mean_protected:.4f}",
            f"{comparison.divergence.std_protected:.4f}",
            f"{comparison.divergence.median_protected:.4f}",
            f"{comparison.divergence.min_protected:.4f}",
            f"{comparison.divergence.max_protected:.4f}",
        ],
        "Difference": [
            f"{comparison.divergence.mean_difference:.4f}",
            f"{comparison.divergence.std_difference:.4f}",
            f"{comparison.divergence.median_protected - comparison.divergence.median_original:.4f}",  # noqa: E501
            f"{comparison.divergence.min_protected - comparison.divergence.min_original:.4f}",  # noqa: E501
            f"{comparison.divergence.max_protected - comparison.divergence.max_original:.4f}",  # noqa: E501
        ],
    }

    st.dataframe(pd.DataFrame(stats_data), hide_index=True)


def render_categorical_comparison(
    original: pd.Series,
    protected: pd.Series,
    comparison: CategoricalComparison,
) -> None:
    """Render comparison for a categorical column."""
    st.markdown(f"#### {comparison.column_name} (Categorical)")

    # Metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Categories", comparison.divergence.cardinality_original)

    with col2:
        st.metric("Frequency MAE", f"{comparison.divergence.frequency_mae:.4f}")

    with col3:
        cat_drift = comparison.divergence.category_drift
        st.metric("Category Drift", f"{cat_drift:.4f}")

    st.markdown("---")

    # Category frequency chart
    st.markdown("##### Category Distribution")

    try:
        fig = create_category_bar_chart(
            original=original,
            protected=protected,
            column_name=comparison.column_name,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"Could not create category chart: {e}")

    # Category frequency table
    st.markdown("##### Category Frequency Comparison")

    freq_data = []
    for cat, (orig_freq, prot_freq) in comparison.frequency_comparison.items():
        diff = prot_freq - orig_freq
        freq_data.append({
            "Category": str(cat),
            "Original %": f"{orig_freq * 100:.2f}%",
            "Protected %": f"{prot_freq * 100:.2f}%",
            "Difference": f"{diff * 100:+.2f}%",
        })

    if freq_data:
        st.dataframe(
            pd.DataFrame(freq_data),
            hide_index=True,
        )


# =============================================================================
# Correlations Tab
# =============================================================================


def render_correlations_tab(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    comparison: DatasetComparison,
) -> None:
    """Render the correlations analysis tab."""
    st.markdown("### Correlation Analysis")

    if comparison.correlation_preservation is None:
        st.info(
            "Correlation analysis requires at least 2 numeric columns. "
            "Make sure multiple numeric columns are set to 'Protect'."
        )
        return

    corr_pres = comparison.correlation_preservation

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Correlations Analyzed", corr_pres.total_count)

    with col2:
        st.metric("Preserved", corr_pres.preserved_count)

    with col3:
        st.metric("Preservation Rate", f"{corr_pres.preservation_rate * 100:.1f}%")

    with col4:
        st.metric("Max Difference", f"{corr_pres.max_absolute_error:.4f}")

    st.markdown("---")

    # Correlation heatmap comparison
    st.markdown("##### Correlation Heatmaps")

    # Get numeric columns
    numeric_cols = [c.column_name for c in comparison.numeric_comparisons]

    if len(numeric_cols) >= 2:
        try:
            fig = create_correlation_comparison(
                original=original_df[numeric_cols],
                protected=protected_df[numeric_cols],
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not create correlation heatmap: {e}")

    st.markdown("---")

    # Correlation difference table
    st.markdown("##### Correlation Differences")

    corr_data = []
    for (col1_name, col2_name), diff in corr_pres.correlation_differences.items():
        orig_corr = corr_pres.original_correlations.get((col1_name, col2_name), 0)
        prot_corr = corr_pres.protected_correlations.get((col1_name, col2_name), 0)
        corr_data.append({
            "Column 1": col1_name,
            "Column 2": col2_name,
            "Original": f"{orig_corr:.4f}",
            "Protected": f"{prot_corr:.4f}",
            "Difference": f"{diff:.4f}",
            "Preserved": "Yes" if abs(diff) <= 0.1 else "No",
        })

    if corr_data:
        st.dataframe(
            pd.DataFrame(corr_data),
            hide_index=True,
        )

    # Accuracy metrics
    st.markdown("---")
    st.markdown("##### Correlation Accuracy Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE",
            f"{corr_pres.mae:.4f}",
            help="Mean Absolute Error of correlations",
        )

    with col2:
        st.metric(
            "RMSE",
            f"{corr_pres.rmse:.4f}",
            help="Root Mean Square Error of correlations",
        )


# =============================================================================
# Navigation
# =============================================================================


def render_navigation_buttons() -> None:
    """Render navigation buttons."""
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back to Configure", width="stretch"):
            st.session_state.current_step = 2
            st.rerun()

    with col3:
        if st.button(
            "Continue to Export →",
            type="primary",
            width="stretch",
        ):
            st.session_state.transform_complete = True
            st.session_state.current_step = 4
            st.rerun()


# =============================================================================
# Main Page Function
# =============================================================================


def render_analyze_page() -> None:
    """Render the complete analyze page."""
    st.header("Step 3: Analyze Results")

    # Check prerequisites
    if not st.session_state.get("config_complete", False):
        st.warning("Please configure privacy settings first.")
        if st.button("Go to Configure"):
            st.session_state.current_step = 2
            st.rerun()
        return

    # Get data
    original_df = st.session_state.get("original_df")
    column_configs = st.session_state.get("column_configs", {})

    if original_df is None:
        st.error("Original dataset not found. Please re-upload your dataset.")
        return

    st.markdown(
        "Compare original and protected datasets to understand the impact of "
        "differential privacy on your data."
    )

    st.markdown("---")

    # Check if we need to run transformation
    protected_df = get_protected_df()
    comparison = get_comparison_results()

    if protected_df is None or comparison is None:
        st.info(
            "Click 'Run Analysis' to apply differential privacy and compare."
        )

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Applying differential privacy..."):
                try:
                    protected_df = apply_dp_transformation(original_df, column_configs)
                    set_protected_df(protected_df)
                except Exception as e:
                    st.error(f"Error applying DP: {e}")
                    return

            with st.spinner("Comparing datasets..."):
                try:
                    comparison = run_comparison(
                        original_df, protected_df, column_configs
                    )
                    set_comparison_results(comparison)
                except Exception as e:
                    st.error(f"Error comparing datasets: {e}")
                    return

            st.success("Analysis complete!")
            st.rerun()

        st.markdown("---")
        render_navigation_buttons()
        return

    # Show re-run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Re-run Analysis", width="stretch"):
            # Clear cached results
            st.session_state.protected_df = None
            st.session_state.comparison_results = None
            st.rerun()

    st.markdown("---")

    # Render tabs
    tab1, tab2, tab3 = st.tabs(["Summary", "Per-Column", "Correlations"])

    with tab1:
        render_summary_tab(comparison, column_configs)

    with tab2:
        render_per_column_tab(original_df, protected_df, comparison)

    with tab3:
        render_correlations_tab(original_df, protected_df, comparison)

    st.markdown("---")

    # Navigation
    render_navigation_buttons()
