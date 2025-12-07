"""Configure page for DPtoolkit.

This page handles column configuration for differential privacy,
including protection modes, epsilon values, and mechanism selection.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import streamlit as st

from dp_toolkit.recommendations.advisor import (
    RecommendationAdvisor,
    DatasetRecommendation,
    RecommendedMechanism,
)
from dp_toolkit.recommendations.classifier import SensitivityLevel

# Import UI utilities
try:
    from utils.ui_components import (
        ErrorMessages,
        HELP_TEXTS,
        help_tooltip,
        info_button,
        metric_with_info,
    )
except ImportError:
    # Fallback if running standalone
    ErrorMessages = None  # type: ignore
    HELP_TEXTS = {}
    help_tooltip = lambda x: ""  # type: ignore # noqa: E731
    info_button = lambda *args, **kwargs: None  # type: ignore # noqa: E731
    metric_with_info = lambda *args, **kwargs: None  # type: ignore # noqa: E731


# =============================================================================
# Enums and Constants
# =============================================================================


class ProtectionMode(Enum):
    """Column protection modes."""

    PROTECT = "protect"
    PASSTHROUGH = "passthrough"
    EXCLUDE = "exclude"


# Display names for protection modes
PROTECTION_MODE_LABELS = {
    ProtectionMode.PROTECT: "Protect (Apply DP)",
    ProtectionMode.PASSTHROUGH: "Passthrough (Keep Original)",
    ProtectionMode.EXCLUDE: "Exclude (Remove)",
}

# Display names for mechanisms
MECHANISM_LABELS = {
    RecommendedMechanism.LAPLACE: "Laplace",
    RecommendedMechanism.GAUSSIAN: "Gaussian",
    RecommendedMechanism.EXPONENTIAL: "Exponential",
}

# Sensitivity level colors
SENSITIVITY_COLORS = {
    "HIGH": "#dc3545",
    "MEDIUM": "#ffc107",
    "LOW": "#28a745",
    "UNKNOWN": "#6c757d",
}


# =============================================================================
# Session State Helpers
# =============================================================================


def get_column_configs() -> Dict[str, Dict[str, Any]]:
    """Get column configurations from session state."""
    if "column_configs" not in st.session_state:
        st.session_state.column_configs = {}
    return cast(Dict[str, Dict[str, Any]], st.session_state.column_configs)


def set_column_config(column: str, config: Dict[str, Any]) -> None:
    """Set configuration for a column."""
    if "column_configs" not in st.session_state:
        st.session_state.column_configs = {}
    st.session_state.column_configs[column] = config


def get_global_epsilon() -> float:
    """Get the global epsilon value."""
    return float(st.session_state.get("global_epsilon", 1.0))


def set_global_epsilon(epsilon: float) -> None:
    """Set the global epsilon value."""
    st.session_state.global_epsilon = epsilon


def get_recommendations() -> Optional[DatasetRecommendation]:
    """Get stored recommendations."""
    return st.session_state.get("recommendations")


def set_recommendations(recommendations: DatasetRecommendation) -> None:
    """Store recommendations in session state."""
    st.session_state.recommendations = recommendations


# =============================================================================
# Recommendation Functions
# =============================================================================


def generate_recommendations(df: pd.DataFrame) -> DatasetRecommendation:
    """Generate recommendations for a dataset."""
    advisor = RecommendationAdvisor()
    return advisor.recommend_for_dataset(df)


def apply_recommendations_to_configs(
    recommendations: DatasetRecommendation,
    columns: List[str],
) -> None:
    """Apply recommendations to column configurations."""
    for col in columns:
        if col in recommendations.column_recommendations:
            rec = recommendations.column_recommendations[col]
            config = {
                "mode": ProtectionMode.PROTECT.value,
                "epsilon": rec.epsilon_recommendation.epsilon,
                "mechanism": rec.mechanism_recommendation.mechanism.value,
                "sensitivity": rec.epsilon_recommendation.sensitivity.value,
                "data_type": rec.mechanism_recommendation.data_type.value,
                "recommended": True,
            }
            set_column_config(col, config)


def initialize_column_configs(columns: List[str], column_types: Dict[str, str]) -> None:
    """Initialize column configs with defaults if not already set."""
    configs = get_column_configs()

    for col in columns:
        if col not in configs:
            # Default configuration
            configs[col] = {
                "mode": ProtectionMode.PROTECT.value,
                "epsilon": 1.0,
                "mechanism": RecommendedMechanism.LAPLACE.value,
                "sensitivity": SensitivityLevel.UNKNOWN.value,
                "data_type": column_types.get(col, "unknown"),
                "recommended": False,
            }

    st.session_state.column_configs = configs


# =============================================================================
# UI Components
# =============================================================================


def render_global_settings() -> None:
    """Render global privacy settings."""
    st.markdown("### Global Settings")

    # Show comprehensive help for understanding privacy settings
    info_button("epsilon", "Learn about Epsilon (ε) - The Privacy Parameter")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        global_eps = st.slider(
            "Default Epsilon (ε)",
            min_value=0.1,
            max_value=10.0,
            value=get_global_epsilon(),
            step=0.1,
            help="Privacy parameter. Lower = stronger privacy (more noise). Higher = more utility (less noise).",
            key="global_epsilon_slider",
        )
        set_global_epsilon(global_eps)

    with col2:
        if global_eps <= 0.5:
            privacy_level = "High Privacy"
            level_desc = "Strong protection, more noise"
        elif global_eps <= 2.0:
            privacy_level = "Balanced"
            level_desc = "Good balance of privacy & utility"
        else:
            privacy_level = "High Utility"
            level_desc = "More accuracy, less privacy"
        st.metric("Privacy Level", privacy_level, help=level_desc)

    with col3:
        # Calculate total epsilon for protected columns
        configs = get_column_configs()
        protected = [c for c, cfg in configs.items()
                     if cfg.get("mode") == ProtectionMode.PROTECT.value]
        total_eps = sum(
            configs[c].get("epsilon", global_eps) for c in protected
        )
        st.metric(
            "Total ε (Protected)",
            f"{total_eps:.2f}",
            help="Sum of epsilon across all protected columns"
        )

    # Add info button for total epsilon
    info_button("total_epsilon", "What does Total Epsilon mean?")


def render_auto_recommend_button(df: pd.DataFrame, columns: List[str]) -> None:
    """Render the auto-recommend button."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "Auto-Recommend Settings",
            type="primary",
            width="stretch",
            help="Analyze columns and recommend privacy settings",
        ):
            with st.spinner("Analyzing columns..."):
                recommendations = generate_recommendations(df)
                set_recommendations(recommendations)
                apply_recommendations_to_configs(recommendations, columns)
                st.success("Recommendations applied!")
                st.rerun()


def render_recommendations_summary(recommendations: DatasetRecommendation) -> None:
    """Render a summary of recommendations."""
    st.markdown("### Recommendation Summary")

    # Count sensitivity levels
    high_count = 0
    med_count = 0
    low_count = 0

    for rec in recommendations.column_recommendations.values():
        sens = rec.epsilon_recommendation.sensitivity
        if sens == SensitivityLevel.HIGH:
            high_count += 1
        elif sens == SensitivityLevel.MEDIUM:
            med_count += 1
        elif sens == SensitivityLevel.LOW:
            low_count += 1

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "High Sensitivity",
            high_count,
            help="Columns requiring strong privacy protection",
        )

    with col2:
        st.metric(
            "Medium Sensitivity",
            med_count,
            help="Columns requiring balanced protection",
        )

    with col3:
        st.metric(
            "Low Sensitivity",
            low_count,
            help="Columns requiring minimal protection",
        )

    with col4:
        st.metric(
            "Recommended Total ε",
            f"{recommendations.total_epsilon:.2f}",
            help="Sum of recommended epsilon values",
        )

    # Add explanation for sensitivity levels
    info_button("sensitivity", "What do the sensitivity levels mean?")


def render_column_config_table(
    columns: List[str],
    column_types: Dict[str, str],
) -> None:
    """Render the column configuration table."""
    st.markdown("### Column Configuration")

    # Add help expanders for protection modes and mechanisms
    col_help1, col_help2 = st.columns(2)
    with col_help1:
        info_button("mechanism", "Understanding DP Mechanisms")
    with col_help2:
        with st.expander("ℹ️ Protection Mode Options"):
            st.markdown(
                "**Protect**: Apply differential privacy noise to protect individual values\n\n"
                "**Passthrough**: Keep original values unchanged (no privacy protection)\n\n"
                "**Exclude**: Remove this column entirely from the output"
            )

    configs = get_column_configs()
    global_eps = get_global_epsilon()

    # Column headers
    header_cols = st.columns([2, 2, 1.5, 1.5, 1])
    with header_cols[0]:
        st.markdown("**Column**")
    with header_cols[1]:
        st.markdown("**Protection Mode**")
    with header_cols[2]:
        st.markdown("**Epsilon**")
    with header_cols[3]:
        st.markdown("**Mechanism**")
    with header_cols[4]:
        st.markdown("**Sensitivity**")

    st.markdown("---")

    # Render each column
    for col in columns:
        config = configs.get(col, {})
        render_column_row(col, config, column_types.get(col, "unknown"), global_eps)


def render_column_row(
    column: str,
    config: Dict[str, Any],
    col_type: str,
    global_eps: float,
) -> None:
    """Render a single column configuration row."""
    cols = st.columns([2, 2, 1.5, 1.5, 1])

    current_mode = config.get("mode", ProtectionMode.PROTECT.value)
    current_eps = config.get("epsilon", global_eps)
    current_mech = config.get("mechanism", RecommendedMechanism.LAPLACE.value)
    sensitivity = config.get("sensitivity", "unknown")

    with cols[0]:
        # Column name with type indicator
        type_emoji = {
            "numeric": "🔢",
            "categorical": "📋",
            "date": "📅",
            "unknown": "❓",
        }.get(col_type, "❓")
        st.markdown(f"{type_emoji} **{column}**")

    with cols[1]:
        # Protection mode selector
        mode_options = list(PROTECTION_MODE_LABELS.keys())
        current_idx = next(
            (i for i, m in enumerate(mode_options) if m.value == current_mode),
            0
        )
        new_mode: ProtectionMode = st.selectbox(
            "Mode",
            options=mode_options,
            format_func=lambda x: PROTECTION_MODE_LABELS[x],
            index=current_idx,
            key=f"mode_{column}",
            label_visibility="collapsed",
        )

    with cols[2]:
        # Epsilon input (only for Protect mode)
        if new_mode == ProtectionMode.PROTECT:
            new_eps = st.number_input(
                "Epsilon",
                min_value=0.01,
                max_value=10.0,
                value=float(current_eps),
                step=0.1,
                key=f"eps_{column}",
                label_visibility="collapsed",
            )
        else:
            st.markdown("—")
            new_eps = current_eps

    with cols[3]:
        # Mechanism selector (only for Protect mode)
        new_mech: RecommendedMechanism
        if new_mode == ProtectionMode.PROTECT:
            mech_options = list(MECHANISM_LABELS.keys())
            current_mech_idx = next(
                (i for i, m in enumerate(mech_options) if m.value == current_mech),
                0
            )
            new_mech = st.selectbox(
                "Mechanism",
                options=mech_options,
                format_func=lambda x: MECHANISM_LABELS[x],
                index=current_mech_idx,
                key=f"mech_{column}",
                label_visibility="collapsed",
            )
        else:
            st.markdown("—")
            new_mech = RecommendedMechanism.LAPLACE

    with cols[4]:
        # Sensitivity badge
        sens_upper = sensitivity.upper()
        color = SENSITIVITY_COLORS.get(sens_upper, "#6c757d")
        st.markdown(
            f'<span style="background-color: {color}; color: white; '
            f'padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">'
            f'{sens_upper}</span>',
            unsafe_allow_html=True,
        )

    # Update config if changed
    new_config = {
        "mode": new_mode.value,
        "epsilon": new_eps,
        "mechanism": new_mech.value if isinstance(new_mech, RecommendedMechanism)
        else current_mech,
        "sensitivity": sensitivity,
        "data_type": col_type,
        "recommended": config.get("recommended", False),
    }

    if new_config != config:
        set_column_config(column, new_config)


def render_configuration_summary() -> None:
    """Render a summary of the current configuration."""
    st.markdown("### Configuration Summary")

    configs = get_column_configs()

    # Count by mode
    mode_counts = {mode: 0 for mode in ProtectionMode}
    for cfg in configs.values():
        mode_val = cfg.get("mode", ProtectionMode.PROTECT.value)
        for m in ProtectionMode:
            if m.value == mode_val:
                mode_counts[m] += 1
                break

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Protected", mode_counts[ProtectionMode.PROTECT])

    with col2:
        st.metric("Passthrough", mode_counts[ProtectionMode.PASSTHROUGH])

    with col3:
        st.metric("Excluded", mode_counts[ProtectionMode.EXCLUDE])

    # Calculate total epsilon
    total_eps = sum(
        cfg.get("epsilon", 1.0)
        for cfg in configs.values()
        if cfg.get("mode") == ProtectionMode.PROTECT.value
    )

    st.markdown(f"**Total Privacy Budget (ε):** {total_eps:.2f}")


def render_validation_warnings() -> bool:
    """Check configuration and display warnings. Returns True if valid."""
    configs = get_column_configs()
    warnings = []
    errors = []

    # Check for at least one protected column
    protected = [c for c, cfg in configs.items()
                 if cfg.get("mode") == ProtectionMode.PROTECT.value]

    if len(protected) == 0:
        warnings.append(
            "No columns are set to 'Protect'. At least one column should have "
            "differential privacy applied."
        )

    # Check for very high total epsilon
    total_eps = sum(
        configs[c].get("epsilon", 1.0) for c in protected
    )
    if total_eps > 10.0:
        warnings.append(
            f"Total epsilon ({total_eps:.2f}) is high. This may not provide "
            "adequate privacy protection for sensitive data."
        )

    # Check for very low epsilon on any column
    for col, cfg in configs.items():
        if cfg.get("mode") == ProtectionMode.PROTECT.value:
            eps = cfg.get("epsilon", 1.0)
            if eps < 0.1:
                warnings.append(
                    f"Column '{col}' has very low epsilon ({eps}). "
                    "This will add significant noise."
                )

    # Check if all columns are excluded
    excluded = [c for c, cfg in configs.items()
                if cfg.get("mode") == ProtectionMode.EXCLUDE.value]
    if len(excluded) == len(configs):
        errors.append("All columns are excluded. No output data will be generated.")

    # Display
    for error in errors:
        st.error(error)

    for warning in warnings:
        st.warning(warning)

    return len(errors) == 0


def render_navigation_buttons() -> None:
    """Render navigation buttons."""
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back to Upload", width="stretch"):
            st.session_state.current_step = 1
            st.rerun()

    with col3:
        is_valid = render_validation_warnings()

        if st.button(
            "Apply & Continue →",
            type="primary",
            disabled=not is_valid,
            width="stretch",
        ):
            # Store final configuration
            configs = get_column_configs()
            total_eps = sum(
                cfg.get("epsilon", 1.0)
                for cfg in configs.values()
                if cfg.get("mode") == ProtectionMode.PROTECT.value
            )
            st.session_state.total_epsilon = total_eps
            st.session_state.config_complete = True
            st.session_state.current_step = 3
            st.rerun()


# =============================================================================
# Main Page Function
# =============================================================================


def render_configure_page() -> None:
    """Render the complete configure page."""
    st.header("Step 2: Configure Privacy Settings")

    # Check if dataset is loaded
    if not st.session_state.get("dataset_loaded", False):
        st.warning("Please upload a dataset first.")
        if st.button("Go to Upload"):
            st.session_state.current_step = 1
            st.rerun()
        return

    # Get dataset info
    df = st.session_state.get("original_df")
    dataset_info = st.session_state.get("dataset_info", {})
    columns = dataset_info.get("columns", [])
    column_types = dataset_info.get("column_types", {})

    if df is None or not columns:
        st.error("Dataset information not found. Please re-upload your dataset.")
        return

    st.markdown(
        "Configure how differential privacy will be applied to each column. "
        "Use auto-recommend for intelligent suggestions based on column content."
    )

    st.markdown("---")

    # Initialize configs if needed
    initialize_column_configs(columns, column_types)

    # Global settings
    render_global_settings()

    st.markdown("---")

    # Auto-recommend button
    render_auto_recommend_button(df, columns)

    # Show recommendations summary if available
    recommendations = get_recommendations()
    if recommendations is not None:
        st.markdown("---")
        render_recommendations_summary(recommendations)

    st.markdown("---")

    # Column configuration table
    render_column_config_table(columns, column_types)

    st.markdown("---")

    # Configuration summary
    render_configuration_summary()

    st.markdown("---")

    # Navigation
    render_navigation_buttons()
