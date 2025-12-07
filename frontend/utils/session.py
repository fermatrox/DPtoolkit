"""Session state management for DPtoolkit.

This module provides utilities for managing Streamlit session state,
including initialization, data storage, and cleanup.

Session data is stored only in memory and cleared on application close
or timeout, as required by the PRD security specifications.
"""

from typing import Any, Dict, Optional, cast

import streamlit as st
import pandas as pd


# =============================================================================
# Session State Keys
# =============================================================================


# Core workflow state
WORKFLOW_KEYS = [
    "current_step",
    "dataset_loaded",
    "config_complete",
    "transform_complete",
]

# Dataset state
DATASET_KEYS = [
    "original_df",
    "protected_df",
    "dataset_info",
    "dataset_profile",
]

# Configuration state
CONFIG_KEYS = [
    "column_configs",
    "global_epsilon",
    "total_epsilon",
    "recommendations",
]

# Analysis state
ANALYSIS_KEYS = [
    "comparison_results",
    "divergence_metrics",
    "visualizations",
]

# All session keys
ALL_SESSION_KEYS = WORKFLOW_KEYS + DATASET_KEYS + CONFIG_KEYS + ANALYSIS_KEYS


# =============================================================================
# Session State Initialization
# =============================================================================


def initialize_session_state() -> None:
    """Initialize all session state variables with defaults.

    This should be called at the start of the app to ensure all
    required state variables exist.
    """
    # Workflow state
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    if "dataset_loaded" not in st.session_state:
        st.session_state.dataset_loaded = False

    if "config_complete" not in st.session_state:
        st.session_state.config_complete = False

    if "transform_complete" not in st.session_state:
        st.session_state.transform_complete = False

    # Dataset state
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    if "protected_df" not in st.session_state:
        st.session_state.protected_df = None

    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = {}

    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None

    # Configuration state
    if "column_configs" not in st.session_state:
        st.session_state.column_configs = {}

    if "global_epsilon" not in st.session_state:
        st.session_state.global_epsilon = 1.0

    if "total_epsilon" not in st.session_state:
        st.session_state.total_epsilon = 0.0

    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None

    # Analysis state
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    if "divergence_metrics" not in st.session_state:
        st.session_state.divergence_metrics = None

    if "visualizations" not in st.session_state:
        st.session_state.visualizations = {}


# =============================================================================
# Session State Accessors
# =============================================================================


def get_session_value(key: str, default: Any = None) -> Any:
    """Get a value from session state.

    Args:
        key: The session state key.
        default: Default value if key doesn't exist.

    Returns:
        The session state value or default.
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """Set a value in session state.

    Args:
        key: The session state key.
        value: The value to set.
    """
    st.session_state[key] = value


def has_session_key(key: str) -> bool:
    """Check if a key exists in session state.

    Args:
        key: The session state key.

    Returns:
        True if key exists, False otherwise.
    """
    return key in st.session_state


# =============================================================================
# Dataset Management
# =============================================================================


def store_dataset(
    df: pd.DataFrame,
    filename: str,
    file_format: str,
) -> None:
    """Store a dataset in session state.

    Args:
        df: The DataFrame to store.
        filename: Original filename.
        file_format: File format (csv, xlsx, parquet).
    """
    st.session_state.original_df = df
    st.session_state.dataset_info = {
        "filename": filename,
        "file_format": file_format,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }
    st.session_state.dataset_loaded = True


def get_original_dataframe() -> Optional[pd.DataFrame]:
    """Get the original (unprotected) DataFrame.

    Returns:
        The original DataFrame or None if not loaded.
    """
    return st.session_state.get("original_df")


def get_protected_dataframe() -> Optional[pd.DataFrame]:
    """Get the protected DataFrame.

    Returns:
        The protected DataFrame or None if not generated.
    """
    return st.session_state.get("protected_df")


def store_protected_dataframe(df: pd.DataFrame) -> None:
    """Store the protected DataFrame.

    Args:
        df: The protected DataFrame.
    """
    st.session_state.protected_df = df


def get_dataset_info() -> Dict[str, Any]:
    """Get dataset metadata.

    Returns:
        Dictionary with dataset information.
    """
    return cast(Dict[str, Any], st.session_state.get("dataset_info", {}))


# =============================================================================
# Configuration Management
# =============================================================================


def store_column_config(column: str, config: Dict) -> None:
    """Store configuration for a single column.

    Args:
        column: Column name.
        config: Configuration dictionary.
    """
    if "column_configs" not in st.session_state:
        st.session_state.column_configs = {}
    st.session_state.column_configs[column] = config


def get_column_config(column: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a single column.

    Args:
        column: Column name.

    Returns:
        Configuration dictionary or None.
    """
    raw = st.session_state.get("column_configs", {})
    configs = cast(Dict[str, Dict[str, Any]], raw)
    return configs.get(column)


def get_all_column_configs() -> Dict[str, Dict[str, Any]]:
    """Get all column configurations.

    Returns:
        Dictionary of column configurations.
    """
    return cast(Dict[str, Dict[str, Any]], st.session_state.get("column_configs", {}))


def store_recommendations(recommendations: Any) -> None:
    """Store recommendations from the advisor.

    Args:
        recommendations: DatasetRecommendation object.
    """
    st.session_state.recommendations = recommendations


def get_recommendations() -> Optional[Any]:
    """Get stored recommendations.

    Returns:
        DatasetRecommendation object or None.
    """
    return st.session_state.get("recommendations")


# =============================================================================
# Analysis Results Management
# =============================================================================


def store_comparison_results(results: Any) -> None:
    """Store comparison results.

    Args:
        results: DatasetComparison object.
    """
    st.session_state.comparison_results = results


def get_comparison_results() -> Optional[Any]:
    """Get stored comparison results.

    Returns:
        DatasetComparison object or None.
    """
    return st.session_state.get("comparison_results")


def store_visualization(key: str, figure: Any) -> None:
    """Store a visualization figure.

    Args:
        key: Unique key for the visualization.
        figure: Plotly figure object.
    """
    if "visualizations" not in st.session_state:
        st.session_state.visualizations = {}
    st.session_state.visualizations[key] = figure


def get_visualization(key: str) -> Optional[Any]:
    """Get a stored visualization.

    Args:
        key: Unique key for the visualization.

    Returns:
        Plotly figure or None.
    """
    return st.session_state.get("visualizations", {}).get(key)


# =============================================================================
# Workflow State Management
# =============================================================================


def get_current_step() -> int:
    """Get the current workflow step.

    Returns:
        Current step number (1-4).
    """
    return cast(int, st.session_state.get("current_step", 1))


def set_current_step(step: int) -> None:
    """Set the current workflow step.

    Args:
        step: Step number (1-4).
    """
    if 1 <= step <= 4:
        st.session_state.current_step = step


def can_proceed_to_step(step: int) -> bool:
    """Check if user can proceed to a specific step.

    Args:
        step: Target step number.

    Returns:
        True if user can proceed, False otherwise.
    """
    if step == 1:
        return True
    elif step == 2:
        return bool(st.session_state.get("dataset_loaded", False))
    elif step == 3:
        return bool(st.session_state.get("config_complete", False))
    elif step == 4:
        return bool(st.session_state.get("transform_complete", False))
    return False


def complete_step(step: int) -> None:
    """Mark a step as complete.

    Args:
        step: Step number to mark complete.
    """
    if step == 1:
        st.session_state.dataset_loaded = True
    elif step == 2:
        st.session_state.config_complete = True
    elif step == 3:
        st.session_state.transform_complete = True


# =============================================================================
# Session Cleanup
# =============================================================================


def clear_session_data() -> None:
    """Clear all session data.

    This resets the application to its initial state.
    Called on user request or session timeout.
    """
    # Clear all known keys
    for key in ALL_SESSION_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize with defaults
    initialize_session_state()


def clear_dataset_data() -> None:
    """Clear only dataset-related data.

    Useful when loading a new dataset without clearing config.
    """
    for key in DATASET_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.original_df = None
    st.session_state.protected_df = None
    st.session_state.dataset_info = {}
    st.session_state.dataset_profile = None


def clear_analysis_data() -> None:
    """Clear only analysis-related data.

    Useful when re-running analysis with new config.
    """
    for key in ANALYSIS_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.comparison_results = None
    st.session_state.divergence_metrics = None
    st.session_state.visualizations = {}


# =============================================================================
# Session State Summary
# =============================================================================


def get_session_summary() -> Dict:
    """Get a summary of current session state.

    Returns:
        Dictionary with session summary information.
    """
    return {
        "current_step": get_current_step(),
        "dataset_loaded": st.session_state.get("dataset_loaded", False),
        "config_complete": st.session_state.get("config_complete", False),
        "transform_complete": st.session_state.get("transform_complete", False),
        "dataset_info": get_dataset_info(),
        "total_epsilon": st.session_state.get("total_epsilon", 0.0),
        "num_column_configs": len(get_all_column_configs()),
        "has_comparison": get_comparison_results() is not None,
    }
