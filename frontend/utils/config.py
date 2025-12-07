"""Configuration settings for DPtoolkit frontend.

This module contains all configuration constants for the Streamlit
application, including page settings, theming, and UI constants.
"""

from typing import Dict, Any


# =============================================================================
# Page Configuration
# =============================================================================


PAGE_CONFIG: Dict[str, Any] = {
    "page_title": "DPtoolkit - Differential Privacy Toolkit",
    "page_icon": "🔒",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


# =============================================================================
# Theme Configuration
# =============================================================================


THEME_CONFIG: Dict[str, str] = {
    "primary_color": "#1f77b4",
    "background_color": "#ffffff",
    "secondary_background_color": "#f8f9fa",
    "text_color": "#262730",
    "font": "sans-serif",
}


# =============================================================================
# Workflow Steps
# =============================================================================


WORKFLOW_STEPS = [
    {
        "number": 1,
        "name": "Upload Data",
        "description": "Upload your dataset (CSV, Excel, or Parquet)",
        "icon": "📁",
    },
    {
        "number": 2,
        "name": "Configure",
        "description": "Set privacy parameters for each column",
        "icon": "⚙️",
    },
    {
        "number": 3,
        "name": "Analyze",
        "description": "Review the impact of differential privacy",
        "icon": "📊",
    },
    {
        "number": 4,
        "name": "Export",
        "description": "Download protected dataset and reports",
        "icon": "💾",
    },
]


# =============================================================================
# File Upload Settings
# =============================================================================


UPLOAD_CONFIG: Dict[str, Any] = {
    "allowed_extensions": ["csv", "xlsx", "xls", "parquet"],
    "max_file_size_mb": 200,
    "max_rows": 1_000_000,
    "max_columns": 200,
}


# =============================================================================
# Privacy Parameter Defaults
# =============================================================================


PRIVACY_DEFAULTS: Dict[str, Any] = {
    "default_epsilon": 1.0,
    "min_epsilon": 0.01,
    "max_epsilon": 10.0,
    "epsilon_step": 0.1,
    "default_delta": 1e-5,
}


# =============================================================================
# Display Settings
# =============================================================================


DISPLAY_CONFIG: Dict[str, Any] = {
    "preview_rows": 10,
    "max_display_columns": 50,
    "float_precision": 4,
    "table_height": 400,
}


# =============================================================================
# Sensitivity Level Colors
# =============================================================================


SENSITIVITY_COLORS: Dict[str, str] = {
    "HIGH": "#dc3545",      # Red
    "MEDIUM": "#ffc107",    # Yellow/Orange
    "LOW": "#28a745",       # Green
    "UNKNOWN": "#6c757d",   # Gray
}


# =============================================================================
# Mechanism Display Names
# =============================================================================


MECHANISM_NAMES: Dict[str, str] = {
    "LAPLACE": "Laplace Mechanism",
    "GAUSSIAN": "Gaussian Mechanism",
    "EXPONENTIAL": "Exponential Mechanism",
}


# =============================================================================
# Help Text
# =============================================================================


HELP_TEXT: Dict[str, str] = {
    "epsilon": (
        "Epsilon (ε) is the privacy parameter that controls the privacy-utility "
        "tradeoff. Lower values provide stronger privacy but may reduce data utility."
    ),
    "delta": (
        "Delta (δ) is used with the Gaussian mechanism. It represents the probability "
        "of privacy loss exceeding the epsilon bound."
    ),
    "laplace": (
        "The Laplace mechanism adds noise drawn from a Laplace distribution. "
        "Best for bounded numeric data with known min/max values."
    ),
    "gaussian": (
        "The Gaussian mechanism adds noise drawn from a Gaussian distribution. "
        "Better for unbounded numeric data or when (ε,δ)-DP is acceptable."
    ),
    "exponential": (
        "The Exponential mechanism selects categories with probabilities based on "
        "utility scores. Used for categorical data."
    ),
    "sensitivity_high": (
        "High sensitivity columns (e.g., SSN, patient ID) require strong privacy "
        "protection with low epsilon values (0.1-0.5)."
    ),
    "sensitivity_medium": (
        "Medium sensitivity columns (e.g., age, ZIP code) need moderate protection "
        "with epsilon values around 0.5-2.0."
    ),
    "sensitivity_low": (
        "Low sensitivity columns (e.g., country, product category) can use higher "
        "epsilon values (2.0-5.0) for better utility."
    ),
}
