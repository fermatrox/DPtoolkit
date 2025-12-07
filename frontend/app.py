"""DPtoolkit - Differential Privacy Toolkit.

Main entry point for the Streamlit application.
This app provides a wizard-style interface for applying differential
privacy to healthcare datasets.

Run with: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Add frontend directory to path for imports
frontend_dir = Path(__file__).parent
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))

import streamlit as st  # noqa: E402

from utils.session import initialize_session_state, clear_session_data  # noqa: E402
from utils.config import PAGE_CONFIG  # noqa: E402
from pages.upload import render_upload_page  # noqa: E402
from pages.configure import render_configure_page  # noqa: E402
from pages.analyze import render_analyze_page  # noqa: E402
from pages.export import render_export_page  # noqa: E402


# =============================================================================
# Page Configuration
# =============================================================================


def configure_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title=PAGE_CONFIG["page_title"],
        page_icon=PAGE_CONFIG["page_icon"],
        layout=PAGE_CONFIG["layout"],
        initial_sidebar_state=PAGE_CONFIG["initial_sidebar_state"],
    )


# =============================================================================
# Custom CSS
# =============================================================================


def apply_custom_css() -> None:
    """Apply custom CSS styling."""
    st.markdown(
        """
        <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }

        /* Header styling */
        h1 {
            color: #1f77b4;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 4px;
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background-color: #1f77b4;
            color: white;
        }

        /* Success/warning/error message styling */
        .stSuccess, .stWarning, .stError {
            border-radius: 4px;
        }

        /* Data frame styling */
        .stDataFrame {
            border-radius: 4px;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        /* Progress bar styling */
        .stProgress > div > div {
            background-color: #1f77b4;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
        }

        /* Hide Streamlit branding (optional) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Sidebar Navigation
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with navigation and session info."""
    with st.sidebar:
        st.title("DP Toolkit")
        st.markdown("---")

        # Navigation
        st.subheader("Navigation")

        # Get current step from session state
        current_step = st.session_state.get("current_step", 1)
        dataset_loaded = st.session_state.get("dataset_loaded", False)
        config_complete = st.session_state.get("config_complete", False)
        transform_complete = st.session_state.get("transform_complete", False)

        # Step indicators with status
        steps = [
            ("1. Upload Data", 1, True),
            ("2. Configure", 2, dataset_loaded),
            ("3. Analyze", 3, config_complete),
            ("4. Export", 4, transform_complete),
        ]

        for step_name, step_num, enabled in steps:
            if step_num == current_step:
                st.markdown(f"**{step_name}** (current)")
            elif enabled:
                if st.button(step_name, key=f"nav_{step_num}"):
                    st.session_state.current_step = step_num
                    st.rerun()
            else:
                st.markdown(f"~~{step_name}~~")

        st.markdown("---")

        # Session info
        st.subheader("Session Info")

        if dataset_loaded:
            dataset_info = st.session_state.get("dataset_info", {})
            st.markdown(f"**Dataset:** {dataset_info.get('filename', 'N/A')}")
            st.markdown(f"**Rows:** {dataset_info.get('row_count', 0):,}")
            st.markdown(f"**Columns:** {dataset_info.get('column_count', 0)}")

            if config_complete:
                epsilon = st.session_state.get("total_epsilon", 0)
                st.markdown(f"**Total ε:** {epsilon:.2f}")
        else:
            st.info("No dataset loaded")

        st.markdown("---")

        # Clear session button
        if st.button("Clear Session", type="secondary"):
            clear_session_data()
            st.rerun()

        # Help section
        st.markdown("---")
        with st.expander("Help", expanded=False):
            st.markdown(
                """
                **How to use:**
                1. **Upload** your dataset (CSV, Excel, Parquet)
                2. **Configure** privacy settings per column
                3. **Analyze** the impact of DP on your data
                4. **Export** the protected dataset

                **Privacy Parameters:**
                - **Epsilon (ε):** Controls privacy-utility tradeoff.
                  Lower = more privacy, more noise.
                - **Mechanism:** Algorithm used to add noise.

                **Need more help?**
                Click the ℹ️ icons throughout the app for detailed explanations.
                """
            )

        # Version info
        st.caption("v1.0.0 | Differential Privacy Toolkit")


# =============================================================================
# Main Content Router
# =============================================================================


def render_main_content() -> None:
    """Render the main content based on current step."""
    current_step = st.session_state.get("current_step", 1)

    if current_step == 1:
        render_upload_page()
    elif current_step == 2:
        render_configure_page()
    elif current_step == 3:
        render_analyze_page()
    elif current_step == 4:
        render_export_page()
    else:
        st.error("Unknown step")


# =============================================================================
# Main Application
# =============================================================================


def main() -> None:
    """Main application entry point."""
    # Configure page (must be first Streamlit command)
    configure_page()

    # Initialize session state
    initialize_session_state()

    # Apply custom styling
    apply_custom_css()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
