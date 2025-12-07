"""UI utility components for DPtoolkit.

This module provides reusable UI components for consistent error handling,
loading states, tooltips, and help text throughout the application.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

import streamlit as st

T = TypeVar("T")


# =============================================================================
# Error Messages
# =============================================================================


class ErrorMessages:
    """Comprehensive error messages with helpful hints."""

    # File upload errors
    FILE_TOO_LARGE = (
        "File too large ({size:.1f} MB). Maximum supported size is {max_size} MB.\n\n"
        "**Suggestions:**\n"
        "- Split your dataset into smaller files\n"
        "- Remove unnecessary columns before upload\n"
        "- Use Parquet format for better compression"
    )

    FILE_FORMAT_UNSUPPORTED = (
        "Unsupported file format: `{extension}`\n\n"
        "**Supported formats:**\n"
        "- CSV (.csv, .tsv, .txt)\n"
        "- Excel (.xlsx, .xls)\n"
        "- Parquet (.parquet, .pq)"
    )

    FILE_ENCODING_ERROR = (
        "Could not read file with any supported encoding.\n\n"
        "**Suggestions:**\n"
        "- Open the file in a text editor and save as UTF-8\n"
        "- Check for special characters in the file\n"
        "- Try converting the file to a different format"
    )

    FILE_PARSE_ERROR = (
        "Error parsing file: {error}\n\n"
        "**Suggestions:**\n"
        "- Check that the file is not corrupted\n"
        "- For CSV files, ensure consistent delimiters\n"
        "- For Excel files, ensure the sheet is not password-protected"
    )

    EMPTY_DATASET = (
        "Dataset is empty (0 rows).\n\n"
        "Please upload a file containing data."
    )

    NO_COLUMNS = (
        "Dataset has no columns.\n\n"
        "Please check the file format and ensure headers are present."
    )

    # Configuration errors
    NO_PROTECTED_COLUMNS = (
        "No columns are configured for protection.\n\n"
        "**Tip:** Select 'Protect (Apply DP)' for at least one column "
        "to apply differential privacy."
    )

    ALL_COLUMNS_EXCLUDED = (
        "All columns are set to 'Exclude'.\n\n"
        "This would result in an empty output dataset. "
        "Please keep at least one column as 'Protect' or 'Passthrough'."
    )

    HIGH_TOTAL_EPSILON = (
        "Total epsilon ({epsilon:.2f}) is high.\n\n"
        "**Privacy considerations:**\n"
        "- High epsilon = less privacy protection\n"
        "- For sensitive healthcare data, consider ε < 5.0\n"
        "- Use 'Auto-Recommend' for suggested values"
    )

    LOW_COLUMN_EPSILON = (
        "Column '{column}' has very low epsilon ({epsilon}).\n\n"
        "**Impact:** This will add significant noise and may reduce data utility.\n"
        "Consider increasing epsilon if data accuracy is important."
    )

    # Transformation errors
    TRANSFORM_ERROR = (
        "Error applying differential privacy: {error}\n\n"
        "**Suggestions:**\n"
        "- Check column configurations\n"
        "- Ensure numeric columns contain valid numbers\n"
        "- Try using a different mechanism"
    )

    COMPARISON_ERROR = (
        "Error comparing datasets: {error}\n\n"
        "**This may occur when:**\n"
        "- Column types changed during transformation\n"
        "- All values in a column are null\n"
        "- Numeric columns contain non-numeric data"
    )

    # Session errors
    SESSION_EXPIRED = (
        "Session data not found.\n\n"
        "Your session may have expired or been cleared. "
        "Please start again from the Upload step."
    )

    DATASET_NOT_FOUND = (
        "Dataset not found in session.\n\n"
        "Please re-upload your dataset to continue."
    )

    CONFIG_NOT_FOUND = (
        "Configuration not found.\n\n"
        "Please configure privacy settings before proceeding."
    )

    # PDF generation errors
    PDF_GENERATION_ERROR = (
        "Error generating PDF report: {error}\n\n"
        "**Suggestions:**\n"
        "- Ensure all analysis has completed\n"
        "- Try running the analysis again\n"
        "- Check that visualization data is valid"
    )


def show_error(
    message: str,
    *,
    icon: str = "🚨",
    expanded: bool = True,
    key: Optional[str] = None,
) -> None:
    """Display a comprehensive error message.

    Args:
        message: Error message with markdown formatting.
        icon: Icon to display with the error.
        expanded: Whether to show the full message expanded.
        key: Optional unique key for the component.
    """
    st.error(f"{icon} **Error**", icon=None)
    st.markdown(message)


def show_warning(
    message: str,
    *,
    icon: str = "⚠️",
    key: Optional[str] = None,
) -> None:
    """Display a warning message.

    Args:
        message: Warning message with markdown formatting.
        icon: Icon to display with the warning.
        key: Optional unique key for the component.
    """
    st.warning(f"{icon} {message}", icon=None)


def show_info(
    message: str,
    *,
    icon: str = "ℹ️",
    key: Optional[str] = None,
) -> None:
    """Display an informational message.

    Args:
        message: Info message with markdown formatting.
        icon: Icon to display with the info.
        key: Optional unique key for the component.
    """
    st.info(f"{icon} {message}", icon=None)


def show_success(
    message: str,
    *,
    icon: str = "✅",
    key: Optional[str] = None,
) -> None:
    """Display a success message.

    Args:
        message: Success message.
        icon: Icon to display with the message.
        key: Optional unique key for the component.
    """
    st.success(f"{icon} {message}", icon=None)


# =============================================================================
# Loading States and Progress
# =============================================================================


@contextmanager
def loading_state(
    message: str = "Processing...",
    success_message: Optional[str] = None,
) -> Iterator[None]:
    """Context manager for displaying loading state.

    Args:
        message: Message to display while loading.
        success_message: Optional message to display on success.

    Usage:
        with loading_state("Loading dataset..."):
            df = load_data()
    """
    with st.spinner(message):
        yield

    if success_message:
        st.success(success_message)


class ProgressTracker:
    """Track and display progress for multi-step operations."""

    def __init__(
        self,
        steps: List[str],
        title: str = "Progress",
    ) -> None:
        """Initialize the progress tracker.

        Args:
            steps: List of step names.
            title: Title for the progress section.
        """
        self.steps = steps
        self.title = title
        self.current_step = 0
        self._progress_bar: Optional[Any] = None
        self._status_text: Optional[Any] = None

    def start(self) -> None:
        """Start tracking progress."""
        st.markdown(f"**{self.title}**")
        self._progress_bar = st.progress(0)
        self._status_text = st.empty()

    def update(self, step: int, message: Optional[str] = None) -> None:
        """Update progress to a specific step.

        Args:
            step: Step number (0-indexed).
            message: Optional custom message.
        """
        self.current_step = step
        progress = (step + 1) / len(self.steps)

        if self._progress_bar:
            self._progress_bar.progress(progress)

        if self._status_text:
            msg = message or f"Step {step + 1}/{len(self.steps)}: {self.steps[step]}"
            self._status_text.markdown(f"*{msg}*")

    def complete(self, message: str = "Complete!") -> None:
        """Mark progress as complete.

        Args:
            message: Completion message.
        """
        if self._progress_bar:
            self._progress_bar.progress(1.0)

        if self._status_text:
            self._status_text.markdown(f"**{message}**")


def run_with_progress(
    operations: List[Tuple[str, Callable[[], T]]],
    title: str = "Progress",
) -> List[T]:
    """Run multiple operations with progress tracking.

    Args:
        operations: List of (step_name, callable) tuples.
        title: Title for the progress section.

    Returns:
        List of results from each operation.
    """
    tracker = ProgressTracker([name for name, _ in operations], title)
    tracker.start()

    results: List[T] = []
    for i, (name, func) in enumerate(operations):
        tracker.update(i, f"Running: {name}")
        result = func()
        results.append(result)

    tracker.complete()
    return results


# =============================================================================
# Help Text and Tooltips
# =============================================================================


HELP_TEXTS = {
    # Privacy concepts
    "epsilon": (
        "**Epsilon (ε)** is the privacy parameter that controls the tradeoff "
        "between privacy and data utility.\n\n"
        "- **Lower ε (0.1-0.5)**: Stronger privacy, more noise, less accurate data\n"
        "- **Medium ε (0.5-2.0)**: Balanced privacy and utility\n"
        "- **Higher ε (2.0-10.0)**: Weaker privacy, less noise, more accurate data\n\n"
        "For sensitive healthcare data, we recommend ε ≤ 1.0."
    ),
    "delta": (
        "**Delta (δ)** is an additional privacy parameter used with the "
        "Gaussian mechanism.\n\n"
        "It represents the probability that pure ε-differential privacy is violated. "
        "Typical values are very small, like 10⁻⁵ or 10⁻⁶."
    ),
    "mechanism": (
        "**Differential Privacy Mechanisms** determine how noise is added:\n\n"
        "- **Laplace**: Best for bounded numeric data (counts, ages)\n"
        "- **Gaussian**: Better for unbounded data with known variance\n"
        "- **Exponential**: Designed for categorical data"
    ),
    "sensitivity": (
        "**Sensitivity** measures how much one person's data can change a result.\n\n"
        "Higher sensitivity = more noise needed for the same privacy.\n"
        "- **High**: SSN, names, exact addresses\n"
        "- **Medium**: Age, ZIP code, diagnosis\n"
        "- **Low**: General statistics, counts"
    ),
    # Protection modes
    "protect": (
        "**Protect (Apply DP)**: Differential privacy noise will be added "
        "to this column to protect individual values."
    ),
    "passthrough": (
        "**Passthrough**: This column will be included unchanged.\n\n"
        "Use for columns that don't contain sensitive information, like IDs."
    ),
    "exclude": (
        "**Exclude**: This column will be removed from the output.\n\n"
        "Use for highly sensitive columns that shouldn't be shared at all."
    ),
    # Data quality
    "mae": (
        "**Mean Absolute Error (MAE)**: Average difference between original "
        "and protected values. Lower is better."
    ),
    "rmse": (
        "**Root Mean Square Error (RMSE)**: Similar to MAE but penalizes "
        "larger errors more heavily. Lower is better."
    ),
    "correlation_preservation": (
        "**Correlation Preservation**: How well relationships between columns "
        "are maintained after adding noise.\n\n"
        "A value of 100% means correlations are perfectly preserved."
    ),
}


def show_help(topic: str, *, inline: bool = False) -> None:
    """Display help text for a topic.

    Args:
        topic: Help topic key from HELP_TEXTS.
        inline: If True, show inline instead of expander.
    """
    text = HELP_TEXTS.get(topic, f"No help available for '{topic}'")

    if inline:
        st.markdown(text)
    else:
        with st.expander("Help", expanded=False):
            st.markdown(text)


def help_tooltip(topic: str) -> str:
    """Get a short tooltip for a topic.

    Args:
        topic: Help topic key.

    Returns:
        Short tooltip text suitable for `help` parameter.
    """
    # Extract first line of help text as tooltip
    full_text = HELP_TEXTS.get(topic, "")
    first_line = full_text.split("\n")[0] if full_text else ""
    return first_line.replace("**", "").replace("*", "")


def show_help_icon(topic: str) -> None:
    """Display a help icon that expands to show help text.

    Args:
        topic: Help topic key.
    """
    text = HELP_TEXTS.get(topic, "")
    if text:
        st.markdown(
            f'<span title="{help_tooltip(topic)}" style="cursor: help;">ℹ️</span>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Input Validation
# =============================================================================


def validate_epsilon(value: float) -> Tuple[bool, Optional[str]]:
    """Validate epsilon value.

    Args:
        value: Epsilon value to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if value <= 0:
        return False, "Epsilon must be positive"
    if value > 100:
        return False, "Epsilon too high (max: 100)"
    return True, None


def validate_file_size(size_bytes: int, max_mb: int = 200) -> Tuple[bool, Optional[str]]:
    """Validate file size.

    Args:
        size_bytes: File size in bytes.
        max_mb: Maximum size in MB.

    Returns:
        Tuple of (is_valid, error_message).
    """
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_mb:
        return False, ErrorMessages.FILE_TOO_LARGE.format(
            size=size_mb,
            max_size=max_mb,
        )
    return True, None


# =============================================================================
# Edge Case Handlers
# =============================================================================


def safe_metric(
    label: str,
    value: Any,
    *,
    delta: Optional[Any] = None,
    help: Optional[str] = None,
    fallback: str = "N/A",
) -> None:
    """Display a metric with safe handling for None/NaN values.

    Args:
        label: Metric label.
        value: Metric value.
        delta: Optional delta value.
        help: Optional help text.
        fallback: Value to display if value is None/NaN.
    """
    import math

    # Check for None or NaN
    display_value = fallback
    if value is not None:
        try:
            if not math.isnan(float(value)):
                display_value = value
        except (TypeError, ValueError):
            display_value = value

    if delta is not None:
        try:
            if math.isnan(float(delta)):
                delta = None
        except (TypeError, ValueError):
            pass

    st.metric(label, display_value, delta=delta, help=help)


def safe_dataframe(
    df: Any,
    *,
    empty_message: str = "No data to display",
    **kwargs: Any,
) -> None:
    """Display a dataframe with safe handling for empty data.

    Args:
        df: DataFrame to display.
        empty_message: Message to show if empty.
        **kwargs: Additional arguments for st.dataframe.
    """
    if df is None or len(df) == 0:
        st.info(empty_message)
        return

    st.dataframe(df, **kwargs)


def safe_chart(
    chart_func: Callable[..., Any],
    *args: Any,
    error_message: str = "Could not create chart",
    **kwargs: Any,
) -> None:
    """Display a chart with safe error handling.

    Args:
        chart_func: Function to create the chart.
        *args: Arguments for the chart function.
        error_message: Message to show on error.
        **kwargs: Keyword arguments for the chart function.
    """
    try:
        fig = chart_func(*args, **kwargs)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"{error_message}: {str(e)}")


# =============================================================================
# Confirmation Dialogs
# =============================================================================


def confirm_action(
    message: str,
    *,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    key: str = "confirm_dialog",
) -> Optional[bool]:
    """Display a confirmation dialog.

    Args:
        message: Confirmation message.
        confirm_label: Label for confirm button.
        cancel_label: Label for cancel button.
        key: Unique key for the dialog state.

    Returns:
        True if confirmed, False if cancelled, None if pending.
    """
    st.warning(message)

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button(confirm_label, key=f"{key}_confirm", type="primary"):
            return True

    with col2:
        if st.button(cancel_label, key=f"{key}_cancel"):
            return False

    return None


# =============================================================================
# Session State Utilities
# =============================================================================


def require_session_data(
    *keys: str,
    error_redirect: Optional[int] = None,
) -> bool:
    """Check that required session data exists.

    Args:
        *keys: Session state keys to check.
        error_redirect: Step to redirect to on error.

    Returns:
        True if all keys exist and are not None.
    """
    missing = [k for k in keys if not st.session_state.get(k)]

    if missing:
        st.error(ErrorMessages.SESSION_EXPIRED)
        if error_redirect is not None:
            if st.button(f"Go to Step {error_redirect}"):
                st.session_state.current_step = error_redirect
                st.rerun()
        return False

    return True


# =============================================================================
# Accessibility Helpers
# =============================================================================


def screen_reader_text(text: str) -> None:
    """Add text visible only to screen readers.

    Args:
        text: Text for screen readers.
    """
    st.markdown(
        f'<span class="sr-only" style="position: absolute; left: -10000px; '
        f'top: auto; width: 1px; height: 1px; overflow: hidden;">{text}</span>',
        unsafe_allow_html=True,
    )


def aria_label(element_id: str, label: str) -> None:
    """Add ARIA label via JavaScript injection.

    Args:
        element_id: Element ID to label.
        label: ARIA label text.
    """
    st.markdown(
        f"""
        <script>
            document.getElementById("{element_id}")?.setAttribute("aria-label", "{label}");
        </script>
        """,
        unsafe_allow_html=True,
    )
