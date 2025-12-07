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
    # ==========================================================================
    # Privacy Concepts - Core Understanding
    # ==========================================================================
    "epsilon": (
        "## What is Epsilon (ε)?\n\n"
        "Epsilon is the **privacy budget** - the most important parameter in "
        "differential privacy. It controls how much information about any "
        "individual could potentially leak from your data.\n\n"
        "### How to interpret epsilon:\n"
        "| Epsilon | Privacy Level | What it means |\n"
        "|---------|--------------|---------------|\n"
        "| 0.1-0.5 | **Very High Privacy** | Almost impossible to learn about individuals. Data may be quite noisy. |\n"
        "| 0.5-1.0 | **High Privacy** | Strong protection. Good for sensitive healthcare data. |\n"
        "| 1.0-2.0 | **Moderate Privacy** | Balanced tradeoff. Suitable for most use cases. |\n"
        "| 2.0-5.0 | **Lower Privacy** | More accurate data, but less individual protection. |\n"
        "| 5.0-10.0 | **Minimal Privacy** | Data stays close to original. Limited privacy guarantee. |\n\n"
        "### Why it matters:\n"
        "- **For HIPAA/GDPR compliance**: Use ε ≤ 1.0 for sensitive healthcare data\n"
        "- **For research sharing**: ε = 1.0-2.0 often provides good utility\n"
        "- **For internal analysis**: Higher ε may be acceptable\n\n"
        "### The key tradeoff:\n"
        "**Lower ε = More Privacy = More Noise = Less Accurate Data**\n\n"
        "Think of it like blurring a photo - more blur (lower ε) makes it "
        "harder to identify individuals, but also harder to see details."
    ),
    "delta": (
        "## What is Delta (δ)?\n\n"
        "Delta is an additional privacy parameter used with the Gaussian mechanism. "
        "It represents the probability that the privacy guarantee could fail.\n\n"
        "### Typical values:\n"
        "- **1e-5 (0.00001)**: Good default for most datasets\n"
        "- **1e-6 (0.000001)**: Stronger guarantee for sensitive data\n"
        "- Should always be smaller than 1/n where n is your dataset size\n\n"
        "### Why it matters:\n"
        "A smaller delta means stronger privacy, but requires slightly more noise. "
        "For most practical purposes, the default value works well."
    ),
    "mechanism": (
        "## Differential Privacy Mechanisms\n\n"
        "The mechanism determines *how* noise is added to protect your data. "
        "Different mechanisms work better for different types of data.\n\n"
        "### Laplace Mechanism\n"
        "**Best for:** Numeric data with known bounds (ages, counts, scores)\n"
        "- Adds noise from a Laplace distribution\n"
        "- Provides pure ε-differential privacy\n"
        "- Simple and widely used\n\n"
        "### Gaussian Mechanism\n"
        "**Best for:** Numeric data without strict bounds\n"
        "- Adds noise from a Normal (Gaussian) distribution\n"
        "- Provides (ε,δ)-differential privacy\n"
        "- Can produce tighter results for high-dimensional data\n\n"
        "### Exponential Mechanism\n"
        "**Best for:** Categorical data (diagnoses, departments, categories)\n"
        "- Randomly selects categories based on their frequency\n"
        "- Preserves the overall distribution shape\n"
        "- Essential for non-numeric data\n\n"
        "### Recommendation:\n"
        "Use **Auto-Recommend** to let the system choose the best mechanism "
        "for each column based on its data type."
    ),
    "sensitivity": (
        "## What is Sensitivity?\n\n"
        "Sensitivity measures how much one person's data could affect the results. "
        "It determines how much noise is needed to hide individual contributions.\n\n"
        "### Sensitivity Levels:\n\n"
        "**High Sensitivity** (Red badge)\n"
        "- Direct identifiers: SSN, name, email, phone\n"
        "- Unique characteristics: Exact address, medical record number\n"
        "- **Recommendation**: Exclude or use very low epsilon (ε ≤ 0.5)\n\n"
        "**Medium Sensitivity** (Yellow badge)\n"
        "- Quasi-identifiers: Age, ZIP code, birth date\n"
        "- Health information: Diagnosis codes, procedures\n"
        "- **Recommendation**: Use moderate epsilon (ε = 0.5-2.0)\n\n"
        "**Low Sensitivity** (Green badge)\n"
        "- Aggregate data: Counts, averages\n"
        "- Common attributes: Gender, broad categories\n"
        "- **Recommendation**: Can use higher epsilon (ε = 2.0-5.0)\n\n"
        "### Why it matters:\n"
        "Properly classifying sensitivity helps you allocate your privacy budget "
        "wisely - more protection where it's needed most."
    ),
    "total_epsilon": (
        "## Total Privacy Budget\n\n"
        "The **total epsilon** is the sum of epsilon values across all protected "
        "columns. This represents your overall privacy expenditure.\n\n"
        "### Why it matters:\n"
        "In differential privacy, each query or column transformation 'spends' "
        "some of your privacy budget. The total tells you the cumulative "
        "privacy cost of your entire transformation.\n\n"
        "### Guidelines:\n"
        "- **Total ε < 5**: Generally considered good privacy practice\n"
        "- **Total ε 5-10**: Moderate privacy, acceptable for less sensitive data\n"
        "- **Total ε > 10**: Consider reducing epsilon per column or excluding more columns\n\n"
        "### Tips to reduce total epsilon:\n"
        "1. Exclude highly sensitive columns instead of protecting them\n"
        "2. Use passthrough for non-sensitive columns\n"
        "3. Lower individual column epsilon values"
    ),
    # ==========================================================================
    # Protection Modes - What to do with each column
    # ==========================================================================
    "protect": (
        "## Protect (Apply DP)\n\n"
        "This column will have **differential privacy noise added** to protect "
        "individual values while preserving statistical properties.\n\n"
        "### What happens:\n"
        "- Numeric columns: Random noise is added to each value\n"
        "- Categorical columns: Some values may be randomly changed\n"
        "- Overall distributions are approximately preserved\n\n"
        "### Use this when:\n"
        "- The column contains useful information for analysis\n"
        "- Individual values need protection\n"
        "- You want to keep the column in your output\n\n"
        "### Example:\n"
        "Age values like [25, 30, 35] might become [24.3, 31.2, 34.1] - "
        "the average and distribution stay similar, but exact values change."
    ),
    "passthrough": (
        "## Passthrough (Keep Original)\n\n"
        "This column will be included **exactly as-is** without any modification.\n\n"
        "### ⚠️ Important:\n"
        "No privacy protection is applied! Only use for columns that:\n"
        "- Don't contain sensitive information\n"
        "- Are required for data linkage (non-sensitive IDs)\n"
        "- Contain public or non-personal information\n\n"
        "### Good candidates for passthrough:\n"
        "- Record sequence numbers (not SSN or patient IDs!)\n"
        "- Timestamps for time-series analysis\n"
        "- Public reference codes\n\n"
        "### ⛔ Never passthrough:\n"
        "- Personal identifiers (names, SSN, email)\n"
        "- Exact dates of birth\n"
        "- Precise locations"
    ),
    "exclude": (
        "## Exclude (Remove)\n\n"
        "This column will be **completely removed** from the output dataset.\n\n"
        "### When to exclude:\n"
        "- Direct identifiers (SSN, name, email, phone)\n"
        "- Columns not needed for your analysis\n"
        "- Highly sensitive data that can't be adequately protected\n"
        "- Free-text fields that might contain identifying information\n\n"
        "### Benefits:\n"
        "- Zero privacy risk for excluded columns\n"
        "- Reduces total privacy budget needed\n"
        "- Simplifies the output dataset\n\n"
        "### This is often the safest choice for:\n"
        "- Patient names\n"
        "- Social Security Numbers\n"
        "- Medical record numbers\n"
        "- Contact information"
    ),
    # ==========================================================================
    # Data Quality Metrics - Understanding the Analysis Results
    # ==========================================================================
    "mae": (
        "## Mean Absolute Error (MAE)\n\n"
        "MAE measures the **average difference** between original and protected values.\n\n"
        "### How to interpret:\n"
        "- **MAE = 0**: Perfect match (no privacy protection)\n"
        "- **Lower MAE**: Protected data is closer to original\n"
        "- **Higher MAE**: More noise was added (stronger privacy)\n\n"
        "### Example:\n"
        "If original ages are [25, 30, 35] and protected are [24, 32, 34]:\n"
        "- Differences: |25-24|=1, |30-32|=2, |35-34|=1\n"
        "- MAE = (1+2+1)/3 = **1.33**\n\n"
        "### What's a good MAE?\n"
        "It depends on your data and analysis needs:\n"
        "- For ages: MAE of 2-5 years is often acceptable\n"
        "- For counts: MAE of 5-10% of the mean is typical\n"
        "- Compare to your analysis requirements"
    ),
    "rmse": (
        "## Root Mean Square Error (RMSE)\n\n"
        "RMSE is similar to MAE but **penalizes larger errors more heavily**.\n\n"
        "### How it works:\n"
        "1. Calculate squared differences\n"
        "2. Take the average\n"
        "3. Take the square root\n\n"
        "### Why use RMSE over MAE?\n"
        "- RMSE is more sensitive to outliers\n"
        "- If you see RMSE >> MAE, there are some large individual errors\n"
        "- If RMSE ≈ MAE, errors are consistent across values\n\n"
        "### Interpretation:\n"
        "- RMSE is in the same units as your data\n"
        "- Lower is better\n"
        "- Compare to the standard deviation of your original data"
    ),
    "correlation_preservation": (
        "## Correlation Preservation\n\n"
        "This measures how well **relationships between columns** are maintained "
        "after adding differential privacy noise.\n\n"
        "### Why it matters:\n"
        "Many analyses depend on relationships between variables:\n"
        "- Does age correlate with diagnosis?\n"
        "- Is income related to health outcomes?\n"
        "- Do certain departments have higher costs?\n\n"
        "### How to interpret:\n"
        "- **90-100%**: Excellent - correlations well preserved\n"
        "- **70-90%**: Good - most relationships intact\n"
        "- **50-70%**: Moderate - some relationships may be affected\n"
        "- **<50%**: Poor - consider using higher epsilon\n\n"
        "### If preservation is low:\n"
        "1. Increase epsilon for correlated columns\n"
        "2. Consider which correlations are most important\n"
        "3. Test if your specific analysis is affected"
    ),
    "mean_difference": (
        "## Mean Difference\n\n"
        "The difference between the **average value** in the original vs protected data.\n\n"
        "### Why it matters:\n"
        "- If you're reporting averages, this shows how close you'll be\n"
        "- Differential privacy is designed to preserve means well\n"
        "- Large differences suggest potential issues\n\n"
        "### Interpretation:\n"
        "- Small difference = good for average-based analyses\n"
        "- The sign (+/-) shows direction of change\n"
        "- Compare to standard deviation for context"
    ),
    "std_difference": (
        "## Standard Deviation Difference\n\n"
        "How much the **spread of values** changed after protection.\n\n"
        "### Why it matters:\n"
        "Adding noise typically increases variability slightly. This metric "
        "shows if your data spread has changed significantly.\n\n"
        "### Interpretation:\n"
        "- Small increase: Normal, expected\n"
        "- Large increase: Epsilon may be too low\n"
        "- Decrease: Unusual, check your data"
    ),
    "frequency_mae": (
        "## Frequency MAE (Categorical)\n\n"
        "For categorical columns, this measures how well the **distribution of "
        "categories** is preserved.\n\n"
        "### Example:\n"
        "If original: 40% Category A, 60% Category B\n"
        "And protected: 42% Category A, 58% Category B\n"
        "Frequency MAE = (|40-42| + |60-58|)/2 = **2%**\n\n"
        "### Good values:\n"
        "- <5%: Excellent preservation\n"
        "- 5-10%: Good for most analyses\n"
        "- >10%: May affect category-based analyses"
    ),
    "category_drift": (
        "## Category Drift\n\n"
        "Measures how much individual records **changed categories** during protection.\n\n"
        "### Why it matters:\n"
        "Even if overall proportions are similar, individual values may have changed. "
        "This affects analyses that depend on specific category combinations.\n\n"
        "### Interpretation:\n"
        "- 0%: No values changed (unlikely with DP)\n"
        "- 10-20%: Low drift, good preservation\n"
        "- 20-40%: Moderate drift\n"
        "- >40%: High drift, consider higher epsilon"
    ),
    # ==========================================================================
    # Dataset Information
    # ==========================================================================
    "row_count": (
        "## Row Count\n\n"
        "The number of records (rows) in your dataset.\n\n"
        "### Why it matters for privacy:\n"
        "- Larger datasets generally allow for more accurate results with the same privacy\n"
        "- Very small datasets (<100 rows) may have limited utility after DP\n"
        "- Consider if your sample size is sufficient for your analysis"
    ),
    "column_count": (
        "## Column Count\n\n"
        "The number of variables (columns) in your dataset.\n\n"
        "### Privacy implications:\n"
        "- More protected columns = higher total privacy budget\n"
        "- Consider excluding columns not needed for analysis\n"
        "- Each column's epsilon contributes to the total"
    ),
    "missing_values": (
        "## Missing Values\n\n"
        "Percentage of cells that contain null/empty values.\n\n"
        "### Why it matters:\n"
        "- Null values are preserved during DP transformation\n"
        "- High missing rates may affect analysis quality\n"
        "- Consider imputation before applying DP if appropriate"
    ),
    "column_types": (
        "## Column Types\n\n"
        "How each column is classified for differential privacy:\n\n"
        "- **Numeric (🔢)**: Continuous values like age, income, measurements\n"
        "- **Categorical (📋)**: Discrete categories like gender, diagnosis, department\n"
        "- **Date (📅)**: Temporal values (converted to numeric for DP)\n"
        "- **Unknown (❓)**: Type couldn't be determined - please review\n\n"
        "### Why it matters:\n"
        "The type determines which DP mechanism is used. Misclassification "
        "can lead to poor results."
    ),
    # ==========================================================================
    # Export Information
    # ==========================================================================
    "export_format": (
        "## Export Formats\n\n"
        "Choose the format that works best for your workflow:\n\n"
        "### CSV\n"
        "- Universal compatibility\n"
        "- Human-readable\n"
        "- Best for sharing and simple analysis\n\n"
        "### Excel\n"
        "- Includes metadata sheet with privacy settings\n"
        "- Good for documentation and audit trails\n"
        "- Compatible with spreadsheet software\n\n"
        "### Parquet\n"
        "- Best compression and performance\n"
        "- Preserves data types exactly\n"
        "- Ideal for large datasets and data pipelines"
    ),
    "pdf_report": (
        "## PDF Privacy Report\n\n"
        "Generate a comprehensive report documenting:\n\n"
        "- **Executive Summary**: Privacy settings and key findings\n"
        "- **Configuration Details**: What was done to each column\n"
        "- **Statistical Comparison**: How the data changed\n"
        "- **Visualizations**: Charts showing before/after\n\n"
        "### Use this report for:\n"
        "- Compliance documentation (HIPAA, GDPR)\n"
        "- Sharing methodology with collaborators\n"
        "- Audit trails\n"
        "- Institutional review board (IRB) submissions"
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


def info_button(
    topic: str,
    label: str = "What does this mean?",
    *,
    expanded: bool = False,
    icon: str = "ℹ️",
) -> None:
    """Display an info button/expander for a topic.

    Args:
        topic: Help topic key from HELP_TEXTS.
        label: Label for the expander.
        expanded: Whether to show expanded by default.
        icon: Icon to show next to label.
    """
    text = HELP_TEXTS.get(topic, f"No explanation available for '{topic}'")
    with st.expander(f"{icon} {label}", expanded=expanded):
        st.markdown(text)


def metric_with_info(
    label: str,
    value: Any,
    topic: str,
    *,
    delta: Optional[Any] = None,
    help: Optional[str] = None,
) -> None:
    """Display a metric with an info expander below it.

    Args:
        label: Metric label.
        value: Metric value.
        topic: Help topic key for explanation.
        delta: Optional delta value.
        help: Optional short tooltip.
    """
    st.metric(label, value, delta=delta, help=help)
    text = HELP_TEXTS.get(topic)
    if text:
        with st.expander(f"ℹ️ What is {label}?", expanded=False):
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
