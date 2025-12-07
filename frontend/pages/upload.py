"""Upload page for DPtoolkit.

This page handles file upload, preview, and validation before
proceeding to the configuration step.
"""

import io
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from dp_toolkit.data.loader import ColumnInfo, ColumnType, DatasetInfo

# Import UI utilities
try:
    from utils.ui_components import (
        ErrorMessages,
        show_error,
        show_warning,
        show_info,
        show_success,
        loading_state,
        HELP_TEXTS,
    )
except ImportError:
    # Fallback if running standalone
    ErrorMessages = None  # type: ignore


# =============================================================================
# Constants
# =============================================================================

MAX_PREVIEW_ROWS = 100
MAX_FILE_SIZE_MB = 200
SUPPORTED_FORMATS = {
    "csv": [".csv", ".tsv", ".txt"],
    "excel": [".xlsx", ".xls"],
    "parquet": [".parquet", ".pq"],
}


# =============================================================================
# File Loading Utilities
# =============================================================================


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension from filename."""
    if "." in filename:
        return "." + filename.rsplit(".", 1)[1].lower()
    return ""


def detect_format(filename: str) -> Optional[str]:
    """Detect file format from filename extension."""
    ext = get_file_extension(filename)
    for fmt, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return fmt
    return None


def load_uploaded_file(
    uploaded_file,
    file_format: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load an uploaded file into a DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        file_format: Detected format ('csv', 'excel', 'parquet').

    Returns:
        Tuple of (DataFrame, error_message). DataFrame is None if error.
    """
    try:
        if file_format == "csv":
            # Try common encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df, None
                except UnicodeDecodeError:
                    continue
            return None, "Could not decode file with any supported encoding"

        elif file_format == "excel":
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return df, None

        elif file_format == "parquet":
            uploaded_file.seek(0)
            df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
            return df, None

        else:
            return None, f"Unsupported format: {file_format}"

    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def analyze_column(series: pd.Series, column_name: str) -> ColumnInfo:
    """Analyze a column and return ColumnInfo."""
    dtype_str = str(series.dtype)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique())

    # Sample values
    non_null = series.dropna()
    sample_values = non_null.unique()[:5].tolist() if len(non_null) > 0 else []

    # Detect type
    column_type = detect_column_type(series, column_name)

    return ColumnInfo(
        name=column_name,
        dtype=dtype_str,
        column_type=column_type,
        null_count=null_count,
        unique_count=unique_count,
        sample_values=sample_values,
    )


def detect_column_type(series: pd.Series, column_name: str) -> ColumnType:
    """Detect the DP-relevant type of a column."""
    # Date keywords
    date_keywords = ["date", "time", "timestamp", "created", "updated", "dob"]

    # Check datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATE

    # Check column name for date hints
    col_lower = column_name.lower()
    if any(kw in col_lower for kw in date_keywords):
        non_null = series.dropna()
        if len(non_null) > 0:
            try:
                parsed = pd.to_datetime(non_null.head(100), errors="coerce")
                if parsed.notna().sum() / len(parsed) >= 0.8:
                    return ColumnType.DATE
            except Exception:
                pass

    # Boolean -> Categorical
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.CATEGORICAL

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            unique_count = series.nunique()
            unique_ratio = unique_count / non_null_count
            if unique_ratio < 0.05 and unique_count <= 100 and non_null_count >= 20:
                return ColumnType.CATEGORICAL
            if unique_count <= 5 and non_null_count >= 50:
                return ColumnType.CATEGORICAL
        return ColumnType.NUMERIC

    # String/Object -> Categorical
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return ColumnType.CATEGORICAL

    if pd.api.types.is_categorical_dtype(series):
        return ColumnType.CATEGORICAL

    return ColumnType.UNKNOWN


def create_dataset_info(
    df: pd.DataFrame,
    filename: str,
    file_format: str,
) -> DatasetInfo:
    """Create a DatasetInfo object from a DataFrame."""
    columns = [analyze_column(df[col], col) for col in df.columns]

    return DatasetInfo(
        dataframe=df,
        file_path=None,
        file_format=file_format,
        encoding=None,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        memory_usage_bytes=df.memory_usage(deep=True).sum(),
    )


# =============================================================================
# UI Components
# =============================================================================


def render_file_uploader() -> Any:
    """Render the file upload widget."""
    st.markdown("### Select Your Dataset")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Supported formats: CSV, Excel (.xlsx, .xls), Parquet",
    )

    st.info(
        "**Privacy Note:** Your data is processed in memory only and "
        "never written to disk. Session data is cleared on close."
    )

    return uploaded_file


def render_file_info(uploaded_file: Any) -> None:
    """Display basic file information."""
    file_size_mb = uploaded_file.size / (1024 * 1024)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{file_size_mb:.2f} MB")
    with col3:
        fmt = detect_format(uploaded_file.name) or "Unknown"
        st.metric("Format", fmt.upper())


def render_dataset_preview(df: pd.DataFrame) -> None:
    """Render a preview of the dataset."""
    st.markdown("### Data Preview")

    # Show first N rows
    preview_rows = min(MAX_PREVIEW_ROWS, len(df))
    st.dataframe(
        df.head(preview_rows),
        width="stretch",
        height=400,
    )

    if len(df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing first {MAX_PREVIEW_ROWS} of {len(df):,} rows")


def render_dataset_summary(dataset_info: DatasetInfo) -> None:
    """Render dataset summary statistics."""
    st.markdown("### Dataset Summary")

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{dataset_info.row_count:,}")
    with col2:
        st.metric("Columns", dataset_info.column_count)
    with col3:
        mem_mb = dataset_info.memory_usage_bytes / (1024 * 1024)
        st.metric("Memory", f"{mem_mb:.2f} MB")
    with col4:
        null_total = sum(c.null_count for c in dataset_info.columns)
        total_cells = dataset_info.row_count * dataset_info.column_count
        null_pct = (null_total / total_cells * 100) if total_cells > 0 else 0
        st.metric("Missing Values", f"{null_pct:.1f}%")


def render_column_summary(dataset_info: DatasetInfo) -> None:
    """Render per-column summary."""
    st.markdown("### Column Details")

    # Create summary table
    col_data = []
    for col in dataset_info.columns:
        col_data.append({
            "Column": col.name,
            "Type": col.column_type.value.capitalize(),
            "Data Type": col.dtype,
            "Unique": col.unique_count,
            "Missing": col.null_count,
            "Missing %": f"{col.null_count / dataset_info.row_count * 100:.1f}%"
            if dataset_info.row_count > 0 else "0%",
        })

    col_df = pd.DataFrame(col_data)

    # Color-code by type
    st.dataframe(
        col_df,
        width="stretch",
        hide_index=True,
    )

    # Type distribution
    st.markdown("#### Column Types")
    type_counts: Dict[str, int] = {}
    for col in dataset_info.columns:
        t = col.column_type.value.capitalize()
        type_counts[t] = type_counts.get(t, 0) + 1

    cols = st.columns(len(type_counts))
    for i, (type_name, count) in enumerate(type_counts.items()):
        with cols[i]:
            st.metric(type_name, count)


def render_validation_warnings(dataset_info: DatasetInfo) -> bool:
    """Check and display any validation warnings. Returns True if valid."""
    warnings = []
    errors = []

    # Check for empty dataset
    if dataset_info.row_count == 0:
        errors.append("Dataset is empty (0 rows)")

    if dataset_info.column_count == 0:
        errors.append("Dataset has no columns")

    # Check for high missing value columns
    for col in dataset_info.columns:
        if dataset_info.row_count > 0:
            missing_pct = col.null_count / dataset_info.row_count * 100
            if missing_pct > 90:
                warnings.append(
                    f"Column '{col.name}' has {missing_pct:.0f}% missing values"
                )

    # Check for unknown types
    unknown_cols = [c.name for c in dataset_info.columns
                    if c.column_type == ColumnType.UNKNOWN]
    if unknown_cols:
        warnings.append(
            f"Could not detect type for: {', '.join(unknown_cols[:3])}"
            + (f" and {len(unknown_cols) - 3} more" if len(unknown_cols) > 3 else "")
        )

    # Display errors
    for error in errors:
        st.error(error)

    # Display warnings
    for warning in warnings:
        st.warning(warning)

    return len(errors) == 0


# =============================================================================
# Main Page Function
# =============================================================================


def render_upload_page() -> None:
    """Render the complete upload page."""
    st.header("Step 1: Upload Data")
    st.markdown(
        "Upload your dataset to apply differential privacy protection. "
        "Supported formats include CSV, Excel, and Parquet files."
    )

    # File uploader
    uploaded_file = render_file_uploader()

    if uploaded_file is None:
        # Show placeholder when no file
        st.markdown("---")
        st.markdown(
            "**Getting Started:**\n\n"
            "1. Upload a dataset file (CSV, Excel, or Parquet)\n"
            "2. Review the data preview and column types\n"
            "3. Click 'Continue' to configure privacy settings"
        )
        return

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        if ErrorMessages:
            st.error(ErrorMessages.FILE_TOO_LARGE.format(
                size=file_size_mb,
                max_size=MAX_FILE_SIZE_MB,
            ))
        else:
            st.error(f"File too large ({file_size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB")
        return

    st.markdown("---")

    # Show file info
    render_file_info(uploaded_file)

    # Detect format
    file_format = detect_format(uploaded_file.name)
    if file_format is None:
        ext = get_file_extension(uploaded_file.name)
        if ErrorMessages:
            st.error(ErrorMessages.FILE_FORMAT_UNSUPPORTED.format(extension=ext))
        else:
            st.error(f"Unsupported file format: {ext}")
        return

    # Load file with progress
    with st.spinner("Loading dataset..."):
        df, error = load_uploaded_file(uploaded_file, file_format)

    if error:
        st.error(error)
        return

    if df is None:
        st.error("Failed to load dataset")
        return

    # Create dataset info
    dataset_info = create_dataset_info(df, uploaded_file.name, file_format)

    st.markdown("---")

    # Show preview
    render_dataset_preview(df)

    st.markdown("---")

    # Show summary
    render_dataset_summary(dataset_info)

    st.markdown("---")

    # Show column details
    render_column_summary(dataset_info)

    st.markdown("---")

    # Validation
    is_valid = render_validation_warnings(dataset_info)

    # Continue button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Continue to Configure",
            type="primary",
            disabled=not is_valid,
            width="stretch",
        ):
            # Store in session state
            st.session_state.original_df = df
            st.session_state.dataset_info = {
                "filename": uploaded_file.name,
                "file_format": file_format,
                "row_count": dataset_info.row_count,
                "column_count": dataset_info.column_count,
                "columns": [col.name for col in dataset_info.columns],
                "column_types": {
                    col.name: col.column_type.value
                    for col in dataset_info.columns
                },
                "dtypes": {col.name: col.dtype for col in dataset_info.columns},
                "memory_mb": dataset_info.memory_usage_bytes / (1024 * 1024),
            }
            st.session_state.dataset_loaded = True
            st.session_state.current_step = 2
            st.rerun()
