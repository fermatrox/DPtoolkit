"""Export page for DPtoolkit.

This page handles exporting the protected dataset to various formats
and displays a summary of the privacy configuration applied.
"""

import io
from datetime import datetime
from typing import Any, Dict, Optional, cast

import pandas as pd
import streamlit as st

from utils.session import clear_session_data
from dp_toolkit.reports import PDFReportGenerator, ReportMetadata
from dp_toolkit.analysis.comparator import DatasetComparison


# =============================================================================
# Session State Helpers
# =============================================================================


def get_protected_df() -> Optional[pd.DataFrame]:
    """Get the protected DataFrame from session state."""
    return st.session_state.get("protected_df")


def get_original_df() -> Optional[pd.DataFrame]:
    """Get the original DataFrame from session state."""
    return st.session_state.get("original_df")


def get_column_configs() -> Dict[str, Dict[str, Any]]:
    """Get column configurations from session state."""
    return cast(Dict[str, Dict[str, Any]], st.session_state.get("column_configs", {}))


def get_dataset_info() -> Dict[str, Any]:
    """Get dataset info from session state."""
    return cast(Dict[str, Any], st.session_state.get("dataset_info", {}))


def get_comparison_results() -> Optional[DatasetComparison]:
    """Get comparison results from session state."""
    return st.session_state.get("comparison_results")


# =============================================================================
# Export Functions
# =============================================================================


def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    """Generate CSV file as bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def generate_excel_bytes(
    df: pd.DataFrame,
    include_metadata: bool = True,
    column_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate Excel file as bytes with optional metadata sheet."""
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Write data sheet
        df.to_excel(writer, sheet_name="Protected Data", index=False)

        # Write metadata sheet if requested
        if include_metadata and column_configs:
            metadata_rows = []

            # Dataset info
            if dataset_info:
                metadata_rows.append({
                    "Category": "Dataset",
                    "Property": "Original Filename",
                    "Value": dataset_info.get("filename", "N/A"),
                })
                metadata_rows.append({
                    "Category": "Dataset",
                    "Property": "Original Rows",
                    "Value": str(dataset_info.get("row_count", "N/A")),
                })
                metadata_rows.append({
                    "Category": "Dataset",
                    "Property": "Original Columns",
                    "Value": str(dataset_info.get("column_count", "N/A")),
                })

            # Export info
            metadata_rows.append({
                "Category": "Export",
                "Property": "Export Date",
                "Value": datetime.now().isoformat(),
            })
            metadata_rows.append({
                "Category": "Export",
                "Property": "Protected Rows",
                "Value": str(len(df)),
            })
            metadata_rows.append({
                "Category": "Export",
                "Property": "Protected Columns",
                "Value": str(len(df.columns)),
            })

            # Privacy budget
            total_eps = sum(
                cfg.get("epsilon", 0)
                for cfg in column_configs.values()
                if cfg.get("mode") == "protect"
            )
            metadata_rows.append({
                "Category": "Privacy",
                "Property": "Total Epsilon",
                "Value": f"{total_eps:.4f}",
            })

            # Column configurations
            for col, cfg in column_configs.items():
                mode = cfg.get("mode", "unknown")
                if mode == "protect":
                    metadata_rows.append({
                        "Category": "Column Config",
                        "Property": col,
                        "Value": f"Protected (ε={cfg.get('epsilon', 'N/A')}, "
                                 f"mechanism={cfg.get('mechanism', 'N/A')})",
                    })
                elif mode == "passthrough":
                    metadata_rows.append({
                        "Category": "Column Config",
                        "Property": col,
                        "Value": "Passthrough (unchanged)",
                    })
                elif mode == "exclude":
                    metadata_rows.append({
                        "Category": "Column Config",
                        "Property": col,
                        "Value": "Excluded (removed)",
                    })

            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

    return buffer.getvalue()


def generate_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Generate Parquet file as bytes."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="snappy")
    return buffer.getvalue()


# =============================================================================
# UI Components
# =============================================================================


def render_configuration_summary(
    column_configs: Dict[str, Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> None:
    """Render a summary of the privacy configuration applied."""
    st.markdown("### Configuration Summary")

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Rows", f"{dataset_info.get('row_count', 0):,}")

    with col2:
        st.metric("Original Columns", dataset_info.get("column_count", 0))

    with col3:
        # Count protected columns
        protected_count = sum(
            1 for cfg in column_configs.values()
            if cfg.get("mode") == "protect"
        )
        st.metric("Protected Columns", protected_count)

    with col4:
        # Total epsilon
        total_eps = sum(
            cfg.get("epsilon", 0)
            for cfg in column_configs.values()
            if cfg.get("mode") == "protect"
        )
        st.metric("Total ε", f"{total_eps:.2f}")

    st.markdown("---")

    # Column details
    st.markdown("#### Column Protection Details")

    # Group by mode
    protected = []
    passthrough = []
    excluded = []

    for col, cfg in column_configs.items():
        mode = cfg.get("mode", "unknown")
        if mode == "protect":
            protected.append({
                "Column": col,
                "Epsilon": cfg.get("epsilon", "N/A"),
                "Mechanism": cfg.get("mechanism", "N/A"),
                "Sensitivity": cfg.get("sensitivity", "N/A"),
            })
        elif mode == "passthrough":
            passthrough.append(col)
        elif mode == "exclude":
            excluded.append(col)

    # Show protected columns table
    if protected:
        st.markdown("**Protected Columns (DP Applied):**")
        st.dataframe(
            pd.DataFrame(protected),
            hide_index=True,
            width="stretch",
        )

    # Show passthrough columns
    if passthrough:
        st.markdown("**Passthrough Columns (Unchanged):**")
        st.markdown(", ".join(f"`{c}`" for c in passthrough))

    # Show excluded columns
    if excluded:
        st.markdown("**Excluded Columns (Removed):**")
        st.markdown(", ".join(f"`{c}`" for c in excluded))


def render_export_options(
    protected_df: pd.DataFrame,
    column_configs: Dict[str, Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> None:
    """Render export format options and download buttons."""
    st.markdown("### Export Protected Dataset")

    # Format selection
    col1, col2 = st.columns([1, 2])

    with col1:
        export_format: str = st.selectbox(
            "Export Format",
            options=["CSV", "Excel", "Parquet"],
            help="Choose the output format for your protected dataset",
        ) or "CSV"

    with col2:
        include_metadata = st.checkbox(
            "Include metadata",
            value=True,
            help="Include privacy configuration metadata with the export",
        )

    st.markdown("---")

    # Generate filename
    original_name = dataset_info.get("filename", "dataset")
    if "." in original_name:
        base_name = original_name.rsplit(".", 1)[0]
    else:
        base_name = original_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format-specific export
    if export_format == "CSV":
        filename = f"{base_name}_protected_{timestamp}.csv"
        file_bytes = generate_csv_bytes(protected_df)
        mime_type = "text/csv"

        if include_metadata:
            st.info(
                "For CSV exports, metadata will be displayed below. "
                "Consider Excel format for embedded metadata."
            )

    elif export_format == "Excel":
        filename = f"{base_name}_protected_{timestamp}.xlsx"
        file_bytes = generate_excel_bytes(
            protected_df,
            include_metadata=include_metadata,
            column_configs=column_configs,
            dataset_info=dataset_info,
        )
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    else:  # Parquet
        filename = f"{base_name}_protected_{timestamp}.parquet"
        file_bytes = generate_parquet_bytes(protected_df)
        mime_type = "application/octet-stream"

        if include_metadata:
            st.info(
                "Parquet metadata is embedded in the file schema. "
                "Use a Parquet reader to access it."
            )

    # File info
    file_size_kb = len(file_bytes) / 1024
    file_size_str = (
        f"{file_size_kb:.1f} KB" if file_size_kb < 1024
        else f"{file_size_kb / 1024:.2f} MB"
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Filename:** `{filename}`")

    with col2:
        st.markdown(f"**Size:** {file_size_str}")

    with col3:
        st.markdown(f"**Rows:** {len(protected_df):,}")

    # Download button
    st.download_button(
        label=f"Download {export_format}",
        data=file_bytes,
        file_name=filename,
        mime=mime_type,
        type="primary",
    )


def generate_pdf_bytes(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    comparison: DatasetComparison,
    column_configs: Dict[str, Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> bytes:
    """Generate PDF report as bytes."""
    metadata = ReportMetadata(
        title="Differential Privacy Analysis Report",
        original_filename=dataset_info.get("filename"),
    )

    generator = PDFReportGenerator()
    report = generator.generate(
        original_df=original_df,
        protected_df=protected_df,
        comparison=comparison,
        column_configs=column_configs,
        metadata=metadata,
        dataset_info=dataset_info,
    )

    return report.content


def render_pdf_report_section(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    column_configs: Dict[str, Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> None:
    """Render PDF report generation section."""
    st.markdown("### Generate Report")

    comparison = get_comparison_results()

    if comparison is None:
        st.warning(
            "Comparison results not available. "
            "Please run the analysis again to generate a PDF report."
        )
        return

    st.markdown(
        "Generate a comprehensive PDF report including:\n"
        "- Executive summary with privacy assessment\n"
        "- Configuration details per column\n"
        "- Statistical comparison tables\n"
        "- Distribution and correlation visualizations"
    )

    # Generate filename
    original_name = dataset_info.get("filename", "dataset")
    if "." in original_name:
        base_name = original_name.rsplit(".", 1)[0]
    else:
        base_name = original_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{base_name}_privacy_report_{timestamp}.pdf"

    # Generate PDF button
    if st.button("Generate PDF Report", type="secondary"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_bytes(
                    original_df=original_df,
                    protected_df=protected_df,
                    comparison=comparison,
                    column_configs=column_configs,
                    dataset_info=dataset_info,
                )
                st.session_state["pdf_report"] = pdf_bytes
                st.session_state["pdf_filename"] = pdf_filename
                st.success("PDF report generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                return

    # Download button (if PDF was generated)
    if "pdf_report" in st.session_state:
        pdf_bytes = st.session_state["pdf_report"]
        filename = st.session_state.get("pdf_filename", "report.pdf")

        file_size_kb = len(pdf_bytes) / 1024
        st.markdown(f"**Report size:** {file_size_kb:.1f} KB")

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )


def render_session_actions() -> None:
    """Render session management actions."""
    st.markdown("---")
    st.markdown("### Session Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("← Back to Analysis"):
            st.session_state.current_step = 3
            st.rerun()

    with col2:
        if st.button("Start New Session"):
            clear_session_data()
            st.rerun()

    with col3:
        st.markdown("")  # Spacer

    # Success message
    st.markdown("---")
    st.success(
        "Your protected dataset is ready for download. "
        "The differential privacy transformation has been applied according to "
        "your configuration."
    )


# =============================================================================
# Main Page Function
# =============================================================================


def render_export_page() -> None:
    """Render the complete export page."""
    st.header("Step 4: Export Results")

    # Check prerequisites
    if not st.session_state.get("transform_complete", False):
        st.warning("Please complete the analysis first.")
        if st.button("Go to Analyze"):
            st.session_state.current_step = 3
            st.rerun()
        return

    # Get data from session
    original_df = get_original_df()
    protected_df = get_protected_df()
    column_configs = get_column_configs()
    dataset_info = get_dataset_info()

    if protected_df is None or original_df is None:
        st.error(
            "Protected dataset not found. Please run the analysis again."
        )
        if st.button("Go to Analyze"):
            st.session_state.current_step = 3
            st.rerun()
        return

    st.markdown(
        "Download your privacy-protected dataset and review the "
        "configuration summary."
    )

    st.markdown("---")

    # Configuration summary
    render_configuration_summary(column_configs, dataset_info)

    st.markdown("---")

    # Export options
    render_export_options(protected_df, column_configs, dataset_info)

    st.markdown("---")

    # PDF report section
    render_pdf_report_section(original_df, protected_df, column_configs, dataset_info)

    # Session actions
    render_session_actions()
