"""PDF report generator for differential privacy analysis.

This module generates compliance-ready PDF reports containing:
- Executive summary with privacy budget overview
- Configuration details per column
- Statistical comparison results
- Embedded visualizations

Example:
    >>> generator = PDFReportGenerator()
    >>> report = generator.generate(
    ...     original_df=original,
    ...     protected_df=protected,
    ...     comparison=comparison,
    ...     column_configs=configs,
    ... )
    >>> report.save("privacy_report.pdf")
"""

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from dp_toolkit.analysis.comparator import DatasetComparison  # noqa: E402


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PAGE_SIZE = A4
DEFAULT_MARGIN = 0.75 * inch
DEFAULT_TITLE = "Differential Privacy Analysis Report"
DEFAULT_DPI = 150
CHART_WIDTH = 6 * inch
CHART_HEIGHT = 3 * inch


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ReportMetadata:
    """Metadata for the PDF report.

    Attributes:
        title: Report title.
        author: Report author.
        subject: Report subject.
        generated_at: Generation timestamp.
        original_filename: Original dataset filename.
    """

    title: str = DEFAULT_TITLE
    author: str = "DPtoolkit"
    subject: str = "Differential Privacy Analysis"
    generated_at: datetime = field(default_factory=datetime.now)
    original_filename: Optional[str] = None


@dataclass
class ReportConfig:
    """Configuration for PDF report generation.

    Attributes:
        page_size: Page size (A4, letter, etc.).
        margins: Page margins (top, right, bottom, left).
        include_charts: Whether to include visualizations.
        include_raw_stats: Whether to include detailed statistics.
        max_columns_per_page: Maximum columns shown per page in tables.
        dpi: DPI for embedded images.
    """

    page_size: Tuple[float, float] = DEFAULT_PAGE_SIZE
    margins: Tuple[float, float, float, float] = (
        DEFAULT_MARGIN,
        DEFAULT_MARGIN,
        DEFAULT_MARGIN,
        DEFAULT_MARGIN,
    )
    include_charts: bool = True
    include_raw_stats: bool = True
    max_columns_per_page: int = 10
    dpi: int = DEFAULT_DPI


@dataclass
class GeneratedReport:
    """Result of PDF report generation.

    Attributes:
        content: PDF content as bytes.
        metadata: Report metadata.
        page_count: Number of pages.
        generation_time_ms: Time taken to generate in milliseconds.
    """

    content: bytes
    metadata: ReportMetadata
    page_count: int
    generation_time_ms: float

    def save(self, path: Union[str, Path]) -> Path:
        """Save the PDF to a file.

        Args:
            path: Output file path.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.write_bytes(self.content)
        return path


# =============================================================================
# Chart Generation (Matplotlib)
# =============================================================================


def create_histogram_chart(
    original: pd.Series,
    protected: pd.Series,
    column_name: str,
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Create a histogram comparison chart as PNG bytes.

    Args:
        original: Original column data.
        protected: Protected column data.
        column_name: Name of the column.
        dpi: Image resolution.

    Returns:
        PNG image as bytes.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)

    # Remove NaN values
    orig_clean = original.dropna()
    prot_clean = protected.dropna()

    if len(orig_clean) == 0 or len(prot_clean) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        # Determine bins
        all_data = pd.concat([orig_clean, prot_clean])
        bins = np.histogram_bin_edges(all_data, bins="auto")

        ax.hist(
            orig_clean, bins=bins, alpha=0.5, label="Original",
            color="#1f77b4", edgecolor="white"
        )
        ax.hist(
            prot_clean, bins=bins, alpha=0.5, label="Protected",
            color="#ff7f0e", edgecolor="white"
        )

        ax.set_xlabel(column_name)
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")
        ax.set_title(f"Distribution: {column_name}")

    plt.tight_layout()

    # Save to bytes
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def create_correlation_heatmap(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    numeric_columns: List[str],
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Create a side-by-side correlation heatmap as PNG bytes.

    Args:
        original_df: Original DataFrame.
        protected_df: Protected DataFrame.
        numeric_columns: List of numeric column names.
        dpi: Image resolution.

    Returns:
        PNG image as bytes.
    """
    if len(numeric_columns) < 2:
        # Not enough columns for correlation
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation",
                ha="center", va="center")
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()

    # Compute correlations
    orig_corr = original_df[numeric_columns].corr()
    prot_corr = protected_df[numeric_columns].corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)

    # Original correlation
    ax1.imshow(orig_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(numeric_columns)))
    ax1.set_yticks(range(len(numeric_columns)))
    ax1.set_xticklabels(numeric_columns, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(numeric_columns, fontsize=8)
    ax1.set_title("Original Correlations")

    # Protected correlation
    im2 = ax2.imshow(prot_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(numeric_columns)))
    ax2.set_yticks(range(len(numeric_columns)))
    ax2.set_xticklabels(numeric_columns, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(numeric_columns, fontsize=8)
    ax2.set_title("Protected Correlations")

    # Colorbar
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8)

    # Note: skip tight_layout() as it conflicts with multi-axis colorbar
    # bbox_inches="tight" in savefig() handles layout properly

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def create_privacy_budget_chart(
    column_configs: Dict[str, Dict[str, Any]],
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Create a privacy budget breakdown chart as PNG bytes.

    Args:
        column_configs: Column configuration dictionary.
        dpi: Image resolution.

    Returns:
        PNG image as bytes.
    """
    # Extract protected columns with epsilon
    protected = [
        (col, cfg.get("epsilon", 1.0))
        for col, cfg in column_configs.items()
        if cfg.get("mode") == "protect"
    ]

    if not protected:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
        ax.text(0.5, 0.5, "No protected columns", ha="center", va="center")
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.getvalue()

    # Sort by epsilon (descending)
    protected.sort(key=lambda x: x[1], reverse=True)
    columns = [p[0] for p in protected]
    epsilons = [p[1] for p in protected]

    fig, ax = plt.subplots(figsize=(8, max(3, len(columns) * 0.4)), dpi=dpi)

    # Horizontal bar chart
    y_pos = range(len(columns))
    bars = ax.barh(y_pos, epsilons, color="#1f77b4", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(columns, fontsize=9)
    ax.set_xlabel("Epsilon (ε)")
    ax.set_title("Privacy Budget by Column")
    ax.invert_yaxis()  # Top to bottom

    # Add value labels
    for bar, eps in zip(bars, epsilons):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{eps:.2f}", va="center", fontsize=8)

    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# PDF Report Generator
# =============================================================================


class PDFReportGenerator:
    """Generate PDF reports for differential privacy analysis.

    Example:
        >>> generator = PDFReportGenerator()
        >>> report = generator.generate(
        ...     original_df=original,
        ...     protected_df=protected,
        ...     comparison=comparison,
        ...     column_configs=configs,
        ... )
        >>> report.save("report.pdf")
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        """Initialize the report generator.

        Args:
            config: Report configuration. Uses defaults if None.
        """
        self._config = config or ReportConfig()
        self._styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Set up custom paragraph styles."""
        self._styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self._styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor("#1f77b4"),
        ))
        self._styles.add(ParagraphStyle(
            name="SectionTitle",
            parent=self._styles["Heading2"],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor("#333333"),
        ))
        self._styles.add(ParagraphStyle(
            name="SubSection",
            parent=self._styles["Heading3"],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=8,
        ))
        self._styles.add(ParagraphStyle(
            name="ReportBody",
            parent=self._styles["Normal"],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            name="ReportSmall",
            parent=self._styles["Normal"],
            fontSize=8,
            textColor=colors.gray,
        ))

    def generate(
        self,
        original_df: pd.DataFrame,
        protected_df: pd.DataFrame,
        comparison: DatasetComparison,
        column_configs: Dict[str, Dict[str, Any]],
        metadata: Optional[ReportMetadata] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> GeneratedReport:
        """Generate a PDF report.

        Args:
            original_df: Original DataFrame.
            protected_df: Protected DataFrame.
            comparison: Dataset comparison results.
            column_configs: Column configuration dictionary.
            metadata: Report metadata.
            dataset_info: Optional dataset information.

        Returns:
            GeneratedReport with PDF content.
        """
        start_time = datetime.now()
        metadata = metadata or ReportMetadata()

        if dataset_info:
            metadata.original_filename = dataset_info.get("filename")

        # Build document
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self._config.page_size,
            topMargin=self._config.margins[0],
            rightMargin=self._config.margins[1],
            bottomMargin=self._config.margins[2],
            leftMargin=self._config.margins[3],
        )

        # Build story (content)
        story = []

        # Title page
        story.extend(self._build_title_section(metadata))

        # Executive summary
        story.extend(self._build_executive_summary(
            original_df, protected_df, comparison, column_configs
        ))

        # Configuration details
        story.append(PageBreak())
        story.extend(self._build_configuration_section(column_configs))

        # Statistical comparisons
        story.append(PageBreak())
        story.extend(self._build_statistics_section(comparison))

        # Visualizations
        if self._config.include_charts:
            story.append(PageBreak())
            story.extend(self._build_visualizations_section(
                original_df, protected_df, comparison, column_configs
            ))

        # Build PDF
        doc.build(story)

        # Calculate metrics
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds() * 1000

        pdf_content = buffer.getvalue()

        # Estimate page count (rough)
        page_count = max(1, pdf_content.count(b"/Page") - 1)

        return GeneratedReport(
            content=pdf_content,
            metadata=metadata,
            page_count=page_count,
            generation_time_ms=generation_time,
        )

    def _build_title_section(
        self,
        metadata: ReportMetadata,
    ) -> List[Any]:
        """Build the title section."""
        elements = []

        elements.append(Spacer(1, 1 * inch))
        elements.append(Paragraph(metadata.title, self._styles["ReportTitle"]))
        elements.append(Spacer(1, 0.3 * inch))

        if metadata.original_filename:
            elements.append(Paragraph(
                f"Dataset: {metadata.original_filename}",
                self._styles["ReportBody"]
            ))

        elements.append(Paragraph(
            f"Generated: {metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            self._styles["ReportBody"]
        ))
        elements.append(Paragraph(
            f"Generated by: {metadata.author}",
            self._styles["ReportSmall"]
        ))

        elements.append(Spacer(1, 0.5 * inch))

        return elements

    def _build_executive_summary(
        self,
        original_df: pd.DataFrame,
        protected_df: pd.DataFrame,
        comparison: DatasetComparison,
        column_configs: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        """Build the executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self._styles["SectionTitle"]))

        # Privacy budget summary
        total_epsilon = sum(
            cfg.get("epsilon", 0)
            for cfg in column_configs.values()
            if cfg.get("mode") == "protect"
        )

        protected_count = sum(
            1 for cfg in column_configs.values()
            if cfg.get("mode") == "protect"
        )
        passthrough_count = sum(
            1 for cfg in column_configs.values()
            if cfg.get("mode") == "passthrough"
        )
        excluded_count = sum(
            1 for cfg in column_configs.values()
            if cfg.get("mode") == "exclude"
        )

        # Summary table
        summary_data = [
            ["Metric", "Value"],
            ["Original Rows", f"{len(original_df):,}"],
            ["Original Columns", str(len(original_df.columns))],
            ["Protected Columns", str(protected_count)],
            ["Passthrough Columns", str(passthrough_count)],
            ["Excluded Columns", str(excluded_count)],
            ["Total Privacy Budget (ε)", f"{total_epsilon:.4f}"],
        ]

        table = Table(summary_data, colWidths=[2.5 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        # Privacy interpretation
        if total_epsilon <= 1.0:
            privacy_level = "Strong privacy protection"
            privacy_desc = (
                "With ε ≤ 1.0, this dataset provides strong privacy guarantees. "
                "Individual records are well-protected against inference attacks."
            )
        elif total_epsilon <= 5.0:
            privacy_level = "Moderate privacy protection"
            privacy_desc = (
                "With ε between 1.0 and 5.0, this dataset provides balanced "
                "privacy-utility trade-off suitable for many analytical tasks."
            )
        else:
            privacy_level = "Utility-focused protection"
            privacy_desc = (
                "With ε > 5.0, this configuration prioritizes data utility. "
                "Consider lower epsilon values for sensitive applications."
            )

        elements.append(Paragraph(
            f"<b>Privacy Assessment:</b> {privacy_level}",
            self._styles["ReportBody"]
        ))
        elements.append(Paragraph(privacy_desc, self._styles["ReportBody"]))

        # Utility summary
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(
            "<b>Data Utility Summary</b>",
            self._styles["SubSection"]
        ))

        if comparison.numeric_comparisons:
            avg_mae = sum(
                c.divergence.mae for c in comparison.numeric_comparisons
            ) / len(comparison.numeric_comparisons)
            elements.append(Paragraph(
                f"Average numeric column MAE: {avg_mae:.4f}",
                self._styles["ReportBody"]
            ))

        if comparison.categorical_comparisons:
            avg_drift = sum(
                c.divergence.category_drift for c in comparison.categorical_comparisons
            ) / len(comparison.categorical_comparisons)
            elements.append(Paragraph(
                f"Average categorical drift: {avg_drift:.4f}",
                self._styles["ReportBody"]
            ))

        return elements

    def _build_configuration_section(
        self,
        column_configs: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        """Build the configuration details section."""
        elements = []

        elements.append(Paragraph(
            "Configuration Details",
            self._styles["SectionTitle"]
        ))

        elements.append(Paragraph(
            "The following table shows the privacy configuration applied "
            "to each column in the dataset.",
            self._styles["ReportBody"]
        ))

        # Build table data
        table_data = [["Column", "Mode", "Epsilon", "Mechanism", "Sensitivity"]]

        for col, cfg in column_configs.items():
            mode = cfg.get("mode", "unknown")
            if mode == "protect":
                epsilon = f"{cfg.get('epsilon', 'N/A'):.2f}" if isinstance(
                    cfg.get('epsilon'), (int, float)
                ) else "N/A"
                mechanism = cfg.get("mechanism", "N/A")
                sensitivity = cfg.get("sensitivity", "N/A")
            else:
                epsilon = "—"
                mechanism = "—"
                sensitivity = "—"

            mode_display = {
                "protect": "Protected",
                "passthrough": "Passthrough",
                "exclude": "Excluded",
            }.get(mode, mode)

            table_data.append([col, mode_display, epsilon, mechanism, sensitivity])

        # Create table
        col_widths = [1.8 * inch, 1.2 * inch, 0.8 * inch, 1.2 * inch, 1 * inch]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (2, 1), (2, -1), "CENTER"),  # Epsilon centered
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.HexColor("#f8f9fa"),
                colors.white,
            ]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
        ]))

        elements.append(table)

        return elements

    def _build_statistics_section(
        self,
        comparison: DatasetComparison,
    ) -> List[Any]:
        """Build the statistical comparisons section."""
        elements = []

        elements.append(Paragraph(
            "Statistical Comparison",
            self._styles["SectionTitle"]
        ))

        elements.append(Paragraph(
            "Comparison of statistical properties between original and "
            "protected datasets.",
            self._styles["ReportBody"]
        ))

        # Numeric columns
        if comparison.numeric_comparisons:
            elements.append(Paragraph(
                "Numeric Columns",
                self._styles["SubSection"]
            ))

            table_data = [["Column", "MAE", "RMSE", "Mean Δ", "Std Δ"]]

            for comp in comparison.numeric_comparisons:
                table_data.append([
                    comp.column_name,
                    f"{comp.divergence.mae:.4f}",
                    f"{comp.divergence.rmse:.4f}",
                    f"{comp.divergence.mean_difference:.4f}",
                    f"{comp.divergence.std_difference:.4f}",
                ])

            table = Table(table_data, colWidths=[1.8 * inch] + [1.1 * inch] * 4)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28a745")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.HexColor("#f8f9fa"),
                    colors.white,
                ]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))

        # Categorical columns
        if comparison.categorical_comparisons:
            elements.append(Paragraph(
                "Categorical Columns",
                self._styles["SubSection"]
            ))

            table_data = [["Column", "Categories", "Drift", "Freq MAE", "Mode OK"]]

            for cat_comp in comparison.categorical_comparisons:
                mode_ok = "Yes" if cat_comp.divergence.mode_preserved else "No"
                table_data.append([
                    cat_comp.column_name,
                    str(cat_comp.divergence.cardinality_original),
                    f"{cat_comp.divergence.category_drift:.4f}",
                    f"{cat_comp.divergence.frequency_mae:.4f}",
                    mode_ok,
                ])

            table = Table(table_data, colWidths=[1.8 * inch] + [1.1 * inch] * 4)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17a2b8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.HexColor("#f8f9fa"),
                    colors.white,
                ]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
            ]))

            elements.append(table)

        return elements

    def _build_visualizations_section(
        self,
        original_df: pd.DataFrame,
        protected_df: pd.DataFrame,
        comparison: DatasetComparison,
        column_configs: Dict[str, Dict[str, Any]],
    ) -> List[Any]:
        """Build the visualizations section."""
        elements = []

        elements.append(Paragraph(
            "Visualizations",
            self._styles["SectionTitle"]
        ))

        # Privacy budget chart
        elements.append(Paragraph(
            "Privacy Budget Distribution",
            self._styles["SubSection"]
        ))

        budget_chart = create_privacy_budget_chart(
            column_configs,
            dpi=self._config.dpi,
        )
        img = Image(io.BytesIO(budget_chart), width=CHART_WIDTH, height=CHART_HEIGHT)
        elements.append(img)
        elements.append(Spacer(1, 0.3 * inch))

        # Histogram for first few numeric columns
        numeric_cols = [c.column_name for c in comparison.numeric_comparisons[:3]]
        for col in numeric_cols:
            if col in original_df.columns and col in protected_df.columns:
                elements.append(Paragraph(
                    f"Distribution: {col}",
                    self._styles["SubSection"]
                ))

                hist_chart = create_histogram_chart(
                    original_df[col],
                    protected_df[col],
                    col,
                    dpi=self._config.dpi,
                )
                img = Image(
                    io.BytesIO(hist_chart),
                    width=CHART_WIDTH,
                    height=CHART_HEIGHT
                )
                elements.append(img)
                elements.append(Spacer(1, 0.2 * inch))

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            elements.append(PageBreak())
            elements.append(Paragraph(
                "Correlation Preservation",
                self._styles["SubSection"]
            ))

            all_numeric = [c.column_name for c in comparison.numeric_comparisons]
            corr_chart = create_correlation_heatmap(
                original_df,
                protected_df,
                all_numeric[:8],  # Limit to 8 columns for readability
                dpi=self._config.dpi,
            )
            img = Image(
                io.BytesIO(corr_chart),
                width=6.5 * inch,
                height=3.5 * inch
            )
            elements.append(img)

        return elements


# =============================================================================
# Convenience Function
# =============================================================================


def generate_pdf_report(
    original_df: pd.DataFrame,
    protected_df: pd.DataFrame,
    comparison: DatasetComparison,
    column_configs: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> GeneratedReport:
    """Generate a PDF report (convenience function).

    Args:
        original_df: Original DataFrame.
        protected_df: Protected DataFrame.
        comparison: Dataset comparison results.
        column_configs: Column configuration dictionary.
        output_path: Optional path to save the PDF.
        **kwargs: Additional arguments for ReportMetadata.

    Returns:
        GeneratedReport object.
    """
    metadata = ReportMetadata(**kwargs) if kwargs else None
    generator = PDFReportGenerator()
    report = generator.generate(
        original_df=original_df,
        protected_df=protected_df,
        comparison=comparison,
        column_configs=column_configs,
        metadata=metadata,
    )

    if output_path:
        report.save(output_path)

    return report
