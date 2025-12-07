"""PDF report generation.

This module provides PDF report generation for differential privacy analysis.

Example:
    >>> from dp_toolkit.reports import PDFReportGenerator, generate_pdf_report
    >>> generator = PDFReportGenerator()
    >>> report = generator.generate(
    ...     original_df=original,
    ...     protected_df=protected,
    ...     comparison=comparison,
    ...     column_configs=configs,
    ... )
    >>> report.save("report.pdf")
"""

from dp_toolkit.reports.pdf_generator import (
    PDFReportGenerator,
    GeneratedReport,
    ReportConfig,
    ReportMetadata,
    generate_pdf_report,
)

__all__ = [
    "PDFReportGenerator",
    "GeneratedReport",
    "ReportConfig",
    "ReportMetadata",
    "generate_pdf_report",
]
