"""Unit tests for PDF report generator."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.reports.pdf_generator import (
    PDFReportGenerator,
    GeneratedReport,
    ReportConfig,
    ReportMetadata,
    generate_pdf_report,
    create_histogram_chart,
    create_correlation_heatmap,
    create_privacy_budget_chart,
)
from dp_toolkit.analysis.comparator import DatasetComparator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_original_df():
    """Create a sample original DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "score": np.random.uniform(0, 100, 100),
    })


@pytest.fixture
def sample_protected_df(sample_original_df):
    """Create a sample protected DataFrame (with noise)."""
    df = sample_original_df.copy()
    df["age"] = df["age"] + np.random.laplace(0, 2, len(df))
    df["income"] = df["income"] + np.random.laplace(0, 1000, len(df))
    df["score"] = df["score"] + np.random.laplace(0, 5, len(df))
    # Categorical stays similar for this test
    return df


@pytest.fixture
def sample_comparison(sample_original_df, sample_protected_df):
    """Create a sample comparison result."""
    comparator = DatasetComparator()
    return comparator.compare(
        original=sample_original_df,
        protected=sample_protected_df,
        numeric_columns=["age", "income", "score"],
        categorical_columns=["category"],
    )


@pytest.fixture
def sample_column_configs():
    """Create sample column configurations."""
    return {
        "age": {
            "mode": "protect",
            "epsilon": 1.0,
            "mechanism": "laplace",
            "sensitivity": "medium",
        },
        "income": {
            "mode": "protect",
            "epsilon": 0.5,
            "mechanism": "laplace",
            "sensitivity": "high",
        },
        "category": {
            "mode": "passthrough",
        },
        "score": {
            "mode": "protect",
            "epsilon": 2.0,
            "mechanism": "laplace",
            "sensitivity": "low",
        },
    }


@pytest.fixture
def sample_dataset_info():
    """Create sample dataset info."""
    return {
        "filename": "test_data.csv",
        "row_count": 100,
        "column_count": 4,
    }


# =============================================================================
# ReportMetadata Tests
# =============================================================================


class TestReportMetadata:
    """Tests for ReportMetadata dataclass."""

    def test_default_values(self):
        """Test default metadata values."""
        metadata = ReportMetadata()
        assert metadata.title == "Differential Privacy Analysis Report"
        assert metadata.author == "DPtoolkit"
        assert metadata.subject == "Differential Privacy Analysis"
        assert metadata.original_filename is None
        assert isinstance(metadata.generated_at, datetime)

    def test_custom_values(self):
        """Test custom metadata values."""
        metadata = ReportMetadata(
            title="Custom Report",
            author="Test Author",
            original_filename="data.csv",
        )
        assert metadata.title == "Custom Report"
        assert metadata.author == "Test Author"
        assert metadata.original_filename == "data.csv"


# =============================================================================
# ReportConfig Tests
# =============================================================================


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = ReportConfig()
        assert config.include_charts is True
        assert config.include_raw_stats is True
        assert config.max_columns_per_page == 10
        assert config.dpi == 150

    def test_custom_values(self):
        """Test custom config values."""
        config = ReportConfig(
            include_charts=False,
            dpi=300,
        )
        assert config.include_charts is False
        assert config.dpi == 300


# =============================================================================
# GeneratedReport Tests
# =============================================================================


class TestGeneratedReport:
    """Tests for GeneratedReport dataclass."""

    def test_save_to_file(self, tmp_path):
        """Test saving report to file."""
        content = b"%PDF-1.4 test content"
        report = GeneratedReport(
            content=content,
            metadata=ReportMetadata(),
            page_count=1,
            generation_time_ms=100.0,
        )

        output_path = tmp_path / "test_report.pdf"
        result_path = report.save(output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == content


# =============================================================================
# Chart Generation Tests
# =============================================================================


class TestChartGeneration:
    """Tests for chart generation functions."""

    def test_create_histogram_chart(self, sample_original_df, sample_protected_df):
        """Test histogram chart generation."""
        chart_bytes = create_histogram_chart(
            original=sample_original_df["age"],
            protected=sample_protected_df["age"],
            column_name="age",
        )

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0
        # PNG signature
        assert chart_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_create_histogram_chart_empty_data(self):
        """Test histogram with empty data."""
        empty_series = pd.Series([], dtype=float)
        chart_bytes = create_histogram_chart(
            original=empty_series,
            protected=empty_series,
            column_name="empty",
        )

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0

    def test_create_correlation_heatmap(
        self, sample_original_df, sample_protected_df
    ):
        """Test correlation heatmap generation."""
        chart_bytes = create_correlation_heatmap(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            numeric_columns=["age", "income", "score"],
        )

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0
        assert chart_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_create_correlation_heatmap_single_column(
        self, sample_original_df, sample_protected_df
    ):
        """Test correlation heatmap with single column (not enough for correlation)."""
        chart_bytes = create_correlation_heatmap(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            numeric_columns=["age"],
        )

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0

    def test_create_privacy_budget_chart(self, sample_column_configs):
        """Test privacy budget chart generation."""
        chart_bytes = create_privacy_budget_chart(sample_column_configs)

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0
        assert chart_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_create_privacy_budget_chart_no_protected(self):
        """Test budget chart with no protected columns."""
        configs = {
            "col1": {"mode": "passthrough"},
            "col2": {"mode": "exclude"},
        }
        chart_bytes = create_privacy_budget_chart(configs)

        assert isinstance(chart_bytes, bytes)
        assert len(chart_bytes) > 0


# =============================================================================
# PDFReportGenerator Tests
# =============================================================================


class TestPDFReportGenerator:
    """Tests for PDFReportGenerator class."""

    def test_generate_basic_report(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test basic report generation."""
        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
        )

        assert isinstance(report, GeneratedReport)
        assert isinstance(report.content, bytes)
        assert len(report.content) > 0
        # Check PDF signature
        assert report.content[:4] == b"%PDF"
        assert report.generation_time_ms > 0
        assert report.page_count >= 1

    def test_generate_with_metadata(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test report generation with custom metadata."""
        generator = PDFReportGenerator()
        metadata = ReportMetadata(
            title="Custom Report Title",
            author="Test Suite",
        )

        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
            metadata=metadata,
        )

        assert report.metadata.title == "Custom Report Title"
        assert report.metadata.author == "Test Suite"

    def test_generate_with_dataset_info(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
        sample_dataset_info,
    ):
        """Test report generation with dataset info."""
        generator = PDFReportGenerator()

        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
            dataset_info=sample_dataset_info,
        )

        assert report.metadata.original_filename == "test_data.csv"

    def test_generate_without_charts(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test report generation without charts."""
        config = ReportConfig(include_charts=False)
        generator = PDFReportGenerator(config=config)

        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
        )

        assert isinstance(report.content, bytes)
        assert len(report.content) > 0

    def test_generate_saves_to_file(
        self,
        tmp_path,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test that generated report can be saved."""
        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
        )

        output_path = tmp_path / "test_output.pdf"
        report.save(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGeneratePdfReport:
    """Tests for generate_pdf_report convenience function."""

    def test_generate_pdf_report(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test convenience function."""
        report = generate_pdf_report(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
        )

        assert isinstance(report, GeneratedReport)
        assert len(report.content) > 0

    def test_generate_pdf_report_with_save(
        self,
        tmp_path,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test convenience function with automatic save."""
        output_path = tmp_path / "auto_save.pdf"

        report = generate_pdf_report(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size == len(report.content)

    def test_generate_pdf_report_with_kwargs(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test convenience function with metadata kwargs."""
        report = generate_pdf_report(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
            title="Custom Title via Kwargs",
        )

        assert report.metadata.title == "Custom Title via Kwargs"


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for PDF generation."""

    def test_generation_time_under_20_seconds(
        self,
        sample_original_df,
        sample_protected_df,
        sample_comparison,
        sample_column_configs,
    ):
        """Test that PDF generation completes within time limit."""
        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=sample_original_df,
            protected_df=sample_protected_df,
            comparison=sample_comparison,
            column_configs=sample_column_configs,
        )

        # Should complete in under 20 seconds (20000 ms)
        assert report.generation_time_ms < 20000

    def test_large_dataset_generation(self):
        """Test PDF generation with larger dataset."""
        np.random.seed(42)
        n_rows = 10000
        n_cols = 20

        # Create larger dataset
        original_df = pd.DataFrame({
            f"col_{i}": np.random.normal(0, 1, n_rows)
            for i in range(n_cols)
        })
        protected_df = original_df + np.random.laplace(0, 0.1, original_df.shape)

        # Create comparison
        comparator = DatasetComparator()
        comparison = comparator.compare(
            original=original_df,
            protected=protected_df,
            numeric_columns=[f"col_{i}" for i in range(n_cols)],
        )

        # Create configs
        column_configs = {
            f"col_{i}": {"mode": "protect", "epsilon": 1.0, "mechanism": "laplace"}
            for i in range(n_cols)
        }

        # Generate report
        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=original_df,
            protected_df=protected_df,
            comparison=comparison,
            column_configs=column_configs,
        )

        assert len(report.content) > 0
        assert report.generation_time_ms < 20000  # Under 20 seconds


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframes(self):
        """Test with empty DataFrames."""
        original_df = pd.DataFrame()
        protected_df = pd.DataFrame()

        comparator = DatasetComparator()
        comparison = comparator.compare(
            original=original_df,
            protected=protected_df,
        )

        column_configs = {}

        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=original_df,
            protected_df=protected_df,
            comparison=comparison,
            column_configs=column_configs,
        )

        assert len(report.content) > 0

    def test_all_columns_excluded(self, sample_original_df):
        """Test with all columns excluded."""
        protected_df = sample_original_df.copy()

        comparator = DatasetComparator()
        comparison = comparator.compare(
            original=sample_original_df,
            protected=protected_df,
        )

        column_configs = {
            col: {"mode": "exclude"}
            for col in sample_original_df.columns
        }

        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=sample_original_df,
            protected_df=protected_df,
            comparison=comparison,
            column_configs=column_configs,
        )

        assert len(report.content) > 0

    def test_special_characters_in_column_names(self):
        """Test with special characters in column names."""
        original_df = pd.DataFrame({
            "Column (with) parentheses": [1, 2, 3],
            "Column/with/slashes": [4, 5, 6],
            "Column & special <chars>": [7, 8, 9],
        })
        protected_df = original_df.copy()

        comparator = DatasetComparator()
        comparison = comparator.compare(
            original=original_df,
            protected=protected_df,
            numeric_columns=list(original_df.columns),
        )

        column_configs = {
            col: {"mode": "protect", "epsilon": 1.0}
            for col in original_df.columns
        }

        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=original_df,
            protected_df=protected_df,
            comparison=comparison,
            column_configs=column_configs,
        )

        assert len(report.content) > 0
