"""Tests for dp_toolkit.data.exporter module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.exporter import (
    ExportFormat,
    ExportMetadata,
    ExportResult,
    DataExporter,
    export_csv,
    export_excel,
    export_parquet,
    export_data,
    read_export_metadata,
    METADATA_SUFFIX,
)
from dp_toolkit.data.transformer import (
    DatasetConfig,
    DatasetTransformer,
    ProtectionMode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
    })


@pytest.fixture
def sample_transform_result(sample_dataframe):
    """Create a sample transform result for testing."""
    transformer = DatasetTransformer()
    config = DatasetConfig(global_epsilon=1.0)
    config.protect_columns(["age", "salary"])
    config.passthrough_columns(["id", "name"])
    return transformer.transform(sample_dataframe, config)


@pytest.fixture
def exporter():
    """Create a DataExporter instance."""
    return DataExporter()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# ExportMetadata Tests
# =============================================================================


class TestExportMetadata:
    """Tests for ExportMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = ExportMetadata(
            export_timestamp="2024-01-01T00:00:00",
            source_row_count=100,
            source_column_count=5,
            exported_column_count=4,
            total_epsilon=1.0,
            total_delta=1e-6,
            protected_columns=["age", "salary"],
            passthrough_columns=["id", "name"],
            excluded_columns=["ssn"],
        )
        d = meta.to_dict()
        assert d["source_row_count"] == 100
        assert d["protected_columns"] == ["age", "salary"]

    def test_to_json(self):
        """Test conversion to JSON."""
        meta = ExportMetadata(
            export_timestamp="2024-01-01T00:00:00",
            source_row_count=100,
            source_column_count=5,
            exported_column_count=4,
            total_epsilon=1.0,
            total_delta=None,
            protected_columns=[],
            passthrough_columns=[],
            excluded_columns=[],
        )
        json_str = meta.to_json()
        parsed = json.loads(json_str)
        assert parsed["source_row_count"] == 100

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "export_timestamp": "2024-01-01T00:00:00",
            "source_row_count": 100,
            "source_column_count": 5,
            "exported_column_count": 4,
            "total_epsilon": 1.0,
            "total_delta": None,
            "protected_columns": ["age"],
            "passthrough_columns": ["name"],
            "excluded_columns": [],
        }
        meta = ExportMetadata.from_dict(d)
        assert meta.source_row_count == 100
        assert meta.protected_columns == ["age"]

    def test_from_json(self):
        """Test creation from JSON."""
        json_str = '{"export_timestamp": "2024-01-01", "source_row_count": 50, "source_column_count": 3, "exported_column_count": 3, "total_epsilon": 0.5, "total_delta": null, "protected_columns": [], "passthrough_columns": [], "excluded_columns": []}'
        meta = ExportMetadata.from_json(json_str)
        assert meta.source_row_count == 50
        assert meta.total_epsilon == 0.5

    def test_from_transform_result(self, sample_transform_result):
        """Test creation from transform result."""
        meta = ExportMetadata.from_transform_result(
            result=sample_transform_result,
            format="csv",
        )
        assert meta.source_row_count == 5
        assert meta.total_epsilon > 0
        assert "age" in meta.protected_columns
        assert "id" in meta.passthrough_columns


# =============================================================================
# CSV Export Tests
# =============================================================================


class TestCSVExport:
    """Tests for CSV export functionality."""

    def test_basic_export(self, exporter, sample_dataframe, temp_dir):
        """Test basic CSV export."""
        path = temp_dir / "test.csv"
        result = exporter.export_csv(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        assert result.path.exists()
        assert result.format == ExportFormat.CSV
        assert result.row_count == 5
        assert result.column_count == 4
        assert result.file_size_bytes > 0

    def test_export_with_metadata(self, exporter, sample_dataframe, temp_dir):
        """Test CSV export with metadata sidecar."""
        path = temp_dir / "test.csv"
        result = exporter.export_csv(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        assert result.metadata_path is not None
        assert result.metadata_path.exists()
        assert result.metadata_path.suffix == ".json"

    def test_export_with_transform_result(
        self, exporter, sample_dataframe, sample_transform_result, temp_dir
    ):
        """Test CSV export with transform result."""
        path = temp_dir / "test.csv"
        result = exporter.export_csv(
            df=sample_transform_result.data,
            path=path,
            transform_result=sample_transform_result,
            include_metadata=True,
        )

        assert result.metadata.total_epsilon > 0
        assert len(result.metadata.protected_columns) == 2

    def test_round_trip_csv(self, exporter, sample_dataframe, temp_dir):
        """Test CSV round-trip integrity."""
        path = temp_dir / "test.csv"
        exporter.export_csv(df=sample_dataframe, path=path, include_metadata=False)

        # Read back
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(sample_dataframe, loaded)

    def test_custom_encoding(self, exporter, temp_dir):
        """Test CSV export with custom encoding."""
        df = pd.DataFrame({"name": ["Ångström", "Müller", "Café"]})
        path = temp_dir / "test.csv"

        result = exporter.export_csv(
            df=df,
            path=path,
            encoding="utf-8",
            include_metadata=False,
        )

        # Read back with same encoding
        loaded = pd.read_csv(path, encoding="utf-8")
        pd.testing.assert_frame_equal(df, loaded)


# =============================================================================
# Excel Export Tests
# =============================================================================


class TestExcelExport:
    """Tests for Excel export functionality."""

    def test_basic_export(self, exporter, sample_dataframe, temp_dir):
        """Test basic Excel export."""
        path = temp_dir / "test.xlsx"
        result = exporter.export_excel(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        assert result.path.exists()
        assert result.format == ExportFormat.EXCEL
        assert result.row_count == 5
        assert result.column_count == 4

    def test_export_with_metadata_sheet(self, exporter, sample_dataframe, temp_dir):
        """Test Excel export with metadata sheet."""
        path = temp_dir / "test.xlsx"
        result = exporter.export_excel(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        # Read back and check sheets - use context manager to properly close
        with pd.ExcelFile(path) as xl:
            assert "Data" in xl.sheet_names
            assert "Metadata" in xl.sheet_names

    def test_custom_sheet_names(self, exporter, sample_dataframe, temp_dir):
        """Test Excel export with custom sheet names."""
        path = temp_dir / "test.xlsx"
        exporter.export_excel(
            df=sample_dataframe,
            path=path,
            sheet_name="MyData",
            metadata_sheet_name="MyMetadata",
            include_metadata=True,
        )

        # Use context manager to properly close file
        with pd.ExcelFile(path) as xl:
            assert "MyData" in xl.sheet_names
            assert "MyMetadata" in xl.sheet_names

    def test_round_trip_excel(self, exporter, sample_dataframe, temp_dir):
        """Test Excel round-trip integrity."""
        path = temp_dir / "test.xlsx"
        exporter.export_excel(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        # Read back - check_dtype=False because Excel doesn't perfectly preserve dtypes
        loaded = pd.read_excel(path)
        pd.testing.assert_frame_equal(
            sample_dataframe, loaded, check_dtype=False
        )


# =============================================================================
# Parquet Export Tests
# =============================================================================


class TestParquetExport:
    """Tests for Parquet export functionality."""

    def test_basic_export(self, exporter, sample_dataframe, temp_dir):
        """Test basic Parquet export."""
        path = temp_dir / "test.parquet"
        result = exporter.export_parquet(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        assert result.path.exists()
        assert result.format == ExportFormat.PARQUET
        assert result.row_count == 5
        assert result.column_count == 4

    def test_export_with_embedded_metadata(
        self, exporter, sample_dataframe, temp_dir
    ):
        """Test Parquet export with embedded metadata."""
        path = temp_dir / "test.parquet"
        result = exporter.export_parquet(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        # Metadata should be embedded, not in separate file
        assert result.metadata_path is None
        assert result.metadata is not None

    def test_round_trip_parquet(self, exporter, sample_dataframe, temp_dir):
        """Test Parquet round-trip integrity."""
        path = temp_dir / "test.parquet"
        exporter.export_parquet(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        # Read back
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(sample_dataframe, loaded)

    def test_compression_options(self, exporter, sample_dataframe, temp_dir):
        """Test Parquet export with different compression."""
        for compression in ["snappy", "gzip", "none"]:
            path = temp_dir / f"test_{compression}.parquet"
            result = exporter.export_parquet(
                df=sample_dataframe,
                path=path,
                compression=compression,
                include_metadata=False,
            )
            assert result.path.exists()


# =============================================================================
# Generic Export Tests
# =============================================================================


class TestGenericExport:
    """Tests for generic export functionality."""

    def test_auto_detect_csv(self, exporter, sample_dataframe, temp_dir):
        """Test auto-detection of CSV format."""
        path = temp_dir / "test.csv"
        result = exporter.export(df=sample_dataframe, path=path)
        assert result.format == ExportFormat.CSV

    def test_auto_detect_excel(self, exporter, sample_dataframe, temp_dir):
        """Test auto-detection of Excel format."""
        path = temp_dir / "test.xlsx"
        result = exporter.export(df=sample_dataframe, path=path)
        assert result.format == ExportFormat.EXCEL

    def test_auto_detect_parquet(self, exporter, sample_dataframe, temp_dir):
        """Test auto-detection of Parquet format."""
        path = temp_dir / "test.parquet"
        result = exporter.export(df=sample_dataframe, path=path)
        assert result.format == ExportFormat.PARQUET

    def test_explicit_format_override(self, exporter, sample_dataframe, temp_dir):
        """Test explicit format override."""
        path = temp_dir / "test.data"  # Unknown extension
        result = exporter.export(
            df=sample_dataframe,
            path=path,
            format=ExportFormat.CSV,
        )
        assert result.format == ExportFormat.CSV

    def test_unknown_extension_raises(self, exporter, sample_dataframe, temp_dir):
        """Test that unknown extension raises error."""
        path = temp_dir / "test.unknown"
        with pytest.raises(ValueError, match="Cannot determine format"):
            exporter.export(df=sample_dataframe, path=path)


# =============================================================================
# Metadata Reading Tests
# =============================================================================


class TestReadMetadata:
    """Tests for reading export metadata."""

    def test_read_csv_metadata(self, exporter, sample_dataframe, temp_dir):
        """Test reading CSV metadata."""
        path = temp_dir / "test.csv"
        exporter.export_csv(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        meta = read_export_metadata(path)
        assert meta is not None
        assert meta.source_row_count == 5

    def test_read_excel_metadata(self, exporter, sample_dataframe, temp_dir):
        """Test reading Excel metadata."""
        path = temp_dir / "test.xlsx"
        exporter.export_excel(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        meta = read_export_metadata(path)
        assert meta is not None
        assert meta.source_row_count == 5

    def test_read_parquet_metadata(self, exporter, sample_dataframe, temp_dir):
        """Test reading Parquet metadata."""
        path = temp_dir / "test.parquet"
        exporter.export_parquet(
            df=sample_dataframe,
            path=path,
            include_metadata=True,
        )

        meta = read_export_metadata(path)
        assert meta is not None
        assert meta.source_row_count == 5

    def test_read_missing_metadata(self, exporter, sample_dataframe, temp_dir):
        """Test reading metadata when none exists."""
        path = temp_dir / "test.csv"
        exporter.export_csv(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )

        meta = read_export_metadata(path)
        assert meta is None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_export_csv_convenience(self, sample_dataframe, temp_dir):
        """Test export_csv convenience function."""
        path = temp_dir / "test.csv"
        result = export_csv(df=sample_dataframe, path=path, include_metadata=False)
        assert result.format == ExportFormat.CSV

    def test_export_excel_convenience(self, sample_dataframe, temp_dir):
        """Test export_excel convenience function."""
        path = temp_dir / "test.xlsx"
        result = export_excel(df=sample_dataframe, path=path, include_metadata=False)
        assert result.format == ExportFormat.EXCEL

    def test_export_parquet_convenience(self, sample_dataframe, temp_dir):
        """Test export_parquet convenience function."""
        path = temp_dir / "test.parquet"
        result = export_parquet(df=sample_dataframe, path=path, include_metadata=False)
        assert result.format == ExportFormat.PARQUET

    def test_export_data_convenience(self, sample_dataframe, temp_dir):
        """Test export_data convenience function."""
        path = temp_dir / "test.csv"
        result = export_data(df=sample_dataframe, path=path, include_metadata=False)
        assert result.format == ExportFormat.CSV


# =============================================================================
# Type Preservation Tests
# =============================================================================


class TestTypePreservation:
    """Tests for data type preservation."""

    def test_integer_types_csv(self, exporter, temp_dir):
        """Test integer type preservation in CSV."""
        df = pd.DataFrame({"int_col": [1, 2, 3, 4, 5]})
        path = temp_dir / "test.csv"

        exporter.export_csv(df=df, path=path, include_metadata=False)
        loaded = pd.read_csv(path)

        assert loaded["int_col"].dtype == np.int64

    def test_float_types_csv(self, exporter, temp_dir):
        """Test float type preservation in CSV."""
        df = pd.DataFrame({"float_col": [1.1, 2.2, 3.3, 4.4, 5.5]})
        path = temp_dir / "test.csv"

        exporter.export_csv(df=df, path=path, include_metadata=False)
        loaded = pd.read_csv(path)

        assert np.issubdtype(loaded["float_col"].dtype, np.floating)

    def test_datetime_types_parquet(self, exporter, temp_dir):
        """Test datetime type preservation in Parquet."""
        df = pd.DataFrame({
            "date_col": pd.date_range("2020-01-01", periods=5)
        })
        path = temp_dir / "test.parquet"

        exporter.export_parquet(df=df, path=path, include_metadata=False)
        loaded = pd.read_parquet(path)

        assert pd.api.types.is_datetime64_any_dtype(loaded["date_col"])

    def test_string_types_parquet(self, exporter, temp_dir):
        """Test string type preservation in Parquet."""
        df = pd.DataFrame({"str_col": ["a", "b", "c", "d", "e"]})
        path = temp_dir / "test.parquet"

        exporter.export_parquet(df=df, path=path, include_metadata=False)
        loaded = pd.read_parquet(path)

        # Parquet typically loads strings as object or string dtype
        assert loaded["str_col"].dtype in [object, "string"]


# =============================================================================
# Column Order Preservation Tests
# =============================================================================


class TestColumnOrderPreservation:
    """Tests for column order preservation."""

    def test_column_order_csv(self, exporter, temp_dir):
        """Test column order preservation in CSV."""
        df = pd.DataFrame({
            "z_col": [1, 2, 3],
            "a_col": [4, 5, 6],
            "m_col": [7, 8, 9],
        })
        path = temp_dir / "test.csv"

        exporter.export_csv(df=df, path=path, include_metadata=False)
        loaded = pd.read_csv(path)

        assert list(loaded.columns) == ["z_col", "a_col", "m_col"]

    def test_column_order_excel(self, exporter, temp_dir):
        """Test column order preservation in Excel."""
        df = pd.DataFrame({
            "z_col": [1, 2, 3],
            "a_col": [4, 5, 6],
            "m_col": [7, 8, 9],
        })
        path = temp_dir / "test.xlsx"

        exporter.export_excel(df=df, path=path, include_metadata=False)
        loaded = pd.read_excel(path)

        assert list(loaded.columns) == ["z_col", "a_col", "m_col"]

    def test_column_order_parquet(self, exporter, temp_dir):
        """Test column order preservation in Parquet."""
        df = pd.DataFrame({
            "z_col": [1, 2, 3],
            "a_col": [4, 5, 6],
            "m_col": [7, 8, 9],
        })
        path = temp_dir / "test.parquet"

        exporter.export_parquet(df=df, path=path, include_metadata=False)
        loaded = pd.read_parquet(path)

        assert list(loaded.columns) == ["z_col", "a_col", "m_col"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self, exporter, temp_dir):
        """Test exporting empty DataFrame."""
        df = pd.DataFrame()
        path = temp_dir / "test.csv"

        result = exporter.export_csv(df=df, path=path, include_metadata=False)
        assert result.row_count == 0
        assert result.column_count == 0

    def test_single_row(self, exporter, temp_dir):
        """Test exporting single row DataFrame."""
        df = pd.DataFrame({"col": [42]})
        path = temp_dir / "test.csv"

        result = exporter.export_csv(df=df, path=path, include_metadata=False)
        assert result.row_count == 1

    def test_single_column(self, exporter, temp_dir):
        """Test exporting single column DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        path = temp_dir / "test.csv"

        result = exporter.export_csv(df=df, path=path, include_metadata=False)
        assert result.column_count == 1

    def test_large_dataframe_performance(self, exporter, temp_dir):
        """Test exporting larger DataFrame."""
        import time

        # Create 100K rows x 10 columns
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(100000) for i in range(10)
        })
        path = temp_dir / "test.parquet"

        start = time.time()
        result = exporter.export_parquet(
            df=df,
            path=path,
            include_metadata=False,
        )
        elapsed = time.time() - start

        assert result.row_count == 100000
        # Should complete reasonably quickly
        assert elapsed < 30.0

    def test_special_characters_in_path(self, exporter, sample_dataframe, temp_dir):
        """Test export with special characters in path."""
        path = temp_dir / "test data (1).csv"
        result = exporter.export_csv(
            df=sample_dataframe,
            path=path,
            include_metadata=False,
        )
        assert result.path.exists()

    def test_null_values_preserved(self, exporter, temp_dir):
        """Test that null values are preserved."""
        df = pd.DataFrame({
            "col": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        path = temp_dir / "test.parquet"

        exporter.export_parquet(df=df, path=path, include_metadata=False)
        loaded = pd.read_parquet(path)

        assert pd.isna(loaded["col"].iloc[1])
        assert pd.isna(loaded["col"].iloc[3])
