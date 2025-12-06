"""Unit tests for Excel and Parquet loading functionality."""

import pytest
import pandas as pd

from dp_toolkit.data.loader import (
    ColumnType,
    DataLoader,
    DatasetInfo,
    load,
    load_excel,
    load_parquet,
)


class TestExcelLoader:
    """Tests for Excel loading functionality."""

    def test_load_excel_basic(self, sample_excel):
        """Test basic Excel loading."""
        result = load_excel(sample_excel)

        assert isinstance(result, DatasetInfo)
        assert result.file_format == "excel"
        assert result.row_count == 5
        assert result.column_count == 3
        assert result.encoding is None  # Excel doesn't have text encoding

    def test_load_excel_returns_dataframe(self, sample_excel):
        """Test that loaded result contains valid DataFrame."""
        result = load_excel(sample_excel)

        assert isinstance(result.dataframe, pd.DataFrame)
        assert len(result.dataframe) == 5
        assert list(result.dataframe.columns) == ["id", "name", "value"]

    def test_load_excel_column_info(self, sample_excel):
        """Test that column info is populated correctly."""
        result = load_excel(sample_excel)

        assert len(result.columns) == 3
        column_names = [col.name for col in result.columns]
        assert column_names == ["id", "name", "value"]

    def test_load_excel_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_excel(temp_dir / "nonexistent.xlsx")

    def test_load_excel_empty_file(self, empty_excel):
        """Test loading an empty Excel file (headers only)."""
        result = load_excel(empty_excel)

        assert result.row_count == 0
        assert result.column_count == 3


class TestExcelSheetSelection:
    """Tests for Excel sheet selection functionality."""

    def test_load_first_sheet_by_default(self, multi_sheet_excel):
        """Test that first sheet is loaded by default."""
        result = load_excel(multi_sheet_excel)

        assert result.row_count == 3
        assert list(result.dataframe.columns) == ["a", "b"]

    def test_load_sheet_by_index(self, multi_sheet_excel):
        """Test loading sheet by index."""
        result = load_excel(multi_sheet_excel, sheet_name=1)

        assert list(result.dataframe.columns) == ["x", "y"]

    def test_load_sheet_by_name(self, multi_sheet_excel):
        """Test loading sheet by name."""
        result = load_excel(multi_sheet_excel, sheet_name="Summary")

        assert list(result.dataframe.columns) == ["num", "cat"]
        assert result.row_count == 2

    def test_get_sheet_names(self, multi_sheet_excel):
        """Test getting list of sheet names."""
        loader = DataLoader()
        sheets = loader.get_excel_sheet_names(multi_sheet_excel)

        assert sheets == ["Sheet1", "Data", "Summary"]

    def test_get_sheet_names_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing files."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.get_excel_sheet_names(temp_dir / "nonexistent.xlsx")


class TestParquetLoader:
    """Tests for Parquet loading functionality."""

    def test_load_parquet_basic(self, sample_parquet):
        """Test basic Parquet loading."""
        result = load_parquet(sample_parquet)

        assert isinstance(result, DatasetInfo)
        assert result.file_format == "parquet"
        assert result.row_count == 5
        assert result.column_count == 3
        assert result.encoding is None  # Parquet is binary

    def test_load_parquet_returns_dataframe(self, sample_parquet):
        """Test that loaded result contains valid DataFrame."""
        result = load_parquet(sample_parquet)

        assert isinstance(result.dataframe, pd.DataFrame)
        assert len(result.dataframe) == 5
        assert list(result.dataframe.columns) == ["id", "name", "value"]

    def test_load_parquet_with_types(self, parquet_with_types):
        """Test loading Parquet with various data types."""
        result = load_parquet(parquet_with_types)

        assert result.row_count == 5
        assert result.column_count == 5

        # Check type detection
        type_map = {col.name: col.column_type for col in result.columns}
        assert type_map["int_col"] == ColumnType.NUMERIC
        assert type_map["float_col"] == ColumnType.NUMERIC
        assert type_map["str_col"] == ColumnType.CATEGORICAL
        assert type_map["date_col"] == ColumnType.DATE
        assert type_map["bool_col"] == ColumnType.CATEGORICAL

    def test_load_parquet_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_parquet(temp_dir / "nonexistent.parquet")

    def test_load_parquet_performance(self, large_parquet):
        """Test loading a larger Parquet file."""
        import time

        start = time.time()
        result = load_parquet(large_parquet)
        elapsed = time.time() - start

        assert result.row_count == 10000
        assert elapsed < 5.0  # Should be fast


class TestFormatDetection:
    """Tests for automatic format detection."""

    def test_detect_csv_format(self, sample_csv):
        """Test CSV format detection."""
        result = load(sample_csv)
        assert result.file_format == "csv"

    def test_detect_excel_format(self, sample_excel):
        """Test Excel format detection."""
        result = load(sample_excel)
        assert result.file_format == "excel"

    def test_detect_parquet_format(self, sample_parquet):
        """Test Parquet format detection."""
        result = load(sample_parquet)
        assert result.file_format == "parquet"

    def test_explicit_format_override(self, sample_csv):
        """Test that explicit format parameter works."""
        # This should work even though extension detection would also work
        result = load(sample_csv, file_format="csv")
        assert result.file_format == "csv"

    def test_unknown_extension_raises(self, temp_dir):
        """Test that unknown extension raises ValueError."""
        # Create a file with unknown extension
        unknown_file = temp_dir / "data.unknown"
        unknown_file.write_text("test")

        with pytest.raises(ValueError, match="Cannot detect format"):
            load(unknown_file)

    def test_file_not_found(self, temp_dir):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load(temp_dir / "nonexistent.csv")


class TestRoundTrip:
    """Tests for round-trip loading (load → export → load)."""

    def test_csv_to_excel_roundtrip(self, sample_csv, temp_dir):
        """Test loading CSV and exporting to Excel."""
        # Load CSV
        csv_result = load(sample_csv)

        # Export to Excel
        excel_path = temp_dir / "exported.xlsx"
        csv_result.dataframe.to_excel(excel_path, index=False)

        # Load Excel
        excel_result = load(excel_path)

        # Compare structure (types may differ slightly between formats)
        assert csv_result.row_count == excel_result.row_count
        assert csv_result.column_count == excel_result.column_count
        assert list(csv_result.dataframe.columns) == list(excel_result.dataframe.columns)

        # Compare values (allow type coercion)
        for col in csv_result.dataframe.columns:
            csv_vals = csv_result.dataframe[col].tolist()
            excel_vals = excel_result.dataframe[col].tolist()
            assert csv_vals == excel_vals, f"Values differ for column {col}"

    def test_csv_to_parquet_roundtrip(self, sample_csv, temp_dir):
        """Test loading CSV and exporting to Parquet."""
        # Load CSV
        csv_result = load(sample_csv)

        # Export to Parquet
        parquet_path = temp_dir / "exported.parquet"
        csv_result.dataframe.to_parquet(parquet_path, index=False)

        # Load Parquet
        parquet_result = load(parquet_path)

        # Compare
        assert csv_result.row_count == parquet_result.row_count
        assert csv_result.column_count == parquet_result.column_count
        pd.testing.assert_frame_equal(
            csv_result.dataframe.reset_index(drop=True),
            parquet_result.dataframe.reset_index(drop=True),
        )

    def test_excel_to_parquet_roundtrip(self, sample_excel, temp_dir):
        """Test loading Excel and exporting to Parquet."""
        # Load Excel
        excel_result = load(sample_excel)

        # Export to Parquet
        parquet_path = temp_dir / "exported.parquet"
        excel_result.dataframe.to_parquet(parquet_path, index=False)

        # Load Parquet
        parquet_result = load(parquet_path)

        # Compare
        assert excel_result.row_count == parquet_result.row_count
        assert excel_result.column_count == parquet_result.column_count
        pd.testing.assert_frame_equal(
            excel_result.dataframe.reset_index(drop=True),
            parquet_result.dataframe.reset_index(drop=True),
        )

    def test_parquet_preserves_types(self, parquet_with_types, temp_dir):
        """Test that Parquet preserves data types through round-trip."""
        # Load original
        original = load(parquet_with_types)

        # Export and reload
        export_path = temp_dir / "roundtrip.parquet"
        original.dataframe.to_parquet(export_path, index=False)
        reloaded = load(export_path)

        # Check dtypes are preserved
        for col in original.dataframe.columns:
            assert (
                original.dataframe[col].dtype == reloaded.dataframe[col].dtype
            ), f"Type mismatch for column {col}"


class TestUnifiedInterface:
    """Tests for the unified load() interface."""

    def test_load_with_csv_kwargs(self, temp_dir):
        """Test that kwargs are passed through for CSV."""
        # Create a TSV file
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tsv_path = temp_dir / "data.tsv"
        df.to_csv(tsv_path, index=False, sep="\t")

        # Load with separator kwarg
        result = load(tsv_path, sep="\t")
        assert result.row_count == 2
        assert list(result.dataframe.columns) == ["a", "b"]

    def test_load_with_excel_kwargs(self, multi_sheet_excel):
        """Test that kwargs are passed through for Excel."""
        result = load(multi_sheet_excel, sheet_name="Data")
        assert list(result.dataframe.columns) == ["x", "y"]
