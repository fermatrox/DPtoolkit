"""Unit tests for the data loader module."""

import pytest
import pandas as pd

from dp_toolkit.data.loader import (
    ColumnInfo,
    ColumnType,
    DataLoader,
    DatasetInfo,
    load_csv,
)


class TestDataLoaderCSV:
    """Tests for CSV loading functionality."""

    def test_load_csv_basic(self, sample_csv):
        """Test basic CSV loading."""
        result = load_csv(sample_csv)

        assert isinstance(result, DatasetInfo)
        assert result.file_format == "csv"
        assert result.row_count == 5
        assert result.column_count == 7
        assert result.file_path == sample_csv

    def test_load_csv_returns_dataframe(self, sample_csv):
        """Test that loaded result contains valid DataFrame."""
        result = load_csv(sample_csv)

        assert isinstance(result.dataframe, pd.DataFrame)
        assert len(result.dataframe) == 5
        assert list(result.dataframe.columns) == [
            "id",
            "name",
            "age",
            "salary",
            "department",
            "hire_date",
            "is_active",
        ]

    def test_load_csv_column_info(self, sample_csv):
        """Test that column info is populated correctly."""
        result = load_csv(sample_csv)

        assert len(result.columns) == 7
        assert all(isinstance(col, ColumnInfo) for col in result.columns)

        # Check column names match
        column_names = [col.name for col in result.columns]
        assert column_names == [
            "id",
            "name",
            "age",
            "salary",
            "department",
            "hire_date",
            "is_active",
        ]

    def test_load_csv_memory_usage(self, sample_csv):
        """Test that memory usage is calculated."""
        result = load_csv(sample_csv)

        assert result.memory_usage_bytes > 0

    def test_load_csv_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_csv(temp_dir / "nonexistent.csv")

    def test_load_csv_with_loader_class(self, sample_csv):
        """Test using DataLoader class directly."""
        loader = DataLoader()
        result = loader.load_csv(sample_csv)

        assert isinstance(result, DatasetInfo)
        assert result.row_count == 5


class TestTypeDetection:
    """Tests for column type detection."""

    def test_detect_numeric_columns(self, numeric_csv):
        """Test detection of numeric columns."""
        result = load_csv(numeric_csv)

        for col in result.columns:
            assert col.column_type == ColumnType.NUMERIC, f"{col.name} should be numeric"

    def test_detect_categorical_columns(self, categorical_csv):
        """Test detection of categorical columns."""
        result = load_csv(categorical_csv)

        color_col = next(c for c in result.columns if c.name == "color")
        size_col = next(c for c in result.columns if c.name == "size")

        assert color_col.column_type == ColumnType.CATEGORICAL
        assert size_col.column_type == ColumnType.CATEGORICAL

    def test_detect_low_cardinality_numeric_as_categorical(self, categorical_csv):
        """Test that low cardinality numeric columns are detected as categorical."""
        result = load_csv(categorical_csv)

        code_col = next(c for c in result.columns if c.name == "code")
        # Low cardinality (2 unique values out of 5) should be categorical
        assert code_col.column_type == ColumnType.CATEGORICAL

    def test_detect_date_columns(self, date_csv):
        """Test detection of date columns."""
        result = load_csv(date_csv)

        for col in result.columns:
            assert col.column_type == ColumnType.DATE, f"{col.name} should be date"

    def test_detect_date_by_column_name(self, sample_csv):
        """Test date detection based on column name hints."""
        result = load_csv(sample_csv)

        hire_date_col = next(c for c in result.columns if c.name == "hire_date")
        assert hire_date_col.column_type == ColumnType.DATE

    def test_detect_boolean_as_categorical(self, sample_csv):
        """Test that boolean columns are detected as categorical."""
        result = load_csv(sample_csv)

        is_active_col = next(c for c in result.columns if c.name == "is_active")
        assert is_active_col.column_type == ColumnType.CATEGORICAL

    def test_detect_mixed_types(self, sample_csv):
        """Test detection in a dataset with mixed column types."""
        result = load_csv(sample_csv)

        type_map = {col.name: col.column_type for col in result.columns}

        assert type_map["id"] == ColumnType.NUMERIC
        assert type_map["name"] == ColumnType.CATEGORICAL
        assert type_map["age"] == ColumnType.NUMERIC
        assert type_map["salary"] == ColumnType.NUMERIC
        assert type_map["department"] == ColumnType.CATEGORICAL
        assert type_map["hire_date"] == ColumnType.DATE
        assert type_map["is_active"] == ColumnType.CATEGORICAL


class TestEncodingHandling:
    """Tests for file encoding handling."""

    def test_load_utf8_csv(self, utf8_csv):
        """Test loading UTF-8 encoded file."""
        result = load_csv(utf8_csv)

        assert result.encoding == "utf-8"
        assert result.row_count == 5

        # Verify special characters loaded correctly
        names = result.dataframe["name"].tolist()
        assert "日本語" in names

    def test_load_latin1_csv(self, latin1_csv):
        """Test loading Latin-1 encoded file."""
        result = load_csv(latin1_csv)

        # Should auto-detect latin-1 encoding
        assert result.encoding in ["latin-1", "cp1252", "iso-8859-1"]
        assert result.row_count == 5

        # Verify special characters loaded correctly
        names = result.dataframe["name"].tolist()
        assert "José" in names
        assert "Müller" in names

    def test_explicit_encoding(self, latin1_csv):
        """Test loading with explicit encoding specified."""
        result = load_csv(latin1_csv, encoding="latin-1")

        assert result.encoding == "latin-1"
        assert result.row_count == 5

    def test_encoding_stored_in_result(self, sample_csv):
        """Test that detected encoding is stored in result."""
        result = load_csv(sample_csv)

        assert result.encoding is not None
        assert result.encoding in DataLoader.ENCODINGS


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_file(self, empty_csv):
        """Test loading an empty CSV (headers only)."""
        result = load_csv(empty_csv)

        assert result.row_count == 0
        assert result.column_count == 3
        assert len(result.columns) == 3

    def test_single_column(self, single_column_csv):
        """Test loading a CSV with a single column."""
        result = load_csv(single_column_csv)

        assert result.column_count == 1
        assert result.columns[0].name == "only_column"
        assert result.columns[0].column_type == ColumnType.NUMERIC

    def test_all_nulls_column(self, nulls_csv):
        """Test handling of columns with all null values."""
        result = load_csv(nulls_csv)

        all_nulls_col = next(c for c in result.columns if c.name == "all_nulls")
        assert all_nulls_col.null_count == 5
        assert all_nulls_col.unique_count == 0
        assert all_nulls_col.sample_values == []

    def test_some_nulls_column(self, nulls_csv):
        """Test handling of columns with some null values."""
        result = load_csv(nulls_csv)

        some_nulls_col = next(c for c in result.columns if c.name == "some_nulls")
        assert some_nulls_col.null_count == 2
        assert some_nulls_col.unique_count == 3

    def test_no_nulls_column(self, nulls_csv):
        """Test handling of columns with no null values."""
        result = load_csv(nulls_csv)

        no_nulls_col = next(c for c in result.columns if c.name == "no_nulls")
        assert no_nulls_col.null_count == 0

    def test_unique_count_tracking(self, sample_csv):
        """Test that unique value counts are tracked correctly."""
        result = load_csv(sample_csv)

        # Department has 3 unique values: HR, IT, Finance
        dept_col = next(c for c in result.columns if c.name == "department")
        assert dept_col.unique_count == 3

    def test_sample_values(self, sample_csv):
        """Test that sample values are captured."""
        result = load_csv(sample_csv)

        name_col = next(c for c in result.columns if c.name == "name")
        assert len(name_col.sample_values) > 0
        assert len(name_col.sample_values) <= 5


class TestPerformance:
    """Performance-related tests."""

    def test_load_larger_file(self, large_csv):
        """Test loading a larger file (10K rows)."""
        result = load_csv(large_csv)

        assert result.row_count == 10000
        assert result.column_count == 3

    def test_column_analysis_performance(self, large_csv):
        """Test that column analysis completes in reasonable time."""
        import time

        start = time.time()
        result = load_csv(large_csv)
        elapsed = time.time() - start

        # Should complete in under 5 seconds for 10K rows
        assert elapsed < 5.0
        assert len(result.columns) == 3
