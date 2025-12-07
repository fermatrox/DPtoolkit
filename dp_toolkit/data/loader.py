"""Data loading module for DPtoolkit.

Provides functionality to load datasets from various file formats
with automatic column type detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class ColumnType(Enum):
    """Detected column data type for DP mechanism selection."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a single column in a dataset.

    Attributes:
        name: Column name.
        dtype: Original pandas dtype.
        column_type: Detected DP-relevant type (numeric, categorical, date).
        null_count: Number of null/missing values.
        unique_count: Number of unique values.
        sample_values: Sample of non-null values for preview.
    """

    name: str
    dtype: str
    column_type: ColumnType
    null_count: int
    unique_count: int
    sample_values: list = field(default_factory=list)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset.

    Attributes:
        dataframe: The loaded pandas DataFrame.
        file_path: Path to the source file.
        file_format: Detected file format (csv, excel, parquet).
        encoding: Detected or used encoding.
        row_count: Number of rows.
        column_count: Number of columns.
        columns: List of ColumnInfo for each column.
        memory_usage_bytes: Approximate memory usage in bytes.
    """

    dataframe: pd.DataFrame
    file_path: Optional[Path]
    file_format: str
    encoding: Optional[str]
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    memory_usage_bytes: int


class DataLoader:
    """Load datasets from files with automatic type detection.

    Supports CSV, Excel, and Parquet files with automatic format detection
    and column type inference for differential privacy mechanism selection.
    """

    # Supported file extensions by format
    FORMAT_EXTENSIONS = {
        "csv": [".csv", ".tsv", ".txt"],
        "excel": [".xlsx", ".xls", ".xlsm", ".xlsb"],
        "parquet": [".parquet", ".pq"],
    }

    # Common encodings to try in order
    ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    # Thresholds for type detection
    CATEGORICAL_UNIQUE_RATIO = 0.05  # If unique/total < 5%, likely categorical
    CATEGORICAL_MAX_UNIQUE = 100  # Max unique values for auto-categorical
    DATE_KEYWORDS = ["date", "time", "timestamp", "created", "updated", "born", "dob"]

    def __init__(self) -> None:
        """Initialize the DataLoader."""
        pass

    def load(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        **kwargs,
    ) -> DatasetInfo:
        """Load a file with automatic format detection.

        Args:
            file_path: Path to the file.
            file_format: Explicit format ('csv', 'excel', 'parquet').
                If None, detect from file extension.
            **kwargs: Additional arguments passed to the format-specific loader.

        Returns:
            DatasetInfo containing the loaded data and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format cannot be detected or is unsupported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format if not specified
        if file_format is None:
            file_format = self._detect_format(file_path)

        # Route to appropriate loader
        if file_format == "csv":
            return self.load_csv(file_path, **kwargs)
        elif file_format == "excel":
            return self.load_excel(file_path, **kwargs)
        elif file_format == "parquet":
            return self.load_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension.

        Args:
            file_path: Path to the file.

        Returns:
            Format string ('csv', 'excel', 'parquet').

        Raises:
            ValueError: If format cannot be detected.
        """
        ext = file_path.suffix.lower()
        for fmt, extensions in self.FORMAT_EXTENSIONS.items():
            if ext in extensions:
                return fmt
        raise ValueError(
            f"Cannot detect format for extension '{ext}'. "
            f"Supported: {list(self.FORMAT_EXTENSIONS.keys())}"
        )

    def load_csv(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        **kwargs,
    ) -> DatasetInfo:
        """Load a CSV file with automatic encoding detection.

        Args:
            file_path: Path to the CSV file.
            encoding: Specific encoding to use. If None, auto-detect.
            **kwargs: Additional arguments passed to pandas.read_csv.

        Returns:
            DatasetInfo containing the loaded data and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed with any encoding.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df, detected_encoding = self._read_csv_with_encoding(
            file_path, encoding, **kwargs
        )

        return self._create_dataset_info(
            df=df,
            file_path=file_path,
            file_format="csv",
            encoding=detected_encoding,
        )

    def load_excel(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[str, int, None] = 0,
        **kwargs,
    ) -> DatasetInfo:
        """Load an Excel file.

        Args:
            file_path: Path to the Excel file.
            sheet_name: Sheet to load. Can be:
                - int: Sheet index (0-based)
                - str: Sheet name
                - None: Load first sheet
            **kwargs: Additional arguments passed to pandas.read_excel.

        Returns:
            DatasetInfo containing the loaded data and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the sheet does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

        return self._create_dataset_info(
            df=df,
            file_path=file_path,
            file_format="excel",
            encoding=None,  # Excel files don't have text encoding
        )

    def get_excel_sheet_names(self, file_path: Union[str, Path]) -> list[str]:
        """Get list of sheet names in an Excel file.

        Args:
            file_path: Path to the Excel file.

        Returns:
            List of sheet names.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with pd.ExcelFile(file_path) as excel_file:
            return list(excel_file.sheet_names)

    def load_parquet(
        self,
        file_path: Union[str, Path],
        **kwargs,
    ) -> DatasetInfo:
        """Load a Parquet file.

        Args:
            file_path: Path to the Parquet file.
            **kwargs: Additional arguments passed to pandas.read_parquet.

        Returns:
            DatasetInfo containing the loaded data and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_parquet(file_path, **kwargs)

        return self._create_dataset_info(
            df=df,
            file_path=file_path,
            file_format="parquet",
            encoding=None,  # Parquet is binary
        )

    def _read_csv_with_encoding(
        self,
        file_path: Path,
        encoding: Optional[str],
        **kwargs,
    ) -> tuple[pd.DataFrame, str]:
        """Read CSV file, trying multiple encodings if needed.

        Args:
            file_path: Path to the CSV file.
            encoding: Specific encoding or None for auto-detection.
            **kwargs: Additional pandas.read_csv arguments.

        Returns:
            Tuple of (DataFrame, encoding used).

        Raises:
            ValueError: If no encoding works.
        """
        if encoding is not None:
            # Use specified encoding
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            return df, encoding

        # Try encodings in order
        errors = []
        for enc in self.ENCODINGS:
            try:
                df = pd.read_csv(file_path, encoding=enc, **kwargs)
                return df, enc
            except (UnicodeDecodeError, UnicodeError) as e:
                errors.append(f"{enc}: {e}")
            except Exception:
                # Other errors (like parsing) should be raised
                raise

        raise ValueError(
            f"Could not read file with any encoding. Tried: {', '.join(self.ENCODINGS)}"
        )

    def _create_dataset_info(
        self,
        df: pd.DataFrame,
        file_path: Optional[Path],
        file_format: str,
        encoding: Optional[str],
    ) -> DatasetInfo:
        """Create DatasetInfo from a DataFrame.

        Args:
            df: The loaded DataFrame.
            file_path: Source file path.
            file_format: Format string (csv, excel, parquet).
            encoding: Encoding used.

        Returns:
            DatasetInfo with full metadata.
        """
        columns = [self._analyze_column(df, col) for col in df.columns]

        return DatasetInfo(
            dataframe=df,
            file_path=file_path,
            file_format=file_format,
            encoding=encoding,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            memory_usage_bytes=df.memory_usage(deep=True).sum(),
        )

    def _analyze_column(self, df: pd.DataFrame, column: str) -> ColumnInfo:
        """Analyze a single column and detect its type.

        Args:
            df: The DataFrame containing the column.
            column: Column name to analyze.

        Returns:
            ColumnInfo with detected type and statistics.
        """
        series = df[column]
        dtype_str = str(series.dtype)
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique())

        # Get sample values (up to 5 non-null unique values)
        non_null = series.dropna()
        sample_values = non_null.unique()[:5].tolist() if len(non_null) > 0 else []

        # Detect column type
        column_type = self._detect_column_type(series, column)

        return ColumnInfo(
            name=column,
            dtype=dtype_str,
            column_type=column_type,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=sample_values,
        )

    def _detect_column_type(self, series: pd.Series, column_name: str) -> ColumnType:
        """Detect the DP-relevant type of a column.

        Uses dtype, column name hints, and value distribution to determine
        whether a column should be treated as numeric, categorical, or date.

        Args:
            series: The pandas Series to analyze.
            column_name: Name of the column (for keyword hints).

        Returns:
            ColumnType indicating the detected type.
        """
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATE

        # Check column name for date hints
        col_lower = column_name.lower()
        if any(keyword in col_lower for keyword in self.DATE_KEYWORDS):
            # Try to parse as date
            if self._can_parse_as_date(series):
                return ColumnType.DATE

        # Check for boolean FIRST (before numeric, as bool is considered numeric)
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.CATEGORICAL

        # Check numeric types
        if pd.api.types.is_numeric_dtype(series):
            # Even numeric columns might be categorical (e.g., codes)
            non_null_count = series.notna().sum()
            if non_null_count > 0:
                unique_count = series.nunique()
                unique_ratio = unique_count / non_null_count

                # Low cardinality numeric = likely categorical
                # Must have low unique ratio AND reasonable number of values
                # Also require min dataset size to avoid false positives
                if (
                    unique_ratio < self.CATEGORICAL_UNIQUE_RATIO
                    and unique_count <= self.CATEGORICAL_MAX_UNIQUE
                    and non_null_count >= 20  # Minimum rows for reliable detection
                ):
                    return ColumnType.CATEGORICAL

                # Very small number of unique values in larger dataset = categorical
                if unique_count <= 5 and non_null_count >= 50:
                    return ColumnType.CATEGORICAL

            return ColumnType.NUMERIC

        # Check if object/string type might be date
        if series.dtype == object:
            if self._can_parse_as_date(series):
                return ColumnType.DATE

        # Object/string types - categorical
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            return ColumnType.CATEGORICAL

        # Category dtype
        if pd.api.types.is_categorical_dtype(series):
            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _can_parse_as_date(self, series: pd.Series, sample_size: int = 100) -> bool:
        """Check if a series can be parsed as dates.

        Args:
            series: The series to check.
            sample_size: Number of values to sample for testing.

        Returns:
            True if the series appears to contain dates.
        """
        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        # Sample for performance
        sample = non_null.head(sample_size)

        try:
            # Try parsing - let pandas infer the format
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            # If more than 80% parse successfully, consider it a date column
            success_rate = float(parsed.notna().sum()) / len(sample)
            return bool(success_rate >= 0.8)
        except (ValueError, TypeError, OverflowError):
            # Parsing failed - not a date column
            return False


def load(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs,
) -> DatasetInfo:
    """Convenience function to load a file with automatic format detection.

    Args:
        file_path: Path to the file.
        file_format: Explicit format ('csv', 'excel', 'parquet').
            If None, detect from file extension.
        **kwargs: Additional arguments passed to the format-specific loader.

    Returns:
        DatasetInfo containing the loaded data and metadata.
    """
    loader = DataLoader()
    return loader.load(file_path, file_format=file_format, **kwargs)


def load_csv(
    file_path: Union[str, Path],
    encoding: Optional[str] = None,
    **kwargs,
) -> DatasetInfo:
    """Convenience function to load a CSV file.

    Args:
        file_path: Path to the CSV file.
        encoding: Specific encoding to use. If None, auto-detect.
        **kwargs: Additional arguments passed to pandas.read_csv.

    Returns:
        DatasetInfo containing the loaded data and metadata.
    """
    loader = DataLoader()
    return loader.load_csv(file_path, encoding=encoding, **kwargs)


def load_excel(
    file_path: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
    **kwargs,
) -> DatasetInfo:
    """Convenience function to load an Excel file.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Sheet to load (index, name, or None for first).
        **kwargs: Additional arguments passed to pandas.read_excel.

    Returns:
        DatasetInfo containing the loaded data and metadata.
    """
    loader = DataLoader()
    return loader.load_excel(file_path, sheet_name=sheet_name, **kwargs)


def load_parquet(
    file_path: Union[str, Path],
    **kwargs,
) -> DatasetInfo:
    """Convenience function to load a Parquet file.

    Args:
        file_path: Path to the Parquet file.
        **kwargs: Additional arguments passed to pandas.read_parquet.

    Returns:
        DatasetInfo containing the loaded data and metadata.
    """
    loader = DataLoader()
    return loader.load_parquet(file_path, **kwargs)
