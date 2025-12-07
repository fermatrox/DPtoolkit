"""Data exporter for differential privacy protected datasets.

This module provides export functionality for protected datasets,
supporting multiple formats:
- CSV: Comma-separated values with optional metadata sidecar
- Excel: XLSX with optional metadata sheet
- Parquet: Columnar format with embedded metadata

Key features:
- Preserve column order and data types
- Include privacy metadata (epsilon, delta, mechanisms used)
- Round-trip integrity with loader module
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from dp_toolkit.data.transformer import DatasetTransformResult


# =============================================================================
# Enums and Constants
# =============================================================================


class ExportFormat(Enum):
    """Supported export formats."""

    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"


# Default encoding for CSV files
DEFAULT_ENCODING = "utf-8"

# Metadata file suffix
METADATA_SUFFIX = "_metadata.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExportMetadata:
    """Metadata about an exported dataset.

    Attributes:
        export_timestamp: When the export was created.
        source_row_count: Number of rows in source data.
        source_column_count: Number of columns in source data.
        exported_column_count: Number of columns in export.
        total_epsilon: Total privacy budget consumed.
        total_delta: Total delta consumed (if applicable).
        protected_columns: List of columns that were protected.
        passthrough_columns: List of columns passed through unchanged.
        excluded_columns: List of columns excluded from export.
        column_metadata: Per-column metadata (mechanism, epsilon, etc.).
        format: Export format used.
        encoding: Character encoding (for CSV).
        version: Exporter version.
    """

    export_timestamp: str
    source_row_count: int
    source_column_count: int
    exported_column_count: int
    total_epsilon: float
    total_delta: Optional[float]
    protected_columns: List[str]
    passthrough_columns: List[str]
    excluded_columns: List[str]
    column_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    format: str = "csv"
    encoding: str = DEFAULT_ENCODING
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportMetadata":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ExportMetadata":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_transform_result(
        cls,
        result: DatasetTransformResult,
        format: str = "csv",
        encoding: str = DEFAULT_ENCODING,
    ) -> "ExportMetadata":
        """Create metadata from a DatasetTransformResult.

        Args:
            result: Transform result to extract metadata from.
            format: Export format being used.
            encoding: Character encoding.

        Returns:
            ExportMetadata instance.
        """
        # Build per-column metadata
        column_metadata = {}
        for col_name, summary in result.column_summaries.items():
            col_meta: Dict[str, Any] = {
                "mode": summary.mode.value,
                "null_count": summary.null_count,
            }
            if summary.mechanism_type is not None:
                col_meta["mechanism"] = summary.mechanism_type.value
            if summary.epsilon is not None:
                col_meta["epsilon"] = summary.epsilon
            if summary.delta is not None:
                col_meta["delta"] = summary.delta
            if summary.error is not None:
                col_meta["error"] = summary.error
            column_metadata[col_name] = col_meta

        return cls(
            export_timestamp=datetime.now().isoformat(),
            source_row_count=result.row_count,
            source_column_count=(
                len(result.protected_columns)
                + len(result.passthrough_columns)
                + len(result.excluded_columns)
            ),
            exported_column_count=result.column_count,
            total_epsilon=result.total_epsilon,
            total_delta=result.total_delta,
            protected_columns=result.protected_columns,
            passthrough_columns=result.passthrough_columns,
            excluded_columns=result.excluded_columns,
            column_metadata=column_metadata,
            format=format,
            encoding=encoding,
        )


@dataclass
class ExportResult:
    """Result of an export operation.

    Attributes:
        path: Path to the exported data file.
        metadata_path: Path to metadata file (if created).
        format: Export format used.
        row_count: Number of rows exported.
        column_count: Number of columns exported.
        file_size_bytes: Size of exported file in bytes.
        metadata: Export metadata.
    """

    path: Path
    metadata_path: Optional[Path]
    format: ExportFormat
    row_count: int
    column_count: int
    file_size_bytes: int
    metadata: ExportMetadata


# =============================================================================
# Data Exporter
# =============================================================================


class DataExporter:
    """Export protected datasets to various formats.

    Supports CSV, Excel, and Parquet formats with optional metadata.

    Example:
        >>> exporter = DataExporter()
        >>> result = exporter.export_csv(
        ...     df=protected_df,
        ...     path="output.csv",
        ...     transform_result=transform_result,
        ... )
        >>> print(f"Exported to {result.path}")

    Attributes:
        default_encoding: Default encoding for CSV exports.
    """

    def __init__(self, default_encoding: str = DEFAULT_ENCODING) -> None:
        """Initialize the exporter.

        Args:
            default_encoding: Default encoding for CSV files.
        """
        self._default_encoding = default_encoding

    @property
    def default_encoding(self) -> str:
        """Get default encoding."""
        return self._default_encoding

    # -------------------------------------------------------------------------
    # CSV Export
    # -------------------------------------------------------------------------

    def export_csv(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        transform_result: Optional[DatasetTransformResult] = None,
        include_metadata: bool = True,
        encoding: Optional[str] = None,
        index: bool = False,
        **csv_kwargs: Any,
    ) -> ExportResult:
        """Export DataFrame to CSV format.

        Args:
            df: DataFrame to export.
            path: Output file path.
            transform_result: Optional transform result for metadata.
            include_metadata: Whether to write metadata sidecar file.
            encoding: Character encoding (uses default if None).
            index: Whether to include row index.
            **csv_kwargs: Additional arguments for pandas to_csv().

        Returns:
            ExportResult with export details.
        """
        path = Path(path)
        encoding = encoding or self._default_encoding

        # Export data
        df.to_csv(path, encoding=encoding, index=index, **csv_kwargs)

        # Build metadata
        if transform_result is not None:
            metadata = ExportMetadata.from_transform_result(
                result=transform_result,
                format="csv",
                encoding=encoding,
            )
        else:
            metadata = self._create_basic_metadata(
                df=df,
                format="csv",
                encoding=encoding,
            )

        # Write metadata sidecar
        metadata_path = None
        if include_metadata:
            metadata_path = path.with_suffix(path.suffix + METADATA_SUFFIX)
            metadata_path.write_text(metadata.to_json(), encoding="utf-8")

        return ExportResult(
            path=path,
            metadata_path=metadata_path,
            format=ExportFormat.CSV,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Excel Export
    # -------------------------------------------------------------------------

    def export_excel(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        transform_result: Optional[DatasetTransformResult] = None,
        include_metadata: bool = True,
        sheet_name: str = "Data",
        metadata_sheet_name: str = "Metadata",
        index: bool = False,
        **excel_kwargs: Any,
    ) -> ExportResult:
        """Export DataFrame to Excel format.

        Args:
            df: DataFrame to export.
            path: Output file path.
            transform_result: Optional transform result for metadata.
            include_metadata: Whether to include metadata sheet.
            sheet_name: Name for the data sheet.
            metadata_sheet_name: Name for the metadata sheet.
            index: Whether to include row index.
            **excel_kwargs: Additional arguments for pandas to_excel().

        Returns:
            ExportResult with export details.
        """
        path = Path(path)

        # Build metadata
        if transform_result is not None:
            metadata = ExportMetadata.from_transform_result(
                result=transform_result,
                format="excel",
            )
        else:
            metadata = self._create_basic_metadata(df=df, format="excel")

        # Export with or without metadata sheet
        if include_metadata:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index, **excel_kwargs)
                # Create metadata DataFrame
                metadata_df = self._metadata_to_dataframe(metadata)
                metadata_df.to_excel(
                    writer, sheet_name=metadata_sheet_name, index=False
                )
        else:
            df.to_excel(path, sheet_name=sheet_name, index=index, **excel_kwargs)

        return ExportResult(
            path=path,
            metadata_path=None,  # Metadata is embedded in Excel
            format=ExportFormat.EXCEL,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Parquet Export
    # -------------------------------------------------------------------------

    def export_parquet(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        transform_result: Optional[DatasetTransformResult] = None,
        include_metadata: bool = True,
        compression: str = "snappy",
        index: bool = False,
        **parquet_kwargs: Any,
    ) -> ExportResult:
        """Export DataFrame to Parquet format.

        Metadata is embedded in the Parquet file's schema metadata.

        Args:
            df: DataFrame to export.
            path: Output file path.
            transform_result: Optional transform result for metadata.
            include_metadata: Whether to embed metadata in file.
            compression: Compression codec (snappy, gzip, brotli, none).
            index: Whether to include row index.
            **parquet_kwargs: Additional arguments for pandas to_parquet().

        Returns:
            ExportResult with export details.
        """
        path = Path(path)

        # Build metadata
        if transform_result is not None:
            metadata = ExportMetadata.from_transform_result(
                result=transform_result,
                format="parquet",
            )
        else:
            metadata = self._create_basic_metadata(df=df, format="parquet")

        # For Parquet, we embed metadata in the schema
        if include_metadata:
            # PyArrow allows custom metadata
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=index)

            # Add custom metadata
            custom_meta = {b"dp_toolkit_metadata": metadata.to_json().encode("utf-8")}
            existing_meta = table.schema.metadata or {}
            merged_meta = {**existing_meta, **custom_meta}
            table = table.replace_schema_metadata(merged_meta)

            pq.write_table(table, path, compression=compression, **parquet_kwargs)
        else:
            df.to_parquet(
                path, compression=compression, index=index, **parquet_kwargs
            )

        return ExportResult(
            path=path,
            metadata_path=None,  # Metadata is embedded
            format=ExportFormat.PARQUET,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=path.stat().st_size,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Generic Export
    # -------------------------------------------------------------------------

    def export(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        format: Optional[ExportFormat] = None,
        transform_result: Optional[DatasetTransformResult] = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> ExportResult:
        """Export DataFrame with automatic format detection.

        Format is detected from file extension if not specified.

        Args:
            df: DataFrame to export.
            path: Output file path.
            format: Export format (auto-detected if None).
            transform_result: Optional transform result for metadata.
            include_metadata: Whether to include metadata.
            **kwargs: Format-specific arguments.

        Returns:
            ExportResult with export details.

        Raises:
            ValueError: If format cannot be determined.
        """
        path = Path(path)

        if format is None:
            format = self._detect_format(path)

        if format == ExportFormat.CSV:
            return self.export_csv(
                df=df,
                path=path,
                transform_result=transform_result,
                include_metadata=include_metadata,
                **kwargs,
            )
        elif format == ExportFormat.EXCEL:
            return self.export_excel(
                df=df,
                path=path,
                transform_result=transform_result,
                include_metadata=include_metadata,
                **kwargs,
            )
        elif format == ExportFormat.PARQUET:
            return self.export_parquet(
                df=df,
                path=path,
                transform_result=transform_result,
                include_metadata=include_metadata,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _detect_format(self, path: Path) -> ExportFormat:
        """Detect export format from file extension.

        Args:
            path: File path.

        Returns:
            Detected ExportFormat.

        Raises:
            ValueError: If extension is not recognized.
        """
        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv", ".txt"):
            return ExportFormat.CSV
        elif suffix in (".xlsx", ".xls", ".xlsm"):
            return ExportFormat.EXCEL
        elif suffix in (".parquet", ".pq"):
            return ExportFormat.PARQUET
        else:
            raise ValueError(
                f"Cannot determine format from extension '{suffix}'. "
                f"Supported: .csv, .xlsx, .parquet"
            )

    def _create_basic_metadata(
        self,
        df: pd.DataFrame,
        format: str,
        encoding: str = DEFAULT_ENCODING,
    ) -> ExportMetadata:
        """Create basic metadata when no transform result is available.

        Args:
            df: DataFrame being exported.
            format: Export format.
            encoding: Character encoding.

        Returns:
            Basic ExportMetadata.
        """
        return ExportMetadata(
            export_timestamp=datetime.now().isoformat(),
            source_row_count=len(df),
            source_column_count=len(df.columns),
            exported_column_count=len(df.columns),
            total_epsilon=0.0,
            total_delta=None,
            protected_columns=[],
            passthrough_columns=list(df.columns),
            excluded_columns=[],
            format=format,
            encoding=encoding,
        )

    def _metadata_to_dataframe(self, metadata: ExportMetadata) -> pd.DataFrame:
        """Convert metadata to DataFrame for Excel export.

        Args:
            metadata: Export metadata.

        Returns:
            DataFrame with metadata in key-value format.
        """
        rows = [
            ("Export Timestamp", metadata.export_timestamp),
            ("Source Row Count", metadata.source_row_count),
            ("Source Column Count", metadata.source_column_count),
            ("Exported Column Count", metadata.exported_column_count),
            ("Total Epsilon", metadata.total_epsilon),
            ("Total Delta", metadata.total_delta or "N/A"),
            ("Protected Columns", ", ".join(metadata.protected_columns) or "None"),
            ("Passthrough Columns", ", ".join(metadata.passthrough_columns) or "None"),
            ("Excluded Columns", ", ".join(metadata.excluded_columns) or "None"),
            ("Format", metadata.format),
            ("Version", metadata.version),
        ]

        return pd.DataFrame(rows, columns=["Property", "Value"])


# =============================================================================
# Metadata Reader
# =============================================================================


def read_export_metadata(path: Union[str, Path]) -> Optional[ExportMetadata]:
    """Read export metadata from a file.

    Supports:
    - CSV: Reads sidecar JSON file
    - Excel: Reads Metadata sheet
    - Parquet: Reads embedded metadata

    Args:
        path: Path to exported data file.

    Returns:
        ExportMetadata if found, None otherwise.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        # Look for sidecar file
        metadata_path = path.with_suffix(path.suffix + METADATA_SUFFIX)
        if metadata_path.exists():
            return ExportMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
        return None

    elif suffix in (".xlsx", ".xls", ".xlsm"):
        # Try to read Metadata sheet
        try:
            with pd.ExcelFile(path) as xl:
                if "Metadata" in xl.sheet_names:
                    meta_df = pd.read_excel(xl, sheet_name="Metadata")
                    # Convert back from key-value format
                    meta_dict = dict(zip(meta_df["Property"], meta_df["Value"]))

                    # Helper to parse column lists (handles NaN and "None")
                    def parse_column_list(value: Any) -> List[str]:
                        if pd.isna(value) or value in ("None", "", None):
                            return []
                        return str(value).split(", ")

                    # Helper to parse delta (handles "N/A" and NaN)
                    def parse_delta(value: Any) -> Optional[float]:
                        if pd.isna(value) or value == "N/A":
                            return None
                        return float(value)

                    return ExportMetadata(
                        export_timestamp=str(meta_dict.get("Export Timestamp", "")),
                        source_row_count=int(meta_dict.get("Source Row Count", 0)),
                        source_column_count=int(
                            meta_dict.get("Source Column Count", 0)
                        ),
                        exported_column_count=int(
                            meta_dict.get("Exported Column Count", 0)
                        ),
                        total_epsilon=float(meta_dict.get("Total Epsilon", 0)),
                        total_delta=parse_delta(meta_dict.get("Total Delta")),
                        protected_columns=parse_column_list(
                            meta_dict.get("Protected Columns")
                        ),
                        passthrough_columns=parse_column_list(
                            meta_dict.get("Passthrough Columns")
                        ),
                        excluded_columns=parse_column_list(
                            meta_dict.get("Excluded Columns")
                        ),
                        format=str(meta_dict.get("Format", "excel")),
                    )
        except (KeyError, ValueError, TypeError):
            # Metadata parsing failed - file may not have valid metadata
            pass
        return None

    elif suffix in (".parquet", ".pq"):
        # Read embedded metadata
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            schema_meta = parquet_file.schema_arrow.metadata
            if schema_meta and b"dp_toolkit_metadata" in schema_meta:
                json_str = schema_meta[b"dp_toolkit_metadata"].decode("utf-8")
                return ExportMetadata.from_json(json_str)
        except (ImportError, KeyError, ValueError, OSError):
            # Metadata reading failed - pyarrow not available or invalid file
            pass
        return None

    return None


# =============================================================================
# Convenience Functions
# =============================================================================


def export_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    transform_result: Optional[DatasetTransformResult] = None,
    include_metadata: bool = True,
    **kwargs: Any,
) -> ExportResult:
    """Export DataFrame to CSV format.

    Convenience function for quick exports.

    Args:
        df: DataFrame to export.
        path: Output file path.
        transform_result: Optional transform result for metadata.
        include_metadata: Whether to write metadata sidecar file.
        **kwargs: Additional arguments for pandas to_csv().

    Returns:
        ExportResult with export details.
    """
    exporter = DataExporter()
    return exporter.export_csv(
        df=df,
        path=path,
        transform_result=transform_result,
        include_metadata=include_metadata,
        **kwargs,
    )


def export_excel(
    df: pd.DataFrame,
    path: Union[str, Path],
    transform_result: Optional[DatasetTransformResult] = None,
    include_metadata: bool = True,
    **kwargs: Any,
) -> ExportResult:
    """Export DataFrame to Excel format.

    Convenience function for quick exports.

    Args:
        df: DataFrame to export.
        path: Output file path.
        transform_result: Optional transform result for metadata.
        include_metadata: Whether to include metadata sheet.
        **kwargs: Additional arguments for pandas to_excel().

    Returns:
        ExportResult with export details.
    """
    exporter = DataExporter()
    return exporter.export_excel(
        df=df,
        path=path,
        transform_result=transform_result,
        include_metadata=include_metadata,
        **kwargs,
    )


def export_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    transform_result: Optional[DatasetTransformResult] = None,
    include_metadata: bool = True,
    **kwargs: Any,
) -> ExportResult:
    """Export DataFrame to Parquet format.

    Convenience function for quick exports.

    Args:
        df: DataFrame to export.
        path: Output file path.
        transform_result: Optional transform result for metadata.
        include_metadata: Whether to embed metadata.
        **kwargs: Additional arguments for pandas to_parquet().

    Returns:
        ExportResult with export details.
    """
    exporter = DataExporter()
    return exporter.export_parquet(
        df=df,
        path=path,
        transform_result=transform_result,
        include_metadata=include_metadata,
        **kwargs,
    )


def export_data(
    df: pd.DataFrame,
    path: Union[str, Path],
    transform_result: Optional[DatasetTransformResult] = None,
    include_metadata: bool = True,
    **kwargs: Any,
) -> ExportResult:
    """Export DataFrame with automatic format detection.

    Convenience function for quick exports.

    Args:
        df: DataFrame to export.
        path: Output file path.
        transform_result: Optional transform result for metadata.
        include_metadata: Whether to include metadata.
        **kwargs: Format-specific arguments.

    Returns:
        ExportResult with export details.
    """
    exporter = DataExporter()
    return exporter.export(
        df=df,
        path=path,
        transform_result=transform_result,
        include_metadata=include_metadata,
        **kwargs,
    )
