"""Data loading, profiling, transformation, and export."""

from dp_toolkit.data.loader import (
    ColumnInfo,
    ColumnType,
    DataLoader,
    DatasetInfo,
    load,
    load_csv,
    load_excel,
    load_parquet,
)
from dp_toolkit.data.transformer import (
    # Enums
    MechanismType,
    TransformColumnType,
    ProtectionMode,
    # Data classes - Column
    ColumnConfig,
    TransformResult,
    # Data classes - Dataset
    DatasetColumnConfig,
    DatasetConfig,
    ColumnTransformSummary,
    DatasetTransformResult,
    # Transformer classes
    ColumnTransformer,
    DatasetTransformer,
    # Convenience functions - Column
    transform_numeric,
    transform_categorical,
    transform_date,
    # Convenience functions - Dataset
    transform_dataset,
)
from dp_toolkit.data.profiler import (
    # Enums
    ProfileType,
    # Numeric profiling
    NumericProfile,
    NumericProfiler,
    profile_numeric,
    profile_numeric_columns,
    # Categorical profiling
    CategoricalProfile,
    CategoricalProfiler,
    profile_categorical,
    profile_categorical_columns,
    # Date profiling
    DateProfile,
    DateProfiler,
    profile_date,
    profile_date_columns,
    # Unified profiling
    ColumnProfile,
    ColumnProfiler,
    profile_column,
    profile_columns,
    # Dataset-level profiling
    MissingValueSummary,
    CorrelationMatrix,
    DatasetProfile,
    DatasetProfiler,
    profile_dataset,
    calculate_missing_summary,
    calculate_correlation_matrix,
)

__all__ = [
    # Loader
    "ColumnInfo",
    "ColumnType",
    "DataLoader",
    "DatasetInfo",
    "load",
    "load_csv",
    "load_excel",
    "load_parquet",
    # Transformer - Column
    "MechanismType",
    "TransformColumnType",
    "ColumnConfig",
    "TransformResult",
    "ColumnTransformer",
    "transform_numeric",
    "transform_categorical",
    "transform_date",
    # Transformer - Dataset
    "ProtectionMode",
    "DatasetColumnConfig",
    "DatasetConfig",
    "ColumnTransformSummary",
    "DatasetTransformResult",
    "DatasetTransformer",
    "transform_dataset",
    # Profiler - Enums
    "ProfileType",
    # Profiler - Numeric
    "NumericProfile",
    "NumericProfiler",
    "profile_numeric",
    "profile_numeric_columns",
    # Profiler - Categorical
    "CategoricalProfile",
    "CategoricalProfiler",
    "profile_categorical",
    "profile_categorical_columns",
    # Profiler - Date
    "DateProfile",
    "DateProfiler",
    "profile_date",
    "profile_date_columns",
    # Profiler - Unified
    "ColumnProfile",
    "ColumnProfiler",
    "profile_column",
    "profile_columns",
    # Profiler - Dataset
    "MissingValueSummary",
    "CorrelationMatrix",
    "DatasetProfile",
    "DatasetProfiler",
    "profile_dataset",
    "calculate_missing_summary",
    "calculate_correlation_matrix",
]
