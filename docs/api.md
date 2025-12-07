# DPtoolkit API Reference

This document provides detailed API documentation for the `dp_toolkit` library.

## Table of Contents

- [Data Loading (`dp_toolkit.data.loader`)](#data-loading)
- [Data Transformation (`dp_toolkit.data.transformer`)](#data-transformation)
- [Data Export (`dp_toolkit.data.exporter`)](#data-export)
- [DP Mechanisms (`dp_toolkit.core.mechanisms`)](#dp-mechanisms)
- [Privacy Budget (`dp_toolkit.core.budget`)](#privacy-budget)
- [Analysis & Comparison (`dp_toolkit.analysis`)](#analysis--comparison)
- [Recommendations (`dp_toolkit.recommendations`)](#recommendations)
- [PDF Reports (`dp_toolkit.reports`)](#pdf-reports)

---

## Data Loading

### `dp_toolkit.data.loader`

Module for loading datasets from various file formats with automatic type detection.

#### Classes

##### `ColumnType`

Enum for detected column data types.

```python
class ColumnType(Enum):
    NUMERIC = "numeric"       # Continuous numeric values
    CATEGORICAL = "categorical"  # Discrete categories
    DATE = "date"             # Date/datetime values
    UNKNOWN = "unknown"       # Could not determine type
```

##### `ColumnInfo`

Information about a single column.

```python
@dataclass
class ColumnInfo:
    name: str              # Column name
    dtype: str             # Original pandas dtype
    column_type: ColumnType  # Detected DP-relevant type
    null_count: int        # Number of null values
    unique_count: int      # Number of unique values
    sample_values: list    # Sample values for preview
```

##### `DatasetInfo`

Information about a loaded dataset.

```python
@dataclass
class DatasetInfo:
    dataframe: pd.DataFrame   # The loaded data
    file_path: Optional[Path] # Source file path
    file_format: str          # Format (csv, excel, parquet)
    encoding: Optional[str]   # File encoding used
    row_count: int            # Number of rows
    column_count: int         # Number of columns
    columns: list[ColumnInfo] # Column information
    memory_usage_bytes: int   # Memory usage
```

##### `DataLoader`

Main class for loading files.

```python
class DataLoader:
    def load(
        self,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        sheet_name: Optional[Union[str, int]] = None,
        **kwargs
    ) -> DatasetInfo:
        """Load a dataset from file.

        Args:
            file_path: Path to the file
            format: File format (auto-detected if None)
            encoding: File encoding (auto-detected if None)
            sheet_name: Sheet name/index for Excel files
            **kwargs: Additional arguments passed to pandas

        Returns:
            DatasetInfo with loaded data and metadata
        """
```

#### Convenience Functions

```python
def load_csv(
    file_path: Union[str, Path],
    encoding: Optional[str] = None,
    **kwargs
) -> DatasetInfo:
    """Load a CSV file."""

def load_excel(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = None,
    **kwargs
) -> DatasetInfo:
    """Load an Excel file."""

def load_parquet(
    file_path: Union[str, Path],
    **kwargs
) -> DatasetInfo:
    """Load a Parquet file."""
```

---

## Data Transformation

### `dp_toolkit.data.transformer`

Module for applying differential privacy transformations to datasets.

#### Enums

##### `ProtectionMode`

How to handle each column.

```python
class ProtectionMode(Enum):
    PROTECT = "protect"       # Apply DP noise
    PASSTHROUGH = "passthrough"  # Keep original values
    EXCLUDE = "exclude"       # Remove from output
```

##### `MechanismType`

DP mechanism to use.

```python
class MechanismType(Enum):
    LAPLACE = "laplace"       # Laplace mechanism
    GAUSSIAN = "gaussian"     # Gaussian mechanism
    EXPONENTIAL = "exponential"  # Exponential mechanism
```

#### Classes

##### `DatasetColumnConfig`

Configuration for a single column.

```python
@dataclass
class DatasetColumnConfig:
    mode: ProtectionMode = ProtectionMode.PROTECT
    epsilon: Optional[float] = None  # Uses global if None
    delta: Optional[float] = None    # For Gaussian mechanism
    mechanism: Optional[MechanismType] = None  # Auto-detect if None
    lower: Optional[float] = None    # Lower bound for numeric
    upper: Optional[float] = None    # Upper bound for numeric
```

##### `DatasetConfig`

Configuration for entire dataset.

```python
@dataclass
class DatasetConfig:
    global_epsilon: float = 1.0
    global_delta: Optional[float] = None
    column_configs: Dict[str, DatasetColumnConfig]

    def set_column_mode(
        self,
        column: str,
        mode: ProtectionMode,
        epsilon: Optional[float] = None,
        **kwargs
    ) -> None:
        """Set protection mode for a column."""
```

##### `DatasetTransformResult`

Result of transforming a dataset.

```python
@dataclass
class DatasetTransformResult:
    data: pd.DataFrame           # Transformed data
    total_epsilon: float         # Total privacy budget used
    total_delta: Optional[float] # Total delta (if Gaussian used)
    column_results: Dict[str, ColumnTransformResult]
    excluded_columns: List[str]  # Columns removed
    passthrough_columns: List[str]  # Columns kept unchanged
```

##### `DatasetTransformer`

Main class for dataset transformation.

```python
class DatasetTransformer:
    def transform(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        progress_callback: Optional[Callable] = None
    ) -> DatasetTransformResult:
        """Apply differential privacy to a dataset.

        Args:
            df: Input DataFrame
            config: Transformation configuration
            progress_callback: Optional callback for progress updates

        Returns:
            DatasetTransformResult with protected data
        """
```

---

## Data Export

### `dp_toolkit.data.exporter`

Module for exporting protected datasets.

```python
class DataExporter:
    def export_csv(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        **kwargs
    ) -> None:
        """Export to CSV format."""

    def export_excel(
        self,
        transform_result: DatasetTransformResult,
        path: Union[str, Path],
        include_metadata: bool = True,
        **kwargs
    ) -> None:
        """Export to Excel with optional metadata sheet."""

    def export_parquet(
        self,
        df: pd.DataFrame,
        path: Union[str, Path],
        compression: str = "snappy",
        **kwargs
    ) -> None:
        """Export to Parquet format."""
```

---

## DP Mechanisms

### `dp_toolkit.core.mechanisms`

Low-level differential privacy mechanisms wrapping OpenDP.

#### Constants

```python
EPSILON_MIN = 0.01   # Minimum epsilon value
EPSILON_MAX = 100.0  # Maximum epsilon value
DELTA_MIN = 1e-10    # Minimum delta value
DELTA_MAX = 1.0      # Maximum delta value
```

#### Classes

##### `LaplaceMechanism`

Laplace noise for bounded numeric data.

```python
class LaplaceMechanism:
    def __init__(
        self,
        epsilon: float,
        lower: float,
        upper: float
    ):
        """Initialize Laplace mechanism.

        Args:
            epsilon: Privacy parameter
            lower: Lower bound of data
            upper: Upper bound of data
        """

    def add_noise(self, values: np.ndarray) -> np.ndarray:
        """Add Laplace noise to values."""

    @property
    def sensitivity(self) -> float:
        """Return the sensitivity (upper - lower)."""

    @property
    def scale(self) -> float:
        """Return the noise scale (sensitivity / epsilon)."""
```

##### `GaussianMechanism`

Gaussian noise for (ε,δ)-differential privacy.

```python
class GaussianMechanism:
    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float
    ):
        """Initialize Gaussian mechanism.

        Args:
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta
            sensitivity: L2 sensitivity of the query
        """

    def add_noise(self, values: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to values."""

    @property
    def sigma(self) -> float:
        """Return the noise standard deviation."""
```

##### `ExponentialMechanism`

Exponential mechanism for categorical data.

```python
class ExponentialMechanism:
    def __init__(
        self,
        epsilon: float,
        categories: List[Any],
        scores: Optional[np.ndarray] = None
    ):
        """Initialize exponential mechanism.

        Args:
            epsilon: Privacy parameter
            categories: List of possible categories
            scores: Utility scores (frequency-based if None)
        """

    def select(self) -> Any:
        """Select a category with DP."""

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Transform a categorical series with DP."""
```

---

## Privacy Budget

### `dp_toolkit.core.budget`

Privacy budget tracking and composition.

#### Classes

##### `PrivacyBudget`

Represents a privacy budget.

```python
@dataclass
class PrivacyBudget:
    epsilon: float
    delta: float = 0.0

    @property
    def is_pure_dp(self) -> bool:
        """True if this is pure ε-DP (delta = 0)."""

    def __add__(self, other: PrivacyBudget) -> PrivacyBudget:
        """Sequential composition."""

    def __le__(self, other: PrivacyBudget) -> bool:
        """Check if this budget fits within another."""
```

##### `PrivacyBudgetTracker`

Track privacy budget consumption.

```python
class PrivacyBudgetTracker:
    def __init__(
        self,
        total_budget: Optional[PrivacyBudget] = None,
        composition_method: str = "basic"
    ):
        """Initialize tracker.

        Args:
            total_budget: Maximum budget (unlimited if None)
            composition_method: "basic", "advanced", or "parallel"
        """

    def record_query(
        self,
        epsilon: float,
        delta: float = 0.0,
        column: Optional[str] = None,
        mechanism: Optional[str] = None
    ) -> str:
        """Record a privacy-consuming query."""

    @property
    def consumed_budget(self) -> PrivacyBudget:
        """Total budget consumed so far."""

    @property
    def remaining_budget(self) -> Optional[PrivacyBudget]:
        """Remaining budget (None if unlimited)."""

    def can_afford(self, budget: PrivacyBudget) -> bool:
        """Check if we can afford a query."""
```

---

## Analysis & Comparison

### `dp_toolkit.analysis.comparator`

Compare original and protected datasets.

#### Classes

##### `DatasetComparator`

```python
class DatasetComparator:
    def compare(
        self,
        original: pd.DataFrame,
        protected: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> DatasetComparison:
        """Compare original and protected datasets.

        Args:
            original: Original DataFrame
            protected: Protected DataFrame
            numeric_columns: Columns to compare as numeric
            categorical_columns: Columns to compare as categorical

        Returns:
            DatasetComparison with all metrics
        """
```

##### `DatasetComparison`

```python
@dataclass
class DatasetComparison:
    row_count: int
    column_count: int
    numeric_comparisons: List[NumericComparison]
    categorical_comparisons: List[CategoricalComparison]
    date_comparisons: List[DateComparison]
    correlation_preservation: Optional[CorrelationPreservation]
    overall_numeric_mae: Optional[float]
    overall_numeric_rmse: Optional[float]
```

### `dp_toolkit.analysis.divergence`

Statistical divergence metrics.

```python
def calculate_mae(original: np.ndarray, protected: np.ndarray) -> float:
    """Mean Absolute Error."""

def calculate_rmse(original: np.ndarray, protected: np.ndarray) -> float:
    """Root Mean Square Error."""

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence."""

def calculate_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""

def calculate_wasserstein(original: np.ndarray, protected: np.ndarray) -> float:
    """Wasserstein (Earth Mover's) distance."""
```

### `dp_toolkit.analysis.visualizer`

Create comparison visualizations.

```python
def create_histogram_overlay(
    original: pd.Series,
    protected: pd.Series,
    column_name: str,
    bins: int = 50
) -> go.Figure:
    """Create overlaid histogram comparing distributions."""

def create_correlation_comparison(
    original: pd.DataFrame,
    protected: pd.DataFrame
) -> go.Figure:
    """Create side-by-side correlation heatmaps."""

def create_category_bar_chart(
    original: pd.Series,
    protected: pd.Series,
    column_name: str
) -> go.Figure:
    """Create grouped bar chart for categorical comparison."""

def create_box_comparison(
    original: pd.Series,
    protected: pd.Series,
    column_name: str
) -> go.Figure:
    """Create side-by-side box plots."""
```

---

## Recommendations

### `dp_toolkit.recommendations.classifier`

Automatic column sensitivity classification.

```python
class SensitivityLevel(Enum):
    HIGH = "high"       # Direct identifiers, PII
    MEDIUM = "medium"   # Quasi-identifiers
    LOW = "low"         # General attributes
    UNKNOWN = "unknown" # Could not classify
```

```python
class ColumnClassifier:
    def classify(
        self,
        column_name: str,
        series: Optional[pd.Series] = None
    ) -> ClassificationResult:
        """Classify a column's sensitivity level."""
```

### `dp_toolkit.recommendations.advisor`

Recommend privacy parameters.

```python
class RecommendationAdvisor:
    def recommend_for_column(
        self,
        column_name: str,
        series: Optional[pd.Series] = None,
        sensitivity: Optional[SensitivityLevel] = None,
        prefer_privacy: bool = False
    ) -> ColumnRecommendation:
        """Get recommendations for a single column."""

    def recommend_for_dataset(
        self,
        df: pd.DataFrame,
        prefer_privacy: bool = False
    ) -> DatasetRecommendation:
        """Get recommendations for entire dataset."""
```

#### Convenience Functions

```python
def recommend_for_dataset(
    df: pd.DataFrame,
    prefer_privacy: bool = False
) -> DatasetRecommendation:
    """Get recommendations for a DataFrame."""

def recommend_epsilon(
    sensitivity: SensitivityLevel,
    prefer_privacy: bool = False
) -> float:
    """Get recommended epsilon for sensitivity level."""
```

---

## PDF Reports

### `dp_toolkit.reports.pdf_generator`

Generate compliance-ready PDF reports.

```python
@dataclass
class ReportMetadata:
    title: str = "Differential Privacy Analysis Report"
    author: Optional[str] = None
    organization: Optional[str] = None
    original_filename: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class GeneratedReport:
    content: bytes          # PDF content as bytes
    filename: str           # Suggested filename
    page_count: int         # Number of pages
    generation_time: float  # Time to generate (seconds)
```

```python
class PDFReportGenerator:
    def generate(
        self,
        original_df: pd.DataFrame,
        protected_df: pd.DataFrame,
        comparison: DatasetComparison,
        column_configs: Dict[str, Dict[str, Any]],
        metadata: Optional[ReportMetadata] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> GeneratedReport:
        """Generate a comprehensive PDF report.

        Args:
            original_df: Original DataFrame
            protected_df: Protected DataFrame
            comparison: Comparison results
            column_configs: Column configuration used
            metadata: Report metadata
            dataset_info: Additional dataset information

        Returns:
            GeneratedReport with PDF content
        """
```

---

## Error Handling

All modules raise descriptive exceptions:

- `ValueError`: Invalid parameters (epsilon out of range, invalid mode, etc.)
- `FileNotFoundError`: File does not exist
- `RuntimeError`: Operation failed (e.g., transformation error)

Example:

```python
from dp_toolkit.data.transformer import DatasetTransformer, DatasetConfig

try:
    transformer = DatasetTransformer()
    result = transformer.transform(df, config)
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Transformation failed: {e}")
```

---

## Type Hints

All public APIs are fully typed. Use with mypy for static type checking:

```bash
mypy dp_toolkit
```
