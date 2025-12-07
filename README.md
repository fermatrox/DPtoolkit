# DPtoolkit - Differential Privacy Toolkit

A privacy-preserving data analysis tool for healthcare datasets, enabling secure sharing and analysis of sensitive patient data using formal differential privacy guarantees.

## Features

- **Multiple File Formats**: Support for CSV, Excel (.xlsx, .xls), and Parquet files
- **Automatic Type Detection**: Intelligent classification of numeric, categorical, and date columns
- **DP Mechanisms**: Laplace (bounded numeric), Gaussian (unbounded numeric), and Exponential (categorical)
- **Flexible Configuration**: Global or per-column epsilon settings with protection modes (Protect, Passthrough, Exclude)
- **Statistical Comparison**: Compare original vs. protected datasets with MAE, RMSE, correlation preservation metrics
- **Interactive Visualizations**: Histogram overlays, correlation heatmaps, and distribution comparisons
- **PDF Reports**: Generate compliance-ready reports documenting privacy measures applied
- **Recommendations Engine**: Automatic sensitivity classification and epsilon recommendations

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/fermatrox/DPtoolkit.git
cd DPtoolkit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation

```bash
# Check that the package is installed
python -c "import dp_toolkit; print(f'DPtoolkit v{dp_toolkit.__version__}')"

# Run tests
pytest tests/unit -v
```

## Quick Start

### Web Interface (Streamlit)

The easiest way to use DPtoolkit is through the web interface:

```bash
streamlit run frontend/app.py
```

This opens a browser with a 4-step wizard:
1. **Upload** - Load your dataset (CSV, Excel, or Parquet)
2. **Configure** - Set protection modes and epsilon values per column
3. **Analyze** - Compare original vs. protected data with visualizations
4. **Export** - Download protected dataset and PDF report

### Python Library

Use DPtoolkit programmatically in your data pipelines:

```python
import pandas as pd
from dp_toolkit.data.loader import load_csv
from dp_toolkit.data.transformer import DatasetTransformer, DatasetConfig, ProtectionMode
from dp_toolkit.analysis.comparator import DatasetComparator

# Load dataset
dataset_info = load_csv("patient_data.csv")
df = dataset_info.dataframe

# Configure protection
config = DatasetConfig(global_epsilon=1.0)
config.set_column_mode("patient_id", ProtectionMode.EXCLUDE)  # Remove sensitive ID
config.set_column_mode("age", ProtectionMode.PROTECT, epsilon=0.5)
config.set_column_mode("diagnosis_code", ProtectionMode.PROTECT, epsilon=1.0)
config.set_column_mode("department", ProtectionMode.PASSTHROUGH)  # Keep as-is

# Apply differential privacy
transformer = DatasetTransformer()
result = transformer.transform(df, config)
protected_df = result.data

# Compare datasets
comparator = DatasetComparator()
comparison = comparator.compare(df, protected_df)

print(f"Total epsilon used: {result.total_epsilon:.2f}")
print(f"Correlation preservation: {comparison.correlation_preservation.preservation_rate:.1%}")

# Export
protected_df.to_csv("protected_patient_data.csv", index=False)
```

## Usage Guide

### Protection Modes

Each column can be configured with one of three modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Protect** | Apply differential privacy noise | Sensitive data you need in output |
| **Passthrough** | Keep original values unchanged | Non-sensitive data (e.g., timestamps) |
| **Exclude** | Remove column from output | Direct identifiers (SSN, name, email) |

### Epsilon (ε) Guidelines

Epsilon controls the privacy-utility tradeoff:

| Epsilon | Privacy Level | Recommendation |
|---------|--------------|----------------|
| 0.1-0.5 | Very High | Highly sensitive data (SSN, diagnoses) |
| 0.5-1.0 | High | Sensitive healthcare data |
| 1.0-2.0 | Moderate | General research use |
| 2.0-5.0 | Lower | Less sensitive aggregated data |
| 5.0-10.0 | Minimal | Internal analysis only |

**Rule of thumb**: For HIPAA/GDPR compliance, keep total epsilon ≤ 5.0 for sensitive healthcare data.

### DP Mechanisms

The toolkit automatically selects the appropriate mechanism based on data type:

- **Laplace**: For bounded numeric columns (age, counts, scores)
- **Gaussian**: For unbounded numeric columns (requires additional δ parameter)
- **Exponential**: For categorical columns (diagnoses, departments, categories)

### Recommendations Engine

Let the toolkit recommend settings based on column names and data patterns:

```python
from dp_toolkit.recommendations.advisor import recommend_for_dataset

df = pd.read_csv("patient_data.csv")
recommendations = recommend_for_dataset(df)

print(f"Recommended total epsilon: {recommendations.total_epsilon:.2f}")
for col, rec in recommendations.column_recommendations.items():
    print(f"  {col}: ε={rec.epsilon_recommendation.epsilon}, "
          f"mechanism={rec.mechanism_recommendation.mechanism.value}")
```

## API Reference

### Core Modules

#### `dp_toolkit.data.loader`
- `load_csv(path)` - Load CSV files with automatic type detection
- `load_excel(path)` - Load Excel files
- `load_parquet(path)` - Load Parquet files

#### `dp_toolkit.data.transformer`
- `DatasetTransformer` - Apply DP transformations to datasets
- `DatasetConfig` - Configuration for dataset transformation
- `ProtectionMode` - Enum: PROTECT, PASSTHROUGH, EXCLUDE

#### `dp_toolkit.core.mechanisms`
- `LaplaceMechanism` - Laplace noise for bounded numeric data
- `GaussianMechanism` - Gaussian noise for (ε,δ)-DP
- `ExponentialMechanism` - Category selection for categorical data

#### `dp_toolkit.analysis.comparator`
- `DatasetComparator` - Compare original and protected datasets
- `DatasetComparison` - Results including MAE, RMSE, correlation preservation

#### `dp_toolkit.recommendations.advisor`
- `RecommendationAdvisor` - Generate privacy recommendations
- `recommend_for_dataset(df)` - Get recommendations for a DataFrame

#### `dp_toolkit.reports.pdf_generator`
- `PDFReportGenerator` - Generate compliance PDF reports

For detailed API documentation, see [docs/api.md](docs/api.md).

## Examples

See the [examples/](examples/) directory for Jupyter notebooks demonstrating:
- Basic usage workflow
- Custom mechanism configuration
- Batch processing multiple files
- Integration with pandas pipelines

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DPTOOLKIT_MAX_ROWS` | Maximum rows to process | 1,000,000 |
| `DPTOOLKIT_MAX_MEMORY_GB` | Memory limit in GB | 4 |

### pytest.ini

Test configuration is in `pytest.ini`. Run specific test suites:

```bash
pytest tests/unit              # Unit tests only
pytest tests/integration       # Integration tests
pytest --cov=dp_toolkit       # With coverage report
```

## Performance

Tested performance targets:

| Operation | Dataset Size | Target Time |
|-----------|-------------|-------------|
| File upload & profiling | 1M rows | < 30 seconds |
| DP transformation | 1M rows × 50 columns | < 60 seconds |
| Comparison analysis | 1M rows | < 30 seconds |
| PDF report generation | Any size | < 20 seconds |

## Project Structure

```
DPtoolkit/
├── dp_toolkit/              # Core library
│   ├── core/               # DP mechanisms, budget tracking
│   ├── data/               # Loading, transformation, export
│   ├── analysis/           # Comparison, divergence, visualization
│   ├── recommendations/    # Sensitivity classification, advisor
│   └── reports/            # PDF generation
├── frontend/               # Streamlit web interface
│   ├── app.py             # Main application
│   ├── pages/             # Wizard pages (upload, configure, analyze, export)
│   └── utils/             # UI utilities
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   └── integration/       # End-to-end tests
├── examples/               # Jupyter notebooks
└── docs/                   # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Run linters (`flake8 dp_toolkit && black dp_toolkit && mypy dp_toolkit`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [OpenDP](https://opendp.org/) for differential privacy mechanisms
- Web interface powered by [Streamlit](https://streamlit.io/)
- Visualizations created with [Plotly](https://plotly.com/)

## Support

- **Issues**: [GitHub Issues](https://github.com/fermatrox/DPtoolkit/issues)
- **Documentation**: See the [docs/](docs/) directory
