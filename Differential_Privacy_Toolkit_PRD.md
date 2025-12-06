# Product Requirements Document
## Differential Privacy Toolkit
### Privacy-Preserving Data Analysis for Healthcare

**Version:** 1.0  
**Date:** December 2024  
**Status:** Draft

---

## Document Control

| Field | Value |
|-------|-------|
| Document Owner | [Product Owner Name] |
| Technical Lead | [Technical Lead Name] |
| Target Users | Data Scientists, Data Analysts |
| Compliance | GDPR, HIPAA |

---

## 1. Executive Summary

This document defines the requirements for a Differential Privacy Toolkit designed to enable data scientists and analysts to apply differential privacy mechanisms to healthcare datasets. The tool addresses the critical need to share and analyze sensitive healthcare data while maintaining compliance with GDPR and HIPAA regulations.

The toolkit allows users to upload datasets, apply differential privacy noise with configurable epsilon values, compare statistical properties between original and protected datasets, and export both the privacy-protected data and comprehensive analysis reports.

The architecture separates backend logic from the frontend interface, enabling the differential privacy engine to be reused across multiple projects and integrated into data pipelines.

---

## 2. Problem Statement

Healthcare organizations face a fundamental tension: they need to share and analyze patient data for research, quality improvement, and operational purposes, but must protect individual patient privacy under strict regulatory requirements.

Current approaches often involve either complete data restriction (preventing valuable analysis) or ad-hoc anonymization techniques that provide no formal privacy guarantees and may be vulnerable to re-identification attacks.

Differential privacy provides a mathematically rigorous solution, but existing implementations are often too complex for data analysts to use effectively, or lack the visualization tools needed to understand the privacy-utility tradeoff.

---

## 3. Objectives

1. Provide a user-friendly interface for applying differential privacy to healthcare datasets without requiring deep expertise in privacy-preserving techniques.
2. Enable users to understand and visualize the impact of privacy parameters (epsilon) on data utility through side-by-side statistical comparisons.
3. Support compliance documentation by generating exportable reports showing privacy measures applied and their effects.
4. Create a reusable backend library that can be integrated into other data processing pipelines and applications.
5. Guide users toward appropriate privacy settings through intelligent recommendations based on data sensitivity heuristics.

---

## 4. Scope

### 4.1 In Scope

1. File upload support for CSV, Excel (.xlsx, .xls), and Parquet formats.
2. Support for numeric, categorical, and date column types.
3. Differential privacy mechanisms: Laplace, Gaussian, and Exponential.
4. Per-column or global epsilon configuration.
5. Column-level protection marking (protect vs. passthrough).
6. Statistical profiling and comparison visualizations.
7. Export of privacy-protected datasets and PDF reports.
8. Datasets up to 1 million rows.

### 4.2 Out of Scope

1. Free-text field processing (NLP-based anonymization).
2. Database connections (only file upload supported).
3. User authentication and access control.
4. Session persistence between application restarts.
5. Real-time streaming data processing.
6. Federated learning or distributed privacy.

---

## 5. Functional Requirements

### 5.1 FR1: Data Ingestion

**Description:** The system shall accept dataset uploads and perform initial data validation and profiling.

**Requirements:**

1. Accept file uploads in CSV, Excel (.xlsx, .xls), and Parquet formats.
2. Automatically detect column data types (numeric, categorical, date).
3. Validate file structure and report parsing errors with actionable messages.
4. Display dataset preview (first 100 rows) and basic metadata (row count, column count, memory usage).
5. Handle datasets up to 1 million rows with appropriate memory management.
6. Support UTF-8 and common encodings; detect and handle encoding issues gracefully.

### 5.2 FR2: Statistical Profiling

**Description:** The system shall generate comprehensive statistical profiles for both original and privacy-protected datasets.

**Requirements:**

1. Generate statistical summaries appropriate to column data type (see Section 6.1 for detailed metrics).
2. Compute correlation matrices for numeric columns.
3. Detect and report missing values, outliers, and data quality issues.
4. Calculate divergence metrics between original and protected datasets (see Section 6.2).
5. Profile generation shall complete within 60 seconds for datasets up to 1 million rows.

### 5.3 FR3: Column Configuration

**Description:** The system shall allow users to configure privacy settings at the column level.

**Requirements:**

1. Allow users to mark each column as: Protected (apply DP), Passthrough (keep unchanged), or Exclude (remove from output).
2. Support per-column epsilon values or a single global epsilon applied to all protected columns.
3. Allow users to select DP mechanism per column (Laplace, Gaussian, Exponential) with system recommendations.
4. Provide sensitivity classification heuristics based on column name patterns and content analysis.
5. Display recommended epsilon values with explanations; allow user override.
6. Validate configuration before processing (e.g., warn if mechanism incompatible with data type).

### 5.4 FR4: Differential Privacy Engine

**Description:** The system shall apply differential privacy mechanisms to generate protected datasets.

**Requirements:**

1. Implement Laplace mechanism for numeric data with bounded sensitivity.
2. Implement Gaussian mechanism for numeric data requiring (ε, δ)-differential privacy.
3. Implement Exponential mechanism for categorical data.
4. Use OpenDP library (MIT License) as the core privacy engine.
5. Automatically compute sensitivity bounds from data or accept user-specified bounds.
6. Handle date columns by converting to numeric epoch values, applying DP, and converting back.
7. Preserve data types in output (e.g., integer columns remain integers after noise addition with rounding).
8. Track and report total privacy budget consumed across all columns.

### 5.5 FR5: Comparison and Visualization

**Description:** The system shall provide visual comparison between original and protected datasets.

**Requirements:**

1. Display side-by-side statistical summary tables for original vs. protected data.
2. Generate overlay histograms comparing distributions before and after DP application.
3. Display correlation matrix heatmaps for both datasets with difference highlighting.
4. Provide interactive epsilon sensitivity analysis showing utility degradation across epsilon values.
5. Show per-column impact metrics with visual indicators (green/yellow/red for utility preservation).
6. Support toggling between individual column views and dataset-wide summary views.

### 5.6 FR6: Export Functionality

**Description:** The system shall export protected datasets and comprehensive analysis reports.

**Requirements:**

1. Export protected dataset in CSV, Excel, or Parquet format (matching input format by default).
2. Generate PDF reports containing: executive summary, configuration details, statistical comparisons, visualizations, and privacy budget accounting.
3. Include metadata in exports documenting privacy parameters applied to each column.
4. Provide downloadable report suitable for compliance documentation and audit trails.

### 5.7 FR7: Recommendations Engine

**Description:** The system shall provide intelligent recommendations for privacy settings.

**Requirements:**

1. Classify column sensitivity using heuristics based on column names (e.g., 'ssn', 'diagnosis', 'dob' → high sensitivity; 'department', 'visit_type' → medium; 'record_id' → low).
2. Recommend epsilon values based on sensitivity classification: High (ε = 0.1–0.5), Medium (ε = 0.5–2.0), Low (ε = 2.0–5.0).
3. Recommend appropriate DP mechanism based on data type: Laplace for bounded numeric, Gaussian for unbounded numeric, Exponential for categorical.
4. Provide explanatory tooltips for each recommendation explaining the rationale.
5. Allow users to override any recommendation while logging the deviation.

---

## 6. Statistical Specifications

### 6.1 Statistical Metrics by Data Type

#### Numeric Columns

- Count, count of non-null values, count of missing values, missing percentage
- Mean, median, mode, standard deviation, variance
- Minimum, maximum, range
- Percentiles: 1st, 5th, 25th (Q1), 50th (median), 75th (Q3), 95th, 99th
- Interquartile range (IQR), skewness, kurtosis
- Outlier count (values beyond 1.5 × IQR)

#### Categorical Columns

- Count of unique values (cardinality)
- Mode and mode frequency
- Value frequency distribution (top 10 values with counts and percentages)
- Missing value count and percentage
- Entropy (measure of distribution uniformity)

#### Date Columns

- Minimum date, maximum date, date range (span in days)
- Count of unique dates
- Distribution by year, month, day of week
- Missing value count and percentage

### 6.2 Divergence Metrics

The following metrics shall be computed to quantify the difference between original and protected datasets:

| Metric | Description |
|--------|-------------|
| Mean Absolute Error | Average absolute difference in column means |
| RMSE | Root mean square error between original and protected values |
| KL Divergence | Kullback-Leibler divergence between distributions |
| Jensen-Shannon Distance | Symmetric measure of distribution similarity (0 = identical, 1 = completely different) |
| Wasserstein Distance | Earth mover's distance between distributions |
| Correlation Preservation | Frobenius norm of difference between correlation matrices |
| Category Drift | For categorical: percentage of rows where category changed |

### 6.3 Visualizations

- **Overlay histograms:** Original distribution (blue) vs. protected distribution (orange) with transparency
- **Box plots:** Side-by-side comparison of quartiles and outliers
- **Correlation heatmaps:** Original matrix, protected matrix, and difference matrix
- **Epsilon impact curves:** Line charts showing utility metrics vs. epsilon values (0.1 to 10.0)
- **Category frequency bar charts:** Grouped bars showing original vs. protected frequencies
- **QQ plots:** Quantile-quantile plots comparing original and protected distributions

---

## 7. Non-Functional Requirements

### 7.1 Performance

1. File upload and initial profiling: < 30 seconds for 1M rows
2. DP transformation: < 60 seconds for 1M rows with 50 columns
3. Statistical comparison generation: < 30 seconds
4. PDF report generation: < 20 seconds
5. UI responsiveness: < 100ms for user interactions

### 7.2 Scalability

1. Support datasets up to 1 million rows
2. Support up to 200 columns per dataset
3. Memory usage shall not exceed 4GB for maximum dataset size

### 7.3 Usability

1. Users should be able to complete a basic workflow (upload → configure → export) in under 5 minutes
2. All technical terms shall have explanatory tooltips
3. Error messages shall be actionable and user-friendly

### 7.4 Maintainability

1. Code coverage: minimum 80% for unit tests, 60% for integration tests
2. All public functions shall have docstrings following Google Python Style Guide
3. Architecture shall follow separation of concerns with clearly defined module boundaries

### 7.5 Security

1. Original datasets shall never be persisted to disk beyond the user session
2. Session data shall be cleared on application close or timeout
3. No telemetry or external data transmission

---

## 8. Technical Architecture

### 8.1 System Overview

The system follows a layered architecture separating the user interface from core business logic, enabling backend reuse across multiple applications.

### 8.2 Backend Modules

The backend shall be organized into the following modules:

```
dp_toolkit/
├── core/                      # Core differential privacy implementations
│   ├── mechanisms.py          # Laplace, Gaussian, Exponential mechanism wrappers around OpenDP
│   ├── sensitivity.py         # Sensitivity computation and bounds estimation
│   └── budget.py              # Privacy budget tracking and composition
│
├── data/                      # Data handling and transformation
│   ├── loader.py              # File parsing (CSV, Excel, Parquet)
│   ├── profiler.py            # Statistical profiling engine
│   ├── transformer.py         # DP transformation pipeline
│   └── exporter.py            # Dataset and report export
│
├── analysis/                  # Comparison and visualization
│   ├── comparator.py          # Statistical comparison and divergence metrics
│   └── visualizer.py          # Chart generation (matplotlib/plotly)
│
├── recommendations/           # Intelligent recommendations
│   ├── classifier.py          # Column sensitivity classification heuristics
│   └── advisor.py             # Epsilon and mechanism recommendations
│
└── reports/                   # Report generation
    └── pdf_generator.py       # PDF report creation
```

### 8.3 Frontend Structure

The Streamlit frontend shall be organized as:

```
frontend/
├── app.py                     # Main Streamlit application entry point
├── pages/                     # Multi-page application structure
│   ├── 1_upload.py            # File upload and preview
│   ├── 2_configure.py         # Column configuration and epsilon settings
│   ├── 3_analyze.py           # Statistical comparison and visualizations
│   └── 4_export.py            # Export dataset and reports
├── components/                # Reusable UI components
└── utils/                     # Frontend utilities and session state management
```

### 8.4 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opendp | ≥ 0.10 | Core differential privacy library |
| pandas | ≥ 2.0 | Data manipulation and analysis |
| numpy | ≥ 1.24 | Numerical computations |
| scipy | ≥ 1.10 | Statistical functions and divergence metrics |
| streamlit | ≥ 1.28 | Web UI framework |
| plotly | ≥ 5.15 | Interactive visualizations |
| matplotlib | ≥ 3.7 | Static chart generation for PDF |
| reportlab | ≥ 4.0 | PDF report generation |
| pyarrow | ≥ 12.0 | Parquet file support |
| openpyxl | ≥ 3.1 | Excel file support |
| pytest | ≥ 7.0 | Testing framework |
| pytest-cov | ≥ 4.0 | Test coverage reporting |

---

## 9. Testing Requirements

### 9.1 Unit Tests

Unit tests shall cover all backend modules with minimum 80% code coverage:

- **Mechanisms:** Verify noise distribution properties, epsilon bounds, sensitivity handling
- **Loader:** File format parsing, encoding handling, error cases, type detection
- **Profiler:** Statistical accuracy for all metrics against known datasets
- **Comparator:** Divergence metric calculations against analytical solutions
- **Classifier:** Sensitivity heuristic accuracy on labeled test columns
- **Advisor:** Recommendation logic and edge cases

### 9.2 Integration Tests

Integration tests shall verify end-to-end workflows with minimum 60% coverage:

- **Full pipeline:** Upload → Configure → Transform → Compare → Export
- **File format round-trips:** Load and export each supported format
- **Large dataset handling:** Performance tests with 100K and 1M row datasets
- **PDF generation:** Verify report completeness and formatting
- **Privacy guarantees:** Statistical tests verifying DP properties hold

### 9.3 Test Data

The test suite shall include synthetic healthcare datasets:

- Small dataset (1,000 rows) for fast unit tests
- Medium dataset (100,000 rows) for integration tests
- Large dataset (1,000,000 rows) for performance benchmarks
- Edge case datasets: all nulls, single value, extreme outliers, high cardinality

---

## 10. User Interface Specifications

### 10.1 Page Flow

The application follows a linear wizard-style flow with four main pages. Users can navigate back to previous pages to modify settings.

### Page 1: Upload

- File upload widget with drag-and-drop support
- Dataset preview table (first 100 rows, scrollable)
- Metadata summary: row count, column count, file size, detected types
- Data quality alerts (missing values, type issues)
- "Continue to Configuration" button (disabled until valid file loaded)

### Page 2: Configure

- Column configuration table with columns: Name, Type, Protection (dropdown), Mechanism (dropdown), Epsilon (input), Sensitivity (detected), Recommendation (tooltip)
- Global epsilon toggle: "Apply single epsilon to all columns" with slider (0.1–10.0)
- "Apply Recommendations" button to auto-fill based on heuristics
- Privacy budget summary showing total epsilon consumed
- Validation warnings (incompatible mechanism/type combinations)
- "Generate Protected Dataset" button

### Page 3: Analyze

- Tab layout: Summary | Per-Column | Correlations | Epsilon Impact
- **Summary tab:** Side-by-side statistical tables, overall utility score
- **Per-Column tab:** Column selector, overlay histograms, divergence metrics
- **Correlations tab:** Original and protected correlation heatmaps, difference matrix
- **Epsilon Impact tab:** Interactive charts showing utility vs. epsilon curves

### Page 4: Export

- Export format selector (CSV, Excel, Parquet)
- "Download Protected Dataset" button
- "Generate PDF Report" button
- Report preview (embedded PDF viewer)
- "Download PDF Report" button
- Configuration summary (what was applied)

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Utility loss too high for practical use | High | Provide epsilon impact visualization; allow per-column tuning; document tradeoffs |
| Users misconfigure epsilon (too high = weak privacy) | High | Strong recommendations with explanations; warnings for weak settings; compliance-friendly defaults |
| Performance issues with large datasets | Medium | Implement chunked processing; progress indicators; memory profiling |
| OpenDP API changes break compatibility | Medium | Pin dependency versions; abstract OpenDP behind internal interface; integration tests |
| Heuristic recommendations are inaccurate | Medium | Allow full manual override; log deviations; iterate on heuristics based on user feedback |
| Categorical columns with high cardinality | Low | Warn users; suggest aggregation or exclusion; implement top-K with "other" bucket |

---

## 12. Success Metrics

1. **Adoption:** Tool is used by at least 5 internal teams within 3 months of release
2. **Efficiency:** Average time from upload to export < 10 minutes
3. **Utility preservation:** Protected datasets maintain > 90% correlation structure at recommended epsilon
4. **User satisfaction:** Positive feedback from > 80% of users in post-release survey
5. **Code quality:** Maintain > 80% unit test coverage; zero critical bugs in production
6. **Backend reuse:** Backend library integrated into at least 2 other projects within 6 months

---

## 13. Future Enhancements

The following features are out of scope for v1.0 but may be considered for future releases:

1. Database connectivity (PostgreSQL, SQL Server, Snowflake)
2. User authentication and role-based access control
3. Analysis history and saved configurations
4. Batch processing mode for automated pipelines
5. Synthetic data generation as alternative to noise addition
6. Free-text field anonymization using NLP techniques
7. Docker container deployment
8. API endpoint exposure for programmatic access

---

## 14. Appendix

### 14.1 Glossary

| Term | Definition |
|------|------------|
| Differential Privacy (DP) | Mathematical framework providing provable privacy guarantees by adding calibrated noise to data or query results |
| Epsilon (ε) | Privacy parameter controlling the privacy-utility tradeoff; smaller values = stronger privacy, more noise |
| Sensitivity | Maximum possible change in a query result from adding or removing one individual's data |
| Laplace Mechanism | DP mechanism adding noise from Laplace distribution; provides pure ε-DP for numeric queries |
| Gaussian Mechanism | DP mechanism adding noise from normal distribution; provides (ε,δ)-DP with better composition |
| Exponential Mechanism | DP mechanism for categorical outputs; selects options with probability weighted by utility score |
| Privacy Budget | Total epsilon "spent" across all queries/transformations; budgets compose (add up) across operations |

### 14.2 References

- OpenDP Documentation: https://docs.opendp.org/
- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy
- HIPAA Privacy Rule: https://www.hhs.gov/hipaa/for-professionals/privacy/
- GDPR Article 25: Data Protection by Design and by Default

---

## 15. Approvals

| Role | Name | Date |
|------|------|------|
| Product Owner | | |
| Technical Lead | | |
| Data Privacy Officer | | |
| Development Lead | | |
