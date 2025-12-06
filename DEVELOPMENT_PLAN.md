# DPtoolkit Development Plan

This document outlines the incremental development plan with testing checkpoints after each step.

---

## Phase 1: Project Foundation

### Step 1.1: Project Structure & Configuration
**Goal:** Set up the Python project skeleton with all configuration files.

**Tasks:**
- Create directory structure (`dp_toolkit/`, `frontend/`, `tests/`)
- Create `setup.py` and `pyproject.toml`
- Create `requirements.txt` and `requirements-dev.txt`
- Set up `.gitignore`
- Create `pytest.ini` configuration
- Add empty `__init__.py` files

**Testing Checkpoint:**
- [ ] `pip install -e .` succeeds
- [ ] `pytest` runs (even with no tests)
- [ ] Project imports work: `import dp_toolkit`

---

### Step 1.2: Data Loader - CSV Support
**Goal:** Load CSV files with automatic type detection.

**Tasks:**
- Implement `dp_toolkit/data/loader.py`
- Support CSV parsing with pandas
- Detect column types (numeric, categorical, date)
- Handle common encodings (UTF-8, Latin-1)
- Return structured DatasetInfo object

**Testing Checkpoint:**
- [ ] Unit tests for CSV loading
- [ ] Type detection accuracy tests
- [ ] Encoding handling tests
- [ ] Edge cases: empty file, single column, all nulls

---

### Step 1.3: Data Loader - Excel & Parquet
**Goal:** Extend loader to support Excel and Parquet formats.

**Tasks:**
- Add Excel support via openpyxl
- Add Parquet support via pyarrow
- Unified interface for all formats
- Sheet selection for Excel files

**Testing Checkpoint:**
- [ ] Unit tests for Excel loading
- [ ] Unit tests for Parquet loading
- [ ] Format detection tests
- [ ] Round-trip tests (load → export → load)

---

## Phase 2: Statistical Profiling

### Step 2.1: Numeric Column Profiler
**Goal:** Calculate comprehensive statistics for numeric columns.

**Tasks:**
- Implement `dp_toolkit/data/profiler.py`
- Calculate: count, mean, median, std, min, max, percentiles
- Calculate: skewness, kurtosis, IQR
- Detect outliers (IQR method)
- Handle nulls correctly

**Testing Checkpoint:**
- [ ] Accuracy tests against known datasets
- [ ] Null handling tests
- [ ] Edge cases: single value, all same values
- [ ] Performance test with 100K rows

---

### Step 2.2: Categorical & Date Profiler
**Goal:** Profile categorical and date columns.

**Tasks:**
- Categorical: cardinality, mode, frequency distribution, entropy
- Date: min/max, range, temporal distribution
- Unified ColumnProfile dataclass

**Testing Checkpoint:**
- [ ] Categorical profiling accuracy
- [ ] Date profiling accuracy
- [ ] High cardinality handling (>1000 unique values)
- [ ] Mixed type handling

---

### Step 2.3: Dataset-Level Profiling
**Goal:** Full dataset profiling including correlations.

**Tasks:**
- Correlation matrix for numeric columns
- Missing value summary
- Dataset-level statistics
- Profile caching for performance

**Testing Checkpoint:**
- [ ] Correlation matrix accuracy
- [ ] Full dataset profile generation
- [ ] Performance: < 30s for 1M rows
- [ ] Memory usage under 4GB

---

## Phase 3: Core DP Mechanisms

### Step 3.1: Laplace Mechanism
**Goal:** Implement Laplace noise for bounded numeric data.

**Tasks:**
- Implement `dp_toolkit/core/mechanisms.py`
- Wrap OpenDP Laplace mechanism
- Sensitivity calculation for bounded data
- Epsilon parameter validation

**Testing Checkpoint:**
- [ ] Noise distribution follows Laplace
- [ ] Privacy guarantee verification (statistical test)
- [ ] Sensitivity bounds respected
- [ ] Edge cases: epsilon limits (0.01, 10.0)

---

### Step 3.2: Gaussian Mechanism
**Goal:** Implement Gaussian noise for (ε,δ)-DP.

**Tasks:**
- Wrap OpenDP Gaussian mechanism
- Delta parameter handling
- Unbounded data support

**Testing Checkpoint:**
- [ ] Noise distribution follows Gaussian
- [ ] (ε,δ)-DP guarantee verification
- [ ] Comparison with Laplace for same epsilon

---

### Step 3.3: Exponential Mechanism
**Goal:** Implement exponential mechanism for categorical data.

**Tasks:**
- Wrap OpenDP exponential mechanism
- Utility function for category selection
- Preserve category distribution shape

**Testing Checkpoint:**
- [ ] Category selection probabilities correct
- [ ] Privacy guarantee verification
- [ ] High cardinality performance

---

### Step 3.4: Privacy Budget Tracker
**Goal:** Track and compose privacy budgets.

**Tasks:**
- Implement `dp_toolkit/core/budget.py`
- Sequential composition tracking
- Per-column budget allocation
- Budget exhaustion warnings

**Testing Checkpoint:**
- [ ] Composition calculations correct
- [ ] Budget tracking across operations
- [ ] Warning thresholds work

---

## Phase 4: DP Transformation Pipeline

### Step 4.1: Column Transformer
**Goal:** Apply DP to individual columns.

**Tasks:**
- Implement `dp_toolkit/data/transformer.py`
- Apply mechanism based on column type
- Preserve data types (round integers)
- Handle dates via epoch conversion

**Testing Checkpoint:**
- [ ] Numeric transformation preserves type
- [ ] Date transformation accuracy
- [ ] Null preservation
- [ ] Output validates against input schema

---

### Step 4.2: Dataset Transformer
**Goal:** Transform entire datasets with configuration.

**Tasks:**
- Process all columns based on config
- Support: Protect, Passthrough, Exclude modes
- Global vs per-column epsilon
- Progress callback for UI

**Testing Checkpoint:**
- [ ] Full dataset transformation
- [ ] Configuration modes work correctly
- [ ] Performance: < 60s for 1M × 50
- [ ] Memory efficiency

---

### Step 4.3: Data Exporter
**Goal:** Export protected datasets.

**Tasks:**
- Implement `dp_toolkit/data/exporter.py`
- Export to CSV, Excel, Parquet
- Include metadata in export
- Preserve column order and types

**Testing Checkpoint:**
- [ ] Round-trip integrity (export → import)
- [ ] Format-specific tests
- [ ] Large file export performance

---

## Phase 5: Analysis & Comparison

### Step 5.1: Statistical Comparator
**Goal:** Compare original vs protected datasets.

**Tasks:**
- Implement `dp_toolkit/analysis/comparator.py`
- Per-column comparison tables
- Calculate divergence metrics (MAE, RMSE)
- Correlation preservation metric

**Testing Checkpoint:**
- [ ] Divergence calculations accurate
- [ ] Comparison with known transformations
- [ ] Edge cases: identical data, completely different

---

### Step 5.2: Advanced Divergence Metrics
**Goal:** Implement KL, Jensen-Shannon, Wasserstein.

**Tasks:**
- KL divergence (with smoothing)
- Jensen-Shannon distance
- Wasserstein distance
- Category drift metric

**Testing Checkpoint:**
- [ ] Mathematical accuracy vs scipy
- [ ] Symmetry tests (JS)
- [ ] Edge cases: zero probabilities

---

### Step 5.3: Visualization Generator
**Goal:** Create comparison visualizations.

**Tasks:**
- Implement `dp_toolkit/analysis/visualizer.py`
- Overlay histograms (original vs protected)
- Correlation heatmaps with diff
- Use plotly for interactive charts

**Testing Checkpoint:**
- [ ] Charts render without errors
- [ ] Data accuracy in visualizations
- [ ] Performance with large datasets

---

## Phase 6: Recommendations Engine

### Step 6.1: Column Classifier
**Goal:** Classify column sensitivity automatically.

**Tasks:**
- Implement `dp_toolkit/recommendations/classifier.py`
- Pattern matching for column names
- Sensitivity levels: High, Medium, Low
- Healthcare-specific patterns (SSN, diagnosis, DOB)

**Testing Checkpoint:**
- [ ] Known patterns correctly classified
- [ ] Edge cases: ambiguous names
- [ ] Override capability works

---

### Step 6.2: Epsilon Advisor
**Goal:** Recommend epsilon values and mechanisms.

**Tasks:**
- Implement `dp_toolkit/recommendations/advisor.py`
- Epsilon by sensitivity level
- Mechanism by data type
- Explanatory text generation

**Testing Checkpoint:**
- [ ] Recommendations match PRD specifications
- [ ] All column types covered
- [ ] Explanation quality

---

## Phase 7: Frontend - Core Pages

### Step 7.1: Streamlit App Shell
**Goal:** Create the basic Streamlit application structure.

**Tasks:**
- Implement `frontend/app.py`
- Multi-page navigation setup
- Session state management
- Basic styling

**Testing Checkpoint:**
- [ ] App starts without errors
- [ ] Navigation works
- [ ] Session state persists

---

### Step 7.2: Upload Page
**Goal:** File upload and preview functionality.

**Tasks:**
- Implement `frontend/pages/1_upload.py`
- Drag-and-drop file upload
- Dataset preview table (first 100 rows)
- Metadata summary display
- Validation before proceeding

**Testing Checkpoint:**
- [ ] All file formats upload correctly
- [ ] Preview displays accurately
- [ ] Large file handling (progress indicator)
- [ ] Error handling for invalid files

---

### Step 7.3: Configure Page
**Goal:** Column configuration interface.

**Tasks:**
- Implement `frontend/pages/2_configure.py`
- Column configuration table
- Protection mode selection
- Epsilon slider (global and per-column)
- Auto-recommendations button

**Testing Checkpoint:**
- [ ] All configuration options work
- [ ] Validation warnings display
- [ ] State persists between pages
- [ ] UI responsiveness < 100ms

---

### Step 7.4: Analyze Page
**Goal:** Statistical comparison and visualization.

**Tasks:**
- Implement `frontend/pages/3_analyze.py`
- Tabs: Summary, Per-Column, Correlations
- Interactive visualizations
- Divergence metrics display

**Testing Checkpoint:**
- [ ] All tabs render correctly
- [ ] Visualizations interactive
- [ ] Handles edge cases gracefully

---

### Step 7.5: Export Page
**Goal:** Export datasets and generate reports.

**Tasks:**
- Implement `frontend/pages/4_export.py`
- Format selection
- Download buttons
- Configuration summary

**Testing Checkpoint:**
- [ ] Downloads work in all formats
- [ ] Summary accurate
- [ ] Session cleanup on exit

---

## Phase 8: PDF Reports

### Step 8.1: PDF Generator
**Goal:** Generate compliance-ready PDF reports.

**Tasks:**
- Implement `dp_toolkit/reports/pdf_generator.py`
- Executive summary section
- Configuration details
- Statistical comparisons
- Embedded visualizations

**Testing Checkpoint:**
- [ ] PDF generates without errors
- [ ] All sections present
- [ ] Visualizations render correctly
- [ ] Performance: < 20s generation

---

## Phase 9: Integration & Polish

### Step 9.1: End-to-End Integration Tests
**Goal:** Verify complete workflows.

**Tasks:**
- Full pipeline tests (upload → export)
- Multi-format round-trip tests
- Performance benchmarks
- Memory profiling

**Testing Checkpoint:**
- [ ] Complete workflow succeeds
- [ ] Performance targets met
- [ ] No memory leaks
- [ ] Error recovery works

---

### Step 9.2: UI Polish & Error Handling
**Goal:** Production-ready user experience.

**Tasks:**
- Comprehensive error messages
- Loading states and progress bars
- Tooltips and help text
- Edge case handling

**Testing Checkpoint:**
- [ ] All error paths tested
- [ ] UI feedback appropriate
- [ ] Accessibility basics

---

### Step 9.3: Documentation
**Goal:** Complete project documentation.

**Tasks:**
- Update README.md with usage guide
- API documentation for dp_toolkit
- Example notebooks
- Deployment guide

**Testing Checkpoint:**
- [ ] Examples run successfully
- [ ] Documentation accurate
- [ ] Installation instructions work

---

## Summary: Development Milestones

| Milestone | Steps | Key Deliverable |
|-----------|-------|-----------------|
| M1: Foundation | 1.1-1.3 | Data loading works |
| M2: Profiling | 2.1-2.3 | Statistical profiling complete |
| M3: DP Core | 3.1-3.4 | All mechanisms working |
| M4: Pipeline | 4.1-4.3 | End-to-end transformation |
| M5: Analysis | 5.1-5.3 | Comparison & visualization |
| M6: Recommendations | 6.1-6.2 | Auto-recommendations working |
| M7: Frontend | 7.1-7.5 | Full UI functional |
| M8: Reports | 8.1 | PDF generation complete |
| M9: Polish | 9.1-9.3 | Production ready |

---

## Estimated Effort

Each step is designed to be completable in 1-3 focused sessions. The entire plan contains 25 steps across 9 phases. With proper testing after each step, expect:

- **Phase 1-2:** Foundation & Profiling - Core data handling
- **Phase 3-4:** DP Implementation - The heart of the toolkit
- **Phase 5-6:** Analysis & Intelligence - User value features
- **Phase 7-8:** Frontend & Reports - User-facing deliverables
- **Phase 9:** Integration - Production readiness
