# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DPtoolkit (Differential Privacy Toolkit) is a privacy-preserving data analysis tool for healthcare datasets. It enables secure sharing and analysis of sensitive patient data using formal differential privacy guarantees, targeting GDPR and HIPAA compliance.

**Status:** Pre-development (PRD completed, implementation pending)

## Tech Stack

- **Backend:** Python with OpenDP (≥0.10) as the core differential privacy engine
- **Frontend:** Streamlit (≥1.28) web UI
- **Data:** pandas, numpy, scipy for data manipulation and statistics
- **Visualization:** plotly (interactive), matplotlib (PDF reports)
- **Reports:** reportlab for PDF generation
- **Testing:** pytest with pytest-cov

## Build & Development Commands

```bash
# Environment setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .

# Run the Streamlit frontend
streamlit run frontend/app.py

# Run all tests
pytest

# Run tests with coverage
pytest tests/unit --cov=dp_toolkit --cov-report=html

# Run a single test file
pytest tests/unit/test_mechanisms.py

# Run integration tests only
pytest tests/integration

# Linting
flake8 dp_toolkit
black dp_toolkit
mypy dp_toolkit
```

## Architecture

The codebase separates a reusable backend library (`dp_toolkit/`) from the frontend (`frontend/`):

```
dp_toolkit/
├── core/           # DP mechanisms (Laplace, Gaussian, Exponential), sensitivity, budget tracking
├── data/           # Data loading, profiling, transformation, export
├── analysis/       # Statistical comparison, divergence metrics, visualization
├── recommendations/# Column sensitivity classification, epsilon/mechanism advisor
└── reports/        # PDF report generation

frontend/
├── app.py          # Streamlit entry point
├── pages/          # Wizard pages: upload → configure → analyze → export
├── components/     # Reusable UI components
└── utils/          # Frontend utilities
```

### Key Design Decisions

- **OpenDP abstraction:** Wrap OpenDP behind internal interfaces in `core/mechanisms.py` to isolate API changes
- **Backend-frontend separation:** The `dp_toolkit` package should be usable independently without Streamlit
- **Session-only data:** No dataset persistence beyond user session; memory cleared on close/timeout

## DP Mechanisms

| Mechanism | Use Case | Privacy Model |
|-----------|----------|---------------|
| Laplace | Bounded numeric columns | ε-DP |
| Gaussian | Unbounded numeric columns | (ε,δ)-DP |
| Exponential | Categorical columns | ε-DP |

Dates are handled via epoch conversion. Integer types require rounding after noise addition.

## Performance Targets

- File upload & profiling: < 30s for 1M rows
- DP transformation: < 60s for 1M rows × 50 columns
- Max dataset size: 1M rows, 200 columns, 4GB memory

## Testing Requirements

- Unit test coverage: ≥ 80%
- Integration test coverage: ≥ 60%
- Test data sizes: 1K (small), 100K (medium), 1M (large) rows
- Privacy guarantee verification required for mechanism tests

## Code Style

- Google Python Style Guide for docstrings
- Type hints on all public functions
- Modules should be focused and loosely coupled
