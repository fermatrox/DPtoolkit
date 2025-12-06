"""Pytest fixtures for test data."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv(temp_dir):
    """Create a sample CSV file with mixed column types."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "department": ["HR", "IT", "IT", "HR", "Finance"],
            "hire_date": [
                "2020-01-15",
                "2019-06-20",
                "2018-03-10",
                "2021-09-01",
                "2017-12-05",
            ],
            "is_active": [True, True, False, True, True],
        }
    )
    path = temp_dir / "sample.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def numeric_csv(temp_dir):
    """Create a CSV with only numeric columns."""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "negative": [-10, -5, 0, 5, 10],
        }
    )
    path = temp_dir / "numeric.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def categorical_csv(temp_dir):
    """Create a CSV with categorical columns."""
    import numpy as np

    n_rows = 100  # Enough rows for reliable type detection
    df = pd.DataFrame(
        {
            "color": np.random.choice(["red", "blue", "green"], n_rows),
            "size": np.random.choice(["S", "M", "L"], n_rows),
            "code": np.random.choice([1, 2, 3], n_rows),  # Low cardinality numeric
        }
    )
    path = temp_dir / "categorical.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def date_csv(temp_dir):
    """Create a CSV with date columns."""
    df = pd.DataFrame(
        {
            "created_date": ["2023-01-01", "2023-02-15", "2023-03-20"],
            "timestamp": [
                "2023-01-01 10:30:00",
                "2023-02-15 14:45:00",
                "2023-03-20 09:15:00",
            ],
            "dob": ["1990-05-15", "1985-10-20", "1995-03-08"],
            "event": ["2023/06/01", "2023/07/15", "2023/08/20"],
        }
    )
    path = temp_dir / "dates.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def latin1_csv(temp_dir):
    """Create a CSV with Latin-1 encoded characters."""
    df = pd.DataFrame(
        {
            "name": ["José", "François", "Müller", "Señora", "Øresund"],
            "city": ["São Paulo", "Paris", "München", "Madrid", "København"],
        }
    )
    path = temp_dir / "latin1.csv"
    df.to_csv(path, index=False, encoding="latin-1")
    return path


@pytest.fixture
def utf8_csv(temp_dir):
    """Create a CSV with UTF-8 encoded characters."""
    df = pd.DataFrame(
        {
            "name": ["日本語", "中文", "한국어", "Ελληνικά", "العربية"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    path = temp_dir / "utf8.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def empty_csv(temp_dir):
    """Create an empty CSV (headers only)."""
    df = pd.DataFrame(columns=["col1", "col2", "col3"])
    path = temp_dir / "empty.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def single_column_csv(temp_dir):
    """Create a CSV with a single column."""
    df = pd.DataFrame({"only_column": [1, 2, 3, 4, 5]})
    path = temp_dir / "single.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def nulls_csv(temp_dir):
    """Create a CSV with null values."""
    df = pd.DataFrame(
        {
            "all_nulls": [None, None, None, None, None],
            "some_nulls": [1, None, 3, None, 5],
            "no_nulls": [1, 2, 3, 4, 5],
        }
    )
    path = temp_dir / "nulls.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def large_csv(temp_dir):
    """Create a larger CSV for performance testing."""
    import numpy as np

    n_rows = 10000
    df = pd.DataFrame(
        {
            "id": range(n_rows),
            "value": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows),
        }
    )
    path = temp_dir / "large.csv"
    df.to_csv(path, index=False)
    return path
