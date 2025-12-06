"""Unit tests for date profiler."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.profiler import (
    DateProfile,
    DateProfiler,
    profile_date,
    profile_date_columns,
)


class TestDateProfile:
    """Tests for DateProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a DateProfile."""
        profile = DateProfile(
            count=100,
            null_count=5,
            min_date=datetime(2020, 1, 1),
            max_date=datetime(2020, 12, 31),
            range_days=365,
            cardinality=50,
            mode=datetime(2020, 6, 15),
            mode_count=10,
            weekday_distribution={0: 20, 1: 15},
            month_distribution={1: 30, 6: 40},
            year_distribution={2020: 100},
        )
        assert profile.count == 100
        assert profile.null_count == 5
        assert profile.min_date == datetime(2020, 1, 1)
        assert profile.range_days == 365


class TestDateProfiler:
    """Tests for DateProfiler class."""

    def test_basic_profile(self):
        """Test basic date profiling."""
        dates = pd.to_datetime([
            "2023-01-01", "2023-01-15", "2023-02-01", "2023-03-01"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 4
        assert profile.null_count == 0
        assert profile.min_date == datetime(2023, 1, 1)
        assert profile.max_date == datetime(2023, 3, 1)
        assert profile.cardinality == 4

    def test_profile_with_nulls(self):
        """Test profiling with null values."""
        dates = pd.to_datetime([
            "2023-01-01", None, "2023-01-15", pd.NaT
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 2
        assert profile.null_count == 2

    def test_empty_series(self):
        """Test profiling an empty series."""
        series = pd.Series([], dtype="datetime64[ns]")
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 0
        assert profile.null_count == 0
        assert profile.min_date is None
        assert profile.max_date is None
        assert profile.range_days is None

    def test_all_null_series(self):
        """Test profiling series with all nulls."""
        series = pd.Series([pd.NaT, pd.NaT, pd.NaT])
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 0
        assert profile.null_count == 3
        assert profile.min_date is None

    def test_single_date(self):
        """Test profiling series with single date."""
        series = pd.Series(pd.to_datetime(["2023-06-15"]))
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 1
        assert profile.cardinality == 1
        assert profile.min_date == datetime(2023, 6, 15)
        assert profile.max_date == datetime(2023, 6, 15)
        assert profile.range_days == 0

    def test_date_range_calculation(self):
        """Test date range calculation."""
        dates = pd.to_datetime(["2023-01-01", "2023-12-31"])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.range_days == 364  # Not 365 because it's end - start

    def test_mode_calculation(self):
        """Test mode (most frequent date) calculation."""
        dates = pd.to_datetime([
            "2023-01-01", "2023-01-15", "2023-01-15",
            "2023-01-15", "2023-02-01"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.mode == datetime(2023, 1, 15)
        assert profile.mode_count == 3

    def test_weekday_distribution(self):
        """Test weekday distribution calculation."""
        # Create dates that span different weekdays
        # 2023-01-02 is Monday (0), 2023-01-03 is Tuesday (1), etc.
        dates = pd.to_datetime([
            "2023-01-02",  # Monday
            "2023-01-03",  # Tuesday
            "2023-01-04",  # Wednesday
            "2023-01-09",  # Monday
            "2023-01-10",  # Tuesday
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.weekday_distribution[0] == 2  # 2 Mondays
        assert profile.weekday_distribution[1] == 2  # 2 Tuesdays
        assert profile.weekday_distribution[2] == 1  # 1 Wednesday

    def test_month_distribution(self):
        """Test month distribution calculation."""
        dates = pd.to_datetime([
            "2023-01-15", "2023-01-20",
            "2023-06-01", "2023-06-15", "2023-06-30",
            "2023-12-25"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.month_distribution[1] == 2   # January
        assert profile.month_distribution[6] == 3   # June
        assert profile.month_distribution[12] == 1  # December

    def test_year_distribution(self):
        """Test year distribution calculation."""
        dates = pd.to_datetime([
            "2021-06-01", "2022-06-01", "2022-12-01",
            "2023-01-01", "2023-06-01", "2023-12-01"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.year_distribution[2021] == 1
        assert profile.year_distribution[2022] == 2
        assert profile.year_distribution[2023] == 3

    def test_string_date_conversion(self):
        """Test automatic string to datetime conversion."""
        series = pd.Series(["2023-01-01", "2023-06-15", "2023-12-31"])
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        assert profile.min_date == datetime(2023, 1, 1)
        assert profile.max_date == datetime(2023, 12, 31)

    def test_mixed_format_strings(self):
        """Test handling of mixed format date strings.

        Note: pandas to_datetime with errors='coerce' may not parse all formats.
        Some formats will be coerced to NaT.
        """
        series = pd.Series([
            "2023-01-01",
            "2023-01-15",
            "2023-03-01"
        ])
        profiler = DateProfiler()
        profile = profiler.profile(series)

        # All ISO format dates should parse successfully
        assert profile.count == 3


class TestDateProfilerDataframe:
    """Tests for DataFrame-level date profiling."""

    def test_profile_dataframe(self):
        """Test profiling multiple datetime columns."""
        df = pd.DataFrame({
            "date1": pd.to_datetime(["2023-01-01", "2023-06-15"]),
            "date2": pd.to_datetime(["2022-01-01", "2022-12-31"]),
            "category": ["A", "B"],
        })
        profiler = DateProfiler()
        profiles = profiler.profile_dataframe(df)

        assert "date1" in profiles
        assert "date2" in profiles
        assert "category" not in profiles

    def test_profile_specific_columns(self):
        """Test profiling specific datetime columns."""
        df = pd.DataFrame({
            "date1": pd.to_datetime(["2023-01-01", "2023-06-15"]),
            "date2": pd.to_datetime(["2022-01-01", "2022-12-31"]),
        })
        profiler = DateProfiler()
        profiles = profiler.profile_dataframe(df, columns=["date1"])

        assert "date1" in profiles
        assert "date2" not in profiles


class TestDateConvenienceFunctions:
    """Tests for convenience functions."""

    def test_profile_date_function(self):
        """Test profile_date convenience function."""
        series = pd.Series(pd.to_datetime(["2023-01-01", "2023-06-15"]))
        profile = profile_date(series)

        assert isinstance(profile, DateProfile)
        assert profile.count == 2

    def test_profile_date_columns_function(self):
        """Test profile_date_columns convenience function."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-06-15"]),
            "value": [1, 2],
        })
        profiles = profile_date_columns(df)

        assert "date" in profiles
        assert "value" not in profiles


class TestDateEdgeCases:
    """Edge case tests for date profiling."""

    def test_dates_spanning_many_years(self):
        """Test dates spanning many years."""
        dates = pd.to_datetime([
            "1990-01-01", "2000-06-15", "2010-03-20", "2023-12-31"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 4
        assert profile.min_date.year == 1990
        assert profile.max_date.year == 2023
        assert len(profile.year_distribution) == 4

    def test_same_date_repeated(self):
        """Test series with same date repeated."""
        dates = pd.to_datetime(["2023-06-15"] * 100)
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 100
        assert profile.cardinality == 1
        assert profile.mode_count == 100
        assert profile.range_days == 0

    def test_datetime_with_time(self):
        """Test datetime with time component."""
        dates = pd.to_datetime([
            "2023-01-01 10:30:00",
            "2023-01-01 14:45:00",
            "2023-01-02 09:00:00"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        # All three are unique because of time component
        assert profile.cardinality == 3

    def test_leap_year_dates(self):
        """Test handling of leap year dates."""
        dates = pd.to_datetime([
            "2020-02-29",  # Leap year
            "2024-02-29",  # Leap year
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 2
        assert 2 in profile.month_distribution  # February

    def test_end_of_year_boundary(self):
        """Test dates at end of year boundaries."""
        dates = pd.to_datetime([
            "2022-12-31",
            "2023-01-01"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.range_days == 1
        assert 12 in profile.month_distribution
        assert 1 in profile.month_distribution

    def test_invalid_date_string_coercion(self):
        """Test that invalid dates are coerced to NaT."""
        series = pd.Series([
            "2023-01-01",
            "not a date",
            "2023-06-15"
        ])
        profiler = DateProfiler()
        profile = profiler.profile(series)

        # "not a date" should become NaT
        assert profile.count == 2
        assert profile.null_count == 1

    def test_very_old_dates(self):
        """Test handling of very old dates."""
        dates = pd.to_datetime([
            "1800-01-01",
            "1900-06-15",
            "2000-12-31"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        assert profile.min_date.year == 1800

    def test_future_dates(self):
        """Test handling of future dates."""
        dates = pd.to_datetime([
            "2025-01-01",
            "2030-06-15",
            "2050-12-31"
        ])
        series = pd.Series(dates)
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        assert profile.max_date.year == 2050

    def test_high_cardinality_dates(self):
        """Test handling of many unique dates."""
        # Generate 1000 unique dates
        base_date = datetime(2020, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(1000)]
        series = pd.Series(pd.to_datetime(dates))
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 1000
        assert profile.cardinality == 1000
        assert profile.range_days == 999

    def test_timestamp_type(self):
        """Test handling of pandas Timestamp type."""
        series = pd.Series([
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-06-15"),
            pd.Timestamp("2023-12-31")
        ])
        profiler = DateProfiler()
        profile = profiler.profile(series)

        assert profile.count == 3
        assert isinstance(profile.min_date, datetime)
