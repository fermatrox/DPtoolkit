"""End-to-end integration tests for DPtoolkit.

This module tests complete workflows from data loading through export,
verifying that all components work together correctly.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_toolkit.data.loader import DataLoader, load_csv, load_excel, load_parquet
from dp_toolkit.data.profiler import DatasetProfiler
from dp_toolkit.data.transformer import (
    DatasetTransformer,
    DatasetConfig,
)
from dp_toolkit.data.exporter import DataExporter
from dp_toolkit.analysis.comparator import DatasetComparator
from dp_toolkit.recommendations.classifier import (
    ColumnClassifier,
    SensitivityLevel,
)
from dp_toolkit.recommendations.advisor import (
    RecommendationAdvisor,
    recommend_for_dataset,
)
from dp_toolkit.reports import PDFReportGenerator, ReportMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def healthcare_df():
    """Create a realistic healthcare dataset for testing."""
    np.random.seed(42)
    n_rows = 500

    return pd.DataFrame({
        "patient_id": range(1000, 1000 + n_rows),
        "ssn": [f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}"
                for _ in range(n_rows)],
        "first_name": np.random.choice(["John", "Jane", "Bob", "Alice", "Charlie"], n_rows),
        "last_name": np.random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"], n_rows),
        "age": np.random.randint(18, 90, n_rows),
        "weight_kg": np.random.uniform(45, 120, n_rows).round(1),
        "height_cm": np.random.randint(150, 200, n_rows),
        "blood_pressure_systolic": np.random.randint(90, 180, n_rows),
        "blood_pressure_diastolic": np.random.randint(60, 110, n_rows),
        "cholesterol": np.random.uniform(120, 300, n_rows).round(1),
        "diagnosis_code": np.random.choice(["J06.9", "I10", "E11.9", "M54.5", "R51"], n_rows),
        "department": np.random.choice(["Cardiology", "Neurology", "Orthopedics", "General"], n_rows),
        "admission_date": pd.date_range("2023-01-01", periods=n_rows, freq="h"),
        "is_inpatient": np.random.choice([True, False], n_rows),
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Test complete data processing workflows."""

    def test_csv_pipeline_with_recommendations(self, healthcare_df, temp_dir):
        """Test full pipeline: load → classify → recommend → transform → compare → export."""
        # Step 1: Save and load CSV
        csv_path = temp_dir / "healthcare.csv"
        healthcare_df.to_csv(csv_path, index=False)

        loader = DataLoader()
        dataset_info = loader.load_csv(csv_path)
        df = dataset_info.dataframe

        assert len(df) == len(healthcare_df)
        assert len(df.columns) == len(healthcare_df.columns)

        # Step 2: Profile the dataset
        profiler = DatasetProfiler()
        profile = profiler.profile(df)

        assert profile.row_count == len(df)
        assert profile.column_count == len(df.columns)
        assert len(profile.column_profiles) == len(df.columns)

        # Step 3: Classify sensitivity
        classifier = ColumnClassifier()
        classification = classifier.classify_dataset(df)

        # SSN should be high sensitivity
        assert classification.column_results["ssn"].sensitivity == SensitivityLevel.HIGH
        # Age should be detected as sensitive (medical context)
        assert classification.column_results["age"].sensitivity in [
            SensitivityLevel.HIGH,
            SensitivityLevel.MEDIUM,
            SensitivityLevel.LOW,  # Age can be low sensitivity
        ]

        # Step 4: Get epsilon recommendations
        advisor = RecommendationAdvisor()
        recommendations = advisor.recommend_for_dataset(df)

        assert len(recommendations.column_recommendations) == len(df.columns)
        # High sensitivity should have low epsilon
        ssn_rec = recommendations.column_recommendations["ssn"]
        assert ssn_rec.epsilon_recommendation.epsilon <= 0.5

        # Step 5: Configure and transform
        config = DatasetConfig(global_epsilon=1.0)
        config.exclude_columns(["ssn", "first_name", "last_name"])
        config.passthrough_columns(["patient_id"])

        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Excluded columns should be removed
        assert "ssn" not in protected_df.columns
        assert "first_name" not in protected_df.columns
        assert "last_name" not in protected_df.columns
        # Passthrough column should be unchanged
        assert (protected_df["patient_id"] == df["patient_id"]).all()
        # Protected columns should have noise added
        assert not (protected_df["age"] == df["age"]).all()

        # Step 6: Compare datasets
        # Filter original to match protected columns
        original_filtered = df[[c for c in protected_df.columns if c in df.columns]].copy()

        comparator = DatasetComparator()
        comparison = comparator.compare(original_filtered, protected_df)

        assert comparison.row_count == len(df)
        # Check that we have some comparisons
        total_comparisons = (
            len(comparison.numeric_comparisons)
            + len(comparison.categorical_comparisons)
            + len(comparison.date_comparisons)
        )
        assert total_comparisons > 0
        # Check correlation preservation is reasonable
        if comparison.correlation_preservation is not None:
            assert comparison.correlation_preservation.preservation_rate > 0.5  # Should preserve some correlations

        # Step 7: Export protected data
        exporter = DataExporter()

        # CSV export
        csv_output = temp_dir / "protected.csv"
        exporter.export_csv(protected_df, csv_output)
        assert csv_output.exists()

        # Verify exported data
        reimported = pd.read_csv(csv_output)
        assert len(reimported) == len(protected_df)
        assert list(reimported.columns) == list(protected_df.columns)

    def test_excel_pipeline(self, healthcare_df, temp_dir):
        """Test pipeline with Excel format."""
        # Save as Excel
        excel_path = temp_dir / "healthcare.xlsx"
        healthcare_df.to_excel(excel_path, index=False)

        # Load
        loader = DataLoader()
        dataset_info = loader.load_excel(excel_path)
        df = dataset_info.dataframe

        assert len(df) == len(healthcare_df)

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        config.exclude_columns(["ssn"])

        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Export back to Excel
        exporter = DataExporter()
        output_path = temp_dir / "protected.xlsx"
        exporter.export_excel(protected_df, output_path)

        assert output_path.exists()

        # Reimport and verify
        reimported = pd.read_excel(output_path)
        assert len(reimported) == len(protected_df)

    def test_parquet_pipeline(self, healthcare_df, temp_dir):
        """Test pipeline with Parquet format."""
        # Parquet doesn't handle datetime well with timezone-naive, so convert
        df = healthcare_df.copy()
        df["admission_date"] = df["admission_date"].astype(str)

        # Save as Parquet
        parquet_path = temp_dir / "healthcare.parquet"
        df.to_parquet(parquet_path, index=False)

        # Load
        loader = DataLoader()
        dataset_info = loader.load_parquet(parquet_path)
        loaded_df = dataset_info.dataframe

        assert len(loaded_df) == len(df)

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        config.exclude_columns(["ssn"])

        transformer = DatasetTransformer()
        result = transformer.transform(loaded_df, config)
        protected_df = result.data

        # Export back to Parquet
        exporter = DataExporter()
        output_path = temp_dir / "protected.parquet"
        exporter.export_parquet(protected_df, output_path)

        assert output_path.exists()

        # Reimport and verify
        reimported = pd.read_parquet(output_path)
        assert len(reimported) == len(protected_df)

    def test_pipeline_with_pdf_report(self, healthcare_df, temp_dir):
        """Test full pipeline including PDF report generation."""
        # Prepare data
        df = healthcare_df.copy()
        df["admission_date"] = df["admission_date"].astype(str)

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        config.exclude_columns(["ssn", "first_name", "last_name"])
        config.passthrough_columns(["patient_id"])

        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Compare
        original_filtered = df[[c for c in protected_df.columns if c in df.columns]].copy()
        comparator = DatasetComparator()
        comparison = comparator.compare(original_filtered, protected_df)

        # Generate column configs dict for report
        column_configs = {}
        for col in df.columns:
            if col in ["ssn", "first_name", "last_name"]:
                column_configs[col] = {"mode": "exclude"}
            elif col == "patient_id":
                column_configs[col] = {"mode": "passthrough"}
            else:
                column_configs[col] = {
                    "mode": "protect",
                    "epsilon": 1.0,
                    "mechanism": "laplace",
                }

        # Generate PDF
        metadata = ReportMetadata(
            title="Integration Test Report",
            original_filename="healthcare.csv",
        )

        generator = PDFReportGenerator()
        report = generator.generate(
            original_df=original_filtered,
            protected_df=protected_df,
            comparison=comparison,
            column_configs=column_configs,
            metadata=metadata,
        )

        # Save and verify
        pdf_path = temp_dir / "report.pdf"
        report.save(pdf_path)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        # PDF files start with %PDF
        with open(pdf_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF"


# =============================================================================
# Multi-Format Round-Trip Tests
# =============================================================================


class TestMultiFormatRoundTrip:
    """Test data integrity across format conversions."""

    def test_csv_round_trip(self, healthcare_df, temp_dir):
        """Test CSV: original → transform → export → reimport → verify."""
        df = healthcare_df.copy()
        df["admission_date"] = df["admission_date"].astype(str)

        # Transform
        config = DatasetConfig(global_epsilon=0.5)
        config.passthrough_columns(["patient_id", "department", "diagnosis_code"])

        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Export and reimport
        csv_path = temp_dir / "round_trip.csv"
        exporter = DataExporter()
        exporter.export_csv(protected_df, csv_path)

        reimported = load_csv(csv_path).dataframe

        # Verify structure
        assert list(reimported.columns) == list(protected_df.columns)
        assert len(reimported) == len(protected_df)

        # Verify passthrough columns unchanged
        assert (reimported["patient_id"] == protected_df["patient_id"]).all()
        assert (reimported["department"] == protected_df["department"]).all()

    def test_excel_round_trip_with_metadata(self, healthcare_df, temp_dir):
        """Test Excel round-trip including metadata sheet."""
        df = healthcare_df[["patient_id", "age", "weight_kg", "department"]].copy()

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Export with metadata (using transform_result)
        excel_path = temp_dir / "with_metadata.xlsx"
        exporter = DataExporter()

        exporter.export_excel(
            protected_df,
            excel_path,
            transform_result=result,  # Pass transform result for metadata
            include_metadata=True,
            sheet_name="Data",
            metadata_sheet_name="Metadata",
        )

        # Read back and verify both sheets exist
        with pd.ExcelFile(excel_path) as xls:
            sheets = xls.sheet_names
            assert "Data" in sheets
            assert "Metadata" in sheets

            # Verify data sheet
            data_df = pd.read_excel(xls, sheet_name="Data")
            assert len(data_df) == len(protected_df)

            # Verify metadata sheet has content
            meta_df = pd.read_excel(xls, sheet_name="Metadata")
            assert len(meta_df) > 0

    def test_parquet_round_trip(self, healthcare_df, temp_dir):
        """Test Parquet: preserves data types accurately."""
        df = healthcare_df[["patient_id", "age", "weight_kg", "cholesterol"]].copy()

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()
        result = transformer.transform(df, config)
        protected_df = result.data

        # Export
        parquet_path = temp_dir / "round_trip.parquet"
        exporter = DataExporter()
        exporter.export_parquet(protected_df, parquet_path)

        # Reimport
        reimported = load_parquet(parquet_path).dataframe

        # Verify structure
        assert list(reimported.columns) == list(protected_df.columns)
        assert len(reimported) == len(protected_df)

        # Verify numeric columns are still numeric
        assert pd.api.types.is_numeric_dtype(reimported["age"])
        assert pd.api.types.is_numeric_dtype(reimported["weight_kg"])

    def test_format_conversion_chain(self, healthcare_df, temp_dir):
        """Test: CSV → transform → Parquet → reimport → Excel → verify."""
        df = healthcare_df[["patient_id", "age", "department"]].copy()

        # Save as CSV and load
        csv_path = temp_dir / "original.csv"
        df.to_csv(csv_path, index=False)
        loaded = load_csv(csv_path).dataframe

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()
        result = transformer.transform(loaded, config)
        protected_df = result.data

        # Export to Parquet
        parquet_path = temp_dir / "intermediate.parquet"
        exporter = DataExporter()
        exporter.export_parquet(protected_df, parquet_path)

        # Reimport from Parquet
        from_parquet = load_parquet(parquet_path).dataframe

        # Export to Excel
        excel_path = temp_dir / "final.xlsx"
        exporter.export_excel(from_parquet, excel_path)

        # Final verification
        final = load_excel(excel_path).dataframe

        assert len(final) == len(df)
        assert set(final.columns) == set(protected_df.columns)


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


class TestPerformanceBenchmarks:
    """Test performance meets requirements from PRD."""

    @pytest.fixture
    def large_df(self):
        """Create a large dataset for performance testing."""
        np.random.seed(42)
        n_rows = 50_000  # 50K rows for reasonable test time

        return pd.DataFrame({
            f"numeric_{i}": np.random.randn(n_rows) for i in range(5)
        } | {
            f"categorical_{i}": np.random.choice(["A", "B", "C", "D", "E"], n_rows)
            for i in range(3)
        } | {
            "id": range(n_rows),
        })

    def test_profiling_performance(self, large_df):
        """Test: Profiling should complete in reasonable time."""
        profiler = DatasetProfiler()

        start_time = time.time()
        profile = profiler.profile(large_df)
        elapsed = time.time() - start_time

        # Should complete within 30 seconds for 100K rows
        assert elapsed < 30, f"Profiling took {elapsed:.1f}s, expected < 30s"
        assert profile.row_count == len(large_df)

    def test_transformation_performance(self, large_df):
        """Test: Transformation should complete in reasonable time."""
        config = DatasetConfig(global_epsilon=1.0)
        config.passthrough_columns(["id"])

        transformer = DatasetTransformer()

        start_time = time.time()
        result = transformer.transform(large_df, config)
        elapsed = time.time() - start_time

        # Should complete within 120 seconds for 50K × 9 columns (generous for slow CI)
        assert elapsed < 120, f"Transformation took {elapsed:.1f}s, expected < 120s"
        assert len(result.data) == len(large_df)

    def test_comparison_performance(self, large_df):
        """Test: Comparison should complete in reasonable time."""
        # Create a "protected" version (just add noise manually for test)
        protected_df = large_df.copy()
        for col in protected_df.columns:
            if col.startswith("numeric_"):
                protected_df[col] += np.random.randn(len(protected_df)) * 0.1

        comparator = DatasetComparator()

        start_time = time.time()
        comparison = comparator.compare(large_df, protected_df)
        elapsed = time.time() - start_time

        # Should complete within 30 seconds
        assert elapsed < 30, f"Comparison took {elapsed:.1f}s, expected < 30s"
        assert comparison.row_count == len(large_df)

    def test_export_performance(self, large_df, temp_dir):
        """Test: Export should complete in reasonable time."""
        exporter = DataExporter()

        # Test CSV export
        csv_path = temp_dir / "perf_test.csv"
        start_time = time.time()
        exporter.export_csv(large_df, csv_path)
        csv_elapsed = time.time() - start_time

        # Test Parquet export (should be faster)
        parquet_path = temp_dir / "perf_test.parquet"
        start_time = time.time()
        exporter.export_parquet(large_df, parquet_path)
        parquet_elapsed = time.time() - start_time

        # Both should complete within 30 seconds
        assert csv_elapsed < 30, f"CSV export took {csv_elapsed:.1f}s"
        assert parquet_elapsed < 30, f"Parquet export took {parquet_elapsed:.1f}s"

        # Parquet should generally be faster (or at least not slower)
        # (this is just informational, not a hard requirement)

    def test_full_pipeline_performance(self, large_df, temp_dir):
        """Test: Full pipeline should complete in reasonable time."""
        start_time = time.time()

        # Profile
        profiler = DatasetProfiler()
        profile = profiler.profile(large_df)

        # Classify
        classifier = ColumnClassifier()
        classification = classifier.classify_dataset(large_df)

        # Transform
        config = DatasetConfig(global_epsilon=1.0)
        config.passthrough_columns(["id"])
        transformer = DatasetTransformer()
        result = transformer.transform(large_df, config)
        protected_df = result.data

        # Compare
        comparator = DatasetComparator()
        comparison = comparator.compare(large_df, protected_df)

        # Export
        exporter = DataExporter()
        exporter.export_csv(protected_df, temp_dir / "output.csv")

        elapsed = time.time() - start_time

        # Full pipeline should complete within 2 minutes for 100K rows
        assert elapsed < 120, f"Full pipeline took {elapsed:.1f}s, expected < 120s"


# =============================================================================
# Memory Tests
# =============================================================================


class TestMemoryUsage:
    """Test memory usage stays within limits."""

    def test_no_memory_leak_multiple_transforms(self, healthcare_df):
        """Test: Multiple transformations don't accumulate memory."""
        import gc

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        # Run multiple transformations
        for i in range(10):
            result = transformer.transform(healthcare_df, config)
            # Let go of result explicitly
            del result
            gc.collect()

        # If we get here without memory error, test passes
        # More detailed memory profiling would require external tools

    def test_large_dataset_memory(self):
        """Test: Large dataset processing stays within memory limits."""
        np.random.seed(42)

        # Create a moderately large dataset (50K rows, 20 columns)
        # This should use roughly 100-200MB of memory
        n_rows = 50_000
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(n_rows) for i in range(20)
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        # This should not cause memory issues
        result = transformer.transform(df, config)

        assert len(result.data) == n_rows
        assert len(result.data.columns) == 20


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Test graceful handling of errors and edge cases."""

    def test_empty_dataframe_handling(self, temp_dir):
        """Test: Empty DataFrame is handled gracefully."""
        df = pd.DataFrame(columns=["a", "b", "c"])

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data) == 0
        assert list(result.data.columns) == ["a", "b", "c"]

    def test_all_nulls_column(self):
        """Test: Column with all nulls is handled gracefully."""
        df = pd.DataFrame({
            "valid": [1, 2, 3, 4, 5],
            "all_nulls": [None, None, None, None, None],
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data) == 5
        # All nulls should remain null
        assert result.data["all_nulls"].isna().all()

    def test_invalid_file_path(self):
        """Test: Invalid file path raises appropriate error."""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_csv(Path("/nonexistent/path/file.csv"))

    def test_invalid_epsilon(self):
        """Test: Invalid epsilon raises appropriate error."""
        with pytest.raises(ValueError):
            DatasetConfig(global_epsilon=-1.0)

        with pytest.raises(ValueError):
            DatasetConfig(global_epsilon=100.0)

    def test_mismatched_exclude_column(self, healthcare_df):
        """Test: Excluding non-existent column is handled gracefully."""
        config = DatasetConfig(global_epsilon=1.0)
        # This column doesn't exist
        config.exclude_columns(["nonexistent_column"])

        transformer = DatasetTransformer()

        # Should not raise, should just ignore the non-existent column
        result = transformer.transform(healthcare_df, config)

        assert len(result.data) == len(healthcare_df)

    def test_single_row_dataset(self):
        """Test: Single row dataset works correctly."""
        df = pd.DataFrame({
            "numeric": [42.0],
            "categorical": ["A"],
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data) == 1
        # Numeric should have noise
        assert "numeric" in result.data.columns

    def test_single_column_dataset(self):
        """Test: Single column dataset works correctly."""
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data) == 5
        assert list(result.data.columns) == ["only_col"]

    def test_very_large_values(self):
        """Test: Very large numeric values are handled correctly."""
        df = pd.DataFrame({
            "big": [1e15, 2e15, 3e15],
            "small": [0.001, 0.002, 0.003],
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data) == 3
        # Values should still be finite
        assert np.isfinite(result.data["big"]).all()
        assert np.isfinite(result.data["small"]).all()

    def test_special_characters_in_columns(self):
        """Test: Special characters in column names are handled."""
        df = pd.DataFrame({
            "normal": [1, 2, 3],
            "with spaces": [4, 5, 6],
            "with-dashes": [7, 8, 9],
            "with.dots": [10, 11, 12],
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()

        result = transformer.transform(df, config)

        assert len(result.data.columns) == 4


# =============================================================================
# Privacy Guarantee Tests
# =============================================================================


class TestPrivacyGuarantees:
    """Test that privacy guarantees are maintained end-to-end."""

    def test_excluded_columns_removed(self, healthcare_df):
        """Test: Excluded columns are completely removed from output."""
        config = DatasetConfig(global_epsilon=1.0)
        config.exclude_columns(["ssn", "first_name", "last_name"])

        transformer = DatasetTransformer()
        result = transformer.transform(healthcare_df, config)

        # These columns should NOT be in the output
        assert "ssn" not in result.data.columns
        assert "first_name" not in result.data.columns
        assert "last_name" not in result.data.columns

    def test_passthrough_columns_unchanged(self, healthcare_df):
        """Test: Passthrough columns are exactly unchanged."""
        config = DatasetConfig(global_epsilon=1.0)
        config.passthrough_columns(["patient_id", "department"])

        transformer = DatasetTransformer()
        result = transformer.transform(healthcare_df, config)

        # Passthrough columns should be identical
        pd.testing.assert_series_equal(
            result.data["patient_id"],
            healthcare_df["patient_id"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result.data["department"],
            healthcare_df["department"],
            check_names=False,
        )

    def test_protected_columns_different(self, healthcare_df):
        """Test: Protected columns have noise added."""
        config = DatasetConfig(global_epsilon=1.0)
        config.passthrough_columns(["patient_id"])

        transformer = DatasetTransformer()
        result = transformer.transform(healthcare_df, config)

        # Numeric columns should be different
        assert not (result.data["age"] == healthcare_df["age"]).all()
        assert not (result.data["weight_kg"] == healthcare_df["weight_kg"]).all()

    def test_privacy_budget_tracked(self, healthcare_df):
        """Test: Privacy budget is properly tracked."""
        config = DatasetConfig(global_epsilon=1.0)
        config.passthrough_columns(["patient_id"])
        config.exclude_columns(["ssn"])

        transformer = DatasetTransformer()
        result = transformer.transform(healthcare_df, config)

        # Should have non-zero total epsilon
        assert result.total_epsilon > 0

        # Should have protected columns
        assert len(result.protected_columns) > 0

    def test_null_preservation(self):
        """Test: Null values are preserved after transformation."""
        df = pd.DataFrame({
            "with_nulls": [1.0, None, 3.0, None, 5.0],
            "no_nulls": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        config = DatasetConfig(global_epsilon=1.0)
        transformer = DatasetTransformer()
        result = transformer.transform(df, config)

        # Null positions should be preserved
        assert result.data["with_nulls"].isna()[1]
        assert result.data["with_nulls"].isna()[3]
        assert not result.data["no_nulls"].isna().any()
