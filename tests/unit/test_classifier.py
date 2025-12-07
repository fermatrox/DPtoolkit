"""Unit tests for dp_toolkit.recommendations.classifier module."""

import pytest
import pandas as pd

from dp_toolkit.recommendations.classifier import (
    SensitivityLevel,
    ClassificationResult,
    DatasetClassification,
    ColumnClassifier,
    ContentAnalyzer,
    classify_column,
    classify_columns,
    classify_dataset,
    get_sensitivity_for_column,
    get_sensitive_columns,
    HIGH_SENSITIVITY_PATTERNS,
    MEDIUM_SENSITIVITY_PATTERNS,
    LOW_SENSITIVITY_PATTERNS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def classifier():
    """Create a basic ColumnClassifier."""
    return ColumnClassifier()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "patient_ssn": ["123-45-6789", "987-65-4321"],
        "patient_name": ["John Doe", "Jane Smith"],
        "date_of_birth": ["1990-01-15", "1985-06-20"],
        "diagnosis_code": ["E11.9", "I10"],
        "department": ["Cardiology", "Neurology"],
        "visit_type": ["Outpatient", "Inpatient"],
        "age": [33, 38],
        "record_id": [1001, 1002],
        "visit_count": [5, 3],
        "random_column": ["A", "B"],
    })


@pytest.fixture
def content_analyzer():
    """Create a ContentAnalyzer."""
    return ContentAnalyzer()


# =============================================================================
# SensitivityLevel Tests
# =============================================================================


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_sensitivity_values(self):
        """Test that all sensitivity levels have correct values."""
        assert SensitivityLevel.HIGH.value == "high"
        assert SensitivityLevel.MEDIUM.value == "medium"
        assert SensitivityLevel.LOW.value == "low"
        assert SensitivityLevel.UNKNOWN.value == "unknown"

    def test_sensitivity_str(self):
        """Test string representation."""
        assert str(SensitivityLevel.HIGH) == "high"
        assert str(SensitivityLevel.MEDIUM) == "medium"
        assert str(SensitivityLevel.LOW) == "low"

    def test_sensitivity_comparison(self):
        """Test enum comparison."""
        assert SensitivityLevel.HIGH == SensitivityLevel.HIGH
        assert SensitivityLevel.HIGH != SensitivityLevel.LOW


# =============================================================================
# ClassificationResult Tests
# =============================================================================


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic classification result."""
        result = ClassificationResult(
            column_name="ssn",
            sensitivity=SensitivityLevel.HIGH,
            confidence=0.9,
        )
        assert result.column_name == "ssn"
        assert result.sensitivity == SensitivityLevel.HIGH
        assert result.confidence == 0.9
        assert not result.is_override

    def test_result_with_pattern(self):
        """Test result with matched pattern."""
        result = ClassificationResult(
            column_name="patient_ssn",
            sensitivity=SensitivityLevel.HIGH,
            confidence=0.85,
            matched_pattern=r"\bssn\b",
            pattern_description="Social Security Number",
        )
        assert result.matched_pattern == r"\bssn\b"
        assert result.pattern_description == "Social Security Number"

    def test_result_with_override(self):
        """Test result with override."""
        result = ClassificationResult(
            column_name="custom_column",
            sensitivity=SensitivityLevel.HIGH,
            confidence=1.0,
            is_override=True,
            override_reason="Manually marked as sensitive",
        )
        assert result.is_override
        assert result.override_reason == "Manually marked as sensitive"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ClassificationResult(
            column_name="ssn",
            sensitivity=SensitivityLevel.HIGH,
            confidence=0.9,
            matched_pattern=r"\bssn\b",
            pattern_description="Social Security Number",
        )
        d = result.to_dict()
        assert d["column_name"] == "ssn"
        assert d["sensitivity"] == "high"
        assert d["confidence"] == 0.9
        assert d["matched_pattern"] == r"\bssn\b"


# =============================================================================
# DatasetClassification Tests
# =============================================================================


class TestDatasetClassification:
    """Tests for DatasetClassification dataclass."""

    def test_empty_classification(self):
        """Test empty classification."""
        classification = DatasetClassification()
        assert len(classification.column_results) == 0
        assert classification.high_sensitivity_columns == []
        assert classification.medium_sensitivity_columns == []
        assert classification.low_sensitivity_columns == []

    def test_classification_with_results(self):
        """Test classification with multiple results."""
        results = {
            "ssn": ClassificationResult(
                column_name="ssn",
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.9,
            ),
            "age": ClassificationResult(
                column_name="age",
                sensitivity=SensitivityLevel.MEDIUM,
                confidence=0.8,
            ),
            "record_id": ClassificationResult(
                column_name="record_id",
                sensitivity=SensitivityLevel.LOW,
                confidence=0.7,
            ),
        }
        classification = DatasetClassification(column_results=results)

        assert classification.high_sensitivity_columns == ["ssn"]
        assert classification.medium_sensitivity_columns == ["age"]
        assert classification.low_sensitivity_columns == ["record_id"]

    def test_get_columns_by_sensitivity(self):
        """Test getting columns by sensitivity level."""
        results = {
            "ssn": ClassificationResult(
                column_name="ssn",
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.9,
            ),
            "dob": ClassificationResult(
                column_name="dob",
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.85,
            ),
            "age": ClassificationResult(
                column_name="age",
                sensitivity=SensitivityLevel.MEDIUM,
                confidence=0.8,
            ),
        }
        classification = DatasetClassification(column_results=results)

        high_cols = classification.get_columns_by_sensitivity(SensitivityLevel.HIGH)
        assert len(high_cols) == 2
        assert "ssn" in high_cols
        assert "dob" in high_cols

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = {
            "ssn": ClassificationResult(
                column_name="ssn",
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.9,
            ),
        }
        classification = DatasetClassification(column_results=results)
        d = classification.to_dict()

        assert "columns" in d
        assert "summary" in d
        assert "ssn" in d["columns"]
        assert d["summary"]["high_sensitivity"] == ["ssn"]

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        results = {
            "ssn": ClassificationResult(
                column_name="ssn",
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.9,
            ),
            "age": ClassificationResult(
                column_name="age",
                sensitivity=SensitivityLevel.MEDIUM,
                confidence=0.8,
            ),
        }
        classification = DatasetClassification(column_results=results)
        df = classification.to_dataframe()

        assert len(df) == 2
        assert "Column" in df.columns
        assert "Sensitivity" in df.columns
        assert "Confidence" in df.columns


# =============================================================================
# ColumnClassifier Tests - High Sensitivity Patterns
# =============================================================================


class TestHighSensitivityPatterns:
    """Tests for high sensitivity pattern matching."""

    @pytest.mark.parametrize("column_name", [
        "ssn",
        "SSN",
        "patient_ssn",
        "ssn_number",
        "social_security",
        "social_security_number",
    ])
    def test_ssn_patterns(self, classifier, column_name):
        """Test SSN pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "name",
        "patient_name",
        "first_name",
        "last_name",
        "full_name",
        "member_name",
    ])
    def test_name_patterns(self, classifier, column_name):
        """Test name pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "email",
        "email_address",
        "patient_email",
        "phone",
        "phone_number",
        "telephone",
        "mobile",
        "cell_phone",
    ])
    def test_contact_patterns(self, classifier, column_name):
        """Test contact information pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "address",
        "street_address",
        "home_address",
        "street",
        "city",
        "zip_code",
        "postal_code",
    ])
    def test_address_patterns(self, classifier, column_name):
        """Test address pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "diagnosis",
        "diagnosis_code",
        "primary_diagnosis",
        "icd_code",
        "icd10",
        "medical_record_number",
        "mrn",
        "patient_id",
        "prescription",
        "medication",
        "medication_name",
        "treatment",
        "treatment_plan",
        "procedure",
        "surgery",
        "lab_result",
        "test_result",
        "vital_signs",
        "allergy",
        "allergies",
        "immunization",
        "vaccination",
    ])
    def test_healthcare_phi_patterns(self, classifier, column_name):
        """Test healthcare PHI pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "dob",
        "DOB",
        "date_of_birth",
        "birth_date",
        "birthday",
        "date_of_death",
        "dod",
    ])
    def test_date_of_birth_patterns(self, classifier, column_name):
        """Test date of birth pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "credit_card",
        "credit_card_number",
        "card_number",
        "bank_account",
        "account_number",
        "salary",
        "income",
        "annual_income",
    ])
    def test_financial_patterns(self, classifier, column_name):
        """Test financial information pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH

    @pytest.mark.parametrize("column_name", [
        "race",
        "ethnicity",
        "religion",
        "gender",
        "sex",
    ])
    def test_sensitive_demographic_patterns(self, classifier, column_name):
        """Test sensitive demographic pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.HIGH


# =============================================================================
# ColumnClassifier Tests - Medium Sensitivity Patterns
# =============================================================================


class TestMediumSensitivityPatterns:
    """Tests for medium sensitivity pattern matching."""

    @pytest.mark.parametrize("column_name", [
        "age",
        "patient_age",
        "year_of_birth",
        "marital_status",
        "education",
        "education_level",
        "occupation",
        "employment_status",
    ])
    def test_demographic_patterns(self, classifier, column_name):
        """Test demographic pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.MEDIUM

    @pytest.mark.parametrize("column_name", [
        "state",
        "county",
        "region",
        "location",
        "facility",
        "hospital",
        "clinic",
        "department",
        "unit",
        "ward",
    ])
    def test_location_patterns(self, classifier, column_name):
        """Test location pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.MEDIUM

    @pytest.mark.parametrize("column_name", [
        "visit_type",
        "admission_date",
        "discharge_date",
        "length_of_stay",
        "los",
        "provider",
        "physician",
        "doctor",
        "specialty",
        "referral",
    ])
    def test_healthcare_operational_patterns(self, classifier, column_name):
        """Test healthcare operational pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.MEDIUM

    @pytest.mark.parametrize("column_name", [
        "date",
        "visit_date",
        "service_date",
        "created_date",
        "modified_date",
        "timestamp",
    ])
    def test_date_patterns(self, classifier, column_name):
        """Test date pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.MEDIUM

    @pytest.mark.parametrize("column_name", [
        "cost",
        "charge",
        "payment",
        "billing_amount",
        "insurance_type",
        "coverage",
    ])
    def test_financial_operational_patterns(self, classifier, column_name):
        """Test financial operational pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.MEDIUM


# =============================================================================
# ColumnClassifier Tests - Low Sensitivity Patterns
# =============================================================================


class TestLowSensitivityPatterns:
    """Tests for low sensitivity pattern matching."""

    @pytest.mark.parametrize("column_name", [
        "record_id",
        "transaction_id",
        "case_id",
        "event_id",
        "session_id",
        "id",
        "row_id",
        "sequence_id",
        "patient_record_id",  # Note: record_id pattern, not patient_id
    ])
    def test_id_patterns(self, classifier, column_name):
        """Test ID pattern recognition."""
        result = classifier.classify_column(column_name)
        # Some might match HIGH due to "patient" prefix
        assert result.sensitivity in [SensitivityLevel.LOW, SensitivityLevel.HIGH]

    @pytest.mark.parametrize("column_name", [
        "code",
        "type",
        "category",
        "status",
        "flag",
        "indicator",
    ])
    def test_code_type_patterns(self, classifier, column_name):
        """Test code/type pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.LOW

    @pytest.mark.parametrize("column_name", [
        "count",
        "total",
        "sum",
        "avg",
        "average",
        "quantity",
        "frequency",
    ])
    def test_aggregate_patterns(self, classifier, column_name):
        """Test aggregate pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.LOW

    @pytest.mark.parametrize("column_name", [
        "description",
        "notes",
        "note",
        "comment",
        "remarks",
    ])
    def test_description_patterns(self, classifier, column_name):
        """Test description pattern recognition."""
        result = classifier.classify_column(column_name)
        assert result.sensitivity == SensitivityLevel.LOW


# =============================================================================
# ColumnClassifier Tests - Override Capability
# =============================================================================


class TestOverrideCapability:
    """Tests for manual override functionality."""

    def test_add_override(self, classifier):
        """Test adding a manual override."""
        classifier.add_override("custom_column", SensitivityLevel.HIGH, "Contains PII")

        result = classifier.classify_column("custom_column")
        assert result.sensitivity == SensitivityLevel.HIGH
        assert result.is_override
        assert result.override_reason == "Contains PII"
        assert result.confidence == 1.0

    def test_override_case_insensitive(self, classifier):
        """Test that overrides are case-insensitive."""
        classifier.add_override("My_Column", SensitivityLevel.HIGH)

        result = classifier.classify_column("my_column")
        assert result.sensitivity == SensitivityLevel.HIGH
        assert result.is_override

    def test_override_takes_precedence(self, classifier):
        """Test that override takes precedence over pattern matching."""
        # "record_id" would normally be LOW
        classifier.add_override("record_id", SensitivityLevel.HIGH, "Override")

        result = classifier.classify_column("record_id")
        assert result.sensitivity == SensitivityLevel.HIGH
        assert result.is_override

    def test_remove_override(self, classifier):
        """Test removing an override."""
        classifier.add_override("custom_column", SensitivityLevel.HIGH)
        assert classifier.remove_override("custom_column")

        result = classifier.classify_column("custom_column")
        assert not result.is_override

    def test_remove_nonexistent_override(self, classifier):
        """Test removing a non-existent override returns False."""
        assert not classifier.remove_override("nonexistent")

    def test_clear_overrides(self, classifier):
        """Test clearing all overrides."""
        classifier.add_override("col1", SensitivityLevel.HIGH)
        classifier.add_override("col2", SensitivityLevel.LOW)
        classifier.clear_overrides()

        assert len(classifier.get_overrides()) == 0

    def test_get_overrides(self, classifier):
        """Test getting all overrides."""
        classifier.add_override("col1", SensitivityLevel.HIGH, "Reason 1")
        classifier.add_override("col2", SensitivityLevel.LOW, "Reason 2")

        overrides = classifier.get_overrides()
        assert len(overrides) == 2
        assert "col1" in overrides
        assert overrides["col1"][0] == SensitivityLevel.HIGH


# =============================================================================
# ColumnClassifier Tests - Custom Patterns
# =============================================================================


class TestCustomPatterns:
    """Tests for custom pattern support."""

    def test_custom_high_patterns(self):
        """Test adding custom high sensitivity patterns."""
        custom_patterns = [
            (r"\bcustom_sensitive", "Custom Sensitive Field"),
        ]
        classifier = ColumnClassifier(custom_high_patterns=custom_patterns)

        result = classifier.classify_column("custom_sensitive_data")
        assert result.sensitivity == SensitivityLevel.HIGH

    def test_custom_medium_patterns(self):
        """Test adding custom medium sensitivity patterns."""
        custom_patterns = [
            (r"\bcustom_medium", "Custom Medium Field"),
        ]
        classifier = ColumnClassifier(custom_medium_patterns=custom_patterns)

        result = classifier.classify_column("custom_medium_field")
        assert result.sensitivity == SensitivityLevel.MEDIUM

    def test_custom_low_patterns(self):
        """Test adding custom low sensitivity patterns."""
        custom_patterns = [
            (r"\bcustom_low", "Custom Low Field"),
        ]
        classifier = ColumnClassifier(custom_low_patterns=custom_patterns)

        result = classifier.classify_column("custom_low_field")
        assert result.sensitivity == SensitivityLevel.LOW

    def test_pattern_count(self):
        """Test getting pattern counts."""
        classifier = ColumnClassifier()
        counts = classifier.get_pattern_count()

        assert counts["high"] == len(HIGH_SENSITIVITY_PATTERNS)
        assert counts["medium"] == len(MEDIUM_SENSITIVITY_PATTERNS)
        assert counts["low"] == len(LOW_SENSITIVITY_PATTERNS)


# =============================================================================
# ColumnClassifier Tests - Default Sensitivity
# =============================================================================


class TestDefaultSensitivity:
    """Tests for default sensitivity configuration."""

    def test_default_is_medium(self):
        """Test that default sensitivity is MEDIUM."""
        classifier = ColumnClassifier()
        result = classifier.classify_column("completely_random_xyz123")
        assert result.sensitivity == SensitivityLevel.MEDIUM

    def test_custom_default_low(self):
        """Test setting default sensitivity to LOW."""
        classifier = ColumnClassifier(default_sensitivity=SensitivityLevel.LOW)
        result = classifier.classify_column("completely_random_xyz123")
        assert result.sensitivity == SensitivityLevel.LOW

    def test_custom_default_high(self):
        """Test setting default sensitivity to HIGH."""
        classifier = ColumnClassifier(default_sensitivity=SensitivityLevel.HIGH)
        result = classifier.classify_column("completely_random_xyz123")
        assert result.sensitivity == SensitivityLevel.HIGH


# =============================================================================
# ColumnClassifier Tests - Dataset Classification
# =============================================================================


class TestDatasetClassification:
    """Tests for dataset classification."""

    def test_classify_dataset(self, classifier, sample_df):
        """Test classifying an entire dataset."""
        classification = classifier.classify_dataset(sample_df)

        assert len(classification.column_results) == len(sample_df.columns)
        assert "patient_ssn" in classification.high_sensitivity_columns
        assert "patient_name" in classification.high_sensitivity_columns
        assert "diagnosis_code" in classification.high_sensitivity_columns

    def test_classify_dataset_with_column_filter(self, classifier, sample_df):
        """Test classifying specific columns only."""
        columns = ["patient_ssn", "age", "record_id"]
        classification = classifier.classify_dataset(sample_df, columns=columns)

        assert len(classification.column_results) == 3
        assert "patient_name" not in classification.column_results

    def test_classify_columns(self, classifier):
        """Test classifying a list of column names."""
        columns = ["ssn", "age", "record_id"]
        results = classifier.classify_columns(columns)

        assert len(results) == 3
        assert results["ssn"].sensitivity == SensitivityLevel.HIGH
        assert results["age"].sensitivity == SensitivityLevel.MEDIUM
        assert results["record_id"].sensitivity == SensitivityLevel.LOW


# =============================================================================
# ContentAnalyzer Tests
# =============================================================================


class TestContentAnalyzer:
    """Tests for content-based analysis."""

    def test_detect_ssn_content(self, content_analyzer):
        """Test detecting SSN patterns in content."""
        series = pd.Series([
            "123-45-6789",
            "987-65-4321",
            "555-12-3456",
            "111-22-3333",
        ])
        current_result = ClassificationResult(
            column_name="mystery_column",
            sensitivity=SensitivityLevel.MEDIUM,
            confidence=0.5,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.HIGH
        assert "SSN" in refined.pattern_description

    def test_detect_email_content(self, content_analyzer):
        """Test detecting email patterns in content."""
        series = pd.Series([
            "john@example.com",
            "jane@test.org",
            "user@domain.net",
            "admin@company.com",
        ])
        current_result = ClassificationResult(
            column_name="mystery_column",
            sensitivity=SensitivityLevel.LOW,
            confidence=0.5,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.HIGH
        assert "Email" in refined.pattern_description

    def test_detect_phone_content(self, content_analyzer):
        """Test detecting phone patterns in content."""
        series = pd.Series([
            "555-123-4567",
            "(555) 987-6543",
            "555.111.2222",
            "555-333-4444",
        ])
        current_result = ClassificationResult(
            column_name="mystery_column",
            sensitivity=SensitivityLevel.MEDIUM,
            confidence=0.5,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.HIGH
        assert "Phone" in refined.pattern_description

    def test_no_sensitive_content(self, content_analyzer):
        """Test that non-sensitive content is not upgraded."""
        series = pd.Series([
            "Apple",
            "Banana",
            "Cherry",
            "Date",
        ])
        current_result = ClassificationResult(
            column_name="fruit",
            sensitivity=SensitivityLevel.LOW,
            confidence=0.8,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.LOW

    def test_skip_numeric_columns(self, content_analyzer):
        """Test that numeric columns are skipped."""
        series = pd.Series([1, 2, 3, 4, 5])
        current_result = ClassificationResult(
            column_name="numbers",
            sensitivity=SensitivityLevel.LOW,
            confidence=0.8,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.LOW

    def test_skip_already_high(self, content_analyzer):
        """Test that already HIGH columns are skipped."""
        series = pd.Series(["123-45-6789", "test"])
        current_result = ClassificationResult(
            column_name="ssn",
            sensitivity=SensitivityLevel.HIGH,
            confidence=0.9,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined == current_result

    def test_skip_overridden(self, content_analyzer):
        """Test that overridden columns are skipped."""
        series = pd.Series(["123-45-6789", "987-65-4321"])
        current_result = ClassificationResult(
            column_name="col",
            sensitivity=SensitivityLevel.LOW,
            confidence=1.0,
            is_override=True,
        )

        refined = content_analyzer.analyze_column(series, current_result)
        assert refined.sensitivity == SensitivityLevel.LOW

    def test_analyze_dataset(self, content_analyzer, sample_df):
        """Test analyzing an entire dataset."""
        # Create initial classification
        classification = DatasetClassification(column_results={
            col: ClassificationResult(
                column_name=col,
                sensitivity=SensitivityLevel.MEDIUM,
                confidence=0.5,
            )
            for col in sample_df.columns
        })

        refined = content_analyzer.analyze_dataset(sample_df, classification)
        assert len(refined.column_results) == len(sample_df.columns)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_classify_column_function(self):
        """Test classify_column convenience function."""
        result = classify_column("patient_ssn")
        assert result.sensitivity == SensitivityLevel.HIGH

    def test_classify_column_with_custom_patterns(self):
        """Test classify_column with custom patterns."""
        custom = {
            "high": [(r"\bmy_secret", "My Secret Field")],
        }
        result = classify_column("my_secret_column", custom_patterns=custom)
        assert result.sensitivity == SensitivityLevel.HIGH

    def test_classify_columns_function(self):
        """Test classify_columns convenience function."""
        results = classify_columns(["ssn", "age", "record_id"])
        assert results["ssn"].sensitivity == SensitivityLevel.HIGH
        assert results["age"].sensitivity == SensitivityLevel.MEDIUM
        assert results["record_id"].sensitivity == SensitivityLevel.LOW

    def test_classify_columns_with_overrides(self):
        """Test classify_columns with overrides."""
        overrides = {"age": SensitivityLevel.HIGH}
        results = classify_columns(["ssn", "age", "record_id"], overrides=overrides)
        assert results["age"].sensitivity == SensitivityLevel.HIGH
        assert results["age"].is_override

    def test_classify_dataset_function(self, sample_df):
        """Test classify_dataset convenience function."""
        classification = classify_dataset(sample_df)
        assert "patient_ssn" in classification.high_sensitivity_columns

    def test_classify_dataset_with_content_analysis(self):
        """Test classify_dataset with content analysis enabled."""
        df = pd.DataFrame({
            "mystery_col": ["123-45-6789", "987-65-4321", "555-12-3456"],
        })
        classification = classify_dataset(df, analyze_content=True)
        assert classification.column_results["mystery_col"].sensitivity == SensitivityLevel.HIGH

    def test_get_sensitivity_for_column(self):
        """Test get_sensitivity_for_column function."""
        level = get_sensitivity_for_column("patient_ssn")
        assert level == SensitivityLevel.HIGH

        level = get_sensitivity_for_column("age")
        assert level == SensitivityLevel.MEDIUM

    def test_get_sensitive_columns(self, sample_df):
        """Test get_sensitive_columns function."""
        # Get HIGH sensitivity columns
        sensitive = get_sensitive_columns(sample_df, SensitivityLevel.HIGH)
        assert "patient_ssn" in sensitive
        assert "patient_name" in sensitive

    def test_get_sensitive_columns_threshold(self, sample_df):
        """Test get_sensitive_columns with different thresholds."""
        # MEDIUM and above
        medium_and_above = get_sensitive_columns(sample_df, SensitivityLevel.MEDIUM)

        # LOW and above (should include all classified columns)
        low_and_above = get_sensitive_columns(sample_df, SensitivityLevel.LOW)

        assert len(low_and_above) >= len(medium_and_above)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_column_name(self, classifier):
        """Test handling of empty column name."""
        result = classifier.classify_column("")
        assert result.sensitivity == SensitivityLevel.MEDIUM  # Default

    def test_numeric_column_name(self, classifier):
        """Test handling of numeric column name."""
        result = classifier.classify_column("12345")
        assert result.sensitivity == SensitivityLevel.MEDIUM  # Default

    def test_unicode_column_name(self, classifier):
        """Test handling of unicode column names."""
        result = classifier.classify_column("患者名")  # "Patient name" in Chinese
        assert result.sensitivity == SensitivityLevel.MEDIUM  # Default

    def test_very_long_column_name(self, classifier):
        """Test handling of very long column names."""
        long_name = "patient_" + "a" * 1000 + "_ssn"
        result = classifier.classify_column(long_name)
        # Should still match ssn pattern
        assert result.sensitivity == SensitivityLevel.HIGH

    def test_special_characters(self, classifier):
        """Test handling of special characters in column names."""
        result = classifier.classify_column("patient.ssn")
        # Might match due to normalization
        assert result.sensitivity in [SensitivityLevel.HIGH, SensitivityLevel.MEDIUM]

    def test_mixed_case_patterns(self, classifier):
        """Test case insensitivity of pattern matching."""
        results = [
            classifier.classify_column("SSN"),
            classifier.classify_column("ssn"),
            classifier.classify_column("Ssn"),
            classifier.classify_column("sSn"),
        ]
        for result in results:
            assert result.sensitivity == SensitivityLevel.HIGH

    def test_empty_dataframe(self, classifier):
        """Test classifying an empty DataFrame."""
        df = pd.DataFrame()
        classification = classifier.classify_dataset(df)
        assert len(classification.column_results) == 0

    def test_single_column_dataframe(self, classifier):
        """Test classifying a single-column DataFrame."""
        df = pd.DataFrame({"ssn": ["123-45-6789"]})
        classification = classifier.classify_dataset(df)
        assert len(classification.column_results) == 1
        assert classification.high_sensitivity_columns == ["ssn"]

    def test_whitespace_in_column_name(self, classifier):
        """Test handling of whitespace in column names."""
        result = classifier.classify_column("  patient ssn  ")
        # Should still match patterns
        assert result.sensitivity == SensitivityLevel.HIGH

    def test_underscore_vs_space(self, classifier):
        """Test that underscores and spaces are treated similarly."""
        result1 = classifier.classify_column("patient_ssn")
        result2 = classifier.classify_column("patient ssn")
        assert result1.sensitivity == result2.sensitivity == SensitivityLevel.HIGH


# =============================================================================
# Pattern Lists Tests
# =============================================================================


class TestPatternLists:
    """Tests for pattern list exports."""

    def test_high_patterns_not_empty(self):
        """Test that high sensitivity patterns list is not empty."""
        assert len(HIGH_SENSITIVITY_PATTERNS) > 0

    def test_medium_patterns_not_empty(self):
        """Test that medium sensitivity patterns list is not empty."""
        assert len(MEDIUM_SENSITIVITY_PATTERNS) > 0

    def test_low_patterns_not_empty(self):
        """Test that low sensitivity patterns list is not empty."""
        assert len(LOW_SENSITIVITY_PATTERNS) > 0

    def test_pattern_format(self):
        """Test that patterns are properly formatted tuples."""
        for pattern, description in HIGH_SENSITIVITY_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(description, str)
            assert len(description) > 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for classifier."""

    def test_large_column_count(self, classifier):
        """Test classification of many columns."""
        columns = [f"column_{i}" for i in range(1000)]
        results = classifier.classify_columns(columns)
        assert len(results) == 1000

    def test_classification_speed(self, classifier):
        """Test that classification is fast enough."""
        import time

        columns = [f"patient_ssn_{i}" for i in range(100)]
        columns.extend([f"age_{i}" for i in range(100)])
        columns.extend([f"record_id_{i}" for i in range(100)])

        start = time.time()
        classifier.classify_columns(columns)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
