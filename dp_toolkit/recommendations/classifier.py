"""Column sensitivity classifier for automatic privacy recommendations.

This module provides automatic classification of column sensitivity levels
based on column name patterns and optional content analysis. It supports
healthcare-specific patterns for HIPAA/GDPR compliance.

Sensitivity levels:
- HIGH: Direct identifiers, protected health information (PHI)
- MEDIUM: Quasi-identifiers, demographic data
- LOW: General operational data, aggregated metrics
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Enums
# =============================================================================


class SensitivityLevel(Enum):
    """Sensitivity classification levels for columns."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


# =============================================================================
# Pattern Definitions
# =============================================================================


# High sensitivity patterns - Direct identifiers and PHI
HIGH_SENSITIVITY_PATTERNS: List[Tuple[str, str]] = [
    # Direct identifiers
    (r"\bssn\b", "Social Security Number"),
    (r"\bsocial.*security", "Social Security Number"),
    (r"\bnational.*id", "National ID"),
    (r"\bpassport", "Passport Number"),
    (r"\bdriver.*licen[cs]e", "Driver's License"),
    (r"\blicen[cs]e.*num", "License Number"),
    # Names
    (r"\b(first|last|middle|full|patient|member).*name\b", "Personal Name"),
    (r"\bname\b", "Name field"),
    (r"\bsurname\b", "Surname"),
    (r"\bgiven.*name", "Given Name"),
    # Contact information
    (r"\bemail", "Email Address"),
    (r"\bphone", "Phone Number"),
    (r"\btelephone", "Telephone Number"),
    (r"\bmobile", "Mobile Number"),
    (r"\bcell", "Cell Number"),
    (r"\bfax", "Fax Number"),
    (r"\baddress", "Physical Address"),
    (r"\bstreet", "Street Address"),
    (r"\bcity\b", "City"),
    (r"\bzip", "ZIP Code"),
    (r"\bpostal", "Postal Code"),
    # Healthcare specific - PHI under HIPAA
    (r"\bdiagnos", "Diagnosis"),
    (r"\bicd", "ICD Code"),
    (r"\bmedical.*record", "Medical Record Number"),
    (r"\bmrn\b", "Medical Record Number"),
    (r"\bpatient.*id", "Patient ID"),
    (r"\bhealth.*id", "Health ID"),
    (r"\binsurance.*id", "Insurance ID"),
    (r"\bpolicy.*num", "Policy Number"),
    (r"\bprescription", "Prescription"),
    (r"\bmedication", "Medication"),
    (r"\bdrug\b", "Drug"),
    (r"\btreatment", "Treatment"),
    (r"\bprocedure", "Procedure"),
    (r"\bsurgery", "Surgery"),
    (r"\blab.*result", "Lab Result"),
    (r"\btest.*result", "Test Result"),
    (r"\bvital", "Vital Signs"),
    (r"\ballerg", "Allergy"),
    (r"\bimmuni[zs]", "Immunization"),
    (r"\bvaccin", "Vaccination"),
    # Dates of birth / sensitive dates
    (r"\bdob\b", "Date of Birth"),
    (r"\bdate.*birth", "Date of Birth"),
    (r"\bbirth.*date", "Birth Date"),
    (r"\bbirthday", "Birthday"),
    (r"\bborn\b", "Birth date"),
    (r"\bdeath", "Date of Death"),
    (r"\bdod\b", "Date of Death"),
    # Financial
    (r"\bcredit.*card", "Credit Card"),
    (r"\bcard.*num", "Card Number"),
    (r"\bbank.*account", "Bank Account"),
    (r"\biban\b", "IBAN"),
    (r"\brouting.*num", "Routing Number"),
    (r"\baccount.*num", "Account Number"),
    (r"\bsalary", "Salary"),
    (r"\bincome", "Income"),
    (r"\bwage", "Wage"),
    # Biometric / genetic
    (r"\bbiometric", "Biometric Data"),
    (r"\bfingerprint", "Fingerprint"),
    (r"\bretina", "Retina Scan"),
    (r"\bgenetic", "Genetic Data"),
    (r"\bdna\b", "DNA"),
    (r"\bgenome", "Genome"),
    # Sensitive demographics
    (r"\brace\b", "Race"),
    (r"\bethnicit", "Ethnicity"),
    (r"\breligion", "Religion"),
    (r"\bsex\b", "Sex"),
    (r"\bgender", "Gender"),
    (r"\bsexual.*orient", "Sexual Orientation"),
    (r"\bpolitical", "Political Affiliation"),
    (r"\bunion.*member", "Union Membership"),
]

# Medium sensitivity patterns - Quasi-identifiers
MEDIUM_SENSITIVITY_PATTERNS: List[Tuple[str, str]] = [
    # Demographics (non-sensitive)
    (r"\bage\b", "Age"),
    (r"\byear.*birth", "Year of Birth"),
    (r"\bmarital", "Marital Status"),
    (r"\beducation", "Education Level"),
    (r"\boccupation", "Occupation"),
    (r"\bemployment", "Employment Status"),
    (r"\bjob\b", "Job"),
    # Location (general)
    (r"\bstate\b", "State"),
    (r"\bcounty", "County"),
    (r"\bregion", "Region"),
    (r"\barea\b", "Area"),
    (r"\blocation", "Location"),
    (r"\bfacility", "Facility"),
    (r"\bclinic", "Clinic"),
    (r"\bhospital", "Hospital"),
    (r"\bdepartment", "Department"),
    (r"\bunit\b", "Unit"),
    (r"\bward\b", "Ward"),
    # Healthcare operational
    (r"\bvisit.*type", "Visit Type"),
    (r"\badmission", "Admission"),
    (r"\bdischarge", "Discharge"),
    (r"\blength.*stay", "Length of Stay"),
    (r"\blos\b", "Length of Stay"),
    (r"\bprovider", "Provider"),
    (r"\bphysician", "Physician"),
    (r"\bdoctor", "Doctor"),
    (r"\bnurse", "Nurse"),
    (r"\bspecialt", "Specialty"),
    (r"\breferral", "Referral"),
    # Dates (non-birth)
    (r"\bdate\b", "Date"),
    (r"\btime\b", "Time"),
    (r"\btimestamp", "Timestamp"),
    (r"\bvisit.*date", "Visit Date"),
    (r"\bservice.*date", "Service Date"),
    (r"\bcreated", "Created Date"),
    (r"\bmodified", "Modified Date"),
    (r"\bupdated", "Updated Date"),
    # Financial (non-personal)
    (r"\bcost\b", "Cost"),
    (r"\bcharge", "Charge"),
    (r"\bpayment", "Payment"),
    (r"\bbilling", "Billing"),
    (r"\binsurance", "Insurance"),
    (r"\bcoverage", "Coverage"),
    # Healthcare metrics
    (r"\bscore\b", "Score"),
    (r"\brating", "Rating"),
    (r"\brisk\b", "Risk"),
    (r"\bseverity", "Severity"),
    (r"\bstage\b", "Stage"),
    (r"\bgrade\b", "Grade"),
    (r"\blevel\b", "Level"),
]

# Low sensitivity patterns - Non-sensitive operational data
LOW_SENSITIVITY_PATTERNS: List[Tuple[str, str]] = [
    # Identifiers (non-personal)
    (r"\brecord.*id", "Record ID"),
    (r"\btransaction.*id", "Transaction ID"),
    (r"\bcase.*id", "Case ID"),
    (r"\bevent.*id", "Event ID"),
    (r"\bsession.*id", "Session ID"),
    (r"\b(row|seq|sequence).*id", "Sequence ID"),
    (r"\bid\b$", "ID field"),
    (r"^id\b", "ID field"),
    (r"_id$", "ID suffix"),
    # Codes and types
    (r"\bcode\b", "Code"),
    (r"\btype\b", "Type"),
    (r"\bcategory", "Category"),
    (r"\bclass\b", "Class"),
    (r"\bstatus", "Status"),
    (r"\bflag\b", "Flag"),
    (r"\bindicator", "Indicator"),
    # Counts and aggregates
    (r"\bcount\b", "Count"),
    (r"\btotal\b", "Total"),
    (r"\bsum\b", "Sum"),
    (r"\bavg\b", "Average"),
    (r"\baverage", "Average"),
    (r"\bnum\b", "Number"),
    (r"\bnumber\b", "Number"),
    (r"\bquantity", "Quantity"),
    (r"\bfrequency", "Frequency"),
    # Descriptions
    (r"\bdescription", "Description"),
    (r"\bnotes?\b", "Notes"),
    (r"\bcomment", "Comment"),
    (r"\bremark", "Remark"),
    # System fields
    (r"\bversion", "Version"),
    (r"\bsource\b", "Source"),
    (r"\borigin", "Origin"),
    (r"\bformat\b", "Format"),
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ClassificationResult:
    """Result of classifying a single column.

    Attributes:
        column_name: Original column name.
        sensitivity: Classified sensitivity level.
        confidence: Confidence score (0.0 to 1.0).
        matched_pattern: Pattern that matched (if any).
        pattern_description: Human-readable description of the match.
        is_override: Whether this was manually overridden.
        override_reason: Reason for override (if applicable).
    """

    column_name: str
    sensitivity: SensitivityLevel
    confidence: float
    matched_pattern: Optional[str] = None
    pattern_description: Optional[str] = None
    is_override: bool = False
    override_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "column_name": self.column_name,
            "sensitivity": self.sensitivity.value,
            "confidence": self.confidence,
            "matched_pattern": self.matched_pattern,
            "pattern_description": self.pattern_description,
            "is_override": self.is_override,
            "override_reason": self.override_reason,
        }


@dataclass
class DatasetClassification:
    """Classification results for an entire dataset.

    Attributes:
        column_results: Dictionary mapping column names to classification results.
        high_sensitivity_columns: List of high sensitivity column names.
        medium_sensitivity_columns: List of medium sensitivity column names.
        low_sensitivity_columns: List of low sensitivity column names.
        unknown_columns: List of columns that couldn't be classified.
    """

    column_results: Dict[str, ClassificationResult] = field(default_factory=dict)

    @property
    def high_sensitivity_columns(self) -> List[str]:
        """Get list of high sensitivity columns."""
        return [
            name
            for name, result in self.column_results.items()
            if result.sensitivity == SensitivityLevel.HIGH
        ]

    @property
    def medium_sensitivity_columns(self) -> List[str]:
        """Get list of medium sensitivity columns."""
        return [
            name
            for name, result in self.column_results.items()
            if result.sensitivity == SensitivityLevel.MEDIUM
        ]

    @property
    def low_sensitivity_columns(self) -> List[str]:
        """Get list of low sensitivity columns."""
        return [
            name
            for name, result in self.column_results.items()
            if result.sensitivity == SensitivityLevel.LOW
        ]

    @property
    def unknown_columns(self) -> List[str]:
        """Get list of unclassified columns."""
        return [
            name
            for name, result in self.column_results.items()
            if result.sensitivity == SensitivityLevel.UNKNOWN
        ]

    def get_columns_by_sensitivity(
        self, sensitivity: SensitivityLevel
    ) -> List[str]:
        """Get columns with a specific sensitivity level."""
        return [
            name
            for name, result in self.column_results.items()
            if result.sensitivity == sensitivity
        ]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "columns": {
                name: result.to_dict()
                for name, result in self.column_results.items()
            },
            "summary": {
                "high_sensitivity": self.high_sensitivity_columns,
                "medium_sensitivity": self.medium_sensitivity_columns,
                "low_sensitivity": self.low_sensitivity_columns,
                "unknown": self.unknown_columns,
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for display."""
        rows = []
        for name, result in self.column_results.items():
            rows.append({
                "Column": name,
                "Sensitivity": result.sensitivity.value.upper(),
                "Confidence": f"{result.confidence:.0%}",
                "Pattern": result.pattern_description or "-",
                "Override": "Yes" if result.is_override else "No",
            })
        return pd.DataFrame(rows)


# =============================================================================
# Column Classifier
# =============================================================================


class ColumnClassifier:
    """Classifies column sensitivity based on name patterns.

    This classifier uses regex patterns to match column names against
    known sensitive data types. It supports healthcare-specific patterns
    for HIPAA compliance.

    Example:
        >>> classifier = ColumnClassifier()
        >>> result = classifier.classify_column("patient_ssn")
        >>> print(result.sensitivity)
        SensitivityLevel.HIGH

        >>> df = pd.DataFrame({"ssn": [], "age": [], "record_id": []})
        >>> classification = classifier.classify_dataset(df)
        >>> print(classification.high_sensitivity_columns)
        ['ssn']
    """

    def __init__(
        self,
        custom_high_patterns: Optional[List[Tuple[str, str]]] = None,
        custom_medium_patterns: Optional[List[Tuple[str, str]]] = None,
        custom_low_patterns: Optional[List[Tuple[str, str]]] = None,
        default_sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
    ):
        """Initialize the classifier.

        Args:
            custom_high_patterns: Additional high sensitivity patterns.
            custom_medium_patterns: Additional medium sensitivity patterns.
            custom_low_patterns: Additional low sensitivity patterns.
            default_sensitivity: Default level for unmatched columns.
        """
        self._high_patterns = list(HIGH_SENSITIVITY_PATTERNS)
        self._medium_patterns = list(MEDIUM_SENSITIVITY_PATTERNS)
        self._low_patterns = list(LOW_SENSITIVITY_PATTERNS)

        if custom_high_patterns:
            self._high_patterns.extend(custom_high_patterns)
        if custom_medium_patterns:
            self._medium_patterns.extend(custom_medium_patterns)
        if custom_low_patterns:
            self._low_patterns.extend(custom_low_patterns)

        # Compile patterns for efficiency
        self._compiled_high = [
            (re.compile(pattern, re.IGNORECASE), desc)
            for pattern, desc in self._high_patterns
        ]
        self._compiled_medium = [
            (re.compile(pattern, re.IGNORECASE), desc)
            for pattern, desc in self._medium_patterns
        ]
        self._compiled_low = [
            (re.compile(pattern, re.IGNORECASE), desc)
            for pattern, desc in self._low_patterns
        ]

        self._default_sensitivity = default_sensitivity
        self._overrides: Dict[str, Tuple[SensitivityLevel, str]] = {}

    def add_override(
        self,
        column_name: str,
        sensitivity: SensitivityLevel,
        reason: Optional[str] = None,
    ) -> None:
        """Add a manual override for a specific column.

        Args:
            column_name: Name of the column to override.
            sensitivity: Override sensitivity level.
            reason: Optional reason for the override.
        """
        self._overrides[column_name.lower()] = (
            sensitivity,
            reason or "Manual override",
        )

    def remove_override(self, column_name: str) -> bool:
        """Remove an override for a column.

        Args:
            column_name: Name of the column.

        Returns:
            True if override was removed, False if not found.
        """
        key = column_name.lower()
        if key in self._overrides:
            del self._overrides[key]
            return True
        return False

    def clear_overrides(self) -> None:
        """Clear all manual overrides."""
        self._overrides.clear()

    def get_overrides(self) -> Dict[str, Tuple[SensitivityLevel, str]]:
        """Get all current overrides."""
        return dict(self._overrides)

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching.

        Converts to lowercase, replaces underscores/hyphens with spaces.
        """
        normalized = name.lower()
        normalized = normalized.replace("_", " ")
        normalized = normalized.replace("-", " ")
        return normalized

    def _match_patterns(
        self,
        name: str,
        patterns: List[Tuple[re.Pattern, str]],
    ) -> Optional[Tuple[str, str, float]]:
        """Match column name against a list of patterns.

        Returns:
            Tuple of (matched_pattern, description, confidence) or None.
        """
        normalized = self._normalize_column_name(name)

        for pattern, description in patterns:
            # Try matching against normalized name
            if pattern.search(normalized):
                # Calculate confidence based on match quality
                match = pattern.search(normalized)
                if match:
                    # Full match gives higher confidence
                    match_len = match.end() - match.start()
                    name_len = len(normalized)
                    confidence = min(0.9, 0.6 + (match_len / name_len) * 0.3)
                    return (pattern.pattern, description, confidence)

            # Also try original name
            if pattern.search(name):
                return (pattern.pattern, description, 0.7)

        return None

    def classify_column(self, column_name: str) -> ClassificationResult:
        """Classify a single column by name.

        Args:
            column_name: Name of the column to classify.

        Returns:
            ClassificationResult with sensitivity level and metadata.
        """
        # Check for manual override first
        key = column_name.lower()
        if key in self._overrides:
            sensitivity, reason = self._overrides[key]
            return ClassificationResult(
                column_name=column_name,
                sensitivity=sensitivity,
                confidence=1.0,
                is_override=True,
                override_reason=reason,
            )

        # Try high sensitivity patterns first (most restrictive)
        match = self._match_patterns(column_name, self._compiled_high)
        if match:
            pattern, description, confidence = match
            return ClassificationResult(
                column_name=column_name,
                sensitivity=SensitivityLevel.HIGH,
                confidence=confidence,
                matched_pattern=pattern,
                pattern_description=description,
            )

        # Try medium sensitivity patterns
        match = self._match_patterns(column_name, self._compiled_medium)
        if match:
            pattern, description, confidence = match
            return ClassificationResult(
                column_name=column_name,
                sensitivity=SensitivityLevel.MEDIUM,
                confidence=confidence,
                matched_pattern=pattern,
                pattern_description=description,
            )

        # Try low sensitivity patterns
        match = self._match_patterns(column_name, self._compiled_low)
        if match:
            pattern, description, confidence = match
            return ClassificationResult(
                column_name=column_name,
                sensitivity=SensitivityLevel.LOW,
                confidence=confidence,
                matched_pattern=pattern,
                pattern_description=description,
            )

        # No match - use default
        return ClassificationResult(
            column_name=column_name,
            sensitivity=self._default_sensitivity,
            confidence=0.5,
            pattern_description="No pattern match - using default",
        )

    def classify_columns(
        self,
        column_names: List[str],
    ) -> Dict[str, ClassificationResult]:
        """Classify multiple columns.

        Args:
            column_names: List of column names to classify.

        Returns:
            Dictionary mapping column names to classification results.
        """
        return {name: self.classify_column(name) for name in column_names}

    def classify_dataset(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> DatasetClassification:
        """Classify all columns in a DataFrame.

        Args:
            df: DataFrame to classify.
            columns: Optional list of columns to classify (default: all).

        Returns:
            DatasetClassification with results for all columns.
        """
        cols = columns if columns is not None else list(df.columns)
        results = self.classify_columns(cols)

        return DatasetClassification(column_results=results)

    def get_pattern_count(self) -> Dict[str, int]:
        """Get the count of patterns by sensitivity level."""
        return {
            "high": len(self._high_patterns),
            "medium": len(self._medium_patterns),
            "low": len(self._low_patterns),
        }


# =============================================================================
# Content-Based Classifier
# =============================================================================


class ContentAnalyzer:
    """Analyzes column content to refine sensitivity classification.

    This analyzer examines actual data values to detect sensitive
    content that might not be apparent from the column name alone.
    """

    # Patterns for content analysis
    SSN_PATTERN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    PHONE_PATTERN = re.compile(r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
    EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")
    CREDIT_CARD_PATTERN = re.compile(r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$")
    DATE_PATTERN = re.compile(
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$|^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
    )

    def __init__(self, sample_size: int = 100):
        """Initialize the content analyzer.

        Args:
            sample_size: Number of rows to sample for analysis.
        """
        self.sample_size = sample_size

    def analyze_column(
        self,
        series: pd.Series,
        current_result: ClassificationResult,
    ) -> ClassificationResult:
        """Analyze column content to refine classification.

        Args:
            series: Column data to analyze.
            current_result: Current classification result from name-based analysis.

        Returns:
            Refined ClassificationResult.
        """
        # Skip if already high sensitivity or overridden
        if (
            current_result.sensitivity == SensitivityLevel.HIGH
            or current_result.is_override
        ):
            return current_result

        # Skip non-string columns
        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(
            series
        ):
            return current_result

        # Sample data for analysis
        sample = series.dropna().head(self.sample_size)
        if len(sample) == 0:
            return current_result

        # Convert to strings
        sample = sample.astype(str)

        # Check for sensitive patterns in content
        sensitive_type = self._detect_sensitive_content(sample)
        if sensitive_type:
            return ClassificationResult(
                column_name=current_result.column_name,
                sensitivity=SensitivityLevel.HIGH,
                confidence=0.85,
                matched_pattern=f"content:{sensitive_type}",
                pattern_description=f"Content analysis detected: {sensitive_type}",
            )

        return current_result

    def _detect_sensitive_content(
        self,
        sample: pd.Series,
    ) -> Optional[str]:
        """Detect sensitive content patterns in a sample.

        Returns:
            Type of sensitive content detected, or None.
        """
        # Count matches for each pattern
        matches = {
            "SSN": 0,
            "Phone": 0,
            "Email": 0,
            "ZIP Code": 0,
            "Credit Card": 0,
        }

        for value in sample:
            if self.SSN_PATTERN.match(value):
                matches["SSN"] += 1
            elif self.PHONE_PATTERN.match(value):
                matches["Phone"] += 1
            elif self.EMAIL_PATTERN.match(value):
                matches["Email"] += 1
            elif self.ZIP_PATTERN.match(value):
                matches["ZIP Code"] += 1
            elif self.CREDIT_CARD_PATTERN.match(value):
                matches["Credit Card"] += 1

        # Require at least 50% match rate for detection
        threshold = len(sample) * 0.5
        for content_type, count in matches.items():
            if count >= threshold:
                return content_type

        return None

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        classification: DatasetClassification,
    ) -> DatasetClassification:
        """Analyze all columns in a dataset.

        Args:
            df: DataFrame to analyze.
            classification: Current classification results.

        Returns:
            Refined DatasetClassification.
        """
        refined_results: Dict[str, ClassificationResult] = {}

        for column_name, current_result in classification.column_results.items():
            if column_name in df.columns:
                refined_results[column_name] = self.analyze_column(
                    df[column_name],
                    current_result,
                )
            else:
                refined_results[column_name] = current_result

        return DatasetClassification(column_results=refined_results)


# =============================================================================
# Convenience Functions
# =============================================================================


def classify_column(
    column_name: str,
    custom_patterns: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> ClassificationResult:
    """Classify a single column by name.

    Args:
        column_name: Name of the column.
        custom_patterns: Optional custom patterns by level (high, medium, low).

    Returns:
        ClassificationResult with sensitivity level.

    Example:
        >>> result = classify_column("patient_ssn")
        >>> print(result.sensitivity)
        SensitivityLevel.HIGH
    """
    custom_high: Optional[List[Tuple[str, str]]] = None
    custom_medium: Optional[List[Tuple[str, str]]] = None
    custom_low: Optional[List[Tuple[str, str]]] = None

    if custom_patterns:
        custom_high = custom_patterns.get("high")
        custom_medium = custom_patterns.get("medium")
        custom_low = custom_patterns.get("low")

    classifier = ColumnClassifier(
        custom_high_patterns=custom_high,
        custom_medium_patterns=custom_medium,
        custom_low_patterns=custom_low,
    )
    return classifier.classify_column(column_name)


def classify_columns(
    column_names: List[str],
    overrides: Optional[Dict[str, SensitivityLevel]] = None,
) -> Dict[str, ClassificationResult]:
    """Classify multiple columns.

    Args:
        column_names: List of column names.
        overrides: Optional dictionary of manual overrides.

    Returns:
        Dictionary mapping column names to classification results.

    Example:
        >>> results = classify_columns(["ssn", "age", "record_id"])
        >>> print(results["ssn"].sensitivity)
        SensitivityLevel.HIGH
    """
    classifier = ColumnClassifier()

    if overrides:
        for name, level in overrides.items():
            classifier.add_override(name, level)

    return classifier.classify_columns(column_names)


def classify_dataset(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, SensitivityLevel]] = None,
    analyze_content: bool = False,
) -> DatasetClassification:
    """Classify all columns in a DataFrame.

    Args:
        df: DataFrame to classify.
        overrides: Optional dictionary of manual overrides.
        analyze_content: Whether to analyze column content for sensitive patterns.

    Returns:
        DatasetClassification with results for all columns.

    Example:
        >>> df = pd.DataFrame({"ssn": ["123-45-6789"], "age": [25]})
        >>> classification = classify_dataset(df)
        >>> print(classification.high_sensitivity_columns)
        ['ssn']
    """
    classifier = ColumnClassifier()

    if overrides:
        for name, level in overrides.items():
            classifier.add_override(name, level)

    classification = classifier.classify_dataset(df)

    if analyze_content:
        analyzer = ContentAnalyzer()
        classification = analyzer.analyze_dataset(df, classification)

    return classification


def get_sensitivity_for_column(column_name: str) -> SensitivityLevel:
    """Quick utility to get sensitivity level for a column name.

    Args:
        column_name: Name of the column.

    Returns:
        SensitivityLevel enum value.

    Example:
        >>> level = get_sensitivity_for_column("patient_ssn")
        >>> print(level)
        SensitivityLevel.HIGH
    """
    result = classify_column(column_name)
    return result.sensitivity


def get_sensitive_columns(
    df: pd.DataFrame,
    min_sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM,
) -> List[str]:
    """Get list of columns at or above a sensitivity threshold.

    Args:
        df: DataFrame to analyze.
        min_sensitivity: Minimum sensitivity level to include.

    Returns:
        List of column names at or above the threshold.

    Example:
        >>> df = pd.DataFrame({"ssn": [], "age": [], "record_id": []})
        >>> sensitive = get_sensitive_columns(df, SensitivityLevel.HIGH)
        >>> print(sensitive)
        ['ssn']
    """
    classification = classify_dataset(df)

    sensitivity_order = {
        SensitivityLevel.HIGH: 3,
        SensitivityLevel.MEDIUM: 2,
        SensitivityLevel.LOW: 1,
        SensitivityLevel.UNKNOWN: 0,
    }

    min_level = sensitivity_order[min_sensitivity]

    return [
        name
        for name, result in classification.column_results.items()
        if sensitivity_order[result.sensitivity] >= min_level
    ]
