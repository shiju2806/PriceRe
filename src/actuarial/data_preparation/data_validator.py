"""
Actuarial Data Validator
Comprehensive validation for life and retirement reinsurance data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date

class DataQualityLevel(Enum):
    """Data quality classification levels"""
    PRISTINE = "pristine"  # 100% complete, no issues
    GOOD = "good"  # >95% complete, minor issues
    ACCEPTABLE = "acceptable"  # >90% complete, some issues  
    POOR = "poor"  # <90% complete, major issues
    REJECTED = "rejected"  # Unusable data

class FieldType(Enum):
    """Actuarial field types"""
    POLICY_ID = "policy_id"
    ISSUE_DATE = "issue_date"
    MATURITY_DATE = "maturity_date"
    ISSUE_AGE = "issue_age"
    CURRENT_AGE = "current_age"
    GENDER = "gender"
    SMOKER_STATUS = "smoker_status"
    FACE_AMOUNT = "face_amount"
    PREMIUM = "premium"
    PRODUCT_TYPE = "product_type"
    MORTALITY_CLASS = "mortality_class"
    RESERVE_BASIS = "reserve_basis"
    POLICY_STATUS = "policy_status"
    DEATH_BENEFIT = "death_benefit"
    CASH_VALUE = "cash_value"
    LAPSE_INDICATOR = "lapse_indicator"
    CLAIM_AMOUNT = "claim_amount"
    ECONOMIC_INDICATOR = "economic_indicator"

@dataclass
class ValidationRule:
    """Single validation rule"""
    field: str
    rule_type: str
    condition: Any
    severity: str  # "error", "warning", "info"
    message: str

@dataclass
class ValidationResult:
    """Results from data validation"""
    quality_level: DataQualityLevel
    total_records: int
    valid_records: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    field_stats: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    is_usable: bool

class ActuarialDataValidator:
    """
    Comprehensive data validator for actuarial pricing
    Ensures data meets SOA standards and regulatory requirements
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.required_fields = self._define_required_fields()
        self.soa_standards = self._load_soa_standards()
        
    def _initialize_validation_rules(self) -> Dict[FieldType, List[ValidationRule]]:
        """Initialize comprehensive validation rules"""
        
        rules = {
            FieldType.ISSUE_AGE: [
                ValidationRule(
                    field="issue_age",
                    rule_type="range",
                    condition=(0, 100),
                    severity="error",
                    message="Issue age must be between 0 and 100"
                ),
                ValidationRule(
                    field="issue_age",
                    rule_type="typical_range",
                    condition=(18, 75),
                    severity="warning",
                    message="Issue age outside typical range (18-75)"
                )
            ],
            FieldType.FACE_AMOUNT: [
                ValidationRule(
                    field="face_amount",
                    rule_type="positive",
                    condition=lambda x: x > 0,
                    severity="error",
                    message="Face amount must be positive"
                ),
                ValidationRule(
                    field="face_amount",
                    rule_type="reasonable_range",
                    condition=(10000, 100000000),
                    severity="warning",
                    message="Face amount outside reasonable range ($10K - $100M)"
                )
            ],
            FieldType.PREMIUM: [
                ValidationRule(
                    field="premium",
                    rule_type="positive",
                    condition=lambda x: x >= 0,
                    severity="error",
                    message="Premium cannot be negative"
                ),
                ValidationRule(
                    field="premium",
                    rule_type="ratio_to_face",
                    condition=lambda p, f: 0.0001 < p/f < 0.2 if f > 0 else False,
                    severity="warning",
                    message="Premium to face amount ratio unusual"
                )
            ],
            FieldType.GENDER: [
                ValidationRule(
                    field="gender",
                    rule_type="valid_values",
                    condition=["M", "F", "U"],
                    severity="error",
                    message="Gender must be M, F, or U (unspecified)"
                )
            ],
            FieldType.SMOKER_STATUS: [
                ValidationRule(
                    field="smoker_status",
                    rule_type="valid_values",
                    condition=["NS", "S", "U"],  # Non-smoker, Smoker, Unknown
                    severity="error",
                    message="Smoker status must be NS, S, or U"
                )
            ],
            FieldType.PRODUCT_TYPE: [
                ValidationRule(
                    field="product_type",
                    rule_type="valid_values",
                    condition=["TERM", "WHOLE_LIFE", "UNIVERSAL_LIFE", "VARIABLE_LIFE", 
                              "ANNUITY", "PENSION", "401K", "IRA"],
                    severity="error",
                    message="Invalid product type"
                )
            ],
            FieldType.POLICY_STATUS: [
                ValidationRule(
                    field="policy_status",
                    rule_type="valid_values",
                    condition=["ACTIVE", "LAPSED", "SURRENDERED", "MATURED", "CLAIM", "PENDING"],
                    severity="error",
                    message="Invalid policy status"
                )
            ]
        }
        
        return rules
    
    def _define_required_fields(self) -> Dict[str, List[str]]:
        """Define required fields by product type"""
        
        return {
            "LIFE": [
                "policy_id", "issue_date", "issue_age", "gender", 
                "face_amount", "premium", "product_type", "policy_status"
            ],
            "ANNUITY": [
                "policy_id", "issue_date", "issue_age", "gender",
                "premium", "product_type", "policy_status", "cash_value"
            ],
            "RETIREMENT": [
                "policy_id", "issue_date", "issue_age", 
                "contribution_amount", "account_balance", "product_type"
            ]
        }
    
    def _load_soa_standards(self) -> Dict[str, Any]:
        """Load SOA (Society of Actuaries) data standards"""
        
        return {
            "mortality_tables": ["2017CSO", "2001CSO", "2015VBT", "2008VBT"],
            "reserve_methods": ["CRVM", "PBR", "GAAP_LDTI", "STATUTORY"],
            "interest_rates": {
                "valuation_rate_min": 0.005,
                "valuation_rate_max": 0.06,
                "discount_rate_range": (0.02, 0.05)
            },
            "underwriting_classes": [
                "PREFERRED_PLUS", "PREFERRED", "STANDARD_PLUS", 
                "STANDARD", "SUBSTANDARD", "DECLINED"
            ]
        }
    
    def validate_dataset(self, df: pd.DataFrame, 
                        product_category: str = "LIFE") -> ValidationResult:
        """
        Perform comprehensive validation on dataset
        
        Args:
            df: Input dataframe
            product_category: Type of insurance product
            
        Returns:
            ValidationResult with detailed analysis
        """
        
        errors = []
        warnings = []
        field_stats = {}
        
        # Check required fields
        required = self.required_fields.get(product_category, self.required_fields["LIFE"])
        missing_fields = [f for f in required if f not in df.columns]
        
        if missing_fields:
            errors.append({
                "type": "missing_fields",
                "fields": missing_fields,
                "message": f"Required fields missing: {', '.join(missing_fields)}"
            })
        
        # Validate each field
        for col in df.columns:
            field_stats[col] = self._analyze_field(df[col])
            
            # Apply validation rules
            field_errors, field_warnings = self._validate_field(df, col)
            errors.extend(field_errors)
            warnings.extend(field_warnings)
        
        # Check data relationships
        relationship_issues = self._validate_relationships(df)
        errors.extend(relationship_issues.get("errors", []))
        warnings.extend(relationship_issues.get("warnings", []))
        
        # Determine quality level
        completeness = self._calculate_completeness(df)
        quality_level = self._determine_quality_level(completeness, len(errors), len(warnings))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, errors, warnings, field_stats)
        
        return ValidationResult(
            quality_level=quality_level,
            total_records=len(df),
            valid_records=len(df) - len(errors),
            errors=errors,
            warnings=warnings,
            field_stats=field_stats,
            recommendations=recommendations,
            is_usable=quality_level not in [DataQualityLevel.POOR, DataQualityLevel.REJECTED]
        )
    
    def _analyze_field(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze individual field statistics"""
        
        stats = {
            "count": len(series),
            "missing": series.isna().sum(),
            "missing_pct": series.isna().mean() * 100,
            "unique": series.nunique(),
            "dtype": str(series.dtype)
        }
        
        # Numeric fields
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "q25": series.quantile(0.25),
                "q50": series.quantile(0.50),
                "q75": series.quantile(0.75),
                "outliers": self._detect_outliers(series)
            })
        
        # Date fields
        elif pd.api.types.is_datetime64_any_dtype(series):
            stats.update({
                "min_date": series.min(),
                "max_date": series.max(),
                "date_range_years": (series.max() - series.min()).days / 365.25
            })
        
        # Categorical fields
        else:
            value_counts = series.value_counts()
            stats.update({
                "top_values": value_counts.head(10).to_dict(),
                "mode": value_counts.index[0] if len(value_counts) > 0 else None
            })
        
        return stats
    
    def _validate_field(self, df: pd.DataFrame, field: str) -> Tuple[List[Dict], List[Dict]]:
        """Validate individual field against rules"""
        
        errors = []
        warnings = []
        
        # Check if field maps to a known type
        field_type = self._identify_field_type(field)
        if field_type and field_type in self.validation_rules:
            rules = self.validation_rules[field_type]
            
            for rule in rules:
                violations = self._apply_rule(df[field], rule)
                
                if violations:
                    issue = {
                        "field": field,
                        "rule": rule.rule_type,
                        "message": rule.message,
                        "count": len(violations),
                        "sample_rows": violations[:5]
                    }
                    
                    if rule.severity == "error":
                        errors.append(issue)
                    else:
                        warnings.append(issue)
        
        return errors, warnings
    
    def _validate_relationships(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Validate relationships between fields"""
        
        errors = []
        warnings = []
        
        # Check issue_age vs current_age
        if "issue_age" in df.columns and "current_age" in df.columns:
            invalid = df[df["current_age"] < df["issue_age"]]
            if not invalid.empty:
                errors.append({
                    "type": "relationship",
                    "message": "Current age less than issue age",
                    "count": len(invalid)
                })
        
        # Check dates consistency
        if "issue_date" in df.columns and "maturity_date" in df.columns:
            df["issue_date"] = pd.to_datetime(df["issue_date"], errors='coerce')
            df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors='coerce')
            
            invalid = df[df["maturity_date"] < df["issue_date"]]
            if not invalid.empty:
                errors.append({
                    "type": "relationship",
                    "message": "Maturity date before issue date",
                    "count": len(invalid)
                })
        
        # Check premium reasonableness vs face amount
        if "premium" in df.columns and "face_amount" in df.columns:
            df["premium_ratio"] = df["premium"] / df["face_amount"]
            unusual = df[(df["premium_ratio"] < 0.0001) | (df["premium_ratio"] > 0.2)]
            if not unusual.empty:
                warnings.append({
                    "type": "relationship",
                    "message": "Unusual premium to face amount ratio",
                    "count": len(unusual),
                    "details": f"Ratios range from {unusual['premium_ratio'].min():.4f} to {unusual['premium_ratio'].max():.4f}"
                })
        
        return {"errors": errors, "warnings": warnings}
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(series) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate overall data completeness"""
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        return (total_cells - missing_cells) / total_cells * 100
    
    def _determine_quality_level(self, completeness: float, 
                                 error_count: int, warning_count: int) -> DataQualityLevel:
        """Determine overall data quality level"""
        
        if error_count > 10 or completeness < 70:
            return DataQualityLevel.REJECTED
        elif error_count > 5 or completeness < 90:
            return DataQualityLevel.POOR
        elif error_count > 0 or completeness < 95:
            return DataQualityLevel.ACCEPTABLE
        elif warning_count > 5 or completeness < 98:
            return DataQualityLevel.GOOD
        else:
            return DataQualityLevel.PRISTINE
    
    def _identify_field_type(self, field_name: str) -> Optional[FieldType]:
        """Map field name to FieldType enum"""
        
        field_lower = field_name.lower()
        
        mappings = {
            "issue_age": FieldType.ISSUE_AGE,
            "age_at_issue": FieldType.ISSUE_AGE,
            "current_age": FieldType.CURRENT_AGE,
            "attained_age": FieldType.CURRENT_AGE,
            "gender": FieldType.GENDER,
            "sex": FieldType.GENDER,
            "smoker": FieldType.SMOKER_STATUS,
            "smoker_status": FieldType.SMOKER_STATUS,
            "face_amount": FieldType.FACE_AMOUNT,
            "death_benefit": FieldType.DEATH_BENEFIT,
            "premium": FieldType.PREMIUM,
            "annual_premium": FieldType.PREMIUM,
            "product_type": FieldType.PRODUCT_TYPE,
            "product": FieldType.PRODUCT_TYPE,
            "policy_status": FieldType.POLICY_STATUS,
            "status": FieldType.POLICY_STATUS
        }
        
        for key, field_type in mappings.items():
            if key in field_lower:
                return field_type
        
        return None
    
    def _apply_rule(self, series: pd.Series, rule: ValidationRule) -> List[int]:
        """Apply validation rule and return violating indices"""
        
        violations = []
        
        if rule.rule_type == "range":
            min_val, max_val = rule.condition
            mask = (series < min_val) | (series > max_val)
            violations = series[mask].index.tolist()
            
        elif rule.rule_type == "positive":
            mask = series <= 0
            violations = series[mask].index.tolist()
            
        elif rule.rule_type == "valid_values":
            mask = ~series.isin(rule.condition)
            violations = series[mask].index.tolist()
        
        return violations
    
    def _generate_recommendations(self, df: pd.DataFrame, errors: List[Dict], 
                                 warnings: List[Dict], field_stats: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # High missing data
        for field, stats in field_stats.items():
            if stats["missing_pct"] > 10:
                recommendations.append(
                    f"Field '{field}' has {stats['missing_pct']:.1f}% missing values - consider imputation or exclusion"
                )
        
        # Outliers
        for field, stats in field_stats.items():
            if "outliers" in stats and stats["outliers"]["percentage"] > 5:
                recommendations.append(
                    f"Field '{field}' has {stats['outliers']['percentage']:.1f}% outliers - review for data quality issues"
                )
        
        # Critical errors
        if any(e["type"] == "missing_fields" for e in errors):
            recommendations.append(
                "Critical: Required fields are missing - ensure proper field mapping before proceeding"
            )
        
        # Date range issues
        date_fields = [f for f in field_stats if "date_range_years" in field_stats[f]]
        for field in date_fields:
            if field_stats[field]["date_range_years"] > 50:
                recommendations.append(
                    f"Field '{field}' spans {field_stats[field]['date_range_years']:.1f} years - verify date accuracy"
                )
        
        # Product mix
        if "product_type" in df.columns:
            product_dist = df["product_type"].value_counts(normalize=True)
            if any(product_dist > 0.8):
                recommendations.append(
                    "Dataset is heavily concentrated in one product type - ensure adequate representation for modeling"
                )
        
        return recommendations

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("ACTUARIAL DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nData Quality Level: {result.quality_level.value.upper()}")
        report.append(f"Total Records: {result.total_records:,}")
        report.append(f"Valid Records: {result.valid_records:,}")
        report.append(f"Validation Status: {'✅ PASSED' if result.is_usable else '❌ FAILED'}")
        
        if result.errors:
            report.append(f"\n⚠️  ERRORS ({len(result.errors)})")
            report.append("-" * 40)
            for error in result.errors[:5]:
                report.append(f"• {error['message']}")
                if "count" in error:
                    report.append(f"  Affected records: {error['count']}")
        
        if result.warnings:
            report.append(f"\n⚠️  WARNINGS ({len(result.warnings)})")
            report.append("-" * 40)
            for warning in result.warnings[:5]:
                report.append(f"• {warning['message']}")
                if "count" in warning:
                    report.append(f"  Affected records: {warning['count']}")
        
        if result.recommendations:
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-" * 40)
            for rec in result.recommendations:
                report.append(f"• {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)