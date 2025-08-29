"""
Data Upload and Validation System

Handles real data uploads for reinsurance pricing with:
- Multi-format file support (CSV, Excel, Parquet)
- Data quality validation
- Schema validation and mapping
- Data profiling and summary statistics
- Missing data imputation recommendations
- Outlier detection and flagging
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class DataType(Enum):
    TREATY = "treaty"
    CLAIMS = "claims"
    PORTFOLIO = "portfolio"
    CATASTROPHE = "catastrophe"
    DEVELOPMENT = "development"


class ValidationLevel(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Single validation result"""
    level: ValidationLevel
    message: str
    column: Optional[str] = None
    affected_rows: int = 0
    suggested_action: Optional[str] = None


@dataclass
class DataValidationReport:
    """Complete data validation report"""
    file_name: str
    data_type: DataType
    total_rows: int
    total_columns: int
    validation_results: List[ValidationResult]
    data_profile: Dict[str, Any]
    schema_mapping: Dict[str, str]
    recommended_actions: List[str]
    is_valid: bool
    confidence_score: float


class DataValidator:
    """Validates and profiles uploaded reinsurance data"""
    
    def __init__(self):
        """Initialize validator with schemas and rules"""
        self.schema_definitions = self._load_schema_definitions()
        self.validation_rules = self._load_validation_rules()
        
    def validate_upload(
        self,
        file_path: Union[str, Path],
        data_type: DataType,
        user_schema_mapping: Optional[Dict[str, str]] = None
    ) -> DataValidationReport:
        """Main validation entry point"""
        
        file_path = Path(file_path)
        
        try:
            # Load data
            df = self._load_file(file_path)
            
            # Basic validation
            if df.is_empty():
                return self._create_error_report(
                    file_path.name, data_type, "File is empty"
                )
            
            # Schema validation and mapping
            schema_mapping = self._validate_and_map_schema(
                df, data_type, user_schema_mapping
            )
            
            # Apply schema mapping
            mapped_df = self._apply_schema_mapping(df, schema_mapping)
            
            # Data quality validation
            validation_results = self._run_data_quality_checks(mapped_df, data_type)
            
            # Data profiling
            data_profile = self._profile_data(mapped_df, data_type)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, data_profile)
            
            # Calculate scores
            is_valid = self._calculate_validity(validation_results)
            confidence_score = self._calculate_confidence(validation_results, data_profile)
            
            return DataValidationReport(
                file_name=file_path.name,
                data_type=data_type,
                total_rows=len(mapped_df),
                total_columns=len(mapped_df.columns),
                validation_results=validation_results,
                data_profile=data_profile,
                schema_mapping=schema_mapping,
                recommended_actions=recommendations,
                is_valid=is_valid,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            return self._create_error_report(
                file_path.name, data_type, f"Error processing file: {str(e)}"
            )
    
    def _load_file(self, file_path: Path) -> pl.DataFrame:
        """Load file based on extension"""
        
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return pl.read_csv(file_path, ignore_errors=True)
        elif extension in ['.xlsx', '.xls']:
            # Use pandas for Excel then convert to polars
            pandas_df = pd.read_excel(file_path)
            return pl.from_pandas(pandas_df)
        elif extension == '.parquet':
            return pl.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _validate_and_map_schema(
        self,
        df: pl.DataFrame,
        data_type: DataType,
        user_mapping: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """Validate schema and create column mapping"""
        
        expected_schema = self.schema_definitions[data_type.value]
        actual_columns = df.columns
        
        # Start with user mapping if provided
        mapping = user_mapping.copy() if user_mapping else {}
        
        # Auto-detect unmapped columns
        for expected_col in expected_schema["required"] + expected_schema["optional"]:
            if expected_col not in mapping.values():
                # Try to find matching column
                best_match = self._find_best_column_match(expected_col, actual_columns)
                if best_match:
                    # Find the key that maps to this best_match
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    if best_match not in reverse_mapping:
                        mapping[expected_col] = best_match
        
        return mapping
    
    def _find_best_column_match(self, target: str, candidates: List[str]) -> Optional[str]:
        """Find best matching column name using fuzzy matching"""
        
        target_lower = target.lower().replace("_", "").replace(" ", "")
        best_score = 0
        best_match = None
        
        for candidate in candidates:
            candidate_lower = candidate.lower().replace("_", "").replace(" ", "")
            
            # Exact match
            if target_lower == candidate_lower:
                return candidate
            
            # Contains match
            if target_lower in candidate_lower or candidate_lower in target_lower:
                score = min(len(target_lower), len(candidate_lower)) / max(len(target_lower), len(candidate_lower))
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = candidate
        
        return best_match
    
    def _apply_schema_mapping(self, df: pl.DataFrame, mapping: Dict[str, str]) -> pl.DataFrame:
        """Apply column mapping and rename columns"""
        
        # Create reverse mapping (current_name -> standard_name)
        rename_dict = {v: k for k, v in mapping.items() if v in df.columns}
        
        if rename_dict:
            df = df.rename(rename_dict)
        
        return df
    
    def _run_data_quality_checks(
        self,
        df: pl.DataFrame,
        data_type: DataType
    ) -> List[ValidationResult]:
        """Run comprehensive data quality checks"""
        
        results = []
        rules = self.validation_rules[data_type.value]
        
        # Required column checks
        for col in rules["required_columns"]:
            if col not in df.columns:
                results.append(ValidationResult(
                    level=ValidationLevel.CRITICAL,
                    message=f"Required column '{col}' is missing",
                    column=col,
                    suggested_action=f"Add column '{col}' or update column mapping"
                ))
        
        # Data type validation (more flexible for reinsurance)
        for col, expected_type in rules["column_types"].items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    # For reinsurance data, be more lenient with numeric types
                    if expected_type == "Float" and "Int" in actual_type:
                        results.append(ValidationResult(
                            level=ValidationLevel.INFO,
                            message=f"Column '{col}' has type {actual_type}, expected {expected_type} (acceptable for reinsurance data)",
                            column=col,
                            suggested_action=f"Column '{col}' can be used as-is, or convert to {expected_type} if needed"
                        ))
                    else:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Column '{col}' has type {actual_type}, expected {expected_type}",
                            column=col,
                            suggested_action=f"Convert column '{col}' to {expected_type}"
                        ))
        
        # Null value checks (with reinsurance-specific logic)
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = null_count / len(df) * 100
            
            # Skip validation for columns that are legitimately null for certain treaty types
            if data_type == DataType.TREATY and col in ['limit', 'cession_rate', 'minimum_premium', 'maximum_premium', 'profit_commission', 'aggregate_limit']:
                # These fields are legitimately null for certain treaty types
                if null_pct > 80:  # Only flag if almost all are missing
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Column '{col}' has {null_pct:.1f}% missing values (may be expected for certain treaty types)",
                        column=col,
                        affected_rows=null_count,
                        suggested_action=f"Review if '{col}' should be populated for your treaty types"
                    ))
            else:
                # Standard null value validation for other columns/data types
                if null_pct > 50:
                    results.append(ValidationResult(
                        level=ValidationLevel.CRITICAL,
                        message=f"Column '{col}' has {null_pct:.1f}% missing values",
                        column=col,
                        affected_rows=null_count,
                        suggested_action="Consider removing column or imputing missing values"
                    ))
                elif null_pct > 20:  # Relaxed threshold for reinsurance data
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Column '{col}' has {null_pct:.1f}% missing values",
                        column=col,
                        affected_rows=null_count,
                        suggested_action="Consider imputing missing values"
                    ))
        
        # Range validation
        results.extend(self._validate_ranges(df, rules.get("ranges", {})))
        
        # Business logic validation
        results.extend(self._validate_business_logic(df, data_type))
        
        # Outlier detection
        results.extend(self._detect_outliers(df, data_type))
        
        return results
    
    def _validate_ranges(self, df: pl.DataFrame, range_rules: Dict) -> List[ValidationResult]:
        """Validate numeric ranges"""
        
        results = []
        
        for col, range_def in range_rules.items():
            if col not in df.columns:
                continue
                
            try:
                col_data = df[col].drop_nulls()
                if len(col_data) == 0:
                    continue
                
                min_val = col_data.min()
                max_val = col_data.max()
                
                expected_min = range_def.get("min")
                expected_max = range_def.get("max")
                
                if expected_min is not None and min_val < expected_min:
                    invalid_count = (df[col] < expected_min).sum()
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Column '{col}' has values below minimum {expected_min}",
                        column=col,
                        affected_rows=invalid_count,
                        suggested_action=f"Review values below {expected_min} in column '{col}'"
                    ))
                
                if expected_max is not None and max_val > expected_max:
                    invalid_count = (df[col] > expected_max).sum()
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Column '{col}' has values above maximum {expected_max}",
                        column=col,
                        affected_rows=invalid_count,
                        suggested_action=f"Review values above {expected_max} in column '{col}'"
                    ))
                    
            except Exception:
                # Skip if column is not numeric
                pass
        
        return results
    
    def _validate_business_logic(self, df: pl.DataFrame, data_type: DataType) -> List[ValidationResult]:
        """Validate business-specific logic"""
        
        results = []
        
        if data_type == DataType.TREATY:
            # Treaty-specific validations
            if "loss_ratio" in df.columns:
                high_lr_count = (df["loss_ratio"] > 2.0).sum()
                if high_lr_count > 0:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"{high_lr_count} treaties have loss ratios > 200%",
                        column="loss_ratio",
                        affected_rows=high_lr_count,
                        suggested_action="Review treaties with extremely high loss ratios"
                    ))
            
            if "combined_ratio" in df.columns:
                negative_cr_count = (df["combined_ratio"] < 0).sum()
                if negative_cr_count > 0:
                    results.append(ValidationResult(
                        level=ValidationLevel.CRITICAL,
                        message=f"{negative_cr_count} treaties have negative combined ratios",
                        column="combined_ratio",
                        affected_rows=negative_cr_count,
                        suggested_action="Fix negative combined ratio values"
                    ))
        
        elif data_type == DataType.CLAIMS:
            # Claims-specific validations
            if "gross_claim_amount" in df.columns and "reinsurance_recovery" in df.columns:
                over_recovery_count = (df["reinsurance_recovery"] > df["gross_claim_amount"]).sum()
                if over_recovery_count > 0:
                    results.append(ValidationResult(
                        level=ValidationLevel.CRITICAL,
                        message=f"{over_recovery_count} claims have recoveries exceeding gross amounts",
                        column="reinsurance_recovery",
                        affected_rows=over_recovery_count,
                        suggested_action="Fix claims where recovery > gross amount"
                    ))
        
        return results
    
    def _detect_outliers(self, df: pl.DataFrame, data_type: DataType) -> List[ValidationResult]:
        """Detect statistical outliers"""
        
        results = []
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        
        for col in numeric_columns:
            try:
                col_data = df[col].drop_nulls()
                if len(col_data) < 10:  # Skip if too few values
                    continue
                
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outlier_pct = outlier_count / len(col_data) * 100
                
                # Be more lenient with outliers for reinsurance data (higher thresholds)
                outlier_threshold = 25 if data_type == DataType.TREATY else 15  # Reinsurance has more natural variation
                
                if outlier_pct > outlier_threshold:
                    results.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        message=f"Column '{col}' has {outlier_pct:.1f}% outliers (normal for reinsurance data with high variation)",
                        column=col,
                        affected_rows=outlier_count,
                        suggested_action=f"Review extreme values in column '{col}' if needed"
                    ))
            except Exception:
                pass
        
        return results
    
    def _profile_data(self, df: pl.DataFrame, data_type: DataType) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        profile = {
            "basic_stats": {},
            "column_profiles": {},
            "data_types": {},
            "missing_values": {},
            "unique_values": {},
            "date_ranges": {}
        }
        
        for col in df.columns:
            col_data = df[col]
            
            # Basic stats
            profile["data_types"][col] = str(col_data.dtype)
            profile["missing_values"][col] = {
                "count": col_data.null_count(),
                "percentage": col_data.null_count() / len(df) * 100
            }
            profile["unique_values"][col] = col_data.n_unique()
            
            # Column-specific profiling
            if col_data.dtype.is_numeric():
                try:
                    non_null_data = col_data.drop_nulls()
                    if len(non_null_data) > 0:
                        profile["column_profiles"][col] = {
                            "mean": float(non_null_data.mean()),
                            "median": float(non_null_data.median()),
                            "std": float(non_null_data.std()),
                            "min": float(non_null_data.min()),
                            "max": float(non_null_data.max()),
                            "q25": float(non_null_data.quantile(0.25)),
                            "q75": float(non_null_data.quantile(0.75))
                        }
                except Exception:
                    pass
            
            # Date range profiling
            if col_data.dtype == pl.Date or "date" in col.lower():
                try:
                    non_null_dates = col_data.drop_nulls()
                    if len(non_null_dates) > 0:
                        profile["date_ranges"][col] = {
                            "earliest": str(non_null_dates.min()),
                            "latest": str(non_null_dates.max())
                        }
                except Exception:
                    pass
        
        return profile
    
    def _generate_recommendations(
        self,
        validation_results: List[ValidationResult],
        data_profile: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Count issues by level
        critical_count = sum(1 for r in validation_results if r.level == ValidationLevel.CRITICAL)
        warning_count = sum(1 for r in validation_results if r.level == ValidationLevel.WARNING)
        
        if critical_count > 0:
            recommendations.append(f"Fix {critical_count} critical data issues before proceeding")
        
        if warning_count > 0:
            recommendations.append(f"Review {warning_count} data warnings for better model performance")
        
        # Missing value recommendations
        high_missing_cols = []
        for col, missing_info in data_profile["missing_values"].items():
            if missing_info["percentage"] > 25:
                high_missing_cols.append(col)
        
        if high_missing_cols:
            recommendations.append(f"Consider imputing or removing columns with high missing values: {', '.join(high_missing_cols[:3])}")
        
        # Data type recommendations
        date_cols = [col for col in data_profile["data_types"] if "date" in col.lower()]
        non_date_types = [col for col in date_cols if data_profile["data_types"][col] != "Date"]
        
        if non_date_types:
            recommendations.append(f"Convert date columns to proper date format: {', '.join(non_date_types[:3])}")
        
        return recommendations
    
    def _calculate_validity(self, validation_results: List[ValidationResult]) -> bool:
        """Calculate if data is valid for processing"""
        
        critical_issues = [r for r in validation_results if r.level == ValidationLevel.CRITICAL]
        return len(critical_issues) == 0
    
    def _calculate_confidence(
        self,
        validation_results: List[ValidationResult],
        data_profile: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for data quality"""
        
        score = 100.0
        
        # Deduct for issues (more lenient scoring)
        critical_count = sum(1 for r in validation_results if r.level == ValidationLevel.CRITICAL)
        warning_count = sum(1 for r in validation_results if r.level == ValidationLevel.WARNING)
        info_count = sum(1 for r in validation_results if r.level == ValidationLevel.INFO)
        
        score -= critical_count * 15  # Reduced from 20 to 15
        score -= warning_count * 3    # Reduced from 5 to 3  
        score -= info_count * 1       # Only 1 point for info issues
        
        # More lenient missing value penalty
        missing_values = [info["percentage"] for info in data_profile["missing_values"].values()]
        # Only penalize if average missing is high
        avg_missing = np.mean(missing_values)
        if avg_missing > 30:  # Only start penalizing after 30% missing
            score -= (avg_missing - 30) * 0.3    # Reduced penalty
        
        return max(0.0, min(100.0, score))
    
    def _create_error_report(
        self,
        file_name: str,
        data_type: DataType,
        error_message: str
    ) -> DataValidationReport:
        """Create error report for failed validation"""
        
        return DataValidationReport(
            file_name=file_name,
            data_type=data_type,
            total_rows=0,
            total_columns=0,
            validation_results=[ValidationResult(
                level=ValidationLevel.CRITICAL,
                message=error_message
            )],
            data_profile={},
            schema_mapping={},
            recommended_actions=[f"Fix error: {error_message}"],
            is_valid=False,
            confidence_score=0.0
        )
    
    def _is_compatible_type(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected"""
        
        type_mappings = {
            "String": ["Utf8", "String", "str"],
            "Float": ["Float64", "Float32", "Float", "float"],
            "Int": ["Int64", "Int32", "Int", "int"],
            "Date": ["Date", "Datetime", "datetime"]
        }
        
        return actual in type_mappings.get(expected, [expected])
    
    def _load_schema_definitions(self) -> Dict[str, Dict]:
        """Load schema definitions for different data types"""
        
        return {
            "treaty": {
                "required": ["treaty_id", "premium", "loss_ratio"],
                "optional": ["treaty_type", "business_line", "cedant", "reinsurer"]
            },
            "claims": {
                "required": ["claim_id", "treaty_id", "gross_claim_amount"],
                "optional": ["occurrence_date", "report_date", "reinsurance_recovery"]
            },
            "portfolio": {
                "required": ["portfolio_id", "total_sum_insured", "number_of_risks"],
                "optional": ["business_line", "territory", "premium_rate"]
            },
            "catastrophe": {
                "required": ["event_id", "event_type", "occurrence_date"],
                "optional": ["industry_loss", "modeled_loss", "location"]
            },
            "development": {
                "required": ["claim_id", "valuation_date", "incurred_amount"],
                "optional": ["paid_amount", "case_reserves", "development_quarter"]
            }
        }
    
    def _load_validation_rules(self) -> Dict[str, Dict]:
        """Load validation rules for different data types"""
        
        return {
            "treaty": {
                "required_columns": ["premium"],
                "column_types": {"premium": "Float", "loss_ratio": "Float"},
                "ranges": {"loss_ratio": {"min": 0, "max": 5}, "premium": {"min": 0}}
            },
            "claims": {
                "required_columns": ["gross_claim_amount"],
                "column_types": {"gross_claim_amount": "Float"},
                "ranges": {"gross_claim_amount": {"min": 0}}
            },
            "portfolio": {
                "required_columns": ["total_sum_insured"],
                "column_types": {"total_sum_insured": "Float", "number_of_risks": "Int"},
                "ranges": {"total_sum_insured": {"min": 0}, "number_of_risks": {"min": 1}}
            },
            "catastrophe": {
                "required_columns": ["event_id"],
                "column_types": {"industry_loss": "Float", "modeled_loss": "Float"},
                "ranges": {"industry_loss": {"min": 0}, "modeled_loss": {"min": 0}}
            },
            "development": {
                "required_columns": ["incurred_amount"],
                "column_types": {"incurred_amount": "Float", "paid_amount": "Float"},
                "ranges": {"incurred_amount": {"min": 0}, "paid_amount": {"min": 0}}
            }
        }


def export_validation_report(report: DataValidationReport, output_path: str) -> None:
    """Export validation report to JSON file"""
    
    # Convert to serializable format
    report_dict = asdict(report)
    
    # Convert validation results
    report_dict["validation_results"] = [
        {
            "level": result.level.value,
            "message": result.message,
            "column": result.column,
            "affected_rows": result.affected_rows,
            "suggested_action": result.suggested_action
        }
        for result in report.validation_results
    ]
    
    report_dict["data_type"] = report.data_type.value
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)