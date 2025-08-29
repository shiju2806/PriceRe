"""
Actuarial Data Cleaner
Automated cleaning and standardization for actuarial data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import re

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations"""
    handle_missing: bool = True
    standardize_dates: bool = True
    normalize_amounts: bool = True
    fix_data_types: bool = True
    remove_duplicates: bool = True
    standardize_categories: bool = True
    handle_outliers: bool = True
    imputation_method: str = "smart"  # smart, mean, median, forward_fill, drop

@dataclass 
class CleaningResult:
    """Results from data cleaning"""
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    cleaning_actions: List[Dict[str, Any]]
    imputed_values: Dict[str, int]
    standardizations: Dict[str, Any]
    removed_records: int
    data_quality_before: float
    data_quality_after: float
    cleaned_df: pd.DataFrame

class ActuarialDataCleaner:
    """
    Comprehensive data cleaner for actuarial data
    Implements industry-standard cleaning procedures
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.cleaning_log = []
        self.actuarial_mappings = self._initialize_mappings()
        
    def _initialize_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standardization mappings"""
        
        return {
            "gender": {
                "M": ["M", "MALE", "1", "Male", "male", "m"],
                "F": ["F", "FEMALE", "2", "Female", "female", "f"],
                "U": ["U", "UNKNOWN", "0", "Unknown", "unknown", "u", None, np.nan]
            },
            "smoker_status": {
                "NS": ["NS", "N", "NO", "NON-SMOKER", "Non-Smoker", "0", False],
                "S": ["S", "Y", "YES", "SMOKER", "Smoker", "1", True],
                "U": ["U", "UNKNOWN", "Unknown", None, np.nan]
            },
            "policy_status": {
                "ACTIVE": ["ACTIVE", "A", "INFORCE", "IF", "In Force", "1"],
                "LAPSED": ["LAPSED", "L", "LAPSE", "Lapsed", "2"],
                "SURRENDERED": ["SURRENDERED", "S", "SURRENDER", "Surrendered", "3"],
                "MATURED": ["MATURED", "M", "MATURE", "Matured", "4"],
                "CLAIM": ["CLAIM", "C", "DEATH", "Death Claim", "5"],
                "PENDING": ["PENDING", "P", "Pending", "0"]
            },
            "product_type": {
                "TERM": ["TERM", "TL", "TERM LIFE", "Term Life", "Term"],
                "WHOLE_LIFE": ["WL", "WHOLE", "WHOLE LIFE", "Whole Life"],
                "UNIVERSAL_LIFE": ["UL", "UNIVERSAL", "UNIVERSAL LIFE", "Universal Life"],
                "VARIABLE_LIFE": ["VL", "VARIABLE", "VARIABLE LIFE", "Variable Life"],
                "ANNUITY": ["ANNUITY", "ANN", "Annuity"],
                "PENSION": ["PENSION", "PEN", "Pension"],
                "401K": ["401K", "401(k)", "401k"],
                "IRA": ["IRA", "Individual Retirement Account"]
            },
            "underwriting_class": {
                "PREFERRED_PLUS": ["PP", "PREFERRED PLUS", "Preferred Plus", "Super Preferred", "1"],
                "PREFERRED": ["P", "PREFERRED", "Preferred", "2"],
                "STANDARD_PLUS": ["SP", "STANDARD PLUS", "Standard Plus", "3"],
                "STANDARD": ["S", "STANDARD", "Standard", "4"],
                "SUBSTANDARD": ["SS", "SUBSTANDARD", "Substandard", "Rated", "5"],
                "DECLINED": ["D", "DECLINED", "Declined", "6"]
            }
        }
    
    def clean_dataset(self, df: pd.DataFrame) -> CleaningResult:
        """
        Perform comprehensive cleaning on actuarial dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            CleaningResult with cleaned data and documentation
        """
        
        # Track original state
        original_shape = df.shape
        original_quality = self._calculate_data_quality(df)
        df_cleaned = df.copy()
        
        # Initialize tracking
        cleaning_actions = []
        imputed_values = {}
        standardizations = {}
        
        # Step 1: Remove duplicates
        if self.config.remove_duplicates:
            df_cleaned, action = self._remove_duplicates(df_cleaned)
            cleaning_actions.append(action)
        
        # Step 2: Fix data types
        if self.config.fix_data_types:
            df_cleaned, action = self._fix_data_types(df_cleaned)
            cleaning_actions.append(action)
        
        # Step 3: Standardize dates
        if self.config.standardize_dates:
            df_cleaned, action = self._standardize_dates(df_cleaned)
            cleaning_actions.append(action)
        
        # Step 4: Standardize categories
        if self.config.standardize_categories:
            df_cleaned, std_results = self._standardize_categories(df_cleaned)
            standardizations.update(std_results)
            cleaning_actions.append({
                "action": "standardize_categories",
                "fields_standardized": list(std_results.keys())
            })
        
        # Step 5: Normalize amounts
        if self.config.normalize_amounts:
            df_cleaned, action = self._normalize_amounts(df_cleaned)
            cleaning_actions.append(action)
        
        # Step 6: Handle missing values
        if self.config.handle_missing:
            df_cleaned, imp_results = self._handle_missing_values(df_cleaned)
            imputed_values.update(imp_results)
            cleaning_actions.append({
                "action": "handle_missing",
                "method": self.config.imputation_method,
                "fields_imputed": list(imp_results.keys())
            })
        
        # Step 7: Handle outliers
        if self.config.handle_outliers:
            df_cleaned, action = self._handle_outliers(df_cleaned)
            cleaning_actions.append(action)
        
        # Step 8: Actuarial-specific validations
        df_cleaned = self._apply_actuarial_rules(df_cleaned)
        
        # Calculate final quality
        final_quality = self._calculate_data_quality(df_cleaned)
        
        return CleaningResult(
            original_shape=original_shape,
            cleaned_shape=df_cleaned.shape,
            cleaning_actions=cleaning_actions,
            imputed_values=imputed_values,
            standardizations=standardizations,
            removed_records=original_shape[0] - df_cleaned.shape[0],
            data_quality_before=original_quality,
            data_quality_after=final_quality,
            cleaned_df=df_cleaned
        )
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate records"""
        
        initial_count = len(df)
        
        # Identify key fields for duplicate detection
        key_fields = []
        if "policy_id" in df.columns:
            key_fields.append("policy_id")
        if "issue_date" in df.columns:
            key_fields.append("issue_date")
        
        if key_fields:
            df = df.drop_duplicates(subset=key_fields, keep='last')
        else:
            df = df.drop_duplicates()
        
        removed = initial_count - len(df)
        
        return df, {
            "action": "remove_duplicates",
            "removed_count": removed,
            "method": f"key_fields: {key_fields}" if key_fields else "all_columns"
        }
    
    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Fix data types based on field names"""
        
        type_fixes = {}
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Numeric fields
            if any(term in col_lower for term in ["age", "amount", "premium", "value", "benefit", "rate"]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    type_fixes[col] = "numeric"
                except:
                    pass
            
            # Date fields
            elif any(term in col_lower for term in ["date", "time", "dob", "birth"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    type_fixes[col] = "datetime"
                except:
                    pass
            
            # Boolean fields
            elif any(term in col_lower for term in ["is_", "has_", "flag"]):
                try:
                    df[col] = df[col].astype(bool)
                    type_fixes[col] = "boolean"
                except:
                    pass
            
            # String fields
            elif df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                type_fixes[col] = "string"
        
        return df, {
            "action": "fix_data_types",
            "fields_fixed": type_fixes
        }
    
    def _standardize_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Standardize date formats"""
        
        date_columns = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower():
                date_columns.append(col)
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Check for future dates (likely errors)
                future_mask = df[col] > pd.Timestamp.now()
                if future_mask.any():
                    # Assume year typo (e.g., 2092 instead of 1992)
                    df.loc[future_mask, col] = df.loc[future_mask, col].apply(
                        lambda x: x.replace(year=x.year - 100) if x.year > 2050 else x
                    )
        
        return df, {
            "action": "standardize_dates",
            "date_columns": date_columns
        }
    
    def _standardize_categories(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Standardize categorical values"""
        
        standardizations = {}
        
        for field, mappings in self.actuarial_mappings.items():
            # Find matching columns
            matching_cols = [col for col in df.columns if field in col.lower()]
            
            for col in matching_cols:
                if col in df.columns:
                    # Create reverse mapping
                    reverse_map = {}
                    for standard_value, variations in mappings.items():
                        for variation in variations:
                            reverse_map[variation] = standard_value
                    
                    # Apply mapping
                    original_unique = df[col].nunique()
                    df[col] = df[col].map(lambda x: reverse_map.get(x, x))
                    new_unique = df[col].nunique()
                    
                    standardizations[col] = {
                        "original_unique": original_unique,
                        "standardized_unique": new_unique,
                        "mapping_applied": field
                    }
        
        return df, standardizations
    
    def _normalize_amounts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize monetary amounts"""
        
        amount_fields = []
        
        for col in df.columns:
            if any(term in col.lower() for term in ["amount", "premium", "value", "benefit"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    amount_fields.append(col)
                    
                    # Remove negative values for amounts that should be positive
                    if "face_amount" in col.lower() or "death_benefit" in col.lower():
                        df[col] = df[col].abs()
                    
                    # Round to 2 decimal places for currency
                    df[col] = df[col].round(2)
        
        return df, {
            "action": "normalize_amounts",
            "fields_normalized": amount_fields
        }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values with smart imputation"""
        
        imputed = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            
            if missing_count > 0:
                if self.config.imputation_method == "drop":
                    df = df.dropna(subset=[col])
                    imputed[col] = {"method": "dropped", "count": missing_count}
                    
                elif self.config.imputation_method == "smart":
                    # Smart imputation based on field type
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Use median for amounts, mean for rates
                        if "amount" in col.lower() or "premium" in col.lower():
                            fill_value = df[col].median()
                            df[col].fillna(fill_value, inplace=True)
                            imputed[col] = {"method": "median", "value": fill_value, "count": missing_count}
                        else:
                            fill_value = df[col].mean()
                            df[col].fillna(fill_value, inplace=True)
                            imputed[col] = {"method": "mean", "value": fill_value, "count": missing_count}
                    
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        # Forward fill for dates
                        df[col].fillna(method='ffill', inplace=True)
                        imputed[col] = {"method": "forward_fill", "count": missing_count}
                    
                    else:
                        # Mode for categorical
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                        df[col].fillna(mode_value, inplace=True)
                        imputed[col] = {"method": "mode", "value": mode_value, "count": missing_count}
        
        return df, imputed
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers in numeric fields"""
        
        outliers_handled = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # Using 3x IQR for actuarial data
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    
                    outliers_handled[col] = {
                        "count": outlier_count,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound
                    }
        
        return df, {
            "action": "handle_outliers",
            "method": "capping",
            "fields_handled": outliers_handled
        }
    
    def _apply_actuarial_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply actuarial-specific business rules"""
        
        # Fix age inconsistencies
        if "issue_age" in df.columns and "current_age" in df.columns:
            # Current age should be >= issue age
            df.loc[df["current_age"] < df["issue_age"], "current_age"] = df["issue_age"]
        
        # Fix premium/face amount relationships
        if "premium" in df.columns and "face_amount" in df.columns:
            # Premium should be reasonable percentage of face amount
            df["premium_ratio"] = df["premium"] / df["face_amount"]
            
            # Cap unreasonable ratios
            df.loc[df["premium_ratio"] > 0.2, "premium"] = df["face_amount"] * 0.2
            df.loc[df["premium_ratio"] < 0.0001, "premium"] = df["face_amount"] * 0.0001
            
            df.drop("premium_ratio", axis=1, inplace=True)
        
        # Ensure positive values
        positive_fields = ["face_amount", "death_benefit", "cash_value", "premium"]
        for field in positive_fields:
            if field in df.columns:
                df[field] = df[field].abs()
        
        # Age bounds
        age_fields = ["issue_age", "current_age", "attained_age"]
        for field in age_fields:
            if field in df.columns:
                df[field] = df[field].clip(0, 120)
        
        return df
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        completeness = (total_cells - missing_cells) / total_cells
        
        # Additional quality factors
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency_score = 1.0
        
        for col in numeric_cols:
            if col in df.columns and len(df[col]) > 0:
                # Check for negative values in fields that should be positive
                if any(term in col.lower() for term in ["amount", "age", "premium"]):
                    negative_ratio = (df[col] < 0).sum() / len(df[col])
                    consistency_score *= (1 - negative_ratio)
        
        # Weighted quality score
        quality_score = (completeness * 0.7 + consistency_score * 0.3) * 100
        
        return quality_score
    
    def generate_cleaning_report(self, result: CleaningResult) -> str:
        """Generate human-readable cleaning report"""
        
        report = []
        report.append("=" * 60)
        report.append("ACTUARIAL DATA CLEANING REPORT")
        report.append("=" * 60)
        
        report.append(f"\n📊 DATA SHAPE")
        report.append(f"Original: {result.original_shape[0]:,} rows × {result.original_shape[1]} columns")
        report.append(f"Cleaned:  {result.cleaned_shape[0]:,} rows × {result.cleaned_shape[1]} columns")
        report.append(f"Records Removed: {result.removed_records:,}")
        
        report.append(f"\n📈 DATA QUALITY")
        report.append(f"Before: {result.data_quality_before:.1f}%")
        report.append(f"After:  {result.data_quality_after:.1f}%")
        report.append(f"Improvement: {result.data_quality_after - result.data_quality_before:+.1f}%")
        
        report.append(f"\n🔧 CLEANING ACTIONS")
        for action in result.cleaning_actions:
            report.append(f"• {action['action'].replace('_', ' ').title()}")
            if "removed_count" in action:
                report.append(f"  - Removed {action['removed_count']} duplicates")
            if "fields_fixed" in action:
                report.append(f"  - Fixed {len(action['fields_fixed'])} field types")
        
        if result.imputed_values:
            report.append(f"\n🔮 IMPUTED VALUES")
            for field, details in list(result.imputed_values.items())[:5]:
                report.append(f"• {field}: {details['method']} ({details['count']} values)")
        
        if result.standardizations:
            report.append(f"\n📐 STANDARDIZATIONS")
            for field, details in list(result.standardizations.items())[:5]:
                report.append(f"• {field}: {details['original_unique']} → {details['standardized_unique']} unique values")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)