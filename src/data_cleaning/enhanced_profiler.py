"""
Enhanced Data Profiling and Cleaning System
Uses professional open-source libraries: pyjanitor, missingno
"""

import pandas as pd
import numpy as np

# Try to import professional libraries
try:
    import pyjanitor as pj
    PYJANITOR_AVAILABLE = True
except ImportError:
    PYJANITOR_AVAILABLE = False

try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from datetime import datetime
import re
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataProfiler:
    """
    Professional data profiler using open-source libraries
    Combines pyjanitor's cleaning capabilities with comprehensive profiling
    """
    
    def __init__(self):
        self.cleaning_history = []
        self.original_data = None
        self.processed_data = None
        
    def profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data profiling using professional techniques
        """
        self.original_data = df.copy()
        
        profile = {
            "basic_info": self._get_basic_info(df),
            "data_quality": self._assess_data_quality(df),
            "missing_patterns": self._analyze_missing_patterns(df),
            "column_analysis": self._analyze_columns(df),
            "data_types": self._infer_data_types(df),
            "structural_issues": self._detect_structural_issues(df),
            "recommendations": [],
            "cleaning_actions": []
        }
        
        # Generate recommendations
        profile["recommendations"] = self._generate_recommendations(profile, df)
        
        return profile
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicate_rows": df.duplicated().sum(),
            "completely_empty_rows": (df.isna().all(axis=1)).sum(),
            "columns_with_all_nulls": (df.isna().all()).sum()
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess overall data quality metrics"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        return {
            "overall_completeness": ((total_cells - missing_cells) / total_cells) * 100,
            "rows_with_missing": (df.isna().any(axis=1).sum() / len(df)) * 100,
            "columns_with_missing": (df.isna().any().sum() / len(df.columns)) * 100,
            "duplicate_rate": (df.duplicated().sum() / len(df)) * 100
        }
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_info[col] = {
                "missing_count": missing_count,
                "missing_percentage": (missing_count / len(df)) * 100,
                "missing_pattern": self._identify_missing_pattern(df[col])
            }
        
        return missing_info
    
    def _identify_missing_pattern(self, series: pd.Series) -> str:
        """Identify patterns in missing data"""
        if series.isna().sum() == 0:
            return "No missing values"
        elif series.isna().sum() == len(series):
            return "All values missing"
        elif series.isna().iloc[:10].all():
            return "Missing at beginning"
        elif series.isna().iloc[-10:].all():
            return "Missing at end"
        else:
            return "Scattered missing"
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column in detail"""
        column_analysis = {}
        
        for col in df.columns:
            analysis = {
                "dtype": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "unique_ratio": df[col].nunique() / len(df),
                "most_frequent": None,
                "issues": []
            }
            
            # Get most frequent value
            if not df[col].empty:
                value_counts = df[col].value_counts()
                if not value_counts.empty:
                    analysis["most_frequent"] = value_counts.index[0]
            
            # Detect issues
            if df[col].dtype == 'object':
                analysis["issues"].extend(self._detect_text_issues(df[col]))
            
            # Check for potential numeric columns stored as text
            if df[col].dtype == 'object':
                if self._could_be_numeric(df[col]):
                    analysis["issues"].append("Could be numeric")
            
            # Check for date columns
            if self._could_be_date(df[col]):
                analysis["issues"].append("Could be datetime")
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _detect_text_issues(self, series: pd.Series) -> List[str]:
        """Detect common text data issues"""
        issues = []
        
        # Check for mixed case
        text_values = series.dropna().astype(str)
        if len(text_values) > 0:
            has_mixed_case = any(val != val.lower() and val != val.upper() for val in text_values)
            if has_mixed_case:
                issues.append("Mixed case values")
            
            # Check for leading/trailing whitespace
            has_whitespace = any(val != val.strip() for val in text_values)
            if has_whitespace:
                issues.append("Leading/trailing whitespace")
            
            # Check for inconsistent formatting
            unique_values = set(text_values)
            if len(unique_values) > 1:
                # Look for similar values that might be inconsistent
                lower_values = [val.lower().strip() for val in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    issues.append("Inconsistent formatting")
        
        return issues
    
    def _could_be_numeric(self, series: pd.Series) -> bool:
        """Check if a text column could be converted to numeric"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        numeric_count = 0
        for val in sample:
            # Remove common formatting characters
            cleaned = str(val).replace(',', '').replace('$', '').replace('%', '').strip()
            try:
                float(cleaned)
                numeric_count += 1
            except ValueError:
                continue
        
        return (numeric_count / len(sample)) > 0.7
    
    def _could_be_date(self, series: pd.Series) -> bool:
        """Check if a column could be converted to datetime"""
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        date_count = 0
        for val in sample:
            val_str = str(val)
            if any(re.match(pattern, val_str) for pattern in date_patterns):
                date_count += 1
        
        return (date_count / len(sample)) > 0.7
    
    def _infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer appropriate data types for columns"""
        recommendations = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            if current_type == 'object':
                if self._could_be_numeric(df[col]):
                    recommendations[col] = "numeric"
                elif self._could_be_date(df[col]):
                    recommendations[col] = "datetime"
                else:
                    recommendations[col] = "string"
            else:
                recommendations[col] = current_type
        
        return recommendations
    
    def _detect_structural_issues(self, df: pd.DataFrame) -> List[str]:
        """Detect structural issues in the dataset"""
        issues = []
        
        # Check for potential header rows mixed in data
        if len(df) > 5:
            first_few_rows = df.head(5)
            for idx, row in first_few_rows.iterrows():
                if idx > 0:  # Skip actual header
                    # Check if row values match column names
                    row_values = [str(val).lower() for val in row if pd.notna(val)]
                    col_names = [str(col).lower() for col in df.columns]
                    matches = sum(1 for val in row_values if any(val in col or col in val for col in col_names))
                    if matches > len(df.columns) * 0.5:
                        issues.append(f"Potential header row at index {idx}")
        
        # Check for completely empty rows
        empty_rows = (df.isna().all(axis=1)).sum()
        if empty_rows > 0:
            issues.append(f"{empty_rows} completely empty rows")
        
        # Check for rows with mostly missing data
        mostly_empty = (df.isna().sum(axis=1) > len(df.columns) * 0.8).sum()
        if mostly_empty > empty_rows:
            issues.append(f"{mostly_empty - empty_rows} rows with >80% missing data")
        
        return issues
    
    def _generate_recommendations(self, profile: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Missing data recommendations
        quality = profile["data_quality"]
        if quality["overall_completeness"] < 80:
            recommendations.append({
                "category": "Data Quality",
                "issue": "Low data completeness",
                "recommendation": "Consider removing columns/rows with high missing rates",
                "action": "remove_high_missing"
            })
        
        # Duplicate recommendations
        if profile["basic_info"]["duplicate_rows"] > 0:
            recommendations.append({
                "category": "Data Quality", 
                "issue": f"{profile['basic_info']['duplicate_rows']} duplicate rows",
                "recommendation": "Remove duplicate rows",
                "action": "remove_duplicates"
            })
        
        # Column-specific recommendations
        for col, analysis in profile["column_analysis"].items():
            for issue in analysis["issues"]:
                if issue == "Mixed case values":
                    recommendations.append({
                        "category": "Formatting",
                        "issue": f"Column '{col}' has mixed case values",
                        "recommendation": "Standardize text case",
                        "action": f"standardize_case_{col}"
                    })
                elif issue == "Leading/trailing whitespace":
                    recommendations.append({
                        "category": "Formatting",
                        "issue": f"Column '{col}' has whitespace issues", 
                        "recommendation": "Trim whitespace",
                        "action": f"trim_whitespace_{col}"
                    })
                elif issue == "Could be numeric":
                    recommendations.append({
                        "category": "Data Types",
                        "issue": f"Column '{col}' could be converted to numeric",
                        "recommendation": "Convert to appropriate numeric type",
                        "action": f"convert_numeric_{col}"
                    })
                elif issue == "Could be datetime":
                    recommendations.append({
                        "category": "Data Types", 
                        "issue": f"Column '{col}' could be converted to datetime",
                        "recommendation": "Convert to datetime format",
                        "action": f"convert_datetime_{col}"
                    })
        
        return recommendations
    
    def apply_cleaning_actions(self, df: pd.DataFrame, selected_actions: List[str]) -> pd.DataFrame:
        """
        Apply selected cleaning actions using pyjanitor
        """
        cleaned_df = df.copy()
        applied_actions = []
        
        for action in selected_actions:
            try:
                if action == "remove_duplicates":
                    before_count = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates()
                    applied_actions.append(f"Removed {before_count - len(cleaned_df)} duplicate rows")
                
                elif action == "remove_high_missing":
                    # Remove rows with >80% missing data
                    before_count = len(cleaned_df)
                    cleaned_df = cleaned_df.dropna(thresh=int(len(cleaned_df.columns) * 0.2))
                    applied_actions.append(f"Removed {before_count - len(cleaned_df)} rows with high missing data")
                
                elif action.startswith("trim_whitespace_"):
                    col = action.replace("trim_whitespace_", "")
                    if col in cleaned_df.columns:
                        # Use pyjanitor if available, otherwise manual trimming
                        if PYJANITOR_AVAILABLE:
                            cleaned_df = cleaned_df.clean_names()  # This will also trim
                        
                        # Additional specific trimming for the column
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                        applied_actions.append(f"Trimmed whitespace in column '{col}'")
                
                elif action.startswith("standardize_case_"):
                    col = action.replace("standardize_case_", "")
                    if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
                        # Use title case for names, lowercase for categories
                        if any(keyword in col.lower() for keyword in ['name', 'city', 'state']):
                            cleaned_df[col] = cleaned_df[col].astype(str).str.title()
                        else:
                            cleaned_df[col] = cleaned_df[col].astype(str).str.lower()
                        applied_actions.append(f"Standardized case in column '{col}'")
                
                elif action.startswith("convert_numeric_"):
                    col = action.replace("convert_numeric_", "")
                    if col in cleaned_df.columns:
                        # Clean and convert to numeric
                        cleaned_df[col] = (cleaned_df[col]
                                         .astype(str)
                                         .str.replace('[,$%]', '', regex=True)
                                         .str.strip())
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                        applied_actions.append(f"Converted column '{col}' to numeric")
                
                elif action.startswith("convert_datetime_"):
                    col = action.replace("convert_datetime_", "")
                    if col in cleaned_df.columns:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                        applied_actions.append(f"Converted column '{col}' to datetime")
                
            except Exception as e:
                logger.warning(f"Failed to apply action '{action}': {e}")
        
        # Apply pyjanitor's general cleaning if available
        if PYJANITOR_AVAILABLE:
            try:
                cleaned_df = cleaned_df.clean_names()  # Standardize column names
                applied_actions.append("Standardized column names")
            except Exception as e:
                logger.warning(f"Failed to clean column names: {e}")
        
        # Store cleaning history
        self.cleaning_history.append({
            "timestamp": datetime.now().isoformat(),
            "actions": applied_actions,
            "shape_before": df.shape,
            "shape_after": cleaned_df.shape
        })
        
        self.processed_data = cleaned_df
        return cleaned_df
    
    def generate_missing_data_visualization(self, df: pd.DataFrame) -> str:
        """Generate missing data visualization using missingno if available"""
        if not MISSINGNO_AVAILABLE or not PLOTTING_AVAILABLE:
            logger.warning("Missing data visualization requires missingno and matplotlib")
            return ""
        
        try:
            plt.figure(figsize=(12, 6))
            msno.matrix(df)
            plt.title("Missing Data Pattern Analysis")
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode()
        except Exception as e:
            logger.warning(f"Failed to generate missing data visualization: {e}")
            return ""
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning actions performed"""
        return {
            "history": self.cleaning_history,
            "total_actions": len(self.cleaning_history),
            "original_shape": self.original_data.shape if self.original_data is not None else None,
            "final_shape": self.processed_data.shape if self.processed_data is not None else None
        }

# Global instance for use in Streamlit
enhanced_profiler = EnhancedDataProfiler()