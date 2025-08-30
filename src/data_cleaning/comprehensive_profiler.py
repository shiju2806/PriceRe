"""
Comprehensive Data Profiler
Deep analysis of data quality issues across multiple dimensions
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import logging

@dataclass
class DataIssue:
    """Represents a specific data quality issue"""
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    column: Optional[str]
    description: str
    affected_rows: int
    examples: List[str]
    suggested_fix: str

@dataclass
class ColumnProfile:
    """Profile of a single column"""
    name: str
    dtype: str
    inferred_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[str]
    issues: List[DataIssue]

@dataclass
class DataProfile:
    """Complete profile of a dataset"""
    filename: str
    shape: Tuple[int, int]
    structural_issues: List[DataIssue]
    column_profiles: List[ColumnProfile]
    overall_quality_score: int
    recommendations: List[str]

class ComprehensiveDataProfiler:
    """
    Comprehensive data profiler that analyzes:
    - Structural issues (empty rows, misplaced headers, etc.)
    - Column-level quality (types, formats, consistency)
    - Value-level issues (spaces, cases, formats)
    - Business rule violations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced patterns for comprehensive data type inference
        self.patterns = {
            'date': [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
                r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # M/D/YYYY
                r'^\d{2}/\d{2}/\d{2}$',  # MM/DD/YY
                r'^\d{1,2}/\d{1,2}/\d{2}$',  # M/D/YY
                r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
                r'^\d{1,2}-\d{1,2}-\d{4}$',  # M-D-YYYY
                r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$',  # Jan 1, 2023
                r'^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$',  # 1 Jan 2023
                r'^\d{4}-\d{2}$',  # YYYY-MM
                r'^[A-Za-z]{3}\s+\d{4}$',  # Jan 2023
                r'^Q[1-4]\s+\d{4}$',  # Q1 2023
            ],
            'phone': [
                r'^\(\d{3}\)\s*\d{3}-\d{4}$',  # (555) 123-4567
                r'^\d{3}-\d{3}-\d{4}$',  # 555-123-4567
                r'^\d{3}\.\d{3}\.\d{4}$',  # 555.123.4567
                r'^\d{10}$',  # 5551234567
                r'^\+1\d{10}$',  # +15551234567
            ],
            'email': [
                r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            ],
            'currency': [
                r'^\$[\d,]+\.?\d*$',  # $1,234.56
                r'^[\d,]+\.?\d*\$?$',  # 1234.56 or 1234$
                r'^USD\s*[\d,]+\.?\d*$',  # USD 1234.56
                r'^[\d,]+\.?\d*\s*USD$',  # 1234.56 USD
            ],
            'policy_id': [
                r'^[A-Z]{2,4}\d{6,}$',  # ABC123456
                r'^POL[_-]?\d+$',  # POL_123456 or POL123456
                r'^\d{8,12}$',  # 12345678901
            ],
            'ssn': [
                r'^\d{3}-\d{2}-\d{4}$',  # 123-45-6789
                r'^\d{9}$',  # 123456789
            ]
        }
        
        # Business domain knowledge for insurance data
        self.insurance_column_patterns = {
            'policy': ['policy', 'pol_', 'policy_id', 'policy_num', 'contract'],
            'claim': ['claim', 'claim_id', 'claim_num', 'loss'],
            'premium': ['premium', 'prem', 'payment', 'amount'],
            'date_fields': ['date', 'effective', 'expiry', 'issue', 'birth', 'claim_date'],
            'person': ['name', 'first', 'last', 'insured', 'beneficiary'],
            'address': ['address', 'street', 'city', 'state', 'zip', 'postal'],
            'coverage': ['coverage', 'limit', 'deductible', 'face_amount', 'benefit']
        }
        
        # Junk row patterns
        self.junk_patterns = [
            r'^total:?$',
            r'^summary:?$', 
            r'^end\s+of\s+report$',
            r'^report\s+generated',
            r'^page\s+\d+',
            r'^\d+\s+records?$',
            r'^test$',
            r'^[x]+$',  # xxxx
            r'^[-]+$',  # ----
            r'^[=]+$',  # ====
            r'^\d{1,3}$',  # Just numbers like 1, 12, 123
            r'^[a-z]{1,4}$',  # Random letters like asdf, test
            r'^\s*note:',
            r'^\s*disclaimer',
            r'confidential',
            r'internal\s+use',
        ]
    
    def profile_dataset(self, df: pd.DataFrame, filename: str = "dataset") -> DataProfile:
        """
        Perform comprehensive profiling of a dataset
        """
        self.logger.info(f"Starting comprehensive profiling of {filename}")
        
        # 1. Structural analysis
        structural_issues = self._analyze_structure(df)
        
        # 2. Column-by-column analysis
        column_profiles = []
        for column in df.columns:
            profile = self._profile_column(df, column)
            column_profiles.append(profile)
        
        # 3. Calculate overall quality score
        quality_score = self._calculate_quality_score(structural_issues, column_profiles)
        
        # 4. Generate recommendations
        recommendations = self._generate_recommendations(structural_issues, column_profiles)
        
        return DataProfile(
            filename=filename,
            shape=df.shape,
            structural_issues=structural_issues,
            column_profiles=column_profiles,
            overall_quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _analyze_structure(self, df: pd.DataFrame) -> List[DataIssue]:
        """Analyze structural issues in the dataset"""
        issues = []
        
        # 1. Empty rows
        completely_empty = df.isnull().all(axis=1)
        empty_count = completely_empty.sum()
        if empty_count > 0:
            empty_indices = df[completely_empty].index.tolist()
            issues.append(DataIssue(
                issue_type='empty_rows',
                severity='critical',
                column=None,
                description=f'{empty_count} completely empty rows found',
                affected_rows=empty_count,
                examples=[f"Row {i}" for i in empty_indices[:3]],
                suggested_fix='Remove empty rows'
            ))
        
        # 2. Mostly empty rows (>80% missing)
        mostly_empty = (df.isnull().sum(axis=1) / len(df.columns)) > 0.8
        mostly_empty_count = mostly_empty.sum()
        if mostly_empty_count > 0:
            mostly_empty_indices = df[mostly_empty].index.tolist()
            issues.append(DataIssue(
                issue_type='mostly_empty_rows',
                severity='warning',
                column=None,
                description=f'{mostly_empty_count} rows with >80% missing data',
                affected_rows=mostly_empty_count,
                examples=[f"Row {i}" for i in mostly_empty_indices[:3]],
                suggested_fix='Review and potentially remove sparse rows'
            ))
        
        # 3. Potential misplaced headers
        header_issues = self._detect_misplaced_headers(df)
        issues.extend(header_issues)
        
        # 4. Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(DataIssue(
                issue_type='duplicate_rows',
                severity='warning',
                column=None,
                description=f'{duplicate_count} duplicate rows found',
                affected_rows=duplicate_count,
                examples=['Exact duplicates across all columns'],
                suggested_fix='Remove duplicate rows'
            ))
        
        return issues
    
    def _detect_misplaced_headers(self, df: pd.DataFrame) -> List[DataIssue]:
        """Detect if headers are misplaced (data in header position or vice versa)"""
        issues = []
        
        # Check if first few rows contain potential header-like data
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            text_values = row.astype(str).tolist()
            
            # Look for header-like patterns (mostly text, no numbers)
            non_numeric_count = sum(1 for val in text_values if not self._is_numeric(val) and val.strip() != '')
            if non_numeric_count > len(text_values) * 0.7 and i > 0:  # Not the actual header
                issues.append(DataIssue(
                    issue_type='misplaced_header',
                    severity='critical',
                    column=None,
                    description=f'Row {i+1} appears to contain column headers',
                    affected_rows=1,
                    examples=text_values[:3],
                    suggested_fix=f'Promote row {i+1} to column headers'
                ))
        
        return issues
    
    def _profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Create detailed profile for a single column"""
        
        col_data = df[column]
        
        # Basic statistics
        null_count = col_data.isnull().sum()
        null_percentage = (null_count / len(col_data)) * 100
        unique_count = col_data.nunique()
        unique_percentage = (unique_count / len(col_data)) * 100
        
        # Sample values (non-null)
        sample_values = col_data.dropna().astype(str).head(5).tolist()
        
        # Infer actual data type
        inferred_type = self._infer_column_type(col_data)
        
        # Detect column-specific issues
        issues = self._detect_column_issues(col_data, column, inferred_type)
        
        return ColumnProfile(
            name=column,
            dtype=str(col_data.dtype),
            inferred_type=inferred_type,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            sample_values=sample_values,
            issues=issues
        )
    
    def _infer_column_type(self, col_data: pd.Series) -> str:
        """Infer the actual data type based on content patterns"""
        
        # Get non-null string values for pattern matching
        non_null_str = col_data.dropna().astype(str)
        
        if len(non_null_str) == 0:
            return 'empty'
        
        # Test against patterns
        for data_type, patterns in self.patterns.items():
            matches = 0
            for pattern in patterns:
                pattern_matches = non_null_str.str.match(pattern, na=False).sum()
                matches += pattern_matches
            
            # If >70% match any pattern for this type
            if matches / len(non_null_str) > 0.7:
                return data_type
        
        # Fallback to pandas dtype analysis
        if col_data.dtype in ['int64', 'float64']:
            return 'numeric'
        elif col_data.dtype == 'object':
            # Check if it's categorical (low unique count)
            if col_data.nunique() / len(col_data) < 0.1:
                return 'categorical'
            else:
                return 'text'
        else:
            return 'other'
    
    def _detect_column_issues(self, col_data: pd.Series, column_name: str, inferred_type: str) -> List[DataIssue]:
        """Detect issues within a specific column"""
        issues = []
        
        # Get non-null values as strings for analysis
        non_null_values = col_data.dropna().astype(str)
        
        if len(non_null_values) == 0:
            return issues
        
        # 1. Leading/trailing spaces
        spaces_count = non_null_values.str.strip().ne(non_null_values).sum()
        if spaces_count > 0:
            examples = non_null_values[non_null_values.str.strip().ne(non_null_values)].head(3).tolist()
            issues.append(DataIssue(
                issue_type='whitespace',
                severity='warning',
                column=column_name,
                description=f'{spaces_count} values with leading/trailing spaces',
                affected_rows=spaces_count,
                examples=[f'"{val}"' for val in examples],
                suggested_fix='Trim whitespace from values'
            ))
        
        # 2. Mixed case issues (for text columns)
        if inferred_type in ['text', 'categorical']:
            case_issues = self._detect_case_issues(non_null_values)
            if case_issues:
                issues.append(case_issues)
        
        # 3. Format inconsistency
        if inferred_type == 'date':
            format_issues = self._detect_date_format_issues(non_null_values, column_name)
            if format_issues:
                issues.append(format_issues)
        
        # 4. Missing value patterns
        missing_issue = self._analyze_missing_patterns(col_data, column_name)
        if missing_issue:
            issues.append(missing_issue)
        
        return issues
    
    def _detect_case_issues(self, values: pd.Series) -> Optional[DataIssue]:
        """Detect inconsistent casing in text values"""
        
        case_types = {
            'lower': values.str.islower().sum(),
            'upper': values.str.isupper().sum(), 
            'title': values.str.istitle().sum(),
            'mixed': len(values)
        }
        
        # Calculate mixed case (not consistently one type)
        consistent_cases = case_types['lower'] + case_types['upper'] + case_types['title']
        case_types['mixed'] = len(values) - consistent_cases
        
        # If we have significant mixing of cases
        if case_types['mixed'] > len(values) * 0.2:  # >20% mixed case
            examples = []
            for case_type in ['lower', 'upper', 'title']:
                if case_types[case_type] > 0:
                    sample = values[getattr(values.str, f'is{case_type}')()].iloc[0] if case_types[case_type] > 0 else None
                    if sample:
                        examples.append(f"{case_type}: '{sample}'")
            
            return DataIssue(
                issue_type='inconsistent_case',
                severity='warning',
                column=None,
                description=f'Mixed case formatting detected',
                affected_rows=case_types['mixed'],
                examples=examples[:3],
                suggested_fix='Standardize to consistent case (Title Case recommended)'
            )
        
        return None
    
    def _detect_date_format_issues(self, values: pd.Series, column_name: str) -> Optional[DataIssue]:
        """Detect multiple date formats in the same column"""
        
        format_counts = {}
        for pattern_name, patterns in [('standard', [r'^\d{4}-\d{2}-\d{2}$']),
                                      ('us_format', [r'^\d{2}/\d{2}/\d{4}$']),
                                      ('written', [r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$'])]:
            count = 0
            for pattern in patterns:
                count += values.str.match(pattern, na=False).sum()
            if count > 0:
                format_counts[pattern_name] = count
        
        if len(format_counts) > 1:  # Multiple formats detected
            examples = []
            for fmt, count in format_counts.items():
                examples.append(f"{fmt}: {count} values")
            
            return DataIssue(
                issue_type='mixed_date_formats',
                severity='warning',
                column=column_name,
                description=f'Multiple date formats detected',
                affected_rows=sum(format_counts.values()),
                examples=examples,
                suggested_fix='Standardize all dates to YYYY-MM-DD format'
            )
        
        return None
    
    def _analyze_missing_patterns(self, col_data: pd.Series, column_name: str) -> Optional[DataIssue]:
        """Analyze patterns in missing data"""
        
        null_count = col_data.isnull().sum()
        if null_count == 0:
            return None
        
        null_percentage = (null_count / len(col_data)) * 100
        
        # Check for different types of "missing" values
        missing_variants = []
        if col_data.dtype == 'object':
            str_data = col_data.astype(str)
            variants = {
                'empty_string': (str_data == '').sum(),
                'na_string': str_data.str.lower().isin(['na', 'n/a', 'null', 'none', '-', 'tbd']).sum(),
                'whitespace': str_data.str.strip().eq('').sum()
            }
            missing_variants = [(k, v) for k, v in variants.items() if v > 0]
        
        severity = 'critical' if null_percentage > 50 else 'warning' if null_percentage > 20 else 'info'
        
        examples = [f"{null_count} null values ({null_percentage:.1f}%)"]
        examples.extend([f"{variant}: {count}" for variant, count in missing_variants])
        
        return DataIssue(
            issue_type='missing_data',
            severity=severity,
            column=column_name,
            description=f'{null_percentage:.1f}% missing data',
            affected_rows=null_count,
            examples=examples[:3],
            suggested_fix='Review missing data pattern and choose appropriate handling'
        )
    
    def _calculate_quality_score(self, structural_issues: List[DataIssue], column_profiles: List[ColumnProfile]) -> int:
        """Calculate overall data quality score (0-100)"""
        
        # Start with perfect score
        score = 100
        
        # Deduct for structural issues
        for issue in structural_issues:
            if issue.severity == 'critical':
                score -= 15
            elif issue.severity == 'warning':
                score -= 8
            elif issue.severity == 'info':
                score -= 3
        
        # Deduct for column issues
        for column in column_profiles:
            for issue in column.issues:
                if issue.severity == 'critical':
                    score -= 10
                elif issue.severity == 'warning':
                    score -= 5
                elif issue.severity == 'info':
                    score -= 2
        
        return max(0, score)
    
    def _generate_recommendations(self, structural_issues: List[DataIssue], column_profiles: List[ColumnProfile]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # High-priority structural fixes
        critical_structural = [issue for issue in structural_issues if issue.severity == 'critical']
        if critical_structural:
            recommendations.append("🚨 Fix critical structural issues first (empty rows, misplaced headers)")
        
        # Column-specific recommendations
        high_missing_cols = [col for col in column_profiles if col.null_percentage > 30]
        if high_missing_cols:
            recommendations.append(f"📊 Review high missing data in: {', '.join([col.name for col in high_missing_cols])}")
        
        # Format standardization
        format_issues = []
        for col in column_profiles:
            format_problems = [issue for issue in col.issues if issue.issue_type in ['mixed_date_formats', 'inconsistent_case']]
            if format_problems:
                format_issues.append(col.name)
        
        if format_issues:
            recommendations.append(f"🔧 Standardize formats in: {', '.join(format_issues)}")
        
        return recommendations
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a numeric value"""
        try:
            float(str(value).replace(',', '').replace('$', ''))
            return True
        except (ValueError, AttributeError):
            return False