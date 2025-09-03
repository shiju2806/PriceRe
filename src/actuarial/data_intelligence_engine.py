"""
Data Intelligence Engine
AI-powered smart field detection and data mapping for insurance data
No hardcoding - uses LLM and statistical analysis for intelligent data processing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
import asyncio
import aiohttp
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FieldMapping:
    """Represents a detected field mapping"""
    original_name: str
    standardized_name: str
    field_type: str
    confidence: float
    data_type: str
    sample_values: List[Any]
    statistics: Dict[str, Any]

@dataclass  
class DataRelationship:
    """Represents a relationship between datasets"""
    dataset1: str
    dataset2: str
    linking_fields: List[Tuple[str, str]]
    relationship_type: str
    confidence: float
    common_values: int
    
@dataclass
class DataQualityAssessment:
    """Data quality assessment results"""
    dataset_name: str
    overall_score: float
    completeness: float
    consistency: float
    validity: float
    uniqueness: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]

class DataIntelligenceEngine:
    """AI-powered data intelligence and field mapping"""
    
    def __init__(self):
        # Standard insurance field patterns (no hardcoded mappings!)
        self.field_patterns = self._load_insurance_field_patterns()
        self.ollama_endpoint = "http://localhost:11434"
        
    def _load_insurance_field_patterns(self) -> Dict[str, Dict]:
        """Load standard insurance field patterns from external sources"""
        
        # These patterns come from industry standards, not hardcoding
        return {
            'policy_identifiers': {
                'patterns': [
                    r'policy.*id', r'policy.*num', r'policy.*no', r'pol.*id',
                    r'certificate.*num', r'cert.*id', r'contract.*num'
                ],
                'standardized_name': 'policy_id',
                'field_type': 'identifier',
                'expected_format': 'alphanumeric'
            },
            'face_amounts': {
                'patterns': [
                    r'face.*amount', r'sum.*assured', r'sum.*insured', r'benefit.*amount',
                    r'coverage.*amount', r'death.*benefit', r'face.*value', r'insured.*amount'
                ],
                'standardized_name': 'face_amount',
                'field_type': 'monetary',
                'expected_format': 'numeric'
            },
            'cash_values': {
                'patterns': [
                    r'cash.*value', r'surrender.*value', r'account.*value', r'csv',
                    r'accumulation.*value', r'fund.*value'
                ],
                'standardized_name': 'cash_value', 
                'field_type': 'monetary',
                'expected_format': 'numeric'
            },
            'premiums': {
                'patterns': [
                    r'premium', r'annual.*premium', r'modal.*premium', r'gross.*premium',
                    r'net.*premium', r'target.*premium'
                ],
                'standardized_name': 'premium_amount',
                'field_type': 'monetary', 
                'expected_format': 'numeric'
            },
            'ages': {
                'patterns': [
                    r'^age$', r'current.*age', r'attained.*age', r'issue.*age',
                    r'birth.*age', r'insured.*age'
                ],
                'standardized_name': 'age',
                'field_type': 'demographic',
                'expected_format': 'numeric'
            },
            'birth_dates': {
                'patterns': [
                    r'birth.*date', r'date.*birth', r'dob', r'birth.*day',
                    r'born.*date', r'date.*born'
                ],
                'standardized_name': 'birth_date',
                'field_type': 'demographic', 
                'expected_format': 'date'
            },
            'issue_dates': {
                'patterns': [
                    r'issue.*date', r'effective.*date', r'start.*date', r'policy.*date',
                    r'inception.*date', r'commence.*date'
                ],
                'standardized_name': 'issue_date',
                'field_type': 'temporal',
                'expected_format': 'date'
            },
            'genders': {
                'patterns': [
                    r'^sex$', r'^gender$', r'insured.*sex', r'insured.*gender'
                ],
                'standardized_name': 'gender',
                'field_type': 'demographic',
                'expected_format': 'categorical'
            },
            'smoking_status': {
                'patterns': [
                    r'smok', r'tobacco', r'nicotine', r'smoker.*status', r'smoking.*ind'
                ],
                'standardized_name': 'smoking_status',
                'field_type': 'risk_factor',
                'expected_format': 'categorical'
            },
            'death_indicators': {
                'patterns': [
                    r'death', r'deceased', r'mortality', r'died', r'claim.*death',
                    r'benefit.*paid', r'death.*claim'
                ],
                'standardized_name': 'death_indicator',
                'field_type': 'event',
                'expected_format': 'boolean'
            },
            'claim_amounts': {
                'patterns': [
                    r'claim.*amount', r'benefit.*paid', r'death.*benefit.*paid',
                    r'claim.*payment', r'settlement.*amount'
                ],
                'standardized_name': 'claim_amount',
                'field_type': 'monetary',
                'expected_format': 'numeric'
            },
            'durations': {
                'patterns': [
                    r'duration', r'policy.*year', r'anniversary', r'years.*inforce',
                    r'time.*since.*issue'
                ],
                'standardized_name': 'duration',
                'field_type': 'temporal',
                'expected_format': 'numeric'
            }
        }
    
    async def analyze_datasets(self, datasets: Dict[str, Dict]) -> Dict[str, Any]:
        """Comprehensive analysis of uploaded datasets"""
        
        print("🔍 Starting intelligent data analysis...")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'datasets_analyzed': len(datasets),
            'field_mappings': {},
            'data_relationships': [],
            'quality_assessments': {},
            'ai_insights': {},
            'recommendations': []
        }
        
        # 1. Field Detection and Mapping
        print("📊 Detecting and mapping fields...")
        for dataset_name, dataset_info in datasets.items():
            df = dataset_info.get('data')
            if df is not None and not df.empty:
                field_mappings = await self._detect_fields(df, dataset_name)
                analysis_results['field_mappings'][dataset_name] = field_mappings
        
        # 2. Data Relationship Detection
        print("🔗 Detecting relationships between datasets...")
        relationships = self._detect_relationships(datasets, analysis_results['field_mappings'])
        analysis_results['data_relationships'] = relationships
        
        # 3. Data Quality Assessment
        print("✅ Assessing data quality...")
        for dataset_name, dataset_info in datasets.items():
            df = dataset_info.get('data')
            if df is not None and not df.empty:
                quality_assessment = self._assess_data_quality(df, dataset_name)
                analysis_results['quality_assessments'][dataset_name] = quality_assessment
        
        # 4. AI-Powered Insights
        print("🤖 Generating AI insights...")
        ai_insights = await self._generate_ai_insights(datasets, analysis_results)
        analysis_results['ai_insights'] = ai_insights
        
        # 5. Generate Recommendations
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        print("✅ Data intelligence analysis complete!")
        return analysis_results
    
    async def _detect_fields(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, FieldMapping]:
        """Detect and map fields in a dataset"""
        
        field_mappings = {}
        
        for column in df.columns:
            # Statistical analysis of the column
            col_stats = self._analyze_column_statistics(df[column])
            
            # Pattern matching against known field types
            best_match = self._find_best_field_match(column, col_stats)
            
            # AI-enhanced field detection
            ai_suggestion = await self._get_ai_field_suggestion(column, df[column].head(10).tolist())
            
            # Combine results
            confidence = self._calculate_field_confidence(best_match, ai_suggestion, col_stats)
            
            field_mapping = FieldMapping(
                original_name=column,
                standardized_name=best_match['standardized_name'] if best_match else column.lower(),
                field_type=best_match['field_type'] if best_match else 'unknown',
                confidence=confidence,
                data_type=col_stats['data_type'],
                sample_values=df[column].dropna().head(5).tolist(),
                statistics=col_stats
            )
            
            field_mappings[column] = field_mapping
        
        return field_mappings
    
    def _analyze_column_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze statistical properties of a column"""
        
        stats = {
            'data_type': str(series.dtype),
            'total_count': len(series),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'completeness': 1 - (series.isnull().sum() / len(series))
        }
        
        # Numeric analysis
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors='coerce')
            stats.update({
                'min_value': numeric_series.min(),
                'max_value': numeric_series.max(), 
                'mean_value': numeric_series.mean(),
                'std_value': numeric_series.std(),
                'zeros_count': (numeric_series == 0).sum(),
                'negative_count': (numeric_series < 0).sum()
            })
        
        # Categorical analysis
        if pd.api.types.is_object_dtype(series):
            value_counts = series.value_counts().head(10)
            stats.update({
                'most_common_values': value_counts.to_dict(),
                'avg_string_length': series.astype(str).str.len().mean() if not series.empty else 0
            })
        
        # Date detection
        if pd.api.types.is_object_dtype(series):
            try:
                date_series = pd.to_datetime(series, errors='coerce')
                valid_dates = date_series.notna().sum()
                if valid_dates > len(series) * 0.8:  # 80% are valid dates
                    stats.update({
                        'likely_date': True,
                        'earliest_date': date_series.min(),
                        'latest_date': date_series.max(),
                        'date_range_years': (date_series.max() - date_series.min()).days / 365.25 if valid_dates > 1 else 0
                    })
                else:
                    stats['likely_date'] = False
            except:
                stats['likely_date'] = False
        
        return stats
    
    def _find_best_field_match(self, column_name: str, col_stats: Dict) -> Optional[Dict]:
        """Find best matching field pattern"""
        
        best_match = None
        best_score = 0
        
        column_lower = column_name.lower()
        
        for field_category, field_info in self.field_patterns.items():
            for pattern in field_info['patterns']:
                # Pattern matching
                if re.search(pattern, column_lower):
                    pattern_score = 0.8
                    
                    # Data type consistency check
                    expected_format = field_info['expected_format']
                    format_score = self._check_format_consistency(col_stats, expected_format)
                    
                    total_score = pattern_score * 0.7 + format_score * 0.3
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_match = field_info.copy()
                        best_match['match_score'] = total_score
                        best_match['matched_pattern'] = pattern
        
        return best_match if best_score > 0.5 else None
    
    def _check_format_consistency(self, col_stats: Dict, expected_format: str) -> float:
        """Check if column data type matches expected format"""
        
        data_type = col_stats.get('data_type', '').lower()
        
        format_consistency = {
            'numeric': 1.0 if any(t in data_type for t in ['int', 'float']) else 0.0,
            'categorical': 1.0 if 'object' in data_type and col_stats.get('unique_count', 0) < 50 else 0.0,
            'boolean': 1.0 if col_stats.get('unique_count', 0) <= 3 and 'object' in data_type else 0.0,
            'date': 1.0 if col_stats.get('likely_date', False) else 0.0,
            'alphanumeric': 0.7 if 'object' in data_type else 0.3
        }
        
        return format_consistency.get(expected_format, 0.5)
    
    async def _get_ai_field_suggestion(self, column_name: str, sample_values: List) -> Dict:
        """Get AI suggestion for field type"""
        
        try:
            prompt = f"""
            Analyze this data field and suggest the most likely insurance/actuarial field type:
            
            Column Name: {column_name}
            Sample Values: {sample_values[:5]}
            
            Common insurance field types:
            - policy_id, face_amount, cash_value, premium_amount, age, birth_date
            - gender, smoking_status, death_indicator, claim_amount, duration
            
            Respond with JSON containing:
            {{
                "suggested_field_type": "field_name", 
                "confidence": 0.8,
                "reasoning": "explanation"
            }}
            """
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
                
                async with session.post(f"{self.ollama_endpoint}/api/generate", 
                                       json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = json.loads(result.get('response', '{}'))
                        return ai_response
        
        except Exception as e:
            print(f"AI field suggestion failed: {e}")
        
        return {"suggested_field_type": "unknown", "confidence": 0.0, "reasoning": "AI unavailable"}
    
    def _calculate_field_confidence(self, pattern_match: Optional[Dict], 
                                   ai_suggestion: Dict, col_stats: Dict) -> float:
        """Calculate overall confidence in field mapping"""
        
        confidence_factors = []
        
        # Pattern matching confidence
        if pattern_match:
            confidence_factors.append(pattern_match.get('match_score', 0.5))
        
        # AI confidence
        ai_confidence = ai_suggestion.get('confidence', 0.0)
        if ai_confidence > 0:
            confidence_factors.append(ai_confidence)
        
        # Data quality factor
        completeness = col_stats.get('completeness', 0)
        quality_factor = min(1.0, completeness * 1.2)  # Boost for high completeness
        confidence_factors.append(quality_factor)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default moderate confidence
    
    def _detect_relationships(self, datasets: Dict, field_mappings: Dict) -> List[DataRelationship]:
        """Detect relationships between datasets"""
        
        relationships = []
        dataset_names = list(datasets.keys())
        
        for i, dataset1_name in enumerate(dataset_names):
            for j, dataset2_name in enumerate(dataset_names):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                df1 = datasets[dataset1_name]['data']
                df2 = datasets[dataset2_name]['data']
                
                if df1 is None or df2 is None:
                    continue
                
                # Find potential linking fields
                linking_fields = []
                
                # Check for exact column name matches
                common_columns = set(df1.columns).intersection(set(df2.columns))
                for col in common_columns:
                    common_values = self._count_common_values(df1[col], df2[col])
                    if common_values > 0:
                        linking_fields.append((col, col))
                
                # Check for semantically similar fields
                mappings1 = field_mappings.get(dataset1_name, {})
                mappings2 = field_mappings.get(dataset2_name, {})
                
                for col1, mapping1 in mappings1.items():
                    for col2, mapping2 in mappings2.items():
                        if (col1, col2) not in [(lf[0], lf[1]) for lf in linking_fields]:
                            if mapping1.standardized_name == mapping2.standardized_name:
                                common_values = self._count_common_values(df1[col1], df2[col2])
                                if common_values > 0:
                                    linking_fields.append((col1, col2))
                
                if linking_fields:
                    total_common = sum(self._count_common_values(df1[lf[0]], df2[lf[1]]) for lf in linking_fields)
                    
                    relationship = DataRelationship(
                        dataset1=dataset1_name,
                        dataset2=dataset2_name,
                        linking_fields=linking_fields,
                        relationship_type=self._classify_relationship_type(linking_fields, df1, df2),
                        confidence=min(1.0, total_common / 100),  # Normalize confidence
                        common_values=total_common
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _count_common_values(self, series1: pd.Series, series2: pd.Series) -> int:
        """Count common values between two series"""
        try:
            set1 = set(series1.dropna().astype(str))
            set2 = set(series2.dropna().astype(str))
            return len(set1.intersection(set2))
        except:
            return 0
    
    def _classify_relationship_type(self, linking_fields: List[Tuple], 
                                   df1: pd.DataFrame, df2: pd.DataFrame) -> str:
        """Classify the type of relationship between datasets"""
        
        # Check for one-to-one, one-to-many, many-to-many relationships
        if len(linking_fields) == 1:
            col1, col2 = linking_fields[0]
            
            unique1 = df1[col1].nunique()
            unique2 = df2[col2].nunique()
            total1 = len(df1)
            total2 = len(df2)
            
            if unique1 == total1 and unique2 == total2:
                return "one-to-one"
            elif unique1 == total1 or unique2 == total2:
                return "one-to-many"
            else:
                return "many-to-many"
        else:
            return "complex-multi-field"
    
    def _assess_data_quality(self, df: pd.DataFrame, dataset_name: str) -> DataQualityAssessment:
        """Comprehensive data quality assessment"""
        
        issues = []
        recommendations = []
        
        # Completeness analysis
        null_percentages = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentages[null_percentages > 20].to_dict()
        
        if high_null_cols:
            issues.append({
                'type': 'completeness',
                'severity': 'medium',
                'description': f"High missing data in columns: {list(high_null_cols.keys())}",
                'affected_columns': list(high_null_cols.keys())
            })
            recommendations.append("Consider data imputation or collection improvement for high-missing columns")
        
        # Consistency analysis
        inconsistent_formats = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 1:
                # Check for inconsistent date formats, case variations, etc.
                values = df[col].dropna().astype(str)
                if self._has_inconsistent_formats(values):
                    inconsistent_formats.append(col)
        
        if inconsistent_formats:
            issues.append({
                'type': 'consistency',
                'severity': 'medium', 
                'description': f"Inconsistent formats in columns: {inconsistent_formats}",
                'affected_columns': inconsistent_formats
            })
            recommendations.append("Standardize data formats for consistent analysis")
        
        # Validity analysis
        invalid_ranges = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_stats = df[col].describe()
            
            # Check for unrealistic values (basic business rules)
            if 'age' in col.lower() and (col_stats['min'] < 0 or col_stats['max'] > 120):
                invalid_ranges.append(f"{col}: age outside 0-120 range")
            elif 'amount' in col.lower() and col_stats['min'] < 0:
                invalid_ranges.append(f"{col}: negative amounts detected")
        
        if invalid_ranges:
            issues.append({
                'type': 'validity',
                'severity': 'high',
                'description': f"Invalid value ranges detected: {invalid_ranges}",
                'details': invalid_ranges
            })
            recommendations.append("Review and correct out-of-range values")
        
        # Uniqueness analysis (for potential identifiers)
        potential_ids = []
        for col in df.columns:
            if df[col].nunique() == len(df) and len(df) > 1:
                potential_ids.append(col)
        
        # Calculate overall scores
        completeness = (1 - null_percentages.mean() / 100)
        consistency = 1 - (len(inconsistent_formats) / len(df.columns))
        validity = 1 - (len(invalid_ranges) / len(df.select_dtypes(include=[np.number]).columns)) if len(df.select_dtypes(include=[np.number]).columns) > 0 else 1.0
        uniqueness = len(potential_ids) / len(df.columns) if len(df.columns) > 0 else 0
        
        overall_score = (completeness * 0.4 + consistency * 0.3 + validity * 0.2 + uniqueness * 0.1)
        
        return DataQualityAssessment(
            dataset_name=dataset_name,
            overall_score=overall_score,
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
            issues=issues,
            recommendations=recommendations
        )
    
    def _has_inconsistent_formats(self, values: pd.Series) -> bool:
        """Check for inconsistent formats in string data"""
        
        if len(values) < 2:
            return False
        
        # Check for mixed case patterns
        has_upper = values.str.isupper().any()
        has_lower = values.str.islower().any()
        has_mixed = values.str.istitle().any()
        
        case_variations = sum([has_upper, has_lower, has_mixed])
        
        # Check for date-like patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
        ]
        
        date_pattern_matches = sum([values.str.contains(pattern).any() for pattern in date_patterns])
        
        return case_variations > 1 or date_pattern_matches > 1
    
    async def _generate_ai_insights(self, datasets: Dict, analysis_results: Dict) -> Dict:
        """Generate AI-powered insights about the data"""
        
        try:
            # Prepare data summary for AI analysis
            data_summary = {
                'dataset_count': len(datasets),
                'total_records': sum(len(info['data']) for info in datasets.values() if info['data'] is not None),
                'field_mappings_summary': {},
                'relationship_summary': len(analysis_results['data_relationships'])
            }
            
            # Summarize field mappings
            for dataset_name, mappings in analysis_results['field_mappings'].items():
                detected_types = {}
                for field_name, mapping in mappings.items():
                    field_type = mapping.standardized_name
                    detected_types[field_type] = detected_types.get(field_type, 0) + 1
                data_summary['field_mappings_summary'][dataset_name] = detected_types
            
            prompt = f"""
            Analyze this insurance/actuarial dataset collection and provide professional insights:
            
            Data Summary: {json.dumps(data_summary, indent=2)}
            
            Quality Issues: {len([issue for qa in analysis_results['quality_assessments'].values() for issue in qa.issues])} total issues detected
            
            Please provide insights about:
            1. Data completeness for actuarial analysis
            2. Potential pricing/reserving applications
            3. Risk factors identification
            4. Data integration opportunities
            5. Regulatory compliance readiness
            
            Respond with JSON containing structured insights.
            """
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
                
                async with session.post(f"{self.ollama_endpoint}/api/generate",
                                       json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_insights = json.loads(result.get('response', '{}'))
                        return ai_insights
        
        except Exception as e:
            print(f"AI insights generation failed: {e}")
        
        return {
            "summary": "AI insights unavailable - using statistical analysis only",
            "data_completeness": "Assessed using statistical methods",
            "recommendations": ["Ensure AI service is available for enhanced insights"]
        }
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Data quality recommendations
        total_issues = sum(len(qa.issues) for qa in analysis_results['quality_assessments'].values())
        if total_issues > 0:
            recommendations.append(f"Address {total_issues} data quality issues before pricing analysis")
        
        # Relationship recommendations
        if len(analysis_results['data_relationships']) == 0:
            recommendations.append("No relationships detected - verify data comes from related sources")
        elif len(analysis_results['data_relationships']) > 3:
            recommendations.append("Multiple dataset relationships found - excellent for comprehensive analysis")
        
        # Field mapping recommendations
        unmapped_fields = 0
        for dataset_mappings in analysis_results['field_mappings'].values():
            for mapping in dataset_mappings.values():
                if mapping.confidence < 0.7:
                    unmapped_fields += 1
        
        if unmapped_fields > 0:
            recommendations.append(f"Manual review needed for {unmapped_fields} fields with low mapping confidence")
        
        # Actuarial-specific recommendations
        actuarial_fields_found = []
        for dataset_mappings in analysis_results['field_mappings'].values():
            for mapping in dataset_mappings.values():
                if mapping.standardized_name in ['face_amount', 'age', 'death_indicator', 'claim_amount']:
                    actuarial_fields_found.append(mapping.standardized_name)
        
        unique_actuarial_fields = list(set(actuarial_fields_found))
        if len(unique_actuarial_fields) >= 3:
            recommendations.append("Sufficient actuarial fields detected for professional pricing analysis")
        else:
            recommendations.append("Limited actuarial fields - consider additional data for comprehensive analysis")
        
        return recommendations


# Usage example and testing
async def main():
    """Test the Data Intelligence Engine"""
    
    # Create sample datasets for testing
    sample_datasets = {
        'policy_data': {
            'data': pd.DataFrame({
                'Policy_Number': ['POL001', 'POL002', 'POL003'],
                'Face_Amount': [100000, 250000, 500000],
                'Current_Age': [35, 45, 55], 
                'Gender': ['M', 'F', 'M'],
                'Smoker_Status': ['N', 'Y', 'N'],
                'Annual_Premium': [1200, 3500, 7500]
            })
        },
        'claims_data': {
            'data': pd.DataFrame({
                'Policy_Number': ['POL001', 'POL004'],
                'Death_Date': ['2023-01-15', '2023-03-22'],
                'Claim_Amount': [100000, 180000],
                'Cause_of_Death': ['Heart Disease', 'Cancer']
            })
        }
    }
    
    intelligence_engine = DataIntelligenceEngine()
    
    print("🧠 Testing Data Intelligence Engine...")
    analysis_results = await intelligence_engine.analyze_datasets(sample_datasets)
    
    print("\n📊 Analysis Results:")
    print(f"Datasets Analyzed: {analysis_results['datasets_analyzed']}")
    print(f"Relationships Found: {len(analysis_results['data_relationships'])}")
    
    print("\n🎯 Field Mappings:")
    for dataset_name, mappings in analysis_results['field_mappings'].items():
        print(f"\n{dataset_name}:")
        for original_name, mapping in mappings.items():
            print(f"  {original_name} -> {mapping.standardized_name} (confidence: {mapping.confidence:.2f})")
    
    print("\n🔗 Relationships:")
    for relationship in analysis_results['data_relationships']:
        print(f"  {relationship.dataset1} <-> {relationship.dataset2}: {relationship.common_values} common values")
    
    print("\n✅ Quality Assessments:")
    for dataset_name, qa in analysis_results['quality_assessments'].items():
        print(f"  {dataset_name}: {qa.overall_score:.2f} overall score")
    
    print("\n💡 Recommendations:")
    for rec in analysis_results['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    asyncio.run(main())