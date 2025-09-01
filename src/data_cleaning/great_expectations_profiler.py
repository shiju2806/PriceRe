"""
Enterprise-Grade Data Profiler using Great Expectations
Production-ready data validation and profiling for reinsurance data
"""

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset
import streamlit as st
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GreatExpectationsProfiler:
    """
    Enterprise-grade data profiler using Great Expectations
    Designed for production reinsurance data quality assessment
    """
    
    def __init__(self):
        self.context = None
        self.suite_name = "pricere_data_quality_suite"
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize Great Expectations context"""
        try:
            # Create a simple in-memory context for Streamlit compatibility
            self.context = gx.get_context()
            logger.info("Great Expectations context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Great Expectations context: {e}")
            self.context = None
    
    def generate_enterprise_profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive enterprise-grade data profile
        
        Args:
            df: DataFrame to profile
            target_column: Optional target column for specialized analysis
            
        Returns:
            Comprehensive profile with business-ready insights
        """
        
        if self.context is None:
            return self._create_fallback_profile(df)
        
        try:
            # Create PandasDataset for Great Expectations
            dataset = PandasDataset(df)
            
            # Generate comprehensive expectations
            expectations = self._generate_comprehensive_expectations(dataset, target_column)
            
            # Run validation
            validation_results = dataset.validate(expectations)
            
            # Create comprehensive profile
            profile = {
                "enterprise_summary": self._create_enterprise_summary(df, validation_results),
                "data_quality_metrics": self._extract_quality_metrics(validation_results),
                "business_rules_validation": self._validate_business_rules(df, target_column),
                "regulatory_compliance": self._check_regulatory_compliance(df),
                "risk_indicators": self._identify_risk_indicators(df),
                "actionable_recommendations": self._generate_enterprise_recommendations(validation_results, df),
                "validation_results": validation_results,
                "profiling_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Enterprise profile generated with {len(expectations.expectations)} expectations")
            return profile
            
        except Exception as e:
            logger.error(f"Enterprise profiling failed: {e}")
            return self._create_fallback_profile(df)
    
    def _generate_comprehensive_expectations(self, dataset: PandasDataset, target_column: Optional[str] = None) -> ExpectationSuite:
        """Generate comprehensive expectations for insurance data"""
        
        # Create expectation suite
        suite = ExpectationSuite(expectation_suite_name=self.suite_name)
        expectations = []
        
        # Basic data integrity expectations
        expectations.append(dataset.expect_table_row_count_to_be_between(min_value=1, max_value=10000000))
        expectations.append(dataset.expect_table_column_count_to_be_between(min_value=1, max_value=1000))
        
        # Column-level expectations
        for column in dataset.columns:
            try:
                # Basic column expectations
                expectations.append(dataset.expect_column_to_exist(column))
                
                # Data type and completeness expectations
                null_percent = (dataset[column].isnull().sum() / len(dataset)) * 100
                
                if null_percent < 50:  # Only add completeness expectations for columns with reasonable data
                    expectations.append(dataset.expect_column_values_to_not_be_null(column, mostly=0.8))
                
                # Specific expectations based on column patterns
                column_lower = column.lower()
                
                # Insurance-specific validation rules
                if 'policy' in column_lower and 'number' in column_lower:
                    # Policy numbers should be unique and not null
                    expectations.append(dataset.expect_column_values_to_be_unique(column))
                    expectations.append(dataset.expect_column_values_to_not_be_null(column))
                
                elif 'premium' in column_lower or 'amount' in column_lower:
                    # Premium amounts should be positive
                    expectations.append(dataset.expect_column_values_to_be_between(column, min_value=0, mostly=0.95))
                
                elif 'age' in column_lower:
                    # Age should be within reasonable bounds
                    expectations.append(dataset.expect_column_values_to_be_between(column, min_value=0, max_value=120))
                
                elif 'date' in column_lower:
                    # Dates should be in valid format
                    expectations.append(dataset.expect_column_values_to_not_be_null(column, mostly=0.9))
                
                elif 'email' in column_lower:
                    # Email format validation
                    expectations.append(dataset.expect_column_values_to_match_regex(
                        column, 
                        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                        mostly=0.8
                    ))
                
                elif column_lower in ['gender', 'sex']:
                    # Gender values should be standardized
                    expected_values = ['M', 'F', 'Male', 'Female', 'male', 'female', 'm', 'f']
                    expectations.append(dataset.expect_column_values_to_be_in_set(column, expected_values, mostly=0.9))
                
                elif 'state' in column_lower or 'province' in column_lower:
                    # State codes validation
                    us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
                    if dataset[column].dtype == 'object':
                        unique_values = dataset[column].dropna().unique()
                        if len(unique_values) < 60:  # Likely state codes
                            expectations.append(dataset.expect_column_values_to_be_in_set(column, us_states, mostly=0.7))
                
            except Exception as e:
                logger.warning(f"Could not generate expectations for column {column}: {e}")
                continue
        
        # Target-specific expectations
        if target_column and target_column in dataset.columns:
            expectations.append(dataset.expect_column_values_to_not_be_null(target_column))
            
        return ExpectationSuite(expectation_suite_name=self.suite_name, expectations=expectations)
    
    def _create_enterprise_summary(self, df: pd.DataFrame, validation_results: Dict) -> Dict[str, Any]:
        """Create enterprise-level summary"""
        
        total_expectations = len(validation_results.get('results', []))
        successful_expectations = sum(1 for result in validation_results.get('results', []) if result.get('success', False))
        
        success_rate = (successful_expectations / total_expectations) if total_expectations > 0 else 0
        data_quality_score = success_rate * 100
        
        return {
            "dataset_shape": df.shape,
            "data_quality_score": data_quality_score,
            "overall_quality_score": data_quality_score,  # Alias for compatibility
            "compliance_score": min(100, data_quality_score + 10) if success_rate > 0.8 else data_quality_score * 0.8,
            "risk_score": max(0, 100 - data_quality_score) if success_rate < 0.9 else 10,
            "total_validations": total_expectations,
            "passed_validations": successful_expectations,
            "failed_validations": total_expectations - successful_expectations,
            "data_quality_issues": total_expectations - successful_expectations,
            "business_readiness": "Production Ready" if success_rate > 0.9 else "Needs Attention",
            "risk_level": "Low" if success_rate > 0.95 else "Medium" if success_rate > 0.8 else "High"
        }
    
    def _extract_quality_metrics(self, validation_results: Dict) -> Dict[str, Any]:
        """Extract detailed quality metrics"""
        
        metrics = {
            "completeness_issues": [],
            "uniqueness_violations": [],
            "range_violations": [],
            "format_violations": [],
            "business_rule_failures": []
        }
        
        for result in validation_results.get('results', []):
            expectation_config = result.get('expectation_config', {})
            expectation_type = expectation_config.get('expectation_type', '')
            
            if not result.get('success', True):
                # Try different ways to get the column name
                column = 'Unknown'
                if hasattr(expectation_config, 'kwargs') and 'column' in expectation_config.kwargs:
                    column = expectation_config.kwargs['column']
                elif isinstance(expectation_config, dict) and 'kwargs' in expectation_config:
                    column = expectation_config['kwargs'].get('column', 'Unknown')
                elif hasattr(result, 'expectation_config') and hasattr(result.expectation_config, 'column'):
                    column = result.expectation_config.column
                
                if 'null' in expectation_type:
                    metrics["completeness_issues"].append({
                        "column": column,
                        "expectation": expectation_type,
                        "details": result.get('result', {})
                    })
                elif 'unique' in expectation_type:
                    metrics["uniqueness_violations"].append({
                        "column": column,
                        "expectation": expectation_type,
                        "details": result.get('result', {})
                    })
                elif 'between' in expectation_type:
                    metrics["range_violations"].append({
                        "column": column,
                        "expectation": expectation_type,
                        "details": result.get('result', {})
                    })
                elif 'regex' in expectation_type or 'format' in expectation_type:
                    metrics["format_violations"].append({
                        "column": column,
                        "expectation": expectation_type,
                        "details": result.get('result', {})
                    })
                else:
                    metrics["business_rule_failures"].append({
                        "column": column,
                        "expectation": expectation_type,
                        "details": result.get('result', {})
                    })
        
        return metrics
    
    def _validate_business_rules(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Validate insurance-specific business rules"""
        
        business_rules = {
            "insurance_data_rules": [],
            "financial_data_rules": [],
            "regulatory_rules": []
        }
        
        # Insurance-specific business rules
        columns = [col.lower() for col in df.columns]
        
        # Check for required insurance fields
        required_fields = ['policy', 'premium', 'coverage', 'insured']
        for field in required_fields:
            field_present = any(field in col for col in columns)
            business_rules["insurance_data_rules"].append({
                "rule": f"Required field '{field}' present",
                "status": "PASS" if field_present else "FAIL",
                "severity": "HIGH" if not field_present else "INFO"
            })
        
        # Financial validation rules
        financial_columns = [col for col in df.columns if any(term in col.lower() for term in ['premium', 'amount', 'value', 'cost', 'price'])]
        
        for col in financial_columns:
            if df[col].dtype in ['int64', 'float64']:
                negative_count = (df[col] < 0).sum()
                business_rules["financial_data_rules"].append({
                    "rule": f"No negative values in {col}",
                    "status": "PASS" if negative_count == 0 else "WARN",
                    "details": f"{negative_count} negative values found" if negative_count > 0 else "All values positive"
                })
        
        return business_rules
    
    def _check_regulatory_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check regulatory compliance requirements"""
        
        compliance = {
            "data_privacy": [],
            "reporting_requirements": [],
            "audit_trail": []
        }
        
        columns = [col.lower() for col in df.columns]
        
        # Data privacy checks
        sensitive_fields = ['ssn', 'social_security', 'tax_id', 'phone', 'email', 'address']
        for field in sensitive_fields:
            field_present = any(field in col for col in columns)
            if field_present:
                compliance["data_privacy"].append({
                    "requirement": f"PII field '{field}' detected",
                    "action_required": "Ensure proper encryption and access controls",
                    "severity": "HIGH"
                })
        
        # Reporting requirements
        if len(df) > 1000:
            compliance["reporting_requirements"].append({
                "requirement": "Large dataset reporting",
                "status": "Dataset qualifies for enhanced reporting requirements",
                "action_required": "Ensure proper data governance procedures"
            })
        
        return compliance
    
    def _identify_risk_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify data quality risk indicators"""
        
        risks = []
        
        # High missing data risk
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 30:
                risks.append({
                    "risk_type": "Data Completeness",
                    "severity": "HIGH" if missing_pct > 50 else "MEDIUM",
                    "description": f"Column '{col}' has {missing_pct:.1f}% missing data",
                    "business_impact": "May affect pricing accuracy and regulatory reporting",
                    "recommended_action": "Investigate data collection process and implement validation rules"
                })
        
        # Data freshness risk
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df_dates = pd.to_datetime(df[col], errors='coerce')
                if not df_dates.isnull().all():
                    latest_date = df_dates.max()
                    days_old = (datetime.now() - latest_date).days
                    if days_old > 90:
                        risks.append({
                            "risk_type": "Data Freshness",
                            "severity": "MEDIUM",
                            "description": f"Latest data in '{col}' is {days_old} days old",
                            "business_impact": "Outdated data may affect pricing decisions",
                            "recommended_action": "Update data refresh procedures"
                        })
            except:
                pass
        
        return risks
    
    def _generate_enterprise_recommendations(self, validation_results: Dict, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate enterprise-level recommendations"""
        
        recommendations = []
        
        # Analyze validation failures
        for result in validation_results.get('results', []):
            if not result.get('success', True):
                expectation_config = result.get('expectation_config', {})
                expectation_type = expectation_config.get('expectation_type', '')
                
                # Try different ways to get the column name
                column = 'Unknown'
                if hasattr(expectation_config, 'kwargs') and 'column' in expectation_config.kwargs:
                    column = expectation_config.kwargs['column']
                elif isinstance(expectation_config, dict) and 'kwargs' in expectation_config:
                    column = expectation_config['kwargs'].get('column', 'Unknown')
                elif hasattr(result, 'expectation_config') and hasattr(result.expectation_config, 'column'):
                    column = result.expectation_config.column
                
                if 'null' in expectation_type:
                    recommendations.append({
                        "category": "Data Completeness",
                        "priority": "HIGH",
                        "issue": f"Missing data in column '{column}'",
                        "recommendation": "Implement data validation at source",
                        "business_benefit": "Improved pricing accuracy and regulatory compliance",
                        "estimated_effort": "Medium",
                        "action": f"implement_null_validation_{column}"
                    })
                
                elif 'unique' in expectation_type:
                    recommendations.append({
                        "category": "Data Uniqueness",
                        "priority": "HIGH",
                        "issue": f"Duplicate values in column '{column}'",
                        "recommendation": "Implement uniqueness constraints",
                        "business_benefit": "Prevents data quality issues and processing errors",
                        "estimated_effort": "Low",
                        "action": f"implement_uniqueness_{column}"
                    })
        
        # General recommendations based on dataset characteristics
        if len(df) > 10000:
            recommendations.append({
                "category": "Performance",
                "priority": "MEDIUM",
                "issue": "Large dataset detected",
                "recommendation": "Consider implementing data partitioning strategies",
                "business_benefit": "Improved processing performance and scalability",
                "estimated_effort": "High",
                "action": "implement_data_partitioning"
            })
        
        return recommendations
    
    def _create_fallback_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic profile when Great Expectations fails"""
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        return {
            "enterprise_summary": {
                "dataset_shape": df.shape,
                "data_quality_score": completeness,
                "business_readiness": "Needs Assessment",
                "risk_level": "Unknown"
            },
            "data_quality_metrics": {"note": "Limited analysis - Great Expectations unavailable"},
            "business_rules_validation": {"note": "Business rule validation unavailable"},
            "regulatory_compliance": {"note": "Compliance checking unavailable"},
            "risk_indicators": [],
            "actionable_recommendations": [{
                "category": "System",
                "priority": "HIGH",
                "issue": "Great Expectations profiling unavailable",
                "recommendation": "Install and configure Great Expectations properly",
                "action": "configure_great_expectations"
            }],
            "validation_results": None,
            "profiling_timestamp": datetime.now().isoformat()
        }
    
    def display_enterprise_results(self, profile: Dict[str, Any]) -> None:
        """Display enterprise-grade results in Streamlit"""
        
        st.markdown("## 🏢 Enterprise Data Quality Assessment")
        
        # Executive summary
        summary = profile["enterprise_summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{summary['data_quality_score']:.1f}%")
        
        with col2:
            st.metric("Business Readiness", summary['business_readiness'])
        
        with col3:
            risk_color = "🟢" if summary['risk_level'] == 'Low' else "🟡" if summary['risk_level'] == 'Medium' else "🔴"
            st.metric("Risk Level", f"{risk_color} {summary['risk_level']}")
        
        with col4:
            st.metric("Validations", f"{summary.get('passed_validations', 0)}/{summary.get('total_validations', 0)}")
        
        # Risk indicators
        risks = profile.get("risk_indicators", [])
        if risks:
            st.markdown("### ⚠️ Risk Indicators")
            for risk in risks[:3]:  # Show top 3 risks
                severity_color = "🔴" if risk['severity'] == 'HIGH' else "🟡"
                st.warning(f"{severity_color} **{risk['risk_type']}**: {risk['description']}")
                st.caption(f"Business Impact: {risk['business_impact']}")
        
        # Enterprise recommendations
        recommendations = profile.get("actionable_recommendations", [])
        if recommendations:
            st.markdown("### 💼 Enterprise Recommendations")
            
            high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
            
            for rec in high_priority[:3]:  # Show top 3 high-priority recommendations
                st.info(f"**{rec['category']}**: {rec['recommendation']}")
                st.caption(f"Business Benefit: {rec.get('business_benefit', 'Not specified')}")

# Global instance
great_expectations_profiler = GreatExpectationsProfiler()