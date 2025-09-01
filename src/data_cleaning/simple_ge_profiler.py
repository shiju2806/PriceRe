"""
Simplified Great Expectations Profiler for MVP testing
"""

import pandas as pd
import great_expectations as gx
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGreatExpectationsProfiler:
    """Simplified Great Expectations profiler for testing"""
    
    def __init__(self):
        try:
            self.context = gx.get_context()
            logger.info("Great Expectations context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Great Expectations context: {e}")
            self.context = None
    
    def generate_enterprise_profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Generate simplified enterprise profile"""
        
        if self.context is None:
            return self._create_fallback_profile(df, target_column)
        
        try:
            # Create dataset
            dataset = self.context.sources.pandas_default.read_dataframe(df)
            
            # Simple expectations
            expectations = []
            
            # Basic completeness checks for each column
            for column in df.columns:
                try:
                    expectations.append(dataset.expect_column_to_exist(column))
                    if df[column].dtype in ['object', 'string']:
                        expectations.append(dataset.expect_column_values_to_not_be_null(column, mostly=0.8))
                    elif df[column].dtype in ['int64', 'float64']:
                        expectations.append(dataset.expect_column_values_to_not_be_null(column, mostly=0.9))
                except Exception as e:
                    logger.warning(f"Could not create expectations for {column}: {e}")
                    continue
            
            # Validate expectations
            validation_results = dataset.validate(expectations)
            
            # Create simplified profile
            total_expectations = len(expectations)
            passed_expectations = sum(1 for result in validation_results['results'] if result.get('success', False))
            success_rate = passed_expectations / total_expectations if total_expectations > 0 else 0
            
            profile = {
                "enterprise_summary": {
                    "dataset_shape": df.shape,
                    "data_quality_score": success_rate * 100,
                    "overall_quality_score": success_rate * 100,
                    "compliance_score": min(100, success_rate * 100 + 10) if success_rate > 0.8 else success_rate * 80,
                    "risk_score": max(0, 100 - success_rate * 100) if success_rate < 0.9 else 10,
                    "total_validations": total_expectations,
                    "passed_validations": passed_expectations,
                    "failed_validations": total_expectations - passed_expectations,
                    "data_quality_issues": total_expectations - passed_expectations,
                    "business_readiness": "Production Ready" if success_rate > 0.9 else "Needs Attention",
                    "risk_level": "Low" if success_rate > 0.95 else "Medium" if success_rate > 0.8 else "High"
                },
                "business_rules_validation": self._simple_business_validation(df, target_column),
                "data_quality_metrics": self._extract_simple_metrics(validation_results),
                "regulatory_compliance": self._simple_compliance_check(df),
                "risk_indicators": self._identify_simple_risks(df, success_rate)
            }
            
            logger.info(f"Simple GE profile generated: {success_rate:.2%} success rate")
            return profile
            
        except Exception as e:
            logger.error(f"Simple GE profiling failed: {e}")
            return self._create_fallback_profile(df, target_column)
    
    def _simple_business_validation(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Simple business rules validation"""
        
        columns = [col.lower() for col in df.columns]
        
        # Determine data domain
        if any('policy' in col for col in columns):
            data_domain = 'insurance_policy'
        elif any('claim' in col for col in columns):
            data_domain = 'claims'
        elif any('mortality' in col for col in columns):
            data_domain = 'mortality'
        elif any('premium' in col or 'amount' in col for col in columns):
            data_domain = 'financial'
        else:
            data_domain = 'general'
        
        return {
            "data_domain": data_domain,
            "business_context": f"Detected as {data_domain.replace('_', ' ')} data",
            "validation_results": {
                "has_key_columns": {"passed": len(columns) >= 3, "message": f"Found {len(columns)} columns"}
            }
        }
    
    def _extract_simple_metrics(self, validation_results: Dict) -> Dict[str, Any]:
        """Extract simple quality metrics"""
        
        passed = []
        failed = []
        
        for result in validation_results.get('results', []):
            expectation_type = result.get('expectation_config', {}).get('expectation_type', 'unknown')
            
            if result.get('success', False):
                passed.append({"expectation_type": expectation_type})
            else:
                failed.append({
                    "expectation_type": expectation_type,
                    "observed_value": str(result.get('result', {}))
                })
        
        return {
            "passed_expectations": passed,
            "failed_expectations": failed
        }
    
    def _simple_compliance_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple compliance checking"""
        
        compliance_items = []
        
        # Check for PII columns (basic check)
        pii_columns = [col for col in df.columns if any(pii in col.lower() for pii in ['ssn', 'social', 'license', 'id'])]
        
        if pii_columns:
            compliance_items.append({
                "requirement": "PII Data Protection",
                "compliant": False,
                "issue": f"Potential PII columns detected: {', '.join(pii_columns)}"
            })
        else:
            compliance_items.append({
                "requirement": "PII Data Protection", 
                "compliant": True,
                "issue": None
            })
        
        return {
            "compliance_status": compliance_items
        }
    
    def _identify_simple_risks(self, df: pd.DataFrame, success_rate: float) -> List[Dict[str, Any]]:
        """Identify simple risk indicators"""
        
        risks = []
        
        if success_rate < 0.8:
            risks.append({
                "type": "data_quality_risk",
                "severity": "high",
                "description": f"Low data quality score: {success_rate:.1%}",
                "recommendation": "Implement comprehensive data validation"
            })
        
        # Check for high missing data
        missing_percentage = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_percentage > 0.2:
            risks.append({
                "type": "missing_data_risk",
                "severity": "medium",
                "description": f"High missing data rate: {missing_percentage:.1%}",
                "recommendation": "Review data collection processes"
            })
        
        return risks
    
    def _create_fallback_profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Create fallback profile when GE fails"""
        
        missing_percentage = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        quality_score = (1 - missing_percentage) * 100
        
        return {
            "enterprise_summary": {
                "dataset_shape": df.shape,
                "data_quality_score": quality_score,
                "overall_quality_score": quality_score,
                "compliance_score": quality_score * 0.9,
                "risk_score": 100 - quality_score,
                "total_validations": 1,
                "passed_validations": 1 if quality_score > 50 else 0,
                "failed_validations": 0 if quality_score > 50 else 1,
                "data_quality_issues": 0 if quality_score > 80 else 1,
                "business_readiness": "Needs Attention",
                "risk_level": "High" if quality_score < 60 else "Medium"
            },
            "business_rules_validation": self._simple_business_validation(df, target_column),
            "data_quality_metrics": {
                "passed_expectations": [{"expectation_type": "Basic validation"}] if quality_score > 50 else [],
                "failed_expectations": [{"expectation_type": "Quality threshold", "observed_value": f"{quality_score:.1f}%"}] if quality_score <= 50 else []
            },
            "regulatory_compliance": self._simple_compliance_check(df),
            "risk_indicators": self._identify_simple_risks(df, quality_score / 100)
        }

# Global instance
simple_ge_profiler = SimpleGreatExpectationsProfiler()