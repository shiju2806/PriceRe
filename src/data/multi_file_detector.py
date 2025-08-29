"""
Multi-File Data Type Detection for Reinsurance Data

Automatically detects file types based on column patterns to enable
realistic multi-file workflows that mirror production systems.
"""

import polars as pl
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported reinsurance data file types"""
    TREATY_MASTER = "treaty_master"
    CLAIMS_HISTORY = "claims_history"
    POLICY_EXPOSURES = "policy_exposures"
    MARKET_DATA = "market_data"
    COMBINED = "combined"  # Single file with everything
    UNKNOWN = "unknown"


@dataclass
class FileTypeScore:
    """File type detection score with confidence metrics"""
    file_type: FileType
    confidence: float
    matched_columns: List[str]
    missing_columns: List[str]
    total_columns: int


@dataclass
class DetectionResult:
    """Complete file detection result"""
    filename: str
    detected_type: FileType
    confidence: float
    recommendations: List[str]
    column_analysis: Dict[str, str]


class ReinsuranceFileDetector:
    """Detects reinsurance file types based on column patterns"""
    
    def __init__(self):
        # Define column patterns for each file type
        self.file_type_patterns = {
            FileType.TREATY_MASTER: {
                'required': [
                    'treaty_id', 'treaty_name', 'treaty_type',
                    'inception_date', 'premium', 'limit'
                ],
                'optional': [
                    'expiry_date', 'retention', 'commission', 'brokerage',
                    'cedant', 'reinsurer', 'currency', 'territory',
                    'business_line', 'cession_rate', 'minimum_premium',
                    'maximum_premium', 'profit_commission'
                ],
                'indicators': [
                    'treaty', 'inception', 'expiry', 'cedant', 'reinsurer'
                ]
            },
            
            FileType.CLAIMS_HISTORY: {
                'required': [
                    'claim_id', 'treaty_id', 'loss_date', 'claim_amount'
                ],
                'optional': [
                    'reported_date', 'paid_date', 'reserve_amount',
                    'cause_of_loss', 'catastrophe_code', 'latitude',
                    'longitude', 'status', 'recovery_amount', 'adjuster',
                    'claim_type', 'development_year', 'ultimate_loss'
                ],
                'indicators': [
                    'claim', 'loss_date', 'paid', 'reserve', 'catastrophe',
                    'recovery', 'ultimate', 'development'
                ]
            },
            
            FileType.POLICY_EXPOSURES: {
                'required': [
                    'policy_id', 'sum_insured'
                ],
                'optional': [
                    'treaty_id', 'deductible', 'latitude', 'longitude',
                    'occupancy', 'construction_type', 'year_built',
                    'protection_class', 'coverage_type', 'policy_limits',
                    'address', 'zip_code', 'state', 'country'
                ],
                'indicators': [
                    'policy', 'sum_insured', 'deductible', 'occupancy',
                    'construction', 'latitude', 'longitude', 'address',
                    'coverage', 'protection'
                ]
            },
            
            FileType.MARKET_DATA: {
                'required': [
                    'date'
                ],
                'optional': [
                    'gdp_growth', 'interest_rate_10y', 'cat_pcs_index',
                    'insurance_stocks_index', 'hard_market_indicator',
                    'regulatory_capital_ratio', 'inflation_rate',
                    'unemployment_rate', 'sp500_index', 'vix_index'
                ],
                'indicators': [
                    'gdp', 'interest_rate', 'market', 'index', 'inflation',
                    'economic', 'regulatory', 'capital', 'vix'
                ]
            },
            
            FileType.COMBINED: {
                'required': [
                    'treaty_id', 'premium', 'loss_ratio'
                ],
                'optional': [
                    'treaty_name', 'treaty_type', 'business_line',
                    'cedant', 'reinsurer', 'commission', 'brokerage',
                    'expense_ratio', 'combined_ratio'
                ],
                'indicators': [
                    'loss_ratio', 'expense_ratio', 'combined_ratio'
                ]
            }
        }
    
    def detect_file_type(self, df, filename: str = "") -> DetectionResult:
        """
        Detect the type of a reinsurance data file
        
        Args:
            df: DataFrame to analyze
            filename: Optional filename for additional hints
            
        Returns:
            DetectionResult with file type and confidence
        """
        
        # Get column names (handle both pandas and polars)
        if hasattr(df, 'columns'):
            columns = list(df.columns)
        else:
            columns = df.columns.tolist()
        
        # Convert to lowercase for matching
        columns_lower = [col.lower().replace('_', '').replace(' ', '') for col in columns]
        
        # Score each file type
        scores = []
        for file_type, patterns in self.file_type_patterns.items():
            score = self._calculate_file_type_score(columns, columns_lower, patterns, file_type)
            scores.append(score)
        
        # Find best match
        best_score = max(scores, key=lambda x: x.confidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(best_score, columns)
        
        # Analyze columns
        column_analysis = self._analyze_columns(columns, best_score.file_type)
        
        return DetectionResult(
            filename=filename,
            detected_type=best_score.file_type,
            confidence=best_score.confidence,
            recommendations=recommendations,
            column_analysis=column_analysis
        )
    
    def _calculate_file_type_score(self, columns: List[str], columns_lower: List[str], 
                                 patterns: Dict, file_type: FileType) -> FileTypeScore:
        """Calculate confidence score for a specific file type"""
        
        required_cols = patterns.get('required', [])
        optional_cols = patterns.get('optional', [])
        indicators = patterns.get('indicators', [])
        
        # Score required columns
        required_matches = []
        for req_col in required_cols:
            if self._column_matches(req_col, columns_lower):
                required_matches.append(req_col)
        
        # Score optional columns
        optional_matches = []
        for opt_col in optional_cols:
            if self._column_matches(opt_col, columns_lower):
                optional_matches.append(opt_col)
        
        # Score indicators (partial matches)
        indicator_matches = []
        for indicator in indicators:
            if any(indicator.lower() in col_lower for col_lower in columns_lower):
                indicator_matches.append(indicator)
        
        # Calculate confidence
        required_score = len(required_matches) / max(len(required_cols), 1)
        optional_score = len(optional_matches) / max(len(optional_cols), 1) * 0.5
        indicator_score = len(indicator_matches) / max(len(indicators), 1) * 0.3
        
        total_confidence = (required_score * 0.7 + optional_score * 0.2 + indicator_score * 0.1)
        
        # Penalty for missing required columns
        if required_score < 0.5:  # Less than half required columns
            total_confidence *= 0.3
        
        all_matches = required_matches + optional_matches
        missing_required = [col for col in required_cols if col not in required_matches]
        
        return FileTypeScore(
            file_type=file_type,
            confidence=total_confidence,
            matched_columns=all_matches,
            missing_columns=missing_required,
            total_columns=len(columns)
        )
    
    def _column_matches(self, pattern: str, columns_lower: List[str]) -> bool:
        """Check if a pattern matches any column"""
        pattern_clean = pattern.lower().replace('_', '').replace(' ', '')
        
        # Exact match
        if pattern_clean in columns_lower:
            return True
        
        # Partial match for key terms
        key_terms = {
            'treatyid': ['treaty', 'id'],
            'claimid': ['claim', 'id'],
            'policyid': ['policy', 'id'],
            'lossdate': ['loss', 'date'],
            'claimamount': ['claim', 'amount'],
            'suminsured': ['sum', 'insured'],
            'inceptiondate': ['inception', 'date'],
            'expirydate': ['expiry', 'date'],
        }
        
        if pattern_clean in key_terms:
            terms = key_terms[pattern_clean]
            return any(all(term in col for term in terms) for col in columns_lower)
        
        # Fuzzy matching for common variations
        variations = {
            'treaty_id': ['treatyid', 'treaty', 'contractid'],
            'claim_amount': ['claimamount', 'amount', 'loss'],
            'loss_ratio': ['lossratio', 'lr'],
            'expense_ratio': ['expenseratio', 'er'],
            'premium': ['prem', 'premium'],
            'commission': ['comm', 'commission'],
        }
        
        if pattern in variations:
            return any(var in col for var in variations[pattern] for col in columns_lower)
        
        return False
    
    def _generate_recommendations(self, score: FileTypeScore, columns: List[str]) -> List[str]:
        """Generate recommendations based on detection results"""
        recommendations = []
        
        if score.confidence < 0.3:
            recommendations.append("⚠️ Low confidence detection - file type may be incorrect")
        
        if score.missing_columns:
            recommendations.append(f"📋 Missing recommended columns: {', '.join(score.missing_columns[:3])}")
        
        if score.file_type == FileType.TREATY_MASTER and score.confidence > 0.7:
            recommendations.append("✅ Excellent treaty master file - ready for processing")
        
        if score.file_type == FileType.CLAIMS_HISTORY and 'ultimate_loss' not in [c.lower() for c in columns]:
            recommendations.append("💡 Consider adding ultimate loss estimates for better reserving")
        
        if score.file_type == FileType.POLICY_EXPOSURES and not any('lat' in c.lower() for c in columns):
            recommendations.append("🗺️ Adding latitude/longitude would enable catastrophe modeling")
        
        return recommendations
    
    def _analyze_columns(self, columns: List[str], file_type: FileType) -> Dict[str, str]:
        """Analyze columns and categorize them"""
        analysis = {}
        
        patterns = self.file_type_patterns.get(file_type, {})
        required = patterns.get('required', [])
        optional = patterns.get('optional', [])
        
        for col in columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            
            if any(self._column_matches(req, [col_lower]) for req in required):
                analysis[col] = "🔴 Required"
            elif any(self._column_matches(opt, [col_lower]) for opt in optional):
                analysis[col] = "🟡 Optional"
            else:
                analysis[col] = "🔵 Additional"
        
        return analysis


class MultiFileIntegrator:
    """Integrates multiple reinsurance data files into unified dataset"""
    
    def __init__(self):
        self.detector = ReinsuranceFileDetector()
        self.last_integration_report = None
    
    def integrate_files(self, file_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Integrate multiple files into unified dataset for ML training
        
        Args:
            file_data: Dictionary of {filename: dataframe}
            
        Returns:
            Integrated dataframe ready for ML processing
        """
        
        integration_report = {
            'files_processed': len(file_data),
            'file_types_detected': {},
            'records_integrated': 0,
            'data_completeness': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Step 1: Detect file types
        detected_files = {}
        for filename, df in file_data.items():
            detection = self.detector.detect_file_type(df, filename)
            detected_files[filename] = {
                'dataframe': df,
                'detection': detection
            }
            integration_report['file_types_detected'][filename] = {
                'type': detection.detected_type.value,
                'confidence': detection.confidence
            }
        
        # Step 2: Find treaty master (required)
        treaty_master = None
        treaty_master_name = None
        
        for filename, file_info in detected_files.items():
            if file_info['detection'].detected_type == FileType.TREATY_MASTER:
                treaty_master = file_info['dataframe']
                treaty_master_name = filename
                break
            elif file_info['detection'].detected_type == FileType.COMBINED:
                # Combined file can serve as treaty master
                treaty_master = file_info['dataframe']
                treaty_master_name = filename
                break
        
        if treaty_master is None:
            raise ValueError("No treaty master or combined data file found. At least one is required.")
        
        # Start with treaty master as base
        integrated_data = treaty_master.clone()
        integration_report['records_integrated'] = len(integrated_data)
        
        # Step 3: Integrate claims history if available
        claims_data = None
        for filename, file_info in detected_files.items():
            if file_info['detection'].detected_type == FileType.CLAIMS_HISTORY:
                claims_data = file_info['dataframe']
                break
        
        if claims_data is not None:
            claims_agg = self._aggregate_claims_by_treaty(claims_data)
            integrated_data = self._safe_join(integrated_data, claims_agg, 'treaty_id', 'claims')
            integration_report['data_completeness']['claims_history'] = "Available"
        else:
            integration_report['warnings'].append("No claims history file found - using treaty-level loss ratios")
            integration_report['data_completeness']['claims_history'] = "Missing"
        
        # Step 4: Integrate exposure data if available  
        exposure_data = None
        for filename, file_info in detected_files.items():
            if file_info['detection'].detected_type == FileType.POLICY_EXPOSURES:
                exposure_data = file_info['dataframe']
                break
        
        if exposure_data is not None:
            exposure_agg = self._aggregate_exposures_by_treaty(exposure_data)
            integrated_data = self._safe_join(integrated_data, exposure_agg, 'treaty_id', 'exposures')
            integration_report['data_completeness']['policy_exposures'] = "Available"
        else:
            integration_report['warnings'].append("No exposure data file found - using treaty limits for exposure estimation")
            integration_report['data_completeness']['policy_exposures'] = "Missing"
        
        # Step 5: Add market data if available
        market_data = None
        for filename, file_info in detected_files.items():
            if file_info['detection'].detected_type == FileType.MARKET_DATA:
                market_data = file_info['dataframe']
                break
        
        if market_data is not None:
            # Add market indicators (simplified - would need date matching in production)
            market_summary = self._summarize_market_data(market_data)
            integration_report['data_completeness']['market_data'] = "Available"
        else:
            integration_report['data_completeness']['market_data'] = "Missing"
        
        # Step 6: Calculate data quality score
        completeness_score = sum([
            1 if integration_report['data_completeness'].get('claims_history') == "Available" else 0.3,
            1 if integration_report['data_completeness'].get('policy_exposures') == "Available" else 0.5,
            1 if integration_report['data_completeness'].get('market_data') == "Available" else 0.8,
        ]) / 3.0
        
        integration_report['overall_quality_score'] = completeness_score
        
        # Generate recommendations
        if completeness_score < 0.6:
            integration_report['recommendations'].append("Consider uploading additional files for better analysis")
        if integration_report['data_completeness'].get('claims_history') == "Missing":
            integration_report['recommendations'].append("Claims history would enable better loss development modeling")
        if integration_report['data_completeness'].get('policy_exposures') == "Missing":
            integration_report['recommendations'].append("Exposure data would enable catastrophe risk modeling")
        
        # Store integration report in global state for UI access
        self.last_integration_report = integration_report
        
        return integrated_data
    
    def get_integration_report(self) -> Dict:
        """Get the last integration report"""
        return self.last_integration_report or {}
    
    def _aggregate_claims_by_treaty(self, claims_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate individual claims to treaty level"""
        
        if 'treaty_id' not in claims_df.columns:
            raise ValueError("Claims data must include 'treaty_id' column")
        
        return claims_df.group_by('treaty_id').agg([
            pl.col('claim_amount').sum().alias('total_historical_claims'),
            pl.col('claim_amount').count().alias('total_claim_count'),
            pl.col('claim_amount').max().alias('largest_historical_claim'),
            pl.col('claim_amount').mean().alias('average_claim_size'),
            pl.col('claim_amount').std().alias('claim_volatility'),
            # Calculate frequency (claims per year - simplified)
            (pl.col('claim_amount').count() / 5).alias('annual_claim_frequency'),  # Assume 5 years
        ])
    
    def _aggregate_exposures_by_treaty(self, exposure_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate policy exposures to treaty level"""
        
        # Group by treaty if treaty_id available, otherwise create summary stats
        if 'treaty_id' in exposure_df.columns:
            return exposure_df.group_by('treaty_id').agg([
                pl.col('sum_insured').sum().alias('total_sum_insured'),
                pl.col('sum_insured').count().alias('total_policy_count'),
                pl.col('sum_insured').mean().alias('average_policy_size'),
                pl.col('sum_insured').max().alias('largest_policy_size'),
            ])
        else:
            # Create portfolio-level summary to merge with all treaties
            summary = exposure_df.select([
                pl.col('sum_insured').sum().alias('portfolio_sum_insured'),
                pl.col('sum_insured').count().alias('portfolio_policy_count'),
                pl.col('sum_insured').mean().alias('portfolio_avg_policy_size'),
            ])
            # This would be applied to all treaties (simplified approach)
            return summary
    
    def _summarize_market_data(self, market_df: pl.DataFrame) -> Dict:
        """Summarize market data for integration"""
        
        summary = {}
        
        # Calculate recent market indicators (simplified)
        if 'hard_market_indicator' in market_df.columns:
            summary['current_market_cycle'] = market_df['hard_market_indicator'][-1:].to_list()[0]
        
        if 'interest_rate_10y' in market_df.columns:
            summary['current_interest_rate'] = market_df['interest_rate_10y'][-1:].to_list()[0]
        
        return summary
    
    def _safe_join(self, left_df: pl.DataFrame, right_df: pl.DataFrame, 
                  join_key: str, join_type: str) -> pl.DataFrame:
        """Safely join dataframes with error handling"""
        
        try:
            if join_key in left_df.columns and join_key in right_df.columns:
                return left_df.join(right_df, on=join_key, how='left')
            else:
                logger.warning(f"Cannot join {join_type} data - missing {join_key} column")
                return left_df
        except Exception as e:
            logger.error(f"Error joining {join_type} data: {e}")
            return left_df


# Example usage and testing
if __name__ == "__main__":
    detector = ReinsuranceFileDetector()
    
    # Test with sample data
    sample_treaty_data = pl.DataFrame({
        'treaty_id': ['T001', 'T002'],
        'treaty_name': ['Property QS 2024', 'CAT XOL 2024'],
        'treaty_type': ['Quota Share', 'Excess of Loss'],
        'premium': [1000000, 2000000],
        'limit': [10000000, 50000000],
        'inception_date': ['2024-01-01', '2024-01-01']
    })
    
    result = detector.detect_file_type(sample_treaty_data, "treaties.csv")
    
    print(f"🔍 File Type Detection Results:")
    print(f"Detected Type: {result.detected_type.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Column Analysis: {result.column_analysis}")