"""
Robust Reinsurance Feature Engineering
Works with integrated datasets and creates 100+ meaningful features
"""

import polars as pl
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class RobustReinsuranceFeatures:
    """Robust feature engineering that works with available columns"""
    
    def __init__(self):
        self.feature_descriptions = {}
    
    def create_treaty_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create comprehensive features from available columns"""
        
        # Get available columns to avoid errors
        available_cols = df.columns
        print(f"Available columns: {available_cols}")
        
        # Start with original data
        features = df.clone()
        
        # 1. BASIC FINANCIAL RATIOS (always available)
        if all(col in available_cols for col in ['premium', 'loss_ratio', 'expense_ratio']):
            features = features.with_columns([
                pl.col("combined_ratio").alias("combined_ratio_orig"),
                (pl.col("loss_ratio") + pl.col("expense_ratio")).alias("combined_ratio_calc"),
                (1 - pl.col("combined_ratio")).alias("underwriting_profit_margin"),
                (pl.col("premium") * (1 - pl.col("combined_ratio"))).alias("underwriting_profit"),
                (pl.col("premium") * pl.col("loss_ratio")).alias("expected_losses"),
                (pl.col("premium") * pl.col("expense_ratio")).alias("expense_amount"),
                pl.when(pl.col("combined_ratio") < 1.0).then(1).otherwise(0).alias("is_profitable"),
                pl.when(pl.col("combined_ratio") > 1.2).then(1).otherwise(0).alias("is_loss_making"),
                pl.when(pl.col("loss_ratio") > 0.8).then(1).otherwise(0).alias("high_loss_ratio"),
                pl.when(pl.col("expense_ratio") > 0.35).then(1).otherwise(0).alias("high_expense_ratio"),
            ])
        
        # 2. TREATY STRUCTURE FEATURES
        if 'treaty_type' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("treaty_type") == "Quota Share").then(1).otherwise(0).alias("is_quota_share"),
                pl.when(pl.col("treaty_type") == "Surplus").then(1).otherwise(0).alias("is_surplus"),
                pl.when(pl.col("treaty_type") == "Excess of Loss").then(1).otherwise(0).alias("is_excess_of_loss"),
                pl.when(pl.col("treaty_type") == "Catastrophe").then(1).otherwise(0).alias("is_catastrophe"),
                pl.when(pl.col("treaty_type").is_in(["Life Quota Share", "Life Coinsurance"])).then(1).otherwise(0).alias("is_life_treaty"),
            ])
        
        # 3. COVERAGE AND CAPACITY FEATURES
        if all(col in available_cols for col in ['limit', 'retention']):
            features = features.with_columns([
                (pl.col("limit") / pl.col("retention")).fill_null(0).alias("leverage_ratio"),
                pl.col("limit").log().alias("log_limit"),
                pl.col("retention").log().alias("log_retention"),
                pl.when(pl.col("limit") > 50000000).then(1).otherwise(0).alias("large_limit"),
                pl.when(pl.col("retention") > 1000000).then(1).otherwise(0).alias("high_retention"),
            ])
        
        if all(col in available_cols for col in ['premium', 'limit']):
            features = features.with_columns([
                (pl.col("premium") / pl.col("limit")).fill_null(0).alias("rate_on_line"),
                pl.when(pl.col("premium") / pl.col("limit") > 0.1).then(1).otherwise(0).alias("high_rate_on_line"),
            ])
        
        # 4. CESSION FEATURES
        if 'cession_rate' in available_cols:
            features = features.with_columns([
                pl.col("cession_rate").alias("cession_rate_orig"),
                (1 - pl.col("cession_rate")).fill_null(1).alias("retention_rate"),
                pl.when(pl.col("cession_rate") > 0.5).then(1).otherwise(0).alias("high_cession"),
                pl.when(pl.col("cession_rate") < 0.25).then(1).otherwise(0).alias("low_cession"),
                (pl.col("cession_rate") * pl.col("premium")).fill_null(0).alias("ceded_premium"),
            ])
        
        # 5. COMMISSION AND COST FEATURES
        if all(col in available_cols for col in ['commission', 'brokerage']):
            features = features.with_columns([
                (pl.col("commission") + pl.col("brokerage")).alias("total_acquisition_cost"),
                pl.when(pl.col("commission") > 0.25).then(1).otherwise(0).alias("high_commission"),
                pl.when(pl.col("brokerage") > 0.03).then(1).otherwise(0).alias("high_brokerage"),
                (pl.col("premium") * (pl.col("commission") + pl.col("brokerage"))).alias("acquisition_cost_amount"),
            ])
        
        # 6. REINSTATEMENT FEATURES
        if 'reinstatements' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("reinstatements") > 0).then(1).otherwise(0).alias("has_reinstatements"),
                pl.col("reinstatements").alias("reinstatement_count"),
                (pl.col("reinstatements") * pl.col("premium") * 0.1).alias("estimated_reinstatement_cost"),
                pl.when(pl.col("reinstatements") > 2).then(1).otherwise(0).alias("multiple_reinstatements"),
            ])
        
        # 7. AGGREGATE LIMIT FEATURES
        if 'aggregate_limit' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("aggregate_limit") > 0).then(1).otherwise(0).alias("has_aggregate"),
                (pl.col("premium") / pl.col("aggregate_limit")).fill_null(0).alias("aggregate_burn_rate"),
                (pl.col("aggregate_limit") / pl.col("limit")).fill_null(1).alias("aggregate_multiple"),
                pl.when(pl.col("aggregate_limit") > 100000000).then(1).otherwise(0).alias("large_aggregate"),
            ])
        
        # 8. BUSINESS LINE FEATURES
        if 'business_line' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("business_line") == "Property").then(1).otherwise(0).alias("is_property"),
                pl.when(pl.col("business_line") == "Casualty").then(1).otherwise(0).alias("is_casualty"),
                pl.when(pl.col("business_line") == "Motor").then(1).otherwise(0).alias("is_motor"),
                pl.when(pl.col("business_line") == "Marine").then(1).otherwise(0).alias("is_marine"),
                pl.when(pl.col("business_line") == "Aviation").then(1).otherwise(0).alias("is_aviation"),
                pl.when(pl.col("business_line") == "Health").then(1).otherwise(0).alias("is_health"),
                pl.when(pl.col("business_line").is_in(["Property", "Catastrophe"])).then(1).otherwise(0).alias("is_cat_exposed"),
            ])
        
        # 9. GEOGRAPHIC FEATURES
        if 'territory' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("territory") == "United States").then(1).otherwise(0).alias("is_us"),
                pl.when(pl.col("territory").is_in(["United Kingdom", "Germany", "France"])).then(1).otherwise(0).alias("is_europe"),
                pl.when(pl.col("territory") == "Japan").then(1).otherwise(0).alias("is_japan"),
                pl.when(pl.col("territory").is_in(["Australia", "Canada"])).then(1).otherwise(0).alias("is_developed"),
            ])
        
        # 10. REINSURER FEATURES
        if 'reinsurer' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("reinsurer").is_in(["Swiss Re", "Munich Re"])).then(1).otherwise(0).alias("top_tier_reinsurer"),
                pl.when(pl.col("reinsurer") == "Lloyd's of London").then(1).otherwise(0).alias("is_lloyds"),
                pl.when(pl.col("reinsurer").is_in(["Swiss Re", "Munich Re", "Hannover Re"])).then(1).otherwise(0).alias("traditional_reinsurer"),
            ])
        
        # 11. CEDANT FEATURES (simplified to avoid string operations)
        if 'cedant' in available_cols:
            features = features.with_columns([
                pl.when(pl.col("cedant").is_in(["State Farm", "Allstate"])).then(1).otherwise(0).alias("major_insurer"),
                pl.when(pl.col("cedant").is_in(["AIG", "Chubb"])).then(1).otherwise(0).alias("specialty_insurer"),
            ])
        
        # 12. PREMIUM SIZE FEATURES
        if 'premium' in available_cols:
            features = features.with_columns([
                pl.col("premium").log().alias("log_premium"),
                pl.when(pl.col("premium") > 50000000).then(1).otherwise(0).alias("large_premium"),
                pl.when(pl.col("premium") < 5000000).then(1).otherwise(0).alias("small_premium"),
                (pl.col("premium") / 1000000).alias("premium_millions"),
                pl.col("premium").rank("ordinal").alias("premium_rank"),
            ])
        
        # 13. CLAIMS-BASED FEATURES (from integration)
        claims_cols = [col for col in available_cols if 'claim' in col.lower() or col.startswith('total_historical')]
        if claims_cols:
            print(f"Found claims columns: {claims_cols}")
            
            if 'total_historical_claims' in available_cols:
                features = features.with_columns([
                    pl.col("total_historical_claims").log().alias("log_historical_claims"),
                    (pl.col("total_historical_claims") / pl.col("premium")).fill_null(0).alias("claims_to_premium_ratio"),
                    pl.when(pl.col("total_historical_claims") > pl.col("premium")).then(1).otherwise(0).alias("claims_exceed_premium"),
                ])
            
            if 'total_claim_count' in available_cols:
                features = features.with_columns([
                    pl.col("total_claim_count").alias("claim_frequency"),
                    pl.when(pl.col("total_claim_count") > 20).then(1).otherwise(0).alias("high_frequency"),
                    pl.when(pl.col("total_claim_count") == 0).then(1).otherwise(0).alias("no_claims"),
                ])
            
            if 'average_claim_size' in available_cols:
                features = features.with_columns([
                    pl.col("average_claim_size").log().alias("log_avg_claim_size"),
                    pl.when(pl.col("average_claim_size") > 500000).then(1).otherwise(0).alias("large_avg_claims"),
                ])
            
            if 'largest_historical_claim' in available_cols:
                features = features.with_columns([
                    pl.col("largest_historical_claim").log().alias("log_largest_claim"),
                    (pl.col("largest_historical_claim") / pl.col("premium")).fill_null(0).alias("largest_claim_to_premium"),
                ])
            
            if 'claim_volatility' in available_cols:
                features = features.with_columns([
                    pl.when(pl.col("claim_volatility") > pl.col("average_claim_size")).then(1).otherwise(0).alias("high_volatility"),
                ])
        
        # 14. EXPOSURE-BASED FEATURES (from integration)
        exposure_cols = [col for col in available_cols if 'sum_insured' in col or 'policy' in col]
        if exposure_cols:
            print(f"Found exposure columns: {exposure_cols}")
            
            if 'total_sum_insured' in available_cols:
                features = features.with_columns([
                    pl.col("total_sum_insured").log().alias("log_total_exposure"),
                    (pl.col("total_sum_insured") / pl.col("premium")).fill_null(0).alias("exposure_to_premium_ratio"),
                    pl.when(pl.col("total_sum_insured") > 1000000000).then(1).otherwise(0).alias("billion_plus_exposure"),
                ])
            
            if 'total_policy_count' in available_cols:
                features = features.with_columns([
                    pl.col("total_policy_count").log().alias("log_policy_count"),
                    pl.when(pl.col("total_policy_count") > 1000).then(1).otherwise(0).alias("high_policy_count"),
                ])
            
            if all(col in available_cols for col in ['total_sum_insured', 'total_policy_count']):
                features = features.with_columns([
                    (pl.col("total_sum_insured") / pl.col("total_policy_count")).fill_null(0).alias("average_policy_size_calc"),
                ])
        
        # 15. INTERACTION FEATURES
        if all(col in available_cols for col in ['loss_ratio', 'business_line']):
            features = features.with_columns([
                (pl.col("loss_ratio") * pl.when(pl.col("business_line") == "Property").then(1).otherwise(0)).alias("property_loss_ratio"),
                (pl.col("loss_ratio") * pl.when(pl.col("business_line") == "Casualty").then(1).otherwise(0)).alias("casualty_loss_ratio"),
            ])
        
        # 16. RISK SCORES
        features = features.with_columns([
            # Simple risk score based on available features
            (pl.col("loss_ratio") * 0.4 + 
             pl.col("expense_ratio") * 0.2 + 
             pl.when(pl.col("combined_ratio") > 1).then(0.3).otherwise(0) +
             pl.when('total_claim_count' in available_cols and pl.col("total_claim_count") > 15).then(0.1).otherwise(0)).alias("basic_risk_score"),
        ])
        
        print(f"Final feature count: {len(features.columns)}")
        return features
    
    def create_feature_summary(self) -> Dict[str, List[str]]:
        """Create summary of feature categories"""
        return {
            "financial_ratios": ["combined_ratio", "underwriting_profit_margin", "rate_on_line"],
            "treaty_structure": ["is_quota_share", "is_excess_of_loss", "leverage_ratio"],
            "cost_features": ["total_acquisition_cost", "high_commission"],
            "risk_indicators": ["high_loss_ratio", "is_loss_making", "basic_risk_score"],
            "business_context": ["is_property", "is_us", "top_tier_reinsurer"],
            "claims_features": ["claim_frequency", "high_frequency", "large_avg_claims"],
            "exposure_features": ["log_total_exposure", "high_policy_count"]
        }