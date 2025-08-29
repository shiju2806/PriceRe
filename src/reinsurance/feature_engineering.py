"""
Reinsurance Feature Engineering

Specialized feature engineering for reinsurance pricing and modeling:
- Treaty-specific features
- Portfolio risk features  
- Claims development features
- Catastrophe features
- Cedent quality features
- Market cycle features
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ReinsuranceFeatures:
    """Advanced feature engineering for reinsurance data"""
    
    def __init__(self):
        """Initialize feature engineering engine"""
        self.feature_descriptions = {}
    
    def create_treaty_features(self, treaty_df: pl.DataFrame) -> pl.DataFrame:
        """Create treaty-specific features"""
        
        features = treaty_df.with_columns([
            # Treaty structure features
            (pl.col("limit") / pl.col("retention")).alias("leverage_ratio"),
            (pl.col("premium") / pl.col("limit")).alias("rate_on_line"),
            (pl.col("commission") + pl.col("brokerage")).alias("total_acquisition_cost"),
            
            # Coverage features
            pl.when(pl.col("treaty_type") == "Quota Share")
              .then(pl.col("cession_rate"))
              .otherwise(0).alias("qs_cession_rate"),
            
            pl.when(pl.col("treaty_type").str.contains("Excess"))
              .then(1)
              .otherwise(0).alias("is_excess_treaty"),
            
            # Premium adequacy
            (pl.col("loss_ratio") + pl.col("expense_ratio")).alias("combined_ratio"),
            pl.when(pl.col("combined_ratio") < 1.0)
              .then(1)
              .otherwise(0).alias("is_profitable"),
            
            # Treaty vintage
            (pl.col("inception_date").str.strptime(pl.Date, "%Y-%m-%d")
             .dt.year()).alias("treaty_year"),
            
            # Risk concentration
            pl.when(pl.col("aggregate_limit").is_not_null())
              .then(pl.col("premium") / pl.col("aggregate_limit"))
              .otherwise(0).alias("aggregate_burn_rate"),
        ])
        
        # Add reinstatement features
        features = features.with_columns([
            pl.when(pl.col("reinstatements") > 0)
              .then(1)
              .otherwise(0).alias("has_reinstatements"),
              
            (pl.col("reinstatements") * pl.col("premium") * 0.1).alias("reinstatement_cost_estimate")
        ])
        
        # Add market cycle features
        features = self._add_market_cycle_features(features)
        
        # Add cedant quality features  
        features = self._add_cedant_features(features)
        
        return features
    
    def create_portfolio_features(self, portfolio_df: pl.DataFrame) -> pl.DataFrame:
        """Create portfolio risk features"""
        
        features = portfolio_df.with_columns([
            # Scale features
            (pl.col("total_sum_insured") / pl.col("number_of_risks")).alias("average_risk_size"),
            pl.col("total_sum_insured").log().alias("log_total_exposure"),
            
            # Concentration features
            (pl.col("geographic_concentration") * pl.col("industry_concentration")).alias("total_concentration"),
            
            pl.when(pl.col("geographic_concentration") > 0.5)
              .then(1)
              .otherwise(0).alias("high_geo_concentration"),
              
            # Risk quality features
            (pl.col("deductible_average") / pl.col("average_sum_insured")).alias("deductible_ratio"),
            (pl.col("policy_limit_average") / pl.col("average_sum_insured")).alias("limit_adequacy"),
            
            # Experience features
            pl.when(pl.col("historical_loss_ratio") > 1.0)
              .then(1)
              .otherwise(0).alias("loss_making_portfolio"),
              
            (pl.col("volatility") * pl.col("historical_loss_ratio")).alias("risk_adjusted_loss_ratio"),
            
            # Pricing features
            (pl.col("premium_rate") / pl.col("historical_loss_ratio")).alias("pricing_adequacy_ratio"),
            
            pl.when(pl.col("pricing_adequacy") < 0.9)
              .then(1)
              .otherwise(0).alias("underpriced_flag"),
        ])
        
        # Add catastrophe exposure features
        features = self._add_catastrophe_features(features)
        
        # Add correlation features
        features = self._add_correlation_features(features)
        
        return features
    
    def create_claims_features(self, claims_df: pl.DataFrame) -> pl.DataFrame:
        """Create claims-based features"""
        
        # Basic claim features
        features = claims_df.with_columns([
            # Timing features
            (pl.col("report_date").str.strptime(pl.Date, "%Y-%m-%d") -
             pl.col("occurrence_date").str.strptime(pl.Date, "%Y-%m-%d")).dt.total_days().alias("report_lag_days"),
            
            # Severity features
            pl.col("gross_claim_amount").log().alias("log_gross_claim"),
            (pl.col("reinsurance_recovery") / pl.col("gross_claim_amount")).alias("recovery_ratio"),
            
            # Reserve adequacy
            (pl.col("case_reserves") / pl.col("gross_claim_amount")).alias("reserve_ratio"),
            (pl.col("outstanding_reserves") / pl.col("case_reserves")).alias("reserve_development"),
            
            # Claim characteristics
            pl.when(pl.col("claim_type") == "Catastrophe")
              .then(1)
              .otherwise(0).alias("is_cat_claim"),
              
            pl.when(pl.col("gross_claim_amount") > 1000000)
              .then(1)
              .otherwise(0).alias("is_large_claim"),
            
            # Recovery features
            (pl.col("salvage_subrogation") / pl.col("gross_claim_amount")).alias("salvage_ratio"),
        ])
        
        # Add claim status features
        features = self._add_claim_status_features(features)
        
        # Add development features
        features = self._add_development_features(features)
        
        return features
    
    def create_loss_development_features(self, development_df: pl.DataFrame) -> pl.DataFrame:
        """Create loss development triangle features"""
        
        features = development_df.with_columns([
            # Development patterns
            (pl.col("incurred_amount") / pl.col("incurred_amount").first()).over("claim_id").alias("cumulative_development_factor"),
            
            (pl.col("paid_amount") / pl.col("incurred_amount")).alias("paid_ratio"),
            
            # Payment patterns
            pl.when(pl.col("payment_pattern") == "Fast")
              .then(1.2)
              .when(pl.col("payment_pattern") == "Medium")
              .then(1.0)
              .otherwise(0.8).alias("payment_speed_factor"),
            
            # Reserve adequacy over time
            pl.when(pl.col("reserve_adequacy") < 0.9)
              .then(1)
              .otherwise(0).alias("reserve_deficiency_flag"),
            
            # Development volatility
            (pl.col("development_factor") - pl.col("development_factor").mean()).over("claim_id").alias("development_volatility"),
        ])
        
        # Add tail development features
        features = self._add_tail_development_features(features)
        
        return features
    
    def create_catastrophe_features(self, cat_df: pl.DataFrame) -> pl.DataFrame:
        """Create catastrophe event features"""
        
        features = cat_df.with_columns([
            # Event characteristics
            pl.col("magnitude").alias("cat_magnitude"),
            
            pl.when(pl.col("event_type").is_in(["Hurricane", "Earthquake", "Tsunami"]))
              .then(1)
              .otherwise(0).alias("is_major_peril"),
            
            # Loss features
            (pl.col("industry_loss") / pl.col("modeled_loss")).alias("model_vs_industry_ratio"),
            
            pl.when(pl.col("industry_loss") > 1_000_000_000)
              .then(1)
              .otherwise(0).alias("is_billion_dollar_event"),
            
            # Geographic features
            pl.when(pl.col("geographic_spread") == "National")
              .then(3)
              .when(pl.col("geographic_spread") == "Regional")
              .then(2)
              .otherwise(1).alias("geographic_impact_score"),
            
            # Development features
            pl.when(pl.col("loss_development_pattern") == "Fast")
              .then(1.3)
              .when(pl.col("loss_development_pattern") == "Medium")
              .then(1.0)
              .otherwise(0.7).alias("cat_development_factor"),
        ])
        
        # Add seasonal features
        features = self._add_seasonal_cat_features(features)
        
        return features
    
    def create_aggregate_features(
        self,
        claims_df: pl.DataFrame,
        groupby_cols: List[str]
    ) -> pl.DataFrame:
        """Create aggregated features by treaty/portfolio"""
        
        aggregated = claims_df.group_by(groupby_cols).agg([
            # Claim counts
            pl.len().alias("claim_count"),
            pl.col("is_cat_claim").sum().alias("cat_claim_count"),
            pl.col("is_large_claim").sum().alias("large_claim_count"),
            
            # Loss aggregates
            pl.col("gross_claim_amount").sum().alias("total_gross_losses"),
            pl.col("reinsurance_recovery").sum().alias("total_recoveries"),
            pl.col("gross_claim_amount").mean().alias("average_claim_size"),
            pl.col("gross_claim_amount").std().alias("claim_size_volatility"),
            
            # Ratios
            pl.col("recovery_ratio").mean().alias("average_recovery_ratio"),
            pl.col("reserve_ratio").mean().alias("average_reserve_ratio"),
            
            # Development metrics
            pl.col("report_lag_days").mean().alias("average_report_lag"),
            pl.col("report_lag_days").quantile(0.9).alias("report_lag_p90"),
        ])
        
        # Add derived features
        aggregated = aggregated.with_columns([
            # Frequency features
            pl.when(pl.col("claim_count") > 0)
              .then(pl.col("cat_claim_count") / pl.col("claim_count"))
              .otherwise(0).alias("cat_claim_frequency"),
              
            pl.when(pl.col("claim_count") > 0)
              .then(pl.col("large_claim_count") / pl.col("claim_count"))
              .otherwise(0).alias("large_claim_frequency"),
            
            # Severity ratios
            pl.when(pl.col("total_gross_losses") > 0)
              .then(pl.col("total_recoveries") / pl.col("total_gross_losses"))
              .otherwise(0).alias("overall_recovery_ratio"),
            
            # Volatility measures
            pl.when(pl.col("average_claim_size") > 0)
              .then(pl.col("claim_size_volatility") / pl.col("average_claim_size"))
              .otherwise(0).alias("coefficient_of_variation"),
        ])
        
        return aggregated
    
    def _add_market_cycle_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add insurance market cycle features"""
        
        # Market cycle mapping (simplified)
        cycle_mapping = {
            2020: "Soft", 2021: "Soft", 2022: "Hardening", 
            2023: "Hard", 2024: "Hard"
        }
        
        cycle_features = df.with_columns([
            pl.col("treaty_year").replace(cycle_mapping, default="Unknown").alias("market_cycle"),
            
            pl.when(pl.col("treaty_year").is_in([2022, 2023, 2024]))
              .then(1)
              .otherwise(0).alias("hard_market_period"),
        ])
        
        return cycle_features
    
    def _add_cedant_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cedant quality and relationship features"""
        
        # Simplified cedant scoring
        tier1_cedants = ["AIG", "Travelers", "Chubb", "Zurich"]
        
        cedant_features = df.with_columns([
            pl.when(pl.col("cedant").is_in(tier1_cedants))
              .then(1)
              .otherwise(0).alias("is_tier1_cedant"),
              
            # Historical relationship indicator (simplified)
            pl.when(pl.col("combined_ratio") < 1.0)
              .then(1)
              .otherwise(0).alias("profitable_relationship"),
        ])
        
        return cedant_features
    
    def _add_catastrophe_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add catastrophe exposure features"""
        
        cat_features = df.with_columns([
            pl.when(pl.col("cat_exposure") == True)
              .then(1.5)
              .otherwise(1.0).alias("cat_exposure_multiplier"),
              
            # Territory-based cat risk
            pl.when(pl.col("territory").is_in(["United States", "Japan"]))
              .then(1)
              .otherwise(0).alias("high_cat_risk_territory"),
        ])
        
        return cat_features
    
    def _add_correlation_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add correlation risk features"""
        
        correlation_features = df.with_columns([
            pl.when(pl.col("correlation_factor") > 0.5)
              .then(1)
              .otherwise(0).alias("high_correlation_risk"),
              
            (pl.col("correlation_factor") * pl.col("geographic_concentration")).alias("correlation_concentration_interaction"),
        ])
        
        return correlation_features
    
    def _add_claim_status_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add claim status-based features"""
        
        status_features = df.with_columns([
            pl.when(pl.col("claim_status") == "Open")
              .then(1)
              .otherwise(0).alias("is_open_claim"),
              
            pl.when(pl.col("claim_status") == "IBNR")
              .then(1)
              .otherwise(0).alias("is_ibnr_claim"),
              
            pl.when(pl.col("claim_status") == "Reopened")
              .then(1)
              .otherwise(0).alias("is_reopened_claim"),
        ])
        
        return status_features
    
    def _add_development_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add claim development features"""
        
        development_features = df.with_columns([
            # Payment pattern indicators
            pl.when(pl.col("paid_to_date") / pl.col("gross_claim_amount") > 0.8)
              .then(1)
              .otherwise(0).alias("high_payment_ratio"),
              
            # Reserve movements
            pl.when(pl.col("outstanding_reserves") > pl.col("case_reserves"))
              .then(1)
              .otherwise(0).alias("reserves_increased"),
        ])
        
        return development_features
    
    def _add_tail_development_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add tail development features"""
        
        tail_features = df.with_columns([
            # Late development indicator
            pl.when((pl.col("development_quarter") > 8) & 
                   (pl.col("development_factor") > pl.col("development_factor").shift(1)))
              .then(1)
              .otherwise(0).alias("late_development_flag"),
              
            # Ultimate loss estimate stability
            pl.when(pl.col("development_quarter") > 4)
              .then(pl.col("incurred_amount").std().over("claim_id"))
              .otherwise(0).alias("ultimate_volatility"),
        ])
        
        return tail_features
    
    def _add_seasonal_cat_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add seasonal catastrophe features"""
        
        seasonal_features = df.with_columns([
            # Extract month from occurrence date
            pl.col("occurrence_date").str.strptime(pl.Date, "%Y-%m-%d").dt.month().alias("occurrence_month"),
        ])
        
        # Add hurricane season indicator
        seasonal_features = seasonal_features.with_columns([
            pl.when(pl.col("occurrence_month").is_in([6, 7, 8, 9, 10, 11]))
              .then(1)
              .otherwise(0).alias("hurricane_season"),
              
            pl.when(pl.col("occurrence_month").is_in([12, 1, 2, 3]))
              .then(1)
              .otherwise(0).alias("winter_season"),
        ])
        
        return seasonal_features
    
    def create_feature_summary(self) -> Dict[str, List[str]]:
        """Return summary of all available features"""
        
        return {
            "treaty_features": [
                "leverage_ratio", "rate_on_line", "total_acquisition_cost",
                "qs_cession_rate", "is_excess_treaty", "combined_ratio",
                "is_profitable", "treaty_year", "aggregate_burn_rate",
                "has_reinstatements", "reinstatement_cost_estimate"
            ],
            "portfolio_features": [
                "average_risk_size", "log_total_exposure", "total_concentration",
                "high_geo_concentration", "deductible_ratio", "limit_adequacy",
                "loss_making_portfolio", "risk_adjusted_loss_ratio",
                "pricing_adequacy_ratio", "underpriced_flag"
            ],
            "claims_features": [
                "report_lag_days", "log_gross_claim", "recovery_ratio",
                "reserve_ratio", "reserve_development", "is_cat_claim",
                "is_large_claim", "salvage_ratio"
            ],
            "development_features": [
                "cumulative_development_factor", "paid_ratio", "payment_speed_factor",
                "reserve_deficiency_flag", "development_volatility"
            ],
            "catastrophe_features": [
                "cat_magnitude", "is_major_peril", "model_vs_industry_ratio",
                "is_billion_dollar_event", "geographic_impact_score",
                "cat_development_factor"
            ],
            "aggregate_features": [
                "claim_count", "cat_claim_count", "large_claim_count",
                "total_gross_losses", "total_recoveries", "average_claim_size",
                "claim_size_volatility", "cat_claim_frequency", "large_claim_frequency"
            ]
        }