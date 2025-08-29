"""
Actuarial Feature Engineering for Insurance Pricing

Provides domain-specific feature engineering for life insurance and annuity pricing.
Includes pre-built actuarial calculations, risk factors, and interaction terms.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import math
from dataclasses import dataclass

from ..core.actuarial_engine import ActuarialEngine, MortalityTable

class LifeInsuranceFeatureEngine:
    """
    Feature engineering specifically for life insurance pricing models
    """
    
    def __init__(self):
        self.actuarial_engine = ActuarialEngine()
    
    def create_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create complete set of life insurance pricing features"""
        
        print("Creating actuarial features for life insurance...")
        
        # Core demographic features
        df = self.create_demographic_features(df)
        
        # Health and risk features
        df = self.create_health_risk_features(df)
        
        # Financial features
        df = self.create_financial_features(df)
        
        # Policy structure features
        df = self.create_policy_features(df)
        
        # Actuarial calculation features
        df = self.create_actuarial_features(df)
        
        # Geographic risk features
        df = self.create_geographic_features(df)
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        print(f"Feature engineering completed. Total columns: {len(df.columns)}")
        return df
    
    def create_demographic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create demographic-based features"""
        
        # Age-based features
        if 'age_at_issue' in df.columns:
            df = df.with_columns([
                # Age transformations
                pl.col('age_at_issue').alias('age'),
                (pl.col('age_at_issue') ** 2).alias('age_squared'),
                pl.col('age_at_issue').map_elements(lambda x: math.log(max(x, 1)), return_dtype=pl.Float64).alias('age_log'),
                
                # Age bands
                (pl.col('age_at_issue') // 5 * 5).alias('age_band_5yr'),
                (pl.col('age_at_issue') // 10 * 10).alias('age_band_10yr'),
                
                # Life stage indicators
                (pl.col('age_at_issue') < 30).alias('young_adult'),
                ((pl.col('age_at_issue') >= 30) & (pl.col('age_at_issue') < 50)).alias('middle_age'),
                ((pl.col('age_at_issue') >= 50) & (pl.col('age_at_issue') < 65)).alias('pre_retirement'),
                (pl.col('age_at_issue') >= 65).alias('senior'),
                
                # Retirement proximity
                pl.max_horizontal([pl.lit(0), pl.lit(65) - pl.col('age_at_issue')]).alias('years_to_retirement'),
                (pl.col('age_at_issue') >= 62).alias('retirement_eligible'),
                
                # Mortality risk indicators
                (pl.col('age_at_issue') < 40).alias('low_mortality_age'),
                ((pl.col('age_at_issue') >= 40) & (pl.col('age_at_issue') < 60)).alias('medium_mortality_age'),
                (pl.col('age_at_issue') >= 60).alias('high_mortality_age')
            ])
        
        # Gender features
        if 'gender' in df.columns:
            df = df.with_columns([
                (pl.col('gender') == 'M').alias('gender_male'),
                (pl.col('gender') == 'F').alias('gender_female')
            ])
        
        # Marital status features (if available)
        if 'marital_status' in df.columns:
            df = df.with_columns([
                (pl.col('marital_status') == 'Married').alias('married'),
                (pl.col('marital_status') == 'Single').alias('single'),
                pl.col('marital_status').is_in(['Divorced', 'Widowed']).alias('previously_married')
            ])
        
        return df
    
    def create_health_risk_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create health and risk-based features"""
        
        # Smoking features
        if 'smoker_status' in df.columns:
            df = df.with_columns([
                pl.col('smoker_status').map_elements(
                    lambda x: 1 if str(x).upper() in ['SMOKER', 'YES', 'TRUE', 'Y'] else 0,
                    return_dtype=pl.Int8
                ).alias('is_smoker'),
                
                pl.col('smoker_status').map_elements(
                    lambda x: 0 if str(x).upper() in ['SMOKER', 'YES', 'TRUE', 'Y'] else 1,
                    return_dtype=pl.Int8
                ).alias('is_nonsmoker')
            ])
        
        # BMI calculation and categories
        if 'height_inches' in df.columns and 'weight_pounds' in df.columns:
            df = df.with_columns([
                # BMI calculation: (weight_lbs / height_inches^2) * 703
                ((pl.col('weight_pounds') / (pl.col('height_inches') ** 2)) * 703).alias('bmi'),
            ])
            
            # BMI categories
            df = df.with_columns([
                (pl.col('bmi') < 18.5).alias('underweight'),
                ((pl.col('bmi') >= 18.5) & (pl.col('bmi') < 25)).alias('normal_weight'),
                ((pl.col('bmi') >= 25) & (pl.col('bmi') < 30)).alias('overweight'),
                (pl.col('bmi') >= 30).alias('obese'),
                (pl.col('bmi') >= 35).alias('severely_obese'),
                
                # BMI risk score (U-shaped relationship with mortality)
                pl.col('bmi').map_elements(self._calculate_bmi_risk_score, return_dtype=pl.Float64).alias('bmi_risk_score')
            ])
        
        # Health rating conversion
        if 'health_rating' in df.columns:
            health_rating_map = {
                'Preferred Plus': 5, 'Preferred': 4, 'Standard Plus': 3, 
                'Standard': 2, 'Substandard': 1, 'Declined': 0
            }
            df = df.with_columns([
                pl.col('health_rating').map_elements(
                    lambda x: health_rating_map.get(str(x), 2),
                    return_dtype=pl.Int8
                ).alias('health_rating_numeric'),
                
                (pl.col('health_rating').is_in(['Preferred Plus', 'Preferred'])).alias('preferred_health'),
                (pl.col('health_rating') == 'Substandard').alias('substandard_health')
            ])
        
        # Medical history indicators (if available)
        medical_conditions = [
            'hypertension', 'diabetes', 'heart_disease', 'cancer_history',
            'asthma', 'depression', 'anxiety', 'high_cholesterol'
        ]
        
        condition_counts = []
        condition_severity = []
        
        for condition in medical_conditions:
            col_name = f'history_{condition}'
            if col_name in df.columns:
                df = df.with_columns([
                    pl.col(col_name).cast(pl.Int8).alias(f'has_{condition}')
                ])
                condition_counts.append(pl.col(f'has_{condition}'))
                
                # Assign severity weights
                severity_weights = {
                    'diabetes': 3, 'heart_disease': 4, 'cancer_history': 3,
                    'hypertension': 2, 'high_cholesterol': 1, 'asthma': 1,
                    'depression': 2, 'anxiety': 1
                }
                weight = severity_weights.get(condition, 1)
                condition_severity.append(pl.col(f'has_{condition}') * weight)
        
        # Create aggregate health metrics
        if condition_counts:
            df = df.with_columns([
                pl.sum_horizontal(condition_counts).alias('total_health_conditions'),
                pl.sum_horizontal(condition_severity).alias('health_severity_score'),
                (pl.sum_horizontal(condition_counts) == 0).alias('no_health_conditions')
            ])
        
        return df
    
    def create_financial_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create financial capacity and risk features"""
        
        # Income-based features
        if 'annual_income' in df.columns:
            df = df.with_columns([
                pl.col('annual_income').map_elements(lambda x: math.log(max(x, 1000)), return_dtype=pl.Float64).alias('income_log'),
                
                # Income brackets
                (pl.col('annual_income') < 35000).alias('low_income'),
                ((pl.col('annual_income') >= 35000) & (pl.col('annual_income') < 75000)).alias('middle_income'),
                ((pl.col('annual_income') >= 75000) & (pl.col('annual_income') < 150000)).alias('high_income'),
                (pl.col('annual_income') >= 150000).alias('very_high_income'),
                
                # Income percentiles (approximate)
                pl.col('annual_income').map_elements(self._income_to_percentile, return_dtype=pl.Float64).alias('income_percentile')
            ])
        
        # Face amount features
        if 'face_amount' in df.columns:
            df = df.with_columns([
                pl.col('face_amount').map_elements(lambda x: math.log(max(x, 1000)), return_dtype=pl.Float64).alias('face_amount_log'),
                
                # Coverage levels
                (pl.col('face_amount') < 100000).alias('small_coverage'),
                ((pl.col('face_amount') >= 100000) & (pl.col('face_amount') < 500000)).alias('medium_coverage'),
                ((pl.col('face_amount') >= 500000) & (pl.col('face_amount') < 1000000)).alias('large_coverage'),
                (pl.col('face_amount') >= 1000000).alias('jumbo_coverage'),
                
                # Coverage standardization (per $1000)
                (pl.col('face_amount') / 1000).alias('face_amount_per_1k')
            ])
        
        # Coverage ratios
        if 'annual_income' in df.columns and 'face_amount' in df.columns:
            df = df.with_columns([
                # Coverage multiple
                (pl.col('face_amount') / pl.col('annual_income').clip(lower_bound=1)).alias('coverage_multiple'),
                
                # Coverage adequacy indicators
                (pl.col('face_amount') / pl.col('annual_income') < 3).alias('low_coverage_ratio'),
                ((pl.col('face_amount') / pl.col('annual_income') >= 3) & 
                 (pl.col('face_amount') / pl.col('annual_income') < 10)).alias('adequate_coverage_ratio'),
                (pl.col('face_amount') / pl.col('annual_income') >= 10).alias('high_coverage_ratio'),
                
                # Anti-selection risk
                (pl.col('face_amount') / pl.col('annual_income') > 20).alias('potential_anti_selection')
            ])
        
        # Premium affordability
        if 'annual_premium' in df.columns and 'annual_income' in df.columns:
            df = df.with_columns([
                (pl.col('annual_premium') / pl.col('annual_income').clip(lower_bound=1) * 100).alias('premium_to_income_pct'),
                (pl.col('annual_premium') / pl.col('annual_income') > 0.1).alias('high_premium_burden'),
                (pl.col('annual_premium') / pl.col('annual_income') < 0.02).alias('low_premium_burden')
            ])
        
        # Net worth features (if available)
        if 'net_worth' in df.columns:
            df = df.with_columns([
                pl.col('net_worth').map_elements(lambda x: math.log(max(x, 1000)), return_dtype=pl.Float64).alias('net_worth_log'),
                (pl.col('net_worth') > 1000000).alias('high_net_worth'),
                (pl.col('net_worth') < 0).alias('negative_net_worth')
            ])
        
        return df
    
    def create_policy_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create policy structure and product features"""
        
        # Policy type features
        if 'policy_type' in df.columns:
            policy_types = ['Term', 'Whole Life', 'Universal Life', 'Variable Life', 'Variable Universal Life']
            for policy_type in policy_types:
                df = df.with_columns([
                    (pl.col('policy_type') == policy_type).alias(f'is_{policy_type.lower().replace(" ", "_")}')
                ])
            
            # Policy complexity indicators
            df = df.with_columns([
                (pl.col('policy_type') == 'Term').alias('simple_product'),
                pl.col('policy_type').is_in(['Variable Life', 'Variable Universal Life']).alias('complex_product'),
                pl.col('policy_type').is_in(['Whole Life', 'Universal Life']).alias('permanent_product')
            ])
        
        # Payment mode features
        if 'payment_mode' in df.columns or 'premium_frequency' in df.columns:
            if 'payment_mode' in df.columns:
                df = df.with_columns([
                    (pl.col('payment_mode') == 'Annual').alias('annual_pay'),
                    (pl.col('payment_mode') == 'Monthly').alias('monthly_pay'),
                    pl.col('payment_mode').is_in(['Quarterly', 'Semi-Annual']).alias('semi_frequent_pay')
                ])
            
            if 'premium_frequency' in df.columns:
                df = df.with_columns([
                    (pl.col('premium_frequency') == 1).alias('annual_premium_freq'),
                    (pl.col('premium_frequency') == 12).alias('monthly_premium_freq'),
                    (pl.col('premium_frequency') > 1).alias('frequent_payment'),
                    
                    # Payment frequency risk (monthly payers more likely to lapse)
                    pl.col('premium_frequency').map_elements(
                        lambda x: 1.0 if x == 1 else 1.2 if x <= 4 else 1.5,
                        return_dtype=pl.Float64
                    ).alias('payment_frequency_risk_factor')
                ])
        
        # Underwriting features
        if 'medical_exam' in df.columns:
            df = df.with_columns([
                (pl.col('medical_exam') == 'Yes').alias('had_medical_exam'),
                (pl.col('medical_exam') == 'No').alias('no_medical_exam')
            ])
        
        # Distribution channel features
        if 'agent_id' in df.columns:
            df = df.with_columns([
                pl.col('agent_id').is_not_null().alias('agent_sale'),
                pl.col('agent_id').is_null().alias('direct_sale')
            ])
        
        return df
    
    def create_actuarial_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create actuarial calculation-based features"""
        
        # Mortality-based features
        if all(col in df.columns for col in ['age_at_issue', 'gender']):
            # Calculate expected mortality rates
            mortality_rates = []
            life_expectancies = []
            
            for row in df.iter_rows():
                age = row[df.columns.index('age_at_issue')]
                gender = row[df.columns.index('gender')]
                smoker = False
                
                # Check if smoker status is available
                if 'is_smoker' in df.columns:
                    smoker_idx = df.columns.index('is_smoker')
                    smoker = bool(row[smoker_idx])
                elif 'smoker_status' in df.columns:
                    smoker_idx = df.columns.index('smoker_status')
                    smoker_val = row[smoker_idx]
                    smoker = str(smoker_val).upper() in ['SMOKER', 'YES', 'TRUE', 'Y']
                
                # Get actuarial values
                try:
                    qx = self.actuarial_engine.get_mortality_rate(age, gender, smoker)
                    ex = self.actuarial_engine.calculate_life_expectancy(age, gender, smoker)
                    mortality_rates.append(qx)
                    life_expectancies.append(ex)
                except:
                    mortality_rates.append(0.01)  # Default fallback
                    life_expectancies.append(75 - age)  # Simple fallback
            
            df = df.with_columns([
                pl.Series('mortality_rate_qx', mortality_rates),
                pl.Series('life_expectancy', life_expectancies)
            ])
            
            # Mortality risk categories
            df = df.with_columns([
                (pl.col('mortality_rate_qx') < 0.005).alias('low_mortality_risk'),
                ((pl.col('mortality_rate_qx') >= 0.005) & (pl.col('mortality_rate_qx') < 0.015)).alias('medium_mortality_risk'),
                (pl.col('mortality_rate_qx') >= 0.015).alias('high_mortality_risk'),
                
                # Life expectancy categories
                (pl.col('life_expectancy') > 25).alias('long_life_expectancy'),
                ((pl.col('life_expectancy') >= 15) & (pl.col('life_expectancy') <= 25)).alias('medium_life_expectancy'),
                (pl.col('life_expectancy') < 15).alias('short_life_expectancy')
            ])
        
        # Present value calculations (if face amount available)
        if all(col in df.columns for col in ['age_at_issue', 'gender', 'face_amount']):
            # This is computationally intensive, so we'll create simplified versions
            df = df.with_columns([
                # Simplified net single premium estimate
                (pl.col('face_amount') * pl.col('mortality_rate_qx') * 0.97).alias('approx_net_single_premium'),
                
                # Risk-adjusted coverage
                (pl.col('face_amount') * pl.col('mortality_rate_qx')).alias('risk_adjusted_coverage')
            ])
        
        return df
    
    def create_geographic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create geographic risk features"""
        
        if 'state' in df.columns:
            # High-cost states
            high_cost_states = ['CA', 'NY', 'CT', 'NJ', 'MA', 'HI', 'MD', 'WA']
            df = df.with_columns([
                pl.col('state').is_in(high_cost_states).alias('high_cost_state')
            ])
            
            # Regional indicators
            northeast_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA']
            southeast_states = ['DE', 'MD', 'DC', 'VA', 'WV', 'KY', 'TN', 'NC', 'SC', 'GA', 'FL', 'AL', 'MS', 'AR', 'LA']
            midwest_states = ['OH', 'MI', 'IN', 'WI', 'IL', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS']
            west_states = ['MT', 'WY', 'CO', 'NM', 'ID', 'UT', 'AZ', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
            
            df = df.with_columns([
                pl.col('state').is_in(northeast_states).alias('northeast_region'),
                pl.col('state').is_in(southeast_states).alias('southeast_region'),
                pl.col('state').is_in(midwest_states).alias('midwest_region'),
                pl.col('state').is_in(west_states).alias('west_region')
            ])
        
        # Zip code features (if available)
        if 'zip_code' in df.columns:
            df = df.with_columns([
                # Urban vs rural (rough approximation based on zip code patterns)
                pl.col('zip_code').str.slice(0, 1).cast(pl.Int32).alias('zip_first_digit'),
                pl.col('zip_code').str.slice(0, 3).alias('zip_prefix')
            ])
        
        return df
    
    def create_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create time-based features"""
        
        if 'issue_date' in df.columns:
            df = df.with_columns([
                pl.col('issue_date').dt.year().alias('issue_year'),
                pl.col('issue_date').dt.month().alias('issue_month'),
                pl.col('issue_date').dt.quarter().alias('issue_quarter'),
                pl.col('issue_date').dt.weekday().alias('issue_weekday'),
                
                # Seasonal indicators
                pl.col('issue_date').dt.month().is_in([12, 1, 2]).alias('winter_issue'),
                pl.col('issue_date').dt.month().is_in([3, 4, 5]).alias('spring_issue'),
                pl.col('issue_date').dt.month().is_in([6, 7, 8]).alias('summer_issue'),
                pl.col('issue_date').dt.month().is_in([9, 10, 11]).alias('fall_issue'),
                
                # Year-end rush (common in insurance)
                pl.col('issue_date').dt.month().is_in([11, 12]).alias('year_end_issue'),
                
                # Days from epoch (for trending)
                (pl.col('issue_date').cast(pl.Date).cast(pl.Int32) - date(2000, 1, 1).toordinal()).alias('days_since_2000')
            ])
        
        # Age and date combinations
        if 'birth_date' in df.columns and 'issue_date' in df.columns:
            df = df.with_columns([
                (pl.col('issue_date') - pl.col('birth_date')).dt.total_days().alias('age_in_days'),
                
                # Birth year features
                pl.col('birth_date').dt.year().alias('birth_year'),
                
                # Generation indicators (approximate)
                (pl.col('birth_date').dt.year() >= 1997).alias('gen_z'),
                ((pl.col('birth_date').dt.year() >= 1981) & (pl.col('birth_date').dt.year() <= 1996)).alias('millennial'),
                ((pl.col('birth_date').dt.year() >= 1965) & (pl.col('birth_date').dt.year() <= 1980)).alias('gen_x'),
                ((pl.col('birth_date').dt.year() >= 1946) & (pl.col('birth_date').dt.year() <= 1964)).alias('boomer')
            ])
        
        return df
    
    def create_interaction_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create actuarially meaningful interaction terms"""
        
        interactions = []
        
        # Age-Gender interaction (classic actuarial relationship)
        if 'age_at_issue' in df.columns and 'gender_male' in df.columns:
            df = df.with_columns([
                (pl.col('age_at_issue') * pl.col('gender_male')).alias('age_male_interaction')
            ])
        
        # Age-Smoking interaction
        if 'age_at_issue' in df.columns and 'is_smoker' in df.columns:
            df = df.with_columns([
                (pl.col('age_at_issue') * pl.col('is_smoker')).alias('age_smoker_interaction'),
                ((pl.col('age_at_issue') ** 2) * pl.col('is_smoker')).alias('age_squared_smoker_interaction')
            ])
        
        # Coverage-Age interaction (anti-selection effect)
        if 'face_amount_log' in df.columns and 'age_at_issue' in df.columns:
            df = df.with_columns([
                (pl.col('face_amount_log') * pl.col('age_at_issue')).alias('coverage_age_interaction')
            ])
        
        # Income-Coverage interaction
        if 'income_log' in df.columns and 'face_amount_log' in df.columns:
            df = df.with_columns([
                (pl.col('income_log') * pl.col('face_amount_log')).alias('income_coverage_interaction')
            ])
        
        # Health-Age interaction
        if 'health_rating_numeric' in df.columns and 'age_at_issue' in df.columns:
            df = df.with_columns([
                (pl.col('health_rating_numeric') * pl.col('age_at_issue')).alias('health_age_interaction')
            ])
        
        # Complex interaction: Age * Gender * Smoking
        if all(col in df.columns for col in ['age_at_issue', 'gender_male', 'is_smoker']):
            df = df.with_columns([
                (pl.col('age_at_issue') * pl.col('gender_male') * pl.col('is_smoker')).alias('age_gender_smoker_interaction')
            ])
        
        return df
    
    def _calculate_bmi_risk_score(self, bmi: float) -> float:
        """Calculate BMI risk score (U-shaped relationship with mortality)"""
        if pd.isna(bmi) or bmi <= 0:
            return 1.0
        
        # Optimal BMI around 22-25
        if 22 <= bmi <= 25:
            return 0.9  # Lower risk
        elif 18.5 <= bmi < 22:
            return 1.0 + (22 - bmi) * 0.05  # Slightly higher risk for underweight
        elif 25 < bmi <= 30:
            return 1.0 + (bmi - 25) * 0.04  # Increasing risk for overweight
        elif bmi > 30:
            return 1.0 + (bmi - 25) * 0.08  # Higher risk for obesity
        else:  # BMI < 18.5
            return 1.4  # Underweight risk
    
    def _income_to_percentile(self, income: float) -> float:
        """Convert income to approximate percentile"""
        if pd.isna(income) or income <= 0:
            return 0.0
        
        # Rough income percentile mapping (US household income)
        if income < 25000:
            return 20.0
        elif income < 50000:
            return 40.0
        elif income < 75000:
            return 60.0
        elif income < 100000:
            return 75.0
        elif income < 150000:
            return 85.0
        elif income < 200000:
            return 92.0
        else:
            return 95.0

class AnnuityFeatureEngine:
    """
    Feature engineering specifically for annuity and retirement products
    """
    
    def __init__(self):
        self.actuarial_engine = ActuarialEngine()
    
    def create_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create complete set of annuity pricing features"""
        
        print("Creating actuarial features for annuities...")
        
        # Core demographic features (reuse from life insurance)
        life_engine = LifeInsuranceFeatureEngine()
        df = life_engine.create_demographic_features(df)
        
        # Annuity-specific features
        df = self.create_longevity_features(df)
        df = self.create_retirement_features(df)
        df = self.create_investment_features(df)
        df = self.create_payout_features(df)
        df = self.create_surrender_risk_features(df)
        
        print(f"Annuity feature engineering completed. Total columns: {len(df.columns)}")
        return df
    
    def create_longevity_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create longevity-specific features for annuities"""
        
        if all(col in df.columns for col in ['age_at_contract', 'gender']):
            # Calculate longevity metrics using annuity mortality table
            longevity_rates = []
            remaining_life_exp = []
            
            for row in df.iter_rows():
                age = row[df.columns.index('age_at_contract')]
                gender = row[df.columns.index('gender')]
                
                try:
                    # Use annuity table (better mortality than life insurance)
                    qx = self.actuarial_engine.get_mortality_rate(
                        age, gender, False, MortalityTable.ANNUITY_2000
                    )
                    ex = self.actuarial_engine.calculate_life_expectancy(
                        age, gender, False, MortalityTable.ANNUITY_2000
                    )
                    longevity_rates.append(1 - qx)  # Survival probability
                    remaining_life_exp.append(ex)
                except:
                    longevity_rates.append(0.99)
                    remaining_life_exp.append(max(0, 85 - age))
            
            df = df.with_columns([
                pl.Series('survival_probability', longevity_rates),
                pl.Series('remaining_life_expectancy', remaining_life_exp)
            ])
            
            # Longevity risk categories
            df = df.with_columns([
                (pl.col('remaining_life_expectancy') > 20).alias('high_longevity_risk'),
                ((pl.col('remaining_life_expectancy') >= 10) & (pl.col('remaining_life_expectancy') <= 20)).alias('medium_longevity_risk'),
                (pl.col('remaining_life_expectancy') < 10).alias('low_longevity_risk')
            ])
        
        return df
    
    def create_retirement_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create retirement-specific features"""
        
        if 'age_at_contract' in df.columns:
            df = df.with_columns([
                # Retirement timing
                (pl.col('age_at_contract') < 59.5).alias('early_retirement'),
                ((pl.col('age_at_contract') >= 59.5) & (pl.col('age_at_contract') < 65)).alias('normal_retirement'),
                (pl.col('age_at_contract') >= 65).alias('late_retirement'),
                
                # Required minimum distribution considerations
                (pl.col('age_at_contract') >= 72).alias('rmd_age'),
                
                # Retirement income replacement needs
                pl.col('age_at_contract').map_elements(
                    lambda age: max(0, 25 - (age - 65)) if age >= 65 else 25,
                    return_dtype=pl.Float64
                ).alias('expected_retirement_years')
            ])
        
        return df
    
    def create_investment_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create investment-related features"""
        
        if 'premium_amount' in df.columns:
            df = df.with_columns([
                pl.col('premium_amount').map_elements(lambda x: math.log(max(x, 1000)), return_dtype=pl.Float64).alias('premium_log'),
                
                # Premium size categories
                (pl.col('premium_amount') < 50000).alias('small_premium'),
                ((pl.col('premium_amount') >= 50000) & (pl.col('premium_amount') < 250000)).alias('medium_premium'),
                ((pl.col('premium_amount') >= 250000) & (pl.col('premium_amount') < 1000000)).alias('large_premium'),
                (pl.col('premium_amount') >= 1000000).alias('jumbo_premium')
            ])
        
        if 'investment_option' in df.columns:
            df = df.with_columns([
                (pl.col('investment_option') == 'Conservative').alias('conservative_investor'),
                (pl.col('investment_option') == 'Moderate').alias('moderate_investor'),
                (pl.col('investment_option') == 'Aggressive').alias('aggressive_investor')
            ])
        
        if 'guaranteed_rate' in df.columns:
            df = df.with_columns([
                (pl.col('guaranteed_rate') * 100).alias('guaranteed_rate_pct'),
                (pl.col('guaranteed_rate') > 0.02).alias('high_guaranteed_rate'),
                (pl.col('guaranteed_rate') < 0.015).alias('low_guaranteed_rate')
            ])
        
        return df
    
    def create_payout_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create payout option features"""
        
        if 'payout_option' in df.columns:
            df = df.with_columns([
                (pl.col('payout_option') == 'Life Only').alias('life_only_payout'),
                (pl.col('payout_option').str.contains('Certain')).alias('period_certain_payout'),
                (pl.col('payout_option').str.contains('Joint')).alias('joint_life_payout'),
                
                # Risk level of payout option
                pl.col('payout_option').map_elements(
                    lambda x: 1.0 if 'Life Only' in str(x) else 0.8 if 'Certain' in str(x) else 0.6,
                    return_dtype=pl.Float64
                ).alias('payout_risk_factor')
            ])
        
        if 'monthly_payment' in df.columns and 'premium_amount' in df.columns:
            df = df.with_columns([
                # Payout rate
                ((pl.col('monthly_payment') * 12) / pl.col('premium_amount')).alias('annual_payout_rate'),
                
                # Payout adequacy
                (pl.col('monthly_payment') > 1000).alias('adequate_monthly_income'),
                (pl.col('monthly_payment') < 500).alias('low_monthly_income')
            ])
        
        return df
    
    def create_surrender_risk_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create surrender and lapse risk features"""
        
        if 'surrender_value' in df.columns and 'premium_amount' in df.columns:
            df = df.with_columns([
                # Surrender charge impact
                ((pl.col('premium_amount') - pl.col('surrender_value')) / pl.col('premium_amount')).alias('surrender_charge_rate'),
                
                # High surrender charge (higher lapse risk)
                (((pl.col('premium_amount') - pl.col('surrender_value')) / pl.col('premium_amount')) > 0.1).alias('high_surrender_charges')
            ])
        
        if 'contract_date' in df.columns:
            df = df.with_columns([
                # Contract duration (surrender risk decreases over time)
                (pl.lit(datetime.now().date()) - pl.col('contract_date')).dt.total_days().alias('contract_duration_days'),
                ((pl.lit(datetime.now().date()) - pl.col('contract_date')).dt.total_days() / 365.25).alias('contract_duration_years'),
                
                # Early surrender risk period
                (((pl.lit(datetime.now().date()) - pl.col('contract_date')).dt.total_days() / 365.25) < 5).alias('early_surrender_risk_period')
            ])
        
        return df