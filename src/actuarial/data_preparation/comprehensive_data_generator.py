"""
Comprehensive Actuarial Data Generator
Creates realistic test datasets for life and retirement reinsurance
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
from enum import Enum

class ProductType(Enum):
    TERM_LIFE = "TERM"
    WHOLE_LIFE = "WHOLE_LIFE"
    UNIVERSAL_LIFE = "UNIVERSAL_LIFE"
    VARIABLE_LIFE = "VARIABLE_LIFE"
    IMMEDIATE_ANNUITY = "IMMEDIATE_ANNUITY"
    DEFERRED_ANNUITY = "DEFERRED_ANNUITY"
    PENSION = "PENSION"
    GROUP_LIFE = "GROUP_LIFE"

class PolicyStatus(Enum):
    ACTIVE = "ACTIVE"
    LAPSED = "LAPSED"
    SURRENDERED = "SURRENDERED"
    MATURED = "MATURED"
    CLAIM = "CLAIM"
    PENDING = "PENDING"

@dataclass
class DataGenerationConfig:
    """Configuration for comprehensive data generation"""
    n_policies: int = 50000
    n_companies: int = 25
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2023, 12, 31)
    include_claims: bool = True
    include_lapses: bool = True
    include_economic_data: bool = True
    include_medical_data: bool = True
    include_geographic_data: bool = True
    data_quality_issues: bool = True
    missing_data_pct: float = 0.05
    outlier_pct: float = 0.02
    duplicate_pct: float = 0.01

class ComprehensiveActuarialDataGenerator:
    """
    Generate comprehensive, realistic actuarial datasets
    Includes all major components needed for reinsurance pricing
    """
    
    def __init__(self, config: Optional[DataGenerationConfig] = None):
        self.config = config or DataGenerationConfig()
        self.setup_reference_data()
        
    def setup_reference_data(self):
        """Initialize reference data for realistic generation"""
        
        # US State data with populations
        self.states = {
            'CA': {'name': 'California', 'pop_weight': 0.12, 'mortality_adj': 0.95},
            'TX': {'name': 'Texas', 'pop_weight': 0.09, 'mortality_adj': 1.02},
            'FL': {'name': 'Florida', 'pop_weight': 0.065, 'mortality_adj': 0.98},
            'NY': {'name': 'New York', 'pop_weight': 0.058, 'mortality_adj': 0.93},
            'PA': {'name': 'Pennsylvania', 'pop_weight': 0.038, 'mortality_adj': 1.01},
            'IL': {'name': 'Illinois', 'pop_weight': 0.038, 'mortality_adj': 0.97},
            'OH': {'name': 'Ohio', 'pop_weight': 0.035, 'mortality_adj': 1.03},
            'GA': {'name': 'Georgia', 'pop_weight': 0.032, 'mortality_adj': 1.04},
            'NC': {'name': 'North Carolina', 'pop_weight': 0.032, 'mortality_adj': 1.01},
            'MI': {'name': 'Michigan', 'pop_weight': 0.030, 'mortality_adj': 1.02}
        }
        
        # Insurance companies with market characteristics
        self.companies = [
            {'name': 'MetLife', 'size': 'Large', 'specialty': 'Group Life', 'risk_appetite': 'Conservative'},
            {'name': 'Prudential', 'size': 'Large', 'specialty': 'Individual Life', 'risk_appetite': 'Moderate'},
            {'name': 'New York Life', 'size': 'Large', 'specialty': 'Whole Life', 'risk_appetite': 'Conservative'},
            {'name': 'Northwestern Mutual', 'size': 'Large', 'specialty': 'Whole Life', 'risk_appetite': 'Conservative'},
            {'name': 'MassMutual', 'size': 'Medium', 'specialty': 'Universal Life', 'risk_appetite': 'Moderate'},
            {'name': 'Guardian Life', 'size': 'Medium', 'specialty': 'Term Life', 'risk_appetite': 'Moderate'},
            {'name': 'Lincoln Financial', 'size': 'Medium', 'specialty': 'Variable Life', 'risk_appetite': 'Aggressive'},
            {'name': 'Principal Financial', 'size': 'Medium', 'specialty': 'Pension', 'risk_appetite': 'Moderate'},
            {'name': 'Ameritas', 'size': 'Small', 'specialty': 'Term Life', 'risk_appetite': 'Aggressive'},
            {'name': 'Security Benefit', 'size': 'Small', 'specialty': 'Annuities', 'risk_appetite': 'Aggressive'}
        ]
        
        # Medical conditions with prevalence and mortality impact
        self.medical_conditions = {
            'DIABETES': {'prevalence': 0.11, 'mortality_factor': 1.5, 'age_correlation': 0.7},
            'HYPERTENSION': {'prevalence': 0.45, 'mortality_factor': 1.2, 'age_correlation': 0.8},
            'HEART_DISEASE': {'prevalence': 0.065, 'mortality_factor': 2.0, 'age_correlation': 0.9},
            'CANCER_HISTORY': {'prevalence': 0.04, 'mortality_factor': 1.8, 'age_correlation': 0.6},
            'DEPRESSION': {'prevalence': 0.08, 'mortality_factor': 1.3, 'age_correlation': 0.2},
            'OBESITY': {'prevalence': 0.36, 'mortality_factor': 1.4, 'age_correlation': 0.3},
            'ASTHMA': {'prevalence': 0.08, 'mortality_factor': 1.1, 'age_correlation': -0.2},
            'ARTHRITIS': {'prevalence': 0.23, 'mortality_factor': 1.05, 'age_correlation': 0.9}
        }
        
        # Economic cycles and their impact
        self.economic_periods = [
            {'start': date(2010, 1, 1), 'end': date(2012, 12, 31), 'cycle': 'Recovery', 'lapse_adj': 1.2},
            {'start': date(2013, 1, 1), 'end': date(2019, 12, 31), 'cycle': 'Expansion', 'lapse_adj': 0.8},
            {'start': date(2020, 1, 1), 'end': date(2021, 12, 31), 'cycle': 'Recession', 'lapse_adj': 1.5},
            {'start': date(2022, 1, 1), 'end': date(2023, 12, 31), 'cycle': 'Recovery', 'lapse_adj': 1.1}
        ]
        
    def generate_comprehensive_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete actuarial dataset"""
        
        print("🚀 Generating Comprehensive Actuarial Dataset...")
        print(f"Target size: {self.config.n_policies:,} policies")
        
        # Core policy data
        print("📋 Generating policy data...")
        policy_data = self._generate_policy_data()
        
        # Mortality experience
        print("💀 Generating mortality experience...")
        mortality_data = self._generate_mortality_experience(policy_data)
        
        # Lapse history
        print("📉 Generating lapse history...")
        lapse_data = self._generate_lapse_history(policy_data)
        
        # Claims data
        if self.config.include_claims:
            print("🏥 Generating claims data...")
            claims_data = self._generate_claims_data(policy_data)
        else:
            claims_data = pd.DataFrame()
        
        # Medical underwriting
        if self.config.include_medical_data:
            print("🩺 Generating medical data...")
            medical_data = self._generate_medical_data(policy_data)
        else:
            medical_data = pd.DataFrame()
        
        # Economic data
        if self.config.include_economic_data:
            print("💰 Generating economic scenarios...")
            economic_data = self._generate_economic_data()
        else:
            economic_data = pd.DataFrame()
        
        # Geographic data
        if self.config.include_geographic_data:
            print("🗺️ Generating geographic data...")
            geographic_data = self._generate_geographic_data(policy_data)
        else:
            geographic_data = pd.DataFrame()
        
        # Reinsurance treaties
        print("📜 Generating reinsurance treaties...")
        treaty_data = self._generate_treaty_data()
        
        # Add data quality issues if requested
        if self.config.data_quality_issues:
            print("🔧 Adding realistic data quality issues...")
            policy_data = self._add_data_quality_issues(policy_data)
        
        print("✅ Dataset generation complete!")
        
        return {
            'policy_data': policy_data,
            'mortality_experience': mortality_data,
            'lapse_history': lapse_data,
            'claims_data': claims_data,
            'medical_data': medical_data,
            'economic_data': economic_data,
            'geographic_data': geographic_data,
            'treaty_data': treaty_data
        }
    
    def _generate_policy_data(self) -> pd.DataFrame:
        """Generate core policy data"""
        
        np.random.seed(42)  # For reproducibility
        
        n = self.config.n_policies
        
        # Basic demographics with realistic distributions
        ages = np.random.beta(2, 3, n) * 60 + 18  # Skewed towards younger ages
        issue_ages = ages.astype(int)
        
        # Gender distribution (slightly more female in life insurance)
        genders = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
        
        # Product type distribution based on market data
        product_weights = [0.35, 0.25, 0.15, 0.08, 0.05, 0.05, 0.04, 0.03]
        products = np.random.choice(
            [p.value for p in ProductType], 
            n, 
            p=product_weights
        )
        
        # Issue dates spread over time period
        start_timestamp = datetime.combine(self.config.start_date, datetime.min.time()).timestamp()
        end_timestamp = datetime.combine(self.config.end_date, datetime.min.time()).timestamp()
        issue_timestamps = np.random.uniform(start_timestamp, end_timestamp, n)
        issue_dates = [datetime.fromtimestamp(ts).date() for ts in issue_timestamps]
        
        # Face amounts based on product type and demographics
        base_amounts = np.random.lognormal(11.5, 0.8, n)  # Mean ~$100K
        
        # Adjust by product type
        product_adjustments = {
            'TERM': 1.2,
            'WHOLE_LIFE': 0.8,
            'UNIVERSAL_LIFE': 1.0,
            'VARIABLE_LIFE': 1.5,
            'IMMEDIATE_ANNUITY': 2.0,
            'DEFERRED_ANNUITY': 1.5,
            'PENSION': 3.0,
            'GROUP_LIFE': 0.5
        }
        
        face_amounts = []
        for i, product in enumerate(products):
            adjustment = product_adjustments.get(product, 1.0)
            amount = base_amounts[i] * adjustment
            face_amounts.append(max(10000, min(50000000, amount)))  # Cap between $10K and $50M
        
        # Premium calculations (simplified but realistic)
        annual_premiums = []
        for i in range(n):
            age = issue_ages[i]
            gender = genders[i]
            product = products[i]
            face_amount = face_amounts[i]
            
            # Base mortality cost per $1000
            if gender == 'M':
                base_cost = 0.8 + (age - 25) * 0.02
            else:
                base_cost = 0.6 + (age - 25) * 0.018
            
            # Product-specific adjustments
            if product == 'TERM':
                cost_factor = 1.0
            elif product == 'WHOLE_LIFE':
                cost_factor = 3.5
            elif product == 'UNIVERSAL_LIFE':
                cost_factor = 2.8
            else:
                cost_factor = 2.0
            
            premium = face_amount * base_cost * cost_factor / 1000
            annual_premiums.append(max(100, premium))
        
        # Smoker status (affects pricing significantly)
        smoker_status = np.random.choice(['NS', 'S'], n, p=[0.85, 0.15])
        
        # Underwriting class
        underwriting_classes = np.random.choice(
            ['PREFERRED_PLUS', 'PREFERRED', 'STANDARD_PLUS', 'STANDARD', 'SUBSTANDARD'],
            n, 
            p=[0.15, 0.25, 0.30, 0.25, 0.05]
        )
        
        # Policy status
        # Most policies are active, but some have lapsed/matured/claimed
        status_probs = [0.75, 0.15, 0.05, 0.02, 0.02, 0.01]
        policy_statuses = np.random.choice(
            [s.value for s in PolicyStatus], 
            n, 
            p=status_probs
        )
        
        # Company assignment
        company_names = [comp['name'] for comp in self.companies]
        company_weights = [0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.105, 0.105]
        companies = np.random.choice(company_names, n, p=company_weights)
        
        # Create policy IDs
        policy_ids = [f'POL_{i:08d}' for i in range(1, n + 1)]
        
        # Build dataframe
        policy_df = pd.DataFrame({
            'policy_id': policy_ids,
            'company': companies,
            'product_type': products,
            'issue_date': issue_dates,
            'issue_age': issue_ages,
            'current_age': issue_ages + np.random.randint(0, 5, n),  # Some aging
            'gender': genders,
            'smoker_status': smoker_status,
            'underwriting_class': underwriting_classes,
            'face_amount': face_amounts,
            'annual_premium': annual_premiums,
            'policy_status': policy_statuses,
            'policy_year': np.random.randint(1, 15, n)
        })
        
        # Add derived fields
        policy_df['premium_per_thousand'] = (policy_df['annual_premium'] / policy_df['face_amount'] * 1000).round(2)
        
        # Calculate maturity dates individually
        maturity_dates = []
        for _, row in policy_df.iterrows():
            issue_date = pd.to_datetime(row['issue_date'])
            if row['product_type'] == 'TERM':
                years_to_add = 20
            elif row['product_type'] in ['WHOLE_LIFE', 'UNIVERSAL_LIFE']:
                years_to_add = 100 - row['issue_age']
            else:
                years_to_add = 65 - row['issue_age']
            
            maturity_date = issue_date + pd.DateOffset(years=int(years_to_add))
            maturity_dates.append(maturity_date)
        
        policy_df['maturity_date'] = maturity_dates
        
        return policy_df.round(2)
    
    def _generate_mortality_experience(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic mortality experience data"""
        
        # Select subset of policies for mortality experience
        experience_policies = policy_df.sample(frac=0.6, random_state=42)
        
        mortality_data = []
        
        for _, policy in experience_policies.iterrows():
            # Calculate expected mortality based on standard tables
            age = policy['current_age']
            gender = policy['gender']
            smoker = policy['smoker_status']
            
            # Base qx (probability of death) - simplified 2017 CSO
            if gender == 'M':
                base_qx = 0.0005 * (1.1 ** (age - 20)) if age >= 20 else 0.0001
            else:
                base_qx = 0.0003 * (1.08 ** (age - 20)) if age >= 20 else 0.0001
            
            # Smoker adjustment
            if smoker == 'S':
                base_qx *= 2.5
            
            # Underwriting class adjustment
            uw_class = policy['underwriting_class']
            uw_adjustments = {
                'PREFERRED_PLUS': 0.6,
                'PREFERRED': 0.8,
                'STANDARD_PLUS': 0.9,
                'STANDARD': 1.0,
                'SUBSTANDARD': 2.0
            }
            
            adjusted_qx = base_qx * uw_adjustments.get(uw_class, 1.0)
            
            # Generate actual deaths (Poisson process)
            expected_deaths = adjusted_qx * policy['face_amount'] / 100000  # Per $100K
            actual_deaths = np.random.poisson(expected_deaths)
            
            mortality_data.append({
                'policy_id': policy['policy_id'],
                'experience_year': 2023,
                'age_band': f"{int(age//5)*5}-{int(age//5)*5+4}",
                'expected_deaths': expected_deaths,
                'actual_deaths': actual_deaths,
                'exposure_amount': policy['face_amount'],
                'ae_ratio': actual_deaths / expected_deaths if expected_deaths > 0 else 0
            })
        
        return pd.DataFrame(mortality_data)
    
    def _generate_lapse_history(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate lapse behavior data"""
        
        lapse_data = []
        
        # Select policies that could have lapsed
        active_policies = policy_df[policy_df['policy_status'].isin(['ACTIVE', 'LAPSED'])].copy()
        
        for _, policy in active_policies.iterrows():
            # Lapse probability depends on multiple factors
            base_lapse_prob = 0.08  # 8% base annual lapse rate
            
            # Product type impact
            product_lapse_adj = {
                'TERM': 1.5,
                'WHOLE_LIFE': 0.6,
                'UNIVERSAL_LIFE': 1.2,
                'VARIABLE_LIFE': 1.8,
                'IMMEDIATE_ANNUITY': 0.1,
                'DEFERRED_ANNUITY': 0.8
            }
            
            lapse_prob = base_lapse_prob * product_lapse_adj.get(policy['product_type'], 1.0)
            
            # Age impact (younger people lapse more)
            if policy['issue_age'] < 30:
                lapse_prob *= 1.4
            elif policy['issue_age'] > 50:
                lapse_prob *= 0.7
            
            # Premium affordability impact
            if policy['premium_per_thousand'] > 15:
                lapse_prob *= 1.6
            
            # Economic cycle impact
            issue_date = pd.to_datetime(policy['issue_date']).date()
            for period in self.economic_periods:
                if period['start'] <= issue_date <= period['end']:
                    lapse_prob *= period['lapse_adj']
                    break
            
            # Generate lapse outcome
            lapsed = np.random.random() < lapse_prob
            
            if lapsed or policy['policy_status'] == 'LAPSED':
                lapse_date = pd.to_datetime(policy['issue_date']) + pd.DateOffset(
                    months=np.random.randint(6, 60)
                )
                
                lapse_data.append({
                    'policy_id': policy['policy_id'],
                    'lapse_date': lapse_date.date(),
                    'lapse_year': policy['policy_year'],
                    'lapse_probability': lapse_prob,
                    'lapsed': True,
                    'surrender_value': policy['annual_premium'] * policy['policy_year'] * 0.6,
                    'lapse_reason': np.random.choice([
                        'FINANCIAL_HARDSHIP', 'POLICY_REPLACEMENT', 'DISSATISFACTION',
                        'COVERAGE_NO_LONGER_NEEDED', 'PREMIUM_INCREASE', 'OTHER'
                    ], p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05])
                })
            else:
                lapse_data.append({
                    'policy_id': policy['policy_id'],
                    'lapse_date': None,
                    'lapse_year': None,
                    'lapse_probability': lapse_prob,
                    'lapsed': False,
                    'surrender_value': None,
                    'lapse_reason': None
                })
        
        return pd.DataFrame(lapse_data)
    
    def _generate_claims_data(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive claims data"""
        
        claims_data = []
        
        # Generate death claims
        claimed_policies = policy_df[policy_df['policy_status'] == 'CLAIM'].copy()
        
        for _, policy in claimed_policies.iterrows():
            # Ensure valid date range for claim
            max_months = max(12, policy['policy_year'] * 12)
            min_months = min(12, max_months - 1)
            
            claim_date = pd.to_datetime(policy['issue_date']) + pd.DateOffset(
                months=np.random.randint(min_months, max_months + 1)
            )
            
            # Cause of death distribution
            causes = ['HEART_DISEASE', 'CANCER', 'ACCIDENT', 'STROKE', 'DIABETES', 
                     'PNEUMONIA', 'SUICIDE', 'KIDNEY_DISEASE', 'OTHER']
            cause_weights = [0.23, 0.21, 0.06, 0.05, 0.03, 0.02, 0.015, 0.015, 0.385]
            # Normalize to ensure sum = 1
            cause_weights = np.array(cause_weights)
            cause_weights = cause_weights / cause_weights.sum()
            cause = np.random.choice(causes, p=cause_weights)
            
            # Claim amount (usually face amount, but can vary)
            claim_multiplier = np.random.normal(1.0, 0.05)  # Small variation
            claim_amount = policy['face_amount'] * max(0.95, min(1.05, claim_multiplier))
            
            claims_data.append({
                'policy_id': policy['policy_id'],
                'claim_id': f"CLM_{len(claims_data)+1:08d}",
                'claim_date': claim_date.date(),
                'claim_type': 'DEATH',
                'cause_of_death': cause,
                'claim_amount': claim_amount,
                'paid_amount': claim_amount * np.random.uniform(0.98, 1.0),  # Small deductions
                'days_to_pay': np.random.gamma(2, 15),  # Average 30 days
                'claim_status': 'PAID',
                'investigation_required': cause in ['ACCIDENT', 'SUICIDE'] or claim_amount > 1000000
            })
        
        return pd.DataFrame(claims_data)
    
    def _generate_medical_data(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate medical underwriting data"""
        
        medical_data = []
        
        for _, policy in policy_df.iterrows():
            # Medical exam requirements based on age and face amount
            exam_required = (policy['issue_age'] > 45 or policy['face_amount'] > 500000)
            
            # Generate medical conditions
            conditions = []
            for condition, data in self.medical_conditions.items():
                # Age-correlated probability
                age_factor = 1 + (policy['issue_age'] - 40) * data['age_correlation'] * 0.02
                prob = data['prevalence'] * age_factor
                
                if np.random.random() < prob:
                    conditions.append(condition)
            
            # BMI generation
            if policy['gender'] == 'M':
                bmi = np.random.normal(28.5, 5.5)
            else:
                bmi = np.random.normal(27.2, 6.0)
            bmi = max(16, min(50, bmi))
            
            # Blood pressure
            systolic = np.random.normal(125, 20)
            diastolic = np.random.normal(80, 12)
            
            # Cholesterol
            total_cholesterol = np.random.normal(195, 35)
            hdl_cholesterol = np.random.normal(55, 15)
            
            medical_data.append({
                'policy_id': policy['policy_id'],
                'exam_required': exam_required,
                'bmi': round(bmi, 1),
                'blood_pressure_systolic': max(80, min(200, int(systolic))),
                'blood_pressure_diastolic': max(50, min(120, int(diastolic))),
                'total_cholesterol': max(120, min(400, int(total_cholesterol))),
                'hdl_cholesterol': max(25, min(100, int(hdl_cholesterol))),
                'medical_conditions': ','.join(conditions) if conditions else None,
                'medical_conditions_count': len(conditions),
                'family_history_heart_disease': np.random.choice([True, False], p=[0.3, 0.7]),
                'family_history_cancer': np.random.choice([True, False], p=[0.25, 0.75]),
                'exercise_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'], 
                                                     p=[0.1, 0.2, 0.3, 0.3, 0.1])
            })
        
        return pd.DataFrame(medical_data)
    
    def _generate_economic_data(self) -> pd.DataFrame:
        """Generate economic scenario data"""
        
        # Daily data from start to end date
        date_range = pd.date_range(self.config.start_date, self.config.end_date, freq='D')
        n_days = len(date_range)
        
        # Interest rates (with trend and volatility)
        base_rate = 0.035
        rates_10y = [base_rate]
        
        for i in range(1, n_days):
            # Mean reversion with volatility
            change = np.random.normal(-0.0001, 0.002)  # Slight downward trend
            new_rate = rates_10y[-1] + change
            rates_10y.append(max(0.001, min(0.08, new_rate)))
        
        # Other economic indicators
        unemployment_base = 0.05
        unemployment = []
        current_unemployment = unemployment_base
        
        for i in range(n_days):
            change = np.random.normal(0, 0.0005)
            current_unemployment += change
            current_unemployment = max(0.02, min(0.15, current_unemployment))
            unemployment.append(current_unemployment)
        
        # Stock market returns (daily)
        daily_returns = np.random.normal(0.0008, 0.012, n_days)  # ~8% annual, 19% vol
        
        # Inflation (monthly, interpolated to daily)
        monthly_inflation = np.random.normal(0.002, 0.003, n_days//30 + 1)
        inflation = np.interp(range(n_days), range(0, n_days, 30), monthly_inflation)
        
        economic_df = pd.DataFrame({
            'date': date_range,
            'interest_rate_10y': rates_10y,
            'interest_rate_1y': [r * 0.8 + np.random.normal(0, 0.001) for r in rates_10y],
            'interest_rate_30y': [r * 1.3 + np.random.normal(0, 0.002) for r in rates_10y],
            'unemployment_rate': unemployment,
            'sp500_daily_return': daily_returns,
            'inflation_rate': inflation,
            'gdp_growth_quarterly': np.random.normal(0.005, 0.01, n_days),  # Quarterly rate
            'vix_level': np.random.gamma(2, 10, n_days),  # Volatility index
            'credit_spread': np.random.gamma(1.5, 0.8, n_days) / 100  # Credit spreads
        })
        
        return economic_df.round(6)
    
    def _generate_geographic_data(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate geographic distribution data"""
        
        geographic_data = []
        
        # Assign states based on population weights
        state_codes = list(self.states.keys())
        state_weights = [self.states[s]['pop_weight'] for s in state_codes]
        # Normalize weights
        total_weight = sum(state_weights)
        state_weights = [w/total_weight for w in state_weights]
        
        for _, policy in policy_df.iterrows():
            state = np.random.choice(state_codes, p=state_weights)
            state_info = self.states[state]
            
            # ZIP codes (simplified - just generate realistic looking ones)
            if state == 'CA':
                zip_code = f"9{np.random.randint(0, 9)}{np.random.randint(100, 999)}"
            elif state == 'NY':
                zip_code = f"1{np.random.randint(0, 9)}{np.random.randint(100, 999)}"
            elif state == 'FL':
                zip_code = f"3{np.random.randint(0, 9)}{np.random.randint(100, 999)}"
            else:
                zip_code = f"{np.random.randint(10, 99)}{np.random.randint(100, 999)}"
            
            # Urban vs rural (affects mortality and lapse rates)
            urban = np.random.choice([True, False], p=[0.82, 0.18])  # 82% urban in US
            
            geographic_data.append({
                'policy_id': policy['policy_id'],
                'state_code': state,
                'state_name': state_info['name'],
                'zip_code': zip_code,
                'urban_flag': urban,
                'mortality_adjustment': state_info['mortality_adj'],
                'region': self._get_region(state),
                'cost_of_living_index': np.random.normal(100, 25)
            })
        
        return pd.DataFrame(geographic_data)
    
    def _generate_treaty_data(self) -> pd.DataFrame:
        """Generate reinsurance treaty data"""
        
        treaties = []
        
        # Generate different types of treaties
        treaty_types = ['QUOTA_SHARE', 'SURPLUS', 'XS_OF_LOSS', 'CATASTROPHE']
        
        for i in range(50):  # 50 different treaties
            treaty_type = np.random.choice(treaty_types)
            
            if treaty_type == 'QUOTA_SHARE':
                retention = np.random.uniform(0.1, 0.5)  # 10-50% retention
                limit = None
            elif treaty_type == 'SURPLUS':
                retention = np.random.choice([100000, 250000, 500000, 1000000])
                limit = retention * np.random.randint(5, 20)
            else:  # XS or CAT
                retention = np.random.choice([1000000, 2500000, 5000000, 10000000])
                limit = retention * np.random.randint(2, 10)
            
            treaties.append({
                'treaty_id': f"TRT_{i+1:03d}",
                'treaty_type': treaty_type,
                'effective_date': self.config.start_date + timedelta(days=np.random.randint(0, 1000)),
                'expiry_date': self.config.end_date,
                'retention_amount': retention,
                'treaty_limit': limit,
                'reinsurer': np.random.choice(['Swiss Re', 'Munich Re', 'Gen Re', 'RGA', 'SCOR']),
                'commission_rate': np.random.uniform(0.15, 0.35) if treaty_type in ['QUOTA_SHARE', 'SURPLUS'] else 0,
                'profit_commission': np.random.choice([True, False], p=[0.6, 0.4]),
                'territory': 'USA',
                'currency': 'USD'
            })
        
        return pd.DataFrame(treaties)
    
    def _add_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic data quality issues"""
        
        df_with_issues = df.copy()
        n_rows = len(df_with_issues)
        
        # Add missing values
        missing_columns = ['current_age', 'smoker_status', 'underwriting_class']
        for col in missing_columns:
            if col in df_with_issues.columns:
                n_missing = int(n_rows * self.config.missing_data_pct)
                missing_idx = np.random.choice(df_with_issues.index, n_missing, replace=False)
                df_with_issues.loc[missing_idx, col] = np.nan
        
        # Add outliers
        n_outliers = int(n_rows * self.config.outlier_pct)
        outlier_idx = np.random.choice(df_with_issues.index, n_outliers, replace=False)
        df_with_issues.loc[outlier_idx, 'face_amount'] *= np.random.uniform(5, 20)  # Extreme values
        
        # Add duplicates
        n_duplicates = int(n_rows * self.config.duplicate_pct)
        duplicate_idx = np.random.choice(df_with_issues.index, n_duplicates, replace=False)
        duplicates = df_with_issues.loc[duplicate_idx].copy()
        # Slightly modify policy IDs to make them "near duplicates"
        duplicates['policy_id'] = duplicates['policy_id'] + '_DUP'
        df_with_issues = pd.concat([df_with_issues, duplicates], ignore_index=True)
        
        # Add inconsistent formats
        # Gender variants
        gender_variants = {'M': ['M', 'Male', 'MALE', '1'], 'F': ['F', 'Female', 'FEMALE', '2']}
        for idx, row in df_with_issues.iterrows():
            if np.random.random() < 0.05:  # 5% chance of format variation
                original_gender = row['gender']
                if original_gender in gender_variants:
                    df_with_issues.loc[idx, 'gender'] = np.random.choice(gender_variants[original_gender])
        
        return df_with_issues
    
    def _get_region(self, state_code: str) -> str:
        """Get US region for state"""
        regions = {
            'CA': 'West', 'TX': 'South', 'FL': 'South', 'NY': 'Northeast',
            'PA': 'Northeast', 'IL': 'Midwest', 'OH': 'Midwest', 'GA': 'South',
            'NC': 'South', 'MI': 'Midwest'
        }
        return regions.get(state_code, 'Other')
    
    def generate_summary_report(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """Generate summary report of generated data"""
        
        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE ACTUARIAL DATA GENERATION REPORT")
        report.append("=" * 70)
        
        total_records = sum(len(df) for df in datasets.values())
        report.append(f"\n📊 DATASET OVERVIEW")
        report.append(f"Total Records Generated: {total_records:,}")
        report.append(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append(f"\n📋 DATASET BREAKDOWN")
        for name, df in datasets.items():
            if not df.empty:
                report.append(f"• {name.replace('_', ' ').title()}: {len(df):,} records")
        
        # Policy data details
        if 'policy_data' in datasets and not datasets['policy_data'].empty:
            policy_df = datasets['policy_data']
            report.append(f"\n🏛️ POLICY DATA DETAILS")
            report.append(f"• Face Amount Range: ${policy_df['face_amount'].min():,.0f} - ${policy_df['face_amount'].max():,.0f}")
            report.append(f"• Average Premium: ${policy_df['annual_premium'].mean():,.0f}")
            report.append(f"• Age Range: {policy_df['issue_age'].min()} - {policy_df['issue_age'].max()}")
            
            product_dist = policy_df['product_type'].value_counts()
            report.append(f"• Product Mix:")
            for product, count in product_dist.head().items():
                pct = count / len(policy_df) * 100
                report.append(f"  - {product}: {count:,} ({pct:.1f}%)")
        
        # Data quality info
        if self.config.data_quality_issues:
            report.append(f"\n⚠️ DATA QUALITY ISSUES ADDED")
            report.append(f"• Missing Data: {self.config.missing_data_pct*100:.1f}%")
            report.append(f"• Outliers: {self.config.outlier_pct*100:.1f}%")
            report.append(f"• Duplicates: {self.config.duplicate_pct*100:.1f}%")
            report.append("• Format Inconsistencies: 5% of categorical fields")
        
        report.append("\n✅ DATA GENERATION COMPLETE!")
        report.append("Ready for actuarial validation and ML-enhanced pricing")
        report.append("=" * 70)
        
        return "\n".join(report)

def quick_generate_test_data(size: str = "medium") -> Dict[str, pd.DataFrame]:
    """Quick function to generate test data"""
    
    size_configs = {
        "small": DataGenerationConfig(n_policies=1000, n_companies=5),
        "medium": DataGenerationConfig(n_policies=10000, n_companies=15), 
        "large": DataGenerationConfig(n_policies=50000, n_companies=25),
        "xlarge": DataGenerationConfig(n_policies=100000, n_companies=50)
    }
    
    config = size_configs.get(size, size_configs["medium"])
    generator = ComprehensiveActuarialDataGenerator(config)
    
    return generator.generate_comprehensive_dataset()

if __name__ == "__main__":
    # Generate comprehensive test dataset
    config = DataGenerationConfig(n_policies=5000, data_quality_issues=True)
    generator = ComprehensiveActuarialDataGenerator(config)
    
    datasets = generator.generate_comprehensive_dataset()
    print(generator.generate_summary_report(datasets))
    
    # Save to files
    for name, df in datasets.items():
        if not df.empty:
            filename = f"data/generated_{name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")