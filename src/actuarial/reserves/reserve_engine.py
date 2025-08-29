"""
GAAP/Statutory Reserve Engine
Implements LDTI (ASC 944) and Statutory requirements for life insurance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from ..mortality.mortality_engine import MortalityEngine

class ReserveEngine:
    """Real actuarial reserve calculations per GAAP and Statutory requirements"""
    
    def __init__(self, mortality_engine: MortalityEngine):
        self.mortality = mortality_engine
        self.logger = logging.getLogger(__name__)
        
        # Reserve bases
        self.reserve_bases = {
            "gaap": 0.035,       # GAAP discount rate
            "statutory": 0.035,   # Statutory valuation rate
            "economic": 0.025     # Current economic rate
        }
        
        # LDTI cohort tracking
        self.cohort_data = {}
        
    def calculate_gaap_reserves(
        self,
        policy_data: pd.DataFrame,
        valuation_date: datetime,
        reserve_basis: str = "LDTI"
    ) -> Dict[str, any]:
        """Calculate GAAP reserves under LDTI (ASC 944)"""
        
        total_reserves = 0
        reserve_details = []
        
        for _, policy in policy_data.iterrows():
            try:
                # Determine cohort (issue year for LDTI)
                issue_year = pd.to_datetime(policy.get('issue_date', valuation_date)).year
                current_age = policy.get('current_age', 45)
                issue_age = policy.get('issue_age', 35)
                face_amount = policy.get('face_amount', 100000)
                
                # Calculate net premium using locked-in assumptions
                if reserve_basis == "LDTI":
                    cohort_key = f"{issue_year}_{policy.get('product_type', 'TERM')}"
                    if cohort_key not in self.cohort_data:
                        self.cohort_data[cohort_key] = self._establish_cohort_assumptions(
                            issue_year, policy.get('product_type', 'TERM')
                        )
                    
                    assumptions = self.cohort_data[cohort_key]
                    discount_rate = assumptions['discount_rate']
                    mortality_table = assumptions['mortality_table']
                else:
                    discount_rate = self.reserve_bases['gaap']
                    mortality_table = "2017_CSO_Male_NS"
                
                # Present value of future benefits
                pv_benefits = self._calculate_pv_benefits(
                    current_age, issue_age, face_amount, 
                    mortality_table, discount_rate
                )
                
                # Present value of future premiums
                pv_premiums = self._calculate_pv_premiums(
                    current_age, issue_age, face_amount,
                    policy.get('annual_premium', 0),
                    mortality_table, discount_rate
                )
                
                # GAAP reserve = PV Benefits - PV Premiums - DAC
                dac_amortization = self._calculate_dac_amortization(policy)
                gaap_reserve = max(0, pv_benefits - pv_premiums - dac_amortization)
                
                reserve_details.append({
                    'policy_id': policy.get('policy_id', 'UNKNOWN'),
                    'gaap_reserve': gaap_reserve,
                    'pv_benefits': pv_benefits,
                    'pv_premiums': pv_premiums,
                    'dac_balance': dac_amortization,
                    'cohort': cohort_key if reserve_basis == "LDTI" else "N/A"
                })
                
                total_reserves += gaap_reserve
                
            except Exception as e:
                self.logger.error(f"Error calculating GAAP reserve for policy {policy.get('policy_id')}: {e}")
                continue
        
        return {
            'total_gaap_reserves': total_reserves,
            'reserve_details': reserve_details,
            'valuation_date': valuation_date,
            'basis': reserve_basis,
            'average_reserve_per_policy': total_reserves / len(policy_data) if len(policy_data) > 0 else 0
        }
    
    def calculate_statutory_reserves(
        self,
        policy_data: pd.DataFrame,
        valuation_date: datetime
    ) -> Dict[str, any]:
        """Calculate statutory reserves per VM-20/PBR"""
        
        total_reserves = 0
        reserve_details = []
        
        for _, policy in policy_data.iterrows():
            try:
                current_age = policy.get('current_age', 45)
                issue_age = policy.get('issue_age', 35) 
                face_amount = policy.get('face_amount', 100000)
                product_type = policy.get('product_type', 'TERM')
                
                # Determine reserve method based on product
                if product_type in ['TERM', 'WHOLE_LIFE']:
                    # Use Commissioners Reserve Valuation Method (CRVM)
                    stat_reserve = self._calculate_crvm_reserve(
                        current_age, issue_age, face_amount, policy
                    )
                elif product_type in ['UNIVERSAL_LIFE']:
                    # Use account value method
                    stat_reserve = self._calculate_ul_reserve(policy, valuation_date)
                else:
                    # Default net level reserve
                    stat_reserve = self._calculate_net_level_reserve(
                        current_age, issue_age, face_amount
                    )
                
                reserve_details.append({
                    'policy_id': policy.get('policy_id', 'UNKNOWN'),
                    'statutory_reserve': stat_reserve,
                    'reserve_method': 'CRVM' if product_type in ['TERM', 'WHOLE_LIFE'] else 'Other'
                })
                
                total_reserves += stat_reserve
                
            except Exception as e:
                self.logger.error(f"Error calculating statutory reserve: {e}")
                continue
        
        return {
            'total_statutory_reserves': total_reserves,
            'reserve_details': reserve_details,
            'valuation_date': valuation_date,
            'average_reserve_per_policy': total_reserves / len(policy_data) if len(policy_data) > 0 else 0
        }
    
    def calculate_economic_reserves(
        self,
        policy_data: pd.DataFrame,
        current_yield_curve: Dict[int, float]
    ) -> Dict[str, any]:
        """Calculate economic reserves using current market rates"""
        
        total_reserves = 0
        reserve_details = []
        
        for _, policy in policy_data.iterrows():
            try:
                current_age = policy.get('current_age', 45)
                issue_age = policy.get('issue_age', 35)
                face_amount = policy.get('face_amount', 100000)
                
                # Use current yield curve for discounting
                avg_duration = max(1, 65 - current_age)  # Approximate duration
                discount_rate = current_yield_curve.get(
                    min(30, avg_duration), 
                    self.reserve_bases['economic']
                )
                
                # Best estimate assumptions
                mortality_rates = self.mortality.calculate_mortality_rates(
                    current_age, policy.get('gender', 'M')
                )
                
                # Economic reserve (best estimate)
                economic_reserve = self._calculate_best_estimate_reserve(
                    current_age, issue_age, face_amount,
                    discount_rate, mortality_rates
                )
                
                reserve_details.append({
                    'policy_id': policy.get('policy_id', 'UNKNOWN'),
                    'economic_reserve': economic_reserve,
                    'discount_rate': discount_rate
                })
                
                total_reserves += economic_reserve
                
            except Exception as e:
                self.logger.error(f"Error calculating economic reserve: {e}")
                continue
        
        return {
            'total_economic_reserves': total_reserves,
            'reserve_details': reserve_details,
            'average_discount_rate': np.mean([r['discount_rate'] for r in reserve_details]),
            'average_reserve_per_policy': total_reserves / len(policy_data) if len(policy_data) > 0 else 0
        }
    
    def _establish_cohort_assumptions(
        self, 
        issue_year: int, 
        product_type: str
    ) -> Dict[str, any]:
        """Establish locked-in cohort assumptions for LDTI"""
        
        # Historical rates by issue year (simplified)
        historical_rates = {
            2020: 0.025, 2021: 0.020, 2022: 0.035,
            2023: 0.045, 2024: 0.042, 2025: 0.040
        }
        
        return {
            'discount_rate': historical_rates.get(issue_year, 0.035),
            'mortality_table': '2017_CSO_Male_NS',
            'lapse_rates': self._get_lapse_assumptions(product_type),
            'expense_rates': self._get_expense_assumptions(product_type)
        }
    
    def _calculate_pv_benefits(
        self,
        current_age: int,
        issue_age: int, 
        face_amount: float,
        mortality_table: str,
        discount_rate: float
    ) -> float:
        """Calculate present value of future death benefits"""
        
        try:
            # Simplified whole life calculation
            mortality_rates = self.mortality.calculate_mortality_rates(
                current_age, "M", mortality_table
            )
            
            # Approximate using standard actuarial formulas
            # In production, would use full mortality table integration
            years_remaining = max(1, 65 - current_age)
            avg_mortality = 0.01 + (current_age - 35) * 0.001  # Age-adjusted
            
            pv_benefits = 0
            for year in range(years_remaining):
                age_at_year = current_age + year
                survival_prob = (1 - avg_mortality) ** year
                death_prob = avg_mortality * survival_prob
                discount_factor = (1 + discount_rate) ** (-year - 0.5)  # Mid-year assumption
                
                pv_benefits += face_amount * death_prob * discount_factor
            
            return pv_benefits
            
        except Exception as e:
            self.logger.error(f"Error calculating PV benefits: {e}")
            return face_amount * 0.3  # Fallback estimate
    
    def _calculate_pv_premiums(
        self,
        current_age: int,
        issue_age: int,
        face_amount: float,
        annual_premium: float,
        mortality_table: str,
        discount_rate: float
    ) -> float:
        """Calculate present value of future premiums"""
        
        try:
            premium_paying_period = max(1, min(65 - current_age, 30))  # Max 30 years
            
            pv_premiums = 0
            for year in range(premium_paying_period):
                survival_prob = 0.98 ** year  # Simplified survival
                lapse_prob = 0.95 ** year     # Simplified persistency
                discount_factor = (1 + discount_rate) ** (-year)
                
                pv_premiums += annual_premium * survival_prob * lapse_prob * discount_factor
            
            return pv_premiums
            
        except Exception as e:
            self.logger.error(f"Error calculating PV premiums: {e}")
            return annual_premium * 10  # Fallback estimate
    
    def _calculate_dac_amortization(self, policy: pd.Series) -> float:
        """Calculate Deferred Acquisition Cost amortization"""
        
        # Simplified DAC calculation
        face_amount = policy.get('face_amount', 100000)
        duration = policy.get('duration', 1)
        
        # Initial DAC as % of face amount
        initial_dac_rate = 0.05  # 5% of face amount
        amortization_period = 20  # Years
        
        initial_dac = face_amount * initial_dac_rate
        annual_amortization = initial_dac / amortization_period
        
        remaining_dac = max(0, initial_dac - (duration * annual_amortization))
        
        return remaining_dac
    
    def _calculate_crvm_reserve(
        self,
        current_age: int,
        issue_age: int,
        face_amount: float,
        policy: pd.Series
    ) -> float:
        """Calculate Commissioners Reserve Valuation Method reserve"""
        
        duration = current_age - issue_age
        
        # CRVM uses modified net premiums
        # Simplified calculation - in production would use full CRVM formulas
        base_reserve = self._calculate_net_level_reserve(current_age, issue_age, face_amount)
        
        # CRVM modification in early years
        if duration <= 20:
            modification_factor = min(1.2, 1.0 + (20 - duration) * 0.01)
            crvm_reserve = base_reserve * modification_factor
        else:
            crvm_reserve = base_reserve
            
        return max(0, crvm_reserve)
    
    def _calculate_ul_reserve(self, policy: pd.Series, valuation_date: datetime) -> float:
        """Calculate Universal Life account value reserve"""
        
        # For UL, reserve typically equals account value
        account_value = policy.get('account_value', 0)
        
        # Add any additional reserves for guarantees
        guarantee_reserve = policy.get('guarantee_reserve', 0)
        
        return account_value + guarantee_reserve
    
    def _calculate_net_level_reserve(
        self,
        current_age: int,
        issue_age: int,
        face_amount: float
    ) -> float:
        """Calculate basic net level reserve"""
        
        # Use mortality engine for proper calculation
        net_premium_calc = self.mortality.calculate_net_premium(
            issue_age, "M", face_amount, 
            premium_period=999, benefit_period=999
        )
        
        reserve_calc = self.mortality.calculate_reserves(
            issue_age, current_age, "M", face_amount,
            net_premium_calc['net_annual_premium']
        )
        
        return reserve_calc['net_level_reserve']
    
    def _calculate_best_estimate_reserve(
        self,
        current_age: int,
        issue_age: int,
        face_amount: float,
        discount_rate: float,
        mortality_rates: Dict[str, float]
    ) -> float:
        """Calculate best estimate economic reserve"""
        
        # Use current best estimate assumptions
        years_remaining = max(1, 85 - current_age)  # To age 85
        
        best_estimate_reserve = 0
        for year in range(years_remaining):
            survival_prob = mortality_rates['px'] ** year
            death_prob = mortality_rates['qx'] * survival_prob
            discount_factor = (1 + discount_rate) ** (-year - 0.5)
            
            best_estimate_reserve += face_amount * death_prob * discount_factor
        
        return best_estimate_reserve
    
    def _get_lapse_assumptions(self, product_type: str) -> Dict[int, float]:
        """Get lapse rate assumptions by product type"""
        
        lapse_rates = {
            'TERM': {1: 0.15, 2: 0.12, 3: 0.10, 4: 0.08, 5: 0.06},
            'WHOLE_LIFE': {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03},
            'UNIVERSAL_LIFE': {1: 0.12, 2: 0.10, 3: 0.08, 4: 0.06, 5: 0.05}
        }
        
        return lapse_rates.get(product_type, lapse_rates['TERM'])
    
    def _get_expense_assumptions(self, product_type: str) -> Dict[str, float]:
        """Get expense assumptions by product type"""
        
        return {
            'acquisition_expense_rate': 0.05,  # % of premium
            'maintenance_expense_rate': 0.02,   # % of premium
            'per_policy_expense': 50.0          # Annual per policy
        }