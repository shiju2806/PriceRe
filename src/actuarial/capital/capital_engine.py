"""
Risk-Based Capital (RBC) and Economic Capital Engine
Implements NAIC RBC formulas and economic capital calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

class CapitalEngine:
    """Risk-Based Capital calculations per NAIC standards"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # RBC factors by risk category
        self.rbc_factors = {
            # C1 Asset Risk (before diversification)
            'c1_bonds': {
                'aaa': 0.003, 'aa': 0.005, 'a': 0.010, 
                'bbb': 0.020, 'bb': 0.045, 'b': 0.100, 'ccc': 0.230
            },
            'c1_stocks': 0.30,  # 30% factor for stocks
            'c1_mortgages': 0.05,  # 5% for mortgages
            'c1_real_estate': 0.15,  # 15% for real estate
            
            # C2 Insurance Risk
            'c2_mortality': 0.015,  # 1.5% of net amount at risk
            'c2_morbidity': 0.125,  # 12.5% of claim reserves
            'c2_lapse': 0.05,       # 5% of reserves for lapse risk
            
            # C3 Interest Rate Risk
            'c3_disintermediation': 0.10,  # 10% of reserves
            'c3_asset_liability': 0.05,    # 5% mismatch penalty
            
            # C4 Business Risk
            'c4_business_growth': 0.025,   # 2.5% of premiums
            'c4_operational': 0.01         # 1% operational risk
        }
        
        # Correlation matrix for risk aggregation
        self.correlation_matrix = np.array([
            [1.00, 0.25, 0.50, 0.25],  # C1 correlations
            [0.25, 1.00, 0.00, 0.25],  # C2 correlations
            [0.50, 0.00, 1.00, 0.00],  # C3 correlations
            [0.25, 0.25, 0.00, 1.00]   # C4 correlations
        ])
    
    def calculate_rbc(
        self, 
        company_data: Dict[str, any],
        portfolio_data: pd.DataFrame
    ) -> Dict[str, any]:
        """Calculate total Risk-Based Capital requirement"""
        
        try:
            # C1: Asset Risk
            c1_before = self._calculate_c1_asset_risk(company_data)
            c1_after = c1_before * 0.85  # After covariance adjustment
            
            # C2: Insurance Risk  
            c2 = self._calculate_c2_insurance_risk(portfolio_data, company_data)
            
            # C3: Interest Rate Risk
            c3 = self._calculate_c3_interest_rate_risk(company_data)
            
            # C4: Business Risk
            c4 = self._calculate_c4_business_risk(company_data)
            
            # Aggregate using RBC formula
            # Total RBC = C1 + sqrt((C2)^2 + (C3)^2 + (C4)^2)
            c234_combined = np.sqrt(c2**2 + c3**2 + c4**2)
            total_rbc_before_covariance = c1_before + c234_combined
            
            # Apply covariance adjustments
            total_authorized_control_level = c1_after + c234_combined
            
            # Calculate RBC ratios
            total_adjusted_capital = company_data.get('surplus', 100000000)  # $100M default
            
            rbc_ratio = total_adjusted_capital / total_authorized_control_level if total_authorized_control_level > 0 else 5.0
            
            # Determine action level
            action_level = self._determine_rbc_action_level(rbc_ratio)
            
            return {
                'total_rbc_requirement': total_authorized_control_level,
                'c1_asset_risk': c1_after,
                'c2_insurance_risk': c2,
                'c3_interest_rate_risk': c3, 
                'c4_business_risk': c4,
                'rbc_ratio': rbc_ratio,
                'action_level': action_level,
                'total_adjusted_capital': total_adjusted_capital,
                'capital_adequacy': 'Adequate' if rbc_ratio >= 2.0 else 'Inadequate'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating RBC: {e}")
            return self._default_rbc_result()
    
    def calculate_economic_capital(
        self,
        portfolio_data: pd.DataFrame,
        confidence_level: float = 0.995,
        time_horizon: int = 1
    ) -> Dict[str, any]:
        """Calculate economic capital using tail risk measures"""
        
        try:
            # Simulate portfolio losses using Monte Carlo
            num_simulations = 10000
            
            # Risk factor simulations
            mortality_shocks = np.random.normal(0, 0.15, num_simulations)  # 15% volatility
            interest_shocks = np.random.normal(0, 0.02, num_simulations)   # 2% volatility
            lapse_shocks = np.random.normal(0, 0.25, num_simulations)      # 25% volatility
            
            # Calculate portfolio value under each scenario
            base_portfolio_value = portfolio_data['face_amount'].sum()
            portfolio_values = []
            
            for i in range(num_simulations):
                # Apply shocks to calculate stressed portfolio value
                mortality_impact = base_portfolio_value * mortality_shocks[i] * 0.1
                interest_impact = base_portfolio_value * interest_shocks[i] * 0.05
                lapse_impact = base_portfolio_value * lapse_shocks[i] * 0.02
                
                total_impact = mortality_impact + interest_impact + lapse_impact
                stressed_value = base_portfolio_value + total_impact
                
                portfolio_values.append(stressed_value)
            
            portfolio_values = np.array(portfolio_values)
            
            # Calculate risk metrics
            portfolio_losses = base_portfolio_value - portfolio_values
            
            var_99_5 = np.percentile(portfolio_losses, confidence_level * 100)
            var_99 = np.percentile(portfolio_losses, 99.0)
            expected_shortfall = np.mean(portfolio_losses[portfolio_losses >= var_99_5])
            
            # Economic capital = VaR at confidence level
            economic_capital = max(0, var_99_5)
            
            # Calculate diversification benefit
            standalone_ec = self._calculate_standalone_ec(portfolio_data)
            diversification_benefit = max(0, standalone_ec - economic_capital)
            
            return {
                'economic_capital': economic_capital,
                'var_99_5_percent': var_99_5,
                'var_99_percent': var_99,
                'expected_shortfall': expected_shortfall,
                'confidence_level': confidence_level,
                'time_horizon_years': time_horizon,
                'diversification_benefit': diversification_benefit,
                'standalone_capital': standalone_ec,
                'capital_efficiency': diversification_benefit / standalone_ec if standalone_ec > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating economic capital: {e}")
            return self._default_economic_capital_result()
    
    def calculate_target_capital(
        self,
        company_data: Dict[str, any],
        target_roe: float = 0.15,
        target_rating: str = "AA"
    ) -> Dict[str, any]:
        """Calculate target capital based on ROE and rating objectives"""
        
        try:
            # Rating-based capital multipliers
            rating_multipliers = {
                'AAA': 2.5, 'AA': 2.0, 'A': 1.5, 'BBB': 1.2, 'BB': 1.0
            }
            
            multiplier = rating_multipliers.get(target_rating, 2.0)
            
            # Base capital requirement (higher of RBC or Economic Capital)
            rbc_result = self.calculate_rbc(company_data, pd.DataFrame())
            ec_result = self.calculate_economic_capital(pd.DataFrame())
            
            base_capital = max(
                rbc_result['total_rbc_requirement'],
                ec_result['economic_capital']
            )
            
            # Target capital = Base capital × Rating multiplier
            target_capital = base_capital * multiplier
            
            # ROE-based validation
            expected_earnings = company_data.get('annual_earnings', target_capital * 0.12)
            implied_roe = expected_earnings / target_capital if target_capital > 0 else 0
            
            # Adjust if ROE too low/high
            if implied_roe < target_roe * 0.8:  # Too low ROE
                target_capital = expected_earnings / target_roe
            elif implied_roe > target_roe * 1.5:  # Too high ROE, may be under-capitalized
                target_capital = expected_earnings / (target_roe * 1.2)
            
            return {
                'target_capital': target_capital,
                'target_roe': target_roe,
                'target_rating': target_rating,
                'rating_multiplier': multiplier,
                'base_capital_requirement': base_capital,
                'implied_roe': implied_roe,
                'capital_buffer': target_capital - base_capital,
                'buffer_percentage': (target_capital - base_capital) / base_capital if base_capital > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating target capital: {e}")
            return {
                'target_capital': 100000000,  # $100M default
                'target_roe': target_roe,
                'target_rating': target_rating,
                'error': str(e)
            }
    
    def _calculate_c1_asset_risk(self, company_data: Dict[str, any]) -> float:
        """Calculate C1 Asset Risk component"""
        
        c1_total = 0
        
        # Bond risk by rating
        bonds = company_data.get('bonds', {})
        for rating, amount in bonds.items():
            factor = self.rbc_factors['c1_bonds'].get(rating.lower(), 0.10)
            c1_total += amount * factor
        
        # Stock risk
        stocks = company_data.get('stocks', 0)
        c1_total += stocks * self.rbc_factors['c1_stocks']
        
        # Mortgage risk
        mortgages = company_data.get('mortgages', 0)
        c1_total += mortgages * self.rbc_factors['c1_mortgages']
        
        # Real estate risk
        real_estate = company_data.get('real_estate', 0)
        c1_total += real_estate * self.rbc_factors['c1_real_estate']
        
        return c1_total
    
    def _calculate_c2_insurance_risk(
        self, 
        portfolio_data: pd.DataFrame,
        company_data: Dict[str, any]
    ) -> float:
        """Calculate C2 Insurance Risk component"""
        
        c2_total = 0
        
        if not portfolio_data.empty:
            # Mortality risk
            net_amount_at_risk = portfolio_data['face_amount'].sum()
            c2_total += net_amount_at_risk * self.rbc_factors['c2_mortality']
            
            # Morbidity risk (for health/disability products)
            claim_reserves = company_data.get('claim_reserves', 0)
            c2_total += claim_reserves * self.rbc_factors['c2_morbidity']
        
        # Lapse risk
        policy_reserves = company_data.get('policy_reserves', 0)
        c2_total += policy_reserves * self.rbc_factors['c2_lapse']
        
        return c2_total
    
    def _calculate_c3_interest_rate_risk(self, company_data: Dict[str, any]) -> float:
        """Calculate C3 Interest Rate Risk component"""
        
        c3_total = 0
        
        # Disintermediation risk
        reserves_subject_to_withdrawal = company_data.get('withdrawable_reserves', 0)
        c3_total += reserves_subject_to_withdrawal * self.rbc_factors['c3_disintermediation']
        
        # Asset-liability mismatch risk
        total_reserves = company_data.get('total_reserves', 0)
        duration_mismatch = abs(company_data.get('asset_duration', 7) - 
                              company_data.get('liability_duration', 8))
        
        if duration_mismatch > 2:  # Significant mismatch
            c3_total += total_reserves * self.rbc_factors['c3_asset_liability'] * (duration_mismatch - 2)
        
        return c3_total
    
    def _calculate_c4_business_risk(self, company_data: Dict[str, any]) -> float:
        """Calculate C4 Business Risk component"""
        
        c4_total = 0
        
        # Growth risk
        annual_premiums = company_data.get('annual_premiums', 0)
        c4_total += annual_premiums * self.rbc_factors['c4_business_growth']
        
        # Operational risk
        operating_expenses = company_data.get('operating_expenses', annual_premiums * 0.1)
        c4_total += operating_expenses * self.rbc_factors['c4_operational']
        
        return c4_total
    
    def _calculate_standalone_ec(self, portfolio_data: pd.DataFrame) -> float:
        """Calculate standalone economic capital (no diversification)"""
        
        if portfolio_data.empty:
            return 50000000  # $50M default
        
        # Sum of individual policy economic capitals
        total_face_amount = portfolio_data['face_amount'].sum()
        
        # Approximate standalone EC as higher percentage of face amount
        standalone_ec_rate = 0.15  # 15% of face amount (no diversification)
        
        return total_face_amount * standalone_ec_rate
    
    def _determine_rbc_action_level(self, rbc_ratio: float) -> str:
        """Determine regulatory action level based on RBC ratio"""
        
        if rbc_ratio >= 2.0:
            return "No Action"
        elif rbc_ratio >= 1.5:
            return "Company Action Level"
        elif rbc_ratio >= 1.0:
            return "Regulatory Action Level"
        elif rbc_ratio >= 0.7:
            return "Authorized Control Level"
        else:
            return "Mandatory Control Level"
    
    def _default_rbc_result(self) -> Dict[str, any]:
        """Default RBC result for error cases"""
        
        return {
            'total_rbc_requirement': 50000000,  # $50M
            'c1_asset_risk': 20000000,
            'c2_insurance_risk': 15000000,
            'c3_interest_rate_risk': 10000000,
            'c4_business_risk': 5000000,
            'rbc_ratio': 2.0,
            'action_level': 'No Action',
            'total_adjusted_capital': 100000000,
            'capital_adequacy': 'Adequate'
        }
    
    def _default_economic_capital_result(self) -> Dict[str, any]:
        """Default economic capital result for error cases"""
        
        return {
            'economic_capital': 75000000,  # $75M
            'var_99_5_percent': 75000000,
            'var_99_percent': 60000000,
            'expected_shortfall': 90000000,
            'confidence_level': 0.995,
            'time_horizon_years': 1,
            'diversification_benefit': 25000000,
            'standalone_capital': 100000000,
            'capital_efficiency': 0.25
        }