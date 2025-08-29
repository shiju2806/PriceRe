"""
Core Actuarial Calculation Engine

This module provides the foundational actuarial calculations used across
all insurance products (life insurance, annuities, disability, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

class MortalityTable(Enum):
    """Standard mortality tables used in insurance"""
    CSO_2017 = "2017_cso"
    CSO_2001 = "2001_cso"  
    VBT_2015 = "2015_vbt"
    ANNUITY_2000 = "annuity_2000"
    IAM_2012 = "2012_iam"

class InterestRateScenario(Enum):
    """Interest rate scenarios for present value calculations"""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    FORWARD_CURVE = "forward_curve"

@dataclass
class ActuarialContext:
    """Configuration for actuarial calculations"""
    mortality_table: MortalityTable = MortalityTable.CSO_2017
    interest_rate: float = 0.03
    expense_loading: float = 0.15
    profit_margin: float = 0.05
    improvement_scale: Optional[str] = "MP-2021"
    select_period: int = 25

class ActuarialEngine:
    """
    Core actuarial calculation engine providing mortality, present value,
    and reserve calculations for all insurance products.
    """
    
    def __init__(self):
        self.mortality_tables = {}
        self.interest_curves = {}
        self.load_standard_tables()
    
    def load_standard_tables(self):
        """Load standard mortality and interest rate tables"""
        # In production, these would be loaded from files or databases
        # For now, we'll use approximations
        
        # 2017 CSO Mortality Table (Male, Nonsmoker, Ultimate)
        self.mortality_tables[MortalityTable.CSO_2017] = {
            "male_nonsmoker": self._generate_cso_2017_male_nonsmoker(),
            "female_nonsmoker": self._generate_cso_2017_female_nonsmoker(),
            "male_smoker": self._generate_cso_2017_male_smoker(),
            "female_smoker": self._generate_cso_2017_female_smoker()
        }
        
        # Annuity 2000 Table (for longevity calculations)
        self.mortality_tables[MortalityTable.ANNUITY_2000] = {
            "male": self._generate_annuity_2000_male(),
            "female": self._generate_annuity_2000_female()
        }
    
    def _generate_cso_2017_male_nonsmoker(self) -> Dict[int, float]:
        """Generate approximation of 2017 CSO Male Nonsmoker Ultimate mortality rates"""
        qx = {}
        for age in range(0, 121):
            if age < 30:
                # Low mortality at young ages
                qx[age] = 0.0003 + age * 0.000005
            elif age < 65:
                # Gradual increase through working years  
                qx[age] = 0.0015 + (age - 30) * 0.0003
            else:
                # Exponential increase at older ages (Gompertz law)
                qx[age] = 0.015 * math.exp(0.08 * (age - 65))
        
        # Ensure qx never exceeds 1.0
        for age in qx:
            qx[age] = min(qx[age], 1.0)
            
        return qx
    
    def _generate_cso_2017_female_nonsmoker(self) -> Dict[int, float]:
        """Generate approximation of 2017 CSO Female Nonsmoker Ultimate mortality rates"""
        male_qx = self._generate_cso_2017_male_nonsmoker()
        # Female mortality is typically 85% of male mortality
        return {age: rate * 0.85 for age, rate in male_qx.items()}
    
    def _generate_cso_2017_male_smoker(self) -> Dict[int, float]:
        """Generate approximation of 2017 CSO Male Smoker Ultimate mortality rates"""
        nonsmoker_qx = self._generate_cso_2017_male_nonsmoker()
        # Smoker mortality is typically 2.5x nonsmoker mortality
        return {age: min(rate * 2.5, 1.0) for age, rate in nonsmoker_qx.items()}
    
    def _generate_cso_2017_female_smoker(self) -> Dict[int, float]:
        """Generate approximation of 2017 CSO Female Smoker Ultimate mortality rates"""
        nonsmoker_qx = self._generate_cso_2017_female_nonsmoker()
        # Smoker mortality is typically 2.5x nonsmoker mortality
        return {age: min(rate * 2.5, 1.0) for age, rate in nonsmoker_qx.items()}
    
    def _generate_annuity_2000_male(self) -> Dict[int, float]:
        """Generate Annuity 2000 Male mortality rates (lower than CSO for annuitants)"""
        cso_qx = self._generate_cso_2017_male_nonsmoker()
        # Annuitants have better mortality (selection effect)
        return {age: rate * 0.7 for age, rate in cso_qx.items()}
    
    def _generate_annuity_2000_female(self) -> Dict[int, float]:
        """Generate Annuity 2000 Female mortality rates"""
        cso_qx = self._generate_cso_2017_female_nonsmoker()
        # Annuitants have better mortality (selection effect)
        return {age: rate * 0.7 for age, rate in cso_qx.items()}
    
    def get_mortality_rate(
        self, 
        age: int, 
        gender: str, 
        smoker_status: bool = False,
        table: MortalityTable = MortalityTable.CSO_2017,
        select_duration: Optional[int] = None
    ) -> float:
        """
        Get mortality rate (qx) for given age and characteristics
        
        Args:
            age: Age in years
            gender: 'M' or 'F'
            smoker_status: True if smoker, False if nonsmoker
            table: Which mortality table to use
            select_duration: Duration since policy issue (for select rates)
            
        Returns:
            Annual mortality rate (qx)
        """
        if table not in self.mortality_tables:
            raise ValueError(f"Mortality table {table} not loaded")
        
        table_data = self.mortality_tables[table]
        
        # Determine which subtable to use
        if table == MortalityTable.CSO_2017:
            if gender.upper() == 'M':
                subtable_key = "male_smoker" if smoker_status else "male_nonsmoker"
            else:
                subtable_key = "female_smoker" if smoker_status else "female_nonsmoker"
        elif table == MortalityTable.ANNUITY_2000:
            subtable_key = "male" if gender.upper() == 'M' else "female"
        else:
            raise ValueError(f"Subtable logic not implemented for {table}")
        
        subtable = table_data[subtable_key]
        
        # Get base mortality rate
        if age not in subtable:
            # For ages outside table, use boundary values
            if age < min(subtable.keys()):
                qx = subtable[min(subtable.keys())]
            else:
                qx = subtable[max(subtable.keys())]
        else:
            qx = subtable[age]
        
        # Apply select factors if duration provided
        if select_duration is not None and select_duration <= 25:
            select_factor = 0.5 + 0.02 * select_duration  # Gradual approach to ultimate
            qx *= select_factor
        
        return min(qx, 1.0)
    
    def calculate_life_expectancy(
        self, 
        age: int, 
        gender: str,
        smoker_status: bool = False,
        table: MortalityTable = MortalityTable.CSO_2017
    ) -> float:
        """
        Calculate complete life expectancy at given age
        
        Returns:
            Expected remaining years of life
        """
        life_expectancy = 0.0
        current_age = age
        survival_prob = 1.0
        
        # Calculate until survival probability becomes negligible or age 120
        while current_age <= 120 and survival_prob > 0.0001:
            qx = self.get_mortality_rate(current_age, gender, smoker_status, table)
            survival_prob *= (1 - qx)
            life_expectancy += survival_prob
            current_age += 1
        
        return life_expectancy
    
    def calculate_present_value_annuity(
        self, 
        age: int,
        gender: str,
        annual_payment: float,
        interest_rate: float = 0.03,
        payment_frequency: int = 1,
        smoker_status: bool = False,
        table: MortalityTable = MortalityTable.ANNUITY_2000,
        payment_timing: str = "beginning"  # "beginning" or "end"
    ) -> float:
        """
        Calculate present value of life annuity
        
        Args:
            age: Current age of annuitant
            gender: 'M' or 'F'  
            annual_payment: Annual payment amount
            interest_rate: Discount rate
            payment_frequency: Payments per year (1=annual, 12=monthly)
            smoker_status: Smoking status
            table: Mortality table to use
            payment_timing: When payments are made
            
        Returns:
            Present value of annuity
        """
        pv = 0.0
        current_age = age
        survival_prob = 1.0
        payment_per_period = annual_payment / payment_frequency
        discount_rate_per_period = interest_rate / payment_frequency
        
        # Calculate for up to 50 years or until survival probability negligible
        for year in range(50):
            for period in range(payment_frequency):
                time_periods = year * payment_frequency + period
                
                if payment_timing == "beginning":
                    discount_factor = (1 + discount_rate_per_period) ** -time_periods
                else:
                    discount_factor = (1 + discount_rate_per_period) ** -(time_periods + 1)
                
                # Get mortality rate for this age
                period_age = current_age + time_periods / payment_frequency
                age_floor = int(period_age)
                age_fraction = period_age - age_floor
                
                # Interpolate mortality rate
                qx_floor = self.get_mortality_rate(age_floor, gender, smoker_status, table)
                if age_floor < 120:
                    qx_ceil = self.get_mortality_rate(age_floor + 1, gender, smoker_status, table)
                    qx = qx_floor + age_fraction * (qx_ceil - qx_floor)
                else:
                    qx = qx_floor
                
                # Adjust survival probability
                period_qx = qx / payment_frequency  # Approximate period mortality
                survival_prob *= (1 - period_qx)
                
                # Add to present value
                pv += payment_per_period * survival_prob * discount_factor
                
                if survival_prob < 0.0001:  # Stop when survival probability negligible
                    break
            
            if survival_prob < 0.0001:
                break
                
        return pv
    
    def calculate_present_value_term_insurance(
        self,
        age: int,
        gender: str,
        face_amount: float,
        term_years: int,
        interest_rate: float = 0.03,
        smoker_status: bool = False,
        table: MortalityTable = MortalityTable.CSO_2017
    ) -> float:
        """
        Calculate present value of term life insurance death benefit
        
        Args:
            age: Issue age
            gender: 'M' or 'F'
            face_amount: Death benefit amount
            term_years: Term length in years
            interest_rate: Discount rate
            smoker_status: Smoking status  
            table: Mortality table to use
            
        Returns:
            Present value of death benefits
        """
        pv_benefits = 0.0
        survival_prob = 1.0
        
        for t in range(1, term_years + 1):
            current_age = age + t - 1
            
            # Get mortality rate for this year
            qx = self.get_mortality_rate(current_age, gender, smoker_status, table)
            
            # Probability of death in year t
            prob_death_year_t = survival_prob * qx
            
            # Discount factor (assuming death occurs mid-year)
            discount_factor = (1 + interest_rate) ** -(t - 0.5)
            
            # Add to present value
            pv_benefits += face_amount * prob_death_year_t * discount_factor
            
            # Update survival probability for next year
            survival_prob *= (1 - qx)
        
        return pv_benefits
    
    def calculate_net_single_premium(
        self,
        age: int,
        gender: str,
        face_amount: float,
        product_type: str = "whole_life",
        interest_rate: float = 0.03,
        smoker_status: bool = False,
        term_years: Optional[int] = None
    ) -> float:
        """
        Calculate net single premium (present value of benefits)
        
        Args:
            age: Issue age
            gender: 'M' or 'F'
            face_amount: Death benefit
            product_type: "whole_life", "term", "endowment"
            interest_rate: Valuation interest rate
            smoker_status: Smoking status
            term_years: Term length (for term insurance)
            
        Returns:
            Net single premium
        """
        if product_type == "whole_life":
            # Whole life: death benefit paid whenever death occurs
            return self.calculate_present_value_term_insurance(
                age, gender, face_amount, 120 - age, interest_rate, smoker_status
            )
        elif product_type == "term":
            if term_years is None:
                raise ValueError("term_years required for term insurance")
            return self.calculate_present_value_term_insurance(
                age, gender, face_amount, term_years, interest_rate, smoker_status
            )
        else:
            raise NotImplementedError(f"Product type {product_type} not implemented")
    
    def calculate_annual_premium(
        self,
        age: int,
        gender: str,
        face_amount: float,
        premium_paying_period: int,
        product_type: str = "whole_life",
        interest_rate: float = 0.03,
        smoker_status: bool = False,
        expense_loading: float = 0.15,
        term_years: Optional[int] = None
    ) -> float:
        """
        Calculate level annual premium using equivalence principle
        
        Args:
            age: Issue age
            gender: 'M' or 'F'
            face_amount: Death benefit
            premium_paying_period: Number of years premiums are paid
            product_type: Type of insurance product
            interest_rate: Valuation interest rate
            smoker_status: Smoking status
            expense_loading: Loading for expenses (as percentage)
            term_years: Term length (for term insurance)
            
        Returns:
            Level annual premium
        """
        # Calculate net single premium (PV of benefits)
        nsp = self.calculate_net_single_premium(
            age, gender, face_amount, product_type, interest_rate, 
            smoker_status, term_years
        )
        
        # Calculate present value of annuity due for premium paying period
        pv_annuity = self.calculate_present_value_annuity(
            age, gender, 1.0, interest_rate, 1, smoker_status, 
            MortalityTable.CSO_2017, "beginning"
        )
        
        # Adjust for premium paying period (not lifetime)
        if premium_paying_period < 120:
            # Calculate temporary annuity due
            pv_temp_annuity = 0.0
            survival_prob = 1.0
            
            for t in range(premium_paying_period):
                current_age = age + t
                discount_factor = (1 + interest_rate) ** -t
                pv_temp_annuity += survival_prob * discount_factor
                
                if current_age < 120:
                    qx = self.get_mortality_rate(current_age, gender, smoker_status)
                    survival_prob *= (1 - qx)
            
            pv_annuity = pv_temp_annuity
        
        # Calculate net level premium
        net_premium = nsp / pv_annuity if pv_annuity > 0 else 0
        
        # Add expense loading
        gross_premium = net_premium / (1 - expense_loading)
        
        return gross_premium

# Convenience functions for common calculations
def calculate_life_insurance_premium(
    age: int,
    gender: str, 
    face_amount: float,
    smoker_status: bool = False,
    product_type: str = "term",
    term_years: int = 20,
    expense_loading: float = 0.15
) -> Dict[str, float]:
    """
    Convenience function for life insurance premium calculation
    
    Returns dictionary with premium and key metrics
    """
    engine = ActuarialEngine()
    
    if product_type == "term":
        premium_paying_period = term_years
        premium = engine.calculate_annual_premium(
            age, gender, face_amount, premium_paying_period,
            "term", 0.03, smoker_status, expense_loading, term_years
        )
    else:
        premium_paying_period = 65 - age if age < 65 else 1  # Pay to 65
        premium = engine.calculate_annual_premium(
            age, gender, face_amount, premium_paying_period,
            "whole_life", 0.03, smoker_status, expense_loading
        )
    
    # Calculate other useful metrics
    life_expectancy = engine.calculate_life_expectancy(age, gender, smoker_status)
    mortality_rate = engine.get_mortality_rate(age, gender, smoker_status)
    
    return {
        "annual_premium": premium,
        "monthly_premium": premium / 12,
        "premium_per_1000": (premium / face_amount) * 1000,
        "life_expectancy": life_expectancy,
        "mortality_rate": mortality_rate,
        "premium_paying_years": premium_paying_period
    }

def calculate_annuity_payment(
    age: int,
    gender: str,
    premium_amount: float,
    product_type: str = "immediate",
    payment_frequency: int = 12,
    interest_rate: float = 0.03
) -> Dict[str, float]:
    """
    Convenience function for annuity payment calculation
    
    Returns dictionary with payment amounts and metrics  
    """
    engine = ActuarialEngine()
    
    # Calculate present value of $1 annuity
    pv_unit_annuity = engine.calculate_present_value_annuity(
        age, gender, 1.0, interest_rate, payment_frequency, 
        False, MortalityTable.ANNUITY_2000, "beginning"
    )
    
    # Calculate payment amount
    if pv_unit_annuity > 0:
        annual_payment = premium_amount / pv_unit_annuity
        payment_per_period = annual_payment / payment_frequency
    else:
        annual_payment = 0
        payment_per_period = 0
    
    # Calculate other metrics
    life_expectancy = engine.calculate_life_expectancy(
        age, gender, False, MortalityTable.ANNUITY_2000
    )
    
    return {
        "annual_payment": annual_payment,
        "monthly_payment": payment_per_period if payment_frequency == 12 else annual_payment / 12,
        "payment_per_period": payment_per_period,
        "present_value_factor": pv_unit_annuity,
        "life_expectancy": life_expectancy,
        "expected_total_payments": annual_payment * life_expectancy
    }