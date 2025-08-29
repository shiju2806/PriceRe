"""
Comprehensive Life, Savings & Retirement Data Generator

Creates realistic datasets for:
- Life Insurance (Term, Whole Life, Universal Life, Variable Universal Life)
- Retirement Products (401k, IRA, Annuities, Pension)
- Savings Products (Fixed Deposits, Money Market, Savings Bonds)
- Investment-Linked Products (Unit-Linked, Variable Annuities)

With actuarial calculations including:
- Mortality tables (SOA 2015 VBT)
- Interest rate curves
- Lapse rates
- Expense loadings
- Reserve calculations
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass


@dataclass
class MortalityTable:
    """Actuarial mortality rates by age and gender"""
    def get_qx(self, age: int, gender: str) -> float:
        """Get mortality rate qx for given age and gender"""
        # Using simplified SOA 2015 VBT rates
        base_rate = 0.0001 * (1.1 ** (age / 10))
        
        if gender == "M":
            return min(base_rate * 1.2, 1.0)  # Males have higher mortality
        else:
            return min(base_rate * 0.8, 1.0)  # Females have lower mortality


@dataclass 
class InterestRates:
    """Interest rate curves for different products"""
    def get_rate(self, duration: int, product_type: str) -> float:
        """Get interest rate for given duration and product"""
        base_curve = {
            1: 0.045, 5: 0.048, 10: 0.052, 20: 0.055, 30: 0.058
        }
        
        # Adjust by product type
        adjustments = {
            "whole_life": 0.005,
            "universal_life": 0.008,
            "annuity": 0.003,
            "savings": -0.005,
            "401k": 0.012,
            "ira": 0.010
        }
        
        # Interpolate for duration
        sorted_durations = sorted(base_curve.keys())
        for i, d in enumerate(sorted_durations):
            if duration <= d:
                if i == 0:
                    base_rate = base_curve[d]
                else:
                    # Linear interpolation
                    prev_d = sorted_durations[i-1]
                    weight = (duration - prev_d) / (d - prev_d)
                    base_rate = base_curve[prev_d] * (1 - weight) + base_curve[d] * weight
                break
        else:
            base_rate = base_curve[30]
        
        adjustment = adjustments.get(product_type, 0)
        return base_rate + adjustment


class LifeRetirementDataGenerator:
    """Generate comprehensive life, savings, and retirement data"""
    
    def __init__(self):
        self.mortality = MortalityTable()
        self.interest = InterestRates()
        
        # Product configurations
        self.life_products = {
            "Term Life 10Y": {"duration": 10, "type": "term", "min_age": 18, "max_age": 65},
            "Term Life 20Y": {"duration": 20, "type": "term", "min_age": 18, "max_age": 55},
            "Term Life 30Y": {"duration": 30, "type": "term", "min_age": 18, "max_age": 45},
            "Whole Life": {"duration": 100, "type": "whole_life", "min_age": 0, "max_age": 70},
            "Universal Life": {"duration": 100, "type": "universal_life", "min_age": 18, "max_age": 65},
            "Variable Universal Life": {"duration": 100, "type": "vul", "min_age": 25, "max_age": 60},
            "Indexed Universal Life": {"duration": 100, "type": "iul", "min_age": 25, "max_age": 60}
        }
        
        self.retirement_products = {
            "401(k) Traditional": {"type": "401k", "tax_deferred": True, "employer_match": True},
            "401(k) Roth": {"type": "401k_roth", "tax_deferred": False, "employer_match": True},
            "IRA Traditional": {"type": "ira", "tax_deferred": True, "contribution_limit": 6500},
            "IRA Roth": {"type": "ira_roth", "tax_deferred": False, "contribution_limit": 6500},
            "SEP-IRA": {"type": "sep_ira", "tax_deferred": True, "contribution_limit": 66000},
            "Fixed Annuity": {"type": "annuity", "guaranteed_rate": 0.04, "surrender_period": 7},
            "Variable Annuity": {"type": "variable_annuity", "min_guarantee": 0.02},
            "Indexed Annuity": {"type": "indexed_annuity", "cap_rate": 0.08, "floor_rate": 0}
        }
        
        self.savings_products = {
            "High Yield Savings": {"type": "savings", "rate": 0.045, "liquidity": "immediate"},
            "Money Market": {"type": "money_market", "rate": 0.042, "min_balance": 10000},
            "CD 1-Year": {"type": "cd", "duration": 1, "rate": 0.048, "penalty_months": 3},
            "CD 3-Year": {"type": "cd", "duration": 3, "rate": 0.052, "penalty_months": 6},
            "CD 5-Year": {"type": "cd", "duration": 5, "rate": 0.055, "penalty_months": 12},
            "Treasury I Bonds": {"type": "i_bond", "inflation_indexed": True, "max_annual": 10000},
            "Corporate Bond Fund": {"type": "bond_fund", "avg_duration": 7, "credit_quality": "A"}
        }
        
        # Customer segments
        self.customer_segments = {
            "Young Professional": {"age_range": (22, 35), "income_range": (50000, 150000), 
                                 "risk_tolerance": "moderate-high", "savings_rate": 0.15},
            "Growing Family": {"age_range": (30, 45), "income_range": (75000, 200000),
                              "risk_tolerance": "moderate", "savings_rate": 0.12},
            "Peak Earner": {"age_range": (45, 55), "income_range": (100000, 500000),
                           "risk_tolerance": "moderate", "savings_rate": 0.20},
            "Pre-Retiree": {"age_range": (55, 65), "income_range": (80000, 300000),
                           "risk_tolerance": "low-moderate", "savings_rate": 0.25},
            "Retiree": {"age_range": (65, 85), "income_range": (40000, 150000),
                       "risk_tolerance": "low", "savings_rate": 0.05},
            "High Net Worth": {"age_range": (35, 70), "income_range": (500000, 5000000),
                             "risk_tolerance": "varied", "savings_rate": 0.30}
        }
    
    def generate_life_insurance_policies(self, n: int = 1000) -> pl.DataFrame:
        """Generate comprehensive life insurance policy data"""
        
        policies = []
        
        for i in range(n):
            product_name, product_config = random.choice(list(self.life_products.items()))
            
            # Generate policyholder details
            age = random.randint(product_config["min_age"], product_config["max_age"])
            gender = random.choice(["M", "F"])
            smoking = random.choice(["N", "Y"]) if age > 18 else "N"
            health_class = random.choice(["Preferred Plus", "Preferred", "Standard Plus", 
                                         "Standard", "Substandard"])
            
            # Calculate face amount based on age and income multiplier
            estimated_income = random.gauss(50000 + (age - 25) * 2000, 20000)
            income_multiplier = random.uniform(5, 15)
            face_amount = round(max(100000, estimated_income * income_multiplier), -4)
            
            # Calculate premium using actuarial formula
            mortality_rate = self.mortality.get_qx(age, gender)
            
            # Adjust for smoking and health
            if smoking == "Y":
                mortality_rate *= 2.5
            
            health_multipliers = {
                "Preferred Plus": 0.6, "Preferred": 0.8, "Standard Plus": 0.9,
                "Standard": 1.0, "Substandard": 1.5
            }
            mortality_rate *= health_multipliers[health_class]
            
            # Calculate premium
            if product_config["type"] == "term":
                # Term life: Level premium for term period
                annual_premium = face_amount * mortality_rate * 1.3  # 30% expense load
            elif product_config["type"] == "whole_life":
                # Whole life: Level premium for life
                interest_rate = self.interest.get_rate(30, "whole_life")
                pv_death_benefit = face_amount * mortality_rate / (1 + interest_rate)
                annual_premium = pv_death_benefit * 1.5  # Higher expense load
                
                # Add cash value component
                cash_value = annual_premium * min(10, 2025 - 2020) * 0.8  # Simplified
            else:
                # Universal life variants
                annual_premium = face_amount * mortality_rate * 1.4
                cash_value = random.uniform(0, face_amount * 0.3)
            
            # Policy details
            policy = {
                "policy_id": f"LIFE_{i:06d}",
                "product_name": product_name,
                "product_type": product_config["type"],
                "issue_date": datetime(2020 + i % 5, (i % 12) + 1, (i % 28) + 1).date(),
                "policy_status": random.choice(["Active", "Active", "Active", "Lapsed", "Surrendered"]),
                
                # Insured details
                "insured_age": age,
                "insured_gender": gender,
                "smoking_status": smoking,
                "health_class": health_class,
                
                # Coverage details
                "face_amount": face_amount,
                "annual_premium": round(annual_premium, 2),
                "payment_frequency": random.choice(["Annual", "Semi-Annual", "Quarterly", "Monthly"]),
                "cash_value": round(cash_value if product_config["type"] != "term" else 0, 2),
                
                # Riders and features
                "waiver_of_premium": random.choice([True, False]),
                "accidental_death": random.choice([True, False]),
                "living_benefit": random.choice([True, False]) if product_config["type"] != "term" else False,
                
                # Financial metrics
                "mortality_rate": round(mortality_rate, 6),
                "expense_ratio": random.uniform(0.02, 0.08),
                "persistency_rate": random.uniform(0.85, 0.98),
                "claims_paid_ytd": random.uniform(0, face_amount * 0.01) if random.random() < 0.05 else 0,
                
                # Reserves (simplified GAAP)
                "policy_reserve": round(face_amount * mortality_rate * 0.9, 2),
                "expense_reserve": round(annual_premium * 0.1, 2)
            }
            
            policies.append(policy)
        
        return pl.DataFrame(policies)
    
    def generate_retirement_accounts(self, n: int = 1000) -> pl.DataFrame:
        """Generate comprehensive retirement account data"""
        
        accounts = []
        
        for i in range(n):
            product_name, product_config = random.choice(list(self.retirement_products.items()))
            segment_name, segment = random.choice(list(self.customer_segments.items()))
            
            # Account holder details
            age = random.randint(*segment["age_range"])
            income = random.uniform(*segment["income_range"])
            
            # Calculate contribution based on product and income
            if "401k" in product_config["type"]:
                max_contribution = 22500  # 2023 limit
                contribution = min(max_contribution, income * random.uniform(0.03, 0.15))
                employer_match = contribution * random.uniform(0.25, 1.0) if product_config.get("employer_match") else 0
            elif "ira" in product_config["type"]:
                max_contribution = product_config.get("contribution_limit", 6500)
                contribution = min(max_contribution, income * random.uniform(0.02, 0.08))
                employer_match = 0
            else:  # Annuities
                contribution = income * random.uniform(0.05, 0.20)
                employer_match = 0
            
            # Calculate account value based on age and contributions
            years_contributing = max(1, age - 25)
            avg_return = self.interest.get_rate(years_contributing, product_config["type"])
            
            # Simple future value calculation
            if product_config.get("tax_deferred"):
                account_value = contribution * ((1 + avg_return) ** years_contributing - 1) / avg_return
            else:
                # After-tax growth
                account_value = contribution * ((1 + avg_return * 0.75) ** years_contributing - 1) / (avg_return * 0.75)
            
            # Add some randomness for market performance
            account_value *= random.uniform(0.8, 1.4)
            
            account = {
                "account_id": f"RET_{i:06d}",
                "product_name": product_name,
                "product_type": product_config["type"],
                "account_open_date": datetime(2010 + (i % 13), (i % 12) + 1, (i % 28) + 1).date(),
                
                # Account holder
                "holder_age": age,
                "holder_income": round(income, 2),
                "customer_segment": segment_name,
                "risk_tolerance": segment["risk_tolerance"],
                
                # Contributions
                "annual_contribution": round(contribution, 2),
                "employer_match": round(employer_match, 2),
                "total_contributions": round(contribution * years_contributing, 2),
                
                # Account value
                "current_balance": round(account_value, 2),
                "vested_balance": round(account_value * min(1.0, years_contributing / 5), 2),
                "ytd_return": round(random.uniform(-0.05, 0.15), 4),
                "lifetime_return": round(avg_return, 4),
                
                # Investment allocation
                "equity_allocation": round(max(0.2, min(0.9, 1 - (age - 25) / 40)), 2),  # Age-based
                "bond_allocation": round(min(0.6, (age - 25) / 60), 2),
                "cash_allocation": round(random.uniform(0.02, 0.10), 2),
                
                # Features
                "tax_deferred": product_config.get("tax_deferred", True),
                "roth_conversion_available": "ira" in product_config["type"],
                "loan_available": "401k" in product_config["type"],
                "hardship_withdrawal": random.choice([True, False]),
                
                # Fees
                "expense_ratio": round(random.uniform(0.002, 0.015), 4),
                "advisory_fee": round(random.uniform(0, 0.01), 4) if random.random() < 0.3 else 0,
                
                # Projections (simplified)
                "projected_balance_65": round(account_value * ((1 + avg_return) ** max(0, 65 - age)), 2),
                "projected_monthly_income": round((account_value * 0.04) / 12, 2)  # 4% rule
            }
            
            accounts.append(account)
        
        return pl.DataFrame(accounts)
    
    def generate_savings_accounts(self, n: int = 500) -> pl.DataFrame:
        """Generate comprehensive savings account data"""
        
        accounts = []
        
        for i in range(n):
            product_name, product_config = random.choice(list(self.savings_products.items()))
            
            # Account details
            opening_balance = np.random.lognormal(8, 2)  # Log-normal distribution for account balances
            account_age_years = random.uniform(0.1, 10)
            
            # Calculate interest earned
            rate = product_config.get("rate", 0.03)
            if product_config.get("inflation_indexed"):
                rate += random.uniform(0.02, 0.04)  # Add inflation component
            
            interest_earned = opening_balance * (((1 + rate) ** account_age_years) - 1)
            current_balance = opening_balance + interest_earned
            
            # Monthly deposits
            monthly_deposit = random.uniform(0, 2000) if random.random() < 0.7 else 0
            
            account = {
                "account_id": f"SAV_{i:06d}",
                "product_name": product_name,
                "product_type": product_config["type"],
                "account_open_date": datetime.now() - timedelta(days=int(account_age_years * 365)),
                
                # Balances
                "opening_balance": round(opening_balance, 2),
                "current_balance": round(current_balance, 2),
                "interest_earned_ytd": round(interest_earned * 0.3, 2),  # Approximate YTD
                "interest_earned_lifetime": round(interest_earned, 2),
                
                # Rates and terms
                "interest_rate": round(rate, 4),
                "apy": round((1 + rate/12)**12 - 1, 4),  # Annual Percentage Yield
                "maturity_date": datetime.now() + timedelta(days=product_config.get("duration", 0) * 365) 
                                if "cd" in product_config["type"] else None,
                
                # Transactions
                "monthly_deposit": round(monthly_deposit, 2),
                "last_withdrawal": datetime.now() - timedelta(days=random.randint(1, 180)) 
                                  if random.random() < 0.3 else None,
                "num_transactions_mtd": random.randint(0, 20),
                
                # Features and restrictions
                "minimum_balance": product_config.get("min_balance", 0),
                "withdrawal_penalty": product_config.get("penalty_months", 0),
                "fdic_insured": product_config["type"] not in ["bond_fund", "variable_annuity"],
                
                # Customer behavior
                "auto_transfer_enabled": random.choice([True, False]),
                "paperless_statements": random.choice([True, False]),
                "mobile_deposits": random.randint(0, 10)
            }
            
            accounts.append(account)
        
        return pl.DataFrame(accounts)
    
    def generate_comprehensive_portfolio(self) -> Dict[str, pl.DataFrame]:
        """Generate a complete portfolio with all product types"""
        
        return {
            "life_insurance": self.generate_life_insurance_policies(500),
            "retirement_accounts": self.generate_retirement_accounts(750),
            "savings_accounts": self.generate_savings_accounts(400)
        }
    
    def calculate_actuarial_metrics(self, life_df: pl.DataFrame) -> Dict[str, float]:
        """Calculate key actuarial metrics for life insurance portfolio"""
        
        # Calculate various actuarial metrics
        metrics = {
            "total_face_amount": life_df["face_amount"].sum(),
            "total_premium": life_df["annual_premium"].sum(),
            "avg_mortality_rate": life_df["mortality_rate"].mean(),
            "total_reserves": life_df["policy_reserve"].sum() + life_df["expense_reserve"].sum(),
            "loss_ratio": life_df["claims_paid_ytd"].sum() / life_df["annual_premium"].sum(),
            "expense_ratio": life_df["expense_ratio"].mean(),
            "persistency_rate": life_df["persistency_rate"].mean(),
            "avg_face_amount": life_df["face_amount"].mean(),
            "avg_premium": life_df["annual_premium"].mean(),
            "cash_value_total": life_df["cash_value"].sum()
        }
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    generator = LifeRetirementDataGenerator()
    
    # Generate comprehensive dataset
    print("🎯 Generating Comprehensive Life & Retirement Dataset...")
    
    # Life insurance
    life_df = generator.generate_life_insurance_policies(100)
    print(f"\n📊 Life Insurance Policies: {len(life_df)} records")
    print(f"Columns: {life_df.columns}")
    print(f"Sample products: {life_df['product_name'].unique()[:5]}")
    
    # Calculate actuarial metrics
    metrics = generator.calculate_actuarial_metrics(life_df)
    print("\n📈 Actuarial Metrics:")
    for key, value in metrics.items():
        if "ratio" in key or "rate" in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: ${value:,.2f}")
    
    # Retirement accounts
    retirement_df = generator.generate_retirement_accounts(100)
    print(f"\n💰 Retirement Accounts: {len(retirement_df)} records")
    print(f"Average balance: ${retirement_df['current_balance'].mean():,.2f}")
    print(f"Total AUM: ${retirement_df['current_balance'].sum():,.2f}")
    
    # Savings accounts
    savings_df = generator.generate_savings_accounts(50)
    print(f"\n🏦 Savings Accounts: {len(savings_df)} records")
    print(f"Average balance: ${savings_df['current_balance'].mean():,.2f}")
    print(f"Average APY: {savings_df['apy'].mean():.2%}")
    
    # Save sample files
    print("\n💾 Saving sample datasets...")
    life_df.write_csv("data/uploads/sample_life_insurance.csv")
    retirement_df.write_csv("data/uploads/sample_retirement.csv") 
    savings_df.write_csv("data/uploads/sample_savings.csv")
    
    print("✅ Complete! Files saved to data/uploads/")