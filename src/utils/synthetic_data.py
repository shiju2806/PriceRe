"""
Synthetic Insurance Data Generator

Generates realistic insurance datasets for testing and development.
Includes proper actuarial relationships and industry-standard patterns.
"""

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import random
from faker import Faker
from dataclasses import dataclass
import uuid

from ..core.actuarial_engine import ActuarialEngine, calculate_life_insurance_premium, calculate_annuity_payment

fake = Faker()
fake.seed_instance(42)  # Reproducible data
np.random.seed(42)
random.seed(42)

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    n_records: int = 100000
    start_date: date = date(2015, 1, 1)
    end_date: date = date(2024, 12, 31)
    age_range: Tuple[int, int] = (18, 80)
    smoking_rate: float = 0.15
    gender_split: Tuple[float, float] = (0.52, 0.48)  # Male, Female
    include_medical_history: bool = True
    include_financial_data: bool = True
    realistic_correlations: bool = True

class SyntheticInsuranceDataGenerator:
    """
    Generate synthetic insurance data with realistic actuarial patterns
    """
    
    def __init__(self, config: DataGenerationConfig = None):
        self.config = config or DataGenerationConfig()
        self.actuarial_engine = ActuarialEngine()
        
        # Industry-standard reference data
        self.states = [
            'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
            'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
        ]
        
        self.occupations = [
            'Teacher', 'Engineer', 'Manager', 'Salesperson', 'Nurse', 
            'Accountant', 'Lawyer', 'Doctor', 'Mechanic', 'Clerk',
            'Construction Worker', 'Police Officer', 'Firefighter',
            'Artist', 'Consultant', 'Analyst', 'Developer'
        ]
        
        self.occupation_risk_classes = {
            'Teacher': 1, 'Engineer': 1, 'Manager': 1, 'Accountant': 1, 'Lawyer': 1,
            'Nurse': 2, 'Doctor': 1, 'Analyst': 1, 'Developer': 1, 'Consultant': 1,
            'Salesperson': 2, 'Clerk': 2, 'Artist': 2,
            'Mechanic': 3, 'Construction Worker': 4, 'Police Officer': 4, 'Firefighter': 5
        }
        
        self.health_conditions = [
            'Hypertension', 'Diabetes', 'Heart Disease', 'Cancer History',
            'Asthma', 'Depression', 'Anxiety', 'High Cholesterol',
            'Arthritis', 'Sleep Apnea'
        ]
        
        # Mortality multipliers by health condition
        self.condition_multipliers = {
            'Hypertension': 1.3, 'Diabetes': 2.0, 'Heart Disease': 3.0,
            'Cancer History': 2.5, 'Asthma': 1.2, 'Depression': 1.4,
            'Anxiety': 1.1, 'High Cholesterol': 1.3, 'Arthritis': 1.1,
            'Sleep Apnea': 1.5
        }

    def generate_life_insurance_data(self) -> pl.DataFrame:
        """Generate realistic life insurance dataset"""
        
        print(f"Generating {self.config.n_records} life insurance records...")
        
        records = []
        for i in range(self.config.n_records):
            if i % 10000 == 0:
                print(f"  Generated {i} records...")
            
            record = self._generate_life_insurance_record()
            records.append(record)
        
        df = pl.DataFrame(records)
        print(f"Generated life insurance dataset: {len(df)} records, {len(df.columns)} columns")
        return df
    
    def generate_annuity_data(self) -> pl.DataFrame:
        """Generate realistic annuity/retirement dataset"""
        
        print(f"Generating {self.config.n_records} annuity records...")
        
        records = []
        for i in range(self.config.n_records):
            if i % 10000 == 0:
                print(f"  Generated {i} records...")
            
            record = self._generate_annuity_record()
            records.append(record)
        
        df = pl.DataFrame(records)
        print(f"Generated annuity dataset: {len(df)} records, {len(df.columns)} columns")
        return df
    
    def _generate_life_insurance_record(self) -> Dict:
        """Generate a single life insurance policy record"""
        
        # Basic demographics
        gender = np.random.choice(['M', 'F'], p=self.config.gender_split)
        age = np.random.randint(self.config.age_range[0], self.config.age_range[1] + 1)
        
        # Issue date within specified range
        issue_date = fake.date_between(self.config.start_date, self.config.end_date)
        birth_date = issue_date - timedelta(days=age * 365.25)
        
        # Smoking status with age correlation (younger people smoke less)
        base_smoking_rate = self.config.smoking_rate
        age_adjusted_smoking = base_smoking_rate * (1.5 if age > 50 else 0.8)
        smoker_status = np.random.random() < age_adjusted_smoking
        
        # Health rating based on age and smoking
        health_ratings = ['Preferred Plus', 'Preferred', 'Standard Plus', 'Standard', 'Substandard']
        health_weights = [0.15, 0.25, 0.25, 0.25, 0.10]
        
        if smoker_status:
            # Smokers less likely to get preferred rates
            health_weights = [0.05, 0.15, 0.25, 0.35, 0.20]
        
        if age > 65:
            # Older ages less likely to get preferred rates
            health_weights = [0.05, 0.15, 0.20, 0.40, 0.20]
        
        health_rating = np.random.choice(health_ratings, p=health_weights)
        
        # Policy details
        policy_types = ['Term', 'Whole Life', 'Universal Life', 'Variable Life']
        policy_type = np.random.choice(policy_types, p=[0.6, 0.2, 0.15, 0.05])
        
        # Face amount correlated with income
        income = self._generate_income(age, gender)
        face_amount = self._generate_face_amount(income, age)
        
        # Calculate realistic premium using actuarial engine
        try:
            if policy_type == 'Term':
                term_years = np.random.choice([10, 15, 20, 30], p=[0.2, 0.25, 0.4, 0.15])
                premium_calc = calculate_life_insurance_premium(
                    age=age,
                    gender=gender,
                    face_amount=face_amount,
                    smoker_status=smoker_status,
                    product_type="term",
                    term_years=term_years
                )
                annual_premium = premium_calc['annual_premium']
            else:
                premium_calc = calculate_life_insurance_premium(
                    age=age,
                    gender=gender,
                    face_amount=face_amount,
                    smoker_status=smoker_status,
                    product_type="whole_life"
                )
                annual_premium = premium_calc['annual_premium']
        except:
            # Fallback calculation if actuarial calculation fails
            base_rate = 0.005 if not smoker_status else 0.012
            age_factor = 1 + (age - 30) * 0.02
            annual_premium = face_amount * base_rate * age_factor
        
        # Payment mode
        payment_modes = ['Annual', 'Semi-Annual', 'Quarterly', 'Monthly']
        payment_mode = np.random.choice(payment_modes, p=[0.1, 0.15, 0.25, 0.5])
        
        # Geographic and occupation data
        state = np.random.choice(self.states)
        occupation = np.random.choice(self.occupations)
        occupation_class = self.occupation_risk_classes.get(occupation, 2)
        
        # Contact information
        record = {
            # Policy identification
            'policy_number': f'POL{uuid.uuid4().hex[:8].upper()}',
            'policy_type': policy_type,
            'issue_date': issue_date,
            'policy_status': np.random.choice(['Active', 'Lapsed', 'Paid-up'], p=[0.85, 0.12, 0.03]),
            
            # Insured demographics
            'first_name': fake.first_name_male() if gender == 'M' else fake.first_name_female(),
            'last_name': fake.last_name(),
            'birth_date': birth_date,
            'age_at_issue': age,
            'gender': gender,
            'smoker_status': 'Smoker' if smoker_status else 'Non-Smoker',
            'health_rating': health_rating,
            
            # Address
            'address': fake.street_address(),
            'city': fake.city(),
            'state': state,
            'zip_code': fake.zipcode(),
            
            # Contact
            'phone': fake.phone_number(),
            'email': fake.email(),
            
            # Financial
            'annual_income': income,
            'occupation': occupation,
            'occupation_class': occupation_class,
            'employer': fake.company(),
            
            # Policy details
            'face_amount': face_amount,
            'annual_premium': annual_premium,
            'payment_mode': payment_mode,
            'premium_frequency': {'Annual': 1, 'Semi-Annual': 2, 'Quarterly': 4, 'Monthly': 12}[payment_mode],
            
            # Underwriting
            'medical_exam': np.random.choice(['Yes', 'No'], p=[0.7, 0.3]),
            'agent_id': f'AGT{np.random.randint(1000, 9999)}',
            
            # Financial ratios
            'coverage_ratio': face_amount / income if income > 0 else 0,
            'premium_to_income': annual_premium / income if income > 0 else 0,
        }
        
        # Add medical history if requested
        if self.config.include_medical_history:
            record.update(self._generate_medical_history(age, smoker_status))
        
        # Add financial details if requested
        if self.config.include_financial_data:
            record.update(self._generate_financial_details(income))
        
        return record
    
    def _generate_annuity_record(self) -> Dict:
        """Generate a single annuity contract record"""
        
        # Demographics (annuity buyers tend to be older)
        gender = np.random.choice(['M', 'F'], p=[0.45, 0.55])  # More female annuity buyers
        age = np.random.randint(45, 85)  # Annuity buyers typically older
        
        # Contract date
        contract_date = fake.date_between(self.config.start_date, self.config.end_date)
        birth_date = contract_date - timedelta(days=age * 365.25)
        
        # Product type
        product_types = ['Immediate Annuity', 'Deferred Annuity', 'Variable Annuity', 'Fixed Indexed Annuity']
        product_type = np.random.choice(product_types, p=[0.25, 0.35, 0.25, 0.15])
        
        # Premium amount (higher for annuities)
        premium_amount = self._generate_annuity_premium(age)
        
        # Calculate payments using actuarial engine
        try:
            if product_type == 'Immediate Annuity':
                payment_calc = calculate_annuity_payment(
                    age=age,
                    gender=gender,
                    premium_amount=premium_amount,
                    product_type="immediate",
                    payment_frequency=12
                )
                monthly_payment = payment_calc['monthly_payment']
                annual_payment = payment_calc['annual_payment']
            else:
                # For deferred annuities, calculate future value
                monthly_payment = 0  # No current payments
                annual_payment = 0
        except:
            # Fallback calculation
            payout_rate = 0.06  # 6% payout rate
            annual_payment = premium_amount * payout_rate
            monthly_payment = annual_payment / 12
        
        # Payout options
        payout_options = ['Life Only', 'Life with 10 Years Certain', 'Life with 20 Years Certain', 'Joint Life']
        payout_option = np.random.choice(payout_options, p=[0.2, 0.4, 0.3, 0.1])
        
        # Geographic data
        state = np.random.choice(self.states)
        
        record = {
            # Contract identification
            'contract_number': f'ANN{uuid.uuid4().hex[:8].upper()}',
            'product_type': product_type,
            'contract_date': contract_date,
            'contract_status': np.random.choice(['Active', 'Annuitizing', 'Matured'], p=[0.7, 0.2, 0.1]),
            
            # Annuitant demographics
            'first_name': fake.first_name_male() if gender == 'M' else fake.first_name_female(),
            'last_name': fake.last_name(),
            'birth_date': birth_date,
            'age_at_contract': age,
            'gender': gender,
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], p=[0.2, 0.5, 0.15, 0.15]),
            
            # Address
            'address': fake.street_address(),
            'city': fake.city(),
            'state': state,
            'zip_code': fake.zipcode(),
            
            # Contact
            'phone': fake.phone_number(),
            'email': fake.email(),
            
            # Contract details
            'premium_amount': premium_amount,
            'accumulation_value': premium_amount * np.random.uniform(1.02, 1.15),  # Some growth
            'surrender_value': premium_amount * np.random.uniform(0.85, 0.98),  # Surrender charges
            
            # Payout details
            'payout_option': payout_option,
            'monthly_payment': monthly_payment,
            'annual_payment': annual_payment,
            'payment_start_date': contract_date + timedelta(days=np.random.randint(0, 3650)),  # 0-10 years deferral
            
            # Investment details (for variable products)
            'investment_option': np.random.choice(['Conservative', 'Moderate', 'Aggressive'], p=[0.4, 0.4, 0.2]),
            'guaranteed_rate': np.random.uniform(0.01, 0.03),  # 1-3% guaranteed
            
            # Riders
            'death_benefit_rider': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
            'nursing_home_rider': np.random.choice(['Yes', 'No'], p=[0.2, 0.8]),
            'cola_rider': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            
            # Agent/Distribution
            'agent_id': f'AGT{np.random.randint(1000, 9999)}',
            'distribution_channel': np.random.choice(['Agent', 'Broker', 'Bank', 'Direct'], p=[0.4, 0.3, 0.2, 0.1]),
        }
        
        return record
    
    def _generate_income(self, age: int, gender: str) -> float:
        """Generate realistic income based on age and gender"""
        
        # Base income by age (career progression)
        if age < 25:
            base_income = 35000
        elif age < 35:
            base_income = 55000
        elif age < 45:
            base_income = 75000
        elif age < 55:
            base_income = 90000
        elif age < 65:
            base_income = 85000  # Peak earning years
        else:
            base_income = 40000  # Retirement/part-time
        
        # Gender income gap (unfortunately realistic)
        if gender == 'F':
            base_income *= 0.85
        
        # Add randomness
        income = base_income * np.random.lognormal(0, 0.5)
        
        return max(25000, min(income, 500000))  # Reasonable bounds
    
    def _generate_face_amount(self, income: float, age: int) -> float:
        """Generate realistic face amount based on income and age"""
        
        # Typical coverage multiples
        if age < 35:
            coverage_multiple = np.random.uniform(8, 15)  # Young families need more coverage
        elif age < 55:
            coverage_multiple = np.random.uniform(5, 12)  # Peak earning years
        else:
            coverage_multiple = np.random.uniform(2, 8)   # Lower needs approaching retirement
        
        face_amount = income * coverage_multiple
        
        # Round to standard amounts
        if face_amount < 50000:
            face_amount = round(face_amount / 5000) * 5000
        elif face_amount < 250000:
            face_amount = round(face_amount / 25000) * 25000
        else:
            face_amount = round(face_amount / 50000) * 50000
        
        return max(25000, min(face_amount, 5000000))  # Policy limits
    
    def _generate_annuity_premium(self, age: int) -> float:
        """Generate realistic annuity premium amounts"""
        
        # Annuity premiums tend to be larger
        if age < 55:
            base_premium = 100000
        elif age < 65:
            base_premium = 250000
        else:
            base_premium = 500000  # Retirement rollovers
        
        # Add lognormal variation
        premium = base_premium * np.random.lognormal(0, 0.7)
        
        # Round to standard amounts
        if premium < 25000:
            premium = round(premium / 1000) * 1000
        elif premium < 100000:
            premium = round(premium / 5000) * 5000
        else:
            premium = round(premium / 25000) * 25000
        
        return max(10000, min(premium, 2000000))
    
    def _generate_medical_history(self, age: int, smoker_status: bool) -> Dict:
        """Generate medical history fields"""
        
        # Probability of conditions increases with age
        condition_prob = min(0.1 + age * 0.005, 0.6)
        if smoker_status:
            condition_prob *= 1.5
        
        medical_data = {}
        
        # Height and weight with realistic distributions
        if np.random.random() < 0.52:  # Male
            height = np.random.normal(70, 3)  # inches
            weight = np.random.normal(180, 25)  # pounds
        else:  # Female
            height = np.random.normal(65, 3)
            weight = np.random.normal(150, 20)
        
        medical_data.update({
            'height_inches': max(60, min(height, 84)),
            'weight_pounds': max(100, min(weight, 300)),
            'bmi': (weight / (height ** 2)) * 703,
        })
        
        # Medical conditions
        num_conditions = np.random.poisson(condition_prob * 3)
        conditions = np.random.choice(self.health_conditions, 
                                     size=min(num_conditions, len(self.health_conditions)), 
                                     replace=False)
        
        for condition in self.health_conditions:
            medical_data[f'history_{condition.lower().replace(" ", "_")}'] = condition in conditions
        
        # Vital signs
        medical_data.update({
            'blood_pressure_systolic': np.random.normal(120, 15),
            'blood_pressure_diastolic': np.random.normal(80, 10),
            'cholesterol_total': np.random.normal(200, 40),
            'resting_heart_rate': np.random.normal(70, 12),
        })
        
        return medical_data
    
    def _generate_financial_details(self, income: float) -> Dict:
        """Generate additional financial information"""
        
        # Net worth typically correlates with income and age
        net_worth = income * np.random.uniform(2, 8)
        liquid_assets = net_worth * np.random.uniform(0.1, 0.4)
        
        return {
            'net_worth': net_worth,
            'liquid_assets': liquid_assets,
            'home_value': net_worth * np.random.uniform(0.2, 0.6),
            'mortgage_balance': net_worth * np.random.uniform(0.1, 0.4),
            'investment_assets': liquid_assets * np.random.uniform(0.3, 0.8),
            'retirement_assets': income * np.random.uniform(3, 12),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.4),
            'credit_score': np.random.randint(550, 850),
        }
    
    def save_datasets(self, output_dir: str = "data/synthetic"):
        """Generate and save both life insurance and annuity datasets"""
        
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate life insurance data
        life_df = self.generate_life_insurance_data()
        life_path = f"{output_dir}/life_insurance_synthetic.csv"
        life_df.write_csv(life_path)
        print(f"Saved life insurance data to {life_path}")
        
        # Generate annuity data  
        annuity_df = self.generate_annuity_data()
        annuity_path = f"{output_dir}/annuity_synthetic.csv"
        annuity_df.write_csv(annuity_path)
        print(f"Saved annuity data to {annuity_path}")
        
        # Generate summary statistics
        self._generate_data_summary(life_df, annuity_df, output_dir)
    
    def _generate_data_summary(self, life_df: pl.DataFrame, annuity_df: pl.DataFrame, output_dir: str):
        """Generate summary report of synthetic data"""
        
        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "life_insurance": {
                "record_count": len(life_df),
                "column_count": len(life_df.columns),
                "avg_age": life_df['age_at_issue'].mean(),
                "avg_face_amount": life_df['face_amount'].mean(),
                "avg_premium": life_df['annual_premium'].mean(),
                "smoking_rate": (life_df['smoker_status'] == 'Smoker').sum() / len(life_df),
                "gender_split": {
                    "male": (life_df['gender'] == 'M').sum() / len(life_df),
                    "female": (life_df['gender'] == 'F').sum() / len(life_df)
                }
            },
            "annuity": {
                "record_count": len(annuity_df),
                "column_count": len(annuity_df.columns),
                "avg_age": annuity_df['age_at_contract'].mean(),
                "avg_premium": annuity_df['premium_amount'].mean(),
                "avg_monthly_payment": annuity_df['monthly_payment'].mean(),
            }
        }
        
        import json
        summary_path = f"{output_dir}/data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved data summary to {summary_path}")

# Convenience function for quick data generation
def generate_sample_data(n_life: int = 10000, n_annuity: int = 5000, output_dir: str = "data/synthetic") -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Quick function to generate sample datasets
    
    Args:
        n_life: Number of life insurance records
        n_annuity: Number of annuity records
        output_dir: Output directory for saved files
        
    Returns:
        Tuple of (life_insurance_df, annuity_df)
    """
    config = DataGenerationConfig(n_records=n_life)
    generator = SyntheticInsuranceDataGenerator(config)
    
    # Generate life insurance data
    life_df = generator.generate_life_insurance_data()
    
    # Generate annuity data with different count
    config.n_records = n_annuity
    generator.config = config
    annuity_df = generator.generate_annuity_data()
    
    # Save if output directory provided
    if output_dir:
        generator.save_datasets(output_dir)
    
    return life_df, annuity_df