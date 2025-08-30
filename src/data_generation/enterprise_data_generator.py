"""
Enterprise Data Generator
Creates realistic large-scale datasets for reinsurance pricing testing
Supports millions of records with realistic patterns and relationships
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import random
from faker import Faker

# Initialize Faker for realistic data
fake = Faker('en_US')
Faker.seed(42)  # For reproducible results

@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    base_size: int = 1000  # Base dataset size
    scale_factor: int = 1000  # Multiplier for enterprise scale
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    messy_data_pct: float = 0.15  # 15% messy data
    missing_data_pct: float = 0.08  # 8% missing data
    
class EnterpriseDataGenerator:
    """Generate large-scale realistic insurance datasets"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.logger = self._setup_logging()
        self.start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        # State and company distributions (must sum to 1.0)
        self.state_weights = {
            'CA': 0.15, 'TX': 0.12, 'NY': 0.10, 'FL': 0.09, 'IL': 0.08,
            'PA': 0.07, 'OH': 0.06, 'GA': 0.06, 'NC': 0.05, 'MI': 0.05,
            'NJ': 0.04, 'VA': 0.04, 'WA': 0.04, 'AZ': 0.03, 'MA': 0.01,
            'Other': 0.01
        }
        
        self.company_names = [
            "Metropolitan Life", "Prudential Financial", "New York Life", 
            "Northwestern Mutual", "MassMutual", "Guardian Life", "Pacific Life",
            "Principal Financial", "Lincoln National", "Ameritas Life",
            "John Hancock", "AIG Life", "Transamerica", "Genworth Financial",
            "Unum Group", "Aflac", "Protective Life", "Western & Southern"
        ]
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data generation"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def generate_enterprise_policies(self, size: Optional[int] = None) -> pd.DataFrame:
        """Generate large-scale policy dataset"""
        
        actual_size = size or (self.config.base_size * self.config.scale_factor)
        self.logger.info(f"Generating {actual_size:,} policy records...")
        
        # Generate in chunks to manage memory
        chunk_size = 50000
        chunks = []
        
        for i in range(0, actual_size, chunk_size):
            current_chunk_size = min(chunk_size, actual_size - i)
            chunk = self._generate_policy_chunk(current_chunk_size, i)
            chunks.append(chunk)
            
            if (i // chunk_size + 1) % 10 == 0:
                self.logger.info(f"Generated {i + current_chunk_size:,} policies...")
        
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"✅ Generated {len(df):,} policy records")
        
        # Add realistic data quality issues
        df = self._add_data_quality_issues(df)
        
        return df
    
    def _generate_policy_chunk(self, size: int, offset: int) -> pd.DataFrame:
        """Generate a chunk of policy data"""
        
        # Generate correlated demographics
        ages = np.random.normal(42, 15, size).clip(18, 85).astype(int)
        genders = np.random.choice(['M', 'F'], size, p=[0.52, 0.48])
        
        # Smoking correlated with age/gender
        smoking_probs = []
        for age, gender in zip(ages, genders):
            base_prob = 0.15 if gender == 'M' else 0.12
            age_factor = 1.2 if age < 40 else 0.8
            smoking_probs.append(base_prob * age_factor)
        
        smoking_status = [
            np.random.choice(['Nonsmoker', 'Smoker'], p=[1-prob, prob])
            for prob in smoking_probs
        ]
        
        # Face amounts correlated with age/income
        income_multiplier = np.random.lognormal(0, 0.8, size)
        face_amounts = (ages * 15000 * income_multiplier).clip(25000, 50000000).astype(int)
        
        # Premium calculations with realistic factors
        base_rates = np.where(
            np.array(smoking_status) == 'Smoker',
            ages * 8.5,  # Higher rate for smokers
            ages * 3.2   # Lower rate for non-smokers
        )
        
        annual_premiums = (face_amounts * base_rates / 100000).clip(200, 500000).astype(int)
        
        # Product types based on age
        product_probs = []
        for age in ages:
            if age < 35:
                probs = [0.70, 0.15, 0.10, 0.05]  # Term, UL, Whole, VUL
            elif age < 50:
                probs = [0.50, 0.25, 0.20, 0.05]
            else:
                probs = [0.30, 0.30, 0.35, 0.05]
            product_probs.append(probs)
        
        products = [
            np.random.choice(['Term Life', 'Universal Life', 'Whole Life', 'Variable Life'], 
                           p=probs)
            for probs in product_probs
        ]
        
        # Generate issue dates
        issue_dates = [
            self.start_date + timedelta(days=random.randint(0, 
                (self.end_date - self.start_date).days))
            for _ in range(size)
        ]
        
        # Policy status with realistic lapse patterns
        durations = [(datetime.now().date() - d.date()).days / 365.25 for d in issue_dates]
        lapse_probs = [min(0.25, 0.08 + duration * 0.02) for duration in durations]
        policy_statuses = [
            np.random.choice(['Inforce', 'Lapsed'], p=[1-prob, prob])
            for prob in lapse_probs
        ]
        
        return pd.DataFrame({
            'policy_number': [f'ENT_{offset + i + 1:08d}' for i in range(size)],
            'issue_date': [d.strftime('%Y-%m-%d') for d in issue_dates],
            'product_type': products,
            'face_amount': face_amounts,
            'annual_premium': annual_premiums,
            'issue_age': ages,
            'gender': genders,
            'smoking_status': smoking_status,
            'underwriting_class': np.random.choice(
                ['Super Preferred', 'Preferred Plus', 'Preferred', 'Standard', 'Substandard'],
                size, p=[0.15, 0.20, 0.30, 0.25, 0.10]
            ),
            'policy_status': policy_statuses,
            'state': np.random.choice(
                list(self.state_weights.keys()),
                size, p=list(self.state_weights.values())
            ),
            'distribution_channel': np.random.choice(
                ['Agent', 'Broker', 'Direct', 'Online'],
                size, p=[0.60, 0.25, 0.10, 0.05]
            ),
            'rider_accidental_death': np.random.choice(['Y', 'N'], size, p=[0.35, 0.65]),
            'rider_waiver_premium': np.random.choice(['Y', 'N'], size, p=[0.45, 0.55]),
            'policy_year': [max(1, int(duration)) for duration in durations]
        })
    
    def generate_enterprise_claims(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate claims data correlated with policies"""
        
        self.logger.info("Generating claims data...")
        
        # Select subset of policies for claims (realistic claim rate)
        claim_rate = 0.008  # 0.8% annual claim rate
        years_exposure = policy_df['policy_year'].sum()
        expected_claims = int(years_exposure * claim_rate)
        
        # Sample policies for claims
        claim_policies = policy_df.sample(n=min(expected_claims, len(policy_df)))
        
        claims_data = []
        for _, policy in claim_policies.iterrows():
            # Claim date within policy period
            issue_date = datetime.strptime(policy['issue_date'], '%Y-%m-%d')
            max_claim_date = min(datetime.now(), 
                               issue_date + timedelta(days=policy['policy_year'] * 365))
            
            if max_claim_date > issue_date:
                claim_date = fake.date_between(
                    start_date=issue_date.date(),
                    end_date=max_claim_date.date()
                )
                
                # Claim types based on product and demographics
                if policy['issue_age'] > 65:
                    claim_types = ['Death', 'Accelerated', 'Chronic_Illness']
                    weights = [0.70, 0.15, 0.15]
                else:
                    claim_types = ['Death', 'Accidental', 'Disability', 'Critical_Illness']
                    weights = [0.60, 0.15, 0.15, 0.10]
                
                claim_type = np.random.choice(claim_types, p=weights)
                
                # Claim amount
                base_amount = policy['face_amount']
                if claim_type == 'Death':
                    claim_amount = base_amount
                elif claim_type == 'Accidental' and policy['rider_accidental_death'] == 'Y':
                    claim_amount = base_amount * 2  # Double indemnity
                else:
                    claim_amount = int(base_amount * np.random.uniform(0.25, 1.0))
                
                claims_data.append({
                    'claim_id': f'CLM_{len(claims_data) + 1:08d}',
                    'policy_id': policy['policy_number'],
                    'claim_type': claim_type,
                    'date_of_death': claim_date.strftime('%Y-%m-%d') if claim_type == 'Death' else '',
                    'claim_notification_date': (claim_date + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d'),
                    'claim_approval_date': (claim_date + timedelta(days=random.randint(15, 45))).strftime('%Y-%m-%d'),
                    'claim_payment_date': (claim_date + timedelta(days=random.randint(20, 60))).strftime('%Y-%m-%d'),
                    'face_amount': policy['face_amount'],
                    'rider_benefits': int(base_amount * 0.05) if random.random() < 0.3 else 0,
                    'total_claim_amount': claim_amount,
                    'payment_method': np.random.choice(['Wire', 'ACH', 'Check'], p=[0.60, 0.30, 0.10]),
                    'payment_status': 'Paid',
                    'reinsurance_recoverable': int(claim_amount * 0.75),  # Assuming 75% reinsurance
                    'retention_amount': int(claim_amount * 0.25)
                })
        
        claims_df = pd.DataFrame(claims_data)
        self.logger.info(f"✅ Generated {len(claims_df):,} claim records")
        return claims_df
    
    def generate_premium_transactions(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Generate premium payment history"""
        
        self.logger.info("Generating premium transactions...")
        
        transactions = []
        for _, policy in policy_df.iterrows():
            if policy['policy_status'] == 'Lapsed':
                continue
                
            # Generate monthly payments for recent years
            issue_date = datetime.strptime(policy['issue_date'], '%Y-%m-%d')
            monthly_premium = policy['annual_premium'] / 12
            
            # Generate 24 months of payments
            for month in range(24):
                payment_date = issue_date + timedelta(days=month * 30)
                if payment_date > datetime.now():
                    break
                    
                # Some late payments
                actual_payment_date = payment_date
                if random.random() < 0.12:  # 12% late payments
                    actual_payment_date += timedelta(days=random.randint(1, 30))
                
                transactions.append({
                    'policy_id': policy['policy_number'],
                    'policy_year': (payment_date - issue_date).days // 365 + 1,
                    'policy_month': month % 12 + 1,
                    'premium_due_date': payment_date.strftime('%Y-%m-%d'),
                    'premium_paid_date': actual_payment_date.strftime('%Y-%m-%d'),
                    'premium_mode': 'Monthly',
                    'premium_due': monthly_premium,
                    'premium_paid': monthly_premium,
                    'payment_method': np.random.choice(['ACH', 'Credit Card', 'Check'], p=[0.70, 0.20, 0.10]),
                    'late_fee': 25 if (actual_payment_date - payment_date).days > 10 else 0,
                    'payment_status': 'Paid',
                    'days_late': max(0, (actual_payment_date - payment_date).days)
                })
        
        transactions_df = pd.DataFrame(transactions)
        self.logger.info(f"✅ Generated {len(transactions_df):,} premium transactions")
        return transactions_df
    
    def _add_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic data quality issues"""
        
        df = df.copy()
        n_rows = len(df)
        
        # Missing data
        missing_count = int(n_rows * self.config.missing_data_pct)
        missing_indices = np.random.choice(n_rows, missing_count, replace=False)
        
        for idx in missing_indices:
            # Randomly choose columns to make missing
            missing_cols = np.random.choice(
                ['smoking_status', 'underwriting_class', 'rider_accidental_death'],
                size=random.randint(1, 2), replace=False
            )
            for col in missing_cols:
                df.loc[idx, col] = ''
        
        # Format inconsistencies
        messy_count = int(n_rows * self.config.messy_data_pct)
        messy_indices = np.random.choice(n_rows, messy_count, replace=False)
        
        for idx in messy_indices:
            # Mess up gender
            if random.random() < 0.3:
                current_gender = df.loc[idx, 'gender']
                messy_formats = {
                    'M': ['Male', 'MALE', 'm', '1', 'Man'],
                    'F': ['Female', 'FEMALE', 'f', '0', 'Woman', 'F ']
                }
                df.loc[idx, 'gender'] = random.choice(messy_formats[current_gender])
            
            # Mess up smoking status
            if random.random() < 0.3:
                current_smoking = df.loc[idx, 'smoking_status']
                if current_smoking == 'Smoker':
                    df.loc[idx, 'smoking_status'] = random.choice(['Y', 'Yes', 'S', '1', 'Current'])
                else:
                    df.loc[idx, 'smoking_status'] = random.choice(['N', 'No', 'NS', '0', 'Never', 'Non-Smoker'])
            
            # Mess up dates occasionally
            if random.random() < 0.2:
                issue_date = datetime.strptime(df.loc[idx, 'issue_date'], '%Y-%m-%d')
                messy_formats = [
                    issue_date.strftime('%m/%d/%Y'),
                    issue_date.strftime('%d-%b-%Y'),
                    issue_date.strftime('%Y%m%d'),
                    str(int(issue_date.timestamp()))  # Unix timestamp
                ]
                df.loc[idx, 'issue_date'] = random.choice(messy_formats)
        
        return df
    
    def generate_all_datasets(self, output_dir: str = "enterprise_data") -> Dict[str, str]:
        """Generate complete set of enterprise datasets"""
        
        self.logger.info("🏭 GENERATING ENTERPRISE DATASETS")
        self.logger.info("=" * 50)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        files_created = {}
        
        # 1. Generate policies (foundation dataset)
        policies = self.generate_enterprise_policies()
        policy_file = output_path / "enterprise_policies.csv"
        policies.to_csv(policy_file, index=False)
        files_created['policies'] = str(policy_file)
        
        # 2. Generate claims
        claims = self.generate_enterprise_claims(policies)
        claims_file = output_path / "enterprise_claims.csv"
        claims.to_csv(claims_file, index=False)
        files_created['claims'] = str(claims_file)
        
        # 3. Generate premium transactions
        premiums = self.generate_premium_transactions(policies)
        premiums_file = output_path / "enterprise_premiums.csv"
        premiums.to_csv(premiums_file, index=False)
        files_created['premiums'] = str(premiums_file)
        
        # 4. Generate expanded mortality table
        mortality = self._generate_expanded_mortality_table()
        mortality_file = output_path / "enterprise_mortality_table.csv"
        mortality.to_csv(mortality_file, index=False)
        files_created['mortality'] = str(mortality_file)
        
        # 5. Generate economic scenarios
        scenarios = self._generate_economic_scenarios()
        scenarios_file = output_path / "enterprise_economic_scenarios.csv"
        scenarios.to_csv(scenarios_file, index=False)
        files_created['scenarios'] = str(scenarios_file)
        
        # Summary
        self.logger.info("🎉 ENTERPRISE DATA GENERATION COMPLETE")
        self.logger.info("=" * 50)
        for data_type, file_path in files_created.items():
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            record_count = pd.read_csv(file_path).shape[0]
            self.logger.info(f"{data_type.upper():12}: {record_count:>8,} records ({file_size:.1f} MB)")
        
        return files_created
    
    def _generate_expanded_mortality_table(self) -> pd.DataFrame:
        """Generate comprehensive mortality table"""
        
        data = []
        for age in range(18, 101):
            for gender in ['M', 'F']:
                for smoking in ['NS', 'S']:
                    # Base mortality rate (per 1000)
                    if gender == 'M':
                        base_rate = 0.0005 * (1.085 ** age)  # Male mortality curve
                    else:
                        base_rate = 0.0003 * (1.082 ** age)  # Female mortality curve
                    
                    # Smoking multiplier
                    if smoking == 'S':
                        base_rate *= 2.5
                    
                    data.append({
                        'age': age,
                        'gender': gender,
                        'smoking_status': smoking,
                        'mortality_rate_per_1000': round(base_rate * 1000, 3),
                        'table_year': 2023,
                        'table_type': 'Standard',
                        'country': 'USA',
                        'adjustment_factor': 1.00
                    })
        
        return pd.DataFrame(data)
    
    def _generate_economic_scenarios(self) -> pd.DataFrame:
        """Generate economic scenario projections"""
        
        scenarios = ['Base', 'Optimistic', 'Pessimistic', 'Stress']
        years = list(range(2024, 2035))
        
        data = []
        for scenario in scenarios:
            for year in years:
                if scenario == 'Base':
                    risk_free = 4.0 - (year - 2024) * 0.1
                    equity_return = 8.5 + random.normalvariate(0, 2)
                elif scenario == 'Optimistic':
                    risk_free = 4.5 - (year - 2024) * 0.05
                    equity_return = 12.0 + random.normalvariate(0, 3)
                elif scenario == 'Pessimistic':
                    risk_free = 3.0 - (year - 2024) * 0.15
                    equity_return = 5.0 + random.normalvariate(0, 4)
                else:  # Stress
                    risk_free = 1.5 - (year - 2024) * 0.05
                    equity_return = -2.0 + random.normalvariate(0, 8)
                
                data.append({
                    'scenario': scenario,
                    'projection_year': year,
                    'risk_free_rate': max(0.001, risk_free / 100),
                    'equity_return': equity_return / 100,
                    'inflation_rate': max(0.001, random.normalvariate(0.025, 0.005)),
                    'gdp_growth': max(-0.05, random.normalvariate(0.025, 0.01)),
                    'scenario_probability': {
                        'Base': 0.50, 'Optimistic': 0.25, 
                        'Pessimistic': 0.20, 'Stress': 0.05
                    }[scenario]
                })
        
        return pd.DataFrame(data)

def main():
    """Generate enterprise datasets"""
    
    # Configuration for different scales
    configs = {
        'development': GenerationConfig(base_size=1000, scale_factor=10),      # 10K records
        'testing': GenerationConfig(base_size=1000, scale_factor=100),         # 100K records  
        'production': GenerationConfig(base_size=1000, scale_factor=1000),     # 1M records
        'enterprise': GenerationConfig(base_size=1000, scale_factor=10000)     # 10M records
    }
    
    # Generate different scales
    for scale_name, config in configs.items():
        print(f"\n🏭 Generating {scale_name.upper()} scale datasets...")
        generator = EnterpriseDataGenerator(config)
        files = generator.generate_all_datasets(f"enterprise_data_{scale_name}")
        print(f"✅ {scale_name.capitalize()} datasets ready!")

if __name__ == "__main__":
    main()