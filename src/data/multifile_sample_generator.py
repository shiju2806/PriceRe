"""
Realistic Multi-File Sample Data Generator for Reinsurance

Generates separate files that mirror real-world reinsurance company data structures:
- Treaty Master (contract terms)
- Claims History (individual claims)
- Policy Exposures (underlying risks)
- Market Data (economic indicators)
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """Configuration for multi-file sample generation"""
    num_treaties: int = 100
    claims_per_treaty_avg: int = 12
    policies_per_treaty_avg: int = 250
    years_of_history: int = 5
    missing_data_rate: float = 0.15  # 15% missing data like real world
    include_claims: bool = True
    include_exposures: bool = True
    include_market: bool = False

class MultiFileSampleGenerator:
    """Generates realistic multi-file reinsurance datasets"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        
        # Realistic data patterns
        self.business_lines = [
            "Property", "Casualty", "Motor", "Marine", "Aviation", 
            "Workers Compensation", "Health", "Disability"
        ]
        
        self.treaty_types = [
            "Quota Share", "Surplus", "Excess of Loss", "Catastrophe",
            "Life Quota Share", "Life Coinsurance"
        ]
        
        self.cedants = [
            "State Farm", "Allstate", "Progressive", "Travelers", "Chubb",
            "AIG", "Hartford", "Liberty Mutual", "Zurich", "Munich Re Client"
        ]
        
        self.reinsurers = [
            "Swiss Re", "Munich Re", "Berkshire Hathaway Re", "Lloyd's of London",
            "SCOR", "Hannover Re", "Transatlantic Re", "Partner Re", "Everest Re", 
            "RGA", "Odyssey Re"
        ]
        
        self.territories = [
            "United States", "Canada", "United Kingdom", "Germany", "France",
            "Australia", "Japan", "Netherlands", "Sweden"
        ]
        
        self.currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]
        
        self.loss_causes = [
            "Fire", "Wind", "Flood", "Earthquake", "Theft", "Collision",
            "Liability", "Workers Injury", "Medical Malpractice", "Cyber",
            "Hurricane", "Tornado", "Wildfire", "Hail"
        ]
        
    def generate_all_files(self, output_dir: str = "data/uploads") -> Dict[str, str]:
        """Generate all sample files and return file paths"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # 1. Generate Treaty Master (always required)
        treaty_df = self.generate_treaty_master()
        treaty_path = output_path / "sample_treaty_master.csv"
        treaty_df.to_csv(treaty_path, index=False)
        generated_files['treaty_master'] = str(treaty_path)
        
        # 2. Generate Claims History (if requested)
        if self.config.include_claims:
            claims_df = self.generate_claims_history(treaty_df)
            claims_path = output_path / "sample_claims_history.csv"
            claims_df.to_csv(claims_path, index=False)
            generated_files['claims_history'] = str(claims_path)
        
        # 3. Generate Policy Exposures (if requested)
        if self.config.include_exposures:
            exposures_df = self.generate_policy_exposures(treaty_df)
            exposures_path = output_path / "sample_policy_exposures.csv"
            exposures_df.to_csv(exposures_path, index=False)
            generated_files['policy_exposures'] = str(exposures_path)
        
        # 4. Generate Market Data (if requested)
        if self.config.include_market:
            market_df = self.generate_market_data()
            market_path = output_path / "sample_market_data.csv"
            market_df.to_csv(market_path, index=False)
            generated_files['market_data'] = str(market_path)
        
        return generated_files
    
    def generate_treaty_master(self) -> pd.DataFrame:
        """Generate realistic treaty master data"""
        
        treaties = []
        
        for i in range(self.config.num_treaties):
            # Generate treaty ID
            treaty_id = f"{random.choice(['TR', 'CT', 'RE'])}{random.randint(100000, 999999)}"
            
            # Select random characteristics
            business_line = random.choice(self.business_lines)
            treaty_type = random.choice(self.treaty_types)
            cedant = random.choice(self.cedants)
            reinsurer = random.choice(self.reinsurers)
            territory = random.choice(self.territories)
            currency = random.choice(self.currencies)
            
            # Generate realistic dates
            inception = self.start_date + timedelta(days=random.randint(0, 1460))  # 4 years
            expiry = inception + timedelta(days=365)  # 1 year terms
            
            # Generate realistic financial terms
            if treaty_type in ["Quota Share", "Life Quota Share"]:
                cession_rate = random.uniform(0.15, 0.85)
                retention = 1 - cession_rate
                limit = None
                commission = random.uniform(0.12, 0.35)
            elif treaty_type in ["Surplus"]:
                cession_rate = None
                retention = random.uniform(100000, 2000000)
                limit = retention * random.randint(3, 10)
                commission = random.uniform(0.15, 0.30)
            elif treaty_type == "Excess of Loss":
                cession_rate = None
                retention = random.uniform(1000000, 10000000)
                limit = retention * random.randint(2, 50)
                commission = 0.0  # Usually no commission on XOL
            elif treaty_type == "Catastrophe":
                cession_rate = None
                retention = random.uniform(10000000, 100000000)
                limit = retention * random.randint(5, 50)
                commission = 0.0
            else:  # Life Coinsurance
                cession_rate = random.uniform(0.60, 0.95)
                retention = 1 - cession_rate
                limit = None
                commission = random.uniform(0.02, 0.08)
            
            # Generate premium based on treaty type and size
            if treaty_type == "Catastrophe":
                premium = random.uniform(5000000, 50000000)
            elif treaty_type == "Excess of Loss":
                premium = random.uniform(1000000, 20000000)
            elif "Life" in treaty_type:
                premium = random.uniform(20000000, 200000000)
            else:
                premium = random.uniform(2000000, 100000000)
            
            # Generate realistic ratios
            loss_ratio = np.random.beta(2, 1.5) * 1.2  # Skewed towards higher values
            expense_ratio = random.uniform(0.15, 0.45)
            combined_ratio = loss_ratio + expense_ratio
            
            # Apply missing data patterns (realistic) with consistent types
            def maybe_missing_float(value, prob=self.config.missing_data_rate):
                return 0.0 if random.random() < prob else value
            
            def maybe_missing_string(value, prob=self.config.missing_data_rate):
                return "Not Available" if random.random() < prob else value
            
            def maybe_missing_int(value, prob=self.config.missing_data_rate):
                return 0 if random.random() < prob else value
            
            treaty = {
                'treaty_id': treaty_id,
                'treaty_name': f"{cedant} - {treaty_type} {inception.year}",
                'treaty_type': treaty_type,
                'business_line': business_line,
                'cedant': cedant,
                'reinsurer': reinsurer,
                'currency': currency,
                'territory': territory,
                'inception_date': inception.strftime('%Y-%m-%d'),
                'expiry_date': expiry.strftime('%Y-%m-%d'),
                'premium': premium,
                'commission': maybe_missing_float(commission) if commission is not None else 0.0,
                'brokerage': maybe_missing_float(random.uniform(0.005, 0.05)),
                'retention': maybe_missing_float(retention) if retention is not None else 0.0,
                'limit': maybe_missing_float(limit) if limit is not None else 0.0,
                'cession_rate': maybe_missing_float(cession_rate) if cession_rate is not None else 0.0,
                'minimum_premium': maybe_missing_float(premium * random.uniform(0.8, 0.95)),
                'maximum_premium': maybe_missing_float(premium * random.uniform(1.05, 1.2)),
                'profit_commission': maybe_missing_float(random.uniform(0.05, 0.20)),
                'loss_corridor': maybe_missing_string(f"{random.randint(75, 95)}%-{random.randint(105, 125)}%"),
                'aggregate_limit': maybe_missing_float(limit * random.randint(2, 5)) if limit is not None else 0.0,
                'reinstatements': random.choice([0, 1, 2, 3, 4]) if treaty_type == "Catastrophe" else 0,
                'rating_method': random.choice(['Experience Rating', 'Exposure Rating', 'Original Terms', 'Mortality Tables', 'Cat Modeling']),
                'experience_period': f"{random.choice([3, 5, 10, 15])} years",
                'loss_ratio': loss_ratio,
                'expense_ratio': expense_ratio,
                'combined_ratio': combined_ratio
            }
            
            treaties.append(treaty)
        
        # Create pandas DataFrame directly with explicit dtypes
        df = pd.DataFrame(treaties)
        
        # Define explicit dtypes for better consistency
        dtype_mapping = {
            'treaty_id': 'string',
            'treaty_name': 'string', 
            'treaty_type': 'string',
            'business_line': 'string',
            'cedant': 'string',
            'reinsurer': 'string',
            'currency': 'string',
            'territory': 'string',
            'inception_date': 'string',
            'expiry_date': 'string',
            'rating_method': 'string',
            'experience_period': 'string',
            'loss_corridor': 'string',
            'premium': 'float64',
            'commission': 'float64',
            'brokerage': 'float64', 
            'retention': 'float64',
            'limit': 'float64',
            'cession_rate': 'float64',
            'minimum_premium': 'float64',
            'maximum_premium': 'float64',
            'profit_commission': 'float64',
            'aggregate_limit': 'float64',
            'reinstatements': 'int64',
            'loss_ratio': 'float64',
            'expense_ratio': 'float64',
            'combined_ratio': 'float64'
        }
        
        # Apply dtypes where columns exist
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                if dtype.startswith('float'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                elif dtype.startswith('int'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'string':
                    df[col] = df[col].astype('string').fillna('Not Available')
        
        return df
    
    def generate_claims_history(self, treaty_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic claims history linked to treaties"""
        
        claims = []
        treaty_ids = treaty_df['treaty_id'].tolist()
        
        for treaty_id in treaty_ids:
            # Get treaty info for realistic claim generation
            treaty_info = treaty_df[treaty_df['treaty_id'] == treaty_id].iloc[0].to_dict()
            treaty_type = treaty_info['treaty_type']
            business_line = treaty_info['business_line']
            
            # Generate realistic number of claims based on treaty type
            if treaty_type == "Catastrophe":
                num_claims = max(1, int(np.random.poisson(2)))  # Few but large claims
            elif treaty_type == "Excess of Loss":
                num_claims = max(1, int(np.random.poisson(5)))  # Moderate frequency
            else:
                base_frequency = self.config.claims_per_treaty_avg
                num_claims = max(1, int(np.random.poisson(base_frequency)))
            
            for j in range(num_claims):
                # Generate claim ID
                claim_id = f"CLM{random.randint(1000000, 9999999)}"
                
                # Generate realistic claim dates
                inception_date = datetime.strptime(treaty_info['inception_date'], '%Y-%m-%d')
                expiry_date = datetime.strptime(treaty_info['expiry_date'], '%Y-%m-%d')
                
                loss_date = inception_date + timedelta(
                    days=random.randint(0, (expiry_date - inception_date).days)
                )
                reported_date = loss_date + timedelta(days=random.randint(1, 180))
                paid_date = reported_date + timedelta(days=random.randint(30, 720))
                
                # Generate realistic claim amounts based on treaty type
                if treaty_type == "Catastrophe":
                    claim_amount = np.random.lognormal(15, 1.5)  # Large claims
                elif treaty_type == "Excess of Loss":
                    retention = treaty_info.get('retention', 1000000)
                    if retention:
                        claim_amount = retention * random.uniform(0.5, 3.0)
                    else:
                        claim_amount = np.random.lognormal(13, 1.2)
                elif business_line == "Workers Compensation":
                    claim_amount = np.random.lognormal(10, 1.8)  # Medium-high variability
                elif business_line == "Motor":
                    claim_amount = np.random.lognormal(9, 1.5)   # Medium claims
                else:
                    claim_amount = np.random.lognormal(11, 1.6)  # General
                
                # Reserve amount (often higher than paid initially)
                reserve_amount = claim_amount * random.uniform(1.1, 2.5)
                
                # Recovery amount (subrogation, salvage) - consistent float type
                if random.random() < 0.15:
                    recovery_amount = claim_amount * random.uniform(0, 0.3)
                else:
                    recovery_amount = 0.0  # Use 0.0 instead of 0 for consistent float type
                
                # Geographic coordinates (if property-related) - consistent float types
                if business_line in ["Property", "Catastrophe"]:
                    # Focus on US coordinates for realism
                    latitude = random.uniform(25.0, 49.0)  # Continental US
                    longitude = random.uniform(-125.0, -66.0)
                else:
                    latitude = 0.0  # Use 0.0 instead of None for consistent float type
                    longitude = 0.0  # Use 0.0 instead of None for consistent float type
                
                # Paid date with consistent string type
                if random.random() > 0.1:
                    paid_date_str = paid_date.strftime('%Y-%m-%d')
                else:
                    paid_date_str = "Not Paid"  # Use consistent string instead of None
                
                # Catastrophe code with consistent string type
                if random.random() < 0.05:
                    catastrophe_code = f"CAT{random.randint(2020, 2024)}{random.randint(100, 999)}"
                else:
                    catastrophe_code = "Not Applicable"  # Use consistent string instead of None
                
                # Adjuster with consistent string type
                if random.random() > 0.2:
                    adjuster = random.choice(['Internal', 'External', 'TPA'])
                else:
                    adjuster = "Not Assigned"  # Use consistent string instead of None
                
                # Claim type with consistent string type
                if random.random() > 0.15:
                    claim_type = random.choice(['Property', 'Liability', 'Auto Physical Damage', 'Bodily Injury', 'Medical'])
                else:
                    claim_type = "Not Specified"  # Use consistent string instead of None
                
                # Ultimate loss with consistent float type
                if random.random() > 0.3:
                    ultimate_loss = claim_amount * random.uniform(0.95, 1.15)
                else:
                    ultimate_loss = claim_amount  # Use claim_amount instead of None for consistent float type
                
                claim = {
                    'claim_id': claim_id,
                    'treaty_id': treaty_id,
                    'loss_date': loss_date.strftime('%Y-%m-%d'),
                    'reported_date': reported_date.strftime('%Y-%m-%d'),
                    'paid_date': paid_date_str,
                    'claim_amount': claim_amount,
                    'reserve_amount': reserve_amount,
                    'recovery_amount': recovery_amount,
                    'cause_of_loss': random.choice(self.loss_causes),
                    'catastrophe_code': catastrophe_code,
                    'latitude': latitude,
                    'longitude': longitude,
                    'status': random.choice(['Open', 'Closed', 'Reopened', 'Pending']),
                    'adjuster': adjuster,
                    'claim_type': claim_type,
                    'development_year': (paid_date.year - loss_date.year),
                    'ultimate_loss': ultimate_loss
                }
                
                claims.append(claim)
        
        # Create pandas DataFrame directly with explicit dtypes
        df = pd.DataFrame(claims)
        
        # Define explicit dtypes for claims data
        dtype_mapping = {
            'claim_id': 'string',
            'treaty_id': 'string',
            'loss_date': 'string',
            'reported_date': 'string', 
            'paid_date': 'string',
            'cause_of_loss': 'string',
            'catastrophe_code': 'string',
            'status': 'string',
            'adjuster': 'string',
            'claim_type': 'string',
            'claim_amount': 'float64',
            'reserve_amount': 'float64',
            'recovery_amount': 'float64',
            'latitude': 'float64',
            'longitude': 'float64',
            'development_year': 'int64',
            'ultimate_loss': 'float64'
        }
        
        # Apply dtypes where columns exist
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                if dtype.startswith('float'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                elif dtype.startswith('int'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'string':
                    df[col] = df[col].astype('string').fillna('Not Available')
        
        return df
    
    def generate_policy_exposures(self, treaty_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic policy exposure data"""
        
        exposures = []
        treaty_ids = treaty_df['treaty_id'].tolist()
        
        for treaty_id in treaty_ids:
            # Get treaty info
            treaty_info = treaty_df[treaty_df['treaty_id'] == treaty_id].iloc[0].to_dict()
            business_line = treaty_info['business_line']
            
            # Generate realistic number of policies
            base_policies = self.config.policies_per_treaty_avg
            num_policies = max(10, int(np.random.poisson(base_policies)))
            
            for j in range(num_policies):
                policy_id = f"POL{random.randint(10000000, 99999999)}"
                
                # Generate realistic sum insured based on business line
                if business_line == "Property":
                    sum_insured = np.random.lognormal(12, 1.0)  # Property values
                elif business_line == "Motor":
                    sum_insured = random.uniform(15000, 150000)  # Vehicle values
                elif business_line == "Marine":
                    sum_insured = np.random.lognormal(14, 1.5)  # Vessel/cargo values
                elif business_line == "Aviation":
                    sum_insured = np.random.lognormal(16, 2.0)  # Aircraft values
                else:
                    sum_insured = np.random.lognormal(11, 1.2)  # General
                
                # Deductible (typically 1-10% of sum insured)
                deductible = sum_insured * random.uniform(0.01, 0.10)
                
                # Geographic distribution (realistic US focus)
                if random.random() < 0.7:  # 70% US exposure
                    latitude = random.uniform(25.0, 49.0)
                    longitude = random.uniform(-125.0, -66.0)
                    state = random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'])
                    country = 'US'
                else:  # International exposure
                    latitude = random.uniform(-60, 70)
                    longitude = random.uniform(-180, 180)
                    state = None
                    country = random.choice(['CA', 'UK', 'DE', 'FR', 'AU', 'JP'])
                
                # Generate occupancy with consistent string type
                if business_line == "Property":
                    occupancy = random.choice(['Residential', 'Commercial', 'Industrial', 'Agricultural', 'Mixed Use'])
                else:
                    occupancy = "Not Applicable"  # Use consistent string instead of None
                
                # Generate construction type with consistent string type  
                if business_line == "Property":
                    construction_type = random.choice(['Frame', 'Masonry', 'Steel', 'Concrete', 'Mixed'])
                else:
                    construction_type = "Not Applicable"  # Use consistent string instead of None
                
                # Generate year built with consistent int type
                if business_line in ["Property", "Motor"]:
                    year_built = random.randint(1950, 2023)
                else:
                    year_built = 0  # Use 0 instead of None for consistent int type
                
                # Generate protection class with consistent int type
                if business_line == "Property":
                    protection_class = random.randint(1, 10)
                else:
                    protection_class = 0  # Use 0 instead of None
                
                # Generate address with consistent string type
                if random.random() > 0.3:
                    address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Cedar', 'Elm'])} St"
                else:
                    address = "Not Available"  # Use consistent string instead of None
                
                # Generate zip code with consistent string type
                if random.random() > 0.2:
                    zip_code = f"{random.randint(10000, 99999)}"
                else:
                    zip_code = "00000"  # Use consistent string instead of None
                
                # Generate treaty_id with consistent string type
                if random.random() > 0.1:
                    policy_treaty_id = treaty_id
                else:
                    policy_treaty_id = "UNASSIGNED"  # Use consistent string instead of None
                
                exposure = {
                    'policy_id': policy_id,
                    'treaty_id': policy_treaty_id,
                    'sum_insured': sum_insured,
                    'deductible': deductible,
                    'latitude': latitude,
                    'longitude': longitude,
                    'occupancy': occupancy,
                    'construction_type': construction_type,
                    'year_built': year_built,
                    'protection_class': protection_class,
                    'coverage_type': random.choice(['Comprehensive', 'Named Perils', 'Basic', 'Broad']),
                    'policy_limits': sum_insured,
                    'address': address,
                    'zip_code': zip_code,
                    'state': state,
                    'country': country
                }
                
                exposures.append(exposure)
        
        # Create pandas DataFrame directly with explicit dtypes
        df = pd.DataFrame(exposures)
        
        # Define explicit dtypes for exposures data
        dtype_mapping = {
            'policy_id': 'string',
            'treaty_id': 'string',
            'occupancy': 'string',
            'construction_type': 'string',
            'coverage_type': 'string',
            'address': 'string',
            'zip_code': 'string',
            'state': 'string',
            'country': 'string',
            'sum_insured': 'float64',
            'deductible': 'float64',
            'latitude': 'float64',
            'longitude': 'float64',
            'policy_limits': 'float64',
            'year_built': 'int64',
            'protection_class': 'int64'
        }
        
        # Apply dtypes where columns exist
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                if dtype.startswith('float'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                elif dtype.startswith('int'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'string':
                    df[col] = df[col].astype('string').fillna('Not Available')
        
        return df
    
    def generate_market_data(self) -> pd.DataFrame:
        """Generate realistic market/economic data time series"""
        
        market_data = []
        current_date = self.start_date
        
        # Initialize realistic starting values
        gdp_growth = 2.1
        interest_rate_10y = 2.8
        sp500_level = 2800
        vix_level = 18.5
        
        while current_date <= self.end_date:
            # Simulate economic indicators with realistic volatility
            gdp_growth += random.gauss(0, 0.3)
            gdp_growth = max(-5, min(8, gdp_growth))  # Constrain range
            
            interest_rate_10y += random.gauss(0, 0.15)
            interest_rate_10y = max(0.5, min(6.0, interest_rate_10y))
            
            sp500_level *= (1 + random.gauss(0.0007, 0.012))  # Daily returns
            sp500_level = max(1000, sp500_level)
            
            vix_level *= (1 + random.gauss(-0.001, 0.05))
            vix_level = max(9, min(80, vix_level))
            
            # Insurance-specific indicators
            cat_pcs_index = 100 + random.gauss(0, 25)
            insurance_stocks_index = sp500_level * random.uniform(0.8, 1.2)
            hard_market_indicator = 1 if vix_level > 25 or gdp_growth < 0 else 0
            regulatory_capital_ratio = random.uniform(1.5, 3.0)
            
            market_point = {
                'date': current_date.strftime('%Y-%m-%d'),
                'gdp_growth': gdp_growth,
                'interest_rate_10y': interest_rate_10y,
                'cat_pcs_index': cat_pcs_index,
                'insurance_stocks_index': insurance_stocks_index,
                'hard_market_indicator': hard_market_indicator,
                'regulatory_capital_ratio': regulatory_capital_ratio,
                'inflation_rate': max(0, gdp_growth * 0.6 + random.gauss(0, 0.5)),
                'unemployment_rate': max(2, min(15, 5.5 - gdp_growth * 0.3 + random.gauss(0, 0.3))),
                'sp500_index': sp500_level,
                'vix_index': vix_level
            }
            
            market_data.append(market_point)
            
            # Move to next month
            current_date += timedelta(days=30)
        
        # Create pandas DataFrame directly with explicit dtypes
        df = pd.DataFrame(market_data)
        
        # Define explicit dtypes for market data
        dtype_mapping = {
            'date': 'string',
            'gdp_growth': 'float64',
            'interest_rate_10y': 'float64',
            'cat_pcs_index': 'float64',
            'insurance_stocks_index': 'float64',
            'hard_market_indicator': 'int64',
            'regulatory_capital_ratio': 'float64',
            'inflation_rate': 'float64',
            'unemployment_rate': 'float64',
            'sp500_index': 'float64',
            'vix_index': 'float64'
        }
        
        # Apply dtypes where columns exist
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                if dtype.startswith('float'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                elif dtype.startswith('int'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'string':
                    df[col] = df[col].astype('string').fillna('Not Available')
        
        return df


def generate_realistic_multifile_samples(
    complexity: str = "Simple (100 treaties)",
    include_claims: bool = True,
    include_exposures: bool = True,
    include_market: bool = False,
    output_dir: str = "data/uploads"
) -> Dict[str, str]:
    """
    Generate realistic multi-file sample datasets
    
    Returns dictionary of {file_type: file_path}
    """
    
    # Parse complexity level
    if "100" in complexity:
        num_treaties = 100
    elif "500" in complexity:
        num_treaties = 500
    else:
        num_treaties = 1000
    
    # Configure generation
    config = GenerationConfig(
        num_treaties=num_treaties,
        include_claims=include_claims,
        include_exposures=include_exposures,
        include_market=include_market,
        missing_data_rate=0.12  # Realistic missing data
    )
    
    # Generate files
    generator = MultiFileSampleGenerator(config)
    generated_files = generator.generate_all_files(output_dir)
    
    return generated_files


if __name__ == "__main__":
    # Test generation
    files = generate_realistic_multifile_samples(
        complexity="Simple (100 treaties)",
        include_claims=True,
        include_exposures=True,
        include_market=True
    )
    
    print("Generated files:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")