"""
Simple, robust sample data generator using only pandas
Avoids all Polars schema inference issues
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_simple_multifile_samples(num_treaties: int = 1000, output_dir: str = "data/uploads"):
    """Generate realistic sample data using only pandas - no schema issues"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    # 1. Generate Treaty Master
    print(f"Generating {num_treaties} treaties...")
    treaties = []
    
    business_lines = ["Property", "Casualty", "Motor", "Marine", "Aviation", "Health"]
    treaty_types = ["Quota Share", "Surplus", "Excess of Loss", "Catastrophe"]
    cedants = ["State Farm", "Allstate", "AIG", "Hartford", "Chubb"]
    reinsurers = ["Swiss Re", "Munich Re", "Lloyd's of London", "SCOR", "Hannover Re"]
    territories = ["United States", "Canada", "United Kingdom", "Germany", "France"]
    currencies = ["USD", "EUR", "GBP", "CAD"]
    
    for i in range(num_treaties):
        treaty_id = f"TR{random.randint(100000, 999999)}"
        business_line = random.choice(business_lines)
        treaty_type = random.choice(treaty_types)
        cedant = random.choice(cedants)
        reinsurer = random.choice(reinsurers)
        
        # Generate realistic financial terms based on treaty type
        premium = random.uniform(1000000, 100000000)
        loss_ratio = random.uniform(0.4, 1.2)
        expense_ratio = random.uniform(0.15, 0.45)
        
        # Treaty type specific parameters
        if treaty_type == "Quota Share":
            cession_rate = random.uniform(0.20, 0.80)
            retention = 1 - cession_rate  # For quota share, retention is the percentage kept
            limit = 0.0  # Quota share doesn't have limits
            aggregate_limit = 0.0
            reinstatements = 0
            commission = random.uniform(0.15, 0.35)
        elif treaty_type == "Surplus":
            cession_rate = 0.0
            retention = random.uniform(500000, 2000000)
            limit = retention * random.randint(3, 10)
            aggregate_limit = 0.0
            reinstatements = 0
            commission = random.uniform(0.15, 0.30)
        elif treaty_type == "Excess of Loss":
            cession_rate = 0.0
            retention = random.uniform(1000000, 10000000)
            limit = retention * random.randint(2, 20)
            aggregate_limit = 0.0
            reinstatements = 0
            commission = 0.0  # Usually no commission on XOL
        else:  # Catastrophe
            cession_rate = 0.0
            retention = random.uniform(10000000, 100000000)
            limit = retention * random.randint(5, 20)
            aggregate_limit = limit * random.randint(2, 5)
            reinstatements = random.randint(1, 4)
            commission = 0.0
        
        # Additional realistic columns expected by feature engineering
        minimum_premium = premium * random.uniform(0.8, 0.95) if random.random() > 0.3 else 0.0
        maximum_premium = premium * random.uniform(1.05, 1.2) if random.random() > 0.3 else 0.0
        profit_commission = random.uniform(0.05, 0.20) if random.random() > 0.4 else 0.0
        loss_corridor = f"{random.randint(75, 95)}%-{random.randint(105, 125)}%" if random.random() > 0.5 else "Not Available"
        
        treaties.append({
            'treaty_id': treaty_id,
            'treaty_name': f"{cedant} - {treaty_type} {2024}",
            'treaty_type': treaty_type,
            'business_line': business_line,
            'cedant': cedant,
            'reinsurer': reinsurer,
            'currency': random.choice(currencies),
            'territory': random.choice(territories),
            'inception_date': '2024-01-01',
            'expiry_date': '2024-12-31',
            'premium': premium,
            'commission': commission,
            'brokerage': random.uniform(0.01, 0.05),
            'retention': retention,
            'limit': limit,
            'cession_rate': cession_rate,
            'minimum_premium': minimum_premium,
            'maximum_premium': maximum_premium,
            'profit_commission': profit_commission,
            'loss_corridor': loss_corridor,
            'aggregate_limit': aggregate_limit,
            'reinstatements': reinstatements,
            'rating_method': random.choice(['Experience Rating', 'Exposure Rating', 'Original Terms', 'Cat Modeling']),
            'experience_period': f"{random.choice([3, 5, 10, 15])} years",
            'loss_ratio': loss_ratio,
            'expense_ratio': expense_ratio,
            'combined_ratio': loss_ratio + expense_ratio
        })
    
    # Save treaty master
    treaty_df = pd.DataFrame(treaties)
    treaty_path = output_path / "sample_treaty_master.csv"
    treaty_df.to_csv(treaty_path, index=False)
    generated_files['treaty_master'] = str(treaty_path)
    print(f"✅ Treaty master: {len(treaty_df)} records")
    
    # 2. Generate Claims History
    print("Generating claims history...")
    claims = []
    
    loss_causes = ["Fire", "Wind", "Flood", "Earthquake", "Theft", "Collision", "Liability"]
    
    for _, treaty in treaty_df.iterrows():
        # Generate 5-15 claims per treaty
        num_claims = random.randint(5, 15)
        
        for j in range(num_claims):
            claim_id = f"CLM{random.randint(1000000, 9999999)}"
            claim_amount = random.uniform(10000, 5000000)
            
            claims.append({
                'claim_id': claim_id,
                'treaty_id': treaty['treaty_id'],
                'loss_date': '2024-03-15',
                'reported_date': '2024-03-20',
                'paid_date': '2024-06-15',
                'claim_amount': claim_amount,
                'reserve_amount': claim_amount * 1.2,
                'recovery_amount': 0.0,
                'cause_of_loss': random.choice(loss_causes),
                'catastrophe_code': 'Not Applicable',
                'latitude': random.uniform(25.0, 49.0),
                'longitude': random.uniform(-125.0, -66.0),
                'status': random.choice(['Open', 'Closed', 'Pending']),
                'adjuster': random.choice(['Internal', 'External', 'TPA']),
                'claim_type': random.choice(['Property', 'Liability', 'Auto Physical Damage']),
                'development_year': 1,
                'ultimate_loss': claim_amount
            })
    
    # Save claims
    claims_df = pd.DataFrame(claims)
    claims_path = output_path / "sample_claims_history.csv"
    claims_df.to_csv(claims_path, index=False)
    generated_files['claims_history'] = str(claims_path)
    print(f"✅ Claims history: {len(claims_df)} records")
    
    # 3. Generate Policy Exposures
    print("Generating policy exposures...")
    exposures = []
    
    occupancies = ["Residential", "Commercial", "Industrial", "Agricultural"]
    constructions = ["Frame", "Masonry", "Steel", "Concrete"]
    
    for _, treaty in treaty_df.iterrows():
        # Generate 100-500 policies per treaty
        num_policies = random.randint(100, 500)
        
        for j in range(num_policies):
            policy_id = f"POL{random.randint(10000000, 99999999)}"
            sum_insured = random.uniform(100000, 5000000)
            
            exposures.append({
                'policy_id': policy_id,
                'treaty_id': treaty['treaty_id'],
                'sum_insured': sum_insured,
                'deductible': sum_insured * random.uniform(0.01, 0.10),
                'latitude': random.uniform(25.0, 49.0),
                'longitude': random.uniform(-125.0, -66.0),
                'occupancy': random.choice(occupancies),
                'construction_type': random.choice(constructions),
                'year_built': random.randint(1950, 2023),
                'protection_class': random.randint(1, 10),
                'coverage_type': random.choice(['Comprehensive', 'Named Perils', 'Basic']),
                'policy_limits': sum_insured,
                'address': f"{random.randint(100, 9999)} Main St",
                'zip_code': f"{random.randint(10000, 99999)}",
                'state': random.choice(['CA', 'TX', 'FL', 'NY', 'IL']),
                'country': 'US'
            })
    
    # Save exposures
    exposures_df = pd.DataFrame(exposures)
    exposures_path = output_path / "sample_policy_exposures.csv"
    exposures_df.to_csv(exposures_path, index=False)
    generated_files['policy_exposures'] = str(exposures_path)
    print(f"✅ Policy exposures: {len(exposures_df)} records")
    
    print(f"🎯 Total generated: {len(treaty_df)} treaties, {len(claims_df)} claims, {len(exposures_df)} policies")
    
    return generated_files

def generate_realistic_multifile_samples(
    complexity: str = "Simple (100 treaties)",
    include_claims: bool = True,
    include_exposures: bool = True,
    include_market: bool = False,
    output_dir: str = "data/uploads"
):
    """Main function that mirrors the original interface"""
    
    # Parse complexity
    if "100" in complexity:
        num_treaties = 100
    elif "500" in complexity:  
        num_treaties = 500
    else:
        num_treaties = 1000
        
    print(f"🚀 Generating {complexity} dataset...")
    
    return generate_simple_multifile_samples(num_treaties, output_dir)

if __name__ == "__main__":
    files = generate_realistic_multifile_samples("Large (1,000+ treaties)")
    print("Generated files:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")