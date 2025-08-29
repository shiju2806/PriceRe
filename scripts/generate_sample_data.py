#!/usr/bin/env python3
"""
Generate sample synthetic insurance data for testing PricingFlow
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

try:
    from src.utils.synthetic_data import generate_sample_data, DataGenerationConfig
except ImportError:
    # Try direct import
    from utils.synthetic_data import generate_sample_data, DataGenerationConfig
import polars as pl
from datetime import date

def main():
    print("🚀 PricingFlow Sample Data Generator")
    print("="*50)
    
    # Configuration
    config = DataGenerationConfig(
        n_records=10000,  # Start with 10K records
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        age_range=(18, 75),
        smoking_rate=0.12,
        realistic_correlations=True,
        include_medical_history=True,
        include_financial_data=True
    )
    
    print(f"Generating {config.n_records} life insurance records...")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Age range: {config.age_range[0]} to {config.age_range[1]}")
    
    try:
        # Generate data
        life_df, annuity_df = generate_sample_data(
            n_life=config.n_records,
            n_annuity=config.n_records // 2,  # 5K annuity records
            output_dir="data/synthetic"
        )
        
        print("\n✅ Sample data generation completed successfully!")
        print(f"📊 Life Insurance Dataset: {len(life_df)} records, {len(life_df.columns)} columns")
        print(f"📊 Annuity Dataset: {len(annuity_df)} records, {len(annuity_df.columns)} columns")
        
        # Display sample statistics
        print("\n📈 Life Insurance Sample Statistics:")
        if 'age_at_issue' in life_df.columns:
            print(f"   Average age: {life_df['age_at_issue'].mean():.1f}")
        if 'face_amount' in life_df.columns:
            print(f"   Average face amount: ${life_df['face_amount'].mean():,.0f}")
        if 'annual_premium' in life_df.columns:
            print(f"   Average premium: ${life_df['annual_premium'].mean():,.0f}")
        
        print("\n📈 Annuity Sample Statistics:")
        if 'age_at_contract' in annuity_df.columns:
            print(f"   Average age: {annuity_df['age_at_contract'].mean():.1f}")
        if 'premium_amount' in annuity_df.columns:
            print(f"   Average premium: ${annuity_df['premium_amount'].mean():,.0f}")
        
        print("\n📁 Files saved to data/synthetic/:")
        print("   - life_insurance_synthetic.csv")
        print("   - annuity_synthetic.csv") 
        print("   - data_summary.json")
        
        print("\n🎯 Next steps:")
        print("   1. Run: python scripts/test_pricing_models.py")
        print("   2. Or start the demo: streamlit run ui/demo.py")
        
    except Exception as e:
        print(f"\n❌ Error generating sample data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())