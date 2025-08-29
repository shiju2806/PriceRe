"""
Test script for comprehensive data generator
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.actuarial.data_preparation import (
    ComprehensiveActuarialDataGenerator,
    DataGenerationConfig,
    quick_generate_test_data
)

def test_quick_generation():
    """Test quick data generation"""
    print("🧪 Testing Quick Data Generation...")
    
    datasets = quick_generate_test_data("small")
    
    print(f"✅ Generated {len(datasets)} dataset types:")
    for name, df in datasets.items():
        if not df.empty:
            print(f"  • {name}: {len(df):,} records")
    
    return datasets

def test_comprehensive_generation():
    """Test comprehensive data generation"""
    print("\n🧪 Testing Comprehensive Data Generation...")
    
    config = DataGenerationConfig(
        n_policies=500,
        data_quality_issues=True,
        include_medical_data=True,
        include_geographic_data=True
    )
    
    generator = ComprehensiveActuarialDataGenerator(config)
    datasets = generator.generate_comprehensive_dataset()
    
    # Show summary report
    print(generator.generate_summary_report(datasets))
    
    return datasets

def test_data_quality():
    """Test data quality aspects"""
    print("\n🧪 Testing Data Quality Features...")
    
    datasets = quick_generate_test_data("small")
    policy_df = datasets['policy_data']
    
    print("📊 Data Quality Analysis:")
    print(f"  • Missing values: {policy_df.isnull().sum().sum()}")
    print(f"  • Duplicate policy IDs: {policy_df['policy_id'].duplicated().sum()}")
    print(f"  • Face amount range: ${policy_df['face_amount'].min():,.0f} - ${policy_df['face_amount'].max():,.0f}")
    print(f"  • Age range: {policy_df['issue_age'].min()} - {policy_df['issue_age'].max()}")
    
    # Product distribution
    print("  • Product distribution:")
    product_counts = policy_df['product_type'].value_counts()
    for product, count in product_counts.head().items():
        pct = count / len(policy_df) * 100
        print(f"    - {product}: {count} ({pct:.1f}%)")
    
    return True

if __name__ == "__main__":
    try:
        # Run tests
        test_quick_generation()
        test_comprehensive_generation()
        test_data_quality()
        
        print("\n✅ All tests passed! Data generator is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()