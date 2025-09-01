#!/usr/bin/env python3
"""
Test Excel upload functionality directly
"""

import pandas as pd
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_excel_processing():
    """Test Excel file processing functionality"""
    
    excel_file = "test_messy_data.xlsx"
    print(f"🧪 Testing Excel processing with: {excel_file}")
    
    try:
        # Test basic pandas Excel reading
        print("\n1️⃣ Testing basic Excel reading...")
        df = pd.read_excel(excel_file)
        print(f"✅ Successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        print("\n📊 Raw Excel data:")
        print(df.head(10))
        
        # Test enhanced profiler
        print("\n2️⃣ Testing Enhanced Profiler...")
        from src.data_cleaning.enhanced_profiler import EnhancedDataProfiler
        
        profiler = EnhancedDataProfiler()
        profile = profiler.profile_data(df)
        
        print(f"✅ Profiling completed!")
        print(f"📈 Data Quality: {profile['data_quality']['overall_completeness']:.1f}% complete")
        print(f"🚨 Issues found: {len(profile['recommendations'])}")
        
        # Show key recommendations
        print(f"\n💡 Key recommendations:")
        for i, rec in enumerate(profile['recommendations'][:5], 1):
            print(f"  {i}. [{rec['category']}] {rec['recommendation']}")
        
        # Test cleaning actions
        print(f"\n3️⃣ Testing cleaning actions...")
        test_actions = [rec['action'] for rec in profile['recommendations'][:3]]
        
        if test_actions:
            cleaned_df = profiler.apply_cleaning_actions(df, test_actions)
            print(f"✅ Cleaning applied: {df.shape} → {cleaned_df.shape}")
            
            print(f"\n📊 Sample cleaned data:")
            print(cleaned_df.head(5))
        
        return True
        
    except Exception as e:
        print(f"❌ Excel processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_excel_processing()
    if success:
        print(f"\n✅ Excel upload and processing is working correctly!")
        print(f"🎯 The file upload issue is likely resolved - try uploading Excel files now")
    else:
        print(f"\n❌ Excel processing still has issues")