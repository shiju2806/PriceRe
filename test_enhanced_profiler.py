#!/usr/bin/env python3
"""
Test the Enhanced Data Profiler
"""

import pandas as pd
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_cleaning.enhanced_profiler import EnhancedDataProfiler

def test_enhanced_profiler():
    """Test enhanced profiler with integration test data"""
    
    # Load the messy test data
    test_file = "test_data/integration_test.csv"
    print(f"Loading test data from {test_file}...")
    
    try:
        df = pd.read_csv(test_file)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show first few rows
        print("\n📊 Sample data:")
        print(df.head(3))
        
        # Initialize profiler
        print("\n🔍 Initializing Enhanced Profiler...")
        profiler = EnhancedDataProfiler()
        
        # Profile the data
        print("\n📋 Profiling data...")
        profile = profiler.profile_data(df)
        
        # Show key results
        print("\n📈 PROFILING RESULTS:")
        print("="*50)
        
        print(f"Data Shape: {profile['basic_info']['shape']}")
        print(f"Duplicate Rows: {profile['basic_info']['duplicate_rows']}")
        print(f"Empty Rows: {profile['basic_info']['completely_empty_rows']}")
        print(f"Overall Completeness: {profile['data_quality']['overall_completeness']:.1f}%")
        
        print(f"\n🚨 Structural Issues ({len(profile['structural_issues'])}):")
        for issue in profile['structural_issues']:
            print(f"  • {issue}")
        
        print(f"\n📋 Column Analysis:")
        for col, analysis in profile['column_analysis'].items():
            issues = len(analysis['issues'])
            print(f"  {col} ({analysis['dtype']}): {issues} issues")
            if issues > 0:
                for issue in analysis['issues']:
                    print(f"    - {issue}")
        
        print(f"\n💡 Recommendations ({len(profile['recommendations'])}):")
        for i, rec in enumerate(profile['recommendations'], 1):
            print(f"  {i}. [{rec['category']}] {rec['recommendation']}")
            print(f"     Issue: {rec['issue']}")
            print(f"     Action: {rec['action']}")
        
        # Test cleaning actions
        print(f"\n🧹 TESTING CLEANING ACTIONS:")
        print("="*50)
        
        # Get a few recommended actions to test
        test_actions = []
        for rec in profile['recommendations'][:5]:  # Test first 5
            test_actions.append(rec['action'])
        
        if test_actions:
            print(f"Testing {len(test_actions)} actions:")
            for action in test_actions:
                print(f"  • {action}")
            
            # Apply cleaning
            cleaned_df = profiler.apply_cleaning_actions(df, test_actions)
            
            # Show results
            print(f"\n📊 CLEANING RESULTS:")
            print(f"Before: {df.shape} | After: {cleaned_df.shape}")
            
            # Get cleaning summary
            summary = profiler.get_cleaning_summary()
            if summary['history']:
                latest = summary['history'][-1]
                print(f"\n✅ Applied actions:")
                for action in latest['actions']:
                    print(f"  • {action}")
            
            # Show sample of cleaned data
            print(f"\n📊 Sample cleaned data:")
            print(cleaned_df.head(3))
        
        print(f"\n✅ Enhanced Profiler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_profiler()
    if success:
        print("\n🎉 All tests passed! Enhanced profiler is working correctly.")
    else:
        print("\n💥 Tests failed! Check error messages above.")
        sys.exit(1)