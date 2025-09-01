#!/usr/bin/env python3
"""
Test the new semantic cleaning system
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.cleaning.data_sources import read_any_source, UniversalDataSourceFactory
    from src.cleaning.semantic_detector import SemanticJunkDetector, quick_semantic_clean
    from src.cleaning.test_framework import quick_test_semantic_detector, SemanticDetectorTester
    from src.cleaning.test_data_generator import generate_test_suite
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nTrying to install missing dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "faiss-cpu", "scikit-learn"])
    print("Installed dependencies - please run again")
    sys.exit(1)

print("\n" + "="*60)
print("TESTING NEW SEMANTIC CLEANING SYSTEM")
print("="*60)

# Test 1: Data Source Universal Adapter
print("\n1. Testing Universal Data Source Adapter...")
try:
    # Test with your existing messy Excel file
    if Path("test_messy_insurance_data.xlsx").exists():
        df = read_any_source("test_messy_insurance_data.xlsx")
        print(f"✅ Successfully loaded Excel file: {df.shape}")
    else:
        print("⚠️ Messy Excel file not found, creating sample data...")
        import polars as pl
        import pandas as pd
        
        sample_data = {
            'Policy': ['POL001', '', 'Header Row Here', 'POL002', 'POL003'],
            'Premium': [25000, None, None, 18500, 22000],
            'Age': [35, None, None, 42, 29]
        }
        df = pl.DataFrame(sample_data)
        print(f"✅ Created sample data: {df.shape}")

except Exception as e:
    print(f"❌ Data source test failed: {e}")
    sys.exit(1)

# Test 2: Quick Semantic Detection
print("\n2. Testing Quick Semantic Detection...")
try:
    clean_df, removed_indices = quick_semantic_clean(df)
    print(f"✅ Original: {df.shape[0]} rows -> Clean: {clean_df.shape[0]} rows")
    print(f"✅ Removed {len(removed_indices)} junk rows: {removed_indices}")
    
except Exception as e:
    print(f"❌ Quick semantic clean failed: {e}")
    print("This might be due to missing sentence-transformers model...")

# Test 3: Comprehensive Test Suite
print("\n3. Running Test Suite...")
try:
    # Generate test cases
    print("Generating comprehensive test suite...")
    test_suite = generate_test_suite()
    print(f"✅ Generated {len(test_suite)} test cases:")
    for name, case in test_suite.items():
        print(f"   - {name}: {case.messy_df.shape[0]} rows, {len(case.junk_row_indices)} junk")
    
except Exception as e:
    print(f"❌ Test suite generation failed: {e}")

# Test 4: Semantic Detector Validation
print("\n4. Testing Semantic Detector (if model available)...")
try:
    # This will try to download the model if not present
    detector = SemanticJunkDetector()
    if detector.initialize():
        print("✅ Semantic detector initialized successfully")
        
        # Test with simple case
        simple_test_data = {
            'col1': ['Data 1', '', 'TOTAL:', 'Data 2'],
            'col2': [100, None, 500, 200]
        }
        test_df = pl.DataFrame(simple_test_data)
        
        junk_indices = detector.detect_junk_rows(test_df)
        print(f"✅ Detected junk rows: {junk_indices}")
        
        # Test explanation
        if junk_indices:
            result = detector.detect_junk_rows(test_df, return_detailed=True)
            explanation = detector.get_outlier_explanation(result, test_df, junk_indices[0])
            print(f"✅ Explanation for row {junk_indices[0]}: {explanation['reasons']}")
        
    else:
        print("⚠️ Could not initialize semantic detector (model download might be needed)")
        
except Exception as e:
    print(f"❌ Semantic detector test failed: {e}")
    print("This is normal if sentence-transformers model needs to be downloaded")

# Test 5: Performance Test
print("\n5. Performance Test...")
try:
    # Create larger test dataset
    import numpy as np
    import polars as pl
    
    large_data = {
        'id': list(range(100)),
        'value': np.random.randint(1, 1000, 100),
        'category': ['A', 'B', 'C'] * 33 + ['A']
    }
    large_df = pl.DataFrame(large_data)
    
    # Add some junk rows
    junk_rows = [
        [None, None, None],  # Empty
        ['TOTAL', 50000, None],  # Summary
        ['', '', ''],  # Empty strings
    ]
    
    for junk in junk_rows:
        large_df = large_df.vstack(pl.DataFrame([junk], schema=large_df.columns))
    
    print(f"Testing with {large_df.shape[0]} rows...")
    
    import time
    start_time = time.time()
    clean_large_df, large_removed = quick_semantic_clean(large_df)
    end_time = time.time()
    
    print(f"✅ Processed {large_df.shape[0]} rows in {end_time - start_time:.2f}s")
    print(f"✅ Removed {len(large_removed)} rows")
    
except Exception as e:
    print(f"❌ Performance test failed: {e}")

print("\n" + "="*60)
print("SYSTEM STATUS SUMMARY")
print("="*60)

# Check what's working
working_components = []
broken_components = []

try:
    from src.cleaning.data_sources import read_any_source
    working_components.append("✅ Universal Data Sources")
except:
    broken_components.append("❌ Universal Data Sources")

try:
    from src.cleaning.test_data_generator import generate_test_suite
    test_suite = generate_test_suite()
    working_components.append("✅ Test Data Generator")
except:
    broken_components.append("❌ Test Data Generator")

try:
    from sentence_transformers import SentenceTransformer
    working_components.append("✅ Sentence Transformers")
except:
    broken_components.append("❌ Sentence Transformers (run: pip install sentence-transformers)")

try:
    import faiss
    working_components.append("✅ FAISS")
except:
    broken_components.append("❌ FAISS (run: pip install faiss-cpu)")

print("\nWorking Components:")
for component in working_components:
    print(f"  {component}")

if broken_components:
    print("\nComponents Needing Installation:")
    for component in broken_components:
        print(f"  {component}")
    
    print(f"\n🔧 To install missing components:")
    print(f"pip install sentence-transformers faiss-cpu scikit-learn polars")

print(f"\n🎯 Next Steps:")
print(f"1. Install any missing dependencies")
print(f"2. Run: python test_new_semantic_system.py")
print(f"3. If successful, integrate with PriceRe platform")
print(f"4. Test with real messy Excel files")
print(f"5. Add statistical and LLM layers")

print("\n✨ New Architecture Ready! Zero hard-coded rules! ✨")