"""
Complete System Test
Tests enterprise data generation + Ollama processing + universal upload system
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_generation():
    """Test enterprise data generation"""
    
    print("🏭 TESTING ENTERPRISE DATA GENERATION")
    print("=" * 50)
    
    try:
        from src.data_generation.enterprise_data_generator import EnterpriseDataGenerator, GenerationConfig
        
        # Small test configuration
        config = GenerationConfig(base_size=100, scale_factor=10)  # 1000 records
        generator = EnterpriseDataGenerator(config)
        
        print(f"Generating {config.base_size * config.scale_factor:,} policy records...")
        
        # Generate policies
        policies = generator.generate_enterprise_policies()
        print(f"✅ Generated {len(policies):,} policy records")
        print(f"   Columns: {list(policies.columns)}")
        print(f"   Sample data:")
        print(policies.head(3).to_string())
        
        # Generate claims
        claims = generator.generate_enterprise_claims(policies)
        print(f"✅ Generated {len(claims):,} claim records")
        
        # Save test files
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        policy_file = test_data_dir / "test_policies.csv"
        claims_file = test_data_dir / "test_claims.csv"
        
        policies.to_csv(policy_file, index=False)
        claims.to_csv(claims_file, index=False)
        
        print(f"💾 Saved test files:")
        print(f"   Policies: {policy_file}")
        print(f"   Claims: {claims_file}")
        
        return policy_file, claims_file
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return None, None

def test_ollama_processing():
    """Test Ollama data processing"""
    
    print("\n🧠 TESTING OLLAMA DATA PROCESSING")
    print("=" * 50)
    
    try:
        from src.data_processing.ollama_data_processor import IntelligentDataProcessor, test_ollama_connection
        
        # Test Ollama connection first
        if not test_ollama_connection():
            print("❌ Ollama not available. Please run: ollama serve")
            return False
        
        print("✅ Ollama connection successful")
        
        # Create messy test data
        messy_data = {
            'pol_num': ['POL001', 'POL-002', 'P003'],
            'Age': [35, 42, '28'],  # Mixed types
            'Sex': ['M', 'Female', '1'],  # Various formats
            'Smoker': ['Y', 'No', 'Never'],  # Inconsistent
            'FaceAmt': [100000, '250,000', 150000],  # Mixed formats
            'IssueDate': ['2020-01-01', '1/15/2020', '2021-03-15']  # Different date formats
        }
        
        messy_df = pd.DataFrame(messy_data)
        messy_file = "test_data/messy_sample.csv"
        Path("test_data").mkdir(exist_ok=True)
        messy_df.to_csv(messy_file, index=False)
        
        print(f"📝 Created messy test data:")
        print(messy_df.to_string())
        
        # Process with AI
        processor = IntelligentDataProcessor()
        result = processor.process_uploaded_file(messy_file)
        
        if result.success:
            print(f"\n✅ AI Processing Successful!")
            print(f"   Identified as: {result.data_type}")
            print(f"   Quality score: {result.quality_score:.0f}/100")
            print(f"   Records processed: {len(result.standardized_data):,}")
            
            print(f"\n🔧 Standardized Data:")
            print(result.standardized_data.to_string())
            
            if result.issues:
                print(f"\n⚠️ Issues found: {result.issues}")
            if result.recommendations:
                print(f"\n💡 Recommendations: {result.recommendations}")
                
            return True
        else:
            print(f"❌ Processing failed: {result.issues}")
            return False
            
    except Exception as e:
        print(f"❌ Ollama processing failed: {e}")
        return False

def test_universal_upload():
    """Test universal upload interface"""
    
    print("\n📤 TESTING UNIVERSAL UPLOAD INTERFACE")
    print("=" * 50)
    
    try:
        # Create different file formats
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # 1. CSV with different separators
        csv_data = pd.DataFrame({
            'policy_id': ['POL001', 'POL002', 'POL003'],
            'issue_age': [35, 42, 28],
            'face_amount': [100000, 250000, 150000]
        })
        
        csv_file = test_dir / "test_standard.csv"
        semicolon_file = test_dir / "test_semicolon.csv"
        tab_file = test_dir / "test_tab.csv"
        
        csv_data.to_csv(csv_file, index=False)
        csv_data.to_csv(semicolon_file, index=False, sep=';')
        csv_data.to_csv(tab_file, index=False, sep='\t')
        
        print("✅ Created test files:")
        print(f"   Standard CSV: {csv_file}")
        print(f"   Semicolon CSV: {semicolon_file}")
        print(f"   Tab-separated: {tab_file}")
        
        # 2. Excel file
        excel_file = test_dir / "test_data.xlsx"
        csv_data.to_excel(excel_file, index=False)
        print(f"   Excel file: {excel_file}")
        
        # 3. JSON file
        json_file = test_dir / "test_data.json"
        csv_data.to_json(json_file, orient='records', indent=2)
        print(f"   JSON file: {json_file}")
        
        # 4. Fixed width format
        fixed_width_content = """POL001    35    100000
POL002    42    250000
POL003    28    150000"""
        
        fixed_file = test_dir / "test_fixed_width.txt"
        with open(fixed_file, 'w') as f:
            f.write(fixed_width_content)
        print(f"   Fixed width: {fixed_file}")
        
        print("\n📁 Test files ready for upload interface!")
        print("   Run: streamlit run ui/universal_data_upload.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload test setup failed: {e}")
        return False

def test_integration():
    """Test complete integration"""
    
    print("\n🔗 TESTING COMPLETE INTEGRATION")
    print("=" * 50)
    
    try:
        from src.data_generation.enterprise_data_generator import EnterpriseDataGenerator, GenerationConfig
        from src.data_processing.ollama_data_processor import IntelligentDataProcessor
        
        # 1. Generate realistic data
        config = GenerationConfig(base_size=50, scale_factor=2)  # 100 records
        generator = EnterpriseDataGenerator(config)
        policies = generator.generate_enterprise_policies()
        
        # 2. Add realistic messiness
        policies_messy = policies.copy()
        
        # Mess up some gender values
        policies_messy.loc[0, 'gender'] = 'Male'
        policies_messy.loc[1, 'gender'] = 'm'
        policies_messy.loc[2, 'smoking_status'] = 'Y'
        policies_messy.loc[3, 'smoking_status'] = 'No'
        
        # Save messy data
        messy_file = "test_data/integration_test.csv"
        policies_messy.to_csv(messy_file, index=False)
        
        print(f"📊 Generated {len(policies_messy):,} records with realistic issues")
        
        # 3. Process with AI
        processor = IntelligentDataProcessor()
        result = processor.process_uploaded_file(messy_file)
        
        if result.success:
            print(f"✅ End-to-end processing successful!")
            print(f"   Data type: {result.data_type}")
            print(f"   Quality: {result.quality_score:.0f}/100")
            print(f"   Original shape: {policies_messy.shape}")
            print(f"   Processed shape: {result.standardized_data.shape}")
            
            # Compare original vs standardized
            print(f"\n🔍 Data standardization comparison:")
            print("Original gender values:", policies_messy['gender'].value_counts().to_dict())
            print("Standardized gender values:", result.standardized_data['gender'].value_counts().to_dict())
            
            return True
        else:
            print(f"❌ Integration test failed: {result.issues}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_startup_check():
    """Check system startup requirements"""
    
    print("🔍 SYSTEM STARTUP CHECK")
    print("=" * 30)
    
    # Check Ollama
    try:
        from src.data_processing.ollama_data_processor import test_ollama_connection
        if test_ollama_connection():
            print("✅ Ollama: Connected")
            ollama_ok = True
        else:
            print("❌ Ollama: Not connected")
            print("   → Run: ollama serve")
            ollama_ok = False
    except:
        print("❌ Ollama: Import failed")
        ollama_ok = False
    
    # Check dependencies
    missing_deps = []
    
    try:
        import pandas
        print("✅ Pandas: Available")
    except:
        missing_deps.append("pandas")
    
    try:
        import numpy
        print("✅ NumPy: Available") 
    except:
        missing_deps.append("numpy")
        
    try:
        import streamlit
        print("✅ Streamlit: Available")
    except:
        missing_deps.append("streamlit")
    
    try:
        import plotly
        print("✅ Plotly: Available")
    except:
        missing_deps.append("plotly")
    
    try:
        import requests
        print("✅ Requests: Available")
    except:
        missing_deps.append("requests")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print(f"   → Run: pip install {' '.join(missing_deps)}")
    
    return ollama_ok and len(missing_deps) == 0

def main():
    """Run complete system test"""
    
    print("🚀 COMPLETE SYSTEM TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Startup check
    system_ok = run_startup_check()
    
    if not system_ok:
        print("\n❌ System requirements not met. Please fix issues above.")
        return
    
    print(f"\n✅ System requirements satisfied!")
    
    # Test 1: Data Generation
    policy_file, claims_file = test_data_generation()
    
    # Test 2: Ollama Processing (if available)
    ollama_ok = test_ollama_processing()
    
    # Test 3: Universal Upload Setup
    upload_ok = test_universal_upload()
    
    # Test 4: Integration Test
    if ollama_ok:
        integration_ok = test_integration()
    else:
        integration_ok = False
        print("\n⚠️ Skipping integration test (Ollama not available)")
    
    # Final Summary
    print(f"\n📊 SYSTEM TEST RESULTS")
    print("=" * 40)
    print(f"✅ Data Generation: {'PASS' if policy_file else 'FAIL'}")
    print(f"{'✅' if ollama_ok else '❌'} Ollama Processing: {'PASS' if ollama_ok else 'FAIL'}")
    print(f"✅ Upload Interface: {'PASS' if upload_ok else 'FAIL'}")
    print(f"{'✅' if integration_ok else '❌'} Integration: {'PASS' if integration_ok else 'FAIL'}")
    
    if policy_file and upload_ok:
        success_rate = "75-100%"
        status = "🎉 SYSTEM READY FOR USE!"
    elif policy_file:
        success_rate = "50-75%"
        status = "⚠️ PARTIAL FUNCTIONALITY"
    else:
        success_rate = "0-25%"
        status = "❌ SYSTEM NOT READY"
    
    print(f"\n📈 Overall Success Rate: {success_rate}")
    print(f"{status}")
    
    # Next Steps
    print(f"\n🚀 NEXT STEPS:")
    if ollama_ok:
        print("1. Launch UI: streamlit run ui/universal_data_upload.py")
        print("2. Upload test files from test_data/ folder")
        print("3. Watch AI process any data format automatically!")
    else:
        print("1. Start Ollama: ollama serve")
        print("2. Download model: ollama pull llama3.2")
        print("3. Then launch UI: streamlit run ui/universal_data_upload.py")
    
    print(f"\n💡 TEST DATA LOCATION: test_data/")
    print(f"🏭 ENTERPRISE GENERATOR: src/data_generation/")
    print(f"🧠 AI PROCESSOR: src/data_processing/")
    print(f"📤 UPLOAD UI: ui/universal_data_upload.py")

if __name__ == "__main__":
    main()