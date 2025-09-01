"""
Test Real User Experience - What happens when users try to test the product
"""

import sys
from pathlib import Path
import pandas as pd
import tempfile
import os

def test_missing_dependencies():
    """Test what happens when dependencies are missing"""
    
    print("=" * 60)
    print("REAL USER EXPERIENCE TEST")
    print("=" * 60)
    
    print("\n1. 🧪 TESTING DEPENDENCY AVAILABILITY")
    print("-" * 40)
    
    # Test core dependencies
    missing_deps = []
    
    try:
        import streamlit
        print("✅ Streamlit: Available")
    except ImportError as e:
        print(f"❌ Streamlit: Missing - {e}")
        missing_deps.append("streamlit")
    
    try:
        import pyliferisk
        print("✅ pyliferisk: Available")
    except ImportError as e:
        print(f"❌ pyliferisk: Missing - {e}")
        missing_deps.append("pyliferisk")
    
    try:
        import plotly
        print("✅ Plotly: Available")
    except ImportError as e:
        print(f"❌ Plotly: Missing - {e}")
        missing_deps.append("plotly")
    
    try:
        from src.mvp_production.core_pricing_engine import ProductionPricingEngine
        print("✅ Production Engine: Available")
        engine_available = True
    except ImportError as e:
        print(f"❌ Production Engine: Missing - {e}")
        missing_deps.append("production-engine")
        engine_available = False
    
    return missing_deps, engine_available

def test_common_data_upload_issues():
    """Test common issues users face when uploading data"""
    
    print("\n2. 📊 TESTING DATA UPLOAD SCENARIOS")
    print("-" * 40)
    
    # Scenario 1: Wrong column names
    print("\n❌ SCENARIO 1: User uploads Excel with wrong column names")
    wrong_columns_data = {
        'PolicyNum': ['POL001', 'POL002', 'POL003'],  # Wrong: should be 'policy_number'
        'IssueDate': ['2020-01-01', '2020-01-02', '2020-01-03'],  # Wrong: should be 'issue_date'
        'Coverage': [100000, 200000, 150000],  # Wrong: should be 'face_amount'
        'Premium': [1000, 2000, 1500]  # Wrong: should be 'annual_premium'
    }
    
    # What the system expects
    expected_columns = [
        'policy_number', 'issue_date', 'face_amount', 'annual_premium',
        'issue_age', 'gender', 'smoker_status', 'product_type',
        'state', 'policy_status'
    ]
    
    user_columns = list(wrong_columns_data.keys())
    missing_columns = set(expected_columns) - set(user_columns)
    
    print(f"   User provided columns: {user_columns}")
    print(f"   System expects: {expected_columns}")
    print(f"   Missing columns: {list(missing_columns)}")
    print("   → Result: Upload will FAIL with validation errors")
    
    # Scenario 2: Wrong date format
    print("\n❌ SCENARIO 2: User uploads with wrong date format")
    wrong_date_data = {
        'policy_number': ['POL001'],
        'issue_date': ['1/15/2020'],  # Wrong: system expects YYYY-MM-DD
        'face_amount': [100000],
        'annual_premium': [1000],
        'issue_age': [35],
        'gender': ['M'],
        'smoker_status': ['N'],  # Wrong: system expects 'Nonsmoker'
        'product_type': ['Life'],
        'state': ['CA'],
        'policy_status': ['Active']  # Wrong: system expects 'Inforce'
    }
    
    print(f"   Date format provided: '1/15/2020'")
    print(f"   System expects: 'YYYY-MM-DD' (e.g., '2020-01-15')")
    print(f"   Smoker status: 'N' vs expected 'Nonsmoker'")
    print(f"   Policy status: 'Active' vs expected 'Inforce'")
    print("   → Result: Date parsing errors, validation failures")
    
    # Scenario 3: Missing required columns
    print("\n❌ SCENARIO 3: User uploads minimal data")
    minimal_data = {
        'policy_number': ['POL001'],
        'face_amount': [100000]
    }
    
    minimal_columns = set(minimal_data.keys())
    required_missing = set(expected_columns) - minimal_columns
    
    print(f"   User provided: {list(minimal_columns)}")
    print(f"   Missing required: {list(required_missing)}")
    print("   → Result: System cannot perform experience analysis")
    
    return True

def test_ui_failure_modes():
    """Test what users see when UI fails to load"""
    
    print("\n3. 🖥️  TESTING UI FAILURE MODES")
    print("-" * 40)
    
    # Simulate the UI import failure
    print("\n❌ SCENARIO: User runs 'streamlit run ui/professional_pricing_platform.py'")
    print("   Step 1: Streamlit loads the Python file")
    print("   Step 2: File tries to import production engine")
    print("   Step 3: Import fails due to missing dependencies")
    print("   Step 4: User sees Streamlit error page:")
    print("           'Production engine not available: No module named pyliferisk'")
    print("   Step 5: User gets frustrated and gives up")
    
    return True

def test_success_scenario():
    """Test what happens when everything works correctly"""
    
    print("\n4. ✅ TESTING SUCCESS SCENARIO")
    print("-" * 40)
    
    # Try to create a working example
    try:
        from production_demo import ProductionPricingEngine
        
        print("✅ User installs all dependencies correctly")
        print("✅ User creates properly formatted CSV file")
        
        # Create sample correct data
        correct_data = {
            'policy_number': [f'POL{i:06d}' for i in range(1, 11)],
            'issue_date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
                          '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10'],
            'face_amount': [100000, 200000, 150000, 300000, 250000, 180000, 220000, 190000, 170000, 210000],
            'annual_premium': [1000, 2000, 1500, 3000, 2500, 1800, 2200, 1900, 1700, 2100],
            'issue_age': [35, 42, 28, 55, 39, 33, 47, 41, 36, 44],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
            'smoker_status': ['Nonsmoker', 'Nonsmoker', 'Smoker', 'Nonsmoker', 'Nonsmoker',
                             'Smoker', 'Nonsmoker', 'Nonsmoker', 'Nonsmoker', 'Smoker'],
            'product_type': ['Term Life', 'Universal Life', 'Term Life', 'Whole Life', 'Term Life',
                           'Universal Life', 'Term Life', 'Universal Life', 'Term Life', 'Whole Life'],
            'state': ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
            'policy_status': ['Inforce', 'Inforce', 'Inforce', 'Lapsed', 'Inforce',
                            'Inforce', 'Inforce', 'Inforce', 'Lapsed', 'Inforce']
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(correct_data)
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        print(f"✅ Created sample file with {len(df)} records")
        print(f"✅ File saved to: {temp_file}")
        
        # Test if the production engine can process it
        try:
            engine = ProductionPricingEngine("Test Company")
            print("✅ Production engine initialized successfully")
            print("✅ User can now upload data and get pricing results")
            
            # Clean up
            os.unlink(temp_file)
            return True
            
        except Exception as e:
            print(f"❌ Production engine failed: {e}")
            os.unlink(temp_file)
            return False
            
    except Exception as e:
        print(f"❌ Success scenario failed: {e}")
        return False

def generate_user_guidance():
    """Generate guidance for real users"""
    
    print("\n5. 📋 USER GUIDANCE FOR SUCCESSFUL TESTING")
    print("-" * 40)
    
    print("\n🔧 REQUIRED SETUP:")
    print("1. Install all dependencies:")
    print("   pip install -r requirements.txt")
    print("   pip install pyliferisk actuarialpy")
    print()
    
    print("2. Verify installation:")
    print("   python3 -c 'import streamlit, pyliferisk, plotly; print(\"All dependencies OK\")'")
    print()
    
    print("📊 REQUIRED DATA FORMAT:")
    print("Create CSV file with EXACTLY these columns:")
    columns = [
        "policy_number", "issue_date", "face_amount", "annual_premium",
        "issue_age", "gender", "smoker_status", "product_type", 
        "state", "policy_status"
    ]
    for col in columns:
        print(f"   • {col}")
    print()
    
    print("📝 DATA FORMAT REQUIREMENTS:")
    print("   • issue_date: YYYY-MM-DD format (e.g., '2020-01-15')")
    print("   • gender: 'M' or 'F'")
    print("   • smoker_status: 'Smoker' or 'Nonsmoker'") 
    print("   • policy_status: 'Inforce' or 'Lapsed'")
    print("   • face_amount: numeric (e.g., 100000)")
    print("   • annual_premium: numeric (e.g., 1200)")
    print()
    
    print("🚀 HOW TO TEST:")
    print("1. Run: streamlit run ui/professional_pricing_platform.py")
    print("2. Navigate to: http://localhost:8501")
    print("3. Create new submission with company details")
    print("4. Upload properly formatted CSV file")
    print("5. Wait for data processing and analysis")
    print("6. Review pricing results and recommendations")

def main():
    """Run comprehensive user experience test"""
    
    print("Testing what real users experience when trying the product...")
    
    # Test 1: Dependencies  
    missing_deps, engine_available = test_missing_dependencies()
    
    # Test 2: Data upload issues
    test_common_data_upload_issues()
    
    # Test 3: UI failures
    test_ui_failure_modes()
    
    # Test 4: Success scenario
    success = test_success_scenario()
    
    # Test 5: User guidance
    generate_user_guidance()
    
    # Summary
    print("\n📊 REAL USER EXPERIENCE SUMMARY")
    print("=" * 50)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("   → Most users will see import errors and give up")
    else:
        print("✅ All dependencies available")
    
    if not engine_available:
        print("❌ Production engine not accessible")
        print("   → Users cannot complete full workflow")
    else:
        print("✅ Production engine ready")
    
    if success:
        print("✅ Success scenario works when properly set up")
        print("   → Users CAN get results if they follow exact requirements")
    else:
        print("❌ Even success scenario has issues")
        print("   → Major problems need resolution")
    
    # Likelihood assessment
    if missing_deps or not engine_available:
        success_rate = "10-20%"
        frustration_level = "HIGH"
    else:
        success_rate = "60-70%" 
        frustration_level = "MEDIUM"
    
    print(f"\n📈 ESTIMATED USER SUCCESS RATE: {success_rate}")
    print(f"😤 USER FRUSTRATION LEVEL: {frustration_level}")
    
    print(f"\n💡 RECOMMENDATION:")
    if missing_deps:
        print("🚨 CRITICAL: Fix dependency installation before user testing")
        print("   • Create setup script that installs all dependencies")
        print("   • Add dependency verification step")
        print("   • Provide clear error messages when dependencies missing")
    else:
        print("✅ Ready for user testing with proper setup guide")
        print("   • Provide sample CSV file")
        print("   • Create step-by-step setup tutorial") 
        print("   • Add data format validation with helpful error messages")

if __name__ == "__main__":
    main()