#!/usr/bin/env python3
"""
User Environment Setup Script
Prepares the system for real user testing of the reinsurance pricing platform
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

def install_dependencies():
    """Install all required dependencies"""
    
    print("🔧 INSTALLING DEPENDENCIES...")
    print("-" * 40)
    
    # Core dependencies that might be missing
    critical_deps = [
        "streamlit==1.31.0",
        "plotly==5.18.0", 
        "pandas==2.0.3",
        "numpy==1.24.3",
        "openpyxl==3.1.2"
    ]
    
    # Actuarial dependencies (these often fail)
    actuarial_deps = [
        "pyliferisk==1.12"
    ]
    
    success_count = 0
    total_deps = len(critical_deps) + len(actuarial_deps)
    
    # Install critical dependencies
    for dep in critical_deps:
        try:
            print(f"Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {dep}")
                success_count += 1
            else:
                print(f"❌ {dep}: {result.stderr}")
        except Exception as e:
            print(f"❌ {dep}: {e}")
    
    # Install actuarial dependencies (may fail, provide alternatives)
    for dep in actuarial_deps:
        try:
            print(f"Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {dep}")
                success_count += 1
            else:
                print(f"⚠️ {dep}: Failed (will use fallback calculations)")
                print(f"   Error: {result.stderr[:100]}...")
        except Exception as e:
            print(f"⚠️ {dep}: Failed (will use fallback calculations)")
            print(f"   Error: {e}")
    
    print(f"\n📊 Installation Summary: {success_count}/{total_deps} successful")
    
    return success_count >= len(critical_deps)  # Must have critical deps

def verify_installation():
    """Verify that key components can be imported"""
    
    print("\n🧪 VERIFYING INSTALLATION...")
    print("-" * 40)
    
    tests = [
        ("streamlit", "import streamlit as st"),
        ("plotly", "import plotly.express as px"),
        ("pandas", "import pandas as pd"),
        ("production engine", "from production_demo import ProductionPricingEngine")
    ]
    
    working = []
    broken = []
    
    for name, import_code in tests:
        try:
            exec(import_code)
            print(f"✅ {name}")
            working.append(name)
        except Exception as e:
            print(f"❌ {name}: {e}")
            broken.append(name)
    
    return working, broken

def create_sample_data():
    """Create sample data files for user testing"""
    
    print("\n📊 CREATING SAMPLE DATA FILES...")
    print("-" * 40)
    
    # Create sample_data directory
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # 1. Perfect format sample
    perfect_data = {
        'policy_number': [f'SAMPLE_{i:05d}' for i in range(1, 101)],
        'issue_date': ['2020-01-01'] * 25 + ['2020-06-01'] * 25 + ['2021-01-01'] * 25 + ['2021-06-01'] * 25,
        'face_amount': [100000, 200000, 150000, 300000] * 25,
        'annual_premium': [1000, 2000, 1500, 3000] * 25,
        'issue_age': [25, 35, 45, 55] * 25,
        'gender': ['M', 'F'] * 50,
        'smoker_status': ['Nonsmoker'] * 85 + ['Smoker'] * 15,
        'product_type': ['Term Life'] * 60 + ['Universal Life'] * 25 + ['Whole Life'] * 15,
        'state': ['CA', 'NY', 'TX', 'FL', 'IL'] * 20,
        'policy_status': ['Inforce'] * 90 + ['Lapsed'] * 10
    }
    
    perfect_df = pd.DataFrame(perfect_data)
    perfect_file = sample_dir / "perfect_format_sample.csv"
    perfect_df.to_csv(perfect_file, index=False)
    print(f"✅ Created: {perfect_file} ({len(perfect_df)} records)")
    
    # 2. Larger realistic sample
    import numpy as np
    np.random.seed(42)
    
    n_policies = 1000
    realistic_data = {
        'policy_number': [f'REAL_{i:06d}' for i in range(1, n_policies + 1)],
        'issue_date': pd.date_range(start='2018-01-01', periods=n_policies, freq='D')[:n_policies].strftime('%Y-%m-%d'),
        'face_amount': np.random.lognormal(12.4, 0.5, n_policies).astype(int),
        'annual_premium': np.random.lognormal(8.5, 0.4, n_policies).astype(int),
        'issue_age': np.random.normal(42, 12, n_policies).astype(int).clip(18, 80),
        'gender': np.random.choice(['M', 'F'], n_policies, p=[0.52, 0.48]),
        'smoker_status': np.random.choice(['Nonsmoker', 'Smoker'], n_policies, p=[0.85, 0.15]),
        'product_type': np.random.choice(['Term Life', 'Universal Life', 'Whole Life'], 
                                       n_policies, p=[0.60, 0.30, 0.10]),
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA'], n_policies),
        'policy_status': np.random.choice(['Inforce', 'Lapsed'], n_policies, p=[0.88, 0.12])
    }
    
    realistic_df = pd.DataFrame(realistic_data)
    realistic_file = sample_dir / "realistic_portfolio_sample.csv"
    realistic_df.to_csv(realistic_file, index=False)
    print(f"✅ Created: {realistic_file} ({len(realistic_df)} records)")
    
    # 3. Common mistake examples (for testing error handling)
    wrong_format_data = {
        'PolicyNumber': ['WRONG001', 'WRONG002'],  # Wrong column name
        'IssueDate': ['1/1/2020', '2/1/2020'],    # Wrong date format
        'FaceAmount': ['$100,000', '$200,000'],   # String with currency symbols
        'Premium': [1000, 2000]                   # Missing other required columns
    }
    
    wrong_df = pd.DataFrame(wrong_format_data)
    wrong_file = sample_dir / "common_mistakes_example.csv"
    wrong_df.to_csv(wrong_file, index=False)
    print(f"✅ Created: {wrong_file} (demonstrates common errors)")
    
    return [perfect_file, realistic_file, wrong_file]

def create_user_guide():
    """Create comprehensive user guide"""
    
    print("\n📖 CREATING USER GUIDE...")
    print("-" * 40)
    
    guide_content = """# 🏛️ Reinsurance Pricing Platform - User Guide

## Quick Start

### 1. 🔧 Setup (First Time Only)
```bash
# Install dependencies
python3 setup_user_environment.py

# Verify installation
python3 -c "import streamlit, pandas, plotly; print('✅ Ready to go!')"
```

### 2. 🚀 Launch the Platform
```bash
streamlit run ui/professional_pricing_platform.py
```
Then open: http://localhost:8501

### 3. 📊 Test with Sample Data
Use the provided sample files in `sample_data/`:
- `perfect_format_sample.csv` - Small perfect example (100 policies)  
- `realistic_portfolio_sample.csv` - Large realistic example (1,000 policies)
- `common_mistakes_example.csv` - Shows what NOT to do

## Data Format Requirements

### Required CSV Columns (Exact Names)
```
policy_number    - Unique identifier (e.g., "POL123456")
issue_date       - Date in YYYY-MM-DD format (e.g., "2020-01-15")
face_amount      - Death benefit amount (e.g., 100000)
annual_premium   - Yearly premium (e.g., 1200)
issue_age        - Age at issue (e.g., 35)
gender           - "M" or "F"
smoker_status    - "Smoker" or "Nonsmoker"
product_type     - "Term Life", "Universal Life", or "Whole Life"
state            - 2-letter state code (e.g., "CA")
policy_status    - "Inforce" or "Lapsed"
```

### ❌ Common Mistakes to Avoid
- Wrong column names (PolicyNumber vs policy_number)
- Wrong date format (1/15/2020 vs 2020-01-15)
- Currency symbols in amounts ($100,000 vs 100000)
- Wrong status values (Active vs Inforce)
- Missing required columns

## Typical Workflow

1. **Create Submission**
   - Enter cedent (insurance company) details
   - Specify treaty type and financial information
   - Set submission parameters

2. **Upload Policy Data**
   - Use properly formatted CSV file
   - System validates data quality (target: 90%+ score)
   - Review any validation warnings

3. **Experience Analysis**
   - System calculates mortality statistics
   - Assesses portfolio risk factors  
   - Determines statistical credibility

4. **Final Pricing**
   - Calculates expected loss ratio
   - Applies expense and risk margins
   - Provides pricing recommendations

5. **Review Results**
   - Final gross rate and confidence level
   - Sensitivity analysis scenarios
   - Professional recommendations

## Troubleshooting

### "Production engine not available"
- Run: `python3 setup_user_environment.py`
- Ensure all dependencies installed successfully

### "Data validation failed"
- Check CSV column names match exactly
- Verify date format: YYYY-MM-DD
- Ensure no currency symbols or commas in numeric fields

### "Statistical credibility too low"
- Need minimum ~500 policies for reliable results
- Consider using industry benchmarks
- System will warn about low credibility scenarios

## Support

For issues or questions:
1. Check this guide first
2. Review sample data files for correct format
3. Run diagnostics: `python3 test_real_user_experience.py`
"""
    
    guide_file = Path("USER_GUIDE.md")
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Created: {guide_file}")
    return guide_file

def create_startup_script():
    """Create easy startup script for users"""
    
    print("\n🚀 CREATING STARTUP SCRIPT...")
    print("-" * 40)
    
    # Create startup script
    startup_content = """#!/bin/bash
# Easy startup script for the Reinsurance Pricing Platform

echo "🏛️ Starting Reinsurance Pricing Platform..."
echo "========================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Running setup..."
    python3 setup_user_environment.py
    echo ""
fi

echo "🚀 Launching platform..."
echo "📱 Open browser to: http://localhost:8501"
echo "📖 Press Ctrl+C to stop"
echo ""

streamlit run ui/professional_pricing_platform.py --server.headless false --server.port 8501
"""
    
    startup_file = Path("start_platform.sh")
    with open(startup_file, 'w') as f:
        f.write(startup_content)
    
    # Make executable
    os.chmod(startup_file, 0o755)
    
    print(f"✅ Created: {startup_file}")
    print("   Usage: ./start_platform.sh")
    
    return startup_file

def main():
    """Main setup process"""
    
    print("🏛️ REINSURANCE PRICING PLATFORM - USER ENVIRONMENT SETUP")
    print("=" * 65)
    print("Preparing system for real user testing...")
    print()
    
    # Step 1: Install dependencies
    deps_ok = install_dependencies()
    
    # Step 2: Verify installation  
    working, broken = verify_installation()
    
    # Step 3: Create sample data
    sample_files = create_sample_data()
    
    # Step 4: Create user guide
    guide_file = create_user_guide()
    
    # Step 5: Create startup script
    startup_file = create_startup_script()
    
    # Final assessment
    print("\n📊 SETUP SUMMARY")
    print("=" * 30)
    
    if deps_ok and len(working) >= 3:
        print("✅ SETUP SUCCESSFUL!")
        print("   → Users can now test the platform")
        print()
        print("🚀 NEXT STEPS FOR USERS:")
        print("1. Run: ./start_platform.sh")
        print("2. Open: http://localhost:8501")
        print("3. Upload: sample_data/perfect_format_sample.csv")
        print("4. Follow the workflow in the UI")
        print()
        print(f"📖 User Guide: {guide_file}")
        print(f"📊 Sample Data: {len(sample_files)} files in sample_data/")
        
        estimated_success_rate = "70-80%"
        
    else:
        print("⚠️ PARTIAL SETUP")
        print(f"   Working: {working}")
        print(f"   Broken: {broken}")
        print()
        print("🔧 MANUAL FIXES NEEDED:")
        if not deps_ok:
            print("• Install missing dependencies manually")
        if 'production engine' in broken:
            print("• Fix pyliferisk installation")
        if 'streamlit' in broken:
            print("• Install streamlit: pip install streamlit")
            
        estimated_success_rate = "40-50%"
    
    print(f"\n📈 Estimated User Success Rate: {estimated_success_rate}")
    
    # Provide fallback instructions
    print("\n💡 FALLBACK FOR USERS:")
    print("If UI doesn't work, users can still test core functionality:")
    print("   python3 production_demo.py")
    print("   → This runs the complete workflow without UI dependencies")

if __name__ == "__main__":
    main()