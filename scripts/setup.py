#!/usr/bin/env python3
"""
Setup script for PricingFlow development environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 PricingFlow Setup")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("❌ PricingFlow requires Python 3.10 or higher")
        return 1
    
    # Create necessary directories
    directories = [
        "data/synthetic",
        "data/sample", 
        "data/external",
        "models/saved",
        "logs",
        "outputs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Install core dependencies first
    core_deps = [
        "polars==0.20.26",
        "numpy==1.24.3", 
        "pandas==2.0.3",
        "faker==24.0.0",
        "scikit-learn==1.3.0"
    ]
    
    print(f"\n📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip3 install {dep}", f"Installing {dep.split('==')[0]}"):
            print(f"⚠️ Failed to install {dep}, trying without version pin...")
            dep_name = dep.split('==')[0]
            if not run_command(f"pip3 install {dep_name}", f"Installing {dep_name}"):
                print(f"❌ Could not install {dep_name}")
                return 1
    
    # Try to install all requirements
    print(f"\n📦 Installing all requirements...")
    if Path("requirements.txt").exists():
        success = run_command("pip3 install -r requirements.txt", "Installing all requirements")
        if not success:
            print("⚠️ Some packages failed to install, but core dependencies are ready")
    
    # Test imports
    print(f"\n🧪 Testing core imports...")
    try:
        import polars as pl
        print("✅ Polars imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Polars: {e}")
        return 1
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return 1
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pandas: {e}")
        return 1
    
    try:
        from faker import Faker
        print("✅ Faker imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Faker: {e}")
        return 1
    
    print(f"\n✅ PricingFlow setup completed successfully!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Generate sample data: python3 scripts/generate_sample_data.py")
    print(f"   2. Test the system: python3 scripts/test_basic_functionality.py")
    
    return 0

if __name__ == "__main__":
    exit(main())