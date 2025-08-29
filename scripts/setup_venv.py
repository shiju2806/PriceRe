#!/usr/bin/env python3
"""
Setup PricingFlow with virtual environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, check=True):
    """Run command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"⚠️ {description} completed with warnings")
            if result.stderr:
                print(f"Warning: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 PricingFlow Setup with Virtual Environment")
    print("="*60)
    
    # Check if virtual environment already exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        activate_script = venv_path / "bin" / "activate"
        if not activate_script.exists():
            print("❌ Virtual environment appears corrupted, removing...")
            run_command("rm -rf venv", "Removing corrupted venv")
    
    # Create virtual environment
    if not venv_path.exists():
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            print("❌ Failed to create virtual environment")
            return 1
    
    # Create activation helper script
    activation_script = """#!/bin/bash
echo "🚀 Activating PricingFlow virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "📍 You are now in: $(which python3)"
echo ""
echo "🎯 Available commands:"
echo "   python3 scripts/generate_sample_data.py  # Generate test data"
echo "   python3 scripts/test_basic_functionality.py  # Test system"
echo "   deactivate  # Exit virtual environment"
echo ""
"""
    
    with open("activate_pricingflow.sh", "w") as f:
        f.write(activation_script)
    
    run_command("chmod +x activate_pricingflow.sh", "Making activation script executable")
    
    # Install packages in virtual environment
    pip_path = "venv/bin/pip3"
    
    # Core dependencies first
    core_deps = [
        "polars",
        "numpy", 
        "pandas",
        "faker",
        "scikit-learn",
        "datetime"
    ]
    
    print(f"\n📦 Installing core dependencies in virtual environment...")
    for dep in core_deps:
        if dep == "datetime":
            continue  # datetime is built-in
        success = run_command(f"{pip_path} install {dep}", f"Installing {dep}", check=False)
        if not success:
            print(f"⚠️ {dep} installation had issues, but continuing...")
    
    # Test the virtual environment
    print(f"\n🧪 Testing virtual environment...")
    test_command = """
venv/bin/python3 -c "
try:
    import polars as pl
    print('✅ Polars: OK')
except:
    print('❌ Polars: Failed')

try:
    import numpy as np
    print('✅ NumPy: OK')
except:
    print('❌ NumPy: Failed')
    
try:
    import pandas as pd
    print('✅ Pandas: OK')
except:
    print('❌ Pandas: Failed')

try:
    from faker import Faker
    print('✅ Faker: OK')
except:
    print('❌ Faker: Failed')
"
"""
    
    run_command(test_command, "Testing imports in virtual environment", check=False)
    
    # Create directories
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
    
    print(f"\n✅ PricingFlow setup completed!")
    print(f"\n🎯 To get started:")
    print(f"   1. Activate environment: ./activate_pricingflow.sh")
    print(f"   2. Or manually: source venv/bin/activate")
    print(f"   3. Generate data: python3 scripts/generate_sample_data.py")
    
    return 0

if __name__ == "__main__":
    exit(main())