#!/usr/bin/env python3
"""
Install Streamlit and create demo interface
"""

import subprocess
import sys

def install_streamlit():
    """Install Streamlit in virtual environment"""
    print("📦 Installing Streamlit...")
    try:
        result = subprocess.run([
            "venv/bin/pip3", "install", "streamlit", "plotly"
        ], check=True, capture_output=True, text=True)
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False

if __name__ == "__main__":
    success = install_streamlit()
    if success:
        print("🎯 Run the demo with: streamlit run ui/demo.py")
    else:
        print("❌ Installation failed")
        sys.exit(1)