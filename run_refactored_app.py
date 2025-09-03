#!/usr/bin/env python3
"""
Launch the refactored PriceRe Smart Reinsurance Pricing Platform
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the refactored Streamlit application"""
    
    print("🚀 Starting PriceRe Smart Reinsurance Pricing Platform (Refactored)")
    print("📦 Using modular architecture:")
    print("  ├── ui/config/session_state.py (Session management)")
    print("  ├── ui/components/chat_sidebar.py (Chat integration)")
    print("  ├── ui/components/navigation.py (Navigation & progress)")
    print("  ├── ui/workflow/step4_pricing.py (Pricing calculations)")
    print("  └── ui/app.py (Main entry point)")
    print()
    
    # Get the application path
    app_path = Path(__file__).parent / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Application file not found: {app_path}")
        return 1
    
    print(f"📍 Running: streamlit run {app_path}")
    print("🌐 Access at: http://localhost:8501")
    print("💬 Chat server: http://localhost:8001 (if running)")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 PriceRe application stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return 1

if __name__ == "__main__":
    exit(main())