#!/bin/bash
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
