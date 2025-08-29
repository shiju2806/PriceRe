#!/bin/bash
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
