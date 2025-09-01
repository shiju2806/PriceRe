#!/usr/bin/env python3
"""
Test OpenAI GPT-4o-mini Integration for PriceRe Chat
Quick test to verify the OpenAI API is working correctly
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv not available, try manual loading
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_openai_integration():
    """Test the OpenAI GPT-4o-mini integration"""
    
    print("🧪 Testing OpenAI GPT-4o-mini Integration...")
    print("=" * 50)
    
    # Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Make sure to set it in your .env file")
        return False
    
    print(f"✅ API Key configured: {api_key[:20]}...{api_key[-4:]}")
    
    try:
        # Import OpenAI engine
        from chat.openai_chat_engine import OpenAIChatEngine, OPENAI_AVAILABLE
        
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI package not available")
            print("💡 Run: pip install openai")
            return False
        
        print("✅ OpenAI package available")
        
        # Create engine
        engine = OpenAIChatEngine()
        print(f"✅ Chat engine created with model: {engine.model}")
        
        # Get model info
        info = engine.get_model_info()
        print(f"📊 Model Info: {info}")
        
        # Test connection
        print("\n🧠 Testing GPT-4o-mini connection...")
        test_result = engine.test_connection()
        print(f"🔗 Connection Test: {test_result}")
        
        # Test reinsurance query
        print("\n💼 Testing reinsurance expertise...")
        response = engine.chat("What is the difference between quota share and surplus reinsurance treaties?")
        print(f"🎯 Reinsurance Response: {response[:200]}...")
        
        # Test data cleaning query
        print("\n🧹 Testing data cleaning expertise...")
        response = engine.chat("How would you identify junk rows in an insurance dataset?")
        print(f"📊 Data Cleaning Response: {response[:200]}...")
        
        print("\n" + "=" * 50)
        print("🎉 OpenAI GPT-4o-mini integration test PASSED!")
        print("✅ PriceRe Chat is ready for intelligent conversations")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Check your OpenAI API key is valid")
        print("2. Ensure you have sufficient OpenAI credits")
        print("3. Verify internet connection")
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    sys.exit(0 if success else 1)