#!/usr/bin/env python3
"""
Launch script for PriceRe Chat Server
Starts FastAPI chat server with proper environment setup
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    # Get project root
    project_root = Path(__file__).parent
    chat_server_path = project_root / "api" / "chat_server.py"
    
    print("🚀 PriceRe Chat Server Launcher")
    print(f"📁 Project Root: {project_root}")
    print(f"🔧 Chat Server: {chat_server_path}")
    
    # Check if chat server exists
    if not chat_server_path.exists():
        print(f"❌ Chat server not found at {chat_server_path}")
        sys.exit(1)
    
    # Load environment variables
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"✅ Loading environment from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
        print("   OpenAI API key may not be available")
    
    try:
        # Start the FastAPI chat server
        print("\n🌟 Starting FastAPI Chat Server...")
        print("📍 Chat Interface: http://localhost:8001")
        print("🔌 WebSocket API: ws://localhost:8001/ws/{user_id}")
        print("🌐 REST API: http://localhost:8001/docs")
        print("\n💡 Press Ctrl+C to stop the server\n")
        
        # Run the server
        subprocess.run([
            sys.executable,
            str(chat_server_path)
        ], cwd=str(project_root))
        
    except KeyboardInterrupt:
        print("\n👋 Chat server stopped by user")
    except Exception as e:
        print(f"❌ Error starting chat server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()