#!/usr/bin/env python3
"""
FastAPI Chat Server for PriceRe
Professional chat interface with WebSocket support and OpenAI integration
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our OpenAI chat engine
from src.chat.openai_chat_engine import OpenAIChatEngine

app = FastAPI(title="PriceRe Chat Server", version="1.0.0")

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=project_root / "ui" / "assets"), name="static")

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    user_id: str

# Initialize OpenAI chat engine
try:
    chat_engine = OpenAIChatEngine()
    print("✅ OpenAI Chat Engine initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI Chat Engine: {e}")
    chat_engine = None

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.chat_history: Dict[str, List[dict]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send chat history to new connection
        if user_id in self.chat_history:
            for message in self.chat_history[user_id][-10:]:  # Last 10 messages
                await websocket.send_json(message)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket, user_id: str):
        await websocket.send_json({
            "type": "response",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
    
    def add_to_history(self, user_id: str, message: dict):
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        self.chat_history[user_id].append(message)
        
        # Keep only last 50 messages per user
        if len(self.chat_history[user_id]) > 50:
            self.chat_history[user_id] = self.chat_history[user_id][-50:]

manager = ConnectionManager()

@app.get("/")
async def get_chat_interface():
    """Serve the main chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PriceRe Chat Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .chat-container {
                width: 400px;
                height: 600px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
                font-weight: 600;
                font-size: 18px;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
                line-height: 1.4;
            }
            
            .user-message {
                background: #667eea;
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 4px;
            }
            
            .bot-message {
                background: #f1f3f5;
                color: #333;
                align-self: flex-start;
                border-bottom-left-radius: 4px;
            }
            
            .chat-input-container {
                padding: 20px;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
            }
            
            .chat-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
            }
            
            .send-button {
                padding: 12px 20px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s;
            }
            
            .send-button:hover {
                background: #5a6fd8;
            }
            
            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .typing-indicator {
                display: none;
                padding: 12px 16px;
                background: #f1f3f5;
                border-radius: 18px;
                align-self: flex-start;
                color: #666;
                font-style: italic;
            }
            
            .connection-status {
                position: absolute;
                top: 10px;
                right: 10px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #28a745;
            }
            
            .connection-status.disconnected {
                background: #dc3545;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                💰 PriceRe Chat Assistant
                <div class="connection-status" id="connectionStatus"></div>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    👋 Welcome to PriceRe! I'm your AI assistant for reinsurance questions, LDTI explanations, data cleaning help, and platform guidance.
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                PriceRe is typing...
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="messageInput" placeholder="Ask about LDTI, reinsurance, data cleaning..." autofocus>
                <button class="send-button" id="sendButton">Send</button>
            </div>
        </div>
        
        <script>
            class ChatInterface {
                constructor() {
                    this.websocket = null;
                    this.userId = 'user_' + Date.now();
                    this.initializeElements();
                    this.connect();
                }
                
                initializeElements() {
                    this.messagesContainer = document.getElementById('chatMessages');
                    this.messageInput = document.getElementById('messageInput');
                    this.sendButton = document.getElementById('sendButton');
                    this.typingIndicator = document.getElementById('typingIndicator');
                    this.connectionStatus = document.getElementById('connectionStatus');
                    
                    this.sendButton.addEventListener('click', () => this.sendMessage());
                    this.messageInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') this.sendMessage();
                    });
                }
                
                connect() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/${this.userId}`;
                    
                    this.websocket = new WebSocket(wsUrl);
                    
                    this.websocket.onopen = () => {
                        console.log('Connected to PriceRe Chat');
                        this.connectionStatus.classList.remove('disconnected');
                    };
                    
                    this.websocket.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'response') {
                            this.hideTyping();
                            this.addMessage(data.content, 'bot');
                        }
                    };
                    
                    this.websocket.onclose = () => {
                        console.log('Disconnected from PriceRe Chat');
                        this.connectionStatus.classList.add('disconnected');
                        // Attempt to reconnect after 3 seconds
                        setTimeout(() => this.connect(), 3000);
                    };
                    
                    this.websocket.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.connectionStatus.classList.add('disconnected');
                    };
                }
                
                sendMessage() {
                    const message = this.messageInput.value.trim();
                    if (!message || !this.websocket) return;
                    
                    this.addMessage(message, 'user');
                    this.showTyping();
                    
                    this.websocket.send(JSON.stringify({
                        message: message,
                        user_id: this.userId
                    }));
                    
                    this.messageInput.value = '';
                }
                
                addMessage(content, type) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${type}-message`;
                    messageDiv.textContent = content;
                    
                    this.messagesContainer.appendChild(messageDiv);
                    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
                }
                
                showTyping() {
                    this.typingIndicator.style.display = 'block';
                    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
                }
                
                hideTyping() {
                    this.typingIndicator.style.display = 'none';
                }
            }
            
            // Initialize chat when page loads
            document.addEventListener('DOMContentLoaded', () => {
                new ChatInterface();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Add user message to history
            user_msg = {
                "type": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            manager.add_to_history(user_id, user_msg)
            
            # Get AI response
            if chat_engine:
                try:
                    response = await asyncio.to_thread(
                        chat_engine.chat,  # Changed from process_message to chat
                        message
                    )
                except Exception as e:
                    response = f"I apologize, but I encountered an error processing your request: {str(e)}"
            else:
                response = "I'm currently unable to process your request. Please ensure the OpenAI API key is configured correctly."
            
            # Add bot response to history
            bot_msg = {
                "type": "bot",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
            manager.add_to_history(user_id, bot_msg)
            
            # Send response to client
            await manager.send_personal_message(response, websocket, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"User {user_id} disconnected")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """REST endpoint for chat (alternative to WebSocket)"""
    if not chat_engine:
        raise HTTPException(status_code=503, detail="Chat engine not available")
    
    try:
        response = await asyncio.to_thread(
            chat_engine.chat,  # Changed from process_message to chat
            chat_message.message
        )
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now(),
            user_id=chat_message.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chat_engine": chat_engine is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting PriceRe Chat Server...")
    print("📍 Chat interface: http://localhost:8001")
    print("🔌 WebSocket endpoint: ws://localhost:8001/ws/{user_id}")
    print("🌐 REST API: http://localhost:8001/chat")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )