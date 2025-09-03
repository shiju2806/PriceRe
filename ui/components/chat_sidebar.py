"""
Chat Sidebar Component
Professional AI assistant integration for the PriceRe platform
"""

import streamlit as st
import requests
from typing import List, Dict


def render_professional_floating_chat():
    """Render enhanced chat widget with better visibility and functionality"""
    
    # Initialize chat state
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = True  # Keep chat always visible
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = True
        st.session_state.chat_history.append({
            'role': 'assistant', 
            'content': '👋 Welcome to PriceRe! I can help you with:\n• LDTI (Long-Duration Targeted Improvements)\n• Reinsurance concepts (quota share, surplus, catastrophe modeling)\n• Data cleaning assistance\n• Platform navigation and features'
        })
    
    # Enhanced sidebar chat - always visible
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 💬 PriceRe AI Assistant")
        
        # Always show chat interface (no toggle needed)
        with st.container():
            # Chat history with better formatting
            st.markdown("### 💭 Conversation")
            
            # Create scrollable chat area with more messages visible
            chat_container = st.container(height=400)  # Fixed height scrollable area
            
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        st.markdown(f"**🧑 You:**")
                        st.info(message['content'])
                    else:
                        st.markdown(f"**🤖 PriceRe:**")
                        st.success(message['content'])
                    
                    # Add spacing between messages
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("")  # Empty line for spacing
            
            st.markdown("---")
            
            # Chat input area
            st.markdown("### 💬 Ask a Question")
            
            # Use text area with form for Enter key support and clearing
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Type your message:",
                    placeholder="Ask about LDTI, reinsurance, data cleaning...",
                    height=80,
                    help="Press Ctrl+Enter to send or use the button below"
                )
                
                # Send button inside form
                send_pressed = st.form_submit_button("🚀 Send Message", type="primary", use_container_width=True)
            
            # Process message if sent
            if send_pressed and user_input.strip():
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Show processing message
                with st.spinner("🤔 PriceRe is thinking..."):
                    bot_response = _get_ai_response(user_input)
                
                # Add bot response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': bot_response
                })
                
                # Input is automatically cleared by form, just rerun
                st.rerun()
            
            # Clear chat button
            if len(st.session_state.chat_history) > 1:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.chat_history = [{
                        'role': 'assistant', 
                        'content': '👋 Chat cleared! How can I help you today?'
                    }]
                    st.rerun()


def render_professional_chat_interface():
    """Professional full-width chat interface with proper message display"""
    
    st.markdown("### 💬 PriceRe Chat Assistant")
    st.markdown("**Your AI-powered reinsurance pricing and LDTI assistant**")
    
    # Initialize chat history
    if 'professional_chat_history' not in st.session_state:
        st.session_state.professional_chat_history = []
    
    # Chat container with proper styling
    st.markdown("""
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        text-align: right;
    }
    .bot-message {
        background: #e9ecef;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.professional_chat_history:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.professional_chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="user-message"><strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message"><strong>PriceRe:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Ask about reinsurance, LDTI, data cleaning, or platform features...",
            height=100,
            key="professional_chat_input"
        )
    
    with col2:
        if st.button("🚀 Send", type="primary"):
            if user_input.strip():
                _handle_professional_chat_message(user_input)
        
        if st.button("🗑️ Clear"):
            st.session_state.professional_chat_history = []
            st.rerun()


def render_sidebar_chat():
    """Compact sidebar chat interface"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Pricing Assistant")
    
    user_input = st.sidebar.text_area(
        "Quick question:",
        height=80,
        placeholder="Ask about pricing, risk analysis...",
        key="sidebar_chat_input"
    )
    
    if st.sidebar.button("Send", key="sidebar_send"):
        if user_input.strip():
            if 'sidebar_chat_history' not in st.session_state:
                st.session_state.sidebar_chat_history = []
            
            # Add user message
            st.session_state.sidebar_chat_history.append({'role': 'user', 'content': user_input})
            
            # Process message
            try:
                bot_response = _get_ai_response(user_input)
                st.session_state.sidebar_chat_history.append({'role': 'assistant', 'content': bot_response})
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Chat error: {e}")
    
    # Show recent messages
    if 'sidebar_chat_history' in st.session_state and st.session_state.sidebar_chat_history:
        with st.sidebar.expander("💬 Recent", expanded=False):
            for msg in st.session_state.sidebar_chat_history[-2:]:  # Last 2 messages
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content'][:100]}...")
                else:
                    st.markdown(f"**PriceRe:** {msg['content'][:100]}...")
            
            if st.button("Clear History", key="clear_sidebar"):
                st.session_state.sidebar_chat_history = []
                st.rerun()


def _get_ai_response(user_input: str) -> str:
    """Get AI response from OpenAI chat engine or FastAPI server"""
    try:
        # Direct call to OpenAI via our chat engine
        from src.chat.openai_chat_engine import OpenAIChatEngine
        
        try:
            chat_engine = OpenAIChatEngine()
            return chat_engine.chat(user_input)  # Fixed method name
        except:
            # Fallback to FastAPI server
            response = requests.post(
                "http://localhost:8001/chat",
                json={"message": user_input, "user_id": "streamlit_user"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "I'm having trouble connecting to the chat server. Please check that the FastAPI server is running."
                
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please ensure the chat server is running."


def _handle_professional_chat_message(user_input: str):
    """Handle professional chat message processing"""
    # Add user message
    st.session_state.professional_chat_history.append({'role': 'user', 'content': user_input})
    
    # Process with AI
    try:
        response_text = _get_ai_response(user_input)
        st.session_state.professional_chat_history.append({'role': 'assistant', 'content': response_text})
        
        # Clear input and rerun
        st.session_state.professional_chat_input = ""
        st.rerun()
        
    except Exception as e:
        st.error(f"Chat processing error: {e}")


def get_chat_history() -> List[Dict[str, str]]:
    """Get current chat history"""
    return st.session_state.get('chat_history', [])


def clear_chat_history():
    """Clear all chat histories"""
    for key in ['chat_history', 'professional_chat_history', 'sidebar_chat_history']:
        if key in st.session_state:
            st.session_state[key] = []