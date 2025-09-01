"""
Streamlit Chat Interface for PriceRe Assistant
Integrates with existing comprehensive pricing platform
"""

import streamlit as st
from typing import Optional, Tuple, Any
import polars as pl
from datetime import datetime

try:
    from .chat_assistant import PriceReChatAssistant, ChatAction, ChatIntent
except ImportError:
    from chat_assistant import PriceReChatAssistant, ChatAction, ChatIntent

def initialize_chat_assistant(openai_api_key: Optional[str] = None) -> None:
    """Initialize chat assistant in session state with OpenAI from environment"""
    if "chat_assistant" not in st.session_state:
        # Use environment variable if no key provided
        st.session_state.chat_assistant = PriceReChatAssistant(openai_api_key=openai_api_key)
    
    if "chat_initialized" not in st.session_state:
        st.session_state.chat_initialized = False
    
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

def setup_chat_context(original_df: pl.DataFrame, cleaned_df: pl.DataFrame, 
                      cleaning_result: Any) -> None:
    """Setup chat context with cleaning results"""
    if not st.session_state.chat_initialized:
        st.session_state.chat_assistant.initialize_context(
            original_df, cleaned_df, cleaning_result
        )
        st.session_state.chat_initialized = True

def render_chat_interface() -> Optional[pl.DataFrame]:
    """Render the chat interface and return updated dataframe if changes made"""
    initialize_chat_assistant()
    
    if not st.session_state.chat_initialized:
        st.warning("💬 Initialize chat by uploading and cleaning data first")
        return None
    
    st.markdown("### 💬 PriceRe Cleaning Assistant")
    
    # Show conversation history
    chat_history = st.session_state.chat_assistant.get_conversation_history()
    
    # Create chat container
    chat_container = st.container()
    
    with chat_container:
        # Display conversation
        for message in chat_history:
            if message.role == "user":
                st.chat_message("user").write(message.content)
            else:
                st.chat_message("assistant").write(message.content)
    
    # Chat input
    user_input = st.chat_input("Ask me to refine the cleaning...")
    
    if user_input:
        return process_chat_message(user_input)
    
    # Handle pending actions
    if st.session_state.pending_action:
        return handle_pending_action()
    
    return None

def process_chat_message(user_message: str) -> Optional[pl.DataFrame]:
    """Process user chat message"""
    with st.chat_message("user"):
        st.write(user_message)
    
    # Process message through assistant
    response, action = st.session_state.chat_assistant.process_user_message(user_message)
    
    with st.chat_message("assistant"):
        st.write(response)
        
        # If there's a high-confidence action, show confirmation buttons
        if action and action.confidence > 0.7:
            st.session_state.pending_action = action
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, do it", key=f"confirm_{datetime.now().timestamp()}"):
                    return apply_chat_action(action)
            with col2:
                if st.button("❌ No, cancel", key=f"cancel_{datetime.now().timestamp()}"):
                    st.session_state.pending_action = None
                    st.rerun()
        
        # If medium confidence, show preview and ask for confirmation
        elif action and action.confidence > 0.5:
            if action.preview_data:
                with st.expander("🔍 Preview affected rows"):
                    for item in action.preview_data[:5]:
                        st.json(item['data'])
            
            st.session_state.pending_action = action
    
    return None

def handle_pending_action() -> Optional[pl.DataFrame]:
    """Handle pending action confirmations"""
    action = st.session_state.pending_action
    
    # Check if user confirmed via button clicks (handled in process_chat_message)
    # This function is for handling other confirmation methods
    return None

def apply_chat_action(action: ChatAction) -> Optional[pl.DataFrame]:
    """Apply the confirmed chat action"""
    try:
        updated_df, status_message = st.session_state.chat_assistant.apply_action(action)
        
        # Clear pending action
        st.session_state.pending_action = None
        
        # Show success message
        st.success(status_message)
        
        # Add confirmation to chat
        st.session_state.chat_assistant.add_message("assistant", status_message)
        
        # Rerun to update display
        st.rerun()
        
        return updated_df
        
    except Exception as e:
        st.error(f"❌ Error applying action: {str(e)}")
        st.session_state.pending_action = None
        return None

def render_chat_sidebar(original_df: pl.DataFrame, cleaned_df: pl.DataFrame, 
                       cleaning_result: Any) -> Optional[pl.DataFrame]:
    """Render chat interface in sidebar"""
    initialize_chat_assistant()
    
    # Setup context if not done
    setup_chat_context(original_df, cleaned_df, cleaning_result)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Cleaning Assistant")
    
    # Show basic stats
    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)
    removed_count = original_rows - cleaned_rows
    
    st.sidebar.metric("Rows Removed", removed_count, f"-{removed_count/original_rows*100:.1f}%")
    
    # Chat input in sidebar
    with st.sidebar:
        user_input = st.text_area(
            "Refine cleaning:",
            placeholder="e.g., 'Remove summary rows at bottom'",
            height=80,
            key="sidebar_chat_input"
        )
        
        if st.button("Send", key="sidebar_send"):
            if user_input.strip():
                return process_sidebar_message(user_input, original_df, cleaned_df)
    
    # Show recent messages
    chat_history = st.session_state.chat_assistant.get_conversation_history()
    if len(chat_history) > 1:  # More than just initial message
        st.sidebar.markdown("**Recent:**")
        for message in chat_history[-4:]:  # Show last 4 messages
            if len(message.content) > 100:
                content = message.content[:100] + "..."
            else:
                content = message.content
            
            if message.role == "user":
                st.sidebar.markdown(f"👤 {content}")
            else:
                st.sidebar.markdown(f"🤖 {content}")
    
    return None

def process_sidebar_message(user_message: str, original_df: pl.DataFrame, 
                           cleaned_df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Process message from sidebar chat"""
    # Process through assistant
    response, action = st.session_state.chat_assistant.process_user_message(user_message)
    
    # Show response in main area
    st.info(f"💬 **Assistant:** {response}")
    
    # If there's a confident action, show it in main area
    if action and action.confidence > 0.6:
        st.markdown("**Proposed Action:**")
        st.write(f"- {action.description}")
        
        if action.preview_data:
            with st.expander("Preview affected rows"):
                for item in action.preview_data[:3]:
                    st.json(item['data'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply Changes", key=f"apply_{datetime.now().timestamp()}"):
                return apply_chat_action(action)
        with col2:
            if st.button("❌ Cancel", key=f"cancel_sidebar_{datetime.now().timestamp()}"):
                st.info("Action cancelled")
    
    return None

def show_chat_summary() -> dict:
    """Show summary of chat session for logging/analytics"""
    if "chat_assistant" not in st.session_state:
        return {}
    
    history = st.session_state.chat_assistant.get_conversation_history()
    actions = st.session_state.chat_assistant.action_history
    
    return {
        "total_messages": len(history),
        "user_messages": len([m for m in history if m.role == "user"]),
        "actions_taken": len(actions),
        "action_types": [a.intent.value for a in actions]
    }

# Helper functions for integration with main PriceRe interface

def add_chat_refinement_button() -> bool:
    """Add chat refinement button to main interface"""
    return st.button("💬 PriceRe Chat", help="Use PriceRe Chat assistant to refine the cleaning results")

def show_chat_in_expander(original_df: pl.DataFrame, cleaned_df: pl.DataFrame, 
                         cleaning_result: Any) -> Optional[pl.DataFrame]:
    """Show chat interface in an expander"""
    with st.expander("💬 PriceRe Chat Assistant"):
        initialize_chat_assistant()
        setup_chat_context(original_df, cleaned_df, cleaning_result)
        
        return render_chat_interface()

def reset_chat_session():
    """Reset chat session (useful for new files)"""
    if "chat_assistant" in st.session_state:
        del st.session_state.chat_assistant
    if "chat_initialized" in st.session_state:
        del st.session_state.chat_initialized
    if "pending_action" in st.session_state:
        del st.session_state.pending_action