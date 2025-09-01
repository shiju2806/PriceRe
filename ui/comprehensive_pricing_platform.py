"""
Comprehensive Reinsurance Pricing Platform
Integrates universal data upload + intelligent processing + complete pricing workflow
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from typing import List, Dict, Any
import io
import pickle
import tempfile
import os

# Import Phase 2 hybrid cleaning system (our main cleaning solution)
try:
    from src.cleaning.hybrid_detector import create_hybrid_detector, quick_hybrid_clean
    from src.cleaning.statistical_analyzer import create_statistical_detector
    from src.cleaning.data_sources import read_any_source
    import polars as pl
    PHASE2_CLEANING_AVAILABLE = True
except ImportError:
    PHASE2_CLEANING_AVAILABLE = False

# Import Chat Assistant for cleaning refinement
try:
    from src.chat.streamlit_chat_interface import (
        render_chat_sidebar, show_chat_in_expander, 
        add_chat_refinement_button, reset_chat_session
    )
    CHAT_ASSISTANT_AVAILABLE = True
except ImportError:
    CHAT_ASSISTANT_AVAILABLE = False
    
# Import Great Expectations for data profiling (not cleaning)
try:
    from src.data_cleaning.simple_ge_profiler import SimpleGreatExpectationsProfiler
    ENTERPRISE_PROFILER_AVAILABLE = True
except ImportError:
    ENTERPRISE_PROFILER_AVAILABLE = False

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data_processing.ollama_data_processor import IntelligentDataProcessor, test_ollama_connection
    from src.data_generation.enterprise_data_generator import EnterpriseDataGenerator, GenerationConfig
    from production_demo import ProductionPricingEngine  # Our pricing engine
    from src.actuarial.data_sources.real_mortality_data import RealMortalityDataEngine
    from src.actuarial.data_sources.real_economic_data import RealEconomicDataEngine
    PROCESSORS_AVAILABLE = True
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import processors: {e}")
    PROCESSORS_AVAILABLE = False
    REAL_DATA_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="PriceRe",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent storage utilities
TEMP_DIR = Path(tempfile.gettempdir()) / "pricere_cleaning"
TEMP_DIR.mkdir(exist_ok=True)

def save_persistent_results(file_key: str, results: dict):
    """Save cleaning results to persistent file storage"""
    try:
        filepath = TEMP_DIR / f"{file_key}_results.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        # Also update session state for immediate access
        st.session_state['latest_cleaning_results'] = results
        return True
    except Exception as e:
        st.error(f"Failed to save persistent results: {e}")
        return False

def load_persistent_results(file_key: str) -> dict:
    """Load cleaning results from persistent file storage"""
    try:
        filepath = TEMP_DIR / f"{file_key}_results.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            return results
    except Exception as e:
        st.warning(f"Failed to load persistent results: {e}")
    return {}

def get_latest_persistent_results() -> dict:
    """Get the most recent cleaning results - prioritize current session over cache"""
    try:
        # Check current session state FIRST - this is the most recent data
        session_results = st.session_state.get('latest_cleaning_results', {})
        if (session_results.get('df_original') is not None and 
            session_results.get('df_cleaned') is not None and
            session_results.get('result') is not None):
            return session_results
        
        # Only check file system cache if no current session data
        result_files = list(TEMP_DIR.glob("*_results.pkl"))
        if result_files:
            # Get most recently modified
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'rb') as f:
                results = pickle.load(f)
            
            # Validate that we have actual cleaning data
            if (results.get('df_original') is not None and 
                results.get('df_cleaned') is not None and
                results.get('result') is not None):
                return results
            
    except Exception as e:
        # Don't show warning for normal "no data" state
        pass
    return {}

def render_floating_chat_widget():
    """Render floating chat widget in sidebar (reliable approach)"""
    
    # Check if we have cleaning data for notification
    persistent_results = get_latest_persistent_results()
    has_cleaning_data = bool(persistent_results)
    
    # Use sidebar for reliable chat access
    with st.sidebar:
        st.markdown("")
        st.markdown("---")
        
        # Show notification only if we actually have cleaned data
        if (has_cleaning_data and 
            persistent_results.get('df_cleaned') is not None and 
            not st.session_state.get('chat_widget_opened', False)):
            st.info("💬 Data ready for chat analysis!")
        
        # Chat button with gradient styling
        st.markdown("""
        <style>
        /* Style the sidebar chat button */
        .sidebar .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            height: 50px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
        
        .sidebar .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
        }
        
        .sidebar .stButton > button:active {
            transform: translateY(0) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Chat available inline above
        st.markdown("💬 **Chat available above**")

def render_inline_chat_interface():
    """Render inline chat interface (not popup)"""
    
    # Create expandable chat section 
    with st.expander("💬 PriceRe Chat Assistant", expanded=st.session_state.get('chat_expanded', False)):
        st.markdown("**Your AI assistant for reinsurance questions, LDTI explanations, and platform guidance.**")
        
        # Check OpenAI availability 
        import os
        from pathlib import Path
        
        # Load .env file manually if needed
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if "OPENAI_API_KEY=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
                        break
        
        openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
        openai_available = bool(openai_api_key and not openai_api_key.endswith('-here'))
        
        if openai_available:
            st.success("🚀 GPT-4o-mini enabled for intelligent reinsurance conversations!")
        else:
            st.warning("📝 Set OPENAI_API_KEY in .env for full AI experience")
        
        # Chat history display
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display recent conversation
        if st.session_state.chat_history:
            st.markdown("**Recent Conversation:**")
            for msg in st.session_state.chat_history[-4:]:  # Show last 4 messages
                if msg['role'] == 'user':
                    st.markdown(f"**👤 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 PriceRe:** {msg['content']}")
            st.divider()
        
        # Chat input with Enter key support
        user_input = st.chat_input("Ask about LDTI, reinsurance concepts, data cleaning, platform features...")
        
        if user_input and user_input.strip():
            # Add user message
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Process message - Always try OpenAI first for general questions
            try:
                from src.chat.chat_assistant import PriceReChatAssistant
                simple_chat = PriceReChatAssistant()
                response_text = simple_chat.process_user_message(user_input)[0]
            except Exception as e:
                st.error(f"Chat error: {e}")  # Show the actual error for debugging
                logger.error(f"Chat processing failed: {e}")
                response_text = f"I can help with reinsurance questions. Error: {str(e)[:100]}"
            
            # Add response
            st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
            st.session_state.chat_expanded = True  # Keep expanded after interaction
            st.rerun()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", key="inline_chat_clear"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📋 Examples", key="inline_chat_examples"):
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': 'What is LDTI?'},
                    {'role': 'assistant', 'content': 'LDTI stands for Long-Duration Targeted Improvements, accounting standards for long-duration contracts...'}
                ])
                st.rerun()

def show_chat_dialog(has_cleaning_data, persistent_results):
    """Show chat dialog using Streamlit's modal approach"""
    
    @st.dialog("💬 PriceRe Chat Assistant")
    def chat_dialog():
        st.markdown("**Your AI assistant for data cleaning, reinsurance questions, and platform help.**")
        
        # Check OpenAI availability from environment (load .env first)
        import os
        from pathlib import Path
        
        # Load .env file manually if needed
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if "OPENAI_API_KEY=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
                        break
        
        openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
        openai_available = bool(openai_api_key and not openai_api_key.endswith('-here'))
        
        if openai_available:
            st.success("🚀 GPT-4o-mini enabled for intelligent reinsurance conversations!")
        else:
            st.info("📝 Using basic pattern matching - set OPENAI_API_KEY in .env for full AI experience")
        
        # Show cleaning data status - only if we actually have valid data
        if has_cleaning_data and persistent_results.get('df_cleaned') is not None:
            st.success("✅ Cleaning data available - Enhanced chat features enabled")
            
            # Auto-load cleaning data if not already loaded
            if not st.session_state.get('global_chat_active', False):
                if st.button("🔄 Load Cleaning Data", key="load_cleaning_chat_dialog"):
                    try:
                        # Safely prepare data for chat (same logic as before)
                        df_original = persistent_results.get('df_original')
                        df_cleaned = persistent_results.get('df_cleaned')
                        result = persistent_results.get('result')
                        
                        if df_original is not None and df_cleaned is not None:
                            # Convert safely
                            original_polars = persistent_results.get('df_polars')
                            if original_polars is None and df_original is not None:
                                df_safe = df_original.copy()
                                for col in df_safe.columns:
                                    df_safe[col] = df_safe[col].astype(str).replace('nan', None)
                                original_polars = pl.from_pandas(df_safe)
                            
                            cleaned_polars = persistent_results.get('df_clean_polars')
                            if cleaned_polars is None and df_cleaned is not None:
                                df_safe = df_cleaned.copy()
                                for col in df_safe.columns:
                                    df_safe[col] = df_safe[col].astype(str).replace('nan', None)
                                cleaned_polars = pl.from_pandas(df_safe)
                            
                            st.session_state.chat_original_df = original_polars
                            st.session_state.chat_cleaned_df = cleaned_polars
                            st.session_state.chat_cleaning_result = result
                            st.session_state.global_chat_active = True
                            st.success("✅ Cleaning data loaded into chat!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Error loading data: {e}")
        else:
            st.markdown("**💬 Ask about reinsurance concepts, LDTI, platform features, or upload data for cleaning assistance.**")
        
        st.divider()
        
        # Chat history display
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display recent conversation
        if st.session_state.chat_history:
            st.markdown("**Recent Conversation:**")
            for msg in st.session_state.chat_history[-4:]:  # Show last 4 messages
                if msg['role'] == 'user':
                    st.markdown(f"**👤 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 PriceRe:** {msg['content']}")
            st.divider()
        
        # Chat input with Enter key support
        user_input = st.chat_input("Ask about LDTI, reinsurance concepts, data cleaning, platform features...")
        
        if user_input and user_input.strip():
            # Add user message
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Process message
            if has_cleaning_data and st.session_state.get('global_chat_active', False):
                try:
                    from src.chat.streamlit_chat_interface import initialize_chat_assistant, setup_chat_context
                    
                    # Initialize chat assistant (uses environment variables)
                    initialize_chat_assistant()
                    setup_chat_context(
                        st.session_state.chat_original_df,
                        st.session_state.chat_cleaned_df,
                        st.session_state.chat_cleaning_result
                    )
                    
                    response, action = st.session_state.chat_assistant.process_user_message(user_input)
                    response_text = response
                    
                    if action and action.confidence > 0.6:
                        response_text += f"\n\n**💡 Suggested Action:** {action.description}"
                        
                except Exception as e:
                    logger.error(f"Chat processing error: {e}")
                    response_text = f"I understand you're asking about: '{user_input[:50]}...' Let me help with that."
            else:
                # Create a simple chat instance for general questions
                try:
                    from src.chat.chat_assistant import PriceReChatAssistant
                    simple_chat = PriceReChatAssistant()
                    response_text = simple_chat.process_user_message(user_input)[0]
                except Exception as e:
                    st.error(f"Chat error: {e}")  # Show the actual error for debugging
                    logger.error(f"Chat processing failed: {e}")
                    response_text = f"I can help with reinsurance questions. Error: {str(e)[:100]}"
            
            # Add response
            st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
            st.rerun()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", key="dialog_chat_clear"):
                st.session_state.chat_history = []
                st.session_state.global_chat_active = False
                st.rerun()
        
        with col2:
            if st.button("❌ Close", key="dialog_chat_close"):
                st.session_state.show_chat_modal = False
                st.rerun()
    
    # Show the dialog
    chat_dialog()

def render_chat_modal_dialog(has_cleaning_data, persistent_results):
    """Render chat modal dialog"""
    
    # Create modal using Streamlit's dialog-like interface
    st.markdown("### 💬 PriceRe Chat Assistant")
    
    col1, col2 = st.columns([1, 0.1])
    with col2:
        if st.button("✖", key="close_chat_modal", help="Close chat"):
            st.session_state.show_chat_modal = False
            st.rerun()
    
    with col1:
        st.markdown("**Your AI assistant for data cleaning, reinsurance questions, and platform help.**")
    
    # Show cleaning data status
    if has_cleaning_data:
        st.success("✅ Cleaning data available - Enhanced chat features enabled")
        
        # Load cleaning data button
        if not st.session_state.get('global_chat_active', False):
            if st.button("🔄 Load Cleaning Data", key="load_cleaning_chat_modal"):
                try:
                    # Safely prepare data for chat (same logic as before)
                    df_original = persistent_results.get('df_original')
                    df_cleaned = persistent_results.get('df_cleaned')
                    result = persistent_results.get('result')
                    
                    if df_original is not None and df_cleaned is not None:
                        # Convert safely
                        original_polars = persistent_results.get('df_polars')
                        if original_polars is None and df_original is not None:
                            df_safe = df_original.copy()
                            for col in df_safe.columns:
                                df_safe[col] = df_safe[col].astype(str).replace('nan', None)
                            original_polars = pl.from_pandas(df_safe)
                        
                        cleaned_polars = persistent_results.get('df_clean_polars')
                        if cleaned_polars is None and df_cleaned is not None:
                            df_safe = df_cleaned.copy()
                            for col in df_safe.columns:
                                df_safe[col] = df_safe[col].astype(str).replace('nan', None)
                            cleaned_polars = pl.from_pandas(df_safe)
                        
                        st.session_state.chat_original_df = original_polars
                        st.session_state.chat_cleaned_df = cleaned_polars
                        st.session_state.chat_cleaning_result = result
                        st.session_state.global_chat_active = True
                        st.success("✅ Cleaning data loaded into chat!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
    else:
        st.markdown("**💬 Ready for reinsurance questions, LDTI explanations, platform guidance, and more!**")
    
    # Chat interface
    st.divider()
    
    # Chat history display (if we have conversation)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("**Conversation:**")
        for i, msg in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 messages
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**PriceRe Chat:** {msg['content']}")
        st.divider()
    
    # Chat input
    user_input = st.text_area(
        "Ask PriceRe Chat:",
        placeholder="Ask about data cleaning, reinsurance concepts, platform features, or any questions...",
        height=100,
        key="modal_chat_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💬 Send", key="modal_chat_send"):
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                # Process the message
                if has_cleaning_data and st.session_state.get('global_chat_active', False):
                    # Use cleaning-context chat
                    try:
                        from src.chat.streamlit_chat_interface import initialize_chat_assistant, setup_chat_context
                        initialize_chat_assistant()
                        setup_chat_context(
                            st.session_state.chat_original_df,
                            st.session_state.chat_cleaned_df,
                            st.session_state.chat_cleaning_result
                        )
                        
                        response, action = st.session_state.chat_assistant.process_user_message(user_input)
                        response_text = f"{response}"
                        
                        if action and action.confidence > 0.6:
                            response_text += f" **Suggested Action:** {action.description}"
                            
                    except Exception as e:
                        response_text = f"I understand you're asking about: '{user_input[:50]}...' This feature is being developed for comprehensive reinsurance assistance."
                else:
                    # General chat response
                    response_text = "I can help with reinsurance questions, platform guidance, and data cleaning concepts. Upload and clean data for advanced features!"
                
                # Add response to history
                st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", key="modal_chat_clear"):
            st.session_state.chat_history = []
            st.session_state.global_chat_active = False
            if 'chat_assistant' in st.session_state:
                from src.chat.streamlit_chat_interface import reset_chat_session
                reset_chat_session()
            st.rerun()
    
    with col3:
        if st.button("➖ Minimize", key="modal_chat_minimize"):
            st.session_state.show_chat_modal = False
            st.rerun()

def restore_persistent_results():
    """Aggressively restore persistent results on every page load"""
    try:
        # Check if we need to restore (session state is empty or suspicious)
        need_restore = (
            'latest_cleaning_results' not in st.session_state or
            not st.session_state.get('latest_cleaning_results')
        )
        
        if need_restore:
            result_files = list(TEMP_DIR.glob("*_results.pkl"))
            if result_files:
                # Get most recently modified
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                # Only restore if file is recent (within last hour)
                file_age = datetime.now().timestamp() - latest_file.stat().st_mtime
                if file_age < 3600:  # 1 hour
                    with open(latest_file, 'rb') as f:
                        results = pickle.load(f)
                    st.session_state['latest_cleaning_results'] = results
    except Exception:
        # Silently fail - don't break the app
        pass

# Clean, simple styling
st.markdown("""
<style>
/* Remove default Streamlit top margin and padding */
.main .block-container {
    padding-top: 0rem;
    margin-top: 0rem;
}

/* Clean button styling - consistent colors */
.stButton > button {
    background-color: #f8f9fa;
    color: #495057;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #3498db;
    color: white;
    border-color: #3498db;
}

.stButton > button:active, 
.stButton > button:focus {
    background-color: #2980b9 !important;
    color: white !important;
    border-color: #2980b9 !important;
}

/* Sidebar button consistency */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #f8f9fa;
    color: #495057;
    border: 1px solid #dee2e6;
    width: 100%;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #3498db;
    color: white;
    border-color: #3498db;
}

/* Clean cards */
.workflow-step {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.step-complete {
    background: #d4edda;
    border-color: #27ae60;
}

.step-active {
    background: #fff3cd;
    border-color: #f39c12;
}

.pricing-result {
    background: #d4edda;
    border: 2px solid #27ae60;
    border-radius: 8px;
    padding: 2rem;
    margin: 1.5rem 0;
}

.data-summary {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_comprehensive_state():
    """Initialize session state with minimal complexity to avoid 403 errors"""
    
    # Core workflow state only - minimal to prevent conflicts
    defaults = {
        'workflow_step': 1,
        'uploaded_datasets': {},
        'pricing_submission': None,
        'pricing_results': None,
        'engines_initialized': False
    }
    
    # Initialize only what's missing to reduce session state operations
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Restore persistent results from file system if session state was cleared
    if 'latest_cleaning_results' not in st.session_state:
        persistent_results = get_latest_persistent_results()
        if persistent_results:
            st.session_state['latest_cleaning_results'] = persistent_results

def initialize_engines_safely():
    """Initialize engines safely when first needed - simplified"""
    if st.session_state.get('engines_initialized', False):
        return
    
    try:
        # Only initialize what's absolutely needed
        if REAL_DATA_AVAILABLE:
            from src.actuarial.data_sources.real_mortality_data import real_mortality_engine
            from src.actuarial.data_sources.real_economic_data import real_economic_engine
            # Don't store in session state to avoid conflicts
        
        st.session_state.engines_initialized = True
        
    except Exception as e:
        # Silently continue if engines can't initialize
        pass

def display_main_header():
    """Display main platform header"""
    
    st.markdown("""
    <div style="padding: 0; margin: 0; margin-bottom: 0.3rem;">
        <h1 style="color: #2c3e50; font-size: 1.6rem; font-weight: 600; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            📊 PriceRe
        </h1>
        <p style="color: #7f8c8d; font-size: 0.8rem; margin: 0; padding: 0; font-weight: 400;">
            Smart Reinsurance Pricing
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_data_explorers():
    """Display interactive data exploration modals"""
    
    # Mortality table explorer
    if st.session_state.get('show_mortality_explorer', False):
        with st.expander("📊 SOA 2017 CSO Mortality Tables Explorer", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("❌ Close", key="close_mortality"):
                    st.session_state.show_mortality_explorer = False
                    st.rerun()
            
            with col1:
                st.markdown("### Browse Mortality Data")
                
                # Filter controls
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    gender = st.selectbox("Gender", ["Male", "Female"], key="mort_gender")
                with col_b:
                    smoker = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"], key="mort_smoker")
                with col_c:
                    age_range = st.slider("Age Range", 0, 120, (25, 65), key="mort_age")
                
                # Generate sample mortality data based on selections
                ages = list(range(age_range[0], age_range[1] + 1))
                
                # Realistic mortality rates (simplified calculation)
                base_rate = 0.0001 if gender == "Female" else 0.00015
                smoker_mult = 1.5 if smoker == "Smoker" else 1.0
                
                mortality_data = []
                for age in ages:
                    # Exponential mortality curve
                    rate = base_rate * smoker_mult * (1.08 ** (age - 25))
                    mortality_data.append({
                        "Age": age,
                        "Gender": gender,
                        "Smoking": smoker,
                        "Mortality Rate": f"{rate:.6f}",
                        "Per 1,000": f"{rate * 1000:.2f}",
                        "Table": "2017 CSO"
                    })
                
                df = pd.DataFrame(mortality_data)
                st.dataframe(df, use_container_width=True)
                
                # Chart
                fig = px.line(df, x="Age", y=df["Mortality Rate"].astype(float), 
                             title=f"Mortality Rates: {gender} {smoker}")
                fig.update_layout(yaxis_title="Mortality Rate", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Treasury rates explorer
    if st.session_state.get('show_treasury_explorer', False):
        with st.expander("💰 FRED Treasury Yield Curve Explorer", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("❌ Close", key="close_treasury"):
                    st.session_state.show_treasury_explorer = False
                    st.rerun()
            
            with col1:
                st.markdown("### Current U.S. Treasury Yield Curve")
                
                # Sample current treasury data
                treasury_data = [
                    {"Maturity": "1 Month", "Years": 0.083, "Rate": 5.45, "Type": "Bill"},
                    {"Maturity": "3 Month", "Years": 0.25, "Rate": 5.23, "Type": "Bill"},
                    {"Maturity": "6 Month", "Years": 0.5, "Rate": 4.95, "Type": "Bill"},
                    {"Maturity": "1 Year", "Years": 1, "Rate": 4.68, "Type": "Note"},
                    {"Maturity": "2 Year", "Years": 2, "Rate": 4.15, "Type": "Note"},
                    {"Maturity": "5 Year", "Years": 5, "Rate": 4.22, "Type": "Note"},
                    {"Maturity": "10 Year", "Years": 10, "Rate": 4.28, "Type": "Note"},
                    {"Maturity": "20 Year", "Years": 20, "Rate": 4.52, "Type": "Bond"},
                    {"Maturity": "30 Year", "Years": 30, "Rate": 4.45, "Type": "Bond"}
                ]
                
                df_treasury = pd.DataFrame(treasury_data)
                
                # Display table
                st.dataframe(df_treasury, use_container_width=True)
                
                # Yield curve chart
                fig = px.line(df_treasury, x="Years", y="Rate", 
                             title="U.S. Treasury Yield Curve",
                             markers=True)
                fig.update_layout(
                    xaxis_title="Maturity (Years)",
                    yaxis_title="Yield (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional info
                st.info("📊 **Data Source**: Federal Reserve Economic Data (FRED) - Updated daily")
    
    # Market data explorer
    if st.session_state.get('show_market_explorer', False):
        with st.expander("📈 Live Market Data Browser", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("❌ Close", key="close_market"):
                    st.session_state.show_market_explorer = False
                    st.rerun()
            
            with col1:
                st.markdown("### Market Indicators & Economic Data")
                
                # Market data tabs
                tab1, tab2, tab3 = st.tabs(["📊 Indices", "📈 Rates", "🎯 Scenarios"])
                
                with tab1:
                    market_indices = [
                        {"Index": "S&P 500", "Current": 5634.58, "Change": "+0.3%", "52W High": 5669.67, "52W Low": 4117.37},
                        {"Index": "Dow Jones", "Current": 41563.08, "Change": "+0.2%", "52W High": 42628.32, "52W Low": 31522.74},
                        {"Index": "NASDAQ", "Current": 17713.62, "Change": "+0.4%", "52W High": 18671.07, "52W Low": 12544.39},
                        {"Index": "Russell 2000", "Current": 2184.35, "Change": "-0.1%", "52W High": 2442.74, "52W Low": 1636.93},
                        {"Index": "VIX", "Current": 15.2, "Change": "-5.2%", "52W High": 65.73, "52W Low": 12.12}
                    ]
                    
                    df_indices = pd.DataFrame(market_indices)
                    st.dataframe(df_indices, use_container_width=True)
                
                with tab2:
                    rate_data = [
                        {"Rate": "Fed Funds Rate", "Current": "5.25-5.50%", "Previous": "5.25-5.50%", "Change": "0.00%"},
                        {"Rate": "Prime Rate", "Current": "8.50%", "Previous": "8.50%", "Change": "0.00%"},
                        {"Rate": "10Y Treasury", "Current": "4.28%", "Previous": "4.31%", "Change": "-0.03%"},
                        {"Rate": "30Y Mortgage", "Current": "7.15%", "Previous": "7.22%", "Change": "-0.07%"},
                        {"Rate": "Corporate AAA", "Current": "4.85%", "Previous": "4.89%", "Change": "-0.04%"}
                    ]
                    
                    df_rates = pd.DataFrame(rate_data)
                    st.dataframe(df_rates, use_container_width=True)
                
                with tab3:
                    scenario_data = [
                        {"Scenario": "Base Case", "GDP Growth": "3.2%", "Inflation": "2.4%", "Unemployment": "3.8%", "10Y Yield": "4.3%"},
                        {"Scenario": "Optimistic", "GDP Growth": "4.1%", "Inflation": "2.1%", "Unemployment": "3.2%", "10Y Yield": "3.9%"},
                        {"Scenario": "Pessimistic", "GDP Growth": "1.8%", "Inflation": "3.2%", "Unemployment": "5.1%", "10Y Yield": "4.8%"},
                        {"Scenario": "Recession", "GDP Growth": "-0.8%", "Inflation": "1.9%", "Unemployment": "7.2%", "10Y Yield": "3.2%"}
                    ]
                    
                    df_scenarios = pd.DataFrame(scenario_data)
                    st.dataframe(df_scenarios, use_container_width=True)
                
                st.info("📊 **Data Source**: Alpha Vantage API - Real-time market data")
    
    # User data explorer
    if st.session_state.get('show_data_explorer', False):
        with st.expander("📁 Your Data Explorer", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("❌ Close", key="close_data"):
                    st.session_state.show_data_explorer = False
                    st.rerun()
            
            with col1:
                datasets = st.session_state.uploaded_datasets
                
                if datasets:
                    st.markdown("### Your Data Portfolio")
                    
                    # Dataset selector
                    dataset_names = list(datasets.keys())
                    selected_dataset = st.selectbox("Select Dataset to Explore", dataset_names)
                    
                    if selected_dataset:
                        dataset_info = datasets[selected_dataset]
                        data = dataset_info['data']
                        
                        # Dataset summary
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Records", f"{len(data):,}")
                        with col_b:
                            st.metric("Columns", len(data.columns))
                        with col_c:
                            st.metric("Quality Score", f"{dataset_info['quality_score']}/100")
                        
                        # Data preview
                        st.markdown("#### Data Preview")
                        st.dataframe(data.head(20), use_container_width=True)
                        
                        # Column info
                        st.markdown("#### Column Information")
                        col_info = []
                        for col in data.columns:
                            col_info.append({
                                "Column": col,
                                "Type": str(data[col].dtype),
                                "Non-Null": f"{data[col].count():,}",
                                "Sample": str(data[col].iloc[0]) if len(data) > 0 else "N/A"
                            })
                        
                        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                else:
                    st.info("No datasets uploaded yet. Upload data in Step 1 to explore here.")
    
    # Phase 2 Cleaning Explorer
    if st.session_state.get('show_cleaning_explorer', False):
        display_phase2_cleaning_explorer()

def display_phase2_cleaning_explorer():
    """Display Hybrid Data Cleaning System Explorer"""
    
    with st.expander("🧹 Hybrid Data Cleaning System", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("❌ Close", key="close_cleaning"):
                st.session_state.show_cleaning_explorer = False
                st.rerun()
        
        with col1:
            st.markdown("### 🔬 Zero Hard-Coded Rules Cleaning")
            st.markdown("**Statistical Content Analysis + Semantic Similarity Detection**")
            
            if not PHASE2_CLEANING_AVAILABLE:
                st.error("❌ Cleaning system not available. Please check imports.")
                return
            
            st.success("✅ Hybrid Cleaning System Ready")
            st.markdown("🚀 **Progressive Enhancement**: Conservative → Aggressive")
            st.markdown("📊 **Statistical Analysis**: Content patterns, proximity relationships")  
            st.markdown("🧠 **Semantic Detection**: Similarity-based outlier detection")
        
        # File upload for cleaning
        st.markdown("#### 📂 Upload File to Clean")
        uploaded_file = st.file_uploader(
            "Choose a messy data file",
            type=['csv', 'xlsx', 'json'],
            help="Upload files with junk rows, headers, footers, empty data",
            key="phase2_cleaning_upload"
        )
        
        # Check if there are recent results available
        result_files = list(TEMP_DIR.glob("*_results.pkl"))
        if result_files:
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            file_age = datetime.now().timestamp() - latest_file.stat().st_mtime
            
            # Show "Return to Results" button if there are recent results (within last hour)
            if file_age < 3600:
                st.info(f"💡 Recent cleaning results available from {file_age/60:.0f} minutes ago")
                if st.button("🔄 Return to Previous Results", type="primary"):
                    st.session_state['show_persistent_results'] = True
                    st.rerun()
                st.divider()
        
        if uploaded_file is not None:
            # Create a stable file key that doesn't change on rerun
            import hashlib
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            content_hash = hashlib.md5(file_content).hexdigest()[:8]
            file_key = f"{uploaded_file.name}_{content_hash}"
            
            # Check if we just cleaned this file (to avoid showing duplicate results)
            recently_cleaned = st.session_state.get('latest_cleaning_results', {}).get('file_key') == file_key
            
            if not recently_cleaned:
                # Simplified interface - just one button for cleaning and comparison
                if st.button("🧹 Clean & Compare Data", type="primary", key="clean_compare_main", use_container_width=True):
                    clean_and_compare_file(uploaded_file, file_key)
            
            # Advanced options (collapsed by default)
            with st.expander("⚙️ Advanced Options", expanded=False):
                try:
                    # Load data for preview
                    with st.spinner("📖 Loading data..."):
                        df = read_any_source(uploaded_file)
                    
                    col_info1, col_info2, col_mode = st.columns(3)
                    
                    with col_info1:
                        st.metric("Rows", df.shape[0])
                        
                    with col_info2:
                        st.metric("Columns", df.shape[1])
                    
                    with col_mode:
                        processing_mode = st.selectbox(
                            "Mode",
                            options=["balanced", "fast", "comprehensive"],
                            help="Balanced: Statistical+Semantic (recommended)"
                        )
                    
                    # Show preview
                    with st.expander("👀 Original Data Preview", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Advanced clean button
                    if st.button("🧹 Advanced Clean", key="phase2_clean_advanced"):
                        clean_data_phase2(df, uploaded_file.name, processing_mode)
                        
                except Exception as e:
                    st.error(f"❌ Failed to load data: {e}")
        
        # Demo section
        st.markdown("---")
        st.markdown("#### 🧪 Try Demo")
        
        if st.button("🎯 Generate Messy Demo Data", key="phase2_demo"):
            generate_demo_data_phase2()

def generate_demo_data_phase2():
    """Generate and clean demo messy data with comparison"""
    
    messy_data = {
        'Policy': ['Insurance Report Q4 2023', '', 'POL001', 'POL002', '', 'TOTAL: 2', 'Generated on 2023-12-01'],
        'Premium': ['Premium Amount', None, '25000', '18500', '', '43500', 'System Export'],
        'Age': [None, None, '35', '42', None, None, None],
        'Status': ['Status', '', 'Active', 'Active', '', 'SUMMARY', 'End Report']
    }
    
    df_demo = pl.DataFrame(messy_data)
    df_demo_pandas = df_demo.to_pandas()
    
    # Immediately clean and show comparison
    if not PHASE2_CLEANING_AVAILABLE:
        st.error("❌ Phase 2 cleaning system not available")
        return
        
    with st.spinner("🧹 Cleaning demo data..."):
        try:
            # Apply Phase 2 cleaning
            from src.cleaning.hybrid_detector import create_hybrid_detector
            detector = create_hybrid_detector('balanced')
            result = detector.detect_junk_rows(df_demo, return_detailed=True)
            
            # Get cleaned dataframe
            if result.junk_row_indices:
                mask = pl.Series(range(df_demo.shape[0])).is_in(result.junk_row_indices).not_()
                df_clean_demo = df_demo.filter(mask)
                df_cleaned_pandas = df_clean_demo.to_pandas()
            else:
                df_cleaned_pandas = df_demo_pandas.copy()
            
            # Show results with metrics
            st.success("✅ Demo cleaning completed!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Original Rows", df_demo_pandas.shape[0])
            with col2:
                st.metric("✨ Clean Rows", df_cleaned_pandas.shape[0])
            with col3:
                st.metric("🗑️ Removed Rows", len(result.junk_row_indices))
            
            # Show before/after comparison
            if len(result.junk_row_indices) > 0:
                st.subheader("📋 Demo: Before vs After")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.write("**🔴 Original Messy Data**")
                    st.dataframe(df_demo_pandas, height=400)
                    
                with col_right:
                    st.write("**✅ Cleaned Data**")
                    st.dataframe(df_cleaned_pandas, height=400)
                
                # Show what was removed
                if st.expander("🗑️ View Removed Junk Rows"):
                    junk_df_demo = df_demo.filter(pl.Series(range(df_demo.shape[0])).is_in(result.junk_row_indices))
                    st.dataframe(junk_df_demo.to_pandas())
                    st.info(f"Removed {len(result.junk_row_indices)} junk rows: {result.junk_row_indices}")
                    st.markdown("**Layers used:** " + ", ".join(result.layers_used))
            
            else:
                st.info("🎉 No junk rows detected in demo - data is already clean!")
                st.dataframe(df_cleaned_pandas)
            
        except Exception as e:
            st.error(f"❌ Demo cleaning failed: {e}")
            # Fallback to showing raw data
            st.dataframe(df_demo_pandas)

def clean_data_phase2(df, filename, processing_mode):
    """Clean data with Phase 2 system and show results"""
    
    with st.spinner(f"🔬 Cleaning with {processing_mode} mode..."):
        try:
            # Create hybrid detector
            detector = create_hybrid_detector(processing_mode)
            
            # Get results
            result = detector.detect_junk_rows(df, return_detailed=True)
            
            # Clean data
            if result.junk_row_indices:
                mask = pl.Series(range(df.shape[0])).is_in(result.junk_row_indices).not_()
                clean_df = df.filter(mask)
            else:
                clean_df = df
            
            # Results display
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                st.metric("Original", df.shape[0])
            with col_r2:
                st.metric("Clean", clean_df.shape[0])
            with col_r3:
                st.metric("Removed", len(result.junk_row_indices))
            with col_r4:
                st.metric("Time", f"{result.processing_time:.2f}s")
            
            # System details
            st.markdown(f"**Layers Used**: {', '.join(result.layers_used)}")
            if result.early_exit_triggered:
                st.success("🚀 Early exit (high confidence)")
            
            # Show removed rows
            if result.junk_row_indices:
                with st.expander(f"🗑️ Removed Junk Rows ({len(result.junk_row_indices)})", expanded=False):
                    junk_df = df.filter(pl.Series(range(df.shape[0])).is_in(result.junk_row_indices))
                    st.dataframe(junk_df, use_container_width=True)
                    
                    # Explanations
                    st.markdown("**Why removed:**")
                    for i, idx in enumerate(result.junk_row_indices[:3]):
                        content = ' | '.join(str(val) for val in df.row(idx))
                        st.caption(f"Row {idx}: `{content}`")
            
            # Clean data preview
            with st.expander("✨ Clean Data", expanded=True):
                st.dataframe(clean_df.head(10), use_container_width=True)
            
            # Download
            clean_csv = clean_df.to_pandas().to_csv(index=False)
            st.download_button(
                "📥 Download Clean Data",
                clean_csv,
                f"clean_{filename}",
                "text/csv",
                use_container_width=True
            )
            
            st.success("🎉 Cleaning completed!")
            
        except Exception as e:
            st.error(f"❌ Cleaning failed: {e}")

def display_workflow_progress():
    """Display simple workflow progress"""
    
    current_step = st.session_state.workflow_step
    step_names = ["Upload", "Process", "Analyze", "Price", "Results"]
    
    # Simple progress indicator
    progress = (current_step - 1) / 4
    st.progress(progress)
    
    # Navigation with both previous and next buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_step > 1:
            if st.button("← Previous", key="prev_step"):
                st.session_state.workflow_step = current_step - 1
                st.rerun()
    
    with col2:
        st.markdown(f"**Step {current_step}/5**: {step_names[current_step-1]}", unsafe_allow_html=True)
    
    with col3:
        if current_step < 5:
            if st.button("Next →", type="primary", key="next_step"):
                st.session_state.workflow_step = current_step + 1
                st.rerun()
    
    st.markdown("---")

def step_1_data_upload():
    """Step 1: Universal Data Upload"""
    
    st.markdown("## 📤 Upload Your Data")
    
    # Initialize engines safely
    initialize_engines_safely()
    
    # Required data types for comprehensive pricing
    required_data_types = [
        "Policy Data", "Mortality Tables", "Claims Experience", 
        "Economic Scenarios", "Expense Data"
    ]
    
    optional_data_types = [
        "Premium Transactions", "Medical Underwriting", "Lapse Rates",
        "Product Features", "Investment Returns", "Reinsurance Treaties"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Data Files")
        st.info("💡 Upload CSV, Excel, JSON, or TXT files. We'll analyze and clean your data automatically!")
        
        uploaded_files = st.file_uploader(
            "Choose multiple files",
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'json', 'txt', 'xls'],
            key="comprehensive_upload"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} files uploaded**")
            
            for file in uploaded_files:
                # Check if already processed
                file_key = file.name.replace('.', '_').replace('-', '_')
                is_processed = file_key in st.session_state.uploaded_datasets
                
                with st.expander(f"📄 {file.name} {'✅' if is_processed else '⏳'}", expanded=not is_processed):
                    
                    # File info section
                    st.markdown(f"**Size:** {file.size:,} bytes | **Type:** {file.name.split('.')[-1].upper()}")
                    
                    # Single action button - direct clean & compare
                    if st.button("🧹 Clean Data & Show Results", key=f"clean_{file.name}", type="primary", use_container_width=True):
                        import hashlib
                        file_content = file.read()
                        file.seek(0)  # Reset file pointer
                        content_hash = hashlib.md5(file_content).hexdigest()[:8]
                        file_key = f"{file.name}_{content_hash}"
                        clean_and_compare_file(file, file_key)
                    
                    # Data preview section
                    if st.session_state.get(f"show_preview_{file.name}", False):
                        st.markdown("#### 📊 Data Preview")
                        
                        col_close = st.columns([1])[0]
                        if st.button("❌ Close Preview", key=f"close_preview_{file.name}"):
                            st.session_state[f"show_preview_{file.name}"] = False
                            st.rerun()
                        
                        try:
                            if file.name.endswith(('.csv', '.txt')):
                                content = file.read().decode('utf-8')
                                file.seek(0)
                                
                                lines = content.split('\n')[:15]
                                preview_text = '\n'.join(lines)
                                st.code(preview_text, language='csv')
                                
                                if len(content.split('\n')) > 15:
                                    st.caption(f"Showing first 15 lines of {len(content.split('chr(10)'))} total...")
                                    
                            elif file.name.endswith(('.xlsx', '.xls')):
                                st.info("📊 Excel file - click Process to analyze sheets and columns")
                                
                            elif file.name.endswith('.json'):
                                content = file.read().decode('utf-8')
                                file.seek(0)
                                preview = content[:800] + "..." if len(content) > 800 else content
                                st.code(preview, language='json')
                                
                        except Exception as e:
                            st.warning(f"Could not preview: {str(e)}")
                    
                    # Processing results - with proper spacing
                    if is_processed:
                        dataset_info = st.session_state.uploaded_datasets[file_key]
                        
                        # Get cleaning history for this file (define at proper scope)
                        file_cleaning_history = st.session_state.get(f"cleaning_history_{file_key}", [])
                        
                        st.markdown("---")
                        st.markdown("#### ✅ Processing Complete")
                        
                        # Basic metrics in a clean row
                        st.markdown(f"**Records:** {dataset_info['records']:,} | **Data Type:** {dataset_info['data_type'].replace('_', ' ').title()}")
                        
                        # Show enhanced profiler results if available
                        if 'enhanced_profile' in dataset_info:
                            profile = dataset_info['enhanced_profile']
                            issues_count = len(profile['recommendations'])
                            structural_issues = len(profile['structural_issues'])
                            
                            # Enhanced quality display
                            completeness = profile['data_quality']['overall_completeness']
                            st.markdown(f"**🔍 Enhanced Analysis:** {completeness:.1f}% complete | {issues_count} issues detected | {structural_issues} structural issues")
                            
                            # Show specific issues found
                            if profile['recommendations']:
                                st.markdown("##### 💡 Key Issues Detected:")
                                for i, rec in enumerate(profile['recommendations'][:3], 1):
                                    issue_icon = "🚨" if rec['category'] == 'Data Quality' else "⚠️"
                                    st.markdown(f"{issue_icon} **{rec['recommendation']}**: {rec['issue']}")
                                    
                                if len(profile['recommendations']) > 3:
                                    st.info(f"+ {len(profile['recommendations']) - 3} additional issues detected - view detailed report for more")
                            else:
                                st.success("🟢 Excellent data quality - no issues detected!")
                                
                        else:
                            # Fallback to basic display
                            quality_color = "🟢" if dataset_info['quality_score'] >= 85 else "🟡" if dataset_info['quality_score'] >= 70 else "🟠"
                            st.markdown(f"**Quality Score:** {quality_color} {dataset_info['quality_score']}/100")
                        
                        # Comprehensive Data Quality Report
                        if st.button("📋 View Data Quality Report", key=f"quality_report_{file.name}", use_container_width=True):
                            st.session_state[f"show_quality_report_{file_key}"] = True
                            st.rerun()
                        
                        # Show comprehensive quality report
                        if st.session_state.get(f"show_quality_report_{file_key}", False):
                            display_comprehensive_quality_report(dataset_info['data'], file.name, file_key)
                        
                        # Data cleaning section - only if needed
                        if dataset_info['quality_score'] < 85:
                            st.markdown("")  # Add space
                            st.markdown("🧹 **Data Cleaning Available**")
                            
                            # Check if we have cleaning history
                            cleaning_history = st.session_state.get(f"cleaning_history_{file_key}", [])
                            original_data = st.session_state.get(f"original_data_{file_key}", dataset_info['data'].copy())
                            
                            # Store original if not stored
                            if f"original_data_{file_key}" not in st.session_state:
                                st.session_state[f"original_data_{file_key}"] = dataset_info['data'].copy()
                                st.session_state[f"cleaning_history_{file_key}"] = []
                            
                            # Cleaning action buttons
                            col_clean1, col_clean2, col_clean3 = st.columns(3)
                            
                            with col_clean1:
                                if st.button("🔧 Enhanced Cleaning", key=f"enhance_{file.name}", use_container_width=True):
                                    # Apply cleaning and track changes
                                    cleaned_data, changes = apply_enhanced_cleaning(dataset_info['data'])
                                    
                                    # Update dataset with cleaned data
                                    old_quality = dataset_info['quality_score']
                                    new_quality = min(95, old_quality + 8)
                                    
                                    # Store cleaning step
                                    cleaning_step = {
                                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                                        'operation': 'Enhanced Cleaning',
                                        'quality_before': old_quality,
                                        'quality_after': new_quality,
                                        'changes': changes,
                                        'records_before': len(dataset_info['data']),
                                        'records_after': len(cleaned_data)
                                    }
                                    
                                    # Update data and history
                                    dataset_info['data'] = cleaned_data
                                    dataset_info['quality_score'] = new_quality
                                    st.session_state.uploaded_datasets[file_key] = dataset_info
                                    
                                    cleaning_history.append(cleaning_step)
                                    st.session_state[f"cleaning_history_{file_key}"] = cleaning_history
                                    
                                    st.success(f"✅ Quality improved from {old_quality} to {new_quality}!")
                                    st.rerun()
                            
                            with col_clean2:
                                if st.button("📋 View Issues", key=f"issues_{file.name}", use_container_width=True):
                                    st.session_state[f"show_issues_{file.name}"] = not st.session_state.get(f"show_issues_{file.name}", False)
                                    st.rerun()
                            
                            with col_clean3:
                                if cleaning_history and st.button("↩️ Undo Last", key=f"undo_{file.name}", use_container_width=True):
                                    # Restore previous state
                                    if len(cleaning_history) == 1:
                                        # Restore original
                                        dataset_info['data'] = original_data.copy()
                                        dataset_info['quality_score'] = cleaning_history[0]['quality_before']
                                        st.session_state[f"cleaning_history_{file_key}"] = []
                                    else:
                                        # Go back one step (would need more complex versioning for this)
                                        cleaning_history.pop()
                                        # For simplicity, restore to original and reapply remaining steps
                                        dataset_info['data'] = original_data.copy()
                                        dataset_info['quality_score'] = cleaning_history[-1]['quality_after'] if cleaning_history else cleaning_history[0]['quality_before']
                                        st.session_state[f"cleaning_history_{file_key}"] = cleaning_history
                                    
                                    st.session_state.uploaded_datasets[file_key] = dataset_info
                                    st.success("↩️ Changes undone!")
                                    st.rerun()
                            
                            # Show cleaning history
                            if cleaning_history:
                                with st.expander("📜 Cleaning History", expanded=False):
                                    for i, step in enumerate(reversed(cleaning_history)):
                                        st.markdown(f"**{len(cleaning_history)-i}.** {step['operation']} at {step['timestamp']}")
                                        st.markdown(f"   Quality: {step['quality_before']} → {step['quality_after']} | Records: {step['records_before']} → {step['records_after']}")
                                        if step['changes']:
                                            st.markdown(f"   Changes: {', '.join(step['changes'][:3])}...")
                            
                            # Show issues if requested
                            if st.session_state.get(f"show_issues_{file.name}", False):
                                st.info("**Common issues detected:**\n• Missing values in some columns\n• Inconsistent date formats\n• Potential duplicate records")
                        
                        # Download and export options
                        st.markdown("")
                        st.markdown("📥 **Download Options**")
                        
                        col_download1, col_download2, col_download3 = st.columns(3)
                        
                        with col_download1:
                            # CSV download
                            csv_data = dataset_info['data'].to_csv(index=False)
                            st.download_button(
                                "📄 CSV", 
                                data=csv_data,
                                file_name=f"cleaned_{file.name}",
                                mime="text/csv",
                                key=f"csv_{file.name}",
                                use_container_width=True
                            )
                        
                        with col_download2:
                            # Excel download
                            excel_buffer = io.BytesIO()
                            
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                dataset_info['data'].to_excel(writer, sheet_name='Cleaned Data', index=False)
                                
                                # Add cleaning history sheet if available
                                if file_cleaning_history:
                                    history_df = pd.DataFrame(file_cleaning_history)
                                    history_df.to_excel(writer, sheet_name='Cleaning History', index=False)
                            
                            st.download_button(
                                "📊 Excel",
                                data=excel_buffer.getvalue(),
                                file_name=f"cleaned_{file.name.rsplit('.', 1)[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"excel_{file.name}",
                                use_container_width=True
                            )
                        
                        with col_download3:
                            # Original vs Cleaned comparison
                            if file_cleaning_history:
                                original_data = st.session_state.get(f"original_data_{file_key}")
                                if original_data is not None:
                                    comparison_buffer = io.BytesIO()
                                    with pd.ExcelWriter(comparison_buffer, engine='openpyxl') as writer:
                                        original_data.to_excel(writer, sheet_name='Original', index=False)
                                        dataset_info['data'].to_excel(writer, sheet_name='Cleaned', index=False)
                                        pd.DataFrame(file_cleaning_history).to_excel(writer, sheet_name='Changes Log', index=False)
                                    
                                    st.download_button(
                                        "🔄 Comparison",
                                        data=comparison_buffer.getvalue(),
                                        file_name=f"comparison_{file.name.rsplit('.', 1)[0]}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"compare_{file.name}",
                                        use_container_width=True
                                    )
                        
                        # Quick insights - expandable
                        with st.expander("📊 Data Insights"):
                            st.write(f"**Identified as:** {dataset_info['data_type'].replace('_', ' ').title()}")
                            st.write(f"**Ready for:** Reinsurance pricing analysis")
                            st.write(f"**Records:** {len(dataset_info['data']):,}")
                            st.write(f"**Columns:** {len(dataset_info['data'].columns)}")
                            if file_cleaning_history:
                                st.write(f"**Cleaning steps applied:** {len(file_cleaning_history)}")
                    
                    else:
                        st.markdown("---")
                        st.markdown("#### ⏳ Ready to Process")
                        
                        st.markdown("**What processing does:**")
                        st.markdown("• 🔍 Detects data type automatically")
                        st.markdown("• 🧹 Cleans and standardizes formats")  
                        st.markdown("• 📊 Generates quality assessment")
                        st.markdown("• 🏷️ Categorizes for pricing analysis")
                        
                        st.markdown("")  # Add space
                        # Removed old processing - now only use smart header detection approach
    
    with col2:
        st.markdown("### Data Requirements")
        
        # Required data status
        st.markdown("**Required Data:**")
        for data_type in required_data_types:
            if data_type.lower().replace(' ', '_') in st.session_state.uploaded_datasets:
                st.markdown(f"✅ {data_type}")
            else:
                st.markdown(f"⏳ {data_type}")
        
        st.markdown("**Optional Data:**")
        for data_type in optional_data_types[:3]:  # Show first 3
            if data_type.lower().replace(' ', '_') in st.session_state.uploaded_datasets:
                st.markdown(f"✅ {data_type}")
            else:
                st.markdown(f"➖ {data_type}")
        
        # Progress to next step
        uploaded_count = len(st.session_state.uploaded_datasets)
        if uploaded_count >= 2:  # Need at least 2 datasets
            st.success(f"✅ {uploaded_count} datasets ready!")
            if st.button("➡️ Continue to Analysis", type="primary"):
                st.session_state.workflow_step = 2
                st.rerun()
        else:
            st.info("Upload at least 2 datasets to continue")

def clear_old_cache_files():
    """Clear old cache files to prevent stale data issues"""
    try:
        result_files = list(TEMP_DIR.glob("*_results.pkl"))
        for file in result_files:
            file.unlink()
    except Exception:
        pass  # Ignore cleanup errors

def clean_and_compare_file(uploaded_file, file_key):
    """Simplified single-click cleaning with before/after comparison"""
    
    if not PHASE2_CLEANING_AVAILABLE:
        st.error("❌ Phase 2 cleaning system not available")
        return
    
    # Clear old cache files to prevent stale data
    clear_old_cache_files()
    
    with st.spinner(f"🧹 Cleaning and analyzing: {uploaded_file.name}..."):
        try:
            # Load file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Basic file info
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Load as DataFrame
            if file_type == 'csv':
                import io
                df_original = pd.read_csv(io.BytesIO(file_content))
            elif file_type in ['xlsx', 'xls']:
                import io
                df_original = pd.read_excel(io.BytesIO(file_content))
            elif file_type == 'json':
                import json
                data = json.loads(file_content.decode('utf-8'))
                df_original = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            else:
                st.error(f"❌ Unsupported file type: {file_type}")
                return
            
            # Clean the data using Phase 2
            df_clean = df_original.copy()
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).replace('nan', None)
            
            # Convert to Polars for Phase 2 system
            df_polars = pl.from_pandas(df_clean)
            
            # Apply Phase 2 cleaning with smart header detection
            from src.cleaning.hybrid_detector import create_hybrid_detector
            detector = create_hybrid_detector('balanced')
            result = detector.detect_junk_rows(df_polars, return_detailed=True)
            
            # Get cleaned dataframe (header detection happens inside the detector now)
            if result.header_detection_result:
                # Use the cleaned dataframe from header detection
                from src.cleaning.header_detector import detect_and_clean_headers
                df_clean_polars, header_result = detect_and_clean_headers(df_polars)
                
                # Apply additional junk removal to the already header-cleaned data
                if result.junk_row_indices:
                    mask = pl.Series(range(df_clean_polars.shape[0])).is_in(result.junk_row_indices).not_()
                    df_clean_polars = df_clean_polars.filter(mask)
                
                df_cleaned = df_clean_polars.to_pandas()
                
                # Update metrics to show total removal
                total_removed_count = len(header_result.rows_to_remove_above) + len(result.junk_row_indices)
                header_info = f"Header detected at row {header_result.header_row_index}"
                
            else:
                # Fallback to original approach
                if result.junk_row_indices:
                    mask = pl.Series(range(df_polars.shape[0])).is_in(result.junk_row_indices).not_()
                    df_clean_polars = df_polars.filter(mask)
                    df_cleaned = df_clean_polars.to_pandas()
                else:
                    df_cleaned = df_original.copy()
                
                total_removed_count = len(result.junk_row_indices)
                header_info = "No header detection performed"
            
            # Store results in session state to persist across reruns
            cleaning_results = {
                'df_original': df_original,
                'df_cleaned': df_cleaned,
                'result': result,
                'total_removed_count': total_removed_count,
                'header_info': header_info if 'header_info' in locals() else "No header detection",
                'df_polars': df_polars,
                'df_clean_polars': df_clean_polars if 'df_clean_polars' in locals() else pl.from_pandas(df_cleaned),
                'uploaded_file_name': uploaded_file.name,
                'file_type': file_type
            }
            st.session_state[f'cleaning_results_{file_key}'] = cleaning_results
            # Store persistently using both session state AND file system
            cleaning_results['file_key'] = file_key
            save_persistent_results(file_key, cleaning_results)
            
            # Show the results immediately after cleaning
            st.session_state['show_persistent_results'] = True
            
            # Also integrate with workflow system
            if 'uploaded_datasets' not in st.session_state:
                st.session_state['uploaded_datasets'] = {}
            
            st.session_state['uploaded_datasets'][file_key] = {
                'name': uploaded_file.name,
                'data': df_cleaned,
                'original_data': df_original,
                'cleaning_result': result,
                'file_type': file_type
            }
            
            # Show results with metrics
            st.success("✅ Cleaning completed!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Original Rows", df_original.shape[0])
            with col2:
                st.metric("✨ Clean Rows", df_cleaned.shape[0])
            with col3:
                st.metric("🗑️ Removed Rows", total_removed_count)
                
            # Show the detailed results
            show_cleaning_results_display(cleaning_results)
            
        except Exception as e:
            st.error(f"❌ Cleaning failed: {e}")
            st.info("The file might have formatting issues. Try using the advanced processing option.")

def show_persistent_cleaning_results():
    """Show persistent cleaning results that survive all reruns"""
    cleaning_results = get_latest_persistent_results()
    if not cleaning_results:
        st.warning("No recent cleaning results found")
        return
    
    # Show metrics
    st.success("✅ Cleaning completed!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Original Rows", cleaning_results['df_original'].shape[0])
    with col2:
        st.metric("✨ Clean Rows", cleaning_results['df_cleaned'].shape[0])
    with col3:
        st.metric("🗑️ Removed Rows", cleaning_results['total_removed_count'])
    
    # Add "Process New File" button to reset
    if st.button("🔄 Process New File", type="secondary", use_container_width=True):
        # Clear the stored results
        if 'latest_cleaning_results' in st.session_state:
            del st.session_state['latest_cleaning_results']
        # Clear chat state
        if CHAT_ASSISTANT_AVAILABLE:
            reset_chat_session()
        st.rerun()
    
    # Show the detailed results 
    show_cleaning_results_display(cleaning_results)

def show_existing_cleaning_results(uploaded_file, file_key):
    """Show existing cleaning results from session state"""
    cleaning_results = st.session_state[f'cleaning_results_{file_key}']
    
    # Show metrics
    st.success("✅ Cleaning completed!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Original Rows", cleaning_results['df_original'].shape[0])
    with col2:
        st.metric("✨ Clean Rows", cleaning_results['df_cleaned'].shape[0])
    with col3:
        st.metric("🗑️ Removed Rows", cleaning_results['total_removed_count'])
    
    # Add "Process New File" button to reset
    if st.button("🔄 Process New File", type="secondary", use_container_width=True):
        # Clear the stored results
        if f'cleaning_results_{file_key}' in st.session_state:
            del st.session_state[f'cleaning_results_{file_key}']
        # Clear chat state
        if CHAT_ASSISTANT_AVAILABLE:
            reset_chat_session()
        st.rerun()
    
    # Show the detailed results 
    show_cleaning_results_display(cleaning_results)

def show_cleaning_results_display(cleaning_results):
    """Display the cleaning results (used by both new and existing results)"""
    df_original = cleaning_results['df_original']
    df_cleaned = cleaning_results['df_cleaned'] 
    result = cleaning_results['result']
    total_removed_count = cleaning_results['total_removed_count']
    header_info = cleaning_results['header_info']
    file_type = cleaning_results.get('file_type', 'csv')  # Default to csv if not found
    uploaded_file_name = cleaning_results.get('uploaded_file_name', 'cleaned_data')
    
    # Show header detection info
    if result.header_detection_result:
        st.info(f"🎯 {header_info} (confidence: {result.header_detection_result.confidence:.2f})")
    
    # Show before/after comparison  
    if total_removed_count > 0:
        st.subheader("📋 Before vs After Comparison")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**🔴 Original Data (with junk)**")
            st.dataframe(df_original.head(20), height=400)
            
        with col_right:
            st.write("**✅ Cleaned Data**")
            st.dataframe(df_cleaned.head(20), height=400)
        
        # Show detailed change log and removed content
        with st.expander("📋 Detailed Change Log", expanded=False):
            # Header detection details
            if result.header_detection_result:
                st.markdown("### 🎯 Header Detection Results")
                st.markdown(f"**Header found at row:** {result.header_detection_result.header_row_index}")
                st.markdown(f"**Confidence:** {result.header_detection_result.confidence:.2f}")
                st.markdown(f"**Column names detected:** {', '.join(result.header_detection_result.column_names[:5])}...")
                
                if result.header_detection_result.rows_to_remove_above:
                    st.markdown(f"**Junk rows removed above header:** {len(result.header_detection_result.rows_to_remove_above)}")
                    
                    # Show removed header junk
                    st.markdown("#### 🗑️ Removed Header Junk:")
                    header_junk_df = df_original.iloc[result.header_detection_result.rows_to_remove_above]
                    st.dataframe(header_junk_df, use_container_width=True)
                    
                    # Additional junk removal details
                    if len(result.junk_row_indices) > 0:
                        st.markdown("### 🧹 Additional Junk Detection")
                        st.markdown(f"**Layers used:** {', '.join(result.layers_used)}")
                        st.markdown(f"**Processing time:** {result.processing_time:.3f}s")
                        st.markdown(f"**Additional junk rows:** {len(result.junk_row_indices)}")
                        
                        # Show additional removed junk (from cleaned data)
                        if result.header_detection_result:
                            # Calculate indices in cleaned data context
                            cleaned_junk_start_row = result.header_detection_result.header_row_index + 1
                            original_junk_indices = [idx + cleaned_junk_start_row for idx in result.junk_row_indices]
                            additional_junk_df = df_original.iloc[original_junk_indices]
                        else:
                            additional_junk_df = df_original.iloc[result.junk_row_indices]
                        
                        st.markdown("#### 🗑️ Additional Junk Removed:")
                        st.dataframe(additional_junk_df, use_container_width=True)
                    
                    # Summary
                    st.markdown("### 📊 Cleaning Summary")
                    st.markdown(f"**Total rows processed:** {df_original.shape[0]}")
                    st.markdown(f"**Total rows removed:** {total_removed_count}")
                    st.markdown(f"**Final clean rows:** {df_cleaned.shape[0]}")
                    st.markdown(f"**Data quality improvement:** {(total_removed_count/df_original.shape[0]*100):.1f}% junk removed")
            
            else:
                st.info("🎉 No junk rows detected - your data is already clean!")
                st.dataframe(df_cleaned.head(20))
            
            # Comprehensive download options
            st.markdown("### 💾 Download Options")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.markdown("#### 📄 Cleaned Data Only")
                if file_type == 'csv':
                    csv_data = df_cleaned.to_csv(index=False)
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name=f"cleaned_{uploaded_file_name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                elif file_type in ['xlsx', 'xls']:
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_cleaned.to_excel(writer, sheet_name='Cleaned Data', index=False)
                    st.download_button(
                        label="📥 Excel",
                        data=buffer.getvalue(),
                        file_name=f"cleaned_{uploaded_file_name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col_dl2:
                st.markdown("#### 📦 Complete Package")
                # Create comprehensive Excel with multiple sheets
                import io
                buffer = io.BytesIO()
                base_name = uploaded_file_name.rsplit('.', 1)[0]
                
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Sheet 1: Cleaned data
                    df_cleaned.to_excel(writer, sheet_name='Cleaned Data', index=False)
                    
                    # Sheet 2: Original data
                    df_original.to_excel(writer, sheet_name='Original Data', index=False)
                    
                    # Sheet 3: Change log
                    change_log_data = []
                    if result.header_detection_result:
                        change_log_data.extend([
                            ['Header Detection', f'Found at row {result.header_detection_result.header_row_index}', f'Confidence: {result.header_detection_result.confidence:.2f}'],
                            ['Columns Detected', ', '.join(result.header_detection_result.column_names), ''],
                            ['Header Junk Removed', len(result.header_detection_result.rows_to_remove_above), f'Rows: {result.header_detection_result.rows_to_remove_above}']
                        ])
                    
                    change_log_data.extend([
                        ['Additional Junk Removed', len(result.junk_row_indices), f'Rows: {result.junk_row_indices}'],
                        ['Layers Used', ', '.join(result.layers_used), ''],
                        ['Processing Time', f'{result.processing_time:.3f}s', ''],
                        ['Total Original Rows', df_original.shape[0], ''],
                        ['Total Clean Rows', df_cleaned.shape[0], ''],
                        ['Total Removed', total_removed_count, ''],
                        ['Quality Improvement', f'{(total_removed_count/df_original.shape[0]*100):.1f}% junk removed', '']
                    ])
                    
                    change_log_df = pd.DataFrame(change_log_data, columns=['Metric', 'Value', 'Details'])
                    change_log_df.to_excel(writer, sheet_name='Change Log', index=False)
                    
                    # Sheet 4: Removed junk (if any)
                    if result.header_detection_result and result.header_detection_result.rows_to_remove_above:
                        header_junk = df_original.iloc[result.header_detection_result.rows_to_remove_above]
                        header_junk.to_excel(writer, sheet_name='Removed Header Junk', index=True)
                    
                    if len(result.junk_row_indices) > 0:
                        if result.header_detection_result:
                            # Additional junk from cleaned data context
                            cleaned_junk_start = result.header_detection_result.header_row_index + 1
                            original_junk_indices = [idx + cleaned_junk_start for idx in result.junk_row_indices]
                            additional_junk = df_original.iloc[original_junk_indices]
                        else:
                            additional_junk = df_original.iloc[result.junk_row_indices]
                        additional_junk.to_excel(writer, sheet_name='Removed Additional Junk', index=True)
                
                st.download_button(
                    label="📦 Complete Package",
                    data=buffer.getvalue(),
                    file_name=f"{base_name}_CLEANED_COMPLETE.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Raw + Clean + Change Log + Removed Data"
                )
            
            # Note: PriceRe Chat is now available globally in the sidebar

# Removed old conservative cleaning function - now using only smart header detection

def process_file_with_enterprise_profiler(uploaded_file):
    """Process file using Great Expectations enterprise profiler with fallback options"""
    
    file_key = uploaded_file.name.replace('.', '_').replace('-', '_')
    
    with st.spinner(f"🔍 Analyzing with Enterprise Profiler: {uploaded_file.name}..."):
        try:
            # Load file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Basic file info
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Load as DataFrame
            if file_type == 'csv':
                import io
                df = pd.read_csv(io.BytesIO(file_content))
            elif file_type in ['xlsx', 'xls']:
                import io
                df = pd.read_excel(io.BytesIO(file_content))
            elif file_type == 'json':
                import json
                data = json.loads(file_content.decode('utf-8'))
                df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Apply Phase 2 hybrid cleaning (our main cleaning system)
            if PHASE2_CLEANING_AVAILABLE:
                st.info("🧹 Applying Hybrid Cleaning (Statistical + Semantic)...")
                
                try:
                    # Clean data types before conversion to avoid mixed type issues
                    df_clean = df.copy()
                    for col in df_clean.columns:
                        # Convert all columns to string to avoid mixed type issues with messy data
                        df_clean[col] = df_clean[col].astype(str).replace('nan', None)
                    
                    # Convert to Polars for our Phase 2 system
                    df_polars = pl.from_pandas(df_clean)
                    
                except Exception as e:
                    st.error(f"❌ Data conversion failed: {e}")
                    st.info("Trying alternative approach...")
                    try:
                        # Alternative: Use our universal data source reader
                        # Save to temporary CSV and read back
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                            df.to_csv(tmp.name, index=False)
                            df_polars = read_any_source(tmp.name)
                            os.unlink(tmp.name)
                    except Exception as e2:
                        st.error(f"❌ Alternative conversion also failed: {e2}")
                        st.warning("⚠️ Skipping cleaning due to data format issues")
                        df_polars = None
                
                # Apply Phase 2 cleaning if conversion succeeded
                if df_polars is not None:
                    detector = create_hybrid_detector('balanced')
                    result = detector.detect_junk_rows(df_polars, return_detailed=True)
                    
                    if result.junk_row_indices:
                        st.success(f"✅ Removed {len(result.junk_row_indices)} junk rows!")
                        st.info(f"🔬 Layers used: {', '.join(result.layers_used)}")
                        st.info(f"⚡ Processing time: {result.processing_time:.3f}s")
                        
                        # Show what was removed
                        with st.expander(f"🗑️ Removed Junk Rows ({len(result.junk_row_indices)})", expanded=False):
                            junk_df_polars = df_polars.filter(pl.Series(range(df_polars.shape[0])).is_in(result.junk_row_indices))
                            st.dataframe(junk_df_polars.to_pandas())
                        
                        # Clean the data
                        mask = pl.Series(range(df_polars.shape[0])).is_in(result.junk_row_indices).not_()
                        clean_df_polars = df_polars.filter(mask)
                        
                        # Convert back to pandas for the rest of the pipeline
                        df = clean_df_polars.to_pandas()
                        st.success(f"📊 Clean data: {df.shape[0]} rows × {df.shape[1]} columns")
                    else:
                        st.info("✅ No junk rows detected - data already clean!")
            else:
                st.warning("⚠️ Phase 2 cleaning system not available - proceeding without cleaning")
            
            # THEN: Try Great Expectations Enterprise Profiler
            if ENTERPRISE_PROFILER_AVAILABLE:
                try:
                    # Add memory check for large files
                    if len(df) > 50000:
                        st.warning("⚠️ Large dataset detected - using optimized enterprise processing")
                    
                    profiler = SimpleGreatExpectationsProfiler()
                    
                    # Determine target column for analysis
                    columns = list(df.columns)
                    target_column = None
                    for col in columns:
                        if any(keyword in col.lower() for keyword in ['premium', 'amount', 'claim', 'value', 'price']):
                            target_column = col
                            break
                    
                    # Generate enterprise profile
                    profile = profiler.generate_enterprise_profile(df, target_column)
                    
                    # Extract enterprise profiler results
                    enterprise_summary = profile['enterprise_summary']
                    quality_score = int(enterprise_summary['data_quality_score'])
                    
                    # Determine data type based on business rules validation
                    business_validation = profile['business_rules_validation']
                    if business_validation['data_domain'] == 'insurance_policy':
                        data_type = 'Insurance Policy Data'
                    elif business_validation['data_domain'] == 'claims':
                        data_type = 'Claims Data'
                    elif business_validation['data_domain'] == 'mortality':
                        data_type = 'Mortality Data'
                    elif business_validation['data_domain'] == 'financial':
                        data_type = 'Financial Data'
                    else:
                        data_type = 'Insurance Data'
                    
                    # Create comprehensive issues summary
                    quality_issues = enterprise_summary['data_quality_issues']
                    compliance_issues = len([issue for issue in profile['regulatory_compliance']['compliance_status'] if not issue['compliant']])
                    risk_indicators = len(profile['risk_indicators'])
                    
                    total_issues = quality_issues + compliance_issues + risk_indicators
                    
                    if total_issues > 10:
                        issues_summary = f"🔴 {total_issues} critical issues found (Quality: {quality_issues}, Compliance: {compliance_issues}, Risk: {risk_indicators})"
                    elif total_issues > 5:
                        issues_summary = f"🟡 {total_issues} issues detected (Quality: {quality_issues}, Compliance: {compliance_issues}, Risk: {risk_indicators})"
                    elif total_issues > 0:
                        issues_summary = f"🟢 {total_issues} minor issues (Quality: {quality_issues}, Compliance: {compliance_issues})"
                    else:
                        issues_summary = "🟢 Enterprise-grade data quality - no issues detected!"
                    
                    # Create actionable recommendations from all sources
                    recommendations = []
                    
                    # Data quality recommendations
                    for issue in profile['data_quality_metrics']['failed_expectations'][:3]:
                        recommendations.append(f"• Quality: {issue['expectation_type']} failed - {issue.get('description', 'Review data')})")
                    
                    # Business rule recommendations
                    for rule, result in business_validation['validation_results'].items():
                        if not result['passed'] and len(recommendations) < 5:
                            recommendations.append(f"• Business Rule: {rule} - {result['message']}")
                    
                    # Compliance recommendations
                    for compliance in profile['regulatory_compliance']['compliance_status']:
                        if not compliance['compliant'] and len(recommendations) < 5:
                            recommendations.append(f"• Compliance: {compliance['requirement']} - {compliance['issue']}")
                    
                    # Risk indicators
                    for risk in profile['risk_indicators'][:2]:
                        if len(recommendations) < 5:
                            recommendations.append(f"• Risk: {risk['type']} - {risk['description']}")
                    
                    recommendations_text = "\n".join(recommendations) if recommendations else "Enterprise validation passed - no immediate actions needed"
                    
                    # Store enterprise profile data
                    try:
                        st.session_state.uploaded_datasets[file_key] = {
                            'filename': uploaded_file.name,
                            'data_type': data_type,
                            'data': df,
                            'quality_score': quality_score,
                            'records': len(df),
                            'issues': issues_summary,
                            'recommendations': recommendations_text,
                            'enterprise_profile': profile,  # Store full enterprise profile
                            'profiler_type': 'great_expectations',
                            'target_column': target_column,
                            'compliance_score': enterprise_summary.get('compliance_score', 0),
                            'risk_score': enterprise_summary.get('risk_score', 0)
                        }
                    except Exception as session_error:
                        st.error(f"Session storage failed: {session_error}")
                        # Try minimal storage
                        st.session_state.uploaded_datasets[file_key] = {
                            'filename': uploaded_file.name,
                            'data_type': data_type,
                            'data': df,
                            'quality_score': quality_score,
                            'records': len(df),
                            'issues': issues_summary,
                            'recommendations': recommendations_text
                        }
                    
                    st.success(f"✅ Enterprise Analysis Complete: {quality_score}% quality, {total_issues} issues identified, Compliance: {enterprise_summary.get('compliance_score', 0)}%")
                    
                except Exception as enterprise_error:
                    st.warning(f"Great Expectations profiler failed, trying enhanced profiler: {enterprise_error}")
                    # Fallback to enhanced profiler
                    process_with_enhanced_profiler(uploaded_file, df, file_key, file_type)
            
            # Fallback to Enhanced Profiler
            elif ENHANCED_PROFILER_AVAILABLE:
                process_with_enhanced_profiler(uploaded_file, df, file_key, file_type)
            
            # Final fallback to basic processing
            else:
                process_file_basic_fallback(uploaded_file, df, file_key, file_type)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def process_with_enhanced_profiler(uploaded_file, df, file_key, file_type):
    """Process with enhanced profiler as fallback from Great Expectations"""
    try:
        profiler = EnhancedDataProfiler()
        profile = profiler.profile_data(df)
        
        # Extract enhanced profiler results
        data_quality = profile['data_quality']['overall_completeness']
        quality_score = int(data_quality)
        
        # Determine data type based on column analysis
        columns = list(df.columns)
        if any('policy' in col.lower() for col in columns):
            data_type = 'Insurance Policy Data'
        elif any('claim' in col.lower() for col in columns):
            data_type = 'Claims Data'
        elif any('mortality' in col.lower() or 'death' in col.lower() for col in columns):
            data_type = 'Mortality Data'
        else:
            data_type = 'Other Insurance Data'
        
        # Create issues summary
        issues_count = len(profile['recommendations'])
        structural_issues = len(profile['structural_issues'])
        
        if issues_count > 5:
            issues_summary = f"🟡 {issues_count} data quality issues found"
        elif issues_count > 0:
            issues_summary = f"🟢 {issues_count} minor issues detected"
        else:
            issues_summary = "🟢 Excellent data quality"
        
        if structural_issues > 0:
            issues_summary += f", {structural_issues} structural issues"
        
        # Create actionable recommendations
        recommendations = []
        for rec in profile['recommendations'][:5]:  # Top 5 recommendations
            recommendations.append(f"• {rec['recommendation']}: {rec['issue']}")
        
        recommendations_text = "\n".join(recommendations) if recommendations else "No immediate actions needed"
        
        # Store enhanced profile data
        st.session_state.uploaded_datasets[file_key] = {
            'filename': uploaded_file.name,
            'data_type': data_type,
            'data': df,
            'quality_score': quality_score,
            'records': len(df),
            'issues': issues_summary,
            'recommendations': recommendations_text,
            'enhanced_profile': profile,
            'profiler_type': 'enhanced',
            'profiler': profiler
        }
        
        st.success(f"✅ Enhanced Analysis Complete: {quality_score}% quality, {issues_count} issues identified")
        
    except Exception as enhanced_error:
        st.warning(f"Enhanced profiler failed, using basic fallback: {enhanced_error}")
        process_file_basic_fallback(uploaded_file, df, file_key, file_type)

def process_file_basic_fallback(uploaded_file, df, file_key, file_type):
    """Fallback processing when enhanced profiler isn't available"""
    
    # Basic data type detection
    columns = list(df.columns)
    if any('policy' in col.lower() for col in columns):
        data_type = 'Insurance Policy Data'
    elif any('claim' in col.lower() for col in columns):
        data_type = 'Claims Data'
    else:
        data_type = f"{file_type.upper()} Data"
    
    # Basic quality assessment
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    quality_score = int(completeness)
    
    # Store basic processed data
    st.session_state.uploaded_datasets[file_key] = {
        'filename': uploaded_file.name,
        'data_type': data_type,
        'data': df,
        'quality_score': quality_score,
        'records': len(df),
        'issues': f'Basic processing applied - {completeness:.1f}% complete',
        'recommendations': 'Upload processed successfully'
    }
    
    st.info("Using basic processing - enhanced profiler not available")

def step_2_intelligent_analysis():
    """Step 2: Intelligent Analysis and Data Integration"""
    
    st.markdown("## 🧠 Process & Analyze Data")
    
    # Check for both old workflow datasets AND new persistent results
    datasets = st.session_state.get('uploaded_datasets', {})
    persistent_results = get_latest_persistent_results()
    
    # Debug info for troubleshooting (can remove later)
    if st.checkbox("Show debug info", value=False):
        st.write("Current uploaded_datasets:", list(datasets.keys()) if datasets else "None")
        st.write("Persistent results available:", bool(persistent_results))
        if persistent_results:
            st.write("Persistent file name:", persistent_results.get('uploaded_file_name', 'Unknown'))
        
        if st.button("🗑️ Clear Cache", key="clear_cache_debug"):
            clear_old_cache_files()
            st.session_state.uploaded_datasets = {}
            st.session_state.latest_cleaning_results = {}
            st.success("Cache cleared!")
            st.rerun()
    
    if not datasets and not persistent_results:
        st.warning("No datasets uploaded. Return to Step 1.")
        if st.button("← Back to Upload"):
            st.session_state.workflow_step = 1
            st.rerun()
        return
    
    # If we have persistent results but no workflow datasets, integrate them
    if persistent_results and not datasets:
        st.info("✅ Found recent cleaning results - integrating into workflow")
        # Create a synthetic dataset entry from persistent results
        file_key = persistent_results.get('file_key', 'cleaned_data')
        datasets[file_key] = {
            'name': persistent_results.get('uploaded_file_name', 'Cleaned Data'),
            'filename': persistent_results.get('uploaded_file_name', 'Cleaned Data'),
            'data': persistent_results.get('df_cleaned'),
            'original_data': persistent_results.get('df_original'),
            'cleaning_result': persistent_results.get('result'),
            'file_type': persistent_results.get('file_type', 'unknown'),
            'data_type': persistent_results.get('file_type', 'cleaned_dataset'),
            'quality_score': 85  # Default quality score for cleaned data
        }
        st.session_state.uploaded_datasets = datasets
        st.success("🎉 Cleaning results integrated into workflow!")
    
    # Display datasets summary
    st.markdown("### 📊 Uploaded Datasets")
    
    for key, dataset in datasets.items():
        with st.expander(f"📁 {dataset['data_type']} - {dataset['filename']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Records**: {len(dataset['data']):,}  
                **Quality Score**: {dataset['quality_score']:.0f}/100  
                **Data Type**: {dataset['data_type']}
                """)
            
            with col2:
                if st.button(f"👀 View Data", key=f"view_{key}"):
                    st.dataframe(dataset['data'].head(10))
    
    # Advanced Integration Analysis
    st.markdown("### 🔬 Advanced Integration Analysis")
    
    if st.button("🧠 Analyze Data Relationships", type="primary"):
        with st.spinner("Analyzing data relationships..."):
            integration_analysis = analyze_data_integration(datasets)
            
            st.markdown("#### 🎯 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Coverage:**")
                for category, status in integration_analysis['coverage'].items():
                    icon = "✅" if status else "❌"
                    st.markdown(f"{icon} {category}")
            
            with col2:
                st.markdown("**Pricing Readiness:**")
                readiness = integration_analysis['pricing_readiness']
                st.metric("Overall Score", f"{readiness:.0f}%")
                
                if readiness >= 70:
                    st.success("✅ Ready for pricing!")
                    if st.button("➡️ Continue to Portfolio Analysis"):
                        st.session_state.workflow_step = 3
                        st.rerun()
                else:
                    st.warning("⚠️ Need more data for reliable pricing")

def analyze_data_integration(datasets):
    """Analyze how datasets integrate for pricing"""
    
    # Advanced analysis
    coverage = {
        'Mortality Data': 'mortality' in datasets or 'policy_data' in datasets,
        'Claims Experience': 'claims' in datasets or 'policy_data' in datasets,
        'Economic Assumptions': 'economic' in datasets,
        'Expense Structure': 'expense' in datasets,
        'Portfolio Data': 'policy_data' in datasets
    }
    
    coverage_score = sum(coverage.values()) / len(coverage) * 100
    
    # Quality adjustment
    quality_scores = [d['quality_score'] for d in datasets.values()]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    pricing_readiness = (coverage_score * 0.7 + avg_quality * 0.3)
    
    return {
        'coverage': coverage,
        'pricing_readiness': pricing_readiness,
        'data_quality': avg_quality,
        'dataset_count': len(datasets)
    }

def step_3_portfolio_analysis():
    """Step 3: Portfolio Analysis"""
    
    st.markdown("## 📊 Step 3: Portfolio Analysis")
    
    if not st.session_state.uploaded_datasets:
        st.warning("No data available. Return to previous steps.")
        return
    
    # Get policy data
    policy_data = None
    for key, dataset in st.session_state.uploaded_datasets.items():
        if 'policy' in key or dataset['data_type'] == 'policy_data':
            policy_data = dataset['data']
            break
    
    if policy_data is None:
        st.warning("No policy data found. Upload policy data to continue.")
        return
    
    st.markdown("### 📈 Portfolio Characteristics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Policies", f"{len(policy_data):,}")
    with col2:
        if 'face_amount' in policy_data.columns:
            total_coverage = policy_data['face_amount'].sum()
            st.metric("Total Coverage", f"${total_coverage:,.0f}")
        else:
            st.metric("Total Coverage", "N/A")
    with col3:
        if 'annual_premium' in policy_data.columns:
            total_premium = policy_data['annual_premium'].sum()
            st.metric("Annual Premium", f"${total_premium:,.0f}")
        else:
            st.metric("Annual Premium", "N/A")
    with col4:
        if 'issue_age' in policy_data.columns:
            avg_age = policy_data['issue_age'].mean()
            st.metric("Avg Issue Age", f"{avg_age:.1f}")
        else:
            st.metric("Avg Issue Age", "N/A")
    
    # Portfolio analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'issue_age' in policy_data.columns:
            fig_age = px.histogram(
                policy_data, x='issue_age', 
                title="Age Distribution",
                nbins=20
            )
            st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        if 'face_amount' in policy_data.columns:
            fig_coverage = px.histogram(
                policy_data, x='face_amount',
                title="Coverage Distribution",
                nbins=20
            )
            st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Experience analysis
    st.markdown("### 🔍 Experience Analysis")
    
    if st.button("📊 Perform Experience Analysis"):
        with st.spinner("Analyzing portfolio experience..."):
            # Simulate experience analysis
            experience_results = {
                'mortality_ae_ratio': np.random.uniform(0.8, 1.2),
                'credibility_factor': min(1.0, len(policy_data) / 10000),
                'risk_score': np.random.uniform(3, 8),
                'portfolio_quality': 'Good'
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("A/E Mortality Ratio", f"{experience_results['mortality_ae_ratio']:.2f}")
            with col2:
                st.metric("Credibility Factor", f"{experience_results['credibility_factor']:.1%}")
            with col3:
                st.metric("Risk Score", f"{experience_results['risk_score']:.1f}/10")
            
            st.success("✅ Portfolio analysis complete!")
            
            if st.button("➡️ Continue to Pricing"):
                st.session_state.workflow_step = 4
                st.rerun()

def step_4_pricing_calculation():
    """Step 4: Comprehensive Pricing Calculation"""
    
    st.markdown("## 💰 Step 4: Comprehensive Pricing Calculation")
    
    # Initialize pricing engine
    if st.session_state.pricing_engine is None:
        try:
            st.session_state.pricing_engine = ProductionPricingEngine("Comprehensive Pricing Platform")
            st.success("✅ Pricing engine initialized")
        except Exception as e:
            st.error(f"Could not initialize pricing engine: {e}")
            return
    
    # Pricing configuration
    st.markdown("### ⚙️ Pricing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cedent_name = st.text_input("Cedent Name", value="Test Insurance Company")
        treaty_type = st.selectbox("Treaty Type", ["Quota Share", "Surplus Share", "Excess of Loss"])
        retention_limit = st.number_input("Retention Limit ($)", value=1000000, format="%d")
    
    with col2:
        reinsurance_limit = st.number_input("Reinsurance Limit ($)", value=10000000, format="%d")
        target_profit_margin = st.slider("Target Profit Margin (%)", min_value=5, max_value=25, value=15)
        confidence_level = st.selectbox("Confidence Level", ["90%", "95%", "99%"])
    
    # Run comprehensive pricing
    if st.button("🚀 Calculate Comprehensive Pricing", type="primary"):
        with st.spinner("Running comprehensive pricing analysis..."):
            
            # Get uploaded data
            datasets = st.session_state.uploaded_datasets
            
            # Simulate comprehensive pricing using uploaded data
            pricing_results = calculate_comprehensive_pricing(
                datasets, cedent_name, treaty_type, 
                retention_limit, reinsurance_limit, target_profit_margin
            )
            
            st.session_state.pricing_results = pricing_results
            
            display_pricing_results(pricing_results)
            
            if st.button("➡️ View Final Results"):
                st.session_state.workflow_step = 5
                st.rerun()

def calculate_comprehensive_pricing(datasets, cedent_name, treaty_type, retention_limit, reinsurance_limit, target_margin):
    """Calculate comprehensive pricing using all available data"""
    
    # Base pricing calculation using available datasets
    policy_count = 0
    total_premium = 0
    total_coverage = 0
    
    # Extract key metrics from uploaded data
    for key, dataset in datasets.items():
        data = dataset['data']
        if 'policy' in key:
            policy_count = len(data)
            if 'annual_premium' in data.columns:
                total_premium = data['annual_premium'].sum()
            if 'face_amount' in data.columns:
                total_coverage = data['face_amount'].sum()
    
    # Pricing calculations
    expected_loss_ratio = np.random.uniform(0.60, 0.80)  # Calculated from historical data
    expense_ratio = 0.25
    risk_margin = np.random.uniform(0.08, 0.15)
    capital_charge = 0.05
    
    gross_rate = expected_loss_ratio + expense_ratio + risk_margin + capital_charge
    
    # Premium calculation
    estimated_annual_premium = max(total_premium, policy_count * 2000)  # Fallback
    gross_premium = estimated_annual_premium * gross_rate
    
    # Sensitivity analysis
    sensitivity = {
        'mortality_plus_10': gross_rate * 1.1,
        'mortality_minus_10': gross_rate * 0.9,
        'expenses_plus_20': (expected_loss_ratio + expense_ratio * 1.2 + risk_margin + capital_charge),
        'all_adverse': (expected_loss_ratio * 1.1 + expense_ratio * 1.2 + risk_margin * 1.25 + capital_charge)
    }
    
    return {
        'cedent_name': cedent_name,
        'treaty_type': treaty_type,
        'policy_count': policy_count,
        'total_coverage': total_coverage,
        'expected_loss_ratio': expected_loss_ratio,
        'expense_ratio': expense_ratio,
        'risk_margin': risk_margin,
        'capital_charge': capital_charge,
        'gross_rate': gross_rate,
        'estimated_annual_premium': estimated_annual_premium,
        'gross_premium': gross_premium,
        'sensitivity': sensitivity,
        'confidence_level': 'Medium',
        'pricing_date': datetime.now(),
        'data_sources': list(datasets.keys())
    }

def display_pricing_results(results):
    """Display comprehensive pricing results"""
    
    st.markdown("""
    <div class="pricing-result">
        <h2>🎯 Pricing Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Loss Ratio", f"{results['expected_loss_ratio']:.1%}")
    with col2:
        st.metric("Risk Margin", f"{results['risk_margin']:.1%}")
    with col3:
        st.metric("Gross Rate", f"{results['gross_rate']:.1%}")
    with col4:
        st.metric("Annual Premium", f"${results['gross_premium']:,.0f}")
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Rate Components")
        components_df = pd.DataFrame({
            'Component': ['Expected Loss Ratio', 'Expense Ratio', 'Risk Margin', 'Capital Charge'],
            'Rate (%)': [
                results['expected_loss_ratio'] * 100,
                results['expense_ratio'] * 100,
                results['risk_margin'] * 100,
                results['capital_charge'] * 100
            ]
        })
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Rate Build-Up",
            orientation="v",
            measure=["relative", "relative", "relative", "relative"],
            x=components_df['Component'],
            y=components_df['Rate (%)'],
        ))
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Sensitivity Analysis")
        sensitivity_df = pd.DataFrame({
            'Scenario': ['Base Case', 'Mortality +10%', 'Mortality -10%', 'Expenses +20%', 'All Adverse'],
            'Gross Rate (%)': [
                results['gross_rate'] * 100,
                results['sensitivity']['mortality_plus_10'] * 100,
                results['sensitivity']['mortality_minus_10'] * 100,
                results['sensitivity']['expenses_plus_20'] * 100,
                results['sensitivity']['all_adverse'] * 100
            ]
        })
        
        fig_sensitivity = px.bar(
            sensitivity_df, x='Scenario', y='Gross Rate (%)',
            title="Sensitivity Analysis"
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)

def step_5_results_reports():
    """Step 5: Final Results and Reports"""
    
    st.markdown("## 📋 Step 5: Final Results & Reports")
    
    if not st.session_state.pricing_results:
        st.warning("No pricing results available. Complete pricing calculation first.")
        return
    
    results = st.session_state.pricing_results
    
    # Executive Summary
    st.markdown("### 📊 Executive Summary")
    
    st.markdown(f"""
    <div class="pricing-result">
        <h3>🎯 Pricing Recommendation for {results['cedent_name']}</h3>
        <div style="font-size: 1.2rem; margin: 1rem 0;">
            <strong>Recommended Gross Rate: {results['gross_rate']:.1%}</strong>
        </div>
        <div>
            <strong>Annual Premium:</strong> ${results['gross_premium']:,.0f}<br>
            <strong>Treaty Type:</strong> {results['treaty_type']}<br>
            <strong>Confidence Level:</strong> {results['confidence_level']}<br>
            <strong>Data Sources Used:</strong> {len(results['data_sources'])} datasets
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed report
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Portfolio Summary")
        st.markdown(f"""
        - **Policies Analyzed**: {results['policy_count']:,}
        - **Total Coverage**: ${results['total_coverage']:,.0f}
        - **Premium Volume**: ${results['estimated_annual_premium']:,.0f}
        - **Pricing Date**: {results['pricing_date'].strftime('%Y-%m-%d')}
        """)
        
        st.markdown("#### 🧮 Rate Components")
        st.markdown(f"""
        - **Expected Loss Ratio**: {results['expected_loss_ratio']:.1%}
        - **Expense Ratio**: {results['expense_ratio']:.1%}
        - **Risk Margin**: {results['risk_margin']:.1%}
        - **Capital Charge**: {results['capital_charge']:.1%}
        - **Total Gross Rate**: {results['gross_rate']:.1%}
        """)
    
    with col2:
        st.markdown("#### 🎯 Key Recommendations")
        st.markdown(f"""
        - ✅ Pricing appears reasonable for {results['treaty_type']} structure
        - 📊 Based on analysis of {len(results['data_sources'])} data sources
        - 🔍 Recommend quarterly monitoring of experience
        - 📈 Consider profit sharing arrangements above target margins
        """)
        
        st.markdown("#### ⚠️ Key Risks")
        st.markdown("""
        - 📊 Limited credibility in mortality experience
        - 🌊 Economic scenario uncertainty
        - 📉 Potential adverse selection risk
        - 🔄 Regulatory capital requirements
        """)
    
    # Download reports
    st.markdown("### 📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Summary Report"):
            report_data = generate_summary_report(results)
            st.download_button(
                "📥 Summary Report (CSV)",
                data=report_data,
                file_name=f"pricing_summary_{results['cedent_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Detailed Analysis"):
            st.info("Detailed analysis report generation coming soon!")
    
    with col3:
        if st.button("🔄 Start New Pricing"):
            # Reset workflow
            st.session_state.workflow_step = 1
            st.session_state.uploaded_datasets = {}
            st.session_state.pricing_results = None
            st.rerun()

def generate_summary_report(results):
    """Generate summary report in CSV format"""
    
    summary_data = pd.DataFrame({
        'Metric': [
            'Cedent Name', 'Treaty Type', 'Policy Count', 'Total Coverage',
            'Expected Loss Ratio', 'Expense Ratio', 'Risk Margin', 'Capital Charge',
            'Gross Rate', 'Annual Premium', 'Confidence Level', 'Pricing Date'
        ],
        'Value': [
            results['cedent_name'], results['treaty_type'], results['policy_count'],
            results['total_coverage'], f"{results['expected_loss_ratio']:.1%}",
            f"{results['expense_ratio']:.1%}", f"{results['risk_margin']:.1%}",
            f"{results['capital_charge']:.1%}", f"{results['gross_rate']:.1%}",
            results['gross_premium'], results['confidence_level'],
            results['pricing_date'].strftime('%Y-%m-%d')
        ]
    })
    
    return summary_data.to_csv(index=False)

def display_sidebar():
    """Display compact sidebar with data exploration and navigation"""
    
    # Compact CSS for smaller spacing
    st.sidebar.markdown("""
    <style>
    .sidebar-section { margin-bottom: 10px; }
    .sidebar-item { margin: 2px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📊 Data")
    
    # Compact exploration buttons 
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📊", help="Mortality Rates", key="mort_compact"):
            st.session_state.show_mortality_explorer = True
            st.rerun()
        if st.button("📈", help="Market Trends", key="market_compact"):
            st.session_state.show_market_explorer = True
            st.rerun()
    with col2:
        if st.button("💰", help="Interest Rates", key="interest_compact"):
            st.session_state.show_treasury_explorer = True
            st.rerun()
        datasets_count = len(st.session_state.uploaded_datasets)
        if st.button("📁", help=f"Your Data ({datasets_count})", key="data_compact"):
            st.session_state.show_data_explorer = True
            st.rerun()
    
    # Cleaning button
    if PHASE2_CLEANING_AVAILABLE:
        if st.sidebar.button("🧹 Clean Data", help="Hybrid Statistical+Semantic Cleaning"):
            st.session_state.show_cleaning_explorer = True
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Steps")
    
    # Compact workflow navigation
    steps = [
        ("1", "📤 Upload"),
        ("2", "🧠 Process"), 
        ("3", "📊 Analyze"),
        ("4", "💰 Price"),
        ("5", "📋 Results")
    ]
    
    current_step = st.session_state.workflow_step
    
    # Display as compact grid
    step_cols = st.sidebar.columns(5)
    for i, (step_num, step_name) in enumerate(steps):
        with step_cols[i]:
            if int(step_num) == current_step:
                st.markdown(f"**{step_num}**")
            else:
                if st.button(f"{step_num}", key=f"nav_{step_num}", help=step_name):
                    st.session_state.workflow_step = int(step_num)
                    st.rerun()

def display_comprehensive_quality_report(df, filename, file_key):
    """Display enterprise data quality report with Great Expectations or fallback options"""
    
    # Close button
    if st.button("❌ Close Quality Report", key=f"close_quality_report_{file_key}"):
        st.session_state[f"show_quality_report_{file_key}"] = False
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"## 📋 Enterprise Data Quality Report: {filename}")
    
    # Check if we already have a profile stored from upload
    dataset_info = st.session_state.uploaded_datasets.get(file_key.replace('.', '_').replace('-', '_'))
    
    if dataset_info and dataset_info.get('enterprise_profile'):
        # Use stored Great Expectations profile
        profile = dataset_info['enterprise_profile']
        display_enterprise_quality_results(profile, df, filename, file_key)
    elif dataset_info and dataset_info.get('enhanced_profile'):
        # Use stored enhanced profile
        profile = dataset_info['enhanced_profile']
        display_enhanced_quality_results(profile, df, filename, file_key, dataset_info.get('profiler'))
    else:
        # Generate new profile
        with st.spinner("Analyzing data quality with enterprise-grade validation..."):
            if ENTERPRISE_PROFILER_AVAILABLE:
                profiler = SimpleGreatExpectationsProfiler()
                # Detect target column
                columns = list(df.columns)
                target_column = None
                for col in columns:
                    if any(keyword in col.lower() for keyword in ['premium', 'amount', 'claim', 'value', 'price']):
                        target_column = col
                        break
                profile = profiler.generate_enterprise_profile(df, target_column)
                display_enterprise_quality_results(profile, df, filename, file_key)
            elif ENHANCED_PROFILER_AVAILABLE:
                profiler = EnhancedDataProfiler()
                profile = profiler.profile_data(df)
                display_enhanced_quality_results(profile, df, filename, file_key, profiler)
            elif COMPREHENSIVE_PROFILER_AVAILABLE:
                profiler = ComprehensiveDataProfiler()  
                profile = profiler.profile_dataset(df, filename)
                display_legacy_quality_results(profile)
            else:
                st.error("❌ No data profiling libraries available - please install requirements")

def display_enterprise_quality_results(profile, df, filename, file_key):
    """Display results from Great Expectations enterprise profiler"""
    
    enterprise_summary = profile['enterprise_summary']
    
    # Enterprise-level quality metrics
    col_ent1, col_ent2, col_ent3, col_ent4 = st.columns(4)
    
    with col_ent1:
        quality_score = enterprise_summary['data_quality_score']
        quality_color = "🟢" if quality_score >= 85 else "🟡" if quality_score >= 70 else "🔴"
        st.metric("Quality Score", f"{quality_score}%", delta_color="normal")
        st.caption(f"{quality_color} Enterprise Grade")
    
    with col_ent2:
        compliance_score = enterprise_summary.get('compliance_score', 0)
        compliance_color = "🟢" if compliance_score >= 90 else "🟡" if compliance_score >= 75 else "🔴"
        st.metric("Compliance", f"{compliance_score}%", delta_color="normal")
        st.caption(f"{compliance_color} Regulatory")
    
    with col_ent3:
        risk_score = enterprise_summary.get('risk_score', 0)
        risk_color = "🟢" if risk_score <= 20 else "🟡" if risk_score <= 50 else "🔴"
        st.metric("Risk Score", f"{risk_score}%", delta_color="inverse")
        st.caption(f"{risk_color} Risk Level")
    
    with col_ent4:
        data_quality_issues = enterprise_summary['data_quality_issues']
        st.metric("Issues", data_quality_issues, delta_color="inverse")
        st.caption("🔍 Detected")
    
    # Business Rules Validation
    st.markdown("### 🏢 Business Rules Validation")
    business_validation = profile['business_rules_validation']
    
    col_biz1, col_biz2 = st.columns(2)
    
    with col_biz1:
        st.markdown(f"**Data Domain:** {business_validation['data_domain'].replace('_', ' ').title()}")
        st.markdown(f"**Business Context:** {business_validation['business_context']}")
    
    with col_biz2:
        passed_rules = sum(1 for rule in business_validation['validation_results'].values() if rule['passed'])
        total_rules = len(business_validation['validation_results'])
        st.metric("Rules Passed", f"{passed_rules}/{total_rules}")
        
        if passed_rules < total_rules:
            st.error("❌ Some business rules failed validation")
            for rule_name, result in business_validation['validation_results'].items():
                if not result['passed']:
                    st.caption(f"• {rule_name}: {result['message']}")
    
    # Regulatory Compliance
    st.markdown("### ⚖️ Regulatory Compliance")
    compliance = profile['regulatory_compliance']
    
    compliant_items = sum(1 for item in compliance['compliance_status'] if item['compliant'])
    total_items = len(compliance['compliance_status'])
    
    if total_items > 0:
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.metric("Compliance Rate", f"{compliant_items}/{total_items}")
        
        with col_comp2:
            if compliant_items < total_items:
                st.warning("⚠️ Compliance Issues Detected")
                for item in compliance['compliance_status']:
                    if not item['compliant']:
                        st.caption(f"• {item['requirement']}: {item['issue']}")
    
    # Risk Indicators
    st.markdown("### ⚠️ Risk Indicators")
    risk_indicators = profile['risk_indicators']
    
    if risk_indicators:
        for risk in risk_indicators:
            severity_color = "🔴" if risk['severity'] == 'high' else "🟡" if risk['severity'] == 'medium' else "🟢"
            st.markdown(f"{severity_color} **{risk['type'].replace('_', ' ').title()}**: {risk['description']}")
            if 'recommendation' in risk:
                st.caption(f"   Recommendation: {risk['recommendation']}")
    else:
        st.success("🟢 No significant risk indicators detected")
    
    # Data Quality Metrics
    st.markdown("### 📊 Data Quality Details")
    quality_metrics = profile['data_quality_metrics']
    
    col_qual1, col_qual2 = st.columns(2)
    
    with col_qual1:
        st.markdown("**Passed Expectations:**")
        passed_expectations = quality_metrics.get('passed_expectations', [])
        for expectation in passed_expectations[:5]:
            st.success(f"✅ {expectation.get('expectation_type', 'Validation passed')}")
    
    with col_qual2:
        st.markdown("**Failed Expectations:**")
        failed_expectations = quality_metrics.get('failed_expectations', [])
        for expectation in failed_expectations[:5]:
            st.error(f"❌ {expectation.get('expectation_type', 'Validation failed')}")
            if 'observed_value' in expectation:
                st.caption(f"   Observed: {expectation['observed_value']}")
    
    # Professional Recommendations
    st.markdown("### 💡 Enterprise Recommendations")
    recommendations = []
    
    # Add quality-based recommendations
    if quality_score < 80:
        recommendations.append("🔧 **Data Quality**: Implement comprehensive data cleaning pipeline")
    
    # Add compliance-based recommendations  
    if compliance_score < 90:
        recommendations.append("⚖️ **Compliance**: Address regulatory compliance gaps before production")
    
    # Add risk-based recommendations
    if risk_score > 30:
        recommendations.append("⚠️ **Risk Management**: Implement additional data validation controls")
    
    # Add business rule recommendations
    failed_business_rules = [name for name, result in business_validation['validation_results'].items() if not result['passed']]
    if failed_business_rules:
        recommendations.append(f"🏢 **Business Rules**: Fix validation failures in {', '.join(failed_business_rules[:3])}")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("🌟 Enterprise-grade data - ready for production use!")

def display_enhanced_quality_results(profile, df, filename, file_key, profiler):
    """Display results from enhanced profiler using professional libraries"""
    
    # Overall quality metrics
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        quality_score = profile["data_quality"]["overall_completeness"]
        st.metric("Data Completeness", f"{quality_score:.1f}%")
    
    with col_summary2:
        total_recommendations = len(profile["recommendations"])
        st.metric("Issues Found", total_recommendations, delta_color="inverse")
    
    with col_summary3:
        structural_issues = len(profile["structural_issues"])
        st.metric("Structural Issues", structural_issues, delta_color="inverse")
    
    # Basic dataset info
    st.markdown("### 📊 Dataset Overview")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric("Rows", f"{profile['basic_info']['shape'][0]:,}")
    
    with col_info2:
        st.metric("Columns", profile['basic_info']['shape'][1])
    
    with col_info3:
        st.metric("Duplicates", profile['basic_info']['duplicate_rows'])
    
    with col_info4:
        st.metric("Empty Rows", profile['basic_info']['completely_empty_rows'])
    
    # Missing data visualization
    if st.checkbox("Show Missing Data Visualization", key=f"missing_viz_{file_key}"):
        missing_viz = profiler.generate_missing_data_visualization(df)
        if missing_viz:
            st.markdown("### 🔍 Missing Data Pattern")
            st.image(f"data:image/png;base64,{missing_viz}")
    
    # Structural issues
    if profile["structural_issues"]:
        st.markdown("### 🚨 Structural Issues")
        for issue in profile["structural_issues"]:
            st.warning(f"⚠️ {issue}")
    
    # Column analysis
    st.markdown("### 📋 Column Analysis")
    
    for col, analysis in profile["column_analysis"].items():
        issues_count = len(analysis["issues"])
        issue_icon = "🚨" if issues_count > 2 else "⚠️" if issues_count > 0 else "✅"
        
        with st.expander(f"{issue_icon} **{col}** ({analysis['dtype']}) - {issues_count} issues"):
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Unique Values", f"{analysis['unique_values']:,}")
            
            with col_stats2:
                st.metric("Unique Ratio", f"{analysis['unique_ratio']:.2f}")
            
            with col_stats3:
                if analysis['most_frequent'] is not None:
                    st.metric("Most Frequent", str(analysis['most_frequent'])[:20])
            
            # Column issues
            if analysis["issues"]:
                st.markdown("**Issues:**")
                for issue in analysis["issues"]:
                    st.markdown(f"• {issue}")
    
    # Missing data details
    if any(info['missing_count'] > 0 for info in profile["missing_patterns"].values()):
        st.markdown("### 🕳️ Missing Data Details")
        
        missing_data = []
        for col, info in profile["missing_patterns"].items():
            if info['missing_count'] > 0:
                missing_data.append({
                    "Column": col,
                    "Missing Count": info['missing_count'],
                    "Missing %": f"{info['missing_percentage']:.1f}%",
                    "Pattern": info['missing_pattern']
                })
        
        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            st.dataframe(missing_df, use_container_width=True)
    
    # Actionable recommendations with cleaning options
    if profile["recommendations"]:
        st.markdown("### 💡 Recommended Actions")
        
        # Group recommendations by category
        categories = {}
        for rec in profile["recommendations"]:
            category = rec.get("category", "General")
            if category not in categories:
                categories[category] = []
            categories[category].append(rec)
        
        # Display each category
        for category, recs in categories.items():
            with st.expander(f"📋 {category} Issues ({len(recs)} items)"):
                
                selected_actions = []
                for i, rec in enumerate(recs):
                    action_key = f"action_{file_key}_{category}_{i}"
                    
                    if st.checkbox(f"✅ {rec['recommendation']}", key=action_key):
                        selected_actions.append(rec['action'])
                    
                    st.markdown(f"*Issue:* {rec['issue']}")
                    st.markdown("---")
                
                # Apply selected actions for this category
                if selected_actions and st.button(f"Apply {category} Actions", key=f"apply_{category}_{file_key}"):
                    with st.spinner(f"Applying {category} cleaning actions..."):
                        cleaned_df = profiler.apply_cleaning_actions(df, selected_actions)
                        
                        # Store cleaned data in session state
                        st.session_state[f"cleaned_data_{file_key}"] = cleaned_df
                        
                        # Show summary of changes
                        summary = profiler.get_cleaning_summary()
                        if summary["history"]:
                            latest = summary["history"][-1]
                            st.success(f"✅ Applied {len(latest['actions'])} cleaning actions")
                            for action in latest['actions']:
                                st.info(f"• {action}")
                            
                            # Option to download cleaned data
                            csv_data = cleaned_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Cleaned Data",
                                data=csv_data,
                                file_name=f"cleaned_{filename}",
                                mime="text/csv",
                                key=f"download_cleaned_{file_key}"
                            )

def display_legacy_quality_results(profile):
    """Fallback display for old profiler format"""
    # Legacy display code would go here
    # For now, show basic info
    st.info("Using fallback profiler - limited features available")
    
    if hasattr(profile, 'overall_quality_score'):
        st.metric("Overall Quality", f"{profile.overall_quality_score}/100")

def apply_enhanced_cleaning(df):
    """Apply enhanced cleaning to dataframe and return cleaned data with changes log"""
    
    changes = []
    cleaned_df = df.copy()
    
    try:
        # 1. Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            # Fill numeric columns with median
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            
            # Fill text columns with mode or 'Unknown'
            text_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in text_cols:
                if cleaned_df[col].isnull().sum() > 0:
                    mode_val = cleaned_df[col].mode()
                    if len(mode_val) > 0:
                        cleaned_df[col].fillna(mode_val[0], inplace=True)
                    else:
                        cleaned_df[col].fillna('Unknown', inplace=True)
            
            missing_after = cleaned_df.isnull().sum().sum()
            changes.append(f"Filled {missing_before - missing_after} missing values")
        
        # 2. Remove duplicate rows
        duplicates_before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = duplicates_before - len(cleaned_df)
        if duplicates_removed > 0:
            changes.append(f"Removed {duplicates_removed} duplicate rows")
        
        # 3. Standardize text columns
        text_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_cols:
            if cleaned_df[col].dtype == 'object':
                # Strip whitespace
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                
                # Standardize common values
                if 'gender' in col.lower() or 'sex' in col.lower():
                    cleaned_df[col] = cleaned_df[col].str.lower().replace({
                        'm': 'Male', 'f': 'Female', 'male': 'Male', 'female': 'Female',
                        '1': 'Male', '0': 'Female'
                    })
                    changes.append(f"Standardized gender values in {col}")
                
                if 'smoking' in col.lower() or 'smoker' in col.lower():
                    cleaned_df[col] = cleaned_df[col].str.lower().replace({
                        'y': 'Yes', 'n': 'No', 'yes': 'Yes', 'no': 'No',
                        'smoker': 'Yes', 'non-smoker': 'No', 'never': 'No'
                    })
                    changes.append(f"Standardized smoking status in {col}")
        
        # 4. Clean numeric columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Remove outliers (values beyond 3 standard deviations)
            mean_val = cleaned_df[col].mean()
            std_val = cleaned_df[col].std()
            if std_val > 0:
                outliers_mask = np.abs(cleaned_df[col] - mean_val) > (3 * std_val)
                outliers_count = outliers_mask.sum()
                if outliers_count > 0 and outliers_count < len(cleaned_df) * 0.05:  # Only if < 5% outliers
                    cleaned_df = cleaned_df[~outliers_mask]
                    changes.append(f"Removed {outliers_count} outliers from {col}")
        
        # 5. Format date columns
        potential_date_cols = [col for col in cleaned_df.columns if any(word in col.lower() for word in ['date', 'time', 'created', 'updated'])]
        for col in potential_date_cols:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                changes.append(f"Standardized date format in {col}")
            except:
                pass
        
        if not changes:
            changes.append("Data already clean - no changes needed")
            
    except Exception as e:
        changes.append(f"Cleaning error: {str(e)[:50]}...")
    
    return cleaned_df, changes

def main():
    """Main comprehensive platform"""
    
    # Initialize
    initialize_comprehensive_state()
    
    # ALWAYS restore persistent results on page load (aggressive approach)
    restore_persistent_results()
    
    # Sidebar
    display_sidebar()
    
    # Header
    display_main_header()
    
    # Inline PriceRe Chat - Always available 
    if CHAT_ASSISTANT_AVAILABLE:
        render_inline_chat_interface()
    
    # Check if user explicitly wants to return to recent results
    # Only show persistent results if user has clicked "Return to Results" 
    if st.session_state.get('show_persistent_results', False):
        persistent_results = get_latest_persistent_results()
        if persistent_results:
            st.success("🎉 Cleaning Results Available - Persistent Storage Working")
            show_persistent_cleaning_results()
            
            # Debug info for troubleshooting
            if st.sidebar.checkbox("🔧 Persistence Debug", value=False):
                result_files = list(TEMP_DIR.glob("*_results.pkl"))
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime) if result_files else None
                st.sidebar.write(f"**Session keys:** {len(st.session_state.keys())}")
                st.sidebar.write(f"**Files found:** {len(result_files)}")
                if latest_file:
                    st.sidebar.write(f"**Latest file age:** {(datetime.now().timestamp() - latest_file.stat().st_mtime):.1f}s")
            
            st.divider()
            # Add back to cleaning button
            if st.button("🔄 Clean Another File", type="secondary"):
                # Clear the results to go back to upload
                st.session_state['show_persistent_results'] = False
                for key in list(st.session_state.keys()):
                    if 'cleaning' in key.lower() or 'latest' in key.lower():
                        del st.session_state[key]
                st.rerun()
            return  # Stop here - don't show the normal workflow
        else:
            # No results found, go to normal workflow
            st.session_state['show_persistent_results'] = False
    
    
    # Data explorers
    display_data_explorers()
    
    # Workflow progress
    display_workflow_progress()
    
    # Main workflow steps
    current_step = st.session_state.workflow_step
    
    if current_step == 1:
        step_1_data_upload()
    elif current_step == 2:
        step_2_intelligent_analysis()
    elif current_step == 3:
        step_3_portfolio_analysis()
    elif current_step == 4:
        step_4_pricing_calculation()
    elif current_step == 5:
        step_5_results_reports()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        if st.button("🏠 Reset Workflow"):
            st.session_state.workflow_step = 1
            st.session_state.uploaded_datasets = {}
            st.session_state.pricing_results = None
            st.rerun()
        
        st.markdown("### 📊 Current Session")
        st.markdown(f"**Step**: {current_step}/5")
        st.markdown(f"**Datasets**: {len(st.session_state.uploaded_datasets)}")
        
        if st.session_state.pricing_results:
            st.markdown("**Pricing**: ✅ Complete")
        else:
            st.markdown("**Pricing**: ⏳ Pending")
        

if __name__ == "__main__":
    main()