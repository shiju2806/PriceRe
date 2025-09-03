"""
Session State Management
Centralized configuration for Streamlit session state variables
"""

import streamlit as st
from pathlib import Path
import json


def initialize_comprehensive_state():
    """Initialize session state with minimal complexity to avoid 403 errors"""
    
    # Core workflow state only - minimal to prevent conflicts
    defaults = {
        'workflow_step': 1,
        'uploaded_datasets': {},
        'pricing_submission': None,
        'pricing_results': None,
        'pricing_engine': None,
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


def initialize_chat_state():
    """Initialize chat-related session state variables"""
    chat_defaults = {
        'chat_open': False,
        'chat_history': [],
        'chat_initialized': False,
        'professional_chat_history': [],
        'sidebar_chat_history': [],
        'global_chat_active': False
    }
    
    for key, default_value in chat_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_latest_persistent_results():
    """Get latest persistent results from file system"""
    try:
        results_dir = Path(__file__).parent.parent.parent / "results" / "persistent"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if result_files:
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                with open(latest_file) as f:
                    return json.load(f)
    except Exception:
        pass
    return None


def save_persistent_results(results, filename: str):
    """Save results to persistent storage"""
    try:
        results_dir = Path(__file__).parent.parent.parent / "results" / "persistent"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / f"{filename}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Could not save persistent results: {e}")


def clear_session_state():
    """Clear all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def get_session_info():
    """Get current session state information for debugging"""
    return {
        'keys': list(st.session_state.keys()),
        'workflow_step': st.session_state.get('workflow_step', 'Not set'),
        'uploaded_datasets_count': len(st.session_state.get('uploaded_datasets', {})),
        'engines_initialized': st.session_state.get('engines_initialized', False)
    }