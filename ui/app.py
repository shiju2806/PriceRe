"""
PriceRe Smart Reinsurance Pricing Platform
Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration and components
from ui.config.session_state import initialize_comprehensive_state, initialize_chat_state
from ui.components.chat_sidebar import render_professional_floating_chat
from ui.components.navigation import render_workflow_progress, render_quick_navigation, render_workflow_summary
from ui.components.data_explorers import render_all_data_explorers

# Import all refactored workflow steps
from ui.workflow.step1_upload_complete import step_1_data_upload
from ui.workflow.step2_analysis import step_2_intelligent_analysis
from ui.workflow.step3_actuarial import step_3_portfolio_analysis
from ui.workflow.step4_pricing import step_4_pricing_calculation
from ui.workflow.step5_results import step_5_results_reports

# Import remaining functions from original file
from ui.comprehensive_pricing_platform import initialize_engines_safely


def configure_page():
    """Configure Streamlit page settings and styling"""
    
    st.set_page_config(
        page_title="PriceRe Smart Reinsurance Pricing",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    /* Make sidebar significantly wider for better chat */
    section[data-testid="stSidebar"] {
        width: 450px !important;
        min-width: 450px !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Reduce top padding to move header closer to top */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Professional metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e0e6ed;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: #f8f9fa;
        border-radius: 10px;
        border: 2px solid transparent;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def render_main_header():
    """Render the main application header"""
    
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em;">📊 PriceRe</h1>
        <p style="margin: 5px 0 0 0; font-size: 1.2em; opacity: 0.9;">
            Smart Reinsurance Pricing
        </p>
    </div>
    """, unsafe_allow_html=True)


def route_workflow_step():
    """Route to the appropriate workflow step"""
    
    current_step = st.session_state.get('workflow_step', 1)
    
    if current_step == 1:
        step_1_data_upload()  # Now using the complete refactored module
    elif current_step == 2:
        step_2_intelligent_analysis()
    elif current_step == 3:
        step_3_portfolio_analysis()
    elif current_step == 4:
        step_4_pricing_calculation()  # Using the refactored module
    elif current_step == 5:
        step_5_results_reports()
    else:
        st.error(f"Invalid workflow step: {current_step}")
        st.session_state.workflow_step = 1


def main():
    """Main application entry point"""
    
    # Configure page
    configure_page()
    
    # Initialize session state
    initialize_comprehensive_state()
    initialize_chat_state()
    
    # Initialize engines safely
    initialize_engines_safely()
    
    # Render main header
    render_main_header()
    
    # Check if any data explorer is active
    any_explorer_active = (
        st.session_state.get('show_mortality_explorer', False) or
        st.session_state.get('show_treasury_explorer', False) or  
        st.session_state.get('show_market_explorer', False) or
        st.session_state.get('show_data_explorer', False)
    )
    
    if any_explorer_active:
        # Show data explorer in full width when active
        render_all_data_explorers()
    else:
        # Workflow progress indicator - full width like PriceRe header
        current_step, step_name = render_workflow_progress()
        
        # Step header - full width like PriceRe header  
        if current_step == 1:
            st.markdown("## 📤 Upload Your Data")
        elif current_step == 2:
            st.markdown("## 🧠 Process & Analyze Data")
        elif current_step == 3:
            st.markdown("## 📊 Actuarial Analysis")
            st.markdown("*Professional actuarial analytics with comprehensive insights*") 
        elif current_step == 4:
            st.markdown("## 💰 Pricing Calculation")
        elif current_step == 5:
            st.markdown("## 📋 Final Results")
        
        # Handle Step 3 with full width
        if current_step == 3:
            # Step 3 gets full width like data explorers
            route_workflow_step()
        else:
            # Other steps use column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Route to appropriate step
                route_workflow_step()
            
            with col2:
                # Quick navigation
                render_quick_navigation()
                
                # Show data requirements for Step 1
                if current_step == 1:
                    from ui.workflow.step1_upload_complete import _render_data_requirements_panel
                    st.markdown("---")
                    _render_data_requirements_panel()
    
    # Render professional floating chat widget (always visible across all platform sections)
    render_professional_floating_chat()


if __name__ == "__main__":
    main()