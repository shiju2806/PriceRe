"""
Navigation Components
Step progression and workflow management for PriceRe platform
"""

import streamlit as st
from typing import Tuple, List


def render_workflow_progress() -> Tuple[int, str]:
    """Render workflow progress indicator and return current step info"""
    
    current_step = st.session_state.get('workflow_step', 1)
    
    # Step definitions
    steps = [
        {"num": 1, "name": "Upload Data", "icon": "📁", "desc": "Upload and validate files"},
        {"num": 2, "name": "Process & Analyze", "icon": "🧠", "desc": "Clean and analyze data"},
        {"num": 3, "name": "Actuarial Analysis", "icon": "📊", "desc": "Comprehensive actuarial workspace"},
        {"num": 4, "name": "Pricing Calculation", "icon": "💰", "desc": "Calculate reinsurance pricing"},
        {"num": 5, "name": "Final Results", "icon": "📋", "desc": "Review and export results"}
    ]
    
    # Use native Streamlit columns for better compatibility
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 2px solid #dee2e6;
    ">
    </div>
    """, unsafe_allow_html=True)
    
    # Create 5 equal columns
    cols = st.columns(5)
    
    for i, step in enumerate(steps):
        with cols[i]:
            if step["num"] < current_step:
                # Completed step
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                    padding: 15px 10px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 24px; font-weight: bold;">✅ {step["num"]}</div>
                    <div style="font-size: 14px; font-weight: 600; margin-top: 5px;">{step["name"]}</div>
                    <div style="font-size: 12px; opacity: 0.9; margin-top: 3px;">{step["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)
            elif step["num"] == current_step:
                # Active step
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px 10px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    transform: scale(1.05);
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 24px; font-weight: bold;">{step["icon"]} {step["num"]}</div>
                    <div style="font-size: 14px; font-weight: 600; margin-top: 5px;">{step["name"]}</div>
                    <div style="font-size: 12px; opacity: 0.9; margin-top: 3px;">{step["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Pending step
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    color: #6c757d;
                    border: 2px dashed #dee2e6;
                    padding: 15px 10px;
                    border-radius: 12px;
                    text-align: center;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 24px; font-weight: bold;">{step["icon"]} {step["num"]}</div>
                    <div style="font-size: 14px; font-weight: 600; margin-top: 5px;">{step["name"]}</div>
                    <div style="font-size: 12px; opacity: 0.9; margin-top: 3px;">{step["desc"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Current step info
    current_step_info = next((s for s in steps if s["num"] == current_step), steps[0])
    
    return current_step, current_step_info["name"]


def render_step_navigation(show_back: bool = True, show_next: bool = True, 
                         next_disabled: bool = False, back_disabled: bool = False):
    """Render step navigation buttons"""
    
    current_step = st.session_state.get('workflow_step', 1)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if show_back and current_step > 1:
            if st.button("⬅️ Previous Step", disabled=back_disabled, use_container_width=True):
                st.session_state.workflow_step = max(1, current_step - 1)
                st.rerun()
    
    with col3:
        if show_next and current_step < 5:
            if st.button("Next Step ➡️", type="primary", disabled=next_disabled, use_container_width=True):
                st.session_state.workflow_step = min(5, current_step + 1)
                st.rerun()


def render_step_header(step_num: int, title: str, description: str = ""):
    """Render consistent step header"""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2.5em;">Step {step_num}</h1>
        <h2 style="margin: 10px 0 0 0; font-weight: 300;">{title}</h2>
        {f'<p style="margin: 10px 0 0 0; opacity: 0.9;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)


def render_quick_navigation():
    """Render quick navigation sidebar with data explorers"""
    
    current_step = st.session_state.get('workflow_step', 1)
    
    with st.sidebar:
        # Data Explorers Section
        st.markdown("## 📊 Data Explorers")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Mortality", key="mort_explorer", use_container_width=True):
                st.session_state.show_mortality_explorer = True
                st.rerun()
            if st.button("💹 Market", key="market_explorer", use_container_width=True):
                st.session_state.show_market_explorer = True
                st.rerun()
        with col2:
            if st.button("💰 Treasury", key="treasury_explorer", use_container_width=True):
                st.session_state.show_treasury_explorer = True
                st.rerun()
            if st.button("📁 Your Data", key="data_explorer", use_container_width=True):
                st.session_state.show_data_explorer = True
                st.rerun()
        
        st.markdown("---")
        
        # Navigation Section
        st.markdown("## 🚀 Quick Navigation")
        
        steps = [
            {"num": 1, "name": "Upload", "icon": "📁"},
            {"num": 2, "name": "Process", "icon": "🧠"},
            {"num": 3, "name": "Analyze", "icon": "📊"},
            {"num": 4, "name": "Price", "icon": "💰"},
            {"num": 5, "name": "Results", "icon": "📋"}
        ]
        
        # Create 3x2 grid for 5 steps (3 top, 2 bottom)
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)
        
        cols = [col1, col2, col3, col4, col5]
        
        for i, step in enumerate(steps):
            with cols[i]:
                if step["num"] == current_step:
                    # Current step - special styling
                    st.markdown(f"**{step['icon']} {step['name']}**")
                    st.caption("← Current")
                elif step["num"] < current_step:
                    # Completed steps - clickable with checkmark
                    if st.button(f"✅ {step['name']}", key=f"nav_{step['num']}", use_container_width=True):
                        st.session_state.workflow_step = step["num"]
                        st.rerun()
                else:
                    # Future steps - check if we can proceed
                    can_proceed, missing_reqs = can_proceed_to_step(step["num"])
                    if can_proceed:
                        # Allow navigation to next step if requirements are met
                        if st.button(f"{step['icon']} {step['name']}", key=f"nav_{step['num']}", use_container_width=True):
                            st.session_state.workflow_step = step["num"]
                            st.rerun()
                    else:
                        # Show as disabled
                        st.button(f"⏳ {step['name']}", key=f"nav_disabled_{step['num']}", disabled=True, use_container_width=True)


def get_step_requirements(step_num: int) -> List[str]:
    """Get requirements for a specific step"""
    
    requirements = {
        1: [],
        2: ["At least one file uploaded"],
        3: ["Data processed and cleaned"],
        4: ["Actuarial analysis completed"],
        5: ["Pricing calculation completed"]
    }
    
    return requirements.get(step_num, [])


def can_proceed_to_step(target_step: int) -> Tuple[bool, List[str]]:
    """Check if user can proceed to target step"""
    
    current_step = st.session_state.get('workflow_step', 1)
    missing_requirements = []
    
    # Check basic progression
    if target_step > current_step + 1:
        missing_requirements.append(f"Complete steps {current_step} through {target_step - 1} first")
    
    # Check specific requirements
    if target_step >= 2:
        if not st.session_state.get('uploaded_datasets', {}):
            missing_requirements.append("Upload data files in Step 1")
    
    if target_step >= 4:
        if not st.session_state.get('pricing_engine'):
            missing_requirements.append("Complete actuarial analysis in Step 3")
    
    if target_step >= 5:
        if not st.session_state.get('pricing_results'):
            missing_requirements.append("Complete pricing calculation in Step 4")
    
    return len(missing_requirements) == 0, missing_requirements


def render_workflow_summary():
    """Render summary of workflow progress"""
    
    current_step = st.session_state.get('workflow_step', 1)
    uploaded_count = len(st.session_state.get('uploaded_datasets', {}))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 Workflow Summary")
    
    # Progress metrics
    st.sidebar.metric("Current Step", f"{current_step}/5")
    st.sidebar.metric("Files Uploaded", uploaded_count)
    
    # Quick stats
    if st.session_state.get('pricing_results'):
        results = st.session_state.pricing_results
        st.sidebar.metric("Gross Premium", f"${results.get('gross_premium', 0):,.0f}")
    
    # Completion status
    completion = (current_step - 1) / 4 * 100
    st.sidebar.progress(completion / 100)
    st.sidebar.caption(f"Workflow {completion:.0f}% complete")