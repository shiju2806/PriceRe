#!/usr/bin/env python3
"""
Fixed Upload Platform - Simplified version to resolve 403 errors
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import enhanced profiler
try:
    from src.data_cleaning.enhanced_profiler import EnhancedDataProfiler
    ENHANCED_PROFILER_AVAILABLE = True
except ImportError as e:
    ENHANCED_PROFILER_AVAILABLE = False
    st.error(f"Enhanced profiler not available: {e}")

# Page config
st.set_page_config(
    page_title="PriceRe - Fixed Upload",
    page_icon="💰", 
    layout="wide"
)

# Simple session state initialization
def init_simple_state():
    """Minimal session state to avoid conflicts"""
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = {}

# Initialize
init_simple_state()

st.title("💰 PriceRe - Enhanced Data Upload")
st.markdown("**Fixed version to resolve file upload issues**")

# File upload section
st.markdown("## 📁 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=['csv', 'xlsx', 'xls', 'json', 'txt'],
    key="fixed_upload"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")
    
    # File info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Size", f"{uploaded_file.size:,} bytes")
    with col2:
        st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
    with col3:
        file_key = uploaded_file.name.replace('.', '_').replace('-', '_')
        processed = file_key in st.session_state.uploaded_data
        st.metric("Status", "✅ Processed" if processed else "⏳ Ready")
    
    # Process button
    col_process, col_download = st.columns(2)
    
    with col_process:
        if st.button("🔍 Analyze with Enhanced Profiler", key="analyze_btn", use_container_width=True):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    # Load file
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_type in ['xlsx', 'xls']:
                        df = pd.read_excel(uploaded_file)
                    elif file_type == 'json':
                        import json
                        uploaded_file.seek(0)
                        data = json.load(uploaded_file)
                        df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
                    else:
                        st.error(f"Unsupported file type: {file_type}")
                        st.stop()
                    
                    st.success(f"📊 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Store data
                    st.session_state.uploaded_data[file_key] = {
                        'dataframe': df,
                        'filename': uploaded_file.name,
                        'analysis': None,
                        'cleaned_data': None
                    }
                    
                    # Show preview
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Enhanced profiling if available
                    if ENHANCED_PROFILER_AVAILABLE:
                        st.markdown("### 🔍 Enhanced Data Analysis")
                        
                        profiler = EnhancedDataProfiler()
                        profile = profiler.profile_data(df)
                        
                        # Store analysis
                        st.session_state.uploaded_data[file_key]['analysis'] = profile
                        st.session_state.uploaded_data[file_key]['profiler'] = profiler
                        
                        # Quality metrics
                        col_q1, col_q2, col_q3 = st.columns(3)
                        
                        with col_q1:
                            completeness = profile['data_quality']['overall_completeness']
                            st.metric("Data Completeness", f"{completeness:.1f}%")
                        
                        with col_q2:
                            issues = len(profile['recommendations'])
                            st.metric("Issues Found", issues, delta_color="inverse")
                        
                        with col_q3:
                            structural = len(profile['structural_issues'])
                            st.metric("Structural Issues", structural, delta_color="inverse")
                        
                        # Recommendations
                        if profile['recommendations']:
                            st.markdown("### 💡 Data Quality Recommendations")
                            
                            selected_actions = []
                            
                            for i, rec in enumerate(profile['recommendations']):
                                col_check, col_desc = st.columns([0.1, 0.9])
                                
                                with col_check:
                                    if st.checkbox("", key=f"rec_{i}"):
                                        selected_actions.append(rec['action'])
                                
                                with col_desc:
                                    st.markdown(f"**{rec['recommendation']}**")
                                    st.caption(f"Issue: {rec['issue']}")
                            
                            # Apply cleaning
                            if selected_actions:
                                if st.button("🧹 Apply Selected Cleaning Actions", use_container_width=True):
                                    with st.spinner("Applying cleaning actions..."):
                                        cleaned_df = profiler.apply_cleaning_actions(df, selected_actions)
                                        
                                        # Store cleaned data
                                        st.session_state.uploaded_data[file_key]['cleaned_data'] = cleaned_df
                                        
                                        # Show results
                                        st.success(f"✅ Cleaning completed: {df.shape} → {cleaned_df.shape}")
                                        
                                        # Show changes
                                        summary = profiler.get_cleaning_summary()
                                        if summary['history']:
                                            latest = summary['history'][-1]
                                            st.markdown("**Applied Actions:**")
                                            for action in latest['actions']:
                                                st.info(f"• {action}")
                                        
                                        # Show cleaned preview
                                        st.markdown("### 🧽 Cleaned Data Preview")
                                        st.dataframe(cleaned_df.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    
    # Download section
    with col_download:
        if file_key in st.session_state.uploaded_data:
            data_info = st.session_state.uploaded_data[file_key]
            
            if data_info.get('cleaned_data') is not None:
                # Download cleaned data
                cleaned_csv = data_info['cleaned_data'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Cleaned Data",
                    data=cleaned_csv,
                    file_name=f"cleaned_{uploaded_file.name.replace('.xlsx', '.csv')}",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Clean data first to enable download")

else:
    st.info("👆 Upload a file to get started")
    
    # Show sample files
    st.markdown("### 📋 Sample Files Available")
    st.markdown("You can test with these files in the test_data folder:")
    
    sample_files = [
        "integration_test.csv - Insurance policy data with mixed formatting",
        "test_messy_data.xlsx - Excel file with header rows and mixed formats",
    ]
    
    for sample in sample_files:
        st.markdown(f"• {sample}")

# Debug section
with st.expander("🔧 Debug Information"):
    st.write("**Session State Keys:**")
    st.write(list(st.session_state.keys()))
    
    st.write("**Available Libraries:**")
    st.write(f"Enhanced Profiler: {ENHANCED_PROFILER_AVAILABLE}")
    
    st.write("**Uploaded Data:**")
    st.write(f"Files in memory: {len(st.session_state.uploaded_data)}")