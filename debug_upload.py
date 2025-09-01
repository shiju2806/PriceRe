#!/usr/bin/env python3
"""
Debug the 403 upload issue
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="Debug Upload", layout="wide")

st.title("🐛 Debug Upload Issue")

# Test imports
st.markdown("### Import Test")
try:
    from src.data_cleaning.enhanced_profiler import EnhancedDataProfiler
    st.success("✅ Enhanced profiler imported successfully")
    ENHANCED_AVAILABLE = True
except Exception as e:
    st.error(f"❌ Enhanced profiler import failed: {e}")
    ENHANCED_AVAILABLE = False

# Simple upload test
st.markdown("### File Upload Test")
uploaded_file = st.file_uploader("Upload a file", type=['xlsx', 'csv'])

if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Test processing
    if st.button("Test Processing"):
        try:
            # Load file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {df.shape}")
            st.dataframe(df.head())
            
            # Test enhanced profiler
            if ENHANCED_AVAILABLE:
                st.markdown("### Enhanced Profiler Test")
                with st.spinner("Testing enhanced profiler..."):
                    profiler = EnhancedDataProfiler()
                    profile = profiler.profile_data(df)
                    
                    st.success("✅ Enhanced profiler worked!")
                    st.json({
                        "completeness": profile['data_quality']['overall_completeness'],
                        "recommendations": len(profile['recommendations']),
                        "structural_issues": len(profile['structural_issues'])
                    })
            
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Session state debug
with st.sidebar:
    st.markdown("### Debug Info")
    st.write("Session state keys:", len(st.session_state.keys()))
    if st.button("Clear Session State"):
        st.session_state.clear()
        st.rerun()