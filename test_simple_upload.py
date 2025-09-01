#!/usr/bin/env python3
"""
Simple file upload test to isolate the 403 error
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Upload Test",
    layout="wide"
)

st.title("🧪 File Upload Test")

st.markdown("""
This is a minimal test to isolate the 403 error issue.
""")

# Simple file uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'json', 'txt'],
    key="simple_test_upload"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size:,} bytes")
    st.write(f"File type: {uploaded_file.type}")
    
    # Try to process the file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write("📊 CSV Data Preview:")
            st.dataframe(df.head())
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.write("📊 Excel Data Preview:")
            st.dataframe(df.head())
            
        elif uploaded_file.name.endswith('.json'):
            import json
            data = json.load(uploaded_file)
            st.write("📊 JSON Data Preview:")
            st.json(data)
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Please upload a file to test")

# Show Streamlit version and configuration
st.sidebar.markdown("### 🔧 Debug Info")
st.sidebar.write(f"Streamlit version: {st.__version__}")

# Test session state
if 'test_counter' not in st.session_state:
    st.session_state.test_counter = 0

if st.sidebar.button("Test Session State"):
    st.session_state.test_counter += 1

st.sidebar.write(f"Session counter: {st.session_state.test_counter}")