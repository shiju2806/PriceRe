#!/usr/bin/env python3
"""
Simple test interface for Phase 2 cleaning system
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.cleaning.hybrid_detector import create_hybrid_detector
    import polars as pl
    st.success("✅ Phase 2 cleaning system imported successfully")
    
    st.title("🧹 Phase 2 Cleaning Test")
    
    # Generate test data
    if st.button("🎯 Generate Test Data"):
        messy_data = {
            'Policy': ['Insurance Report Q4 2023', '', 'POL001', 'POL002', '', 'TOTAL: 2'],
            'Premium': ['Premium Amount', None, '25000', '18500', '', '43500'],
            'Age': [None, None, '35', '42', None, None],
            'Status': ['Status', '', 'Active', 'Active', '', 'SUMMARY']
        }
        
        st.session_state.test_data = pl.DataFrame(messy_data)
        st.success(f"✅ Generated messy data: {st.session_state.test_data.shape}")
    
    if 'test_data' in st.session_state:
        st.subheader("Original Messy Data")
        st.dataframe(st.session_state.test_data)
        
        if st.button("🧹 CLEAN DATA NOW"):
            with st.spinner("Cleaning..."):
                # Create detector
                detector = create_hybrid_detector('balanced')
                
                # Get results
                result = detector.detect_junk_rows(st.session_state.test_data, return_detailed=True)
                
                st.success(f"🎉 Detected {len(result.junk_row_indices)} junk rows!")
                st.write(f"Junk indices: {result.junk_row_indices}")
                st.write(f"Layers used: {result.layers_used}")
                
                # Clean data
                if result.junk_row_indices:
                    mask = pl.Series(range(st.session_state.test_data.shape[0])).is_in(result.junk_row_indices).not_()
                    clean_df = st.session_state.test_data.filter(mask)
                    
                    st.subheader("🗑️ REMOVED JUNK ROWS:")
                    junk_df = st.session_state.test_data.filter(pl.Series(range(st.session_state.test_data.shape[0])).is_in(result.junk_row_indices))
                    st.dataframe(junk_df)
                    
                    st.subheader("✨ CLEAN DATA:")
                    st.dataframe(clean_df)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original", st.session_state.test_data.shape[0])
                    with col2:
                        st.metric("Clean", clean_df.shape[0])
                    with col3:
                        st.metric("Removed", len(result.junk_row_indices))
                else:
                    st.warning("No junk detected")
    
except ImportError as e:
    st.error(f"❌ Import failed: {e}")
    st.info("Make sure you're running from the correct directory with src/cleaning modules available")

except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.code(traceback.format_exc())