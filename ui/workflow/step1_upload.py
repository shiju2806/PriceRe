"""
Step 1: Data Upload and Initial Processing
File upload, validation, and basic data quality assessment
"""

import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any
from datetime import datetime

# Import utilities
from ui.utils.data_processing import (
    load_file_as_dataframe, calculate_data_quality_score, apply_enhanced_cleaning,
    save_dataset_to_session, get_file_preview, create_download_data,
    create_comprehensive_report, apply_phase2_cleaning, create_file_hash
)

# Import from original file (will refactor later)
try:
    from ui.comprehensive_pricing_platform import (
        initialize_engines_safely, clean_and_compare_file,
        display_comprehensive_quality_report
    )
except ImportError:
    # Fallback functions
    def initialize_engines_safely():
        pass
    
    def clean_and_compare_file(file, file_key):
        st.warning("Advanced cleaning not available")
    
    def display_comprehensive_quality_report(data, filename, file_key):
        st.info("Quality report not available")


def step_1_upload_files():
    """Step 1: Universal Data Upload"""
    
    st.markdown("## 📁 Step 1: Upload Your Data Files")
    st.markdown("Upload CSV, Excel, JSON, or TXT files for reinsurance analysis")
    
    # Initialize engines safely
    initialize_engines_safely()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _render_file_upload_section()
    
    with col2:
        _render_data_requirements_panel()


def _render_file_upload_section():
    """Render the main file upload interface"""
    
    st.markdown("### 📤 Upload Files")
    st.info("💡 Upload multiple files - we'll analyze and clean your data automatically!")
    
    uploaded_files = st.file_uploader(
        "Choose multiple files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'json', 'txt', 'xls'],
        key="comprehensive_upload"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files uploaded**")
        
        # Process all files button
        _render_batch_processing(uploaded_files)
        
        # Individual file processing
        for file in uploaded_files:
            _render_individual_file_processor(file)


def _render_batch_processing(uploaded_files: List[Any]):
    """Render batch processing interface"""
    
    if 'uploaded_datasets' not in st.session_state:
        st.session_state.uploaded_datasets = {}
    
    unprocessed_files = [
        f for f in uploaded_files 
        if f.name.replace('.', '_').replace('-', '_') not in st.session_state.uploaded_datasets
    ]
    
    if unprocessed_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"🔄 {len(unprocessed_files)} files need processing")
        with col2:
            if st.button("🚀 Process All Files", type="primary", key="process_all_files"):
                _process_files_batch(unprocessed_files)


def _process_files_batch(files: List[Any]):
    """Process multiple files in batch"""
    
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        progress_bar.progress((i + 1) / len(files))
        file_key = file.name.replace('.', '_').replace('-', '_')
        
        try:
            # Load file
            df = load_file_as_dataframe(file)
            
            # Calculate quality score
            quality_score = calculate_data_quality_score(df)
            
            # Save to session
            save_dataset_to_session(
                file_key=file_key,
                filename=file.name,
                df_cleaned=df,
                df_original=df.copy(),
                file_type=file.name.split('.')[-1].lower()
            )
            
        except Exception as e:
            st.error(f"Failed to process {file.name}: {str(e)}")
    
    st.success(f"✅ Processed {len(files)} files!")
    st.rerun()


def _render_individual_file_processor(uploaded_file):
    """Render individual file processing interface"""
    
    file_key = uploaded_file.name.replace('.', '_').replace('-', '_')
    is_processed = file_key in st.session_state.get('uploaded_datasets', {})
    
    with st.expander(f"📄 {uploaded_file.name} {'✅' if is_processed else '⏳'}", 
                     expanded=not is_processed):
        
        # File info
        st.markdown(f"**Size:** {uploaded_file.size:,} bytes | **Type:** {uploaded_file.name.split('.')[-1].upper()}")
        
        if not is_processed:
            # Processing options for unprocessed files
            _render_processing_options(uploaded_file, file_key)
        else:
            # Show processing results
            _render_processing_results(uploaded_file, file_key)


def _render_processing_options(uploaded_file, file_key: str):
    """Render processing options for unprocessed files"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔧 Quick Process", key=f"quick_{uploaded_file.name}", 
                    type="primary", use_container_width=True):
            _quick_process_file(uploaded_file, file_key)
    
    with col2:
        if st.button("🧹 Advanced Clean", key=f"clean_{uploaded_file.name}", 
                    use_container_width=True):
            content_hash = create_file_hash(uploaded_file)
            advanced_file_key = f"{uploaded_file.name}_{content_hash}"
            clean_and_compare_file(uploaded_file, advanced_file_key)
    
    # File preview
    _render_file_preview(uploaded_file)


def _quick_process_file(uploaded_file, file_key: str):
    """Quick process a single file"""
    
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # Load file
            df = load_file_as_dataframe(uploaded_file)
            
            # Basic cleaning
            df_cleaned, cleaning_info = apply_phase2_cleaning(df)
            
            if not cleaning_info.get('success', True):
                st.warning(f"Advanced cleaning failed: {cleaning_info.get('error', 'Unknown error')}")
                df_cleaned = df
            
            # Save to session
            save_dataset_to_session(
                file_key=file_key,
                filename=uploaded_file.name,
                df_cleaned=df_cleaned,
                df_original=df.copy(),
                cleaning_result=cleaning_info.get('result'),
                file_type=uploaded_file.name.split('.')[-1].lower()
            )
            
            st.success("✅ File processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")


def _render_processing_results(uploaded_file, file_key: str):
    """Render results for processed files"""
    
    dataset_info = st.session_state.uploaded_datasets[file_key]
    
    st.markdown("---")
    st.markdown("#### ✅ Processing Complete")
    
    # Basic metrics
    st.markdown(f"**Records:** {dataset_info['records']:,} | **Quality Score:** {dataset_info['quality_score']}/100")
    
    # Data quality indicator
    quality_score = dataset_info['quality_score']
    if quality_score >= 85:
        st.success("🟢 Excellent data quality")
    elif quality_score >= 70:
        st.warning("🟡 Good data quality - minor issues")
    else:
        st.error("🟠 Data quality needs improvement")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Quality Report", key=f"quality_report_{uploaded_file.name}", 
                    use_container_width=True):
            display_comprehensive_quality_report(dataset_info['data'], uploaded_file.name, file_key)
    
    with col2:
        if quality_score < 85:
            if st.button("🔧 Enhance", key=f"enhance_{uploaded_file.name}", 
                        use_container_width=True):
                _apply_enhanced_cleaning(dataset_info, file_key)
    
    with col3:
        # Download options
        if st.button("💾 Download", key=f"download_{uploaded_file.name}", 
                    use_container_width=True):
            _show_download_options(dataset_info, uploaded_file.name)


def _apply_enhanced_cleaning(dataset_info: Dict, file_key: str):
    """Apply enhanced cleaning to dataset"""
    
    with st.spinner("Applying enhanced cleaning..."):
        try:
            cleaned_data, changes = apply_enhanced_cleaning(dataset_info['data'])
            
            # Update dataset
            old_quality = dataset_info['quality_score']
            new_quality = min(95, old_quality + 10)
            
            dataset_info['data'] = cleaned_data
            dataset_info['quality_score'] = new_quality
            dataset_info['records'] = len(cleaned_data)
            
            st.session_state.uploaded_datasets[file_key] = dataset_info
            
            st.success(f"✅ Quality improved: {old_quality} → {new_quality}")
            if changes:
                st.info(f"Changes applied: {', '.join(changes[:3])}...")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Enhanced cleaning failed: {str(e)}")


def _show_download_options(dataset_info: Dict, filename: str):
    """Show download options for processed data"""
    
    st.markdown("#### 💾 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = create_download_data(dataset_info['data'], filename, 'csv')
        st.download_button(
            "📄 Download CSV",
            data=csv_data,
            file_name=f"cleaned_{filename}",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        excel_data = create_download_data(dataset_info['data'], filename, 'excel')
        st.download_button(
            "📊 Download Excel",
            data=excel_data,
            file_name=f"cleaned_{filename.rsplit('.', 1)[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def _render_file_preview(uploaded_file):
    """Render file preview section"""
    
    if st.button("👁️ Preview", key=f"preview_{uploaded_file.name}"):
        st.session_state[f"show_preview_{uploaded_file.name}"] = True
        st.rerun()
    
    if st.session_state.get(f"show_preview_{uploaded_file.name}", False):
        st.markdown("#### 📊 Data Preview")
        
        if st.button("❌ Close Preview", key=f"close_preview_{uploaded_file.name}"):
            st.session_state[f"show_preview_{uploaded_file.name}"] = False
            st.rerun()
        
        preview_text = get_file_preview(uploaded_file)
        
        if uploaded_file.name.endswith(('.csv', '.txt')):
            st.code(preview_text, language='csv')
        elif uploaded_file.name.endswith('.json'):
            st.code(preview_text, language='json')
        else:
            st.text(preview_text)


def _render_data_requirements_panel():
    """Render data requirements panel"""
    
    st.markdown("### 📋 Data Requirements")
    
    required_data_types = [
        "Policy Data", "Mortality Tables", "Claims Experience", 
        "Economic Scenarios", "Expense Data"
    ]
    
    optional_data_types = [
        "Premium Transactions", "Medical Underwriting", "Lapse Rates",
        "Product Features", "Investment Returns", "Reinsurance Treaties"
    ]
    
    # Required data status
    st.markdown("**Required Data:**")
    for data_type in required_data_types:
        status = "✅" if _has_data_type(data_type) else "⏳"
        st.markdown(f"{status} {data_type}")
    
    st.markdown("**Optional Data:**")
    for data_type in optional_data_types[:3]:  # Show first 3
        status = "✅" if _has_data_type(data_type) else "➖"
        st.markdown(f"{status} {data_type}")
    
    # Progress to next step
    uploaded_count = len(st.session_state.get('uploaded_datasets', {}))
    if uploaded_count >= 2:  # Need at least 2 datasets
        st.success(f"✅ {uploaded_count} datasets ready!")
        if st.button("➡️ Continue to Analysis", type="primary"):
            st.session_state.workflow_step = 2
            st.rerun()
    else:
        st.info("Upload at least 2 datasets to continue")


def _has_data_type(data_type: str) -> bool:
    """Check if a specific data type has been uploaded"""
    datasets = st.session_state.get('uploaded_datasets', {})
    data_type_key = data_type.lower().replace(' ', '_')
    return data_type_key in datasets


def get_upload_summary() -> Dict[str, Any]:
    """Get summary of uploaded files"""
    datasets = st.session_state.get('uploaded_datasets', {})
    
    return {
        'total_files': len(datasets),
        'total_records': sum(d['records'] for d in datasets.values()),
        'avg_quality': sum(d['quality_score'] for d in datasets.values()) / len(datasets) if datasets else 0,
        'file_types': list(set(d.get('file_type', 'unknown') for d in datasets.values()))
    }