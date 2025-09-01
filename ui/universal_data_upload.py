"""
Universal Data Upload Interface
AI-powered frontend for uploading and processing any data format
Supports multiple files, various formats, and intelligent data understanding
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any
import io

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data_processing.ollama_data_processor import IntelligentDataProcessor, test_ollama_connection
    from src.data_generation.enterprise_data_generator import EnterpriseDataGenerator, GenerationConfig
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import processors: {e}")
    PROCESSORS_AVAILABLE = False

# Import our new hybrid cleaning system
try:
    from src.cleaning.hybrid_detector import create_hybrid_detector, quick_hybrid_clean
    from src.cleaning.statistical_analyzer import create_statistical_detector
    from src.cleaning.data_sources import read_any_source
    import polars as pl
    CLEANING_SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import cleaning system: {e}")
    CLEANING_SYSTEM_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Universal Data Upload Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.upload-zone {
    border: 2px dashed #1f77b4;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    margin: 1rem 0;
}

.success-box {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #28a745;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.info-card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}

.metric-card {
    text-align: center;
    padding: 1rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin: 0.5rem;
}

.data-preview {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #dee2e6;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    
    if 'ollama_status' not in st.session_state:
        st.session_state.ollama_status = None
    
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = {}

def check_ollama_status():
    """Check if Ollama is available"""
    
    if st.session_state.ollama_status is None:
        with st.spinner("Checking Ollama connection..."):
            st.session_state.ollama_status = test_ollama_connection() if PROCESSORS_AVAILABLE else False
    
    return st.session_state.ollama_status

def display_header():
    """Display main header"""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; margin-bottom: 0.5rem;">🚀 Universal Data Upload Platform</h1>
        <p style="color: #e8f4f8; font-size: 1.2rem; margin: 0;">
            AI-Powered Actuarial Data Processing with Ollama + Llama 3.2
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Ollama status
    ollama_ok = check_ollama_status()
    with col1:
        if ollama_ok:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <h3>🧠 Ollama</h3>
                <p>Connected</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);">
                <h3>🧠 Ollama</h3>
                <p>Disconnected</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Processors status
    with col2:
        if PROCESSORS_AVAILABLE:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <h3>⚙️ Processors</h3>
                <p>Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);">
                <h3>⚙️ Processors</h3>
                <p>Error</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Uploaded files count
    with col3:
        file_count = len(st.session_state.uploaded_files)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📁 Files</h3>
            <p>{file_count} Uploaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Processing results count
    with col4:
        results_count = len(st.session_state.processing_results)
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Processed</h3>
            <p>{results_count} Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cleaning system status
    with col5:
        if CLEANING_SYSTEM_AVAILABLE:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <h3>🧹 Cleaning</h3>
                <p>Phase 2 Ready</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);">
                <h3>🧹 Cleaning</h3>
                <p>Error</p>
            </div>
            """, unsafe_allow_html=True)

def display_data_generator():
    """Display data generation interface"""
    
    st.markdown("## 🏭 Generate Enterprise Test Data")
    
    with st.expander("Generate Large-Scale Datasets", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Generation Configuration")
            
            scale = st.selectbox(
                "Dataset Scale",
                options=['development', 'testing', 'production', 'enterprise'],
                help="Development: 10K records, Testing: 100K, Production: 1M, Enterprise: 10M+"
            )
            
            scale_configs = {
                'development': GenerationConfig(base_size=1000, scale_factor=10),
                'testing': GenerationConfig(base_size=1000, scale_factor=100),
                'production': GenerationConfig(base_size=1000, scale_factor=1000),
                'enterprise': GenerationConfig(base_size=1000, scale_factor=10000)
            }
            
            config = scale_configs[scale]
            expected_records = config.base_size * config.scale_factor
            
            st.info(f"**{scale.title()} Scale**: ~{expected_records:,} policy records")
            
            data_types = st.multiselect(
                "Data Types to Generate",
                options=[
                    'Policies', 'Claims', 'Premiums', 'Mortality Tables', 
                    'Economic Scenarios', 'Underwriting', 'Account Values'
                ],
                default=['Policies', 'Claims', 'Premiums']
            )
            
            messy_data = st.checkbox("Include realistic data quality issues", value=True)
            
        with col2:
            st.markdown("### Generation Preview")
            
            if scale == 'development':
                est_time = "30 seconds"
                file_size = "5-10 MB"
            elif scale == 'testing':
                est_time = "2-3 minutes"
                file_size = "50-100 MB"
            elif scale == 'production':
                est_time = "10-15 minutes"
                file_size = "500 MB - 1 GB"
            else:  # enterprise
                est_time = "1-2 hours"
                file_size = "5-10 GB"
            
            st.markdown(f"""
            **Estimated Generation Time**: {est_time}  
            **Estimated File Size**: {file_size}  
            **Data Quality Issues**: {'Yes' if messy_data else 'No'}  
            **Output Format**: CSV files
            """)
        
        if st.button("🚀 Generate Enterprise Data", type="primary"):
            if PROCESSORS_AVAILABLE:
                with st.spinner(f"Generating {scale} scale datasets..."):
                    try:
                        generator = EnterpriseDataGenerator(config)
                        
                        # Generate just policies for demo (full generation takes too long for UI)
                        if 'Policies' in data_types:
                            policies = generator.generate_enterprise_policies(
                                size=min(10000, expected_records)  # Limit for demo
                            )
                            
                            # Store in session state
                            st.session_state.generated_data['policies'] = policies
                            
                            st.success(f"✅ Generated {len(policies):,} policy records!")
                            
                            # Show preview
                            st.markdown("### Generated Data Preview")
                            st.dataframe(policies.head(10), use_container_width=True)
                            
                            # Download option
                            csv_data = policies.to_csv(index=False)
                            st.download_button(
                                "📥 Download Generated Policies CSV",
                                data=csv_data,
                                file_name=f"generated_policies_{scale}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
            else:
                st.error("Data processors not available. Please check the system status.")

def display_file_upload():
    """Display file upload interface"""
    
    st.markdown("## 📤 Upload Data Files")
    
    # File upload area
    st.markdown("""
    <div class="upload-zone">
        <h3>🎯 Drop Files Here or Browse</h3>
        <p>Supports: CSV, Excel, JSON, TXT, and more formats</p>
        <p>AI will automatically detect and process any data format</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=None,  # Accept any file type
        help="Upload any data files. The AI will automatically detect the format and content."
    )
    
    if uploaded_files:
        st.markdown("### 📋 Uploaded Files")
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)", expanded=i == 0):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # File details
                    st.markdown(f"""
                    **Filename**: {uploaded_file.name}  
                    **Size**: {uploaded_file.size:,} bytes  
                    **Type**: {uploaded_file.type or 'Unknown'}
                    """)
                    
                    # Preview file content
                    if st.checkbox(f"Preview {uploaded_file.name}", key=f"preview_{i}"):
                        try:
                            # Read file content
                            content = uploaded_file.read()
                            
                            # Try to display as text first
                            if uploaded_file.name.lower().endswith(('.csv', '.txt', '.json')):
                                # Show first 1000 characters
                                preview_text = content.decode('utf-8')[:1000]
                                st.text_area("File Preview", preview_text, height=200)
                            else:
                                st.info(f"Binary file - {len(content):,} bytes")
                            
                            # Reset file pointer
                            uploaded_file.seek(0)
                            
                        except Exception as e:
                            st.warning(f"Could not preview file: {e}")
                
                with col2:
                    # Process button
                    if st.button(f"🧠 Process with AI", key=f"process_{i}"):
                        if not check_ollama_status():
                            st.error("Ollama not connected. Please start Ollama first.")
                        elif not PROCESSORS_AVAILABLE:
                            st.error("Data processors not available.")
                        else:
                            process_file_with_ai(uploaded_file, i)

def process_file_with_ai(uploaded_file, file_index: int):
    """Process uploaded file using AI"""
    
    with st.spinner(f"🧠 AI is analyzing {uploaded_file.name}..."):
        try:
            # Get file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Initialize processor
            processor = IntelligentDataProcessor()
            
            # Process the file
            result = processor.process_uploaded_file(
                file_path=uploaded_file.name,
                file_content=file_content
            )
            
            # Store result in session state
            st.session_state.processing_results[uploaded_file.name] = result
            
            # Display results
            display_processing_results(uploaded_file.name, result)
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)

def display_processing_results(filename: str, result):
    """Display AI processing results"""
    
    if result.success:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Processing Successful</h4>
            <p><strong>File</strong>: {filename}</p>
            <p><strong>Data Type Identified</strong>: {result.data_type}</p>
            <p><strong>Quality Score</strong>: {result.quality_score:.0f}/100</p>
            <p><strong>Records</strong>: {len(result.standardized_data):,}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quality breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Analysis")
            
            # Quality gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Data Details")
            
            # Data characteristics
            if result.standardized_data is not None:
                df = result.standardized_data
                
                st.markdown(f"""
                **Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns  
                **Data Type**: {result.data_type}  
                **Missing Values**: {df.isnull().sum().sum():,}  
                **Duplicate Rows**: {df.duplicated().sum():,}
                """)
                
                # Column info
                st.markdown("**Columns**:")
                for col in df.columns[:10]:  # Show first 10 columns
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    st.text(f"  • {col}: {dtype} ({null_count} nulls)")
                
                if len(df.columns) > 10:
                    st.text(f"  ... and {len(df.columns) - 10} more columns")
        
        # Issues and recommendations
        if result.issues or result.recommendations:
            col1, col2 = st.columns(2)
            
            if result.issues:
                with col1:
                    st.markdown("#### ⚠️ Issues Found")
                    for issue in result.issues[:5]:  # Show first 5
                        st.warning(f"• {issue}")
            
            if result.recommendations:
                with col2:
                    st.markdown("#### 💡 Recommendations")
                    for rec in result.recommendations[:5]:  # Show first 5
                        st.info(f"• {rec}")
        
        # Data preview
        if result.standardized_data is not None:
            st.markdown("#### 👀 Data Preview")
            
            # Sample data
            preview_df = result.standardized_data.head(100)
            st.dataframe(preview_df, use_container_width=True)
            
            # Download processed data
            csv_data = result.standardized_data.to_csv(index=False)
            st.download_button(
                "📥 Download Processed Data",
                data=csv_data,
                file_name=f"processed_{filename}",
                mime="text/csv",
                key=f"download_{filename}"
            )
        
    else:
        st.markdown(f"""
        <div class="warning-box">
            <h4>❌ Processing Failed</h4>
            <p><strong>File</strong>: {filename}</p>
            <p><strong>Issues</strong>: {', '.join(result.issues)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if result.recommendations:
            st.markdown("**Recommendations:**")
            for rec in result.recommendations:
                st.info(f"• {rec}")

def display_data_cleaning_interface():
    """Display the Phase 2 Hybrid Data Cleaning Interface"""
    
    st.markdown("## 🧹 Hybrid Data Cleaning System (Phase 2)")
    
    if not CLEANING_SYSTEM_AVAILABLE:
        st.error("❌ Cleaning system not available. Please check the imports.")
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin-bottom: 0.5rem;">✨ Zero Hard-Coded Rules</h3>
        <p style="color: #e8f4f8; margin: 0;">
            🔬 Statistical Content Analysis + 🧠 Semantic Similarity Detection<br/>
            🚀 Progressive Enhancement: Conservative → Aggressive Cleaning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File Upload for Cleaning
    st.markdown("### 📂 Upload File to Clean")
    uploaded_file = st.file_uploader(
        "Choose a messy data file",
        type=['csv', 'xlsx', 'json'],
        help="Upload CSV, Excel, or JSON files with messy data (empty rows, headers, footers, etc.)"
    )
    
    if uploaded_file is not None:
        # Show file info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **📁 File**: {uploaded_file.name}  
            **📏 Size**: {uploaded_file.size:,} bytes  
            **🏷️ Type**: {uploaded_file.type or 'Unknown'}
            """)
        
        with col2:
            # Processing mode selection
            processing_mode = st.selectbox(
                "🎛️ Cleaning Mode",
                options=["fast", "balanced", "comprehensive"],
                index=1,  # Default to balanced
                help="""
                • **Fast**: Statistical analysis only (fastest)
                • **Balanced**: Statistical + Semantic analysis (recommended)  
                • **Comprehensive**: All layers + Future LLM (coming soon)
                """
            )
        
        # Load and preview data
        try:
            # Load data using our universal adapter
            with st.spinner("📖 Loading data..."):
                df = read_any_source(uploaded_file)
                
            st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show data preview
            with st.expander("👀 Preview Original Data", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
            
            # Clean Data Button
            if st.button("🧹 Clean Data", type="primary", use_container_width=True):
                clean_data_with_hybrid_system(df, uploaded_file.name, processing_mode)
                
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
    
    # Demo Section
    st.markdown("---")
    st.markdown("### 🧪 Try Demo with Sample Messy Data")
    
    if st.button("🎯 Generate Sample Messy Data"):
        generate_demo_messy_data()

def generate_demo_messy_data():
    """Generate sample messy data for demonstration"""
    
    with st.spinner("🏭 Generating messy demo data..."):
        # Create realistic messy insurance data
        messy_data = {
            'Policy_Number': [
                'Insurance Company Report',  # Header junk
                '',  # Empty row
                'POL001', 'POL002', '', 'POL003',
                'TOTAL POLICIES: 3',  # Summary row
                '', 'Report generated on 2023-12-01'  # Footer junk
            ],
            'Premium': [
                'Q4 2023',  # Header junk
                None,  # Empty
                '25000', '18500', '', '22000', 
                '65500',  # Total
                '', 'System Export'  # Footer
            ],
            'Age': [
                None, None,
                '35', '42', None, '29',
                None, None, None
            ],
            'Status': [
                'Policy Status', '',
                'Active', 'Active', '', 'Pending',
                'SUMMARY', '', 'End of Report'
            ]
        }
        
        demo_df = pl.DataFrame(messy_data)
        st.session_state.demo_data = demo_df
        
    st.success("✅ Generated sample messy data!")
    
    # Show the messy data
    with st.expander("👀 Sample Messy Data", expanded=True):
        st.dataframe(st.session_state.demo_data, use_container_width=True)
    
    # Clean the demo data
    if st.button("🧹 Clean Demo Data"):
        clean_data_with_hybrid_system(st.session_state.demo_data, "demo_messy_data.csv", "balanced")

def clean_data_with_hybrid_system(df, filename, processing_mode):
    """Clean data using the hybrid system and display results"""
    
    with st.spinner(f"🔬 Cleaning data with {processing_mode} mode..."):
        try:
            # Create hybrid detector
            detector = create_hybrid_detector(processing_mode)
            
            # Get detailed results
            result = detector.detect_junk_rows(df, return_detailed=True)
            
            # Clean the data
            if result.junk_row_indices:
                mask = pl.Series(range(df.shape[0])).is_in(result.junk_row_indices).not_()
                clean_df = df.filter(mask)
            else:
                clean_df = df
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Cleaning Results")
                st.metric("Original Rows", df.shape[0])
                st.metric("Clean Rows", clean_df.shape[0]) 
                st.metric("Junk Rows Removed", len(result.junk_row_indices))
                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                
            with col2:
                st.markdown("### ⚙️ System Details")
                st.metric("Processing Mode", processing_mode.title())
                st.metric("Layers Used", len(result.layers_used))
                st.write("**Layers**: " + ", ".join(result.layers_used))
                if result.early_exit_triggered:
                    st.success("🚀 Early exit triggered (high confidence)")
            
            # Show junk rows that were removed
            if result.junk_row_indices:
                with st.expander(f"🗑️ Removed Junk Rows ({len(result.junk_row_indices)})", expanded=False):
                    junk_df = df.filter(pl.Series(range(df.shape[0])).is_in(result.junk_row_indices))
                    st.dataframe(junk_df, use_container_width=True)
                    
                    # Show explanations for first few junk rows
                    st.markdown("#### 🕵️ Why These Rows Were Removed:")
                    for i, row_idx in enumerate(result.junk_row_indices[:3]):  # Show first 3
                        try:
                            explanation = detector.get_detection_explanation(df, result, row_idx)
                            confidence = explanation.get('final_confidence', 0)
                            layers = explanation.get('layers_used', [])
                            
                            st.markdown(f"""
                            **Row {row_idx}** (Confidence: {confidence:.2f})
                            - Layers used: {', '.join(layers)}
                            - Content: `{' | '.join(str(val) for val in df.row(row_idx))}`
                            """)
                        except Exception as e:
                            st.write(f"Row {row_idx}: Detection triggered (explanation error: {e})")
                    
                    if len(result.junk_row_indices) > 3:
                        st.info(f"... and {len(result.junk_row_indices) - 3} more rows")
            
            # Show clean data
            with st.expander("✨ Clean Data Preview", expanded=True):
                st.dataframe(clean_df.head(20), use_container_width=True)
            
            # Feature importance (if available)
            if hasattr(result, 'statistical_result') and result.statistical_result:
                if result.statistical_result.feature_importance:
                    with st.expander("📈 Feature Importance Analysis", expanded=False):
                        importance = result.statistical_result.feature_importance
                        
                        # Create bar chart
                        import plotly.express as px
                        importance_df = pd.DataFrame(
                            list(importance.items()),
                            columns=['Feature', 'Importance']
                        )
                        
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Statistical Feature Importance for Junk Detection"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Download clean data
            st.markdown("### 💾 Download Clean Data")
            
            # Convert to pandas for download
            clean_pandas_df = clean_df.to_pandas()
            
            # CSV download
            csv_data = clean_pandas_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"clean_{filename}",
                mime="text/csv",
                use_container_width=True
            )
            
            # Excel download  
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                clean_pandas_df.to_excel(writer, sheet_name='Clean Data', index=False)
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"clean_{filename.replace('.csv', '.xlsx')}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.success("🎉 Data cleaning completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Cleaning failed: {e}")
            st.exception(e)

def display_processed_data_dashboard():
    """Display dashboard of processed data"""
    
    if not st.session_state.processing_results:
        st.info("No processed data yet. Upload and process files to see results here.")
        return
    
    st.markdown("## 📊 Processed Data Dashboard")
    
    # Summary metrics
    total_files = len(st.session_state.processing_results)
    successful_files = sum(1 for r in st.session_state.processing_results.values() if r.success)
    total_records = sum(
        len(r.standardized_data) 
        for r in st.session_state.processing_results.values() 
        if r.success and r.standardized_data is not None
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Processed", total_files)
    with col2:
        st.metric("Successful", successful_files)
    with col3:
        st.metric("Total Records", f"{total_records:,}")
    
    # Data type breakdown
    data_types = {}
    quality_scores = []
    
    for filename, result in st.session_state.processing_results.items():
        if result.success:
            data_types[result.data_type] = data_types.get(result.data_type, 0) + 1
            quality_scores.append(result.quality_score)
    
    if data_types:
        col1, col2 = st.columns(2)
        
        with col1:
            # Data types pie chart
            fig_pie = px.pie(
                values=list(data_types.values()),
                names=list(data_types.keys()),
                title="Data Types Processed"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Quality scores histogram
            fig_hist = px.histogram(
                x=quality_scores,
                title="Data Quality Distribution",
                labels={'x': 'Quality Score', 'y': 'Number of Files'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # File details table
    st.markdown("### 📋 File Processing Details")
    
    details_data = []
    for filename, result in st.session_state.processing_results.items():
        details_data.append({
            'Filename': filename,
            'Status': '✅ Success' if result.success else '❌ Failed',
            'Data Type': result.data_type,
            'Records': len(result.standardized_data) if result.standardized_data is not None else 0,
            'Quality Score': f"{result.quality_score:.0f}/100",
            'Issues': len(result.issues)
        })
    
    if details_data:
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df, use_container_width=True)

def main():
    """Main application"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    display_header()
    
    # System status
    display_system_status()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Action",
        options=[
            "📤 Upload Data Files",
            "🧹 Clean Data (Phase 2)",
            "🏭 Generate Test Data", 
            "📊 View Dashboard",
            "⚙️ System Settings"
        ]
    )
    
    # Main content based on page selection
    if page == "📤 Upload Data Files":
        display_file_upload()
        
    elif page == "🧹 Clean Data (Phase 2)":
        display_data_cleaning_interface()
        
    elif page == "🏭 Generate Test Data":
        display_data_generator()
        
    elif page == "📊 View Dashboard":
        display_processed_data_dashboard()
        
    elif page == "⚙️ System Settings":
        st.markdown("## ⚙️ System Settings")
        
        st.markdown("### Ollama Configuration")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        
        if st.button("🔄 Test Connection"):
            # Test connection
            status = check_ollama_status()
            if status:
                st.success("✅ Ollama connection successful!")
            else:
                st.error("❌ Could not connect to Ollama")
                st.info("Make sure Ollama is running: `ollama serve`")
        
        st.markdown("### Data Processing Settings")
        max_file_size = st.slider("Max File Size (MB)", min_value=1, max_value=1000, value=100)
        processing_timeout = st.slider("Processing Timeout (seconds)", min_value=30, max_value=600, value=120)
        
        st.markdown("### Clear Data")
        if st.button("🗑️ Clear All Processing Results", type="secondary"):
            st.session_state.processing_results = {}
            st.session_state.generated_data = {}
            st.success("✅ All data cleared!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🚀 Universal Data Upload Platform | Powered by Ollama + Llama 3.2 | 
        <a href="https://github.com/anthropics/claude-code">Built with Claude Code</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()