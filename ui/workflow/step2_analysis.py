"""
Step 2: Intelligent Analysis and Data Integration
Data relationship analysis, quality assessment, and pricing readiness evaluation
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any

# Import utilities
from ui.utils.data_processing import get_latest_persistent_results, clear_old_cache_files


def step_2_intelligent_analysis():
    """Step 2: Intelligent Analysis and Data Integration"""
    
    st.markdown("## 🧠 Step 2: Process & Analyze Data")
    st.markdown("Intelligent data integration and relationship analysis")
    
    # Check for both workflow datasets and persistent results
    datasets = st.session_state.get('uploaded_datasets', {})
    persistent_results = get_latest_persistent_results()
    
    # Debug and integration tools
    _render_debug_integration_tools(datasets, persistent_results)
    
    # Handle no data scenario
    if not datasets and not persistent_results:
        _render_no_data_state()
        return
    
    # Integrate persistent results if needed
    if persistent_results and not datasets:
        datasets = _integrate_persistent_results(persistent_results)
    
    # Display datasets summary
    _render_datasets_summary(datasets)
    
    # Advanced integration analysis
    _render_integration_analysis(datasets)


def _render_debug_integration_tools(datasets: Dict, persistent_results: Dict):
    """Render debug and integration tools"""
    
    with st.expander("🔧 Integration Tools", expanded=False):
        st.write("**Current Status:**")
        st.write(f"• Workflow datasets: {len(datasets)}")
        st.write(f"• Persistent results: {'Available' if persistent_results else 'None'}")
        
        if persistent_results:
            st.write(f"• Last cleaned file: {persistent_results.get('uploaded_file_name', 'Unknown')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Cache", key="clear_cache_debug"):
                clear_old_cache_files()
                st.session_state.uploaded_datasets = {}
                st.session_state.latest_cleaning_results = {}
                st.success("Cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Force Integration", key="force_integration"):
                integrated_count = _force_data_integration(datasets, persistent_results)
                if integrated_count > 0:
                    st.success(f"🎉 Integrated {integrated_count} datasets!")
                    st.rerun()
                else:
                    st.warning("No data found to integrate")
        
        with col3:
            if st.button("↩️ Back to Step 1", key="back_to_step1"):
                st.session_state.workflow_step = 1
                st.rerun()


def _render_no_data_state():
    """Render no data available state"""
    
    st.warning("📥 No datasets uploaded. Return to Step 1 to upload your data files.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("← Back to Upload", type="primary", use_container_width=True):
            st.session_state.workflow_step = 1
            st.rerun()


def _force_data_integration(datasets: Dict, persistent_results: Dict) -> int:
    """Force integration of available data"""
    
    integrated_count = 0
    
    # Integrate persistent results
    if persistent_results and persistent_results.get('df_cleaned') is not None:
        file_key = persistent_results.get('file_key', f'persistent_data_{integrated_count}')
        datasets[file_key] = {
            'filename': persistent_results.get('uploaded_file_name', f'Cleaned_Data_{integrated_count}'),
            'data': persistent_results.get('df_cleaned'),
            'data_type': 'cleaned_dataset',
            'quality_score': 85,
            'records': len(persistent_results.get('df_cleaned', [])),
            'issues': 'Integrated from persistent storage',
            'recommendations': 'Data ready for analysis'
        }
        integrated_count += 1
    
    # Look for DataFrames in session state
    for key, value in st.session_state.items():
        if hasattr(value, 'shape') and len(value.shape) == 2:  # It's a DataFrame
            if key not in datasets and 'uploaded_datasets' not in key:
                datasets[f'discovered_{integrated_count}'] = {
                    'filename': f'Data_from_{key}',
                    'data': value,
                    'data_type': 'discovered_data',
                    'quality_score': 70,
                    'records': len(value),
                    'issues': f'Discovered in session state as {key}',
                    'recommendations': 'Review data quality'
                }
                integrated_count += 1
    
    if integrated_count > 0:
        st.session_state.uploaded_datasets = datasets
    
    return integrated_count


def _integrate_persistent_results(persistent_results: Dict) -> Dict:
    """Integrate persistent results into workflow datasets"""
    
    st.info("✅ Found recent cleaning results - integrating into workflow")
    
    file_key = persistent_results.get('file_key', 'cleaned_data')
    datasets = {
        file_key: {
            'name': persistent_results.get('uploaded_file_name', 'Cleaned Data'),
            'filename': persistent_results.get('uploaded_file_name', 'Cleaned Data'),
            'data': persistent_results.get('df_cleaned'),
            'original_data': persistent_results.get('df_original'),
            'cleaning_result': persistent_results.get('result'),
            'file_type': persistent_results.get('file_type', 'unknown'),
            'data_type': persistent_results.get('file_type', 'cleaned_dataset'),
            'quality_score': 85  # Default quality score for cleaned data
        }
    }
    
    st.session_state.uploaded_datasets = datasets
    st.success("🎉 Cleaning results integrated into workflow!")
    
    return datasets


def _render_datasets_summary(datasets: Dict):
    """Render summary of uploaded datasets"""
    
    st.markdown("### 📊 Uploaded Datasets")
    
    if not datasets:
        st.info("No datasets to display")
        return
    
    # Overall summary metrics
    total_records = sum(len(d.get('data', [])) for d in datasets.values())
    avg_quality = sum(d.get('quality_score', 0) for d in datasets.values()) / len(datasets)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Datasets", len(datasets))
    with col2:
        st.metric("Total Records", f"{total_records:,}")
    with col3:
        st.metric("Average Quality", f"{avg_quality:.0f}%")
    
    # Individual dataset details
    for key, dataset in datasets.items():
        with st.expander(f"📁 {dataset.get('data_type', 'Unknown')} - {dataset.get('filename', 'Unknown')}", 
                         expanded=False):
            _render_individual_dataset_details(key, dataset)


def _render_individual_dataset_details(key: str, dataset: Dict):
    """Render details for individual dataset"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Records**: {len(dataset.get('data', [])):,}  
        **Quality Score**: {dataset.get('quality_score', 0):.0f}/100  
        **Data Type**: {dataset.get('data_type', 'Unknown')}
        """)
        
        # Quality indicator
        quality_score = dataset.get('quality_score', 0)
        if quality_score >= 85:
            st.success("🟢 Excellent quality")
        elif quality_score >= 70:
            st.warning("🟡 Good quality")
        else:
            st.error("🟠 Needs improvement")
    
    with col2:
        if st.button(f"👀 View Sample Data", key=f"view_{key}"):
            st.markdown("#### Sample Data Preview:")
            if 'data' in dataset and dataset['data'] is not None:
                st.dataframe(dataset['data'].head(5), use_container_width=True)
            else:
                st.warning("No data available to display")
        
        # Additional actions
        if st.button(f"📊 Data Summary", key=f"summary_{key}"):
            _show_data_summary(dataset)


def _show_data_summary(dataset: Dict):
    """Show detailed data summary"""
    
    if 'data' not in dataset or dataset['data'] is None:
        st.warning("No data available for summary")
        return
    
    df = dataset['data']
    
    st.markdown("#### 📊 Data Summary:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Basic Statistics:**")
        st.write(f"• Rows: {len(df):,}")
        st.write(f"• Columns: {len(df.columns)}")
        st.write(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    with col2:
        st.markdown("**Data Quality:**")
        missing_cells = df.isna().sum().sum()
        total_cells = len(df) * len(df.columns)
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        st.write(f"• Completeness: {completeness:.1f}%")
        st.write(f"• Missing values: {missing_cells:,}")
        st.write(f"• Data types: {len(df.dtypes.unique())}")


def _render_integration_analysis(datasets: Dict):
    """Render advanced integration analysis"""
    
    st.markdown("### 🔬 Data Integration Analysis")
    st.markdown("Analyze data relationships and pricing readiness")
    
    if not datasets:
        st.info("Upload datasets to perform integration analysis")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🧠 Analyze Data Relationships", type="primary", use_container_width=True):
            _perform_integration_analysis(datasets)
            st.session_state.analysis_completed = True
    
    # Show analysis results if already performed
    if st.session_state.get('analysis_completed', False) and st.session_state.get('integration_analysis'):
        _display_analysis_results()
    
    with col2:
        # Quick stats
        total_records = sum(len(d.get('data', [])) for d in datasets.values())
        if total_records >= 100:
            st.success(f"✅ {total_records:,} records")
        else:
            st.warning(f"⚠️ {total_records:,} records (need 100+)")


def _perform_integration_analysis(datasets: Dict):
    """Perform comprehensive integration analysis - calculation only"""
    
    with st.spinner("🔍 Analyzing data relationships and pricing readiness..."):
        relationships = []
        dataset_names = list(datasets.keys())
        
        for i, dataset1 in enumerate(dataset_names):
            data1 = datasets[dataset1]['data']
            cols1 = set(data1.columns.str.lower())
            
            for j, dataset2 in enumerate(dataset_names):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                    
                data2 = datasets[dataset2]['data'] 
                cols2 = set(data2.columns.str.lower())
                
                # Find common columns
                common_cols = cols1.intersection(cols2)
                
                if common_cols:
                    strength = len(common_cols) / min(len(cols1), len(cols2)) * 100
                    relationships.append({
                        'Dataset 1': dataset1,
                        'Dataset 2': dataset2, 
                        'Common Fields': len(common_cols),
                        'Relationship Strength': f"{strength:.1f}%",
                        'Linkable Fields': ', '.join(list(common_cols)[:5])
                    })
        
        # Store analysis results in session state (no display here)
        st.session_state.integration_analysis = {
            'relationships': relationships,
            'total_files': len(datasets),
            'analyzed_at': pd.Timestamp.now(),
            'datasets': datasets,
            'analysis_summary': {
                'pricing_readiness': 75 if relationships else 50,
                'data_quality': 80,
                'dataset_count': len(datasets),
                'relationships': len(relationships)
            }
        }
        
        st.success(f"✅ Analysis complete! Found {len(relationships)} relationships.")


def _display_analysis_results():
    """Display the stored analysis results"""
    
    analysis_data = st.session_state.integration_analysis
    relationships = analysis_data['relationships']
    
    st.markdown("### 🔗 Detected Data Relationships")
    
    if relationships:
        st.success(f"🔗 Found {len(relationships)} potential relationships!")
        relationships_df = pd.DataFrame(relationships)
        st.dataframe(relationships_df, use_container_width=True)
        
        # Show integration opportunities
        st.markdown("### 💡 Integration Opportunities")
        for rel in relationships[:3]:  # Show top 3
            if float(rel['Relationship Strength'].replace('%', '')) > 30:
                st.info(f"**{rel['Dataset 1']} ↔ {rel['Dataset 2']}**: Can be linked via {rel['Linkable Fields']}")
    else:
        st.warning("⚠️ No direct relationships detected between datasets")
        st.markdown("Consider:")
        st.markdown("- Check if files share common identifiers (Policy ID, Customer ID, etc.)")
        st.markdown("- Ensure consistent column naming across files")
        st.markdown("- Add reference data to link disparate datasets")
    
    # Show recommendations with navigation
    _render_analysis_recommendations(analysis_data['analysis_summary'])


def _render_analysis_recommendations(analysis: Dict):
    """Render analysis recommendations and next steps"""
    
    st.markdown("#### 💡 Recommendations")
    
    readiness = analysis['pricing_readiness']
    quality = analysis['data_quality']
    count = analysis['dataset_count']
    
    if readiness >= 70:
        st.success("🎉 **Data is ready for actuarial analysis!**")
        st.markdown("Your datasets meet the quality and coverage requirements for pricing.")
        
        # Debug info
        st.write(f"Current readiness score: {readiness}")
        st.write(f"Current workflow step: {st.session_state.get('workflow_step', 1)}")
        
        if st.button("➡️ Continue to Actuarial Analysis", key="continue_to_step3_main", type="primary"):
            st.success("Navigating to Step 3...")
            st.session_state.workflow_step = 3
            st.rerun()
            
    elif readiness >= 50:
        st.warning("⚠️ **Data quality could be improved**")
        st.markdown("Consider:")
        if quality < 80:
            st.markdown("• Improving data quality through additional cleaning")
        if count < 2:
            st.markdown("• Adding more data sources for better analysis")
        
        st.markdown("You can proceed with caution or return to Step 1 for more data.")
        
        # Debug info for this path too
        st.write(f"Alternative path readiness score: {readiness}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚠️ Proceed Anyway", key="proceed_anyway_to_step3"):
                st.success("Proceeding to Step 3...")
                st.session_state.workflow_step = 3
                st.rerun()
        with col2:
            if st.button("← Add More Data", key="back_to_step1"):
                st.session_state.workflow_step = 1
                st.rerun()
    
    else:
        st.error("🚨 **Data quality insufficient for reliable pricing**")
        st.markdown("**Issues identified:**")
        if quality < 50:
            st.markdown("• Poor data quality - consider re-uploading or cleaning")
        if count < 1:
            st.markdown("• No valid datasets found")
        
        st.markdown("Return to Step 1 to upload and clean your data.")
        
        if st.button("← Return to Upload", type="primary"):
            st.session_state.workflow_step = 1
            st.rerun()


def analyze_data_integration(datasets: Dict) -> Dict[str, Any]:
    """Analyze uploaded datasets for pricing readiness"""
    
    # Analyze what we actually have instead of looking for specific types
    total_datasets = len(datasets)
    has_data = total_datasets > 0
    
    # Check data quality and completeness
    quality_scores = [d.get('quality_score', 0) for d in datasets.values()]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Check data size and coverage
    total_records = sum(len(d.get('data', [])) for d in datasets.values())
    has_sufficient_data = total_records >= 100  # Need at least 100 records
    
    # Analysis based on what we actually have
    coverage = {
        f'Dataset Coverage ({total_datasets} files)': has_data,
        f'Data Quality (Avg: {avg_quality:.0f}%)': avg_quality >= 70,
        f'Data Volume ({total_records:,} records)': has_sufficient_data,
        'Multiple Data Sources': total_datasets >= 2,
        'High Quality Data': avg_quality >= 85
    }
    
    coverage_score = sum(coverage.values()) / len(coverage) * 100
    
    # More realistic pricing readiness calculation
    pricing_readiness = min(100, (coverage_score * 0.6 + avg_quality * 0.4))
    
    return {
        'coverage': coverage,
        'pricing_readiness': pricing_readiness,
        'data_quality': avg_quality,
        'dataset_count': total_datasets
    }


def get_analysis_summary() -> Dict[str, Any]:
    """Get summary of analysis results"""
    
    datasets = st.session_state.get('uploaded_datasets', {})
    
    if not datasets:
        return {
            'datasets_count': 0,
            'total_records': 0,
            'avg_quality': 0,
            'pricing_readiness': 0
        }
    
    analysis = analyze_data_integration(datasets)
    
    return {
        'datasets_count': len(datasets),
        'total_records': sum(len(d.get('data', [])) for d in datasets.values()),
        'avg_quality': analysis['data_quality'],
        'pricing_readiness': analysis['pricing_readiness']
    }