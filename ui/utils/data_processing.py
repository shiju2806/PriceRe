"""
Data Processing Utilities
Common functions for file loading, cleaning, and data quality assessment
"""

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import io
import json
import hashlib
import tempfile
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import cleaning system
try:
    from src.cleaning.hybrid_detector import create_hybrid_detector
    from src.cleaning.header_detector import detect_and_clean_headers
    from src.data_sources.universal_reader import read_any_source
    PHASE2_CLEANING_AVAILABLE = True
except ImportError:
    PHASE2_CLEANING_AVAILABLE = False


def load_file_as_dataframe(uploaded_file) -> pd.DataFrame:
    """Load uploaded file as pandas DataFrame"""
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        return pd.read_csv(io.BytesIO(file_content))
    elif file_type in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_content))
    elif file_type == 'json':
        data = json.loads(file_content.decode('utf-8'))
        return pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    else:
        # Try CSV as fallback
        return pd.read_csv(io.BytesIO(file_content))


def calculate_data_quality_score(df: pd.DataFrame) -> int:
    """Calculate basic data quality score"""
    if df.empty:
        return 0
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    return int(completeness)


def apply_phase2_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply Phase 2 hybrid cleaning system"""
    if not PHASE2_CLEANING_AVAILABLE:
        return df, {"error": "Phase 2 cleaning system not available"}
    
    try:
        # Clean data types before conversion
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).replace('nan', None)
        
        # Convert to Polars
        df_polars = pl.from_pandas(df_clean)
        
        # Apply hybrid cleaning
        detector = create_hybrid_detector('balanced')
        result = detector.detect_junk_rows(df_polars, return_detailed=True)
        
        # Handle header detection
        if result.header_detection_result:
            df_clean_polars, header_result = detect_and_clean_headers(df_polars)
            
            # Apply additional junk removal
            if result.junk_row_indices:
                mask = pl.Series(range(df_clean_polars.shape[0])).is_in(result.junk_row_indices).not_()
                df_clean_polars = df_clean_polars.filter(mask)
            
            df_cleaned = df_clean_polars.to_pandas()
            total_removed = len(header_result.rows_to_remove_above) + len(result.junk_row_indices)
            header_info = f"Header detected at row {header_result.header_row_index}"
        else:
            # No header detection
            if result.junk_row_indices:
                mask = pl.Series(range(df_polars.shape[0])).is_in(result.junk_row_indices).not_()
                df_clean_polars = df_polars.filter(mask)
                df_cleaned = df_clean_polars.to_pandas()
            else:
                df_cleaned = df.copy()
            
            total_removed = len(result.junk_row_indices)
            header_info = "No header detection performed"
        
        return df_cleaned, {
            "result": result,
            "total_removed": total_removed,
            "header_info": header_info,
            "success": True
        }
        
    except Exception as e:
        return df, {"error": str(e), "success": False}


def apply_enhanced_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Apply enhanced cleaning operations"""
    changes = []
    df_cleaned = df.copy()
    
    # Remove duplicates
    original_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    if len(df_cleaned) < original_count:
        changes.append(f"Removed {original_count - len(df_cleaned)} duplicate rows")
    
    # Handle missing values
    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isna().sum()
        if missing_count > 0:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                changes.append(f"Filled {missing_count} missing values in {col} with median")
            else:
                df_cleaned[col].fillna("Unknown", inplace=True)
                changes.append(f"Filled {missing_count} missing values in {col} with 'Unknown'")
    
    # Standardize text columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Remove extra whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            changes.append(f"Standardized text formatting in {col}")
    
    return df_cleaned, changes


def create_file_hash(uploaded_file) -> str:
    """Create unique hash for uploaded file"""
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()[:8]


def save_dataset_to_session(file_key: str, filename: str, df_cleaned: pd.DataFrame, 
                           df_original: pd.DataFrame, cleaning_result: Any = None,
                           file_type: str = "csv") -> None:
    """Save dataset to session state"""
    if 'uploaded_datasets' not in st.session_state:
        st.session_state['uploaded_datasets'] = {}
    
    quality_score = calculate_data_quality_score(df_cleaned)
    
    st.session_state['uploaded_datasets'][file_key] = {
        'filename': filename,
        'data_type': 'uploaded_data',
        'data': df_cleaned,
        'original_data': df_original,
        'cleaning_result': cleaning_result,
        'quality_score': quality_score,
        'records': len(df_cleaned),
        'file_type': file_type,
        'issues': f'Auto-processed - {quality_score}% complete',
        'recommendations': 'Ready for analysis'
    }


def get_file_preview(uploaded_file, max_lines: int = 15) -> str:
    """Get file preview for display"""
    try:
        if uploaded_file.name.endswith(('.csv', '.txt')):
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            lines = content.split('\n')[:max_lines]
            preview_text = '\n'.join(lines)
            
            if len(content.split('\n')) > max_lines:
                preview_text += f"\n... ({len(content.split(chr(10))) - max_lines} more lines)"
            
            return preview_text
            
        elif uploaded_file.name.endswith('.json'):
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            return content[:800] + "..." if len(content) > 800 else content
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return "Excel file - click Process to analyze sheets and columns"
            
    except Exception as e:
        return f"Could not preview: {str(e)}"
    
    return "Preview not available for this file type"


def create_download_data(df: pd.DataFrame, filename: str, format_type: str) -> bytes:
    """Create downloadable data in specified format"""
    if format_type == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    
    elif format_type == 'excel':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cleaned Data', index=False)
        return buffer.getvalue()
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def create_comprehensive_report(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                               cleaning_history: List[Dict], filename: str) -> bytes:
    """Create comprehensive Excel report with multiple sheets"""
    buffer = io.BytesIO()
    base_name = filename.rsplit('.', 1)[0]
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: Cleaned data
        df_cleaned.to_excel(writer, sheet_name='Cleaned Data', index=False)
        
        # Sheet 2: Original data
        df_original.to_excel(writer, sheet_name='Original Data', index=False)
        
        # Sheet 3: Cleaning history
        if cleaning_history:
            history_df = pd.DataFrame(cleaning_history)
            history_df.to_excel(writer, sheet_name='Cleaning History', index=False)
        
        # Sheet 4: Summary statistics
        summary_data = {
            'Metric': [
                'Original Rows',
                'Cleaned Rows', 
                'Rows Removed',
                'Original Columns',
                'Cleaned Columns',
                'Data Quality Score'
            ],
            'Value': [
                len(df_original),
                len(df_cleaned),
                len(df_original) - len(df_cleaned),
                len(df_original.columns),
                len(df_cleaned.columns),
                calculate_data_quality_score(df_cleaned)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return buffer.getvalue()


# Persistent storage utilities
TEMP_DIR = Path(tempfile.gettempdir()) / "pricere_cleaning"
TEMP_DIR.mkdir(exist_ok=True)


def save_persistent_results(file_key: str, results: dict) -> bool:
    """Save cleaning results to persistent file storage"""
    try:
        filepath = TEMP_DIR / f"{file_key}_results.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Also update session state for immediate access
        st.session_state['latest_cleaning_results'] = results
        return True
    except Exception as e:
        st.error(f"Failed to save persistent results: {e}")
        return False


def load_persistent_results(file_key: str) -> dict:
    """Load cleaning results from persistent file storage"""
    try:
        filepath = TEMP_DIR / f"{file_key}_results.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load persistent results: {e}")
    return {}


def get_latest_persistent_results() -> dict:
    """Get the most recent cleaning results"""
    try:
        # Check current session state first
        session_results = st.session_state.get('latest_cleaning_results', {})
        if (session_results.get('df_original') is not None and 
            session_results.get('df_cleaned') is not None and
            session_results.get('result') is not None):
            return session_results
        
        # Check file system cache
        result_files = list(TEMP_DIR.glob("*_results.pkl"))
        if result_files:
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'rb') as f:
                results = pickle.load(f)
            
            # Validate data
            if (results.get('df_original') is not None and 
                results.get('df_cleaned') is not None and
                results.get('result') is not None):
                return results
    except Exception:
        pass
    return {}


def clear_old_cache_files():
    """Clear old cache files to prevent stale data issues"""
    try:
        result_files = list(TEMP_DIR.glob("*_results.pkl"))
        for file in result_files:
            file.unlink()
    except Exception:
        pass