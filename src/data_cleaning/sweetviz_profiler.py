"""
SweetViz Data Profiler - Streamlit Compatible
Professional data profiling using SweetViz library
"""

import pandas as pd
import sweetviz as sv
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SweetVizProfiler:
    """
    Professional data profiler using SweetViz - designed for Streamlit compatibility
    """
    
    def __init__(self):
        self.reports = {}
        
    def generate_profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profile using SweetViz
        
        Args:
            df: DataFrame to profile
            target_column: Optional target column for analysis
            
        Returns:
            Profile results dictionary
        """
        
        try:
            # Create SweetViz report
            if target_column and target_column in df.columns:
                # Create report with target analysis
                report = sv.analyze(df, target_feat=target_column)
                analysis_type = f"Target Analysis ({target_column})"
            else:
                # Create general analysis report
                report = sv.analyze(df)
                analysis_type = "General Analysis"
            
            # Generate basic statistics for our interface
            profile_data = {
                "analysis_type": analysis_type,
                "basic_info": self._get_basic_info(df),
                "data_quality": self._assess_data_quality(df),
                "column_analysis": self._analyze_columns(df),
                "recommendations": self._generate_recommendations(df),
                "sweetviz_report": report
            }
            
            logger.info(f"SweetViz profile generated successfully for {df.shape[0]}x{df.shape[1]} dataset")
            return profile_data
            
        except Exception as e:
            logger.error(f"SweetViz profiling failed: {e}")
            return self._create_fallback_profile(df)
    
    def display_in_streamlit(self, profile_data: Dict[str, Any], height: int = 1000) -> None:
        """
        Display SweetViz report in Streamlit using components
        """
        
        try:
            # Get the SweetViz report
            report = profile_data.get("sweetviz_report")
            
            if report:
                # Create temporary file for the HTML report
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                    # Generate HTML report
                    report.show_html(tmp_file.name, open_browser=False)
                    
                    # Read the HTML content
                    with open(tmp_file.name, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Display in Streamlit using components
                    components.html(html_content, height=height, scrolling=True)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            st.error(f"Could not display SweetViz report: {e}")
            # Show basic profile data as fallback
            self._display_basic_profile(profile_data)
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicate_rows": df.duplicated().sum(),
            "completely_empty_rows": (df.isna().all(axis=1)).sum()
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Assess overall data quality metrics"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        return {
            "overall_completeness": ((total_cells - missing_cells) / total_cells) * 100,
            "rows_with_missing": (df.isna().any(axis=1).sum() / len(df)) * 100,
            "columns_with_missing": (df.isna().any().sum() / len(df.columns)) * 100,
            "duplicate_rate": (df.duplicated().sum() / len(df)) * 100
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column"""
        column_analysis = {}
        
        for col in df.columns:
            analysis = {
                "dtype": str(df[col].dtype),
                "unique_values": df[col].nunique(),
                "unique_ratio": df[col].nunique() / len(df),
                "missing_count": df[col].isna().sum(),
                "missing_percentage": (df[col].isna().sum() / len(df)) * 100
            }
            
            # Add most frequent value
            if not df[col].empty:
                value_counts = df[col].value_counts()
                if not value_counts.empty:
                    analysis["most_frequent"] = value_counts.index[0]
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _generate_recommendations(self, df: pd.DataFrame) -> list:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Check for high missing data
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                recommendations.append({
                    "category": "Data Quality",
                    "issue": f"Column '{col}' has {missing_pct:.1f}% missing values",
                    "recommendation": f"Consider removing or imputing missing values in {col}",
                    "action": f"high_missing_{col}"
                })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append({
                "category": "Data Quality",
                "issue": f"{duplicate_count} duplicate rows found",
                "recommendation": "Remove duplicate rows to improve data quality",
                "action": "remove_duplicates"
            })
        
        # Check for potential data type issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if could be numeric
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', '').replace('$', ''))
                        numeric_count += 1
                    except:
                        pass
                
                if numeric_count / len(sample) > 0.7:
                    recommendations.append({
                        "category": "Data Types",
                        "issue": f"Column '{col}' appears to contain numeric data but is stored as text",
                        "recommendation": f"Convert {col} to numeric type",
                        "action": f"convert_numeric_{col}"
                    })
        
        return recommendations
    
    def _create_fallback_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic profile when SweetViz fails"""
        return {
            "analysis_type": "Basic Analysis (SweetViz unavailable)",
            "basic_info": self._get_basic_info(df),
            "data_quality": self._assess_data_quality(df),
            "column_analysis": self._analyze_columns(df),
            "recommendations": self._generate_recommendations(df),
            "sweetviz_report": None
        }
    
    def _display_basic_profile(self, profile_data: Dict[str, Any]) -> None:
        """Display basic profile data when SweetViz report can't be shown"""
        
        st.markdown("### 📊 Basic Data Profile")
        
        # Basic info
        basic_info = profile_data["basic_info"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", f"{basic_info['shape'][0]:,}")
        with col2:
            st.metric("Columns", basic_info['shape'][1])
        with col3:
            st.metric("Duplicates", basic_info['duplicate_rows'])
        
        # Data quality
        quality = profile_data["data_quality"]
        st.markdown("### 🔍 Data Quality")
        st.write(f"**Completeness:** {quality['overall_completeness']:.1f}%")
        st.write(f"**Rows with missing data:** {quality['rows_with_missing']:.1f}%")
        
        # Recommendations
        recommendations = profile_data["recommendations"]
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for i, rec in enumerate(recommendations[:5], 1):
                st.write(f"**{i}.** {rec['recommendation']}")
                st.caption(f"Issue: {rec['issue']}")

# Global instance
sweetviz_profiler = SweetVizProfiler()