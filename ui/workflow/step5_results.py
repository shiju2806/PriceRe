"""
Step 5: Final Results and Reports
Comprehensive reporting, executive summary, and download options
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def step_5_results_reports():
    """Step 5: Final Results and Reports"""
    
    st.markdown("## 📋 Step 5: Final Results & Reports")
    st.markdown("*Comprehensive pricing analysis results and professional reporting*")
    
    # Check for pricing results
    if not st.session_state.get('pricing_results'):
        _render_no_results_state()
        return
    
    results = st.session_state.pricing_results
    
    # Render main results sections
    _render_executive_summary(results)
    _render_detailed_analysis(results)
    _render_download_options(results)


def _render_no_results_state():
    """Render no pricing results available state"""
    
    st.warning("📊 No pricing results available.")
    st.info("Complete the pricing calculation in Step 4 to view results.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("← Return to Pricing", type="primary", use_container_width=True):
            st.session_state.workflow_step = 4
            st.rerun()


def _render_executive_summary(results: Dict[str, Any]):
    """Render executive summary section"""
    
    st.markdown("### 🎯 Executive Summary")
    
    # Main pricing recommendation card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    ">
        <h3 style="margin: 0 0 15px 0;">🎯 Pricing Recommendation for {results['cedent_name']}</h3>
        <div style="font-size: 1.5rem; margin: 15px 0;">
            <strong>Recommended Gross Rate: {results['gross_rate']:.1%}</strong>
        </div>
        <div style="font-size: 1.1rem; opacity: 0.9;">
            <strong>Annual Premium:</strong> ${results['gross_premium']:,.0f}<br>
            <strong>Treaty Type:</strong> {results['treaty_type']}<br>
            <strong>Confidence Level:</strong> {results['confidence_level']}<br>
            <strong>Data Sources Used:</strong> {len(results['data_sources'])} datasets
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gross Rate", f"{results['gross_rate']:.1%}")
    with col2:
        st.metric("Annual Premium", f"${results['gross_premium']:,.0f}")
    with col3:
        st.metric("Policy Count", f"{results['policy_count']:,}")
    with col4:
        st.metric("Coverage", f"${results['total_coverage']:,.0f}")


def _render_detailed_analysis(results: Dict[str, Any]):
    """Render detailed analysis section"""
    
    st.markdown("### 📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        _render_portfolio_summary(results)
        _render_rate_components(results)
    
    with col2:
        _render_recommendations(results)
        _render_risk_assessment()


def _render_portfolio_summary(results: Dict[str, Any]):
    """Render portfolio summary"""
    
    st.markdown("#### 📈 Portfolio Summary")
    st.markdown(f"""
    - **Policies Analyzed**: {results['policy_count']:,}
    - **Total Coverage**: ${results['total_coverage']:,.0f}
    - **Premium Volume**: ${results['estimated_annual_premium']:,.0f}
    - **Pricing Date**: {results['pricing_date'].strftime('%Y-%m-%d')}
    - **Data Quality**: High
    - **Analysis Method**: Comprehensive Actuarial
    """)


def _render_rate_components(results: Dict[str, Any]):
    """Render rate component breakdown"""
    
    st.markdown("#### 🧮 Rate Component Analysis")
    
    components = [
        ("Expected Loss Ratio", results['expected_loss_ratio'], "🔴"),
        ("Expense Ratio", results['expense_ratio'], "🟡"),
        ("Risk Margin", results['risk_margin'], "🟠"),
        ("Capital Charge", results['capital_charge'], "🔵")
    ]
    
    for name, value, color in components:
        st.markdown(f"{color} **{name}**: {value:.1%}")
    
    st.markdown("---")
    st.markdown(f"🎯 **Total Gross Rate**: {results['gross_rate']:.1%}")


def _render_recommendations(results: Dict[str, Any]):
    """Render key recommendations"""
    
    st.markdown("#### 🎯 Professional Recommendations")
    
    # Generate recommendations based on results
    gross_rate = results['gross_rate']
    treaty_type = results['treaty_type']
    
    if gross_rate < 1.0:
        st.success("✅ **RECOMMENDED FOR ACCEPTANCE**")
        st.markdown("**Pricing Assessment:**")
        st.markdown("• Competitive and sustainable pricing")
        st.markdown("• Adequate profit margins maintained")
        st.markdown("• Risk profile within acceptable parameters")
    elif gross_rate < 1.2:
        st.warning("⚠️ **PROCEED WITH CAUTION**")
        st.markdown("**Pricing Assessment:**")
        st.markdown("• Elevated rate reflects increased risk")
        st.markdown("• Consider additional risk mitigation")
        st.markdown("• Monitor experience closely")
    else:
        st.error("🚨 **RECOMMEND DECLINE OR RESTRUCTURE**")
        st.markdown("**Pricing Assessment:**")
        st.markdown("• Rate indicates high risk exposure")
        st.markdown("• Consider alternative structures")
        st.markdown("• Additional capital requirements")
    
    st.markdown("**Implementation Guidance:**")
    st.markdown(f"• {treaty_type} structure appropriate for risk profile")
    st.markdown(f"• Based on {len(results['data_sources'])} data sources")
    st.markdown("• Recommend quarterly experience reviews")
    st.markdown("• Consider profit-sharing arrangements")


def _render_risk_assessment():
    """Render risk assessment section"""
    
    st.markdown("#### ⚠️ Key Risk Factors")
    
    risk_factors = [
        {
            "factor": "Data Credibility Risk",
            "level": "Medium",
            "description": "Limited historical experience data",
            "mitigation": "Use industry benchmarks for validation"
        },
        {
            "factor": "Economic Environment",
            "level": "Medium", 
            "description": "Interest rate and inflation uncertainty",
            "mitigation": "Regular assumption updates"
        },
        {
            "factor": "Adverse Selection",
            "level": "Low",
            "description": "Quality underwriting standards",
            "mitigation": "Maintain current selection criteria"
        },
        {
            "factor": "Regulatory Capital",
            "level": "Low",
            "description": "Stable regulatory environment",
            "mitigation": "Monitor regulatory changes"
        }
    ]
    
    for risk in risk_factors:
        level_color = "🟢" if risk["level"] == "Low" else "🟡" if risk["level"] == "Medium" else "🔴"
        
        with st.expander(f"{level_color} {risk['factor']} - {risk['level']} Risk"):
            st.markdown(f"**Description:** {risk['description']}")
            st.markdown(f"**Mitigation:** {risk['mitigation']}")


def _render_download_options(results: Dict[str, Any]):
    """Render download and export options"""
    
    st.markdown("### 📥 Report Downloads")
    st.markdown("Export professional reports and analysis results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Executive Summary")
        if st.button("Generate Summary Report", use_container_width=True):
            report_data = _generate_executive_summary_csv(results)
            st.download_button(
                "📥 Download Summary (CSV)",
                data=report_data,
                file_name=f"pricing_summary_{results['cedent_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("#### 📈 Detailed Analysis")
        if st.button("Generate Detailed Report", use_container_width=True):
            detailed_data = _generate_detailed_analysis_csv(results)
            st.download_button(
                "📥 Download Analysis (CSV)",
                data=detailed_data,
                file_name=f"pricing_analysis_{results['cedent_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        st.markdown("#### 📋 Complete Package")
        if st.button("Generate Complete Package", use_container_width=True):
            package_data = _generate_complete_package_excel(results)
            st.download_button(
                "📥 Download Package (Excel)",
                data=package_data,
                file_name=f"pricing_package_{results['cedent_name']}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Sensitivity analysis download
    st.markdown("#### 🔍 Sensitivity Analysis")
    if 'sensitivity' in results:
        sensitivity_data = _generate_sensitivity_csv(results['sensitivity'])
        st.download_button(
            "📊 Download Sensitivity Analysis (CSV)",
            data=sensitivity_data,
            file_name=f"sensitivity_analysis_{results['cedent_name']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📧 Email Results", use_container_width=True):
            st.info("Email functionality would integrate with your email system")
    
    with col2:
        if st.button("💾 Save to Database", use_container_width=True):
            st.info("Database integration would store results for future reference")
    
    with col3:
        if st.button("🔄 Start New Analysis", type="primary", use_container_width=True):
            _reset_workflow()


def _generate_executive_summary_csv(results: Dict[str, Any]) -> str:
    """Generate executive summary report in CSV format"""
    
    summary_data = pd.DataFrame({
        'Metric': [
            'Cedent Name', 'Treaty Type', 'Policy Count', 'Total Coverage',
            'Expected Loss Ratio', 'Expense Ratio', 'Risk Margin', 'Capital Charge',
            'Gross Rate', 'Annual Premium', 'Confidence Level', 'Pricing Date'
        ],
        'Value': [
            results['cedent_name'], 
            results['treaty_type'], 
            results['policy_count'],
            f"${results['total_coverage']:,.0f}",
            f"{results['expected_loss_ratio']:.1%}",
            f"{results['expense_ratio']:.1%}", 
            f"{results['risk_margin']:.1%}",
            f"{results['capital_charge']:.1%}", 
            f"{results['gross_rate']:.1%}",
            f"${results['gross_premium']:,.0f}",
            results['confidence_level'],
            results['pricing_date'].strftime('%Y-%m-%d')
        ]
    })
    
    return summary_data.to_csv(index=False)


def _generate_detailed_analysis_csv(results: Dict[str, Any]) -> str:
    """Generate detailed analysis report in CSV format"""
    
    # Create detailed breakdown
    detailed_data = []
    
    # Rate components
    detailed_data.extend([
        ['Rate Components', '', ''],
        ['Expected Loss Ratio', f"{results['expected_loss_ratio']:.3f}", f"{results['expected_loss_ratio']:.1%}"],
        ['Expense Ratio', f"{results['expense_ratio']:.3f}", f"{results['expense_ratio']:.1%}"],
        ['Risk Margin', f"{results['risk_margin']:.3f}", f"{results['risk_margin']:.1%}"],
        ['Capital Charge', f"{results['capital_charge']:.3f}", f"{results['capital_charge']:.1%}"],
        ['Total Gross Rate', f"{results['gross_rate']:.3f}", f"{results['gross_rate']:.1%}"],
        ['', '', ''],
    ])
    
    # Portfolio metrics
    detailed_data.extend([
        ['Portfolio Metrics', '', ''],
        ['Policy Count', f"{results['policy_count']:,}", 'policies'],
        ['Total Coverage', f"{results['total_coverage']:,.0f}", 'USD'],
        ['Annual Premium', f"{results['gross_premium']:,.0f}", 'USD'],
        ['Average Policy Size', f"{results['total_coverage']/results['policy_count']:,.0f}" if results['policy_count'] > 0 else "0", 'USD'],
        ['', '', ''],
    ])
    
    # Data sources
    detailed_data.extend([
        ['Data Sources', '', ''],
        ['Number of Sources', len(results['data_sources']), 'count'],
        ['Data Sources Used', ', '.join(results['data_sources']), 'list'],
    ])
    
    df = pd.DataFrame(detailed_data, columns=['Category', 'Value', 'Unit'])
    return df.to_csv(index=False)


def _generate_sensitivity_csv(sensitivity: Dict[str, float]) -> str:
    """Generate sensitivity analysis CSV"""
    
    sensitivity_data = pd.DataFrame({
        'Scenario': list(sensitivity.keys()),
        'Rate': [f"{rate:.3f}" for rate in sensitivity.values()],
        'Percentage': [f"{rate:.1%}" for rate in sensitivity.values()],
        'Change_from_Base': [f"{(rate - list(sensitivity.values())[0]):.3f}" for rate in sensitivity.values()]
    })
    
    return sensitivity_data.to_csv(index=False)


def _generate_complete_package_excel(results: Dict[str, Any]) -> bytes:
    """Generate complete Excel package with multiple sheets"""
    
    import io
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Executive Summary sheet
        summary_df = pd.DataFrame({
            'Metric': [
                'Cedent Name', 'Treaty Type', 'Policy Count', 'Total Coverage',
                'Expected Loss Ratio', 'Expense Ratio', 'Risk Margin', 'Capital Charge',
                'Gross Rate', 'Annual Premium', 'Confidence Level', 'Pricing Date'
            ],
            'Value': [
                results['cedent_name'], results['treaty_type'], results['policy_count'],
                results['total_coverage'], results['expected_loss_ratio'],
                results['expense_ratio'], results['risk_margin'], results['capital_charge'],
                results['gross_rate'], results['gross_premium'], results['confidence_level'],
                results['pricing_date'].strftime('%Y-%m-%d')
            ]
        })
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Rate Components sheet
        components_df = pd.DataFrame({
            'Component': ['Expected Loss Ratio', 'Expense Ratio', 'Risk Margin', 'Capital Charge'],
            'Rate': [results['expected_loss_ratio'], results['expense_ratio'], 
                    results['risk_margin'], results['capital_charge']],
            'Percentage': [f"{results['expected_loss_ratio']:.1%}", f"{results['expense_ratio']:.1%}",
                          f"{results['risk_margin']:.1%}", f"{results['capital_charge']:.1%}"]
        })
        components_df.to_excel(writer, sheet_name='Rate Components', index=False)
        
        # Sensitivity Analysis sheet
        if 'sensitivity' in results:
            sensitivity_df = pd.DataFrame({
                'Scenario': list(results['sensitivity'].keys()),
                'Rate': list(results['sensitivity'].values()),
                'Percentage': [f"{rate:.1%}" for rate in results['sensitivity'].values()]
            })
            sensitivity_df.to_excel(writer, sheet_name='Sensitivity Analysis', index=False)
        
        # Recommendations sheet
        recommendations_df = pd.DataFrame({
            'Area': ['Pricing', 'Risk Management', 'Monitoring', 'Structure'],
            'Recommendation': [
                f"Rate of {results['gross_rate']:.1%} recommended for {results['treaty_type']}",
                'Quarterly experience monitoring recommended',
                'Regular assumption validation against industry benchmarks', 
                'Current treaty structure appropriate for risk profile'
            ],
            'Priority': ['High', 'Medium', 'Medium', 'Low']
        })
        recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    return buffer.getvalue()


def _reset_workflow():
    """Reset the entire workflow for new analysis"""
    
    # Clear all workflow-related session state
    keys_to_clear = [
        'workflow_step', 'uploaded_datasets', 'pricing_results', 'pricing_engine',
        'latest_cleaning_results', 'actuarial_metrics'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset to step 1
    st.session_state.workflow_step = 1
    
    st.success("🔄 Workflow reset successfully!")
    st.rerun()


def get_results_summary() -> Dict[str, Any]:
    """Get summary of results for navigation/status display"""
    
    results = st.session_state.get('pricing_results')
    
    if not results:
        return {
            'has_results': False,
            'gross_rate': 0,
            'annual_premium': 0,
            'cedent_name': '',
            'recommendation': 'No results available'
        }
    
    # Determine recommendation based on gross rate
    gross_rate = results['gross_rate']
    if gross_rate < 1.0:
        recommendation = "Recommended"
    elif gross_rate < 1.2:
        recommendation = "Proceed with caution"
    else:
        recommendation = "Recommend decline"
    
    return {
        'has_results': True,
        'gross_rate': gross_rate,
        'annual_premium': results['gross_premium'],
        'cedent_name': results['cedent_name'],
        'treaty_type': results['treaty_type'],
        'recommendation': recommendation
    }