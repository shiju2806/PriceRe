"""
Professional Actuarial Workbench
Advanced technical dashboard for actuarial professionals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, date
import scipy.stats as stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def professional_workbench():
    """Professional actuarial workbench interface"""
    
    st.markdown("""
    <style>
    .workbench-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c2d12 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .tool-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .metric-dashboard {
        background: linear-gradient(135deg, #065f46 0%, #1f2937 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="workbench-header">', unsafe_allow_html=True)
    st.markdown("# 🔬 Professional Actuarial Workbench")
    st.markdown("**Advanced Technical Analysis & Model Diagnostics**")
    st.markdown("*For Actuarial Professionals & Model Validators*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main workbench tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance Monitor", 
        "🔬 Assumption Testing Lab", 
        "📈 Risk Analytics", 
        "🏗️ Model Architecture",
        "📋 Regulatory Dashboard"
    ])
    
    with tab1:
        model_performance_monitor()
    
    with tab2:
        assumption_testing_lab()
    
    with tab3:
        risk_analytics_dashboard()
    
    with tab4:
        model_architecture_analyzer()
    
    with tab5:
        regulatory_compliance_dashboard()

def model_performance_monitor():
    """Real-time model performance monitoring"""
    st.markdown("## 📊 Real-time Model Performance Monitor")
    
    # Generate realistic model performance data
    dates = pd.date_range('2024-01-01', '2024-08-29', freq='D')
    
    # Model performance metrics over time
    auc_scores = 0.918 + np.random.normal(0, 0.005, len(dates))
    gini_scores = auc_scores * 2 - 1
    log_loss = 0.25 + np.random.normal(0, 0.01, len(dates))
    
    performance_df = pd.DataFrame({
        'date': dates,
        'auc_score': auc_scores.clip(0.8, 1.0),
        'gini_coefficient': gini_scores.clip(0.6, 1.0),
        'log_loss': log_loss.clip(0.1, 0.4)
    })
    
    # Current vs target metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_auc = performance_df['auc_score'].iloc[-1]
    current_gini = performance_df['gini_coefficient'].iloc[-1]
    current_log_loss = performance_df['log_loss'].iloc[-1]
    
    with col1:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric(
            "Current AUC Score", 
            f"{current_auc:.3f}",
            delta=f"{current_auc - 0.918:+.3f}",
            help="Target: >0.900, Industry Benchmark: 0.850"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric(
            "Gini Coefficient", 
            f"{current_gini:.3f}",
            delta=f"{current_gini - 0.836:+.3f}",
            help="Target: >0.800, Industry Benchmark: 0.700"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric(
            "Log Loss", 
            f"{current_log_loss:.3f}",
            delta=f"{current_log_loss - 0.251:+.3f}" if current_log_loss > 0.251 else f"{current_log_loss - 0.251:.3f}",
            delta_color="inverse",
            help="Target: <0.300, Lower is better"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # Model health score
        health_score = (
            (current_auc - 0.5) * 0.4 +  # AUC contribution
            (current_gini) * 0.4 +        # Gini contribution  
            (0.5 - current_log_loss) * 0.2  # Log loss contribution (inverted)
        ) * 100
        
        if health_score > 85:
            health_color = "🟢"
            health_status = "Excellent"
        elif health_score > 75:
            health_color = "🟡"
            health_status = "Good"
        else:
            health_color = "🔴"
            health_status = "Needs Attention"
        
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric(
            "Model Health", 
            f"{health_color} {health_score:.1f}%",
            help=f"Overall model health: {health_status}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        # AUC and Gini trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_df['date'],
            y=performance_df['auc_score'],
            mode='lines+markers',
            name='AUC Score',
            line=dict(color='#10b981', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_df['date'],
            y=performance_df['gini_coefficient'],
            mode='lines+markers',
            name='Gini Coefficient',
            line=dict(color='#3b82f6', width=2),
            yaxis='y2'
        ))
        
        # Add target lines
        fig.add_hline(y=0.900, line_dash="dash", line_color="green", annotation_text="AUC Target")
        
        fig.update_layout(
            title="Model Performance Trends",
            xaxis_title="Date",
            yaxis_title="AUC Score",
            yaxis2=dict(title="Gini Coefficient", overlaying='y', side='right'),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model stability metrics
        rolling_std = performance_df['auc_score'].rolling(30).std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_df['date'],
            y=rolling_std,
            mode='lines+markers',
            name='30-Day Rolling Std Dev',
            line=dict(color='#ef4444', width=2)
        ))
        
        fig.add_hline(y=0.01, line_dash="dash", line_color="orange", annotation_text="Stability Threshold")
        
        fig.update_layout(
            title="Model Stability (AUC Volatility)",
            xaxis_title="Date",
            yaxis_title="Rolling Standard Deviation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature drift monitoring
    st.markdown("### 🚨 Feature Drift Detection")
    
    feature_drift_data = pd.DataFrame({
        'Feature': ['Issue Age', 'BMI', 'Smoker Status', 'Medical Conditions', 'Blood Pressure', 
                   'Cholesterol', 'Exercise Frequency', 'Geographic Region', 'Family History'],
        'Current Mean': [45.2, 27.8, 0.15, 1.8, 125.4, 195.2, 2.4, 0.35, 0.42],
        'Training Mean': [44.8, 27.5, 0.16, 1.7, 124.8, 193.1, 2.3, 0.33, 0.41],
        'Drift Score': [0.012, 0.008, 0.021, 0.031, 0.018, 0.025, 0.009, 0.038, 0.019],
        'Status': ['✅ Stable', '✅ Stable', '⚠️ Monitor', '🔴 Alert', '✅ Stable', 
                  '⚠️ Monitor', '✅ Stable', '🔴 Alert', '✅ Stable']
    })
    
    # Color code the drift scores
    def highlight_drift(val):
        if '🔴' in str(val):
            return 'background-color: #fee2e2'
        elif '⚠️' in str(val):
            return 'background-color: #fef3cd'
        else:
            return 'background-color: #d1fae5'
    
    styled_df = feature_drift_data.style.applymap(highlight_drift, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Alerts and recommendations
    st.markdown("### 🚨 Current Alerts")
    
    alerts = [
        {"severity": "🔴 HIGH", "message": "Geographic Region feature drift detected (score: 0.038)", "action": "Recommend model retraining"},
        {"severity": "🔴 HIGH", "message": "Medical Conditions distribution shift (score: 0.031)", "action": "Review recent underwriting changes"},
        {"severity": "⚠️ MEDIUM", "message": "Cholesterol levels trending higher", "action": "Monitor for population health trends"},
        {"severity": "⚠️ MEDIUM", "message": "Smoker prevalence decreasing", "action": "Validate against industry data"}
    ]
    
    for alert in alerts:
        if "HIGH" in alert["severity"]:
            st.error(f"**{alert['severity']}**: {alert['message']}")
            st.write(f"   → **Recommended Action**: {alert['action']}")
        else:
            st.warning(f"**{alert['severity']}**: {alert['message']}")
            st.write(f"   → **Recommended Action**: {alert['action']}")

def assumption_testing_lab():
    """Interactive assumption testing interface"""
    st.markdown("## 🔬 Assumption Testing Laboratory")
    
    st.markdown("""
    **Test the impact of changing key actuarial assumptions on pricing and reserves.**
    Perform sensitivity analysis and scenario testing with real-time calculations.
    """)
    
    # Base scenario setup
    st.markdown("### 📋 Base Scenario Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Mortality Assumptions**")
        base_mortality_table = st.selectbox("Mortality Table", ["2017 CSO", "2015 VBT", "2001 CSO"])
        mortality_improvement = st.slider("Mortality Improvement %", -20.0, 20.0, 0.0, 0.5)
        credibility_factor = st.slider("Experience Credibility", 0.0, 1.0, 0.75, 0.05)
    
    with col2:
        st.markdown("**Economic Assumptions**")
        discount_rate = st.slider("Discount Rate %", 1.0, 8.0, 3.2, 0.1)
        inflation_rate = st.slider("Inflation Rate %", 0.0, 5.0, 2.5, 0.1)
        investment_return = st.slider("Investment Return %", 2.0, 10.0, 4.5, 0.1)
    
    with col3:
        st.markdown("**Behavioral Assumptions**")
        base_lapse_rate = st.slider("Base Lapse Rate %", 2.0, 15.0, 8.0, 0.5)
        lapse_sensitivity = st.slider("Economic Sensitivity", -2.0, 2.0, 0.0, 0.1)
        surrender_charge = st.slider("Surrender Charge %", 0.0, 10.0, 2.0, 0.5)
    
    # Scenario comparison
    st.markdown("### 📊 Scenario Testing")
    
    scenario_type = st.selectbox("Select Scenario Type", [
        "Stress Testing",
        "Best/Worst Case Analysis", 
        "Regulatory Scenarios",
        "Custom Sensitivity Analysis"
    ])
    
    if scenario_type == "Stress Testing":
        st.markdown("#### 🌪️ Regulatory Stress Scenarios")
        
        scenarios = {
            "Base Case": {"mortality": 0, "lapse": 0, "rates": 0},
            "Mortality Shock (+25%)": {"mortality": 25, "lapse": 0, "rates": 0},
            "Mass Lapse (+50%)": {"mortality": 0, "lapse": 50, "rates": 0},
            "Interest Rate Shock (-200bp)": {"mortality": 0, "lapse": 0, "rates": -200},
            "Combined Stress": {"mortality": 15, "lapse": 25, "rates": -100}
        }
        
        # Calculate impacts for each scenario
        scenario_results = []
        
        for scenario_name, shocks in scenarios.items():
            # Mock calculation - in reality would call actual pricing functions
            base_premium = 1000
            base_reserve = 15000
            base_capital = 2500
            
            # Apply shocks
            mortality_impact = 1 + (shocks["mortality"] / 100)
            lapse_impact = 1 + (shocks["lapse"] / 100) * 0.3  # Lapse reduces reserves
            rate_impact = 1 + (shocks["rates"] / 10000) * 2  # Rate sensitivity
            
            stressed_premium = base_premium * mortality_impact * rate_impact
            stressed_reserve = base_reserve * mortality_impact * lapse_impact
            stressed_capital = base_capital * mortality_impact * (rate_impact ** 0.5)
            
            scenario_results.append({
                "Scenario": scenario_name,
                "Premium": f"${stressed_premium:.0f}",
                "Premium Impact": f"{(stressed_premium/base_premium - 1)*100:+.1f}%",
                "Reserve": f"${stressed_reserve:.0f}",
                "Reserve Impact": f"{(stressed_reserve/base_reserve - 1)*100:+.1f}%",
                "Capital": f"${stressed_capital:.0f}",
                "Capital Impact": f"{(stressed_capital/base_capital - 1)*100:+.1f}%"
            })
        
        results_df = pd.DataFrame(scenario_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        impacts = [
            float(row["Premium Impact"].replace('%', '').replace('+', '')) 
            for row in scenario_results
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=list(scenarios.keys()), y=impacts, 
                   marker_color=['green' if x >= 0 else 'red' for x in impacts])
        ])
        
        fig.update_layout(
            title="Premium Impact by Stress Scenario",
            xaxis_title="Scenario",
            yaxis_title="Premium Impact (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif scenario_type == "Custom Sensitivity Analysis":
        st.markdown("#### 🎛️ Custom Sensitivity Analysis")
        
        # Interactive tornado diagram
        sensitivity_var = st.selectbox("Variable to Analyze", [
            "Mortality Rate", "Lapse Rate", "Discount Rate", 
            "Expense Ratio", "Profit Margin"
        ])
        
        base_value = st.number_input("Base Value", value=100.0)
        shock_range = st.slider("Shock Range (%)", 5, 50, 20)
        
        # Generate tornado data
        shocks = np.arange(-shock_range, shock_range + 1, shock_range//5)
        impacts = []
        
        for shock in shocks:
            # Simplified sensitivity calculation
            if sensitivity_var == "Mortality Rate":
                impact = shock * 0.8  # High sensitivity
            elif sensitivity_var == "Lapse Rate":
                impact = shock * -0.3  # Inverse relationship
            elif sensitivity_var == "Discount Rate":
                impact = shock * -1.2  # Strong inverse
            else:
                impact = shock * 0.5
            
            impacts.append(base_value * (1 + impact/100))
        
        # Tornado chart
        fig = go.Figure()
        
        colors = ['red' if x < 0 else 'green' for x in shocks]
        
        fig.add_trace(go.Bar(
            x=impacts,
            y=[f"{shock:+}%" for shock in shocks],
            orientation='h',
            marker_color=colors,
            text=[f"${impact:.0f}" for impact in impacts],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Sensitivity Analysis: {sensitivity_var}",
            xaxis_title="Output Value",
            yaxis_title="Shock Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export sensitivity results
        sensitivity_df = pd.DataFrame({
            "Shock Level": [f"{shock:+}%" for shock in shocks],
            "Input Value": [base_value * (1 + shock/100) for shock in shocks],
            "Output Value": impacts,
            "Impact": [f"{(impact/base_value - 1)*100:+.1f}%" for impact in impacts]
        })
        
        csv_data = sensitivity_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Sensitivity Analysis",
            data=csv_data,
            file_name=f"sensitivity_analysis_{sensitivity_var.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def risk_analytics_dashboard():
    """Advanced risk analytics and concentration analysis"""
    st.markdown("## 📈 Advanced Risk Analytics")
    
    # Risk decomposition
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Risk Decomposition")
        
        # Generate risk factor data
        risk_factors = {
            'Mortality Risk': 45.2,
            'Longevity Risk': 23.1, 
            'Lapse Risk': 18.7,
            'Expense Risk': 8.9,
            'Interest Rate Risk': 12.3,
            'Credit Risk': 6.2,
            'Operational Risk': 4.8,
            'Model Risk': 3.1
        }
        
        fig = px.pie(
            values=list(risk_factors.values()),
            names=list(risk_factors.keys()),
            title="Risk Factor Contribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Risk Concentration Metrics")
        
        concentration_metrics = pd.DataFrame({
            'Metric': [
                'Geographic Concentration (Herfindahl)',
                'Product Concentration',
                'Age Band Concentration', 
                'Large Policy Concentration',
                'Reinsurer Concentration'
            ],
            'Current Value': [0.15, 0.32, 0.28, 0.42, 0.18],
            'Target': [0.20, 0.35, 0.30, 0.45, 0.25],
            'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good', '✅ Good']
        })
        
        # Add color coding
        def highlight_status(val):
            if '✅' in str(val):
                return 'background-color: #d1fae5'
            elif '⚠️' in str(val):
                return 'background-color: #fef3cd'
            else:
                return 'background-color: #fee2e2'
        
        styled_metrics = concentration_metrics.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_metrics, use_container_width=True)
    
    # VaR and tail risk analysis
    st.markdown("### 📉 Value at Risk (VaR) Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate VaR data
        confidence_levels = [90, 95, 99, 99.5, 99.9]
        var_values = [42.3, 67.8, 121.4, 156.7, 203.9]  # in millions
        
        var_df = pd.DataFrame({
            'Confidence Level': [f"{cl}%" for cl in confidence_levels],
            'VaR ($M)': var_values,
            'Expected Shortfall ($M)': [v * 1.2 for v in var_values]
        })
        
        st.markdown("**VaR by Confidence Level**")
        st.dataframe(var_df, use_container_width=True)
    
    with col2:
        # Risk factor correlation matrix
        st.markdown("**Risk Factor Correlations**")
        
        factors = ['Mortality', 'Lapse', 'Interest', 'Expense']
        correlation_matrix = np.array([
            [1.00, -0.15, 0.23, 0.31],
            [-0.15, 1.00, 0.45, 0.12],
            [0.23, 0.45, 1.00, 0.08],
            [0.31, 0.12, 0.08, 1.00]
        ])
        
        fig = px.imshow(
            correlation_matrix,
            x=factors,
            y=factors,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Tail risk metrics
        st.markdown("**Tail Risk Metrics**")
        
        tail_metrics = pd.DataFrame({
            'Metric': ['Expected Shortfall (99%)', 'Maximum Drawdown', 'Tail Expectation', 'Risk Contribution'],
            'Value': ['$145.2M', '$89.3M', '$167.8M', '23.4%'],
            'Benchmark': ['$120M', '$75M', '$140M', '25%'],
            'Status': ['⚠️ Above', '⚠️ Above', '⚠️ Above', '✅ Good']
        })
        
        styled_tail = tail_metrics.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_tail, use_container_width=True)
    
    # Monte Carlo simulation results
    st.markdown("### 🎲 Monte Carlo Simulation Results")
    
    # Generate simulation data
    np.random.seed(42)
    n_sims = 10000
    
    # Simulate portfolio outcomes
    mortality_shocks = np.random.normal(1.0, 0.15, n_sims)
    lapse_shocks = np.random.lognormal(0, 0.3, n_sims)
    rate_shocks = np.random.normal(0, 0.02, n_sims)
    
    portfolio_outcomes = (
        mortality_shocks * 0.6 +
        lapse_shocks * 0.3 +
        (1 + rate_shocks) * 0.4
    ) * 100  # Convert to millions
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of outcomes
        fig = px.histogram(
            x=portfolio_outcomes,
            nbins=50,
            title="Distribution of Portfolio Outcomes",
            labels={'x': 'Portfolio Value ($M)', 'y': 'Frequency'}
        )
        
        # Add percentile lines
        p95 = np.percentile(portfolio_outcomes, 5)
        p99 = np.percentile(portfolio_outcomes, 1)
        
        fig.add_vline(x=p95, line_dash="dash", line_color="orange", 
                     annotation_text="95th Percentile")
        fig.add_vline(x=p99, line_dash="dash", line_color="red", 
                     annotation_text="99th Percentile")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Percentile table
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(portfolio_outcomes, p) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Portfolio Value ($M)': [f"${v:.1f}" for v in percentile_values],
            'Loss from Mean ($M)': [f"${np.mean(portfolio_outcomes) - v:.1f}" if v < np.mean(portfolio_outcomes) else f"+${v - np.mean(portfolio_outcomes):.1f}" for v in percentile_values]
        })
        
        st.markdown("**Simulation Percentiles**")
        st.dataframe(percentile_df, use_container_width=True)
        
        # Key statistics
        st.markdown("**Key Statistics**")
        st.write(f"• **Mean:** ${np.mean(portfolio_outcomes):.1f}M")
        st.write(f"• **Std Dev:** ${np.std(portfolio_outcomes):.1f}M")
        st.write(f"• **Skewness:** {stats.skew(portfolio_outcomes):.2f}")
        st.write(f"• **Kurtosis:** {stats.kurtosis(portfolio_outcomes):.2f}")

def model_architecture_analyzer():
    """Model architecture analysis and documentation"""
    st.markdown("## 🏗️ Model Architecture Analyzer")
    
    # Model inventory
    st.markdown("### 📋 Model Inventory")
    
    models = pd.DataFrame({
        'Model Name': ['Mortality XGBoost', 'Lapse Random Forest', 'Economic LSTM', 'Reserve Calculator', 'Capital Model'],
        'Version': ['v2.3.1', 'v1.8.2', 'v3.1.0', 'v4.2.1', 'v2.1.0'],
        'Type': ['ML', 'ML', 'ML', 'Deterministic', 'Deterministic'],
        'Status': ['Production', 'Production', 'UAT', 'Production', 'Production'],
        'Last Updated': ['2024-01-15', '2024-02-01', '2024-08-15', '2024-03-01', '2024-01-30'],
        'Performance': [0.918, 0.845, 0.762, 'N/A', 'N/A'],
        'Risk Rating': ['Low', 'Medium', 'High', 'Low', 'Low']
    })
    
    # Color code by risk rating
    def highlight_risk(val):
        if val == 'High':
            return 'background-color: #fee2e2'
        elif val == 'Medium':
            return 'background-color: #fef3cd'
        elif val == 'Low':
            return 'background-color: #d1fae5'
        else:
            return ''
    
    styled_models = models.style.applymap(highlight_risk, subset=['Risk Rating'])
    st.dataframe(styled_models, use_container_width=True)
    
    # Model dependency graph
    st.markdown("### 🔗 Model Dependency Graph")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Simplified dependency visualization
        st.markdown("""
        ```mermaid
        graph TD
            A[Policy Data] --> B[Mortality Model]
            A --> C[Lapse Model] 
            A --> D[Economic Model]
            B --> E[Reserve Calculator]
            C --> E
            D --> E
            E --> F[Capital Model]
            F --> G[Treaty Pricing]
            
            style B fill:#e1f5fe
            style C fill:#e1f5fe
            style D fill:#fff3e0
            style E fill:#e8f5e8
            style F fill:#e8f5e8
            style G fill:#f3e5f5
        ```
        """)
        
        st.info("🔵 ML Models | 🟡 Economic Models | 🟢 Actuarial Models | 🟣 Business Logic")
    
    with col2:
        st.markdown("### 📊 Model Complexity")
        
        complexity_metrics = pd.DataFrame({
            'Model': ['Mortality', 'Lapse', 'Economic', 'Reserve', 'Capital'],
            'Parameters': [500, 100, 2500, 50, 25],
            'Features': [23, 15, 8, 12, 6],
            'Complexity': ['High', 'Medium', 'Very High', 'Low', 'Low']
        })
        
        fig = px.scatter(
            complexity_metrics,
            x='Parameters',
            y='Features',
            size='Parameters',
            color='Complexity',
            hover_name='Model',
            title='Model Complexity Matrix'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model validation status
    st.markdown("### ✅ Model Validation Status")
    
    validation_status = pd.DataFrame({
        'Model': ['Mortality XGBoost', 'Lapse Random Forest', 'Economic LSTM', 'Reserve Calculator', 'Capital Model'],
        'Development': ['✅ Complete', '✅ Complete', '⚠️ In Progress', '✅ Complete', '✅ Complete'],
        'Validation': ['✅ Passed', '✅ Passed', '⏳ Pending', '✅ Passed', '✅ Passed'],
        'Documentation': ['✅ Complete', '⚠️ Update Due', '⏳ In Progress', '✅ Complete', '✅ Complete'],
        'Approval': ['✅ Approved', '✅ Approved', '❌ Not Ready', '✅ Approved', '✅ Approved'],
        'Next Review': ['Q1 2025', 'Q4 2024', 'Q3 2024', 'Q2 2025', 'Q1 2025']
    })
    
    # Apply styling
    def highlight_validation(val):
        if '✅' in str(val):
            return 'background-color: #d1fae5'
        elif '⚠️' in str(val):
            return 'background-color: #fef3cd'
        elif '⏳' in str(val):
            return 'background-color: #dbeafe'
        elif '❌' in str(val):
            return 'background-color: #fee2e2'
        else:
            return ''
    
    styled_validation = validation_status.style.applymap(highlight_validation)
    st.dataframe(styled_validation, use_container_width=True)
    
    # Model change log
    st.markdown("### 📝 Model Change Log")
    
    change_log = pd.DataFrame({
        'Date': ['2024-08-15', '2024-07-01', '2024-05-15', '2024-03-01', '2024-01-15'],
        'Model': ['Economic LSTM', 'Lapse RF', 'Reserve Calc', 'Capital Model', 'Mortality XGB'],
        'Change Type': ['Major Update', 'Parameter Tuning', 'Bug Fix', 'Enhancement', 'New Version'],
        'Description': [
            'Added cryptocurrency correlation features',
            'Adjusted regularization parameters',
            'Fixed leap year calculation error',
            'Added C4 business risk component',
            'Upgraded to XGBoost 2.0 with new features'
        ],
        'Impact': ['High', 'Low', 'Medium', 'Medium', 'High'],
        'Approver': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown', 'John Smith']
    })
    
    st.dataframe(change_log, use_container_width=True)
    
    # Export model documentation
    st.markdown("### 📥 Export Documentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Model inventory export
        inventory_csv = models.to_csv(index=False)
        st.download_button(
            label="📊 Model Inventory (CSV)",
            data=inventory_csv,
            file_name=f"model_inventory_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Validation status export
        validation_csv = validation_status.to_csv(index=False)
        st.download_button(
            label="✅ Validation Status (CSV)",
            data=validation_csv,
            file_name=f"validation_status_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Change log export
        changelog_csv = change_log.to_csv(index=False)
        st.download_button(
            label="📝 Change Log (CSV)",
            data=changelog_csv,
            file_name=f"model_changelog_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def regulatory_compliance_dashboard():
    """Regulatory compliance monitoring dashboard"""
    st.markdown("## 📋 Regulatory Compliance Dashboard")
    
    # Compliance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric("SOA Compliance", "✅ 100%", help="Society of Actuaries standards")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric("NAIC Compliance", "✅ 95%", delta="+5%", help="National Association of Insurance Commissioners")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric("Model Governance", "⚠️ 87%", delta="-3%", delta_color="inverse", help="Internal model governance standards")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-dashboard">', unsafe_allow_html=True)
        st.metric("Audit Readiness", "✅ Ready", help="External audit preparation status")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed compliance breakdown
    st.markdown("### 📊 Detailed Compliance Status")
    
    compliance_details = pd.DataFrame({
        'Regulation': [
            'SOA - Mortality Tables',
            'SOA - Experience Studies', 
            'SOA - Model Validation',
            'NAIC - RBC Requirements',
            'NAIC - Asset Adequacy Testing',
            'NAIC - PBR Implementation',
            'GAAP - LDTI Compliance',
            'Model Governance - SR 11-7',
            'Data Governance - GDPR',
            'Audit Trail - SOX'
        ],
        'Requirement': [
            'Use current approved tables',
            'Min 3 years credible data',
            'Annual back-testing required',
            'C1-C4 risk factors current',
            'Stress testing scenarios',
            'Principle-based reserves',
            'Cohort tracking implemented',
            'Model risk management',
            'Data privacy compliance',
            'Financial reporting controls'
        ],
        'Status': [
            '✅ Compliant',
            '✅ Compliant', 
            '✅ Compliant',
            '✅ Compliant',
            '⚠️ Due Soon',
            '✅ Compliant',
            '✅ Compliant',
            '⚠️ Gap Identified',
            '✅ Compliant',
            '✅ Compliant'
        ],
        'Last Review': [
            '2024-01-15',
            '2024-02-28',
            '2024-07-15',
            '2024-01-15',
            '2023-12-15',
            '2024-03-01',
            '2024-01-15',
            '2024-06-01',
            '2024-05-15',
            '2024-07-01'
        ],
        'Next Due': [
            '2025-01-15',
            '2024-11-30',
            '2024-10-15',
            '2025-01-15',
            '2024-09-15',
            '2025-03-01',
            '2025-01-15',
            '2024-09-01',
            '2024-11-15',
            '2024-10-01'
        ]
    })
    
    # Color coding compliance status
    def highlight_compliance(val):
        if '✅' in str(val):
            return 'background-color: #d1fae5'
        elif '⚠️' in str(val):
            return 'background-color: #fef3cd'
        elif '❌' in str(val):
            return 'background-color: #fee2e2'
        else:
            return ''
    
    styled_compliance = compliance_details.style.applymap(highlight_compliance, subset=['Status'])
    st.dataframe(styled_compliance, use_container_width=True)
    
    # Upcoming compliance deadlines
    st.markdown("### ⏰ Upcoming Compliance Deadlines")
    
    # Calculate days until next due dates
    today = datetime.now().date()
    compliance_details['Next Due'] = pd.to_datetime(compliance_details['Next Due']).dt.date
    compliance_details['Days Until Due'] = (compliance_details['Next Due'] - today).dt.days
    
    upcoming = compliance_details[compliance_details['Days Until Due'] <= 90].sort_values('Days Until Due')
    
    if not upcoming.empty:
        for _, item in upcoming.iterrows():
            days_left = item['Days Until Due']
            
            if days_left < 0:
                st.error(f"🚨 **OVERDUE**: {item['Regulation']} was due {abs(days_left)} days ago")
            elif days_left <= 30:
                st.error(f"🔴 **{days_left} days**: {item['Regulation']} - {item['Requirement']}")
            elif days_left <= 60:
                st.warning(f"🟡 **{days_left} days**: {item['Regulation']} - {item['Requirement']}")
            else:
                st.info(f"🔵 **{days_left} days**: {item['Regulation']} - {item['Requirement']}")
    else:
        st.success("✅ No compliance deadlines in the next 90 days")
    
    # Regulatory change monitoring
    st.markdown("### 📢 Recent Regulatory Updates")
    
    reg_updates = pd.DataFrame({
        'Date': ['2024-08-15', '2024-07-01', '2024-06-01', '2024-05-15'],
        'Source': ['NAIC', 'SOA', 'FASB', 'Fed'],
        'Update': [
            'Updated RBC factors for cyber risk',
            'New mortality improvement scales released',
            'GAAP LDTI implementation guidance',
            'Interest rate stress testing scenarios'
        ],
        'Impact': ['Medium', 'High', 'Medium', 'Low'],
        'Action Required': [
            'Update capital models by Q4 2024',
            'Review mortality assumptions',
            'Validate LDTI implementation', 
            'Update economic scenarios'
        ],
        'Status': ['⏳ In Progress', '✅ Complete', '✅ Complete', '⏳ In Progress']
    })
    
    st.dataframe(reg_updates, use_container_width=True)
    
    # Export compliance reports
    st.markdown("### 📥 Export Compliance Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compliance_csv = compliance_details.to_csv(index=False)
        st.download_button(
            label="📋 Full Compliance Report (CSV)",
            data=compliance_csv,
            file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        reg_updates_csv = reg_updates.to_csv(index=False)
        st.download_button(
            label="📢 Regulatory Updates (CSV)",
            data=reg_updates_csv,
            file_name=f"regulatory_updates_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    professional_workbench()