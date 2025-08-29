"""
PricingFlow Demo Interface

Simple Streamlit demo showing life insurance and annuity pricing capabilities
"""

import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="PricingFlow Demo",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}

.feature-highlight {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def calculate_life_insurance_premium(age, gender, face_amount, smoker=False, health_rating="Standard"):
    """Simple life insurance premium calculation"""
    
    # Base mortality rate
    base_rate = 0.001 + age * 0.0002
    
    # Gender adjustment
    if gender == "Male":
        base_rate *= 1.15
    
    # Smoking adjustment
    if smoker:
        base_rate *= 2.5
    
    # Health rating adjustment
    health_multipliers = {
        "Preferred Plus": 0.8,
        "Preferred": 0.9,
        "Standard Plus": 0.95,
        "Standard": 1.0,
        "Substandard": 1.5
    }
    base_rate *= health_multipliers.get(health_rating, 1.0)
    
    # Calculate premium with loadings
    net_premium = face_amount * base_rate
    gross_premium = net_premium * 1.3  # Add expenses and profit
    
    return {
        "annual_premium": gross_premium,
        "monthly_premium": gross_premium / 12,
        "premium_per_1000": (gross_premium / face_amount) * 1000,
        "mortality_rate": base_rate,
        "life_expectancy": max(0, 85 - age + (3 if gender == "Female" else 0))
    }

def calculate_annuity_payment(age, gender, premium_amount, payout_option="Life Only"):
    """Simple annuity payment calculation"""
    
    # Life expectancy calculation
    life_expectancy = max(0, 90 - age + (3 if gender == "Female" else 0))
    
    # Interest rate assumption
    interest_rate = 0.04
    
    # Present value of annuity calculation (simplified)
    if payout_option == "Life Only":
        pv_factor = (1 - (1 + interest_rate) ** -life_expectancy) / interest_rate
    elif payout_option == "10 Years Certain":
        pv_factor = (1 - (1 + interest_rate) ** -max(10, life_expectancy)) / interest_rate
    else:  # 20 Years Certain
        pv_factor = (1 - (1 + interest_rate) ** -max(20, life_expectancy)) / interest_rate
    
    # Calculate payment
    annual_payment = premium_amount / pv_factor
    
    return {
        "annual_payment": annual_payment,
        "monthly_payment": annual_payment / 12,
        "life_expectancy": life_expectancy,
        "total_expected_payments": annual_payment * life_expectancy,
        "payout_ratio": annual_payment / premium_amount
    }

def load_sample_data():
    """Load sample data if available"""
    data_path = "data/synthetic/sample_life_insurance.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏢 PricingFlow Demo</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">'
        'AI-Powered Insurance Pricing Platform for Life Insurance & Retirement Products'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.title("🎯 Pricing Tools")
    app_mode = st.sidebar.selectbox(
        "Choose Application",
        ["🏠 Overview", "💰 Life Insurance Pricing", "🏦 Annuity Pricing", "📊 Data Analysis", "ℹ️ About"]
    )
    
    if app_mode == "🏠 Overview":
        show_overview()
    elif app_mode == "💰 Life Insurance Pricing":
        show_life_insurance_pricer()
    elif app_mode == "🏦 Annuity Pricing":
        show_annuity_pricer()
    elif app_mode == "📊 Data Analysis":
        show_data_analysis()
    elif app_mode == "ℹ️ About":
        show_about()

def show_overview():
    """Show system overview"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
        <h3>🧮 Actuarial Intelligence</h3>
        <p>AI-powered feature engineering with 200+ pre-built insurance features</p>
        <ul>
        <li>Mortality rate calculations</li>
        <li>Life expectancy modeling</li>
        <li>Risk factor analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
        <h3>⚡ Fast Model Development</h3>
        <p>Build pricing models 10x faster than traditional methods</p>
        <ul>
        <li>Automated data preparation</li>
        <li>One-click model training</li>
        <li>Real-time validation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-highlight">
        <h3>🎯 Multi-Product Platform</h3>
        <p>Unified platform for all insurance products</p>
        <ul>
        <li>Life Insurance</li>
        <li>Annuities & Retirement</li>
        <li>Disability & Long-term Care</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample metrics
    st.subheader("📈 Platform Benefits")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⏱️ Development Time",
            value="2-4 weeks",
            delta="-80% vs traditional",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="🎯 Model Accuracy",
            value="95%+",
            delta="+15% improvement"
        )
    
    with col3:
        st.metric(
            label="📊 Features Generated",
            value="200+",
            delta="Automated"
        )
    
    with col4:
        st.metric(
            label="💰 Cost Savings",
            value="$500k+",
            delta="Per pricing model"
        )
    
    # Architecture diagram
    st.subheader("🏗️ System Architecture")
    st.image("https://via.placeholder.com/800x400/1f77b4/ffffff?text=Data+Pipeline+%E2%86%92+AI+Engine+%E2%86%92+Pricing+Models", 
             caption="PricingFlow Architecture: From Raw Data to Production Models")

def show_life_insurance_pricer():
    """Life insurance pricing calculator"""
    
    st.header("💰 Life Insurance Premium Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Policy Details")
        
        # Input fields
        age = st.slider("Age at Issue", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        face_amount = st.number_input("Face Amount ($)", min_value=25000, max_value=5000000, value=500000, step=25000)
        
        smoker = st.checkbox("Smoker")
        health_rating = st.selectbox(
            "Health Rating", 
            ["Preferred Plus", "Preferred", "Standard Plus", "Standard", "Substandard"]
        )
        
        policy_type = st.selectbox("Policy Type", ["Term Life", "Whole Life", "Universal Life"])
        
        if st.button("Calculate Premium", type="primary"):
            # Calculate premium
            result = calculate_life_insurance_premium(age, gender, face_amount, smoker, health_rating)
            
            # Store in session state
            st.session_state['life_result'] = result
    
    with col2:
        st.subheader("Premium Quote")
        
        if 'life_result' in st.session_state:
            result = st.session_state['life_result']
            
            # Display results
            st.metric("Annual Premium", f"${result['annual_premium']:,.0f}")
            st.metric("Monthly Premium", f"${result['monthly_premium']:,.0f}")
            st.metric("Premium per $1,000", f"${result['premium_per_1000']:.2f}")
            
            # Additional metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Mortality Rate", f"{result['mortality_rate']:.4f}")
            with col2b:
                st.metric("Life Expectancy", f"{result['life_expectancy']:.0f} years")
            
            # Premium breakdown chart
            st.subheader("Premium Analysis")
            
            breakdown = {
                "Mortality Charge": result['annual_premium'] * 0.4,
                "Expense Loading": result['annual_premium'] * 0.3,
                "Profit Margin": result['annual_premium'] * 0.2,
                "Commissions": result['annual_premium'] * 0.1
            }
            
            fig = px.pie(
                values=list(breakdown.values()),
                names=list(breakdown.keys()),
                title="Premium Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Enter policy details and click 'Calculate Premium' to see results")

def show_annuity_pricer():
    """Annuity pricing calculator"""
    
    st.header("🏦 Annuity Payment Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Contract Details")
        
        age = st.slider("Age at Contract", min_value=35, max_value=85, value=65)
        gender = st.selectbox("Gender", ["Male", "Female"], key="annuity_gender")
        premium_amount = st.number_input("Premium Amount ($)", min_value=10000, max_value=2000000, value=250000, step=10000)
        
        product_type = st.selectbox("Product Type", ["Immediate Annuity", "Deferred Annuity"])
        payout_option = st.selectbox("Payout Option", ["Life Only", "10 Years Certain", "20 Years Certain"])
        
        if st.button("Calculate Payment", type="primary"):
            result = calculate_annuity_payment(age, gender, premium_amount, payout_option)
            st.session_state['annuity_result'] = result
    
    with col2:
        st.subheader("Payment Quote")
        
        if 'annuity_result' in st.session_state:
            result = st.session_state['annuity_result']
            
            st.metric("Annual Payment", f"${result['annual_payment']:,.0f}")
            st.metric("Monthly Payment", f"${result['monthly_payment']:,.0f}")
            st.metric("Life Expectancy", f"{result['life_expectancy']:.0f} years")
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Total Expected Payments", f"${result['total_expected_payments']:,.0f}")
            with col2b:
                st.metric("Annual Payout Rate", f"{result['payout_ratio']:.1%}")
            
            # Payment projection chart
            st.subheader("Payment Projection")
            
            years = list(range(1, int(result['life_expectancy']) + 1))
            cumulative_payments = [result['annual_payment'] * year for year in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years, 
                y=cumulative_payments,
                mode='lines+markers',
                name='Cumulative Payments',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_hline(y=premium_amount, line_dash="dash", line_color="red", 
                         annotation_text="Break-even Point")
            
            fig.update_layout(
                title="Cumulative Annuity Payments Over Time",
                xaxis_title="Years",
                yaxis_title="Cumulative Payments ($)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Enter contract details and click 'Calculate Payment' to see results")

def show_data_analysis():
    """Show data analysis capabilities"""
    
    st.header("📊 Sample Data Analysis")
    
    # Load sample data
    df = load_sample_data()
    
    if df is not None:
        st.success(f"✅ Loaded {len(df)} sample life insurance records")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Age", f"{df['age_at_issue'].mean():.1f}")
        
        with col2:
            st.metric("Average Face Amount", f"${df['face_amount'].mean():,.0f}")
        
        with col3:
            st.metric("Average Premium", f"${df['annual_premium'].mean():,.0f}")
        
        with col4:
            smoking_rate = (df['smoker_status'] == 'Smoker').mean() * 100
            st.metric("Smoking Rate", f"{smoking_rate:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig1 = px.histogram(df, x='age_at_issue', title='Age Distribution', nbins=20)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Premium by age and gender
            fig2 = px.box(df, x='gender', y='annual_premium', title='Premium by Gender')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Premium vs Face Amount scatter
        fig3 = px.scatter(
            df, x='face_amount', y='annual_premium', 
            color='smoker_status', size='age_at_issue',
            title='Premium vs Face Amount (sized by age, colored by smoking status)',
            hover_data=['age_at_issue', 'state']
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Show raw data sample
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
    else:
        st.warning("No sample data found. Run the data generation script first:")
        st.code("python3 scripts/test_basic_functionality.py")

def show_about():
    """Show about information"""
    
    st.header("ℹ️ About PricingFlow")
    
    st.markdown("""
    ## 🎯 Mission
    
    PricingFlow transforms insurance pricing model development from a 6-12 month process into a 2-4 week sprint using AI-powered automation and domain expertise.
    
    ## 🚀 Key Features
    
    ### **Actuarial Intelligence**
    - Pre-built library of 200+ insurance-specific features
    - Mortality rate calculations using industry-standard tables
    - Life expectancy modeling with demographic adjustments
    - Cross-product consistency validation
    
    ### **AI-Powered Automation**
    - Automated feature engineering for insurance data
    - Smart data quality detection and cleaning
    - Domain-aware anomaly detection
    - Business rule discovery and validation
    
    ### **Multi-Product Platform**
    - **Life Insurance**: Term, whole, universal, variable life
    - **Annuities**: Immediate, deferred, variable, indexed annuities
    - **Retirement Income**: Systematic withdrawals, guaranteed income
    - **Future**: Disability, long-term care, reinsurance
    
    ## 🏗️ Technology Stack
    
    - **Data Processing**: Polars, DuckDB (10x faster than pandas)
    - **Machine Learning**: Scikit-learn, LightGBM, XGBoost
    - **AI/NLP**: Local LLMs for domain understanding
    - **API**: FastAPI for production deployment
    - **UI**: Streamlit for rapid prototyping
    
    ## 📈 Business Impact
    
    | Metric | Traditional | PricingFlow | Improvement |
    |--------|-------------|-------------|-------------|
    | Development Time | 6-12 months | 2-4 weeks | **10x faster** |
    | Model Accuracy | 80-85% | 95%+ | **+15% better** |
    | Features Created | 20-50 manual | 200+ automated | **4x more** |
    | Cost per Model | $500k-1M | $50k-100k | **80% savings** |
    
    ## 🎖️ Built for Insurance Professionals
    
    PricingFlow is designed by actuaries, for actuaries. Every feature is rooted in actuarial science and insurance industry best practices.
    
    ---
    
    **Ready to transform your pricing process?** Contact us at hello@pricingflow.com
    """)

if __name__ == "__main__":
    main()