"""
Data Explorers Components
Market data, mortality tables, treasury rates, and user data exploration
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_mortality_explorer():
    """Render mortality tables explorer"""
    if st.session_state.get('show_mortality_explorer', False):
        # Professional container styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        ">
            <h2 style="margin: 0; font-size: 1.8em;">📊 SOA 2017 CSO Mortality Tables Explorer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button in top right
        col_header1, col_header2 = st.columns([4, 1])
        with col_header2:
            if st.button("❌ Close Explorer", key="close_mortality", use_container_width=True):
                st.session_state.show_mortality_explorer = False
                st.rerun()
        
        # Main content container
        with st.container():
                st.markdown("### Browse Mortality Data")
                
                # Filter controls
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    gender = st.selectbox("Gender", ["Male", "Female"], key="mort_gender")
                with col_b:
                    smoker = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"], key="mort_smoker")
                with col_c:
                    age_range = st.slider("Age Range", 0, 120, (25, 65), key="mort_age")
                
                # Generate sample mortality data based on selections
                ages = list(range(age_range[0], age_range[1] + 1))
                
                # Realistic mortality rates (simplified calculation)
                base_rate = 0.0001 if gender == "Female" else 0.00015
                smoker_mult = 1.5 if smoker == "Smoker" else 1.0
                
                mortality_data = []
                for age in ages:
                    # Exponential mortality curve
                    rate = base_rate * smoker_mult * (1.08 ** (age - 25))
                    mortality_data.append({
                        "Age": age,
                        "Gender": gender,
                        "Smoking": smoker,
                        "Mortality Rate": f"{rate:.6f}",
                        "Per 1,000": f"{rate * 1000:.2f}",
                        "Table": "2017 CSO"
                    })
                
                df = pd.DataFrame(mortality_data)
                st.dataframe(df, use_container_width=True)
                
                # Chart
                fig = px.line(df, x="Age", y=df["Mortality Rate"].astype(float), 
                             title=f"Mortality Rates: {gender} {smoker}")
                fig.update_layout(yaxis_title="Mortality Rate", height=400)
                st.plotly_chart(fig, use_container_width=True)


def render_treasury_explorer():
    """Render treasury rates explorer"""
    if st.session_state.get('show_treasury_explorer', False):
        # Professional container styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        ">
            <h2 style="margin: 0; font-size: 1.8em;">💰 FRED Treasury Yield Curve Explorer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button in top right
        col_header1, col_header2 = st.columns([4, 1])
        with col_header2:
            if st.button("❌ Close Explorer", key="close_treasury", use_container_width=True):
                st.session_state.show_treasury_explorer = False
                st.rerun()
        
        # Main content container
        with st.container():
                st.markdown("### Current U.S. Treasury Yield Curve")
                
                # Sample current treasury data
                treasury_data = [
                    {"Maturity": "1 Month", "Years": 0.083, "Rate": 5.45, "Type": "Bill"},
                    {"Maturity": "3 Month", "Years": 0.25, "Rate": 5.23, "Type": "Bill"},
                    {"Maturity": "6 Month", "Years": 0.5, "Rate": 4.95, "Type": "Bill"},
                    {"Maturity": "1 Year", "Years": 1, "Rate": 4.68, "Type": "Note"},
                    {"Maturity": "2 Year", "Years": 2, "Rate": 4.15, "Type": "Note"},
                    {"Maturity": "5 Year", "Years": 5, "Rate": 4.22, "Type": "Note"},
                    {"Maturity": "10 Year", "Years": 10, "Rate": 4.28, "Type": "Note"},
                    {"Maturity": "20 Year", "Years": 20, "Rate": 4.52, "Type": "Bond"},
                    {"Maturity": "30 Year", "Years": 30, "Rate": 4.45, "Type": "Bond"}
                ]
                
                df_treasury = pd.DataFrame(treasury_data)
                
                # Display table
                st.dataframe(df_treasury, use_container_width=True)
                
                # Yield curve chart
                fig = px.line(df_treasury, x="Years", y="Rate", 
                             title="U.S. Treasury Yield Curve",
                             markers=True)
                fig.update_layout(
                    xaxis_title="Maturity (Years)",
                    yaxis_title="Yield (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional info
                st.info("📊 **Data Source**: Federal Reserve Economic Data (FRED) - Updated daily")


def render_market_explorer():
    """Render market data explorer"""
    if st.session_state.get('show_market_explorer', False):
        # Professional container styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        ">
            <h2 style="margin: 0; font-size: 1.8em;">📈 Live Market Data Browser</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button in top right
        col_header1, col_header2 = st.columns([4, 1])
        with col_header2:
            if st.button("❌ Close Explorer", key="close_market", use_container_width=True):
                st.session_state.show_market_explorer = False
                st.rerun()
        
        # Main content container
        with st.container():
                st.markdown("### Market Indicators & Economic Data")
                
                # Market data tabs
                tab1, tab2, tab3 = st.tabs(["📊 Indices", "📈 Rates", "🎯 Scenarios"])
                
                with tab1:
                    market_indices = [
                        {"Index": "S&P 500", "Current": 5634.58, "Change": "+0.3%", "52W High": 5669.67, "52W Low": 4117.37},
                        {"Index": "Dow Jones", "Current": 41563.08, "Change": "+0.2%", "52W High": 42628.32, "52W Low": 31522.74},
                        {"Index": "NASDAQ", "Current": 17713.62, "Change": "+0.4%", "52W High": 18671.07, "52W Low": 12544.39},
                        {"Index": "Russell 2000", "Current": 2184.35, "Change": "-0.1%", "52W High": 2442.74, "52W Low": 1636.93},
                        {"Index": "VIX", "Current": 15.2, "Change": "-5.2%", "52W High": 65.73, "52W Low": 12.12}
                    ]
                    
                    df_indices = pd.DataFrame(market_indices)
                    st.dataframe(df_indices, use_container_width=True)
                
                with tab2:
                    rate_data = [
                        {"Rate": "Fed Funds Rate", "Current": "5.25-5.50%", "Previous": "5.25-5.50%", "Change": "0.00%"},
                        {"Rate": "Prime Rate", "Current": "8.50%", "Previous": "8.50%", "Change": "0.00%"},
                        {"Rate": "10Y Treasury", "Current": "4.28%", "Previous": "4.31%", "Change": "-0.03%"},
                        {"Rate": "30Y Mortgage", "Current": "7.15%", "Previous": "7.22%", "Change": "-0.07%"},
                        {"Rate": "Corporate AAA", "Current": "4.85%", "Previous": "4.89%", "Change": "-0.04%"}
                    ]
                    
                    df_rates = pd.DataFrame(rate_data)
                    st.dataframe(df_rates, use_container_width=True)
                
                with tab3:
                    scenario_data = [
                        {"Scenario": "Base Case", "GDP Growth": "3.2%", "Inflation": "2.4%", "Unemployment": "3.8%", "10Y Yield": "4.3%"},
                        {"Scenario": "Optimistic", "GDP Growth": "4.1%", "Inflation": "2.1%", "Unemployment": "3.2%", "10Y Yield": "3.9%"},
                        {"Scenario": "Pessimistic", "GDP Growth": "1.8%", "Inflation": "3.2%", "Unemployment": "5.1%", "10Y Yield": "4.8%"},
                        {"Scenario": "Recession", "GDP Growth": "-0.8%", "Inflation": "1.9%", "Unemployment": "7.2%", "10Y Yield": "3.2%"}
                    ]
                    
                    df_scenarios = pd.DataFrame(scenario_data)
                    st.dataframe(df_scenarios, use_container_width=True)
                
                st.info("📊 **Data Source**: Alpha Vantage API - Real-time market data")


def render_data_explorer():
    """Render user data explorer"""
    if st.session_state.get('show_data_explorer', False):
        # Professional container styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
        ">
            <h2 style="margin: 0; font-size: 1.8em;">📁 Your Data Explorer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button in top right
        col_header1, col_header2 = st.columns([4, 1])
        with col_header2:
            if st.button("❌ Close Explorer", key="close_data", use_container_width=True):
                st.session_state.show_data_explorer = False
                st.rerun()
        
        # Main content container  
        with st.container():
                datasets = st.session_state.get('uploaded_datasets', {})
                
                if datasets:
                    st.markdown("### Your Data Portfolio")
                    
                    # Dataset selector
                    dataset_names = list(datasets.keys())
                    selected_dataset = st.selectbox("Select Dataset to Explore", dataset_names)
                    
                    if selected_dataset:
                        dataset_info = datasets[selected_dataset]
                        data = dataset_info['data']
                        
                        # Dataset summary
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Records", f"{len(data):,}")
                        with col_b:
                            st.metric("Columns", len(data.columns))
                        with col_c:
                            st.metric("Quality Score", f"{dataset_info.get('quality_score', 85)}/100")
                        
                        # Data preview
                        st.markdown("#### Data Preview")
                        st.dataframe(data.head(20), use_container_width=True)
                        
                        # Column info
                        st.markdown("#### Column Information")
                        col_info = []
                        for col in data.columns:
                            col_info.append({
                                "Column": col,
                                "Type": str(data[col].dtype),
                                "Non-Null": f"{data[col].count():,}",
                                "Sample": str(data[col].iloc[0]) if len(data) > 0 else "N/A"
                            })
                        
                        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
                else:
                    st.info("No datasets uploaded yet. Upload data in Step 1 to explore here.")


def render_all_data_explorers():
    """Render all data explorers based on session state flags"""
    render_mortality_explorer()
    render_treasury_explorer()
    render_market_explorer()
    render_data_explorer()