"""
ML-Enhanced Economic Forecasting for Actuarial Modeling
Predicts interest rates, yield curves, and economic scenarios for pricing and reserving
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import VectorAutoregression
    VAR_AVAILABLE = True
except ImportError:
    VAR_AVAILABLE = False

try:
    import yfinance as yf
    YAHOO_FINANCE_AVAILABLE = True
except ImportError:
    YAHOO_FINANCE_AVAILABLE = False

class EconomicForecastingEngine:
    """ML-powered economic scenario generation for actuarial modeling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        
        # Economic indicators to track
        self.economic_indicators = {
            'interest_rates': ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y'],
            'equity_indices': ['^GSPC', '^IXIC', '^DJI'],  # S&P500, NASDAQ, DOW
            'volatility_indices': ['^VIX'],
            'commodity_prices': ['GC=F', 'CL=F'],  # Gold, Oil
            'fx_rates': ['EURUSD=X', 'GBPUSD=X']
        }
        
        # Historical relationships for fallback
        self.historical_correlations = {
            ('10Y_rate', 'equity_return'): -0.3,
            ('vix', 'equity_return'): -0.7,
            ('inflation', '10Y_rate'): 0.6,
            ('unemployment', 'equity_return'): -0.4
        }
        
        # Economic regimes
        self.economic_regimes = {
            'expansion': {'prob': 0.6, 'duration_months': 60, 'gdp_growth': 0.025},
            'recession': {'prob': 0.15, 'duration_months': 12, 'gdp_growth': -0.015},
            'recovery': {'prob': 0.2, 'duration_months': 24, 'gdp_growth': 0.035},
            'stagflation': {'prob': 0.05, 'duration_months': 36, 'gdp_growth': 0.005}
        }
    
    def load_economic_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load historical economic data for training"""
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            if YAHOO_FINANCE_AVAILABLE:
                # Use real market data
                return self._load_real_market_data(start_date, end_date)
            else:
                # Generate synthetic historical data
                return self._generate_synthetic_economic_data(start_date, end_date)
                
        except Exception as e:
            self.logger.error(f"Error loading economic data: {e}")
            return self._generate_synthetic_economic_data(start_date, end_date)
    
    def train_interest_rate_models(
        self,
        economic_data: pd.DataFrame,
        forecast_horizon: int = 12
    ) -> Dict[str, Any]:
        """Train ML models for interest rate forecasting"""
        
        results = {}
        
        # Prepare features and targets
        features, targets = self._prepare_rate_modeling_data(
            economic_data, forecast_horizon
        )
        
        # Train models for each maturity
        for maturity in ['3M', '1Y', '5Y', '10Y', '30Y']:
            if f'{maturity}_rate' in targets.columns:
                
                target = targets[f'{maturity}_rate'].dropna()
                feature_subset = features.loc[target.index]
                
                if len(target) < 100:  # Insufficient data
                    continue
                
                # Split data (80% train, 20% test)
                split_idx = int(len(target) * 0.8)
                X_train, X_test = feature_subset[:split_idx], feature_subset[split_idx:]
                y_train, y_test = target[:split_idx], target[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
                
                rf_model.fit(X_train_scaled, y_train)
                predictions = rf_model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                
                # Store model and results
                model_key = f'interest_rate_{maturity.lower()}'
                self.models[model_key] = rf_model
                self.scalers[model_key] = scaler
                
                results[maturity] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'feature_importance': dict(zip(
                        features.columns, rf_model.feature_importances_
                    ))
                }
                
                self.logger.info(f"Trained {maturity} rate model - RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def train_equity_models(self, economic_data: pd.DataFrame) -> Dict[str, Any]:
        """Train models for equity market forecasting"""
        
        # Prepare equity features
        equity_features = self._prepare_equity_features(economic_data)
        
        # Calculate equity returns
        if 'sp500' in economic_data.columns:
            equity_returns = economic_data['sp500'].pct_change(periods=252).dropna()  # Annual returns
        else:
            # Synthetic equity returns
            equity_returns = pd.Series(
                np.random.normal(0.08, 0.16, len(economic_data)),
                index=economic_data.index
            )
        
        # Align features and targets
        common_index = equity_features.index.intersection(equity_returns.index)
        X = equity_features.loc[common_index]
        y = equity_returns.loc[common_index]
        
        if len(X) < 50:
            self.logger.warning("Insufficient data for equity model training")
            return {'equity': {'error': 'Insufficient data'}}
        
        # Train model
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        equity_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        equity_model.fit(X_train_scaled, y_train)
        predictions = equity_model.predict(X_test_scaled)
        
        # Store model
        self.models['equity_returns'] = equity_model
        self.scalers['equity_returns'] = scaler
        
        return {
            'equity': {
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'feature_importance': dict(zip(X.columns, equity_model.feature_importances_))
            }
        }
    
    def generate_economic_scenarios(
        self,
        num_scenarios: int = 1000,
        time_horizon_years: int = 30,
        scenario_type: str = 'stochastic'
    ) -> Dict[str, np.ndarray]:
        """Generate economic scenarios for actuarial modeling"""
        
        try:
            if scenario_type == 'deterministic':
                return self._generate_deterministic_scenarios(time_horizon_years)
            elif scenario_type == 'stochastic':
                return self._generate_stochastic_scenarios(num_scenarios, time_horizon_years)
            elif scenario_type == 'stress':
                return self._generate_stress_scenarios(time_horizon_years)
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
                
        except Exception as e:
            self.logger.error(f"Error generating scenarios: {e}")
            return self._generate_fallback_scenarios(num_scenarios, time_horizon_years)
    
    def forecast_yield_curve(
        self,
        current_rates: Dict[str, float],
        forecast_months: int = 12
    ) -> Dict[str, np.ndarray]:
        """Forecast entire yield curve evolution"""
        
        maturities = ['3M', '1Y', '2Y', '5Y', '10Y', '30Y']
        forecasts = {}
        
        for maturity in maturities:
            model_key = f'interest_rate_{maturity.lower()}'
            
            if model_key in self.models and model_key in self.scalers:
                # Use trained ML model
                forecasts[maturity] = self._ml_rate_forecast(
                    maturity, current_rates, forecast_months
                )
            else:
                # Use economic theory-based forecast
                forecasts[maturity] = self._theory_based_forecast(
                    maturity, current_rates.get(maturity, 0.03), forecast_months
                )
        
        return forecasts
    
    def assess_economic_regime(
        self,
        current_indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess current economic regime and transition probabilities"""
        
        # Economic indicators analysis
        gdp_growth = current_indicators.get('gdp_growth', 0.02)
        unemployment = current_indicators.get('unemployment', 0.04)
        inflation = current_indicators.get('inflation', 0.02)
        yield_curve_slope = current_indicators.get('yield_curve_slope', 0.01)
        
        # Regime scoring
        regime_scores = {}
        
        # Expansion: positive growth, low unemployment, normal inflation
        expansion_score = (
            max(0, gdp_growth) * 2 +
            max(0, 0.08 - unemployment) * 3 +
            max(0, 0.03 - abs(inflation - 0.02)) * 2 +
            max(0, yield_curve_slope) * 1
        )
        regime_scores['expansion'] = min(1.0, expansion_score / 10)
        
        # Recession: negative growth, high unemployment
        recession_score = (
            max(0, -gdp_growth) * 4 +
            max(0, unemployment - 0.05) * 3 +
            max(0, -yield_curve_slope) * 2
        )
        regime_scores['recession'] = min(1.0, recession_score / 8)
        
        # Recovery: improving but below trend
        recovery_score = (
            max(0, gdp_growth) * 1.5 +
            max(0, 0.06 - unemployment) * 2 +
            max(0, yield_curve_slope) * 1.5
        ) if gdp_growth > 0 and unemployment > 0.045 else 0
        regime_scores['recovery'] = min(1.0, recovery_score / 6)
        
        # Stagflation: low growth, high inflation
        stagflation_score = (
            max(0, inflation - 0.04) * 4 +
            max(0, 0.01 - gdp_growth) * 2
        )
        regime_scores['stagflation'] = min(1.0, stagflation_score / 6)
        
        # Normalize probabilities
        total_score = sum(regime_scores.values())
        if total_score > 0:
            regime_probs = {k: v/total_score for k, v in regime_scores.items()}
        else:
            regime_probs = {'expansion': 0.6, 'recession': 0.15, 'recovery': 0.2, 'stagflation': 0.05}
        
        # Determine most likely regime
        current_regime = max(regime_probs.keys(), key=lambda x: regime_probs[x])
        
        return {
            'current_regime': current_regime,
            'regime_probabilities': regime_probs,
            'confidence': max(regime_probs.values()),
            'transition_matrix': self._get_regime_transition_matrix(),
            'expected_duration_months': self.economic_regimes[current_regime]['duration_months']
        }
    
    def _load_real_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load real market data using yfinance"""
        
        data_dict = {}
        
        # Download equity data
        try:
            sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
            data_dict['sp500'] = sp500
            
            vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
            data_dict['vix'] = vix
        except:
            pass
        
        # Create combined dataset
        if data_dict:
            df = pd.DataFrame(data_dict)
            df = df.fillna(method='ffill').dropna()
            
            # Add calculated features
            df['sp500_return'] = df['sp500'].pct_change(periods=252)  # Annual returns
            df['vix_level'] = df.get('vix', 20)  # VIX level
            
            return df
        else:
            raise Exception("Could not load real market data")
    
    def _generate_synthetic_economic_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Generate realistic synthetic economic data"""
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_periods = len(dates)
        
        # Interest rate simulation using Vasicek model
        rates_3m = self._simulate_vasicek_rates(n_periods, initial_rate=0.01, mean_rate=0.025)
        rates_10y = self._simulate_vasicek_rates(n_periods, initial_rate=0.03, mean_rate=0.035)
        rates_30y = self._simulate_vasicek_rates(n_periods, initial_rate=0.035, mean_rate=0.040)
        
        # Equity returns simulation
        sp500_returns = np.random.normal(0.08/252, 0.16/np.sqrt(252), n_periods)
        sp500_levels = 3000 * np.exp(np.cumsum(sp500_returns))
        
        # VIX simulation
        vix_levels = np.maximum(10, 20 + np.cumsum(np.random.normal(0, 0.1, n_periods)))
        
        # Create DataFrame
        data = pd.DataFrame({
            '3M_rate': rates_3m,
            '1Y_rate': rates_3m * 1.1 + np.random.normal(0, 0.002, n_periods),
            '5Y_rate': rates_3m * 1.3 + np.random.normal(0, 0.003, n_periods),
            '10Y_rate': rates_10y,
            '30Y_rate': rates_30y,
            'sp500': sp500_levels,
            'vix': vix_levels,
            'inflation': np.maximum(0, 0.02 + np.random.normal(0, 0.005, n_periods)),
            'unemployment': np.maximum(0.03, 0.05 + np.random.normal(0, 0.002, n_periods))
        }, index=dates)
        
        # Add calculated features
        data['sp500_return'] = data['sp500'].pct_change(periods=252)
        data['yield_curve_slope'] = data['10Y_rate'] - data['3M_rate']
        data['term_spread'] = data['30Y_rate'] - data['10Y_rate']
        data['credit_spread'] = np.maximum(0.005, 0.01 + np.random.normal(0, 0.002, n_periods))
        
        return data.fillna(method='ffill').dropna()
    
    def _simulate_vasicek_rates(
        self, 
        n_periods: int, 
        initial_rate: float, 
        mean_rate: float,
        speed: float = 0.1,
        volatility: float = 0.01
    ) -> np.ndarray:
        """Simulate interest rates using Vasicek model"""
        
        dt = 1/252  # Daily time step
        rates = np.zeros(n_periods)
        rates[0] = initial_rate
        
        for t in range(1, n_periods):
            dr = speed * (mean_rate - rates[t-1]) * dt + volatility * np.sqrt(dt) * np.random.normal()
            rates[t] = max(0.001, rates[t-1] + dr)  # Floor at 0.1%
        
        return rates
    
    def _prepare_rate_modeling_data(
        self, 
        data: pd.DataFrame, 
        forecast_horizon: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets for interest rate modeling"""
        
        # Features (lagged variables)
        features = pd.DataFrame(index=data.index)
        
        # Current level features
        rate_cols = [col for col in data.columns if '_rate' in col]
        for col in rate_cols:
            if col in data.columns:
                features[f'{col}_current'] = data[col]
                features[f'{col}_lag1'] = data[col].shift(1)
                features[f'{col}_lag5'] = data[col].shift(5)
                features[f'{col}_ma10'] = data[col].rolling(10).mean()
        
        # Economic indicators
        if 'sp500_return' in data.columns:
            features['equity_return'] = data['sp500_return']
            features['equity_volatility'] = data['sp500'].pct_change().rolling(20).std()
        
        if 'vix' in data.columns:
            features['vix_level'] = data['vix']
            features['vix_change'] = data['vix'].diff()
        
        if 'inflation' in data.columns:
            features['inflation'] = data['inflation']
            features['real_rate_10y'] = data.get('10Y_rate', 0.03) - data['inflation']
        
        # Yield curve features
        if 'yield_curve_slope' in data.columns:
            features['yield_curve_slope'] = data['yield_curve_slope']
            features['curve_level'] = data.get('10Y_rate', 0.03)
            features['curve_curvature'] = (
                data.get('5Y_rate', 0.03) - 
                (data.get('3M_rate', 0.02) + data.get('30Y_rate', 0.04)) / 2
            )
        
        # Targets (forward rates)
        targets = pd.DataFrame(index=data.index)
        for col in rate_cols:
            if col in data.columns:
                targets[col] = data[col].shift(-forecast_horizon)
        
        return features.fillna(method='ffill').dropna(), targets
    
    def _prepare_equity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for equity modeling"""
        
        features = pd.DataFrame(index=data.index)
        
        # Interest rate features
        if '10Y_rate' in data.columns:
            features['long_rate'] = data['10Y_rate']
            features['rate_change'] = data['10Y_rate'].diff()
        
        if '3M_rate' in data.columns and '10Y_rate' in data.columns:
            features['yield_spread'] = data['10Y_rate'] - data['3M_rate']
        
        # Volatility features
        if 'vix' in data.columns:
            features['volatility'] = data['vix']
            features['vol_regime'] = (data['vix'] > data['vix'].rolling(252).mean()).astype(int)
        
        # Economic features
        if 'inflation' in data.columns:
            features['inflation'] = data['inflation']
        
        if 'unemployment' in data.columns:
            features['unemployment'] = data['unemployment']
        
        # Technical features
        if 'sp500' in data.columns:
            features['momentum'] = data['sp500'].pct_change(20)  # 20-day momentum
            features['rsi'] = self._calculate_rsi(data['sp500'])
        
        return features.fillna(method='ffill').dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _generate_stochastic_scenarios(
        self, 
        num_scenarios: int, 
        time_horizon_years: int
    ) -> Dict[str, np.ndarray]:
        """Generate stochastic economic scenarios"""
        
        n_periods = time_horizon_years * 12  # Monthly scenarios
        
        # Initialize scenario arrays
        scenarios = {
            'short_rates': np.zeros((num_scenarios, n_periods)),
            'long_rates': np.zeros((num_scenarios, n_periods)),
            'equity_returns': np.zeros((num_scenarios, n_periods)),
            'inflation': np.zeros((num_scenarios, n_periods))
        }
        
        # Generate scenarios
        for scenario in range(num_scenarios):
            # Interest rates using Vasicek
            scenarios['short_rates'][scenario] = self._simulate_vasicek_rates(
                n_periods, 0.02, 0.025, speed=0.2, volatility=0.015
            )
            scenarios['long_rates'][scenario] = self._simulate_vasicek_rates(
                n_periods, 0.035, 0.04, speed=0.1, volatility=0.012
            )
            
            # Equity returns (correlated with rates)
            rate_impact = -(scenarios['long_rates'][scenario] - 0.035) * 2
            equity_base = np.random.normal(0.08/12, 0.16/np.sqrt(12), n_periods)
            scenarios['equity_returns'][scenario] = equity_base + rate_impact/12
            
            # Inflation
            scenarios['inflation'][scenario] = np.maximum(
                0, np.random.normal(0.025, 0.01, n_periods)
            )
        
        return scenarios
    
    def _generate_deterministic_scenarios(
        self, 
        time_horizon_years: int
    ) -> Dict[str, np.ndarray]:
        """Generate deterministic base case scenario"""
        
        n_periods = time_horizon_years * 12
        
        return {
            'short_rates': np.full(n_periods, 0.025),
            'long_rates': np.full(n_periods, 0.035),
            'equity_returns': np.full(n_periods, 0.08/12),
            'inflation': np.full(n_periods, 0.025)
        }
    
    def _generate_stress_scenarios(
        self, 
        time_horizon_years: int
    ) -> Dict[str, np.ndarray]:
        """Generate stress test scenarios"""
        
        n_periods = time_horizon_years * 12
        
        scenarios = {}
        
        # Interest rate shock scenarios
        scenarios['rates_up_200bp'] = {
            'short_rates': np.full(n_periods, 0.045),  # +200bp
            'long_rates': np.full(n_periods, 0.055),   # +200bp
            'equity_returns': np.full(n_periods, 0.04/12),  # Lower equity returns
            'inflation': np.full(n_periods, 0.035)
        }
        
        scenarios['rates_down_100bp'] = {
            'short_rates': np.maximum(0.001, np.full(n_periods, 0.015)),  # -100bp, floored
            'long_rates': np.full(n_periods, 0.025),  # -100bp
            'equity_returns': np.full(n_periods, 0.10/12),  # Higher equity returns
            'inflation': np.full(n_periods, 0.015)
        }
        
        # Equity crash scenario
        scenarios['equity_crash'] = {
            'short_rates': np.full(n_periods, 0.015),  # Flight to quality
            'long_rates': np.full(n_periods, 0.025),   # Flight to quality
            'equity_returns': np.concatenate([
                np.full(12, -0.30/12),  # -30% in first year
                np.full(n_periods-12, 0.12/12)  # Recovery
            ]),
            'inflation': np.full(n_periods, 0.015)
        }
        
        return scenarios
    
    def _generate_fallback_scenarios(
        self, 
        num_scenarios: int, 
        time_horizon_years: int
    ) -> Dict[str, np.ndarray]:
        """Generate simple fallback scenarios"""
        
        n_periods = time_horizon_years * 12
        
        return {
            'short_rates': np.random.normal(0.025, 0.01, (num_scenarios, n_periods)),
            'long_rates': np.random.normal(0.035, 0.012, (num_scenarios, n_periods)),
            'equity_returns': np.random.normal(0.08/12, 0.16/np.sqrt(12), (num_scenarios, n_periods)),
            'inflation': np.maximum(0, np.random.normal(0.025, 0.008, (num_scenarios, n_periods)))
        }
    
    def _ml_rate_forecast(
        self, 
        maturity: str, 
        current_rates: Dict[str, float],
        forecast_months: int
    ) -> np.ndarray:
        """Forecast rates using trained ML models"""
        
        model_key = f'interest_rate_{maturity.lower()}'
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        # Create feature vector (simplified)
        features = np.array([
            current_rates.get('3M', 0.02),
            current_rates.get('10Y', 0.035),
            current_rates.get(maturity, 0.03),
            0.025,  # inflation assumption
            20      # VIX assumption
        ]).reshape(1, -1)
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        base_forecast = model.predict(features_scaled)[0]
        
        # Add some path variability
        forecasts = np.full(forecast_months, base_forecast)
        forecasts += np.random.normal(0, 0.005, forecast_months)  # Add noise
        
        return np.maximum(0.001, forecasts)  # Floor at 0.1%
    
    def _theory_based_forecast(
        self, 
        maturity: str, 
        current_rate: float, 
        forecast_months: int
    ) -> np.ndarray:
        """Simple theory-based rate forecast"""
        
        # Mean reversion toward long-term average
        long_term_rates = {'3M': 0.025, '1Y': 0.028, '5Y': 0.032, '10Y': 0.035, '30Y': 0.040}
        target_rate = long_term_rates.get(maturity, 0.035)
        
        # Gradual mean reversion
        reversion_speed = 0.1  # 10% per month
        forecasts = np.zeros(forecast_months)
        
        rate = current_rate
        for month in range(forecast_months):
            rate += reversion_speed * (target_rate - rate) + np.random.normal(0, 0.003)
            rate = max(0.001, rate)  # Floor at 0.1%
            forecasts[month] = rate
        
        return forecasts
    
    def _get_regime_transition_matrix(self) -> np.ndarray:
        """Get transition probabilities between economic regimes"""
        
        # Transition matrix (from regime to regime)
        # Rows: current regime, Columns: next regime
        # Order: expansion, recession, recovery, stagflation
        return np.array([
            [0.85, 0.10, 0.03, 0.02],  # From expansion
            [0.05, 0.60, 0.30, 0.05],  # From recession
            [0.40, 0.15, 0.40, 0.05],  # From recovery
            [0.20, 0.25, 0.20, 0.35]   # From stagflation
        ])