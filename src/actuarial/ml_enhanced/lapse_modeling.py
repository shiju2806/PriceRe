"""
ML-Enhanced Lapse Modeling for Life Insurance
Predicts policy lapse behavior using policyholder characteristics and economic conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False

class LapseModelingEngine:
    """ML-powered lapse prediction for life insurance policies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.calibrators = {}
        
        # Industry lapse benchmarks by product type
        self.industry_benchmarks = {
            'TERM': {
                'year_1': 0.12, 'year_2': 0.10, 'year_3': 0.08,
                'year_5': 0.06, 'year_10': 0.05, 'ultimate': 0.04
            },
            'WHOLE_LIFE': {
                'year_1': 0.08, 'year_2': 0.06, 'year_3': 0.05,
                'year_5': 0.04, 'year_10': 0.03, 'ultimate': 0.02
            },
            'UNIVERSAL_LIFE': {
                'year_1': 0.10, 'year_2': 0.08, 'year_3': 0.06,
                'year_5': 0.05, 'year_10': 0.04, 'ultimate': 0.03
            }
        }
        
        # Lapse sensitivity factors
        self.sensitivity_factors = {
            'interest_rate': {
                'direction': 'positive',  # Higher rates = more lapses
                'elasticity': 1.5
            },
            'unemployment': {
                'direction': 'positive',  # Higher unemployment = more lapses
                'elasticity': 0.8
            },
            'income_change': {
                'direction': 'negative',  # Income decrease = more lapses
                'elasticity': -1.2
            },
            'competitive_rates': {
                'direction': 'positive',  # Better alternatives = more lapses
                'elasticity': 2.0
            }
        }
    
    def prepare_lapse_training_data(
        self,
        policy_data: pd.DataFrame,
        lapse_history: pd.DataFrame,
        economic_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for lapse modeling"""
        
        try:
            # Merge policy and lapse data
            training_data = policy_data.merge(
                lapse_history,
                on=['policy_id'],
                how='left'
            )
            
            # Add economic context at policy level
            training_data = self._add_economic_context(training_data, economic_data)
            
            # Engineer lapse features
            features = self._engineer_lapse_features(training_data)
            
            # Create target variable (1 = lapsed, 0 = active)
            target = training_data['lapsed'].fillna(0).astype(int)
            
            # Balance dataset if needed
            features, target = self._balance_dataset(features, target)
            
            self.logger.info(f"Prepared {len(features)} samples, lapse rate: {target.mean():.3f}")
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error preparing lapse training data: {e}")
            return self._generate_sample_lapse_data()
    
    def train_lapse_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        optimize_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """Train multiple models for lapse prediction"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, stratify=target
        )
        
        # Scale features
        self.scalers['lapse'] = StandardScaler()
        X_train_scaled = self.scalers['lapse'].fit_transform(X_train)
        X_test_scaled = self.scalers['lapse'].transform(X_test)
        
        results = {}
        
        # 1. Logistic Regression (baseline, highly interpretable)
        lr_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models['lapse_logistic'] = lr_model
        results['logistic_regression'] = self._evaluate_classification_model(
            y_test, lr_pred, features.columns, lr_model.coef_[0]
        )
        
        # 2. Random Forest (interpretable, robust)
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        if optimize_hyperparameters:
            rf_params = self._optimize_rf_params(X_train_scaled, y_train)
        
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models['lapse_rf'] = rf_model
        results['random_forest'] = self._evaluate_classification_model(
            y_test, rf_pred, features.columns, rf_model.feature_importances_
        )
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models['lapse_gb'] = gb_model
        results['gradient_boosting'] = self._evaluate_classification_model(
            y_test, gb_pred, features.columns, gb_model.feature_importances_
        )
        
        # 4. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            self.models['lapse_xgb'] = xgb_model
            results['xgboost'] = self._evaluate_classification_model(
                y_test, xgb_pred, features.columns, xgb_model.feature_importances_
            )
        
        # 5. Neural Network (if available)
        if NEURAL_NETWORK_AVAILABLE:
            nn_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                learning_rate_init=0.001,
                random_state=42
            )
            nn_model.fit(X_train_scaled, y_train)
            nn_pred = nn_model.predict_proba(X_test_scaled)[:, 1]
            
            self.models['lapse_nn'] = nn_model
            results['neural_network'] = self._evaluate_classification_model(
                y_test, nn_pred, features.columns, None  # No feature importance for NN
            )
        
        # Select best model based on AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        self.best_lapse_model = best_model_name
        
        # Calibrate best model for better probability estimates
        best_model = self.models[f'lapse_{best_model_name.replace("_", "_").split("_")[-1]}']
        calibrator = CalibratedClassifierCV(best_model, cv=3, method='isotonic')
        calibrator.fit(X_train_scaled, y_train)
        self.calibrators['lapse'] = calibrator
        
        self.logger.info(f"Best lapse model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.3f})")
        
        return results
    
    def predict_lapse_probability(
        self,
        policy_features: Dict[str, Any],
        economic_scenario: Optional[Dict[str, float]] = None,
        time_horizon_months: int = 12
    ) -> Dict[str, Any]:
        """Predict lapse probability for a policy"""
        
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame([policy_features])
            
            # Add economic scenario if provided
            if economic_scenario:
                for key, value in economic_scenario.items():
                    features_df[f'economic_{key}'] = value
            
            # Process features
            processed_features = self._process_lapse_prediction_features(features_df)
            
            # Scale features
            if 'lapse' in self.scalers:
                scaled_features = self.scalers['lapse'].transform(processed_features)
            else:
                scaled_features = processed_features
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                if 'lapse' in model_name:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(scaled_features)[0, 1]
                    else:
                        prob = model.predict(scaled_features)[0]
                    
                    model_key = model_name.replace('lapse_', '')
                    predictions[model_key] = prob
            
            # Use calibrated prediction if available
            if 'lapse' in self.calibrators:
                calibrated_prob = self.calibrators['lapse'].predict_proba(scaled_features)[0, 1]
                predictions['calibrated'] = calibrated_prob
            
            # Calculate ensemble prediction
            ensemble_prob = np.mean(list(predictions.values()))
            
            # Adjust for time horizon (compound over months)
            monthly_lapse_rate = 1 - (1 - ensemble_prob) ** (1/12)
            cumulative_lapse_prob = 1 - (1 - monthly_lapse_rate) ** time_horizon_months
            
            # Risk assessment
            risk_category = self._assess_lapse_risk(ensemble_prob, policy_features)
            
            # Key drivers analysis
            drivers = self._identify_lapse_drivers(policy_features, processed_features)
            
            return {
                'annual_lapse_probability': ensemble_prob,
                'monthly_lapse_rate': monthly_lapse_rate,
                f'{time_horizon_months}_month_lapse_probability': cumulative_lapse_prob,
                'individual_predictions': predictions,
                'risk_category': risk_category,
                'key_drivers': drivers,
                'confidence_level': self._calculate_lapse_confidence(policy_features),
                'benchmark_comparison': self._compare_to_benchmark(
                    ensemble_prob, policy_features.get('product_type', 'TERM')
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting lapse probability: {e}")
            return self._default_lapse_prediction(policy_features)
    
    def analyze_lapse_sensitivity(
        self,
        base_features: Dict[str, Any],
        sensitivity_variables: List[str],
        variation_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Analyze lapse sensitivity to key variables"""
        
        sensitivity_results = {}
        base_prediction = self.predict_lapse_probability(base_features)
        base_prob = base_prediction['annual_lapse_probability']
        
        for variable in sensitivity_variables:
            if variable in variation_ranges:
                min_val, max_val = variation_ranges[variable]
                
                # Test multiple points
                test_values = np.linspace(min_val, max_val, 11)
                probabilities = []
                
                for test_val in test_values:
                    test_features = base_features.copy()
                    test_features[variable] = test_val
                    
                    pred = self.predict_lapse_probability(test_features)
                    probabilities.append(pred['annual_lapse_probability'])
                
                # Calculate elasticity
                elasticity = self._calculate_elasticity(
                    base_features[variable], base_prob,
                    test_values, probabilities
                )
                
                sensitivity_results[variable] = {
                    'base_value': base_features[variable],
                    'base_probability': base_prob,
                    'test_values': test_values.tolist(),
                    'test_probabilities': probabilities,
                    'elasticity': elasticity,
                    'impact_direction': 'positive' if elasticity > 0 else 'negative'
                }
        
        return sensitivity_results
    
    def generate_lapse_curves(
        self,
        policy_cohort: pd.DataFrame,
        projection_years: int = 20
    ) -> Dict[str, np.ndarray]:
        """Generate lapse curves for a cohort of policies"""
        
        # Group policies by characteristics
        cohort_groups = self._segment_cohort(policy_cohort)
        
        lapse_curves = {}
        
        for group_name, group_policies in cohort_groups.items():
            
            # Calculate average lapse probability for this group
            group_features = group_policies.mean().to_dict()
            
            # Project lapse rates by duration
            annual_lapse_rates = []
            
            for year in range(1, projection_years + 1):
                # Adjust features for duration
                duration_features = group_features.copy()
                duration_features['policy_duration'] = year
                duration_features['age'] = group_features.get('issue_age', 35) + year
                
                # Predict lapse probability
                prediction = self.predict_lapse_probability(duration_features)
                annual_lapse_rates.append(prediction['annual_lapse_probability'])
            
            lapse_curves[group_name] = np.array(annual_lapse_rates)
        
        return lapse_curves
    
    def _engineer_lapse_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for lapse prediction"""
        
        features = data.copy()
        
        # Policy characteristics
        if 'face_amount' in features.columns:
            features['log_face_amount'] = np.log1p(features['face_amount'])
            features['high_face_amount'] = (features['face_amount'] > 1000000).astype(int)
        
        if 'annual_premium' in features.columns and 'face_amount' in features.columns:
            features['premium_per_1000'] = (features['annual_premium'] / features['face_amount']) * 1000
            features['high_premium_ratio'] = (features['premium_per_1000'] > 20).astype(int)
        
        # Duration features
        if 'policy_duration' in features.columns:
            features['early_duration'] = (features['policy_duration'] <= 2).astype(int)
            features['mature_policy'] = (features['policy_duration'] >= 10).astype(int)
            features['duration_squared'] = features['policy_duration'] ** 2
        
        # Age-related features
        if 'current_age' in features.columns:
            features['retirement_age'] = (features['current_age'] >= 65).astype(int)
            features['young_adult'] = (features['current_age'] <= 35).astype(int)
            features['middle_aged'] = ((features['current_age'] > 35) & 
                                     (features['current_age'] < 65)).astype(int)
        
        # Economic sensitivity features
        if 'interest_rate_10y' in features.columns:
            features['high_interest_environment'] = (features['interest_rate_10y'] > 0.04).astype(int)
            features['low_interest_environment'] = (features['interest_rate_10y'] < 0.02).astype(int)
        
        # Competitive environment
        if 'competitive_rate_spread' in features.columns:
            features['poor_competitive_position'] = (features['competitive_rate_spread'] > 0.005).astype(int)
            features['strong_competitive_position'] = (features['competitive_rate_spread'] < -0.005).astype(int)
        
        # Financial stress indicators
        if 'unemployment_rate' in features.columns:
            features['high_unemployment'] = (features['unemployment_rate'] > 0.06).astype(int)
        
        if 'credit_score' in features.columns:
            features['low_credit_score'] = (features['credit_score'] < 650).astype(int)
            features['excellent_credit'] = (features['credit_score'] >= 750).astype(int)
        
        # Behavioral features
        if 'payment_history' in features.columns:
            features['perfect_payment_history'] = (features['payment_history'] == 1.0).astype(int)
            features['payment_issues'] = (features['payment_history'] < 0.9).astype(int)
        
        # Product type encoding
        if 'product_type' in features.columns:
            product_dummies = pd.get_dummies(features['product_type'], prefix='product')
            features = pd.concat([features, product_dummies], axis=1)
        
        # Select numeric features
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)
        
        return numeric_features
    
    def _add_economic_context(
        self, 
        policy_data: pd.DataFrame, 
        economic_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add economic context to policy data"""
        
        # Merge based on time period (simplified)
        if 'valuation_date' in policy_data.columns and not economic_data.empty:
            
            # Get closest economic data for each policy
            merged_data = policy_data.copy()
            
            # Add key economic indicators
            if 'interest_rate_10y' in economic_data.columns:
                merged_data['interest_rate_10y'] = economic_data['interest_rate_10y'].iloc[-1]
            
            if 'unemployment_rate' in economic_data.columns:
                merged_data['unemployment_rate'] = economic_data['unemployment_rate'].iloc[-1]
            
            if 'sp500_return' in economic_data.columns:
                merged_data['equity_performance'] = economic_data['sp500_return'].iloc[-1]
            
            return merged_data
        
        return policy_data
    
    def _balance_dataset(
        self, 
        features: pd.DataFrame, 
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance the dataset if lapse rate is too low/high"""
        
        lapse_rate = target.mean()
        
        # If lapse rate is extremely imbalanced, apply sampling
        if lapse_rate < 0.05 or lapse_rate > 0.5:
            
            from sklearn.utils import resample
            
            # Separate classes
            majority_class = target.value_counts().idxmax()
            minority_class = target.value_counts().idxmin()
            
            majority_features = features[target == majority_class]
            majority_target = target[target == majority_class]
            minority_features = features[target == minority_class]
            minority_target = target[target == minority_class]
            
            # Downsample majority class to achieve ~15% lapse rate
            target_ratio = 0.15
            target_majority_size = int(len(minority_features) * (1 - target_ratio) / target_ratio)
            
            if target_majority_size < len(majority_features):
                majority_downsampled = resample(
                    majority_features,
                    n_samples=target_majority_size,
                    random_state=42
                )
                majority_target_downsampled = resample(
                    majority_target,
                    n_samples=target_majority_size,
                    random_state=42
                )
                
                # Combine
                balanced_features = pd.concat([minority_features, majority_downsampled])
                balanced_target = pd.concat([minority_target, majority_target_downsampled])
                
                return balanced_features, balanced_target
        
        return features, target
    
    def _generate_sample_lapse_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate sample lapse data for testing"""
        
        n_samples = 5000
        
        # Generate synthetic features
        features = pd.DataFrame({
            'age': np.random.normal(45, 12, n_samples),
            'policy_duration': np.random.exponential(5, n_samples),
            'face_amount': np.random.lognormal(11, 1, n_samples),  # ~100k average
            'annual_premium': np.random.lognormal(7, 0.8, n_samples),  # ~1k average
            'interest_rate_10y': np.random.normal(0.035, 0.01, n_samples),
            'unemployment_rate': np.random.normal(0.05, 0.02, n_samples),
            'credit_score': np.random.normal(720, 80, n_samples),
            'payment_history': np.random.beta(9, 1, n_samples)  # Mostly good payment history
        })
        
        # Generate synthetic target with realistic relationships
        lapse_propensity = (
            -2.0 +  # Base intercept
            0.02 * features['age'] +
            -0.1 * features['policy_duration'] +
            0.3 * features['interest_rate_10y'] * 100 +
            0.5 * features['unemployment_rate'] * 100 +
            -0.005 * features['credit_score'] +
            -1.0 * features['payment_history'] +
            np.random.normal(0, 0.5, n_samples)  # Noise
        )
        
        # Convert to probabilities
        lapse_probs = 1 / (1 + np.exp(-lapse_propensity))
        target = (np.random.random(n_samples) < lapse_probs).astype(int)
        
        return features, target
    
    def _optimize_rf_params(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest parameters"""
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [20, 50],
            'min_samples_leaf': [10, 20]
        }
        
        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_params.update({
            'class_weight': 'balanced',
            'random_state': 42
        })
        
        return best_params
    
    def _evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        feature_names: List[str],
        feature_importance: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Evaluate classification model performance"""
        
        # AUC score
        auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find optimal threshold (F1 maximization)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Binary predictions at optimal threshold
        y_pred_binary = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Classification report
        report = classification_report(y_true, y_pred_binary, output_dict=True)
        
        result = {
            'auc_score': auc,
            'optimal_threshold': optimal_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        # Add feature importance if available
        if feature_importance is not None:
            importance_dict = dict(zip(feature_names, feature_importance))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            result['feature_importance'] = dict(sorted_importance[:10])  # Top 10
        
        return result
    
    def _process_lapse_prediction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Process features for prediction"""
        
        # Apply same feature engineering as training
        processed = self._engineer_lapse_features(features_df)
        
        # Ensure all expected features are present
        expected_features = [
            'age', 'policy_duration', 'face_amount', 'annual_premium',
            'interest_rate_10y', 'unemployment_rate', 'credit_score', 'payment_history'
        ]
        
        for feature in expected_features:
            if feature not in processed.columns:
                if feature == 'age':
                    processed[feature] = processed.get('current_age', 45)
                elif feature == 'policy_duration':
                    processed[feature] = processed.get('duration', 5)
                elif feature == 'face_amount':
                    processed[feature] = processed.get('face_amount', 100000)
                elif feature == 'annual_premium':
                    processed[feature] = processed.get('annual_premium', 1000)
                else:
                    # Set reasonable defaults
                    defaults = {
                        'interest_rate_10y': 0.035,
                        'unemployment_rate': 0.05,
                        'credit_score': 720,
                        'payment_history': 0.95
                    }
                    processed[feature] = defaults.get(feature, 0)
        
        return processed[expected_features].fillna(0)
    
    def _assess_lapse_risk(
        self, 
        lapse_probability: float, 
        policy_features: Dict[str, Any]
    ) -> str:
        """Assess lapse risk category"""
        
        product_type = policy_features.get('product_type', 'TERM')
        benchmark = self.industry_benchmarks[product_type]['ultimate']
        
        if lapse_probability >= benchmark * 2:
            return 'high_risk'
        elif lapse_probability >= benchmark * 1.5:
            return 'moderate_risk'
        elif lapse_probability >= benchmark * 0.5:
            return 'standard_risk'
        else:
            return 'low_risk'
    
    def _identify_lapse_drivers(
        self, 
        original_features: Dict[str, Any],
        processed_features: pd.DataFrame
    ) -> List[str]:
        """Identify key drivers of lapse probability"""
        
        drivers = []
        
        # High interest rate environment
        if original_features.get('interest_rate_10y', 0.035) > 0.045:
            drivers.append("High interest rate environment")
        
        # Economic stress
        if original_features.get('unemployment_rate', 0.05) > 0.065:
            drivers.append("Economic stress (high unemployment)")
        
        # Early policy duration
        if original_features.get('policy_duration', 5) <= 2:
            drivers.append("Early policy duration")
        
        # Credit issues
        if original_features.get('credit_score', 720) < 650:
            drivers.append("Credit quality concerns")
        
        # Payment history
        if original_features.get('payment_history', 0.95) < 0.9:
            drivers.append("Payment history issues")
        
        # High premium ratio
        if (original_features.get('annual_premium', 1000) / 
            original_features.get('face_amount', 100000) * 1000) > 25:
            drivers.append("High premium-to-coverage ratio")
        
        return drivers if drivers else ["Standard risk factors"]
    
    def _calculate_lapse_confidence(self, policy_features: Dict[str, Any]) -> float:
        """Calculate confidence in lapse prediction"""
        
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for missing features
        required_features = ['age', 'policy_duration', 'face_amount']
        missing_features = sum(1 for f in required_features if policy_features.get(f) is None)
        confidence -= missing_features * 0.1
        
        # Reduce confidence for edge cases
        age = policy_features.get('age', 45)
        if age < 18 or age > 85:
            confidence -= 0.15
        
        duration = policy_features.get('policy_duration', 5)
        if duration > 30:  # Very old policy
            confidence -= 0.1
        
        return max(0.3, min(0.95, confidence))
    
    def _compare_to_benchmark(
        self, 
        predicted_prob: float, 
        product_type: str
    ) -> Dict[str, Any]:
        """Compare prediction to industry benchmarks"""
        
        if product_type not in self.industry_benchmarks:
            product_type = 'TERM'
        
        benchmark = self.industry_benchmarks[product_type]['ultimate']
        
        return {
            'industry_benchmark': benchmark,
            'predicted_probability': predicted_prob,
            'relative_to_benchmark': predicted_prob / benchmark if benchmark > 0 else 1.0,
            'assessment': (
                'Above benchmark' if predicted_prob > benchmark * 1.2 else
                'Near benchmark' if predicted_prob > benchmark * 0.8 else
                'Below benchmark'
            )
        }
    
    def _default_lapse_prediction(self, policy_features: Dict[str, Any]) -> Dict[str, Any]:
        """Default lapse prediction for error cases"""
        
        product_type = policy_features.get('product_type', 'TERM')
        default_prob = self.industry_benchmarks[product_type]['ultimate']
        
        return {
            'annual_lapse_probability': default_prob,
            'monthly_lapse_rate': 1 - (1 - default_prob) ** (1/12),
            '12_month_lapse_probability': default_prob,
            'individual_predictions': {'default': default_prob},
            'risk_category': 'standard_risk',
            'key_drivers': ['Industry average assumption'],
            'confidence_level': 0.5,
            'benchmark_comparison': {
                'industry_benchmark': default_prob,
                'predicted_probability': default_prob,
                'relative_to_benchmark': 1.0,
                'assessment': 'Industry benchmark'
            }
        }
    
    def _calculate_elasticity(
        self,
        base_value: float,
        base_prob: float,
        test_values: np.ndarray,
        test_probs: List[float]
    ) -> float:
        """Calculate elasticity of lapse probability"""
        
        if len(test_values) < 3 or base_value == 0:
            return 0.0
        
        # Linear regression to estimate elasticity
        log_values = np.log(np.maximum(0.001, test_values / base_value))
        log_probs = np.log(np.maximum(0.001, np.array(test_probs) / base_prob))
        
        # Simple linear fit
        if len(log_values) > 1 and np.std(log_values) > 0:
            elasticity = np.corrcoef(log_values, log_probs)[0, 1] * np.std(log_probs) / np.std(log_values)
            return elasticity
        
        return 0.0
    
    def _segment_cohort(self, cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Segment cohort for lapse curve generation"""
        
        segments = {}
        
        # By product type
        if 'product_type' in cohort.columns:
            for product_type in cohort['product_type'].unique():
                segments[f'product_{product_type}'] = cohort[cohort['product_type'] == product_type]
        
        # By age group
        if 'issue_age' in cohort.columns:
            segments['young'] = cohort[cohort['issue_age'] <= 35]
            segments['middle_aged'] = cohort[(cohort['issue_age'] > 35) & (cohort['issue_age'] <= 55)]
            segments['mature'] = cohort[cohort['issue_age'] > 55]
        
        # By face amount
        if 'face_amount' in cohort.columns:
            face_median = cohort['face_amount'].median()
            segments['small_policies'] = cohort[cohort['face_amount'] <= face_median]
            segments['large_policies'] = cohort[cohort['face_amount'] > face_median]
        
        return segments if segments else {'all_policies': cohort}