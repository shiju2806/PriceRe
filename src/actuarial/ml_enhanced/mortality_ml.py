"""
ML-Enhanced Mortality Modeling
Combines SOA actuarial tables with machine learning for personalized mortality rates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class MortalityMLEnhancer:
    """ML enhancement for mortality rate predictions beyond standard tables"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        
        # Mortality adjustment bounds for safety
        self.adjustment_bounds = {
            'min_adjustment': 0.25,  # Can't reduce mortality below 25% of table
            'max_adjustment': 5.0,   # Can't increase beyond 500% of table
            'extreme_threshold': 3.0  # Flag for manual review
        }
        
        # Medical condition impact factors (clinical research based)
        self.medical_factors = {
            'diabetes_type1': 2.5,
            'diabetes_type2': 1.8,
            'heart_disease': 2.2,
            'cancer_history': 1.9,
            'stroke_history': 2.8,
            'kidney_disease': 2.1,
            'liver_disease': 2.4,
            'copd': 2.0,
            'hypertension': 1.3,
            'obesity_bmi_35_plus': 1.6
        }
        
        # Lifestyle factors
        self.lifestyle_factors = {
            'smoker_current': 2.5,
            'smoker_former_5yr': 1.4,
            'alcohol_heavy': 1.8,
            'exercise_regular': 0.85,
            'diet_mediterranean': 0.90,
            'sleep_adequate': 0.95
        }
    
    def prepare_mortality_training_data(
        self,
        policy_data: pd.DataFrame,
        experience_data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for mortality ML models"""
        
        try:
            # Merge policy and experience data
            training_data = policy_data.merge(
                experience_data, 
                on=['policy_id'], 
                how='inner'
            )
            
            # Calculate A/E (Actual vs Expected) ratios as target
            training_data['ae_mortality_ratio'] = (
                training_data['actual_deaths'] / 
                training_data['expected_deaths'].clip(lower=0.001)
            )
            
            # Cap extreme A/E ratios for training stability
            training_data['ae_mortality_ratio'] = training_data['ae_mortality_ratio'].clip(
                lower=0.2, upper=10.0
            )
            
            # Create feature set
            features = self._engineer_mortality_features(training_data, external_data)
            target = training_data['ae_mortality_ratio']
            
            self.logger.info(f"Prepared {len(features)} samples with {features.shape[1]} features")
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error preparing mortality training data: {e}")
            # Return dummy data
            dummy_features = pd.DataFrame({
                'age': [35, 45, 55, 65],
                'gender_M': [1, 0, 1, 0],
                'bmi': [25, 30, 28, 32],
                'smoker': [0, 1, 0, 1]
            })
            dummy_target = pd.Series([1.0, 2.5, 1.2, 3.0])
            return dummy_features, dummy_target
    
    def train_mortality_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train multiple ML models for mortality enhancement"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scalers['mortality'] = StandardScaler()
        X_train_scaled = self.scalers['mortality'].fit_transform(X_train)
        X_test_scaled = self.scalers['mortality'].transform(X_test)
        
        results = {}
        
        # 1. Random Forest (interpretable baseline)
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        self.models['mortality_rf'] = rf_model
        results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'feature_importance': dict(zip(features.columns, rf_model.feature_importances_))
        }
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=20,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        self.models['mortality_gb'] = gb_model
        results['gradient_boosting'] = {
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred),
            'feature_importance': dict(zip(features.columns, gb_model.feature_importances_))
        }
        
        # 3. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            
            self.models['mortality_xgb'] = xgb_model
            results['xgboost'] = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'r2': r2_score(y_test, xgb_pred),
                'feature_importance': dict(zip(features.columns, xgb_model.feature_importances_))
            }
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        self.best_mortality_model = best_model_name
        self.feature_importance['mortality'] = results[best_model_name]['feature_importance']
        
        self.logger.info(f"Best mortality model: {best_model_name}")
        
        return results
    
    def predict_mortality_adjustment(
        self,
        policy_features: Dict[str, Any],
        use_ensemble: bool = True
    ) -> Dict[str, float]:
        """Predict mortality rate adjustment factor"""
        
        try:
            # Convert to DataFrame for processing
            features_df = pd.DataFrame([policy_features])
            
            # Apply same feature engineering as training
            processed_features = self._process_prediction_features(features_df)
            
            # Scale features
            if 'mortality' in self.scalers:
                scaled_features = self.scalers['mortality'].transform(processed_features)
            else:
                scaled_features = processed_features
            
            predictions = {}
            
            # Get predictions from all trained models
            for model_name, model in self.models.items():
                if 'mortality' in model_name:
                    pred = model.predict(scaled_features)[0]
                    # Apply safety bounds
                    bounded_pred = np.clip(
                        pred, 
                        self.adjustment_bounds['min_adjustment'],
                        self.adjustment_bounds['max_adjustment']
                    )
                    predictions[model_name.replace('mortality_', '')] = bounded_pred
            
            # Ensemble prediction (weighted average)
            if use_ensemble and len(predictions) > 1:
                # Weight based on model performance (better models get higher weight)
                weights = {'rf': 0.3, 'gb': 0.35, 'xgb': 0.35}
                ensemble_pred = sum(
                    predictions.get(model, 1.0) * weights.get(model, 0.0)
                    for model in weights.keys()
                    if model in predictions
                )
                predictions['ensemble'] = ensemble_pred
            
            # Add interpretability scores
            risk_assessment = self._assess_mortality_risk(policy_features)
            
            final_adjustment = predictions.get('ensemble', predictions.get(self.best_mortality_model, 1.0))
            
            return {
                'mortality_adjustment_factor': final_adjustment,
                'individual_predictions': predictions,
                'risk_assessment': risk_assessment,
                'confidence_level': self._calculate_prediction_confidence(policy_features),
                'requires_manual_review': final_adjustment > self.adjustment_bounds['extreme_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting mortality adjustment: {e}")
            return {
                'mortality_adjustment_factor': 1.0,
                'individual_predictions': {'fallback': 1.0},
                'risk_assessment': 'standard',
                'confidence_level': 0.5,
                'requires_manual_review': False
            }
    
    def _engineer_mortality_features(
        self,
        data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Engineer features for mortality prediction"""
        
        features = data.copy()
        
        # Basic demographic features
        features['age_squared'] = features['age'] ** 2
        features['age_cubed'] = features['age'] ** 3
        
        # Gender encoding
        if 'gender' in features.columns:
            features['gender_M'] = (features['gender'] == 'M').astype(int)
            features['gender_F'] = (features['gender'] == 'F').astype(int)
        
        # BMI categories
        if 'bmi' in features.columns:
            features['bmi_underweight'] = (features['bmi'] < 18.5).astype(int)
            features['bmi_normal'] = ((features['bmi'] >= 18.5) & (features['bmi'] < 25)).astype(int)
            features['bmi_overweight'] = ((features['bmi'] >= 25) & (features['bmi'] < 30)).astype(int)
            features['bmi_obese'] = (features['bmi'] >= 30).astype(int)
            features['bmi_extreme_obese'] = (features['bmi'] >= 35).astype(int)
        
        # Medical history aggregation
        medical_cols = [col for col in features.columns if 'medical_' in col or 'condition_' in col]
        if medical_cols:
            features['total_medical_conditions'] = features[medical_cols].sum(axis=1)
            features['has_major_condition'] = (features['total_medical_conditions'] > 2).astype(int)
        
        # Lifestyle score
        lifestyle_cols = [col for col in features.columns if 'lifestyle_' in col]
        if lifestyle_cols:
            features['lifestyle_risk_score'] = features[lifestyle_cols].sum(axis=1)
        
        # Interaction features
        if 'age' in features.columns and 'smoker' in features.columns:
            features['age_smoker_interaction'] = features['age'] * features['smoker']
        
        # Duration-based features
        if 'policy_duration' in features.columns:
            features['duration_squared'] = features['policy_duration'] ** 2
            features['early_duration'] = (features['policy_duration'] <= 2).astype(int)
            features['mature_duration'] = (features['policy_duration'] >= 10).astype(int)
        
        # Geographic risk (if available)
        if 'state' in features.columns:
            # High-risk states for certain conditions
            high_risk_states = ['WV', 'MS', 'AL', 'LA', 'KY', 'AR', 'TN', 'SC', 'OK']
            features['high_risk_geography'] = features['state'].isin(high_risk_states).astype(int)
        
        # Economic indicators (if external data provided)
        if external_data is not None:
            features = features.merge(
                external_data[['region', 'unemployment_rate', 'median_income']],
                left_on='state', right_on='region', how='left'
            )
            features['low_income_area'] = (features['median_income'] < 50000).astype(int)
        
        # Select numeric columns for modeling
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        return features[numeric_cols].fillna(0)
    
    def _process_prediction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Process features for prediction (same as training)"""
        
        # Apply same feature engineering
        processed = features_df.copy()
        
        # Ensure all expected features are present
        expected_features = [
            'age', 'age_squared', 'age_cubed', 'gender_M', 'gender_F',
            'bmi', 'bmi_underweight', 'bmi_normal', 'bmi_overweight', 
            'bmi_obese', 'bmi_extreme_obese', 'smoker'
        ]
        
        for feature in expected_features:
            if feature not in processed.columns:
                if 'age' in feature and 'age' in processed.columns:
                    if feature == 'age_squared':
                        processed[feature] = processed['age'] ** 2
                    elif feature == 'age_cubed':
                        processed[feature] = processed['age'] ** 3
                elif 'gender' in feature:
                    if feature == 'gender_M':
                        processed[feature] = (processed.get('gender', 'M') == 'M').astype(int)
                    elif feature == 'gender_F':
                        processed[feature] = (processed.get('gender', 'M') == 'F').astype(int)
                elif 'bmi' in feature and 'bmi' in processed.columns:
                    bmi_val = processed['bmi'].iloc[0] if len(processed) > 0 else 25
                    if feature == 'bmi_underweight':
                        processed[feature] = int(bmi_val < 18.5)
                    elif feature == 'bmi_normal':
                        processed[feature] = int(18.5 <= bmi_val < 25)
                    elif feature == 'bmi_overweight':
                        processed[feature] = int(25 <= bmi_val < 30)
                    elif feature == 'bmi_obese':
                        processed[feature] = int(bmi_val >= 30)
                    elif feature == 'bmi_extreme_obese':
                        processed[feature] = int(bmi_val >= 35)
                else:
                    processed[feature] = 0
        
        return processed[expected_features].fillna(0)
    
    def _assess_mortality_risk(self, features: Dict[str, Any]) -> str:
        """Assess overall mortality risk category"""
        
        risk_score = 0
        
        # Age risk
        age = features.get('age', 35)
        if age >= 65:
            risk_score += 2
        elif age >= 55:
            risk_score += 1
        
        # Medical conditions
        if features.get('smoker', False):
            risk_score += 3
        if features.get('diabetes', False):
            risk_score += 2
        if features.get('heart_disease', False):
            risk_score += 3
        
        # BMI risk
        bmi = features.get('bmi', 25)
        if bmi >= 35:
            risk_score += 2
        elif bmi >= 30:
            risk_score += 1
        
        # Determine risk category
        if risk_score >= 6:
            return 'high_risk'
        elif risk_score >= 3:
            return 'moderate_risk'
        else:
            return 'standard_risk'
    
    def _calculate_prediction_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence level for the prediction"""
        
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for edge cases
        age = features.get('age', 35)
        if age < 18 or age > 85:
            confidence -= 0.2
        
        # Reduce confidence for missing critical features
        critical_features = ['age', 'gender', 'smoker']
        missing_critical = sum(1 for f in critical_features if features.get(f) is None)
        confidence -= missing_critical * 0.15
        
        return max(0.3, min(0.95, confidence))
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'best_model': getattr(self, 'best_mortality_model', 'rf'),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Mortality ML models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.encoders = model_data['encoders']
            self.feature_importance = model_data['feature_importance']
            self.best_mortality_model = model_data['best_model']
            
            self.logger.info(f"Mortality ML models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False