"""
Life Insurance Pricing Engine

Provides end-to-end pricing model development for life insurance products.
Includes feature engineering, model training, and validation specifically
designed for life insurance actuarial work.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import joblib
from datetime import datetime, date
import warnings

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Actuarial libraries
try:
    from ..core.actuarial_engine import ActuarialEngine, MortalityTable
    ACTUARIAL_ENGINE_AVAILABLE = True
except ImportError:
    ACTUARIAL_ENGINE_AVAILABLE = False

try:
    from ..validation.actuarial_validation import LifeInsuranceValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
from ..utils.feature_engineering import LifeInsuranceFeatureEngine

warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class PricingModelConfig:
    """Configuration for life insurance pricing models"""
    target_variable: str = "annual_premium"
    model_type: str = "lightgbm"  # "glm", "lightgbm", "xgboost", "ensemble"
    validation_split: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # GLM specific
    glm_distribution: str = "gamma"  # "gamma", "tweedie", "poisson"
    glm_link: str = "log"
    
    # Tree-based model specific
    max_depth: int = 8
    n_estimators: int = 1000
    learning_rate: float = 0.1
    early_stopping_rounds: int = 50
    
    # Feature engineering
    create_interactions: bool = True
    polynomial_features: bool = False
    feature_selection: bool = True
    min_feature_importance: float = 0.001

@dataclass
class PricingModelResult:
    """Results from pricing model training"""
    model: Any
    model_type: str
    features: List[str]
    target: str
    training_score: float
    validation_score: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    residuals: np.ndarray
    validation_results: Dict[str, Any]
    model_metadata: Dict[str, Any]

class LifeInsurancePricer:
    """
    Complete life insurance pricing model development system
    """
    
    def __init__(self, config: PricingModelConfig = None):
        self.config = config or PricingModelConfig()
        self.actuarial_engine = ActuarialEngine()
        self.feature_engine = LifeInsuranceFeatureEngine()
        self.validator = LifeInsuranceValidator()
        
        self.model = None
        self.features = []
        self.scaler = None
        self.label_encoders = {}
        self.training_metadata = {}
    
    def load_and_prepare_data(
        self, 
        data_source: Union[str, pl.DataFrame, pd.DataFrame],
        data_format: str = "auto"
    ) -> pl.DataFrame:
        """
        Load and prepare data for pricing model development
        
        Args:
            data_source: File path, DataFrame, or data source
            data_format: "csv", "excel", "parquet", "auto"
            
        Returns:
            Cleaned and prepared Polars DataFrame
        """
        print(f"Loading data from {data_source}...")
        
        # Load data based on source type
        if isinstance(data_source, str):
            # File path
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {data_source}")
            
            if data_format == "auto":
                data_format = path.suffix.lower()[1:]  # Remove dot
            
            if data_format == "csv":
                df = pl.read_csv(data_source)
            elif data_format in ["xlsx", "xls"]:
                df = pl.read_excel(data_source)
            elif data_format == "parquet":
                df = pl.read_parquet(data_source)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
                
        elif isinstance(data_source, pl.DataFrame):
            df = data_source
        elif isinstance(data_source, pd.DataFrame):
            df = pl.from_pandas(data_source)
        else:
            raise ValueError("data_source must be file path, Polars DataFrame, or Pandas DataFrame")
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Basic data cleaning
        df = self._clean_data(df)
        
        print(f"After cleaning: {len(df)} records with {len(df.columns)} columns")
        return df
    
    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Basic data cleaning for life insurance data"""
        
        print("Performing basic data cleaning...")
        
        # Remove records with missing critical fields
        critical_fields = ['age_at_issue', 'gender', 'face_amount']
        available_critical = [f for f in critical_fields if f in df.columns]
        
        for field in available_critical:
            before_count = len(df)
            df = df.filter(pl.col(field).is_not_null())
            after_count = len(df)
            if before_count != after_count:
                print(f"  Removed {before_count - after_count} records with missing {field}")
        
        # Clean age values
        if 'age_at_issue' in df.columns:
            df = df.filter((pl.col('age_at_issue') >= 0) & (pl.col('age_at_issue') <= 100))
        
        # Clean face amounts
        if 'face_amount' in df.columns:
            df = df.filter((pl.col('face_amount') > 0) & (pl.col('face_amount') <= 50000000))
        
        # Clean premiums if they exist
        if self.config.target_variable in df.columns:
            df = df.filter(pl.col(self.config.target_variable) > 0)
        
        # Standardize gender values
        if 'gender' in df.columns:
            df = df.with_columns(
                pl.col('gender').map_elements(
                    lambda x: 'M' if str(x).upper().startswith('M') else 'F',
                    return_dtype=pl.Utf8
                )
            )
        
        # Standardize smoker status
        if 'smoker_status' in df.columns:
            df = df.with_columns(
                pl.col('smoker_status').map_elements(
                    lambda x: str(x).upper() in ['SMOKER', 'YES', 'TRUE', 'Y', '1'],
                    return_dtype=pl.Boolean
                )
            )
        
        return df
    
    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create actuarial features for life insurance pricing
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering actuarial features...")
        
        # Use feature engineering module
        df = self.feature_engine.create_all_features(df)
        
        print(f"After feature engineering: {len(df.columns)} columns")
        return df
    
    def train_pricing_model(
        self, 
        df: pl.DataFrame,
        features: Optional[List[str]] = None
    ) -> PricingModelResult:
        """
        Train life insurance pricing model
        
        Args:
            df: Training data
            features: List of feature columns to use (auto-detect if None)
            
        Returns:
            PricingModelResult with trained model and metrics
        """
        print(f"Training {self.config.model_type} pricing model...")
        
        # Prepare training data
        X, y, feature_names = self._prepare_training_data(df, features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=None  # Can't stratify continuous targets
        )
        
        print(f"Training set: {len(X_train)} records")
        print(f"Validation set: {len(X_val)} records")
        
        # Train model based on type
        if self.config.model_type == "glm":
            model = self._train_glm_model(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "lightgbm":
            model = self._train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names)
        elif self.config.model_type == "xgboost":
            model = self._train_xgboost_model(X_train, y_train, X_val, y_val, feature_names)
        elif self.config.model_type == "ensemble":
            model = self._train_ensemble_model(X_train, y_train, X_val, y_val, feature_names)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Generate predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Calculate metrics
        train_score = r2_score(y_train, y_pred_train)
        val_score = r2_score(y_val, y_pred_val)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=self.config.cross_validation_folds, scoring='r2')
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)
        
        # Residuals
        residuals = y_val - y_pred_val
        
        # Actuarial validation
        validation_results = self._validate_model_actuarially(
            df, model, feature_names, X_val, y_val, y_pred_val
        )
        
        # Store model and metadata
        self.model = model
        self.features = feature_names
        
        result = PricingModelResult(
            model=model,
            model_type=self.config.model_type,
            features=feature_names,
            target=self.config.target_variable,
            training_score=train_score,
            validation_score=val_score,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
            predictions=y_pred_val,
            residuals=residuals,
            validation_results=validation_results,
            model_metadata={
                "training_timestamp": datetime.now().isoformat(),
                "training_records": len(X_train),
                "validation_records": len(X_val),
                "feature_count": len(feature_names),
                "config": self.config.__dict__
            }
        )
        
        print(f"Model training completed!")
        print(f"Training R²: {train_score:.4f}")
        print(f"Validation R²: {val_score:.4f}")
        print(f"CV R² (mean): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return result
    
    def _prepare_training_data(
        self, 
        df: pl.DataFrame, 
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for model"""
        
        # Identify target variable
        if self.config.target_variable not in df.columns:
            raise ValueError(f"Target variable '{self.config.target_variable}' not found in data")
        
        # Auto-detect features if not provided
        if features is None:
            # Exclude target and non-predictive columns
            exclude_columns = {
                self.config.target_variable, 'policy_number', 'first_name', 'last_name',
                'address', 'city', 'phone', 'email', 'agent_id'
            }
            
            numeric_columns = []
            categorical_columns = []
            
            for col in df.columns:
                if col in exclude_columns:
                    continue
                    
                dtype = df[col].dtype
                if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                    numeric_columns.append(col)
                elif dtype in [pl.Utf8, pl.Boolean]:
                    categorical_columns.append(col)
            
            features = numeric_columns + categorical_columns
            print(f"Auto-detected {len(features)} features ({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)")
        
        # Convert to pandas for sklearn compatibility
        df_pandas = df.select([self.config.target_variable] + features).to_pandas()
        
        # Handle categorical variables
        X = df_pandas[features].copy()
        for col in features:
            if X[col].dtype == 'object' or X[col].dtype == 'bool':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        y = df_pandas[self.config.target_variable].values
        
        return X.values, y, features
    
    def _train_glm_model(self, X_train, y_train, X_val, y_val):
        """Train GLM model for life insurance pricing"""
        
        if self.config.glm_distribution == "tweedie":
            model = TweedieRegressor(
                power=1.5,  # Between Poisson (1) and Gamma (2)
                alpha=1.0,
                link='log',
                max_iter=1000
            )
        elif self.config.glm_distribution == "gamma":
            from sklearn.linear_model import GammaRegressor
            model = GammaRegressor(alpha=1.0, max_iter=1000)
        else:
            # Default to Tweedie
            model = TweedieRegressor(power=1.5, alpha=1.0, link='log', max_iter=1000)
        
        model.fit(X_train, y_train)
        return model
    
    def _train_lightgbm_model(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM model"""
        
        model = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            num_leaves=2**self.config.max_depth - 1,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_names=['validation'],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, verbose=0),
                lgb.log_evaluation(0)
            ]
        )
        
        return model
    
    def _train_xgboost_model(self, X_train, y_train, X_val, y_val, feature_names):
        """Train XGBoost model"""
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False
        )
        
        return model
    
    def _train_ensemble_model(self, X_train, y_train, X_val, y_val, feature_names):
        """Train ensemble of models"""
        from sklearn.ensemble import VotingRegressor
        
        # Create individual models
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=self.config.random_state, verbose=-1
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=self.config.random_state, verbosity=0
        )
        
        glm_model = TweedieRegressor(power=1.5, alpha=1.0, link='log')
        
        # Create ensemble
        ensemble = VotingRegressor([
            ('lightgbm', lgb_model),
            ('xgboost', xgb_model), 
            ('glm', glm_model)
        ])
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Extract feature importance from model"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models - use absolute coefficients
                importance = np.abs(model.coef_)
            else:
                # Ensemble models
                importance = np.zeros(len(feature_names))
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importance += estimator.feature_importances_
                    elif hasattr(estimator, 'coef_'):
                        importance += np.abs(estimator.coef_)
                importance = importance / len(model.named_estimators_)
            
            # Normalize to sum to 1
            if importance.sum() > 0:
                importance = importance / importance.sum()
            
            # Create dictionary
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            print(f"Warning: Could not extract feature importance: {e}")
            return {name: 1/len(feature_names) for name in feature_names}
    
    def _validate_model_actuarially(
        self, 
        df: pl.DataFrame, 
        model, 
        feature_names: List[str], 
        X_val: np.ndarray,
        y_val: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Perform actuarial validation of the model"""
        
        # Convert validation data back to DataFrame for validation
        val_df = pd.DataFrame(X_val, columns=feature_names)
        val_df[self.config.target_variable] = y_val
        val_df['predicted_premium'] = y_pred
        
        # Use validation module
        validation_results = self.validator.validate_pricing_model(
            val_df, model, feature_names
        )
        
        return validation_results
    
    def predict_premium(
        self, 
        policy_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict premium for a single policy
        
        Args:
            policy_data: Dictionary with policy characteristics
            
        Returns:
            Dictionary with premium and confidence metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_pricing_model() first.")
        
        # Create DataFrame from single record
        df = pl.DataFrame([policy_data])
        
        # Engineer features
        df = self.feature_engine.create_all_features(df)
        
        # Prepare features
        df_pandas = df.to_pandas()
        X = df_pandas[self.features].copy()
        
        # Handle categorical encoding
        for col in self.features:
            if col in self.label_encoders:
                if col in X.columns:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(0)  # Simple imputation for single record
        
        # Make prediction
        premium_pred = self.model.predict(X.values)[0]
        
        # Calculate confidence (simplified)
        confidence = 0.85  # Placeholder - would need proper uncertainty quantification
        
        return {
            'predicted_premium': premium_pred,
            'monthly_premium': premium_pred / 12,
            'premium_per_1000': (premium_pred / policy_data.get('face_amount', 100000)) * 1000,
            'confidence': confidence,
            'model_type': self.config.model_type
        }
    
    def save_model(self, filepath: str):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'label_encoders': self.label_encoders,
            'config': self.config,
            'training_metadata': self.training_metadata,
            'save_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and metadata"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.features = model_data['features']
        self.label_encoders = model_data['label_encoders']
        self.config = model_data['config']
        self.training_metadata = model_data['training_metadata']
        
        print(f"Model loaded from {filepath}")
        print(f"Model type: {self.config.model_type}")
        print(f"Features: {len(self.features)}")
    
    def create_pricing_pipeline(
        self,
        data_source: Union[str, pl.DataFrame, pd.DataFrame],
        **kwargs
    ) -> PricingModelResult:
        """
        Complete end-to-end pricing model pipeline
        
        Args:
            data_source: Input data
            **kwargs: Additional configuration options
            
        Returns:
            Complete pricing model results
        """
        print("Starting life insurance pricing pipeline...")
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Step 1: Load and prepare data
        df = self.load_and_prepare_data(data_source)
        
        # Step 2: Engineer features
        df = self.engineer_features(df)
        
        # Step 3: Train model
        results = self.train_pricing_model(df)
        
        print("Pricing pipeline completed successfully!")
        return results

# Convenience function for quick model training
def train_life_insurance_model(
    data_source: Union[str, pl.DataFrame, pd.DataFrame],
    model_type: str = "lightgbm",
    target_variable: str = "annual_premium",
    **kwargs
) -> Tuple[LifeInsurancePricer, PricingModelResult]:
    """
    Quick function to train a life insurance pricing model
    
    Returns:
        Tuple of (trained_pricer, results)
    """
    config = PricingModelConfig(
        model_type=model_type,
        target_variable=target_variable,
        **kwargs
    )
    
    pricer = LifeInsurancePricer(config)
    results = pricer.create_pricing_pipeline(data_source)
    
    return pricer, results