"""
Reinsurance Model Training Workflow

Complete ML pipeline for reinsurance pricing models:
- Automated feature engineering
- Model selection and training
- Cross-validation and hyperparameter tuning
- Model interpretation and validation
- Production deployment preparation
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pickle
import json
from datetime import datetime
from dataclasses import dataclass
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Optional ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

warnings.filterwarnings('ignore')


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    rmse: float
    mae: float
    r2: float
    mape: float
    cv_score: float
    cv_std: float


@dataclass
class ModelResults:
    """Complete model training results"""
    model_name: str
    model_object: Any
    metrics: ModelMetrics
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    actual_values: np.ndarray
    training_time: float
    model_params: Dict[str, Any]


class ReinsuranceModelTrainer:
    """Complete ML pipeline for reinsurance pricing"""
    
    def __init__(self, random_state: int = 42):
        """Initialize model trainer"""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
        # Available models
        self.model_configs = {
            "linear_regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "elastic_net": {
                "model": ElasticNet(random_state=random_state),
                "params": {
                    "alpha": [0.1, 0.5, 1.0, 2.0],
                    "l1_ratio": [0.1, 0.5, 0.7, 0.9]
                }
            },
            "random_forest": {
                "model": RandomForestRegressor(random_state=random_state, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=random_state),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            }
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.model_configs["lightgbm"] = {
                "model": lgb.LGBMRegressor(random_state=random_state, verbose=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "num_leaves": [31, 50, 100],
                    "max_depth": [5, 10, 15]
                }
            }
        
        # Add XGBoost if available (Industry standard!)
        if XGBOOST_AVAILABLE:
            self.model_configs["xgboost"] = {
                "model": xgb.XGBRegressor(
                    random_state=random_state,
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    min_child_weight=5
                ),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.7, 0.8, 0.9]
                }
            }
        
        # Add CatBoost if available (Best for mixed data types!)
        if CATBOOST_AVAILABLE:
            self.model_configs["catboost"] = {
                "model": CatBoostRegressor(
                    random_state=random_state,
                    verbose=False,
                    iterations=200,
                    learning_rate=0.1,
                    depth=6
                ),
                "params": {
                    "iterations": [100, 200, 300],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1]
                }
            }
    
    def train_pricing_model(
        self,
        data: pl.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        models_to_train: Optional[List[str]] = None,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, ModelResults]:
        """Train multiple pricing models and return results"""
        
        print(f"🎯 Training reinsurance pricing models on {len(data)} records...")
        
        # Prepare data
        X, y = self._prepare_data(data, target_column, feature_columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Select models to train
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())
        
        results = {}
        
        for model_name in models_to_train:
            print(f"   Training {model_name}...")
            
            start_time = datetime.now()
            
            # Get model configuration
            config = self.model_configs[model_name]
            
            # Train with hyperparameter tuning
            if config["params"]:
                best_model = self._train_with_tuning(
                    config["model"], config["params"], 
                    X_train, y_train, cv_folds
                )
            else:
                best_model = config["model"]
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, best_model, X_train, y_train, cv_folds)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(best_model, self.feature_names)
            
            # Training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            results[model_name] = ModelResults(
                model_name=model_name,
                model_object=best_model,
                metrics=metrics,
                feature_importance=feature_importance,
                predictions=y_pred,
                actual_values=y_test,
                training_time=training_time,
                model_params=best_model.get_params() if hasattr(best_model, 'get_params') else {}
            )
            
            print(f"      RMSE: {metrics.rmse:.4f}, R²: {metrics.r2:.4f}")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x].metrics.rmse)
        print(f"   🏆 Best model: {best_model_name}")
        
        return results
    
    def train_frequency_severity_models(
        self,
        claims_data: pl.DataFrame,
        portfolio_data: pl.DataFrame
    ) -> Dict[str, Any]:
        """Train separate frequency and severity models"""
        
        print("🎯 Training frequency and severity models...")
        
        # Prepare frequency data (claims per portfolio)
        frequency_data = self._prepare_frequency_data(claims_data, portfolio_data)
        
        # Determine available models
        available_models = ["random_forest", "gradient_boosting"]
        if LIGHTGBM_AVAILABLE:
            available_models.append("lightgbm")
        
        # Train frequency model
        print("   Training frequency model...")
        freq_results = self.train_pricing_model(
            frequency_data,
            target_column="claim_frequency",
            models_to_train=available_models
        )
        
        # Prepare severity data (claim amounts given occurrence)
        severity_data = self._prepare_severity_data(claims_data, portfolio_data)
        
        # Train severity model  
        print("   Training severity model...")
        sev_results = self.train_pricing_model(
            severity_data,
            target_column="log_claim_amount",
            models_to_train=available_models
        )
        
        return {
            "frequency_models": freq_results,
            "severity_models": sev_results,
            "modeling_approach": "frequency_severity"
        }
    
    def validate_model_performance(
        self,
        model_results: Dict[str, ModelResults],
        validation_data: Optional[pl.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validate model performance with business metrics"""
        
        validation_report = {
            "model_comparison": {},
            "business_validation": {},
            "production_readiness": {}
        }
        
        # Model comparison
        for model_name, result in model_results.items():
            validation_report["model_comparison"][model_name] = {
                "rmse": result.metrics.rmse,
                "r2": result.metrics.r2,
                "cv_score": result.metrics.cv_score,
                "training_time": result.training_time
            }
        
        # Business validation
        best_model_name = min(model_results.keys(), key=lambda x: model_results[x].metrics.rmse)
        best_result = model_results[best_model_name]
        
        # Check prediction reasonableness
        pred_mean = np.mean(best_result.predictions)
        actual_mean = np.mean(best_result.actual_values)
        
        validation_report["business_validation"] = {
            "prediction_vs_actual_ratio": pred_mean / actual_mean,
            "prediction_stability": np.std(best_result.predictions) / pred_mean,
            "residual_analysis": self._analyze_residuals(
                best_result.actual_values, best_result.predictions
            )
        }
        
        # Production readiness
        validation_report["production_readiness"] = {
            "model_size_mb": self._estimate_model_size(best_result.model_object),
            "prediction_speed": self._estimate_prediction_speed(best_result.model_object),
            "feature_stability": self._check_feature_stability(best_result.feature_importance)
        }
        
        return validation_report
    
    def save_production_model(
        self,
        model_result: ModelResults,
        model_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> None:
        """Save model for production deployment"""
        
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model object
        with open(model_path / "model.pkl", "wb") as f:
            pickle.dump(model_result.model_object, f)
        
        # Save preprocessing objects
        if self.scalers:
            with open(model_path / "scalers.pkl", "wb") as f:
                pickle.dump(self.scalers, f)
        
        if self.encoders:
            with open(model_path / "encoders.pkl", "wb") as f:
                pickle.dump(self.encoders, f)
        
        # Save metadata
        model_metadata = {
            "model_name": model_result.model_name,
            "training_date": datetime.now().isoformat(),
            "feature_names": self.feature_names,
            "metrics": {
                "rmse": model_result.metrics.rmse,
                "r2": model_result.metrics.r2,
                "cv_score": model_result.metrics.cv_score
            },
            "model_params": model_result.model_params,
            "feature_importance": model_result.feature_importance
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        with open(model_path / "metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ Model saved to {model_path}")
    
    def _prepare_data(
        self,
        data: pl.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for model training"""
        
        # Convert to pandas for sklearn compatibility
        df = data.to_pandas()
        
        # Initial data validation - check for infinity values
        print("🔍 Initial Data Validation:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"  ⚠️  {col}: {inf_count} infinite values detected")
                # Replace infinity values immediately
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Handle missing target values
        df = df.dropna(subset=[target_column])
        
        # Prepare features
        X = df[feature_columns].copy()
        y = df[target_column].values
        
        # Final validation before processing
        print(f"📊 Data shape: {X.shape}, Target shape: {y.shape}")
        total_inf = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if total_inf > 0:
            print(f"⚠️  {total_inf} infinite values still present in features")
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            
            # Handle missing values
            X[col] = X[col].fillna('MISSING')
            X[col] = self.encoders[col].fit_transform(X[col])
        
        # Handle numeric missing values and infinity values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # First, replace infinity values with NaN
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            
            # Check for any remaining infinite values and handle them
            if np.any(np.isinf(X[col].fillna(0))):
                print(f"Warning: Column {col} still has infinite values after replacement")
                X[col] = X[col].fillna(0)
                X[col] = np.where(np.isinf(X[col]), 0, X[col])
            
            # Cap extreme values to reasonable ranges (only if we have valid data)
            valid_data = X[col].dropna()
            if len(valid_data) > 0:
                q99 = valid_data.quantile(0.99)
                q01 = valid_data.quantile(0.01)
                # Ensure quantiles are finite
                if np.isfinite(q99) and np.isfinite(q01):
                    X[col] = X[col].clip(lower=q01, upper=q99)
                else:
                    # Fallback to median-based capping
                    median_val = valid_data.median()
                    std_val = valid_data.std()
                    if np.isfinite(median_val) and np.isfinite(std_val):
                        X[col] = X[col].clip(lower=median_val - 3*std_val, upper=median_val + 3*std_val)
            
            # Final check for infinite values before imputation
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            
            if col not in self.scalers:
                self.scalers[col] = SimpleImputer(strategy='median')
            
            # Brute force final cleanup before sklearn
            temp_data = X[[col]].copy()
            temp_data = temp_data.replace([np.inf, -np.inf], np.nan)
            temp_data = np.where(np.isinf(temp_data), np.nan, temp_data)
            
            X[col] = self.scalers[col].fit_transform(temp_data).ravel()
        
        # Final comprehensive check
        print("🔍 Final Data Validation Before Model Training:")
        for col in X.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(X[col]).sum()
            nan_count = np.isnan(X[col]).sum()
            if inf_count > 0 or nan_count > len(X) * 0.5:
                print(f"  ⚠️  {col}: {inf_count} inf, {nan_count} nan values")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        return X, y
    
    def _train_with_tuning(
        self,
        base_model,
        param_grid: Dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int
    ):
        """Train model with hyperparameter tuning"""
        
        # Use a smaller parameter grid for speed if dataset is large
        if len(X_train) > 10000:
            # Reduce parameter grid size for large datasets
            reduced_grid = {}
            for param, values in param_grid.items():
                if len(values) > 3:
                    reduced_grid[param] = values[::2]  # Take every 2nd value
                else:
                    reduced_grid[param] = values
            param_grid = reduced_grid
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=min(cv_folds, 3),  # Limit CV folds for speed
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_folds: int
    ) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=min(cv_folds, 3), scoring='neg_mean_squared_error'
            )
            cv_score = -cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_score = rmse
            cv_std = 0
        
        return ModelMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            cv_score=cv_score,
            cv_std=cv_std
        )
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
            else:
                return {}
            
            # Normalize to sum to 1
            if np.sum(importances) > 0:
                importances = importances / np.sum(importances)
            
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
            
        except Exception:
            pass
        
        return importance_dict
    
    def _prepare_frequency_data(
        self,
        claims_data: pl.DataFrame,
        portfolio_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Prepare data for frequency modeling"""
        
        # Aggregate claims by portfolio
        claim_counts = claims_data.group_by("treaty_id").agg([
            pl.len().alias("claim_count")
        ])
        
        # Join with portfolio data
        frequency_data = portfolio_data.join(claim_counts, on="portfolio_id", how="left")
        
        # Fill missing claim counts with 0
        frequency_data = frequency_data.with_columns([
            pl.col("claim_count").fill_null(0)
        ])
        
        # Calculate claim frequency
        frequency_data = frequency_data.with_columns([
            (pl.col("claim_count") / pl.col("number_of_risks")).alias("claim_frequency")
        ])
        
        return frequency_data
    
    def _prepare_severity_data(
        self,
        claims_data: pl.DataFrame,
        portfolio_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Prepare data for severity modeling"""
        
        # Filter to claims with positive amounts
        severity_data = claims_data.filter(pl.col("gross_claim_amount") > 0)
        
        # Add log transformation
        severity_data = severity_data.with_columns([
            pl.col("gross_claim_amount").log().alias("log_claim_amount")
        ])
        
        # Join with portfolio characteristics
        severity_data = severity_data.join(
            portfolio_data.select(["portfolio_id", "business_line", "territory", "average_sum_insured"]),
            on="portfolio_id",
            how="left"
        )
        
        return severity_data
    
    def _analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze model residuals"""
        
        residuals = y_true - y_pred
        
        return {
            "mean_residual": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_skewness": float(self._calculate_skewness(residuals)),
            "residual_kurtosis": float(self._calculate_kurtosis(residuals))
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB"""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
        except:
            return 0.0
    
    def _estimate_prediction_speed(self, model) -> float:
        """Estimate prediction speed (predictions per second)"""
        try:
            # Create dummy data
            dummy_X = np.random.random((100, len(self.feature_names)))
            
            # Time predictions
            start_time = datetime.now()
            model.predict(dummy_X)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            return 100 / duration if duration > 0 else 0
        except:
            return 0.0
    
    def _check_feature_stability(self, feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """Check feature stability for production"""
        
        # Get top features
        top_features = list(feature_importance.keys())[:10]
        
        # Calculate importance concentration
        top_10_importance = sum(feature_importance[f] for f in top_features[:10])
        
        return {
            "top_10_features": top_features,
            "top_10_concentration": top_10_importance,
            "feature_count": len(feature_importance),
            "min_importance": min(feature_importance.values()) if feature_importance else 0
        }