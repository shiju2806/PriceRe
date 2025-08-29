"""
Model training and prediction modules
"""

# Use try/except to handle optional dependencies
try:
    from .reinsurance_model import ReinsuranceModelTrainer, ModelResults, ModelMetrics
    _REINSURANCE_MODEL_AVAILABLE = True
    __all__ = [
        "ReinsuranceModelTrainer",
        "ModelResults", 
        "ModelMetrics"
    ]
except ImportError as e:
    _REINSURANCE_MODEL_AVAILABLE = False
    __all__ = []
    
    # Create stub classes for graceful degradation
    class ReinsuranceModelTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ReinsuranceModelTrainer unavailable due to missing dependencies: {e}")
    
    class ModelResults:
        pass
    
    class ModelMetrics:
        pass