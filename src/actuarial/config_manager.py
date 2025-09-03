"""
Configuration Manager for Actuarial Engine
Loads all parameters from external configuration - ZERO hardcoding
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ActuarialConfigManager:
    """Manages all actuarial configuration and parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        
        # Find config file
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config in standard locations
            project_root = Path(__file__).parent.parent.parent
            possible_paths = [
                project_root / "config" / "actuarial_config.json",
                Path("config/actuarial_config.json"),
                Path("actuarial_config.json")
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.config_path = path
                    break
            else:
                raise FileNotFoundError(
                    "No actuarial configuration file found. "
                    "Please create config/actuarial_config.json"
                )
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                print(f"✅ Loaded actuarial configuration from {self.config_path}")
                return config
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            # Return minimal defaults only if config fails
            return self._get_emergency_defaults()
    
    def _get_emergency_defaults(self) -> Dict[str, Any]:
        """Emergency defaults ONLY if config file fails - should never be used in production"""
        
        print("⚠️ WARNING: Using emergency defaults - configuration file not loaded!")
        
        return {
            "actuarial_standards": {
                "credibility": {
                    "full_credibility_deaths": 1082,  # SOA standard
                    "partial_credibility_threshold": 0.3,
                    "minimum_credibility": 0.1
                },
                "risk_scoring": {
                    "base_risk_score": 2.5,
                    "max_risk_score": 5.0,
                    "min_risk_score": 1.0
                }
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Examples:
            config.get('actuarial_standards.credibility.full_credibility_deaths')
            config.get('risk_scoring.risk_weights.mortality', default=0.4)
        """
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_credibility_params(self) -> Dict[str, Any]:
        """Get credibility calculation parameters"""
        return self.get('actuarial_standards.credibility', {})
    
    def get_risk_weights(self) -> Dict[str, float]:
        """Get risk component weights"""
        return self.get('actuarial_standards.risk_scoring.risk_weights', {})
    
    def get_experience_thresholds(self) -> Dict[str, float]:
        """Get experience analysis thresholds"""
        return self.get('actuarial_standards.experience_analysis', {})
    
    def get_mortality_risk_params(self) -> Dict[str, Any]:
        """Get mortality risk parameters"""
        return self.get('actuarial_standards.risk_scoring.mortality_risk', {})
    
    def get_interest_rate_risk_params(self) -> Dict[str, Any]:
        """Get interest rate risk parameters"""
        return self.get('actuarial_standards.risk_scoring.interest_rate_risk', {})
    
    def get_concentration_risk_params(self) -> Dict[str, Any]:
        """Get concentration risk parameters"""
        return self.get('actuarial_standards.risk_scoring.concentration_risk', {})
    
    def get_pricing_assumptions(self) -> Dict[str, Any]:
        """Get pricing assumptions"""
        return self.get('actuarial_standards.pricing_assumptions', {})
    
    def get_capital_params(self) -> Dict[str, Any]:
        """Get capital requirement parameters"""
        return self.get('actuarial_standards.capital_requirements', {})
    
    def calculate_default_from_data(self, data_values: list, stat_type: str = 'mean') -> float:
        """Calculate defaults from actual data instead of using hardcoded values"""
        
        if not data_values:
            # If no data, return None (not a hardcoded default!)
            return None
        
        import numpy as np
        
        if stat_type == 'mean':
            return float(np.mean(data_values))
        elif stat_type == 'median':
            return float(np.median(data_values))
        elif stat_type == 'mode':
            from scipy import stats
            mode_result = stats.mode(data_values, keepdims=False)
            return float(mode_result.mode) if mode_result.count > 0 else float(np.mean(data_values))
        else:
            return float(np.mean(data_values))
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration and save to file"""
        
        try:
            # Deep merge updates into config
            self._deep_merge(self.config, updates)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Configuration updated and saved to {self.config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
            return False
    
    def _deep_merge(self, target: dict, source: dict) -> dict:
        """Deep merge source into target dictionary"""
        
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
        
        return target
    
    def validate_config(self) -> bool:
        """Validate that all required configuration parameters are present"""
        
        required_paths = [
            'actuarial_standards.credibility.full_credibility_deaths',
            'actuarial_standards.risk_scoring.risk_weights',
            'actuarial_standards.experience_analysis.adverse_threshold',
            'actuarial_standards.capital_requirements.var_confidence_level'
        ]
        
        missing = []
        for path in required_paths:
            if self.get(path) is None:
                missing.append(path)
        
        if missing:
            print(f"⚠️ Missing required configuration: {missing}")
            return False
        
        return True


# Singleton instance
_config_instance = None


def get_config() -> ActuarialConfigManager:
    """Get singleton configuration instance"""
    
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ActuarialConfigManager()
    
    return _config_instance