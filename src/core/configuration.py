"""
Central Configuration System
Single source of truth for all system configuration
No hardcoded values anywhere in the system
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for external data sources"""
    name: str
    type: str  # 'api', 'file', 'database'
    connection_params: Dict[str, Any] = field(default_factory=dict)
    refresh_interval_hours: int = 24
    timeout_seconds: int = 30
    retry_count: int = 3
    fallback_enabled: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    name: str
    type: str  # 'xgboost', 'random_forest', 'neural_network'
    parameters: Dict[str, Any] = field(default_factory=dict)
    training_schedule: str = "monthly"
    validation_threshold: float = 0.85
    auto_retrain: bool = True
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""

@dataclass
class BusinessConfig:
    """Business logic configuration"""
    company_name: str = "Default Insurance Company"
    regulatory_jurisdiction: str = "US"
    base_currency: str = "USD"
    fiscal_year_end: str = "12-31"
    
    # Default business assumptions
    default_profit_margin: float = 0.15
    default_expense_ratio: float = 0.05
    default_commission_structure: Dict[str, float] = field(default_factory=lambda: {
        "first_year": 0.90,
        "renewal": 0.05
    })

@dataclass
class SystemConfig:
    """Overall system configuration"""
    version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    debug_mode: bool = True
    
    # Storage paths (all configurable)
    data_storage_path: str = "data"
    model_storage_path: str = "models"
    log_storage_path: str = "logs"
    cache_storage_path: str = "cache"
    
    # Event system configuration
    event_history_size: int = 10000
    event_processing_enabled: bool = True
    
    # API configuration
    api_rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "fred_api": 1000,  # calls per day
        "alpha_vantage": 500
    })

class ConfigurationManager:
    """
    Central configuration manager - single source of truth
    All hardcoded values removed from the system
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/system_config.yaml"
        self.config_path = Path(self.config_file)
        
        # Default configurations
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.models: Dict[str, ModelConfig] = {}
        self.business: BusinessConfig = BusinessConfig()
        self.system: SystemConfig = SystemConfig()
        
        # Environment overrides
        self.env_overrides: Dict[str, Any] = {}
        
        # Load configuration
        self._load_configuration()
        self._load_environment_overrides()
        
        logger.info(f"Configuration loaded from {self.config_file}")
    
    def _load_configuration(self):
        """Load configuration from file"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_file} not found, creating defaults")
            self._create_default_configuration()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._parse_configuration(config_data)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _parse_configuration(self, config_data: Dict[str, Any]):
        """Parse configuration data into structured objects"""
        
        # Parse data sources
        for name, ds_config in config_data.get('data_sources', {}).items():
            self.data_sources[name] = DataSourceConfig(
                name=name,
                **ds_config
            )
        
        # Parse models
        for name, model_config in config_data.get('models', {}).items():
            self.models[name] = ModelConfig(
                name=name,
                **model_config
            )
        
        # Parse business configuration
        business_config = config_data.get('business', {})
        # Filter out any unknown fields for BusinessConfig
        try:
            valid_business_fields = set(BusinessConfig.__dataclass_fields__.keys())
            filtered_business_config = {k: v for k, v in business_config.items() if k in valid_business_fields}
            self.business = BusinessConfig(**filtered_business_config)
        except Exception as e:
            logger.warning(f"Error parsing business config: {e}, using defaults")
            self.business = BusinessConfig()
        
        # Parse system configuration
        system_config = config_data.get('system', {})
        self.system = SystemConfig(**system_config)
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # Pattern: MRCLEAN_SECTION_KEY=value
        for key, value in os.environ.items():
            if key.startswith('MRCLEAN_'):
                config_path = key[8:].lower().split('_')  # Remove MRCLEAN_ prefix
                self._set_nested_config(config_path, value)
        
        logger.debug(f"Loaded {len(self.env_overrides)} environment overrides")
    
    def _set_nested_config(self, path: List[str], value: str):
        """Set nested configuration value from environment variable"""
        # Convert string value to appropriate type
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
        
        # Store in environment overrides
        current = self.env_overrides
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _create_default_configuration(self):
        """Create default configuration file"""
        default_config = {
            'system': {
                'version': '1.0.0',
                'environment': 'development',
                'debug_mode': True,
                'data_storage_path': 'data',
                'model_storage_path': 'models',
                'log_storage_path': 'logs'
            },
            'business': {
                'company_name': 'Mr.Clean Insurance Platform',
                'regulatory_jurisdiction': 'US',
                'base_currency': 'USD',
                'default_profit_margin': 0.15,
                'default_expense_ratio': 0.05
            },
            'data_sources': {
                'fred_api': {
                    'type': 'api',
                    'connection_params': {
                        'base_url': 'https://api.stlouisfed.org/fred',
                        'api_key': '${FRED_API_KEY}',
                        'format': 'json'
                    },
                    'refresh_interval_hours': 24,
                    'timeout_seconds': 30,
                    'retry_count': 3,
                    'fallback_enabled': True
                },
                'alpha_vantage': {
                    'type': 'api',
                    'connection_params': {
                        'base_url': 'https://www.alphavantage.co/query',
                        'api_key': '${ALPHA_VANTAGE_KEY}'
                    },
                    'refresh_interval_hours': 1,
                    'timeout_seconds': 30,
                    'retry_count': 3,
                    'fallback_enabled': True
                },
                'soa_mortality_tables': {
                    'type': 'file',
                    'connection_params': {
                        'data_path': 'data/mortality_tables',
                        'table_version': '2017_CSO'
                    },
                    'refresh_interval_hours': 8760,  # Annual
                    'fallback_enabled': True
                }
            },
            'models': {
                'mortality_enhancement': {
                    'type': 'xgboost',
                    'parameters': {
                        'n_estimators': 500,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42
                    },
                    'training_schedule': 'monthly',
                    'validation_threshold': 0.85,
                    'auto_retrain': True,
                    'feature_columns': ['age', 'gender', 'smoker', 'bmi', 'medical_history']
                },
                'lapse_prediction': {
                    'type': 'random_forest',
                    'parameters': {
                        'n_estimators': 200,
                        'max_depth': 10,
                        'random_state': 42
                    },
                    'training_schedule': 'quarterly',
                    'validation_threshold': 0.80,
                    'auto_retrain': True
                },
                'economic_forecasting': {
                    'type': 'neural_network',
                    'parameters': {
                        'hidden_layers': [64, 32, 16],
                        'dropout_rate': 0.2,
                        'epochs': 100
                    },
                    'training_schedule': 'weekly',
                    'validation_threshold': 0.75,
                    'auto_retrain': True
                }
            }
        }
        
        # Create config directory
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        # Load the defaults we just created
        self._parse_configuration(default_config)
    
    def get_data_source_config(self, name: str) -> Optional[DataSourceConfig]:
        """Get configuration for specific data source"""
        config = self.data_sources.get(name)
        if config:
            # Apply environment overrides
            env_path = ['data_sources', name]
            if env_path[0] in self.env_overrides and env_path[1] in self.env_overrides[env_path[0]]:
                overrides = self.env_overrides[env_path[0]][env_path[1]]
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        return config
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        config = self.models.get(name)
        if config:
            # Apply environment overrides
            env_path = ['models', name]
            if env_path[0] in self.env_overrides and env_path[1] in self.env_overrides[env_path[0]]:
                overrides = self.env_overrides[env_path[0]][env_path[1]]
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        return config
    
    def get_business_config(self) -> BusinessConfig:
        """Get business configuration with environment overrides"""
        config = self.business
        if 'business' in self.env_overrides:
            for key, value in self.env_overrides['business'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration with environment overrides"""
        config = self.system
        if 'system' in self.env_overrides:
            for key, value in self.env_overrides['system'].items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for service (with environment variable substitution)"""
        service_config = self.get_data_source_config(service)
        if not service_config:
            return None
        
        api_key = service_config.connection_params.get('api_key')
        if not api_key:
            return None
        
        # Handle environment variable substitution
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            return os.environ.get(env_var)
        
        return api_key
    
    def update_configuration(self, section: str, key: str, value: Any, persist: bool = True):
        """Update configuration value at runtime"""
        # Update in-memory configuration
        if section == 'system':
            if hasattr(self.system, key):
                setattr(self.system, key, value)
        elif section == 'business':
            if hasattr(self.business, key):
                setattr(self.business, key, value)
        
        # Persist to file if requested
        if persist:
            self._save_configuration()
        
        logger.info(f"Updated configuration: {section}.{key} = {value}")
    
    def _save_configuration(self):
        """Save current configuration to file"""
        config_data = {
            'system': self.system.__dict__,
            'business': self.business.__dict__,
            'data_sources': {name: config.__dict__ for name, config in self.data_sources.items()},
            'models': {name: config.__dict__ for name, config in self.models.items()}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Validate data source configurations
        for name, config in self.data_sources.items():
            if config.type == 'api':
                if 'api_key' in config.connection_params:
                    api_key = self.get_api_key(name)
                    if not api_key:
                        issues['errors'].append(f"Missing API key for {name}")
                
                if 'base_url' not in config.connection_params:
                    issues['errors'].append(f"Missing base_url for {name}")
        
        # Validate model configurations
        for name, config in self.models.items():
            if not config.feature_columns:
                issues['warnings'].append(f"No feature columns specified for model {name}")
        
        # Validate paths exist
        system_config = self.get_system_config()
        for path_name in ['data_storage_path', 'model_storage_path', 'log_storage_path']:
            path_value = getattr(system_config, path_name)
            path_obj = Path(path_value)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    issues['warnings'].append(f"Created missing directory: {path_value}")
                except Exception as e:
                    issues['errors'].append(f"Cannot create directory {path_value}: {e}")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            'system': {
                'version': self.system.version,
                'environment': self.system.environment,
                'debug_mode': self.system.debug_mode
            },
            'data_sources': list(self.data_sources.keys()),
            'models': list(self.models.keys()),
            'business': {
                'company_name': self.business.company_name,
                'jurisdiction': self.business.regulatory_jurisdiction
            },
            'environment_overrides_count': len(self.env_overrides),
            'config_file': str(self.config_path),
            'last_loaded': datetime.now().isoformat()
        }

# Global configuration manager - single source of truth for all configuration
config_manager = ConfigurationManager()