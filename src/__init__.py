"""
PricingFlow: AI-Powered Insurance Pricing Platform

Copyright 2024 PricingFlow. All rights reserved.
"""

__version__ = "0.1.0"
__author__ = "PricingFlow Team"

# Import available modules
try:
    from .engines.life_insurance import LifeInsurancePricer
    _LIFE_INSURANCE_AVAILABLE = True
except ImportError:
    _LIFE_INSURANCE_AVAILABLE = False

try:
    from .reinsurance.data_generator import ReinsuranceDataGenerator
    from .reinsurance.treaty_pricer import TreatyPricer
    from .reinsurance.feature_engineering import ReinsuranceFeatures
    _REINSURANCE_AVAILABLE = True
except ImportError:
    _REINSURANCE_AVAILABLE = False

__all__ = []

if _LIFE_INSURANCE_AVAILABLE:
    __all__.append("LifeInsurancePricer")

if _REINSURANCE_AVAILABLE:
    __all__.extend(["ReinsuranceDataGenerator", "TreatyPricer", "ReinsuranceFeatures"])