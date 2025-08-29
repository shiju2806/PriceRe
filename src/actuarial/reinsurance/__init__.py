"""
Reinsurance Pricing Module
Specialized pricing for life and savings & retirement reinsurance
"""

from .treaty_pricing_engine import (
    TreatyPricingEngine, TreatyResult, TreatyTerms, CedentExperience,
    TreatyType, BusinessLine
)
from .cedent_data_processor import CedentDataProcessor, CedentProfile
from .portfolio_analysis import PortfolioAnalyzer, PortfolioRisk
from .catastrophe_modeling import CatastropheModeler, CatRisk

__version__ = "1.0.0"
__author__ = "Claude & Shiju - Reinsurance Pricing Specialists"

# Available reinsurance capabilities
REINSURANCE_CAPABILITIES = {
    "treaty_pricing": {
        "description": "Complete reinsurance treaty pricing",
        "types": ["Quota Share", "Surplus", "XS of Loss", "Stop Loss", "Catastrophe"],
        "features": ["Portfolio modeling", "Aggregate limits", "Profit commission", "Experience rating"]
    },
    "cedent_analysis": {
        "description": "Ceding company risk assessment",
        "features": ["Underwriting quality", "Claims experience", "Lapse patterns", "Geographic concentration"],
        "scoring": "Comprehensive cedent risk scoring system"
    },
    "portfolio_modeling": {
        "description": "Block of business analysis",
        "features": ["Aggregate mortality", "Correlation modeling", "Concentration risk", "Diversification benefits"],
        "methods": ["Monte Carlo simulation", "Copula modeling", "Stress testing"]
    },
    "catastrophe_modeling": {
        "description": "Large loss event modeling",
        "features": ["Pandemic risk", "Natural disasters", "Terrorism", "Economic shocks"],
        "models": ["Frequency-severity", "Extreme value theory", "Scenario analysis"]
    }
}

def get_reinsurance_capabilities():
    """Return available reinsurance pricing capabilities"""
    return REINSURANCE_CAPABILITIES