"""
Real Data Sources Module
Professional actuarial data integration with real APIs and official tables
"""

from .real_mortality_data import real_mortality_engine, RealMortalityDataEngine
from .real_economic_data import real_economic_engine, RealEconomicDataEngine

__all__ = [
    'real_mortality_engine',
    'RealMortalityDataEngine', 
    'real_economic_engine',
    'RealEconomicDataEngine'
]