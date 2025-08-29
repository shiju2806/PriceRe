"""
Real Actuarial Engineering Platform
Built with industry-standard methodologies and open-source libraries

This module provides proper actuarial calculations for life reinsurance:
- SOA mortality tables and life contingency formulas
- GAAP/Statutory reserve calculations (LDTI compliant)
- NAIC Risk-Based Capital requirements
- Economic capital modeling
- Regulatory compliance validation

Uses established libraries:
- lifelib: Professional actuarial modeling
- pyliferisk: Life contingency calculations  
- chainladder: Reserve development
- tmval: Time value of money

All calculations follow actuarial standards and can be audited.
"""

from .mortality.mortality_engine import MortalityEngine, LifeContingencies
from .reserves.reserve_engine import ReserveEngine  
from .capital.capital_engine import CapitalEngine

__version__ = "1.0.0"
__author__ = "Claude & Shiju - Real Actuarial Implementation"

# Available engines
ACTUARIAL_ENGINES = {
    "mortality": MortalityEngine,
    "reserves": ReserveEngine, 
    "capital": CapitalEngine
}

def get_engine_status():
    """Return status of all actuarial engines"""
    
    try:
        from pyliferisk import qx, SULT80CNSMT
        mortality_status = "✅ Available"
    except:
        mortality_status = "❌ Error"
        
    try:
        import lifelib
        lifelib_status = "✅ Available" 
    except:
        lifelib_status = "❌ Error"
        
    try:
        import chainladder as cl
        chainladder_status = "✅ Available"
    except:
        chainladder_status = "❌ Error"
        
    return {
        "mortality_calculations": mortality_status,
        "lifelib_models": lifelib_status, 
        "reserve_triangles": chainladder_status,
        "total_engines": len(ACTUARIAL_ENGINES)
    }