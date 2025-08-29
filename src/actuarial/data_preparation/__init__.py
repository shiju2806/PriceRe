"""
Actuarial Data Preparation Module
Professional data validation, cleaning, and generation
"""

from .data_validator import ActuarialDataValidator, DataQualityLevel, ValidationResult
from .data_cleaner import ActuarialDataCleaner, CleaningConfig, CleaningResult
from .comprehensive_data_generator import (
    ComprehensiveActuarialDataGenerator, 
    DataGenerationConfig, 
    quick_generate_test_data,
    ProductType,
    PolicyStatus
)

__version__ = "1.0.0"
__author__ = "Claude & Shiju - Actuarial Data Processing"

# Available data preparation capabilities
DATA_CAPABILITIES = {
    "validation": {
        "description": "SOA-compliant actuarial data validation",
        "features": ["Field validation", "Relationship checks", "Quality scoring", "Regulatory compliance"],
        "standards": ["SOA", "NAIC", "GAAP"]
    },
    "cleaning": {
        "description": "Industry-standard data cleaning and standardization", 
        "features": ["Smart imputation", "Outlier handling", "Category standardization", "Type conversion"],
        "methods": ["Statistical imputation", "Actuarial rules", "Quality tracking"]
    },
    "generation": {
        "description": "Comprehensive test data generation",
        "features": ["Realistic distributions", "Multi-table relationships", "Economic scenarios", "Quality issues"],
        "scale": "Up to 100,000+ policies with full actuarial context"
    }
}

def get_data_prep_capabilities():
    """Return available data preparation capabilities"""
    return DATA_CAPABILITIES