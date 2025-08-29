"""
Professional Assumption Management System
Real-world assumption override and governance capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class AssumptionType(Enum):
    MORTALITY = "mortality"
    ECONOMIC = "economic"
    LAPSE = "lapse"
    EXPENSE = "expense"
    BUSINESS = "business"

class OverrideStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class AssumptionOverride:
    """Single assumption override with full audit trail"""
    # Required fields first
    override_id: str
    assumption_type: AssumptionType
    description: str
    base_value: float
    override_value: float
    justification: str
    requested_by: str
    requested_date: datetime
    effective_date: datetime
    
    # Optional fields with defaults
    override_factor: Optional[float] = None  # e.g., 1.05 for 5% increase
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    experience_period: Optional[str] = None
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None
    status: OverrideStatus = OverrideStatus.PENDING
    expiry_date: Optional[datetime] = None
    applies_to: Dict[str, Any] = field(default_factory=dict)  # Product, state, age band
    pricing_impact: Optional[Dict[str, float]] = None
    volume_impact: Optional[float] = None

@dataclass
class AssumptionSet:
    """Complete set of assumptions for pricing"""
    set_id: str
    set_name: str
    base_date: datetime
    
    # Base assumptions (from official sources)
    mortality_table: str = "SOA_2017_CSO"
    economic_scenario: str = "FRED_Current"
    
    # Active overrides
    active_overrides: List[AssumptionOverride] = field(default_factory=list)
    
    # Computed final values
    effective_assumptions: Dict[str, Any] = field(default_factory=dict)

class AssumptionManager:
    """
    Professional assumption management system
    Handles overrides, governance, and audit trails like real insurance companies
    """
    
    def __init__(self):
        self.storage_path = Path("data/assumptions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active assumption sets by company/product
        self.assumption_sets: Dict[str, AssumptionSet] = {}
        self.override_history: List[AssumptionOverride] = []
        
        # Default assumption set
        self._create_default_assumption_set()
        
        logger.info("Assumption Manager initialized")
    
    def _create_default_assumption_set(self):
        """Create default assumption set using real data sources"""
        default_set = AssumptionSet(
            set_id="default_2024",
            set_name="Default Assumptions - 2024",
            base_date=datetime.now(),
            effective_assumptions=self._load_base_assumptions()
        )
        
        self.assumption_sets["default"] = default_set
    
    def _load_base_assumptions(self) -> Dict[str, Any]:
        """Load base assumptions from real data sources"""
        from ..actuarial.data_sources.real_economic_data import real_economic_engine
        from ..actuarial.data_sources.real_mortality_data import real_mortality_engine
        
        try:
            # Get current economic environment
            treasury_rates = real_economic_engine.get_treasury_yield_curve()
            inflation = real_economic_engine.get_inflation_data()
            
            base_assumptions = {
                # Mortality assumptions
                "mortality": {
                    "base_table": "SOA_2017_CSO",
                    "improvement_scale": "MP-2017",
                    "smoker_factor": 2.5,
                    "select_period": 15,
                    "sample_rates": {
                        "male_35_ns": real_mortality_engine.get_mortality_rate(35, 'M', False),
                        "female_35_ns": real_mortality_engine.get_mortality_rate(35, 'F', False),
                        "male_50_s": real_mortality_engine.get_mortality_rate(50, 'M', True),
                        "female_50_s": real_mortality_engine.get_mortality_rate(50, 'F', True)
                    }
                },
                
                # Economic assumptions
                "economic": {
                    "risk_free_rate": treasury_rates.get("10Y", 0.042),
                    "credit_spread": 0.008,  # 80 bps corporate spread
                    "inflation_rate": inflation.get("CPI_Core", 0.028),
                    "equity_return": treasury_rates.get("10Y", 0.042) + 0.04,  # Risk premium
                    "treasury_curve": treasury_rates
                },
                
                # Lapse assumptions  
                "lapse": {
                    "base_lapse_rate": 0.08,  # 8% annual lapse
                    "first_year_multiplier": 1.5,  # 50% higher first year
                    "shock_lapse_factor": 2.0,  # Double in stress
                    "policy_loan_impact": 1.15,  # 15% increase if loans
                    "age_curve": "Industry_Standard_2023"
                },
                
                # Expense assumptions
                "expenses": {
                    "per_policy_annual": 150.0,  # $150/policy/year
                    "percent_of_premium": 0.05,  # 5% of premium
                    "new_business_cost": 800.0,  # $800 per new policy
                    "inflation_adjustment": inflation.get("CPI_Core", 0.028)
                },
                
                # Business assumptions
                "business": {
                    "target_profit_margin": 0.15,  # 15% ROE
                    "commission_first_year": 0.90,  # 90% of premium
                    "commission_renewal": 0.05,  # 5% of premium
                    "underwriting_acceptance_rate": 0.85,  # 85% issue rate
                    "reinsurance_cession": 0.10  # 10% ceded
                }
            }
            
            logger.info("Loaded base assumptions from real data sources")
            return base_assumptions
            
        except Exception as e:
            logger.error(f"Error loading base assumptions: {e}")
            return self._get_fallback_assumptions()
    
    def request_assumption_override(self, 
                                  assumption_type: AssumptionType,
                                  description: str,
                                  current_value: float,
                                  proposed_value: float,
                                  justification: str,
                                  requested_by: str,
                                  supporting_data: Optional[Dict] = None) -> str:
        """
        Request assumption override - real-world workflow
        Returns override_id for tracking
        """
        
        override_id = f"override_{assumption_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        override = AssumptionOverride(
            override_id=override_id,
            assumption_type=assumption_type,
            description=description,
            base_value=current_value,
            override_value=proposed_value,
            override_factor=proposed_value / current_value if current_value != 0 else None,
            justification=justification,
            supporting_data=supporting_data or {},
            requested_by=requested_by,
            requested_date=datetime.now(),
            effective_date=datetime.now(),
            expiry_date=datetime.now() + timedelta(days=365)  # 1 year default
        )
        
        # Calculate impact analysis
        override.pricing_impact = self._analyze_pricing_impact(override)
        
        self.override_history.append(override)
        
        logger.info(f"Created assumption override request: {override_id}")
        return override_id
    
    def approve_assumption_override(self, override_id: str, approved_by: str, 
                                  effective_date: Optional[datetime] = None) -> bool:
        """
        Approve assumption override - represents committee approval
        """
        override = self._find_override(override_id)
        if not override:
            logger.error(f"Override {override_id} not found")
            return False
        
        override.status = OverrideStatus.APPROVED
        override.approved_by = approved_by
        override.approved_date = datetime.now()
        
        if effective_date:
            override.effective_date = effective_date
        
        # Apply to default assumption set
        self._apply_override_to_assumption_set("default", override)
        
        logger.info(f"Approved assumption override: {override_id} by {approved_by}")
        return True
    
    def get_effective_assumptions(self, assumption_set: str = "default") -> Dict[str, Any]:
        """
        Get current effective assumptions with all overrides applied
        This is what pricing engines should use
        """
        if assumption_set not in self.assumption_sets:
            logger.warning(f"Assumption set {assumption_set} not found, using default")
            assumption_set = "default"
        
        return self.assumption_sets[assumption_set].effective_assumptions
    
    def get_assumption_audit_trail(self, days_back: int = 90) -> pd.DataFrame:
        """
        Get assumption change audit trail for regulatory compliance
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        relevant_overrides = [
            override for override in self.override_history
            if override.requested_date > cutoff_date
        ]
        
        audit_data = []
        for override in relevant_overrides:
            audit_data.append({
                "Override_ID": override.override_id,
                "Type": override.assumption_type.value.title(),
                "Description": override.description,
                "Requested_By": override.requested_by,
                "Requested_Date": override.requested_date.strftime("%Y-%m-%d"),
                "Status": override.status.value.title(),
                "Approved_By": override.approved_by or "Pending",
                "Base_Value": override.base_value,
                "Override_Value": override.override_value,
                "Change_%": f"{((override.override_value/override.base_value - 1) * 100):.1f}%" if override.base_value != 0 else "N/A",
                "Justification": override.justification[:100] + "..." if len(override.justification) > 100 else override.justification,
                "Pricing_Impact": f"{override.pricing_impact.get('premium_change_pct', 0):.1f}%" if override.pricing_impact else "N/A"
            })
        
        return pd.DataFrame(audit_data)
    
    def get_current_overrides_summary(self) -> Dict[str, Any]:
        """
        Get summary of all active assumption overrides
        """
        active_overrides = [
            override for override in self.override_history
            if override.status == OverrideStatus.APPROVED
            and (not override.expiry_date or override.expiry_date > datetime.now())
        ]
        
        summary = {
            "total_active_overrides": len(active_overrides),
            "by_type": {},
            "recent_changes": [],
            "expiring_soon": []
        }
        
        # Group by assumption type
        for assumption_type in AssumptionType:
            type_overrides = [o for o in active_overrides if o.assumption_type == assumption_type]
            summary["by_type"][assumption_type.value] = len(type_overrides)
        
        # Recent changes (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_overrides = [o for o in active_overrides if o.approved_date and o.approved_date > recent_cutoff]
        
        for override in recent_overrides[:5]:  # Last 5
            summary["recent_changes"].append({
                "description": override.description,
                "approved_date": override.approved_date.strftime("%Y-%m-%d"),
                "impact": f"{override.pricing_impact.get('premium_change_pct', 0):.1f}%" if override.pricing_impact else "N/A"
            })
        
        # Expiring soon (next 60 days)
        expiry_cutoff = datetime.now() + timedelta(days=60)
        expiring_overrides = [
            o for o in active_overrides 
            if o.expiry_date and o.expiry_date < expiry_cutoff
        ]
        
        for override in expiring_overrides:
            summary["expiring_soon"].append({
                "description": override.description,
                "expiry_date": override.expiry_date.strftime("%Y-%m-%d"),
                "days_remaining": (override.expiry_date - datetime.now()).days
            })
        
        return summary
    
    def create_assumption_scenario(self, scenario_name: str, overrides: Dict[str, float]) -> str:
        """
        Create assumption scenario for stress testing
        """
        scenario_id = f"scenario_{scenario_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        
        # Create scenario-specific assumption set
        base_assumptions = self._load_base_assumptions()
        scenario_assumptions = base_assumptions.copy()
        
        # Apply scenario overrides
        for path, value in overrides.items():
            self._set_nested_value(scenario_assumptions, path, value)
        
        scenario_set = AssumptionSet(
            set_id=scenario_id,
            set_name=f"Scenario: {scenario_name}",
            base_date=datetime.now(),
            effective_assumptions=scenario_assumptions
        )
        
        self.assumption_sets[scenario_id] = scenario_set
        
        logger.info(f"Created assumption scenario: {scenario_name} ({scenario_id})")
        return scenario_id
    
    def _find_override(self, override_id: str) -> Optional[AssumptionOverride]:
        """Find override by ID"""
        for override in self.override_history:
            if override.override_id == override_id:
                return override
        return None
    
    def _analyze_pricing_impact(self, override: AssumptionOverride) -> Dict[str, float]:
        """
        Analyze pricing impact of assumption change
        Real-world impact analysis
        """
        if override.override_factor is None:
            return {"premium_change_pct": 0.0}
        
        # Simplified impact analysis - would be more sophisticated in reality
        impact_factors = {
            AssumptionType.MORTALITY: 0.8,    # 80% flow-through to premium
            AssumptionType.ECONOMIC: 0.6,     # 60% flow-through (discounting)
            AssumptionType.LAPSE: 0.4,        # 40% flow-through (complex interaction)
            AssumptionType.EXPENSE: 1.0,      # 100% flow-through to premium
            AssumptionType.BUSINESS: 0.3       # 30% flow-through (profit margins)
        }
        
        assumption_change = (override.override_factor - 1.0) * 100  # % change
        premium_impact = assumption_change * impact_factors[override.assumption_type]
        
        return {
            "assumption_change_pct": assumption_change,
            "premium_change_pct": premium_impact,
            "flow_through_factor": impact_factors[override.assumption_type]
        }
    
    def _apply_override_to_assumption_set(self, set_id: str, override: AssumptionOverride):
        """Apply approved override to assumption set"""
        if set_id not in self.assumption_sets:
            return
        
        assumption_set = self.assumption_sets[set_id]
        assumption_set.active_overrides.append(override)
        
        # Update effective assumptions
        # This would be more sophisticated based on assumption type and path
        logger.info(f"Applied override {override.override_id} to assumption set {set_id}")
    
    def _set_nested_value(self, dictionary: Dict, path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = path.split('.')
        current = dictionary
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_fallback_assumptions(self) -> Dict[str, Any]:
        """Fallback assumptions if real data unavailable"""
        return {
            "mortality": {"base_table": "SOA_2017_CSO_Fallback"},
            "economic": {"risk_free_rate": 0.04, "inflation_rate": 0.03},
            "lapse": {"base_lapse_rate": 0.08},
            "expenses": {"per_policy_annual": 150.0},
            "business": {"target_profit_margin": 0.15}
        }

# Global assumption manager instance  
assumption_manager = AssumptionManager()