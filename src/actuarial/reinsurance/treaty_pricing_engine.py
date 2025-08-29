"""
Reinsurance Treaty Pricing Engine
Specialized pricing for life and retirement reinsurance treaties
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
import logging

class TreatyType(Enum):
    QUOTA_SHARE = "quota_share"
    SURPLUS = "surplus"
    XS_OF_LOSS = "xs_of_loss"
    STOP_LOSS = "stop_loss"
    CATASTROPHE = "catastrophe"
    COINSURANCE = "coinsurance"

class BusinessLine(Enum):
    INDIVIDUAL_LIFE = "individual_life"
    GROUP_LIFE = "group_life"
    ANNUITIES = "annuities"
    PENSION = "pension"
    DISABILITY = "disability"
    CRITICAL_ILLNESS = "critical_illness"

@dataclass
class TreatyTerms:
    """Reinsurance treaty terms and conditions"""
    treaty_type: TreatyType
    business_lines: List[BusinessLine]
    effective_date: date
    expiry_date: date
    
    # Retention and limits
    retention_basis: str  # "amount", "percentage", "first_loss"
    retention_amount: Optional[float] = None
    retention_percentage: Optional[float] = None
    treaty_limit: Optional[float] = None
    
    # Financial terms
    reinsurance_premium_rate: float = 0.0
    commission_rate: float = 0.0
    profit_commission_rate: float = 0.0
    profit_commission_threshold: float = 0.75  # Loss ratio threshold
    
    # Experience rating
    experience_rating: bool = False
    experience_period_years: int = 3
    credibility_factor: float = 0.5
    
    # Territory and currency
    territory: str = "USA"
    currency: str = "USD"
    
    # Special provisions
    aggregate_deductible: Optional[float] = None
    aggregate_limit: Optional[float] = None
    reinstatement_premium: float = 1.0
    catastrophe_provision: bool = False

@dataclass
class CedentExperience:
    """Ceding company historical experience"""
    cedent_name: str
    business_line: BusinessLine
    experience_years: List[int]
    
    # Volume metrics
    premium_volume: List[float]
    face_amount_inforce: List[float]
    policy_count: List[int]
    
    # Loss experience
    incurred_claims: List[float]
    paid_claims: List[float]
    loss_ratios: List[float]
    
    # Lapse experience
    lapse_rates: List[float]
    surrender_rates: List[float]
    
    # Underwriting quality
    av_mortality_ratios: List[float] = field(default_factory=list)  # A/E ratios
    underwriting_grade: str = "B"  # A, B, C, D rating
    
    # Portfolio characteristics
    avg_face_amount: float = 250000
    avg_issue_age: float = 42
    smoker_percentage: float = 0.15
    male_percentage: float = 0.52
    
    # Geographic concentration
    top_state_concentration: float = 0.25
    urban_percentage: float = 0.80

@dataclass
class TreatyResult:
    """Comprehensive treaty pricing result"""
    treaty_id: str
    cedent_name: str
    treaty_terms: TreatyTerms
    
    # Pricing components
    expected_loss_ratio: float
    expense_ratio: float
    profit_margin: float
    risk_margin: float
    
    # Final pricing
    technical_rate: float  # Pure risk rate
    loading_factor: float  # Total loadings
    gross_rate: float  # Final reinsurance rate
    
    # Capital requirements
    required_capital: float
    capital_charge: float
    
    # Risk metrics
    var_99_5: float  # 99.5% VaR
    expected_shortfall: float  # Conditional tail expectation
    diversification_benefit: float
    
    # Profit analysis
    expected_profit: float
    profit_margin_pct: float
    roe_target: float
    
    # Sensitivity analysis
    rate_sensitivities: Dict[str, float] = field(default_factory=dict)
    break_even_loss_ratio: float = 0.0
    
    # Recommendations
    pricing_confidence: str = "Medium"  # High, Medium, Low
    key_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Supporting data
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)
    pricing_assumptions: Dict[str, Any] = field(default_factory=dict)

class TreatyPricingEngine:
    """
    Comprehensive reinsurance treaty pricing engine
    Handles all major treaty types for life and retirement business
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Industry benchmarks
        self.industry_benchmarks = self._load_industry_benchmarks()
        
        # Risk factors
        self.risk_factors = self._initialize_risk_factors()
        
        # Capital requirements
        self.capital_factors = self._load_capital_factors()
    
    def _load_industry_benchmarks(self) -> Dict[str, Any]:
        """Load industry loss ratio and expense benchmarks"""
        
        return {
            "loss_ratios": {
                BusinessLine.INDIVIDUAL_LIFE: {"mean": 0.68, "std": 0.15, "75th": 0.75, "90th": 0.85},
                BusinessLine.GROUP_LIFE: {"mean": 0.72, "std": 0.18, "75th": 0.82, "90th": 0.92},
                BusinessLine.ANNUITIES: {"mean": 0.45, "std": 0.12, "75th": 0.52, "90th": 0.60},
                BusinessLine.PENSION: {"mean": 0.55, "std": 0.10, "75th": 0.62, "90th": 0.68},
                BusinessLine.DISABILITY: {"mean": 0.78, "std": 0.20, "75th": 0.88, "90th": 1.00},
            },
            "expense_ratios": {
                TreatyType.QUOTA_SHARE: 0.25,
                TreatyType.SURPLUS: 0.20,
                TreatyType.XS_OF_LOSS: 0.15,
                TreatyType.STOP_LOSS: 0.12,
                TreatyType.CATASTROPHE: 0.10
            },
            "profit_margins": {
                "conservative": 0.08,
                "standard": 0.12,
                "aggressive": 0.18
            }
        }
    
    def _initialize_risk_factors(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk adjustment factors"""
        
        return {
            "cedent_quality": {
                "A": 0.90,  # 10% reduction in expected loss
                "B": 1.00,  # Baseline
                "C": 1.15,  # 15% increase
                "D": 1.35   # 35% increase
            },
            "geographic_concentration": {
                "low": 1.00,      # <20% in any state
                "medium": 1.05,   # 20-40% in top state
                "high": 1.15,     # >40% in top state
                "extreme": 1.30   # >60% in top state
            },
            "product_mix": {
                "term_heavy": 1.20,      # >60% term life
                "balanced": 1.00,        # Mixed portfolio
                "permanent_heavy": 0.85, # >60% permanent
                "annuity_heavy": 0.70    # >60% annuities
            },
            "size_adjustment": {
                "small": 1.20,    # <$10M premium
                "medium": 1.05,   # $10M-$100M
                "large": 1.00,    # $100M-$1B
                "jumbo": 0.95     # >$1B
            }
        }
    
    def _load_capital_factors(self) -> Dict[str, float]:
        """Load capital requirement factors by business line"""
        
        return {
            BusinessLine.INDIVIDUAL_LIFE.value: 0.045,     # 4.5% of net amount at risk
            BusinessLine.GROUP_LIFE.value: 0.060,          # 6.0% - higher volatility
            BusinessLine.ANNUITIES.value: 0.025,           # 2.5% - lower risk
            BusinessLine.PENSION.value: 0.035,             # 3.5% - moderate risk
            BusinessLine.DISABILITY.value: 0.085,          # 8.5% - high volatility
            BusinessLine.CRITICAL_ILLNESS.value: 0.070     # 7.0% - high severity
        }
    
    def price_treaty(self, 
                    treaty_terms: TreatyTerms,
                    cedent_experience: CedentExperience,
                    portfolio_data: Optional[pd.DataFrame] = None) -> TreatyResult:
        """
        Price a reinsurance treaty comprehensively
        
        Args:
            treaty_terms: Treaty structure and terms
            cedent_experience: Historical experience of ceding company
            portfolio_data: Optional detailed portfolio data
            
        Returns:
            Complete treaty pricing with all components
        """
        
        self.logger.info(f"Pricing {treaty_terms.treaty_type.value} treaty for {cedent_experience.cedent_name}")
        
        # Step 1: Analyze cedent experience
        cedent_analysis = self._analyze_cedent_experience(cedent_experience)
        
        # Step 2: Calculate expected loss ratio
        expected_loss_ratio = self._calculate_expected_loss_ratio(
            treaty_terms, cedent_experience, cedent_analysis
        )
        
        # Step 3: Determine expense ratio
        expense_ratio = self._calculate_expense_ratio(treaty_terms, cedent_experience)
        
        # Step 4: Apply risk adjustments
        risk_adjusted_loss_ratio = self._apply_risk_adjustments(
            expected_loss_ratio, cedent_analysis, treaty_terms
        )
        
        # Step 5: Calculate capital requirements
        required_capital, capital_charge = self._calculate_capital_requirements(
            treaty_terms, cedent_experience, portfolio_data
        )
        
        # Step 6: Determine profit margin
        profit_margin = self._calculate_profit_margin(
            treaty_terms, cedent_analysis, required_capital
        )
        
        # Step 7: Calculate technical rate
        technical_rate = risk_adjusted_loss_ratio
        
        # Step 8: Apply loadings
        loading_factor = expense_ratio + profit_margin + capital_charge / 100
        gross_rate = technical_rate + loading_factor
        
        # Step 9: Risk metrics
        var_99_5, expected_shortfall = self._calculate_risk_metrics(
            risk_adjusted_loss_ratio, cedent_experience, portfolio_data
        )
        
        # Step 10: Sensitivity analysis
        sensitivities = self._perform_sensitivity_analysis(
            gross_rate, risk_adjusted_loss_ratio, loading_factor
        )
        
        # Step 11: Generate recommendations
        recommendations, key_risks, confidence = self._generate_recommendations(
            cedent_analysis, treaty_terms, gross_rate
        )
        
        return TreatyResult(
            treaty_id=f"TRT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            cedent_name=cedent_experience.cedent_name,
            treaty_terms=treaty_terms,
            
            expected_loss_ratio=risk_adjusted_loss_ratio,
            expense_ratio=expense_ratio,
            profit_margin=profit_margin,
            risk_margin=capital_charge / 100,
            
            technical_rate=technical_rate,
            loading_factor=loading_factor,
            gross_rate=gross_rate,
            
            required_capital=required_capital,
            capital_charge=capital_charge,
            
            var_99_5=var_99_5,
            expected_shortfall=expected_shortfall,
            diversification_benefit=0.15,  # Assume 15% diversification
            
            expected_profit=gross_rate - risk_adjusted_loss_ratio - expense_ratio,
            profit_margin_pct=profit_margin * 100,
            roe_target=15.0,  # 15% ROE target
            
            rate_sensitivities=sensitivities,
            break_even_loss_ratio=gross_rate - expense_ratio,
            
            pricing_confidence=confidence,
            key_risks=key_risks,
            recommendations=recommendations,
            
            portfolio_summary=self._create_portfolio_summary(cedent_experience),
            pricing_assumptions=self._document_assumptions(treaty_terms)
        )
    
    def _analyze_cedent_experience(self, cedent_exp: CedentExperience) -> Dict[str, Any]:
        """Comprehensive analysis of cedent's historical experience"""
        
        analysis = {}
        
        # Loss ratio trends
        if len(cedent_exp.loss_ratios) >= 3:
            recent_avg = np.mean(cedent_exp.loss_ratios[-3:])
            historical_avg = np.mean(cedent_exp.loss_ratios)
            trend = "improving" if recent_avg < historical_avg * 0.95 else \
                   "deteriorating" if recent_avg > historical_avg * 1.05 else "stable"
            
            analysis["loss_ratio_trend"] = trend
            analysis["recent_loss_ratio"] = recent_avg
            analysis["historical_loss_ratio"] = historical_avg
        
        # Volume stability
        premium_cv = np.std(cedent_exp.premium_volume) / np.mean(cedent_exp.premium_volume)
        analysis["volume_stability"] = "stable" if premium_cv < 0.2 else \
                                     "moderate" if premium_cv < 0.4 else "volatile"
        
        # Underwriting quality score
        if cedent_exp.av_mortality_ratios:
            avg_ae_ratio = np.mean(cedent_exp.av_mortality_ratios)
            uw_score = "excellent" if avg_ae_ratio < 0.85 else \
                      "good" if avg_ae_ratio < 1.0 else \
                      "poor" if avg_ae_ratio > 1.2 else "average"
            analysis["underwriting_quality"] = uw_score
            analysis["avg_ae_ratio"] = avg_ae_ratio
        
        # Portfolio risk assessment
        analysis["concentration_risk"] = "high" if cedent_exp.top_state_concentration > 0.4 else \
                                       "medium" if cedent_exp.top_state_concentration > 0.25 else "low"
        
        # Size classification
        recent_premium = cedent_exp.premium_volume[-1] if cedent_exp.premium_volume else 0
        analysis["size_category"] = "large" if recent_premium > 100_000_000 else \
                                   "medium" if recent_premium > 10_000_000 else "small"
        
        return analysis
    
    def _calculate_expected_loss_ratio(self, 
                                     terms: TreatyTerms, 
                                     cedent_exp: CedentExperience,
                                     analysis: Dict[str, Any]) -> float:
        """Calculate base expected loss ratio"""
        
        # Start with cedent's historical experience
        if len(cedent_exp.loss_ratios) >= 3:
            # Weight recent years more heavily
            weights = np.array([0.5, 0.3, 0.2][:len(cedent_exp.loss_ratios[-3:])])
            base_loss_ratio = np.average(cedent_exp.loss_ratios[-3:], weights=weights)
        else:
            # Use industry benchmark
            business_line = terms.business_lines[0] if terms.business_lines else BusinessLine.INDIVIDUAL_LIFE
            base_loss_ratio = self.industry_benchmarks["loss_ratios"][business_line]["mean"]
        
        # Adjust for treaty type (different treaties see different loss patterns)
        treaty_adjustments = {
            TreatyType.QUOTA_SHARE: 1.00,      # Sees all losses proportionally
            TreatyType.SURPLUS: 0.85,          # Excludes smaller losses
            TreatyType.XS_OF_LOSS: 0.45,       # Only large losses
            TreatyType.STOP_LOSS: 0.25,        # Only catastrophic years
            TreatyType.CATASTROPHE: 0.15       # Only extreme events
        }
        
        adjusted_loss_ratio = base_loss_ratio * treaty_adjustments.get(terms.treaty_type, 1.0)
        
        return max(0.05, min(2.0, adjusted_loss_ratio))  # Reasonable bounds
    
    def _calculate_expense_ratio(self, terms: TreatyTerms, cedent_exp: CedentExperience) -> float:
        """Calculate expense ratio based on treaty type and size"""
        
        base_expense = self.industry_benchmarks["expense_ratios"][terms.treaty_type]
        
        # Size adjustment (larger treaties have economies of scale)
        recent_premium = cedent_exp.premium_volume[-1] if cedent_exp.premium_volume else 10_000_000
        size_adjustment = 1.0
        
        if recent_premium > 500_000_000:
            size_adjustment = 0.8  # 20% reduction for jumbo treaties
        elif recent_premium > 100_000_000:
            size_adjustment = 0.9  # 10% reduction for large treaties
        elif recent_premium < 10_000_000:
            size_adjustment = 1.3  # 30% increase for small treaties
        
        return base_expense * size_adjustment
    
    def _apply_risk_adjustments(self, 
                               base_loss_ratio: float,
                               analysis: Dict[str, Any],
                               terms: TreatyTerms) -> float:
        """Apply various risk adjustments to base loss ratio"""
        
        adjustment_factor = 1.0
        
        # Underwriting quality adjustment
        uw_grade = analysis.get("underwriting_quality", "average")
        if uw_grade == "excellent":
            adjustment_factor *= 0.90
        elif uw_grade == "good":
            adjustment_factor *= 0.95
        elif uw_grade == "poor":
            adjustment_factor *= 1.20
        
        # Concentration risk adjustment
        conc_risk = analysis.get("concentration_risk", "medium")
        concentration_adj = {"low": 1.0, "medium": 1.05, "high": 1.15}
        adjustment_factor *= concentration_adj[conc_risk]
        
        # Loss trend adjustment
        trend = analysis.get("loss_ratio_trend", "stable")
        if trend == "deteriorating":
            adjustment_factor *= 1.10
        elif trend == "improving":
            adjustment_factor *= 0.95
        
        # Size adjustment
        size_cat = analysis.get("size_category", "medium")
        size_adj = self.risk_factors["size_adjustment"]
        if size_cat in size_adj:
            adjustment_factor *= size_adj[size_cat]
        
        return base_loss_ratio * adjustment_factor
    
    def _calculate_capital_requirements(self, 
                                       terms: TreatyTerms,
                                       cedent_exp: CedentExperience,
                                       portfolio_data: Optional[pd.DataFrame]) -> Tuple[float, float]:
        """Calculate required capital and capital charge"""
        
        # Estimate net amount at risk
        recent_premium = cedent_exp.premium_volume[-1] if cedent_exp.premium_volume else 10_000_000
        
        # Rough estimate: net amount at risk = premium / (mortality rate * loading)
        # For life insurance, assume ~0.8% mortality and 40% loading
        estimated_nar = recent_premium / (0.008 * 1.4)
        
        # Apply capital factor based on business line
        business_line = terms.business_lines[0] if terms.business_lines else BusinessLine.INDIVIDUAL_LIFE
        capital_factor = self.capital_factors.get(business_line.value, 0.045)
        
        # Adjust for treaty type
        treaty_capital_adj = {
            TreatyType.QUOTA_SHARE: 1.0,
            TreatyType.SURPLUS: 0.8,
            TreatyType.XS_OF_LOSS: 1.5,
            TreatyType.STOP_LOSS: 2.0,
            TreatyType.CATASTROPHE: 2.5
        }
        
        adjusted_factor = capital_factor * treaty_capital_adj.get(terms.treaty_type, 1.0)
        required_capital = estimated_nar * adjusted_factor
        
        # Capital charge as percentage of premium (assume 12% cost of capital)
        capital_charge = (required_capital * 0.12) / recent_premium * 100
        
        return required_capital, capital_charge
    
    def _calculate_profit_margin(self, 
                                terms: TreatyTerms,
                                analysis: Dict[str, Any], 
                                required_capital: float) -> float:
        """Calculate appropriate profit margin"""
        
        # Base profit margin
        base_margin = self.industry_benchmarks["profit_margins"]["standard"]
        
        # Risk adjustments
        risk_multiplier = 1.0
        
        # Treaty type risk
        treaty_risk = {
            TreatyType.QUOTA_SHARE: 1.0,
            TreatyType.SURPLUS: 1.1,
            TreatyType.XS_OF_LOSS: 1.4,
            TreatyType.STOP_LOSS: 1.8,
            TreatyType.CATASTROPHE: 2.2
        }
        
        risk_multiplier *= treaty_risk.get(terms.treaty_type, 1.0)
        
        # Cedent quality adjustment
        uw_quality = analysis.get("underwriting_quality", "average")
        if uw_quality in ["excellent", "good"]:
            risk_multiplier *= 0.9
        elif uw_quality == "poor":
            risk_multiplier *= 1.3
        
        return base_margin * risk_multiplier
    
    def _calculate_risk_metrics(self, 
                               loss_ratio: float,
                               cedent_exp: CedentExperience,
                               portfolio_data: Optional[pd.DataFrame]) -> Tuple[float, float]:
        """Calculate VaR and Expected Shortfall"""
        
        # Simplified calculation - in practice would use Monte Carlo
        # Assume loss ratios follow a lognormal distribution
        
        # Historical volatility
        if len(cedent_exp.loss_ratios) >= 3:
            volatility = np.std(cedent_exp.loss_ratios)
        else:
            volatility = 0.3  # Default assumption
        
        # 99.5% VaR (approximate)
        import scipy.stats as stats
        var_99_5 = loss_ratio * np.exp(stats.norm.ppf(0.995) * volatility - 0.5 * volatility**2)
        
        # Expected Shortfall (approximate)
        expected_shortfall = var_99_5 * 1.2  # Rough approximation
        
        return var_99_5, expected_shortfall
    
    def _perform_sensitivity_analysis(self, 
                                     gross_rate: float,
                                     loss_ratio: float, 
                                     loading_factor: float) -> Dict[str, float]:
        """Perform sensitivity analysis on key variables"""
        
        sensitivities = {}
        
        # Loss ratio sensitivity
        sensitivities["loss_ratio_+10%"] = (loss_ratio * 1.1 + loading_factor) / gross_rate - 1
        sensitivities["loss_ratio_-10%"] = (loss_ratio * 0.9 + loading_factor) / gross_rate - 1
        
        # Expense sensitivity
        sensitivities["expenses_+20%"] = (loss_ratio + loading_factor * 1.2) / gross_rate - 1
        
        # Volume sensitivity
        sensitivities["volume_+50%"] = -0.15  # Economies of scale
        sensitivities["volume_-25%"] = +0.12  # Diseconomies
        
        return sensitivities
    
    def _generate_recommendations(self, 
                                 analysis: Dict[str, Any],
                                 terms: TreatyTerms,
                                 gross_rate: float) -> Tuple[List[str], List[str], str]:
        """Generate pricing recommendations and identify key risks"""
        
        recommendations = []
        key_risks = []
        
        # Pricing confidence
        confidence_factors = []
        
        # Data quality assessment
        if analysis.get("loss_ratio_trend") == "stable":
            confidence_factors.append("stable")
        else:
            key_risks.append("Loss ratio trend volatility")
        
        # Experience credibility
        uw_quality = analysis.get("underwriting_quality", "average")
        if uw_quality in ["excellent", "good"]:
            confidence_factors.append("good_uw")
            recommendations.append("Consider preferential terms due to strong underwriting")
        else:
            key_risks.append("Underwriting quality concerns")
        
        # Concentration risk
        if analysis.get("concentration_risk") == "high":
            key_risks.append("Geographic concentration risk")
            recommendations.append("Consider aggregate limits or territorial restrictions")
        
        # Treaty structure
        if terms.treaty_type in [TreatyType.XS_OF_LOSS, TreatyType.CATASTROPHE]:
            key_risks.append("Low frequency, high severity exposure")
            recommendations.append("Ensure adequate catastrophe reserves")
        
        # Rate level assessment
        if gross_rate > 0.8:
            recommendations.append("Rate appears high - consider market competitiveness")
        elif gross_rate < 0.3:
            recommendations.append("Rate may be insufficient - verify loss projections")
        
        # Confidence determination
        if len(confidence_factors) >= 2 and len(key_risks) <= 2:
            confidence = "High"
        elif len(key_risks) >= 4:
            confidence = "Low"
        else:
            confidence = "Medium"
        
        return recommendations, key_risks, confidence
    
    def _create_portfolio_summary(self, cedent_exp: CedentExperience) -> Dict[str, Any]:
        """Create summary of portfolio characteristics"""
        
        return {
            "business_line": cedent_exp.business_line.value,
            "recent_premium": cedent_exp.premium_volume[-1] if cedent_exp.premium_volume else 0,
            "avg_face_amount": cedent_exp.avg_face_amount,
            "avg_issue_age": cedent_exp.avg_issue_age,
            "smoker_mix": f"{cedent_exp.smoker_percentage*100:.1f}%",
            "male_percentage": f"{cedent_exp.male_percentage*100:.1f}%",
            "geographic_concentration": f"{cedent_exp.top_state_concentration*100:.1f}%",
            "underwriting_grade": cedent_exp.underwriting_grade
        }
    
    def _document_assumptions(self, terms: TreatyTerms) -> Dict[str, Any]:
        """Document key pricing assumptions"""
        
        return {
            "treaty_type": terms.treaty_type.value,
            "territory": terms.territory,
            "effective_date": terms.effective_date.isoformat(),
            "expiry_date": terms.expiry_date.isoformat(),
            "commission_rate": terms.commission_rate,
            "profit_commission": terms.profit_commission_rate > 0,
            "experience_rating": terms.experience_rating,
            "aggregate_provisions": terms.aggregate_limit is not None
        }