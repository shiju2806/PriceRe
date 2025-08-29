"""
Unified ML-Enhanced Actuarial Pricing Engine
Integrates mortality enhancement, economic forecasting, and lapse modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..mortality.mortality_engine import MortalityEngine
from ..reserves.reserve_engine import ReserveEngine
from ..capital.capital_engine import CapitalEngine
from .mortality_ml import MortalityMLEnhancer
from .economic_forecasting import EconomicForecastingEngine
from .lapse_modeling import LapseModelingEngine

# Import real data sources
from ..data_sources.real_mortality_data import real_mortality_engine
from ..data_sources.real_economic_data import real_economic_engine

@dataclass
class PricingResult:
    """Comprehensive pricing result with ML enhancements"""
    policy_id: str
    product_type: str
    
    # Traditional actuarial components
    base_net_premium: float
    base_reserves: Dict[str, float]
    base_capital_requirement: float
    
    # ML-enhanced components
    mortality_adjustment_factor: float
    lapse_probability: float
    economic_scenario_impact: float
    
    # Final pricing
    ml_enhanced_premium: float
    commercial_premium: float
    profit_margin: float
    
    # Risk metrics
    risk_metrics: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    # Explanations
    pricing_drivers: List[str]
    risk_assessment: str
    
    # Validation
    regulatory_compliant: bool
    audit_trail: Dict[str, Any]

class UnifiedMLActuarialPricingEngine:
    """Complete ML-enhanced actuarial pricing system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize core actuarial engines
        self.mortality_engine = MortalityEngine()
        self.reserve_engine = ReserveEngine(self.mortality_engine)
        self.capital_engine = CapitalEngine()
        
        # Initialize ML enhancement engines
        self.mortality_ml = MortalityMLEnhancer()
        self.economic_ml = EconomicForecastingEngine()
        self.lapse_ml = LapseModelingEngine()
        
        # Pricing parameters
        self.pricing_config = {
            'profit_margin_target': 0.15,    # 15% profit margin
            'expense_loading': 0.05,         # 5% expense loading
            'contingency_margin': 0.03,      # 3% contingency
            'min_premium_rate': 0.001,       # 0.1% minimum rate
            'max_premium_rate': 0.10         # 10% maximum rate
        }
        
        # Model weights for ensemble
        self.ml_weights = {
            'mortality_weight': 0.4,
            'lapse_weight': 0.3,
            'economic_weight': 0.3
        }
        
        # Validation rules
        self.validation_rules = {
            'max_mortality_adjustment': 5.0,    # Max 5x mortality
            'min_mortality_adjustment': 0.25,   # Min 25% of standard
            'max_lapse_rate': 0.5,              # Max 50% annual lapse
            'min_profit_margin': 0.05,          # Min 5% profit margin
            'max_premium_multiple': 3.0         # Max 3x standard premium
        }
    
    def train_all_ml_models(
        self,
        historical_data: Dict[str, pd.DataFrame],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train all ML enhancement models"""
        
        training_results = {}
        
        try:
            # 1. Train Mortality ML Models
            if 'policy_data' in historical_data and 'mortality_experience' in historical_data:
                self.logger.info("Training mortality enhancement models...")
                
                mortality_features, mortality_target = self.mortality_ml.prepare_mortality_training_data(
                    historical_data['policy_data'],
                    historical_data['mortality_experience'],
                    historical_data.get('external_data')
                )
                
                mortality_results = self.mortality_ml.train_mortality_models(
                    mortality_features, mortality_target, validation_split
                )
                training_results['mortality'] = mortality_results
            
            # 2. Train Economic Forecasting Models
            if 'economic_data' in historical_data:
                self.logger.info("Training economic forecasting models...")
                
                economic_data = historical_data['economic_data']
                
                rate_results = self.economic_ml.train_interest_rate_models(economic_data)
                equity_results = self.economic_ml.train_equity_models(economic_data)
                
                training_results['economic'] = {
                    'interest_rates': rate_results,
                    'equity_models': equity_results
                }
            
            # 3. Train Lapse Models
            if all(key in historical_data for key in ['policy_data', 'lapse_history']):
                self.logger.info("Training lapse prediction models...")
                
                lapse_features, lapse_target = self.lapse_ml.prepare_lapse_training_data(
                    historical_data['policy_data'],
                    historical_data['lapse_history'],
                    historical_data.get('economic_data', pd.DataFrame())
                )
                
                lapse_results = self.lapse_ml.train_lapse_models(
                    lapse_features, lapse_target, validation_split
                )
                training_results['lapse'] = lapse_results
            
            self.logger.info("All ML models trained successfully")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            return {'error': str(e)}
    
    def price_policy(
        self,
        policy_data: Dict[str, Any],
        economic_scenario: Optional[Dict[str, float]] = None,
        use_ml_enhancements: bool = True
    ) -> PricingResult:
        """Complete policy pricing with ML enhancements"""
        
        try:
            # Extract key policy information
            policy_id = policy_data.get('policy_id', 'UNKNOWN')
            product_type = policy_data.get('product_type', 'TERM')
            issue_age = policy_data.get('issue_age', 35)
            gender = policy_data.get('gender', 'M')
            face_amount = policy_data.get('face_amount', 100000)
            
            self.logger.info(f"Pricing policy {policy_id}: {product_type}, Age {issue_age}, ${face_amount:,.0f}")
            
            # Step 1: Calculate base actuarial premium
            base_pricing = self._calculate_base_actuarial_premium(policy_data)
            
            # Step 2: Apply ML enhancements if enabled
            if use_ml_enhancements:
                ml_adjustments = self._calculate_ml_adjustments(policy_data, economic_scenario)
            else:
                ml_adjustments = self._get_default_adjustments()
            
            # Step 3: Calculate reserves
            reserve_calculations = self._calculate_comprehensive_reserves(
                policy_data, base_pricing, ml_adjustments
            )
            
            # Step 4: Calculate capital requirements
            capital_calculations = self._calculate_capital_requirements(
                policy_data, base_pricing, ml_adjustments
            )
            
            # Step 5: Apply ML adjustments to final pricing
            ml_enhanced_premium = self._apply_ml_enhancements(
                base_pricing, ml_adjustments
            )
            
            # Step 6: Calculate commercial premium with loadings
            commercial_premium = self._calculate_commercial_premium(
                ml_enhanced_premium, policy_data
            )
            
            # Step 7: Risk assessment and validation
            risk_assessment = self._assess_overall_risk(policy_data, ml_adjustments)
            validation_result = self._validate_pricing_result(
                base_pricing, ml_enhanced_premium, commercial_premium, policy_data
            )
            
            # Step 8: Generate explanations
            pricing_drivers = self._generate_pricing_explanations(
                policy_data, ml_adjustments, base_pricing, commercial_premium
            )
            
            # Step 9: Create comprehensive result
            result = PricingResult(
                policy_id=policy_id,
                product_type=product_type,
                base_net_premium=base_pricing['net_annual_premium'],
                base_reserves=reserve_calculations,
                base_capital_requirement=capital_calculations['total_requirement'],
                mortality_adjustment_factor=ml_adjustments['mortality_factor'],
                lapse_probability=ml_adjustments['lapse_probability'],
                economic_scenario_impact=ml_adjustments['economic_factor'],
                ml_enhanced_premium=ml_enhanced_premium,
                commercial_premium=commercial_premium,
                profit_margin=(commercial_premium - ml_enhanced_premium) / commercial_premium,
                risk_metrics=self._calculate_risk_metrics(policy_data, ml_adjustments),
                confidence_scores=self._calculate_confidence_scores(policy_data, ml_adjustments),
                pricing_drivers=pricing_drivers,
                risk_assessment=risk_assessment,
                regulatory_compliant=validation_result['compliant'],
                audit_trail=self._create_audit_trail(policy_data, base_pricing, ml_adjustments)
            )
            
            self.logger.info(f"Policy {policy_id} priced successfully: ${commercial_premium:,.0f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error pricing policy {policy_data.get('policy_id', 'UNKNOWN')}: {e}")
            return self._create_fallback_result(policy_data, str(e))
    
    def price_portfolio(
        self,
        portfolio_data: pd.DataFrame,
        economic_scenario: Optional[Dict[str, float]] = None,
        use_ml_enhancements: bool = True
    ) -> Dict[str, Any]:
        """Price entire portfolio with ML enhancements"""
        
        pricing_results = []
        portfolio_metrics = {}
        
        try:
            self.logger.info(f"Pricing portfolio of {len(portfolio_data)} policies")
            
            # Price each policy
            for idx, policy_row in portfolio_data.iterrows():
                policy_dict = policy_row.to_dict()
                
                result = self.price_policy(
                    policy_dict, economic_scenario, use_ml_enhancements
                )
                pricing_results.append(result)
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(pricing_results)
            
            # Portfolio-level risk assessment
            portfolio_risk = self._assess_portfolio_risk(pricing_results)
            
            # Correlation adjustments
            correlation_adjustments = self._calculate_correlation_adjustments(pricing_results)
            
            return {
                'individual_results': pricing_results,
                'portfolio_metrics': portfolio_metrics,
                'portfolio_risk_assessment': portfolio_risk,
                'correlation_adjustments': correlation_adjustments,
                'total_premium': sum(r.commercial_premium for r in pricing_results),
                'average_profit_margin': np.mean([r.profit_margin for r in pricing_results]),
                'pricing_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error pricing portfolio: {e}")
            return {
                'error': str(e),
                'individual_results': pricing_results,
                'portfolio_metrics': {}
            }
    
    def generate_pricing_report(
        self,
        pricing_result: PricingResult,
        include_technical_details: bool = True
    ) -> str:
        """Generate comprehensive pricing report"""
        
        report = f"""
# ML-Enhanced Actuarial Pricing Report

## Policy Information
- **Policy ID**: {pricing_result.policy_id}
- **Product Type**: {pricing_result.product_type}
- **Pricing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pricing Summary
- **Commercial Premium**: ${pricing_result.commercial_premium:,.0f}
- **ML-Enhanced Premium**: ${pricing_result.ml_enhanced_premium:,.0f}
- **Base Actuarial Premium**: ${pricing_result.base_net_premium:,.0f}
- **Profit Margin**: {pricing_result.profit_margin:.1%}

## ML Enhancement Factors
- **Mortality Adjustment**: {pricing_result.mortality_adjustment_factor:.2f}x
- **Lapse Probability**: {pricing_result.lapse_probability:.1%}
- **Economic Impact**: {pricing_result.economic_scenario_impact:.2f}x

## Risk Assessment
- **Overall Risk Level**: {pricing_result.risk_assessment}
- **Regulatory Compliance**: {'✅ Compliant' if pricing_result.regulatory_compliant else '❌ Non-Compliant'}

## Key Pricing Drivers
"""
        for i, driver in enumerate(pricing_result.pricing_drivers, 1):
            report += f"{i}. {driver}\n"
        
        if include_technical_details:
            report += f"""
## Technical Details

### Risk Metrics
"""
            for metric, value in pricing_result.risk_metrics.items():
                report += f"- **{metric}**: {value:.3f}\n"
            
            report += f"""
### Confidence Scores
"""
            for component, score in pricing_result.confidence_scores.items():
                report += f"- **{component}**: {score:.1%}\n"
        
        report += f"""
---
*Report generated by ML-Enhanced Actuarial Pricing Engine*
*Combines traditional actuarial methods with machine learning enhancements*
"""
        
        return report
    
    def _calculate_base_actuarial_premium(self, policy_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate base actuarial premium using traditional methods"""
        
        issue_age = policy_data.get('issue_age', 35)
        gender = policy_data.get('gender', 'M')
        face_amount = policy_data.get('face_amount', 100000)
        premium_period = policy_data.get('premium_period', 999)  # Whole life
        benefit_period = policy_data.get('benefit_period', 999)
        
        # Calculate net premium using mortality engine
        net_premium_calc = self.mortality_engine.calculate_net_premium(
            issue_age, gender, face_amount, premium_period, benefit_period
        )
        
        return net_premium_calc
    
    def _calculate_ml_adjustments(
        self,
        policy_data: Dict[str, Any],
        economic_scenario: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate ML-based adjustments"""
        
        adjustments = {}
        
        # Mortality adjustment
        try:
            mortality_result = self.mortality_ml.predict_mortality_adjustment(policy_data)
            adjustments['mortality_factor'] = mortality_result['mortality_adjustment_factor']
            adjustments['mortality_confidence'] = mortality_result['confidence_level']
        except Exception as e:
            self.logger.warning(f"Mortality ML prediction failed: {e}")
            adjustments['mortality_factor'] = 1.0
            adjustments['mortality_confidence'] = 0.5
        
        # Lapse probability
        try:
            lapse_result = self.lapse_ml.predict_lapse_probability(policy_data, economic_scenario)
            adjustments['lapse_probability'] = lapse_result['annual_lapse_probability']
            adjustments['lapse_confidence'] = lapse_result['confidence_level']
        except Exception as e:
            self.logger.warning(f"Lapse ML prediction failed: {e}")
            adjustments['lapse_probability'] = 0.05
            adjustments['lapse_confidence'] = 0.5
        
        # Economic adjustment
        if economic_scenario:
            try:
                economic_impact = self._calculate_economic_impact(economic_scenario)
                adjustments['economic_factor'] = economic_impact
                adjustments['economic_confidence'] = 0.7
            except Exception as e:
                self.logger.warning(f"Economic adjustment failed: {e}")
                adjustments['economic_factor'] = 1.0
                adjustments['economic_confidence'] = 0.5
        else:
            adjustments['economic_factor'] = 1.0
            adjustments['economic_confidence'] = 0.5
        
        # Apply validation bounds
        adjustments['mortality_factor'] = np.clip(
            adjustments['mortality_factor'],
            self.validation_rules['min_mortality_adjustment'],
            self.validation_rules['max_mortality_adjustment']
        )
        
        adjustments['lapse_probability'] = np.clip(
            adjustments['lapse_probability'], 0.001, self.validation_rules['max_lapse_rate']
        )
        
        return adjustments
    
    def _get_default_adjustments(self) -> Dict[str, float]:
        """Default adjustments when ML is disabled"""
        
        return {
            'mortality_factor': 1.0,
            'lapse_probability': 0.05,
            'economic_factor': 1.0,
            'mortality_confidence': 0.6,
            'lapse_confidence': 0.6,
            'economic_confidence': 0.6
        }
    
    def _calculate_comprehensive_reserves(
        self,
        policy_data: Dict[str, Any],
        base_pricing: Dict[str, float],
        ml_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate comprehensive reserves with ML adjustments"""
        
        # Create policy DataFrame for reserve engine
        policy_df = pd.DataFrame([policy_data])
        
        try:
            # GAAP reserves
            gaap_reserves = self.reserve_engine.calculate_gaap_reserves(
                policy_df, datetime.now()
            )
            
            # Statutory reserves
            statutory_reserves = self.reserve_engine.calculate_statutory_reserves(
                policy_df, datetime.now()
            )
            
            # Adjust reserves for ML-predicted lapse behavior
            lapse_adjustment = 1 - ml_adjustments['lapse_probability'] * 0.1
            
            return {
                'gaap_reserve': gaap_reserves['total_gaap_reserves'] * lapse_adjustment,
                'statutory_reserve': statutory_reserves['total_statutory_reserves'] * lapse_adjustment,
                'economic_reserve': gaap_reserves['total_gaap_reserves'] * 0.9,  # Simplified
                'lapse_adjustment_factor': lapse_adjustment
            }
            
        except Exception as e:
            self.logger.warning(f"Reserve calculation failed: {e}")
            return {
                'gaap_reserve': base_pricing['net_annual_premium'] * 5,
                'statutory_reserve': base_pricing['net_annual_premium'] * 6,
                'economic_reserve': base_pricing['net_annual_premium'] * 4.5,
                'lapse_adjustment_factor': 1.0
            }
    
    def _calculate_capital_requirements(
        self,
        policy_data: Dict[str, Any],
        base_pricing: Dict[str, float],
        ml_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate capital requirements with ML adjustments"""
        
        try:
            # Mock company data for capital calculation
            company_data = {
                'annual_premiums': base_pricing['net_annual_premium'],
                'policy_reserves': base_pricing['net_annual_premium'] * 5,
                'surplus': 100000000
            }
            
            portfolio_df = pd.DataFrame([policy_data])
            
            rbc_result = self.capital_engine.calculate_rbc(company_data, portfolio_df)
            ec_result = self.capital_engine.calculate_economic_capital(portfolio_df)
            
            # Adjust for ML-predicted mortality and lapse risks
            mortality_adj = ml_adjustments['mortality_factor'] - 1.0  # Excess over standard
            lapse_adj = ml_adjustments['lapse_probability'] - 0.05    # Excess over 5% standard
            
            risk_adjustment = 1 + max(0, mortality_adj * 0.5) + max(0, lapse_adj * 2)
            
            return {
                'total_requirement': rbc_result['total_rbc_requirement'] * risk_adjustment,
                'rbc_requirement': rbc_result['total_rbc_requirement'],
                'economic_capital': ec_result['economic_capital'],
                'risk_adjustment_factor': risk_adjustment
            }
            
        except Exception as e:
            self.logger.warning(f"Capital calculation failed: {e}")
            return {
                'total_requirement': base_pricing['net_annual_premium'] * 2,
                'rbc_requirement': base_pricing['net_annual_premium'] * 1.5,
                'economic_capital': base_pricing['net_annual_premium'] * 2.5,
                'risk_adjustment_factor': 1.0
            }
    
    def _apply_ml_enhancements(
        self,
        base_pricing: Dict[str, float],
        ml_adjustments: Dict[str, float]
    ) -> float:
        """Apply ML adjustments to base pricing"""
        
        base_premium = base_pricing['net_annual_premium']
        
        # Apply mortality adjustment
        mortality_adjusted = base_premium * ml_adjustments['mortality_factor']
        
        # Apply lapse adjustment (higher lapse = lower premium needed)
        lapse_adjustment = 1 - (ml_adjustments['lapse_probability'] - 0.05) * 0.3
        lapse_adjusted = mortality_adjusted * max(0.7, lapse_adjustment)
        
        # Apply economic adjustment
        economic_adjusted = lapse_adjusted * ml_adjustments['economic_factor']
        
        # Ensure reasonable bounds
        min_premium = base_premium * 0.5  # Not less than 50% of base
        max_premium = base_premium * self.validation_rules['max_premium_multiple']
        
        ml_enhanced_premium = np.clip(economic_adjusted, min_premium, max_premium)
        
        return ml_enhanced_premium
    
    def _calculate_commercial_premium(
        self,
        ml_enhanced_premium: float,
        policy_data: Dict[str, Any]
    ) -> float:
        """Calculate final commercial premium with loadings"""
        
        # Apply loadings
        expense_loading = ml_enhanced_premium * self.pricing_config['expense_loading']
        profit_margin = ml_enhanced_premium * self.pricing_config['profit_margin_target']
        contingency_margin = ml_enhanced_premium * self.pricing_config['contingency_margin']
        
        commercial_premium = ml_enhanced_premium + expense_loading + profit_margin + contingency_margin
        
        # Apply minimum/maximum rates
        face_amount = policy_data.get('face_amount', 100000)
        min_premium = face_amount * self.pricing_config['min_premium_rate']
        max_premium = face_amount * self.pricing_config['max_premium_rate']
        
        commercial_premium = np.clip(commercial_premium, min_premium, max_premium)
        
        return commercial_premium
    
    def _calculate_economic_impact(self, economic_scenario: Dict[str, float]) -> float:
        """Calculate impact of economic scenario on pricing"""
        
        base_impact = 1.0
        
        # Interest rate impact
        interest_rate = economic_scenario.get('interest_rate_10y', 0.035)
        if interest_rate > 0.05:  # High interest rate environment
            base_impact *= 1.1  # Slightly higher premiums due to lapse risk
        elif interest_rate < 0.02:  # Low interest rate environment
            base_impact *= 1.05  # Higher premiums due to investment risk
        
        # Economic growth impact
        gdp_growth = economic_scenario.get('gdp_growth', 0.025)
        if gdp_growth < 0:  # Recession
            base_impact *= 1.15  # Higher premiums due to increased lapse risk
        
        # Unemployment impact
        unemployment = economic_scenario.get('unemployment', 0.05)
        if unemployment > 0.08:  # High unemployment
            base_impact *= 1.10  # Higher premiums due to lapse risk
        
        return np.clip(base_impact, 0.9, 1.3)  # Reasonable bounds
    
    def _assess_overall_risk(
        self,
        policy_data: Dict[str, Any],
        ml_adjustments: Dict[str, float]
    ) -> str:
        """Assess overall risk level"""
        
        risk_score = 0
        
        # Mortality risk
        mortality_factor = ml_adjustments['mortality_factor']
        if mortality_factor > 2.0:
            risk_score += 3
        elif mortality_factor > 1.5:
            risk_score += 2
        elif mortality_factor > 1.2:
            risk_score += 1
        
        # Lapse risk
        lapse_prob = ml_adjustments['lapse_probability']
        if lapse_prob > 0.15:
            risk_score += 2
        elif lapse_prob > 0.10:
            risk_score += 1
        
        # Age risk
        age = policy_data.get('issue_age', 35)
        if age >= 65:
            risk_score += 2
        elif age >= 55:
            risk_score += 1
        
        # Face amount risk
        face_amount = policy_data.get('face_amount', 100000)
        if face_amount >= 5000000:
            risk_score += 2
        elif face_amount >= 1000000:
            risk_score += 1
        
        # Determine risk category
        if risk_score >= 6:
            return 'high_risk'
        elif risk_score >= 3:
            return 'moderate_risk'
        else:
            return 'standard_risk'
    
    def _validate_pricing_result(
        self,
        base_pricing: Dict[str, float],
        ml_enhanced_premium: float,
        commercial_premium: float,
        policy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate pricing results against regulatory and business rules"""
        
        validation_errors = []
        warnings = []
        
        # Check premium reasonableness
        premium_multiple = ml_enhanced_premium / base_pricing['net_annual_premium']
        if premium_multiple > self.validation_rules['max_premium_multiple']:
            validation_errors.append(f"Premium multiple {premium_multiple:.2f}x exceeds maximum allowed")
        
        # Check profit margin
        profit_margin = (commercial_premium - ml_enhanced_premium) / commercial_premium
        if profit_margin < self.validation_rules['min_profit_margin']:
            warnings.append(f"Profit margin {profit_margin:.1%} below minimum target")
        
        # Check rate per $1000
        face_amount = policy_data.get('face_amount', 100000)
        rate_per_1000 = (commercial_premium / face_amount) * 1000
        if rate_per_1000 > 100:  # $100 per $1000 seems excessive
            validation_errors.append(f"Rate per $1000 of ${rate_per_1000:.0f} appears excessive")
        
        return {
            'compliant': len(validation_errors) == 0,
            'errors': validation_errors,
            'warnings': warnings,
            'premium_multiple': premium_multiple,
            'profit_margin': profit_margin,
            'rate_per_1000': rate_per_1000
        }
    
    def _generate_pricing_explanations(
        self,
        policy_data: Dict[str, Any],
        ml_adjustments: Dict[str, float],
        base_pricing: Dict[str, float],
        commercial_premium: float
    ) -> List[str]:
        """Generate human-readable explanations for pricing"""
        
        explanations = []
        
        # Base premium explanation
        explanations.append(
            f"Base actuarial premium of ${base_pricing['net_annual_premium']:,.0f} "
            f"calculated using {policy_data.get('gender', 'standard')} mortality tables"
        )
        
        # Mortality adjustment explanation
        mortality_factor = ml_adjustments['mortality_factor']
        if mortality_factor > 1.2:
            explanations.append(
                f"Mortality risk increased by {(mortality_factor-1)*100:.0f}% based on "
                f"health and lifestyle factors"
            )
        elif mortality_factor < 0.9:
            explanations.append(
                f"Mortality risk reduced by {(1-mortality_factor)*100:.0f}% based on "
                f"favorable health profile"
            )
        
        # Lapse risk explanation
        lapse_prob = ml_adjustments['lapse_probability']
        if lapse_prob > 0.10:
            explanations.append(
                f"Higher lapse probability of {lapse_prob:.1%} identified, "
                f"affecting reserve requirements"
            )
        elif lapse_prob < 0.03:
            explanations.append(
                f"Lower lapse probability of {lapse_prob:.1%} supports "
                f"more competitive pricing"
            )
        
        # Economic impact explanation
        economic_factor = ml_adjustments['economic_factor']
        if economic_factor > 1.05:
            explanations.append(
                "Current economic conditions increase pricing by "
                f"{(economic_factor-1)*100:.0f}%"
            )
        
        # Commercial loading explanation
        ml_premium = base_pricing['net_annual_premium'] * mortality_factor
        loading = commercial_premium - ml_premium
        if loading > 0:
            explanations.append(
                f"Commercial loading of ${loading:,.0f} includes expenses, "
                f"profit margin, and contingency reserves"
            )
        
        return explanations
    
    def _calculate_risk_metrics(
        self,
        policy_data: Dict[str, Any],
        ml_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        return {
            'mortality_risk_score': (ml_adjustments['mortality_factor'] - 1.0) * 2,
            'lapse_risk_score': ml_adjustments['lapse_probability'] * 10,
            'economic_risk_score': abs(ml_adjustments['economic_factor'] - 1.0) * 5,
            'age_risk_score': max(0, policy_data.get('issue_age', 35) - 50) * 0.1,
            'size_risk_score': min(5, policy_data.get('face_amount', 100000) / 1000000),
            'overall_risk_score': (
                (ml_adjustments['mortality_factor'] - 1.0) * 2 +
                ml_adjustments['lapse_probability'] * 10 +
                abs(ml_adjustments['economic_factor'] - 1.0) * 5
            ) / 3
        }
    
    def _calculate_confidence_scores(
        self,
        policy_data: Dict[str, Any],
        ml_adjustments: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate confidence scores for different components"""
        
        return {
            'mortality_prediction': ml_adjustments.get('mortality_confidence', 0.7),
            'lapse_prediction': ml_adjustments.get('lapse_confidence', 0.7),
            'economic_forecast': ml_adjustments.get('economic_confidence', 0.6),
            'overall_pricing': np.mean([
                ml_adjustments.get('mortality_confidence', 0.7),
                ml_adjustments.get('lapse_confidence', 0.7),
                ml_adjustments.get('economic_confidence', 0.6)
            ])
        }
    
    def _create_audit_trail(
        self,
        policy_data: Dict[str, Any],
        base_pricing: Dict[str, float],
        ml_adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create detailed audit trail for regulatory compliance"""
        
        return {
            'pricing_timestamp': datetime.now().isoformat(),
            'input_data': policy_data,
            'base_actuarial_calculation': base_pricing,
            'ml_adjustments_applied': ml_adjustments,
            'model_versions': {
                'mortality_ml_version': '1.0',
                'lapse_ml_version': '1.0',
                'economic_ml_version': '1.0'
            },
            'validation_rules_applied': self.validation_rules,
            'pricing_config': self.pricing_config
        }
    
    def _create_fallback_result(
        self, 
        policy_data: Dict[str, Any], 
        error_message: str
    ) -> PricingResult:
        """Create fallback pricing result in case of errors"""
        
        # Simple fallback calculation
        face_amount = policy_data.get('face_amount', 100000)
        fallback_premium = face_amount * 0.015  # 1.5% of face amount
        
        return PricingResult(
            policy_id=policy_data.get('policy_id', 'ERROR'),
            product_type=policy_data.get('product_type', 'UNKNOWN'),
            base_net_premium=fallback_premium * 0.8,
            base_reserves={'gaap_reserve': fallback_premium * 5},
            base_capital_requirement=fallback_premium * 2,
            mortality_adjustment_factor=1.0,
            lapse_probability=0.05,
            economic_scenario_impact=1.0,
            ml_enhanced_premium=fallback_premium,
            commercial_premium=fallback_premium * 1.2,
            profit_margin=0.15,
            risk_metrics={'overall_risk_score': 0.5},
            confidence_scores={'overall_pricing': 0.3},
            pricing_drivers=[f"Fallback calculation due to error: {error_message}"],
            risk_assessment='unknown_risk',
            regulatory_compliant=False,
            audit_trail={'error': error_message}
        )
    
    def _calculate_portfolio_metrics(
        self, 
        pricing_results: List[PricingResult]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        
        if not pricing_results:
            return {}
        
        premiums = [r.commercial_premium for r in pricing_results]
        profit_margins = [r.profit_margin for r in pricing_results]
        risk_scores = [r.risk_metrics.get('overall_risk_score', 0) for r in pricing_results]
        
        return {
            'total_premium': sum(premiums),
            'average_premium': np.mean(premiums),
            'premium_std': np.std(premiums),
            'average_profit_margin': np.mean(profit_margins),
            'profit_margin_std': np.std(profit_margins),
            'average_risk_score': np.mean(risk_scores),
            'high_risk_policies': sum(1 for r in pricing_results if r.risk_assessment == 'high_risk'),
            'non_compliant_policies': sum(1 for r in pricing_results if not r.regulatory_compliant)
        }
    
    def _assess_portfolio_risk(
        self, 
        pricing_results: List[PricingResult]
    ) -> Dict[str, Any]:
        """Assess portfolio-level risk"""
        
        risk_levels = [r.risk_assessment for r in pricing_results]
        risk_distribution = {
            'standard_risk': risk_levels.count('standard_risk') / len(risk_levels),
            'moderate_risk': risk_levels.count('moderate_risk') / len(risk_levels),
            'high_risk': risk_levels.count('high_risk') / len(risk_levels)
        }
        
        return {
            'risk_distribution': risk_distribution,
            'concentration_risk': max(risk_distribution.values()),
            'diversification_score': 1 - max(risk_distribution.values()),
            'overall_portfolio_risk': (
                'high' if risk_distribution['high_risk'] > 0.2 else
                'moderate' if risk_distribution['moderate_risk'] > 0.4 else
                'standard'
            )
        }
    
    def _calculate_correlation_adjustments(
        self, 
        pricing_results: List[PricingResult]
    ) -> Dict[str, float]:
        """Calculate correlation-based portfolio adjustments"""
        
        # Simplified correlation adjustments
        n_policies = len(pricing_results)
        
        if n_policies > 100:
            diversification_benefit = 0.05  # 5% reduction due to diversification
        elif n_policies > 50:
            diversification_benefit = 0.03  # 3% reduction
        else:
            diversification_benefit = 0.0   # No benefit for small portfolios
        
        return {
            'diversification_benefit': diversification_benefit,
            'correlation_adjustment': 1 - diversification_benefit,
            'effective_risk_reduction': diversification_benefit
        }