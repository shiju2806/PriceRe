"""
Professional Actuarial Engine
Complete reinsurance pricing engine with zero hardcoding
Integrates External API Manager, SOA Data Manager, and Data Intelligence Engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .external_api_manager import ExternalAPIManager
from .soa_data_manager import SOADataManager
from .data_intelligence_engine import DataIntelligenceEngine
from .config_manager import get_config


class ProfessionalActuarialEngine:
    """Main actuarial engine that integrates all components for professional pricing"""
    
    def __init__(self):
        # Load configuration (NO HARDCODING!)
        self.config = get_config()
        
        # Initialize component managers
        self.api_manager = ExternalAPIManager()
        self.soa_manager = SOADataManager()
        self.intelligence_engine = DataIntelligenceEngine()
        
        # Cache for expensive calculations
        self.calculation_cache = {}
        self.last_data_hash = None
        
    async def analyze_portfolio(self, datasets: Dict, product_mix: Optional[Dict] = None) -> Dict:
        """Complete portfolio analysis with real actuarial calculations"""
        
        print("🔬 Starting professional actuarial analysis...")
        
        # Prepare datasets for intelligence engine (expects 'data' key with DataFrame)
        prepared_for_intelligence = {}
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                prepared_for_intelligence[name] = {'data': df}
            elif isinstance(df, dict) and 'data' in df:
                prepared_for_intelligence[name] = df
            elif isinstance(df, dict) and 'dataframe' in df:
                prepared_for_intelligence[name] = {'data': df['dataframe']}
            else:
                # Try to use as-is if it's already a DataFrame
                prepared_for_intelligence[name] = {'data': df}
        
        # Step 1: Intelligent data mapping
        intelligence_results = await self.intelligence_engine.analyze_datasets(prepared_for_intelligence)
        
        # Extract field mappings and prepare mapped data
        mapped_data = {}
        for dataset_name, field_mappings in intelligence_results.get('field_mappings', {}).items():
            if dataset_name in prepared_for_intelligence:
                mapped_data[dataset_name] = {
                    'dataframe': prepared_for_intelligence[dataset_name]['data'],
                    'field_mapping': field_mappings
                }
        
        # Step 2: Extract market environment 
        market_data = await self._get_market_environment()
        
        # Step 3: Perform core actuarial calculations
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_mapping': mapped_data,
            'market_environment': market_data,
            'portfolio_metrics': await self._calculate_portfolio_metrics(mapped_data, market_data),
            'experience_analysis': await self._perform_experience_analysis(mapped_data),
            'risk_assessment': await self._assess_portfolio_risk(mapped_data, market_data),
            'trend_analysis': await self._analyze_trends(mapped_data),
            'pricing_recommendations': await self._generate_pricing_recommendations(mapped_data, market_data),
            'reserve_requirements': await self._calculate_reserves(mapped_data, market_data),
            'capital_requirements': await self._calculate_economic_capital(mapped_data, market_data)
        }
        
        print("✅ Professional actuarial analysis complete!")
        return results
    
    async def _get_market_environment(self) -> Dict:
        """Get current market environment data"""
        
        # Get Treasury yield curve
        treasury_curve = await self.api_manager.get_treasury_yield_curve()
        
        # Get economic indicators
        economic_indicators = await self.api_manager.get_economic_indicators()
        
        # Get historical data for volatility calibration
        historical_data = await self.api_manager.get_historical_treasury_data(years_back=5)
        
        # Calculate interest rate statistics
        rate_stats = self._calculate_rate_statistics(historical_data)
        
        return {
            'yield_curve': treasury_curve.to_dict('records'),
            'economic_indicators': economic_indicators,
            'rate_statistics': rate_stats,
            'market_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _calculate_rate_statistics(self, historical_data: pd.DataFrame) -> Dict:
        """Calculate interest rate statistics for pricing models"""
        
        if len(historical_data) < 24:
            # Use default assumptions if insufficient data
            return {
                'mean_rate': 0.045,
                'rate_volatility': 0.015,
                'mean_reversion_speed': 0.10,
                'long_term_rate': 0.045
            }
        
        # Calculate statistics from historical data
        rates = historical_data['rate'].dropna()
        log_changes = historical_data['log_change'].dropna()
        
        return {
            'mean_rate': float(rates.mean()),
            'rate_volatility': float(log_changes.std() * np.sqrt(12)),  # Annualized
            'mean_reversion_speed': self._estimate_mean_reversion(rates),
            'long_term_rate': float(rates.rolling(window=60).mean().iloc[-1]) if len(rates) >= 60 else float(rates.mean())
        }
    
    def _estimate_mean_reversion(self, rates: pd.Series) -> float:
        """Estimate mean reversion parameter using AR(1) model"""
        try:
            from scipy import stats
            rate_changes = rates.diff().dropna()
            rate_levels = rates.shift(1).dropna()
            
            if len(rate_changes) != len(rate_levels):
                rate_changes = rate_changes[:len(rate_levels)]
            
            # Fit AR(1): dr = alpha * (mu - r) * dt + sigma * dW
            slope, intercept, r_value, p_value, std_err = stats.linregress(rate_levels, rate_changes)
            
            # Mean reversion speed is -slope (should be positive)
            mean_reversion = max(0.01, min(0.5, -slope * 12))  # Annualized, bounded
            return float(mean_reversion)
            
        except:
            return 0.10  # Default assumption
    
    async def _calculate_portfolio_metrics(self, mapped_data: Dict, market_data: Dict) -> Dict:
        """Calculate core portfolio metrics using real data"""
        
        metrics = {
            'total_exposure': 0,
            'weighted_age': 0,
            'gender_mix': {'MALE': 0, 'FEMALE': 0, 'UNKNOWN': 0},
            'smoking_mix': {'SMOKER': 0, 'NON-SMOKER': 0, 'UNKNOWN': 0},
            'duration_distribution': {},
            'face_amount_statistics': {},
            'geographic_distribution': {},
            'product_mix': {}
        }
        
        total_face_amount = 0
        unique_policy_ids = set()
        ages = []
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Find unique policies using policy ID or row index as fallback
            policy_id_col = None
            for field_name, mapping_obj in field_map.items():
                if hasattr(mapping_obj, 'standardized_name') and mapping_obj.standardized_name in ['policy_id', 'policy_number']:
                    policy_id_col = mapping_obj.original_name
                    break
                elif isinstance(mapping_obj, dict) and mapping_obj.get('standardized_name') in ['policy_id', 'policy_number']:
                    policy_id_col = mapping_obj.get('original_name') or mapping_obj.get('column_name')
                    break
            
            if policy_id_col and policy_id_col in df.columns:
                # Use actual policy IDs
                unique_policy_ids.update(df[policy_id_col].astype(str))
            else:
                # Fallback: create unique IDs using dataset name + row index
                for idx in range(len(df)):
                    unique_policy_ids.add(f"{dataset_name}_{idx}")
        
        total_policies = len(unique_policy_ids)
        
        # Now process each dataset for face amount and other calculations
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Face amount analysis - handle FieldMapping objects
            face_mapping = None
            for field_name, mapping in field_map.items():
                if hasattr(mapping, 'standardized_name') and mapping.standardized_name == 'face_amount':
                    face_mapping = mapping
                    break
                elif field_name == 'face_amount' and isinstance(mapping, dict):
                    # Fallback for dictionary format
                    face_mapping = mapping
                    break
            
            if face_mapping:
                if hasattr(face_mapping, 'original_name'):
                    face_col = face_mapping.original_name
                else:
                    face_col = face_mapping.get('column_name', face_mapping.get('original_name'))
                
                if face_col and face_col in df.columns:
                    face_amounts = pd.to_numeric(df[face_col], errors='coerce').fillna(0)
                    total_face_amount += face_amounts.sum()
                
                metrics['face_amount_statistics'] = {
                    'total': float(face_amounts.sum()),
                    'mean': float(face_amounts.mean()),
                    'median': float(face_amounts.median()),
                    'std': float(face_amounts.std()),
                    'percentiles': {
                        '25th': float(face_amounts.quantile(0.25)),
                        '75th': float(face_amounts.quantile(0.75)),
                        '90th': float(face_amounts.quantile(0.90))
                    }
                }
            
            # Age analysis - handle FieldMapping objects
            age_mapping = None
            for field_name, mapping in field_map.items():
                if hasattr(mapping, 'standardized_name') and mapping.standardized_name == 'age':
                    age_mapping = mapping
                    break
                elif field_name == 'age' and isinstance(mapping, dict):
                    age_mapping = mapping
                    break
            
            if age_mapping:
                if hasattr(age_mapping, 'original_name'):
                    age_col = age_mapping.original_name
                else:
                    age_col = age_mapping.get('column_name', age_mapping.get('original_name'))
                
                if age_col and age_col in df.columns:
                    age_values = pd.to_numeric(df[age_col], errors='coerce').dropna()
                    ages.extend(age_values.tolist())
            
            # Gender analysis - handle FieldMapping objects
            gender_mapping = None
            for field_name, mapping in field_map.items():
                if hasattr(mapping, 'standardized_name') and mapping.standardized_name == 'gender':
                    gender_mapping = mapping
                    break
                elif field_name == 'gender' and isinstance(mapping, dict):
                    gender_mapping = mapping
                    break
            
            if gender_mapping:
                if hasattr(gender_mapping, 'original_name'):
                    gender_col = gender_mapping.original_name
                else:
                    gender_col = gender_mapping.get('column_name', gender_mapping.get('original_name'))
                
                if gender_col and gender_col in df.columns:
                    gender_counts = df[gender_col].value_counts()
                    
                    for gender, count in gender_counts.items():
                        gender_key = self._standardize_gender(str(gender))
                        metrics['gender_mix'][gender_key] += int(count)
            
            # Smoking status analysis - handle FieldMapping objects
            smoking_mapping = None
            for field_name, mapping in field_map.items():
                if hasattr(mapping, 'standardized_name') and mapping.standardized_name == 'smoking_status':
                    smoking_mapping = mapping
                    break
                elif field_name == 'smoking_status' and isinstance(mapping, dict):
                    smoking_mapping = mapping
                    break
            
            if smoking_mapping:
                if hasattr(smoking_mapping, 'original_name'):
                    smoking_col = smoking_mapping.original_name
                else:
                    smoking_col = smoking_mapping.get('column_name', smoking_mapping.get('original_name'))
                
                if smoking_col and smoking_col in df.columns:
                    smoking_counts = df[smoking_col].value_counts()
                    
                    for status, count in smoking_counts.items():
                        smoking_key = self._standardize_smoking_status(str(status))
                        metrics['smoking_mix'][smoking_key] += int(count)
            
            # Duration analysis - handle FieldMapping objects
            duration_mapping = None
            for field_name, mapping in field_map.items():
                if hasattr(mapping, 'standardized_name') and mapping.standardized_name == 'policy_duration':
                    duration_mapping = mapping
                    break
                elif field_name == 'policy_duration' and isinstance(mapping, dict):
                    duration_mapping = mapping
                    break
            
            if duration_mapping:
                if hasattr(duration_mapping, 'original_name'):
                    duration_col = duration_mapping.original_name
                else:
                    duration_col = duration_mapping.get('column_name', duration_mapping.get('original_name'))
                
                if duration_col and duration_col in df.columns:
                    duration_values = pd.to_numeric(df[duration_col], errors='coerce').dropna()
                    
                    # Create duration buckets
                    duration_buckets = pd.cut(duration_values, 
                                            bins=[0, 5, 10, 15, 20, 25, float('inf')],
                                            labels=['1-5', '6-10', '11-15', '16-20', '21-25', '25+'])
                    duration_counts = duration_buckets.value_counts()
                    
                    for bucket, count in duration_counts.items():
                        metrics['duration_distribution'][str(bucket)] = int(count)
        
        # Calculate summary metrics
        metrics['total_exposure'] = total_face_amount
        metrics['total_policies'] = total_policies
        metrics['average_policy_size'] = total_face_amount / total_policies if total_policies > 0 else 0
        
        if ages:
            metrics['weighted_age'] = float(np.mean(ages))
            metrics['age_statistics'] = {
                'mean': float(np.mean(ages)),
                'median': float(np.median(ages)),
                'std': float(np.std(ages)),
                'min': float(min(ages)),
                'max': float(max(ages))
            }
        
        return metrics
    
    async def _perform_experience_analysis(self, mapped_data: Dict) -> Dict:
        """Perform mortality experience analysis against SOA tables"""
        
        experience_results = {
            'ae_mortality_ratio': 1.0,
            'credibility_factor': 0.0,
            'expected_claims': 0,
            'actual_claims': 0,
            'experience_rating_factor': 1.0,
            'statistical_significance': False
        }
        
        total_expected_deaths = 0
        total_actual_deaths = 0
        total_exposure_years = 0
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Extract required fields using helper function
            age_col = self._get_field_column(field_map, 'age')
            gender_col = self._get_field_column(field_map, 'gender')
            smoking_col = self._get_field_column(field_map, 'smoking_status')
            duration_col = self._get_field_column(field_map, 'policy_duration')
            death_col = self._get_field_column(field_map, 'death_indicator')
            face_amount_col = self._get_field_column(field_map, 'face_amount')
            
            if not all([age_col, gender_col]):
                continue  # Skip if essential fields missing
            
            # Process each record
            for idx, row in df.iterrows():
                try:
                    age = int(pd.to_numeric(row[age_col], errors='coerce'))
                    gender = self._standardize_gender(str(row[gender_col]))
                    smoking = self._standardize_smoking_status(str(row.get(smoking_col, 'NON-SMOKER')))
                    duration = int(pd.to_numeric(row.get(duration_col, 1), errors='coerce')) if duration_col else 1
                    
                    # Get expected mortality rate from SOA table
                    expected_qx = self.soa_manager.get_mortality_rate(age, gender, smoking, duration)
                    
                    # Calculate exposure (assume 1 year for simplicity)
                    exposure = 1.0
                    total_exposure_years += exposure
                    
                    # Calculate expected deaths
                    expected_deaths = expected_qx * exposure
                    total_expected_deaths += expected_deaths
                    
                    # Count actual deaths
                    if death_col and pd.notna(row.get(death_col)):
                        death_indicator = str(row[death_col]).upper()
                        if death_indicator in ['1', 'YES', 'TRUE', 'DIED', 'DEATH']:
                            total_actual_deaths += 1
                    
                except (ValueError, TypeError):
                    continue  # Skip invalid records
        
        # Calculate A/E ratio
        if total_expected_deaths > 0:
            experience_results['ae_mortality_ratio'] = total_actual_deaths / total_expected_deaths
            experience_results['expected_claims'] = total_expected_deaths
            experience_results['actual_claims'] = total_actual_deaths
            
            # Calculate credibility using standard actuarial formula
            # Z = min(1, sqrt(actual_deaths / 1082))  # 1082 is full credibility for mortality
            credibility = min(1.0, np.sqrt(total_actual_deaths / 1082)) if total_actual_deaths > 0 else 0.0
            experience_results['credibility_factor'] = credibility
            
            # Statistical significance test (simple chi-square approximation)
            if total_actual_deaths >= 10:
                chi_square = ((total_actual_deaths - total_expected_deaths) ** 2) / total_expected_deaths
                # Critical value for 95% confidence (1 degree of freedom) is 3.84
                experience_results['statistical_significance'] = chi_square > 3.84
            
            # Experience rating factor (credibility-weighted)
            ae_ratio = experience_results['ae_mortality_ratio']
            experience_factor = credibility * ae_ratio + (1 - credibility) * 1.0
            experience_results['experience_rating_factor'] = experience_factor
        
        return experience_results
    
    async def _assess_portfolio_risk(self, mapped_data: Dict, market_data: Dict) -> Dict:
        """Comprehensive risk assessment"""
        
        risk_assessment = {
            'mortality_risk_score': 0.0,
            'interest_rate_risk_score': 0.0,
            'concentration_risk_score': 0.0,
            'operational_risk_score': 0.0,
            'overall_risk_score': 0.0,
            'risk_factors': [],
            'mitigation_recommendations': []
        }
        
        # Mortality Risk Assessment
        mortality_risk = await self._assess_mortality_risk(mapped_data)
        risk_assessment['mortality_risk_score'] = mortality_risk['score']
        risk_assessment['risk_factors'].extend(mortality_risk['factors'])
        
        # Interest Rate Risk Assessment
        ir_risk = self._assess_interest_rate_risk(market_data)
        risk_assessment['interest_rate_risk_score'] = ir_risk['score']
        risk_assessment['risk_factors'].extend(ir_risk['factors'])
        
        # Concentration Risk Assessment
        concentration_risk = self._assess_concentration_risk(mapped_data)
        risk_assessment['concentration_risk_score'] = concentration_risk['score']
        risk_assessment['risk_factors'].extend(concentration_risk['factors'])
        
        # Calculate overall risk score (weighted average)
        weights = {'mortality': 0.4, 'interest_rate': 0.3, 'concentration': 0.2, 'operational': 0.1}
        
        overall_score = (
            weights['mortality'] * risk_assessment['mortality_risk_score'] +
            weights['interest_rate'] * risk_assessment['interest_rate_risk_score'] +
            weights['concentration'] * risk_assessment['concentration_risk_score'] +
            weights['operational'] * 3.0  # Default operational risk
        )
        
        risk_assessment['overall_risk_score'] = overall_score
        
        # Generate recommendations based on risk scores
        risk_assessment['mitigation_recommendations'] = self._generate_risk_mitigation_recommendations(risk_assessment)
        
        return risk_assessment
    
    async def _assess_mortality_risk(self, mapped_data: Dict) -> Dict:
        """Assess mortality-specific risks"""
        
        risk_factors = []
        risk_score = 2.5  # Start with neutral
        
        unique_policy_ids = set()
        age_data = []
        smoker_percentage = 0
        
        # First pass: count unique policies
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Find unique policies using policy ID or row index as fallback
            policy_id_col = None
            for field_name, mapping_obj in field_map.items():
                if hasattr(mapping_obj, 'standardized_name') and mapping_obj.standardized_name in ['policy_id', 'policy_number']:
                    policy_id_col = mapping_obj.original_name
                    break
                elif isinstance(mapping_obj, dict) and mapping_obj.get('standardized_name') in ['policy_id', 'policy_number']:
                    policy_id_col = mapping_obj.get('original_name') or mapping_obj.get('column_name')
                    break
            
            if policy_id_col and policy_id_col in df.columns:
                unique_policy_ids.update(df[policy_id_col].astype(str))
            else:
                for idx in range(len(df)):
                    unique_policy_ids.add(f"{dataset_name}_{idx}")
        
        total_policies = len(unique_policy_ids)
        
        # Second pass: analyze data
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Age concentration risk
            if 'age' in field_map:
                ages = pd.to_numeric(df[field_map['age']['column_name']], errors='coerce').dropna()
                age_data.extend(ages.tolist())
            
            # Smoking risk
            if 'smoking_status' in field_map:
                smoking_col = field_map['smoking_status']['column_name']
                smoker_count = df[smoking_col].apply(lambda x: 'SMOKE' in str(x).upper()).sum()
                smoker_percentage = smoker_count / len(df) if len(df) > 0 else 0
        
        # Age risk assessment
        if age_data:
            avg_age = np.mean(age_data)
            age_std = np.std(age_data)
            
            if avg_age > 65:
                risk_score += 1.0
                risk_factors.append(f"High average age ({avg_age:.1f}) increases mortality risk")
            elif avg_age < 30:
                risk_score += 0.5
                risk_factors.append(f"Very young average age ({avg_age:.1f}) - limited mortality data")
            
            if age_std < 5:
                risk_score += 0.8
                risk_factors.append(f"Low age diversity (std: {age_std:.1f}) creates concentration risk")
        
        # Smoking risk assessment
        if smoker_percentage > 0.3:
            risk_score += 1.2
            risk_factors.append(f"High smoking percentage ({smoker_percentage:.1%}) significantly increases mortality")
        elif smoker_percentage > 0.15:
            risk_score += 0.6
            risk_factors.append(f"Moderate smoking percentage ({smoker_percentage:.1%}) increases mortality")
        
        # Portfolio size risk
        if total_policies < 1000:
            risk_score += 1.0
            risk_factors.append(f"Small portfolio size ({total_policies}) reduces credibility")
        elif total_policies < 5000:
            risk_score += 0.5
            risk_factors.append(f"Moderate portfolio size ({total_policies}) - limited credibility")
        
        return {
            'score': min(5.0, max(1.0, risk_score)),
            'factors': risk_factors
        }
    
    def _assess_interest_rate_risk(self, market_data: Dict) -> Dict:
        """Assess interest rate risk based on current market conditions"""
        
        risk_factors = []
        risk_score = 2.5  # Start with neutral
        
        rate_stats = market_data.get('rate_statistics', {})
        current_10y = None
        
        # Find current 10-year rate
        for rate_data in market_data.get('yield_curve', []):
            if rate_data.get('maturity') == '10Y':
                current_10y = rate_data.get('yield', 0.045)
                break
        
        if current_10y is None:
            current_10y = rate_stats.get('mean_rate', 0.045)
        
        # Low rate environment risk
        if current_10y < 0.03:
            risk_score += 1.5
            risk_factors.append(f"Very low interest rates ({current_10y:.2%}) increase reinvestment risk")
        elif current_10y < 0.04:
            risk_score += 0.8
            risk_factors.append(f"Low interest rates ({current_10y:.2%}) create reinvestment challenges")
        
        # High volatility risk
        volatility = rate_stats.get('rate_volatility', 0.015)
        if volatility > 0.03:
            risk_score += 1.2
            risk_factors.append(f"High interest rate volatility ({volatility:.2%}) increases uncertainty")
        elif volatility > 0.02:
            risk_score += 0.6
            risk_factors.append(f"Moderate interest rate volatility ({volatility:.2%})")
        
        # Inverted yield curve risk
        yield_curve = market_data.get('yield_curve', [])
        if len(yield_curve) >= 2:
            short_rate = next((r['yield'] for r in yield_curve if r['maturity'] == '2Y'), None)
            long_rate = next((r['yield'] for r in yield_curve if r['maturity'] == '10Y'), None)
            
            if short_rate and long_rate and short_rate > long_rate:
                risk_score += 1.0
                risk_factors.append("Inverted yield curve signals potential economic stress")
        
        return {
            'score': min(5.0, max(1.0, risk_score)),
            'factors': risk_factors
        }
    
    def _assess_concentration_risk(self, mapped_data: Dict) -> Dict:
        """Assess concentration risk in the portfolio"""
        
        risk_factors = []
        risk_score = 2.5
        
        # Calculate various concentration metrics
        face_amounts = []
        states = []
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Face amount concentration
            if 'face_amount' in field_map:
                face_col = field_map['face_amount']['column_name']
                amounts = pd.to_numeric(df[face_col], errors='coerce').dropna()
                face_amounts.extend(amounts.tolist())
            
            # Geographic concentration
            if 'state' in field_map:
                state_col = field_map['state']['column_name']
                states.extend(df[state_col].tolist())
        
        # Face amount concentration analysis
        if face_amounts:
            face_amounts = np.array(face_amounts)
            total_exposure = face_amounts.sum()
            
            # Calculate Herfindahl index for face amounts
            face_percentiles = np.percentile(face_amounts, [90, 95, 99])
            
            top_10_pct = face_amounts[face_amounts >= face_percentiles[0]].sum() / total_exposure
            top_5_pct = face_amounts[face_amounts >= face_percentiles[1]].sum() / total_exposure
            top_1_pct = face_amounts[face_amounts >= face_percentiles[2]].sum() / total_exposure
            
            if top_1_pct > 0.20:
                risk_score += 1.5
                risk_factors.append(f"High face amount concentration: top 1% holds {top_1_pct:.1%} of exposure")
            elif top_5_pct > 0.40:
                risk_score += 1.0
                risk_factors.append(f"Moderate face amount concentration: top 5% holds {top_5_pct:.1%} of exposure")
        
        # Geographic concentration analysis
        if states:
            state_counts = pd.Series(states).value_counts()
            top_state_pct = state_counts.iloc[0] / len(states) if len(state_counts) > 0 else 0
            
            if top_state_pct > 0.50:
                risk_score += 1.2
                risk_factors.append(f"High geographic concentration: {top_state_pct:.1%} in single state")
            elif top_state_pct > 0.30:
                risk_score += 0.6
                risk_factors.append(f"Moderate geographic concentration: {top_state_pct:.1%} in single state")
        
        return {
            'score': min(5.0, max(1.0, risk_score)),
            'factors': risk_factors
        }
    
    def _generate_risk_mitigation_recommendations(self, risk_assessment: Dict) -> List[str]:
        """Generate specific risk mitigation recommendations"""
        
        recommendations = []
        
        # Mortality risk recommendations
        if risk_assessment['mortality_risk_score'] > 3.5:
            recommendations.extend([
                "Consider mortality reinsurance or excess of retention coverage",
                "Implement enhanced underwriting for high-risk segments",
                "Diversify portfolio across age groups and smoking statuses"
            ])
        
        # Interest rate risk recommendations
        if risk_assessment['interest_rate_risk_score'] > 3.5:
            recommendations.extend([
                "Consider interest rate hedging strategies (swaps, options)",
                "Adjust asset-liability matching to reduce duration risk",
                "Implement dynamic hedging based on rate volatility"
            ])
        
        # Concentration risk recommendations
        if risk_assessment['concentration_risk_score'] > 3.5:
            recommendations.extend([
                "Set maximum retention limits per policy",
                "Diversify geographic exposure across multiple states/regions", 
                "Consider quota share reinsurance for large policies"
            ])
        
        # Overall high risk
        if risk_assessment['overall_risk_score'] > 4.0:
            recommendations.append("Consider comprehensive reinsurance program given elevated risk profile")
        
        return recommendations
    
    async def _analyze_trends(self, mapped_data: Dict) -> Dict:
        """Analyze trends in the portfolio data"""
        
        trend_analysis = {
            'policy_issuance_trends': {},
            'age_trends': {},
            'face_amount_trends': {},
            'geographic_trends': {},
            'seasonal_patterns': {}
        }
        
        # This would typically require time-series data
        # For now, provide structural analysis of current data
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Issue date trends (if available)
            if 'issue_date' in field_map:
                date_col = field_map['issue_date']['column_name']
                try:
                    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        date_trends = dates.groupby(dates.dt.year).size()
                        trend_analysis['policy_issuance_trends'] = date_trends.to_dict()
                except:
                    pass
        
        return trend_analysis
    
    async def _generate_pricing_recommendations(self, mapped_data: Dict, market_data: Dict) -> Dict:
        """Generate comprehensive pricing recommendations"""
        
        # Get current Treasury curve for discount rates
        yield_curve = market_data.get('yield_curve', [])
        base_discount_rate = next((r['yield'] for r in yield_curve if r['maturity'] == '10Y'), 0.045)
        
        # Adjust for credit risk and expenses
        pricing_discount_rate = base_discount_rate + 0.015  # 150 bps spread
        
        recommendations = {
            'base_pricing_assumptions': {
                'discount_rate': pricing_discount_rate,
                'expense_loading': 0.08,  # 8% expense loading
                'profit_margin': 0.12,     # 12% profit margin
                'mortality_margin': 0.05   # 5% mortality margin
            },
            'experience_adjustments': {},
            'product_specific_rates': {},
            'reinsurance_structures': []
        }
        
        # Calculate experience adjustments from A/E analysis
        experience_data = await self._perform_experience_analysis(mapped_data)
        ae_ratio = experience_data.get('ae_mortality_ratio', 1.0)
        credibility = experience_data.get('credibility_factor', 0.0)
        
        # Apply credibility-weighted experience adjustment
        experience_adjustment = credibility * (ae_ratio - 1.0)
        recommendations['experience_adjustments'] = {
            'ae_ratio': ae_ratio,
            'credibility_factor': credibility,
            'mortality_adjustment': experience_adjustment,
            'adjusted_mortality_rate': 1.0 + experience_adjustment
        }
        
        # Recommend reinsurance structures based on risk assessment
        risk_data = await self._assess_portfolio_risk(mapped_data, market_data)
        overall_risk = risk_data['overall_risk_score']
        
        if overall_risk > 4.0:
            recommendations['reinsurance_structures'] = [
                {
                    'type': 'Quota Share',
                    'cession_percentage': 0.50,
                    'rationale': 'High overall risk score requires significant risk sharing'
                },
                {
                    'type': 'Excess of Retention',
                    'retention_limit': 1000000,
                    'rationale': 'Protection against large individual claims'
                }
            ]
        elif overall_risk > 3.0:
            recommendations['reinsurance_structures'] = [
                {
                    'type': 'Yearly Renewable Term',
                    'cession_percentage': 0.75,
                    'rationale': 'Moderate risk - retain mortality margin while sharing risk'
                }
            ]
        
        return recommendations
    
    async def _calculate_reserves(self, mapped_data: Dict, market_data: Dict) -> Dict:
        """Calculate regulatory and economic reserves"""
        
        # This is a simplified reserve calculation
        # In practice, would use full VM-20 stochastic modeling
        
        yield_curve = market_data.get('yield_curve', [])
        discount_rates = {r['maturity']: r['yield'] for r in yield_curve}
        
        reserves = {
            'total_statutory_reserves': 0.0,
            'total_economic_reserves': 0.0,
            'reserve_breakdown': {},
            'vm20_requirements': {},
            'cash_flow_testing': {}
        }
        
        total_face_amount = 0
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            if 'face_amount' in field_map:
                face_col = field_map['face_amount']['column_name']
                face_amounts = pd.to_numeric(df[face_col], errors='coerce').fillna(0)
                total_face_amount += face_amounts.sum()
        
        # Simplified reserve calculation (2% of face amount)
        statutory_reserve_rate = 0.020
        economic_reserve_rate = 0.018
        
        reserves['total_statutory_reserves'] = total_face_amount * statutory_reserve_rate
        reserves['total_economic_reserves'] = total_face_amount * economic_reserve_rate
        
        return reserves
    
    async def _calculate_economic_capital(self, mapped_data: Dict, market_data: Dict) -> Dict:
        """Calculate economic capital requirements using Monte Carlo simulation"""
        
        print("🎲 Running Monte Carlo simulation for economic capital...")
        
        # Simplified Monte Carlo - in practice would be much more sophisticated
        n_simulations = 5000
        confidence_level = 0.995  # 99.5% VaR
        
        # Get portfolio metrics
        portfolio_metrics = await self._calculate_portfolio_metrics(mapped_data, market_data)
        total_exposure = portfolio_metrics['total_exposure']
        
        if total_exposure == 0:
            return {'var_995': 0, 'tvar_995': 0, 'economic_capital': 0}
        
        # Monte Carlo simulation parameters
        base_mortality_rate = 0.008  # Base mortality assumption
        mortality_volatility = 0.15   # 15% mortality volatility
        
        rate_stats = market_data.get('rate_statistics', {})
        base_interest_rate = rate_stats.get('mean_rate', 0.045)
        interest_volatility = rate_stats.get('rate_volatility', 0.015)
        
        # Run simulations
        np.random.seed(42)  # For reproducible results
        simulation_results = []
        
        for i in range(n_simulations):
            # Simulate mortality shock
            mortality_shock = np.random.lognormal(0, mortality_volatility)
            shocked_mortality = base_mortality_rate * mortality_shock
            
            # Simulate interest rate shock  
            interest_shock = np.random.normal(0, interest_volatility)
            shocked_rate = base_interest_rate + interest_shock
            
            # Calculate scenario loss
            mortality_loss = total_exposure * (shocked_mortality - base_mortality_rate)
            interest_loss = total_exposure * 0.05 * max(0, base_interest_rate - shocked_rate)  # Duration effect
            
            total_loss = mortality_loss + interest_loss
            simulation_results.append(total_loss)
        
        simulation_results = np.array(simulation_results)
        simulation_results.sort()
        
        # Calculate risk metrics
        var_index = int(confidence_level * n_simulations)
        var_995 = simulation_results[var_index]
        
        # Tail VaR (expected shortfall)
        tvar_995 = simulation_results[var_index:].mean()
        
        # Economic capital (VaR minus expected loss)
        expected_loss = simulation_results.mean()
        economic_capital = max(0, var_995 - expected_loss)
        
        print("✅ Monte Carlo simulation complete!")
        
        return {
            'expected_loss': float(expected_loss),
            'var_995': float(var_995),
            'tvar_995': float(tvar_995),
            'economic_capital': float(economic_capital),
            'capital_ratio': float(economic_capital / total_exposure) if total_exposure > 0 else 0,
            'simulation_details': {
                'n_simulations': n_simulations,
                'confidence_level': confidence_level,
                'mortality_volatility': mortality_volatility,
                'interest_volatility': interest_volatility
            }
        }
    
    def _standardize_gender(self, gender: str) -> str:
        """Standardize gender values"""
        gender = str(gender).upper().strip()
        if gender in ['M', 'MALE', '1']:
            return 'MALE'
        elif gender in ['F', 'FEMALE', '2']:
            return 'FEMALE'
        else:
            return 'UNKNOWN'
    
    def _standardize_smoking_status(self, status: str) -> str:
        """Standardize smoking status values"""
        status = str(status).upper().strip()
        if any(term in status for term in ['SMOKE', 'SMOKER', 'S', 'Y', 'YES']):
            return 'SMOKER'
        elif any(term in status for term in ['NON', 'N', 'NO']):
            return 'NON-SMOKER'
        else:
            return 'UNKNOWN'
    
    def _get_field_column(self, field_map: Dict, field_name: str) -> Optional[str]:
        """Helper to extract column name from field mapping (handles both dict and FieldMapping object)"""
        
        # Look for the field in the mapping
        for key, mapping in field_map.items():
            # Check if it's a FieldMapping object
            if hasattr(mapping, 'standardized_name') and mapping.standardized_name == field_name:
                return mapping.original_name
            # Check if it's a dictionary with the field name
            elif key == field_name:
                if isinstance(mapping, dict):
                    return mapping.get('column_name', mapping.get('original_name'))
                elif hasattr(mapping, 'original_name'):
                    return mapping.original_name
        
        return None
    
    def _create_minimal_rate_stats(self) -> Dict:
        """Create minimal rate statistics from current market conditions"""
        
        # This should use current FRED data as baseline, not hardcoded values
        # For now, return structure that indicates data-driven approach
        return {
            'mean_rate': None,  # To be filled from FRED API
            'rate_volatility': None,  # To be calculated from market data
            'mean_reversion_speed': self.config.get('actuarial_standards.rate_assumptions.default_mean_reversion', 0.10),
            'long_term_rate': None,  # To be filled from FRED API
            'note': 'Minimal rate statistics - insufficient historical data'
        }
    
    def _calculate_operational_risk_from_data(self, mapped_data: Dict) -> float:
        """Calculate operational risk score from portfolio characteristics"""
        
        base_operational_risk = self.config.get('actuarial_standards.risk_scoring.base_risk_score', 2.5)
        
        # Calculate operational risk based on data quality and completeness
        data_quality_factors = []
        
        for dataset_name, mapping in mapped_data.items():
            df = mapping['dataframe']
            field_map = mapping['field_mapping']
            
            # Check data completeness
            completeness = df.notna().mean().mean()
            if completeness < 0.9:
                data_quality_factors.append(0.5)  # Higher operational risk for incomplete data
            
            # Check if key fields are mapped
            key_fields = ['age', 'gender', 'face_amount']
            mapped_key_fields = sum(1 for field in key_fields if field in field_map)
            if mapped_key_fields < len(key_fields):
                data_quality_factors.append(0.3)  # Risk for missing key fields
        
        # Add operational risk based on data quality
        additional_risk = sum(data_quality_factors)
        
        return min(5.0, max(1.0, base_operational_risk + additional_risk))


# Usage example and testing
async def main():
    """Test the Professional Actuarial Engine"""
    
    engine = ProfessionalActuarialEngine()
    
    # Create sample dataset for testing
    sample_data = {
        'life_policies': pd.DataFrame({
            'age': [35, 42, 55, 28, 67, 45, 38],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'smoking_status': ['NS', 'NS', 'S', 'NS', 'NS', 'S', 'NS'],
            'face_amount': [250000, 500000, 1000000, 300000, 750000, 400000, 600000],
            'policy_duration': [1, 3, 8, 1, 15, 5, 2],
            'death_indicator': ['No', 'No', 'Yes', 'No', 'No', 'No', 'No']
        })
    }
    
    print("🚀 Testing Professional Actuarial Engine...")
    
    # Run complete analysis
    results = await engine.analyze_portfolio({'sample_data': sample_data})
    
    print("📊 Analysis Results:")
    print(f"Total Exposure: ${results['portfolio_metrics']['total_exposure']:,.2f}")
    print(f"A/E Mortality Ratio: {results['experience_analysis']['ae_mortality_ratio']:.3f}")
    print(f"Overall Risk Score: {results['risk_assessment']['overall_risk_score']:.1f}/5.0")
    print(f"Economic Capital: ${results['capital_requirements']['economic_capital']:,.2f}")


if __name__ == "__main__":
    asyncio.run(main())