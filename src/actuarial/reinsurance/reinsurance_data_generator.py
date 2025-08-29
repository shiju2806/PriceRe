"""
Reinsurance Data Generator
Creates realistic reinsurance treaty and cedent data
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass

from .treaty_pricing_engine import TreatyType, BusinessLine, TreatyTerms, CedentExperience

@dataclass
class ReinsuranceDataConfig:
    """Configuration for reinsurance data generation"""
    n_cedents: int = 25
    n_treaties: int = 50
    years_of_history: int = 10
    include_cat_events: bool = True
    include_pandemic: bool = True
    realistic_market_cycles: bool = True

class ReinsuranceDataGenerator:
    """Generate comprehensive reinsurance datasets"""
    
    def __init__(self, config: Optional[ReinsuranceDataConfig] = None):
        self.config = config or ReinsuranceDataConfig()
        self.setup_market_data()
    
    def setup_market_data(self):
        """Setup market reference data"""
        
        # Major insurance companies (cedents)
        self.cedent_companies = [
            {'name': 'MetLife', 'size': 'Large', 'quality': 'A', 'specialty': 'Group Life'},
            {'name': 'Prudential Financial', 'size': 'Large', 'quality': 'A', 'specialty': 'Individual Life'},
            {'name': 'New York Life', 'size': 'Large', 'quality': 'A', 'specialty': 'Whole Life'},
            {'name': 'Northwestern Mutual', 'size': 'Large', 'quality': 'A', 'specialty': 'Individual Life'},
            {'name': 'MassMutual', 'size': 'Medium', 'quality': 'B', 'specialty': 'Universal Life'},
            {'name': 'Guardian Life', 'size': 'Medium', 'quality': 'B', 'specialty': 'Term Life'},
            {'name': 'Lincoln Financial', 'size': 'Medium', 'quality': 'B', 'specialty': 'Annuities'},
            {'name': 'Principal Financial', 'size': 'Medium', 'quality': 'B', 'specialty': 'Pension'},
            {'name': 'Pacific Life', 'size': 'Medium', 'quality': 'B', 'specialty': 'Annuities'},
            {'name': 'Ameritas', 'size': 'Small', 'quality': 'C', 'specialty': 'Term Life'},
            {'name': 'Mutual of Omaha', 'size': 'Small', 'quality': 'B', 'specialty': 'Individual Life'},
            {'name': 'Security Benefit', 'size': 'Small', 'quality': 'C', 'specialty': 'Annuities'},
            {'name': 'Foresters', 'size': 'Small', 'quality': 'C', 'specialty': 'Fraternal'},
            {'name': 'Penn Mutual', 'size': 'Small', 'quality': 'B', 'specialty': 'Whole Life'},
            {'name': 'Symetra', 'size': 'Small', 'quality': 'C', 'specialty': 'Annuities'}
        ]
        
        # Market events affecting mortality/morbidity
        self.market_events = [
            {'year': 2020, 'event': 'COVID-19 Pandemic', 'mortality_impact': 1.15, 'duration': 2},
            {'year': 2008, 'event': 'Financial Crisis', 'lapse_impact': 1.4, 'duration': 2},
            {'year': 2001, 'event': '9/11 Attacks', 'mortality_impact': 1.02, 'duration': 1},
            {'year': 2018, 'event': 'Flu Season', 'mortality_impact': 1.08, 'duration': 1},
            {'year': 2017, 'event': 'Hurricane Season', 'mortality_impact': 1.03, 'duration': 1}
        ]
        
        # Reinsurers
        self.reinsurers = ['Swiss Re', 'Munich Re', 'RGA', 'Gen Re', 'SCOR', 'Hannover Re', 'Optimum Re']
    
    def generate_reinsurance_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete reinsurance dataset"""
        
        print("🏢 Generating Reinsurance Dataset...")
        
        # Generate cedent profiles
        print("🏛️ Generating cedent companies...")
        cedent_profiles = self._generate_cedent_profiles()
        
        # Generate historical experience
        print("📊 Generating historical experience...")
        experience_data = self._generate_experience_data(cedent_profiles)
        
        # Generate treaty data
        print("📜 Generating reinsurance treaties...")
        treaty_data = self._generate_treaty_data(cedent_profiles)
        
        # Generate portfolio data
        print("💼 Generating portfolio exposures...")
        portfolio_data = self._generate_portfolio_data(cedent_profiles)
        
        # Generate claims data
        print("🏥 Generating reinsurance claims...")
        claims_data = self._generate_reinsurance_claims(treaty_data, experience_data)
        
        # Generate catastrophe events
        print("🌪️ Generating catastrophe events...")
        cat_events = self._generate_cat_events()
        
        print("✅ Reinsurance dataset generation complete!")
        
        return {
            'cedent_profiles': cedent_profiles,
            'experience_data': experience_data,
            'treaty_data': treaty_data,
            'portfolio_data': portfolio_data,
            'reinsurance_claims': claims_data,
            'catastrophe_events': cat_events
        }
    
    def _generate_cedent_profiles(self) -> pd.DataFrame:
        """Generate ceding company profiles"""
        
        profiles = []
        
        for i in range(self.config.n_cedents):
            if i < len(self.cedent_companies):
                company = self.cedent_companies[i]
            else:
                # Generate additional fictional companies
                company = {
                    'name': f'Regional Life {i+1}',
                    'size': np.random.choice(['Small', 'Medium'], p=[0.6, 0.4]),
                    'quality': np.random.choice(['B', 'C'], p=[0.7, 0.3]),
                    'specialty': np.random.choice(['Individual Life', 'Group Life', 'Annuities'])
                }
            
            # Premium volume based on size
            if company['size'] == 'Large':
                base_premium = np.random.uniform(500_000_000, 2_000_000_000)
            elif company['size'] == 'Medium':
                base_premium = np.random.uniform(50_000_000, 500_000_000)
            else:
                base_premium = np.random.uniform(5_000_000, 50_000_000)
            
            # Geographic concentration
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
            primary_state = np.random.choice(states)
            
            if company['size'] == 'Large':
                concentration = np.random.uniform(0.15, 0.35)
            else:
                concentration = np.random.uniform(0.30, 0.65)
            
            profiles.append({
                'cedent_id': f'CED_{i+1:03d}',
                'cedent_name': company['name'],
                'size_category': company['size'],
                'underwriting_grade': company['quality'],
                'specialty': company['specialty'],
                'base_annual_premium': base_premium,
                'primary_state': primary_state,
                'geographic_concentration': concentration,
                'years_in_business': np.random.randint(10, 75),
                'am_best_rating': self._assign_am_best_rating(company['quality']),
                'active_since': 2023 - np.random.randint(5, 25)
            })
        
        return pd.DataFrame(profiles)
    
    def _generate_experience_data(self, cedent_profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate historical experience data for each cedent"""
        
        experience_data = []
        current_year = 2023
        
        for _, cedent in cedent_profiles.iterrows():
            base_premium = cedent['base_annual_premium']
            quality_grade = cedent['underwriting_grade']
            
            # Generate experience for each year
            for year in range(current_year - self.config.years_of_history, current_year):
                # Premium growth (realistic patterns)
                years_from_base = current_year - year
                growth_rate = np.random.normal(0.05, 0.03)  # 5% average growth
                premium = base_premium * (1 + growth_rate) ** (-years_from_base)
                
                # Face amount in force
                face_inforce = premium * np.random.uniform(150, 250)  # Premium to face ratio
                
                # Policy count
                avg_face = np.random.uniform(200_000, 400_000)
                policy_count = int(face_inforce / avg_face)
                
                # Loss ratio based on quality and market events
                base_loss_ratio = self._get_base_loss_ratio(quality_grade, cedent['specialty'])
                
                # Apply market event impacts
                loss_ratio = base_loss_ratio
                for event in self.market_events:
                    if event['year'] == year:
                        if 'mortality_impact' in event:
                            loss_ratio *= event['mortality_impact']
                        if 'lapse_impact' in event:
                            # Lapse impacts affect different products differently
                            if cedent['specialty'] in ['Individual Life', 'Whole Life']:
                                loss_ratio *= 0.9  # Lower losses due to lapses
                
                # Add random variation
                loss_ratio *= np.random.uniform(0.85, 1.15)
                
                # Calculate claims
                incurred_claims = premium * loss_ratio
                paid_claims = incurred_claims * np.random.uniform(0.95, 1.0)
                
                # Lapse rates
                base_lapse_rate = 0.08  # 8% base
                if cedent['specialty'] == 'Term Life':
                    base_lapse_rate = 0.12
                elif cedent['specialty'] == 'Whole Life':
                    base_lapse_rate = 0.05
                elif cedent['specialty'] == 'Annuities':
                    base_lapse_rate = 0.03
                
                lapse_rate = base_lapse_rate * np.random.uniform(0.7, 1.3)
                
                # A/E mortality ratios (for life business)
                if 'Life' in cedent['specialty']:
                    if quality_grade == 'A':
                        ae_ratio = np.random.normal(0.85, 0.1)
                    elif quality_grade == 'B':
                        ae_ratio = np.random.normal(0.95, 0.12)
                    else:  # Grade C
                        ae_ratio = np.random.normal(1.15, 0.15)
                else:
                    ae_ratio = None
                
                experience_data.append({
                    'cedent_id': cedent['cedent_id'],
                    'cedent_name': cedent['cedent_name'],
                    'experience_year': year,
                    'business_line': self._map_specialty_to_business_line(cedent['specialty']),
                    'premium_volume': premium,
                    'face_amount_inforce': face_inforce,
                    'policy_count': policy_count,
                    'incurred_claims': incurred_claims,
                    'paid_claims': paid_claims,
                    'loss_ratio': loss_ratio,
                    'lapse_rate': lapse_rate,
                    'surrender_rate': lapse_rate * 0.6,  # 60% of lapses are surrenders
                    'ae_mortality_ratio': ae_ratio,
                    'avg_face_amount': face_inforce / policy_count if policy_count > 0 else 0
                })
        
        return pd.DataFrame(experience_data)
    
    def _generate_treaty_data(self, cedent_profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate reinsurance treaty data"""
        
        treaties = []
        treaty_id = 1
        
        for _, cedent in cedent_profiles.iterrows():
            # Number of treaties per cedent based on size
            if cedent['size_category'] == 'Large':
                n_treaties = np.random.randint(3, 8)
            elif cedent['size_category'] == 'Medium':
                n_treaties = np.random.randint(2, 5)
            else:
                n_treaties = np.random.randint(1, 3)
            
            for t in range(n_treaties):
                # Treaty type distribution
                treaty_types = list(TreatyType)
                if cedent['size_category'] == 'Small':
                    # Small companies more likely to use quota share
                    weights = [0.4, 0.2, 0.2, 0.1, 0.1, 0.0]  # 6 weights for 6 treaty types
                else:
                    weights = [0.2, 0.3, 0.3, 0.1, 0.1, 0.0]  # 6 weights for 6 treaty types
                
                # Ensure weights match treaty types length and sum to 1
                weights = weights[:len(treaty_types)]
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                treaty_type = np.random.choice(treaty_types, p=weights)
                
                # Business line based on cedent specialty
                business_line = self._map_specialty_to_business_line(cedent['specialty'])
                
                # Treaty terms
                effective_date = date(2024, 1, 1)
                expiry_date = date(2024, 12, 31)
                
                # Retention and limits based on treaty type and cedent size
                retention_amount, treaty_limit = self._calculate_treaty_terms(
                    treaty_type, cedent['size_category'], cedent['base_annual_premium']
                )
                
                # Rates and commissions
                reinsurance_rate = self._calculate_treaty_rate(
                    treaty_type, business_line, cedent['underwriting_grade']
                )
                
                commission_rate = self._calculate_commission_rate(treaty_type, business_line)
                
                treaties.append({
                    'treaty_id': f'TRT_{treaty_id:04d}',
                    'cedent_id': cedent['cedent_id'],
                    'cedent_name': cedent['cedent_name'],
                    'treaty_type': treaty_type.value,
                    'business_line': business_line.value,
                    'effective_date': effective_date,
                    'expiry_date': expiry_date,
                    'retention_amount': retention_amount,
                    'treaty_limit': treaty_limit,
                    'reinsurance_premium_rate': reinsurance_rate,
                    'commission_rate': commission_rate,
                    'profit_commission_rate': 0.2 if np.random.random() < 0.6 else 0,
                    'profit_commission_threshold': 0.75,
                    'reinsurer': np.random.choice(self.reinsurers),
                    'territory': 'USA',
                    'currency': 'USD',
                    'experience_rating': np.random.choice([True, False], p=[0.3, 0.7]),
                    'aggregate_limit': treaty_limit * 2 if treaty_type in [TreatyType.CATASTROPHE, TreatyType.STOP_LOSS] else None
                })
                
                treaty_id += 1
        
        return pd.DataFrame(treaties)
    
    def _generate_portfolio_data(self, cedent_profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed portfolio exposure data"""
        
        portfolio_data = []
        
        for _, cedent in cedent_profiles.iterrows():
            # Break down portfolio by age bands, product types, etc.
            age_bands = ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76+']
            product_types = ['Term', 'Whole Life', 'Universal Life', 'Variable Life', 'Annuities']
            
            total_premium = cedent['base_annual_premium']
            
            for age_band in age_bands:
                for product in product_types:
                    # Realistic distribution weights
                    age_weight = self._get_age_weight(age_band)
                    product_weight = self._get_product_weight(product, cedent['specialty'])
                    
                    allocation = age_weight * product_weight
                    segment_premium = total_premium * allocation
                    
                    if segment_premium > 1000:  # Only include meaningful segments
                        # Calculate segment characteristics
                        avg_face = self._get_avg_face_amount(age_band, product)
                        policy_count = int(segment_premium * np.random.uniform(120, 180) / avg_face)
                        
                        portfolio_data.append({
                            'cedent_id': cedent['cedent_id'],
                            'age_band': age_band,
                            'product_type': product,
                            'premium_volume': segment_premium,
                            'policy_count': policy_count,
                            'avg_face_amount': avg_face,
                            'total_face_amount': policy_count * avg_face,
                            'avg_issue_age': self._get_avg_issue_age(age_band),
                            'male_percentage': np.random.uniform(0.45, 0.55),
                            'smoker_percentage': np.random.uniform(0.10, 0.20),
                            'geographic_concentration': cedent['geographic_concentration'],
                            'primary_state': cedent['primary_state']
                        })
        
        return pd.DataFrame(portfolio_data)
    
    def _generate_reinsurance_claims(self, treaty_data: pd.DataFrame, experience_data: pd.DataFrame) -> pd.DataFrame:
        """Generate reinsurance claims data"""
        
        claims_data = []
        claim_id = 1
        
        for _, treaty in treaty_data.iterrows():
            cedent_id = treaty['cedent_id']
            treaty_type = TreatyType(treaty['treaty_type'])
            retention = treaty['retention_amount']
            
            # Get cedent's claims experience
            cedent_claims = experience_data[
                (experience_data['cedent_id'] == cedent_id) & 
                (experience_data['experience_year'] >= 2020)
            ]
            
            for _, exp in cedent_claims.iterrows():
                total_claims = exp['incurred_claims']
                
                # Simulate individual claims that would hit the treaty
                if treaty_type == TreatyType.QUOTA_SHARE:
                    # All claims participate proportionally
                    n_claims = np.random.poisson(total_claims / 50000)  # Average claim size $50K
                    
                    for c in range(min(n_claims, 20)):  # Limit for processing
                        claim_amount = np.random.lognormal(10.8, 0.8)  # ~$50K average
                        
                        if treaty['retention_amount']:
                            retention_pct = treaty['retention_amount']
                        else:
                            retention_pct = 0.5  # 50% quota share
                        
                        reins_amount = claim_amount * (1 - retention_pct)
                        
                        if reins_amount > 1000:  # Material claims only
                            claims_data.append({
                                'claim_id': f'CLM_{claim_id:06d}',
                                'treaty_id': treaty['treaty_id'],
                                'cedent_id': cedent_id,
                                'claim_year': exp['experience_year'],
                                'claim_date': self._random_date_in_year(exp['experience_year']),
                                'gross_claim_amount': claim_amount,
                                'retention_amount': claim_amount * retention_pct,
                                'reinsurance_amount': reins_amount,
                                'claim_cause': self._random_claim_cause(),
                                'claim_status': 'Paid',
                                'days_to_settlement': np.random.gamma(2, 20)
                            })
                            claim_id += 1
                
                elif treaty_type == TreatyType.XS_OF_LOSS:
                    # Only large claims
                    n_large_claims = max(1, int(total_claims / 500000))  # Fewer, larger claims
                    
                    for c in range(n_large_claims):
                        claim_amount = np.random.lognormal(12.5, 0.6)  # ~$300K average
                        
                        if claim_amount > retention:
                            reins_amount = min(claim_amount - retention, treaty['treaty_limit'])
                            
                            claims_data.append({
                                'claim_id': f'CLM_{claim_id:06d}',
                                'treaty_id': treaty['treaty_id'],
                                'cedent_id': cedent_id,
                                'claim_year': exp['experience_year'],
                                'claim_date': self._random_date_in_year(exp['experience_year']),
                                'gross_claim_amount': claim_amount,
                                'retention_amount': retention,
                                'reinsurance_amount': reins_amount,
                                'claim_cause': self._random_claim_cause(),
                                'claim_status': 'Paid',
                                'days_to_settlement': np.random.gamma(3, 25)
                            })
                            claim_id += 1
        
        return pd.DataFrame(claims_data)
    
    def _generate_cat_events(self) -> pd.DataFrame:
        """Generate catastrophic events affecting multiple cedents"""
        
        events = []
        
        # Historical events
        major_events = [
            {'year': 2020, 'event': 'COVID-19 Pandemic', 'type': 'Pandemic', 'severity': 'Extreme'},
            {'year': 2017, 'event': 'Hurricane Harvey/Irma', 'type': 'Natural Disaster', 'severity': 'High'},
            {'year': 2012, 'event': 'Hurricane Sandy', 'type': 'Natural Disaster', 'severity': 'High'},
            {'year': 2005, 'event': 'Hurricane Katrina', 'type': 'Natural Disaster', 'severity': 'Extreme'},
            {'year': 2001, 'event': '9/11 Terrorist Attacks', 'type': 'Man-made', 'severity': 'High'}
        ]
        
        for event_data in major_events:
            # Estimate industry losses
            if event_data['severity'] == 'Extreme':
                industry_loss = np.random.uniform(50_000_000_000, 200_000_000_000)
            else:
                industry_loss = np.random.uniform(5_000_000_000, 50_000_000_000)
            
            events.append({
                'event_id': f"CAT_{event_data['year']}_{event_data['event'][:5]}",
                'event_name': event_data['event'],
                'event_year': event_data['year'],
                'event_type': event_data['type'],
                'severity_level': event_data['severity'],
                'estimated_industry_loss': industry_loss,
                'duration_months': 3 if event_data['type'] == 'Pandemic' else 1,
                'geographic_impact': 'National' if event_data['type'] in ['Pandemic', 'Man-made'] else 'Regional',
                'affected_lines': 'Life, Group Life' if event_data['type'] != 'Natural Disaster' else 'All Lines'
            })
        
        return pd.DataFrame(events)
    
    # Helper methods
    def _assign_am_best_rating(self, quality_grade: str) -> str:
        """Assign AM Best rating based on quality grade"""
        if quality_grade == 'A':
            return np.random.choice(['A++', 'A+', 'A'], p=[0.1, 0.3, 0.6])
        elif quality_grade == 'B':
            return np.random.choice(['A-', 'B++', 'B+'], p=[0.2, 0.4, 0.4])
        else:
            return np.random.choice(['B', 'B-', 'C++'], p=[0.5, 0.3, 0.2])
    
    def _get_base_loss_ratio(self, quality_grade: str, specialty: str) -> float:
        """Get base loss ratio by quality and specialty"""
        base_ratios = {
            'Individual Life': 0.68,
            'Group Life': 0.72,
            'Whole Life': 0.65,
            'Term Life': 0.70,
            'Universal Life': 0.69,
            'Annuities': 0.45,
            'Pension': 0.55
        }
        
        base = base_ratios.get(specialty, 0.68)
        
        # Quality adjustment
        if quality_grade == 'A':
            return base * 0.92
        elif quality_grade == 'C':
            return base * 1.15
        else:
            return base
    
    def _map_specialty_to_business_line(self, specialty: str) -> BusinessLine:
        """Map cedent specialty to business line"""
        mapping = {
            'Individual Life': BusinessLine.INDIVIDUAL_LIFE,
            'Group Life': BusinessLine.GROUP_LIFE,
            'Whole Life': BusinessLine.INDIVIDUAL_LIFE,
            'Term Life': BusinessLine.INDIVIDUAL_LIFE,
            'Universal Life': BusinessLine.INDIVIDUAL_LIFE,
            'Annuities': BusinessLine.ANNUITIES,
            'Pension': BusinessLine.PENSION
        }
        return mapping.get(specialty, BusinessLine.INDIVIDUAL_LIFE)
    
    def _calculate_treaty_terms(self, treaty_type: TreatyType, size: str, base_premium: float) -> Tuple[float, Optional[float]]:
        """Calculate retention and limits"""
        if treaty_type == TreatyType.QUOTA_SHARE:
            retention_pct = np.random.uniform(0.2, 0.7)
            return retention_pct, None
        elif treaty_type == TreatyType.SURPLUS:
            if size == 'Large':
                retention = np.random.choice([500_000, 1_000_000, 2_000_000])
            else:
                retention = np.random.choice([100_000, 250_000, 500_000])
            limit = retention * np.random.randint(5, 20)
            return retention, limit
        else:  # XS or Cat
            if size == 'Large':
                retention = np.random.choice([2_000_000, 5_000_000, 10_000_000])
            else:
                retention = np.random.choice([500_000, 1_000_000, 2_000_000])
            limit = retention * np.random.randint(2, 10)
            return retention, limit
    
    def _calculate_treaty_rate(self, treaty_type: TreatyType, business_line: BusinessLine, grade: str) -> float:
        """Calculate reinsurance rate"""
        base_rates = {
            TreatyType.QUOTA_SHARE: 0.65,
            TreatyType.SURPLUS: 0.45,
            TreatyType.XS_OF_LOSS: 0.25,
            TreatyType.STOP_LOSS: 0.15,
            TreatyType.CATASTROPHE: 0.08
        }
        
        base = base_rates[treaty_type]
        
        # Quality adjustment
        if grade == 'A':
            base *= 0.9
        elif grade == 'C':
            base *= 1.2
        
        return base * np.random.uniform(0.9, 1.1)
    
    def _calculate_commission_rate(self, treaty_type: TreatyType, business_line: BusinessLine) -> float:
        """Calculate commission rate"""
        if treaty_type in [TreatyType.QUOTA_SHARE, TreatyType.SURPLUS]:
            return np.random.uniform(0.15, 0.35)
        else:
            return 0.0  # XS treaties typically don't pay commission
    
    def _get_age_weight(self, age_band: str) -> float:
        """Get portfolio weight by age band"""
        weights = {
            '18-25': 0.08, '26-35': 0.22, '36-45': 0.28, 
            '46-55': 0.25, '56-65': 0.12, '66-75': 0.04, '76+': 0.01
        }
        return weights.get(age_band, 0.1)
    
    def _get_product_weight(self, product: str, specialty: str) -> float:
        """Get product weight based on cedent specialty"""
        if specialty == 'Term Life':
            weights = {'Term': 0.7, 'Whole Life': 0.1, 'Universal Life': 0.1, 'Variable Life': 0.05, 'Annuities': 0.05}
        elif specialty == 'Whole Life':
            weights = {'Term': 0.2, 'Whole Life': 0.6, 'Universal Life': 0.1, 'Variable Life': 0.05, 'Annuities': 0.05}
        elif specialty == 'Annuities':
            weights = {'Term': 0.1, 'Whole Life': 0.1, 'Universal Life': 0.1, 'Variable Life': 0.1, 'Annuities': 0.6}
        else:  # Individual Life, Group Life
            weights = {'Term': 0.35, 'Whole Life': 0.25, 'Universal Life': 0.2, 'Variable Life': 0.1, 'Annuities': 0.1}
        
        return weights.get(product, 0.2)
    
    def _get_avg_face_amount(self, age_band: str, product: str) -> float:
        """Get average face amount by age and product"""
        base_amounts = {
            'Term': 300_000, 'Whole Life': 150_000, 'Universal Life': 250_000,
            'Variable Life': 400_000, 'Annuities': 200_000
        }
        
        age_multipliers = {
            '18-25': 0.6, '26-35': 0.8, '36-45': 1.2, 
            '46-55': 1.4, '56-65': 1.1, '66-75': 0.8, '76+': 0.5
        }
        
        return base_amounts[product] * age_multipliers.get(age_band, 1.0)
    
    def _get_avg_issue_age(self, age_band: str) -> float:
        """Get average issue age for age band"""
        midpoints = {
            '18-25': 22, '26-35': 31, '36-45': 41, 
            '46-55': 51, '56-65': 61, '66-75': 71, '76+': 78
        }
        return midpoints.get(age_band, 45)
    
    def _random_date_in_year(self, year: int) -> date:
        """Generate random date within a year"""
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        days_diff = (end_date - start_date).days
        random_days = np.random.randint(0, days_diff)
        return start_date + timedelta(days=random_days)
    
    def _random_claim_cause(self) -> str:
        """Generate random claim cause"""
        causes = [
            'Heart Disease', 'Cancer', 'Stroke', 'Accident', 'Respiratory Disease',
            'Diabetes', 'Kidney Disease', 'Suicide', 'Homicide', 'Other'
        ]
        weights = [0.25, 0.22, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.3]
        return np.random.choice(causes, p=weights)