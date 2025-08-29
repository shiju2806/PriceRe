"""
Reinsurance Data Generator

Generates realistic reinsurance treaty datasets including:
- Quota Share treaties
- Surplus treaties  
- Excess of Loss treaties
- Catastrophe reinsurance
- Life reinsurance
- Aggregate claims data
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple
import uuid


class TreatyType(Enum):
    QUOTA_SHARE = "Quota Share"
    SURPLUS = "Surplus"
    EXCESS_OF_LOSS = "Excess of Loss"
    CATASTROPHE = "Catastrophe"
    LIFE_QUOTA_SHARE = "Life Quota Share"
    LIFE_COINSURANCE = "Life Coinsurance"


class BusinessLine(Enum):
    PROPERTY = "Property"
    CASUALTY = "Casualty"
    MOTOR = "Motor"
    MARINE = "Marine"
    AVIATION = "Aviation"
    WORKERS_COMP = "Workers Compensation"
    LIFE = "Life"
    HEALTH = "Health"
    DISABILITY = "Disability"


class ReinsuranceDataGenerator:
    """Generates realistic reinsurance treaty and claims data"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        
        # Market parameters
        self.currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY"]
        self.territories = [
            "United States", "United Kingdom", "Germany", "France", "Canada",
            "Australia", "Japan", "Switzerland", "Netherlands", "Sweden"
        ]
        self.cedants = [
            "Global Insurance Co", "National Mutual", "Regional General",
            "Pacific Life", "Atlantic Casualty", "Continental Re", 
            "Metropolitan Insurance", "Liberty Mutual", "State Farm",
            "AIG", "Travelers", "Hartford", "Chubb", "Zurich"
        ]
        self.reinsurers = [
            "Munich Re", "Swiss Re", "Hannover Re", "Lloyd's of London",
            "Berkshire Hathaway Re", "General Re", "SCOR", "RGA",
            "Partner Re", "Everest Re", "Transatlantic Re", "Odyssey Re"
        ]
    
    def generate_treaty_data(self, n_treaties: int = 100) -> pl.DataFrame:
        """Generate realistic reinsurance treaty data"""
        
        treaties = []
        for _ in range(n_treaties):
            treaty_type = random.choice(list(TreatyType))
            business_line = random.choice(list(BusinessLine))
            
            # Generate treaty terms based on type
            terms = self._generate_treaty_terms(treaty_type, business_line)
            
            treaty = {
                "treaty_id": str(uuid.uuid4())[:8],
                "treaty_name": f"{terms['cedant']} - {treaty_type.value} {random.randint(2020, 2024)}",
                "treaty_type": treaty_type.value,
                "business_line": business_line.value,
                "cedant": terms['cedant'],
                "reinsurer": terms['reinsurer'],
                "currency": terms['currency'],
                "territory": terms['territory'],
                "inception_date": terms['inception_date'],
                "expiry_date": terms['expiry_date'],
                "premium": terms['premium'],
                "commission": terms['commission'],
                "brokerage": terms['brokerage'],
                "retention": terms['retention'],
                "limit": terms['limit'],
                "cession_rate": terms['cession_rate'],
                "minimum_premium": terms['minimum_premium'],
                "maximum_premium": terms['maximum_premium'],
                "profit_commission": terms['profit_commission'],
                "loss_corridor": terms['loss_corridor'],
                "aggregate_limit": terms['aggregate_limit'],
                "reinstatements": terms['reinstatements'],
                "rating_method": terms['rating_method'],
                "experience_period": terms['experience_period'],
                "loss_ratio": terms['loss_ratio'],
                "expense_ratio": terms['expense_ratio'],
                "combined_ratio": terms['combined_ratio']
            }
            treaties.append(treaty)
        
        return pl.DataFrame(treaties)
    
    def generate_claims_data(self, treaty_df: pl.DataFrame, claims_per_treaty: int = 50) -> pl.DataFrame:
        """Generate realistic claims data for treaties"""
        
        claims = []
        for treaty_row in treaty_df.iter_rows(named=True):
            n_claims = max(1, int(np.random.poisson(claims_per_treaty)))
            
            for _ in range(n_claims):
                claim = self._generate_claim(treaty_row)
                claims.append(claim)
        
        return pl.DataFrame(claims)
    
    def generate_loss_development_data(self, claims_df: pl.DataFrame) -> pl.DataFrame:
        """Generate loss development triangles for claims"""
        
        developments = []
        for claim_row in claims_df.iter_rows(named=True):
            claim_developments = self._generate_loss_development(claim_row)
            developments.extend(claim_developments)
        
        return pl.DataFrame(developments)
    
    def _generate_treaty_terms(self, treaty_type: TreatyType, business_line: BusinessLine) -> Dict:
        """Generate realistic treaty terms based on type and business line"""
        
        cedant = random.choice(self.cedants)
        reinsurer = random.choice(self.reinsurers)
        currency = random.choice(self.currencies)
        territory = random.choice(self.territories)
        
        # Generate dates
        inception_year = random.randint(2020, 2024)
        inception_date = datetime(inception_year, 1, 1) + timedelta(days=random.randint(0, 365))
        expiry_date = inception_date + timedelta(days=365)
        
        # Base terms
        terms = {
            "cedant": cedant,
            "reinsurer": reinsurer,
            "currency": currency,
            "territory": territory,
            "inception_date": inception_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d")
        }
        
        # Treaty-specific terms
        if treaty_type == TreatyType.QUOTA_SHARE:
            cession_rate = random.uniform(0.2, 0.8)
            premium = random.uniform(1_000_000, 50_000_000)
            terms.update({
                "premium": premium,
                "cession_rate": cession_rate,
                "commission": random.uniform(0.15, 0.35),
                "brokerage": random.uniform(0.02, 0.05),
                "retention": 1 - cession_rate,
                "limit": None,
                "minimum_premium": premium * 0.8,
                "maximum_premium": premium * 1.2,
                "profit_commission": random.uniform(0.1, 0.2),
                "loss_corridor": f"{random.randint(75, 85)}%-{random.randint(105, 115)}%",
                "aggregate_limit": None,
                "reinstatements": 0,
                "rating_method": "Original Terms",
                "experience_period": "3 years"
            })
            
        elif treaty_type == TreatyType.SURPLUS:
            lines = random.randint(2, 10)
            retention = random.uniform(100_000, 1_000_000)
            terms.update({
                "premium": random.uniform(2_000_000, 20_000_000),
                "cession_rate": None,
                "commission": random.uniform(0.18, 0.32),
                "brokerage": random.uniform(0.02, 0.05),
                "retention": retention,
                "limit": retention * lines,
                "minimum_premium": None,
                "maximum_premium": None,
                "profit_commission": random.uniform(0.08, 0.15),
                "loss_corridor": None,
                "aggregate_limit": None,
                "reinstatements": 0,
                "rating_method": "Original Terms",
                "experience_period": "5 years"
            })
            
        elif treaty_type == TreatyType.EXCESS_OF_LOSS:
            attachment = random.uniform(500_000, 5_000_000)
            limit = random.uniform(5_000_000, 50_000_000)
            terms.update({
                "premium": random.uniform(500_000, 5_000_000),
                "cession_rate": None,
                "commission": 0.0,
                "brokerage": random.uniform(0.05, 0.1),
                "retention": attachment,
                "limit": limit,
                "minimum_premium": None,
                "maximum_premium": None,
                "profit_commission": None,
                "loss_corridor": None,
                "aggregate_limit": random.uniform(10_000_000, 100_000_000),
                "reinstatements": random.randint(1, 3),
                "rating_method": "Experience Rating",
                "experience_period": "10 years"
            })
        
        elif treaty_type == TreatyType.CATASTROPHE:
            attachment = random.uniform(10_000_000, 100_000_000)
            limit = random.uniform(50_000_000, 500_000_000)
            terms.update({
                "premium": random.uniform(5_000_000, 50_000_000),
                "cession_rate": None,
                "commission": 0.0,
                "brokerage": random.uniform(0.08, 0.15),
                "retention": attachment,
                "limit": limit,
                "minimum_premium": None,
                "maximum_premium": None,
                "profit_commission": None,
                "loss_corridor": None,
                "aggregate_limit": random.uniform(100_000_000, 1_000_000_000),
                "reinstatements": random.randint(2, 5),
                "rating_method": "Cat Modeling",
                "experience_period": "20 years"
            })
        
        elif treaty_type == TreatyType.LIFE_QUOTA_SHARE:
            cession_rate = random.uniform(0.3, 0.9)
            premium = random.uniform(5_000_000, 100_000_000)
            terms.update({
                "premium": premium,
                "cession_rate": cession_rate,
                "commission": random.uniform(0.05, 0.15),
                "brokerage": random.uniform(0.01, 0.03),
                "retention": 1 - cession_rate,
                "limit": None,
                "minimum_premium": premium * 0.9,
                "maximum_premium": premium * 1.1,
                "profit_commission": random.uniform(0.05, 0.10),
                "loss_corridor": f"{random.randint(85, 95)}%-{random.randint(105, 115)}%",
                "aggregate_limit": None,
                "reinstatements": 0,
                "rating_method": "Mortality Tables",
                "experience_period": "5 years"
            })
        
        elif treaty_type == TreatyType.LIFE_COINSURANCE:
            cession_rate = random.uniform(0.5, 0.95)
            premium = random.uniform(10_000_000, 200_000_000)
            terms.update({
                "premium": premium,
                "cession_rate": cession_rate,
                "commission": random.uniform(0.02, 0.08),
                "brokerage": random.uniform(0.005, 0.02),
                "retention": 1 - cession_rate,
                "limit": None,
                "minimum_premium": premium * 0.95,
                "maximum_premium": premium * 1.05,
                "profit_commission": random.uniform(0.02, 0.05),
                "loss_corridor": f"{random.randint(90, 95)}%-{random.randint(105, 110)}%",
                "aggregate_limit": None,
                "reinstatements": 0,
                "rating_method": "Experience Rating",
                "experience_period": "10 years"
            })
        
        else:
            # Default case - use quota share structure
            cession_rate = random.uniform(0.2, 0.5)
            premium = random.uniform(1_000_000, 10_000_000)
            terms.update({
                "premium": premium,
                "cession_rate": cession_rate,
                "commission": random.uniform(0.20, 0.30),
                "brokerage": random.uniform(0.02, 0.05),
                "retention": 1 - cession_rate,
                "limit": None,
                "minimum_premium": premium * 0.8,
                "maximum_premium": premium * 1.2,
                "profit_commission": random.uniform(0.10, 0.20),
                "loss_corridor": f"{random.randint(75, 85)}%-{random.randint(105, 115)}%",
                "aggregate_limit": None,
                "reinstatements": 0,
                "rating_method": "Original Terms",
                "experience_period": "3 years"
            })
        
        # Add loss ratios based on business line
        base_loss_ratio = {
            BusinessLine.PROPERTY: 0.65,
            BusinessLine.CASUALTY: 0.72,
            BusinessLine.MOTOR: 0.68,
            BusinessLine.MARINE: 0.58,
            BusinessLine.AVIATION: 0.45,
            BusinessLine.WORKERS_COMP: 0.75,
            BusinessLine.LIFE: 0.85,
            BusinessLine.HEALTH: 0.80,
            BusinessLine.DISABILITY: 0.75
        }.get(business_line, 0.65)
        
        loss_ratio = max(0.2, min(1.5, np.random.normal(base_loss_ratio, 0.15)))
        expense_ratio = random.uniform(0.25, 0.35)
        
        terms.update({
            "loss_ratio": loss_ratio,
            "expense_ratio": expense_ratio,
            "combined_ratio": loss_ratio + expense_ratio
        })
        
        return terms
    
    def _generate_claim(self, treaty: Dict) -> Dict:
        """Generate a realistic claim for a treaty"""
        
        # Claim occurrence date within treaty period
        inception = datetime.strptime(treaty["inception_date"], "%Y-%m-%d")
        expiry = datetime.strptime(treaty["expiry_date"], "%Y-%m-%d")
        occurrence_date = inception + timedelta(
            days=random.randint(0, (expiry - inception).days)
        )
        
        # Report date (after occurrence)
        report_delay = int(np.random.exponential(30))  # 30 day average delay
        report_date = occurrence_date + timedelta(days=report_delay)
        
        # Claim severity based on business line
        business_line = BusinessLine(treaty["business_line"])
        if business_line in [BusinessLine.PROPERTY, BusinessLine.CASUALTY]:
            gross_claim = max(1000, int(np.random.lognormal(10, 2)))
        elif business_line == BusinessLine.MOTOR:
            gross_claim = max(500, int(np.random.lognormal(8, 1.5)))
        elif business_line in [BusinessLine.MARINE, BusinessLine.AVIATION]:
            gross_claim = max(10000, int(np.random.lognormal(12, 2.5)))
        else:  # Life, Health, Disability
            gross_claim = max(5000, int(np.random.lognormal(9, 1.8)))
        
        # Calculate reinsurance recovery
        retention = treaty.get("retention", 0)
        limit = treaty.get("limit")
        cession_rate = treaty.get("cession_rate")
        
        if cession_rate:  # Quota Share
            reinsurance_recovery = gross_claim * cession_rate
        elif limit:  # Surplus or XoL
            recovery = max(0, min(gross_claim - retention, limit))
            reinsurance_recovery = recovery
        else:
            reinsurance_recovery = 0
        
        net_claim = gross_claim - reinsurance_recovery
        
        return {
            "claim_id": str(uuid.uuid4())[:12],
            "treaty_id": treaty["treaty_id"],
            "occurrence_date": occurrence_date.strftime("%Y-%m-%d"),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "claim_status": random.choice(["Open", "Closed", "IBNR", "Reopened"]),
            "claim_type": random.choice(["Attritional", "Large", "Catastrophe"]),
            "cause_of_loss": random.choice([
                "Fire", "Wind", "Water", "Theft", "Collision", "Liability",
                "Medical", "Death", "Disability", "Other"
            ]),
            "territory": treaty["territory"],
            "currency": treaty["currency"],
            "gross_claim_amount": gross_claim,
            "reinsurance_recovery": reinsurance_recovery,
            "net_claim_amount": net_claim,
            "case_reserves": gross_claim * random.uniform(0.8, 1.2),
            "paid_to_date": gross_claim * random.uniform(0.0, 0.9),
            "outstanding_reserves": gross_claim * random.uniform(0.1, 0.3),
            "expense_reserves": gross_claim * random.uniform(0.05, 0.15),
            "salvage_subrogation": gross_claim * random.uniform(0.0, 0.1)
        }
    
    def _generate_loss_development(self, claim: Dict) -> List[Dict]:
        """Generate loss development pattern for a claim"""
        
        developments = []
        occurrence_date = datetime.strptime(claim["occurrence_date"], "%Y-%m-%d")
        gross_ultimate = claim["gross_claim_amount"]
        
        # Generate development pattern (12 quarters)
        development_factors = [
            0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.91, 0.94, 0.96, 0.98, 0.99, 1.0
        ]
        
        for quarter in range(12):
            valuation_date = occurrence_date + timedelta(days=quarter * 90)
            development_factor = development_factors[quarter]
            
            # Add some randomness
            factor_noise = np.random.normal(0, 0.05)
            actual_factor = max(0, min(1, development_factor + factor_noise))
            
            incurred_amount = gross_ultimate * actual_factor
            paid_amount = incurred_amount * random.uniform(0.6, 0.9)
            
            development = {
                "claim_id": claim["claim_id"],
                "treaty_id": claim["treaty_id"],
                "valuation_date": valuation_date.strftime("%Y-%m-%d"),
                "development_quarter": quarter + 1,
                "incurred_amount": incurred_amount,
                "paid_amount": paid_amount,
                "case_reserves": incurred_amount - paid_amount,
                "development_factor": actual_factor,
                "cumulative_paid_ratio": paid_amount / gross_ultimate,
                "payment_pattern": random.choice(["Fast", "Medium", "Slow"]),
                "reserve_adequacy": random.uniform(0.85, 1.15)
            }
            developments.append(development)
        
        return developments
    
    def generate_catastrophe_events(self, n_events: int = 20) -> pl.DataFrame:
        """Generate catastrophe events affecting multiple treaties"""
        
        events = []
        event_types = [
            "Hurricane", "Earthquake", "Flood", "Wildfire", "Tornado",
            "Hail Storm", "Winter Storm", "Tsunami", "Volcanic Eruption"
        ]
        
        for _ in range(n_events):
            event_date = datetime(2020, 1, 1) + timedelta(
                days=random.randint(0, 4 * 365)
            )
            
            event = {
                "event_id": f"CAT{random.randint(2020, 2024)}{random.randint(1, 999):03d}",
                "event_name": f"{random.choice(event_types)} {event_date.strftime('%B %Y')}",
                "event_type": random.choice(event_types),
                "occurrence_date": event_date.strftime("%Y-%m-%d"),
                "location": random.choice(self.territories),
                "magnitude": random.uniform(1, 10),
                "affected_lines": random.choice([
                    "Property", "Property,Motor", "Property,Marine", "All Lines"
                ]),
                "industry_loss": random.uniform(100_000_000, 50_000_000_000),
                "modeled_loss": random.uniform(50_000_000, 20_000_000_000),
                "actual_loss": None,  # Would be filled in as claims develop
                "loss_development_pattern": random.choice(["Fast", "Medium", "Slow"]),
                "geographic_spread": random.choice(["Localized", "Regional", "National"]),
                "loss_driver": random.choice(["Wind", "Water", "Ground_Up", "Fire"])
            }
            events.append(event)
        
        return pl.DataFrame(events)
    
    def generate_portfolio_data(self, n_portfolios: int = 50) -> pl.DataFrame:
        """Generate reinsurance portfolio data"""
        
        portfolios = []
        for _ in range(n_portfolios):
            portfolio = {
                "portfolio_id": str(uuid.uuid4())[:8],
                "portfolio_name": f"Portfolio {random.choice(['A', 'B', 'C', 'D'])}-{random.randint(1, 99)}",
                "business_line": random.choice(list(BusinessLine)).value,
                "territory": random.choice(self.territories),
                "currency": random.choice(self.currencies),
                "total_sum_insured": random.uniform(100_000_000, 10_000_000_000),
                "number_of_risks": random.randint(100, 10000),
                "average_sum_insured": random.uniform(50_000, 2_000_000),
                "premium_rate": random.uniform(0.001, 0.05),
                "deductible_average": random.uniform(1000, 50000),
                "policy_limit_average": random.uniform(100_000, 5_000_000),
                "geographic_concentration": random.uniform(0.1, 0.8),
                "industry_concentration": random.uniform(0.1, 0.6),
                "historical_loss_ratio": random.uniform(0.4, 1.2),
                "volatility": random.uniform(0.1, 0.4),
                "correlation_factor": random.uniform(-0.2, 0.8),
                "cat_exposure": random.choice([True, False]),
                "pricing_adequacy": random.uniform(0.8, 1.3)
            }
            portfolios.append(portfolio)
        
        return pl.DataFrame(portfolios)