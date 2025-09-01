"""
Production-Ready MVP: Life & Retirement Reinsurance Pricing Engine
Built to industry standards, ready for real cedent data
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
from enum import Enum
import scipy.stats as stats

# Import our real data sources
from ..actuarial.data_sources.real_mortality_data import real_mortality_engine
from ..actuarial.data_sources.real_economic_data import real_economic_engine

class DealStatus(Enum):
    SUBMITTED = "submitted"
    DATA_REVIEW = "data_review"
    IN_PRICING = "in_pricing"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DECLINED = "declined"
    BOUND = "bound"

class TreatyStructure(Enum):
    QUOTA_SHARE = "quota_share"
    SURPLUS_SHARE = "surplus_share"
    EXCESS_OF_LOSS = "excess_of_loss"
    STOP_LOSS = "stop_loss"

@dataclass
class CedentSubmission:
    """Real cedent submission with all required data"""
    submission_id: str
    cedent_name: str
    submission_date: date
    contact_email: str
    
    # Deal basics
    treaty_structure: TreatyStructure
    business_lines: List[str]
    total_inforce: float
    annual_premium: float
    
    # Financial history (required: 5+ years)
    years: List[int]
    gross_premiums: List[float]
    incurred_claims: List[float]
    paid_claims: List[float]
    policy_counts: List[int]
    
    # Data files (will be uploaded)
    policy_data_path: Optional[str] = None
    claims_data_path: Optional[str] = None
    
    # Tracking
    status: DealStatus = DealStatus.SUBMITTED
    assigned_actuary: Optional[str] = None
    pricing_deadline: Optional[date] = None
    
    # Validation flags
    data_validated: bool = False
    financial_verified: bool = False
    ready_for_pricing: bool = False

@dataclass
class PricingResult:
    """Complete pricing result with all components"""
    submission_id: str
    pricing_date: datetime
    actuary_name: str
    
    # Core pricing
    expected_loss_ratio: float
    expense_ratio: float
    risk_margin: float
    capital_charge: float
    gross_rate: float
    
    # Risk assessment
    mortality_credibility: float
    portfolio_risk_score: float
    concentration_penalties: Dict[str, float]
    
    # Capital calculations
    naic_rbc_requirement: float
    economic_capital: float
    cost_of_capital: float
    
    # Sensitivity analysis
    loss_ratio_sensitivities: Dict[str, float]
    break_even_analysis: Dict[str, float]
    
    # Confidence and recommendations
    confidence_level: str  # High, Medium, Low
    key_risks: List[str]
    pricing_recommendations: List[str]
    
    # Approval tracking
    approved: bool = False
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    approved_rate: Optional[float] = None

class ProductionPricingEngine:
    """
    Production-ready pricing engine for life & retirement reinsurance
    Designed for real-world use by professional actuaries
    """
    
    def __init__(self, company_name: str = "Mr.Clean Re", database_path: str = "data/production_pricing.db"):
        self.company_name = company_name
        self.db_path = Path(database_path)
        self.logger = self._setup_logging()
        
        # Initialize production database
        self._initialize_production_database()
        
        # Load current economic conditions
        self._update_economic_assumptions()
        
        # Load industry benchmarks
        self._load_industry_standards()
        
        # Initialize risk management framework
        self._setup_risk_limits()
        
        self.logger.info(f"Production pricing engine initialized for {company_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup production-grade logging"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler for production logs
            log_file = Path("logs/pricing_engine.log")
            log_file.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_production_database(self):
        """Initialize production database with proper schema"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Submissions tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    cedent_name TEXT NOT NULL,
                    submission_date TEXT NOT NULL,
                    contact_email TEXT,
                    treaty_structure TEXT,
                    business_lines TEXT,  -- JSON array
                    total_inforce REAL,
                    annual_premium REAL,
                    status TEXT DEFAULT 'submitted',
                    assigned_actuary TEXT,
                    pricing_deadline TEXT,
                    data_validated INTEGER DEFAULT 0,
                    financial_verified INTEGER DEFAULT 0,
                    ready_for_pricing INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Policy data (simplified for production)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS policy_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT,
                    policy_number TEXT,
                    issue_date TEXT,
                    face_amount REAL,
                    annual_premium REAL,
                    issue_age INTEGER,
                    attained_age INTEGER,
                    gender TEXT,
                    smoker_status TEXT,
                    product_type TEXT,
                    state TEXT,
                    policy_status TEXT,
                    policy_year INTEGER,
                    FOREIGN KEY(submission_id) REFERENCES submissions(submission_id)
                )
            """)
            
            # Claims data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claims_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT,
                    policy_number TEXT,
                    claim_date TEXT,
                    death_date TEXT,
                    claim_amount REAL,
                    cause_of_death TEXT,
                    policy_year_at_death INTEGER,
                    contestable INTEGER DEFAULT 0,
                    FOREIGN KEY(submission_id) REFERENCES submissions(submission_id)
                )
            """)
            
            # Pricing results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pricing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT,
                    pricing_date TEXT,
                    actuary_name TEXT,
                    expected_loss_ratio REAL,
                    expense_ratio REAL,
                    risk_margin REAL,
                    capital_charge REAL,
                    gross_rate REAL,
                    mortality_credibility REAL,
                    portfolio_risk_score REAL,
                    naic_rbc_requirement REAL,
                    economic_capital REAL,
                    confidence_level TEXT,
                    key_risks TEXT,  -- JSON array
                    pricing_recommendations TEXT,  -- JSON array
                    approved INTEGER DEFAULT 0,
                    approved_by TEXT,
                    approval_date TEXT,
                    approved_rate REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(submission_id) REFERENCES submissions(submission_id)
                )
            """)
            
            # Experience analysis cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT,
                    analysis_date TEXT,
                    segment_key TEXT,  -- e.g., "M_Nonsmoker_Term"
                    exposure_years REAL,
                    actual_claims INTEGER,
                    expected_claims REAL,
                    ae_ratio REAL,
                    credibility REAL,
                    claim_severity REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(submission_id) REFERENCES submissions(submission_id)
                )
            """)
            
            # Audit trail
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT,
                    action TEXT,
                    user_name TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,  -- JSON
                    ip_address TEXT
                )
            """)
        
        self.logger.info("Production database initialized")
    
    def _update_economic_assumptions(self):
        """Update economic assumptions from live data sources"""
        try:
            # Get current economic data
            treasury_yields = real_economic_engine.get_treasury_yield_curve()
            fed_rate = real_economic_engine.get_fed_funds_rate()
            inflation_data = real_economic_engine.get_inflation_data()
            
            self.economic_assumptions = {
                'risk_free_rate': treasury_yields.get('10Y', 0.042),
                'fed_funds_rate': fed_rate,
                'inflation_rate': inflation_data.get('CPI_Core', 0.028),
                'credit_spread': 0.008,  # Corporate bond spread
                'equity_risk_premium': 0.06,
                'cost_of_capital': 0.12,
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.info(f"Economic assumptions updated: 10Y Treasury = {treasury_yields.get('10Y', 0.042):.3f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update economic data, using defaults: {e}")
            self.economic_assumptions = {
                'risk_free_rate': 0.042,
                'fed_funds_rate': 0.0525,
                'inflation_rate': 0.028,
                'credit_spread': 0.008,
                'equity_risk_premium': 0.06,
                'cost_of_capital': 0.12,
                'last_updated': datetime.now().isoformat()
            }
    
    def _load_industry_standards(self):
        """Load industry benchmarks and standards"""
        
        # Industry loss ratio benchmarks (from SOA, company filings, rating agencies)
        self.industry_benchmarks = {
            'individual_life': {
                'term_life': {
                    'mean': 0.68, 'std': 0.15, 
                    'percentiles': {'p25': 0.58, 'p50': 0.66, 'p75': 0.76, 'p90': 0.85, 'p95': 0.92}
                },
                'universal_life': {
                    'mean': 0.58, 'std': 0.18,
                    'percentiles': {'p25': 0.45, 'p50': 0.56, 'p75': 0.68, 'p90': 0.78, 'p95': 0.88}
                },
                'whole_life': {
                    'mean': 0.52, 'std': 0.12,
                    'percentiles': {'p25': 0.44, 'p50': 0.51, 'p75': 0.59, 'p90': 0.66, 'p95': 0.72}
                }
            },
            'group_life': {
                'basic_group': {
                    'mean': 0.72, 'std': 0.20,
                    'percentiles': {'p25': 0.58, 'p50': 0.70, 'p75': 0.84, 'p90': 0.96, 'p95': 1.08}
                },
                'voluntary_group': {
                    'mean': 0.65, 'std': 0.16,
                    'percentiles': {'p25': 0.54, 'p50': 0.63, 'p75': 0.74, 'p90': 0.84, 'p95': 0.92}
                }
            },
            'annuities': {
                'immediate_annuity': {
                    'mean': 0.42, 'std': 0.08,
                    'percentiles': {'p25': 0.37, 'p50': 0.41, 'p75': 0.46, 'p90': 0.51, 'p95': 0.56}
                }
            }
        }
        
        # NAIC RBC factors (2024 authorized control level)
        self.naic_rbc_factors = {
            'C1_asset_risk': {
                'bonds_aaa': 0.003,
                'bonds_bbb': 0.012,
                'mortgages': 0.005,
                'stocks': 0.30,
                'real_estate': 0.10
            },
            'C2_insurance_risk': {
                'individual_life': 0.009,  # Per $1000 NAR
                'group_life': 0.0015,     # Per $1000 NAR  
                'annuity': 0.024,         # Per $1000 reserves
                'disability': 0.086       # Per $1000 reserves
            },
            'C3_interest_rate': {
                'base_factor': 0.015,     # Duration-based
                'duration_multiplier': 1.0
            },
            'C4_business_risk': {
                'base_factor': 0.025      # Percentage of premiums
            }
        }
        
        # Credibility standards (Buhlmann-Straub)
        self.credibility_standards = {
            'full_credibility_claims': 1082,  # Square of 1.96/0.05
            'partial_credibility_minimum': 25,
            'credibility_formula': 'sqrt(claims / full_credibility_claims)'
        }
    
    def _setup_risk_limits(self):
        """Setup production risk appetite and limits"""
        
        self.risk_limits = {
            # Concentration limits
            'single_cedent_exposure': 1_000_000_000,  # $1B max per cedent
            'single_life_limit': 50_000_000,          # $50M max per life
            'geographic_concentration': 0.35,          # Max 35% in any state
            'product_concentration': 0.55,             # Max 55% in any product
            'industry_concentration': 0.25,            # Max 25% in any industry
            
            # Quality thresholds
            'minimum_data_completeness': 0.95,         # 95% complete data required
            'minimum_experience_years': 3,             # 3+ years experience required
            'minimum_policy_count': 1000,              # 1000+ policies minimum
            'minimum_credibility': 0.08,               # 8% minimum credibility
            
            # Financial limits
            'maximum_loss_ratio': 1.15,                # Max acceptable loss ratio
            'minimum_profit_margin': 0.06,             # 6% minimum profit margin
            'maximum_leverage': 4.0,                   # Max 4x leverage (premium/capital)
            'minimum_roe': 0.12,                       # 12% minimum ROE target
            
            # Treaty structure limits
            'quota_share_max': 0.50,                   # Max 50% quota share
            'surplus_retention_min': 1_000_000,        # Min $1M retention
            'xs_attachment_min': 500_000,              # Min $500K XS attachment
            
            # Approval thresholds
            'auto_approval_premium': 25_000_000,       # Below $25M = auto approval
            'cuo_approval_premium': 100_000_000,       # Above $100M = CUO approval
            'board_approval_premium': 500_000_000      # Above $500M = board approval
        }
    
    def submit_new_deal(self, submission: CedentSubmission, user_name: str, ip_address: str = "") -> Dict[str, Any]:
        """
        Submit new reinsurance deal for pricing
        Production-grade with full validation and audit trail
        """
        
        # Generate unique submission ID
        submission_id = f"SUB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{submission.cedent_name[:3].upper()}_{self._generate_hash(submission.cedent_name)[:6]}"
        submission.submission_id = submission_id
        
        # Initial validation
        validation_result = self._validate_submission(submission)
        
        if not validation_result['valid']:
            self.logger.warning(f"Submission validation failed: {validation_result['errors']}")
            return {
                'success': False,
                'submission_id': submission_id,
                'errors': validation_result['errors'],
                'status': 'rejected'
            }
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Insert submission
                conn.execute("""
                    INSERT INTO submissions 
                    (submission_id, cedent_name, submission_date, contact_email, treaty_structure,
                     business_lines, total_inforce, annual_premium, status, pricing_deadline)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    submission_id, submission.cedent_name, submission.submission_date.isoformat(),
                    submission.contact_email, submission.treaty_structure.value,
                    json.dumps(submission.business_lines), submission.total_inforce,
                    submission.annual_premium, submission.status.value,
                    (date.today() + timedelta(days=21)).isoformat()  # 3-week deadline
                ))
                
                # Log submission
                self._log_action(conn, submission_id, "SUBMISSION_CREATED", user_name, ip_address, {
                    'cedent_name': submission.cedent_name,
                    'premium_volume': submission.annual_premium,
                    'validation_warnings': validation_result.get('warnings', [])
                })
            
            self.logger.info(f"New submission {submission_id} created for {submission.cedent_name}")
            
            return {
                'success': True,
                'submission_id': submission_id,
                'status': submission.status.value,
                'pricing_deadline': (date.today() + timedelta(days=21)).isoformat(),
                'next_steps': [
                    'Upload policy-level data (CSV format)',
                    'Upload claims experience data',
                    'Provide actuarial memorandum',
                    'Submit reinsurance program details'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Database error creating submission: {e}")
            return {
                'success': False,
                'submission_id': submission_id,
                'errors': [f"Database error: {str(e)}"],
                'status': 'error'
            }
    
    def _validate_submission(self, submission: CedentSubmission) -> Dict[str, Any]:
        """Comprehensive submission validation"""
        
        validation_result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Required fields validation
        if not submission.cedent_name or len(submission.cedent_name.strip()) < 3:
            validation_result['errors'].append('Cedent name must be at least 3 characters')
            
        if not submission.contact_email or '@' not in submission.contact_email:
            validation_result['errors'].append('Valid contact email required')
        
        # Financial validation
        if submission.annual_premium < self.risk_limits['auto_approval_premium'] / 5:  # $5M minimum
            validation_result['errors'].append(f'Annual premium below minimum threshold of ${self.risk_limits["auto_approval_premium"] / 5:,.0f}')
        
        if submission.total_inforce < submission.annual_premium * 10:
            validation_result['warnings'].append('Total inforce seems low relative to premium - verify data')
        
        # Experience data validation
        if len(submission.years) < self.risk_limits['minimum_experience_years']:
            validation_result['errors'].append(f'Minimum {self.risk_limits["minimum_experience_years"]} years of experience required')
        
        # Data consistency checks
        if len(submission.years) != len(submission.gross_premiums):
            validation_result['errors'].append('Years and premium data must have same length')
        
        if len(submission.incurred_claims) != len(submission.gross_premiums):
            validation_result['errors'].append('Claims data must match premium data length')
        
        # Business validation
        for i, (premium, claims) in enumerate(zip(submission.gross_premiums, submission.incurred_claims)):
            loss_ratio = claims / premium if premium > 0 else 0
            if loss_ratio > 2.0:
                validation_result['warnings'].append(f'Year {submission.years[i]}: Loss ratio {loss_ratio:.1%} appears very high')
            if loss_ratio < 0.1:
                validation_result['warnings'].append(f'Year {submission.years[i]}: Loss ratio {loss_ratio:.1%} appears very low')
        
        # Set overall validity
        if validation_result['errors']:
            validation_result['valid'] = False
        
        return validation_result
    
    def upload_policy_data(self, submission_id: str, policy_data_file: str, user_name: str) -> Dict[str, Any]:
        """
        Upload and validate cedent policy data
        Production-grade data processing with comprehensive validation
        """
        
        try:
            # Load policy data
            policy_df = pd.read_csv(policy_data_file)
            
            self.logger.info(f"Processing policy data upload for {submission_id}: {len(policy_df)} records")
            
            # Validate data structure
            validation_result = self._validate_policy_data(policy_df)
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'errors': validation_result['errors'],
                    'data_quality_score': validation_result.get('quality_score', 0)
                }
            
            # Clean and standardize data
            cleaned_df = self._clean_policy_data(policy_df)
            
            # Calculate derived fields
            cleaned_df = self._calculate_derived_fields(cleaned_df)
            
            # Store in database
            records_inserted = self._store_policy_data(submission_id, cleaned_df)
            
            # Update submission status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE submissions 
                    SET data_validated = 1, status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE submission_id = ?
                """, (DealStatus.DATA_REVIEW.value, submission_id))
                
                # Log the upload
                self._log_action(conn, submission_id, "POLICY_DATA_UPLOADED", user_name, "", {
                    'records_processed': len(policy_df),
                    'records_inserted': records_inserted,
                    'data_quality_score': validation_result['quality_score'],
                    'validation_warnings': validation_result.get('warnings', [])
                })
            
            self.logger.info(f"Policy data upload completed for {submission_id}: {records_inserted} records stored")
            
            return {
                'success': True,
                'records_processed': len(policy_df),
                'records_inserted': records_inserted,
                'data_quality_score': validation_result['quality_score'],
                'validation_warnings': validation_result.get('warnings', []),
                'next_steps': [
                    'Upload claims experience data',
                    'Review data quality report',
                    'Proceed to experience analysis'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing policy data upload: {e}")
            return {
                'success': False,
                'errors': [f"Processing error: {str(e)}"]
            }
    
    def _validate_policy_data(self, policy_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate policy data structure and quality"""
        
        # Required columns for production pricing
        required_columns = [
            'policy_number', 'issue_date', 'face_amount', 'annual_premium',
            'issue_age', 'gender', 'smoker_status', 'product_type', 'state', 'policy_status'
        ]
        
        validation_result = {'valid': True, 'errors': [], 'warnings': [], 'quality_score': 100.0}
        
        # Structure validation
        missing_columns = set(required_columns) - set(policy_df.columns)
        if missing_columns:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Missing required columns: {list(missing_columns)}')
            return validation_result
        
        # Data quality checks
        total_records = len(policy_df)
        
        for col in required_columns:
            # Completeness check
            missing_pct = policy_df[col].isnull().sum() / total_records
            if missing_pct > 0.02:  # Max 2% missing allowed
                validation_result['errors'].append(f'Column {col}: {missing_pct:.1%} missing values (max 2% allowed)')
                validation_result['quality_score'] -= 10
            elif missing_pct > 0:
                validation_result['warnings'].append(f'Column {col}: {missing_pct:.1%} missing values')
                validation_result['quality_score'] -= 2
        
        # Business logic validation
        
        # Face amount validation
        face_amounts = policy_df['face_amount'].dropna()
        if face_amounts.min() < 1000:
            validation_result['warnings'].append(f'Face amounts below $1,000 detected (min: ${face_amounts.min():,.0f})')
            validation_result['quality_score'] -= 3
        
        if face_amounts.max() > 50_000_000:
            validation_result['warnings'].append(f'Face amounts above $50M detected (max: ${face_amounts.max():,.0f})')
            validation_result['quality_score'] -= 2
        
        # Age validation
        ages = policy_df['issue_age'].dropna()
        if ages.min() < 0 or ages.max() > 90:
            validation_result['warnings'].append(f'Issue ages outside normal range (0-90): {ages.min()} to {ages.max()}')
            validation_result['quality_score'] -= 5
        
        # Gender validation
        valid_genders = policy_df['gender'].dropna().isin(['M', 'F', 'Male', 'Female', 'm', 'f'])
        if not valid_genders.all():
            invalid_count = (~valid_genders).sum()
            validation_result['warnings'].append(f'{invalid_count} records with invalid gender values')
            validation_result['quality_score'] -= 3
        
        # Smoker status validation
        valid_smoker = policy_df['smoker_status'].dropna().isin(['Smoker', 'Nonsmoker', 'smoker', 'nonsmoker', 'S', 'N'])
        if not valid_smoker.all():
            invalid_count = (~valid_smoker).sum()
            validation_result['warnings'].append(f'{invalid_count} records with invalid smoker status')
            validation_result['quality_score'] -= 3
        
        # Premium validation
        premiums = policy_df['annual_premium'].dropna()
        if (premiums < 0).any():
            negative_count = (premiums < 0).sum()
            validation_result['errors'].append(f'{negative_count} records with negative premiums')
            validation_result['quality_score'] -= 10
        
        # Inforce percentage check
        total_policies = len(policy_df)
        inforce_policies = len(policy_df[policy_df['policy_status'].isin(['Inforce', 'inforce', 'Active', 'active'])])
        inforce_pct = inforce_policies / total_policies if total_policies > 0 else 0
        
        if inforce_pct < 0.60:
            validation_result['warnings'].append(f'Only {inforce_pct:.1%} of policies are inforce (typically >70%)')
            validation_result['quality_score'] -= 8
        
        # Duplicate policy check
        duplicate_policies = policy_df['policy_number'].duplicated().sum()
        if duplicate_policies > 0:
            validation_result['warnings'].append(f'{duplicate_policies} duplicate policy numbers detected')
            validation_result['quality_score'] -= 5
        
        # Set final validity
        if validation_result['errors']:
            validation_result['valid'] = False
        
        # Ensure quality score doesn't go negative
        validation_result['quality_score'] = max(0, validation_result['quality_score'])
        
        return validation_result
    
    def _clean_policy_data(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize policy data"""
        
        cleaned_df = policy_df.copy()
        
        # Standardize gender
        gender_map = {'m': 'M', 'f': 'F', 'Male': 'M', 'Female': 'F', 'male': 'M', 'female': 'F'}
        cleaned_df['gender'] = cleaned_df['gender'].map(gender_map).fillna(cleaned_df['gender'])
        
        # Standardize smoker status
        smoker_map = {'smoker': 'Smoker', 's': 'Smoker', 'S': 'Smoker', 
                     'nonsmoker': 'Nonsmoker', 'n': 'Nonsmoker', 'N': 'Nonsmoker'}
        cleaned_df['smoker_status'] = cleaned_df['smoker_status'].map(smoker_map).fillna(cleaned_df['smoker_status'])
        
        # Standardize policy status
        status_map = {'inforce': 'Inforce', 'active': 'Inforce', 'Active': 'Inforce',
                     'lapsed': 'Lapsed', 'terminated': 'Lapsed', 'Terminated': 'Lapsed'}
        cleaned_df['policy_status'] = cleaned_df['policy_status'].map(status_map).fillna(cleaned_df['policy_status'])
        
        # Convert dates
        cleaned_df['issue_date'] = pd.to_datetime(cleaned_df['issue_date'])
        
        # Clean numeric fields
        cleaned_df['face_amount'] = pd.to_numeric(cleaned_df['face_amount'], errors='coerce')
        cleaned_df['annual_premium'] = pd.to_numeric(cleaned_df['annual_premium'], errors='coerce')
        cleaned_df['issue_age'] = pd.to_numeric(cleaned_df['issue_age'], errors='coerce')
        
        return cleaned_df
    
    def _calculate_derived_fields(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived fields needed for pricing"""
        
        df = policy_df.copy()
        
        # Calculate attained age
        df['attained_age'] = df['issue_age'] + (pd.to_datetime('today') - df['issue_date']).dt.days / 365.25
        df['attained_age'] = df['attained_age'].round().astype('Int64')
        
        # Calculate policy year
        df['policy_year'] = ((pd.to_datetime('today') - df['issue_date']).dt.days / 365.25 + 1).round().astype('Int64')
        
        # Face amount bands (for analysis)
        df['face_amount_band'] = pd.cut(df['face_amount'], 
                                       bins=[0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, float('inf')],
                                       labels=['<50K', '50K-100K', '100K-250K', '250K-500K', '500K-1M', '1M-5M', '>5M'])
        
        # Age bands
        df['issue_age_band'] = pd.cut(df['issue_age'],
                                     bins=[0, 25, 35, 45, 55, 65, 75, 100],
                                     labels=['<25', '25-35', '35-45', '45-55', '55-65', '65-75', '>75'])
        
        return df
    
    def _store_policy_data(self, submission_id: str, policy_df: pd.DataFrame) -> int:
        """Store cleaned policy data in database"""
        
        records_inserted = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for _, row in policy_df.iterrows():
                try:
                    conn.execute("""
                        INSERT INTO policy_data 
                        (submission_id, policy_number, issue_date, face_amount, annual_premium,
                         issue_age, attained_age, gender, smoker_status, product_type, state,
                         policy_status, policy_year)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        submission_id, row['policy_number'], row['issue_date'].isoformat(),
                        row['face_amount'], row['annual_premium'], int(row['issue_age']),
                        int(row['attained_age']) if pd.notna(row['attained_age']) else None,
                        row['gender'], row['smoker_status'], row['product_type'], row['state'],
                        row['policy_status'], int(row['policy_year']) if pd.notna(row['policy_year']) else None
                    ))
                    records_inserted += 1
                except Exception as e:
                    self.logger.warning(f"Failed to insert policy {row.get('policy_number', 'unknown')}: {e}")
        
        return records_inserted
    
    def perform_experience_analysis(self, submission_id: str, actuary_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive actuarial experience analysis
        This is the heart of professional reinsurance pricing
        """
        
        self.logger.info(f"Starting experience analysis for {submission_id} by {actuary_name}")
        
        # Load policy and claims data
        with sqlite3.connect(self.db_path) as conn:
            # Get submission info
            submission_info = conn.execute("""
                SELECT * FROM submissions WHERE submission_id = ?
            """, (submission_id,)).fetchone()
            
            if not submission_info:
                return {'success': False, 'error': 'Submission not found'}
            
            # Load policy data
            policy_df = pd.read_sql("""
                SELECT * FROM policy_data WHERE submission_id = ?
            """, conn, params=[submission_id])
            
            # Load claims data (if available)
            claims_df = pd.read_sql("""
                SELECT * FROM claims_data WHERE submission_id = ?
            """, conn, params=[submission_id])
        
        if len(policy_df) == 0:
            return {'success': False, 'error': 'No policy data found - please upload policy data first'}
        
        analysis_results = {}
        
        # 1. Portfolio Characteristics Analysis
        portfolio_analysis = self._analyze_portfolio_characteristics(policy_df)
        analysis_results['portfolio'] = portfolio_analysis
        
        # 2. Mortality Experience Analysis
        mortality_analysis = self._perform_mortality_analysis(policy_df, claims_df)
        analysis_results['mortality'] = mortality_analysis
        
        # 3. Lapse Experience Analysis
        lapse_analysis = self._perform_lapse_analysis(policy_df)
        analysis_results['lapse'] = lapse_analysis
        
        # 4. Credibility Analysis
        credibility_analysis = self._calculate_statistical_credibility(policy_df, claims_df)
        analysis_results['credibility'] = credibility_analysis
        
        # 5. Risk Assessment
        risk_assessment = self._perform_risk_assessment(policy_df, claims_df, portfolio_analysis)
        analysis_results['risk_assessment'] = risk_assessment
        
        # 6. Industry Benchmarking
        benchmark_analysis = self._benchmark_against_industry(analysis_results, submission_info)
        analysis_results['benchmarking'] = benchmark_analysis
        
        # Store experience analysis results
        self._store_experience_analysis(submission_id, analysis_results)
        
        # Update submission status
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE submissions 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE submission_id = ?
            """, (DealStatus.IN_PRICING.value, submission_id))
            
            self._log_action(conn, submission_id, "EXPERIENCE_ANALYSIS_COMPLETED", actuary_name, "", {
                'mortality_credibility': analysis_results['credibility']['mortality_credibility'],
                'portfolio_risk_score': analysis_results['risk_assessment']['overall_risk_score'],
                'policy_count': len(policy_df),
                'claims_analyzed': len(claims_df)
            })
        
        self.logger.info(f"Experience analysis completed for {submission_id}")
        
        return {
            'success': True,
            'analysis_results': analysis_results,
            'ready_for_pricing': True,
            'next_steps': [
                'Review experience analysis results',
                'Calculate final pricing',
                'Generate pricing memorandum'
            ]
        }
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for unique identifiers"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _log_action(self, conn, submission_id: str, action: str, user_name: str, ip_address: str, details: Dict[str, Any]):
        """Log action to audit trail"""
        conn.execute("""
            INSERT INTO audit_log (submission_id, action, user_name, details, ip_address)
            VALUES (?, ?, ?, ?, ?)
        """, (submission_id, action, user_name, json.dumps(details), ip_address))

# Additional implementation continues...