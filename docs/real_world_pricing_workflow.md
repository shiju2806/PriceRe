# Real-World Life & Retirement Reinsurance Pricing Workflow

## Phase 1: Deal Origination & Initial Assessment (Week 1-2)

### 1.1 Business Development
- **Cedent Submission**: RFP with basic portfolio information
- **Initial Screening**: Size, geography, business lines, renewal vs. new business
- **Market Intelligence**: Competitive landscape, incumbent reinsurer performance
- **Decision Gate**: Go/No-Go based on strategic fit and capacity

### 1.2 Information Request
**Standard Data Package Required:**
- 5+ years of audited financial statements
- Monthly/quarterly loss development triangles
- Policy-level data files (encrypted, limited to statistical summary)
- Underwriting guidelines and recent changes
- Claims handling procedures
- Reinsurance program structure
- Regulatory capital calculations

### 1.3 Preliminary Assessment
- **High-Level Pricing**: Back-of-envelope using industry benchmarks
- **Risk Appetite Check**: Against company guidelines and limits
- **Resource Allocation**: Assign pricing actuary, underwriter, and legal support

## Phase 2: Deep Data Analysis (Week 3-4)

### 2.1 Data Quality Review
**Policy Data Analysis:**
```
Required Fields (Individual Life):
- Policy Number, Issue Date, Face Amount, Premium
- Insured: Age, Gender, Smoker Status, Health Class
- Geographic: State, County, Urban/Rural
- Product: Product Code, Premium Type, Underwriting Class
- Current Status: Inforce, Claims, Lapses, Surrenders
```

**Experience Analysis:**
```
Claims Experience (5+ years):
- Claim Date, Cause of Death, Claim Amount
- Policy Duration, Contestability Period
- Claims by Age, Gender, Smoker Status, Geography
- Large Claim Analysis (>$1M)
- Claim Frequency and Severity Trends
```

### 2.2 Actuarial Analysis
**Mortality Analysis:**
- Actual vs Expected (A/E) ratios by demographic segments
- Credibility analysis using Buhlmann-Straub method
- Mortality improvement projections using SOA MP-2021
- Anti-selection analysis for recent vintages

**Lapse Analysis:**
- Lapse rates by policy year, premium mode, face amount
- Economic sensitivity (interest rate impact)
- Shock lapse scenarios for universal life products

**Profitability Analysis:**
- Embedded value calculations
- Profit emergence patterns
- Sensitivity testing (mortality +/- 10%, lapses +/- 20%)

### 2.3 Risk Assessment
**Concentration Limits:**
- Single life limits (typically $25M-$100M retention)
- Geographic concentration (max 30-40% in any state)
- Product concentration (avoid >60% term life)
- Industry/occupation concentration analysis

## Phase 3: Pricing Model Development (Week 5-6)

### 3.1 Base Pricing
**Expected Mortality Costs:**
```python
# Simplified calculation structure
def calculate_mortality_cost(policy_data):
    base_mortality = soa_cso_2017_mortality_table(age, gender, smoker)
    underwriting_adjustment = cedent_ae_ratio * credibility + (1-credibility) * industry_benchmark
    mortality_improvement = mp2021_improvement_factors(projection_years)
    expected_claims = face_amount * base_mortality * underwriting_adjustment * mortality_improvement
    return expected_claims
```

**Expense Loadings:**
- Acquisition costs: Commission, underwriting, issue expenses
- Maintenance costs: Policy administration, claims processing
- Investment expenses: Asset management fees
- Overhead allocation: Corporate expenses, taxes

### 3.2 Risk Margins
**Capital Requirements (NAIC RBC):**
- C1 (Asset Risk): Credit risk, interest rate risk
- C2 (Insurance Risk): Mortality risk, morbidity risk  
- C3 (Interest Rate Risk): Asset-liability mismatch
- C4 (Business Risk): General business risk

**Cost of Capital:**
- Target ROE: 12-15% for life reinsurance
- Risk-free rate + risk premium approach
- Economic capital vs regulatory capital

### 3.3 Treaty Structure Optimization
**Retention Analysis:**
- Optimal retention to balance profit vs volatility
- Diversification benefits across business lines
- Regulatory constraints (affiliated reinsurance limits)

## Phase 4: Commercial Negotiation (Week 7-8)

### 4.1 Proposal Development
**Technical Proposal:**
- Pricing summary with key assumptions
- Experience analysis and benchmarking
- Risk assessment and mitigation strategies
- Regulatory compliance confirmation

**Commercial Terms:**
- Reinsurance premium rates by product/age bands
- Commission structure and overrides
- Profit sharing arrangements (50/50 split at 75% loss ratio)
- Experience rating provisions
- Termination and commutation clauses

### 4.2 Due Diligence
**Financial Review:**
- AM Best rating and outlook
- Statutory capital ratios
- Investment portfolio quality
- Management assessment

**Operational Review:**
- Claims paying history
- System capabilities for data exchange
- Regulatory compliance record
- Key person risk assessment

## Phase 5: Final Pricing & Approval (Week 9-10)

### 5.1 Pricing Committee Review
**Required Attendees:**
- Chief Actuary, Chief Underwriter, CRO
- Business line heads
- Finance (CFO or delegate)
- Legal (for contract terms)

**Decision Criteria:**
- Expected profitability vs hurdle rates
- Risk-adjusted returns and capital efficiency
- Strategic value (client relationship, market position)
- Competitive dynamics and market share

### 5.2 Final Terms
**Documentation:**
- Reinsurance agreement (50-100 pages)
- Schedule of business (specific coverage terms)
- Claims procedures manual
- Reporting requirements specification
- Data exchange protocols

### 5.3 Implementation Planning
**Systems Setup:**
- Policy data feed establishment
- Claims reporting procedures
- Financial reporting (quarterly statements)
- Experience monitoring dashboards

## Key Success Factors

### Data Quality Requirements
1. **Completeness**: <2% missing critical fields
2. **Accuracy**: Reconciliation to financial statements
3. **Timeliness**: Monthly reporting within 30 days
4. **Consistency**: Standardized formats and definitions

### Pricing Model Validation
1. **Back-testing**: Model performance on historical data
2. **Sensitivity Analysis**: Key variable stress testing
3. **Peer Review**: Independent actuarial review
4. **Regulatory Approval**: Meet all jurisdictional requirements

### Risk Management
1. **Monitoring Systems**: Real-time portfolio dashboards
2. **Alert Mechanisms**: Early warning indicators
3. **Remedial Actions**: Predefined response procedures
4. **Regular Reviews**: Quarterly experience updates

## Technology Requirements

### Core Systems
- **Policy Administration**: For treaty management
- **Claims System**: Automated claims processing
- **Financial Reporting**: Statutory and GAAP reporting
- **Risk Management**: Concentration and limit monitoring

### Analytics Platform
- **Data Warehouse**: Centralized policy and claims data
- **Modeling Tools**: R/Python for actuarial analysis
- **Visualization**: Tableau/PowerBI for dashboards
- **Machine Learning**: Predictive modeling capabilities

### External Data Sources
- **Mortality Tables**: SOA, industry experience studies
- **Economic Data**: Federal Reserve, Bureau of Labor Statistics
- **Market Data**: Rating agencies, industry reports
- **Regulatory Updates**: NAIC, state insurance departments