# 🏛️ Reinsurance Pricing Platform - User Guide

## Quick Start

### 1. 🔧 Setup (First Time Only)
```bash
# Install dependencies
python3 setup_user_environment.py

# Verify installation
python3 -c "import streamlit, pandas, plotly; print('✅ Ready to go!')"
```

### 2. 🚀 Launch the Platform
```bash
streamlit run ui/professional_pricing_platform.py
```
Then open: http://localhost:8501

### 3. 📊 Test with Sample Data
Use the provided sample files in `sample_data/`:
- `perfect_format_sample.csv` - Small perfect example (100 policies)  
- `realistic_portfolio_sample.csv` - Large realistic example (1,000 policies)
- `common_mistakes_example.csv` - Shows what NOT to do

## Data Format Requirements

### Required CSV Columns (Exact Names)
```
policy_number    - Unique identifier (e.g., "POL123456")
issue_date       - Date in YYYY-MM-DD format (e.g., "2020-01-15")
face_amount      - Death benefit amount (e.g., 100000)
annual_premium   - Yearly premium (e.g., 1200)
issue_age        - Age at issue (e.g., 35)
gender           - "M" or "F"
smoker_status    - "Smoker" or "Nonsmoker"
product_type     - "Term Life", "Universal Life", or "Whole Life"
state            - 2-letter state code (e.g., "CA")
policy_status    - "Inforce" or "Lapsed"
```

### ❌ Common Mistakes to Avoid
- Wrong column names (PolicyNumber vs policy_number)
- Wrong date format (1/15/2020 vs 2020-01-15)
- Currency symbols in amounts ($100,000 vs 100000)
- Wrong status values (Active vs Inforce)
- Missing required columns

## Typical Workflow

1. **Create Submission**
   - Enter cedent (insurance company) details
   - Specify treaty type and financial information
   - Set submission parameters

2. **Upload Policy Data**
   - Use properly formatted CSV file
   - System validates data quality (target: 90%+ score)
   - Review any validation warnings

3. **Experience Analysis**
   - System calculates mortality statistics
   - Assesses portfolio risk factors  
   - Determines statistical credibility

4. **Final Pricing**
   - Calculates expected loss ratio
   - Applies expense and risk margins
   - Provides pricing recommendations

5. **Review Results**
   - Final gross rate and confidence level
   - Sensitivity analysis scenarios
   - Professional recommendations

## Troubleshooting

### "Production engine not available"
- Run: `python3 setup_user_environment.py`
- Ensure all dependencies installed successfully

### "Data validation failed"
- Check CSV column names match exactly
- Verify date format: YYYY-MM-DD
- Ensure no currency symbols or commas in numeric fields

### "Statistical credibility too low"
- Need minimum ~500 policies for reliable results
- Consider using industry benchmarks
- System will warn about low credibility scenarios

## Support

For issues or questions:
1. Check this guide first
2. Review sample data files for correct format
3. Run diagnostics: `python3 test_real_user_experience.py`
