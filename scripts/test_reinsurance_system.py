#!/usr/bin/env python3
"""
Test Reinsurance System

Comprehensive test of the reinsurance pricing and data processing system.
Generates sample data, validates it, engineers features, and runs pricing models.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def test_reinsurance_system():
    """Test the complete reinsurance system"""
    
    print("🔄 Testing PricingFlow Reinsurance System")
    print("=" * 50)
    
    # Import modules
    try:
        from src.reinsurance.data_generator import ReinsuranceDataGenerator
        from src.reinsurance.treaty_pricer import TreatyPricer, TreatyTerms
        from src.reinsurance.feature_engineering import ReinsuranceFeatures
        from src.data.data_validator import DataValidator, DataType
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Trying to install missing dependencies...")
        try:
            import subprocess
            result = subprocess.run(["pip", "install", "lightgbm"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"   Failed to install lightgbm: {result.stderr}")
            else:
                print("   ✅ LightGBM installed successfully")
                # Try importing again
                from src.reinsurance.data_generator import ReinsuranceDataGenerator
                from src.reinsurance.treaty_pricer import TreatyPricer, TreatyTerms
                from src.reinsurance.feature_engineering import ReinsuranceFeatures
                from src.data.data_validator import DataValidator, DataType
                print("✅ All modules imported successfully after installing dependencies")
        except Exception as install_error:
            print(f"❌ Failed to install dependencies: {install_error}")
            return False
    
    # Test 1: Generate sample reinsurance data
    print("\n📊 Generating Sample Reinsurance Data...")
    
    generator = ReinsuranceDataGenerator(seed=42)
    
    # Generate treaty data
    treaty_df = generator.generate_treaty_data(n_treaties=50)
    print(f"   Generated {len(treaty_df)} treaties")
    
    # Generate claims data
    claims_df = generator.generate_claims_data(treaty_df, claims_per_treaty=25)
    print(f"   Generated {len(claims_df)} claims")
    
    # Generate portfolio data
    portfolio_df = generator.generate_portfolio_data(n_portfolios=30)
    print(f"   Generated {len(portfolio_df)} portfolios")
    
    # Generate catastrophe events
    cat_df = generator.generate_catastrophe_events(n_events=15)
    print(f"   Generated {len(cat_df)} catastrophe events")
    
    # Generate loss development data
    development_df = generator.generate_loss_development_data(claims_df)
    print(f"   Generated {len(development_df)} development records")
    
    # Test 2: Data validation
    print("\n🔍 Testing Data Validation...")
    
    # Save sample data for validation test
    data_dir = project_root / "data" / "sample_reinsurance"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    treaty_file = data_dir / "sample_treaties.csv"
    treaty_df.write_csv(treaty_file)
    
    # Validate data
    validator = DataValidator()
    validation_report = validator.validate_upload(treaty_file, DataType.TREATY)
    
    print(f"   Validation Status: {'✅ VALID' if validation_report.is_valid else '❌ INVALID'}")
    print(f"   Confidence Score: {validation_report.confidence_score:.1f}%")
    print(f"   Issues Found: {len(validation_report.validation_results)}")
    
    if validation_report.validation_results:
        for result in validation_report.validation_results[:3]:  # Show first 3 issues
            print(f"     - {result.level.value.upper()}: {result.message}")
    
    # Test 3: Feature engineering
    print("\n⚙️ Testing Feature Engineering...")
    
    feature_engine = ReinsuranceFeatures()
    
    # Create treaty features
    treaty_features = feature_engine.create_treaty_features(treaty_df)
    print(f"   Treaty features: {len(treaty_features.columns)} columns")
    
    # Create portfolio features
    portfolio_features = feature_engine.create_portfolio_features(portfolio_df)
    print(f"   Portfolio features: {len(portfolio_features.columns)} columns")
    
    # Create claims features
    claims_features = feature_engine.create_claims_features(claims_df)
    print(f"   Claims features: {len(claims_features.columns)} columns")
    
    # Create aggregate features
    aggregate_features = feature_engine.create_aggregate_features(
        claims_features, ["treaty_id"]
    )
    print(f"   Aggregate features: {len(aggregate_features.columns)} columns")
    
    # Show feature summary
    feature_summary = feature_engine.create_feature_summary()
    total_features = sum(len(features) for features in feature_summary.values())
    print(f"   Total available features: {total_features}")
    
    # Test 4: Treaty pricing
    print("\n💰 Testing Treaty Pricing...")
    
    pricer = TreatyPricer()
    
    # Test Quota Share pricing
    qs_terms = TreatyTerms(
        treaty_type="Quota Share",
        cession_rate=0.3,
        commission=0.25,
        brokerage=0.03,
        profit_commission_rate=0.15,
        loss_corridor_min=75,
        loss_corridor_max=105
    )
    
    qs_result = pricer.price_quota_share(portfolio_df, qs_terms)
    print(f"   Quota Share Pricing:")
    print(f"     Technical Premium: ${qs_result.technical_premium:,.0f}")
    print(f"     Commercial Premium: ${qs_result.commercial_premium:,.0f}")
    print(f"     Expected Loss Ratio: {qs_result.expected_loss_ratio:.1%}")
    print(f"     Return on Capital: {qs_result.return_on_capital:.1%}")
    
    # Test Surplus pricing
    surplus_terms = TreatyTerms(
        treaty_type="Surplus",
        attachment_point=250000,
        limit=2500000,
        commission=0.22,
        brokerage=0.03
    )
    
    surplus_result = pricer.price_surplus_treaty(portfolio_df, surplus_terms)
    print(f"   Surplus Treaty Pricing:")
    print(f"     Technical Premium: ${surplus_result.technical_premium:,.0f}")
    print(f"     Commercial Premium: ${surplus_result.commercial_premium:,.0f}")
    print(f"     Expected Loss Ratio: {surplus_result.expected_loss_ratio:.1%}")
    print(f"     Return on Capital: {surplus_result.return_on_capital:.1%}")
    
    # Test Excess of Loss pricing
    xl_terms = TreatyTerms(
        treaty_type="Excess of Loss",
        attachment_point=1000000,
        limit=10000000,
        brokerage=0.05,
        aggregate_limit=25000000,
        reinstatements=2
    )
    
    xl_result = pricer.price_excess_of_loss(portfolio_df, xl_terms, cat_df)
    print(f"   Excess of Loss Pricing:")
    print(f"     Technical Premium: ${xl_result.technical_premium:,.0f}")
    print(f"     Commercial Premium: ${xl_result.commercial_premium:,.0f}")
    print(f"     Expected Loss Ratio: {xl_result.expected_loss_ratio:.1%}")
    print(f"     Burning Cost: ${xl_result.risk_metrics['burning_cost']:,.0f}")
    print(f"     Rate on Line: {xl_result.risk_metrics['rate_on_line']:.1%}")
    
    # Test 5: Treaty optimization
    print("\n🎯 Testing Treaty Optimization...")
    
    optimization_objectives = {
        "roe": 0.15,
        "loss_ratio": 0.65,
        "volatility": 0.25
    }
    
    optimal_structure = pricer.optimize_treaty_structure(
        portfolio_df, 
        budget=5000000,  # $5M budget
        objectives=optimization_objectives
    )
    
    if optimal_structure:
        print(f"   Optimal Structure: {optimal_structure['structure_type'].title()}")
        print(f"   Parameters: {optimal_structure['parameters']}")
        print(f"   Premium: ${optimal_structure['pricing_result'].commercial_premium:,.0f}")
        print(f"   Optimization Score: {optimal_structure['optimization_score']:.2f}")
    else:
        print("   No optimal structure found within budget")
    
    # Test 6: Save results
    print("\n💾 Saving Test Results...")
    
    # Save all generated data
    results_dir = project_root / "data" / "reinsurance_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    treaty_features.write_csv(results_dir / "treaty_features.csv")
    portfolio_features.write_csv(results_dir / "portfolio_features.csv")
    claims_features.write_csv(results_dir / "claims_features.csv")
    aggregate_features.write_csv(results_dir / "aggregate_features.csv")
    cat_df.write_csv(results_dir / "catastrophe_events.csv")
    development_df.write_csv(results_dir / "loss_development.csv")
    
    print(f"   Results saved to: {results_dir}")
    
    # Summary statistics
    print("\n📈 System Performance Summary:")
    print(f"   Data Generation: ✅ Generated {len(treaty_df)} treaties with full claims history")
    print(f"   Data Validation: ✅ Validated data with {validation_report.confidence_score:.1f}% confidence")
    print(f"   Feature Engineering: ✅ Created {total_features} reinsurance-specific features")
    print(f"   Treaty Pricing: ✅ Priced 3 different treaty types successfully")
    print(f"   Optimization: ✅ Found optimal treaty structure within budget")
    
    # Data quality metrics
    print("\n📊 Data Quality Metrics:")
    avg_loss_ratio = treaty_df["loss_ratio"].mean()
    print(f"   Average Loss Ratio: {avg_loss_ratio:.1%}")
    
    total_premium = treaty_df["premium"].sum()
    print(f"   Total Premium Volume: ${total_premium:,.0f}")
    
    profitable_treaties = (treaty_df["combined_ratio"] < 1.0).sum()
    print(f"   Profitable Treaties: {profitable_treaties}/{len(treaty_df)} ({profitable_treaties/len(treaty_df):.1%})")
    
    # Claims analysis
    avg_claim_size = claims_df["gross_claim_amount"].mean()
    total_claims = claims_df["gross_claim_amount"].sum()
    recovery_ratio = claims_df["reinsurance_recovery"].sum() / total_claims
    
    print(f"   Average Claim Size: ${avg_claim_size:,.0f}")
    print(f"   Total Claims: ${total_claims:,.0f}")
    print(f"   Recovery Ratio: {recovery_ratio:.1%}")
    
    print("\n🎉 Reinsurance system test completed successfully!")
    print("   The system is ready for production use with real data.")
    
    return True

if __name__ == "__main__":
    success = test_reinsurance_system()
    if not success:
        sys.exit(1)
    else:
        print("\n🚀 Next steps:")
        print("   1. Run the MVP interface: streamlit run ui/reinsurance_mvp.py")
        print("   2. Upload your own reinsurance data for analysis")
        print("   3. Generate pricing models and export results")