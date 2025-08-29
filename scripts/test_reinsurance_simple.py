#!/usr/bin/env python3
"""
Simple Reinsurance System Test

Minimal test to verify reinsurance functionality without complex dependencies
"""

import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_reinsurance_simple():
    """Simple test of reinsurance components"""
    
    print("🔄 Testing PricingFlow Reinsurance System (Simple)")
    print("=" * 50)
    
    # Test 1: Data Generation
    print("\n📊 Testing Data Generation...")
    try:
        import sys
        sys.path.append(str(project_root / "src"))
        
        # Direct imports to bypass __init__.py issues
        from reinsurance.data_generator import ReinsuranceDataGenerator
        from reinsurance.treaty_pricer import TreatyPricer, TreatyTerms
        from reinsurance.feature_engineering import ReinsuranceFeatures
        
        print("✅ All reinsurance modules imported successfully")
        
        # Generate sample data
        generator = ReinsuranceDataGenerator(seed=42)
        
        # Generate treaty data
        try:
            treaty_df = generator.generate_treaty_data(n_treaties=20)
            print(f"✅ Generated {len(treaty_df)} treaties")
            print(f"   Treaty columns: {list(treaty_df.columns)}")
        except Exception as e:
            print(f"   ❌ Treaty generation failed: {e}")
            raise e
        
        # Generate portfolio data first (needed for claims)
        try:
            portfolio_df = generator.generate_portfolio_data(n_portfolios=10)
            print(f"✅ Generated {len(portfolio_df)} portfolios")
            print(f"   Portfolio columns: {list(portfolio_df.columns)}")
        except Exception as e:
            print(f"   ❌ Portfolio generation failed: {e}")
            raise e
        
        # Generate claims data
        try:
            claims_df = generator.generate_claims_data(treaty_df, claims_per_treaty=15)
            print(f"✅ Generated {len(claims_df)} claims")
            print(f"   Claims columns: {list(claims_df.columns)}")
        except Exception as e:
            print(f"   ❌ Claims generation failed: {e}")
            raise e
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    # Test 2: Feature Engineering
    print("\n⚙️ Testing Feature Engineering...")
    try:
        feature_engine = ReinsuranceFeatures()
        
        # Create treaty features
        treaty_features = feature_engine.create_treaty_features(treaty_df)
        print(f"✅ Created treaty features: {len(treaty_features.columns)} columns")
        
        # Create portfolio features
        portfolio_features = feature_engine.create_portfolio_features(portfolio_df)
        print(f"✅ Created portfolio features: {len(portfolio_features.columns)} columns")
        
        # Create claims features
        claims_features = feature_engine.create_claims_features(claims_df)
        print(f"✅ Created claims features: {len(claims_features.columns)} columns")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    # Test 3: Treaty Pricing
    print("\n💰 Testing Treaty Pricing...")
    try:
        pricer = TreatyPricer()
        
        # Test Quota Share pricing
        qs_terms = TreatyTerms(
            treaty_type="Quota Share",
            cession_rate=0.3,
            commission=0.25,
            brokerage=0.03,
            profit_commission_rate=0.15
        )
        
        qs_result = pricer.price_quota_share(portfolio_df, qs_terms)
        print(f"✅ Quota Share Pricing:")
        print(f"   Technical Premium: ${qs_result.technical_premium:,.0f}")
        print(f"   Commercial Premium: ${qs_result.commercial_premium:,.0f}")
        print(f"   Expected Loss Ratio: {qs_result.expected_loss_ratio:.1%}")
        
        # Test Surplus pricing
        surplus_terms = TreatyTerms(
            treaty_type="Surplus",
            attachment_point=250000,
            limit=2500000,
            commission=0.22,
            brokerage=0.03
        )
        
        surplus_result = pricer.price_surplus_treaty(portfolio_df, surplus_terms)
        print(f"✅ Surplus Treaty Pricing:")
        print(f"   Commercial Premium: ${surplus_result.commercial_premium:,.0f}")
        print(f"   Expected Loss Ratio: {surplus_result.expected_loss_ratio:.1%}")
        
    except Exception as e:
        print(f"❌ Treaty pricing failed: {e}")
        return False
    
    # Test 4: Save Results
    print("\n💾 Saving Test Results...")
    try:
        results_dir = project_root / "data" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        treaty_features.write_csv(results_dir / "treaty_features.csv")
        portfolio_features.write_csv(results_dir / "portfolio_features.csv")
        claims_features.write_csv(results_dir / "claims_features.csv")
        
        print(f"✅ Results saved to: {results_dir}")
        
    except Exception as e:
        print(f"❌ Saving results failed: {e}")
        return False
    
    # Summary
    print("\n🎉 Reinsurance System Test Completed Successfully!")
    print("\n📊 System Summary:")
    print(f"   • Generated realistic reinsurance data: {len(treaty_df)} treaties")
    print(f"   • Engineered domain-specific features for pricing models")
    print(f"   • Priced multiple treaty types with advanced algorithms")
    print(f"   • Ready for integration with full MVP interface")
    
    print("\n🚀 Next Steps:")
    print("   1. Run MVP interface: streamlit run ui/reinsurance_mvp.py")
    print("   2. Upload real reinsurance data for analysis")
    print("   3. Train custom pricing models for your portfolio")
    
    return True

if __name__ == "__main__":
    success = test_reinsurance_simple()
    if not success:
        sys.exit(1)