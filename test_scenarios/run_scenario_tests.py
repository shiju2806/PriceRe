"""
Test Runner for Life & Savings/Retirement Scenarios
Validates pricing engine performance with real-world data
"""

import sys
from pathlib import Path
import traceback
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from life_savings_retirement_scenarios import LifeSavingsRetirementScenarios

def run_scenario_validation():
    """Run comprehensive scenario validation"""
    
    print("=" * 60)
    print("LIFE & SAVINGS/RETIREMENT REINSURANCE SCENARIO VALIDATION")
    print("=" * 60)
    print(f"Test Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize scenarios
        scenarios = LifeSavingsRetirementScenarios()
        print("✅ Scenario engine initialized successfully")
        
        # Run all scenarios
        print("\n🔄 Running comprehensive scenario suite...")
        results = scenarios.run_all_scenarios()
        
        print("\n📊 SCENARIO RESULTS:")
        print("-" * 40)
        
        # Display each scenario result
        for i, (scenario_key, scenario) in enumerate(results['scenarios'].items(), 1):
            print(f"\n{i}. {scenario['scenario_name']}")
            
            # Display key information
            premium_key = next((k for k in ['annual_premium', 'reserves_transferred', 'pension_obligation'] if k in scenario), None)
            if premium_key:
                print(f"   📈 {premium_key.replace('_', ' ').title()}: {scenario[premium_key]}")
            
            if 'policy_count' in scenario:
                print(f"   👥 Policies: {scenario['policy_count']}")
            elif 'covered_lives' in scenario:
                print(f"   👥 Covered Lives: {scenario['covered_lives']}")
            elif 'participant_count' in scenario:
                print(f"   👥 Participants: {scenario['participant_count']}")
            elif 'annuitant_count' in scenario:
                print(f"   👥 Annuitants: {scenario['annuitant_count']}")
            
            # Display pricing result
            pricing_result = scenario['pricing_result']
            print(f"   💰 Risk-Adjusted Premium: ${pricing_result.risk_adjusted_premium:,.0f}")
            print(f"   📉 Expected Loss Ratio: {pricing_result.expected_loss_ratio:.1%}")
            print(f"   📊 Technical Result: {pricing_result.technical_result:.1%}")
            
            # Display key metrics if available
            if 'key_metrics' in scenario:
                metrics = scenario['key_metrics']
                for metric, value in metrics.items():
                    if metric not in ['expected_loss_ratio', 'risk_adjusted_premium']:  # Already shown above
                        print(f"   📋 {metric.replace('_', ' ').title()}: {value}")
        
        # Display portfolio summary
        print(f"\n📈 PORTFOLIO SUMMARY:")
        print("-" * 40)
        portfolio = results['portfolio_summary']
        print(f"Total Reinsured Premium: {portfolio['total_reinsured_premium']}")
        print(f"Business Lines: {', '.join(portfolio['business_lines_covered'])}")
        print(f"Combined Metrics:")
        for metric, value in portfolio['combined_metrics'].items():
            print(f"  • {metric.replace('_', ' ').title()}: {value}")
        
        # Display validation info
        print(f"\n✅ VALIDATION STATUS:")
        print("-" * 40)
        validation = results['validation']
        for aspect, status in validation.items():
            print(f"• {aspect.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 All {len(results['scenarios'])} scenarios executed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in scenario validation:")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def validate_individual_scenarios():
    """Test each scenario individually for detailed debugging"""
    
    print("\n" + "=" * 50)
    print("INDIVIDUAL SCENARIO VALIDATION")
    print("=" * 50)
    
    scenarios = LifeSavingsRetirementScenarios()
    
    test_methods = [
        ("Large Term Life Portfolio", scenarios.scenario_1_large_term_life_portfolio),
        ("Universal Life Surplus Treaty", scenarios.scenario_2_universal_life_surplus_treaty),
        ("Group Life Excess of Loss", scenarios.scenario_3_group_life_xs_treaty),
        ("Annuity Mortality Risk", scenarios.scenario_4_annuity_mortality_risk),
        ("Pension Risk Transfer", scenarios.scenario_5_pension_risk_transfer)
    ]
    
    success_count = 0
    
    for name, method in test_methods:
        try:
            print(f"\n🧪 Testing: {name}")
            result = method()
            
            # Basic validation
            assert 'scenario_name' in result, "Missing scenario name"
            assert 'pricing_result' in result, "Missing pricing result"
            
            pricing = result['pricing_result']
            assert hasattr(pricing, 'risk_adjusted_premium'), "Missing risk-adjusted premium"
            assert hasattr(pricing, 'expected_loss_ratio'), "Missing expected loss ratio"
            assert hasattr(pricing, 'technical_result'), "Missing technical result"
            
            # Sanity checks
            assert pricing.risk_adjusted_premium > 0, "Premium must be positive"
            assert 0 < pricing.expected_loss_ratio < 2, "Loss ratio must be reasonable"
            assert -0.5 < pricing.technical_result < 1, "Technical result must be reasonable"
            
            print(f"   ✅ {name} - PASSED")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {name} - FAILED: {str(e)}")
    
    print(f"\n📊 Individual Test Results: {success_count}/{len(test_methods)} scenarios passed")
    return success_count == len(test_methods)

if __name__ == "__main__":
    print("Starting Life & Savings/Retirement Reinsurance Scenario Validation...\n")
    
    # Run comprehensive validation
    comprehensive_success = run_scenario_validation()
    
    # Run individual scenario tests
    individual_success = validate_individual_scenarios()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    
    if comprehensive_success and individual_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Comprehensive scenario suite: PASSED")
        print("✅ Individual scenario validation: PASSED")
        print("\n💼 Ready for production use with life & savings/retirement reinsurance")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"❌ Comprehensive suite: {'PASSED' if comprehensive_success else 'FAILED'}")
        print(f"❌ Individual validation: {'PASSED' if individual_success else 'FAILED'}")
        print("\n🔧 Please review errors above and fix issues")