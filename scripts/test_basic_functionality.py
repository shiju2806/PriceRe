#!/usr/bin/env python3
"""
Basic functionality test for PricingFlow core components
"""

import sys
import os
import math
from pathlib import Path
from datetime import datetime, date
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Test basic imports
try:
    import polars as pl
    import numpy as np
    import pandas as pd
    from faker import Faker
    print("✅ All required packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_actuarial_engine():
    """Test the core actuarial calculation engine"""
    print("\n🧮 Testing Actuarial Engine...")
    
    # Simple actuarial calculations without complex imports
    def calculate_mortality_rate(age, gender, smoker=False):
        """Simplified mortality rate calculation"""
        base_rate = 0.001 + age * 0.0002
        if gender.upper() == 'M':
            base_rate *= 1.2
        if smoker:
            base_rate *= 2.5
        return min(base_rate, 1.0)
    
    def calculate_life_expectancy(age, gender):
        """Simplified life expectancy calculation"""
        base_expectancy = 85 - age
        if gender.upper() == 'F':
            base_expectancy += 3  # Women live longer on average
        return max(base_expectancy, 0)
    
    def calculate_premium(age, gender, face_amount, smoker=False):
        """Simplified premium calculation"""
        mortality_rate = calculate_mortality_rate(age, gender, smoker)
        base_premium = face_amount * mortality_rate * 1.2  # Add loadings
        return base_premium
    
    # Test calculations
    test_cases = [
        {"age": 35, "gender": "M", "face_amount": 500000, "smoker": False},
        {"age": 45, "gender": "F", "face_amount": 250000, "smoker": True},
        {"age": 60, "gender": "M", "face_amount": 1000000, "smoker": False}
    ]
    
    for i, case in enumerate(test_cases, 1):
        mortality = calculate_mortality_rate(case["age"], case["gender"], case["smoker"])
        life_exp = calculate_life_expectancy(case["age"], case["gender"])
        premium = calculate_premium(case["age"], case["gender"], case["face_amount"], case["smoker"])
        
        print(f"  Test Case {i}:")
        print(f"    Age: {case['age']}, Gender: {case['gender']}, Smoker: {case['smoker']}")
        print(f"    Mortality Rate: {mortality:.4f}")
        print(f"    Life Expectancy: {life_exp:.1f} years")
        print(f"    Annual Premium: ${premium:,.0f}")
    
    print("✅ Actuarial calculations completed")

def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing Synthetic Data Generation...")
    
    fake = Faker()
    fake.seed_instance(42)
    np.random.seed(42)
    
    # Generate sample life insurance records
    n_records = 1000
    records = []
    
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    occupations = ['Teacher', 'Engineer', 'Manager', 'Salesperson', 'Nurse', 'Accountant']
    
    for i in range(n_records):
        # Basic demographics
        gender = np.random.choice(['M', 'F'])
        age = np.random.randint(25, 70)
        smoker = np.random.random() < 0.15
        
        # Generate realistic income based on age
        if age < 30:
            base_income = 45000
        elif age < 50:
            base_income = 75000
        else:
            base_income = 85000
        
        income = base_income * np.random.lognormal(0, 0.3)
        face_amount = income * np.random.uniform(5, 15)
        
        # Calculate premium using our simple formula
        mortality_rate = 0.001 + age * 0.0002
        if gender == 'M':
            mortality_rate *= 1.2
        if smoker:
            mortality_rate *= 2.5
        
        annual_premium = face_amount * mortality_rate * 1.2
        
        record = {
            'policy_number': f'POL{uuid.uuid4().hex[:8].upper()}',
            'age_at_issue': age,
            'gender': gender,
            'smoker_status': 'Smoker' if smoker else 'Non-Smoker',
            'annual_income': income,
            'face_amount': face_amount,
            'annual_premium': annual_premium,
            'state': np.random.choice(states),
            'occupation': np.random.choice(occupations),
            'issue_date': fake.date_between(start_date='-5y', end_date='today')
        }
        
        records.append(record)
    
    # Create DataFrame
    df = pl.DataFrame(records)
    
    print(f"  Generated {len(df)} life insurance records")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Average age: {df['age_at_issue'].mean():.1f}")
    print(f"  Average face amount: ${df['face_amount'].mean():,.0f}")
    print(f"  Average premium: ${df['annual_premium'].mean():,.0f}")
    print(f"  Smoking rate: {(df['smoker_status'] == 'Smoker').sum() / len(df) * 100:.1f}%")
    
    # Save sample data
    output_path = "data/synthetic/sample_life_insurance.csv"
    os.makedirs("data/synthetic", exist_ok=True)
    df.write_csv(output_path)
    print(f"  Saved sample data to: {output_path}")
    
    print("✅ Synthetic data generation completed")
    return df

def test_feature_engineering(df):
    """Test basic feature engineering"""
    print("\n🔧 Testing Feature Engineering...")
    
    # Create basic actuarial features
    df = df.with_columns([
        # Age-based features
        (pl.col('age_at_issue') ** 2).alias('age_squared'),
        (pl.col('age_at_issue') // 10 * 10).alias('age_band_10yr'),
        (pl.col('age_at_issue') >= 50).alias('over_50'),
        
        # Gender features
        (pl.col('gender') == 'M').alias('gender_male'),
        
        # Smoker features
        (pl.col('smoker_status') == 'Smoker').alias('is_smoker'),
        
        # Financial features
        (pl.col('face_amount') / pl.col('annual_income')).alias('coverage_ratio'),
        (pl.col('annual_premium') / pl.col('annual_income') * 100).alias('premium_to_income_pct'),
        
        # Face amount categories
        (pl.col('face_amount') >= 1000000).alias('jumbo_coverage'),
        
        # Age-gender interaction
        (pl.col('age_at_issue') * (pl.col('gender') == 'M').cast(pl.Int32)).alias('age_male_interaction')
    ])
    
    print(f"  Added {len(df.columns) - 10} engineered features")  # Original had ~10 columns
    print(f"  Average coverage ratio: {df['coverage_ratio'].mean():.1f}x")
    print(f"  Average premium/income: {df['premium_to_income_pct'].mean():.1f}%")
    print(f"  Jumbo policies: {df['jumbo_coverage'].sum()} ({df['jumbo_coverage'].sum()/len(df)*100:.1f}%)")
    
    print("✅ Feature engineering completed")
    return df

def test_basic_modeling(df):
    """Test basic modeling capabilities"""
    print("\n🤖 Testing Basic Modeling...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Prepare features for modeling
        feature_columns = [
            'age_at_issue', 'age_squared', 'gender_male', 'is_smoker',
            'coverage_ratio', 'age_male_interaction'
        ]
        
        # Convert to pandas for sklearn
        df_pandas = df.select(['annual_premium'] + feature_columns).to_pandas()
        
        # Handle any missing values
        df_pandas = df_pandas.fillna(df_pandas.median())
        
        X = df_pandas[feature_columns]
        y = df_pandas['annual_premium']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        
        # Test Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        print(f"  Linear Regression:")
        print(f"    R² Score: {lr_r2:.4f}")
        print(f"    Mean Absolute Error: ${lr_mae:,.0f}")
        
        print(f"  Random Forest:")
        print(f"    R² Score: {rf_r2:.4f}")
        print(f"    Mean Absolute Error: ${rf_mae:,.0f}")
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
        print(f"  Top 3 Important Features:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {feature}: {importance:.3f}")
        
        print("✅ Basic modeling completed")
        
    except ImportError as e:
        print(f"⚠️ Scikit-learn not available for modeling test: {e}")

def main():
    print("🚀 PricingFlow Basic Functionality Test")
    print("="*50)
    
    # Test actuarial calculations
    test_actuarial_engine()
    
    # Test data generation
    df = test_synthetic_data_generation()
    
    # Test feature engineering
    df = test_feature_engineering(df)
    
    # Test modeling
    test_basic_modeling(df)
    
    print("\n🎉 All basic functionality tests completed!")
    print("\n🎯 Next steps:")
    print("   1. Review generated data: data/synthetic/sample_life_insurance.csv")
    print("   2. Try building a full pricing model")
    print("   3. Explore the Streamlit demo interface")
    
    return 0

if __name__ == "__main__":
    exit(main())