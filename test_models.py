#!/usr/bin/env python3
"""
Quick verification script to check that all model files exist and can be loaded.
Run this after training to verify everything was created correctly.
"""

import joblib
import json
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("MODEL FILES VERIFICATION")
    print("=" * 60)
    print()
    
    # Check if files exist
    files_to_check = [
        'models/logreg_selected.joblib',
        'models/xgb_selected.joblib',
        'models/thresholds.json',
        'data/processed/train_processed.csv',
        'data/processed/test_processed.csv'
    ]
    
    print("1. Checking file existence:")
    print("-" * 60)
    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   ✅ {file}")
            print(f"      Size: {size:.1f} KB")
        else:
            print(f"   ❌ {file} - NOT FOUND")
            all_exist = False
    print()
    
    if not all_exist:
        print("⚠️  Some files are missing. Please run training first:")
        print("   python scripts/run_train.py")
        return False
    
    # Load and display thresholds
    print("2. Loading thresholds:")
    print("-" * 60)
    try:
        with open('models/thresholds.json', 'r') as f:
            thresholds = json.load(f)
        print("   ✅ Thresholds loaded successfully")
        print(f"   - Logistic Regression threshold: {thresholds['logreg']['threshold']}")
        print(f"   - XGBoost threshold: {thresholds['xgb']['threshold']}")
        print(f"   - LR selected features: {len(thresholds['logreg']['selected_features'])} features")
        print(f"   - XGB selected features: {len(thresholds['xgb']['selected_features'])} features")
        print(f"   - LR features: {', '.join(thresholds['logreg']['selected_features'][:5])}...")
        print(f"   - XGB features: {', '.join(thresholds['xgb']['selected_features'][:5])}...")
    except Exception as e:
        print(f"   ❌ Error loading thresholds: {e}")
        return False
    print()
    
    # Test loading models
    print("3. Testing model loading:")
    print("-" * 60)
    try:
        lr_model = joblib.load('models/logreg_selected.joblib')
        print("   ✅ Logistic Regression model loads successfully")
        print(f"      Model type: {type(lr_model).__name__}")
    except Exception as e:
        print(f"   ❌ Error loading LR model: {e}")
        return False
    
    try:
        xgb_model = joblib.load('models/xgb_selected.joblib')
        print("   ✅ XGBoost model loads successfully")
        print(f"      Model type: {type(xgb_model).__name__}")
    except Exception as e:
        print(f"   ❌ Error loading XGB model: {e}")
        return False
    print()
    
    # Check processed data
    print("4. Checking processed data:")
    print("-" * 60)
    try:
        import pandas as pd
        train_df = pd.read_csv('data/processed/train_processed.csv')
        test_df = pd.read_csv('data/processed/test_processed.csv')
        print(f"   ✅ Training data: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"   ✅ Test data: {len(test_df)} rows, {len(test_df.columns)} columns")
    except Exception as e:
        print(f"   ⚠️  Could not read processed data: {e}")
    print()
    
    print("=" * 60)
    print("✅ ALL CHECKS PASSED! Everything is working correctly.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. View evaluation results: python scripts/run_eval.py")
    print("  2. Launch dashboard: streamlit run dashboard.py")
    print("  3. Open notebook: jupyter lab notebooks/03_implementation_details.ipynb")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

