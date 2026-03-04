
import pandas as pd
import numpy as np
import sys
import os

# Add notebooks directory to path to import 
sys.path.append(os.path.abspath('notebooks'))

from enhanced_feature_engineering import SimilarDayPipeline, FeatureConfig

def test_leakage_fix():
    print("Testing SimilarDayPipeline leakage fix...")
    
    # Create dummy data: 100 days of hourly data
    dates = pd.date_range(start='2023-01-01', periods=24*100, freq='H')
    df = pd.DataFrame({'demand': np.arange(len(dates))}, index=dates)
    
    # Config
    config = FeatureConfig(rolling_similar_days=True)
    
    # Case 1: Horizon 24 (Should exclude lag 24)
    print("\n--- Testing Horizon 24 ---")
    pipeline_24 = SimilarDayPipeline(config, forecast_horizon=24)
    features_24 = pipeline_24.transform(df)
    
    col_name = 'demand_same_hour_avg_30d'
    if col_name in features_24.columns:
        # Check the first valid index
        # If it starts at lag 48, first valid should be at index 48 (if we ignore min_periods logic for mean?)
        # Actually it's a mean of 30 columns.
        # Column 0: lag 48. First valid at 48.
        # Column 1: lag 72.
        # ...
        # The mean will be valid as soon as ONE column is valid (since mean uses skipna=True by default? 
        # Wait, inside concat mean(axis=1)). 
        # Pandas mean(axis=1) ignores NaNs by default.
        # So first valid index should be determined by the SMALLEST lag.
        
        # We expect smallest lag to be 48.
        # Let's inspect the calculated values or reconstruct logic.
        
        # We can inspect the private logic? No, let's just infer from data availability.
        first_valid = features_24[col_name].first_valid_index()
        print(f"First valid index: {first_valid}")
        expected_first = df.index[48] # Lag 48
        print(f"Expected approx: {expected_first}")
        
        if first_valid == expected_first:
            print("SUCCESS: Started at lag 48 (Safe for Horizon 24)")
        elif first_valid == df.index[24]:
             print("FAILURE: Started at lag 24 (Unsafe for Horizon 24)")
        else:
             print(f"Started at index position: {df.index.get_loc(first_valid)}")

    # Case 2: Horizon 48 (Should exclude lag 48, start at 72)
    print("\n--- Testing Horizon 48 ---")
    pipeline_48 = SimilarDayPipeline(config, forecast_horizon=48)
    features_48 = pipeline_48.transform(df)
    
    if col_name in features_48.columns:
        first_valid = features_48[col_name].first_valid_index()
        # Lag 72 expected
        print(f"First valid index: {first_valid}")
        expected_first = df.index[72]
        
        if first_valid == expected_first:
            print("SUCCESS: Started at lag 72 (Safe for Horizon 48)")
        else:
            print(f"FAILURE: Started at index position: {df.index.get_loc(first_valid)} (Expected 72)")

if __name__ == "__main__":
    test_leakage_fix()
