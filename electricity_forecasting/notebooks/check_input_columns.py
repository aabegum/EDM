import pandas as pd
import glob
import os

# Define paths
input_path = '../data/input/'
regions = ['aydin', 'denizli', 'mugla']

print("Checking input files for 'demand_lag_1h'...")

for region in regions:
    file_path = f'{input_path}{region}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"\nFile: {file_path}")
        print(f"Columns: {list(df.columns)}")
        if 'demand_lag_1h' in df.columns:
            print(f"!!! ALERT: 'demand_lag_1h' FOUND in {region}.csv !!!")
        else:
            print(f"'demand_lag_1h' NOT found in {region}.csv")
    else:
        print(f"File not found: {file_path}")
