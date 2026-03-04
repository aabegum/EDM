import pandas as pd
import pathlib
import os

processed_dir = pathlib.Path('../data/processed')
files = list(processed_dir.glob('*_enhanced_features.parquet'))

print(f"Found {len(files)} files to combine.")

dfs = []
for f in files:
    print(f"Loading {f.name}...")
    dfs.append(pd.read_parquet(f))

if dfs:
    combined_df = pd.concat(dfs).reset_index()
    combined_df = combined_df.sort_values(['time', 'city'])
    
    # Save to CSV as expected by training script
    output_path = '../data/processed/engineered_features_essential.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path}")
    print(f"Shape: {combined_df.shape}")
else:
    print("No files found to combine!")
