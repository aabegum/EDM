
import json

notebook_path = r'c:\Users\begum.orhan\OneDrive - MRC\Masaüstü\EDM\electricity_forecasting\notebooks\02_feature_engineering_cleaned.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell by unique content
def find_cell_index_by_content(cells, content_snippet):
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source_code = "".join(cell['source'])
            if content_snippet in source_code:
                return i
    return -1

# 1. Add checks to avoid recalculation (Top of the notebook)
# We will wrap the main data loading block
load_data_idx = find_cell_index_by_content(nb['cells'], "data_path = '../data/input/'")
if load_data_idx != -1:
    source = nb['cells'][load_data_idx]['source']
    # Insert check logic at the beginning
    new_source = [
        "import os\n",
        "import pandas as pd\n",
        "\n",
        "processed_data_path = '../data/input/feature_engineered_data.csv'\n",
        "if os.path.exists(processed_data_path):\n",
        "    print(f'Loading pre-computed features from {processed_data_path}...')\n",
        "    df = pd.read_csv(processed_data_path)\n",
        "    df['time'] = pd.to_datetime(df['time'])\n",
        "    print(f'Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns')\n",
        "    # Skip recalculation flag or exit early could be complex in a notebook, \n",
        "    # but we will wrap subsequent heavy steps with 'if not features_exist:' logic if possible\n",
        "    # For now, we continue but columns checks prevent double work.\n",
        "else:\n",
        "    print('No pre-computed data found. Starting feature engineering...')\n"
    ]
    # Append original loading logic inside an 'else' or just keep it for raw loading
    # Ideally, we should restructure. Let's just modify the cell to load raw ONLY if processed doesn't exist?
    # Or simpler: Just check for columns before calculating.
    pass 

# 2. Remove demand_lag_1h and ensure safe lags (Cell 12)
hist_features_idx = find_cell_index_by_content(nb['cells'], "df['demand_lag_1h'] = df['demand'].shift(1)")
if hist_features_idx != -1:
    source = nb['cells'][hist_features_idx]['source']
    new_source = []
    for line in source:
        # Remove or comment out lag_1h and lag_24h for strict safety
        if "df['demand_lag_1h']" in line or "df['demand_lag_24h']" in line:
            new_source.append("# " + line.rstrip() + " # REMOVED for 24h horizon leakage prevention\n")
        elif "df['demand'].shift(1) -" in line: # Hourly deviation
             # This also uses lag 1. Need to shift this deviation to be safe.
             # Or replace it with a safe deviation: lag48 - lag72 ?
             new_source.append("# " + line.rstrip() + " # REMOVED unsafe deviation\n")
             # Add safe deviation
             new_source.append("df['demand_deviation_safe'] = df['demand'].shift(48) - df['demand'].shift(168) # 48h vs Week ago\n")
        else:
            new_source.append(line)
    
    # Update feature list construction
    # We need to make sure we don't accidentally add the removed ones to the list if they are static strings
    # The code uses list comprehension: [col for col in df.columns if ... 'lag' in col ...] 
    # Since we commented out the creation, they won't be in df.columns. Safe.
    
    nb['cells'][hist_features_idx]['source'] = new_source

# 3. Add column existence checks to expensive calculations
# Cyclical Features (Cell 6)
cyclical_idx = find_cell_index_by_content(nb['cells'], "df['hour_sin'] = np.sin")
if cyclical_idx != -1:
    source = nb['cells'][cyclical_idx]['source']
    # Wrap in check
    new_source = ["if 'hour_sin' not in df.columns:\n"]
    new_source.extend(["    " + line for line in source])
    new_source.append("else:\n")
    new_source.append("    print('Cyclical features already exist. Skipping...')\n")
    nb['cells'][cyclical_idx]['source'] = new_source

# Seasonal Features (Cell 11)
seasonal_idx = find_cell_index_by_content(nb['cells'], "df['is_peak_hour'] = df['hour'].isin")
if seasonal_idx != -1:
    source = nb['cells'][seasonal_idx]['source']
    new_source = ["if 'is_peak_hour' not in df.columns:\n"]
    new_source.extend(["    " + line for line in source])
    new_source.append("else:\n")
    new_source.append("    print('Seasonal features already exist. Skipping...')\n")
    nb['cells'][seasonal_idx]['source'] = new_source

# Domain Features (Cell 16)
domain_idx = find_cell_index_by_content(nb['cells'], "population_map = {'aydin': 1100000")
if domain_idx != -1:
    source = nb['cells'][domain_idx]['source']
    # Note: 'population' check is already inside, but we can wrap the whole block
    new_source = ["if 'is_morning_peak' not in df.columns:\n"]
    new_source.extend(["    " + line for line in source])
    new_source.append("else:\n")
    new_source.append("    print('Domain features already exist. Skipping...')\n")
    nb['cells'][domain_idx]['source'] = new_source

# Lag MA Features (Cell 7 - FORECAST_HORIZON)
lag_ma_idx = find_cell_index_by_content(nb['cells'], "FORECAST_HORIZON = 24")
if lag_ma_idx != -1:
    source = nb['cells'][lag_ma_idx]['source']
    new_source = ["# Check if safe lags exist\n", "if 'demand_lag_48h' not in df.columns:\n"]
    new_source.extend(["    " + line for line in source])
    new_source.append("else:\n")
    new_source.append("    print('Lag/MA features already exist. Skipping...')\n")
    nb['cells'][lag_ma_idx]['source'] = new_source

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
