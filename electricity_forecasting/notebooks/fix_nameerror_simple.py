
import json
import os
import sys

# Use raw string for path to avoid escape sequence issues
notebook_path = r'c:/Users/begum.orhan/OneDrive - MRC/Masaüstü/EDM/electricity_forecasting/notebooks/02_feature_engineering_cleaned.ipynb'

if not os.path.exists(notebook_path):
    print(f"Error: File not found at {notebook_path}")
    sys.exit(1)

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = "".join(source)
            if "feature_categories = {" in source_text:
                print(f"Found target cell at index {i}")
                
                # Check if we already have the fix
                if "# robust initialization" in source_text:
                    print("Patch already applied.")
                    break
                
                # Construct the initialization code
                # We place it right at the start of the cell
                init_code = [
                    "# robust initialization of feature lists to prevent NameError\n",
                    "try:\n",
                    "    temp_features\n",
                    "except NameError:\n",
                    "    temp_features = []\n",
                    "    # Try to populate from known columns if possible\n",
                    "    if 'heating_degree_hours_static' in df.columns:\n",
                    "        temp_features.extend(['heating_degree_hours_static', 'cooling_degree_hours_static'])\n",
                    "\n",
                    "try: cyclical_features\n",
                    "except NameError: cyclical_features = []\n",
                    "try: lag_ma_features\n",
                    "except NameError: lag_ma_features = []\n",
                    "try: interaction_features\n",
                    "except NameError: interaction_features = []\n",
                    "try: weather_features\n",
                    "except NameError: weather_features = []\n",
                    "try: season_features\n",
                    "except NameError: season_features = []\n",
                    "try: historical_features\n",
                    "except NameError: historical_features = []\n",
                    "\n"
                ]
                
                cell['source'] = init_code + source
                modified = True
                break
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {notebook_path}")
    else:
        print("Target cell not found or already patched.")

except Exception as e:
    print(f"Error processing notebook: {e}")
    sys.exit(1)
