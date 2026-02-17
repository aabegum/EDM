
import json
import os

notebook_path = r'c:/Users/begum.orhan/OneDrive - MRC/Masaüstü/EDM/electricity_forecasting/notebooks/02_feature_engineering_cleaned.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = "".join(source)
            if "if 'selected_features' not in dir():" in source_text:
                print("Found target cell for IndentationError.")
                
                new_source = []
                # Simple state machine to handle the fix
                pending_if = False
                
                for line in source:
                    if "if 'selected_features' not in dir():" in line:
                        new_source.append(line)
                        # Check if the next line is indented or not
                        # Actually, we can just insert the initialization right here and make it robust
                        new_source.append("    selected_features = []\n")
                        pending_if = True
                    else:
                        # If we just inserted the fix, we don't need to do anything special for subsequent lines
                        # UNLESS the original code had an indentation error where the next line was intended to be in the block but wasn't.
                        # Looking at the traceback:
                        # if 'selected_features' not in dir():
                        #    
                        # # comments
                        # special_calendar_features = ...
                        
                        # It seems 'special_calendar_features' was NOT intended to be inside the if block.
                        # So just adding the initialization line fixes the empty if block.
                        new_source.append(line)
                
                if new_source != source:
                    cell['source'] = new_source
                    modified = True
                break
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {notebook_path}")
    else:
        print("Target cell not found or no changes needed.")

except Exception as e:
    print(f"Error processing notebook: {e}")
