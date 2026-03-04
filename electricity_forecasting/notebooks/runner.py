import traceback
import sys
import subprocess

def run_script(script_name, args):
    print(f"Running {script_name}...")
    try:
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(['python', script_name] + args, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        print(f"[{script_name}] STDOUT:")
        print(result.stdout)
        if result.stderr:
            print(f"[{script_name}] STDERR:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"Error: {script_name} failed with return code {result.returncode}")
            return False
        
        print(f"Successfully completed {script_name}\n")
        return True
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        traceback.print_exc()
        return False

def main():
    print("Starting End-to-End Pipeline\n" + "="*40)
    
    # success = run_script(
    #     'notebooks/enhanced_feature_engineering.py',
    #     ['--config', 'configs/standard.yaml', 
    #      '--input-dir', 'data/input', 
    #      '--output-dir', 'data/processed']
    # )
    
    # if not success:
    #     return
        
    success = run_script('notebooks/03_model_training.py', [])
    
    if not success:
        return
        
    run_script('notebooks/04_model_explainability.py', [])

if __name__ == "__main__":
    main()
