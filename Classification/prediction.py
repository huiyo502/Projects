import rmodel
import argparse
import json
import numpy as np
from datetime import datetime
import os
import sys

np.random.seed(42)

# Comment requirement
parser = argparse.ArgumentParser(description='Script for model prediction and evaluation.')
parser.add_argument('--path', type=str, default=None, help='Path to the main data file.')
parser.add_argument('--json_file', type=str, default="hyperparams_best.json", help='JSON file containing best hyperparameters.')
parser.add_argument('--data_mode', type=int, default=3, help='Data preparation mode (1, 2, or 3).')
parser.add_argument('--scenarios', type=int, default=1, help='Data scenario setting.')
parser.add_argument('--subscenario', type=int, default=1, help='Data subscenario setting.')
parser.add_argument('--model_list', type=str, nargs='+', default=["xgb", "knn", "svc", "rf"], help='List of models to run (e.g., xgb knn).')
parser.add_argument('--name', type=str, default='default_run', help='Name for the result folder.')
args = parser.parse_args()


# Create result folder and redirect stdout 
TODAY_STR = datetime.now().strftime('%m/%d')
directory_path = f"../Result/{TODAY_STR}/{args.name}/mod{args.data_mode}_scen{args.scenarios}_subscen{args.subscenario}"

print("Setting up directory and output stream...")
try:
    # Use exist_ok=True for robustness
    os.makedirs(directory_path, exist_ok=True) 
    print(f"Directory '{directory_path}' created successfully.")
except Exception as e:
    print(f"Error creating directory '{directory_path}': {e}")
    sys.exit(1)

output_filename = f'output_mod{args.data_mode}_scen{args.scenarios}_subscen{args.subscenario}.txt'
output_file_path = os.path.join(directory_path, output_filename)

try:
    output_file = open(output_file_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = output_file
except IOError as e:
    print(f"Error opening output file {output_file_path}: {e}")
    sys.exit(1)


# Model Execution
print(f"Starting model run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
np.random.seed(42)

model_list_to_run = args.model_list

try:
    result_all = rmodel.model_run(
        args.json_file, 
        model_list_to_run, 
        args.path, 
        args.data_mode, 
        args.scenarios, 
        args.subscenario, 
        directory_path
    )
    print("Model run successfully completed.")
except Exception as e:
    print(f"An error occurred during model_run: {e}")
finally:
    sys.stdout = original_stdout
    output_file.close()
    print(f"Model execution complete. Output logged to {output_file_path}")