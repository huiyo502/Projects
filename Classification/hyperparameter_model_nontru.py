import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from lifelines import CoxPHFitter
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import data_pre_classi_nontru as data_pre 
from datetime import datetime
import argparse
import json
import os
import sys
from typing import Dict, Any, List 

np.random.seed(42)

parser = argparse.ArgumentParser(description='Hyperparameter Tuning Script')
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--json_file', type=str, default="hyperparams_best.json") 
parser.add_argument('--data_mode', type=int, default=1)
parser.add_argument('--scenarios', type=int, default=1)
parser.add_argument('--subscenario', type=int, default=1)
args = parser.parse_args()

# Define directory path
TODAY_STR = datetime.now().strftime('%m/%d')
directory_path = f"../Result/{TODAY_STR}/hyper_noupdate_ONLYTTP028/mod{args.data_mode}_scen{args.scenarios}_subscen{args.subscenario}"

# Create folder
try:
    os.makedirs(directory_path, exist_ok=True) 
    print(f"Directory '{directory_path}' created successfully.")
except Exception as e:
    print(f"Error creating directory '{directory_path}': {e}")


# --- Data Preparation ---
def data_prreparation(path, data_mode, scenarios, subscenario, directory_path): 
    """Wrapper function to call the data preparation module."""
    if data_mode == 1:
        return data_pre.data_one_record(path, scenarios)
    elif data_mode == 2:
        return data_pre.data_all(path, scenarios)
    elif data_mode == 3:
        return data_pre.data_two_record(path, scenarios, subscenario, directory_path)
    
    raise ValueError(f"Invalid data_mode: {data_mode}")

# Prepare data
_, x_train, y_train, x_test, y_test = data_prreparation(args.path, args.data_mode, args.scenarios, args.subscenario, directory_path)

# Models dictionary
models = {
    'logistic_regression' : LogisticRegression,
    'rf' : RandomForestClassifier,
    'knn' : KNeighborsClassifier,
    'svc' : SVC,
    "xgb" : xgb.XGBClassifier,
    "lasso" : Lasso,
    "cox": CoxPHFitter 
}

# -- Search Space Definition ---
def search_space(model): 
    """Defines the hyperparameter search space for each model."""
    space = {}
    
    # !!!! NOTE: hp.choice returns an index (0, 1, 2...), which must be mapped back in the model_run script.
    if model == 'knn': 
        space = { 
            'n_neighbors': hp.choice('n_neighbors', range(1, 100)), 
            'weights': hp.choice("weights", ['uniform', "distance"]), 
            'p': hp.choice('p', [1, 2]) 
        } 
    elif model == 'svc': 
        space = { 
            'C': hp.uniform('C', 0, 20), 
            'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']), 
            'gamma': hp.uniform('gamma', 0, 20),
            'class_weight': hp.choice('class_weight', [None, 'balanced']) 
        } 
    elif model == 'logistic_regression': 
        space = { 
            'warm_start' : hp.choice('warm_start', [True, False]), 
            'fit_intercept' : hp.choice('fit_intercept', [True, False]), 
            'tol' : hp.uniform('tol', 0.00001, 0.0001), 
            'C' : hp.uniform('C', 0.05, 3), 
            'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']), 
            'max_iter' : hp.choice('max_iter', range(100, 1000)), 
            'scale': hp.choice('scale', [0, 1]), 
            'normalize': hp.choice('normalize', [0, 1]), 
            'multi_class' : 'auto', 
            'class_weight' : 'balanced' 
        } 
    elif model == 'rf': 
        space = { 
            'max_depth': hp.choice('max_depth', range(1, 20)), 
            'max_features': "auto", 
            'n_estimators': hp.choice('n_estimators', range(10, 50)), 
            'criterion': hp.choice('criterion', ["gini", "entropy"]), 
            'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']) 
        } 
    elif model == "xgb": 
        space = { 
            'objective': 'binary:logistic', 
            'max_depth': hp.choice("max_depth", range(1, 10)), 
            'learning_rate': hp.uniform("learning_rate", 0.01, 0.2), 
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100), 
            'subsample': hp.uniform('subsample', 0.5, 1), 
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1) 
        } 
    elif model == "lasso":
         space = { 
             'alpha': hp.loguniform('alpha', np.log(0.001), np.log(10.0)), 
             'tol': hp.loguniform('tol', np.log(0.00001), np.log(0.001))
         }
    
    space['model'] = model
    return space

# --- Optimization Objective Function ---

def get_recall_status(clf, X, y):
    """Evaluates model performance using Recall"""

    # SVC needs probability=True for cross_val_score on recall
    if isinstance(clf, SVC):
        clf.set_params(probability=True) 

    recall_scorer = make_scorer(recall_score, zero_division=0)

    # Using 5-fold cross-validation
    recall = cross_val_score(clf, X, y, cv=5, scoring=recall_scorer, error_score='raise').mean()
    return {'loss': -recall, 'status': STATUS_OK} 

def obj_fnc(params): 
    model_name = params.get('model')
    X_ = x_train.copy()
    
    # Remove key before passing to the classifier constructor
    clf_params = {k: v for k, v in params.items() if k != 'model'}
    
    # Index for categorical parameters 
    MAPPINGS = {
        'rf': {'criterion': ['gini', 'entropy'], 'class_weight': [None, 'balanced', 'balanced_subsample']},
        'svc': {'class_weight': [None, 'balanced'], 'kernel': ['linear', 'sigmoid', 'poly', 'rbf']},
        'knn': {'weights': ['uniform', 'distance']}
    }
    
    for param_name, options in MAPPINGS.get(model_name, {}).items():
        if param_name in clf_params and isinstance(clf_params[param_name], int) and clf_params[param_name] < len(options):
             clf_params[param_name] = options[clf_params[param_name]]
    
    # Specific fix for KNN p parameter
    if model_name == "knn":
         clf_params['p'] = max(1, clf_params.get('p', 2))

    try:
        clf = models[model_name](**clf_params)
        return get_recall_status(clf, X_, y_train)
    except Exception as e:
        print(f"Error initializing or running model {model_name} with params {clf_params}: {e}")
        return {'loss': 1.0, 'status': STATUS_OK} # Return max loss if initialization fails


# --- JSON Serialization ---
def convert_to_json_serializable(obj): 
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    
# --- Main Tuning Loop ---
model_list = [  "xgb", "knn", "rf", "svc"]
MAX_EVALS = 50 

existing_data = []

for model in model_list:
    print(f"--- Starting hyperparameter tuning for {model} (Max Evals: {MAX_EVALS}) ---")
    hypopt_trials = Trials()
    
    # Find the best parameters
    best_params = fmin(
        fn=obj_fnc, 
        space=search_space(model), 
        algo=tpe.suggest, 
        max_evals=MAX_EVALS, 
        trials=hypopt_trials,
        rstate=np.random.RandomState(42)
    )

    best_trial_result = hypopt_trials.best_trial['result']
    new_dict = {
        "Algo": model,
        "Accuracy": -best_trial_result['loss'], 
        "Best params": best_params
    }
    
    print(new_dict)

    # Load existing data (robustly)
    json_path = f'{directory_path}/{args.json_file}'
    try:
        with open(json_path, "r") as file:
            file_content = file.read().strip()
            if file_content:
                existing_data = json.loads(file_content)
                if not isinstance(existing_data, list):
                    existing_data = [] 
    except FileNotFoundError:
        pass  
    except json.JSONDecodeError:
        pass  

    # Append the new best result (and potentially replace an existing one for the same Algo)
    existing_data = [d for d in existing_data if d.get('Algo') != model]
    existing_data.append(new_dict)

    # Write updated data back to JSON file
    with open(json_path, "w") as file:
        json.dump(existing_data, file, indent=2, default=convert_to_json_serializable)
        
print("Hyperparameter tuning complete. Results saved to JSON.")