import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lifelines.utils import concordance_index
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, make_scorer
import data_pre_classi_nontru as data_pre 
from typing import Dict, Any, List
import time
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


def initialize_model_from_params(keyword, raw_params): 
    """
    Initializes a model using parameters from hyperparameter tunning
    """
    
    # Mapping tables to safely convert hyperpar index back to values
    MAPPINGS = {
        'rf': {
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced', 'balanced_subsample'] 
        },
        'svc': {
            'class_weight': [None, 'balanced'],
            'kernel': ['linear', 'sigmoid', 'poly', 'rbf']
        },
        'knn': {
            'weights': ['uniform', 'distance']
        }
    }

    params = raw_params.copy()
    
    # Convert index back to values
    for param_name, options in MAPPINGS.get(keyword, {}).items():
        if param_name in params and isinstance(params[param_name], int) and params[param_name] < len(options):
             params[param_name] = options[params[param_name]]

    # Ensure p higher than 1 
    if keyword == "knn":
        params['p'] = max(1, params.get('p', 2))
    
    # Model Initialization
    if keyword == "xgb":
        params['objective'] = params.get('objective', 'binary:logistic')
        return xgb.XGBClassifier(random_state=32, use_label_encoder=False, eval_metric=['logloss', 'auc'], **params)
    
    elif keyword == "rf":
        return RandomForestClassifier(random_state=32, **params)
    
    elif keyword == "svc":
        return SVC(probability=True, random_state=32, **params)
    
    elif keyword == "knn":
        return KNeighborsClassifier(**params)
    
    raise ValueError(f"Model initialization failed for keyword: {keyword}")

def cross_validate_model(model, X, y, n_splits = 10):
    """
    Apply K-Fold cross-validation and return statisitc metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'auc': []}
    
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train_fold, y_train_fold)
        preds_fold = model.predict(X_test_fold)
        
        probs_fold = None
        if hasattr(model, "predict_proba"):
            probs_fold = model.predict_proba(X_test_fold)[:, 1] 
        
        # Add the metrics for this fold to the lists
        metrics['accuracy'].append(accuracy_score(y_test_fold, preds_fold))
        metrics['precision'].append(precision_score(y_test_fold, preds_fold, zero_division=0))
        metrics['recall'].append(recall_score(y_test_fold, preds_fold, zero_division=0))
        if probs_fold is not None:
             metrics['auc'].append(roc_auc_score(y_test_fold, probs_fold))
        
    # Compute average of the metrics
    averaged_metrics = {metric: np.mean(values) for metric, values in metrics.items() if values}
    return averaged_metrics

def bootstrap_auc(y_true, y_probs, n_bootstraps = 1000, ci = 95):
    """
    Perform bootstrapping to calculate the confidence interval for ROC-AUC.
    """
    bootstrapped_scores = []
    y_true_array = y_true.values
    y_probs_array = y_probs.values
    
    if y_probs_array is None:
        return [0.0, 0.0]
        
    for _ in range(n_bootstraps):
        
        indices = resample(np.arange(len(y_probs_array)), replace=True, random_state=np.random.randint(10000)) 
        # Check for class imbalance in the resample (Crucial for AUC)
        if len(np.unique(y_true_array[indices])) < 2:
            continue

        score = roc_auc_score(y_true_array[indices], y_probs_array[indices])
        bootstrapped_scores.append(score)
        
    if not bootstrapped_scores:
        return [0.0, 0.0] 
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Calculating the lower and upper bound of the confidence interval
    lower = np.percentile(sorted_scores, (100 - ci) / 2)
    upper = np.percentile(sorted_scores, 100 - (100 - ci) / 2)
    return [lower, upper] 

def data_prreparation(path, data_mode, scenarios, subscenario, directory_path): 
    """Wrapper function to call the data preparation module."""
    if data_mode == 1:
        return data_pre.data_one_record(path, scenarios)
    elif data_mode == 2:
        return data_pre.data_all(path, scenarios)
    elif data_mode == 3:
        return data_pre.data_two_record(path, scenarios, subscenario, directory_path)
    
    raise ValueError(f"Invalid data_mode: {data_mode}")

# ---------------------- Model Run Function -----------------------------------------------------

def model_run(json_file, model_list, path, data_mode, scenarios, subscenario, directory_path): 
    
    # 1. Data Preparation
    df, x_train, y_train, x_test, y_test = data_prreparation(path, data_mode, scenarios, subscenario, directory_path)

    # 2. Load hyperparameter tuning result
    try:
        with open(json_file, 'r') as f:
            dictionaries = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file}. Cannot proceed.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file}.")
        return []

    results_list = []
    
    for keyword in model_list:
        np.random.seed(32)
        print("==========================", keyword, "============================", sep=" ")
        
        # Extract best parameters
        try:
            params = next((result for result in dictionaries if result['Algo'] == keyword))["Best params"]
        except StopIteration:
            print(f"Warning: No hyperparameter results found for {keyword}. Skipping.")
            continue

        # 3. Model Initialization (using the helper)
        try:
            model = initialize_model_from_params(keyword, params)
        except ValueError as e:
            print(f"Error initializing model {keyword}: {e}. Skipping.")
            continue


        # 4. Cross-validation
        cv_metrics = cross_validate_model(model, x_train, y_train)
        print("Cross-validated metrics for", keyword, ":", cv_metrics)
        
        start_time = time.time()
        
        # 5. Final Training and Prediction
        model.fit(x_train, y_train)
        final_preds = model.predict(x_test)
        
        final_probs = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        marginal_pos = model.predict_proba(x_test)[:, 0] if hasattr(model, "predict_proba") else None
        
        end_time = time.time()
        
        # 6. Evaluation metrics for final predictions
        final_accuracy = accuracy_score(y_test, final_preds)
        final_auc = roc_auc_score(y_test, final_probs) if final_probs is not None else None

        # Calculate the confidence interval for the ROC-AUC score (FIXED to use probabilities)
        lower_bound, upper_bound = bootstrap_auc(y_test, pd.Series(final_probs) if final_probs is not None else y_test)

        print("Final Model Test Accuracy: %.4f" % final_accuracy)
        print("Final Model Test ROC AUC: %.4f" % final_auc)
        print("Concordance index: %.4f" % concordance_index(y_test, final_preds))
        print("precision: %.4f" % precision_score(y_test, final_preds, zero_division=0))
        print("recall: %.4f" % recall_score(y_test, final_preds, zero_division=0))
        print("AUC 95%% CI lower bound: %.4f" % lower_bound)
        print("AUC 95%% CI upper bound: %.4f" % upper_bound)
        print("running_time: %.4f seconds" % (end_time - start_time))

        # 7. Results
        data = x_test.copy()
        data["event"] = y_test
        data["pre"] = final_preds
        data["prob_pos"] = final_probs
        data["prob_neg"] = marginal_pos

        results = {
            "model": keyword,
            "cross_validation_metrics": cv_metrics,
            "final_test_accuracy": final_accuracy,
            "final_test_auc": final_auc
        }
        results_list.append(results)

        data.to_csv(f"{directory_path}/scenar{scenarios}_subscenar{subscenario}_result_{keyword}.csv", index=False)
    
    return results_list