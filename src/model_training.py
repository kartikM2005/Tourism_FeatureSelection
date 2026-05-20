import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(X_train, X_test, y_train, y_test, method_name, is_multiclass=False):
    results = {}
    avg_method = 'weighted' if is_multiclass else 'binary'
    
    # Model 1: Logistic Regression
    start_time = time.time()
    lr = LogisticRegression(max_iter=500, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_time = time.time() - start_time
    
    results['Logistic Regression'] = {
        'Accuracy': float(accuracy_score(y_test, lr_pred)),
        'F1 Score': float(f1_score(y_test, lr_pred, average=avg_method)),
        'Time (s)': float(lr_time)
    }
    
    # Model 2: XGBoost
    start_time = time.time()
    metric = 'mlogloss' if is_multiclass else 'logloss'
    xgb = XGBClassifier(eval_metric=metric, n_jobs=-1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_time = time.time() - start_time
    
    results['XGBoost'] = {
        'Accuracy': float(accuracy_score(y_test, xgb_pred)),
        'F1 Score': float(f1_score(y_test, xgb_pred, average=avg_method)),
        'Time (s)': float(xgb_time)
    }
    
    return results

def run_model_training_pipeline(X, y, selected_features):
    """Programmatic API pipeline entrypoint."""
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    is_multiclass = len(np.unique(y)) > 2
    
    all_results = {}
    all_results['Baseline'] = train_and_evaluate(X_train_full, X_test_full, y_train, y_test, "Baseline", is_multiclass)
    
    for method, features in selected_features.items():
        valid_features = [f for f in features if f in X_train_full.columns]
        if len(valid_features) == 0:
            continue
            
        X_train_subset = X_train_full[valid_features]
        X_test_subset = X_test_full[valid_features]
        
        all_results[method] = train_and_evaluate(X_train_subset, X_test_subset, y_train, y_test, method, is_multiclass)
        
    return all_results

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_path = os.path.join(data_dir, "X_processed.csv")
    y_path = os.path.join(data_dir, "y_processed.csv")
    features_path = os.path.join(data_dir, "selected_features.json")
    
    print("Loading data and selected features...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
        
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    is_multiclass = len(np.unique(y)) > 2
    
    all_results = {}
    
    print("Evaluating Baseline (All Features)...")
    all_results['Baseline'] = train_and_evaluate(X_train_full, X_test_full, y_train, y_test, "Baseline", is_multiclass)
    
    for method, features in selected_features.items():
        print(f"Evaluating {method} ({len(features)} features)...")
        valid_features = [f for f in features if f in X_train_full.columns]
        
        X_train_subset = X_train_full[valid_features]
        X_test_subset = X_test_full[valid_features]
        
        all_results[method] = train_and_evaluate(X_train_subset, X_test_subset, y_train, y_test, method, is_multiclass)
        
    output_path = os.path.join(data_dir, "model_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Model evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
