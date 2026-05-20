import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def get_filter_methods(X, y, k=20):
    print("Running Filter Methods...")
    methods = {}
    actual_k = min(k, X.shape[1])
    
    # ANOVA F-value
    selector_anova = SelectKBest(f_classif, k=actual_k)
    selector_anova.fit(X, y)
    methods['ANOVA'] = list(X.columns[selector_anova.get_support()])
    
    # Chi-Square (Needs non-negative values, so we apply MinMaxScaler first to the scaled features)
    scaler = MinMaxScaler()
    X_minmax = scaler.fit_transform(X)
    selector_chi2 = SelectKBest(chi2, k=actual_k)
    selector_chi2.fit(X_minmax, y)
    methods['Chi-Square'] = list(X.columns[selector_chi2.get_support()])
    
    return methods

def get_wrapper_methods(X, y, k=20):
    print("Running Wrapper Methods (RFE)... This may take a while.")
    methods = {}
    actual_k = min(k, X.shape[1])
    
    # RFE can be very slow. Using a random sample for RFE if dataset is too large.
    sample_size = min(20000, len(X))
    if sample_size < len(X):
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[idx]
    else:
        X_sample, y_sample = X, y
        
    # Using Logistic Regression for speed in RFE instead of Random Forest
    # lbfgs solver natively supports multiclass, whereas liblinear does not.
    is_multiclass = len(np.unique(y_sample)) > 2
    if is_multiclass:
        estimator = LogisticRegression(solver='lbfgs', max_iter=100)
    else:
        estimator = LogisticRegression(solver='liblinear', max_iter=100)
        
    selector_rfe = RFE(estimator, n_features_to_select=actual_k, step=10)
    selector_rfe.fit(X_sample, y_sample)
    methods['RFE_LogisticRegression'] = list(X.columns[selector_rfe.get_support()])
    
    return methods

def get_embedded_methods(X, y, k=20):
    print("Running Embedded Methods...")
    methods = {}
    actual_k = min(k, X.shape[1])
    
    # Lasso (L1 Penalty)
    # liblinear doesn't support multiclass directly in newer scikit-learn versions, so we wrap it in OneVsRestClassifier
    is_multiclass = len(np.unique(y)) > 2
    if is_multiclass:
        from sklearn.multiclass import OneVsRestClassifier
        lasso = OneVsRestClassifier(LogisticRegression(penalty='l1', solver='liblinear', max_iter=100))
    else:
        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100)
        
    selector_lasso = SelectFromModel(lasso, max_features=actual_k)
    selector_lasso.fit(X, y)
    methods['Lasso'] = list(X.columns[selector_lasso.get_support()])
    
    # Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    selector_rf = SelectFromModel(rf, max_features=actual_k)
    selector_rf.fit(X, y)
    methods['RandomForest_Importance'] = list(X.columns[selector_rf.get_support()])
    
    return methods

def run_feature_selection_pipeline(X, y, k=20):
    """Programmatic API pipeline entrypoint."""
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    actual_k = min(k, X.shape[1])
    selected_features = {}
    selected_features.update(get_filter_methods(X, y, k=actual_k))
    selected_features.update(get_wrapper_methods(X, y, k=actual_k))
    selected_features.update(get_embedded_methods(X, y, k=actual_k))
    return selected_features

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    X_path = os.path.join(data_dir, "X_processed.csv")
    y_path = os.path.join(data_dir, "y_processed.csv")
    
    print("Loading preprocessed data...")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
        
    # Number of features to select
    K = 20
    print(f"Total features: {X.shape[1]}. Selecting top {K} features per method.")
    
    selected_features = run_feature_selection_pipeline(X, y, k=K)
    
    output_path = os.path.join(data_dir, "selected_features.json")
    with open(output_path, 'w') as f:
        json.dump(selected_features, f, indent=4)
        
    print(f"Feature selection complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
