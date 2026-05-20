import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_and_clean_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # 1. Handle missing values for hotel dataset columns if present
    if 'company' in df.columns:
        df.drop('company', axis=1, inplace=True)
        
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0)
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)
        
    # Drop rows with 0 adults, 0 children, 0 babies if columns exist
    if all(col in df.columns for col in ['adults', 'children', 'babies']):
        zero_guests = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)
        df = df[~zero_guests]
        
    # Generic fallback: handle missing values for any custom dataset columns
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = df[col].fillna(df[col].median())
                except Exception:
                    df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                
    # 2. Drop leakage features if present
    for col in ['reservation_status', 'reservation_status_date']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    return df

def encode_and_scale(df, target_col='is_canceled'):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    # Label encode target variable to support XGBoost multiclass integer requirements
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
        
    # Ensure column names are strings and clean them for XGBoost compatibility later
    X_encoded.columns = X_encoded.columns.astype(str).str.replace('[', '{', regex=False).str.replace(']', '}', regex=False).str.replace('<', '', regex=False)
    
    return X_encoded, y

def preprocess_pipeline(filepath, target_col='is_canceled'):
    """Helper function for API programmatic execution."""
    df = load_and_clean_data(filepath)
    X, y = encode_and_scale(df, target_col)
    return X, y

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    input_filepath = os.path.join(data_dir, "hotel_bookings.csv")
    output_X_path = os.path.join(data_dir, "X_processed.csv")
    output_y_path = os.path.join(data_dir, "y_processed.csv")
    
    print("Loading and cleaning data...")
    df = load_and_clean_data(input_filepath)
    
    print("Encoding and scaling features...")
    X, y = encode_and_scale(df, target_col='is_canceled')
    
    print(f"Processed features shape: {X.shape}")
    print("Saving processed data...")
    X.to_csv(output_X_path, index=False)
    y.to_csv(output_y_path, index=False)
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()
