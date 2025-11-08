"""
Utility module for data preprocessing
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

def create_mock_preprocessors(models_dir='models'):
    """
    Create and save mock scaler and PCA for demo purposes
    In production, these should be the actual fitted preprocessors from training
    """
    # Create mock data with 61 features (57 transaction + 4 network)
    np.random.seed(42)
    n_samples = 1000
    n_features = 61
    
    mock_data = np.random.randn(n_samples, n_features)
    
    # Fit scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(mock_data)
    
    # Fit PCA
    pca = PCA(n_components=45, random_state=42)
    pca.fit(scaled_data)
    
    # Save preprocessors
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(models_dir, 'pca.pkl'))
    
    print(f"[OK] Mock preprocessors created and saved to {models_dir}/")
    return scaler, pca

@st.cache_resource
def load_preprocessors(models_dir='models'):
    """
    Load scaler and PCA preprocessors
    If not found, create mock ones for demo
    """
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    pca_path = os.path.join(models_dir, 'pca.pkl')
    
    try:
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        print(f"[OK] Preprocessors loaded from {models_dir}/")
    except FileNotFoundError:
        print("[WARNING] Preprocessors not found. Creating mock preprocessors for demo...")
        scaler, pca = create_mock_preprocessors(models_dir)
    
    return scaler, pca

def transform_data(df, scaler, pca):
    """
    Transform input data using scaler and PCA
    
    Args:
        df: DataFrame with raw features (61 columns)
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        
    Returns:
        DataFrame with transformed features (45 PCA components)
    """
    # Ensure we have numeric data only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle NaN and Inf values
    numeric_df = numeric_df.fillna(0)
    numeric_df = numeric_df.replace([np.inf, -np.inf], 0)
    
    # Scale
    scaled_data = scaler.transform(numeric_df)
    
    # PCA transform
    pca_data = pca.transform(scaled_data)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        pca_data,
        columns=[f'PC{i+1}' for i in range(pca_data.shape[1])],
        index=df.index
    )
    
    return pca_df

def get_all_feature_names():
    """Get all 61 feature names used in training (from combined_features.csv)"""
    return [
        # Aggregated transaction features (57 total)
        'Late_delivery_risk_mean', 'Late_delivery_risk_sum', 'Late_delivery_risk_std', 
        'Late_delivery_risk_min', 'Late_delivery_risk_max',
        'Benefit per order_mean', 'Benefit per order_sum', 'Benefit per order_std', 
        'Benefit per order_min', 'Benefit per order_max',
        'Order Profit Per Order_mean', 'Order Profit Per Order_sum', 'Order Profit Per Order_std', 
        'Order Profit Per Order_min', 'Order Profit Per Order_max',
        'Order Item Profit Ratio_mean', 'Order Item Profit Ratio_sum', 'Order Item Profit Ratio_std', 
        'Order Item Profit Ratio_min', 'Order Item Profit Ratio_max',
        'Sales_mean', 'Sales_sum', 'Sales_std', 'Sales_min', 'Sales_max',
        'Order Item Total_mean', 'Order Item Total_sum', 'Order Item Total_std', 
        'Order Item Total_min', 'Order Item Total_max',
        'Order Item Quantity_mean', 'Order Item Quantity_sum', 'Order Item Quantity_std', 
        'Order Item Quantity_min', 'Order Item Quantity_max',
        'Order Item Discount_mean', 'Order Item Discount_sum', 'Order Item Discount_std', 
        'Order Item Discount_min', 'Order Item Discount_max',
        'Order Item Discount Rate_mean', 'Order Item Discount Rate_sum', 'Order Item Discount Rate_std', 
        'Order Item Discount Rate_min', 'Order Item Discount Rate_max',
        'Days for shipping (real)_mean', 'Days for shipping (real)_sum', 'Days for shipping (real)_std', 
        'Days for shipping (real)_min', 'Days for shipping (real)_max',
        'Type_<lambda>', 'Delivery Status_<lambda>', 'Shipping Mode_<lambda>', 
        'Customer Segment_<lambda>', 'Market_<lambda>', 'Category Name_<lambda>', 'Department Name_<lambda>',
        # Network features (4 total)
        'degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'community_id'
    ]

def prepare_single_transaction(transaction_dict):
    """
    Prepare a single transaction dictionary for prediction
    
    Args:
        transaction_dict: Dictionary with transaction features
        
    Returns:
        DataFrame with single row ready for transformation (61 features)
    """
    # Create DataFrame from dictionary
    df = pd.DataFrame([transaction_dict])
    
    # Get all 61 features
    all_features = get_all_feature_names()
    
    # Fill missing features with 0
    for feat in all_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Select only the 61 features we need (in correct order)
    df = df[all_features]
    
    return df

def prepare_batch_transactions(uploaded_file):
    """
    Prepare batch transactions from uploaded CSV
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame ready for transformation (61 features)
    """
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Store original columns for later
    original_df = df.copy()
    
    # Get all 61 features
    all_features = get_all_feature_names()
    
    # Fill missing features with 0
    for feat in all_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Select features (in correct order)
    feature_df = df[all_features]
    
    return feature_df, original_df
