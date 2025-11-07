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
    """Get all 61 feature names used in training"""
    return [
        # Transaction features (57)
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        'Benefit per order', 'Sales per customer', 'Late_delivery_risk',
        'Order Item Discount', 'Order Item Discount Rate',
        'Order Item Product Price', 'Order Item Profit Ratio',
        'Order Item Quantity', 'Sales', 'Order Item Total',
        'Order Profit Per Order', 'product_popularity',
        'product_profit_margin', 'product_avg_discount',
        'customer_order_count', 'customer_total_spent',
        'customer_avg_order_value', 'customer_fraud_history',
        'recency_days', 'order_count_last_30d', 'total_spent_last_30d',
        'time_since_last_order', 'is_new_customer', 'rush_order',
        'unusual_quantity', 'high_discount_flag', 'negative_benefit',
        'international_order', 'high_value_order', 'high_risk_combination',
        # Additional transaction features (25 more to make 57 total)
        'order_item_cardprod_id', 'order_item_id', 'product_card_id',
        'product_category_id', 'days_diff', 'profit_ratio_per_item',
        'discount_amount', 'item_value', 'shipping_cost_est',
        'order_complexity', 'customer_lifetime_days', 'avg_days_between_orders',
        'order_frequency', 'preferred_category', 'preferred_department',
        'preferred_market', 'preferred_shipping', 'order_size_category',
        'price_sensitivity', 'discount_hunter', 'bulk_buyer',
        'weekend_shopper', 'holiday_shopper', 'cross_category_buyer',
        'return_risk',
        # Network features (4)
        'degree_centrality', 'betweenness_centrality',
        'closeness_centrality', 'pagerank'
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
