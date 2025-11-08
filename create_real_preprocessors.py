"""
Script to create REAL scaler and PCA from training data
This will replace mock preprocessors with actual fitted ones
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

print("=" * 60)
print("Creating Real Preprocessors from Training Data")
print("=" * 60)

# Load combined_features.csv
combined_path = "d:/Code/School/UIT_kỳ_1_năm_4/Supply Chain Management/project/fraud_supplychain_year4/Fraud_SupplyChain/data/combined_features.csv"

print(f"\n1. Loading data from: {combined_path}")
df = pd.read_csv(combined_path)
print(f"   ✓ Loaded {len(df)} customers with {len(df.columns)} columns")

# Get feature columns (61 features: 57 aggregated + 4 network)
# Based on get_all_feature_names() from preprocessing.py
feature_names = [
    # Late_delivery_risk (5)
    'Late_delivery_risk_mean', 'Late_delivery_risk_sum', 'Late_delivery_risk_std', 
    'Late_delivery_risk_min', 'Late_delivery_risk_max',
    
    # Benefit per order (5)
    'Benefit per order_mean', 'Benefit per order_sum', 'Benefit per order_std',
    'Benefit per order_min', 'Benefit per order_max',
    
    # Order Profit Per Order (5)
    'Order Profit Per Order_mean', 'Order Profit Per Order_sum', 'Order Profit Per Order_std',
    'Order Profit Per Order_min', 'Order Profit Per Order_max',
    
    # Order Item Profit Ratio (5)
    'Order Item Profit Ratio_mean', 'Order Item Profit Ratio_sum', 'Order Item Profit Ratio_std',
    'Order Item Profit Ratio_min', 'Order Item Profit Ratio_max',
    
    # Sales (5)
    'Sales_mean', 'Sales_sum', 'Sales_std', 'Sales_min', 'Sales_max',
    
    # Order Item Total (5)
    'Order Item Total_mean', 'Order Item Total_sum', 'Order Item Total_std',
    'Order Item Total_min', 'Order Item Total_max',
    
    # Order Item Quantity (5)
    'Order Item Quantity_mean', 'Order Item Quantity_sum', 'Order Item Quantity_std',
    'Order Item Quantity_min', 'Order Item Quantity_max',
    
    # Order Item Discount (5)
    'Order Item Discount_mean', 'Order Item Discount_sum', 'Order Item Discount_std',
    'Order Item Discount_min', 'Order Item Discount_max',
    
    # Order Item Discount Rate (5)
    'Order Item Discount Rate_mean', 'Order Item Discount Rate_sum', 'Order Item Discount Rate_std',
    'Order Item Discount Rate_min', 'Order Item Discount Rate_max',
    
    # Days for shipping (real) (5)
    'Days for shipping (real)_mean', 'Days for shipping (real)_sum', 'Days for shipping (real)_std',
    'Days for shipping (real)_min', 'Days for shipping (real)_max',
    
    # Categorical features (7)
    'Type_<lambda>', 'Delivery Status_<lambda>', 'Shipping Mode_<lambda>',
    'Customer Segment_<lambda>', 'Market_<lambda>', 'Category Name_<lambda>',
    'Department Name_<lambda>',
    
    # Network features (4) - IMPORTANT: Include these!
    'degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'community_id'
]

print(f"\n2. Extracting {len(feature_names)} features (57 aggregated + 4 network)...")

# Check which features exist
missing = [f for f in feature_names if f not in df.columns]
if missing:
    print(f"   ⚠ WARNING: Missing features: {missing[:10]}...")
    print(f"   Total missing: {len(missing)}")
    # Use only available features
    feature_names = [f for f in feature_names if f in df.columns]
    print(f"   → Using {len(feature_names)} available features")

# Verify we have exactly 61 features
if len(feature_names) != 61:
    print(f"   ⚠ WARNING: Expected 61 features but have {len(feature_names)}")
    print(f"   This may cause shape mismatch errors!")

X = df[feature_names].values
print(f"   ✓ Feature matrix shape: {X.shape}")

# Handle missing values
if np.isnan(X).any():
    print(f"   ⚠ Found NaN values, filling with 0...")
    X = np.nan_to_num(X, nan=0.0)

# Create and fit StandardScaler
print(f"\n3. Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   ✓ Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
print(f"   ✓ Scaled data mean: {X_scaled.mean():.6f} (should be ~0)")
print(f"   ✓ Scaled data std: {X_scaled.std():.6f} (should be ~1)")

# Create and fit PCA
print(f"\n4. Fitting PCA with 45 components...")
pca = PCA(n_components=45, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"   ✓ PCA output shape: {X_pca.shape}")
print(f"   ✓ Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
print(f"   ✓ PCA range: [{X_pca.min():.2f}, {X_pca.max():.2f}]")

# Save preprocessors
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

scaler_path = os.path.join(output_dir, 'scaler.pkl')
pca_path = os.path.join(output_dir, 'pca.pkl')

print(f"\n5. Saving preprocessors...")
joblib.dump(scaler, scaler_path)
print(f"   ✓ Saved StandardScaler to: {scaler_path}")

joblib.dump(pca, pca_path)
print(f"   ✓ Saved PCA to: {pca_path}")

# Verify by loading back
print(f"\n6. Verification...")
scaler_loaded = joblib.load(scaler_path)
pca_loaded = joblib.load(pca_path)
print(f"   ✓ Successfully loaded scaler")
print(f"   ✓ Successfully loaded PCA")

# Test on first row
X_test = df[feature_names].iloc[:1].values
X_test = np.nan_to_num(X_test, nan=0.0)
X_test_scaled = scaler_loaded.transform(X_test)
X_test_pca = pca_loaded.transform(X_test_scaled)
print(f"   ✓ Test transform successful: {X_test_pca.shape}")

print("\n" + "=" * 60)
print("✅ Real preprocessors created successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Restart Streamlit app (Ctrl+C then rerun)")
print("2. Test with example_fraud_customer.csv")
print("3. Should now see accurate predictions!")
print("\nNote: Old cached preprocessors will be replaced on next load")
