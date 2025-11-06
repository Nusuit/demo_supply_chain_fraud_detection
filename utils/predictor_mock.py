"""
Mock predictor - Temporary solution when TensorFlow DLL fails
Uses simple logistic regression instead of deep learning
"""

import streamlit as st
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def load_models(models_dir='models'):
    """
    Load mock models (logistic regression) when TensorFlow fails
    
    Returns:
        List of 3 mock models
    """
    st.warning("⚠️ TensorFlow unavailable - Using mock models for demo")
    st.info("💡 To use real deep learning models, install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
    # Create 3 simple logistic regression models with different random states
    models = []
    for seed in [42, 123, 456]:
        model = LogisticRegression(random_state=seed, max_iter=100)
        # Mock fit with dummy data
        X_dummy = np.random.rand(100, 45)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        models.append(model)
        print(f"✓ Created mock model with seed {seed}")
    
    return models

def predict_ensemble(df_processed, models):
    """
    Make predictions using ensemble of mock models
    
    Args:
        df_processed: DataFrame with PCA-transformed features (45 components)
        models: List of 3 sklearn models
        
    Returns:
        Dictionary containing ensemble and individual predictions
    """
    if models is None or len(models) == 0:
        st.error("❌ No models available for prediction!")
        return None
    
    # Convert to numpy array
    X = df_processed.values
    
    # Get predictions from each model
    individual_predictions = []
    
    for i, model in enumerate(models):
        try:
            # Use predict_proba for probability scores
            pred = model.predict_proba(X)[:, 1]  # Probability of fraud class
            individual_predictions.append(pred)
        except Exception as e:
            st.warning(f"⚠ Error predicting with model {i+1}: {str(e)}")
            # Use random predictions as fallback
            pred = np.random.uniform(0, 1, size=len(X))
            individual_predictions.append(pred)
    
    # Ensemble: average of all predictions
    ensemble_scores = np.mean(individual_predictions, axis=0)
    
    return {
        'ensemble_scores': ensemble_scores,
        'individual_scores': individual_predictions
    }

def predict_single(transaction_df, models, scaler, pca):
    """
    Make prediction for a single transaction
    
    Args:
        transaction_df: DataFrame with single transaction (raw features)
        models: List of loaded models
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        
    Returns:
        Dictionary with prediction results
    """
    from .preprocessing import transform_data
    
    # Transform data
    df_processed = transform_data(transaction_df, scaler, pca)
    
    # Get predictions
    predictions = predict_ensemble(df_processed, models)
    
    if predictions is None:
        return None
    
    # Extract single values
    ensemble_score = predictions['ensemble_scores'][0]
    individual_scores = [scores[0] for scores in predictions['individual_scores']]
    
    return {
        'ensemble_score': ensemble_score,
        'model_1_score': individual_scores[0] if len(individual_scores) > 0 else 0.0,
        'model_2_score': individual_scores[1] if len(individual_scores) > 1 else 0.0,
        'model_3_score': individual_scores[2] if len(individual_scores) > 2 else 0.0,
    }

def predict_batch(features_df, models, scaler, pca):
    """
    Make predictions for batch of transactions
    
    Args:
        features_df: DataFrame with transaction features (raw)
        models: List of loaded models
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        
    Returns:
        numpy array of ensemble scores
    """
    from .preprocessing import transform_data
    
    # Transform data
    df_processed = transform_data(features_df, scaler, pca)
    
    # Get predictions
    predictions = predict_ensemble(df_processed, models)
    
    if predictions is None:
        return None
    
    return predictions['ensemble_scores']
