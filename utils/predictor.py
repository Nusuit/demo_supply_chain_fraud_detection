"""
Utility module for model prediction
"""

import streamlit as st
import numpy as np
import os
from tensorflow import keras
import tensorflow.keras.backend as K

# Custom loss functions (must be defined before loading models)
def focal_loss(gamma=2.0, alpha=0.75):
    """Focal Loss for imbalanced classification"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma) + \
                 (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        focal_loss_value = weight * cross_entropy
        return K.mean(focal_loss_value)
    return focal_loss_fixed

def cost_sensitive_focal_loss(gamma=1.0, alpha=0.75, fn_cost=10.0):
    """Cost-Sensitive Focal Loss - Heavily penalize False Negatives"""
    def cs_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow((1 - p_t), gamma)
        focal = alpha * focal_weight * cross_entropy
        fn_penalty = y_true * (1 - y_pred) * fn_cost
        total_loss = focal + fn_penalty
        return K.mean(total_loss)
    return cs_focal_loss_fixed

@st.cache_resource
def load_models(models_dir='models'):
    """
    Load the 3 ensemble models
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of 3 loaded Keras models
    """
    model_files = [
        'combined_model_seed42.keras',
        'combined_model_seed123.keras',
        'combined_model_seed456.keras'
    ]
    
    # CRITICAL: Register loss functions in custom_objects with EXACT names
    # Must match how they were saved during training
    custom_objects = {
        'cs_focal_loss_fixed': cost_sensitive_focal_loss(gamma=0.8, alpha=0.80, fn_cost=15.0),
        'focal_loss_fixed': focal_loss(gamma=1.5, alpha=0.65)
    }
    
    models = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        if os.path.exists(model_path):
            try:
                # Use compile=False to skip recompiling, then manually register custom objects
                model = keras.models.load_model(
                    model_path, 
                    custom_objects=custom_objects,
                    compile=False
                )
                # Recompile with custom loss
                model.compile(
                    optimizer='adam',
                    loss=cost_sensitive_focal_loss(gamma=0.8, alpha=0.80, fn_cost=15.0),
                    metrics=['accuracy']
                )
                models.append(model)
                print(f"✓ Loaded model: {model_file}")
            except Exception as e:
                st.warning(f"⚠ Could not load {model_file}: {str(e)}")
                # Create a mock model for demo
                models.append(create_mock_model())
        else:
            st.warning(f"⚠ Model file not found: {model_file}. Using mock model.")
            models.append(create_mock_model())
    
    if len(models) == 0:
        st.error("❌ No models could be loaded!")
        return None
    
    return models

def create_mock_model():
    """
    Create a mock model for demo purposes when actual models are not available
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=(45,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Initialize with random weights
    model.build((None, 45))
    
    return model

def predict_ensemble(df_processed, models):
    """
    Make predictions using ensemble of models
    
    Args:
        df_processed: DataFrame with PCA-transformed features (45 components)
        models: List of 3 Keras models
        
    Returns:
        Dictionary containing:
        - ensemble_scores: Average predictions from all models
        - individual_scores: List of predictions from each model
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
            pred = model.predict(X, verbose=0).flatten()
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
