"""
Page 1 - Prediction Fraud
Live prediction for single or batch transactions
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from utils.constants import get_risk_level, get_fraud_prediction, MODEL_THRESHOLD
from utils import get_preprocessing_module, get_predictor_module
from utils.styling import inject_custom_css

# Get modules
preprocessing = get_preprocessing_module()
predictor = get_predictor_module()

# Extract functions
load_preprocessors = preprocessing.load_preprocessors
prepare_single_transaction = preprocessing.prepare_single_transaction
prepare_batch_transactions = preprocessing.prepare_batch_transactions
load_models = predictor.load_models
predict_single = predictor.predict_single
predict_batch = predictor.predict_batch

# Page configuration
st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="",
    layout="wide"
)

# Apply custom monochrome CSS
inject_custom_css()

# Additional monochrome styling for prediction page
st.markdown("""
<style>
/* Monochrome progress bar */
.monochrome-bar {
height: 10px;
background-color: #E0E0E0;
border-radius: 9999px;
overflow: hidden;
margin: 0.5rem 0;
}
.monochrome-bar-fill {
height: 100%;
background-color: #2196F3;
transition: width 0.3s ease;
}

/* Hide expander arrow icon text */
.streamlit-expander .st-emotion-cache-1gulkj5 {
    display: none;
}
details summary p {
    display: inline;
}
details summary::before {
    content: "" !important;
}

/* Tab styling - Enhanced monochrome design */
.stTabs [data-baseweb="tab-list"] {
gap: 0.5rem;
background-color: #FAFAFA;
padding: 0.75rem 1.5rem;
border-bottom: 2px solid #EDEDED;
border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
color: #1565C0;
font-size: 1rem;
font-weight: 500;
padding: 0.75rem 1.5rem;
background-color: transparent;
border-radius: 10px;
border: 1px solid transparent;
transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
background-color: white;
color: #1565C0;
border: 1px solid #BBDEFB;
transform: translateY(-1px);
box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.stTabs [aria-selected="true"] {
color: #FFFFFF !important;
font-weight: 700;
background-color: #1565C0;
border: 1px solid #1565C0;
box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.stTabs [aria-selected="true"]:hover {
background-color: #404040;
color: #FFFFFF !important;
border: 1px solid #404040;
transform: translateY(-1px);
}
.stTabs [aria-selected="true"] button {
color: #FFFFFF !important;
}

/* Form inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
background-color: white;
border: 1px solid #BBDEFB;
border-radius: 8px;
color: #1565C0;
}

/* Buttons - Enhanced visibility */
.stButton button {
border-radius: 8px;
font-weight: 700;
font-size: 1rem;
border: 2px solid #2196F3;
transition: all 0.2s ease;
}
.stButton button[kind="primary"],
.stButton > button[type="submit"]:not([kind="secondary"]) {
background-color: #2196F3 !important;
color: #FFFFFF !important;
border: 2px solid #2196F3 !important;
}
.stButton button[kind="primary"]:hover,
.stButton > button[type="submit"]:not([kind="secondary"]):hover {
background-color: #1976D2 !important;
color: #FFFFFF !important;
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stButton button[kind="secondary"] {
background-color: white !important;
color: #2196F3 !important;
border: 2px solid #BBDEFB !important;
}
.stButton button[kind="secondary"]:hover {
background-color: #E3F2FD !important;
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Force form submit button text to be white */
button[type="submit"] {
color: #FFFFFF !important;
}
div[data-testid="stFormSubmitButton"] button {
color: #FFFFFF !important;
}
div[data-testid="stFormSubmitButton"] button p {
color: #FFFFFF !important;
}

/* Force ALL primary button text to be white - including "Predict All" */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:active,
.stButton > button[kind="primary"]:focus,
.stButton button[data-baseweb="button"][kind="primary"],
button[kind="primary"] {
    color: #FFFFFF !important;
    background-color: #2196F3 !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1976D2 !important;
}

/* Force white color for all text inside primary buttons */
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<header style="margin-bottom: 2rem;">
<h1 style="font-size: 2.5rem; font-weight: 800; color: #1565C0; letter-spacing: -0.02em; margin-bottom: 0.5rem;">
Fraud Prediction
</h1>
<h4 style="font-size: 1.125rem; font-weight: 600; color: #1565C0; margin: 0;">
Input and analysis tools for fraud detection models.
</h4>
</header>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_resources():
    models = load_models('models')
    scaler, pca = load_preprocessors('models')
    return models, scaler, pca

with st.spinner("Loading models and preprocessors..."):
    models, scaler, pca = load_resources()

if models is None:
    st.markdown("""
    <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;">
    <strong>Error:</strong> Cannot load models! Please check the models/ directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="background: white; padding: 0.75rem; border-radius: 8px; border: 1px solid #BBDEFB; margin-bottom: 1.5rem;">
<span style="color: #1976D2; font-weight: 600;">Models and preprocessors are ready</span>
</div>
""", unsafe_allow_html=True)

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["Single Prediction", "Batch (CSV/XLSX)"])

# ==================== TAB 1: Single Prediction ====================
with tab1:
    st.markdown("""
<h2 style="font-size: 1.875rem; font-weight: 800; color: #1565C0; margin-bottom: 0.5rem;">
Single Customer Prediction
</h2>
<p style="font-size: 0.875rem; color: #1565C0; margin-bottom: 1rem;">
Enter Customer ID to load aggregated features from training data and predict fraud probability.
</p>
""", unsafe_allow_html=True)

    # Info about model requirement
    st.markdown("""
<div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 1.5rem;">
<span style="color: #1565C0; font-weight: 600;">Accurate Prediction Method</span>
<p style="color: #1976D2; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
The model uses <strong>61 aggregated features per customer</strong> (mean, sum, std, min, max from multiple transactions).<br>
Enter a Customer ID to load their real aggregated data from <code>combined_features.csv</code> for accurate predictions.
</p>
</div>
""", unsafe_allow_html=True)

    # Load combined_features.csv
    @st.cache_data
    def load_combined_features():
        from pathlib import Path
        combined_path = Path(__file__).parent.parent.parent / "fraud_supplychain_year4" / "Fraud_SupplyChain" / "data" / "combined_features.csv"
        if combined_path.exists():
            df = pd.read_csv(combined_path)
            # Rename is_fraud to Fraud for consistency
            if 'is_fraud' in df.columns:
                df = df.rename(columns={'is_fraud': 'Fraud'})
            return df
        return None
    
    combined_df = load_combined_features()
    
    if combined_df is None:
        st.error("""
        **Error:** Cannot load combined_features.csv!
        
        This file contains aggregated customer data required for accurate predictions.
        Please ensure the file exists at: `fraud_supplychain_year4/Fraud_SupplyChain/data/combined_features.csv`
        """)
        st.info("**Alternative:** Use the 'Batch (CSV/XLSX)' tab to upload your own aggregated features file.")
    else:
        # Show available customer IDs
        if 'Customer Id' in combined_df.columns:
            available_customers = combined_df['Customer Id'].unique()
            
            # Use styled HTML instead of st.info to avoid black text
            st.markdown(f"""
            <div style="background-color: #E3F2FD; border-left: 4px solid #2196F3; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; color: #1565C0; font-size: 0.9375rem;">
                ✓ Loaded <strong>{len(combined_df)}</strong> customers with <strong>61</strong> aggregated features
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View sample Customer IDs", expanded=False):
                st.markdown(f"""
                <p style="color: #1565C0; font-size: 0.9375rem; margin-bottom: 0.5rem;">
                    Total unique customers: <strong>{len(available_customers):,}</strong>
                </p>
                """, unsafe_allow_html=True)
                st.write("**Sample IDs:**", available_customers[:20].tolist())
        else:
            available_customers = combined_df.index.unique()
            st.warning("No 'Customer Id' column found. Using row index.")
        
        with st.form("single_prediction_form"):
            st.markdown("""
            <h3 style="font-size: 1.125rem; font-weight: 600; color: #1565C0; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
            Enter Customer ID
            </h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'Customer Id' in combined_df.columns:
                    customer_id = st.number_input(
                        "Customer ID",
                        min_value=int(available_customers.min()),
                        max_value=int(available_customers.max()),
                        value=int(available_customers[0]),
                        step=1,
                        help="Enter a Customer ID from the training dataset"
                    )
                else:
                    customer_id = st.number_input(
                        "Row Index",
                        min_value=0,
                        max_value=len(combined_df)-1,
                        value=0,
                        step=1,
                        help="Enter row index (0 to {})".format(len(combined_df)-1)
                    )
            
            with col2:
                st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
                random_btn = st.form_submit_button("Random Customer", use_container_width=True, type="secondary")
            
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            
            # Action Buttons
            col_reset, col_predict = st.columns([1, 1])
            with col_reset:
                reset_btn = st.form_submit_button("Reset", use_container_width=True, type="secondary")
            with col_predict:
                submitted = st.form_submit_button("Predict Fraud", use_container_width=True, type="primary")
            
            if random_btn:
                import numpy as np
                customer_id = int(np.random.choice(available_customers))
                st.rerun()
            
            if submitted:
                # Find customer data
                if 'Customer Id' in combined_df.columns:
                    customer_data = combined_df[combined_df['Customer Id'] == customer_id]
                else:
                    customer_data = combined_df.iloc[[customer_id]]
                
                if len(customer_data) == 0:
                    st.error(f"Customer ID {customer_id} not found in dataset!")
                    st.stop()
                
                # Get customer features (exclude Customer Id and Fraud columns)
                feature_cols = [col for col in customer_data.columns if col not in ['Customer Id', 'Fraud', 'is_fraud']]
                customer_features = customer_data[feature_cols]
                
                # Show customer info
                st.markdown('<div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #EDEDED;"</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #1976D2; margin-bottom: 1rem;">
                Customer Information
                </h3>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Customer ID", customer_id)
                with col2:
                    st.metric("Features", len(feature_cols))
                with col3:
                    if 'Fraud' in customer_data.columns:
                        actual_fraud = customer_data['Fraud'].iloc[0]
                        st.metric("Actual Label", "FRAUD" if actual_fraud == 1 else "SAFE")
                    else:
                        st.metric("Actual Label", "Unknown")
                
                # Show preview of features
                with st.expander("View Customer Features (61 aggregated features)"):
                    st.dataframe(customer_features.T, use_container_width=True)
                
                # Prepare for prediction
                df = prepare_single_transaction(customer_features.iloc[0].to_dict())

                # Predict
                with st.spinner("Predicting..."):
                    prediction_result = predict_single(df, models, scaler, pca)

                if prediction_result:
                    st.markdown('<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #EDEDED;"></div>', unsafe_allow_html=True)

                    st.markdown("""
                    <h3 style="font-size: 1.5rem; font-weight: 700; color: #1976D2; margin-bottom: 1.5rem;">
                    Prediction Result
                    </h3>
                    """, unsafe_allow_html=True)

                    # Get scores
                    ensemble_score = prediction_result['ensemble_score']
                    model_1 = prediction_result['model_1_score']
                    model_2 = prediction_result['model_2_score']
                    model_3 = prediction_result['model_3_score']

                    # Determine if fraud
                    is_fraud = ensemble_score >= MODEL_THRESHOLD

                    if is_fraud:
                        # FRAUD DETECTED
                        result_card = f"""
<div style="padding: 1.5rem; background: #E3F2FD; border-left: 4px solid #2196F3; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<h4 style="font-size: 1.25rem; font-weight: 800; color: #1565C0; margin-bottom: 1rem;">
FRAUD DETECTED
</h4>

<!-- Probability Row -->
<div style="margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
<span>Fraud Probability</span>
<span style="color: #1565C0; font-weight: 800;">{ensemble_score*100:.1f}%</span>
</div>
<div class="monochrome-bar">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>

<!-- Risk Badge -->
<span style="display: inline-block; padding: 0.25rem 0.75rem; background: #2196F3; color: white; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
High Risk
</span>

<!-- Recommendations -->
<div style="margin-top: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #EDEDED;">
<ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem; color: #1565C0;">
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #2196F3; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Temporarily Block Transaction
</li>
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #2196F3; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Manual Investigation Required
</li>
<li style="display: flex; align-items: flex-start;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #2196F3; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Contact Customer for Verification
</li>
</ul>
</div>

<!-- Prediction Details -->
<div style="margin-top: 1rem; padding: 1rem; border: 1px solid #BBDEFB; border-radius: 8px; background: white;">
<h5 style="font-size: 0.875rem; font-weight: 600; color: #1565C0; margin-bottom: 0.75rem;">
Prediction Details (3 models + ensemble)
</h5>

<div style="font-size: 0.875rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<span style="flex: 1;">Model 1 (seed 42):</span>
<span style="font-weight: 700; margin: 0 1rem;">{model_1*100:.1f}%</span>
<div class="monochrome-bar" style="flex: 1;">
<div class="monochrome-bar-fill" style="width: {model_1*100:.1f}%;"></div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<span style="flex: 1;">Model 2 (seed 123):</span>
<span style="font-weight: 700; margin: 0 1rem;">{model_2*100:.1f}%</span>
<div class="monochrome-bar" style="flex: 1;">
<div class="monochrome-bar-fill" style="width: {model_2*100:.1f}%;"></div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<span style="flex: 1;">Model 3 (seed 456):</span>
<span style="font-weight: 700; margin: 0 1rem;">{model_3*100:.1f}%</span>
<div class="monochrome-bar" style="flex: 1;">
<div class="monochrome-bar-fill" style="width: {model_3*100:.1f}%;"></div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 0.5rem; border-top: 1px solid rgba(237,237,237,0.5);">
<span style="flex: 1; font-weight: 800; color: #1565C0;">Ensemble (average):</span>
<span style="font-weight: 800; color: #1565C0; margin: 0 1rem;">{ensemble_score*100:.1f}%</span>
<div class="monochrome-bar" style="flex: 1;">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>
</div>
</div>

<p style="font-size: 0.75rem; color: #1565C0; margin-top: 1rem; opacity: 0.8;">
Threshold = {MODEL_THRESHOLD:.2f}. Probability > {MODEL_THRESHOLD*100:.0f}% → labeled as FRAUD.
</p>
</div>
"""
                else:
                    # NOT FRAUD
                    result_card = f"""
<div style="padding: 1.5rem; background: #E3F2FD; border-left: 4px solid #EDEDED; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<h4 style="font-size: 1.25rem; font-weight: 800; color: #1565C0; margin-bottom: 1rem;">
NO FRAUD DETECTED
</h4>

<!-- Probability Row -->
<div style="margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
<span>Fraud Probability</span>
<span style="color: #1565C0; font-weight: 800;">{ensemble_score*100:.1f}%</span>
</div>
<div class="monochrome-bar">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>

<!-- Risk Badge -->
<span style="display: inline-block; padding: 0.25rem 0.75rem; border: 1px solid #2196F3; color: #1565C0; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
Low Risk
</span>

<!-- Recommendations -->
<div style="margin-top: 1rem;">
<ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem; color: #1565C0;">
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #EDEDED; border: 1px solid #2196F3; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Allow Transaction
</li>
<li style="display: flex; align-items: flex-start;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #EDEDED; border: 1px solid #2196F3; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Proceed as Normal
</li>
</ul>
</div>
</div>
"""

                    st.markdown(result_card, unsafe_allow_html=True)

# ==================== TAB 2: Batch Prediction ====================
with tab2:
    st.markdown("""
<h2 style="font-size: 1.875rem; font-weight: 800; color: #1565C0; margin-bottom: 0.5rem;">
Batch Prediction
</h2>
<p style="font-size: 0.875rem; color: #1565C0; margin-bottom: 1rem;">
Upload CSV with aggregated customer features (61 columns). Models expect per-customer statistics.
</p>
""", unsafe_allow_html=True)

    # Info box about expected format
    st.markdown("""
<div style="background: #FFFFFF; padding: 1rem; border-radius: 8px; border-left: 4px solid #1565C0; border: 1px solid #BBDEFB; margin-bottom: 1.5rem;">
<strong style="color: #1565C0;"> Expected Format: Aggregated Customer Data</strong>
<p style="color: #1565C0; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
Upload CSV with <strong>61 aggregated features</strong> (mean, sum, std, min, max for each metric).<br>
Example files: <code>example_safe_customer.csv</code>, <code>example_fraud_customer.csv</code><br>
<em>View <code>data/README_AGGREGATED_FORMAT.md</code> to understand format details</em>
</p>
</div>
""", unsafe_allow_html=True)

    # Utilities Row - Download examples
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            safe_df = pd.read_csv('data/example_safe_customer.csv')
            csv_safe = safe_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Safe Customer Example",
                data=csv_safe,
                file_name="example_safe_customer.csv",
                mime="text/csv",
                type="secondary",
                help="Customer safe - is_fraud=0"
            )
        except:
            st.info("File example_safe_customer.csv not found")

    with col2:
        try:
            fraud_df = pd.read_csv('data/example_fraud_customer.csv')
            csv_fraud = fraud_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Fraud Customer Example",
                data=csv_fraud,
                file_name="example_fraud_customer.csv",
                mime="text/csv",
                type="secondary",
                help="Customer fraud - is_fraud=1"
            )
        except:
            st.info("File example_fraud_customer.csv not found")

    with col3:
        try:
            subtle_df = pd.read_csv('data/example_subtle_fraud_1.csv')
            csv_subtle = subtle_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Subtle Fraud Example",
                data=csv_subtle,
                file_name="example_subtle_fraud_1.csv",
                mime="text/csv",
                type="secondary",
                help="Subtle fraud - hard to detect"
            )
        except:
            st.info("File example_subtle_fraud_1.csv not found")

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Drag & Drop your file here or click to select", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            st.markdown('<div style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #EDEDED;"></div>', unsafe_allow_html=True)
            
            st.markdown("""
    <h4 style="font-size: 1.125rem; font-weight: 600; color: #1565C0; margin-bottom: 1rem;">
    Preview First 5 Rows
    </h4>
    """, unsafe_allow_html=True)
            
            # Prepare data
            features_df, original_df = prepare_batch_transactions(uploaded_file)
            
            st.dataframe(original_df.head(5), use_container_width=True)
            st.markdown(f"<p style='font-size: 0.75rem; color: #1565C0; margin-top: 0.75rem;'>Total rows: {len(original_df)}</p>", unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 1.5rem 0; padding-top: 1.5rem; border-top: 1px solid #EDEDED;"></div>', unsafe_allow_html=True)
            
            # Predict button
            if st.button("Predict All", use_container_width=True, type="primary"):
                with st.spinner(f"Predicting {len(features_df)} transactions..."):
                    ensemble_scores = predict_batch(features_df, models, scaler, pca)
                
                if ensemble_scores is not None:
                    # Add predictions to original dataframe
                    result_df = original_df.copy()
                    result_df['fraud_score'] = ensemble_scores
                    result_df['fraud_prediction'] = (ensemble_scores >= MODEL_THRESHOLD).astype(int)
                    result_df['risk_level'] = pd.cut(
                        ensemble_scores,
                        bins=[0, 0.2, 0.5, 1.0],
                        labels=['Low', 'Medium', 'High']
                        )

                    st.markdown('<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #EDEDED;"></div>', unsafe_allow_html=True)

                    st.markdown("""
                    <h3 style="font-size: 1.5rem; font-weight: 700; color: #1976D2; margin-bottom: 1.5rem;">
                        Batch Prediction Results
                    </h3>
                    """, unsafe_allow_html=True)

                    # Summary statistics - 3 cards
                    total = len(result_df)
                    fraud_count = (result_df['fraud_prediction'] == 1).sum()
                    not_fraud_count = total - fraud_count

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #E3F2FD; border-radius: 8px; border: 1px solid #BBDEFB;">
                            <span style="font-size: 0.875rem; color: #1565C0;">Total Transactions Predicted:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #1565C0; margin: 0.25rem 0 0 0;">{total}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #E3F2FD; border-radius: 8px; border: 1px solid #BBDEFB;">
                            <span style="font-size: 0.875rem; color: #1565C0;">Fraud Detected:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #1565C0; margin: 0.25rem 0 0 0;">{fraud_count} ({fraud_count/total*100:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #E3F2FD; border-radius: 8px; border: 1px solid #BBDEFB;">
                            <span style="font-size: 0.875rem; color: #1565C0;">Not Fraud:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #1565C0; margin: 0.25rem 0 0 0;">{not_fraud_count} ({not_fraud_count/total*100:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Risk distribution
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    low_risk = (result_df['risk_level'] == 'Low').sum()
                    medium_risk = (result_df['risk_level'] == 'Medium').sum()
                    high_risk = (result_df['risk_level'] == 'High').sum()

                    st.markdown(f"""
                    <div style="padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #BBDEFB; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <h4 style="font-size: 1.125rem; font-weight: 600; color: #1565C0; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
                            Risk Distribution
                        </h4>
                        <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem;">
                            <li style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(237,237,237,0.5); margin-bottom: 0.75rem;">
                                <span style="color: #1565C0; font-weight: 700;">High Risk (&gt;50%):</span>
                                <span style="font-weight: 700; color: #111;">{high_risk}</span>
                            </li>
                            <li style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(237,237,237,0.5); margin-bottom: 0.75rem;">
                                <span style="color: #1565C0; font-weight: 700;">Medium Risk (20%-50%):</span>
                                <span style="font-weight: 700; color: #111;">{medium_risk}</span>
                            </li>
                            <li style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #1565C0; font-weight: 700;">Low Risk (&lt;20%):</span>
                                <span style="font-weight: 700; color: #111;">{low_risk}</span>
                            </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed results table
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    st.markdown("""
                    <h4 style="font-size: 1.125rem; font-weight: 600; color: #1565C0; margin-bottom: 1rem;">
                        Detailed Results
                    </h4>
                    """, unsafe_allow_html=True)

                    # Display columns to show
                    display_cols = ['fraud_score', 'fraud_prediction', 'risk_level']
                    # Add some original columns if they exist
                    for col in ['Customer Id', 'Sales', 'Benefit per order', 'Order Status']:
                        if col in result_df.columns and col not in display_cols:
                            display_cols.insert(0, col)

                    display_df = result_df[display_cols].copy()
                    display_df['fraud_score'] = display_df['fraud_score'].apply(lambda x: f"{x*100:.1f}%")
                    display_df['fraud_prediction'] = display_df['fraud_prediction'].map({0: 'OK', 1: 'FRAUD'})

                    st.dataframe(display_df, use_container_width=True, height=400)

                    # Download results
                    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results (.csv)",
                        data=csv_result,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )

        except Exception as e:
            st.markdown(f"""
            <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3;">
                <strong>Error processing file:</strong> {str(e)}
                <br><br>
                <span style="font-size: 0.875rem; color: #1565C0;">
                    Please check that your CSV file has the required columns and proper formatting.
                </span>
            </div>
            """, unsafe_allow_html=True)
