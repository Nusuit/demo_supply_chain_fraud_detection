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
background-color: #0B0B0B;
transition: width 0.3s ease;
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
color: #5A5A5A;
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
color: #0B0B0B;
border: 1px solid #E0E0E0;
transform: translateY(-1px);
box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}
.stTabs [aria-selected="true"] {
color: #FFFFFF !important;
font-weight: 700;
background-color: #5A5A5A;
border: 1px solid #5A5A5A;
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
background-color: rgba(245, 245, 245, 0.5);
border: 1px solid #EDEDED;
border-radius: 8px;
color: #0B0B0B;
}

/* Buttons - Enhanced visibility */
.stButton button {
border-radius: 8px;
font-weight: 700;
font-size: 1rem;
border: 2px solid #0B0B0B;
transition: all 0.2s ease;
}
.stButton button[kind="primary"],
.stButton > button[type="submit"]:not([kind="secondary"]) {
background-color: #0B0B0B !important;
color: #FFFFFF !important;
border: 2px solid #0B0B0B !important;
}
.stButton button[kind="primary"]:hover,
.stButton > button[type="submit"]:not([kind="secondary"]):hover {
background-color: #1A1A1A !important;
color: #FFFFFF !important;
transform: translateY(-1px);
box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stButton button[kind="secondary"] {
background-color: white !important;
color: #0B0B0B !important;
border: 2px solid #0B0B0B !important;
}
.stButton button[kind="secondary"]:hover {
background-color: #F5F5F5 !important;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<header style="margin-bottom: 2rem;">
<h1 style="font-size: 2.5rem; font-weight: 800; color: #0B0B0B; letter-spacing: -0.02em; margin-bottom: 0.5rem;">
Fraud Prediction
</h1>
<h4 style="font-size: 1.125rem; font-weight: 600; color: #5A5A5A; margin: 0;">
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
    <div style="background: #F5F5F5; padding: 1rem; border-radius: 8px; border-left: 4px solid #0B0B0B;">
    <strong>Error:</strong> Cannot load models! Please check the models/ directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="background: white; padding: 0.75rem; border-radius: 8px; border: 1px solid #EDEDED; margin-bottom: 1.5rem;">
<span style="color: #0B0B0B; font-weight: 600;">Models and preprocessors are ready</span>
</div>
""", unsafe_allow_html=True)

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["Single Prediction", "Batch (CSV/XLSX)"])

# ==================== TAB 1: Single Prediction ====================
with tab1:
    st.markdown("""
<h2 style="font-size: 1.875rem; font-weight: 800; color: #0B0B0B; margin-bottom: 0.5rem;">
Single Transaction Prediction
</h2>
<p style="font-size: 0.875rem; color: #5A5A5A; margin-bottom: 1rem;">
Enter 10 transaction fields. Network features use estimated values for demo.
</p>
""", unsafe_allow_html=True)

    # Warning about model limitation
    st.markdown("""
<div style="background: #FFF9E6; padding: 1rem; border-radius: 8px; border-left: 4px solid #F59E0B; margin-bottom: 1.5rem;">
<sin style="color: #92400E;"> Important Note:</sin>
<p style="color: #78350F; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
AI models trained on <sin>aggregated per-customer data</sin> (mean, sum, std, min, max from multiple transactions).
This form only inputs <sin>1 transaction</sin> so will <sin>mock aggregate values</sin> → predictions may not be accurate.<br><br>
<sin>Recommendation:</sin> Use <sin>Tab "Batch (CSV/XLSX)"</sin> to upload aggregated customer data file for more accurate results.
</p>
</div>
""", unsafe_allow_html=True)

    with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            
            # LEFT COLUMN - Order & Shipping Information
            with col1:
                # Group 1: Order Information
                st.markdown("""
        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
        1. Order Information
        </h3>
        """, unsafe_allow_html=True)
                
                order_value = st.number_input("Order Value (USD)", min_value=0.0, value=250.0, step=0.01,
                                             help="Larger orders may indicate higher risk")
                product_quantity = st.number_input("Product Quantity", min_value=1, max_value=100, value=1)
                discount_rate = st.slider("Discount (%)", min_value=0, max_value=50, value=10,
                                         help="Discounts > 20% carry higher risk")
                
                # Group 2: Shipping Information
                st.markdown("""
        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
        2. Shipping Information
        </h3>
        """, unsafe_allow_html=True)
                
                shipping_days = st.number_input("Shipping Days", min_value=0, max_value=30, value=3,
                                               help="Very fast shipping (<2 days) may be suspicious")
                shipping_method = st.selectbox("Shipping Method",
                                              ["Standard Class", "Second Class", "First Class", "Same Day"])
                late_delivery_risk = st.radio("Late Delivery Risk", ["No", "Yes"], horizontal=True)
            
            # RIGHT COLUMN - Customer Information
            with col2:
                st.markdown("""
        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
        3. Customer Information
        </h3>
        """, unsafe_allow_html=True)
                
                customer_segment = st.selectbox("Customer Segment",
                                               ["Consumer", "Corporate", "Home Office"])
                product_category = st.selectbox("Product Category",
                                               ["Fishing", "Electronics", "Camping & Hiking", "Cleats", "Office Supplies"])
                market_region = st.selectbox("Market Region",
                                            ["Pacific Asia", "Europe", "USCA", "LATAM"])
                
                st.markdown("""
        <p style="font-size: 0.75rem; color: #5A5A5A; font-style: italic; opacity: 0.7; margin-top: 1.5rem;">
        Network features use estimated values for demo.
        </p>
        """, unsafe_allow_html=True)
            
            # Action Buttons
            col_reset, col_predict = st.columns([1, 1])
            with col_reset:
                reset_btn = st.form_submit_button("Reset Form", use_container_width=True, type="secondary")
            with col_predict:
                submitted = st.form_submit_button("Predict Fraud", use_container_width=True, type="primary")
            
            if submitted:
                # Convert user inputs to model features
                late_risk_val = 1 if late_delivery_risk == "Yes" else 0
                discount_decimal = discount_rate / 100.0
                
                # Prepare transaction data (simplified - using key fields from form)
                transaction_data = {
                    'Days for shipping (real)': shipping_days,
                    'Days for shipment (scheduled)': shipping_days,
                    'Benefit per order': order_value * 0.35,  # Estimated
                    'Sales per customer': order_value,
                    'Late_delivery_risk': late_risk_val,
                    'Order Item Discount': order_value * discount_decimal,
                    'Order Item Discount Rate': discount_decimal,
                    'Order Item Product Price': order_value / product_quantity,
                    'Order Item Profit Ratio': 0.35,
                    'Order Item Quantity': product_quantity,
                    'Sales': order_value,
                    'Order Item Total': order_value,
                    'Order Profit Per Order': order_value * 0.35,
                    'product_popularity': 5000,
                    'product_profit_margin': 0.35,
                    'product_avg_discount': discount_decimal,
                    'customer_order_count': 3,
                    'customer_total_spent': order_value * 3,
                    'customer_avg_order_value': order_value,
                    'customer_fraud_history': 0,
                    'recency_days': 30,
                    'order_count_last_30d': 1,
                    'total_spent_last_30d': order_value,
                    'time_since_last_order': 30,
                    'is_new_customer': 0,
                    'rush_order': 0,
                    'unusual_quantity': 1 if product_quantity > 5 else 0,
                    'high_discount_flag': 1 if discount_decimal > 0.2 else 0,
                    'negative_benefit': 0,
                    'international_order': 0,
                    'high_value_order': 1 if order_value > 500 else 0,
                    'high_risk_combination': late_risk_val,
                    'degree_centrality': 0.0045,
                    'betweenness_centrality': 0.00012,
                    'closeness_centrality': 0.38,
                    'pagerank': 0.000048,
                }

                # Prepare for prediction
                df = prepare_single_transaction(transaction_data)

                # Predict
                with st.spinner("Predicting..."):
                    prediction_result = predict_single(df, models, scaler, pca)

                if prediction_result:
                    st.markdown('<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #EDEDED;"></div>', unsafe_allow_html=True)

                    st.markdown("""
                    <h3 style="font-size: 1.5rem; font-weight: 700; color: #0B0B0B; margin-bottom: 1.5rem;">
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
<div style="padding: 1.5rem; background: #F5F5F5; border-left: 4px solid #0B0B0B; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<h4 style="font-size: 1.25rem; font-weight: 800; color: #0B0B0B; margin-bottom: 1rem;">
FRAUD DETECTED
</h4>

<!-- Probability Row -->
<div style="margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
<span>Fraud Probability</span>
<span style="color: #0B0B0B; font-weight: 800;">{ensemble_score*100:.1f}%</span>
</div>
<div class="monochrome-bar">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>

<!-- Risk Badge -->
<span style="display: inline-block; padding: 0.25rem 0.75rem; background: #0B0B0B; color: white; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
High Risk
</span>

<!-- Recommendations -->
<div style="margin-top: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #EDEDED;">
<ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem; color: #0B0B0B;">
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #0B0B0B; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Temporarily Block Transaction
</li>
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #0B0B0B; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Manual Investigation Required
</li>
<li style="display: flex; align-items: flex-start;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #0B0B0B; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Contact Customer for Verification
</li>
</ul>
</div>

<!-- Prediction Details -->
<div style="margin-top: 1rem; padding: 1rem; border: 1px solid #EDEDED; border-radius: 8px; background: white;">
<h5 style="font-size: 0.875rem; font-weight: 600; color: #5A5A5A; margin-bottom: 0.75rem;">
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
<span style="flex: 1; font-weight: 800; color: #0B0B0B;">Ensemble (average):</span>
<span style="font-weight: 800; color: #0B0B0B; margin: 0 1rem;">{ensemble_score*100:.1f}%</span>
<div class="monochrome-bar" style="flex: 1;">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>
</div>
</div>

<p style="font-size: 0.75rem; color: #5A5A5A; margin-top: 1rem; opacity: 0.8;">
Threshold = {MODEL_THRESHOLD:.2f}. Probability > {MODEL_THRESHOLD*100:.0f}% → labeled as FRAUD.
</p>
</div>
"""
                else:
                    # NOT FRAUD
                    result_card = f"""
<div style="padding: 1.5rem; background: #F5F5F5; border-left: 4px solid #EDEDED; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<h4 style="font-size: 1.25rem; font-weight: 800; color: #0B0B0B; margin-bottom: 1rem;">
NO FRAUD DETECTED
</h4>

<!-- Probability Row -->
<div style="margin-bottom: 1rem;">
<div style="display: flex; justify-content: space-between; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
<span>Fraud Probability</span>
<span style="color: #0B0B0B; font-weight: 800;">{ensemble_score*100:.1f}%</span>
</div>
<div class="monochrome-bar">
<div class="monochrome-bar-fill" style="width: {ensemble_score*100:.1f}%;"></div>
</div>
</div>

<!-- Risk Badge -->
<span style="display: inline-block; padding: 0.25rem 0.75rem; border: 1px solid #0B0B0B; color: #0B0B0B; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;">
Low Risk
</span>

<!-- Recommendations -->
<div style="margin-top: 1rem;">
<ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem; color: #0B0B0B;">
<li style="display: flex; align-items: flex-start; margin-bottom: 0.25rem;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #EDEDED; border: 1px solid #0B0B0B; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
Allow Transaction
</li>
<li style="display: flex; align-items: flex-start;">
<span style="width: 8px; height: 8px; border-radius: 9999px; background: #EDEDED; border: 1px solid #0B0B0B; margin-right: 0.75rem; margin-top: 0.25rem; flex-shrink: 0;"></span>
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
<h2 style="font-size: 1.875rem; font-weight: 800; color: #0B0B0B; margin-bottom: 0.5rem;">
Batch Prediction
</h2>
<p style="font-size: 0.875rem; color: #5A5A5A; margin-bottom: 1rem;">
Upload CSV with aggregated customer features (61 columns). Models expect per-customer statistics.
</p>
""", unsafe_allow_html=True)

    # Info box about expected format
    st.markdown("""
<div style="background: #FFFFFF; padding: 1rem; border-radius: 8px; border-left: 4px solid #5A5A5A; border: 1px solid #EDEDED; margin-bottom: 1.5rem;">
<strong style="color: #0B0B0B;"> Expected Format: Aggregated Customer Data</strong>
<p style="color: #5A5A5A; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
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
    <h4 style="font-size: 1.125rem; font-weight: 600; color: #0B0B0B; margin-bottom: 1rem;">
    Preview First 5 Rows
    </h4>
    """, unsafe_allow_html=True)
            
            # Prepare data
            features_df, original_df = prepare_batch_transactions(uploaded_file)
            
            st.dataframe(original_df.head(5), use_container_width=True)
            st.markdown(f"<p style='font-size: 0.75rem; color: #5A5A5A; margin-top: 0.75rem;'>Total rows: {len(original_df)}</p>", unsafe_allow_html=True)
            
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
                    <h3 style="font-size: 1.5rem; font-weight: 700; color: #0B0B0B; margin-bottom: 1.5rem;">
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
                        <div style="padding: 1rem; background: #F5F5F5; border-radius: 8px; border: 1px solid #EDEDED;">
                            <span style="font-size: 0.875rem; color: #5A5A5A;">Total Transactions Predicted:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #0B0B0B; margin: 0.25rem 0 0 0;">{total}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #F5F5F5; border-radius: 8px; border: 1px solid #EDEDED;">
                            <span style="font-size: 0.875rem; color: #5A5A5A;">Fraud Detected:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #0B0B0B; margin: 0.25rem 0 0 0;">{fraud_count} ({fraud_count/total*100:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="padding: 1rem; background: #F5F5F5; border-radius: 8px; border: 1px solid #EDEDED;">
                            <span style="font-size: 0.875rem; color: #5A5A5A;">Not Fraud:</span>
                            <p style="font-size: 1.5rem; font-weight: 800; color: #0B0B0B; margin: 0.25rem 0 0 0;">{not_fraud_count} ({not_fraud_count/total*100:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Risk distribution
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    low_risk = (result_df['risk_level'] == 'Low').sum()
                    medium_risk = (result_df['risk_level'] == 'Medium').sum()
                    high_risk = (result_df['risk_level'] == 'High').sum()

                    st.markdown(f"""
                    <div style="padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #EDEDED; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <h4 style="font-size: 1.125rem; font-weight: 600; color: #0B0B0B; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #EDEDED;">
                            Risk Distribution
                        </h4>
                        <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.875rem;">
                            <li style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(237,237,237,0.5); margin-bottom: 0.75rem;">
                                <span style="color: #0B0B0B; font-weight: 700;">High Risk (&gt;50%):</span>
                                <span style="font-weight: 700; color: #111;">{high_risk}</span>
                            </li>
                            <li style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(237,237,237,0.5); margin-bottom: 0.75rem;">
                                <span style="color: #0B0B0B; font-weight: 700;">Medium Risk (20%-50%):</span>
                                <span style="font-weight: 700; color: #111;">{medium_risk}</span>
                            </li>
                            <li style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #0B0B0B; font-weight: 700;">Low Risk (&lt;20%):</span>
                                <span style="font-weight: 700; color: #111;">{low_risk}</span>
                            </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed results table
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

                    st.markdown("""
                    <h4 style="font-size: 1.125rem; font-weight: 600; color: #0B0B0B; margin-bottom: 1rem;">
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
            <div style="background: #F5F5F5; padding: 1rem; border-radius: 8px; border-left: 4px solid #0B0B0B;">
                <strong>Error processing file:</strong> {str(e)}
                <br><br>
                <span style="font-size: 0.875rem; color: #5A5A5A;">
                    Please check that your CSV file has the required columns and proper formatting.
                </span>
            </div>
            """, unsafe_allow_html=True)
