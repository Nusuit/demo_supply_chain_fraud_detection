"""
About Page - System Information
Project overview, technology, methodology, and performance
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.styling import inject_custom_css

# Page configuration
st.set_page_config(
    page_title="About - Supply Chain Fraud Detection",
    page_icon="⬛",
    layout="wide"
)

# Apply custom styling
inject_custom_css()

# ============================================
# HEADER
# ============================================
st.title("About the System")
st.markdown("Supply Chain Fraud Detection using Deep Learning and Social Network Analysis")
st.divider()

# ============================================
# PROJECT OVERVIEW
# ============================================
st.subheader("Project Overview")

st.markdown("""
This is a **fraud detection system for supply chain transactions** that combines:
- **Deep Learning** (Ensemble of 3 Deep Neural Networks)
- **Social Network Analysis** (Customer-Product network)
- **Cost-Sensitive Learning** (Heavy penalty for missed frauds)

The hybrid approach detects both **individual fraud** and **organized fraud rings**.
""")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# KEY OBJECTIVES
# ============================================
st.subheader("Key Objectives")

col1, col2 = st.columns([1, 2.5], gap="large")

with col1:
    # Target card with monochrome design
    st.markdown(
        '<div style="background: white; border: 1px solid #EDEDED; border-radius: 12px; '
        'padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
        '<div style="color: #5A5A5A; font-size: 11px; font-weight: 600; '
        'text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">TARGET</div>'
        '<div style="color: #0B0B0B; font-size: 36px; font-weight: 800; '
        'line-height: 1.2; margin-bottom: 8px;">≥70%</div>'
        '<div style="color: #0B0B0B; font-size: 18px; font-weight: 600; '
        'margin-bottom: 8px;">Recall</div>'
        '<div style="color: #5A5A5A; font-size: 13px; line-height: 1.5;">'
        'Detect at least 70% of fraudulent transactions</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    # Achievement card with subtle background
    st.markdown(
        '<div style="background: #F5F5F5; border-left: 4px solid #0B0B0B; '
        'border-radius: 8px; padding: 24px 28px;">'
        '<div style="color: #0B0B0B; font-size: 16px; font-weight: 700; '
        'margin-bottom: 16px;">ACHIEVED: 74.83% Recall</div>'
        '<div style="color: #5A5A5A; font-size: 14px; line-height: 1.7;">'
        'Successfully detected <strong style="color: #0B0B0B;">214 out of 286 frauds</strong> '
        '(74.83%), exceeding the 70% target. This means the system catches '
        '<strong style="color: #0B0B0B;">3 out of 4 fraudulent transactions</strong>.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# TECHNOLOGY STACK
# ============================================
st.subheader("Technology & Methodology")

tab1, tab2, tab3, tab4 = st.tabs([
    "Deep Neural Network",
    "Ensemble Learning", 
    "Cost-Sensitive Loss",
    "Social Network Analysis"
])

with tab1:
    st.markdown("##### Deep Neural Network (DNN)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Architecture:**
        - Input: 45 features (after PCA)
        - Layer 1: 256 neurons + BatchNorm + Dropout(0.3) + ReLU
        - Layer 2: 128 neurons + BatchNorm + Dropout(0.3) + ReLU
        - Layer 3: 64 neurons + BatchNorm + Dropout(0.2) + ReLU
        - Output: 1 neuron + Sigmoid (probability)
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - **Activation:** ReLU for hidden layers, Sigmoid for output
        - **Regularization:** Dropout to prevent overfitting
        - **Normalization:** BatchNormalization for stable training
        - **Optimizer:** Adam (learning_rate=0.001)
        """)

with tab2:
    st.markdown("##### Ensemble Learning Strategy")
    
    st.markdown("""
    **Configuration:**
    - Train **3 independent DNN models** with different random seeds: `[42, 123, 456]`
    - Each model learns different patterns from the data
    - **Prediction method:** Simple averaging of 3 model outputs
    - **Benefits:**
        - Reduces variance and overfitting
        - Improves generalization
        - More robust predictions
    
    ```
    Final Prediction = (Model_42 + Model_123 + Model_456) / 3
    ```
    """)

with tab3:
    st.markdown("##### Cost-Sensitive Focal Loss")
    
    st.markdown("""
    **Why Cost-Sensitive?**
    
    In fraud detection, **missing a fraud (False Negative) is much more costly** than a false alarm (False Positive):
    - Missing fraud = Company loses money
    - False alarm = Manual review (small cost)
    
    **Custom Loss Function:**
    ```
    Focal Loss (base) + False Negative Penalty (FN_COST = 15.0)
    
    Loss = -α(1-p)^γ log(p) + y_true * (1 - y_pred) * 15.0
           ↑                    ↑
           Focal Loss           FN Penalty (15x weight)
    ```
    
    **Parameters:**
    - `gamma = 0.8` — Focus on hard examples
    - `alpha = 0.80` — Weight for positive class
    - `FN_COST = 15.0` — **Heavy penalty** for missing frauds
    """)

with tab4:
    st.markdown("##### Social Network Analysis (SNA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Network Construction:**
        - Type: **Bipartite Network**
        - Customer Nodes: 20,652
        - Product Nodes: 118
        - Edges (Transactions): 101,196
        - Communities Detected: 27
        """)
    
    with col2:
        st.markdown("""
        **Network Features (4):**
        - **Degree:** # of different products purchased
        - **Betweenness:** Bridge role in network (+82%!)
        - **Closeness:** Distance to network center
        - **Community:** Which group customer belongs to
        """)
    
    st.info("""
    **Key Finding:** Fraudsters have **82% higher betweenness centrality** than normal customers, 
    meaning they act as "bridges" connecting different parts of the network — a strong indicator of organized fraud!
    """)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# DATASET INFORMATION
# ============================================
st.subheader("Dataset Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", "180,519")
with col2:
    st.metric("Unique Customers", "20,652")
with col3:
    st.metric("Unique Products", "118")
with col4:
    st.metric("Fraud Rate", "2.25%")

st.markdown("""
**Source:** DataCo Supply Chain Dataset (Kaggle)

**Description:** Real-world e-commerce supply chain data containing:
- Order details (value, quantity, discount, date)
- Customer information (segment, location, history)
- Product details (category, department, price)
- Shipping information (mode, days, delivery status)
- Fraud labels (Order Status = SUSPECTED_FRAUD)
""")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# FEATURES BREAKDOWN
# ============================================
st.subheader("Features (61 Total)")

# Create two-column layout
col_left, col_right = st.columns([2.5, 1], gap="large")

with col_left:
    st.markdown("#### Transaction Features (57)")
    st.markdown(
        '<p style="font-size: 13px; color: #5A5A5A; margin-top: -8px; margin-bottom: 20px; font-style: italic;">'
        'Below are key feature categories with examples (not exhaustive)</p>',
        unsafe_allow_html=True
    )
    
    # Feature categories in a compact table format
    feature_categories = [
        ("Order Info", "Sales value, Quantity, Discount, Profit margin"),
        ("Customer Behavior", "Order count, Total spending, Purchase frequency"),
        ("Shipping", "Shipping days, Mode, Delivery status, Late risk"),
        ("Geographic", "Customer location, Market, Region"),
        ("Product", "Category, Department, Price, Popularity"),
        ("Temporal", "Order date, Day of week, Hour, Recency"),
        ("Risk Flags", "High discount, Negative profit, Rush order")
    ]
    
    # Create table with 2 columns to save space
    for i in range(0, len(feature_categories), 2):
        cols = st.columns(2, gap="small")
        
        for idx, col in enumerate(cols):
            if i + idx < len(feature_categories):
                title, examples = feature_categories[i + idx]
                with col:
                    st.markdown(
                        f'<div style="background: white; border-left: 3px solid #0B0B0B; '
                        f'padding: 12px 16px; margin-bottom: 12px;">'
                        f'<div style="font-weight: 700; font-size: 13px; color: #0B0B0B; '
                        f'margin-bottom: 4px; letter-spacing: 0.3px;">{title}</div>'
                        f'<div style="font-size: 12px; color: #5A5A5A; line-height: 1.4;">{examples}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

with col_right:
    st.markdown("#### Network Features (4)")
    st.markdown("")
    
    st.markdown(
        '<div style="background: white; border: 1px solid #EDEDED; border-radius: 8px; '
        'padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">'
        '<div style="font-size: 13px; color: #5A5A5A; margin-bottom: 16px; font-weight: 500;">'
        'Generated from SNA:</div>'
        '<div style="margin-bottom: 12px;">'
        '<span style="display: inline-block; width: 6px; height: 6px; background: #0B0B0B; '
        'border-radius: 50%; margin-right: 10px;"></span>'
        '<span style="font-weight: 600; color: #0B0B0B;">Degree</span></div>'
        '<div style="margin-bottom: 12px;">'
        '<span style="display: inline-block; width: 6px; height: 6px; background: #0B0B0B; '
        'border-radius: 50%; margin-right: 10px;"></span>'
        '<span style="font-weight: 600; color: #0B0B0B;">Betweenness</span>'
        '<span style="display: inline-block; background: #0B0B0B; color: white; font-size: 11px; '
        'padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-weight: 600;">+82%</span></div>'
        '<div style="margin-bottom: 12px;">'
        '<span style="display: inline-block; width: 6px; height: 6px; background: #0B0B0B; '
        'border-radius: 50%; margin-right: 10px;"></span>'
        '<span style="font-weight: 600; color: #0B0B0B;">Closeness</span></div>'
        '<div style="margin-bottom: 20px;">'
        '<span style="display: inline-block; width: 6px; height: 6px; background: #0B0B0B; '
        'border-radius: 50%; margin-right: 10px;"></span>'
        '<span style="font-weight: 600; color: #0B0B0B;">Community</span></div>'
        '<div style="border-top: 1px solid #EDEDED; padding-top: 16px; font-size: 13px; '
        'color: #5A5A5A;"><strong style="color: #0B0B0B;">Note:</strong> PCA reduces 61 → 45 components</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# WORKFLOW
# ============================================
st.subheader("System Workflow")

# Create centered workflow with max width
st.markdown('<div style="max-width: 900px; margin: 0 auto;">', unsafe_allow_html=True)

# Workflow steps
workflow_steps = [
    ("1", "Input", "New Transaction", "Data entry point for a single transaction"),
    ("2", "Feature Extraction", "Extract 57 Transaction Features", "Order info, customer behavior, shipping details, product data, temporal patterns, risk flags"),
    ("3", "Network Lookup", "Lookup 4 Network Features", "Retrieve pre-computed values: degree, betweenness, closeness, community from customer-product network"),
    ("4", "Preprocessing", "StandardScaler + PCA", "Normalize 61 features, then apply PCA dimensionality reduction to 45 components"),
    ("5", "Ensemble Models", "3 Independent DNN Models", "Model 1 (seed=42) → Prediction 1<br>Model 2 (seed=123) → Prediction 2<br>Model 3 (seed=456) → Prediction 3"),
    ("6", "Aggregation", "Ensemble Average", "Calculate mean of 3 predictions for robust final probability score"),
    ("7", "Classification", "Apply Threshold = 0.20", "Compare probability with aggressive threshold to maximize fraud detection"),
    ("8", "Output", "FRAUD / NOT FRAUD", "Final binary classification result")
]

for idx, (num, title, subtitle, description) in enumerate(workflow_steps):
    # Vertical line HTML (separate to avoid backslash in f-string)
    vertical_line = "" if idx == len(workflow_steps) - 1 else '<div style="position: absolute; left: 22px; top: 44px; width: 2px; height: calc(100% + 24px); background: #EDEDED; z-index: 1;"></div>'
    
    # Create card for each step
    st.markdown(
        f'<div style="position: relative; padding-left: 60px; margin-bottom: 24px;">'
        f'<div style="position: absolute; left: 0; top: 0; background: #0B0B0B; color: white; '
        f'width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center; '
        f'justify-content: center; font-weight: 800; font-size: 18px; z-index: 10;">{num}</div>'
        f'{vertical_line}'
        f'<div style="background: white; border: 1px solid #EDEDED; border-radius: 8px; '
        f'padding: 20px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
        f'<div style="font-weight: 700; font-size: 16px; color: #0B0B0B; margin-bottom: 4px;">{title}</div>'
        f'<div style="font-weight: 600; font-size: 14px; color: #5A5A5A; margin-bottom: 8px;">{subtitle}</div>'
        f'<div style="font-size: 13px; color: #5A5A5A; line-height: 1.6;">{description}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# MODEL PERFORMANCE
# ============================================
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Primary Metrics:**
    - **Recall:** 74.83%
    - **ROC-AUC:** 82.16%
    - **Accuracy:** 77.73%
    - **F1-Score:** 31.75%
    - **Precision:** 20.15%
    """)

with col2:
    st.markdown("""
    **Detection Performance:**
    - Frauds Detected: 214 / 286
    - Frauds Missed: 72
    - False Alerts: 848
    - Alert Rate: 25.7%
    """)

with col3:
    st.markdown("""
    **Business Impact:**
    - Net Benefit: **$88,900**
    - Savings vs Baseline
    - ROI: Positive
    - Deployment: Recommended
    """)

st.info("""
**Trade-off Accepted:** Low Precision (20.15%) is acceptable because:
- Cost of missing fraud >> Cost of false alert
- Manual review is inexpensive compared to fraud losses
- Business priority: **Maximize fraud detection (Recall)**
""")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# FRAUD RINGS DETECTED
# ============================================
st.subheader("Fraud Rings Detection")

st.markdown("""
Using **Social Network Analysis**, we identified **3 high-risk communities** (fraud rings):

| Community | Members | Frauds | Fraud Rate | Status |
|-----------|---------|--------|------------|--------|
| Ring #1 | 15 | 12 | **80.0%** | High Risk |
| Ring #2 | 23 | 18 | **78.0%** | High Risk |
| Ring #3 | 8 | 7 | **87.5%** | Very High Risk |

**Average fraud rate across these rings: 80.4%** — indicating highly coordinated, organized fraud networks.
""")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# DEVELOPMENT INFO
# ============================================
st.subheader("Development Information")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Project:** Supply Chain Fraud Detection System  
    **Institution:** UIT (University of Information Technology)  
    **Program:** Year 4, Semester 1 — 2025  
    **Course:** Supply Chain Management Project
    
    **Research Focus:**
    - Deep Learning for fraud detection
    - Social Network Analysis in supply chains
    - Ensemble methods for imbalanced data
    - Cost-sensitive learning
    """)

with col2:
    st.markdown("""
    **Tech Stack:**
    - Python 3.9+
    - TensorFlow/Keras
    - Scikit-learn
    - NetworkX
    - Streamlit
    - Plotly
    """)

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("© 2025 UIT — Supply Chain Management | Built with Streamlit, TensorFlow, and NetworkX")
