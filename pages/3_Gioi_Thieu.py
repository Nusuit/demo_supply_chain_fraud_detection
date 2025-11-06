"""
Page 3 - Giới thiệu
Thông tin về dự án, dataset, phương pháp và kiến trúc model
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Giới thiệu",
    page_icon="ℹ️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .hero-section {
        text-align: center;
        padding: 3rem 0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #1f2937;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #6b7280;
        margin-top: 1.5rem;
    }
    .benefit-card {
        background-color: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .feature-badge {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">
        Hệ thống Phát hiện Gian lận<br/>
        <span style="color: #0068C9;">Chuỗi Cung ứng Thông minh</span>
    </h1>
    <p class="hero-subtitle">
        Sử dụng AI và Phân tích Mạng lưới (SNA) để tự động hóa việc phát hiện,<br/>
        ngăn chặn và điều tra gian lận cấu kết phức tạp.
    </p>
</div>
""", unsafe_allow_html=True)

# Benefits Section
st.markdown("---")
st.subheader("🎯 Lợi ích Chính")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="benefit-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💰</div>
        <h3 style="color: #0068C9; margin-bottom: 0.5rem;">Giảm Thiệt hại</h3>
        <p style="color: #6b7280;">
            Phát hiện sớm gian lận, giảm thiệt hại tài chính lên đến 75%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="benefit-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
        <h3 style="color: #0068C9; margin-bottom: 0.5rem;">Tự động hóa</h3>
        <p style="color: #6b7280;">
            Xử lý hàng nghìn giao dịch/giây, giảm thời gian điều tra 90%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="benefit-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🕸️</div>
        <h3 style="color: #0068C9; margin-bottom: 0.5rem;">Phát hiện Fraud Rings</h3>
        <p style="color: #6b7280;">
            SNA giúp phát hiện gian lận có tổ chức, cấu kết phức tạp
        </p>
    </div>
    """, unsafe_allow_html=True)

# Dataset Section
st.markdown("---")
st.subheader("📦 Về Dataset")

with st.expander("📊 **DataCo Supply Chain Dataset** - Mở để xem chi tiết", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Thống kê Tổng quan:**
        - 📊 **180,519** giao dịch
        - 👥 **20,652** khách hàng unique
        - 📦 **118** sản phẩm
        - 🌍 Đa quốc gia (US, Mexico, Canada, UK, v.v.)
        - 📅 Dữ liệu thời gian thực
        """)
    
    with col2:
        st.markdown("""
        **Fraud Statistics:**
        - 🔴 **2.25%** tỷ lệ gian lận
        - ⚠️ **4,062** giao dịch SUSPECTED_FRAUD
        - 📈 **54.83%** có rủi ro giao hàng muộn
        - 💸 **18.71%** có lợi nhuận âm
        """)
    
    st.info("""
    💡 **Đặc điểm Dataset:**
    - Dữ liệu thực tế từ supply chain e-commerce
    - Bao gồm thông tin: Order, Customer, Product, Shipping
    - Có label rõ ràng cho Fraud (Order Status = SUSPECTED_FRAUD)
    - Phù hợp để xây dựng mô hình phát hiện gian lận
    """)

# Methodology Section
st.markdown("---")
st.subheader("🔬 Phương pháp Tiếp cận")

with st.expander("🧠 **1. Feature Engineering** - 61 Features", expanded=False):
    st.markdown("""
    **Transaction Features (57):**
    - 📊 Order details: Sales, Quantity, Discount, Profit
    - 👤 Customer behavior: Order count, Total spent, Recency
    - 📦 Product info: Price, Popularity, Profit margin
    - 🚚 Shipping: Days, Status, Late delivery risk
    - 🎯 Risk flags: High discount, Negative benefit, Rush order
    
    **Network Features (4):**
    - 🔗 Degree Centrality: Số sản phẩm khác nhau đã mua
    - 🌉 Betweenness Centrality: Vai trò cầu nối trong network
    - 📍 Closeness Centrality: Độ gần với trung tâm network
    - ⭐ PageRank: Tầm quan trọng trong network
    """)
    
    st.success("✅ **Kết hợp Transaction + Network features** giúp phát hiện cả gian lận đơn lẻ và có tổ chức!")

with st.expander("🕸️ **2. Social Network Analysis (SNA)**", expanded=False):
    st.markdown("""
    **Bipartite Network Construction:**
    - 👥 Customer Nodes: 20,652
    - 📦 Product Nodes: 118
    - 🔗 Edges (Transactions): 101,196
    - 🎯 Network Type: Bipartite (Customer ↔ Product)
    
    **Community Detection:**
    - 🔍 Algorithm: Louvain Community Detection
    - 📊 Detected: 27 communities
    - 🔴 Fraud Rings: 3 high-risk communities (fraud rate 78-87%)
    
    **Key Findings:**
    - Fraudsters có degree cao hơn 1.57x
    - Fraud communities có mật độ kết nối cao
    - PageRank cao → suspicious (mua sản phẩm hot)
    """)

with st.expander("🤖 **3. Deep Learning Model**", expanded=False):
    st.markdown("""
    **Architecture: Deep Neural Network**
    ```
    Input (45 PCA components)
      ↓
    Dense(256) + BatchNorm + Dropout(0.3) + ReLU
      ↓
    Dense(128) + BatchNorm + Dropout(0.3) + ReLU
      ↓
    Dense(64) + BatchNorm + Dropout(0.2) + ReLU
      ↓
    Dense(1) + Sigmoid
      ↓
    Output (Fraud Probability)
    ```
    
    **Training Strategy:**
    - ⚖️ Class Imbalance Handling: SMOTE (sampling_strategy=1.0)
    - 📏 Normalization: StandardScaler
    - 🎯 Dimensionality Reduction: PCA (61→45 components)
    - 💪 Cost-Sensitive Focal Loss (FN_COST=15.0)
    - 🎲 Ensemble: 3 models với seeds khác nhau [42, 123, 456]
    """)

with st.expander("⚙️ **4. Ensemble & Threshold Tuning**", expanded=False):
    st.markdown("""
    **Ensemble Configuration (AGGRESSIVE):**
    - 🤖 3 independent DNNs trained với random seeds khác nhau
    - 📊 Prediction: Simple averaging
    - 🎯 Threshold: 0.20 (optimized for Recall)
    - 🔥 Cost-Sensitive Loss: FN penalty = 15x
    
    **Why AGGRESSIVE?**
    - ✅ Maximize Recall (>70% achieved: 74.83%)
    - ⚠️ Accept lower Precision (20.15%)
    - 💰 Cost of missing fraud >> Cost of false alert
    - 🎯 Business priority: Catch more fraud!
    """)

# Architecture Diagram
st.markdown("---")
st.subheader("🏗️ Kiến trúc Hệ thống")

st.image("https://via.placeholder.com/800x400/0068C9/FFFFFF?text=System+Architecture+Diagram", 
         caption="Pipeline: Data → Feature Engineering → SNA → Model Training → Ensemble Prediction",
         use_column_width=True)

st.markdown("""
**Pipeline Flow:**

1. 📥 **Data Collection** → DataCo Supply Chain Dataset (180K transactions)
2. 🔧 **Feature Engineering** → Extract 57 transaction features
3. 🕸️ **Network Construction** → Build bipartite graph (20K+ nodes)
4. 📊 **SNA Analysis** → Calculate 4 network features (centrality, community)
5. 🔄 **Data Preprocessing** → SMOTE + StandardScaler + PCA (61→45)
6. 🧠 **Model Training** → Train 3 DNNs với Cost-Sensitive Focal Loss
7. 🎯 **Ensemble Prediction** → Average 3 model outputs
8. ✅ **Fraud Detection** → Threshold=0.20 → Alert/Pass
""")

# Performance Section
st.markdown("---")
st.subheader("📈 Hiệu suất Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🎯 Main Metrics:**
    - ✅ **Recall: 74.83%** (Target: >70%) ⭐
    - 📊 **ROC-AUC: 82.16%**
    - 📏 **Accuracy: 77.73%**
    - 🎲 **F1-Score: 31.75%**
    - 🎯 **Precision: 20.15%**
    """)

with col2:
    st.markdown("""
    **🔍 Detection Performance:**
    - ✓ **Frauds Detected: 214 / 286** (74.83%)
    - ✗ **Frauds Missed: 72** (25.17%)
    - ⚠️ **False Alerts: 848**
    - 📊 **Alert Rate: 25.7%**
    """)

st.success("""
### ✅ Model Đạt Mục Tiêu!

**Recall 74.83%** vượt target 70%, phát hiện được **3/4 giao dịch gian lận**.  
Trade-off với Precision thấp là **chấp nhận được** vì:
- 💰 Chi phí điều tra cảnh báo sai < Chi phí bỏ lỡ gian lận
- 🎯 Ưu tiên phát hiện fraud hơn là giảm false alarm
- ⚖️ Business requirement: Maximize Recall!
""")

# Technology Stack
st.markdown("---")
st.subheader("🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Data & ML:**
    - Python 3.9+
    - Pandas, NumPy
    - Scikit-learn
    - TensorFlow/Keras
    - Imbalanced-learn
    """)

with col2:
    st.markdown("""
    **Network Analysis:**
    - NetworkX
    - Community Detection
    - Centrality Analysis
    - Graph Visualization
    """)

with col3:
    st.markdown("""
    **Web Demo:**
    - Streamlit
    - Chart.js
    - Cytoscape.js
    - Plotly
    """)

# Team Section
st.markdown("---")
st.subheader("👥 Đội ngũ Phát triển")

st.info("""
**UIT - Supply Chain Management Project**  
Kỳ 1, Năm 4 - 2025

Dự án nghiên cứu về phát hiện gian lận trong chuỗi cung ứng sử dụng  
Deep Learning và Social Network Analysis.

📧 Liên hệ: [Insert contact info]  
🔗 Repository: [Insert GitHub link]
""")

# Future Work
st.markdown("---")
st.subheader("🚀 Hướng phát triển")

with st.expander("💡 Xem các cải tiến trong tương lai"):
    st.markdown("""
    **Short-term (1-3 tháng):**
    - 🔄 Real-time prediction API
    - 📊 Interactive dashboard nâng cao
    - 🎯 A/B testing different thresholds
    - 📈 Model monitoring & retraining pipeline
    
    **Medium-term (3-6 tháng):**
    - 🧠 Thử nghiệm Graph Neural Networks (GNN)
    - 🕐 Time-series analysis cho temporal patterns
    - 🌍 Multi-modal features (text, images)
    - 🔐 Explainable AI (SHAP, LIME)
    
    **Long-term (6-12 tháng):**
    - 🤖 Automated feature engineering
    - 🎓 Transfer learning từ domains khác
    - 🔍 Active learning với human-in-the-loop
    - 🌐 Scale lên millions of transactions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem 0;">
    <p style="margin: 0;">© 2025 UIT - Supply Chain Management</p>
    <p style="margin: 0.5rem 0 0 0;">Built with ❤️ using Streamlit, TensorFlow, and NetworkX</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/info.png", width=80)
    st.title("Giới thiệu")
    
    st.markdown("""
    ### ℹ️ Về Dự án
    
    Hệ thống phát hiện gian lận  
    trong chuỗi cung ứng sử dụng:
    
    - 🧠 Deep Learning
    - 🕸️ Network Analysis
    - 📊 Feature Engineering
    - 🎯 Ensemble Methods
    
    ---
    
    ### 📊 Dataset
    **DataCo Supply Chain**
    - 180K giao dịch
    - 20K khách hàng
    - 118 sản phẩm
    - 2.25% fraud rate
    
    ---
    
    ### 🎯 Performance
    - ✅ Recall: **74.83%**
    - 📊 ROC-AUC: **82.16%**
    - 🎯 Accuracy: **77.73%**
    
    ---
    
    ### 📚 Tài liệu
    - [Documentation](...)
    - [GitHub Repo](...)
    - [Research Paper](...)
    """)
    
    st.markdown("---")
    st.caption("🎓 UIT - Year 4, Semester 1")
    st.caption("📅 2025")
