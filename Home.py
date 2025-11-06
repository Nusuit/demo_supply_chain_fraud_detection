"""
Home Page - Dashboard
Hiển thị các chỉ số hiệu suất model và visualizations
"""

import streamlit as st
import json
import os

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0068C9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0068C9;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Supply Chain Fraud Detection using Deep Learning & Network Analysis</div>', unsafe_allow_html=True)

# Load test results
@st.cache_data
def load_test_results():
    try:
        with open('data/test_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ File test_results.json không tìm thấy!")
        return None

results = load_test_results()

if results:
    # Model Information Section
    st.markdown("---")
    st.subheader("📊 Thông tin Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", results['model_info']['name'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Architecture", results['model_info']['type'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Threshold", f"{results['model_info']['threshold']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Random Seeds", f"{len(results['model_info']['seeds'])} models")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Metrics Section
    st.markdown("---")
    st.subheader("📈 Hiệu suất Model")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = results['metrics']
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics['accuracy']*100:.2f}%",
            delta=None,
            help="Độ chính xác tổng thể của model"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value=f"{metrics['precision']*100:.2f}%",
            delta=None,
            help="Tỷ lệ dự đoán đúng trong số các dự đoán là gian lận"
        )
    
    with col3:
        st.metric(
            label="Recall ⭐",
            value=f"{metrics['recall']*100:.2f}%",
            delta="Target: 70%",
            delta_color="normal",
            help="Tỷ lệ phát hiện được gian lận thực tế (chỉ số quan trọng nhất)"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value=f"{metrics['f1_score']*100:.2f}%",
            delta=None,
            help="Trung bình điều hòa của Precision và Recall"
        )
    
    with col5:
        st.metric(
            label="ROC-AUC",
            value=f"{metrics['roc_auc']*100:.2f}%",
            delta=None,
            help="Khả năng phân biệt giữa gian lận và hợp lệ"
        )
    
    # Confusion Matrix and Detection Stats
    st.markdown("---")
    st.subheader("🎯 Kết quả Phát hiện")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        
        cm = results['confusion_matrix']
        
        # Display confusion matrix metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("True Negative (TN)", f"{cm['true_negative']:,}")
            st.caption("Dự đoán đúng: Hợp lệ")
            
            st.metric("False Negative (FN)", f"{cm['false_negative']:,}")
            st.caption("Dự đoán sai: Bỏ lỡ gian lận ⚠️")
        
        with col_b:
            st.metric("False Positive (FP)", f"{cm['false_positive']:,}")
            st.caption("Dự đoán sai: Cảnh báo nhầm")
            
            st.metric("True Positive (TP)", f"{cm['true_positive']:,}")
            st.caption("Dự đoán đúng: Phát hiện gian lận ✓")
        
        # Confusion matrix visualization placeholder
        st.info("💡 Để xem visualization Confusion Matrix chi tiết, vui lòng thêm file `assets/confusion_matrix.png`")
        
        # Try to show image if exists
        if os.path.exists('assets/confusion_matrix.png'):
            st.image('assets/confusion_matrix.png', use_column_width=True)
    
    with col2:
        st.markdown("#### Thống kê Phát hiện Gian lận")
        
        fd = results['fraud_detection']
        test = results['test_set']
        
        # Detection statistics
        st.metric(
            "Gian lận phát hiện",
            f"{fd['frauds_detected']}/{test['fraud']}",
            delta=f"{fd['detection_rate']*100:.1f}%",
            help="Số giao dịch gian lận được phát hiện"
        )
        
        st.metric(
            "Gian lận bỏ lỡ",
            f"{fd['frauds_missed']}",
            delta=f"{(fd['frauds_missed']/test['fraud'])*100:.1f}%",
            delta_color="inverse",
            help="Số giao dịch gian lận bị bỏ lỡ"
        )
        
        st.metric(
            "Cảnh báo sai",
            f"{fd['false_alerts']}",
            delta=f"{(fd['false_alerts']/test['not_fraud'])*100:.1f}%",
            delta_color="inverse",
            help="Số giao dịch hợp lệ bị cảnh báo nhầm"
        )
        
        st.metric(
            "Tỷ lệ cảnh báo",
            f"{fd['alert_rate']*100:.1f}%",
            help="Tỷ lệ giao dịch được đánh dấu cần kiểm tra"
        )
        
        # ROC Curve placeholder
        st.info("💡 Để xem ROC Curve, vui lòng thêm file `assets/roc_curve.png`")
        
        # Try to show image if exists
        if os.path.exists('assets/roc_curve.png'):
            st.image('assets/roc_curve.png', use_column_width=True)
    
    # Dataset Information
    st.markdown("---")
    st.subheader("📦 Thông tin Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    dataset = results['dataset']
    
    with col1:
        st.metric("Tổng giao dịch", f"{dataset['total_orders']:,}")
    
    with col2:
        st.metric("Tổng khách hàng", f"{dataset['total_customers']:,}")
    
    with col3:
        st.metric("Tỷ lệ gian lận", f"{dataset['fraud_rate']*100:.2f}%")
    
    with col4:
        st.metric("Kích thước test set", f"{dataset['test_size']:,}")
    
    # Feature Information
    st.markdown("---")
    st.subheader("🔧 Đặc trưng (Features)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = results['features']
    
    with col1:
        st.metric("Transaction Features", features['transaction_features'])
    
    with col2:
        st.metric("Network Features", features['network_features'])
    
    with col3:
        st.metric("Total Features", features['total_features'])
    
    with col4:
        st.metric("PCA Components", features['pca_components'])
    
    # Interpretation Section
    st.markdown("---")
    st.subheader("💡 Giải thích Kết quả")
    
    st.markdown(f"""
    <div class="info-box">
    <h4>✅ Model Đạt Mục Tiêu</h4>
    <p>Model ensemble AGGRESSIVE đã đạt được <strong>Recall {metrics['recall']*100:.2f}%</strong>, vượt qua mục tiêu 70% 
    trong phát hiện gian lận. Điều này có nghĩa:</p>
    <ul>
        <li>🎯 Phát hiện được <strong>{fd['frauds_detected']} / {test['fraud']}</strong> giao dịch gian lận 
        ({fd['detection_rate']*100:.1f}%)</li>
        <li>⚠️ Chỉ bỏ lỡ <strong>{fd['frauds_missed']}</strong> giao dịch gian lận ({(fd['frauds_missed']/test['fraud'])*100:.1f}%)</li>
        <li>📊 ROC-AUC {metrics['roc_auc']*100:.1f}% cho thấy khả năng phân biệt tốt</li>
    </ul>
    
    <h4>⚖️ Trade-off Chấp nhận được</h4>
    <p>Precision tương đối thấp ({metrics['precision']*100:.1f}%) là điều chấp nhận được trong fraud detection:</p>
    <ul>
        <li>✓ Chi phí điều tra cảnh báo sai < Chi phí bỏ lỡ gian lận</li>
        <li>✓ {fd['false_alerts']} cảnh báo sai / {test['not_fraud']} giao dịch hợp lệ 
        = {(fd['false_alerts']/test['not_fraud'])*100:.1f}%</li>
        <li>✓ Đổi lại, model phát hiện được {fd['detection_rate']*100:.1f}% gian lận thực tế</li>
    </ul>
    
    <h4>🚀 Ưu điểm của Ensemble</h4>
    <ul>
        <li>📈 Kết hợp 3 models với random seeds khác nhau</li>
        <li>🎲 Giảm overfitting, tăng khả năng tổng quát hóa</li>
        <li>⚡ Cost-Sensitive Focal Loss ưu tiên phát hiện gian lận</li>
        <li>🎯 Threshold thấp (0.20) tối ưu cho Recall</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🔍 Để dự đoán gian lận cho giao dịch mới, vui lòng chuyển sang trang **Dự đoán**")
    st.caption("📊 Để xem phân tích mạng lưới, vui lòng chuyển sang trang **Phân tích Mạng**")

else:
    st.error("❌ Không thể load dữ liệu. Vui lòng kiểm tra file `data/test_results.json`")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/protect.png", width=80)
    st.title("Navigation")
    st.markdown("""
    ### 📑 Các trang
    - 🏠 **Dashboard** (trang hiện tại)
    - 🔮 **Dự đoán**: Phát hiện gian lận realtime
    - 🕸️ **Phân tích Mạng**: Network analysis
    - ℹ️ **Giới thiệu**: Thông tin dự án
    
    ---
    
    ### 📖 Hướng dẫn
    **Dashboard** hiển thị:
    - Các chỉ số hiệu suất model
    - Confusion matrix
    - Thống kê phát hiện gian lận
    - Thông tin dataset
    
    ---
    
    ### 🎯 Mục tiêu chính
    **Maximize Recall** (≥70%)
    
    Phát hiện càng nhiều gian lận càng tốt, 
    chấp nhận trade-off với Precision.
    """)
    
    st.markdown("---")
    st.caption("© 2025 UIT - Supply Chain Management")
