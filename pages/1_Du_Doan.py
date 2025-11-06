"""
Page 1 - Dự đoán Gian lận
Live prediction cho giao dịch đơn lẻ hoặc theo lô
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
    page_title="Dự đoán Gian lận",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .fraud-box {
        background-color: #ffebee;
        border: 3px solid #ef5350;
        color: #c62828;
    }
    .legitimate-box {
        background-color: #e8f5e9;
        border: 3px solid #66bb6a;
        color: #2e7d32;
    }
    .score-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔮 Dự đoán Gian lận")
st.markdown("Phát hiện gian lận trong giao dịch chuỗi cung ứng sử dụng Deep Learning")

# Load models and preprocessors
@st.cache_resource
def load_resources():
    models = load_models('models')
    scaler, pca = load_preprocessors('models')
    return models, scaler, pca

with st.spinner("🔄 Đang load models và preprocessors..."):
    models, scaler, pca = load_resources()

if models is None:
    st.error("❌ Không thể load models! Vui lòng kiểm tra thư mục `models/`")
    st.stop()

st.success("✅ Models và preprocessors đã sẵn sàng!")

# Tabs for different prediction modes
tab1, tab2 = st.tabs(["📝 Dự đoán Đơn lẻ", "📊 Dự đoán Theo lô"])

# ==================== TAB 1: Single Prediction ====================
with tab1:
    st.markdown("### Nhập thông tin giao dịch")
    st.info("💡 Nhập các thông tin cơ bản về giao dịch. Các trường khác sẽ được tính toán tự động.")
    
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📦 Thông tin Đơn hàng**")
            days_shipping_real = st.number_input("Ngày ship thực tế", min_value=0, max_value=30, value=4)
            days_shipping_scheduled = st.number_input("Ngày ship dự kiến", min_value=0, max_value=30, value=4)
            benefit_per_order = st.number_input("Lợi nhuận/đơn ($)", value=85.5)
            sales = st.number_input("Doanh số ($)", min_value=0.0, value=240.0)
            order_quantity = st.number_input("Số lượng", min_value=1, max_value=100, value=2)
        
        with col2:
            st.markdown("**👤 Thông tin Khách hàng**")
            sales_per_customer = st.number_input("Doanh số/khách hàng ($)", min_value=0.0, value=256.8)
            customer_order_count = st.number_input("Số đơn đã mua", min_value=1, max_value=100, value=5)
            customer_total_spent = st.number_input("Tổng chi tiêu ($)", min_value=0.0, value=1284.0)
            recency_days = st.number_input("Ngày từ lần mua cuối", min_value=0, max_value=365, value=45)
            customer_fraud_history = st.selectbox("Lịch sử gian lận", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")
        
        with col3:
            st.markdown("**📊 Thông tin Sản phẩm**")
            product_price = st.number_input("Giá sản phẩm ($)", min_value=0.0, value=120.0)
            product_popularity = st.number_input("Độ phổ biến", min_value=0, max_value=20000, value=8500)
            product_profit_margin = st.slider("Tỷ suất lợi nhuận", 0.0, 1.0, 0.35)
            discount_rate = st.slider("Tỷ lệ giảm giá", 0.0, 1.0, 0.083)
            late_delivery_risk = st.selectbox("Rủi ro giao muộn", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")
        
        # Advanced section (collapsible)
        with st.expander("⚙️ Cài đặt Nâng cao (Network Features)"):
            col_a, col_b = st.columns(2)
            with col_a:
                degree_centrality = st.number_input("Degree Centrality", value=0.0045, format="%.6f")
                betweenness_centrality = st.number_input("Betweenness Centrality", value=0.00012, format="%.6f")
            with col_b:
                closeness_centrality = st.number_input("Closeness Centrality", value=0.38, format="%.4f")
                pagerank = st.number_input("PageRank", value=0.000048, format="%.6f")
        
        submitted = st.form_submit_button("🔍 Phân tích Giao dịch", use_container_width=True)
        
        if submitted:
            # Prepare transaction data
            transaction_data = {
                'Days for shipping (real)': days_shipping_real,
                'Days for shipment (scheduled)': days_shipping_scheduled,
                'Benefit per order': benefit_per_order,
                'Sales per customer': sales_per_customer,
                'Late_delivery_risk': late_delivery_risk,
                'Order Item Discount': product_price * discount_rate,
                'Order Item Discount Rate': discount_rate,
                'Order Item Product Price': product_price,
                'Order Item Profit Ratio': product_profit_margin,
                'Order Item Quantity': order_quantity,
                'Sales': sales,
                'Order Item Total': product_price * order_quantity,
                'Order Profit Per Order': benefit_per_order,
                'product_popularity': product_popularity,
                'product_profit_margin': product_profit_margin,
                'product_avg_discount': discount_rate,
                'customer_order_count': customer_order_count,
                'customer_total_spent': customer_total_spent,
                'customer_avg_order_value': customer_total_spent / customer_order_count if customer_order_count > 0 else 0,
                'customer_fraud_history': customer_fraud_history,
                'recency_days': recency_days,
                'order_count_last_30d': 2 if recency_days < 30 else 0,
                'total_spent_last_30d': sales if recency_days < 30 else 0,
                'time_since_last_order': recency_days,
                'is_new_customer': 1 if customer_order_count == 1 else 0,
                'rush_order': 1 if days_shipping_real < days_shipping_scheduled else 0,
                'unusual_quantity': 1 if order_quantity > 5 else 0,
                'high_discount_flag': 1 if discount_rate > 0.3 else 0,
                'negative_benefit': 1 if benefit_per_order < 0 else 0,
                'international_order': 0,  # Simplified
                'high_value_order': 1 if sales > 500 else 0,
                'high_risk_combination': 1 if (late_delivery_risk == 1 and benefit_per_order < 0) else 0,
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'pagerank': pagerank,
            }
            
            # Prepare for prediction
            df = prepare_single_transaction(transaction_data)
            
            # Predict
            with st.spinner("🔄 Đang phân tích..."):
                prediction_result = predict_single(df, models, scaler, pca)
            
            if prediction_result:
                st.markdown("---")
                st.markdown("### 📊 Kết quả Dự đoán")
                
                # Ensemble score display
                ensemble_score = prediction_result['ensemble_score']
                risk_label, risk_color = get_risk_level(ensemble_score)
                fraud_prediction = get_fraud_prediction(ensemble_score)
                
                # Big score display
                score_color = "#ef5350" if ensemble_score >= MODEL_THRESHOLD else "#66bb6a"
                st.markdown(f"""
                <div class="score-display" style="color: {score_color};">
                    {ensemble_score*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction box
                box_class = "fraud-box" if ensemble_score >= MODEL_THRESHOLD else "legitimate-box"
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    {fraud_prediction}
                </div>
                """, unsafe_allow_html=True)
                
                # Risk level
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric("Mức độ Rủi ro", risk_label)
                
                # Individual model scores
                st.markdown("---")
                st.markdown("#### 🤖 Chi tiết từ các Models")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model 1 (seed 42)", f"{prediction_result['model_1_score']*100:.2f}%")
                
                with col2:
                    st.metric("Model 2 (seed 123)", f"{prediction_result['model_2_score']*100:.2f}%")
                
                with col3:
                    st.metric("Model 3 (seed 456)", f"{prediction_result['model_3_score']*100:.2f}%")
                
                with col4:
                    st.metric("Ensemble (Trung bình)", f"{ensemble_score*100:.2f}%", delta="Final Score")
                
                # Interpretation
                st.markdown("---")
                st.markdown("#### 💡 Giải thích")
                
                if ensemble_score >= MODEL_THRESHOLD:
                    st.warning(f"""
                    ⚠️ **Cảnh báo Gian lận!**
                    
                    Điểm số ensemble ({ensemble_score*100:.2f}%) vượt ngưỡng {MODEL_THRESHOLD*100:.0f}%. 
                    Giao dịch này có khả năng cao là gian lận.
                    
                    **Khuyến nghị:**
                    - 🔍 Xem xét chi tiết thông tin khách hàng
                    - 📞 Liên hệ xác nhận với khách hàng
                    - 🛑 Tạm dừng xử lý đơn hàng cho đến khi xác minh
                    """)
                else:
                    st.success(f"""
                    ✅ **Giao dịch Hợp lệ**
                    
                    Điểm số ensemble ({ensemble_score*100:.2f}%) dưới ngưỡng {MODEL_THRESHOLD*100:.0f}%. 
                    Giao dịch này có khả năng là hợp lệ.
                    
                    **Khuyến nghị:**
                    - ✓ Tiếp tục xử lý đơn hàng bình thường
                    - 👁️ Theo dõi các giao dịch tiếp theo từ khách hàng này
                    """)

# ==================== TAB 2: Batch Prediction ====================
with tab2:
    st.markdown("### Upload file CSV để dự đoán hàng loạt")
    st.info("💡 File CSV cần có các cột tương ứng với features của model. Các features thiếu sẽ tự động điền 0.")
    
    # Sample file download
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**📥 Tải file mẫu để tham khảo định dạng:**")
    with col2:
        # Load sample file
        try:
            sample_df = pd.read_csv('data/sample_transactions.csv')
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Tải file mẫu",
                data=csv_sample,
                file_name="sample_transactions.csv",
                mime="text/csv"
            )
        except:
            pass
    
    # File uploader
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and display data
            st.markdown("---")
            st.markdown("#### 📄 Dữ liệu đã upload")
            
            # Prepare data
            features_df, original_df = prepare_batch_transactions(uploaded_file)
            
            st.dataframe(original_df.head(10), use_container_width=True)
            st.caption(f"Hiển thị 10/{len(original_df)} hàng đầu tiên")
            
            # Predict button
            if st.button("🔍 Dự đoán cho tất cả giao dịch", use_container_width=True):
                with st.spinner(f"🔄 Đang phân tích {len(features_df)} giao dịch..."):
                    # Predict
                    ensemble_scores = predict_batch(features_df, models, scaler, pca)
                
                if ensemble_scores is not None:
                    # Add predictions to original dataframe
                    result_df = original_df.copy()
                    result_df['fraud_score'] = ensemble_scores
                    result_df['fraud_prediction'] = (ensemble_scores >= MODEL_THRESHOLD).astype(int)
                    result_df['risk_level'] = pd.cut(
                        ensemble_scores,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.success("✅ Dự đoán hoàn tất!")
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("#### 📊 Tổng quan Kết quả")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total = len(result_df)
                    fraud_count = (result_df['fraud_prediction'] == 1).sum()
                    legitimate_count = total - fraud_count
                    avg_score = ensemble_scores.mean()
                    
                    with col1:
                        st.metric("Tổng giao dịch", total)
                    
                    with col2:
                        st.metric("Phát hiện Gian lận", fraud_count, 
                                delta=f"{fraud_count/total*100:.1f}%",
                                delta_color="inverse")
                    
                    with col3:
                        st.metric("Giao dịch Hợp lệ", legitimate_count,
                                delta=f"{legitimate_count/total*100:.1f}%")
                    
                    with col4:
                        st.metric("Điểm trung bình", f"{avg_score*100:.2f}%")
                    
                    # Risk distribution
                    st.markdown("---")
                    st.markdown("#### 🎯 Phân bố Rủi ro")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    low_risk = (result_df['risk_level'] == 'Low').sum()
                    medium_risk = (result_df['risk_level'] == 'Medium').sum()
                    high_risk = (result_df['risk_level'] == 'High').sum()
                    
                    with col1:
                        st.metric("🟢 Rủi ro Thấp", low_risk, 
                                delta=f"{low_risk/total*100:.1f}%")
                    
                    with col2:
                        st.metric("🟡 Rủi ro Trung bình", medium_risk,
                                delta=f"{medium_risk/total*100:.1f}%")
                    
                    with col3:
                        st.metric("🔴 Rủi ro Cao", high_risk,
                                delta=f"{high_risk/total*100:.1f}%",
                                delta_color="inverse")
                    
                    # Show results table
                    st.markdown("---")
                    st.markdown("#### 📋 Kết quả Chi tiết")
                    
                    # Display columns to show
                    display_cols = ['Customer Id', 'fraud_score', 'fraud_prediction', 'risk_level']
                    # Add some original columns if they exist
                    for col in ['Sales', 'Benefit per order', 'Order Status']:
                        if col in result_df.columns:
                            display_cols.insert(-3, col)
                    
                    display_df = result_df[display_cols].copy()
                    display_df['fraud_score'] = display_df['fraud_score'].apply(lambda x: f"{x*100:.2f}%")
                    display_df['fraud_prediction'] = display_df['fraud_prediction'].map({0: '✅ Hợp lệ', 1: '⚠️ Gian lận'})
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    st.markdown("#### 💾 Tải xuống Kết quả")
                    
                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Tải file kết quả (CSV)",
                        data=csv_result,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
            st.info("Vui lòng kiểm tra định dạng file CSV và đảm bảo có đủ các cột cần thiết.")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/crystal-ball.png", width=80)
    st.title("Hướng dẫn")
    
    st.markdown("""
    ### 🔮 Dự đoán Gian lận
    
    **2 chế độ:**
    
    #### 📝 Đơn lẻ
    - Nhập thông tin 1 giao dịch
    - Xem chi tiết dự đoán từ 3 models
    - Nhận khuyến nghị xử lý
    
    #### 📊 Theo lô
    - Upload file CSV nhiều giao dịch
    - Dự đoán hàng loạt
    - Tải xuống kết quả
    
    ---
    
    ### 🎯 Threshold
    **{:.0f}%** - Ngưỡng phân loại
    
    - ≥ {:.0f}%: Gian lận ⚠️
    - < {:.0f}%: Hợp lệ ✅
    
    ---
    
    ### 📊 Điểm Rủi ro
    - 🟢 **Thấp**: 0-30%
    - 🟡 **Trung bình**: 30-60%
    - 🔴 **Cao**: 60-100%
    """.format(MODEL_THRESHOLD*100, MODEL_THRESHOLD*100, MODEL_THRESHOLD*100))
    
    st.markdown("---")
    st.caption("💡 Model sử dụng 61 features")
    st.caption("   (57 transaction + 4 network)")
    st.caption("🧠 Ensemble 3 Deep Neural Networks")
    st.caption("📊 Form chỉ hiển thị features chính")
    st.caption("   (Các features khác tự động điền 0)")
