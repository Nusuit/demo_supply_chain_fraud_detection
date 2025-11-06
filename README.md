# Fraud Supply Chain Detection - Web Demo

Web demo tương tác cho hệ thống phát hiện gian lận trong chuỗi cung ứng sử dụng Deep Learning và Social Network Analysis.

## 🎯 Tính năng

- **Dashboard**: Hiển thị các chỉ số hiệu suất model (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Dự đoán**: Phát hiện gian lận realtime cho giao dịch đơn lẻ hoặc theo lô
- **Phân tích Mạng**: Trực quan hóa mạng lưới khách hàng-sản phẩm và phát hiện fraud rings
- **Giới thiệu**: Thông tin về dataset, phương pháp và kiến trúc model

## 🏗️ Kiến trúc

- **Model**: Ensemble 3 Deep Neural Networks (256-128-64-1)
- **Features**: 61 features (57 transaction + 4 network features)
- **Preprocessing**: StandardScaler + PCA (45 components)
- **Performance**: Recall 74.83%, ROC-AUC 82.16%

## 📦 Cài đặt

### Yêu cầu

- Python 3.9+
- pip

### Các bước

1. Clone repository:

```bash
git clone https://github.com/Nusuit/demo_supply_chain_fraud_detection.git
cd demo_supply_chain_fraud_detection
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Chạy ứng dụng

```bash
streamlit run Home.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

## 🔧 Sử dụng

### 1. Dashboard (Trang chủ)

- Xem tổng quan hiệu suất model
- Confusion Matrix và ROC Curve
- Các metrics chính: Accuracy, Precision, Recall, F1, AUC

### 2. Dự đoán

**Tab Đơn lẻ:**

- Nhập thông tin giao dịch qua form
- Model sẽ dự đoán xác suất gian lận
- Hiển thị kết quả từ 3 models và ensemble score

**Tab Theo lô:**

- Upload file CSV chứa nhiều giao dịch
- Xử lý hàng loạt
- Tải về kết quả dự đoán

### 3. Phân tích Mạng

- Thống kê network (nodes, edges, communities)
- Biểu đồ phân tích centrality
- Trực quan hóa fraud rings bằng Cytoscape

### 4. Giới thiệu

- Thông tin dataset (180K transactions, 20K customers)
- Kiến trúc model và pipeline
- Kết quả đánh giá chi tiết

## 📝 Dataset

**DataCo Supply Chain Dataset**

- Tổng số giao dịch: 180,519
- Tổng số khách hàng: 20,652
- Tổng số sản phẩm: 118
- Tỷ lệ gian lận: 2.25%

## 🧠 Model Details

**Ensemble Configuration (AGGRESSIVE):**

- 3 Deep Neural Networks với các random seeds khác nhau
- Cost-Sensitive Focal Loss (FN_COST=15.0)
- Threshold: 0.20 (tối ưu cho Recall)
- Prediction: Average của 3 models

**Performance:**

- Accuracy: 77.73%
- Precision: 20.15%
- **Recall: 74.83%** ✓ (Target: >70%)
- F1-Score: 31.75%
- ROC-AUC: 82.16%

## 🤝 Contributing

Đây là project demo cho mục đích học tập và nghiên cứu.

## 📄 License

MIT License

## 👥 Authors

- UIT - Supply Chain Management Project
- Year 4, Semester 1

## 📧 Contact

Để biết thêm thông tin, vui lòng liên hệ qua repository.
