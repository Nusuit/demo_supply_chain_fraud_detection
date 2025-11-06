# Models Directory

Thư mục này chứa các model files và preprocessors cần thiết cho prediction.

## Yêu cầu Files

### 1. Keras Models (Required)

Copy 3 model files từ `fraud_supplychain_year4/Fraud_SupplyChain/model/best_models/`:

```
combined_model_seed42.keras
combined_model_seed123.keras
combined_model_seed456.keras
```

**Cách copy:**

```bash
# Từ thư mục fraud_demo_streamlit/
cp ../fraud_supplychain_year4/Fraud_SupplyChain/model/best_models/*.keras models/
```

### 2. Preprocessors (Auto-generated)

Các file này sẽ được tự động tạo nếu chưa có:

```
scaler.pkl       # StandardScaler fitted trên training data
pca.pkl          # PCA fitted trên scaled data
```

**Lưu ý:** Nếu bạn có scaler và PCA thực tế từ training, hãy copy chúng vào đây.  
Nếu không, hệ thống sẽ tự động tạo mock preprocessors cho demo.

## Cấu trúc Model

### Architecture

```
Input: 45 features (PCA components)
├── Dense(256) + BatchNorm + Dropout(0.3) + ReLU
├── Dense(128) + BatchNorm + Dropout(0.3) + ReLU
├── Dense(64) + BatchNorm + Dropout(0.2) + ReLU
└── Dense(1) + Sigmoid → Output: Fraud probability [0, 1]
```

### Training Details

- **Loss Function:** Cost-Sensitive Focal Loss (FN_COST=15.0)
- **Optimizer:** Adam
- **Epochs:** 50 (with EarlyStopping)
- **Batch Size:** 32
- **Validation Split:** 0.2
- **Class Balancing:** SMOTE (sampling_strategy=1.0)

### Ensemble Configuration

- **Number of Models:** 3
- **Random Seeds:** [42, 123, 456]
- **Prediction Method:** Simple averaging
- **Threshold:** 0.20 (optimized for Recall)

## Kiểm tra Models

Sau khi copy models, chạy:

```python
import os
from tensorflow import keras

models_dir = 'models'
for seed in [42, 123, 456]:
    model_path = f'{models_dir}/combined_model_seed{seed}.keras'
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded {model_path}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    else:
        print(f"✗ Missing {model_path}")
```

## Troubleshooting

### Lỗi: "Cannot load model file"

- Kiểm tra xem file có tồn tại trong thư mục `models/`
- Đảm bảo TensorFlow version phù hợp (>=2.13.0)
- Thử load model bằng `keras.models.load_model()` trực tiếp

### Lỗi: "Preprocessor not found"

- Nếu không có scaler.pkl và pca.pkl, hệ thống sẽ tự tạo mock preprocessors
- Để sử dụng preprocessors thực tế, copy từ training pipeline

### Performance không đúng

- Đảm bảo đang sử dụng đúng 3 models (seeds 42, 123, 456)
- Kiểm tra threshold = 0.20
- Verify preprocessing pipeline (scaler → PCA)
