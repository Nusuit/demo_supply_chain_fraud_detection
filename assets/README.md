# Assets Directory

Thư mục này chứa các file hình ảnh và visualization assets cho web demo.

## Yêu cầu Files

### 1. Confusion Matrix (confusion_matrix.png)

Hình ảnh confusion matrix từ model evaluation.

**Cách tạo từ Python:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Confusion matrix values (từ test results)
cm = np.array([[2997, 848],
               [72, 214]])

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix - Ensemble Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 2. ROC Curve (roc_curve.png)

Hình ảnh ROC curve và AUC score.

**Cách tạo từ Python:**

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Mock data (thay bằng y_true và y_pred_proba thực tế)
# y_true = ... # True labels
# y_pred_proba = ... # Predicted probabilities

# Tính ROC curve
fpr = np.linspace(0, 1, 100)
tpr = np.power(fpr, 0.5) * 0.9  # Mock curve với AUC ≈ 0.82

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#0068C9', lw=2,
         label=f'ROC Curve (AUC = 0.82)')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
         label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Ensemble Model', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('assets/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Cách sử dụng trong Streamlit

```python
import streamlit as st
import os

# Display confusion matrix
if os.path.exists('assets/confusion_matrix.png'):
    st.image('assets/confusion_matrix.png',
             caption='Confusion Matrix',
             use_column_width=True)

# Display ROC curve
if os.path.exists('assets/roc_curve.png'):
    st.image('assets/roc_curve.png',
             caption='ROC Curve',
             use_column_width=True)
```

## Placeholder Images

Nếu chưa có images thực tế, có thể dùng placeholder:

```python
# Option 1: Tạo placeholder text
st.info("📊 Confusion Matrix visualization sẽ được hiển thị tại đây")

# Option 2: Sử dụng URL placeholder
st.image("https://via.placeholder.com/800x600/0068C9/FFFFFF?text=Confusion+Matrix")
```

## Các assets khác (Optional)

Bạn có thể thêm các visualizations khác:

- `feature_importance.png` - Biểu đồ feature importance
- `training_history.png` - Loss/accuracy curves qua epochs
- `network_graph.png` - Static network visualization
- `logo.png` - Logo của dự án

## Troubleshooting

### Hình ảnh không hiển thị

- Kiểm tra đường dẫn file: `assets/confusion_matrix.png`
- Verify file permissions
- Thử reload trang Streamlit

### Chất lượng ảnh kém

- Tăng DPI khi save: `plt.savefig(..., dpi=300)`
- Sử dụng vector format: `.svg` thay vì `.png`
- Ensure `bbox_inches='tight'` để tránh crop

### File size quá lớn

- Giảm DPI xuống 150-200
- Optimize PNG: `optipng assets/*.png`
- Sử dụng compression: `plt.savefig(..., optimize=True)`
