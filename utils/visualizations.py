"""
Utility module for visualizations
"""

import os

def get_confusion_matrix_image():
    """
    Get path to confusion matrix image
    In production, this would be generated from actual model results
    """
    return os.path.join('assets', 'confusion_matrix.png')

def get_roc_curve_image():
    """
    Get path to ROC curve image
    In production, this would be generated from actual model results
    """
    return os.path.join('assets', 'roc_curve.png')

def check_assets_exist():
    """
    Check if required asset files exist
    """
    assets_dir = 'assets'
    required_files = ['confusion_matrix.png', 'roc_curve.png']
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(assets_dir, file)):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def create_placeholder_message():
    """
    Create message for missing assets
    """
    return """
    📊 **Hình ảnh chưa có sẵn**
    
    Để hiển thị visualization đầy đủ, vui lòng:
    1. Tạo Confusion Matrix và ROC Curve từ model training
    2. Lưu các file PNG vào thư mục `assets/`:
       - `confusion_matrix.png`
       - `roc_curve.png`
    
    Hoặc sử dụng placeholder images trong thời gian demo.
    """
