"""
Utils package for Fraud Supply Chain Detection Demo
"""

# Import only what's needed, avoid importing TensorFlow-dependent modules by default
from .constants import *
from .visualizations import *

# Lazy imports for heavy dependencies
def get_preprocessing_module():
    from . import preprocessing
    return preprocessing

def get_predictor_module():
    """
    Import predictor with automatic fallback to mock if TensorFlow fails
    """
    try:
        from . import predictor
        return predictor
    except (ImportError, OSError) as e:
        # TensorFlow DLL error or import error - use mock
        if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
            print("⚠️ TensorFlow unavailable, using mock predictor")
            from . import predictor_mock
            return predictor_mock
        else:
            raise

__all__ = [
    'FEATURE_COLUMNS',
    'CATEGORICAL_FEATURES',
    'MODEL_THRESHOLD',
    'TOTAL_FEATURES',
    'get_risk_level',
    'get_fraud_prediction',
    'get_confusion_matrix_image',
    'get_roc_curve_image',
    'check_assets_exist',
    'get_preprocessing_module',
    'get_predictor_module',
]
