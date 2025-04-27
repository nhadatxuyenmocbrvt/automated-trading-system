"""
Tiện ích hỗ trợ cho việc tạo và quản lý đặc trưng.
File này export các hàm và tiện ích quan trọng từ các module con
liên quan đến việc xử lý, kiểm tra, và trực quan hóa đặc trưng.
"""

# Import các hàm từ module con
from data_processors.feature_engineering.utils.validation import validate_features, check_feature_integrity
from data_processors.feature_engineering.utils.preprocessing import normalize_features, standardize_features, min_max_scale
from data_processors.feature_engineering.utils.visualization import plot_feature_importance, plot_feature_correlation, visualize_feature_distribution

# Phiên bản
__version__ = '0.1.0'

# Danh sách các hàm chính
__all__ = [
    # Validation
    'validate_features',
    'check_feature_integrity',
    
    # Preprocessing
    'normalize_features',
    'standardize_features',
    'min_max_scale',
    
    # Visualization
    'plot_feature_importance',
    'plot_feature_correlation',
    'visualize_feature_distribution'
]