"""
Module tạo đặc trưng cho dữ liệu thị trường.
File này export tất cả các module và lớp quan trọng của feature engineering,
cung cấp giao diện đơn giản cho việc tạo và quản lý đặc trưng.
"""

# Import các module con
from data_processors.feature_engineering.feature_generator import FeatureGenerator
from data_processors.feature_engineering.utils.validation import validate_features, check_feature_integrity
from data_processors.feature_engineering.utils.preprocessing import normalize_features, standardize_features, min_max_scale
from data_processors.feature_engineering.utils.visualization import plot_feature_importance, plot_feature_correlation, visualize_feature_distribution

# Import khi đã phát triển đầy đủ các module con
try:
    # Technical indicators
    from data_processors.feature_engineering.technical_indicators import *
except ImportError:
    pass

try:
    # Market features
    from data_processors.feature_engineering.market_features import *
except ImportError:
    pass

try:
    # Sentiment features
    from data_processors.feature_engineering.sentiment_features import *
except ImportError:
    pass

try:
    # Feature selector
    from data_processors.feature_engineering.feature_selector import *
except ImportError:
    pass

# Phiên bản
__version__ = '0.1.0'

# Danh sách các tính năng chính
__all__ = [
    'FeatureGenerator',
    'validate_features',
    'check_feature_integrity',
    'normalize_features',
    'standardize_features',
    'min_max_scale',
    'plot_feature_importance',
    'plot_feature_correlation',
    'visualize_feature_distribution',
]