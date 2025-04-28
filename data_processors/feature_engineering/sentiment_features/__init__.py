"""
Đặc trưng tâm lý thị trường.
Module này cung cấp các lớp và hàm để tạo đặc trưng từ dữ liệu tâm lý thị trường,
bao gồm phân tích mạng xã hội, tin tức, và chỉ số tâm lý.
"""

# Import tất cả các module con
from data_processors.feature_engineering.sentiment_features.social_media import (
    SocialMediaFeatures,
    extract_twitter_sentiment_features,
    extract_reddit_sentiment_features,
    compute_social_momentum
)

from data_processors.feature_engineering.sentiment_features.news_analysis import (
    NewsFeatures,
    extract_news_sentiment_features,
    extract_news_topic_features,
    extract_news_volume_features
)

from data_processors.feature_engineering.sentiment_features.market_sentiment import (
    MarketSentimentFeatures,
    extract_fear_greed_features,
    extract_on_chain_sentiment_features,
    extract_sentiment_divergence_features
)

from data_processors.feature_engineering.sentiment_features.text_processors import (
    TextProcessor,
    clean_text,
    tokenize_text,
    extract_keywords,
    calculate_text_sentiment
)

from data_processors.feature_engineering.sentiment_features.event_detection import (
    EventDetector,
    detect_sentiment_shifts,
    detect_abnormal_social_activity,
    detect_news_events,
    compute_event_impact
)

# Danh sách các lớp và hàm cần export
__all__ = [
    # Từ social_media
    'SocialMediaFeatures',
    'extract_twitter_sentiment_features',
    'extract_reddit_sentiment_features',
    'compute_social_momentum',
    
    # Từ news_analysis
    'NewsFeatures',
    'extract_news_sentiment_features',
    'extract_news_topic_features',
    'extract_news_volume_features',
    
    # Từ market_sentiment
    'MarketSentimentFeatures',
    'extract_fear_greed_features',
    'extract_on_chain_sentiment_features',
    'extract_sentiment_divergence_features',
    
    # Từ text_processors
    'TextProcessor',
    'clean_text',
    'tokenize_text',
    'extract_keywords',
    'calculate_text_sentiment',
    
    # Từ event_detection
    'EventDetector',
    'detect_sentiment_shifts',
    'detect_abnormal_social_activity',
    'detect_news_events',
    'compute_event_impact'
]