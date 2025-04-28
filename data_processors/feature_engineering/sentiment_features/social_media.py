"""
Đặc trưng từ dữ liệu mạng xã hội.
File này cung cấp các lớp và hàm để tạo đặc trưng từ dữ liệu mạng xã hội
như Twitter, Reddit, và các nền tảng khác.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime
from pathlib import Path
import logging
import json

# Import các module nội bộ
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from data_collectors.news_collector.sentiment_collector import SentimentData

# Thiết lập logger
logger = get_logger("sentiment_features")

class SocialMediaFeatures:
    """
    Lớp xử lý và tạo đặc trưng từ dữ liệu mạng xã hội.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Khởi tạo đối tượng SocialMediaFeatures.
        
        Args:
            data_dir: Thư mục chứa dữ liệu tâm lý (tùy chọn)
        """
        self.data_dir = data_dir
        
        if self.data_dir is None:
            # Sử dụng thư mục mặc định nếu không được chỉ định
            from config.system_config import BASE_DIR
            self.data_dir = BASE_DIR / "data" / "sentiment"
        
        # Đảm bảo thư mục tồn tại
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    def load_sentiment_data(self, file_path: Optional[Union[str, Path]] = None, 
                           days: int = 30, asset: Optional[str] = None) -> List[SentimentData]:
        """
        Tải dữ liệu tâm lý từ file hoặc lấy dữ liệu gần đây.
        
        Args:
            file_path: Đường dẫn đến file dữ liệu cụ thể (tùy chọn)
            days: Số ngày dữ liệu cần lấy nếu không chỉ định file
            asset: Mã tài sản cần lọc (tùy chọn)
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Chuyển đổi từ dict sang đối tượng SentimentData
                sentiment_data = [SentimentData.from_dict(item) for item in json_data]
                logger.info(f"Đã tải {len(sentiment_data)} bản ghi tâm lý từ {file_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi tải dữ liệu tâm lý: {str(e)}")
                sentiment_data = []
        else:
            # Tìm các file trong khoảng thời gian
            sentiment_data = []
            current_time = datetime.datetime.now()
            start_time = current_time - datetime.timedelta(days=days)
            
            # Tìm tất cả các file JSON và sắp xếp theo thời gian tạo
            json_files = list(self.data_dir.glob("*.json"))
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for json_file in json_files:
                # Kiểm tra thời gian tạo file
                file_time = datetime.datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_time < start_time:
                    continue
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    batch_data = [SentimentData.from_dict(item) for item in json_data]
                    sentiment_data.extend(batch_data)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi tải file {json_file}: {str(e)}")
            
            logger.info(f"Đã tải tổng cộng {len(sentiment_data)} bản ghi tâm lý từ {len(json_files)} file")
        
        # Lọc theo tài sản nếu cần
        if asset:
            sentiment_data = [data for data in sentiment_data if data.asset == asset]
            logger.info(f"Đã lọc {len(sentiment_data)} bản ghi cho tài sản {asset}")
        
        # Lọc chỉ lấy dữ liệu từ mạng xã hội
        social_media_sources = ["Twitter Sentiment", "Reddit Sentiment"]
        sentiment_data = [data for data in sentiment_data 
                         if any(source in data.source for source in social_media_sources)]
        
        logger.info(f"Dữ liệu mạng xã hội: {len(sentiment_data)} bản ghi")
        
        return sentiment_data
    
    def sentiment_to_dataframe(self, sentiment_data: List[SentimentData]) -> pd.DataFrame:
        """
        Chuyển đổi danh sách SentimentData thành DataFrame.
        
        Args:
            sentiment_data: Danh sách dữ liệu tâm lý
            
        Returns:
            DataFrame chứa dữ liệu tâm lý
        """
        if not sentiment_data:
            return pd.DataFrame()
        
        records = []
        for item in sentiment_data:
            record = {
                'timestamp': item.timestamp,
                'value': item.value,
                'label': item.label,
                'source': item.source,
                'asset': item.asset,
                'timeframe': item.timeframe
            }
            
            # Thêm các thông tin từ metadata
            if item.metadata:
                for key, value in item.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        record[f"metadata_{key}"] = value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Đảm bảo timestamp là datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def extract_features(self, price_df: pd.DataFrame, sentiment_data: List[SentimentData],
                        window_sizes: List[int] = [1, 3, 7, 14],
                        normalize: bool = True) -> pd.DataFrame:
        """
        Tạo các đặc trưng tâm lý từ mạng xã hội và kết hợp với dữ liệu giá.
        
        Args:
            price_df: DataFrame dữ liệu giá (phải có cột 'timestamp', 'close')
            sentiment_data: Danh sách dữ liệu tâm lý từ mạng xã hội
            window_sizes: Các kích thước cửa sổ cho tính toán trung bình và xu hướng
            normalize: Chuẩn hóa đặc trưng (True/False)
            
        Returns:
            DataFrame kết hợp giữa dữ liệu giá và đặc trưng tâm lý
        """
        # Chuyển đổi dữ liệu tâm lý thành DataFrame
        sentiment_df = self.sentiment_to_dataframe(sentiment_data)
        
        if sentiment_df.empty:
            logger.warning("Không có dữ liệu tâm lý mạng xã hội để tạo đặc trưng")
            return price_df
        
        # Đảm bảo dữ liệu giá có cột timestamp là datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Tạo bản sao của DataFrame giá để thêm đặc trưng
        result_df = price_df.copy()
        
        # Tách dữ liệu tâm lý theo nguồn
        twitter_df = sentiment_df[sentiment_df['source'].str.contains('Twitter')]
        reddit_df = sentiment_df[sentiment_df['source'].str.contains('Reddit')]
        
        # Tạo đặc trưng Twitter
        if not twitter_df.empty:
            twitter_features = extract_twitter_sentiment_features(twitter_df, result_df, window_sizes)
            # Kết hợp đặc trưng
            for col in twitter_features.columns:
                if col not in result_df.columns:
                    result_df[col] = twitter_features[col]
        
        # Tạo đặc trưng Reddit
        if not reddit_df.empty:
            reddit_features = extract_reddit_sentiment_features(reddit_df, result_df, window_sizes)
            # Kết hợp đặc trưng
            for col in reddit_features.columns:
                if col not in result_df.columns:
                    result_df[col] = reddit_features[col]
        
        # Tính toán đặc trưng động lượng tâm lý
        momentum_features = compute_social_momentum(result_df, window_sizes)
        for col in momentum_features.columns:
            if col not in result_df.columns:
                result_df[col] = momentum_features[col]
        
        # Chuẩn hóa đặc trưng nếu cần
        if normalize:
            sentiment_columns = [col for col in result_df.columns 
                                if 'sentiment' in col.lower() or 'social' in col.lower()]
            
            for col in sentiment_columns:
                if col in result_df.columns and result_df[col].dtype in [np.float64, np.int64]:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std > 0:
                        result_df[col] = (result_df[col] - mean) / std
        
        return result_df


def extract_twitter_sentiment_features(twitter_df: pd.DataFrame, price_df: pd.DataFrame,
                                      window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tạo đặc trưng từ dữ liệu tâm lý Twitter/X.
    
    Args:
        twitter_df: DataFrame dữ liệu tâm lý Twitter
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng Twitter
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    twitter_df['timestamp'] = pd.to_datetime(twitter_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Đặt timestamp làm index cho twitter_df
    twitter_df = twitter_df.set_index('timestamp')
    
    # Tính điểm tâm lý trung bình theo ngày
    daily_sentiment = twitter_df.resample('D')['value'].mean()
    
    # Với mỗi timestamp trong price_df, tìm điểm tâm lý trung bình gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        
        # Tính toán tâm lý trung bình trong ngày
        result_df.loc[idx, 'twitter_sentiment_daily'] = daily_sentiment.asof(timestamp)
        
        # Tính toán thông số vùng (bearish/bullish) dựa trên phân loại
        bearish_count = twitter_df.loc[:timestamp].iloc[-50:]['label'].str.contains('Bearish').sum()
        bullish_count = twitter_df.loc[:timestamp].iloc[-50:]['label'].str.contains('Bullish').sum()
        result_df.loc[idx, 'twitter_bull_bear_ratio'] = bullish_count / max(1, bearish_count)
    
    # Tính toán các chỉ số trung bình cho các khoảng thời gian khác nhau
    for window in window_sizes:
        # Điểm tâm lý trung bình cho cửa sổ
        result_df[f'twitter_sentiment_avg_{window}d'] = result_df['twitter_sentiment_daily'].rolling(window).mean()
        
        # Độ lệch chuẩn của tâm lý
        result_df[f'twitter_sentiment_std_{window}d'] = result_df['twitter_sentiment_daily'].rolling(window).std()
        
        # Xu hướng tâm lý (độ dốc của đường hồi quy tuyến tính)
        result_df[f'twitter_sentiment_trend_{window}d'] = (
            result_df['twitter_sentiment_daily'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
        )
    
    # Lấy thông tin metadata bổ sung
    if 'metadata_followers' in twitter_df.columns:
        # Trọng số tâm lý theo người theo dõi
        weighted_sentiment = (twitter_df['value'] * twitter_df['metadata_followers']).resample('D').sum() / \
                            twitter_df['metadata_followers'].resample('D').sum()
        
        for idx, row in result_df.iterrows():
            timestamp = row['timestamp']
            result_df.loc[idx, 'twitter_weighted_sentiment'] = weighted_sentiment.asof(timestamp)
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df


def extract_reddit_sentiment_features(reddit_df: pd.DataFrame, price_df: pd.DataFrame,
                                     window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tạo đặc trưng từ dữ liệu tâm lý Reddit.
    
    Args:
        reddit_df: DataFrame dữ liệu tâm lý Reddit
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng Reddit
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Đặt timestamp làm index cho reddit_df
    reddit_df = reddit_df.set_index('timestamp')
    
    # Tính điểm tâm lý trung bình theo ngày
    daily_sentiment = reddit_df.resample('D')['value'].mean()
    
    # Với mỗi timestamp trong price_df, tìm điểm tâm lý trung bình gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        
        # Tính toán tâm lý trung bình trong ngày
        result_df.loc[idx, 'reddit_sentiment_daily'] = daily_sentiment.asof(timestamp)
        
        # Tính toán tâm lý theo subreddit (nếu có)
        if 'metadata_subreddit' in reddit_df.columns:
            # Lấy các bản ghi trước timestamp hiện tại
            prev_records = reddit_df.loc[:timestamp].iloc[-100:]
            
            # Nhóm theo subreddit và tính giá trị trung bình
            subreddit_sentiment = prev_records.groupby('metadata_subreddit')['value'].mean()
            
            # Lấy giá trị cho các subreddit phổ biến
            for subreddit in ['Bitcoin', 'CryptoCurrency', 'Ethereum']:
                if subreddit in subreddit_sentiment:
                    result_df.loc[idx, f'reddit_{subreddit.lower()}_sentiment'] = subreddit_sentiment[subreddit]
    
    # Tính các chỉ số tâm lý khác
    if 'metadata_score' in reddit_df.columns:
        # Trọng số tâm lý theo điểm bài viết
        weighted_sentiment = (reddit_df['value'] * reddit_df['metadata_score']).resample('D').sum() / \
                            reddit_df['metadata_score'].resample('D').sum()
        
        for idx, row in result_df.iterrows():
            timestamp = row['timestamp']
            result_df.loc[idx, 'reddit_weighted_sentiment'] = weighted_sentiment.asof(timestamp)
    
    # Tính toán các chỉ số trung bình cho các khoảng thời gian khác nhau
    for window in window_sizes:
        # Điểm tâm lý trung bình cho cửa sổ
        result_df[f'reddit_sentiment_avg_{window}d'] = result_df['reddit_sentiment_daily'].rolling(window).mean()
        
        # Độ lệch chuẩn của tâm lý
        result_df[f'reddit_sentiment_std_{window}d'] = result_df['reddit_sentiment_daily'].rolling(window).std()
        
        # Xu hướng tâm lý (độ dốc của đường hồi quy tuyến tính)
        result_df[f'reddit_sentiment_trend_{window}d'] = (
            result_df['reddit_sentiment_daily'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
        )
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df


def compute_social_momentum(df: pd.DataFrame, window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tính toán động lượng tâm lý xã hội từ các chỉ số đã có.
    
    Args:
        df: DataFrame chứa đặc trưng tâm lý từ Twitter và Reddit
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng động lượng tâm lý
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Kiểm tra các cột tâm lý cần thiết
    twitter_cols = [col for col in df.columns if 'twitter_sentiment' in col]
    reddit_cols = [col for col in df.columns if 'reddit_sentiment' in col]
    
    if not twitter_cols and not reddit_cols:
        logger.warning("Không tìm thấy cột tâm lý từ Twitter hoặc Reddit để tính toán động lượng xã hội")
        return result_df
    
    # Tạo điểm tâm lý tổng hợp nếu có cả Twitter và Reddit
    if 'twitter_sentiment_daily' in df.columns and 'reddit_sentiment_daily' in df.columns:
        df['social_combined_sentiment'] = (df['twitter_sentiment_daily'] + df['reddit_sentiment_daily']) / 2
    elif 'twitter_sentiment_daily' in df.columns:
        df['social_combined_sentiment'] = df['twitter_sentiment_daily']
    elif 'reddit_sentiment_daily' in df.columns:
        df['social_combined_sentiment'] = df['reddit_sentiment_daily']
    
    # Tính toán chỉ số RSI của tâm lý
    if 'social_combined_sentiment' in df.columns:
        # Tính delta
        df['sentiment_delta'] = df['social_combined_sentiment'].diff()
        
        # Tính các giá trị gain và loss
        df['sentiment_gain'] = df['sentiment_delta'].apply(lambda x: max(x, 0))
        df['sentiment_loss'] = df['sentiment_delta'].apply(lambda x: abs(min(x, 0)))
        
        # Tính RSI cho các cửa sổ khác nhau
        for window in window_sizes:
            # Tính trung bình gain và loss
            avg_gain = df['sentiment_gain'].rolling(window=window).mean()
            avg_loss = df['sentiment_loss'].rolling(window=window).mean()
            
            # Tính RS và RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Tránh chia cho 0
            result_df[f'social_sentiment_rsi_{window}d'] = 100 - (100 / (1 + rs))
    
    # Tính chỉ số phân kỳ giữa tâm lý và giá
    if 'social_combined_sentiment' in df.columns and 'close' in df.columns:
        for window in window_sizes:
            # Chuẩn hóa dữ liệu giá và tâm lý
            normalized_price = (df['close'] - df['close'].rolling(window).min()) / \
                              (df['close'].rolling(window).max() - df['close'].rolling(window).min() + 1e-10)
            
            normalized_sentiment = (df['social_combined_sentiment'] - df['social_combined_sentiment'].rolling(window).min()) / \
                                 (df['social_combined_sentiment'].rolling(window).max() - df['social_combined_sentiment'].rolling(window).min() + 1e-10)
            
            # Tính chỉ số phân kỳ
            result_df[f'social_price_divergence_{window}d'] = normalized_sentiment - normalized_price
    
    # Tính chỉ số xung lượng tâm lý
    if 'social_combined_sentiment' in df.columns:
        for window in window_sizes:
            result_df[f'social_sentiment_momentum_{window}d'] = df['social_combined_sentiment'] - df['social_combined_sentiment'].shift(window)
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df