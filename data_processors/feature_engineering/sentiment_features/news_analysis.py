"""
Đặc trưng từ phân tích tin tức.
File này cung cấp các lớp và hàm để tạo đặc trưng từ dữ liệu tin tức
tiền điện tử, bao gồm phân tích nội dung, khối lượng, và chủ đề.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import datetime
from pathlib import Path
import logging
import json
import re
from collections import Counter, defaultdict

# Import các module nội bộ
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("sentiment_features")

class NewsFeatures:
    """
    Lớp xử lý và tạo đặc trưng từ dữ liệu tin tức.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Khởi tạo đối tượng NewsFeatures.
        
        Args:
            data_dir: Thư mục chứa dữ liệu tin tức (tùy chọn)
        """
        self.data_dir = data_dir
        
        if self.data_dir is None:
            # Sử dụng thư mục mặc định nếu không được chỉ định
            from config.system_config import BASE_DIR
            self.data_dir = BASE_DIR / "data" / "news"
        
        # Đảm bảo thư mục tồn tại
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Từ điển các từ khóa tích cực và tiêu cực
        self.positive_keywords = [
            "bullish", "surge", "rally", "gain", "rise", "soar", "jump", "breakout", 
            "uptrend", "strong", "positive", "growth", "adoption", "opportunity", 
            "optimistic", "promising", "success", "partnership", "innovation"
        ]
        
        self.negative_keywords = [
            "bearish", "crash", "plunge", "drop", "fall", "decline", "slump", "correction", 
            "downtrend", "weak", "negative", "loss", "ban", "regulate", "concern", 
            "risk", "warning", "threat", "fear", "panic", "sell-off", "volatile"
        ]
        
        # Từ điển chủ đề và từ khóa liên quan
        self.topic_keywords = {
            "regulation": ["regulation", "regulatory", "sec", "cftc", "compliance", "legal", 
                         "government", "policy", "law", "regulator", "ban", "approve"],
            "adoption": ["adoption", "mainstream", "institutional", "retail", "corporate", 
                       "integrate", "accept", "payment", "wallet", "user"],
            "technology": ["technology", "blockchain", "protocol", "upgrade", "fork", 
                         "layer", "scaling", "network", "node", "development"],
            "market": ["market", "price", "volume", "trading", "exchange", "liquidity", 
                     "volatility", "momentum", "pullback", "consolidation", "accumulation"],
            "defi": ["defi", "decentralized", "finance", "lending", "borrowing", "yield", 
                   "farming", "liquidity", "pool", "swap", "stake", "governance"],
            "nft": ["nft", "non-fungible", "token", "art", "collectible", "game", 
                  "metaverse", "virtual", "digital", "asset", "ownership"],
            "security": ["security", "hack", "breach", "vulnerability", "exploit", 
                       "attack", "stolen", "theft", "phishing", "scam", "fraud"]
        }
    
    def load_news_data(self, file_path: Optional[Union[str, Path]] = None, 
                      days: int = 30, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Tải dữ liệu tin tức từ file hoặc lấy dữ liệu gần đây.
        
        Args:
            file_path: Đường dẫn đến file dữ liệu cụ thể (tùy chọn)
            days: Số ngày dữ liệu cần lấy nếu không chỉ định file
            asset: Mã tài sản cần lọc (tùy chọn)
            
        Returns:
            Danh sách dữ liệu tin tức
        """
        if file_path:
            # Tải dữ liệu từ file cụ thể
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                
                logger.info(f"Đã tải {len(news_data)} bài viết tin tức từ {file_path}")
                
            except Exception as e:
                logger.error(f"Lỗi khi tải dữ liệu tin tức: {str(e)}")
                news_data = []
        else:
            # Tìm các file trong khoảng thời gian
            news_data = []
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
                        batch_data = json.load(f)
                    
                    # Loại bỏ các bài viết không có nội dung hoặc mô tả
                    batch_data = [article for article in batch_data 
                                 if article.get('content') or article.get('description')]
                    
                    news_data.extend(batch_data)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi tải file {json_file}: {str(e)}")
            
            logger.info(f"Đã tải tổng cộng {len(news_data)} bài viết tin tức từ {len(json_files)} file")
        
        # Lọc tin tức theo tài sản nếu cần
        if asset:
            filtered_news = []
            asset_keywords = [asset.lower()]
            
            # Thêm các từ khóa liên quan cho các tài sản phổ biến
            asset_mappings = {
                "btc": ["bitcoin", "btc", "xbt"],
                "eth": ["ethereum", "eth", "ether"],
                "xrp": ["ripple", "xrp"],
                "sol": ["solana", "sol"],
                "ada": ["cardano", "ada"],
                "bnb": ["binance", "bnb", "binance coin"]
            }
            
            if asset.lower() in asset_mappings:
                asset_keywords = asset_mappings[asset.lower()]
            
            # Lọc bài viết
            for article in news_data:
                content = (article.get('content', '') or '') + ' ' + (article.get('description', '') or '') + ' ' + (article.get('title', '') or '')
                content = content.lower()
                
                if any(keyword in content for keyword in asset_keywords):
                    filtered_news.append(article)
            
            news_data = filtered_news
            logger.info(f"Đã lọc {len(news_data)} bài viết cho tài sản {asset}")
        
        return news_data
    
    def news_to_dataframe(self, news_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Chuyển đổi danh sách tin tức thành DataFrame.
        
        Args:
            news_data: Danh sách dữ liệu tin tức
            
        Returns:
            DataFrame chứa dữ liệu tin tức
        """
        if not news_data:
            return pd.DataFrame()
        
        # Tạo các bản ghi chuẩn hóa
        records = []
        for article in news_data:
            try:
                # Đảm bảo các trường cần thiết tồn tại
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                source = article.get('source', '')
                url = article.get('url', '')
                
                # Xử lý timestamp
                published_at = article.get('published_at')
                if published_at:
                    if isinstance(published_at, str):
                        # Thử nhiều định dạng ngày khác nhau
                        try:
                            timestamp = pd.to_datetime(published_at)
                        except:
                            # Fallback nếu không thể parse
                            timestamp = pd.to_datetime(article.get('collected_at', datetime.datetime.now().isoformat()))
                    else:
                        timestamp = pd.to_datetime(published_at)
                else:
                    timestamp = pd.to_datetime(article.get('collected_at', datetime.datetime.now().isoformat()))
                
                # Tính điểm tâm lý
                sentiment_score = self.calculate_sentiment(title, description, content)
                
                # Phát hiện chủ đề
                topics = self.detect_topics(title, description, content)
                
                # Tạo bản ghi
                record = {
                    'timestamp': timestamp,
                    'title': title,
                    'source': source,
                    'url': url,
                    'sentiment_score': sentiment_score,
                    'topics': topics
                }
                
                records.append(record)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý bài viết tin tức: {str(e)}")
        
        # Tạo DataFrame
        df = pd.DataFrame(records)
        
        # Đảm bảo timestamp là datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sắp xếp theo thời gian
        df = df.sort_values('timestamp')
        
        return df
    
    def calculate_sentiment(self, title: str, description: str, content: str) -> float:
        """
        Tính điểm tâm lý cho một bài viết tin tức.
        
        Args:
            title: Tiêu đề bài viết
            description: Mô tả bài viết
            content: Nội dung bài viết
            
        Returns:
            Điểm tâm lý (-1 đến 1)
        """
        # Kết hợp văn bản, với trọng số cho tiêu đề
        text = (title + " " + title + " " + description + " " + (content or "")).lower()
        
        # Đếm từ khóa tích cực và tiêu cực
        positive_count = sum(text.count(word) for word in self.positive_keywords)
        negative_count = sum(text.count(word) for word in self.negative_keywords)
        
        # Tính điểm tâm lý
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0  # Trung tính
        else:
            return (positive_count - negative_count) / total_count
    
    def detect_topics(self, title: str, description: str, content: str) -> List[str]:
        """
        Phát hiện chủ đề của bài viết tin tức.
        
        Args:
            title: Tiêu đề bài viết
            description: Mô tả bài viết
            content: Nội dung bài viết
            
        Returns:
            Danh sách chủ đề
        """
        # Kết hợp văn bản, với trọng số cho tiêu đề
        text = (title + " " + title + " " + description + " " + (content or "")).lower()
        
        # Kiểm tra từng chủ đề
        detected_topics = []
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def extract_features(self, price_df: pd.DataFrame, news_data: List[Dict[str, Any]],
                        window_sizes: List[int] = [1, 3, 7, 14],
                        normalize: bool = True) -> pd.DataFrame:
        """
        Tạo các đặc trưng từ dữ liệu tin tức và kết hợp với dữ liệu giá.
        
        Args:
            price_df: DataFrame dữ liệu giá (phải có cột 'timestamp', 'close')
            news_data: Danh sách dữ liệu tin tức
            window_sizes: Các kích thước cửa sổ cho tính toán
            normalize: Chuẩn hóa đặc trưng (True/False)
            
        Returns:
            DataFrame kết hợp giữa dữ liệu giá và đặc trưng tin tức
        """
        # Chuyển đổi dữ liệu tin tức thành DataFrame
        news_df = self.news_to_dataframe(news_data)
        
        if news_df.empty:
            logger.warning("Không có dữ liệu tin tức để tạo đặc trưng")
            return price_df
        
        # Đảm bảo dữ liệu giá có cột timestamp là datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Tạo bản sao của DataFrame giá để thêm đặc trưng
        result_df = price_df.copy()
        
        # Tạo các đặc trưng tin tức
        sentiment_features = extract_news_sentiment_features(news_df, result_df, window_sizes)
        topic_features = extract_news_topic_features(news_df, result_df, window_sizes)
        volume_features = extract_news_volume_features(news_df, result_df, window_sizes)
        
        # Kết hợp đặc trưng
        feature_dfs = [sentiment_features, topic_features, volume_features]
        for feature_df in feature_dfs:
            for col in feature_df.columns:
                if col not in result_df.columns:
                    result_df[col] = feature_df[col]
        
        # Chuẩn hóa đặc trưng nếu cần
        if normalize:
            news_columns = [col for col in result_df.columns if 'news_' in col]
            
            for col in news_columns:
                if col in result_df.columns and result_df[col].dtype in [np.float64, np.int64]:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std > 0:
                        result_df[col] = (result_df[col] - mean) / std
        
        return result_df


def extract_news_sentiment_features(news_df: pd.DataFrame, price_df: pd.DataFrame,
                                   window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tạo đặc trưng tâm lý từ dữ liệu tin tức.
    
    Args:
        news_df: DataFrame dữ liệu tin tức
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng tâm lý tin tức
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Đặt timestamp làm index cho news_df
    news_df = news_df.sort_values('timestamp')
    
    # Tính điểm tâm lý trung bình theo ngày
    daily_news = news_df.groupby(news_df['timestamp'].dt.date)['sentiment_score'].agg(['mean', 'std', 'count'])
    daily_news['date'] = daily_news.index
    daily_news = daily_news.reset_index(drop=True)
    daily_news['date'] = pd.to_datetime(daily_news['date'])
    
    # Với mỗi timestamp trong price_df, tìm dữ liệu tin tức gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        date = timestamp.date()
        
        # Tìm ngày gần nhất có dữ liệu tin tức
        relevant_news = daily_news[daily_news['date'] <= timestamp].iloc[-1:] if not daily_news[daily_news['date'] <= timestamp].empty else None
        
        if relevant_news is not None and not relevant_news.empty:
            result_df.loc[idx, 'news_sentiment_daily'] = relevant_news['mean'].values[0]
            result_df.loc[idx, 'news_sentiment_std'] = relevant_news['std'].values[0] if not np.isnan(relevant_news['std'].values[0]) else 0
            result_df.loc[idx, 'news_count_daily'] = relevant_news['count'].values[0]
        else:
            # Nếu không có tin tức trước timestamp này, đặt giá trị mặc định
            result_df.loc[idx, 'news_sentiment_daily'] = 0
            result_df.loc[idx, 'news_sentiment_std'] = 0
            result_df.loc[idx, 'news_count_daily'] = 0
    
    # Tính toán các chỉ số trung bình cho các khoảng thời gian khác nhau
    for window in window_sizes:
        # Điểm tâm lý trung bình cho cửa sổ
        result_df[f'news_sentiment_avg_{window}d'] = result_df['news_sentiment_daily'].rolling(window).mean()
        
        # Độ lệch chuẩn của tâm lý
        result_df[f'news_sentiment_std_{window}d'] = result_df['news_sentiment_daily'].rolling(window).std()
        
        # Xu hướng tâm lý (độ dốc của đường hồi quy tuyến tính)
        result_df[f'news_sentiment_trend_{window}d'] = (
            result_df['news_sentiment_daily'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
        )
        
        # Số lượng tin tức
        result_df[f'news_count_avg_{window}d'] = result_df['news_count_daily'].rolling(window).mean()
    
    # Tính chỉ số tích lũy tâm lý tin tức
    result_df['news_sentiment_cumulative'] = result_df['news_sentiment_daily'].ewm(span=30, adjust=False).mean()
    
    # Điều chỉnh tâm lý theo khối lượng tin tức
    result_df['news_sentiment_volume_adjusted'] = result_df['news_sentiment_daily'] * np.log1p(result_df['news_count_daily'])
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return result_df


def extract_news_topic_features(news_df: pd.DataFrame, price_df: pd.DataFrame,
                               window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tạo đặc trưng chủ đề từ dữ liệu tin tức.
    
    Args:
        news_df: DataFrame dữ liệu tin tức
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng chủ đề tin tức
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Tạo cột cho mỗi chủ đề
    all_topics = set()
    for topics in news_df['topics']:
        all_topics.update(topics)
    
    # Chuẩn bị dữ liệu chủ đề theo ngày
    daily_topics = {}
    
    for date, group in news_df.groupby(news_df['timestamp'].dt.date):
        topic_counts = {topic: 0 for topic in all_topics}
        
        for topics_list in group['topics']:
            for topic in topics_list:
                topic_counts[topic] += 1
        
        daily_topics[date] = topic_counts
    
    # Sắp xếp dữ liệu theo ngày
    sorted_dates = sorted(daily_topics.keys())
    
    # Với mỗi timestamp trong price_df, tìm dữ liệu chủ đề gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        date = timestamp.date()
        
        # Tìm ngày gần nhất có dữ liệu tin tức
        closest_date = None
        for d in reversed(sorted_dates):
            if d <= date:
                closest_date = d
                break
        
        if closest_date:
            # Gán số lượng cho mỗi chủ đề
            topic_counts = daily_topics[closest_date]
            for topic in all_topics:
                result_df.loc[idx, f'news_topic_{topic}_count'] = topic_counts.get(topic, 0)
        else:
            # Nếu không có tin tức trước timestamp này, đặt giá trị mặc định
            for topic in all_topics:
                result_df.loc[idx, f'news_topic_{topic}_count'] = 0
    
    # Tính tổng số chủ đề mỗi ngày
    for topic in all_topics:
        topic_col = f'news_topic_{topic}_count'
        
        # Tính tỷ lệ chủ đề
        total_topics = result_df[[col for col in result_df.columns if 'news_topic_' in col and '_count' in col]].sum(axis=1)
        result_df[f'news_topic_{topic}_ratio'] = result_df[topic_col] / total_topics.replace(0, 1)
        
        # Tính trung bình di động cho các cửa sổ
        for window in window_sizes:
            result_df[f'news_topic_{topic}_avg_{window}d'] = result_df[topic_col].rolling(window).mean()
            result_df[f'news_topic_{topic}_ratio_avg_{window}d'] = result_df[f'news_topic_{topic}_ratio'].rolling(window).mean()
    
    # Tính tâm lý theo chủ đề nếu có cột sentiment_score trong news_df
    if 'sentiment_score' in news_df.columns:
        # Tính trung bình tâm lý theo chủ đề và ngày
        daily_topic_sentiment = defaultdict(lambda: defaultdict(list))
        
        for _, row in news_df.iterrows():
            date = row['timestamp'].date()
            sentiment = row['sentiment_score']
            
            for topic in row['topics']:
                daily_topic_sentiment[date][topic].append(sentiment)
        
        # Chuyển đổi danh sách thành trung bình
        for date in daily_topic_sentiment:
            for topic in daily_topic_sentiment[date]:
                daily_topic_sentiment[date][topic] = np.mean(daily_topic_sentiment[date][topic])
        
        # Với mỗi timestamp trong price_df, tìm dữ liệu tâm lý chủ đề gần nhất
        for idx, row in result_df.iterrows():
            timestamp = row['timestamp']
            date = timestamp.date()
            
            # Tìm ngày gần nhất có dữ liệu tin tức
            closest_date = None
            for d in reversed(sorted_dates):
                if d <= date:
                    closest_date = d
                    break
            
            if closest_date and closest_date in daily_topic_sentiment:
                # Gán tâm lý cho mỗi chủ đề
                topic_sentiments = daily_topic_sentiment[closest_date]
                for topic in all_topics:
                    if topic in topic_sentiments:
                        result_df.loc[idx, f'news_topic_{topic}_sentiment'] = topic_sentiments[topic]
                    else:
                        result_df.loc[idx, f'news_topic_{topic}_sentiment'] = 0
            else:
                # Nếu không có tin tức trước timestamp này, đặt giá trị mặc định
                for topic in all_topics:
                    result_df.loc[idx, f'news_topic_{topic}_sentiment'] = 0
        
        # Tính trung bình di động tâm lý chủ đề
        for topic in all_topics:
            topic_sent_col = f'news_topic_{topic}_sentiment'
            
            # Tính trung bình di động cho các cửa sổ
            for window in window_sizes:
                result_df[f'news_topic_{topic}_sentiment_avg_{window}d'] = result_df[topic_sent_col].rolling(window).mean()
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return result_df


def extract_news_volume_features(news_df: pd.DataFrame, price_df: pd.DataFrame,
                               window_sizes: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Tạo đặc trưng khối lượng tin tức.
    
    Args:
        news_df: DataFrame dữ liệu tin tức
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng khối lượng tin tức
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Tính khối lượng tin tức theo ngày
    daily_counts = news_df.groupby(news_df['timestamp'].dt.date).size()
    daily_counts_df = pd.DataFrame({'date': daily_counts.index, 'count': daily_counts.values})
    daily_counts_df['date'] = pd.to_datetime(daily_counts_df['date'])
    
    # Với mỗi timestamp trong price_df, tìm khối lượng tin tức gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        date = timestamp.date()
        
        # Tìm ngày gần nhất có dữ liệu tin tức
        relevant_day = daily_counts_df[daily_counts_df['date'] <= timestamp].iloc[-1:] if not daily_counts_df[daily_counts_df['date'] <= timestamp].empty else None
        
        if relevant_day is not None and not relevant_day.empty:
            result_df.loc[idx, 'news_volume_daily'] = relevant_day['count'].values[0]
        else:
            # Nếu không có tin tức trước timestamp này, đặt giá trị mặc định
            result_df.loc[idx, 'news_volume_daily'] = 0
    
    # Tính toán các chỉ số khối lượng cho các khoảng thời gian khác nhau
    for window in window_sizes:
        # Khối lượng trung bình cho cửa sổ
        result_df[f'news_volume_avg_{window}d'] = result_df['news_volume_daily'].rolling(window).mean()
        
        # Độ lệch chuẩn của khối lượng
        result_df[f'news_volume_std_{window}d'] = result_df['news_volume_daily'].rolling(window).std()
        
        # Xu hướng khối lượng (độ dốc của đường hồi quy tuyến tính)
        result_df[f'news_volume_trend_{window}d'] = (
            result_df['news_volume_daily'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
        )
    
    # Tính chỉ số bất thường khối lượng tin tức (Z-score)
    long_window = max(window_sizes)
    result_df['news_volume_zscore'] = (
        (result_df['news_volume_daily'] - result_df[f'news_volume_avg_{long_window}d']) /
        result_df[f'news_volume_std_{long_window}d'].replace(0, 1)
    )
    
    # Tính chỉ số xung lượng khối lượng tin tức
    result_df['news_volume_momentum'] = result_df['news_volume_daily'] - result_df['news_volume_daily'].shift(3)
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return result_df