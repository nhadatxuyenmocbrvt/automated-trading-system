"""
Phát hiện sự kiện từ dữ liệu tâm lý.
File này cung cấp các lớp và hàm để phát hiện các sự kiện quan trọng
từ dữ liệu tâm lý thị trường, tin tức, và mạng xã hội.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import datetime
from pathlib import Path
import logging
from scipy import signal, stats
from collections import defaultdict

# Import các module nội bộ
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("sentiment_features")

class EventDetector:
    """
    Lớp phát hiện sự kiện từ dữ liệu tâm lý thị trường.
    """
    
    def __init__(self, window_size: int = 7, threshold: float = 2.0):
        """
        Khởi tạo đối tượng EventDetector.
        
        Args:
            window_size: Kích thước cửa sổ để phát hiện sự kiện
            threshold: Ngưỡng để xác định sự kiện (số lần độ lệch chuẩn)
        """
        self.window_size = window_size
        self.threshold = threshold
    
    def detect_sentiment_shifts(self, df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """
        Phát hiện sự thay đổi đột ngột trong tâm lý thị trường.
        
        Args:
            df: DataFrame chứa dữ liệu tâm lý (với một cột timestamp)
            sentiment_col: Tên cột chứa giá trị tâm lý
            
        Returns:
            DataFrame với các cột chỉ định sự kiện
        """
        # Kiểm tra dữ liệu đầu vào
        if sentiment_col not in df.columns:
            logger.error(f"Cột {sentiment_col} không tồn tại trong DataFrame")
            return pd.DataFrame()
        
        # Tạo bản sao DataFrame để lưu kết quả
        result_df = df.copy()
        
        # Tính trung bình di động và độ lệch chuẩn
        result_df[f'{sentiment_col}_ma'] = df[sentiment_col].rolling(window=self.window_size).mean()
        result_df[f'{sentiment_col}_std'] = df[sentiment_col].rolling(window=self.window_size).std()
        
        # Tính z-score
        result_df[f'{sentiment_col}_zscore'] = (
            (df[sentiment_col] - result_df[f'{sentiment_col}_ma']) / 
            result_df[f'{sentiment_col}_std'].replace(0, 1e-10)  # Tránh chia cho 0
        )
        
        # Xác định các sự kiện thay đổi lớn
        result_df[f'{sentiment_col}_shift_up'] = (result_df[f'{sentiment_col}_zscore'] > self.threshold).astype(int)
        result_df[f'{sentiment_col}_shift_down'] = (result_df[f'{sentiment_col}_zscore'] < -self.threshold).astype(int)
        
        # Xác định sự đảo ngược xu hướng
        result_df[f'{sentiment_col}_reversal'] = 0
        
        # Đảo chiều từ tích cực sang tiêu cực
        result_df.loc[
            (result_df[f'{sentiment_col}_zscore'] < 0) & 
            (result_df[f'{sentiment_col}_zscore'].shift(1) > 0) &
            (result_df[f'{sentiment_col}_zscore'].abs() > 1),
            f'{sentiment_col}_reversal'
        ] = -1
        
        # Đảo chiều từ tiêu cực sang tích cực
        result_df.loc[
            (result_df[f'{sentiment_col}_zscore'] > 0) & 
            (result_df[f'{sentiment_col}_zscore'].shift(1) < 0) &
            (result_df[f'{sentiment_col}_zscore'].abs() > 1),
            f'{sentiment_col}_reversal'
        ] = 1
        
        # Tìm các đỉnh và đáy cục bộ
        try:
            # Lọc nhiễu và tìm các đỉnh/đáy
            # Lưu ý: window_length phải là số lẻ
            smooth_data = signal.savgol_filter(
                df[sentiment_col].fillna(method='ffill').fillna(method='bfill'), 
                min(self.window_size * 2 + 1, len(df) if len(df) % 2 == 1 else len(df) - 1), 
                3
            )
            
            # Tìm các đỉnh cục bộ
            peaks, _ = signal.find_peaks(smooth_data, prominence=result_df[f'{sentiment_col}_std'].mean())
            
            # Tìm các đáy cục bộ (đỉnh của dữ liệu nghịch đảo)
            troughs, _ = signal.find_peaks(-smooth_data, prominence=result_df[f'{sentiment_col}_std'].mean())
            
            # Đánh dấu các đỉnh và đáy
            result_df[f'{sentiment_col}_peak'] = 0
            result_df[f'{sentiment_col}_trough'] = 0
            
            result_df.loc[peaks, f'{sentiment_col}_peak'] = 1
            result_df.loc[troughs, f'{sentiment_col}_trough'] = 1
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm đỉnh/đáy: {str(e)}")
        
        return result_df
    
    def detect_abnormal_social_activity(self, df: pd.DataFrame, 
                                      volume_col: str, 
                                      sentiment_col: Optional[str] = None) -> pd.DataFrame:
        """
        Phát hiện hoạt động bất thường trên mạng xã hội.
        
        Args:
            df: DataFrame chứa dữ liệu mạng xã hội (với một cột timestamp)
            volume_col: Tên cột chứa khối lượng hoạt động (số lượng tweet, post, v.v.)
            sentiment_col: Tên cột chứa giá trị tâm lý (tùy chọn)
            
        Returns:
            DataFrame với các cột chỉ định sự kiện
        """
        # Kiểm tra dữ liệu đầu vào
        if volume_col not in df.columns:
            logger.error(f"Cột {volume_col} không tồn tại trong DataFrame")
            return pd.DataFrame()
        
        # Tạo bản sao DataFrame để lưu kết quả
        result_df = df.copy()
        
        # Tính trung bình di động và độ lệch chuẩn cho khối lượng
        result_df[f'{volume_col}_ma'] = df[volume_col].rolling(window=self.window_size).mean()
        result_df[f'{volume_col}_std'] = df[volume_col].rolling(window=self.window_size).std()
        
        # Tính z-score
        result_df[f'{volume_col}_zscore'] = (
            (df[volume_col] - result_df[f'{volume_col}_ma']) / 
            result_df[f'{volume_col}_std'].replace(0, 1e-10)
        )
        
        # Xác định các sự kiện khối lượng bất thường
        result_df[f'{volume_col}_abnormal'] = (result_df[f'{volume_col}_zscore'] > self.threshold).astype(int)
        
        # Kiểm tra xu hướng khối lượng
        result_df[f'{volume_col}_trend'] = np.sign(result_df[volume_col].diff(self.window_size))
        
        # Nếu có cột tâm lý, tìm mối tương quan giữa khối lượng và tâm lý
        if sentiment_col and sentiment_col in df.columns:
            # Tính trung bình di động và độ lệch chuẩn cho tâm lý
            result_df[f'{sentiment_col}_ma'] = df[sentiment_col].rolling(window=self.window_size).mean()
            
            # Tính tương quan giữa khối lượng và tâm lý trong cửa sổ
            def rolling_correlation(x, y, window):
                return pd.Series(x).rolling(window).corr(pd.Series(y))
            
            result_df[f'{volume_col}_{sentiment_col}_corr'] = rolling_correlation(
                df[volume_col], df[sentiment_col], self.window_size
            )
            
            # Phát hiện xu hướng khối lượng tăng đột biến cùng với tâm lý cực đoan
            result_df[f'{volume_col}_{sentiment_col}_event'] = 0
            
            # Sự kiện tăng khối lượng + tâm lý rất tích cực
            result_df.loc[
                (result_df[f'{volume_col}_abnormal'] == 1) & 
                (result_df[f'{sentiment_col}_ma'] > result_df[f'{sentiment_col}_ma'].quantile(0.8)),
                f'{volume_col}_{sentiment_col}_event'
            ] = 1
            
            # Sự kiện tăng khối lượng + tâm lý rất tiêu cực
            result_df.loc[
                (result_df[f'{volume_col}_abnormal'] == 1) & 
                (result_df[f'{sentiment_col}_ma'] < result_df[f'{sentiment_col}_ma'].quantile(0.2)),
                f'{volume_col}_{sentiment_col}_event'
            ] = -1
        
        return result_df
    
    def detect_news_events(self, df: pd.DataFrame, 
                         volume_col: str, 
                         sentiment_col: Optional[str] = None) -> pd.DataFrame:
        """
        Phát hiện sự kiện từ tin tức.
        
        Args:
            df: DataFrame chứa dữ liệu tin tức (với một cột timestamp)
            volume_col: Tên cột chứa khối lượng tin tức
            sentiment_col: Tên cột chứa giá trị tâm lý tin tức (tùy chọn)
            
        Returns:
            DataFrame với các cột chỉ định sự kiện
        """
        # Kiểm tra dữ liệu đầu vào
        if volume_col not in df.columns:
            logger.error(f"Cột {volume_col} không tồn tại trong DataFrame")
            return pd.DataFrame()
        
        # Tạo bản sao DataFrame để lưu kết quả
        result_df = df.copy()
        
        # Tính trung bình di động và độ lệch chuẩn cho khối lượng tin tức
        result_df[f'{volume_col}_ma'] = df[volume_col].rolling(window=self.window_size).mean()
        result_df[f'{volume_col}_std'] = df[volume_col].rolling(window=self.window_size).std()
        
        # Tính z-score
        result_df[f'{volume_col}_zscore'] = (
            (df[volume_col] - result_df[f'{volume_col}_ma']) / 
            result_df[f'{volume_col}_std'].replace(0, 1e-10)
        )
        
        # Xác định các sự kiện khối lượng tin tức bất thường
        result_df[f'{volume_col}_abnormal'] = (result_df[f'{volume_col}_zscore'] > self.threshold).astype(int)
        
        # Nếu có cột tâm lý, tìm các sự kiện dựa trên tâm lý
        if sentiment_col and sentiment_col in df.columns:
            # Tính trung bình di động và độ lệch chuẩn cho tâm lý
            result_df[f'{sentiment_col}_ma'] = df[sentiment_col].rolling(window=self.window_size).mean()
            result_df[f'{sentiment_col}_std'] = df[sentiment_col].rolling(window=self.window_size).std()
            
            # Tính z-score cho tâm lý
            result_df[f'{sentiment_col}_zscore'] = (
                (df[sentiment_col] - result_df[f'{sentiment_col}_ma']) / 
                result_df[f'{sentiment_col}_std'].replace(0, 1e-10)
            )
            
            # Phát hiện các sự kiện đáng chú ý (khối lượng tin cao + tâm lý cực đoan)
            result_df['news_event_score'] = result_df[f'{volume_col}_zscore'] * result_df[f'{sentiment_col}_zscore'].abs()
            
            # Đánh dấu các sự kiện đáng chú ý
            news_event_threshold = self.threshold * 0.5
            result_df['news_significant_event'] = (result_df['news_event_score'] > news_event_threshold).astype(int)
            
            # Phân loại sự kiện theo tính chất tâm lý
            result_df['news_event_type'] = 0
            result_df.loc[
                (result_df['news_significant_event'] == 1) & (result_df[f'{sentiment_col}_zscore'] > 0),
                'news_event_type'
            ] = 1  # Sự kiện tích cực
            
            result_df.loc[
                (result_df['news_significant_event'] == 1) & (result_df[f'{sentiment_col}_zscore'] < 0),
                'news_event_type'
            ] = -1  # Sự kiện tiêu cực
        
        return result_df
    
    def compute_event_impact(self, event_df: pd.DataFrame, price_df: pd.DataFrame, 
                           event_col: str, look_forward: int = 3) -> pd.DataFrame:
        """
        Tính toán tác động của các sự kiện lên giá.
        
        Args:
            event_df: DataFrame chứa dữ liệu sự kiện (với cột timestamp và event_col)
            price_df: DataFrame chứa dữ liệu giá (với cột timestamp và close)
            event_col: Tên cột chỉ định sự kiện
            look_forward: Số ngày nhìn về phía trước để đánh giá tác động
            
        Returns:
            DataFrame với thông tin tác động của sự kiện
        """
        # Kiểm tra dữ liệu đầu vào
        if event_col not in event_df.columns:
            logger.error(f"Cột {event_col} không tồn tại trong event_df")
            return pd.DataFrame()
        
        if 'timestamp' not in event_df.columns or 'timestamp' not in price_df.columns:
            logger.error("Cả hai DataFrame đều phải có cột timestamp")
            return pd.DataFrame()
        
        if 'close' not in price_df.columns:
            logger.error("price_df phải có cột close")
            return pd.DataFrame()
        
        # Đảm bảo timestamp là datetime
        event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Tạo DataFrame kết quả
        result_df = event_df.copy()
        
        # Tính tác động lên giá
        result_df[f'{event_col}_price_impact_pct'] = np.nan
        result_df[f'{event_col}_price_impact_day'] = np.nan
        
        # Duyệt qua từng hàng có sự kiện
        for idx, row in result_df[result_df[event_col] != 0].iterrows():
            event_time = row['timestamp']
            
            # Tìm giá tại thời điểm sự kiện
            price_at_event = price_df[price_df['timestamp'] <= event_time]['close'].iloc[-1] if not price_df[price_df['timestamp'] <= event_time].empty else None
            
            if price_at_event is not None:
                # Tìm giá trong look_forward ngày sau sự kiện
                future_prices = price_df[
                    (price_df['timestamp'] > event_time) & 
                    (price_df['timestamp'] <= event_time + pd.Timedelta(days=look_forward))
                ]['close']
                
                if not future_prices.empty:
                    # Tìm giá cực đại và cực tiểu trong khoảng thời gian này
                    max_price = future_prices.max()
                    min_price = future_prices.min()
                    
                    # Tính tác động phần trăm (lấy tác động lớn nhất, có thể là tích cực hoặc tiêu cực)
                    max_impact_pct = (max_price - price_at_event) / price_at_event * 100
                    min_impact_pct = (min_price - price_at_event) / price_at_event * 100
                    
                    if abs(max_impact_pct) > abs(min_impact_pct):
                        impact_pct = max_impact_pct
                        # Tìm ngày có tác động lớn nhất
                        max_price_idx = future_prices.idxmax()
                        impact_day = (price_df.loc[max_price_idx, 'timestamp'] - event_time).days
                    else:
                        impact_pct = min_impact_pct
                        # Tìm ngày có tác động lớn nhất
                        min_price_idx = future_prices.idxmin()
                        impact_day = (price_df.loc[min_price_idx, 'timestamp'] - event_time).days
                    
                    # Lưu kết quả
                    result_df.loc[idx, f'{event_col}_price_impact_pct'] = impact_pct
                    result_df.loc[idx, f'{event_col}_price_impact_day'] = impact_day
        
        # Tính xác suất và mức độ tác động trung bình theo loại sự kiện
        if f'{event_col}_price_impact_pct' in result_df.columns:
            # Tạo các cột mới để lưu xác suất tác động
            event_types = result_df[event_col].unique()
            
            for event_type in event_types:
                if event_type != 0:  # Bỏ qua trường hợp không có sự kiện
                    # Lọc các sự kiện có cùng loại
                    same_events = result_df[result_df[event_col] == event_type]
                    
                    if not same_events.empty:
                        # Đếm số sự kiện có tác động tích cực/tiêu cực
                        positive_impact = same_events[f'{event_col}_price_impact_pct'] > 0
                        positive_count = positive_impact.sum()
                        negative_count = len(same_events) - positive_count
                        
                        # Tính xác suất tác động tích cực
                        positive_prob = positive_count / len(same_events) if len(same_events) > 0 else 0
                        
                        # Tính mức độ tác động trung bình
                        avg_positive_impact = same_events.loc[positive_impact, f'{event_col}_price_impact_pct'].mean() if positive_count > 0 else 0
                        avg_negative_impact = same_events.loc[~positive_impact, f'{event_col}_price_impact_pct'].mean() if negative_count > 0 else 0
                        
                        # Tạo cột mới cho mỗi loại sự kiện
                        event_type_str = str(event_type).replace('-', 'neg').replace('.', 'p')
                        result_df[f'{event_col}_type{event_type_str}_positive_prob'] = positive_prob
                        result_df[f'{event_col}_type{event_type_str}_avg_pos_impact'] = avg_positive_impact
                        result_df[f'{event_col}_type{event_type_str}_avg_neg_impact'] = avg_negative_impact
        
        return result_df


def detect_sentiment_shifts(df: pd.DataFrame, sentiment_col: str, 
                          window_size: int = 7, threshold: float = 2.0) -> pd.DataFrame:
    """
    Phát hiện sự thay đổi đột ngột trong tâm lý thị trường.
    
    Args:
        df: DataFrame chứa dữ liệu tâm lý (với một cột timestamp)
        sentiment_col: Tên cột chứa giá trị tâm lý
        window_size: Kích thước cửa sổ để phát hiện sự kiện
        threshold: Ngưỡng để xác định sự kiện (số lần độ lệch chuẩn)
        
    Returns:
        DataFrame với các cột chỉ định sự kiện
    """
    detector = EventDetector(window_size=window_size, threshold=threshold)
    return detector.detect_sentiment_shifts(df, sentiment_col)


def detect_abnormal_social_activity(df: pd.DataFrame, volume_col: str, 
                                  sentiment_col: Optional[str] = None,
                                  window_size: int = 7, threshold: float = 2.0) -> pd.DataFrame:
    """
    Phát hiện hoạt động bất thường trên mạng xã hội.
    
    Args:
        df: DataFrame chứa dữ liệu mạng xã hội (với một cột timestamp)
        volume_col: Tên cột chứa khối lượng hoạt động (số lượng tweet, post, v.v.)
        sentiment_col: Tên cột chứa giá trị tâm lý (tùy chọn)
        window_size: Kích thước cửa sổ để phát hiện sự kiện
        threshold: Ngưỡng để xác định sự kiện (số lần độ lệch chuẩn)
        
    Returns:
        DataFrame với các cột chỉ định sự kiện
    """
    detector = EventDetector(window_size=window_size, threshold=threshold)
    return detector.detect_abnormal_social_activity(df, volume_col, sentiment_col)


def detect_news_events(df: pd.DataFrame, volume_col: str, 
                      sentiment_col: Optional[str] = None,
                      window_size: int = 7, threshold: float = 2.0) -> pd.DataFrame:
    """
    Phát hiện sự kiện từ tin tức.
    
    Args:
        df: DataFrame chứa dữ liệu tin tức (với một cột timestamp)
        volume_col: Tên cột chứa khối lượng tin tức
        sentiment_col: Tên cột chứa giá trị tâm lý tin tức (tùy chọn)
        window_size: Kích thước cửa sổ để phát hiện sự kiện
        threshold: Ngưỡng để xác định sự kiện (số lần độ lệch chuẩn)
        
    Returns:
        DataFrame với các cột chỉ định sự kiện
    """
    detector = EventDetector(window_size=window_size, threshold=threshold)
    return detector.detect_news_events(df, volume_col, sentiment_col)


def compute_event_impact(event_df: pd.DataFrame, price_df: pd.DataFrame, 
                        event_col: str, look_forward: int = 3,
                        window_size: int = 7, threshold: float = 2.0) -> pd.DataFrame:
    """
    Tính toán tác động của các sự kiện lên giá.
    
    Args:
        event_df: DataFrame chứa dữ liệu sự kiện (với cột timestamp và event_col)
        price_df: DataFrame chứa dữ liệu giá (với cột timestamp và close)
        event_col: Tên cột chỉ định sự kiện
        look_forward: Số ngày nhìn về phía trước để đánh giá tác động
        window_size: Kích thước cửa sổ để phát hiện sự kiện
        threshold: Ngưỡng để xác định sự kiện (số lần độ lệch chuẩn)
        
    Returns:
        DataFrame với thông tin tác động của sự kiện
    """
    detector = EventDetector(window_size=window_size, threshold=threshold)
    return detector.compute_event_impact(event_df, price_df, event_col, look_forward)


def create_sentiment_event_features(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng sự kiện tâm lý kết hợp với dữ liệu giá.
    
    Args:
        sentiment_df: DataFrame chứa dữ liệu tâm lý
        price_df: DataFrame chứa dữ liệu giá
        
    Returns:
        DataFrame kết hợp với các đặc trưng sự kiện
    """
    # Kiểm tra dữ liệu đầu vào
    required_columns = ['timestamp', 'close']
    for col in required_columns:
        if col not in price_df.columns:
            logger.error(f"Cột {col} không tồn tại trong price_df")
            return price_df
    
    if 'timestamp' not in sentiment_df.columns:
        logger.error("Cột timestamp không tồn tại trong sentiment_df")
        return price_df
    
    # Đảm bảo timestamp là datetime
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    
    # Tạo bản sao DataFrame giá để thêm đặc trưng
    result_df = price_df.copy()
    
    # Tìm các cột tâm lý có thể sử dụng
    sentiment_columns = []
    volume_columns = []
    
    for col in sentiment_df.columns:
        if 'sentiment' in col.lower() and sentiment_df[col].dtype in [np.float64, np.int64]:
            sentiment_columns.append(col)
        
        if any(term in col.lower() for term in ['volume', 'count', 'activity']) and sentiment_df[col].dtype in [np.float64, np.int64]:
            volume_columns.append(col)
    
    # Thông báo các cột sẽ sử dụng
    logger.info(f"Các cột tâm lý: {sentiment_columns}")
    logger.info(f"Các cột khối lượng: {volume_columns}")
    
    # Phát hiện sự kiện từ mỗi cột tâm lý
    event_dfs = []
    
    for sentiment_col in sentiment_columns:
        try:
            # Phát hiện sự thay đổi tâm lý
            event_df = detect_sentiment_shifts(sentiment_df, sentiment_col)
            
            # Tính tác động lên giá
            if f'{sentiment_col}_shift_up' in event_df.columns:
                event_df = compute_event_impact(event_df, price_df, f'{sentiment_col}_shift_up')
            
            if f'{sentiment_col}_shift_down' in event_df.columns:
                event_df = compute_event_impact(event_df, price_df, f'{sentiment_col}_shift_down')
            
            if f'{sentiment_col}_reversal' in event_df.columns:
                event_df = compute_event_impact(event_df, price_df, f'{sentiment_col}_reversal')
            
            # Lấy các cột sự kiện
            event_cols = [col for col in event_df.columns if any(term in col for term in 
                                                             ['_shift_up', '_shift_down', '_reversal', '_peak', '_trough', '_impact'])]
            
            # Chỉ giữ lại cột timestamp và các cột sự kiện
            event_df = event_df[['timestamp'] + event_cols]
            
            event_dfs.append(event_df)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý cột {sentiment_col}: {str(e)}")
    
    # Phát hiện sự kiện từ khối lượng mạng xã hội và tin tức
    for volume_col in volume_columns:
        try:
            # Tìm cột tâm lý tương ứng nếu có
            related_sentiment_col = None
            
            # Ví dụ: volume_col = "twitter_count_daily" -> tìm "twitter_sentiment_daily"
            volume_prefix = volume_col.split('_')[0]
            for col in sentiment_columns:
                if volume_prefix in col:
                    related_sentiment_col = col
                    break
            
            # Phát hiện hoạt động bất thường
            if 'news' in volume_col:
                event_df = detect_news_events(sentiment_df, volume_col, related_sentiment_col)
            else:
                event_df = detect_abnormal_social_activity(sentiment_df, volume_col, related_sentiment_col)
            
            # Tính tác động lên giá
            if f'{volume_col}_abnormal' in event_df.columns:
                event_df = compute_event_impact(event_df, price_df, f'{volume_col}_abnormal')
            
            if 'news_significant_event' in event_df.columns:
                event_df = compute_event_impact(event_df, price_df, 'news_significant_event')
            
            # Lấy các cột sự kiện
            event_cols = [col for col in event_df.columns if any(term in col for term in 
                                                             ['_abnormal', '_event', 'significant_event', '_event_type', '_impact'])]
            
            # Chỉ giữ lại cột timestamp và các cột sự kiện
            event_df = event_df[['timestamp'] + event_cols]
            
            event_dfs.append(event_df)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý cột {volume_col}: {str(e)}")
    
    # Kết hợp tất cả DataFrame sự kiện với DataFrame giá
    if event_dfs:
        # Kết hợp tất cả DataFrame sự kiện
        combined_event_df = event_dfs[0]
        for df in event_dfs[1:]:
            combined_event_df = pd.merge(combined_event_df, df, on='timestamp', how='outer')
        
        # Kết hợp với DataFrame giá
        result_df = pd.merge_asof(result_df.sort_values('timestamp'), 
                                  combined_event_df.sort_values('timestamp'), 
                                  on='timestamp', 
                                  direction='backward')
        
        # Điền các giá trị NaN với 0
        for col in result_df.columns:
            if col != 'timestamp' and col not in price_df.columns:
                result_df[col] = result_df[col].fillna(0)
        
        logger.info(f"Đã thêm {len(result_df.columns) - len(price_df.columns)} cột đặc trưng sự kiện")
    
    return result_df