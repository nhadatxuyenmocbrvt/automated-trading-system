"""
Đặc trưng từ chỉ số tâm lý thị trường.
File này cung cấp các lớp và hàm để tạo đặc trưng từ các chỉ số tâm lý thị trường
như Fear & Greed Index, on-chain sentiment, và các chỉ số tâm lý khác.
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

class MarketSentimentFeatures:
    """
    Lớp xử lý và tạo đặc trưng từ chỉ số tâm lý thị trường.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Khởi tạo đối tượng MarketSentimentFeatures.
        
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
            sentiment_data = [data for data in sentiment_data if data.asset == asset or data.asset is None]
            logger.info(f"Đã lọc {len(sentiment_data)} bản ghi cho tài sản {asset}")
        
        # Lọc chỉ lấy dữ liệu từ các chỉ số tâm lý thị trường
        market_sources = ["Fear and Greed Index", "GlassNode"]
        sentiment_data = [data for data in sentiment_data 
                         if any(source in data.source for source in market_sources)]
        
        logger.info(f"Dữ liệu tâm lý thị trường: {len(sentiment_data)} bản ghi")
        
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
                        window_sizes: List[int] = [1, 3, 7, 14, 30],
                        normalize: bool = True) -> pd.DataFrame:
        """
        Tạo các đặc trưng tâm lý thị trường và kết hợp với dữ liệu giá.
        
        Args:
            price_df: DataFrame dữ liệu giá (phải có cột 'timestamp', 'close')
            sentiment_data: Danh sách dữ liệu tâm lý thị trường
            window_sizes: Các kích thước cửa sổ cho tính toán trung bình và xu hướng
            normalize: Chuẩn hóa đặc trưng (True/False)
            
        Returns:
            DataFrame kết hợp giữa dữ liệu giá và đặc trưng tâm lý
        """
        # Chuyển đổi dữ liệu tâm lý thành DataFrame
        sentiment_df = self.sentiment_to_dataframe(sentiment_data)
        
        if sentiment_df.empty:
            logger.warning("Không có dữ liệu tâm lý thị trường để tạo đặc trưng")
            return price_df
        
        # Đảm bảo dữ liệu giá có cột timestamp là datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Tạo bản sao của DataFrame giá để thêm đặc trưng
        result_df = price_df.copy()
        
        # Tách dữ liệu tâm lý theo nguồn
        fear_greed_df = sentiment_df[sentiment_df['source'].str.contains('Fear and Greed')]
        glassnode_df = sentiment_df[sentiment_df['source'].str.contains('GlassNode')]
        
        # Tạo đặc trưng Fear & Greed
        if not fear_greed_df.empty:
            fg_features = extract_fear_greed_features(fear_greed_df, result_df, window_sizes)
            # Kết hợp đặc trưng
            for col in fg_features.columns:
                if col not in result_df.columns:
                    result_df[col] = fg_features[col]
        
        # Tạo đặc trưng on-chain
        if not glassnode_df.empty:
            onchain_features = extract_on_chain_sentiment_features(glassnode_df, result_df, window_sizes)
            # Kết hợp đặc trưng
            for col in onchain_features.columns:
                if col not in result_df.columns:
                    result_df[col] = onchain_features[col]
        
        # Tính toán đặc trưng phân kỳ tâm lý
        if 'fear_greed_value' in result_df.columns or 'onchain_sentiment_value' in result_df.columns:
            divergence_features = extract_sentiment_divergence_features(result_df, window_sizes)
            for col in divergence_features.columns:
                if col not in result_df.columns:
                    result_df[col] = divergence_features[col]
        
        # Chuẩn hóa đặc trưng nếu cần
        if normalize:
            sentiment_columns = [col for col in result_df.columns 
                                if any(pattern in col.lower() for pattern in 
                                    ['fear_greed', 'onchain', 'sentiment_divergence'])]
            
            for col in sentiment_columns:
                if col in result_df.columns and result_df[col].dtype in [np.float64, np.int64]:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std > 0:
                        result_df[col] = (result_df[col] - mean) / std
        
        return result_df


def extract_fear_greed_features(fear_greed_df: pd.DataFrame, price_df: pd.DataFrame,
                               window_sizes: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Tạo đặc trưng từ chỉ số Fear & Greed.
    
    Args:
        fear_greed_df: DataFrame dữ liệu Fear & Greed
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng Fear & Greed
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Đặt timestamp làm index cho fear_greed_df
    fear_greed_df = fear_greed_df.sort_values('timestamp')
    
    # Tạo numeric value từ dữ liệu Fear & Greed 
    # (từ 0-100, với 0 = Extreme Fear, 100 = Extreme Greed)
    
    # Với mỗi timestamp trong price_df, tìm giá trị Fear & Greed gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        
        # Lấy bản ghi gần nhất trước timestamp hiện tại
        prev_records = fear_greed_df[fear_greed_df['timestamp'] <= timestamp]
        
        if not prev_records.empty:
            latest_record = prev_records.iloc[-1]
            result_df.loc[idx, 'fear_greed_value'] = latest_record['value']
            result_df.loc[idx, 'fear_greed_label'] = latest_record['label']
            
            # Chuyển đổi nhãn thành giá trị số (cần cho các tính toán)
            label_to_numeric = {
                "Extreme Fear": 0,
                "Fear": 25,
                "Neutral": 50,
                "Greed": 75,
                "Extreme Greed": 100
            }
            
            if latest_record['label'] in label_to_numeric:
                result_df.loc[idx, 'fear_greed_value_numeric'] = label_to_numeric[latest_record['label']]
            elif latest_record['value'] >= 0 and latest_record['value'] <= 100:
                # Nếu giá trị đã trong thang 0-100, sử dụng trực tiếp
                result_df.loc[idx, 'fear_greed_value_numeric'] = latest_record['value']
            else:
                # Giá trị mặc định nếu không phù hợp
                result_df.loc[idx, 'fear_greed_value_numeric'] = 50
                
        else:
            # Nếu không có dữ liệu trước timestamp này, đặt giá trị mặc định
            result_df.loc[idx, 'fear_greed_value'] = 0
            result_df.loc[idx, 'fear_greed_label'] = "Unknown"
            result_df.loc[idx, 'fear_greed_value_numeric'] = 50
    
    # Tính toán các chỉ số trung bình cho các khoảng thời gian khác nhau
    for window in window_sizes:
        # Điểm Fear & Greed trung bình cho cửa sổ
        result_df[f'fear_greed_avg_{window}d'] = result_df['fear_greed_value_numeric'].rolling(window).mean()
        
        # Độ lệch chuẩn của Fear & Greed
        result_df[f'fear_greed_std_{window}d'] = result_df['fear_greed_value_numeric'].rolling(window).std()
        
        # Xu hướng Fear & Greed (độ dốc của đường hồi quy tuyến tính)
        result_df[f'fear_greed_trend_{window}d'] = (
            result_df['fear_greed_value_numeric'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
        )
    
    # Tính chỉ số xung lượng Fear & Greed
    for window in [1, 3, 7]:
        result_df[f'fear_greed_momentum_{window}d'] = result_df['fear_greed_value_numeric'] - result_df['fear_greed_value_numeric'].shift(window)
    
    # Lọc các điểm cực trị (đảo chiều tâm lý)
    result_df['fear_greed_is_local_min'] = (
        (result_df['fear_greed_value_numeric'] < result_df['fear_greed_value_numeric'].shift(1)) &
        (result_df['fear_greed_value_numeric'] < result_df['fear_greed_value_numeric'].shift(-1))
    ).astype(int)
    
    result_df['fear_greed_is_local_max'] = (
        (result_df['fear_greed_value_numeric'] > result_df['fear_greed_value_numeric'].shift(1)) &
        (result_df['fear_greed_value_numeric'] > result_df['fear_greed_value_numeric'].shift(-1))
    ).astype(int)
    
    # Chỉ số trạng thái tâm lý cực đoan
    result_df['fear_greed_extreme_level'] = 0
    result_df.loc[result_df['fear_greed_value_numeric'] <= 25, 'fear_greed_extreme_level'] = -2  # Extreme Fear
    result_df.loc[(result_df['fear_greed_value_numeric'] > 25) & (result_df['fear_greed_value_numeric'] <= 40), 'fear_greed_extreme_level'] = -1  # Fear
    result_df.loc[(result_df['fear_greed_value_numeric'] >= 60) & (result_df['fear_greed_value_numeric'] < 75), 'fear_greed_extreme_level'] = 1  # Greed
    result_df.loc[result_df['fear_greed_value_numeric'] >= 75, 'fear_greed_extreme_level'] = 2  # Extreme Greed
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df


def extract_on_chain_sentiment_features(onchain_df: pd.DataFrame, price_df: pd.DataFrame,
                                      window_sizes: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Tạo đặc trưng từ dữ liệu on-chain sentiment.
    
    Args:
        onchain_df: DataFrame dữ liệu on-chain (từ GlassNode)
        price_df: DataFrame dữ liệu giá để căn chỉnh timestamps
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng on-chain sentiment
    """
    # Tạo DataFrame kết quả với cùng index và timestamp như price_df
    result_df = pd.DataFrame(index=price_df.index)
    result_df['timestamp'] = price_df['timestamp']
    
    # Đảm bảo timestamp là datetime
    onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Đặt timestamp làm index cho onchain_df
    onchain_df = onchain_df.sort_values('timestamp')
    
    # Lấy danh sách các chỉ số on-chain (từ metadata nếu có)
    onchain_metrics = set()
    if 'metadata_metric' in onchain_df.columns:
        onchain_metrics = set(onchain_df['metadata_metric'].dropna().unique())
    
    # Nếu không có cột metadata_metric, sử dụng tên nguồn
    if not onchain_metrics and 'source' in onchain_df.columns:
        for source in onchain_df['source'].unique():
            if 'GlassNode' in source:
                metric_name = source.replace('GlassNode - ', '').lower().replace(' ', '_')
                onchain_metrics.add(metric_name)
    
    # Nếu vẫn không có chỉ số, sử dụng giá trị chung
    onchain_metrics = list(onchain_metrics) if onchain_metrics else ['onchain_sentiment']
    
    # Với mỗi timestamp trong price_df, tìm dữ liệu on-chain gần nhất
    for idx, row in result_df.iterrows():
        timestamp = row['timestamp']
        
        # Lấy các bản ghi gần nhất cho từng chỉ số
        for metric in onchain_metrics:
            if 'metadata_metric' in onchain_df.columns:
                metric_data = onchain_df[onchain_df['metadata_metric'] == metric]
            else:
                metric_data = onchain_df
            
            prev_records = metric_data[metric_data['timestamp'] <= timestamp]
            
            if not prev_records.empty:
                latest_record = prev_records.iloc[-1]
                
                # Lưu giá trị chỉ số
                metric_col = f"onchain_{metric}_value"
                result_df.loc[idx, metric_col] = latest_record['value']
                
                # Lưu nhãn chỉ số
                metric_label_col = f"onchain_{metric}_label"
                result_df.loc[idx, metric_label_col] = latest_record['label']
                
            else:
                # Nếu không có dữ liệu trước timestamp này, đặt giá trị mặc định
                metric_col = f"onchain_{metric}_value"
                result_df.loc[idx, metric_col] = 0
                
                metric_label_col = f"onchain_{metric}_label"
                result_df.loc[idx, metric_label_col] = "Unknown"
    
    # Tạo đặc trưng từ các chỉ số on-chain
    for metric in onchain_metrics:
        metric_col = f"onchain_{metric}_value"
        
        # Tính toán các chỉ số trung bình cho các khoảng thời gian khác nhau
        for window in window_sizes:
            # Giá trị trung bình cho cửa sổ
            result_df[f"onchain_{metric}_avg_{window}d"] = result_df[metric_col].rolling(window).mean()
            
            # Độ lệch chuẩn của giá trị
            result_df[f"onchain_{metric}_std_{window}d"] = result_df[metric_col].rolling(window).std()
            
            # Xu hướng giá trị (độ dốc của đường hồi quy tuyến tính)
            result_df[f"onchain_{metric}_trend_{window}d"] = (
                result_df[metric_col].rolling(window).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
                )
            )
        
        # Tạo chỉ số xung lượng on-chain
        for window in [1, 3, 7]:
            result_df[f"onchain_{metric}_momentum_{window}d"] = result_df[metric_col] - result_df[metric_col].shift(window)
        
        # Chỉ số z-score
        result_df[f"onchain_{metric}_zscore"] = (
            (result_df[metric_col] - result_df[f"onchain_{metric}_avg_{max(window_sizes)}d"]) / 
            result_df[f"onchain_{metric}_std_{max(window_sizes)}d"].replace(0, 1)
        )
    
    # Tính tâm lý tổng hợp on-chain nếu có nhiều chỉ số
    if len(onchain_metrics) > 1:
        # Chuẩn hóa các chỉ số
        normalized_metrics = {}
        for metric in onchain_metrics:
            metric_col = f"onchain_{metric}_value"
            if metric_col in result_df.columns:
                series = result_df[metric_col]
                min_val = series.min()
                max_val = series.max()
                
                if max_val > min_val:
                    normalized_metrics[metric] = (series - min_val) / (max_val - min_val)
                else:
                    normalized_metrics[metric] = series
        
        # Tính trung bình của các chỉ số chuẩn hóa
        if normalized_metrics:
            result_df['onchain_sentiment_composite'] = pd.DataFrame(normalized_metrics).mean(axis=1)
            
            # Tính các chỉ số trung bình cho tâm lý tổng hợp
            for window in window_sizes:
                result_df[f'onchain_sentiment_composite_avg_{window}d'] = result_df['onchain_sentiment_composite'].rolling(window).mean()
                result_df[f'onchain_sentiment_composite_trend_{window}d'] = (
                    result_df['onchain_sentiment_composite'].rolling(window).apply(
                        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else np.nan
                    )
                )
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df


def extract_sentiment_divergence_features(df: pd.DataFrame, window_sizes: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Tính toán đặc trưng phân kỳ giữa tâm lý và giá.
    
    Args:
        df: DataFrame chứa cả dữ liệu tâm lý và giá
        window_sizes: Các kích thước cửa sổ cho tính toán
        
    Returns:
        DataFrame chứa đặc trưng phân kỳ tâm lý
    """
    # Tạo DataFrame kết quả với cùng index như df
    result_df = pd.DataFrame(index=df.index)
    
    # Kiểm tra các cột cần thiết
    required_columns = ['close']
    sentiment_columns = []
    
    if 'fear_greed_value_numeric' in df.columns:
        sentiment_columns.append('fear_greed_value_numeric')
    
    for col in df.columns:
        if 'onchain_' in col and '_value' in col and not ('_avg_' in col or '_std_' in col or '_trend_' in col):
            sentiment_columns.append(col)
    
    if not sentiment_columns:
        logger.warning("Không tìm thấy cột tâm lý để tính toán phân kỳ")
        return result_df
    
    if 'close' not in df.columns:
        logger.warning("Không tìm thấy cột giá 'close' để tính toán phân kỳ")
        return result_df
    
    # Tính phân kỳ cho mỗi chỉ số tâm lý
    for sentiment_col in sentiment_columns:
        # Trích xuất tên ngắn gọn cho chỉ số tâm lý
        if 'fear_greed' in sentiment_col:
            sentiment_name = 'fear_greed'
        elif 'onchain_' in sentiment_col:
            parts = sentiment_col.split('_')
            if len(parts) >= 3:
                sentiment_name = f"{parts[0]}_{parts[1]}"
            else:
                sentiment_name = parts[0]
        else:
            sentiment_name = sentiment_col.replace('_value', '')
            
        # Chuẩn hóa cả giá và tâm lý để so sánh
        for window in window_sizes:
            # Chuẩn hóa giá trong cửa sổ
            price_min = df['close'].rolling(window).min()
            price_max = df['close'].rolling(window).max()
            price_range = price_max - price_min
            
            # Tránh chia cho 0
            price_range = price_range.replace(0, 1)
            
            normalized_price = (df['close'] - price_min) / price_range
            
            # Chuẩn hóa tâm lý trong cửa sổ
            sentiment_min = df[sentiment_col].rolling(window).min()
            sentiment_max = df[sentiment_col].rolling(window).max()
            sentiment_range = sentiment_max - sentiment_min
            
            # Tránh chia cho 0
            sentiment_range = sentiment_range.replace(0, 1)
            
            normalized_sentiment = (df[sentiment_col] - sentiment_min) / sentiment_range
            
            # Tính phân kỳ (tâm lý - giá)
            result_df[f'sentiment_divergence_{sentiment_name}_{window}d'] = normalized_sentiment - normalized_price
            
            # Tính xung lượng phân kỳ
            result_df[f'sentiment_divergence_{sentiment_name}_momentum_{window}d'] = (
                result_df[f'sentiment_divergence_{sentiment_name}_{window}d'] - 
                result_df[f'sentiment_divergence_{sentiment_name}_{window}d'].shift(1)
            )
            
            # Phát hiện đảo chiều phân kỳ
            result_df[f'sentiment_divergence_{sentiment_name}_reversal_{window}d'] = (
                (result_df[f'sentiment_divergence_{sentiment_name}_momentum_{window}d'] > 0) & 
                (result_df[f'sentiment_divergence_{sentiment_name}_momentum_{window}d'].shift(1) < 0)
            ).astype(int)
    
    # Tính phân kỳ tổng hợp nếu có nhiều chỉ số
    if len(sentiment_columns) > 1:
        # Lấy tất cả các cột phân kỳ đã tính
        divergence_cols = [col for col in result_df.columns if 'sentiment_divergence_' in col and not ('momentum' in col or 'reversal' in col)]
        
        # Nhóm theo kích thước cửa sổ
        window_groups = {}
        for col in divergence_cols:
            # Trích xuất kích thước cửa sổ
            parts = col.split('_')
            window = parts[-1]  # Lấy "Xd" như "7d"
            
            if window not in window_groups:
                window_groups[window] = []
            
            window_groups[window].append(col)
        
        # Tính phân kỳ tổng hợp cho mỗi kích thước cửa sổ
        for window, cols in window_groups.items():
            if len(cols) > 1:
                result_df[f'sentiment_divergence_composite_{window}'] = result_df[cols].mean(axis=1)
                
                # Tính xung lượng phân kỳ tổng hợp
                result_df[f'sentiment_divergence_composite_momentum_{window}'] = (
                    result_df[f'sentiment_divergence_composite_{window}'] - 
                    result_df[f'sentiment_divergence_composite_{window}'].shift(1)
                )
    
    # Điền các giá trị NaN với phương pháp forward fill và backward fill
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    return result_df