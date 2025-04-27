"""
Đặc trưng thị trường cho dữ liệu giao dịch.
File này cung cấp các lớp và phương thức để tính toán đặc trưng thị trường
từ dữ liệu giao dịch, bao gồm các đặc trưng giá, khối lượng, và biến động.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from datetime import datetime, timedelta

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from data_processors.feature_engineering.technical_indicators import TechnicalIndicators

class MarketFeatures:
    """
    Lớp chính để tạo đặc trưng thị trường.
    """
    
    def __init__(
        self,
        use_technical_indicators: bool = True,
        use_advanced_features: bool = True,
        custom_features: Optional[List[Callable]] = None
    ):
        """
        Khởi tạo bộ tạo đặc trưng thị trường.
        
        Args:
            use_technical_indicators: Sử dụng chỉ báo kỹ thuật
            use_advanced_features: Sử dụng đặc trưng nâng cao
            custom_features: Danh sách các hàm tạo đặc trưng tùy chỉnh
        """
        self.logger = setup_logger("market_features")
        
        self.use_technical_indicators = use_technical_indicators
        self.use_advanced_features = use_advanced_features
        self.custom_features = custom_features or []
        
        # Khởi tạo technical indicators nếu cần
        if self.use_technical_indicators:
            self.tech_indicators = TechnicalIndicators()
        
        self.logger.info(f"Đã khởi tạo MarketFeatures với use_technical_indicators={self.use_technical_indicators}, use_advanced_features={self.use_advanced_features}")
    
    def add_all_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str] = None,
        timeframes: List[int] = None,
        include_price_features: bool = True,
        include_volume_features: bool = True,
        include_volatility_features: bool = True,
        include_pattern_features: bool = True,
        include_time_features: bool = True
    ) -> pd.DataFrame:
        """
        Thêm tất cả các đặc trưng thị trường vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV ('open', 'high', 'low', 'close', 'volume')
                         với tên cột trong DataFrame. Mặc định là các tên cột chữ thường.
            timeframes: Danh sách các khung thời gian (số nến) để tính toán đặc trưng
                      (mặc định: [5, 10, 20, 50, 100])
            include_price_features: Thêm đặc trưng liên quan đến giá
            include_volume_features: Thêm đặc trưng liên quan đến khối lượng
            include_volatility_features: Thêm đặc trưng liên quan đến biến động
            include_pattern_features: Thêm đặc trưng liên quan đến mẫu hình
            include_time_features: Thêm đặc trưng liên quan đến thời gian
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để thêm đặc trưng")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Chuẩn hóa tên cột OHLCV
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'timestamp': 'timestamp'
            }
        
        # Kiểm tra các cột cần thiết
        missing_columns = [col for col, mapped_col in ohlcv_columns.items() 
                          if mapped_col not in result_df.columns 
                          and col not in ['volume', 'timestamp']]  # Volume và timestamp có thể tùy chọn
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột dữ liệu cần thiết: {missing_columns}")
            return df
        
        # Thiết lập timeframes mặc định nếu không được chỉ định
        if timeframes is None:
            timeframes = [5, 10, 20, 50, 100]
        
        # Đảm bảo thứ tự tăng dần của dữ liệu theo timestamp nếu có
        if 'timestamp' in ohlcv_columns and ohlcv_columns['timestamp'] in result_df.columns:
            result_df = result_df.sort_values(by=ohlcv_columns['timestamp'])
        
        # Thêm các đặc trưng theo yêu cầu
        if include_price_features:
            result_df = self.add_price_features(result_df, ohlcv_columns, timeframes)
        
        if include_volume_features and 'volume' in ohlcv_columns and ohlcv_columns['volume'] in result_df.columns:
            result_df = self.add_volume_features(result_df, ohlcv_columns, timeframes)
        
        if include_volatility_features:
            result_df = self.add_volatility_features(result_df, ohlcv_columns, timeframes)
        
        if include_pattern_features:
            result_df = self.add_pattern_features(result_df, ohlcv_columns)
        
        if include_time_features and 'timestamp' in ohlcv_columns and ohlcv_columns['timestamp'] in result_df.columns:
            result_df = self.add_time_features(result_df, ohlcv_columns['timestamp'])
        
        # Thêm các chỉ báo kỹ thuật phổ biến
        if self.use_technical_indicators:
            result_df = self.add_technical_indicators(result_df, ohlcv_columns)
        
        # Thêm các đặc trưng nâng cao
        if self.use_advanced_features:
            result_df = self.add_advanced_features(result_df, ohlcv_columns, timeframes)
        
        # Thêm các đặc trưng tùy chỉnh
        if self.custom_features:
            for feature_func in self.custom_features:
                try:
                    result_df = feature_func(result_df, ohlcv_columns)
                except Exception as e:
                    self.logger.error(f"Lỗi khi thêm đặc trưng tùy chỉnh {feature_func.__name__}: {e}")
        
        # Log số lượng đặc trưng đã thêm
        num_features_added = len(result_df.columns) - len(df.columns)
        self.logger.info(f"Đã thêm {num_features_added} đặc trưng vào DataFrame")
        
        return result_df
    
    def add_price_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str],
        timeframes: List[int]
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến giá vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            timeframes: Danh sách các khung thời gian để tính toán
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        self.logger.debug("Đang thêm đặc trưng liên quan đến giá")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Lấy các cột dữ liệu giá
        open_col = ohlcv_columns['open']
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        close_col = ohlcv_columns['close']
        
        # 1. Tính lợi nhuận (returns)
        # Lợi nhuận phần trăm
        result_df['return_1'] = result_df[close_col].pct_change(1)
        
        for tf in timeframes:
            if tf > 1:  # Đã tính cho tf=1
                result_df[f'return_{tf}'] = result_df[close_col].pct_change(tf)
        
        # 2. Tính log returns
        result_df['log_return_1'] = np.log(result_df[close_col] / result_df[close_col].shift(1))
        
        for tf in timeframes:
            if tf > 1:  # Đã tính cho tf=1
                result_df[f'log_return_{tf}'] = np.log(result_df[close_col] / result_df[close_col].shift(tf))
        
        # 3. Tính khoảng cách giá so với giá trung bình
        for tf in timeframes:
            # SMA
            result_df[f'price_sma_{tf}'] = result_df[close_col].rolling(window=tf).mean()
            result_df[f'price_sma_{tf}_ratio'] = result_df[close_col] / result_df[f'price_sma_{tf}']
            result_df[f'price_sma_{tf}_diff'] = result_df[close_col] - result_df[f'price_sma_{tf}']
            
            # EMA
            result_df[f'price_ema_{tf}'] = result_df[close_col].ewm(span=tf, adjust=False).mean()
            result_df[f'price_ema_{tf}_ratio'] = result_df[close_col] / result_df[f'price_ema_{tf}']
            result_df[f'price_ema_{tf}_diff'] = result_df[close_col] - result_df[f'price_ema_{tf}']
        
        # 4. Tính các đặc trưng từ OHLC
        # Candle body (thân nến)
        result_df['candle_body'] = abs(result_df[close_col] - result_df[open_col])
        result_df['candle_body_ratio'] = result_df['candle_body'] / (result_df[high_col] - result_df[low_col] + 1e-10)
        
        # Upper shadow (bóng trên)
        result_df['upper_shadow'] = result_df[high_col] - result_df[[open_col, close_col]].max(axis=1)
        result_df['upper_shadow_ratio'] = result_df['upper_shadow'] / (result_df[high_col] - result_df[low_col] + 1e-10)
        
        # Lower shadow (bóng dưới)
        result_df['lower_shadow'] = result_df[[open_col, close_col]].min(axis=1) - result_df[low_col]
        result_df['lower_shadow_ratio'] = result_df['lower_shadow'] / (result_df[high_col] - result_df[low_col] + 1e-10)
        
        # Candle direction (hướng nến: 1 tăng, -1 giảm, 0 đứng yên)
        result_df['candle_direction'] = np.sign(result_df[close_col] - result_df[open_col])
        
        # 5. Tính các mức hỗ trợ/kháng cự
        for tf in timeframes:
            # Mức cao nhất
            result_df[f'highest_high_{tf}'] = result_df[high_col].rolling(window=tf).max()
            result_df[f'highest_high_{tf}_ratio'] = result_df[high_col] / result_df[f'highest_high_{tf}']
            
            # Mức thấp nhất
            result_df[f'lowest_low_{tf}'] = result_df[low_col].rolling(window=tf).min()
            result_df[f'lowest_low_{tf}_ratio'] = result_df[low_col] / result_df[f'lowest_low_{tf}']
            
            # Vị trí giá hiện tại trong phạm vi
            result_df[f'price_position_{tf}'] = (result_df[close_col] - result_df[f'lowest_low_{tf}']) / (
                result_df[f'highest_high_{tf}'] - result_df[f'lowest_low_{tf}'] + 1e-10)
        
        # 6. Tính Pivot Points (điểm xoay)
        result_df['pivot'] = (result_df[high_col] + result_df[low_col] + result_df[close_col]) / 3
        result_df['pivot_r1'] = 2 * result_df['pivot'] - result_df[low_col]
        result_df['pivot_s1'] = 2 * result_df['pivot'] - result_df[high_col]
        result_df['pivot_r2'] = result_df['pivot'] + (result_df[high_col] - result_df[low_col])
        result_df['pivot_s2'] = result_df['pivot'] - (result_df[high_col] - result_df[low_col])
        
        return result_df
    
    def add_volume_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str],
        timeframes: List[int]
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến khối lượng vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            timeframes: Danh sách các khung thời gian để tính toán
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        self.logger.debug("Đang thêm đặc trưng liên quan đến khối lượng")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Lấy các cột dữ liệu
        close_col = ohlcv_columns['close']
        volume_col = ohlcv_columns['volume']
        
        # Kiểm tra xem cột volume có tồn tại không
        if volume_col not in result_df.columns:
            self.logger.warning(f"Không tìm thấy cột khối lượng {volume_col} trong DataFrame")
            return result_df
        
        # 1. Tính khối lượng tương đối so với trung bình
        for tf in timeframes:
            # SMA của volume
            result_df[f'volume_sma_{tf}'] = result_df[volume_col].rolling(window=tf).mean()
            result_df[f'volume_ratio_{tf}'] = result_df[volume_col] / result_df[f'volume_sma_{tf}']
            
            # EMA của volume
            result_df[f'volume_ema_{tf}'] = result_df[volume_col].ewm(span=tf, adjust=False).mean()
            result_df[f'volume_ema_ratio_{tf}'] = result_df[volume_col] / result_df[f'volume_ema_{tf}']
        
        # 2. Tính OBV (On-Balance Volume)
        result_df['obv'] = 0
        result_df.loc[1:, 'obv'] = np.where(
            result_df[close_col] > result_df[close_col].shift(1),
            result_df[volume_col],
            np.where(
                result_df[close_col] < result_df[close_col].shift(1),
                -result_df[volume_col],
                0
            )
        ).cumsum()
        
        # 3. Tính VWAP (Volume Weighted Average Price)
        result_df['vwap_daily'] = (result_df[close_col] * result_df[volume_col]).cumsum() / result_df[volume_col].cumsum()
        
        # 4. Tính Money Flow
        typical_price = (result_df[ohlcv_columns['high']] + result_df[ohlcv_columns['low']] + result_df[close_col]) / 3
        result_df['money_flow'] = typical_price * result_df[volume_col]
        
        # 5. Tính khối lượng theo hướng (volume up/down)
        result_df['volume_up'] = np.where(
            result_df[close_col] > result_df[close_col].shift(1),
            result_df[volume_col],
            0
        )
        
        result_df['volume_down'] = np.where(
            result_df[close_col] < result_df[close_col].shift(1),
            result_df[volume_col],
            0
        )
        
        # 6. Tính tỷ lệ khối lượng lên/xuống
        for tf in timeframes:
            result_df[f'volume_up_sum_{tf}'] = result_df['volume_up'].rolling(window=tf).sum()
            result_df[f'volume_down_sum_{tf}'] = result_df['volume_down'].rolling(window=tf).sum()
            result_df[f'volume_ratio_updown_{tf}'] = (
                result_df[f'volume_up_sum_{tf}'] / (result_df[f'volume_down_sum_{tf}'] + 1e-10)
            )
        
        # 7. Tính Chaikin Money Flow (CMF)
        for tf in timeframes:
            money_flow_multiplier = ((result_df[close_col] - result_df[ohlcv_columns['low']]) - 
                                     (result_df[ohlcv_columns['high']] - result_df[close_col])) / (
                                     result_df[ohlcv_columns['high']] - result_df[ohlcv_columns['low']] + 1e-10)
            money_flow_volume = money_flow_multiplier * result_df[volume_col]
            result_df[f'cmf_{tf}'] = money_flow_volume.rolling(window=tf).sum() / result_df[volume_col].rolling(window=tf).sum()
        
        # 8. Tính Price-Volume Trend (PVT)
        result_df['pvt'] = (result_df[close_col].pct_change() * result_df[volume_col]).cumsum()
        
        # 9. Tính Volume Oscillator
        for short_tf, long_tf in zip(timeframes[:-1], timeframes[1:]):
            result_df[f'volume_oscillator_{short_tf}_{long_tf}'] = (
                result_df[f'volume_ema_{short_tf}'] - result_df[f'volume_ema_{long_tf}']
            ) / result_df[f'volume_ema_{long_tf}'] * 100
        
        return result_df
    
    def add_volatility_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str],
        timeframes: List[int]
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến biến động vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            timeframes: Danh sách các khung thời gian để tính toán
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        self.logger.debug("Đang thêm đặc trưng liên quan đến biến động")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Lấy các cột dữ liệu
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        close_col = ohlcv_columns['close']
        
        # 1. Tính True Range (TR)
        result_df['tr'] = np.maximum(
            result_df[high_col] - result_df[low_col],
            np.maximum(
                abs(result_df[high_col] - result_df[close_col].shift(1)),
                abs(result_df[low_col] - result_df[close_col].shift(1))
            )
        )
        
        # 2. Tính Average True Range (ATR)
        for tf in timeframes:
            result_df[f'atr_{tf}'] = result_df['tr'].rolling(window=tf).mean()
            result_df[f'atr_ratio_{tf}'] = result_df['tr'] / result_df[f'atr_{tf}']
            
            # ATR percent (ATR/Close)
            result_df[f'atr_percent_{tf}'] = result_df[f'atr_{tf}'] / result_df[close_col] * 100
        
        # 3. Tính Bollinger Bands
        for tf in timeframes:
            # SMA cho Bollinger Bands
            result_df[f'bb_middle_{tf}'] = result_df[close_col].rolling(window=tf).mean()
            
            # Độ lệch chuẩn
            result_df[f'bb_std_{tf}'] = result_df[close_col].rolling(window=tf).std()
            
            # Bollinger Bands trên và dưới (2 độ lệch chuẩn)
            result_df[f'bb_upper_{tf}'] = result_df[f'bb_middle_{tf}'] + 2 * result_df[f'bb_std_{tf}']
            result_df[f'bb_lower_{tf}'] = result_df[f'bb_middle_{tf}'] - 2 * result_df[f'bb_std_{tf}']
            
            # Bollinger Bandwidth
            result_df[f'bb_width_{tf}'] = (result_df[f'bb_upper_{tf}'] - result_df[f'bb_lower_{tf}']) / result_df[f'bb_middle_{tf}']
            
            # Bollinger %B
            result_df[f'bb_percent_b_{tf}'] = (result_df[close_col] - result_df[f'bb_lower_{tf}']) / (
                result_df[f'bb_upper_{tf}'] - result_df[f'bb_lower_{tf}'] + 1e-10)
        
        # 4. Tính Historical Volatility
        for tf in timeframes:
            # Log returns
            log_returns = np.log(result_df[close_col] / result_df[close_col].shift(1))
            
            # Volatility (annualized standard deviation)
            result_df[f'volatility_{tf}'] = log_returns.rolling(window=tf).std() * np.sqrt(252)
        
        # 5. Tính Relative Volatility
        for short_tf, long_tf in zip(timeframes[:-1], timeframes[1:]):
            result_df[f'relative_volatility_{short_tf}_{long_tf}'] = (
                result_df[f'volatility_{short_tf}'] / result_df[f'volatility_{long_tf}']
            )
        
        # 6. Tính Garman-Klass Volatility
        for tf in timeframes:
            log_high_low = np.log(result_df[high_col] / result_df[low_col]) ** 2
            log_close_open = np.log(result_df[close_col] / result_df[ohlcv_columns['open']]) ** 2
            
            gk_vol = 0.5 * log_high_low - (2 * np.log(2) - 1) * log_close_open
            result_df[f'gk_volatility_{tf}'] = np.sqrt(gk_vol.rolling(window=tf).mean() * 252)
        
        # 7. Tính Beta
        # Cần có dữ liệu thị trường cho chỉ số này, nếu không có có thể tính beta so với một cặp giao dịch khác
        
        # 8. Tính Keltner Channels
        for tf in timeframes:
            result_df[f'keltner_middle_{tf}'] = result_df[close_col].ewm(span=tf, adjust=False).mean()
            result_df[f'keltner_upper_{tf}'] = result_df[f'keltner_middle_{tf}'] + 2 * result_df[f'atr_{tf}']
            result_df[f'keltner_lower_{tf}'] = result_df[f'keltner_middle_{tf}'] - 2 * result_df[f'atr_{tf}']
            
            # Keltner Channel Width
            result_df[f'keltner_width_{tf}'] = (
                result_df[f'keltner_upper_{tf}'] - result_df[f'keltner_lower_{tf}']
            ) / result_df[f'keltner_middle_{tf}']
        
        # 9. Tính Squeeze Momentum
        for tf in timeframes:
            # Squeeze: Bollinger Bands bên trong Keltner Channels
            result_df[f'squeeze_{tf}'] = (result_df[f'bb_lower_{tf}'] > result_df[f'keltner_lower_{tf}']) & \
                                         (result_df[f'bb_upper_{tf}'] < result_df[f'keltner_upper_{tf}'])
            result_df[f'squeeze_{tf}'] = result_df[f'squeeze_{tf}'].astype(int)
        
        return result_df
    
    def add_pattern_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến mẫu hình vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        self.logger.debug("Đang thêm đặc trưng liên quan đến mẫu hình")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Lấy các cột dữ liệu
        open_col = ohlcv_columns['open']
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        close_col = ohlcv_columns['close']
        
        # 1. Xác định các mẫu nến đơn
        # Doji (thân nến rất nhỏ)
        doji_threshold = 0.1  # Thân nến < 10% biên độ
        result_df['doji'] = (
            abs(result_df[close_col] - result_df[open_col]) / 
            (result_df[high_col] - result_df[low_col] + 1e-10) < doji_threshold
        ).astype(int)
        
        # Hammer (nến có bóng dưới dài, thân ngắn, và gần như không có bóng trên)
        result_df['hammer'] = (
            # Thân nến nhỏ: < 30% biên độ
            (abs(result_df[close_col] - result_df[open_col]) / (result_df[high_col] - result_df[low_col] + 1e-10) < 0.3) &
            # Bóng dưới dài: > 2 lần thân nến
            (result_df[[open_col, close_col]].min(axis=1) - result_df[low_col] > 
             2 * abs(result_df[close_col] - result_df[open_col])) &
            # Gần như không có bóng trên: < 10% thân nến
            (result_df[high_col] - result_df[[open_col, close_col]].max(axis=1) < 
             0.1 * abs(result_df[close_col] - result_df[open_col]))
        ).astype(int)
        
        # Inverted Hammer (nến có bóng trên dài, thân ngắn, và gần như không có bóng dưới)
        result_df['inverted_hammer'] = (
            # Thân nến nhỏ: < 30% biên độ
            (abs(result_df[close_col] - result_df[open_col]) / (result_df[high_col] - result_df[low_col] + 1e-10) < 0.3) &
            # Bóng trên dài: > 2 lần thân nến
            (result_df[high_col] - result_df[[open_col, close_col]].max(axis=1) > 
             2 * abs(result_df[close_col] - result_df[open_col])) &
            # Gần như không có bóng dưới: < 10% thân nến
            (result_df[[open_col, close_col]].min(axis=1) - result_df[low_col] < 
             0.1 * abs(result_df[close_col] - result_df[open_col]))
        ).astype(int)
        
        # Bullish Engulfing (nến tăng bao trùm hoàn toàn nến trước đó)
        result_df['bullish_engulfing'] = (
            (result_df[close_col] > result_df[open_col]) &  # Nến hiện tại tăng
            (result_df[close_col].shift(1) < result_df[open_col].shift(1)) &  # Nến trước giảm
            (result_df[close_col] > result_df[open_col].shift(1)) &  # Close hiện tại > Open trước
            (result_df[open_col] < result_df[close_col].shift(1))  # Open hiện tại < Close trước
        ).astype(int)
        
        # Bearish Engulfing (nến giảm bao trùm hoàn toàn nến trước đó)
        result_df['bearish_engulfing'] = (
            (result_df[close_col] < result_df[open_col]) &  # Nến hiện tại giảm
            (result_df[close_col].shift(1) > result_df[open_col].shift(1)) &  # Nến trước tăng
            (result_df[close_col] < result_df[open_col].shift(1)) &  # Close hiện tại < Open trước
            (result_df[open_col] > result_df[close_col].shift(1))  # Open hiện tại > Close trước
        ).astype(int)
        
        # 2. Xác định các mẫu nến phức tạp
        # Morning Star (3 nến: giảm, doji, tăng)
        result_df['morning_star'] = (
            (result_df[close_col].shift(2) < result_df[open_col].shift(2)) &  # Nến 1 giảm
            (abs(result_df[close_col].shift(1) - result_df[open_col].shift(1)) / 
             (result_df[high_col].shift(1) - result_df[low_col].shift(1) + 1e-10) < 0.1) &  # Nến 2 là doji
            (result_df[close_col] > result_df[open_col]) &  # Nến 3 tăng
            (result_df[close_col] > (result_df[open_col].shift(2) + result_df[close_col].shift(2)) / 2)  # Nến 3 phục hồi > 50% nến 1
        ).astype(int)
        
        # Evening Star (3 nến: tăng, doji, giảm)
        result_df['evening_star'] = (
            (result_df[close_col].shift(2) > result_df[open_col].shift(2)) &  # Nến 1 tăng
            (abs(result_df[close_col].shift(1) - result_df[open_col].shift(1)) / 
             (result_df[high_col].shift(1) - result_df[low_col].shift(1) + 1e-10) < 0.1) &  # Nến 2 là doji
            (result_df[close_col] < result_df[open_col]) &  # Nến 3 giảm
            (result_df[close_col] < (result_df[open_col].shift(2) + result_df[close_col].shift(2)) / 2)  # Nến 3 giảm > 50% nến 1
        ).astype(int)
        
        # Three White Soldiers (3 nến tăng liên tiếp, mỗi nến mở cửa trong thân nến trước và đóng cửa cao hơn)
        result_df['three_white_soldiers'] = (
            (result_df[close_col] > result_df[open_col]) &  # Nến hiện tại tăng
            (result_df[close_col].shift(1) > result_df[open_col].shift(1)) &  # Nến -1 tăng
            (result_df[close_col].shift(2) > result_df[open_col].shift(2)) &  # Nến -2 tăng
            (result_df[open_col] > result_df[open_col].shift(1)) &  # Open hiện tại > Open -1
            (result_df[open_col].shift(1) > result_df[open_col].shift(2)) &  # Open -1 > Open -2
            (result_df[close_col] > result_df[close_col].shift(1)) &  # Close hiện tại > Close -1
            (result_df[close_col].shift(1) > result_df[close_col].shift(2))  # Close -1 > Close -2
        ).astype(int)
        
        # Three Black Crows (3 nến giảm liên tiếp, mỗi nến mở cửa trong thân nến trước và đóng cửa thấp hơn)
        result_df['three_black_crows'] = (
            (result_df[close_col] < result_df[open_col]) &  # Nến hiện tại giảm
            (result_df[close_col].shift(1) < result_df[open_col].shift(1)) &  # Nến -1 giảm
            (result_df[close_col].shift(2) < result_df[open_col].shift(2)) &  # Nến -2 giảm
            (result_df[open_col] < result_df[open_col].shift(1)) &  # Open hiện tại < Open -1
            (result_df[open_col].shift(1) < result_df[open_col].shift(2)) &  # Open -1 < Open -2
            (result_df[close_col] < result_df[close_col].shift(1)) &  # Close hiện tại < Close -1
            (result_df[close_col].shift(1) < result_df[close_col].shift(2))  # Close -1 < Close -2
        ).astype(int)
        
        # 3. Xác định các mẫu đảo chiều và tiếp tục xu hướng
        # Breakout (giá phá vỡ mức cao/thấp trong 20 phiên)
        result_df['breakout_up'] = (
            result_df[close_col] > result_df[high_col].rolling(window=20).max().shift(1)
        ).astype(int)
        
        result_df['breakout_down'] = (
            result_df[close_col] < result_df[low_col].rolling(window=20).min().shift(1)
        ).astype(int)
        
        # Double Top (hai đỉnh cùng độ cao, giá giảm xuống dưới mức hỗ trợ)
        # Đây là một mẫu hình phức tạp, chỉ xác định sơ bộ
        result_df['potential_double_top'] = (
            (result_df[high_col] > result_df[high_col].shift(1)) &
            (result_df[high_col] > result_df[high_col].shift(-1)) &
            (result_df[high_col].rolling(window=10).max().shift(-10) - result_df[high_col] < 0.01 * result_df[high_col])
        ).astype(int)
        
        # Double Bottom (hai đáy cùng độ thấp, giá tăng lên trên mức kháng cự)
        result_df['potential_double_bottom'] = (
            (result_df[low_col] < result_df[low_col].shift(1)) &
            (result_df[low_col] < result_df[low_col].shift(-1)) &
            (result_df[low_col] - result_df[low_col].rolling(window=10).min().shift(-10) < 0.01 * result_df[low_col])
        ).astype(int)
        
        # Gap Up (khoảng trống tăng)
        result_df['gap_up'] = (
            result_df[low_col] > result_df[high_col].shift(1)
        ).astype(int)
        
        # Gap Down (khoảng trống giảm)
        result_df['gap_down'] = (
            result_df[high_col] < result_df[low_col].shift(1)
        ).astype(int)
        
        return result_df
    
    def add_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến thời gian vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            timestamp_col: Tên cột thời gian
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        self.logger.debug("Đang thêm đặc trưng liên quan đến thời gian")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Kiểm tra cột timestamp
        if timestamp_col not in result_df.columns:
            self.logger.warning(f"Không tìm thấy cột thời gian {timestamp_col} trong DataFrame")
            return result_df
        
        # Đảm bảo cột timestamp có kiểu datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            try:
                result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
            except Exception as e:
                self.logger.error(f"Không thể chuyển đổi cột {timestamp_col} thành kiểu datetime: {e}")
                return result_df
        
        # 1. Trích xuất các thành phần thời gian
        # Giờ trong ngày
        result_df['hour'] = result_df[timestamp_col].dt.hour
        
        # Ngày trong tuần (0=Monday, 6=Sunday)
        result_df['dayofweek'] = result_df[timestamp_col].dt.dayofweek
        
        # Ngày trong tháng
        result_df['day'] = result_df[timestamp_col].dt.day
        
        # Tuần trong năm
        result_df['week'] = result_df[timestamp_col].dt.isocalendar().week
        
        # Tháng
        result_df['month'] = result_df[timestamp_col].dt.month
        
        # Quý
        result_df['quarter'] = result_df[timestamp_col].dt.quarter
        
        # Năm
        result_df['year'] = result_df[timestamp_col].dt.year
        
        # 2. Đánh dấu các thời điểm đặc biệt
        # Ngày đầu tuần (Monday)
        result_df['is_monday'] = (result_df['dayofweek'] == 0).astype(int)
        
        # Ngày cuối tuần (Friday)
        result_df['is_friday'] = (result_df['dayofweek'] == 4).astype(int)
        
        # Buổi sáng (morning): 9h-12h
        result_df['is_morning'] = ((result_df['hour'] >= 9) & (result_df['hour'] < 12)).astype(int)
        
        # Buổi chiều (afternoon): 12h-16h
        result_df['is_afternoon'] = ((result_df['hour'] >= 12) & (result_df['hour'] < 16)).astype(int)
        
        # Buổi tối (evening): 16h-20h
        result_df['is_evening'] = ((result_df['hour'] >= 16) & (result_df['hour'] < 20)).astype(int)
        
        # Buổi đêm (night): 20h-9h
        result_df['is_night'] = ((result_df['hour'] >= 20) | (result_df['hour'] < 9)).astype(int)
        
        # Ngày đầu tháng (1-5)
        result_df['is_month_start'] = ((result_df['day'] >= 1) & (result_df['day'] <= 5)).astype(int)
        
        # Ngày cuối tháng (24-31)
        result_df['is_month_end'] = (result_df['day'] >= 24).astype(int)
        
        # 3. Biến đổi cyclical (biến thành các thành phần sin và cos để bảo toàn tính chất cyclic)
        # Giờ (0-23 -> sin/cos)
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        
        # Ngày trong tuần (0-6 -> sin/cos)
        result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek'] / 7)
        result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek'] / 7)
        
        # Ngày trong tháng (1-31 -> sin/cos)
        result_df['day_sin'] = np.sin(2 * np.pi * (result_df['day'] - 1) / 31)
        result_df['day_cos'] = np.cos(2 * np.pi * (result_df['day'] - 1) / 31)
        
        # Tháng (1-12 -> sin/cos)
        result_df['month_sin'] = np.sin(2 * np.pi * (result_df['month'] - 1) / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * (result_df['month'] - 1) / 12)
        
        # 4. Tính khoảng cách thời gian
        # Khoảng cách giữa các mốc thời gian (theo giây)
        result_df['time_diff'] = result_df[timestamp_col].diff().dt.total_seconds()
        
        # Khoảng cách với thời điểm trung bình trong ngày (12h)
        mid_day = result_df[timestamp_col].dt.normalize() + pd.Timedelta(hours=12)
        result_df['mid_day_diff'] = (result_df[timestamp_col] - mid_day).dt.total_seconds() / 3600  # theo giờ
        
        return result_df
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Thêm các chỉ báo kỹ thuật vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        if not self.use_technical_indicators:
            return df
        
        self.logger.debug("Đang thêm các chỉ báo kỹ thuật")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Danh sách các chỉ báo kỹ thuật phổ biến cần thêm
        indicators = {
            # RSI - Relative Strength Index
            'rsi_14': {
                'function': 'rsi',
                'params': {'timeperiod': 14},
                'output_names': ['rsi_14']
            },
            # MACD - Moving Average Convergence Divergence
            'macd': {
                'function': 'macd',
                'params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
                'output_names': ['macd', 'macd_signal', 'macd_hist']
            },
            # EMA - Exponential Moving Average
            'ema_9': {
                'function': 'ema',
                'params': {'timeperiod': 9},
                'output_names': ['ema_9']
            },
            'ema_21': {
                'function': 'ema',
                'params': {'timeperiod': 21},
                'output_names': ['ema_21']
            },
            'ema_55': {
                'function': 'ema',
                'params': {'timeperiod': 55},
                'output_names': ['ema_55']
            },
            'ema_200': {
                'function': 'ema',
                'params': {'timeperiod': 200},
                'output_names': ['ema_200']
            },
            # ADX - Average Directional Index
            'adx_14': {
                'function': 'adx',
                'params': {'timeperiod': 14},
                'output_names': ['adx_14']
            },
            # CCI - Commodity Channel Index
            'cci_14': {
                'function': 'cci',
                'params': {'timeperiod': 14},
                'output_names': ['cci_14']
            },
            # ATR - Average True Range
            'atr_14': {
                'function': 'atr',
                'params': {'timeperiod': 14},
                'output_names': ['atr_14']
            },
            # Stochastic
            'stoch': {
                'function': 'stoch',
                'params': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
                'output_names': ['stoch_k', 'stoch_d']
            },
            # Williams %R
            'willr_14': {
                'function': 'willr',
                'params': {'timeperiod': 14},
                'output_names': ['willr_14']
            },
            # OBV - On Balance Volume
            'obv': {
                'function': 'obv',
                'params': {},
                'output_names': ['obv']
            },
            # Bollinger Bands
            'bbands_20': {
                'function': 'bbands',
                'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
                'output_names': ['bbands_upper_20', 'bbands_middle_20', 'bbands_lower_20']
            }
        }
        
        # Thêm các chỉ báo kỹ thuật
        result_df = self.tech_indicators.add_indicators(result_df, indicators, ohlcv_columns)
        
        return result_df
    
    def add_advanced_features(
        self,
        df: pd.DataFrame,
        ohlcv_columns: Dict[str, str],
        timeframes: List[int]
    ) -> pd.DataFrame:
        """
        Thêm các đặc trưng nâng cao vào DataFrame.
        
        Args:
            df: DataFrame cần thêm đặc trưng
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            timeframes: Danh sách các khung thời gian để tính toán
            
        Returns:
            DataFrame với các đặc trưng đã được thêm vào
        """
        if not self.use_advanced_features:
            return df
        
        self.logger.debug("Đang thêm đặc trưng nâng cao")
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Lấy các cột dữ liệu
        close_col = ohlcv_columns['close']
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        volume_col = ohlcv_columns.get('volume')
        
        # 1. Đặc trưng về sự tích tụ/phân phối (Accumulation/Distribution)
        if volume_col and volume_col in result_df.columns:
            # Chaikin Oscillator
            for short_tf, long_tf in zip([3, 5], [10, 20]):
                if f'cmf_{short_tf}' in result_df.columns and f'cmf_{long_tf}' in result_df.columns:
                    result_df[f'chaikin_osc_{short_tf}_{long_tf}'] = result_df[f'cmf_{short_tf}'] - result_df[f'cmf_{long_tf}']
        
        # 2. Đặc trưng về động lượng (Momentum)
        # Tỷ lệ tăng/giảm theo các khung thời gian
        for tf in timeframes:
            # Đếm số nến tăng/giảm trong tf
            result_df[f'up_days_{tf}'] = result_df[close_col].rolling(window=tf).apply(
                lambda x: np.sum(np.diff(x) > 0) / (len(x) - 1) if len(x) > 1 else np.nan
            )
            
            # Hệ số ROC (Rate of Change) cho tf
            result_df[f'roc_{tf}'] = (result_df[close_col] - result_df[close_col].shift(tf)) / result_df[close_col].shift(tf) * 100
            
            # Lũy kế ROC lớn hơn 0 (số lần liên tiếp ROC > 0)
            result_df[f'roc_pos_streak_{tf}'] = result_df[f'roc_{tf}'].gt(0).astype(int)
            result_df[f'roc_pos_streak_{tf}'] = result_df[f'roc_pos_streak_{tf}'].groupby(
                (result_df[f'roc_pos_streak_{tf}'] != result_df[f'roc_pos_streak_{tf}'].shift()).cumsum()
            ).cumsum()
            result_df[f'roc_pos_streak_{tf}'] = result_df[f'roc_pos_streak_{tf}'].where(result_df[f'roc_{tf}'] > 0, 0)
            
            # Lũy kế ROC nhỏ hơn 0 (số lần liên tiếp ROC < 0)
            result_df[f'roc_neg_streak_{tf}'] = result_df[f'roc_{tf}'].lt(0).astype(int)
            result_df[f'roc_neg_streak_{tf}'] = result_df[f'roc_neg_streak_{tf}'].groupby(
                (result_df[f'roc_neg_streak_{tf}'] != result_df[f'roc_neg_streak_{tf}'].shift()).cumsum()
            ).cumsum()
            result_df[f'roc_neg_streak_{tf}'] = result_df[f'roc_neg_streak_{tf}'].where(result_df[f'roc_{tf}'] < 0, 0)
        
        # 3. Đặc trưng về xu hướng (Trend)
        # ADX và DMI
        if 'adx_14' in result_df.columns:
            # Phân loại xu hướng dựa trên ADX
            result_df['trend_strength'] = pd.cut(
                result_df['adx_14'],
                bins=[-float('inf'), 20, 40, 60, float('inf')],
                labels=['weak', 'moderate', 'strong', 'extreme']
            )
            
            # One-hot encoding cho trend_strength
            for trend in ['weak', 'moderate', 'strong', 'extreme']:
                result_df[f'trend_{trend}'] = (result_df['trend_strength'] == trend).astype(int)
        
        # 4. Đặc trưng phức tạp hơn
        # Fractal Dimension Index (FDI) cho timeframe cụ thể
        for tf in timeframes:
            if tf > 2:  # Cần ít nhất 3 điểm
                # Tính Highs và Lows path length
                h = np.array(result_df[high_col].rolling(window=tf).max())
                l = np.array(result_df[low_col].rolling(window=tf).min())
                
                # Chia tf thành N phần
                N = min(tf - 1, 10)  # Giới hạn số phần để tránh tính toán quá nhiều
                
                # Khoảng thời gian cụ thể
                result_df[f'fdi_{tf}'] = np.nan
                
                for i in range(tf - 1, len(result_df)):
                    window_high = result_df[high_col].iloc[i-tf+1:i+1].values
                    window_low = result_df[low_col].iloc[i-tf+1:i+1].values
                    
                    # Tổng khoảng cách euclidean giữa các điểm liên tiếp
                    total_dist = 0
                    for j in range(1, len(window_high)):
                        dist = np.sqrt(1 + (window_high[j] - window_high[j-1])**2 + (window_low[j] - window_low[j-1])**2)
                        total_dist += dist
                    
                    # Khoảng cách endpoints
                    end_dist = np.sqrt(tf**2 + (window_high[-1] - window_high[0])**2 + (window_low[-1] - window_low[0])**2)
                    
                    # Tính FDI
                    if end_dist > 0:
                        result_df.loc[result_df.index[i], f'fdi_{tf}'] = np.log(total_dist / end_dist) / np.log(N)
        
        # 5. Tính Fisher Transform của RSI
        if 'rsi_14' in result_df.columns:
            # Chuẩn hóa RSI về khoảng [-1, 1]
            y = 2 * (result_df['rsi_14'] / 100 - 0.5)
            
            # Fisher Transform: 0.5 * ln((1+y)/(1-y))
            # Tránh giá trị y = ±1 để không bị lỗi
            y = y.clip(-0.999, 0.999)
            result_df['fisher_rsi_14'] = 0.5 * np.log((1 + y) / (1 - y))
        
        # 6. Tính Zigzag Indicator (nhận biết đỉnh đáy quan trọng)
        # Đây là một đặc trưng phức tạp, chỉ xác định một số điểm quan trọng
        threshold = 0.05  # 5% thay đổi
        
        # Xác định các điểm Pivot
        result_df['is_zigzag_pivot'] = 0
        
        # Tìm các đỉnh cục bộ
        result_df['is_local_peak'] = (
            (result_df[high_col] > result_df[high_col].shift(1)) &
            (result_df[high_col] > result_df[high_col].shift(-1))
        )
        
        # Tìm các đáy cục bộ
        result_df['is_local_valley'] = (
            (result_df[low_col] < result_df[low_col].shift(1)) &
            (result_df[low_col] < result_df[low_col].shift(-1))
        )
        
        # 7. Linear Regression Channel
        for tf in timeframes:
            if tf > 2:
                # X values: 0 to tf-1
                x = np.arange(tf)
                
                # Linear regression function
                def calc_linreg(y):
                    if len(y) < tf:
                        return np.array([np.nan, np.nan, np.nan])
                    
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        pred = slope * x + intercept
                        std_dev = np.std(y - pred)
                        return np.array([intercept + slope * (tf - 1), slope, std_dev])
                    except Exception:
                        return np.array([np.nan, np.nan, np.nan])
                
                # Apply linear regression
                lr_results = result_df[close_col].rolling(window=tf).apply(
                    calc_linreg, raw=True
                )
                
                # Extract results
                if isinstance(lr_results, pd.Series) and lr_results.values.ndim > 1:
                    result_df[f'linreg_value_{tf}'] = lr_results.apply(lambda x: x[0])
                    result_df[f'linreg_slope_{tf}'] = lr_results.apply(lambda x: x[1])
                    result_df[f'linreg_std_{tf}'] = lr_results.apply(lambda x: x[2])
                    
                    # Calculate upper and lower channel lines
                    result_df[f'linreg_upper_{tf}'] = result_df[f'linreg_value_{tf}'] + 2 * result_df[f'linreg_std_{tf}']
                    result_df[f'linreg_lower_{tf}'] = result_df[f'linreg_value_{tf}'] - 2 * result_df[f'linreg_std_{tf}']
                    
                    # Position within channel (0 to 1)
                    channel_width = result_df[f'linreg_upper_{tf}'] - result_df[f'linreg_lower_{tf}']
                    result_df[f'linreg_pos_{tf}'] = (result_df[close_col] - result_df[f'linreg_lower_{tf}']) / channel_width
        
        # 8. Market Regime Detection
        # Đơn giản: Xác định chế độ thị trường dựa trên xu hướng (trend) và biến động (volatility)
        if all(col in result_df.columns for col in ['volatility_20', 'ema_9', 'ema_21']):
            # Trend: 1 (up), -1 (down), 0 (sideways)
            result_df['trend_direction'] = np.where(
                result_df['ema_9'] > result_df['ema_21'],
                1,
                np.where(result_df['ema_9'] < result_df['ema_21'], -1, 0)
            )
            
            # Volatility: 1 (high), 0 (low) - so sánh với phân vị 75
            vol_threshold = result_df['volatility_20'].quantile(0.75)
            result_df['is_high_volatility'] = (result_df['volatility_20'] > vol_threshold).astype(int)
            
            # Regime: Kết hợp trend và volatility
            result_df['market_regime'] = np.where(
                result_df['is_high_volatility'] == 1,
                np.where(result_df['trend_direction'] == 1, 'volatile_bullish',
                      np.where(result_df['trend_direction'] == -1, 'volatile_bearish', 'volatile_neutral')),
                np.where(result_df['trend_direction'] == 1, 'calm_bullish',
                      np.where(result_df['trend_direction'] == -1, 'calm_bearish', 'calm_neutral'))
            )
            
            # One-hot encoding cho market_regime
            regimes = ['volatile_bullish', 'volatile_bearish', 'volatile_neutral', 
                     'calm_bullish', 'calm_bearish', 'calm_neutral']
            for regime in regimes:
                result_df[f'regime_{regime}'] = (result_df['market_regime'] == regime).astype(int)
        
        return result_df


def extract_features(
    df: pd.DataFrame,
    ohlcv_columns: Dict[str, str] = None,
    timeframes: List[int] = None,
    use_technical_indicators: bool = True,
    use_advanced_features: bool = True,
    include_price_features: bool = True,
    include_volume_features: bool = True,
    include_volatility_features: bool = True,
    include_pattern_features: bool = True,
    include_time_features: bool = True
) -> pd.DataFrame:
    """
    Hàm tiện ích để trích xuất tất cả các đặc trưng thị trường từ DataFrame.
    
    Args:
        df: DataFrame chứa dữ liệu OHLCV
        ohlcv_columns: Dict ánh xạ tên cột OHLCV ('open', 'high', 'low', 'close', 'volume')
                     với tên cột trong DataFrame.
        timeframes: Danh sách các khung thời gian (số nến) để tính toán đặc trưng
        use_technical_indicators: Sử dụng chỉ báo kỹ thuật
        use_advanced_features: Sử dụng đặc trưng nâng cao
        include_price_features: Thêm đặc trưng liên quan đến giá
        include_volume_features: Thêm đặc trưng liên quan đến khối lượng
        include_volatility_features: Thêm đặc trưng liên quan đến biến động
        include_pattern_features: Thêm đặc trưng liên quan đến mẫu hình
        include_time_features: Thêm đặc trưng liên quan đến thời gian
        
    Returns:
        DataFrame với các đặc trưng đã được thêm vào
    """
    # Khởi tạo MarketFeatures
    mf = MarketFeatures(
        use_technical_indicators=use_technical_indicators,
        use_advanced_features=use_advanced_features
    )
    
    # Thêm tất cả đặc trưng
    df_with_features = mf.add_all_features(
        df=df,
        ohlcv_columns=ohlcv_columns,
        timeframes=timeframes,
        include_price_features=include_price_features,
        include_volume_features=include_volume_features,
        include_volatility_features=include_volatility_features,
        include_pattern_features=include_pattern_features,
        include_time_features=include_time_features
    )
    
    return df_with_features