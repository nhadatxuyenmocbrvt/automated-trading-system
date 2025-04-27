"""
Đặc trưng về giá.
Mô-đun này cung cấp các hàm tạo đặc trưng dựa trên giá cả,
bao gồm lợi nhuận, tỷ lệ giá, xung lượng và các mẫu hình giá.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import setup_logger
from data_processors.feature_engineering.technical_indicators.utils import validate_price_data

# Logger
logger = setup_logger("price_features")

def calculate_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    periods: List[int] = [1, 2, 3, 5, 10, 20],
    percentage: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính lợi nhuận trong nhiều khung thời gian.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        periods: Danh sách các khoảng thời gian (periods)
        percentage: Nếu True, tính lợi nhuận theo phần trăm
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa lợi nhuận theo khoảng thời gian
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    for period in periods:
        try:
            if percentage:
                # Tính lợi nhuận theo phần trăm: (Price_t / Price_t-n - 1) * 100
                result_df[f"{prefix}return_{period}"] = (
                    result_df[price_column].pct_change(period) * 100
                )
            else:
                # Tính lợi nhuận tuyệt đối: Price_t - Price_t-n
                result_df[f"{prefix}return_{period}"] = (
                    result_df[price_column].diff(period)
                )
            
            logger.debug(f"Đã tính lợi nhuận cho khoảng thời gian {period}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính lợi nhuận cho khoảng thời gian {period}: {e}")
    
    return result_df

def calculate_log_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    periods: List[int] = [1, 2, 3, 5, 10, 20],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính log returns trong nhiều khung thời gian.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        periods: Danh sách các khoảng thời gian (periods)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa log returns theo khoảng thời gian
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    for period in periods:
        try:
            # Tính log returns: ln(Price_t / Price_t-n)
            # = ln(Price_t) - ln(Price_t-n)
            result_df[f"{prefix}log_return_{period}"] = (
                np.log(result_df[price_column]) - np.log(result_df[price_column].shift(period))
            )
            
            logger.debug(f"Đã tính log returns cho khoảng thời gian {period}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính log returns cho khoảng thời gian {period}: {e}")
    
    return result_df

def calculate_rsi_features(
    df: pd.DataFrame,
    source_column: str = 'close',
    windows: List[int] = [14, 21, 50],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng dựa trên RSI.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        source_column: Tên cột nguồn sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ tính RSI
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng RSI
    """
    if not validate_price_data(df, [source_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {source_column}")
        return df
    
    result_df = df.copy()
    
    for window in windows:
        try:
            # Tính thay đổi giá trị
            delta = result_df[source_column].diff()
            
            # Tạo chuỗi gains (số dương) và losses (số âm)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            
            # Tính average gain và average loss
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            # Tránh chia cho 0
            avg_loss = avg_loss.replace(0, np.nan)
            
            # Tính Relative Strength (RS)
            rs = avg_gain / avg_loss
            
            # Tính RSI: 100 - (100 / (1 + RS))
            rsi = 100 - (100 / (1 + rs))
            
            # Thêm vào DataFrame kết quả
            result_df[f"{prefix}rsi_{window}"] = rsi
            
            # Tính các đặc trưng bổ sung liên quan đến RSI
            
            # 1. RSI momentum (thay đổi RSI so với n kỳ trước)
            result_df[f"{prefix}rsi_momentum_{window}"] = rsi - rsi.shift(window//2)
            
            # 2. RSI slope (độ dốc của RSI trong n kỳ)
            result_df[f"{prefix}rsi_slope_{window}"] = (rsi - rsi.shift(window//4)) / (window//4)
            
            # 3. RSI chỉ báo vùng mua/bán quá mức
            result_df[f"{prefix}rsi_overbought_{window}"] = (rsi > 70).astype(int)
            result_df[f"{prefix}rsi_oversold_{window}"] = (rsi < 30).astype(int)
            
            # 4. RSI divergence (sai lệch giữa giá và RSI)
            price_direction = np.sign(result_df[source_column] - result_df[source_column].shift(window//2))
            rsi_direction = np.sign(rsi - rsi.shift(window//2))
            result_df[f"{prefix}rsi_divergence_{window}"] = (price_direction != rsi_direction).astype(int)
            
            logger.debug(f"Đã tính đặc trưng RSI cho cửa sổ {window}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng RSI cho cửa sổ {window}: {e}")
    
    return result_df

def calculate_price_momentum(
    df: pd.DataFrame,
    price_column: str = 'close',
    windows: List[int] = [5, 10, 20, 50, 100],
    normalize: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính xung lượng giá (price momentum) và các đặc trưng liên quan.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        normalize: Chuẩn hóa bằng giá hiện tại
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng xung lượng giá
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    for window in windows:
        try:
            # Tính xung lượng giá: Price_t - Price_t-n
            momentum = result_df[price_column] - result_df[price_column].shift(window)
            
            if normalize:
                # Chuẩn hóa bằng giá hiện tại: (Price_t - Price_t-n) / Price_t
                price_non_zero = result_df[price_column].replace(0, np.nan)
                momentum = momentum / price_non_zero
                
                # Chuyển sang phần trăm
                momentum = momentum * 100
                
                # Tên cột kết quả
                result_df[f"{prefix}price_momentum_pct_{window}"] = momentum
            else:
                # Tên cột kết quả
                result_df[f"{prefix}price_momentum_{window}"] = momentum
            
            # Tính các đặc trưng bổ sung liên quan đến xung lượng
            
            # 1. Momentum acceleration (thay đổi xung lượng)
            result_df[f"{prefix}momentum_acceleration_{window}"] = momentum - momentum.shift(window//2)
            
            # 2. Momentum direction (hướng xung lượng: 1 tăng, -1 giảm, 0 không đổi)
            result_df[f"{prefix}momentum_direction_{window}"] = np.sign(momentum)
            
            # 3. Momentum volatility (biến động xung lượng)
            result_df[f"{prefix}momentum_volatility_{window}"] = momentum.rolling(window).std()
            
            # 4. Momentum reversal (số lần đảo chiều xung lượng)
            momentum_sign = np.sign(momentum)
            momentum_sign_changes = (momentum_sign != momentum_sign.shift(1)).astype(int)
            result_df[f"{prefix}momentum_reversal_{window}"] = momentum_sign_changes.rolling(window).sum()
            
            logger.debug(f"Đã tính đặc trưng xung lượng giá cho cửa sổ {window}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng xung lượng giá cho cửa sổ {window}: {e}")
    
    return result_df

def calculate_price_ratios(
    df: pd.DataFrame,
    price_columns: List[str] = ['open', 'high', 'low', 'close'],
    windows: List[int] = [5, 10, 20, 50],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các tỷ lệ giá và đặc trưng tương đối.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_columns: Danh sách các cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng tỷ lệ giá
    """
    if not validate_price_data(df, price_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu một hoặc nhiều cột {price_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # 1. Tỷ lệ các giá trong cùng một nến
        if 'open' in price_columns and 'close' in price_columns:
            result_df[f"{prefix}close_to_open_ratio"] = result_df['close'] / result_df['open']
        
        if 'high' in price_columns and 'low' in price_columns:
            result_df[f"{prefix}high_to_low_ratio"] = result_df['high'] / result_df['low']
        
        if 'close' in price_columns and 'high' in price_columns and 'low' in price_columns:
            # Position in candle: (close - low) / (high - low)
            high_low_diff = result_df['high'] - result_df['low']
            high_low_diff = high_low_diff.replace(0, np.nan)  # Tránh chia cho 0
            result_df[f"{prefix}position_in_candle"] = (result_df['close'] - result_df['low']) / high_low_diff
        
        # 2. Tỷ lệ giá so với giá trung bình
        for column in price_columns:
            for window in windows:
                # Giá hiện tại so với SMA
                sma = result_df[column].rolling(window=window).mean()
                sma = sma.replace(0, np.nan)  # Tránh chia cho 0
                
                result_df[f"{prefix}{column}_to_sma_{window}_ratio"] = result_df[column] / sma
                
                # Tính khoảng cách giá với min/max
                rolling_min = result_df[column].rolling(window=window).min()
                rolling_max = result_df[column].rolling(window=window).max()
                price_range = rolling_max - rolling_min
                price_range = price_range.replace(0, np.nan)  # Tránh chia cho 0
                
                # Position in range: (price - min) / (max - min)
                result_df[f"{prefix}{column}_position_in_range_{window}"] = (
                    (result_df[column] - rolling_min) / price_range
                )
                
        # 3. Chỉ số giá tương đối giữa giá hiện tại và các giá lịch sử
        if 'close' in price_columns:
            for window in windows:
                # Chỉ số giá tương đối: (Close / SMA - 1) * 100
                sma = result_df['close'].rolling(window=window).mean()
                sma = sma.replace(0, np.nan)  # Tránh chia cho 0
                
                result_df[f"{prefix}relative_price_index_{window}"] = ((result_df['close'] / sma) - 1) * 100
        
        logger.debug("Đã tính đặc trưng tỷ lệ giá")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tỷ lệ giá: {e}")
    
    return result_df

def calculate_price_channels(
    df: pd.DataFrame,
    price_columns: List[str] = ['high', 'low', 'close'],
    windows: List[int] = [10, 20, 50],
    atr_factor: float = 2.0,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các kênh giá và vị trí giá trong kênh.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_columns: Danh sách các cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        atr_factor: Hệ số ATR cho kênh biến động
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng kênh giá
    """
    if not validate_price_data(df, price_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu một hoặc nhiều cột {price_columns}")
        return df
    
    result_df = df.copy()
    
    # Cần đảm bảo có đủ high, low, close cho ATR
    atr_available = all(col in result_df.columns for col in ['high', 'low', 'close'])
    
    if atr_available:
        # Tính True Range cho ATR
        high = result_df['high']
        low = result_df['low']
        close = result_df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    for window in windows:
        try:
            for column in price_columns:
                # Upper & Lower Bands dựa trên Min-Max
                upper_band = result_df[column].rolling(window=window).max()
                lower_band = result_df[column].rolling(window=window).min()
                middle_band = (upper_band + lower_band) / 2
                
                result_df[f"{prefix}{column}_upper_band_{window}"] = upper_band
                result_df[f"{prefix}{column}_lower_band_{window}"] = lower_band
                result_df[f"{prefix}{column}_middle_band_{window}"] = middle_band
                
                # Vị trí của giá trong kênh (0-1)
                channel_width = upper_band - lower_band
                channel_width = channel_width.replace(0, np.nan)  # Tránh chia cho 0
                
                result_df[f"{prefix}{column}_position_in_channel_{window}"] = (
                    (result_df[column] - lower_band) / channel_width
                )
                
                # Khoảng cách tới các band, chuẩn hóa theo chiều rộng kênh
                result_df[f"{prefix}{column}_dist_to_upper_{window}"] = (upper_band - result_df[column]) / channel_width
                result_df[f"{prefix}{column}_dist_to_lower_{window}"] = (result_df[column] - lower_band) / channel_width
                result_df[f"{prefix}{column}_dist_to_middle_{window}"] = abs(result_df[column] - middle_band) / channel_width
            
            # Kênh biến động (Volatility Channel) dựa trên ATR
            if atr_available:
                # Tính ATR
                atr = true_range.rolling(window=window).mean()
                
                # Tính kênh biến động
                if 'close' in price_columns:
                    mid_price = result_df['close'].rolling(window=window).mean()
                    
                    result_df[f"{prefix}volatility_channel_upper_{window}"] = mid_price + (atr * atr_factor)
                    result_df[f"{prefix}volatility_channel_lower_{window}"] = mid_price - (atr * atr_factor)
                    
                    # Vị trí trong kênh biến động
                    vol_channel_width = (atr * atr_factor * 2)
                    vol_channel_width = vol_channel_width.replace(0, np.nan)  # Tránh chia cho 0
                    
                    result_df[f"{prefix}position_in_vol_channel_{window}"] = (
                        (result_df['close'] - (mid_price - atr * atr_factor)) / vol_channel_width
                    )
            
            logger.debug(f"Đã tính đặc trưng kênh giá cho cửa sổ {window}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng kênh giá cho cửa sổ {window}: {e}")
    
    return result_df

def calculate_price_crossovers(
    df: pd.DataFrame,
    price_column: str = 'close',
    ma_windows: List[int] = [5, 10, 20, 50, 100, 200],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng giao cắt giá và đường trung bình động.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        ma_windows: Danh sách các kích thước cửa sổ trung bình động
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng giao cắt giá
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Tính các đường trung bình động
    for window in ma_windows:
        result_df[f"sma_{window}"] = result_df[price_column].rolling(window=window).mean()
    
    try:
        # Tính các tín hiệu giao cắt giữa price và MA
        for window in ma_windows:
            ma_col = f"sma_{window}"
            
            # Price crosses above MA (1 nếu giao cắt từ dưới lên, 0 nếu không)
            result_df[f"{prefix}price_cross_above_{ma_col}"] = (
                (result_df[price_column].shift(1) <= result_df[ma_col].shift(1)) & 
                (result_df[price_column] > result_df[ma_col])
            ).astype(int)
            
            # Price crosses below MA (1 nếu giao cắt từ trên xuống, 0 nếu không)
            result_df[f"{prefix}price_cross_below_{ma_col}"] = (
                (result_df[price_column].shift(1) >= result_df[ma_col].shift(1)) & 
                (result_df[price_column] < result_df[ma_col])
            ).astype(int)
            
            # Price above MA (1 nếu giá > MA, 0 nếu không)
            result_df[f"{prefix}price_above_{ma_col}"] = (
                result_df[price_column] > result_df[ma_col]
            ).astype(int)
            
            # Khoảng cách giữa giá và MA, chuẩn hóa theo MA
            result_df[f"{prefix}price_distance_to_{ma_col}"] = (
                (result_df[price_column] - result_df[ma_col]) / result_df[ma_col]
            ) * 100
        
        # Tính các tín hiệu giao cắt giữa các MA
        for i, window1 in enumerate(ma_windows[:-1]):
            for window2 in ma_windows[i+1:]:
                if window1 < window2:  # chỉ tính cho cặp MA ngắn hạn và dài hạn
                    ma_col1 = f"sma_{window1}"
                    ma_col2 = f"sma_{window2}"
                    
                    # Fast MA crosses above slow MA (1 nếu giao cắt từ dưới lên, 0 nếu không)
                    result_df[f"{prefix}{ma_col1}_cross_above_{ma_col2}"] = (
                        (result_df[ma_col1].shift(1) <= result_df[ma_col2].shift(1)) & 
                        (result_df[ma_col1] > result_df[ma_col2])
                    ).astype(int)
                    
                    # Fast MA crosses below slow MA (1 nếu giao cắt từ trên xuống, 0 nếu không)
                    result_df[f"{prefix}{ma_col1}_cross_below_{ma_col2}"] = (
                        (result_df[ma_col1].shift(1) >= result_df[ma_col2].shift(1)) & 
                        (result_df[ma_col1] < result_df[ma_col2])
                    ).astype(int)
                    
                    # Fast MA above slow MA (1 nếu fast MA > slow MA, 0 nếu không)
                    result_df[f"{prefix}{ma_col1}_above_{ma_col2}"] = (
                        result_df[ma_col1] > result_df[ma_col2]
                    ).astype(int)
                    
                    # Khoảng cách giữa hai MA, chuẩn hóa theo slow MA
                    result_df[f"{prefix}distance_{ma_col1}_to_{ma_col2}"] = (
                        (result_df[ma_col1] - result_df[ma_col2]) / result_df[ma_col2]
                    ) * 100
        
        # Xóa các cột trung gian (các SMA đã tính)
        for window in ma_windows:
            if f"sma_{window}" in result_df.columns:
                del result_df[f"sma_{window}"]
        
        logger.debug(f"Đã tính đặc trưng giao cắt giá")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng giao cắt giá: {e}")
    
    return result_df

def calculate_price_divergence(
    df: pd.DataFrame,
    price_column: str = 'close',
    oscillator_columns: List[str] = None,
    windows: List[int] = [14, 21],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng phân kỳ giữa giá và dao động kỹ thuật.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        oscillator_columns: Danh sách cột dao động kỹ thuật, None để tính tự động
        windows: Danh sách các kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng phân kỳ
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Nếu không cung cấp oscillator_columns, tính RSI làm dao động kỹ thuật mặc định
    if oscillator_columns is None:
        oscillator_columns = []
        
        for window in windows:
            # Tính RSI
            delta = result_df[price_column].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            avg_loss = avg_loss.replace(0, np.nan)  # Tránh chia cho 0
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Thêm vào DataFrame và danh sách oscillator
            rsi_col = f"rsi_{window}"
            result_df[rsi_col] = rsi
            oscillator_columns.append(rsi_col)
    
    try:
        for oscillator in oscillator_columns:
            if oscillator not in result_df.columns:
                logger.warning(f"Không tìm thấy cột dao động {oscillator}, bỏ qua")
                continue
            
            # Tính đỉnh và đáy cục bộ cho giá
            result_df[f"price_peak"] = (
                (result_df[price_column] > result_df[price_column].shift(1)) &
                (result_df[price_column] > result_df[price_column].shift(-1))
            )
            
            result_df[f"price_trough"] = (
                (result_df[price_column] < result_df[price_column].shift(1)) &
                (result_df[price_column] < result_df[price_column].shift(-1))
            )
            
            # Tính đỉnh và đáy cục bộ cho dao động
            result_df[f"oscillator_peak"] = (
                (result_df[oscillator] > result_df[oscillator].shift(1)) &
                (result_df[oscillator] > result_df[oscillator].shift(-1))
            )
            
            result_df[f"oscillator_trough"] = (
                (result_df[oscillator] < result_df[oscillator].shift(1)) &
                (result_df[oscillator] < result_df[oscillator].shift(-1))
            )
            
            # Tính phân kỳ giữa giá và dao động
            
            # Phân kỳ giá tăng nhưng dao động giảm (bearish divergence)
            result_df[f"{prefix}bearish_divergence_{oscillator}"] = (
                result_df[f"price_peak"] &
                (result_df[price_column] > result_df[price_column].shift(2)) &
                result_df[f"oscillator_peak"] &
                (result_df[oscillator] < result_df[oscillator].shift(2))
            ).astype(int)
            
            # Phân kỳ giá giảm nhưng dao động tăng (bullish divergence)
            result_df[f"{prefix}bullish_divergence_{oscillator}"] = (
                result_df[f"price_trough"] &
                (result_df[price_column] < result_df[price_column].shift(2)) &
                result_df[f"oscillator_trough"] &
                (result_df[oscillator] > result_df[oscillator].shift(2))
            ).astype(int)
            
            # Tính độ chênh lệch giữa hướng của giá và dao động
            price_direction = np.sign(result_df[price_column].diff())
            oscillator_direction = np.sign(result_df[oscillator].diff())
            
            result_df[f"{prefix}price_oscillator_direction_diff_{oscillator}"] = (
                (price_direction != oscillator_direction).astype(int)
            )
            
            # Xóa các cột trung gian
            del result_df[f"price_peak"]
            del result_df[f"price_trough"]
            del result_df[f"oscillator_peak"]
            del result_df[f"oscillator_trough"]
            
            logger.debug(f"Đã tính đặc trưng phân kỳ cho dao động {oscillator}")
        
        # Xóa các cột RSI tạm thời nếu đã tính
        for window in windows:
            rsi_col = f"rsi_{window}"
            if rsi_col in result_df.columns and rsi_col not in df.columns:
                del result_df[rsi_col]
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng phân kỳ: {e}")
    
    return result_df