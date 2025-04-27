"""
Các chỉ báo biến động.
File này cung cấp các chỉ báo kỹ thuật cho việc phân tích biến động thị trường
như ATR, Bollinger Bandwidth, Keltner Channel, v.v.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any

# Import các hàm tiện ích
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, true_range,
    get_highest_high, get_lowest_low
)

def average_true_range(
    df: pd.DataFrame,
    window: int = 14,
    method: str = 'ema',
    normalize_by_price: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Average True Range (ATR).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ
        method: Phương pháp tính ('ema', 'sma', 'rma', 'wilder')
               'rma'/'wilder' là Running Moving Average kiểu Wilder chính xác
        normalize_by_price: Nếu True, tính thêm ATR chia cho giá đóng cửa
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa ATR
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính True Range
    tr = true_range(result_df['high'], result_df['low'], result_df['close'])
    
    # Tính ATR dựa trên phương pháp chỉ định
    if method.lower() == 'ema':
        # Sử dụng EMA với alpha = 2/(window+1)
        atr = tr.ewm(span=window, min_periods=window, adjust=False).mean()
    elif method.lower() in ['rma', 'wilder']:
        # Sử dụng Wilder's Running Moving Average
        # RMA(current, length) = (RMA(prev, length) * (length - 1) + current) / length
        atr = pd.Series(index=tr.index, dtype=float)
        
        # Tính giá trị đầu tiên (SMA)
        first_valid_idx = tr.first_valid_index()
        if first_valid_idx:
            start_idx = tr.index.get_loc(first_valid_idx)
            if start_idx + window <= len(tr):
                first_value = tr.iloc[start_idx:start_idx+window].mean()
                atr.iloc[start_idx+window-1] = first_value
                
                # Tính các giá trị tiếp theo sử dụng công thức Wilder
                for i in range(start_idx+window, len(tr)):
                    atr.iloc[i] = (atr.iloc[i-1] * (window - 1) + tr.iloc[i]) / window
    else:
        # Mặc định sử dụng SMA
        atr = tr.rolling(window=window, min_periods=window).mean()
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}atr_{window}"] = atr
    
    # Tính ATR chia cho giá (chuẩn hóa)
    if normalize_by_price:
        # Tránh chia cho 0 hoặc NaN
        close_prices = result_df['close'].replace(0, np.nan)
        atr_normalized = atr / close_prices
        
        # ATR dưới dạng phần trăm của giá
        atr_pct = atr_normalized * 100
        result_df[f"{prefix}atr_pct_{window}"] = atr_pct
        
        # ATR chuẩn hóa (dạng thập phân, hữu ích cho neural networks)
        result_df[f"{prefix}atr_norm_{window}"] = atr_normalized
    
    return result_df

def bollinger_bandwidth(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    std_dev: float = 2.0,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Bollinger Bandwidth.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        std_dev: Số độ lệch chuẩn cho các băng trên và dưới
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Bollinger Bandwidth
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính các thành phần của Bollinger Bands
    rolling = result_df[column].rolling(window=window)
    
    # Middle band là SMA
    middle_band = rolling.mean()
    
    # Tính độ lệch chuẩn
    std = rolling.std(ddof=0)  # ddof=0 cho population standard deviation
    
    # Upper và lower bands
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    
    # Tính Bollinger Bandwidth = (Upper Band - Lower Band) / Middle Band
    bandwidth = (upper_band - lower_band) / middle_band
    
    # Tính Bandwidth Percent Rank để đánh giá biến động tương đối
    # %B = (Price - Lower Band) / (Upper Band - Lower Band)
    percent_b = (result_df[column] - lower_band) / (upper_band - lower_band)
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}bbw_{window}"] = bandwidth
    result_df[f"{prefix}bbw_percentb_{window}"] = percent_b
    
    return result_df

def keltner_channel(
    df: pd.DataFrame,
    window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Keltner Channel.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ cho EMA của giá
        atr_window: Kích thước cửa sổ cho ATR
        atr_multiplier: Hệ số nhân cho ATR
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa giá trị middle line, upper channel, lower channel
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính EMA của giá đóng cửa (middle line)
    middle_line = result_df['close'].ewm(span=window, adjust=False).mean()
    
    # Tính ATR
    tr = true_range(result_df['high'], result_df['low'], result_df['close'])
    atr = tr.ewm(alpha=1/atr_window, min_periods=atr_window, adjust=False).mean()
    
    # Tính upper và lower channel
    upper_channel = middle_line + (atr_multiplier * atr)
    lower_channel = middle_line - (atr_multiplier * atr)
    
    # Tính channel width
    channel_width = (upper_channel - lower_channel) / middle_line
    
    # Tính position in channel
    position = (result_df['close'] - lower_channel) / (upper_channel - lower_channel)
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}kc_middle_{window}"] = middle_line
    result_df[f"{prefix}kc_upper_{window}"] = upper_channel
    result_df[f"{prefix}kc_lower_{window}"] = lower_channel
    result_df[f"{prefix}kc_width_{window}"] = channel_width
    result_df[f"{prefix}kc_position_{window}"] = position
    
    return result_df

def donchian_channel(
    df: pd.DataFrame,
    window: int = 20,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Donchian Channel.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa giá trị upper bound, lower bound, middle line
    """
    required_columns = ['high', 'low']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính upper bound, lower bound, và middle line
    upper_bound = get_highest_high(result_df['high'], window)
    lower_bound = get_lowest_low(result_df['low'], window)
    middle_line = (upper_bound + lower_bound) / 2
    
    # Tính channel width
    channel_width = (upper_bound - lower_bound) / middle_line
    
    # Tính position in channel nếu có giá đóng cửa
    if 'close' in result_df.columns:
        position = (result_df['close'] - lower_bound) / (upper_bound - lower_bound)
        result_df[f"{prefix}dc_position_{window}"] = position
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}dc_upper_{window}"] = upper_bound
    result_df[f"{prefix}dc_lower_{window}"] = lower_bound
    result_df[f"{prefix}dc_middle_{window}"] = middle_line
    result_df[f"{prefix}dc_width_{window}"] = channel_width
    
    return result_df

def ulcer_index(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 14,
    handle_na: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Ulcer Index.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        handle_na: Nếu True, xử lý NaN an toàn hơn
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Ulcer Index
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Loại bỏ NaN để tránh lỗi runtime
    price_data = result_df[column]
    if handle_na:
        # Sử dụng .dropna() để loại bỏ các giá trị NaN trước khi tính toán
        valid_price = price_data.dropna()
        if len(valid_price) == 0:
            # Nếu tất cả đều là NaN, trả về NaN
            result_df[f"{prefix}ui_{window}"] = np.nan
            return result_df
    else:
        valid_price = price_data
    
    # Sử dụng valid_price để tính rolling_max
    # min_periods=1 để tránh sinh quá nhiều NaN
    roll_max = valid_price.rolling(window=window, min_periods=1).max()
    
    # Tránh chia cho 0 hoặc NaN
    roll_max_non_zero = roll_max.replace(0, np.nan)
    
    # Tính phần trăm giảm từ đỉnh
    pct_drawdown = ((valid_price - roll_max_non_zero) / roll_max_non_zero) * 100
    
    # Bình phương phần trăm giảm (chỉ các giá trị âm được tính, các giá trị dương thành 0)
    squared_drawdown = pct_drawdown.clip(upper=0) ** 2
    
    # Tính Ulcer Index = sqrt(mean of squared_drawdown)
    # min_periods=1 để sử dụng nhiều dữ liệu có thể
    ulcer_index = np.sqrt(squared_drawdown.rolling(window=window, min_periods=1).mean())
    
    # Thêm bảo vệ tránh giá trị không hợp lệ
    ulcer_index = ulcer_index.replace([np.inf, -np.inf], np.nan)
    
    # Nếu giá trị ban đầu là NaN, kết quả sẽ là NaN
    na_mask = price_data.isna()
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}ui_{window}"] = ulcer_index
    
    # Đảm bảo giá trị NaN được bảo toàn
    if handle_na:
        result_df.loc[na_mask, f"{prefix}ui_{window}"] = np.nan
    
    return result_df

def standard_deviation(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    trading_periods: int = 252,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Standard Deviation và Annualized Volatility.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        trading_periods: Số phiên giao dịch trong năm (252 cho ngày, 52 cho tuần, 12 cho tháng)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa độ lệch chuẩn và biến động hàng năm
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính phần trăm thay đổi giá
    returns = result_df[column].pct_change()
    
    # Tính rolling standard deviation của returns
    stddev = returns.rolling(window=window).std()
    
    # Tính annualized volatility = stddev * sqrt(trading_periods)
    annualized_vol = stddev * np.sqrt(trading_periods)
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}stddev_{window}"] = stddev
    result_df[f"{prefix}annvol_{window}"] = annualized_vol
    
    return result_df

def historical_volatility(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    trading_periods: int = 252,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Historical Volatility.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        trading_periods: Số phiên giao dịch trong năm (252 cho ngày, 52 cho tuần, 12 cho tháng)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Historical Volatility
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính log returns
    log_returns = np.log(result_df[column] / result_df[column].shift(1))
    
    # Tính độ lệch chuẩn của log returns
    rolling_std = log_returns.rolling(window=window).std()
    
    # Tính Historical Volatility = stddev * sqrt(trading_periods)
    hist_vol = rolling_std * np.sqrt(trading_periods)
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}hvol_{window}"] = hist_vol
    
    # Thêm cả phiên bản phần trăm
    result_df[f"{prefix}hvol_pct_{window}"] = hist_vol * 100
    
    return result_df

def volatility_ratio(
    df: pd.DataFrame,
    column: str = 'close',
    short_window: int = 5,
    long_window: int = 20,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Volatility Ratio (tỷ lệ biến động ngắn hạn so với dài hạn).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        short_window: Kích thước cửa sổ ngắn
        long_window: Kích thước cửa sổ dài
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Volatility Ratio
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính phần trăm thay đổi giá
    returns = result_df[column].pct_change()
    
    # Tính độ lệch chuẩn trong cửa sổ ngắn và dài
    short_vol = returns.rolling(window=short_window).std()
    long_vol = returns.rolling(window=long_window).std()
    
    # Tính Volatility Ratio = short_vol / long_vol
    vol_ratio = short_vol / long_vol
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}vol_ratio_{short_window}_{long_window}"] = vol_ratio
    
    return result_df