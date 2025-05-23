"""
Các chỉ báo biến động.
File này cung cấp các chỉ báo kỹ thuật cho việc phân tích biến động thị trường
như ATR, Bollinger Bandwidth, Keltner Channel, v.v.
"""

import pandas as pd
import numpy as np
import logging
from typing import Union, List, Dict, Tuple, Optional, Any

# Import hàm tiện ích xử lý NaN
from data_processors.utils.preprocessing import fill_nan_values, handle_leading_nans

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
    prefix: str = '',
    min_periods: int = 1,  # Thêm tham số để giảm NaN
    handle_leading_nan: bool = True  # Thêm tham số xử lý NaN đầu tiên
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
        min_periods: Số lượng giá trị tối thiểu để tính trong cửa sổ
        handle_leading_nan: Nếu True, điền giá trị NaN đầu bằng giá trị hợp lệ đầu tiên
        
    Returns:
        DataFrame với cột mới chứa ATR
    """
    # Sử dụng logger thay vì print
    logger = logging.getLogger(__name__)
    
    try:
        # Kiểm tra dữ liệu đầu vào
        required_columns = ['high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {missing_cols}")
        
        # Tạo bản sao để tránh thay đổi DataFrame gốc
        result_df = df.copy()
        
        # Tính True Range
        tr = true_range(result_df['high'], result_df['low'], result_df['close'])
        
        # Tính ATR dựa trên phương pháp chỉ định với min_periods giảm NaN đầu
        if method.lower() == 'ema':
            # Sử dụng EMA với min_periods để giảm số lượng NaN
            atr = tr.ewm(span=window, min_periods=min_periods, adjust=False).mean()
        elif method.lower() in ['rma', 'wilder']:
            # Sử dụng Wilder's Running Moving Average với min_periods cải tiến
            atr = pd.Series(index=tr.index, dtype=float)
            
            # Tính giá trị đầu tiên (SMA) với min_periods giảm
            first_valid_idx = tr.first_valid_index()
            if first_valid_idx:
                start_idx = tr.index.get_loc(first_valid_idx)
                
                # Sử dụng min_periods thay vì window để bắt đầu sớm hơn
                if start_idx + min_periods <= len(tr):
                    # Tính SMA đầu tiên dựa trên min_periods
                    first_value = tr.iloc[start_idx:start_idx+min_periods].mean()
                    atr.iloc[start_idx] = first_value
                    
                    # Tính các giá trị RMA tiếp theo từ điểm thứ hai
                    for i in range(start_idx + 1, len(tr)):
                        atr.iloc[i] = (atr.iloc[i-1] * (window - 1) + tr.iloc[i]) / window
        else:
            # Mặc định sử dụng SMA với min_periods giảm NaN
            atr = tr.rolling(window=window, min_periods=min_periods).mean()
        
        # Đặt tên cột kết quả
        atr_col_name = f"{prefix}atr_{window}"
        result_df[atr_col_name] = atr
        
        # Xử lý NaN ở đầu dữ liệu nếu cần
        if handle_leading_nan:
            result_df[atr_col_name] = handle_leading_nans(result_df[atr_col_name])
            logger.info(f"Đã xử lý các giá trị NaN ở đầu cho {atr_col_name}")
        
        # Tính ATR chia cho giá (chuẩn hóa)
        if normalize_by_price:
            # Tránh chia cho 0 hoặc NaN
            close_prices = result_df['close'].replace(0, np.nan)
            atr_normalized = result_df[atr_col_name] / close_prices
            
            # ATR dưới dạng phần trăm của giá
            atr_pct = atr_normalized * 100
            atr_pct_col = f"{prefix}atr_pct_{window}"
            result_df[atr_pct_col] = atr_pct
            
            # ATR chuẩn hóa (dạng thập phân)
            atr_norm_col = f"{prefix}atr_norm_{window}"
            result_df[atr_norm_col] = atr_normalized
            
            # Xử lý giá trị NaN ở đầu dữ liệu cho các cột chuẩn hóa
            if handle_leading_nan:
                result_df[atr_pct_col] = handle_leading_nans(result_df[atr_pct_col])
                result_df[atr_norm_col] = handle_leading_nans(result_df[atr_norm_col])
                logger.info(f"Đã xử lý các giá trị NaN ở đầu cho {atr_pct_col} và {atr_norm_col}")
        
        # Thêm volatility_rank dựa trên ATR
        volatility_rank_col = f"{prefix}volatility_rank"
        result_df[volatility_rank_col] = result_df[atr_col_name].rank(pct=True) * 100
        
        # Xử lý NaN trong volatility_rank
        if handle_leading_nan:
            result_df[volatility_rank_col] = handle_leading_nans(result_df[volatility_rank_col])
            logger.info(f"Đã xử lý các giá trị NaN ở đầu cho {volatility_rank_col}")
        
        # Ghi log các cột mới đã tạo
        new_cols = [col for col in result_df.columns if col not in df.columns]
        logger.info(f"ATR: Đã tạo {len(new_cols)} cột mới: {new_cols}")
        
        return result_df
    
    except Exception as e:
        # Ghi log lỗi
        logger.error(f"Lỗi khi tính ATR: {e}")
        
        # Trả về DataFrame với cột NaN để quá trình không bị gián đoạn
        result_df = df.copy()
        result_df[f"{prefix}atr_{window}"] = np.nan
        
        if normalize_by_price:
            result_df[f"{prefix}atr_pct_{window}"] = np.nan
            result_df[f"{prefix}atr_norm_{window}"] = np.nan
        
        result_df[f"{prefix}volatility_rank"] = np.nan
            
        return result_df

def bollinger_bands(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    std_dev: float = 2.0,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Bollinger Bands.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        std_dev: Số độ lệch chuẩn cho các băng trên và dưới
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
    Returns:
        DataFrame với các cột mới chứa Middle Band, Upper Band, Lower Band, 
        Bandwidth và %B
    """
    # Sử dụng logger của module
    logger = logging.getLogger(__name__)
    
    try:
        # Kiểm tra dữ liệu đầu vào
        if column not in df.columns:
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        # Kiểm tra thêm giá trị NaN
        if df[column].isna().all():
            raise ValueError(f"Cột {column} chỉ chứa giá trị NaN")
        
        # Tạo bản sao để tránh thay đổi DataFrame gốc
        result_df = df.copy()
        
        # Tính các thành phần của Bollinger Bands
        rolling = result_df[column].rolling(window=window, min_periods=1)
        
        # Middle band là SMA
        middle_band = rolling.mean()
        
        # Tính độ lệch chuẩn
        std = rolling.std(ddof=0)  # ddof=0 cho population standard deviation
        
        # Upper và lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Gán các giá trị vào DataFrame với tên cột rõ ràng
        middle_band_col = f"{prefix}bb_middle_{window}"
        upper_band_col = f"{prefix}bb_upper_{window}"
        lower_band_col = f"{prefix}bb_lower_{window}"
        
        result_df[middle_band_col] = middle_band
        result_df[upper_band_col] = upper_band
        result_df[lower_band_col] = lower_band
        
        # Xử lý NaN ở đầu nếu cần
        if handle_leading_nan:
            result_df[middle_band_col] = handle_leading_nans(result_df[middle_band_col])
            result_df[upper_band_col] = handle_leading_nans(result_df[upper_band_col])
            result_df[lower_band_col] = handle_leading_nans(result_df[lower_band_col])
        
        # Tính Bollinger Bandwidth = (Upper Band - Lower Band) / Middle Band
        # Tránh chia cho 0 bằng cách thêm giá trị rất nhỏ
        bandwidth = (upper_band - lower_band) / (middle_band + 1e-10)
        bandwidth_col = f"{prefix}bb_bandwidth_{window}"
        result_df[bandwidth_col] = bandwidth
        
        if handle_leading_nan:
            result_df[bandwidth_col] = handle_leading_nans(result_df[bandwidth_col])
        
        # Tính %B = (Price - Lower Band) / (Upper Band - Lower Band)
        # Tránh chia cho 0 bằng cách thêm giá trị rất nhỏ
        band_diff = upper_band - lower_band + 1e-10
        percent_b = (result_df[column] - lower_band) / band_diff
        percent_b_col = f"{prefix}bb_percent_b_{window}"
        result_df[percent_b_col] = percent_b
        
        if handle_leading_nan:
            result_df[percent_b_col] = handle_leading_nans(result_df[percent_b_col])
        
        # Ghi log các cột mới đã tạo
        new_cols = [col for col in result_df.columns if col not in df.columns]
        logger.info(f"Bollinger Bands: Đã tạo {len(new_cols)} cột mới: {new_cols}")
        
        return result_df
    
    except Exception as e:
        # Ghi log lỗi bằng logger
        logger.error(f"Lỗi khi tính Bollinger Bands: {str(e)}")
        
        # Trả về DataFrame với các cột mặc định để tránh gián đoạn quy trình
        result_df = df.copy()
        result_df[f"{prefix}bb_middle_{window}"] = np.nan
        result_df[f"{prefix}bb_upper_{window}"] = np.nan
        result_df[f"{prefix}bb_lower_{window}"] = np.nan
        result_df[f"{prefix}bb_bandwidth_{window}"] = np.nan
        result_df[f"{prefix}bb_percent_b_{window}"] = np.nan
        
        # Đảm bảo trả về ít nhất một cột cần thiết
        logger.info(f"Đã tạo các cột mặc định cho Bollinger Bands")
        
        return result_df

# Thêm hàm bollinger_bandwidth riêng biệt
def bollinger_bandwidth(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    std_dev: float = 2.0,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Bollinger Bandwidth - đo lường độ rộng của dải Bollinger Bands.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        std_dev: Số độ lệch chuẩn cho các băng trên và dưới
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
    Returns:
        DataFrame với cột mới chứa Bollinger Bandwidth
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Kiểm tra dữ liệu đầu vào
        if column not in df.columns:
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        # Tạo bản sao để tránh thay đổi DataFrame gốc
        result_df = df.copy()
        
        # Tính các thành phần của Bollinger Bands
        rolling = result_df[column].rolling(window=window, min_periods=1)
        
        # Middle band là SMA
        middle_band = rolling.mean()
        
        # Tính độ lệch chuẩn
        std = rolling.std(ddof=0)
        
        # Upper và lower bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Tính Bollinger Bandwidth = (Upper Band - Lower Band) / Middle Band
        # Tránh chia cho 0 bằng cách thêm giá trị rất nhỏ
        bandwidth = (upper_band - lower_band) / (middle_band + 1e-10)
        
        # Gán giá trị vào DataFrame với tên cột rõ ràng
        bandwidth_col = f"{prefix}bb_bandwidth_{window}"
        result_df[bandwidth_col] = bandwidth
        
        if handle_leading_nan:
            result_df[bandwidth_col] = handle_leading_nans(result_df[bandwidth_col])
        
        # Thêm cột chỉ báo biến động tăng/giảm
        bandwidth_delta = bandwidth - bandwidth.shift(1)
        bandwidth_delta_col = f"{prefix}bb_bandwidth_delta_{window}"
        result_df[bandwidth_delta_col] = bandwidth_delta
        
        if handle_leading_nan:
            result_df[bandwidth_delta_col] = handle_leading_nans(result_df[bandwidth_delta_col])
        
        # Thêm cột chỉ báo biến động tương đối
        bandwidth_rolling = bandwidth.rolling(window=window)
        bandwidth_mean = bandwidth_rolling.mean()
        bandwidth_std = bandwidth_rolling.std()
        
        # Tránh chia cho 0
        bandwidth_std_non_zero = bandwidth_std.replace(0, np.nan).fillna(1e-10)
        
        bandwidth_pct = (bandwidth - bandwidth_mean) / bandwidth_std_non_zero
        bandwidth_pct_col = f"{prefix}bb_bandwidth_pct_{window}"
        result_df[bandwidth_pct_col] = bandwidth_pct
        
        if handle_leading_nan:
            result_df[bandwidth_pct_col] = handle_leading_nans(result_df[bandwidth_pct_col])
        
        # Ghi log các cột mới đã tạo
        new_cols = [col for col in result_df.columns if col not in df.columns]
        logger.info(f"Bollinger Bandwidth: Đã tạo {len(new_cols)} cột mới: {new_cols}")
        
        return result_df
    
    except Exception as e:
        # Ghi log lỗi bằng logger
        logger.error(f"Lỗi khi tính Bollinger Bandwidth: {str(e)}")
        
        # Trả về DataFrame với cột mặc định
        result_df = df.copy()
        result_df[f"{prefix}bb_bandwidth_{window}"] = np.nan
        result_df[f"{prefix}bb_bandwidth_delta_{window}"] = np.nan
        result_df[f"{prefix}bb_bandwidth_pct_{window}"] = np.nan
        
        return result_df

def keltner_channel(
    df: pd.DataFrame,
    window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Keltner Channel.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ cho EMA của giá
        atr_window: Kích thước cửa sổ cho ATR
        atr_multiplier: Hệ số nhân cho ATR
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
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
    middle_line_col = f"{prefix}kc_middle_{window}"
    upper_channel_col = f"{prefix}kc_upper_{window}"
    lower_channel_col = f"{prefix}kc_lower_{window}"
    channel_width_col = f"{prefix}kc_width_{window}"
    position_col = f"{prefix}kc_position_{window}"
    
    result_df[middle_line_col] = middle_line
    result_df[upper_channel_col] = upper_channel
    result_df[lower_channel_col] = lower_channel
    result_df[channel_width_col] = channel_width
    result_df[position_col] = position
    
    # Xử lý NaN ở đầu nếu cần
    if handle_leading_nan:
        result_df[middle_line_col] = handle_leading_nans(result_df[middle_line_col])
        result_df[upper_channel_col] = handle_leading_nans(result_df[upper_channel_col])
        result_df[lower_channel_col] = handle_leading_nans(result_df[lower_channel_col])
        result_df[channel_width_col] = handle_leading_nans(result_df[channel_width_col])
        result_df[position_col] = handle_leading_nans(result_df[position_col])
    
    return result_df

def donchian_channel(
    df: pd.DataFrame,
    window: int = 20,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Donchian Channel.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
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
        position_col = f"{prefix}dc_position_{window}"
        result_df[position_col] = position
        
        if handle_leading_nan:
            result_df[position_col] = handle_leading_nans(result_df[position_col])
    
    # Đặt tên các cột kết quả
    upper_bound_col = f"{prefix}dc_upper_{window}"
    lower_bound_col = f"{prefix}dc_lower_{window}"
    middle_line_col = f"{prefix}dc_middle_{window}"
    channel_width_col = f"{prefix}dc_width_{window}"
    
    result_df[upper_bound_col] = upper_bound
    result_df[lower_bound_col] = lower_bound
    result_df[middle_line_col] = middle_line
    result_df[channel_width_col] = channel_width
    
    # Xử lý NaN ở đầu nếu cần
    if handle_leading_nan:
        result_df[upper_bound_col] = handle_leading_nans(result_df[upper_bound_col])
        result_df[lower_bound_col] = handle_leading_nans(result_df[lower_bound_col])
        result_df[middle_line_col] = handle_leading_nans(result_df[middle_line_col])
        result_df[channel_width_col] = handle_leading_nans(result_df[channel_width_col])
    
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
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Standard Deviation và Annualized Volatility.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        trading_periods: Số phiên giao dịch trong năm (252 cho ngày, 52 cho tuần, 12 cho tháng)
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
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
    stddev_col = f"{prefix}stddev_{window}"
    annvol_col = f"{prefix}annvol_{window}"
    
    result_df[stddev_col] = stddev
    result_df[annvol_col] = annualized_vol
    
    # Xử lý NaN ở đầu nếu cần
    if handle_leading_nan:
        result_df[stddev_col] = handle_leading_nans(result_df[stddev_col])
        result_df[annvol_col] = handle_leading_nans(result_df[annvol_col])
    
    return result_df

def historical_volatility(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    trading_periods: int = 252,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Historical Volatility.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        trading_periods: Số phiên giao dịch trong năm (252 cho ngày, 52 cho tuần, 12 cho tháng)
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
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
    hvol_col = f"{prefix}hvol_{window}"
    hvol_pct_col = f"{prefix}hvol_pct_{window}"
    
    result_df[hvol_col] = hist_vol
    result_df[hvol_pct_col] = hist_vol * 100
    
    # Xử lý NaN ở đầu nếu cần
    if handle_leading_nan:
        result_df[hvol_col] = handle_leading_nans(result_df[hvol_col])
        result_df[hvol_pct_col] = handle_leading_nans(result_df[hvol_pct_col])
    
    return result_df

def volatility_ratio(
    df: pd.DataFrame,
    column: str = 'close',
    short_window: int = 5,
    long_window: int = 20,
    prefix: str = '',
    handle_leading_nan: bool = True
) -> pd.DataFrame:
    """
    Tính Volatility Ratio (tỷ lệ biến động ngắn hạn so với dài hạn).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        short_window: Kích thước cửa sổ ngắn
        long_window: Kích thước cửa sổ dài
        prefix: Tiền tố cho tên cột kết quả
        handle_leading_nan: Nếu True, điền giá trị NaN ở đầu
        
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
    
    # Tránh chia cho 0
    long_vol_non_zero = long_vol.replace(0, np.nan).fillna(1e-10)
    
    # Tính Volatility Ratio = short_vol / long_vol
    vol_ratio = short_vol / long_vol_non_zero
    
    # Đặt tên cột kết quả
    vol_ratio_col = f"{prefix}vol_ratio_{short_window}_{long_window}"
    result_df[vol_ratio_col] = vol_ratio
    
    # Xử lý NaN ở đầu nếu cần
    if handle_leading_nan:
        result_df[vol_ratio_col] = handle_leading_nans(result_df[vol_ratio_col])
    
    return result_df