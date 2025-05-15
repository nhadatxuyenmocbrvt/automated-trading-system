"""
Các chỉ báo khối lượng.
File này cung cấp các chỉ báo kỹ thuật cho việc phân tích khối lượng giao dịch
như OBV, A/D Line, Chaikin Money Flow, v.v.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any

# Import các hàm tiện ích
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, true_range
)

def on_balance_volume(
    df: pd.DataFrame,
    close_column: str = 'close',
    volume_column: str = 'volume',
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính On-Balance Volume (OBV).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        close_column: Tên cột giá đóng cửa
        volume_column: Tên cột khối lượng
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa OBV
    """
    if not validate_price_data(df, [close_column, volume_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {close_column} hoặc {volume_column}")
    
    result_df = df.copy()
    
    # Tính OBV
    close_diff = result_df[close_column].diff()
    
    # Tạo Series cho hướng của OBV (1 nếu tăng, -1 nếu giảm, 0 nếu không đổi)
    obv_direction = pd.Series(0, index=result_df.index)
    obv_direction.loc[close_diff > 0] = 1
    obv_direction.loc[close_diff < 0] = -1
    
    # Tính OBV = OBV trước + (Volume * Direction)
    obv = (result_df[volume_column] * obv_direction).cumsum()
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}obv"] = obv
    
    return result_df

def accumulation_distribution_line(
    df: pd.DataFrame,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Accumulation/Distribution Line (A/D Line).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa A/D Line
    """
    required_columns = ['high', 'low', 'close', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    high_low_diff = result_df['high'] - result_df['low']
    
    # Tránh chia cho 0
    high_low_diff = high_low_diff.replace(0, np.nan)
    
    money_flow_multiplier = ((result_df['close'] - result_df['low']) - 
                             (result_df['high'] - result_df['close'])) / high_low_diff
    
    # Tính Money Flow Volume = Money Flow Multiplier * Volume
    money_flow_volume = money_flow_multiplier * result_df['volume']
    
    # Tính A/D Line = Previous A/D Line + Money Flow Volume
    ad_line = money_flow_volume.cumsum()
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}ad_line"] = ad_line
    
    return result_df

def accumulated_distribution_volume(
    df: pd.DataFrame,
    window: int = 14,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Accumulated Distribution Volume (ADV), phiên bản cải tiến của A/D Line
    với độ nhạy tốt hơn dựa trên cộng dồn volume đã điều chỉnh.
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Cửa sổ cho các tính toán trung bình
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa ADV
    """
    required_columns = ['high', 'low', 'close', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    high_low_diff = result_df['high'] - result_df['low']
    
    # Tránh chia cho 0 bằng cách thêm một giá trị rất nhỏ
    epsilon = 1e-10
    safe_high_low_diff = high_low_diff.replace(0, epsilon)
    
    # Tính chỉ số đóng cửa trong khoảng giá (Close Location Value)
    clv = ((result_df['close'] - result_df['low']) - 
           (result_df['high'] - result_df['close'])) / safe_high_low_diff
    
    # Xử lý các giá trị không hợp lệ
    clv = clv.replace([np.inf, -np.inf], np.nan)
    clv = clv.fillna(0)  # Thay thế NaN bằng 0
    
    # Chuẩn hóa về khoảng [-1, 1]
    clv = clv.clip(-1, 1)
    
    # Tính Volume đã điều chỉnh theo hướng (Adjusted Directional Volume)
    adj_dir_volume = clv * result_df['volume']
    
    # Tính ADV - Accumulated Distribution Volume
    adv = adj_dir_volume.cumsum()
    
    # Tính EMA của ADV
    adv_ema = adv.ewm(span=window, min_periods=1).mean()
    
    # Tính chỉ số phân kỳ (divergence) giữa giá và volume
    # Khi giá tăng nhưng ADV giảm -> cảnh báo phân kỳ âm
    price_diff = result_df['close'].diff(window).fillna(0)
    adv_diff = adv.diff(window).fillna(0)
    
    # Chỉ số phân kỳ: 1 cho phân kỳ dương, -1 cho phân kỳ âm, 0 cho không phân kỳ
    divergence = np.zeros(len(result_df))
    
    # Phân kỳ dương: giá giảm nhưng ADV tăng
    pos_div_mask = (price_diff < 0) & (adv_diff > 0)
    divergence[pos_div_mask] = 1
    
    # Phân kỳ âm: giá tăng nhưng ADV giảm
    neg_div_mask = (price_diff > 0) & (adv_diff < 0)
    divergence[neg_div_mask] = -1
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}adv"] = adv
    result_df[f"{prefix}adv_ema_{window}"] = adv_ema
    result_df[f"{prefix}adv_divergence"] = divergence
    
    return result_df

def chaikin_money_flow(
    df: pd.DataFrame,
    window: int = 20,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Chaikin Money Flow (CMF).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa CMF
    """
    required_columns = ['high', 'low', 'close', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    high_low_diff = result_df['high'] - result_df['low']
    
    # Tránh chia cho 0
    high_low_diff = high_low_diff.replace(0, np.nan)
    
    money_flow_multiplier = ((result_df['close'] - result_df['low']) - 
                             (result_df['high'] - result_df['close'])) / high_low_diff
    
    # Tính Money Flow Volume = Money Flow Multiplier * Volume
    money_flow_volume = money_flow_multiplier * result_df['volume']
    
    # Tính Chaikin Money Flow
    # CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
    sum_money_flow_volume = money_flow_volume.rolling(window=window).sum()
    sum_volume = result_df['volume'].rolling(window=window).sum()
    
    cmf = sum_money_flow_volume / sum_volume
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}cmf_{window}"] = cmf
    
    return result_df

def volume_weighted_average_price(
    df: pd.DataFrame,
    window: int = None,
    anchor: str = 'today',
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Volume-Weighted Average Price (VWAP).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Kích thước cửa sổ, nếu None thì tính VWAP từ đầu ngày
        anchor: Điểm neo cho VWAP ('today', 'week', 'month')
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa VWAP
    """
    required_columns = ['high', 'low', 'close', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Typical Price = (High + Low + Close) / 3
    typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # Tính Volume * Typical Price
    vol_price = typical_price * result_df['volume']
    
    # Tạo nhóm dựa trên anchor
    if 'timestamp' in result_df.columns:
        timestamp_col = 'timestamp'
    elif 'time' in result_df.columns:
        timestamp_col = 'time'
    elif 'date' in result_df.columns:
        timestamp_col = 'date'
    else:
        # Nếu không có cột thời gian, sử dụng chỉ mục
        timestamp_col = None
    
    if anchor == 'today' and timestamp_col is not None:
        # Chuyển timestamp thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        
        # Tạo nhóm dựa trên ngày
        result_df['anchor_group'] = result_df[timestamp_col].dt.date
        
    elif anchor == 'week' and timestamp_col is not None:
        # Chuyển timestamp thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        
        # Tạo nhóm dựa trên năm-tuần để tránh lỗi khi chuyển năm
        result_df['anchor_group'] = (
            result_df[timestamp_col].dt.year.astype(str) + "-" + 
            result_df[timestamp_col].dt.isocalendar().week.astype(str)
        )
        
    elif anchor == 'month' and timestamp_col is not None:
        # Chuyển timestamp thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        
        # Tạo nhóm dựa trên tháng
        result_df['anchor_group'] = result_df[timestamp_col].dt.to_period('M')
        
    else:
        # Nếu không có anchor hoặc timestamp, dùng cửa sổ di động
        if window is not None:
            cumsum_vol_price = vol_price.rolling(window=window).sum()
            cumsum_vol = result_df['volume'].rolling(window=window).sum()
            
            # Tránh chia cho 0
            cumsum_vol_non_zero = cumsum_vol.replace(0, np.nan)
            vwap = cumsum_vol_price / cumsum_vol_non_zero
        else:
            # Nếu không có window, tính VWAP từ đầu dữ liệu
            cumsum_vol_price = vol_price.cumsum()
            cumsum_vol = result_df['volume'].cumsum()
            
            # Tránh chia cho 0
            cumsum_vol_non_zero = cumsum_vol.replace(0, np.nan)
            vwap = cumsum_vol_price / cumsum_vol_non_zero
        
        # Đặt tên cột kết quả
        if window is not None:
            result_df[f"{prefix}vwap_{window}"] = vwap
        else:
            result_df[f"{prefix}vwap"] = vwap
        
        return result_df
    
    # Nếu có anchor_group, tính VWAP cho từng nhóm
    result_df['cumsum_vol_price'] = vol_price.groupby(result_df['anchor_group']).cumsum()
    result_df['cumsum_vol'] = result_df['volume'].groupby(result_df['anchor_group']).cumsum()
    
    # Tránh chia cho 0
    cumsum_vol_non_zero = result_df['cumsum_vol'].replace(0, np.nan)
    
    # Tính VWAP
    result_df['vwap'] = result_df['cumsum_vol_price'] / cumsum_vol_non_zero
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}vwap_{anchor}"] = result_df['vwap']
    
    # Xóa các cột tạm thời
    result_df = result_df.drop(['anchor_group', 'cumsum_vol_price', 'cumsum_vol', 'vwap'], axis=1)
    
    return result_df

def ease_of_movement(
    df: pd.DataFrame,
    window: int = 14,
    divisor: int = 10000,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Ease of Movement (EOM).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Kích thước cửa sổ cho việc làm mịn chỉ báo
        divisor: Hệ số chia để tỷ lệ lại giá trị
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa EOM
    """
    required_columns = ['high', 'low', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Mid-point Move
    high_low_avg = (result_df['high'] + result_df['low']) / 2
    mid_point_move = high_low_avg.diff()
    
    # Tính Box Ratio = Volume / (High - Low)
    price_range = result_df['high'] - result_df['low']
    
    # Tránh chia cho 0 bằng cách thêm một giá trị rất nhỏ
    epsilon = 1e-10  # Giá trị rất nhỏ để tránh chia cho 0
    safe_price_range = price_range.replace(0, epsilon)
    
    box_ratio = result_df['volume'] / safe_price_range / divisor
    
    # Tính Ease of Movement = Mid-point Move / Box Ratio
    # Tránh chia cho 0 hoặc NaN
    box_ratio_non_zero = box_ratio.replace(0, np.nan)
    eom = mid_point_move / box_ratio_non_zero
    
    # Xử lý các giá trị vô cùng
    eom = eom.replace([np.inf, -np.inf], np.nan)
    
    # Tính EOM MA, sử dụng min_periods=1 để xử lý tốt hơn các giá trị NaN
    eom_ma = eom.rolling(window=window, min_periods=1).mean()
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}eom"] = eom
    result_df[f"{prefix}eom_ma_{window}"] = eom_ma
    
    return result_df

def volume_oscillator(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    short_window: int = 5,
    long_window: int = 10,
    percentage: bool = True,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Volume Oscillator (VO).
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng
        volume_column: Tên cột khối lượng
        short_window: Kích thước cửa sổ ngắn
        long_window: Kích thước cửa sổ dài
        percentage: Nếu True, kết quả dưới dạng phần trăm
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Volume Oscillator
    """
    if not validate_price_data(df, [volume_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {volume_column}")
    
    result_df = df.copy()
    
    # Tính SMA ngắn và dài của khối lượng
    short_vol_sma = result_df[volume_column].rolling(window=short_window).mean()
    long_vol_sma = result_df[volume_column].rolling(window=long_window).mean()
    
    # Tính Volume Oscillator
    if percentage:
        # Phần trăm: VO = ((Short SMA - Long SMA) / Long SMA) * 100
        vo = ((short_vol_sma - long_vol_sma) / long_vol_sma) * 100
        result_name = f"{prefix}vo_pct_{short_window}_{long_window}"
    else:
        # Tuyệt đối: VO = Short SMA - Long SMA
        vo = short_vol_sma - long_vol_sma
        result_name = f"{prefix}vo_{short_window}_{long_window}"
    
    # Đặt tên cột kết quả
    result_df[result_name] = vo
    
    return result_df

def money_flow_index(
    df: pd.DataFrame,
    window: int = 14,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Money Flow Index (MFI).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa MFI
    """
    required_columns = ['high', 'low', 'close', 'volume']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Typical Price = (High + Low + Close) / 3
    typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # Tính Raw Money Flow = Typical Price * Volume
    raw_money_flow = typical_price * result_df['volume']
    
    # Tính Money Flow Positive và Negative
    # Nếu Typical Price hôm nay > Typical Price hôm qua, đó là Positive Money Flow
    # Nếu Typical Price hôm nay < Typical Price hôm qua, đó là Negative Money Flow
    price_diff = typical_price.diff()
    
    positive_flow = pd.Series(0, index=result_df.index)
    negative_flow = pd.Series(0, index=result_df.index)
    
    positive_flow.loc[price_diff > 0] = raw_money_flow.loc[price_diff > 0]
    negative_flow.loc[price_diff < 0] = raw_money_flow.loc[price_diff < 0]
    
    # Tính tổng Positive và Negative Money Flow trong cửa sổ
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    
    # Tính Money Flow Ratio = Positive Money Flow / Negative Money Flow
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Tính Money Flow Index = 100 - 100 / (1 + Money Flow Ratio)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}mfi_{window}"] = mfi
    
    return result_df

def price_volume_trend(
    df: pd.DataFrame,
    close_column: str = 'close',
    volume_column: str = 'volume',
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Tính Price Volume Trend (PVT).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        close_column: Tên cột giá đóng cửa
        volume_column: Tên cột khối lượng
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa PVT
    """
    if not validate_price_data(df, [close_column, volume_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {close_column} hoặc {volume_column}")
    
    result_df = df.copy()
    
    # Tính phần trăm thay đổi giá
    close_pct_change = result_df[close_column].pct_change()
    
    # Tính PVT từng bước
    pvt_step = result_df[volume_column] * close_pct_change
    
    # Tính PVT luỹ tích
    pvt = pvt_step.cumsum()
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}pvt"] = pvt
    
    return result_df

def log_transform_volume(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    log_base: float = 10.0,
    add_constant: float = 1.0,
    prefix: str = 'volume_'
) -> pd.DataFrame:
    """
    Áp dụng log transform cho dữ liệu khối lượng để ổn định phân phối.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng
        volume_column: Tên cột khối lượng
        log_base: Cơ số log (10.0 cho log10, np.e cho log tự nhiên)
        add_constant: Hằng số cộng thêm để tránh log(0)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa khối lượng đã transform
    """
    if not validate_price_data(df, [volume_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {volume_column}")
    
    result_df = df.copy()
    
    # Tính log(volume + constant)
    if log_base == np.e:  # Sử dụng np.e thay vì math.e
        # Log tự nhiên
        log_volume = np.log1p(result_df[volume_column])
    else:
        # Log với cơ số tùy chọn
        log_volume = np.log10(result_df[volume_column] + add_constant)
    
    # Đặt tên cột kết quả
    result_df[f"{prefix}{volume_column}_log"] = log_volume
    
    return result_df