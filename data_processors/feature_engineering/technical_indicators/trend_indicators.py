"""
Các chỉ báo xu hướng.
File này cung cấp các chỉ báo kỹ thuật cho việc phân tích xu hướng thị trường
như Moving Average, MACD, Bollinger Bands, v.v.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
import numba as nb

# Import các hàm tiện ích
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, exponential_weights,
    calculate_weighted_average, true_range
)

def simple_moving_average(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    min_periods: int = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Simple Moving Average (SMA).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ (số nến)
        min_periods: Số lượng giá trị tối thiểu cần thiết, mặc định là window
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa giá trị SMA
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    # Nếu không chỉ định min_periods, sử dụng window
    if min_periods is None:
        min_periods = window
    
    # Tính SMA
    result_df = df.copy()
    sma = result_df[column].rolling(window=window, min_periods=min_periods).mean()
    
    # Đặt tên cột kết quả
    result_name = f"{prefix}sma_{window}"
    result_df[result_name] = sma
    
    return result_df

def exponential_moving_average(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    alpha: float = None,
    adjust: bool = False,  # Mặc định là False để phù hợp với trading
    min_periods: int = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Exponential Moving Average (EMA).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ (số nến)
        alpha: Hệ số làm mượt (nếu None, sẽ sử dụng span=window)
        adjust: Điều chỉnh trọng số để bù đắp hiệu ứng chuỗi hữu hạn
        min_periods: Số lượng giá trị tối thiểu cần thiết, mặc định là window
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa giá trị EMA
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    # Nếu không chỉ định min_periods, sử dụng window
    if min_periods is None:
        min_periods = window
    
    # Tính EMA
    result_df = df.copy()
    
    # Chỉ sử dụng một trong hai tham số: span hoặc alpha
    if alpha is None:
        # Sử dụng span nếu không có alpha
        ema = result_df[column].ewm(
            span=window,
            min_periods=min_periods,
            adjust=adjust
        ).mean()
    else:
        # Sử dụng alpha nếu được cung cấp
        ema = result_df[column].ewm(
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust
        ).mean()
    
    # Đặt tên cột kết quả
    result_name = f"{prefix}ema_{window}"
    result_df[result_name] = ema
    
    return result_df

def bollinger_bands(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 20,
    std_dev: float = 2.0,
    min_periods: int = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Bollinger Bands.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ (số nến)
        std_dev: Số độ lệch chuẩn cho các băng trên và dưới
        min_periods: Số lượng giá trị tối thiểu cần thiết, mặc định là window
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa giá trị middle band, upper band, lower band
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    # Nếu không chỉ định min_periods, sử dụng window
    if min_periods is None:
        min_periods = window
    
    # Tính các thành phần của Bollinger Bands
    result_df = df.copy()
    
    try:
        rolling = result_df[column].rolling(window=window, min_periods=min_periods)
        
        # Middle band là SMA
        middle_band = rolling.mean()
        
        # Tính độ lệch chuẩn
        std = rolling.std(ddof=0)  # ddof=0 cho population standard deviation
        
        # Upper và lower bands
        upper_band = middle_band + std_dev * std
        lower_band = middle_band - std_dev * std
        
        # Bandwidth và %B
        # Thêm kiểm tra để tránh chia cho 0 
        bandwidth = pd.Series(np.zeros_like(middle_band), index=middle_band.index)
        mask = middle_band != 0  # Tạo mask cho giá trị khác 0
        bandwidth[mask] = (upper_band[mask] - lower_band[mask]) / middle_band[mask]
        
        # Tránh chia cho 0 khi tính %B
        denominator = upper_band - lower_band
        # Thay thế giá trị 0 bằng NaN để tránh lỗi khi chia
        denominator = denominator.replace(0, np.nan)
        percent_b = (result_df[column] - lower_band) / denominator
        
        # Thêm từng cột vào DataFrame kết quả sử dụng phương thức assign()
        # Đây là cách an toàn hơn để tránh lỗi internal block structure
        result_df = result_df.assign(**{
            f"{prefix}bb_middle_{window}": middle_band,
            f"{prefix}bb_upper_{window}": upper_band,
            f"{prefix}bb_lower_{window}": lower_band,
            f"{prefix}bb_bandwidth_{window}": bandwidth,
            f"{prefix}bb_percent_b_{window}": percent_b
        })
    
    except Exception as e:
        import logging
        logging.warning(f"Lỗi khi tính Bollinger Bands: {str(e)}")
        # Trả về DataFrame gốc nếu có lỗi
        return df
    
    return result_df

def moving_average_convergence_divergence(
    df: pd.DataFrame,
    column: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    normalize: bool = False,
    norm_method: str = 'zscore',
    norm_window: int = 100,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Moving Average Convergence Divergence (MACD).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        fast_period: Kích thước cửa sổ của EMA nhanh
        slow_period: Kích thước cửa sổ của EMA chậm
        signal_period: Kích thước cửa sổ của đường tín hiệu
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa giá trị MACD, Signal, Histogram
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    # Tính các EMA - đảm bảo adjust=False cho đúng cách tính trong trading
    result_df = df.copy()
    ema_fast = result_df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = result_df[column].ewm(span=slow_period, adjust=False).mean()
    
    # MACD line = EMA nhanh - EMA chậm
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA của MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Tìm vị trí sau dòng code:
    histogram = macd_line - signal_line

    # Thêm đoạn code sau:
    # Kiểm tra và xử lý các giá trị lệch
    max_diff = 5.0  # Ngưỡng chênh lệch tối đa giữa MACD và Signal
    extreme_diff_mask = np.abs(macd_line - signal_line) > max_diff
    if extreme_diff_mask.any():
        # Ghi log số lượng giá trị lệch
        extreme_count = extreme_diff_mask.sum()
        print(f"Cảnh báo: Phát hiện {extreme_count} giá trị chênh lệch quá lớn giữa MACD và Signal")
        
        # Điều chỉnh histogram trong trường hợp có chênh lệch lớn
        # Giới hạn biên độ histogram để tránh ảnh hưởng đến biểu đồ
        histogram = np.clip(histogram, -max_diff, max_diff)

    # Kiểm tra giá trị không hợp lệ (NaN, Inf)
    invalid_macd = ~np.isfinite(macd_line)
    invalid_signal = ~np.isfinite(signal_line)
    invalid_hist = ~np.isfinite(histogram)

    if invalid_macd.any() or invalid_signal.any() or invalid_hist.any():
        # Đếm số lượng giá trị không hợp lệ
        invalid_count = max(invalid_macd.sum(), invalid_signal.sum(), invalid_hist.sum())
        print(f"Cảnh báo: Phát hiện {invalid_count} giá trị không hợp lệ trong MACD/Signal/Histogram")
        
        # Thay thế các giá trị không hợp lệ bằng 0
        macd_line = np.where(invalid_macd, 0, macd_line)
        signal_line = np.where(invalid_signal, 0, signal_line)
        histogram = np.where(invalid_hist, 0, histogram)    

    # Đặt tên các cột kết quả
    result_df[f"{prefix}macd_line"] = macd_line
    result_df[f"{prefix}macd_signal"] = signal_line
    result_df[f"{prefix}macd_histogram"] = histogram
    
    # Chuẩn hóa nếu yêu cầu
    if normalize:
        result_df = standardize_macd(
            result_df,
            macd_col=f"{prefix}macd_line",
            signal_col=f"{prefix}macd_signal",
            histogram_col=f"{prefix}macd_histogram",
            window=norm_window,
            method=norm_method,
            prefix=""  # Đặt là chuỗi rỗng để tránh lặp prefix
        )
        
    return result_df

def average_directional_index(
    df: pd.DataFrame,
    window: int = 14,
    smooth_period: int = 14,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Average Directional Index (ADX).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ cho việc tính +DI, -DI
        smooth_period: Kích thước cửa sổ cho việc làm mượt ADX
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa giá trị ADX, +DI, -DI
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính True Range
    tr = true_range(result_df['high'], result_df['low'], result_df['close'])
    
    # Tính +DM và -DM (Directional Movement)
    high_diff = result_df['high'].diff()
    low_diff = result_df['low'].diff().multiply(-1)
    
    # Tối ưu việc tính +DM và -DM bằng cách sử dụng numpy
    plus_dm = np.zeros(len(high_diff))
    minus_dm = np.zeros(len(high_diff))
    
    # Tạo mask cho +DM: high_diff > low_diff và high_diff > 0
    plus_mask = (high_diff > low_diff) & (high_diff > 0)
    plus_dm = np.where(plus_mask, high_diff, 0)
    
    # Tạo mask cho -DM: low_diff > high_diff và low_diff > 0
    minus_mask = (low_diff > high_diff) & (low_diff > 0)
    minus_dm = np.where(minus_mask, low_diff, 0)
    
    # Chuyển đổi mảng numpy thành Series
    plus_dm = pd.Series(plus_dm, index=result_df.index)
    minus_dm = pd.Series(minus_dm, index=result_df.index)
    
    # Tính smoothed TR, +DM, -DM sử dụng Wilder's smoothing (EMA với alpha=1/n)
    tr_ema = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_dm_ema = plus_dm.ewm(alpha=1/window, adjust=False).mean()
    minus_dm_ema = minus_dm.ewm(alpha=1/window, adjust=False).mean()
    
    # Đánh dấu chỗ tr_ema là 0 để tránh chia cho 0
    zero_mask = tr_ema == 0
    # Thay thế giá trị 0 bằng NaN để tránh lỗi khi chia
    tr_ema_safe = tr_ema.copy()
    tr_ema_safe[zero_mask] = np.nan
    
    # Tính +DI và -DI
    plus_di = 100 * plus_dm_ema / tr_ema_safe
    minus_di = 100 * minus_dm_ema / tr_ema_safe
    
    # Tính DX (Directional Index)
    dx_numerator = np.abs(plus_di - minus_di)
    dx_denominator = plus_di + minus_di
    
    # Đánh dấu chỗ dx_denominator là 0 để tránh chia cho 0
    zero_mask = dx_denominator == 0
    # Thay thế giá trị 0 bằng NaN để tránh lỗi khi chia
    dx_denominator_safe = dx_denominator.copy()
    dx_denominator_safe[zero_mask] = np.nan
    
    dx = 100 * dx_numerator / dx_denominator_safe
    
    # Tính ADX (Average Directional Index)
    adx = dx.ewm(alpha=1/smooth_period, adjust=False).mean()
    
    # Xử lý NaN
    adx = adx.fillna(20)  # 20 là giá trị ADX trung bình thể hiện không có xu hướng rõ ràng
    plus_di = plus_di.fillna(25)  # 25 là giá trị mặc định cho +DI
    minus_di = minus_di.fillna(25)  # 25 là giá trị mặc định cho -DI
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}adx_{window}"] = adx
    result_df[f"{prefix}plus_di_{window}"] = plus_di
    result_df[f"{prefix}minus_di_{window}"] = minus_di
    
    return result_df

# Hàm trợ giúp cho Parabolic SAR - tối ưu bằng numba
@nb.jit(nopython=True)
def _calculate_psar_numba(high, low, close, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    Tính Parabolic SAR sử dụng numba để tối ưu hóa hiệu suất.
    
    Args:
        high: Mảng giá cao
        low: Mảng giá thấp
        close: Mảng giá đóng cửa
        af_start: Hệ số tăng tốc ban đầu
        af_step: Bước tăng của hệ số tăng tốc
        af_max: Giá trị tối đa của hệ số tăng tốc
        
    Returns:
        Tuple (sar, trend) - mảng giá trị SAR và mảng xu hướng
    """
    n = len(high)
    sar = np.zeros(n)
    trend = np.zeros(n)
    extreme_point = np.zeros(n)
    acceleration_factor = np.zeros(n)
    
    # Khởi tạo cho phiên đầu tiên
    trend[0] = 1  # Giả sử xu hướng tăng ban đầu
    sar[0] = low[0]  # SAR ban đầu ở dưới giá thấp
    extreme_point[0] = high[0]  # EP ban đầu là giá cao
    acceleration_factor[0] = af_start
    
    # Vòng lặp chính
    for i in range(1, n):
        # Nếu xu hướng tăng
        if trend[i-1] == 1:
            # SAR = SAR trước + AF * (EP - SAR trước)
            sar[i] = sar[i-1] + acceleration_factor[i-1] * (extreme_point[i-1] - sar[i-1])
            
            # Giới hạn SAR không được cao hơn low của 2 phiên trước
            if i >= 2:
                sar[i] = min(sar[i], low[i-1], low[i-2])
            
            # Nếu giá thấp phá vỡ SAR, đảo chiều xu hướng
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = extreme_point[i-1]
                extreme_point[i] = low[i]
                acceleration_factor[i] = af_start
            else:
                # Tiếp tục xu hướng tăng
                trend[i] = 1
                
                # Cập nhật EP và AF
                if high[i] > extreme_point[i-1]:
                    extreme_point[i] = high[i]
                    acceleration_factor[i] = min(acceleration_factor[i-1] + af_step, af_max)
                else:
                    extreme_point[i] = extreme_point[i-1]
                    acceleration_factor[i] = acceleration_factor[i-1]
        
        # Nếu xu hướng giảm
        else:
            # SAR = SAR trước + AF * (EP - SAR trước)
            sar[i] = sar[i-1] + acceleration_factor[i-1] * (extreme_point[i-1] - sar[i-1])
            
            # Giới hạn SAR không được thấp hơn high của 2 phiên trước
            if i >= 2:
                sar[i] = max(sar[i], high[i-1], high[i-2])
            
            # Nếu giá cao phá vỡ SAR, đảo chiều xu hướng
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = extreme_point[i-1]
                extreme_point[i] = high[i]
                acceleration_factor[i] = af_start
            else:
                # Tiếp tục xu hướng giảm
                trend[i] = -1
                
                # Cập nhật EP và AF
                if low[i] < extreme_point[i-1]:
                    extreme_point[i] = low[i]
                    acceleration_factor[i] = min(acceleration_factor[i-1] + af_step, af_max)
                else:
                    extreme_point[i] = extreme_point[i-1]
                    acceleration_factor[i] = acceleration_factor[i-1]
    
    return sar, trend

def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Parabolic SAR (Stop and Reverse).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        af_start: Hệ số tăng tốc ban đầu
        af_step: Bước tăng của hệ số tăng tốc
        af_max: Giá trị tối đa của hệ số tăng tốc
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa giá trị SAR
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    try:
        # Tính SAR sử dụng numba để tối ưu hiệu suất
        sar, trend = _calculate_psar_numba(
            result_df['high'].values,
            result_df['low'].values,
            result_df['close'].values,
            af_start, af_step, af_max
        )
        
        # Đặt tên cột kết quả
        result_df[f"{prefix}psar"] = sar
        result_df[f"{prefix}psar_trend"] = trend
    
    except Exception as e:
        # Fallback nếu numba không khả dụng
        print(f"Không thể sử dụng numba, sử dụng phương pháp truyền thống: {e}")
        
        # Khởi tạo các mảng
        high = result_df['high'].values
        low = result_df['low'].values
        close = result_df['close'].values
        
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)
        extreme_point = np.zeros(n)
        acceleration_factor = np.zeros(n)
        
        # Khởi tạo cho phiên đầu tiên
        trend[0] = 1  # Giả sử xu hướng tăng ban đầu
        sar[0] = low[0]  # SAR ban đầu ở dưới giá thấp
        extreme_point[0] = high[0]  # EP ban đầu là giá cao
        acceleration_factor[0] = af_start
        
        # Vòng lặp chính
        for i in range(1, n):
            # Nếu xu hướng tăng
            if trend[i-1] == 1:
                # SAR = SAR trước + AF * (EP - SAR trước)
                sar[i] = sar[i-1] + acceleration_factor[i-1] * (extreme_point[i-1] - sar[i-1])
                
                # Giới hạn SAR không được cao hơn low của 2 phiên trước
                if i >= 2:
                    sar[i] = min(sar[i], low[i-1], low[i-2])
                
                # Nếu giá thấp phá vỡ SAR, đảo chiều xu hướng
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = extreme_point[i-1]
                    extreme_point[i] = low[i]
                    acceleration_factor[i] = af_start
                else:
                    # Tiếp tục xu hướng tăng
                    trend[i] = 1
                    
                    # Cập nhật EP và AF
                    if high[i] > extreme_point[i-1]:
                        extreme_point[i] = high[i]
                        acceleration_factor[i] = min(acceleration_factor[i-1] + af_step, af_max)
                    else:
                        extreme_point[i] = extreme_point[i-1]
                        acceleration_factor[i] = acceleration_factor[i-1]
            
            # Nếu xu hướng giảm
            else:
                # SAR = SAR trước + AF * (EP - SAR trước)
                sar[i] = sar[i-1] + acceleration_factor[i-1] * (extreme_point[i-1] - sar[i-1])
                
                # Giới hạn SAR không được thấp hơn high của 2 phiên trước
                if i >= 2:
                    sar[i] = max(sar[i], high[i-1], high[i-2])
                
                # Nếu giá cao phá vỡ SAR, đảo chiều xu hướng
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = extreme_point[i-1]
                    extreme_point[i] = high[i]
                    acceleration_factor[i] = af_start
                else:
                    # Tiếp tục xu hướng giảm
                    trend[i] = -1
                    
                    # Cập nhật EP và AF
                    if low[i] < extreme_point[i-1]:
                        extreme_point[i] = low[i]
                        acceleration_factor[i] = min(acceleration_factor[i-1] + af_step, af_max)
                    else:
                        extreme_point[i] = extreme_point[i-1]
                        acceleration_factor[i] = acceleration_factor[i-1]
        
        # Đặt tên cột kết quả
        result_df[f"{prefix}psar"] = sar
        result_df[f"{prefix}psar_trend"] = trend
    
    return result_df

def ichimoku_cloud(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    chikou_period: int = 26,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Ichimoku Cloud.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        tenkan_period: Kích thước cửa sổ cho Tenkan-sen (Conversion Line)
        kijun_period: Kích thước cửa sổ cho Kijun-sen (Base Line)
        senkou_b_period: Kích thước cửa sổ cho Senkou Span B
        chikou_period: Kích thước cửa sổ cho Chikou Span
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa các thành phần của Ichimoku Cloud
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tối ưu: Sử dụng numpy để tính toán các giá trị cao nhất và thấp nhất
    high_values = result_df['high'].values
    low_values = result_df['low'].values
    
    # Tính Tenkan-sen (Conversion Line)
    tenkan_high = np.zeros(len(high_values))
    tenkan_low = np.zeros(len(low_values))
    
    for i in range(tenkan_period - 1, len(high_values)):
        tenkan_high[i] = np.max(high_values[i-tenkan_period+1:i+1])
        tenkan_low[i] = np.min(low_values[i-tenkan_period+1:i+1])
    
    tenkan_sen = (pd.Series(tenkan_high, index=result_df.index) + 
                  pd.Series(tenkan_low, index=result_df.index)) / 2
    
    # Tính Kijun-sen (Base Line)
    kijun_high = np.zeros(len(high_values))
    kijun_low = np.zeros(len(low_values))
    
    for i in range(kijun_period - 1, len(high_values)):
        kijun_high[i] = np.max(high_values[i-kijun_period+1:i+1])
        kijun_low[i] = np.min(low_values[i-kijun_period+1:i+1])
    
    kijun_sen = (pd.Series(kijun_high, index=result_df.index) + 
                 pd.Series(kijun_low, index=result_df.index)) / 2
    
    # Tính Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Tính Senkou Span B (Leading Span B)
    senkou_b_high = np.zeros(len(high_values))
    senkou_b_low = np.zeros(len(low_values))
    
    for i in range(senkou_b_period - 1, len(high_values)):
        senkou_b_high[i] = np.max(high_values[i-senkou_b_period+1:i+1])
        senkou_b_low[i] = np.min(low_values[i-senkou_b_period+1:i+1])
    
    senkou_span_b = ((pd.Series(senkou_b_high, index=result_df.index) + 
                      pd.Series(senkou_b_low, index=result_df.index)) / 2).shift(kijun_period)
    
    # Tính Chikou Span (Lagging Span)
    chikou_span = result_df['close'].shift(-chikou_period)
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}ichimoku_tenkan_sen"] = tenkan_sen
    result_df[f"{prefix}ichimoku_kijun_sen"] = kijun_sen
    result_df[f"{prefix}ichimoku_senkou_span_a"] = senkou_span_a
    result_df[f"{prefix}ichimoku_senkou_span_b"] = senkou_span_b
    result_df[f"{prefix}ichimoku_chikou_span"] = chikou_span
    
    # Tính thêm cloud thickness (độ dày của đám mây) cho phân tích xu hướng
    cloud_thickness = senkou_span_a - senkou_span_b
    result_df[f"{prefix}ichimoku_cloud_thickness"] = cloud_thickness
    
    # Tính cloud direction (hướng của đám mây) cho phân tích động lượng
    cloud_direction = cloud_thickness - cloud_thickness.shift(1)
    result_df[f"{prefix}ichimoku_cloud_direction"] = cloud_direction

    return result_df

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    column: str = 'close',
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính chỉ báo SuperTrend.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        period: Kích thước cửa sổ cho ATR
        multiplier: Hệ số nhân cho ATR
        column: Tên cột giá sử dụng để tính toán (mặc định là 'close')
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa giá trị SuperTrend
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính True Range
    tr = true_range(result_df['high'], result_df['low'], result_df['close'])
    
    # Tính ATR
    atr = tr.rolling(window=period).mean()
    
    # Tính các dải băng trên và dưới
    upper_band = ((result_df['high'] + result_df['low']) / 2) + (multiplier * atr)
    lower_band = ((result_df['high'] + result_df['low']) / 2) - (multiplier * atr)
    
    # Tính SuperTrend
    supertrend = pd.Series(np.zeros(len(result_df)), index=result_df.index)
    trend = pd.Series(np.zeros(len(result_df)), index=result_df.index)
    
    # Khởi tạo giá trị đầu tiên
    supertrend.iloc[period-1] = lower_band.iloc[period-1]
    trend.iloc[period-1] = 1  # 1 là xu hướng tăng, -1 là xu hướng giảm
    
    # Tính toán SuperTrend cho các nến tiếp theo
    for i in range(period, len(result_df)):
        # Xu hướng trước đó là tăng
        if trend.iloc[i-1] == 1:
            # Cập nhật SuperTrend
            if result_df['close'].iloc[i] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = -1
            else:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                trend.iloc[i] = 1
        # Xu hướng trước đó là giảm
        else:
            # Cập nhật SuperTrend
            if result_df['close'].iloc[i] >= supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                trend.iloc[i] = 1
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                trend.iloc[i] = -1
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}supertrend_{period}_{multiplier}"] = supertrend
    result_df[f"{prefix}supertrend_trend_{period}_{multiplier}"] = trend
    
    return result_df

def standardize_macd(
    df: pd.DataFrame,
    macd_col: str = 'macd_line',
    signal_col: str = 'macd_signal',
    histogram_col: str = 'macd_histogram',
    window: int = 100,
    method: str = 'zscore',
    prefix: str = ''
) -> pd.DataFrame:
    """
    Chuẩn hóa các thành phần của MACD.
    
    Args:
        df: DataFrame chứa dữ liệu MACD
        macd_col: Tên cột MACD line
        signal_col: Tên cột MACD signal
        histogram_col: Tên cột MACD histogram
        window: Cửa sổ dùng cho chuẩn hóa (nếu sử dụng chuẩn hóa động)
        method: Phương pháp chuẩn hóa ('zscore', 'minmax', 'static')
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa các giá trị MACD đã chuẩn hóa
    """
    result_df = df.copy()
    
    # Danh sách các cột cần chuẩn hóa
    macd_columns = [col for col in [macd_col, signal_col, histogram_col] if col in result_df.columns]
    
    for col in macd_columns:
        if method == 'zscore':
            # Z-score normalization
            if window is None or window <= 0:
                # Sử dụng toàn bộ dữ liệu
                mean = result_df[col].mean()
                std = result_df[col].std()
                if std > 0:  # Tránh chia cho 0
                    result_df[f"{prefix}{col}_norm"] = (result_df[col] - mean) / std
                else:
                    result_df[f"{prefix}{col}_norm"] = 0
            else:
                # Sử dụng rolling window
                mean = result_df[col].rolling(window=window).mean()
                std = result_df[col].rolling(window=window).std()
                # Tránh chia cho 0
                std_non_zero = std.replace(0, np.nan)
                result_df[f"{prefix}{col}_norm"] = (result_df[col] - mean) / std_non_zero
                result_df[f"{prefix}{col}_norm"] = result_df[f"{prefix}{col}_norm"].fillna(0)
        
        elif method == 'minmax':
            # Min-max normalization
            if window is None or window <= 0:
                # Sử dụng toàn bộ dữ liệu
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                range_val = max_val - min_val
                if range_val > 0:  # Tránh chia cho 0
                    result_df[f"{prefix}{col}_norm"] = (result_df[col] - min_val) / range_val
                else:
                    result_df[f"{prefix}{col}_norm"] = 0.5  # Giá trị trung bình nếu không có biến thiên
            else:
                # Sử dụng rolling window
                min_val = result_df[col].rolling(window=window).min()
                max_val = result_df[col].rolling(window=window).max()
                range_val = max_val - min_val
                # Tránh chia cho 0
                range_non_zero = range_val.replace(0, np.nan)
                result_df[f"{prefix}{col}_norm"] = (result_df[col] - min_val) / range_non_zero
                result_df[f"{prefix}{col}_norm"] = result_df[f"{prefix}{col}_norm"].fillna(0.5)
        
        elif method == 'static':
            # Chuẩn hóa dùng biên cố định (-2, 2) là giá trị thường gặp của MACD
            result_df[f"{prefix}{col}_norm"] = (result_df[col] / 2).clip(-1, 1) * 0.5 + 0.5

        # Thêm đoạn code sau:
        elif method == 'robust':
            # Chuẩn hóa dùng median và MAD (Median Absolute Deviation)
            # Phù hợp hơn cho dữ liệu có ngoại lệ
            for col in macd_columns:
                median = result_df[col].median()
                # Sử dụng numpy để tránh phụ thuộc vào scipy
                mad = np.median(np.abs(result_df[col] - median))
                # Điều chỉnh MAD để tương đương với độ lệch chuẩn khi phân phối chuẩn
                mad_adjusted = mad * 1.4826  # Hằng số chuẩn hóa
                
                if mad_adjusted > 1e-8:  # Tránh chia cho giá trị quá nhỏ
                    z_scores = (result_df[col] - median) / mad_adjusted
                    # Giới hạn z-scores trong khoảng [-3, 3]
                    z_scores_clipped = np.clip(z_scores, -3, 3)
                    # Chuyển từ [-3, 3] sang [0, 1]
                    result_df[f"{prefix}{col}_norm"] = (z_scores_clipped + 3) / 6
                else:
                    # Nếu MAD quá nhỏ, đặt tất cả giá trị về 0.5
                    result_df[f"{prefix}{col}_norm"] = 0.5    
                    
    return result_df