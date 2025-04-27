"""
Các chỉ báo hỗ trợ và kháng cự.
File này cung cấp các hàm để phát hiện các mức hỗ trợ và kháng cự,
pivot points, và Fibonacci retracement.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy.signal import argrelextrema

# Import các hàm tiện ích
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, find_local_extrema, true_range
)

def detect_support_resistance(
    df: pd.DataFrame,
    price_column: str = 'close',
    window: int = 10,
    min_touches: int = 2,
    tolerance: float = 0.02,
    adaptive_tolerance: bool = True,
    tolerance_atr_multiplier: float = 0.2,
    atr_window: int = 14,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phát hiện các mức hỗ trợ và kháng cự.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ để tìm cực đại/cực tiểu cục bộ
        min_touches: Số lần tiếp xúc tối thiểu để xác định mức hỗ trợ/kháng cự
        tolerance: Dung sai phần trăm cố định để nhóm các mức gần nhau
        adaptive_tolerance: Sử dụng dung sai thích ứng dựa trên ATR
        tolerance_atr_multiplier: Hệ số nhân ATR cho dung sai thích ứng
        atr_window: Kích thước cửa sổ cho ATR
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa thông tin hỗ trợ và kháng cự
    """
    if not validate_price_data(df, [price_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
    
    result_df = df.copy()
    
    # Nếu sử dụng dung sai thích ứng, tính ATR
    if adaptive_tolerance:
        if all(col in result_df.columns for col in ['high', 'low', 'close']):
            atr = true_range(result_df['high'], result_df['low'], result_df['close']).rolling(window=atr_window).mean()
            # Tính dung sai dựa trên ATR và giá hiện tại
            current_price = result_df[price_column].iloc[-1]
            # Sử dụng ATR trung bình trên toàn bộ dữ liệu
            avg_atr = atr.mean()
            # Tính dung sai thích ứng = ATR * multiplier / current_price
            adaptive_tol = avg_atr * tolerance_atr_multiplier / current_price
            # Đảm bảo dung sai hợp lý (không quá nhỏ, không quá lớn)
            tolerance = min(max(adaptive_tol, 0.005), 0.05)
            
            # Lưu dung sai đã sử dụng
            result_df.attrs[f"{prefix}sr_tolerance_used"] = tolerance
        else:
            # Nếu không có dữ liệu OHLC đầy đủ, quay lại dung sai cố định
            adaptive_tolerance = False
    
    # Tìm các cực đại và cực tiểu cục bộ sử dụng method robust với window
    price_data = result_df[price_column]
    peaks, troughs = find_local_extrema(price_data, window=window, method='robust')
    
    # Lấy giá tại các điểm cực đại và cực tiểu
    peak_prices = price_data[peaks].values
    trough_prices = price_data[troughs].values
    
    # Tối ưu: Sử dụng numpy để nhóm các mức gần nhau
    def cluster_levels_numpy(levels, tolerance, min_touches):
        if len(levels) == 0:
            return []
        
        # Sắp xếp mảng
        sorted_levels = np.sort(levels)
        
        # Khởi tạo các mảng để theo dõi các cụm
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        # Duyệt qua các mức đã sắp xếp
        for i in range(1, len(sorted_levels)):
            level = sorted_levels[i]
            cluster_center = np.mean(current_cluster)
            
            # Nếu mức hiện tại gần với trung tâm cụm
            if level <= cluster_center * (1 + tolerance) and level >= cluster_center * (1 - tolerance):
                current_cluster.append(level)
            else:
                # Nếu cụm đủ lớn, thêm vào danh sách kết quả
                if len(current_cluster) >= min_touches:
                    clusters.append(np.mean(current_cluster))
                # Bắt đầu cụm mới
                current_cluster = [level]
        
        # Xử lý cụm cuối cùng
        if len(current_cluster) >= min_touches:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    # Nhóm các mức hỗ trợ và kháng cự
    resistance_levels = cluster_levels_numpy(peak_prices, tolerance, min_touches)
    support_levels = cluster_levels_numpy(trough_prices, tolerance, min_touches)
    
    # Sắp xếp các mức từ thấp đến cao
    resistance_levels = sorted(resistance_levels)
    support_levels = sorted(support_levels)
    
    # Tìm mức hỗ trợ và kháng cự hiện tại
    current_price = price_data.iloc[-1]
    
    # Tìm mức kháng cự tiếp theo
    next_resistance = None
    for level in resistance_levels:
        if level > current_price:
            next_resistance = level
            break
    
    # Tìm mức hỗ trợ tiếp theo
    next_support = None
    for level in reversed(support_levels):
        if level < current_price:
            next_support = level
            break
    
    # Tính khoảng cách đến mức hỗ trợ và kháng cự tiếp theo
    if next_resistance is not None:
        distance_to_resistance = (next_resistance - current_price) / current_price
        result_df[f"{prefix}next_resistance"] = next_resistance
        result_df[f"{prefix}distance_to_resistance_pct"] = distance_to_resistance * 100
    
    if next_support is not None:
        distance_to_support = (current_price - next_support) / current_price
        result_df[f"{prefix}next_support"] = next_support
        result_df[f"{prefix}distance_to_support_pct"] = distance_to_support * 100
    
    # Tính biên độ giữa hỗ trợ và kháng cự tiếp theo
    if next_resistance is not None and next_support is not None:
        sr_range = (next_resistance - next_support) / current_price
        result_df[f"{prefix}sr_range_pct"] = sr_range * 100
    
    # Đánh dấu các điểm cực đại và cực tiểu
    result_df[f"{prefix}is_peak"] = peaks
    result_df[f"{prefix}is_trough"] = troughs
    
    # Thêm danh sách các mức hỗ trợ và kháng cự được phát hiện
    result_df.attrs[f"{prefix}resistance_levels"] = resistance_levels
    result_df.attrs[f"{prefix}support_levels"] = support_levels
    
    return result_df

def pivot_points(
    df: pd.DataFrame,
    high_column: str = 'high',
    low_column: str = 'low',
    close_column: str = 'close',
    method: str = 'standard',  # 'standard', 'fibonacci', 'woodie', 'camarilla', 'demark'
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Pivot Points.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        high_column: Tên cột giá cao
        low_column: Tên cột giá thấp
        close_column: Tên cột giá đóng cửa
        method: Phương pháp tính ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa Pivot Points
    """
    required_columns = [high_column, low_column, close_column]
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Chuyển đổi thành numpy arrays để tăng hiệu suất
    high = result_df[high_column].values
    low = result_df[low_column].values
    close = result_df[close_column].values
    
    # Mảng kết quả
    pivot = np.zeros_like(high)
    s1 = np.zeros_like(high)
    s2 = np.zeros_like(high)
    s3 = np.zeros_like(high)
    r1 = np.zeros_like(high)
    r2 = np.zeros_like(high)
    r3 = np.zeros_like(high)
    
    # Phương pháp Standard
    if method == 'standard':
        # Pivot point (P) = (High + Low + Close) / 3
        pivot = (high + low + close) / 3
        
        # Support và Resistance levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Đặt tên các cột kết quả
        result_df[f"{prefix}pivot"] = pivot
        result_df[f"{prefix}support1"] = s1
        result_df[f"{prefix}support2"] = s2
        result_df[f"{prefix}support3"] = s3
        result_df[f"{prefix}resistance1"] = r1
        result_df[f"{prefix}resistance2"] = r2
        result_df[f"{prefix}resistance3"] = r3
    
    # Phương pháp Fibonacci
    elif method == 'fibonacci':
        # Pivot point (P) = (High + Low + Close) / 3
        pivot = (high + low + close) / 3
        
        # Support và Resistance levels với hệ số Fibonacci
        s1 = pivot - 0.382 * (high - low)
        s2 = pivot - 0.618 * (high - low)
        s3 = pivot - 1.000 * (high - low)
        
        r1 = pivot + 0.382 * (high - low)
        r2 = pivot + 0.618 * (high - low)
        r3 = pivot + 1.000 * (high - low)
        
        # Đặt tên các cột kết quả
        result_df[f"{prefix}pivot_fib"] = pivot
        result_df[f"{prefix}support1_fib"] = s1
        result_df[f"{prefix}support2_fib"] = s2
        result_df[f"{prefix}support3_fib"] = s3
        result_df[f"{prefix}resistance1_fib"] = r1
        result_df[f"{prefix}resistance2_fib"] = r2
        result_df[f"{prefix}resistance3_fib"] = r3
    
    # Phương pháp Woodie
    elif method == 'woodie':
        # Pivot point (P) = (High + Low + 2 * Close) / 4
        pivot = (high + low + 2 * close) / 4
        
        # Support và Resistance levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        
        # Đặt tên các cột kết quả
        result_df[f"{prefix}pivot_woodie"] = pivot
        result_df[f"{prefix}support1_woodie"] = s1
        result_df[f"{prefix}support2_woodie"] = s2
        result_df[f"{prefix}resistance1_woodie"] = r1
        result_df[f"{prefix}resistance2_woodie"] = r2
    
    # Phương pháp Camarilla
    elif method == 'camarilla':
        # Support và Resistance levels
        r4 = close + (high - low) * 1.5000
        r3 = close + (high - low) * 1.2500
        r2 = close + (high - low) * 1.1666
        r1 = close + (high - low) * 1.0833
        
        s1 = close - (high - low) * 1.0833
        s2 = close - (high - low) * 1.1666
        s3 = close - (high - low) * 1.2500
        s4 = close - (high - low) * 1.5000
        
        # Đặt tên các cột kết quả
        result_df[f"{prefix}resistance4_camarilla"] = r4
        result_df[f"{prefix}resistance3_camarilla"] = r3
        result_df[f"{prefix}resistance2_camarilla"] = r2
        result_df[f"{prefix}resistance1_camarilla"] = r1
        result_df[f"{prefix}support1_camarilla"] = s1
        result_df[f"{prefix}support2_camarilla"] = s2
        result_df[f"{prefix}support3_camarilla"] = s3
        result_df[f"{prefix}support4_camarilla"] = s4
    
    # Phương pháp DeMark
    elif method == 'demark':
        # Nếu Open có sẵn, sử dụng nó, nếu không giả định Open = Close của ngày trước
        if 'open' in result_df.columns:
            open_prices = result_df['open'].values
        else:
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]  # Giả định ngày đầu tiên
        
        # Tính X dựa trên mối quan hệ giữa Close và Open
        x = np.zeros_like(high)
        
        for i in range(len(close)):
            if close[i] < open_prices[i]:  # Ngày giảm
                x[i] = high[i] + 2 * low[i] + close[i]
            elif close[i] > open_prices[i]:  # Ngày tăng
                x[i] = 2 * high[i] + low[i] + close[i]
            else:  # Không đổi
                x[i] = high[i] + low[i] + 2 * close[i]
        
        # Pivot point
        pivot = x / 4
        
        # Support và Resistance levels
        r1 = x / 2 - low
        s1 = x / 2 - high
        
        # Đặt tên các cột kết quả
        result_df[f"{prefix}pivot_demark"] = pivot
        result_df[f"{prefix}support1_demark"] = s1
        result_df[f"{prefix}resistance1_demark"] = r1
    
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}. Phải là một trong 'standard', 'fibonacci', 'woodie', 'camarilla', 'demark'")
    
    return result_df

def fibonacci_retracement(
    df: pd.DataFrame,
    high_column: str = 'high',
    low_column: str = 'low',
    trend: str = 'auto',  # 'up', 'down', 'auto'
    window: int = 100,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Fibonacci Retracement.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        high_column: Tên cột giá cao
        low_column: Tên cột giá thấp
        trend: Xu hướng thị trường ('up', 'down', 'auto')
        window: Kích thước cửa sổ để tìm đỉnh và đáy
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa các mức Fibonacci Retracement
    """
    required_columns = [high_column, low_column]
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Lấy dữ liệu trong cửa sổ - sử dụng numpy để tăng hiệu suất
    high_window = result_df[high_column].tail(window).values
    low_window = result_df[low_column].tail(window).values
    
    # Tìm đỉnh và đáy trong cửa sổ
    swing_high = np.max(high_window)
    swing_low = np.min(low_window)
    
    # Tìm vị trí của đỉnh và đáy
    swing_high_idx = np.argmax(high_window)
    swing_low_idx = np.argmin(low_window)
    
    # Xác định xu hướng
    if trend == 'auto':
        # Tính SMA để xác định xu hướng
        if 'close' in result_df.columns:
            close_values = result_df['close'].values
            # Lấy giá trị gần nhất và giá trị cách đó một khoảng thời gian để xác định xu hướng
            if len(close_values) >= 20:
                trend = 'up' if close_values[-1] > close_values[-20] else 'down'
            else:
                # Nếu không đủ dữ liệu, dựa vào vị trí của đỉnh và đáy
                trend = 'up' if swing_high_idx > swing_low_idx else 'down'
        else:
            # Nếu không có cột close, dựa vào vị trí của đỉnh và đáy
            trend = 'up' if swing_high_idx > swing_low_idx else 'down'
    
    # Trong xu hướng tăng, retracement đi từ đáy đến đỉnh
    # Trong xu hướng giảm, retracement đi từ đỉnh đến đáy
    if trend == 'up':
        price_diff = swing_high - swing_low
        
        # Các mức Fibonacci Retracement cho xu hướng tăng
        retracement_0 = swing_low  # 0.0
        retracement_236 = swing_low + 0.236 * price_diff  # 23.6%
        retracement_382 = swing_low + 0.382 * price_diff  # 38.2%
        retracement_5 = swing_low + 0.5 * price_diff  # 50.0%
        retracement_618 = swing_low + 0.618 * price_diff  # 61.8%
        retracement_786 = swing_low + 0.786 * price_diff  # 78.6%
        retracement_1 = swing_high  # 100.0%
        
        # Các mức Fibonacci Extension
        extension_1618 = swing_high + 0.618 * price_diff  # 161.8%
        extension_2618 = swing_high + 1.618 * price_diff  # 261.8%
        
    else:  # trend == 'down'
        price_diff = swing_high - swing_low
        
        # Các mức Fibonacci Retracement cho xu hướng giảm
        retracement_0 = swing_high  # 0.0
        retracement_236 = swing_high - 0.236 * price_diff  # 23.6%
        retracement_382 = swing_high - 0.382 * price_diff  # 38.2%
        retracement_5 = swing_high - 0.5 * price_diff  # 50.0%
        retracement_618 = swing_high - 0.618 * price_diff  # 61.8%
        retracement_786 = swing_high - 0.786 * price_diff  # 78.6%
        retracement_1 = swing_low  # 100.0%
        
        # Các mức Fibonacci Extension
        extension_1618 = swing_low - 0.618 * price_diff  # 161.8%
        extension_2618 = swing_low - 1.618 * price_diff  # 261.8%
    
    # Đặt tên các cột kết quả
    result_df[f"{prefix}fib_trend"] = trend
    result_df[f"{prefix}fib_swing_high"] = swing_high
    result_df[f"{prefix}fib_swing_low"] = swing_low
    
    result_df[f"{prefix}fib_0"] = retracement_0
    result_df[f"{prefix}fib_236"] = retracement_236
    result_df[f"{prefix}fib_382"] = retracement_382
    result_df[f"{prefix}fib_5"] = retracement_5
    result_df[f"{prefix}fib_618"] = retracement_618
    result_df[f"{prefix}fib_786"] = retracement_786
    result_df[f"{prefix}fib_1"] = retracement_1
    
    result_df[f"{prefix}fib_ext_1618"] = extension_1618
    result_df[f"{prefix}fib_ext_2618"] = extension_2618
    
    return result_df

# Hàm con để phát hiện Double Top
def _detect_double_top(
    price_data: pd.Series, 
    peak_indices: np.ndarray, 
    trough_indices: np.ndarray, 
    last_window: int, 
    tolerance: float
) -> bool:
    """
    Phát hiện mẫu hình Double Top.
    
    Args:
        price_data: Series giá
        peak_indices: Chỉ số các đỉnh
        trough_indices: Chỉ số các đáy
        last_window: Kích thước cửa sổ xem xét
        tolerance: Dung sai cho việc so khớp
        
    Returns:
        True nếu tìm thấy mẫu hình, False nếu không
    """
    if len(peak_indices) < 2:
        return False
    
    # Chuyển dữ liệu sang numpy array để tăng hiệu suất
    price_array = price_data.iloc[-last_window:].values
    
    for i in range(len(peak_indices) - 1):
        idx1 = peak_indices[i]
        idx2 = peak_indices[i+1]
        
        # Khoảng cách giữa hai đỉnh phải đủ xa
        if idx2 - idx1 < last_window // 10:  # Ít nhất 10% kích thước cửa sổ
            continue
        
        price1 = price_array[idx1]
        price2 = price_array[idx2]
        
        # Kiểm tra nếu hai đỉnh có giá trị gần bằng nhau
        if abs(price1 - price2) / max(price1, price2) <= tolerance:
            # Tìm thung lũng giữa hai đỉnh
            for j in trough_indices:
                if idx1 < j < idx2:
                    valley_price = price_array[j]
                    # Nếu thung lũng đủ sâu
                    if (min(price1, price2) - valley_price) / min(price1, price2) >= 0.03:
                        return True
    
    return False

# Hàm con để phát hiện Double Bottom
def _detect_double_bottom(
    price_data: pd.Series, 
    peak_indices: np.ndarray, 
    trough_indices: np.ndarray,
    last_window: int, 
    tolerance: float
) -> bool:
    """
    Phát hiện mẫu hình Double Bottom.
    
    Args:
        price_data: Series giá
        peak_indices: Chỉ số các đỉnh
        trough_indices: Chỉ số các đáy
        last_window: Kích thước cửa sổ xem xét
        tolerance: Dung sai cho việc so khớp
        
    Returns:
        True nếu tìm thấy mẫu hình, False nếu không
    """
    if len(trough_indices) < 2:
        return False
    
    # Chuyển dữ liệu sang numpy array để tăng hiệu suất
    price_array = price_data.iloc[-last_window:].values
    
    for i in range(len(trough_indices) - 1):
        idx1 = trough_indices[i]
        idx2 = trough_indices[i+1]
        
        # Khoảng cách giữa hai đáy phải đủ xa
        if idx2 - idx1 < last_window // 10:  # Ít nhất 10% kích thước cửa sổ
            continue
        
        price1 = price_array[idx1]
        price2 = price_array[idx2]
        
        # Kiểm tra nếu hai đáy có giá trị gần bằng nhau
        if abs(price1 - price2) / min(price1, price2) <= tolerance:
            # Tìm đỉnh giữa hai đáy
            for j in peak_indices:
                if idx1 < j < idx2:
                    peak_price = price_array[j]
                    # Nếu đỉnh đủ cao
                    if (peak_price - max(price1, price2)) / max(price1, price2) >= 0.03:
                        return True
    
    return False

# Hàm con để phát hiện Head and Shoulders
def _detect_head_shoulders(
    price_data: pd.Series, 
    peak_indices: np.ndarray, 
    trough_indices: np.ndarray,
    last_window: int, 
    tolerance: float
) -> bool:
    """
    Phát hiện mẫu hình Head and Shoulders.
    
    Args:
        price_data: Series giá
        peak_indices: Chỉ số các đỉnh
        trough_indices: Chỉ số các đáy
        last_window: Kích thước cửa sổ xem xét
        tolerance: Dung sai cho việc so khớp
        
    Returns:
        True nếu tìm thấy mẫu hình, False nếu không
    """
    if len(peak_indices) < 3:
        return False
    
    # Chuyển dữ liệu sang numpy array để tăng hiệu suất
    price_array = price_data.iloc[-last_window:].values
    
    for i in range(len(peak_indices) - 2):
        left_shoulder_idx = peak_indices[i]
        head_idx = peak_indices[i+1]
        right_shoulder_idx = peak_indices[i+2]
        
        # Khoảng cách giữa các đỉnh phải hợp lý
        if head_idx - left_shoulder_idx < last_window // 20 or right_shoulder_idx - head_idx < last_window // 20:
            continue
        
        left_shoulder = price_array[left_shoulder_idx]
        head = price_array[head_idx]
        right_shoulder = price_array[right_shoulder_idx]
        
        # Kiểm tra nếu head cao hơn hai shoulder và hai shoulder gần bằng nhau
        if (head > left_shoulder and head > right_shoulder and 
            abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) <= tolerance):
            # Tìm các đáy giữa các đỉnh
            left_trough_idx = None
            right_trough_idx = None
            
            for j in trough_indices:
                if left_shoulder_idx < j < head_idx:
                    left_trough_idx = j
                elif head_idx < j < right_shoulder_idx:
                    right_trough_idx = j
            
            if left_trough_idx is not None and right_trough_idx is not None:
                left_trough = price_array[left_trough_idx]
                right_trough = price_array[right_trough_idx]
                
                # Kiểm tra nếu hai đáy gần bằng nhau (neckline ngang)
                if abs(left_trough - right_trough) / max(left_trough, right_trough) <= tolerance:
                    return True
    
    return False

# Hàm con để phát hiện Inverse Head and Shoulders
def _detect_inverse_head_shoulders(
    price_data: pd.Series, 
    peak_indices: np.ndarray, 
    trough_indices: np.ndarray,
    last_window: int, 
    tolerance: float
) -> bool:
    """
    Phát hiện mẫu hình Inverse Head and Shoulders.
    
    Args:
        price_data: Series giá
        peak_indices: Chỉ số các đỉnh
        trough_indices: Chỉ số các đáy
        last_window: Kích thước cửa sổ xem xét
        tolerance: Dung sai cho việc so khớp
        
    Returns:
        True nếu tìm thấy mẫu hình, False nếu không
    """
    if len(trough_indices) < 3:
        return False
    
    # Chuyển dữ liệu sang numpy array để tăng hiệu suất
    price_array = price_data.iloc[-last_window:].values
    
    for i in range(len(trough_indices) - 2):
        left_shoulder_idx = trough_indices[i]
        head_idx = trough_indices[i+1]
        right_shoulder_idx = trough_indices[i+2]
        
        # Khoảng cách giữa các đáy phải hợp lý
        if head_idx - left_shoulder_idx < last_window // 20 or right_shoulder_idx - head_idx < last_window // 20:
            continue
        
        left_shoulder = price_array[left_shoulder_idx]
        head = price_array[head_idx]
        right_shoulder = price_array[right_shoulder_idx]
        
        # Kiểm tra nếu head thấp hơn hai shoulder và hai shoulder gần bằng nhau
        if (head < left_shoulder and head < right_shoulder and 
            abs(left_shoulder - right_shoulder) / min(left_shoulder, right_shoulder) <= tolerance):
            # Tìm các đỉnh giữa các đáy
            left_peak_idx = None
            right_peak_idx = None
            
            for j in peak_indices:
                if left_shoulder_idx < j < head_idx:
                    left_peak_idx = j
                elif head_idx < j < right_shoulder_idx:
                    right_peak_idx = j
            
            if left_peak_idx is not None and right_peak_idx is not None:
                left_peak = price_array[left_peak_idx]
                right_peak = price_array[right_peak_idx]
                
                # Kiểm tra nếu hai đỉnh gần bằng nhau (neckline ngang)
                if abs(left_peak - right_peak) / max(left_peak, right_peak) <= tolerance:
                    return True
    
    return False

# Hàm con để phát hiện Triangle
def _detect_triangle(
    price_data: pd.Series, 
    peak_indices: np.ndarray, 
    trough_indices: np.ndarray,
    last_window: int
) -> bool:
    """
    Phát hiện mẫu hình Triangle.
    
    Args:
        price_data: Series giá
        peak_indices: Chỉ số các đỉnh
        trough_indices: Chỉ số các đáy
        last_window: Kích thước cửa sổ xem xét
        
    Returns:
        True nếu tìm thấy mẫu hình, False nếu không
    """
    if len(peak_indices) < 3 or len(trough_indices) < 3:
        return False
    
    # Chuyển dữ liệu sang numpy array để tăng hiệu suất
    price_array = price_data.iloc[-last_window:].values
    
    # Lấy 3 đỉnh và 3 đáy gần nhất
    recent_peaks = peak_indices[-3:]
    recent_troughs = trough_indices[-3:]
    
    # Lấy giá tại các đỉnh và đáy
    peak_prices = np.array([price_array[idx] for idx in recent_peaks])
    trough_prices = np.array([price_array[idx] for idx in recent_troughs])
    
    # Tính độ dốc của đường nối các đỉnh và các đáy
    if len(peak_prices) >= 2 and len(trough_prices) >= 2:
        # Kiểm tra dạng tam giác hội tụ (Convergent Triangle)
        if peak_prices[0] > peak_prices[-1] and trough_prices[0] < trough_prices[-1]:
            return True
        
        # Kiểm tra dạng tam giác tăng (Ascending Triangle)
        if abs(peak_prices[0] - peak_prices[-1]) / peak_prices[0] < 0.02 and trough_prices[0] < trough_prices[-1]:
            return True
        
        # Kiểm tra dạng tam giác giảm (Descending Triangle)
        if peak_prices[0] > peak_prices[-1] and abs(trough_prices[0] - trough_prices[-1]) / trough_prices[0] < 0.02:
            return True
    
    return False

def find_chart_patterns(
    df: pd.DataFrame,
    price_column: str = 'close',
    window: int = 50,
    tolerance: float = 0.03,
    adaptive_tolerance: bool = True,
    tolerance_atr_multiplier: float = 0.2,
    atr_window: int = 14,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phát hiện các mẫu hình phổ biến trên đồ thị.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ để tìm mẫu hình
        tolerance: Dung sai phần trăm cho việc so khớp mẫu hình
        adaptive_tolerance: Sử dụng dung sai thích ứng dựa trên ATR
        tolerance_atr_multiplier: Hệ số nhân ATR cho dung sai thích ứng
        atr_window: Kích thước cửa sổ cho ATR
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa thông tin mẫu hình
    """
    if not validate_price_data(df, [price_column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
    
    result_df = df.copy()
    price_data = result_df[price_column]
    
    # Nếu sử dụng dung sai thích ứng, tính ATR
    if adaptive_tolerance:
        if all(col in result_df.columns for col in ['high', 'low', 'close']):
            atr = true_range(result_df['high'], result_df['low'], result_df['close']).rolling(window=atr_window).mean()
            # Tính dung sai dựa trên ATR và giá hiện tại
            current_price = price_data.iloc[-1]
            # Sử dụng ATR trung bình trên toàn bộ dữ liệu
            avg_atr = atr.mean()
            # Tính dung sai thích ứng = ATR * multiplier / current_price
            adaptive_tol = avg_atr * tolerance_atr_multiplier / current_price
            # Đảm bảo dung sai hợp lý (không quá nhỏ, không quá lớn)
            tolerance = min(max(adaptive_tol, 0.01), 0.05)
            
            # Lưu dung sai đã sử dụng
            result_df.attrs[f"{prefix}pattern_tolerance_used"] = tolerance
        else:
            # Nếu không có dữ liệu OHLC đầy đủ, quay lại dung sai cố định
            adaptive_tolerance = False
    
    # Tìm các cực đại và cực tiểu cục bộ
    peaks, troughs = find_local_extrema(price_data, window=window//5, method='robust')
    
    # Tạo các cột kết quả
    result_df[f"{prefix}double_top"] = False
    result_df[f"{prefix}double_bottom"] = False
    result_df[f"{prefix}head_shoulders"] = False
    result_df[f"{prefix}inv_head_shoulders"] = False
    result_df[f"{prefix}triangle"] = False
    
    # Chỉ xét trong cửa sổ cuối cùng
    last_window = min(window, len(result_df))
    
    # Chỉ số các cực đại và cực tiểu trong cửa sổ - sử dụng numpy để tăng hiệu suất
    peak_indices = np.where(peaks.iloc[-last_window:].values)[0]
    trough_indices = np.where(troughs.iloc[-last_window:].values)[0]
    
    # Phát hiện các mẫu hình - sử dụng các hàm con
    has_double_top = _detect_double_top(
        price_data, peak_indices, trough_indices, last_window, tolerance
    )
    
    has_double_bottom = _detect_double_bottom(
        price_data, peak_indices, trough_indices, last_window, tolerance
    )
    
    has_head_shoulders = _detect_head_shoulders(
        price_data, peak_indices, trough_indices, last_window, tolerance
    )
    
    has_inv_head_shoulders = _detect_inverse_head_shoulders(
        price_data, peak_indices, trough_indices, last_window, tolerance
    )
    
    has_triangle = _detect_triangle(
        price_data, peak_indices, trough_indices, last_window
    )
    
    # Cập nhật kết quả
    if has_double_top:
        result_df.iloc[-1, result_df.columns.get_loc(f"{prefix}double_top")] = True
    
    if has_double_bottom:
        result_df.iloc[-1, result_df.columns.get_loc(f"{prefix}double_bottom")] = True
    
    if has_head_shoulders:
        result_df.iloc[-1, result_df.columns.get_loc(f"{prefix}head_shoulders")] = True
    
    if has_inv_head_shoulders:
        result_df.iloc[-1, result_df.columns.get_loc(f"{prefix}inv_head_shoulders")] = True
    
    if has_triangle:
        result_df.iloc[-1, result_df.columns.get_loc(f"{prefix}triangle")] = True
    
    # Thêm chỉ báo pattern_found nếu bất kỳ mẫu hình nào được tìm thấy
    result_df[f"{prefix}pattern_found"] = (
        result_df[f"{prefix}double_top"] |
        result_df[f"{prefix}double_bottom"] |
        result_df[f"{prefix}head_shoulders"] |
        result_df[f"{prefix}inv_head_shoulders"] |
        result_df[f"{prefix}triangle"]
    )
    
    return result_df