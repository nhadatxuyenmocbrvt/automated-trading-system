"""
Tiện ích cho các chỉ báo kỹ thuật.
File này cung cấp các hàm tiện ích được sử dụng bởi các module chỉ báo kỹ thuật khác.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any, Callable

def validate_price_data(df: pd.DataFrame, columns: List[str] = None) -> bool:
    """
    Xác thực dữ liệu giá để đảm bảo nó có thể được sử dụng cho các chỉ báo kỹ thuật.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        columns: Danh sách các cột giá cần kiểm tra
        
    Returns:
        True nếu dữ liệu hợp lệ, False nếu không
    """
    if df is None or df.empty:
        return False
    
    if columns is None:
        # Kiểm tra các cột OHLCV tiêu chuẩn
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                return False
    else:
        # Kiểm tra các cột được chỉ định
        if not all(col in df.columns for col in columns):
            return False
    
    return True

def get_candle_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Xác định các cột OHLCV trong DataFrame.
    
    Args:
        df: DataFrame chứa dữ liệu nến
        
    Returns:
        Dict với mapping tên cột chuẩn hóa
    """
    column_mapping = {}
    
    # Các tên phổ biến cho các cột OHLCV
    ohlcv_patterns = {
        'open': ['open', 'Open', 'OPEN', 'open_price', 'Open_Price'],
        'high': ['high', 'High', 'HIGH', 'high_price', 'High_Price'],
        'low': ['low', 'Low', 'LOW', 'low_price', 'Low_Price'],
        'close': ['close', 'Close', 'CLOSE', 'close_price', 'Close_Price'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
    }
    
    # Tìm các cột tương ứng
    for std_name, patterns in ohlcv_patterns.items():
        for pattern in patterns:
            if pattern in df.columns:
                column_mapping[std_name] = pattern
                break
                
    return column_mapping

def prepare_price_data(
    df: pd.DataFrame, 
    use_adjusted: bool = False, 
    normalize_names: bool = True
) -> pd.DataFrame:
    """
    Chuẩn bị dữ liệu giá để tính toán các chỉ báo kỹ thuật.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        use_adjusted: Sử dụng giá đã điều chỉnh nếu có
        normalize_names: Chuẩn hóa tên cột
        
    Returns:
        DataFrame đã được chuẩn bị
    """
    result_df = df.copy()
    
    # Đảm bảo dữ liệu được sắp xếp theo thời gian
    if 'timestamp' in result_df.columns:
        result_df = result_df.sort_values('timestamp')
    elif 'time' in result_df.columns:
        result_df = result_df.sort_values('time')
    elif 'date' in result_df.columns:
        result_df = result_df.sort_values('date')
    
    # Sử dụng giá đã điều chỉnh nếu được yêu cầu và có sẵn
    if use_adjusted and 'adj_close' in result_df.columns:
        # Điều chỉnh tỷ lệ cho open, high, low dựa trên tỷ lệ close/adj_close
        ratio = result_df['adj_close'] / result_df['close']
        
        if 'open' in result_df.columns:
            result_df['open'] = result_df['open'] * ratio
        
        if 'high' in result_df.columns:
            result_df['high'] = result_df['high'] * ratio
        
        if 'low' in result_df.columns:
            result_df['low'] = result_df['low'] * ratio
        
        # Thay thế close bằng adj_close
        result_df['close'] = result_df['adj_close']
    
    # Chuẩn hóa tên cột nếu cần
    if normalize_names:
        column_mapping = get_candle_columns(result_df)
        if column_mapping:
            # Đổi tên cột
            rename_dict = {v: k for k, v in column_mapping.items()}
            result_df = result_df.rename(columns=rename_dict)
    
    return result_df

def exponential_weights(window_size: int, alpha: Optional[float] = None) -> np.ndarray:
    """
    Tạo mảng trọng số mũ cho các phép tính EMA.
    
    Args:
        window_size: Kích thước cửa sổ
        alpha: Tham số làm mượt, nếu None sẽ sử dụng 2/(window_size+1)
        
    Returns:
        Mảng trọng số
    """
    if alpha is None:
        alpha = 2 / (window_size + 1)
    
    # Trọng số giảm dần theo cấp số nhân
    weights = np.array([(1 - alpha) ** i for i in range(window_size)])
    
    # Đảo ngược mảng để trọng số gần nhất có giá trị lớn nhất
    weights = weights[::-1]
    
    # Chuẩn hóa trọng số
    weights /= weights.sum()
    
    return weights

def calculate_weighted_average(
    series: pd.Series, 
    weights: np.ndarray, 
    min_periods: int = 1
) -> pd.Series:
    """
    Tính trung bình trọng số cho Series.
    
    Args:
        series: Series dữ liệu
        weights: Mảng trọng số
        min_periods: Số lượng điểm dữ liệu tối thiểu cần thiết
        
    Returns:
        Series chứa kết quả trung bình trọng số
    """
    window_size = len(weights)
    
    # Sử dụng rolling window với tùy chỉnh
    def weighted_avg(window):
        # Kiểm tra số lượng giá trị không NA
        if window.count() < min_periods:
            return np.nan
        
        # Tính trung bình trọng số
        w = weights[-len(window):]
        return (window.fillna(0) * w).sum() / w.sum()
    
    result = series.rolling(window=window_size, min_periods=min_periods).apply(weighted_avg, raw=False)
    
    return result

def find_local_extrema(
    series: pd.Series, 
    window: int = 5,
    method: str = 'simple'  # 'simple' hoặc 'robust'
) -> Tuple[pd.Series, pd.Series]:
    """
    Tìm các cực đại và cực tiểu cục bộ trong Series.
    
    Args:
        series: Series dữ liệu
        window: Kích thước cửa sổ để xem xét các cực trị
        method: Phương pháp phát hiện ('simple' hoặc 'robust')
        
    Returns:
        Tuple (peaks, troughs) với peaks là Series đánh dấu các cực đại,
        troughs là Series đánh dấu các cực tiểu
    """
    if method == 'simple':
        # Phương pháp đơn giản: so sánh với giá trị trước và sau
        peaks = ((series > series.shift(1)) & 
                 (series > series.shift(-1)))
        
        troughs = ((series < series.shift(1)) & 
                   (series < series.shift(-1)))
    else:
        # Phương pháp robust: so sánh trong cửa sổ
        peaks = pd.Series(False, index=series.index)
        troughs = pd.Series(False, index=series.index)
        
        half_window = window // 2
        
        for i in range(half_window, len(series) - half_window):
            window_data = series.iloc[i-half_window:i+half_window+1]
            if series.iloc[i] == window_data.max():
                peaks.iloc[i] = True
            if series.iloc[i] == window_data.min():
                troughs.iloc[i] = True
    
    return peaks, troughs

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Tính True Range.
    
    Args:
        high: Series giá cao
        low: Series giá thấp
        close: Series giá đóng cửa
        
    Returns:
        Series chứa giá trị True Range
    """
    # Lấy giá đóng cửa trước đó
    prev_close = close.shift(1)
    
    # Tính ba thành phần của True Range
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    # True Range là giá trị lớn nhất trong ba thành phần
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr

def normalize_indicator(
    series: pd.Series, 
    method: str = 'minmax',  # 'minmax', 'zscore', 'percent_rank'
    window: int = None,
    cache_extrema: bool = True    # Cache min/max để tối ưu hiệu suất
) -> pd.Series:
    """
    Chuẩn hóa giá trị chỉ báo.
    
    Args:
        series: Series giá trị chỉ báo
        method: Phương pháp chuẩn hóa
        window: Kích thước cửa sổ cho chuẩn hóa di động
        cache_extrema: Cache giá trị min/max để tối ưu hiệu suất (chỉ cho chuẩn hóa di động)
        
    Returns:
        Series đã chuẩn hóa
    """
    if window is None:
        # Chuẩn hóa toàn bộ Series
        if method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                return (series - min_val) / (max_val - min_val)
            else:
                return pd.Series(0.5, index=series.index)
        
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            if std > 0:
                return (series - mean) / std
            else:
                return pd.Series(0, index=series.index)
        
        elif method == 'percent_rank':
            return series.rank(pct=True)
    
    else:
        # Chuẩn hóa di động
        if method == 'minmax':
            # Nếu dữ liệu lớn và window nhỏ (kích hoạt optimization)
            if cache_extrema and len(series) > 1000 and window < 100:
                # Tính min và max một lần rồi cache lại
                min_vals = series.rolling(window=window).min()
                max_vals = series.rolling(window=window).max()
                range_vals = max_vals - min_vals
                
                # Tránh chia cho 0
                mask = range_vals <= 0
                
                # Sử dụng numpy để tối ưu hóa
                normalized = np.zeros(len(series))
                valid_idx = ~mask & ~np.isnan(series) & ~np.isnan(range_vals)
                
                # Chỉ tính toán cho các giá trị hợp lệ
                normalized[valid_idx] = (
                    (series.values[valid_idx] - min_vals.values[valid_idx]) / 
                    range_vals.values[valid_idx]
                )
                
                # Giá trị mặc định cho các điểm không hợp lệ
                normalized[~valid_idx & ~np.isnan(series)] = 0.5
                
                result = pd.Series(normalized, index=series.index)
                return result
            else:
                # Phương pháp tiêu chuẩn
                min_vals = series.rolling(window=window).min()
                max_vals = series.rolling(window=window).max()
                range_vals = max_vals - min_vals
                # Tránh chia cho 0
                range_vals = range_vals.replace(0, np.nan)
                result = (series - min_vals) / range_vals
                # Fill NaN với 0.5 (giá trị trung bình)
                result = result.fillna(0.5)
                return result
        
        elif method == 'zscore':
            means = series.rolling(window=window).mean()
            stds = series.rolling(window=window).std()
            # Tránh chia cho 0
            stds = stds.replace(0, np.nan)
            result = (series - means) / stds
            # Fill NaN với 0 (giá trị trung bình)
            result = result.fillna(0)
            return result
        
        elif method == 'percent_rank':
            def rolling_rank(x):
                return pd.Series(x).rank(pct=True).iloc[-1]
            
            return series.rolling(window=window).apply(rolling_rank, raw=False)
    
    # Nếu phương pháp không được hỗ trợ, trả về Series gốc
    return series

def crossover(series1: pd.Series, series2: pd.Series, force: bool = False, threshold: float = 0.01) -> pd.Series:
    """
    Phát hiện giao cắt từ dưới lên.
    
    Args:
        series1: Series thứ nhất
        series2: Series thứ hai
        force: Nếu True, sẽ phát hiện giao cắt mạnh (với biên độ lớn)
        threshold: Ngưỡng phần trăm cho giao cắt mạnh (so với giá trị series2)
        
    Returns:
        Series boolean với True khi series1 cắt qua series2 từ dưới lên
    """
    previous = series1.shift(1) < series2.shift(1)
    
    if force:
        # Giao cắt mạnh: series1 vượt series2 một khoảng lớn hơn threshold%
        # Tính toán biên độ vượt
        crossover_margin = (series1 - series2) / series2
        current = crossover_margin >= threshold
        # Và cần thỏa mãn điều kiện trước đó series1 < series2
        crossover = previous & current
    else:
        # Giao cắt thông thường
        current = series1 >= series2
        crossover = previous & current
    
    return crossover

def crossunder(series1: pd.Series, series2: pd.Series, force: bool = False, threshold: float = 0.01) -> pd.Series:
    """
    Phát hiện giao cắt từ trên xuống.
    
    Args:
        series1: Series thứ nhất
        series2: Series thứ hai
        force: Nếu True, sẽ phát hiện giao cắt mạnh (với biên độ lớn)
        threshold: Ngưỡng phần trăm cho giao cắt mạnh (so với giá trị series2)
        
    Returns:
        Series boolean với True khi series1 cắt qua series2 từ trên xuống
    """
    previous = series1.shift(1) > series2.shift(1)
    
    if force:
        # Giao cắt mạnh: series1 giảm dưới series2 một khoảng lớn hơn threshold%
        # Tính toán biên độ giảm
        crossunder_margin = (series2 - series1) / series2
        current = crossunder_margin >= threshold
        # Và cần thỏa mãn điều kiện trước đó series1 > series2
        crossunder = previous & current
    else:
        # Giao cắt thông thường
        current = series1 <= series2
        crossunder = previous & current
    
    return crossunder

def shift_series(series: pd.Series, periods: int = 1, fill_value: Any = np.nan) -> pd.Series:
    """
    Dịch chuyển Series với giá trị điền tùy chỉnh.
    
    Args:
        series: Series cần dịch chuyển
        periods: Số kỳ dịch chuyển (dương để dịch về tương lai, âm để dịch về quá khứ)
        fill_value: Giá trị để điền cho các phần tử mới
        
    Returns:
        Series đã dịch chuyển
    """
    result = series.shift(periods)
    
    if periods > 0:
        result.iloc[:periods] = fill_value
    elif periods < 0:
        result.iloc[periods:] = fill_value
        
    return result

def get_highest_high(
    high: pd.Series, 
    window: int, 
    min_periods: int = 1
) -> pd.Series:
    """
    Lấy giá cao nhất trong cửa sổ di động.
    
    Args:
        high: Series giá cao
        window: Kích thước cửa sổ
        min_periods: Số lượng giá trị tối thiểu cần thiết
        
    Returns:
        Series chứa giá cao nhất
    """
    return high.rolling(window=window, min_periods=min_periods).max()

def get_lowest_low(
    low: pd.Series, 
    window: int, 
    min_periods: int = 1
) -> pd.Series:
    """
    Lấy giá thấp nhất trong cửa sổ di động.
    
    Args:
        low: Series giá thấp
        window: Kích thước cửa sổ
        min_periods: Số lượng giá trị tối thiểu cần thiết
        
    Returns:
        Series chứa giá thấp nhất
    """
    return low.rolling(window=window, min_periods=min_periods).min()

def scale_macd_components(
    macd_line: pd.Series, 
    signal_line: pd.Series, 
    histogram: pd.Series,
    method: str = 'zscore',
    window: int = 50
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Chuẩn hóa các thành phần của MACD để tránh chênh lệch quá lớn.
    
    Args:
        macd_line: Series chứa MACD line
        signal_line: Series chứa Signal line
        histogram: Series chứa Histogram
        method: Phương pháp chuẩn hóa ('zscore', 'minmax', 'robust', 'clipping')
        window: Kích thước cửa sổ cho chuẩn hóa động
        
    Returns:
        Tuple của các Series (macd_line, signal_line, histogram) đã được chuẩn hóa
    """
    # Sao chép để tránh thay đổi dữ liệu gốc
    macd = macd_line.copy()
    signal = signal_line.copy()
    hist = histogram.copy()
    
    # Xử lý các giá trị không hợp lệ
    invalid_mask = ~np.isfinite(macd) | ~np.isfinite(signal) | ~np.isfinite(hist)
    if invalid_mask.any():
        macd[invalid_mask] = 0
        signal[invalid_mask] = 0
        hist[invalid_mask] = 0
    
    if method == 'zscore':
        # Chuẩn hóa theo z-score với cửa sổ động
        def apply_zscore(series):
            means = series.rolling(window=window, min_periods=1).mean()
            stds = series.rolling(window=window, min_periods=1).std()
            stds = stds.replace(0, 1e-8)  # Tránh chia cho 0
            return (series - means) / stds
        
        macd = apply_zscore(macd)
        signal = apply_zscore(signal)
        hist = apply_zscore(hist)
        
        # Giới hạn z-score trong khoảng [-3, 3]
        macd = np.clip(macd, -3, 3)
        signal = np.clip(signal, -3, 3)
        hist = np.clip(hist, -3, 3)
        
    elif method == 'minmax':
        # Chuẩn hóa theo min-max với cửa sổ động
        def apply_minmax(series):
            mins = series.rolling(window=window, min_periods=1).min()
            maxs = series.rolling(window=window, min_periods=1).max()
            ranges = maxs - mins
            ranges = ranges.replace(0, 1e-8)  # Tránh chia cho 0
            return (series - mins) / ranges
        
        macd = apply_minmax(macd)
        signal = apply_minmax(signal)
        hist = apply_minmax(hist)
        
    elif method == 'robust':
        # Chuẩn hóa theo median và MAD với cửa sổ động
        def apply_robust(series):
            def mad_calc(x):
                median = np.median(x)
                return np.median(np.abs(x - median))
            
            medians = series.rolling(window=window, min_periods=1).apply(
                lambda x: np.median(x), raw=True
            )
            mads = series.rolling(window=window, min_periods=1).apply(
                mad_calc, raw=True
            )
            # Điều chỉnh MAD để tương đương với độ lệch chuẩn khi phân phối chuẩn
            mads = mads * 1.4826
            mads = mads.replace(0, 1e-8)  # Tránh chia cho 0
            return (series - medians) / mads
        
        macd = apply_robust(macd)
        signal = apply_robust(signal)
        hist = apply_robust(hist)
        
        # Giới hạn trong khoảng [-3, 3]
        macd = np.clip(macd, -3, 3)
        signal = np.clip(signal, -3, 3)
        hist = np.clip(hist, -3, 3)
        
    elif method == 'clipping':
        # Đơn giản chỉ cắt bớt các giá trị ngoại lệ
        # Tính ngưỡng dựa trên phân vị
        q1_macd, q3_macd = macd.quantile(0.25), macd.quantile(0.75)
        q1_signal, q3_signal = signal.quantile(0.25), signal.quantile(0.75)
        q1_hist, q3_hist = hist.quantile(0.25), hist.quantile(0.75)
        
        iqr_macd = q3_macd - q1_macd
        iqr_signal = q3_signal - q1_signal
        iqr_hist = q3_hist - q1_hist
        
        # Ngưỡng cắt: Q1 - 1.5*IQR và Q3 + 1.5*IQR
        lower_macd, upper_macd = q1_macd - 1.5*iqr_macd, q3_macd + 1.5*iqr_macd
        lower_signal, upper_signal = q1_signal - 1.5*iqr_signal, q3_signal + 1.5*iqr_signal
        lower_hist, upper_hist = q1_hist - 1.5*iqr_hist, q3_hist + 1.5*iqr_hist
        
        # Áp dụng clipping
        macd = np.clip(macd, lower_macd, upper_macd)
        signal = np.clip(signal, lower_signal, upper_signal)
        hist = np.clip(hist, lower_hist, upper_hist)
    
    return macd, signal, hist