"""
Các chỉ báo động lượng.
File này cung cấp các chỉ báo kỹ thuật cho việc phân tích động lượng thị trường
như RSI, Stochastic, CCI, v.v.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any

# Import các hàm tiện ích
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, exponential_weights,
    true_range, get_highest_high, get_lowest_low, normalize_indicator
)

def relative_strength_index(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 14,
    method: str = 'ema',
    normalize: bool = False,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Relative Strength Index (RSI).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ cho tính toán
        method: Phương pháp tính toán 'ema' hoặc 'sma'
        normalize: Chuẩn hóa giá trị về khoảng [0,1] (thay vì [0,100])
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa giá trị RSI
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    # Tạo bản sao an toàn
    result_df = df.copy()
    
    # Tính delta (thay đổi giá)
    delta = result_df[column].diff()
    
    # Phân tách thành gain (dương) và loss (âm)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Tính trung bình của gain và loss
    if method.lower() == 'ema':
        # Sử dụng EMA cho gain và loss
        avg_gain = gain.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
    else:
        # Sử dụng SMA cho gain và loss
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # Tránh chia cho 0
    avg_loss = avg_loss.replace(0, np.nan)
    
    # Tính RS = AvgGain / AvgLoss
    rs = avg_gain / avg_loss
    
    # Tính RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))
    
    # Thay thế NaN bằng 50 (giá trị trung tính cho RSI)
    rsi = rsi.fillna(50)

    # Đảm bảo RSI luôn nằm trong khoảng [0, 100]
    rsi = np.clip(rsi, 0, 100)

    # Kiểm tra và sửa các giá trị ngoại lệ (NaN, Inf)
    is_invalid = ~np.isfinite(rsi)
    if is_invalid.any():
        invalid_count = is_invalid.sum()
        print(f"Cảnh báo: Phát hiện {invalid_count} giá trị RSI không hợp lệ, tự động sửa lại")
        # Đặt giá trị không hợp lệ thành 50 (giá trị trung tính)
        rsi = np.where(is_invalid, 50, rsi)
        
    # Chuẩn hóa về [0,1] nếu yêu cầu
    if normalize:
        rsi = rsi / 100.0
        result_name = f"{prefix}rsi_norm_{window}"
    else:
        result_name = f"{prefix}rsi_{window}"
        result_df[result_name] = rsi
    
    # THÊM DÒNG NÀY: Chuyển đổi rõ ràng thành float
    rsi = rsi.astype(float)
    
    # SỬA DÒNG NÀY: Sử dụng phương thức an toàn để gán giá trị
    # Tạo Series mới với tên cụ thể trước khi gán vào DataFrame
    rsi_series = pd.Series(rsi, index=result_df.index, name=result_name)
    
    # THÊM TRY-EXCEPT để xử lý lỗi và ghi log
    try:
        # Gán Series vào DataFrame bằng loc để tránh vấn đề với cấu trúc nội bộ
        result_df = result_df.assign(**{result_name: rsi_series})
    except Exception as e:
        # Log lỗi chi tiết
        import traceback
        error_trace = traceback.format_exc()
        print(f"Lỗi khi gán RSI: {e}\n{error_trace}")
        
        # Thử phương thức khác nếu cách đầu không thành công
        try:
            # Phương pháp dự phòng
            result_df.loc[:, result_name] = rsi_series.values
        except Exception as e2:
            print(f"Lỗi khi gán RSI (phương pháp dự phòng): {e2}")
            # Nếu vẫn thất bại, trả về DataFrame gốc
            return df
    
    return result_df

def stochastic_oscillator(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 1,
    normalize: bool = False,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Stochastic Oscillator.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        k_period: Kích thước cửa sổ cho %K
        d_period: Kích thước cửa sổ cho %D
        smooth_k: Độ trơn của %K
        normalize: Chuẩn hóa giá trị về khoảng [0,1] (thay vì [0,100])
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa %K và %D
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính các giá trị rolling - chỉ tính một lần để tối ưu hiệu suất
    high_max = result_df['high'].rolling(window=k_period).max()
    low_min = result_df['low'].rolling(window=k_period).min()
    
    # Tính %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    # Tránh chia cho 0
    price_range = high_max - low_min
    price_range = price_range.replace(0, np.nan)
    
    stoch_k = 100 * ((result_df['close'] - low_min) / price_range)
    
    # Làm mịn %K nếu cần
    if smooth_k > 1:
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
    
    # Tính %D = SMA của %K
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    # Thay thế NaN bằng 50 (giá trị trung tính)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_d.fillna(50)
    
    # Chuẩn hóa về [0,1] nếu yêu cầu
    if normalize:
        stoch_k = stoch_k / 100.0
        stoch_d = stoch_d / 100.0
        k_name = f"{prefix}stoch_k_norm_{k_period}"
        d_name = f"{prefix}stoch_d_norm_{k_period}_{d_period}"
    else:
        k_name = f"{prefix}stoch_k_{k_period}"
        d_name = f"{prefix}stoch_d_{k_period}_{d_period}"
    
    # Đặt tên các cột kết quả
    result_df[k_name] = stoch_k
    result_df[d_name] = stoch_d
    
    return result_df

def williams_r(
    df: pd.DataFrame,
    window: int = 14,
    normalize: bool = False,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Williams %R.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ
        normalize: Chuẩn hóa giá trị về khoảng [0,1] (thay vì [-100,0])
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa Williams %R
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính highest high và lowest low trong cửa sổ
    high_max = result_df['high'].rolling(window=window).max()
    low_min = result_df['low'].rolling(window=window).min()
    
    # Tránh chia cho 0
    range_hl = high_max - low_min
    range_hl = range_hl.replace(0, np.nan)
    
    # Tính Williams %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
    williams_r_val = -100 * ((high_max - result_df['close']) / range_hl)
    
    # Thay thế NaN bằng -50 (giá trị trung tính)
    williams_r_val = williams_r_val.fillna(-50)
    
    # Chuẩn hóa về [0,1] nếu yêu cầu
    if normalize:
        # Chuyển từ [-100,0] sang [0,1]
        williams_r_norm = (williams_r_val + 100) / 100.0
        result_name = f"{prefix}williams_r_norm_{window}"
        result_df[result_name] = williams_r_norm
    else:
        result_name = f"{prefix}williams_r_{window}"
        result_df[result_name] = williams_r_val
    
    return result_df

def commodity_channel_index(
    df: pd.DataFrame,
    window: int = 20,
    constant: float = 0.015,
    normalize: bool = False,
    normalize_range: bool = False,
    normalization_period: int = None,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Commodity Channel Index (CCI).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        window: Kích thước cửa sổ
        constant: Hằng số chia (thường là 0.015)
        normalize: Chuẩn hóa giá trị về khoảng [0,1]
        normalize_range: Sử dụng phạm vi động thay vì phạm vi cố định [-100,100]
        normalization_period: Kích thước cửa sổ cho việc chuẩn hóa (mặc định bằng window)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa CCI
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
    
    result_df = df.copy()
    
    # Tính Typical Price = (High + Low + Close) / 3
    typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
    
    # Tính SMA của Typical Price
    tp_sma = typical_price.rolling(window=window).mean()
    
    # Tính Mean Deviation - sử dụng numpy để tối ưu
    # Mean Deviation = SUM(|Typical Price - SMA(Typical Price)|) / Period
    def mean_abs_dev(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    mean_deviation = typical_price.rolling(window=window).apply(
        mean_abs_dev, 
        raw=True  # Sử dụng raw=True để tối ưu hóa hiệu suất
    )
    
    # Tránh chia cho 0
    mean_deviation = mean_deviation.replace(0, np.nan)
    
    # Tính CCI = (Typical Price - SMA(Typical Price)) / (constant * Mean Deviation)
    cci = (typical_price - tp_sma) / (constant * mean_deviation)
    
    # Thay thế NaN bằng 0 (giá trị trung tính)
    cci = cci.fillna(0)
    
    # Chuẩn hóa về [0,1] nếu yêu cầu
    if normalize:
        if normalization_period is None:
            normalization_period = window
        
        if normalize_range:
            # Chuẩn hóa dựa trên phạm vi thực tế trong period
            cci_norm = normalize_indicator(cci, method='minmax', window=normalization_period)
            result_name = f"{prefix}cci_norm_adaptive_{window}"
        else:
            # Chuẩn hóa về khoảng [0,1] từ phạm vi giả định [-100, 100]
            cci_norm = (cci + 100) / 200
            # Giới hạn vào khoảng [0,1]
            cci_norm = cci_norm.clip(0, 1)
            result_name = f"{prefix}cci_norm_{window}"
            result_df[result_name] = cci_norm
        
    else:
        result_name = f"{prefix}cci_{window}"
        result_df[result_name] = cci
    
    return result_df

def rate_of_change(
    df: pd.DataFrame,
    column: str = 'close',
    window: int = 9,
    percentage: bool = True,
    normalize: bool = False,
    normalize_period: int = None,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Rate of Change (ROC).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ
        percentage: Nếu True, tính ROC theo phần trăm
        normalize: Chuẩn hóa giá trị về khoảng [0,1]
        normalize_period: Kích thước cửa sổ cho việc chuẩn hóa
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với cột mới chứa ROC
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính ROC = (Current Price - Price n periods ago) / Price n periods ago * 100 (nếu percentage=True)
    # hoặc ROC = Current Price - Price n periods ago (nếu percentage=False)
    price_n_periods_ago = result_df[column].shift(window)
    
    if percentage:
        # Tránh chia cho 0
        price_n_periods_ago = price_n_periods_ago.replace(0, np.nan)
        roc = (result_df[column] - price_n_periods_ago) / price_n_periods_ago * 100
        roc_name = f"{prefix}roc_pct_{window}"
    else:
        roc = result_df[column] - price_n_periods_ago
        roc_name = f"{prefix}roc_{window}"
    
    # Thay thế NaN bằng 0
    roc = roc.fillna(0)
    
    # Chuẩn hóa nếu yêu cầu
    if normalize:
        if normalize_period is None:
            normalize_period = max(window * 2, 20)  # Mặc định là 2x window hoặc ít nhất 20
        
        roc_norm = normalize_indicator(roc, method='minmax', window=normalize_period)
        norm_name = f"{prefix}roc_norm_{window}"
        result_df[norm_name] = roc_norm
    
    # Đặt tên cột kết quả gốc
    result_df[roc_name] = roc
    
    return result_df

def money_flow_index(
    df: pd.DataFrame,
    window: int = 14,
    normalize: bool = False,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính Money Flow Index (MFI).
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        window: Kích thước cửa sổ
        normalize: Chuẩn hóa giá trị về khoảng [0,1] (thay vì [0,100])
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
    
    # Tối ưu cách tính positive và negative flow
    positive_flow = price_diff.copy()
    positive_flow[price_diff <= 0] = 0
    
    negative_flow = price_diff.copy()
    negative_flow[price_diff >= 0] = 0
    negative_flow = negative_flow.abs()
    
    # Nhân với raw money flow
    positive_flow = positive_flow * raw_money_flow / price_diff
    negative_flow = negative_flow * raw_money_flow / price_diff.abs()
    
    # Xử lý NaN từ phép chia
    positive_flow = positive_flow.fillna(0)
    negative_flow = negative_flow.fillna(0)
    
    # Tính tổng Positive và Negative Money Flow trong cửa sổ
    positive_flow_sum = positive_flow.rolling(window=window).sum()
    negative_flow_sum = negative_flow.rolling(window=window).sum()
    
    # Tránh chia cho 0
    negative_flow_sum = negative_flow_sum.replace(0, np.nan)
    
    # Tính Money Flow Ratio = Positive Money Flow / Negative Money Flow
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Tính Money Flow Index = 100 - 100 / (1 + Money Flow Ratio)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # Thay thế NaN
    mfi = mfi.fillna(50)  # 50 là giá trị trung tính cho MFI
    
    # Chuẩn hóa nếu yêu cầu
    if normalize:
        mfi = mfi / 100.0
        result_name = f"{prefix}mfi_norm_{window}"
    else:
        result_name = f"{prefix}mfi_{window}"
        result_df[result_name] = mfi
    
    return result_df

def true_strength_index(
    df: pd.DataFrame,
    column: str = 'close',
    long_window: int = 25,
    short_window: int = 13,
    signal_window: int = 7,
    normalize: bool = False,
    normalize_period: int = None,
    prefix: str = 'momentum_'
) -> pd.DataFrame:
    """
    Tính True Strength Index (TSI).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        column: Tên cột giá sử dụng để tính toán
        long_window: Kích thước cửa sổ dài
        short_window: Kích thước cửa sổ ngắn
        signal_window: Kích thước cửa sổ cho đường tín hiệu
        normalize: Chuẩn hóa giá trị về khoảng [0,1]
        normalize_period: Kích thước cửa sổ cho việc chuẩn hóa
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa TSI và Signal Line
    """
    if not validate_price_data(df, [column]):
        raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
    
    result_df = df.copy()
    
    # Tính momentum (thay đổi giá)
    momentum = result_df[column].diff()
    
    # Double EMA smoothing trên momentum
    # First smoothing
    first_smooth = momentum.ewm(span=long_window, adjust=False).mean()
    
    # Second smoothing
    double_smooth = first_smooth.ewm(span=short_window, adjust=False).mean()
    
    # Tương tự cho giá trị tuyệt đối của momentum
    # First smoothing
    first_smooth_abs = momentum.abs().ewm(span=long_window, adjust=False).mean()
    
    # Second smoothing
    double_smooth_abs = first_smooth_abs.ewm(span=short_window, adjust=False).mean()
    
    # Tránh chia cho 0
    double_smooth_abs = double_smooth_abs.replace(0, np.nan)
    
    # Tính TSI = (Double Smoothed Momentum / Double Smoothed Absolute Momentum) * 100
    tsi = (double_smooth / double_smooth_abs) * 100
    
    # Tính Signal Line = EMA của TSI
    signal_line = tsi.ewm(span=signal_window, adjust=False).mean()
    
    # Thay thế NaN
    tsi = tsi.fillna(0)  # 0 là giá trị trung tính cho TSI
    signal_line = signal_line.fillna(0)
    
    # Chuẩn hóa nếu yêu cầu
    if normalize:
        if normalize_period is None:
            normalize_period = max(long_window, 30)  # Mặc định là window dài hoặc ít nhất 30
        
        # TSI thường dao động trong khoảng [-100, 100]
        # Có thể chuẩn hóa bằng cách sử dụng phạm vi tĩnh
        tsi_norm = (tsi + 100) / 200
        signal_norm = (signal_line + 100) / 200
        
        # Giới hạn vào khoảng [0,1]
        tsi_norm = tsi_norm.clip(0, 1)
        signal_norm = signal_norm.clip(0, 1)
        
        tsi_name = f"{prefix}tsi_norm_{long_window}_{short_window}"
        signal_name = f"{prefix}tsi_signal_norm_{signal_window}"
        
        result_df[tsi_name] = tsi_norm
        result_df[signal_name] = signal_norm
    else:
        tsi_name = f"{prefix}tsi_{long_window}_{short_window}"
        signal_name = f"{prefix}tsi_signal_{signal_window}"
        
        result_df[tsi_name] = tsi
        result_df[signal_name] = signal_line
    
    # Tính histogram
    if normalize:
        result_df[f"{prefix}tsi_hist_norm"] = tsi_norm - signal_norm
    else:
        result_df[f"{prefix}tsi_hist"] = tsi - signal_line
    
    return result_df