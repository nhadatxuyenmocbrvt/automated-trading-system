"""
Đặc trưng về khối lượng.
Mô-đun này cung cấp các hàm tạo đặc trưng dựa trên khối lượng giao dịch,
bao gồm khối lượng tương đối, tương quan giá-khối lượng, và các biến thể OBV và VWAP.
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
logger = setup_logger("volume_features")

def calculate_volume_features(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    windows: List[int] = [5, 10, 20, 50, 100],
    normalize: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng cơ bản về khối lượng giao dịch.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng
        volume_column: Tên cột khối lượng sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        normalize: Chuẩn hóa khối lượng so với trung bình
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng khối lượng
    """
    if not validate_price_data(df, [volume_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {volume_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính khối lượng log để giảm ảnh hưởng của các giá trị cực đoan
        log_volume = np.log1p(result_df[volume_column])
        result_df[f"{prefix}log_volume"] = log_volume
        
        # Tính thay đổi khối lượng theo phần trăm
        volume_change = result_df[volume_column].pct_change() * 100
        result_df[f"{prefix}volume_change_pct"] = volume_change
        
        for window in windows:
            # 1. Tính SMA của khối lượng
            vol_sma = result_df[volume_column].rolling(window=window).mean()
            
            if normalize:
                # Chuẩn hóa khối lượng hiện tại so với SMA
                vol_sma_non_zero = vol_sma.replace(0, np.nan)
                norm_volume = result_df[volume_column] / vol_sma_non_zero
                
                result_df[f"{prefix}volume_sma_ratio_{window}"] = norm_volume
                
                # Chỉ báo khối lượng cao (>150% trung bình)
                result_df[f"{prefix}high_volume_{window}"] = (norm_volume > 1.5).astype(int)
                
                # Chỉ báo khối lượng thấp (<50% trung bình)
                result_df[f"{prefix}low_volume_{window}"] = (norm_volume < 0.5).astype(int)
            else:
                result_df[f"{prefix}volume_sma_{window}"] = vol_sma
            
            # 2. Tính độ lệch chuẩn của khối lượng
            vol_std = result_df[volume_column].rolling(window=window).std()
            result_df[f"{prefix}volume_std_{window}"] = vol_std
            
            # 3. Z-score của khối lượng
            vol_mean = result_df[volume_column].rolling(window=window).mean()
            vol_std_non_zero = vol_std.replace(0, np.nan)
            vol_zscore = (result_df[volume_column] - vol_mean) / vol_std_non_zero
            
            result_df[f"{prefix}volume_zscore_{window}"] = vol_zscore
            
            # 4. Xếp hạng phần trăm khối lượng trong cửa sổ
            vol_rank = result_df[volume_column].rolling(window=window).rank(pct=True)
            result_df[f"{prefix}volume_rank_{window}"] = vol_rank
            
            # 5. Trung bình khối lượng tích lũy trong n ngày
            vol_sum = result_df[volume_column].rolling(window=window).sum()
            result_df[f"{prefix}volume_sum_{window}"] = vol_sum
            
            # 6. Chỉ báo tăng trưởng khối lượng
            vol_growth = vol_sma / vol_sma.shift(window//2) - 1
            result_df[f"{prefix}volume_growth_{window}"] = vol_growth
        
        logger.debug("Đã tính đặc trưng khối lượng cơ bản")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng khối lượng cơ bản: {e}")
    
    return result_df

def calculate_relative_volume(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    price_column: str = 'close',
    windows: List[int] = [5, 10, 20],
    ref_windows: List[int] = [50, 100],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng khối lượng tương đối.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng và giá
        volume_column: Tên cột khối lượng
        price_column: Tên cột giá
        windows: Danh sách các kích thước cửa sổ ngắn hạn
        ref_windows: Danh sách các kích thước cửa sổ tham chiếu dài hạn
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng khối lượng tương đối
    """
    if not validate_price_data(df, [volume_column, price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {volume_column} hoặc {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Phần 1: Khối lượng tương đối so với giai đoạn khác nhau
        for window in windows:
            for ref_window in ref_windows:
                if window < ref_window:
                    # Tính khối lượng trung bình ngắn hạn và dài hạn
                    short_vol_avg = result_df[volume_column].rolling(window=window).mean()
                    long_vol_avg = result_df[volume_column].rolling(window=ref_window).mean()
                    
                    # Tránh chia cho 0
                    long_vol_avg_non_zero = long_vol_avg.replace(0, np.nan)
                    
                    # Tỷ lệ khối lượng ngắn hạn/dài hạn
                    rel_volume = short_vol_avg / long_vol_avg_non_zero
                    
                    result_df[f"{prefix}rel_volume_{window}_{ref_window}"] = rel_volume
                    
                    # Chỉ báo động lượng khối lượng (thay đổi tỷ lệ khối lượng)
                    vol_momentum = rel_volume - rel_volume.shift(window)
                    
                    result_df[f"{prefix}volume_momentum_{window}_{ref_window}"] = vol_momentum
                    
                    # Khối lượng bất thường (z-score của tỷ lệ khối lượng)
                    rel_vol_mean = rel_volume.rolling(window=ref_window).mean()
                    rel_vol_std = rel_volume.rolling(window=ref_window).std()
                    
                    # Tránh chia cho 0
                    rel_vol_std_non_zero = rel_vol_std.replace(0, np.nan)
                    
                    rel_vol_zscore = (rel_volume - rel_vol_mean) / rel_vol_std_non_zero
                    
                    result_df[f"{prefix}rel_volume_zscore_{window}_{ref_window}"] = rel_vol_zscore
        
        # Phần 2: Phân tích khối lượng dựa trên chuyển động giá
        
        # Tính lợi nhuận
        returns = result_df[price_column].pct_change()
        
        # Khối lượng trong ngày tăng
        up_volume = result_df[volume_column].copy()
        up_volume[returns <= 0] = 0
        
        # Khối lượng trong ngày giảm
        down_volume = result_df[volume_column].copy()
        down_volume[returns >= 0] = 0
        
        # Thêm vào DataFrame
        result_df[f"{prefix}up_volume"] = up_volume
        result_df[f"{prefix}down_volume"] = down_volume
        
        for window in windows:
            # Tính tổng khối lượng trong ngày tăng và giảm
            up_vol_sum = up_volume.rolling(window=window).sum()
            down_vol_sum = down_volume.rolling(window=window).sum()
            
            # Tránh chia cho 0
            sum_non_zero = (up_vol_sum + down_vol_sum).replace(0, np.nan)
            
            # Tỷ lệ khối lượng tăng/giảm
            up_down_ratio = up_vol_sum / down_vol_sum.replace(0, np.nan)
            result_df[f"{prefix}up_down_vol_ratio_{window}"] = up_down_ratio
            
            # Phần trăm khối lượng trong ngày tăng
            up_vol_pct = up_vol_sum / sum_non_zero * 100
            result_df[f"{prefix}up_vol_pct_{window}"] = up_vol_pct
            
            # Tính chỉ báo lực mua/bán (1 = lực mua mạnh, -1 = lực bán mạnh)
            buy_sell_pressure = 2 * (up_vol_pct / 100 - 0.5)
            result_df[f"{prefix}buy_sell_pressure_{window}"] = buy_sell_pressure
        
        logger.debug("Đã tính đặc trưng khối lượng tương đối")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng khối lượng tương đối: {e}")
    
    return result_df

def calculate_volume_price_correlation(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    price_column: str = 'close',
    windows: List[int] = [10, 20, 50],
    use_log: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính tương quan giữa khối lượng và giá.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng và giá
        volume_column: Tên cột khối lượng
        price_column: Tên cột giá
        windows: Danh sách các kích thước cửa sổ
        use_log: Sử dụng log của khối lượng để giảm ảnh hưởng giá trị cực đoan
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng tương quan khối lượng-giá
    """
    if not validate_price_data(df, [volume_column, price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {volume_column} hoặc {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính returns
        returns = result_df[price_column].pct_change()
        
        # Tạo chuỗi mới để tính tương quan
        if use_log:
            volume_series = np.log1p(result_df[volume_column])
        else:
            volume_series = result_df[volume_column]
        
        for window in windows:
            # 1. Tương quan giữa khối lượng và return tuyệt đối (biến động)
            abs_returns = returns.abs()
            
            vol_volatility_corr = abs_returns.rolling(window=window).corr(volume_series)
            result_df[f"{prefix}vol_volatility_corr_{window}"] = vol_volatility_corr
            
            # 2. Tương quan giữa khối lượng và return (giá)
            vol_return_corr = returns.rolling(window=window).corr(volume_series)
            result_df[f"{prefix}vol_return_corr_{window}"] = vol_return_corr
            
            # 3. Tương quan giữa khối lượng và giá
            vol_price_corr = result_df[price_column].rolling(window=window).corr(volume_series)
            result_df[f"{prefix}vol_price_corr_{window}"] = vol_price_corr
            
            # 4. Tương quan giữa thay đổi khối lượng và thay đổi giá
            volume_change = volume_series.pct_change()
            price_change = result_df[price_column].pct_change()
            
            vol_change_price_change_corr = volume_change.rolling(window=window).corr(price_change)
            result_df[f"{prefix}vol_change_price_change_corr_{window}"] = vol_change_price_change_corr
            
            # 5. Money Flow Index đơn giản
            typical_price = result_df[price_column]  # Sử dụng close nếu không có OHLC
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
            
            money_flow = typical_price * result_df[volume_column]
            
            # Xác định ngày dương và âm
            pos_flow = money_flow.copy()
            pos_flow[returns <= 0] = 0
            
            neg_flow = money_flow.copy()
            neg_flow[returns >= 0] = 0
            
            # Tính tổng trong cửa sổ
            pos_flow_sum = pos_flow.rolling(window=window).sum()
            neg_flow_sum = neg_flow.rolling(window=window).sum()
            
            # Tránh chia cho 0
            neg_flow_sum_non_zero = neg_flow_sum.replace(0, np.nan)
            
            # Money Ratio
            money_ratio = pos_flow_sum / neg_flow_sum_non_zero
            
            # Money Flow Index = 100 - (100 / (1 + Money Ratio))
            mfi = 100 - (100 / (1 + money_ratio))
            
            result_df[f"{prefix}mfi_simple_{window}"] = mfi
        
        logger.debug("Đã tính đặc trưng tương quan khối lượng-giá")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tương quan khối lượng-giá: {e}")
    
    return result_df

def calculate_volume_oscillations(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    price_column: str = 'close',
    short_windows: List[int] = [5, 10],
    long_windows: List[int] = [20, 50],
    use_ema: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các dao động khối lượng và chỉ báo phân kỳ.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng và giá
        volume_column: Tên cột khối lượng
        price_column: Tên cột giá
        short_windows: Danh sách các kích thước cửa sổ ngắn
        long_windows: Danh sách các kích thước cửa sổ dài
        use_ema: Sử dụng EMA thay vì SMA
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng dao động khối lượng
    """
    if not validate_price_data(df, [volume_column, price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {volume_column} hoặc {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính các trung bình động của khối lượng
        for short_window in short_windows:
            for long_window in long_windows:
                if short_window < long_window:
                    if use_ema:
                        # Tính EMA
                        vol_short_ema = result_df[volume_column].ewm(span=short_window, adjust=False).mean()
                        vol_long_ema = result_df[volume_column].ewm(span=long_window, adjust=False).mean()
                        
                        # Dao động khối lượng
                        vol_osc = ((vol_short_ema - vol_long_ema) / vol_long_ema) * 100
                        
                        result_df[f"{prefix}vol_ema_osc_{short_window}_{long_window}"] = vol_osc
                    else:
                        # Tính SMA
                        vol_short_sma = result_df[volume_column].rolling(window=short_window).mean()
                        vol_long_sma = result_df[volume_column].rolling(window=long_window).mean()
                        
                        # Dao động khối lượng
                        vol_osc = ((vol_short_sma - vol_long_sma) / vol_long_sma) * 100
                        
                        result_df[f"{prefix}vol_sma_osc_{short_window}_{long_window}"] = vol_osc
                    
                    # Tính các đặc trưng nâng cao từ dao động khối lượng
                    osc_name = f"{prefix}vol_{'ema' if use_ema else 'sma'}_osc_{short_window}_{long_window}"
                    
                    # 1. Tín hiệu giao cắt 0
                    result_df[f"{osc_name}_cross_above_zero"] = (
                        (result_df[osc_name].shift(1) <= 0) & 
                        (result_df[osc_name] > 0)
                    ).astype(int)
                    
                    result_df[f"{osc_name}_cross_below_zero"] = (
                        (result_df[osc_name].shift(1) >= 0) & 
                        (result_df[osc_name] < 0)
                    ).astype(int)
                    
                    # 2. Phát hiện cực trị
                    result_df[f"{osc_name}_peak"] = (
                        (result_df[osc_name] > result_df[osc_name].shift(1)) & 
                        (result_df[osc_name] > result_df[osc_name].shift(-1))
                    ).astype(int)
                    
                    result_df[f"{osc_name}_trough"] = (
                        (result_df[osc_name] < result_df[osc_name].shift(1)) & 
                        (result_df[osc_name] < result_df[osc_name].shift(-1))
                    ).astype(int)
                    
                    # 3. Phát hiện phân kỳ với giá
                    price_diff = result_df[price_column].diff(short_window)
                    osc_diff = result_df[osc_name].diff(short_window)
                    
                    # Phân kỳ tích cực (giá giảm nhưng khối lượng tăng)
                    result_df[f"{osc_name}_bullish_divergence"] = (
                        (price_diff < 0) & (osc_diff > 0)
                    ).astype(int)
                    
                    # Phân kỳ tiêu cực (giá tăng nhưng khối lượng giảm)
                    result_df[f"{osc_name}_bearish_divergence"] = (
                        (price_diff > 0) & (osc_diff < 0)
                    ).astype(int)
        
        logger.debug("Đã tính đặc trưng dao động khối lượng")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng dao động khối lượng: {e}")
    
    return result_df

def calculate_obv_features(
    df: pd.DataFrame,
    volume_column: str = 'volume',
    price_column: str = 'close',
    windows: List[int] = [10, 20, 50],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính On-Balance Volume (OBV) và các biến thể.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng và giá
        volume_column: Tên cột khối lượng
        price_column: Tên cột giá
        windows: Danh sách các kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng OBV
    """
    if not validate_price_data(df, [volume_column, price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {volume_column} hoặc {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính OBV cơ bản
        price_change = result_df[price_column].diff()
        
        obv = pd.Series(0, index=result_df.index)
        
        for i in range(1, len(result_df)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + result_df[volume_column].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - result_df[volume_column].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        result_df[f"{prefix}obv"] = obv
        
        # Tính các đặc trưng nâng cao từ OBV
        for window in windows:
            # 1. OBV bình thường hóa (chuyển về khoảng 0-100 dựa trên min-max trong cửa sổ)
            obv_min = obv.rolling(window=window).min()
            obv_max = obv.rolling(window=window).max()
            
            # Tránh chia cho 0
            obv_range = obv_max - obv_min
            obv_range_non_zero = obv_range.replace(0, np.nan)
            
            obv_normalized = ((obv - obv_min) / obv_range_non_zero) * 100
            
            result_df[f"{prefix}obv_normalized_{window}"] = obv_normalized
            
            # 2. OBV rate of change
            obv_roc = (obv - obv.shift(window)) / obv.shift(window).abs() * 100
            
            result_df[f"{prefix}obv_roc_{window}"] = obv_roc
            
            # 3. Moving average của OBV
            obv_ma = obv.rolling(window=window).mean()
            
            result_df[f"{prefix}obv_ma_{window}"] = obv_ma
            
            # 4. OBV momentum (tương quan giữa OBV và giá)
            obv_momentum = obv.diff(window) / obv.shift(window).abs() * 100
            price_momentum = result_df[price_column].diff(window) / result_df[price_column].shift(window) * 100
            
            # Nếu cùng hướng: giá trị dương, ngược hướng: giá trị âm
            obv_price_agreement = np.sign(obv_momentum) * np.sign(price_momentum)
            
            result_df[f"{prefix}obv_price_agreement_{window}"] = obv_price_agreement
            
            # 5. OBV tương đối so với giá (thước đo của sức mạnh xu hướng)
            price_normalized = ((result_df[price_column] - result_df[price_column].rolling(window=window).min()) /
                               (result_df[price_column].rolling(window=window).max() - 
                                result_df[price_column].rolling(window=window).min()))
            
            obv_price_relative = obv_normalized / 100 - price_normalized
            
            result_df[f"{prefix}obv_price_relative_{window}"] = obv_price_relative
            
            # 6. Phát hiện phân kỳ OBV
            price_high = (result_df[price_column] > result_df[price_column].shift(1)) & \
                         (result_df[price_column] > result_df[price_column].shift(-1))
            price_low = (result_df[price_column] < result_df[price_column].shift(1)) & \
                        (result_df[price_column] < result_df[price_column].shift(-1))
            
            obv_high = (obv > obv.shift(1)) & (obv > obv.shift(-1))
            obv_low = (obv < obv.shift(1)) & (obv < obv.shift(-1))
            
            # Phân kỳ tiêu cực (đỉnh giá tăng nhưng đỉnh OBV giảm)
            bearish_divergence = pd.Series(False, index=result_df.index)
            bullish_divergence = pd.Series(False, index=result_df.index)
            
            # Đơn giản hóa phát hiện phân kỳ
            bearish_divergence = (
                price_high & (result_df[price_column] > result_df[price_column].shift(window//2)) &
                obv_high & (obv < obv.shift(window//2))
            )
            
            bullish_divergence = (
                price_low & (result_df[price_column] < result_df[price_column].shift(window//2)) &
                obv_low & (obv > obv.shift(window//2))
            )
            
            result_df[f"{prefix}obv_bearish_divergence_{window}"] = bearish_divergence.astype(int)
            result_df[f"{prefix}obv_bullish_divergence_{window}"] = bullish_divergence.astype(int)
        
        logger.debug("Đã tính đặc trưng OBV")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng OBV: {e}")
    
    return result_df

def calculate_vwap_features(
    df: pd.DataFrame,
    anchor: str = 'day',
    intraday_periods: Optional[List[int]] = None,
    price_column: str = 'close',
    volume_column: str = 'volume',
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính Volume-Weighted Average Price (VWAP) và các đặc trưng liên quan.
    
    Args:
        df: DataFrame chứa dữ liệu khối lượng và giá
        anchor: Điểm neo cho VWAP ('day', 'week', 'month')
        intraday_periods: Danh sách các khoảng thời gian nội ngày (None nếu dữ liệu không phải nội ngày)
        price_column: Tên cột giá
        volume_column: Tên cột khối lượng
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng VWAP
    """
    required_columns = [price_column, volume_column]
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # Đảm bảo có cột ngày tháng
        has_date = False
        date_column = None
        
        if 'timestamp' in result_df.columns:
            date_column = 'timestamp'
            has_date = True
        elif 'date' in result_df.columns:
            date_column = 'date'
            has_date = True
        elif 'datetime' in result_df.columns:
            date_column = 'datetime'
            has_date = True
        
        if not has_date:
            logger.warning("Không tìm thấy cột ngày tháng. VWAP sẽ được tính trên toàn bộ dữ liệu.")
            
            # Tính typical price
            typical_price = result_df[price_column]
            if all(col in result_df.columns for col in ['high', 'low']):
                typical_price = (result_df['high'] + result_df['low'] + result_df[price_column]) / 3
            
            # Tính VWAP đơn giản
            cumulative_tp_vol = (typical_price * result_df[volume_column]).cumsum()
            cumulative_vol = result_df[volume_column].cumsum()
            
            # Tránh chia cho 0
            cumulative_vol_non_zero = cumulative_vol.replace(0, np.nan)
            
            vwap = cumulative_tp_vol / cumulative_vol_non_zero
            
            result_df[f"{prefix}vwap"] = vwap
            
            # Tính khoảng cách giá hiện tại so với VWAP
            price_to_vwap = (result_df[price_column] - vwap) / vwap * 100
            
            result_df[f"{prefix}price_to_vwap_pct"] = price_to_vwap
        else:
            # Tính VWAP dựa trên anchor
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
                result_df[date_column] = pd.to_datetime(result_df[date_column])
            
            # Tạo key nhóm dựa trên anchor
            if anchor == 'day':
                group_key = result_df[date_column].dt.date
            elif anchor == 'week':
                group_key = result_df[date_column].dt.isocalendar().week
                group_key = result_df[date_column].dt.year.astype(str) + "-" + group_key.astype(str)
            elif anchor == 'month':
                group_key = result_df[date_column].dt.year.astype(str) + "-" + result_df[date_column].dt.month.astype(str)
            else:
                logger.warning(f"Anchor không hợp lệ: {anchor}. Sử dụng 'day'.")
                group_key = result_df[date_column].dt.date
            
            # Thêm group_key vào DataFrame
            result_df['group_key'] = group_key
            
            # Tính typical price
            typical_price = result_df[price_column]
            if all(col in result_df.columns for col in ['high', 'low']):
                typical_price = (result_df['high'] + result_df['low'] + result_df[price_column]) / 3
            
            # Tính VWAP cho mỗi nhóm
            result_df['tp_vol'] = typical_price * result_df[volume_column]
            result_df['cum_tp_vol'] = result_df.groupby('group_key')['tp_vol'].cumsum()
            result_df['cum_vol'] = result_df.groupby('group_key')[volume_column].cumsum()
            
            # Tránh chia cho 0
            result_df['cum_vol_non_zero'] = result_df['cum_vol'].replace(0, np.nan)
            
            result_df[f"{prefix}vwap_{anchor}"] = result_df['cum_tp_vol'] / result_df['cum_vol_non_zero']
            
            # Tính khoảng cách giá hiện tại so với VWAP
            result_df[f"{prefix}price_to_vwap_{anchor}_pct"] = (
                (result_df[price_column] - result_df[f"{prefix}vwap_{anchor}"]) / 
                result_df[f"{prefix}vwap_{anchor}"] * 100
            )
            
            # Tính băng VWAP (similar to Bollinger Bands)
            if intraday_periods is not None:
                for period in intraday_periods:
                    # Tính độ lệch chuẩn của giá so với VWAP
                    result_df[f'price_dev_from_vwap'] = result_df[price_column] - result_df[f"{prefix}vwap_{anchor}"]
                    
                    result_df[f'price_dev_std_{period}'] = result_df.groupby('group_key')['price_dev_from_vwap'].transform(
                        lambda x: x.rolling(period).std()
                    )
                    
                    # Tính băng trên và dưới
                    result_df[f"{prefix}vwap_{anchor}_upper_{period}"] = (
                        result_df[f"{prefix}vwap_{anchor}"] + 2 * result_df[f'price_dev_std_{period}']
                    )
                    
                    result_df[f"{prefix}vwap_{anchor}_lower_{period}"] = (
                        result_df[f"{prefix}vwap_{anchor}"] - 2 * result_df[f'price_dev_std_{period}']
                    )
                    
                    # Xác định vị trí giá trong băng VWAP
                    upper_band = result_df[f"{prefix}vwap_{anchor}_upper_{period}"]
                    lower_band = result_df[f"{prefix}vwap_{anchor}_lower_{period}"]
                    vwap = result_df[f"{prefix}vwap_{anchor}"]
                    price = result_df[price_column]
                    
                    # Phần trăm vị trí trong băng (0-100)
                    band_width = upper_band - lower_band
                    band_width_non_zero = band_width.replace(0, np.nan)
                    
                    price_position = (price - lower_band) / band_width_non_zero * 100
                    result_df[f"{prefix}price_in_vwap_band_{anchor}_{period}"] = price_position
            
            # Xóa các cột tạm thời
            for col in ['group_key', 'tp_vol', 'cum_tp_vol', 'cum_vol', 'cum_vol_non_zero', 
                         'price_dev_from_vwap']:
                if col in result_df.columns:
                    del result_df[col]
            
            for period in intraday_periods or []:
                if f'price_dev_std_{period}' in result_df.columns:
                    del result_df[f'price_dev_std_{period}']
        
        logger.debug(f"Đã tính đặc trưng VWAP với anchor={anchor}")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng VWAP: {e}")
    
    return result_df