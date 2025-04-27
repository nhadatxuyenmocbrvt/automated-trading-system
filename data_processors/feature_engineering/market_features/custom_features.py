"""
Đặc trưng tùy chỉnh.
Mô-đun này cung cấp các hàm tạo đặc trưng nâng cao và tùy chỉnh,
bao gồm đặc trưng hồi quy về giá trị trung bình, sức mạnh xu hướng,
mẫu hình giá, và phân tích chế độ thị trường.
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
logger = setup_logger("custom_features")

def calculate_mean_reversion_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    windows: List[int] = [20, 50, 100, 200],
    bands: List[float] = [1.0, 1.5, 2.0, 2.5],
    use_log_scale: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng liên quan đến hiện tượng hồi quy về giá trị trung bình.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        bands: Danh sách các độ rộng dải (số độ lệch chuẩn)
        use_log_scale: Sử dụng thang logarithm cho giá
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng hồi quy về giá trị trung bình
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Chuyển giá sang thang logarithm nếu cần
        if use_log_scale:
            price_data = np.log(result_df[price_column])
            result_df[f"{prefix}log_price"] = price_data
        else:
            price_data = result_df[price_column]
        
        # Tính các đặc trưng hồi quy về giá trị trung bình cho mỗi cửa sổ
        for window in windows:
            # 1. Tính trung bình động và độ lệch chuẩn
            ma = price_data.rolling(window=window).mean()
            std = price_data.rolling(window=window).std()
            
            if use_log_scale:
                result_df[f"{prefix}log_ma_{window}"] = ma
            else:
                result_df[f"{prefix}ma_{window}"] = ma
            
            # 2. Tính khoảng cách đến trung bình (z-score)
            # Tránh chia cho 0
            std_non_zero = std.replace(0, np.nan)
            z_score = (price_data - ma) / std_non_zero
            
            result_df[f"{prefix}z_score_{window}"] = z_score
            
            # 3. Tính chỉ báo hồi quy về giá trị trung bình
            
            # Mức độ quá mua/bán (overbought/oversold)
            result_df[f"{prefix}overbought_{window}"] = (z_score > 1.0).astype(int)
            result_df[f"{prefix}oversold_{window}"] = (z_score < -1.0).astype(int)
            
            # Mức độ quá mua/bán nghiêm trọng (severe)
            result_df[f"{prefix}severe_overbought_{window}"] = (z_score > 2.0).astype(int)
            result_df[f"{prefix}severe_oversold_{window}"] = (z_score < -2.0).astype(int)
            
            # Tín hiệu đảo chiều hồi quy về giá trị trung bình
            result_df[f"{prefix}mean_reversion_signal_{window}"] = (
                (z_score.shift(1) > 1.0) & (z_score < z_score.shift(1)) |  # Đảo chiều từ quá mua
                (z_score.shift(1) < -1.0) & (z_score > z_score.shift(1))   # Đảo chiều từ quá bán
            ).astype(int)
            
            # 4. Tạo các dải giá dựa trên độ lệch chuẩn (tương tự Bollinger Bands)
            for band in bands:
                result_df[f"{prefix}upper_band_{window}_{band}"] = ma + std * band
                result_df[f"{prefix}lower_band_{window}_{band}"] = ma - std * band
                
                # Chỉ báo chạm dải (1 nếu giá chạm dải, 0 nếu không)
                result_df[f"{prefix}touch_upper_{window}_{band}"] = (
                    (price_data >= result_df[f"{prefix}upper_band_{window}_{band}"]) &
                    (price_data.shift(1) < result_df[f"{prefix}upper_band_{window}_{band}"].shift(1))
                ).astype(int)
                
                result_df[f"{prefix}touch_lower_{window}_{band}"] = (
                    (price_data <= result_df[f"{prefix}lower_band_{window}_{band}"]) &
                    (price_data.shift(1) > result_df[f"{prefix}lower_band_{window}_{band}"].shift(1))
                ).astype(int)
            
            # 5. Tính tốc độ hồi quy về giá trị trung bình
            result_df[f"{prefix}mean_reversion_velocity_{window}"] = (
                (ma - price_data) / std_non_zero
            )
            
            # 6. Tính tốc độ thay đổi của z-score (đạo hàm)
            result_df[f"{prefix}z_score_change_{window}"] = z_score - z_score.shift(1)
            
            # 7. Tính chỉ số Half-Life hồi quy về giá trị trung bình
            # Half-Life = -log(2) / log(1 - AR(1))
            # Trong đó AR(1) là hệ số tự hồi quy bậc 1
            try:
                # Tính hệ số tự hồi quy bằng cách hồi quy z_score(t) trên z_score(t-1)
                # Sử dụng cửa sổ di động để ước lượng AR(1) trong từng giai đoạn
                ar1_values = []
                
                for i in range(window, len(z_score)):
                    if i < window * 2:
                        ar1_values.append(np.nan)
                        continue
                    
                    y = z_score.iloc[i-window:i].values
                    x = z_score.iloc[i-window:i].shift(1).dropna().values
                    
                    if len(x) > 0 and np.var(x) > 0:
                        # Tính hệ số AR(1)
                        ar1 = np.cov(y[1:], x)[0, 1] / np.var(x)
                        
                        # Tính Half-Life
                        if ar1 < 1 and ar1 != 0:  # Đảm bảo AR(1) hợp lệ
                            half_life = -np.log(2) / np.log(1 - ar1)
                            ar1_values.append(half_life)
                        else:
                            ar1_values.append(np.nan)
                    else:
                        ar1_values.append(np.nan)
                
                # Thêm giá trị NaN cho các điểm đầu tiên
                ar1_values = [np.nan] * window + ar1_values
                
                # Thêm vào DataFrame
                result_df[f"{prefix}mean_reversion_half_life_{window}"] = pd.Series(
                    ar1_values[:len(result_df)], index=result_df.index
                )
                
                # Clip half-life để tránh giá trị cực đoan
                result_df[f"{prefix}mean_reversion_half_life_{window}"] = result_df[
                    f"{prefix}mean_reversion_half_life_{window}"
                ].clip(lower=1, upper=window*2)
            
            except Exception as e:
                logger.warning(f"Không thể tính half-life hồi quy: {e}")
            
            # 8. Tính chỉ số hội tụ/phân kỳ (Convergence/Divergence)
            result_df[f"{prefix}price_ma_convergence_{window}"] = (
                np.abs(z_score) < np.abs(z_score.shift(1))
            ).astype(int)
            
            result_df[f"{prefix}price_ma_divergence_{window}"] = (
                np.abs(z_score) > np.abs(z_score.shift(1))
            ).astype(int)
            
            # 9. Tính độ dốc của đường trung bình (Slope)
            result_df[f"{prefix}ma_slope_{window}"] = (ma - ma.shift(window//5)) / (window//5)
            
            # Chuẩn hóa độ dốc
            std_slope = result_df[f"{prefix}ma_slope_{window}"].rolling(window=window).std()
            std_slope_non_zero = std_slope.replace(0, np.nan)
            result_df[f"{prefix}normalized_ma_slope_{window}"] = (
                result_df[f"{prefix}ma_slope_{window}"] / std_slope_non_zero
            )
        
        logger.debug("Đã tính đặc trưng hồi quy về giá trị trung bình")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng hồi quy về giá trị trung bình: {e}")
    
    return result_df

def calculate_trend_strength_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    windows: List[int] = [20, 50, 100, 200],
    use_multiple_mas: bool = True,
    adx_windows: List[int] = [14, 20],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng về sức mạnh xu hướng.
    
    Args:
        df: DataFrame chứa dữ liệu giá OHLC
        price_column: Tên cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ cho MA
        use_multiple_mas: Sử dụng nhiều đường MA để đánh giá xu hướng
        adx_windows: Danh sách các kích thước cửa sổ cho ADX
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng sức mạnh xu hướng
    """
    required_columns = [price_column]
    use_adx = False
    
    if 'high' in df.columns and 'low' in df.columns:
        required_columns.extend(['high', 'low'])
        use_adx = True
    
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # 1. Tính các đường trung bình động
        for window in windows:
            # SMA (Simple Moving Average)
            sma = result_df[price_column].rolling(window=window).mean()
            result_df[f"{prefix}sma_{window}"] = sma
            
            # EMA (Exponential Moving Average)
            ema = result_df[price_column].ewm(span=window, adjust=False).mean()
            result_df[f"{prefix}ema_{window}"] = ema
            
            # HMA (Hull Moving Average) = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
            n_2 = max(2, window // 2)
            sqrt_n = max(2, int(np.sqrt(window)))
            
            # Tính WMA cho n/2
            weights_half = np.arange(1, n_2 + 1)
            wma_half = result_df[price_column].rolling(window=n_2).apply(
                lambda x: np.sum(weights_half * x[-len(weights_half):]) / np.sum(weights_half), raw=True
            )
            
            # Tính WMA cho n
            weights_full = np.arange(1, window + 1)
            wma_full = result_df[price_column].rolling(window=window).apply(
                lambda x: np.sum(weights_full * x[-len(weights_full):]) / np.sum(weights_full), raw=True
            )
            
            # Tính 2*WMA(n/2) - WMA(n)
            hma_inner = 2 * wma_half - wma_full
            
            # Tính WMA cuối cùng với sqrt(n)
            weights_sqrt = np.arange(1, sqrt_n + 1)
            hma_padded = pd.Series(np.nan, index=result_df.index)
            
            for i in range(sqrt_n - 1, len(hma_inner)):
                if i - sqrt_n + 1 >= 0:
                    segment = hma_inner.iloc[i-sqrt_n+1:i+1].values
                    if not np.isnan(segment).any():
                        hma_padded.iloc[i] = np.sum(weights_sqrt * segment) / np.sum(weights_sqrt)
            
            result_df[f"{prefix}hma_{window}"] = hma_padded
            
            # 2. Tính khoảng cách giá đến các đường MA
            
            # Khoảng cách tương đối (%) giá đến SMA
            sma_non_zero = sma.replace(0, np.nan)
            result_df[f"{prefix}price_to_sma_pct_{window}"] = (
                (result_df[price_column] - sma) / sma_non_zero * 100
            )
            
            # Khoảng cách tương đối (%) giá đến EMA
            ema_non_zero = ema.replace(0, np.nan)
            result_df[f"{prefix}price_to_ema_pct_{window}"] = (
                (result_df[price_column] - ema) / ema_non_zero * 100
            )
            
            # 3. Tính độ dốc của các đường MA
            
            # Độ dốc SMA (% thay đổi trên 5 thanh)
            lookback = max(1, window // 10)  # 10% của window
            result_df[f"{prefix}sma_slope_{window}"] = (
                (sma - sma.shift(lookback)) / sma.shift(lookback) * 100
            )
            
            # Độ dốc EMA (% thay đổi trên 5 thanh)
            result_df[f"{prefix}ema_slope_{window}"] = (
                (ema - ema.shift(lookback)) / ema.shift(lookback) * 100
            )
            
            # 4. Tính chỉ báo vị trí giá với MA
            
            # Giá trên SMA
            result_df[f"{prefix}price_above_sma_{window}"] = (
                result_df[price_column] > sma
            ).astype(int)
            
            # Giá trên EMA
            result_df[f"{prefix}price_above_ema_{window}"] = (
                result_df[price_column] > ema
            ).astype(int)
            
            # Giá cắt qua SMA từ dưới lên
            result_df[f"{prefix}price_crossover_sma_{window}"] = (
                (result_df[price_column] > sma) & 
                (result_df[price_column].shift(1) <= sma.shift(1))
            ).astype(int)
            
            # Giá cắt qua SMA từ trên xuống
            result_df[f"{prefix}price_crossunder_sma_{window}"] = (
                (result_df[price_column] < sma) & 
                (result_df[price_column].shift(1) >= sma.shift(1))
            ).astype(int)
        
        # 5. Tính chỉ báo sức mạnh xu hướng dựa trên sự sắp xếp của nhiều đường MA
        if use_multiple_mas and len(windows) >= 3:
            # Sắp xếp các cửa sổ từ nhỏ đến lớn
            sorted_windows = sorted(windows)
            
            # Chỉ báo xu hướng tăng mạnh
            strong_uptrend = pd.Series(1, index=result_df.index)
            
            # Chỉ báo xu hướng giảm mạnh
            strong_downtrend = pd.Series(1, index=result_df.index)
            
            # Kiểm tra sự sắp xếp của các đường MA
            for i in range(len(sorted_windows) - 1):
                small_window = sorted_windows[i]
                large_window = sorted_windows[i+1]
                
                # Xu hướng tăng: MA(nhỏ) > MA(lớn)
                strong_uptrend &= (
                    result_df[f"{prefix}ema_{small_window}"] > result_df[f"{prefix}ema_{large_window}"]
                )
                
                # Xu hướng giảm: MA(nhỏ) < MA(lớn)
                strong_downtrend &= (
                    result_df[f"{prefix}ema_{small_window}"] < result_df[f"{prefix}ema_{large_window}"]
                )
            
            # Lưu kết quả
            result_df[f"{prefix}strong_uptrend"] = strong_uptrend.astype(int)
            result_df[f"{prefix}strong_downtrend"] = strong_downtrend.astype(int)
            
            # Tính thêm chỉ báo xu hướng tổng hợp (-1 đến 1)
            trend_score = pd.Series(0, index=result_df.index)
            
            for window in sorted_windows:
                # Thêm điểm cho mỗi MA nếu giá ở trên MA (xu hướng tăng), -1 nếu ở dưới (xu hướng giảm)
                trend_score += np.where(
                    result_df[f"{prefix}price_above_ema_{window}"],
                    1 / len(sorted_windows),
                    -1 / len(sorted_windows)
                )
            
            result_df[f"{prefix}trend_score"] = trend_score
        
        # 6. Tính ADX nếu có dữ liệu high, low
        if use_adx:
            for window in adx_windows:
                # Tính True Range
                high = result_df['high']
                low = result_df['low']
                close = result_df[price_column]
                
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Tính +DM và -DM
                high_diff = high.diff()
                low_diff = low.diff().multiply(-1)
                
                plus_dm = np.where(
                    (high_diff > low_diff) & (high_diff > 0),
                    high_diff,
                    0
                )
                
                minus_dm = np.where(
                    (low_diff > high_diff) & (low_diff > 0),
                    low_diff,
                    0
                )
                
                # Tính TR14, +DM14, -DM14 với Wilder's smoothing
                tr14 = tr.rolling(window=window).mean()
                plus_dm14 = pd.Series(plus_dm).rolling(window=window).mean()
                minus_dm14 = pd.Series(minus_dm).rolling(window=window).mean()
                
                # Tính +DI14 và -DI14
                tr14_non_zero = tr14.replace(0, np.nan)
                plus_di14 = (plus_dm14 / tr14_non_zero) * 100
                minus_di14 = (minus_dm14 / tr14_non_zero) * 100
                
                # Tính Directional Index (DX)
                di_diff = np.abs(plus_di14 - minus_di14)
                di_sum = plus_di14 + minus_di14
                di_sum_non_zero = di_sum.replace(0, np.nan)
                
                dx = (di_diff / di_sum_non_zero) * 100
                
                # Tính ADX (Average Directional Index)
                adx = dx.rolling(window=window).mean()
                
                # Lưu kết quả
                result_df[f"{prefix}adx_{window}"] = adx
                result_df[f"{prefix}plus_di_{window}"] = plus_di14
                result_df[f"{prefix}minus_di_{window}"] = minus_di14
                
                # Tính chỉ báo hướng xu hướng dựa trên ADX và DI
                
                # Xu hướng mạnh (ADX > 25)
                result_df[f"{prefix}strong_trend_{window}"] = (adx > 25).astype(int)
                
                # Xu hướng rất mạnh (ADX > 50)
                result_df[f"{prefix}very_strong_trend_{window}"] = (adx > 50).astype(int)
                
                # Xu hướng tăng (+DI > -DI)
                result_df[f"{prefix}adx_uptrend_{window}"] = (plus_di14 > minus_di14).astype(int)
                
                # Xu hướng giảm (-DI > +DI)
                result_df[f"{prefix}adx_downtrend_{window}"] = (minus_di14 > plus_di14).astype(int)
                
                # Giao cắt +DI và -DI (Tín hiệu chuyển đổi xu hướng)
                result_df[f"{prefix}di_crossover_{window}"] = (
                    (plus_di14 > minus_di14) & (plus_di14.shift(1) <= minus_di14.shift(1))
                ).astype(int)
                
                result_df[f"{prefix}di_crossunder_{window}"] = (
                    (plus_di14 < minus_di14) & (plus_di14.shift(1) >= minus_di14.shift(1))
                ).astype(int)
        
        logger.debug("Đã tính đặc trưng sức mạnh xu hướng")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng sức mạnh xu hướng: {e}")
    
    return result_df

def calculate_price_pattern_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: Optional[str] = 'volume',
    high_column: Optional[str] = 'high',
    low_column: Optional[str] = 'low',
    window: int = 20,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng mẫu hình giá.
    
    Args:
        df: DataFrame chứa dữ liệu giá OHLC
        price_column: Tên cột giá đóng cửa
        volume_column: Tên cột khối lượng
        high_column: Tên cột giá cao
        low_column: Tên cột giá thấp
        window: Kích thước cửa sổ để xác định mẫu hình
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng mẫu hình giá
    """
    required_columns = [price_column]
    use_ohlc = False
    use_volume = False
    
    if high_column and low_column and high_column in df.columns and low_column in df.columns:
        required_columns.extend([high_column, low_column])
        use_ohlc = True
    
    if volume_column and volume_column in df.columns:
        required_columns.append(volume_column)
        use_volume = True
    
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # 1. Phát hiện mẫu hình nến
        if use_ohlc and 'open' in df.columns:
            # Lấy dữ liệu
            open_data = result_df['open']
            close_data = result_df[price_column]
            high_data = result_df[high_column]
            low_data = result_df[low_column]
            
            # Tính thân nến và bóng nến
            body = close_data - open_data
            body_abs = body.abs()
            upper_shadow = high_data - np.maximum(open_data, close_data)
            lower_shadow = np.minimum(open_data, close_data) - low_data
            
            # Tính kích thước nến trung bình
            avg_body = body_abs.rolling(window=window).mean()
            avg_range = (high_data - low_data).rolling(window=window).mean()
            
            # Phát hiện các mẫu hình nến cơ bản
            
            # 1.1. Nến Doji (thân rất nhỏ)
            result_df[f"{prefix}doji"] = (
                (body_abs < 0.1 * avg_body) &
                (body_abs < 0.1 * (high_data - low_data))
            ).astype(int)
            
            # 1.2. Nến búa (Hammer) và búa ngược (Inverted Hammer)
            result_df[f"{prefix}hammer"] = (
                (body_abs < 0.5 * (high_data - low_data)) &
                (lower_shadow > 2 * body_abs) &
                (upper_shadow < 0.2 * (high_data - low_data)) &
                (body < 0)  # Nến giảm
            ).astype(int)
            
            result_df[f"{prefix}inverted_hammer"] = (
                (body_abs < 0.5 * (high_data - low_data)) &
                (upper_shadow > 2 * body_abs) &
                (lower_shadow < 0.2 * (high_data - low_data)) &
                (body < 0)  # Nến giảm
            ).astype(int)
            
            # 1.3. Nến Shooting Star và Hanging Man
            result_df[f"{prefix}shooting_star"] = (
                (body_abs < 0.5 * (high_data - low_data)) &
                (upper_shadow > 2 * body_abs) &
                (lower_shadow < 0.2 * (high_data - low_data)) &
                (body > 0) &  # Nến tăng
                (close_data.shift(1) < open_data)  # Sau một nến tăng
            ).astype(int)
            
            result_df[f"{prefix}hanging_man"] = (
                (body_abs < 0.5 * (high_data - low_data)) &
                (lower_shadow > 2 * body_abs) &
                (upper_shadow < 0.2 * (high_data - low_data)) &
                (body > 0) &  # Nến tăng
                (close_data.shift(1) > open_data)  # Sau một nến giảm
            ).astype(int)
            
            # 1.4. Nến nuốt chửng (Engulfing)
            result_df[f"{prefix}bullish_engulfing"] = (
                (body > 0) &  # Nến tăng
                (body.shift(1) < 0) &  # Nến trước giảm
                (open_data < close_data.shift(1)) &  # Mở cửa thấp hơn đóng cửa trước
                (close_data > open_data.shift(1))  # Đóng cửa cao hơn mở cửa trước
            ).astype(int)
            
            result_df[f"{prefix}bearish_engulfing"] = (
                (body < 0) &  # Nến giảm
                (body.shift(1) > 0) &  # Nến trước tăng
                (open_data > close_data.shift(1)) &  # Mở cửa cao hơn đóng cửa trước
                (close_data < open_data.shift(1))  # Đóng cửa thấp hơn mở cửa trước
            ).astype(int)
            
            # 1.5. Nến bao phủ (Harami)
            result_df[f"{prefix}bullish_harami"] = (
                (body > 0) &  # Nến tăng
                (body.shift(1) < 0) &  # Nến trước giảm
                (open_data > close_data.shift(1)) &  # Mở cửa cao hơn đóng cửa trước
                (close_data < open_data.shift(1)) &  # Đóng cửa thấp hơn mở cửa trước
                (body_abs < body_abs.shift(1))  # Thân nhỏ hơn nến trước
            ).astype(int)
            
            result_df[f"{prefix}bearish_harami"] = (
                (body < 0) &  # Nến giảm
                (body.shift(1) > 0) &  # Nến trước tăng
                (open_data < close_data.shift(1)) &  # Mở cửa thấp hơn đóng cửa trước
                (close_data > open_data.shift(1)) &  # Đóng cửa cao hơn mở cửa trước
                (body_abs < body_abs.shift(1))  # Thân nhỏ hơn nến trước
            ).astype(int)
            
            # 1.6. Nến sao mai, sao hôm (Morning Star, Evening Star)
            for i in range(2, len(result_df)):
                # Morning Star
                if (
                    result_df[price_column].iloc[i-2] < result_df['open'].iloc[i-2] and  # Nến giảm đầu tiên
                    abs(result_df[price_column].iloc[i-1] - result_df['open'].iloc[i-1]) < 0.3 * avg_body.iloc[i-1] and  # Nến doji ở giữa
                    result_df[price_column].iloc[i] > result_df['open'].iloc[i] and  # Nến tăng cuối cùng
                    result_df[price_column].iloc[i] > result_df['open'].iloc[i-2]  # Đóng cửa cao hơn mở cửa đầu tiên
                ):
                    result_df.loc[result_df.index[i], f"{prefix}morning_star"] = 1
                else:
                    result_df.loc[result_df.index[i], f"{prefix}morning_star"] = 0
                
                # Evening Star
                if (
                    result_df[price_column].iloc[i-2] > result_df['open'].iloc[i-2] and  # Nến tăng đầu tiên
                    abs(result_df[price_column].iloc[i-1] - result_df['open'].iloc[i-1]) < 0.3 * avg_body.iloc[i-1] and  # Nến doji ở giữa
                    result_df[price_column].iloc[i] < result_df['open'].iloc[i] and  # Nến giảm cuối cùng
                    result_df[price_column].iloc[i] < result_df['open'].iloc[i-2]  # Đóng cửa thấp hơn mở cửa đầu tiên
                ):
                    result_df.loc[result_df.index[i], f"{prefix}evening_star"] = 1
                else:
                    result_df.loc[result_df.index[i], f"{prefix}evening_star"] = 0
        
        # 2. Phát hiện mẫu hình xu hướng
        if use_ohlc:
            # 2.1. Tính các đỉnh và đáy cục bộ
            
            # Tìm các đỉnh cục bộ (yêu cầu ít nhất 2 thanh ở mỗi bên)
            peak_condition = (
                (high_data > high_data.shift(2)) &
                (high_data > high_data.shift(1)) &
                (high_data > high_data.shift(-1)) &
                (high_data > high_data.shift(-2))
            )
            
            # Tìm các đáy cục bộ (yêu cầu ít nhất 2 thanh ở mỗi bên)
            trough_condition = (
                (low_data < low_data.shift(2)) &
                (low_data < low_data.shift(1)) &
                (low_data < low_data.shift(-1)) &
                (low_data < low_data.shift(-2))
            )
            
            # Lưu các đỉnh và đáy cục bộ
            result_df[f"{prefix}is_peak"] = peak_condition.astype(int)
            result_df[f"{prefix}is_trough"] = trough_condition.astype(int)
            
            # 2.2. Phát hiện các mẫu hình phổ biến
            
            # Tìm các đỉnh trong cửa sổ
            rolling_window = min(window, len(result_df))
            
            # Xác định mẫu hình Đầu và Vai (Head and Shoulders)
            for i in range(rolling_window, len(result_df)):
                window_df = result_df.iloc[i-rolling_window:i]
                peaks = window_df[window_df[f"{prefix}is_peak"] == 1]
                
                if len(peaks) >= 3:
                    peak_prices = peaks[high_column].values
                    peak_indices = peaks.index
                    
                    # Kiểm tra đỉnh giữa cao hơn 2 đỉnh bên cạnh
                    if len(peak_prices) >= 3:
                        middle_peak = np.argmax(peak_prices)
                        
                        if 0 < middle_peak < len(peak_prices) - 1:
                            left_peak = peak_prices[middle_peak - 1]
                            right_peak = peak_prices[middle_peak + 1]
                            
                            # Đỉnh giữa cao hơn và 2 đỉnh bên tương đương nhau
                            if (
                                peak_prices[middle_peak] > left_peak and
                                peak_prices[middle_peak] > right_peak and
                                abs(left_peak - right_peak) / max(left_peak, right_peak) < 0.05
                            ):
                                result_df.loc[result_df.index[i-1], f"{prefix}head_and_shoulders"] = 1
                                continue
                
                result_df.loc[result_df.index[i-1], f"{prefix}head_and_shoulders"] = 0
                
                # Inverse Head and Shoulders (mẫu hình đáy)
                troughs = window_df[window_df[f"{prefix}is_trough"] == 1]
                
                if len(troughs) >= 3:
                    trough_prices = troughs[low_column].values
                    trough_indices = troughs.index
                    
                    # Kiểm tra đáy giữa thấp hơn 2 đáy bên cạnh
                    if len(trough_prices) >= 3:
                        middle_trough = np.argmin(trough_prices)
                        
                        if 0 < middle_trough < len(trough_prices) - 1:
                            left_trough = trough_prices[middle_trough - 1]
                            right_trough = trough_prices[middle_trough + 1]
                            
                            # Đáy giữa thấp hơn và 2 đáy bên tương đương nhau
                            if (
                                trough_prices[middle_trough] < left_trough and
                                trough_prices[middle_trough] < right_trough and
                                abs(left_trough - right_trough) / min(left_trough, right_trough) < 0.05
                            ):
                                result_df.loc[result_df.index[i-1], f"{prefix}inverse_head_and_shoulders"] = 1
                                continue
                
                result_df.loc[result_df.index[i-1], f"{prefix}inverse_head_and_shoulders"] = 0
                
                # Double Top (mẫu hình đỉnh đôi)
                if len(peaks) >= 2:
                    peak_prices = peaks[high_column].values
                    peak_indices = peaks.index
                    
                    # Kiểm tra 2 đỉnh gần nhau
                    if len(peak_prices) >= 2:
                        # Lấy 2 đỉnh cuối cùng
                        last_peak = peak_prices[-1]
                        second_last_peak = peak_prices[-2]
                        
                        # 2 đỉnh gần bằng nhau
                        if abs(last_peak - second_last_peak) / max(last_peak, second_last_peak) < 0.03:
                            # Tìm đáy giữa 2 đỉnh
                            between_troughs = troughs[
                                (troughs.index > peak_indices[-2]) & 
                                (troughs.index < peak_indices[-1])
                            ]
                            
                            if len(between_troughs) > 0:
                                result_df.loc[result_df.index[i-1], f"{prefix}double_top"] = 1
                                continue
                
                result_df.loc[result_df.index[i-1], f"{prefix}double_top"] = 0
                
                # Double Bottom (mẫu hình đáy đôi)
                if len(troughs) >= 2:
                    trough_prices = troughs[low_column].values
                    trough_indices = troughs.index
                    
                    # Kiểm tra 2 đáy gần nhau
                    if len(trough_prices) >= 2:
                        # Lấy 2 đáy cuối cùng
                        last_trough = trough_prices[-1]
                        second_last_trough = trough_prices[-2]
                        
                        # 2 đáy gần bằng nhau
                        if abs(last_trough - second_last_trough) / min(last_trough, second_last_trough) < 0.03:
                            # Tìm đỉnh giữa 2 đáy
                            between_peaks = peaks[
                                (peaks.index > trough_indices[-2]) & 
                                (peaks.index < trough_indices[-1])
                            ]
                            
                            if len(between_peaks) > 0:
                                result_df.loc[result_df.index[i-1], f"{prefix}double_bottom"] = 1
                                continue
                
                result_df.loc[result_df.index[i-1], f"{prefix}double_bottom"] = 0
        
        # 3. Phát hiện mẫu hình giá-khối lượng
        if use_volume and use_ohlc:
            volume_data = result_df[volume_column]
            
            # Tính khối lượng trung bình
            avg_volume = volume_data.rolling(window=window).mean()
            
            # 3.1. Phân kỳ giá-khối lượng
            
            # Tính returns
            returns = result_df[price_column].pct_change()
            
            # Phân kỳ dương (giá giảm nhưng khối lượng tăng)
            result_df[f"{prefix}bullish_divergence"] = (
                (returns < 0) &  # Giá giảm
                (volume_data > volume_data.shift(1)) &  # Khối lượng tăng
                (volume_data > 1.5 * avg_volume)  # Khối lượng cao
            ).astype(int)
            
            # Phân kỳ âm (giá tăng nhưng khối lượng giảm)
            result_df[f"{prefix}bearish_divergence"] = (
                (returns > 0) &  # Giá tăng
                (volume_data < volume_data.shift(1)) &  # Khối lượng giảm
                (returns > returns.rolling(window=window).mean())  # Return cao hơn trung bình
            ).astype(int)
            
            # 3.2. Mẫu hình bùng nổ khối lượng (Volume Climax)
            result_df[f"{prefix}volume_climax"] = (
                (volume_data > 2.0 * avg_volume) &  # Khối lượng gấp đôi trung bình
                (
                    (abs(returns) > 2.0 * returns.abs().rolling(window=window).mean()) |  # Biến động giá lớn
                    (result_df[high_column] == result_df[high_column].rolling(window=window).max()) |  # Đạt mức cao mới
                    (result_df[low_column] == result_df[low_column].rolling(window=window).min())  # Đạt mức thấp mới
                )
            ).astype(int)
            
            # 3.3. Mẫu hình khối lượng thấp bất thường
            result_df[f"{prefix}unusually_low_volume"] = (
                (volume_data < 0.5 * avg_volume) &  # Khối lượng thấp hơn một nửa trung bình
                (abs(returns) < 0.5 * returns.abs().rolling(window=window).mean())  # Biến động giá thấp
            ).astype(int)
        
        # 4. Tạo đặc trưng mẫu hình tổng hợp
        
        # 4.1. Tính số lượng mẫu hình bullish và bearish trong cửa sổ gần đây
        pattern_columns = [col for col in result_df.columns if col.startswith(f"{prefix}")]
        
        # Xác định các mẫu hình bullish
        bullish_patterns = [
            f"{prefix}hammer", f"{prefix}bullish_engulfing", f"{prefix}bullish_harami",
            f"{prefix}morning_star", f"{prefix}inverse_head_and_shoulders", f"{prefix}double_bottom",
            f"{prefix}bullish_divergence"
        ]
        
        # Xác định các mẫu hình bearish
        bearish_patterns = [
            f"{prefix}shooting_star", f"{prefix}hanging_man", f"{prefix}bearish_engulfing",
            f"{prefix}bearish_harami", f"{prefix}evening_star", f"{prefix}head_and_shoulders",
            f"{prefix}double_top", f"{prefix}bearish_divergence"
        ]
        
        # Tính số lượng mẫu hình bullish và bearish
        for w in [5, 10]:
            # Chỉ tính nếu có đủ dữ liệu
            if len(result_df) >= w:
                # Các cột mẫu hình bullish có trong DataFrame
                valid_bullish = [col for col in bullish_patterns if col in result_df.columns]
                
                # Các cột mẫu hình bearish có trong DataFrame
                valid_bearish = [col for col in bearish_patterns if col in result_df.columns]
                
                if valid_bullish:
                    # Tính tổng mẫu hình bullish trong cửa sổ
                    bullish_sum = result_df[valid_bullish].rolling(window=w).sum().sum(axis=1)
                    result_df[f"{prefix}bullish_patterns_{w}"] = bullish_sum
                
                if valid_bearish:
                    # Tính tổng mẫu hình bearish trong cửa sổ
                    bearish_sum = result_df[valid_bearish].rolling(window=w).sum().sum(axis=1)
                    result_df[f"{prefix}bearish_patterns_{w}"] = bearish_sum
                
                # Tính chỉ số xu hướng mẫu hình (Pattern Trend Index)
                if valid_bullish and valid_bearish:
                    pattern_trend = (bullish_sum - bearish_sum) / (bullish_sum + bearish_sum).replace(0, 1)
                    result_df[f"{prefix}pattern_trend_index_{w}"] = pattern_trend
        
        logger.debug("Đã tính đặc trưng mẫu hình giá")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng mẫu hình giá: {e}")
    
    return result_df

def calculate_market_regime_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: Optional[str] = 'volume',
    windows: List[int] = [20, 50, 100],
    num_regimes: int = 3,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng về chế độ thị trường.
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        windows: Danh sách các kích thước cửa sổ
        num_regimes: Số lượng chế độ thị trường cần phân loại
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng chế độ thị trường
    """
    required_columns = [price_column]
    use_volume = False
    
    if volume_column and volume_column in df.columns:
        required_columns.append(volume_column)
        use_volume = True
    
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # 1. Tính các đặc trưng biến động và xu hướng
        
        # Tính returns
        returns = result_df[price_column].pct_change()
        result_df[f"{prefix}returns"] = returns
        
        for window in windows:
            # 1.1. Biến động (độ lệch chuẩn của returns)
            volatility = returns.rolling(window=window).std()
            result_df[f"{prefix}volatility_{window}"] = volatility
            
            # 1.2. Xu hướng (slope của giá)
            # Tính SMA
            sma = result_df[price_column].rolling(window=window).mean()
            
            # Độ dốc SMA (% thay đổi trên 5 thanh)
            lookback = max(1, window // 5)
            slope = (sma - sma.shift(lookback)) / sma.shift(lookback) * 100
            result_df[f"{prefix}trend_slope_{window}"] = slope
            
            # 1.3. Tính chỉ số sức mạnh tương đối (RSI)
            # Tính biến động tăng/giảm
            up = returns.clip(lower=0)
            down = -1 * returns.clip(upper=0)
            
            # Tính trung bình di động của biến động tăng/giảm
            up_avg = up.rolling(window=window).mean()
            down_avg = down.rolling(window=window).mean()
            
            # Tránh chia cho 0
            down_avg_non_zero = down_avg.replace(0, np.nan)
            
            # Tính RS và RSI
            rs = up_avg / down_avg_non_zero
            rsi = 100 - (100 / (1 + rs))
            
            result_df[f"{prefix}rsi_{window}"] = rsi
            
            # 1.4. Kết hợp biến động và xu hướng để phân loại chế độ
            
            # Chuẩn hóa biến động (z-score)
            vol_mean = volatility.rolling(window=window*2).mean()
            vol_std = volatility.rolling(window=window*2).std()
            
            # Tránh chia cho 0
            vol_std_non_zero = vol_std.replace(0, np.nan)
            vol_z = (volatility - vol_mean) / vol_std_non_zero
            
            result_df[f"{prefix}volatility_regime_{window}"] = pd.cut(
                vol_z,
                bins=[-float('inf'), -0.5, 0.5, float('inf')],
                labels=['low', 'normal', 'high']
            )
            
            # Chuẩn hóa độ dốc xu hướng
            slope_mean = slope.rolling(window=window*2).mean()
            slope_std = slope.rolling(window=window*2).std()
            
            # Tránh chia cho 0
            slope_std_non_zero = slope_std.replace(0, np.nan)
            slope_z = (slope - slope_mean) / slope_std_non_zero
            
            result_df[f"{prefix}trend_regime_{window}"] = pd.cut(
                slope_z,
                bins=[-float('inf'), -0.5, 0.5, float('inf')],
                labels=['downtrend', 'neutral', 'uptrend']
            )
            
            # 1.5. Kết hợp biến động và xu hướng thành chế độ thị trường
            
            # Chế độ thị trường dựa trên biến động và xu hướng
            vol_regime = result_df[f"{prefix}volatility_regime_{window}"]
            trend_regime = result_df[f"{prefix}trend_regime_{window}"]
            
            # Tạo chế độ thị trường kết hợp
            market_regime = pd.Series(index=result_df.index, dtype='object')
            
            # Điền các chế độ
            market_regime[(vol_regime == 'low') & (trend_regime == 'neutral')] = 'sideways_low_vol'
            market_regime[(vol_regime == 'normal') & (trend_regime == 'neutral')] = 'sideways'
            market_regime[(vol_regime == 'high') & (trend_regime == 'neutral')] = 'sideways_high_vol'
            
            market_regime[(vol_regime == 'low') & (trend_regime == 'uptrend')] = 'uptrend_low_vol'
            market_regime[(vol_regime == 'normal') & (trend_regime == 'uptrend')] = 'uptrend'
            market_regime[(vol_regime == 'high') & (trend_regime == 'uptrend')] = 'uptrend_high_vol'
            
            market_regime[(vol_regime == 'low') & (trend_regime == 'downtrend')] = 'downtrend_low_vol'
            market_regime[(vol_regime == 'normal') & (trend_regime == 'downtrend')] = 'downtrend'
            market_regime[(vol_regime == 'high') & (trend_regime == 'downtrend')] = 'downtrend_high_vol'
            
            result_df[f"{prefix}market_regime_{window}"] = market_regime
            
            # 1.6. Mã hóa One-Hot cho chế độ thị trường
            for regime in market_regime.dropna().unique():
                result_df[f"{prefix}regime_{regime}_{window}"] = (market_regime == regime).astype(int)
            
            # 1.7. Bộ chỉ báo chuyển đổi chế độ
            result_df[f"{prefix}regime_change_{window}"] = (
                market_regime != market_regime.shift(1)
            ).astype(int)
            
            # Thời gian kể từ lần thay đổi chế độ cuối cùng
            regime_changes = result_df[f"{prefix}regime_change_{window}"]
            time_since_change = pd.Series(0, index=result_df.index)
            
            count = 0
            for i in range(len(regime_changes)):
                if regime_changes.iloc[i] == 1:
                    count = 0
                else:
                    count += 1
                time_since_change.iloc[i] = count
            
            result_df[f"{prefix}time_since_regime_change_{window}"] = time_since_change
        
        # 2. Phát hiện thị trường tăng/giảm (Bull/Bear Markets)
        
        # Tính đỉnh và đáy lớn
        for window in windows:
            # Tính giá cao nhất và thấp nhất trong cửa sổ
            rolling_max = result_df[price_column].rolling(window=window).max()
            rolling_min = result_df[price_column].rolling(window=window).min()
            
            # Giảm từ đỉnh (Drawdown)
            drawdown = (result_df[price_column] - rolling_max) / rolling_max * 100
            result_df[f"{prefix}drawdown_{window}"] = drawdown
            
            # Tăng từ đáy (Drawup)
            drawup = (result_df[price_column] - rolling_min) / rolling_min * 100
            result_df[f"{prefix}drawup_{window}"] = drawup
            
            # Phát hiện thị trường giảm (Bear Market) - Drawdown > 20%
            result_df[f"{prefix}bear_market_{window}"] = (drawdown <= -20).astype(int)
            
            # Phát hiện thị trường tăng (Bull Market) - Drawup > 20%
            result_df[f"{prefix}bull_market_{window}"] = (drawup >= 20).astype(int)
            
            # Phát hiện thị trường đi ngang (Sideways) - Drawdown > -10% và Drawup < 10%
            result_df[f"{prefix}sideways_market_{window}"] = (
                (drawdown > -10) & (drawup < 10)
            ).astype(int)
        
        # 3. Phân tích theo khối lượng (nếu có)
        if use_volume:
            # Tính khối lượng tương đối (so với trung bình)
            for window in windows:
                avg_volume = result_df[volume_column].rolling(window=window).mean()
                rel_volume = result_df[volume_column] / avg_volume
                
                result_df[f"{prefix}relative_volume_{window}"] = rel_volume
                
                # Tính độ tương quan giữa khối lượng và biến động giá
                abs_returns = returns.abs()
                vol_ret_corr = abs_returns.rolling(window=window).corr(rel_volume)
                
                result_df[f"{prefix}volume_volatility_correlation_{window}"] = vol_ret_corr
                
                # Phân loại chế độ khối lượng
                result_df[f"{prefix}volume_regime_{window}"] = pd.cut(
                    rel_volume,
                    bins=[0, 0.5, 1.5, float('inf')],
                    labels=['low', 'normal', 'high']
                )
                
                # Mã hóa One-Hot cho chế độ khối lượng
                for regime in ['low', 'normal', 'high']:
                    result_df[f"{prefix}volume_{regime}_{window}"] = (
                        result_df[f"{prefix}volume_regime_{window}"] == regime
                    ).astype(int)
        
        logger.debug("Đã tính đặc trưng chế độ thị trường")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng chế độ thị trường: {e}")
    
    return result_df

def calculate_event_impact_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: Optional[str] = 'volume',
    event_windows: List[int] = [1, 3, 5, 10],
    reference_windows: List[int] = [20, 50],
    thresholds: Dict[str, float] = {
        'price': 0.03,  # 3%
        'volume': 2.0,  # 200%
        'volatility': 2.0  # 200%
    },
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng về tác động của sự kiện thị trường.
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        event_windows: Danh sách các kích thước cửa sổ sau sự kiện
        reference_windows: Danh sách các kích thước cửa sổ tham chiếu
        thresholds: Ngưỡng xác định sự kiện (price: biến động giá, volume: tăng khối lượng, volatility: tăng biến động)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng tác động sự kiện
    """
    required_columns = [price_column]
    use_volume = False
    
    if volume_column and volume_column in df.columns:
        required_columns.append(volume_column)
        use_volume = True
    
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    try:
        # 1. Tính các đặc trưng cơ bản
        
        # Returns hàng ngày
        returns = result_df[price_column].pct_change()
        result_df[f"{prefix}returns"] = returns
        
        # Biến động hàng ngày
        daily_volatility = returns.rolling(window=5).std()
        result_df[f"{prefix}daily_volatility"] = daily_volatility
        
        # 2. Phát hiện các sự kiện lớn
        
        for ref_window in reference_windows:
            # 2.1. Phát hiện sự kiện biến động giá lớn
            
            # Tính trung bình và độ lệch chuẩn của return
            avg_returns = returns.rolling(window=ref_window).mean()
            std_returns = returns.rolling(window=ref_window).std()
            
            # Tính z-score của return
            returns_z = (returns - avg_returns) / std_returns.replace(0, np.nan)
            
            # Phát hiện sự kiện tăng giá mạnh (Positive price shock)
            result_df[f"{prefix}positive_price_shock_{ref_window}"] = (
                returns_z > thresholds['price']
            ).astype(int)
            
            # Phát hiện sự kiện giảm giá mạnh (Negative price shock)
            result_df[f"{prefix}negative_price_shock_{ref_window}"] = (
                returns_z < -thresholds['price']
            ).astype(int)
            
            # 2.2. Phát hiện sự kiện biến động lớn
            
            # Tính trung bình và độ lệch chuẩn của biến động
            avg_volatility = daily_volatility.rolling(window=ref_window).mean()
            std_volatility = daily_volatility.rolling(window=ref_window).std()
            
            # Tính z-score của biến động
            volatility_z = (daily_volatility - avg_volatility) / std_volatility.replace(0, np.nan)
            
            # Phát hiện sự kiện biến động lớn (Volatility shock)
            result_df[f"{prefix}volatility_shock_{ref_window}"] = (
                volatility_z > thresholds['volatility']
            ).astype(int)
            
            # 2.3. Phát hiện sự kiện khối lượng lớn (nếu có dữ liệu khối lượng)
            if use_volume:
                # Tính trung bình và độ lệch chuẩn của khối lượng
                avg_volume = result_df[volume_column].rolling(window=ref_window).mean()
                std_volume = result_df[volume_column].rolling(window=ref_window).std()
                
                # Tính z-score của khối lượng
                volume_z = (result_df[volume_column] - avg_volume) / std_volume.replace(0, np.nan)
                
                # Phát hiện sự kiện khối lượng lớn (Volume shock)
                result_df[f"{prefix}volume_shock_{ref_window}"] = (
                    volume_z > thresholds['volume']
                ).astype(int)
            
            # 2.4. Phát hiện các sự kiện phức hợp
            
            # Sự kiện tăng giá kèm khối lượng lớn (Bullish event)
            if use_volume:
                result_df[f"{prefix}bullish_event_{ref_window}"] = (
                    (returns_z > thresholds['price']) &
                    (volume_z > thresholds['volume'])
                ).astype(int)
                
                # Sự kiện giảm giá kèm khối lượng lớn (Bearish event)
                result_df[f"{prefix}bearish_event_{ref_window}"] = (
                    (returns_z < -thresholds['price']) &
                    (volume_z > thresholds['volume'])
                ).astype(int)
            
            # Sự kiện biến động lớn không có xu hướng rõ ràng (Volatile event)
            result_df[f"{prefix}volatile_event_{ref_window}"] = (
                (volatility_z > thresholds['volatility']) &
                (abs(returns_z) < thresholds['price'] / 2)
            ).astype(int)
        
        # 3. Phân tích tác động của sự kiện
        
        for ref_window in reference_windows:
            for event_window in event_windows:
                if event_window >= ref_window:
                    continue  # Bỏ qua trường hợp cửa sổ sự kiện lớn hơn cửa sổ tham chiếu
                
                # 3.1. Tính hiệu suất sau sự kiện tăng giá
                pos_event_col = f"{prefix}positive_price_shock_{ref_window}"
                if pos_event_col in result_df.columns:
                    # Tính hiệu suất tích lũy trong n ngày sau sự kiện
                    cumulative_returns = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[pos_event_col].iloc[i] == 1:
                            # Tính hiệu suất tích lũy trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            cumulative_returns.iloc[i] = (1 + ret_series).prod() - 1
                    
                    result_df[f"{prefix}pos_event_return_{ref_window}_{event_window}"] = cumulative_returns
                    
                    # Tính biến động sau sự kiện
                    post_event_volatility = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[pos_event_col].iloc[i] == 1:
                            # Tính độ lệch chuẩn của returns trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            post_event_volatility.iloc[i] = ret_series.std()
                    
                    result_df[f"{prefix}pos_event_volatility_{ref_window}_{event_window}"] = post_event_volatility
                
                # 3.2. Tính hiệu suất sau sự kiện giảm giá
                neg_event_col = f"{prefix}negative_price_shock_{ref_window}"
                if neg_event_col in result_df.columns:
                    # Tính hiệu suất tích lũy trong n ngày sau sự kiện
                    cumulative_returns = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[neg_event_col].iloc[i] == 1:
                            # Tính hiệu suất tích lũy trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            cumulative_returns.iloc[i] = (1 + ret_series).prod() - 1
                    
                    result_df[f"{prefix}neg_event_return_{ref_window}_{event_window}"] = cumulative_returns
                    
                    # Tính biến động sau sự kiện
                    post_event_volatility = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[neg_event_col].iloc[i] == 1:
                            # Tính độ lệch chuẩn của returns trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            post_event_volatility.iloc[i] = ret_series.std()
                    
                    result_df[f"{prefix}neg_event_volatility_{ref_window}_{event_window}"] = post_event_volatility
                
                # 3.3. Tính hiệu suất sau sự kiện biến động
                vol_event_col = f"{prefix}volatility_shock_{ref_window}"
                if vol_event_col in result_df.columns:
                    # Tính hiệu suất tích lũy trong n ngày sau sự kiện
                    cumulative_returns = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[vol_event_col].iloc[i] == 1:
                            # Tính hiệu suất tích lũy trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            cumulative_returns.iloc[i] = (1 + ret_series).prod() - 1
                    
                    result_df[f"{prefix}vol_event_return_{ref_window}_{event_window}"] = cumulative_returns
                    
                    # Tính biến động sau sự kiện
                    post_event_volatility = pd.Series(index=result_df.index)
                    for i in range(len(result_df) - event_window):
                        if result_df[vol_event_col].iloc[i] == 1:
                            # Tính độ lệch chuẩn của returns trong n ngày tiếp theo
                            ret_series = result_df[f"{prefix}returns"].iloc[i+1:i+event_window+1]
                            post_event_volatility.iloc[i] = ret_series.std()
                    
                    result_df[f"{prefix}vol_event_volatility_{ref_window}_{event_window}"] = post_event_volatility
        
        logger.debug("Đã tính đặc trưng tác động sự kiện")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tác động sự kiện: {e}")
    
    return result_df

def calculate_correlation_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    reference_dfs: Dict[str, pd.DataFrame] = {},
    reference_columns: Dict[str, str] = {},
    windows: List[int] = [20, 50, 100],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng tương quan với các tài sản/chỉ số tham chiếu.
    
    Args:
        df: DataFrame chứa dữ liệu giá của tài sản chính
        price_column: Tên cột giá sử dụng để tính toán
        reference_dfs: Dict các DataFrame tham chiếu, key là tên tài sản
        reference_columns: Dict các cột giá trong DataFrame tham chiếu
        windows: Danh sách các kích thước cửa sổ tương quan
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng tương quan
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Kiểm tra các DataFrame tham chiếu
    for ref_name, ref_df in reference_dfs.items():
        ref_col = reference_columns.get(ref_name, price_column)
        
        if ref_col not in ref_df.columns:
            logger.warning(f"Không tìm thấy cột {ref_col} trong DataFrame tham chiếu {ref_name}")
    
    try:
        # 1. Tính returns của tài sản chính
        main_returns = result_df[price_column].pct_change()
        result_df[f"{prefix}returns"] = main_returns
        
        # 2. Tính returns và tương quan với từng tài sản tham chiếu
        for ref_name, ref_df in reference_dfs.items():
            ref_col = reference_columns.get(ref_name, price_column)
            
            if ref_col not in ref_df.columns:
                continue
            
            # Tính returns của tài sản tham chiếu
            ref_returns = ref_df[ref_col].pct_change()
            
            # Điều chỉnh index của reference DataFrame để khớp với main DataFrame
            if ref_df.index.equals(result_df.index):
                aligned_ref_returns = ref_returns
            else:
                # Sử dụng reindex để căn chỉnh chuỗi thời gian
                aligned_ref_returns = ref_returns.reindex(result_df.index)
            
            # Lưu returns của tài sản tham chiếu
            result_df[f"{prefix}{ref_name}_returns"] = aligned_ref_returns
            
            # Tính tương quan rolling
            for window in windows:
                # Tính tương quan Pearson
                correlation = main_returns.rolling(window=window).corr(aligned_ref_returns)
                result_df[f"{prefix}correlation_{ref_name}_{window}"] = correlation
                
                # Tính tương quan beta (hệ số hồi quy)
                cov = main_returns.rolling(window=window).cov(aligned_ref_returns)
                ref_var = aligned_ref_returns.rolling(window=window).var()
                
                # Tránh chia cho 0
                ref_var_non_zero = ref_var.replace(0, np.nan)
                beta = cov / ref_var_non_zero
                
                result_df[f"{prefix}beta_{ref_name}_{window}"] = beta
                
                # Tính chỉ số tương quan thay đổi
                corr_change = correlation - correlation.shift(window//5)
                result_df[f"{prefix}correlation_change_{ref_name}_{window}"] = corr_change
                
                # Phân loại trạng thái tương quan
                result_df[f"{prefix}high_correlation_{ref_name}_{window}"] = (correlation > 0.6).astype(int)
                result_df[f"{prefix}negative_correlation_{ref_name}_{window}"] = (correlation < -0.3).astype(int)
                result_df[f"{prefix}no_correlation_{ref_name}_{window}"] = (
                    (correlation >= -0.3) & (correlation <= 0.3)
                ).astype(int)
                
                # Phát hiện phân kỳ tương quan (correlation divergence)
                # Phân kỳ: giá tăng nhưng tương quan giảm hoặc ngược lại
                result_df[f"{prefix}correlation_divergence_{ref_name}_{window}"] = (
                    (np.sign(main_returns.rolling(window=window//2).mean()) != 
                     np.sign(corr_change))
                ).astype(int)
        
        # 3. Tính ma trận tương quan giữa các tài sản tham chiếu
        if len(reference_dfs) >= 2:
            ref_names = list(reference_dfs.keys())
            
            for window in windows:
                for i, ref1 in enumerate(ref_names):
                    for j, ref2 in enumerate(ref_names[i+1:], i+1):
                        ref1_col = reference_columns.get(ref1, price_column)
                        ref2_col = reference_columns.get(ref2, price_column)
                        
                        if (ref1_col in reference_dfs[ref1].columns and 
                            ref2_col in reference_dfs[ref2].columns):
                            
                            # Lấy returns của hai tài sản tham chiếu
                            ref1_returns = result_df[f"{prefix}{ref1}_returns"]
                            ref2_returns = result_df[f"{prefix}{ref2}_returns"]
                            
                            # Tính tương quan giữa chúng
                            cross_correlation = ref1_returns.rolling(window=window).corr(ref2_returns)
                            result_df[f"{prefix}correlation_{ref1}_{ref2}_{window}"] = cross_correlation
        
        # 4. Tính chỉ số tương quan tổng hợp (basket correlation)
        if len(reference_dfs) >= 2:
            for window in windows:
                # Tính trung bình các hệ số tương quan
                corr_columns = [col for col in result_df.columns if col.startswith(f"{prefix}correlation_") and col.endswith(f"_{window}")]
                
                if corr_columns:
                    avg_correlation = result_df[corr_columns].mean(axis=1)
                    result_df[f"{prefix}avg_correlation_{window}"] = avg_correlation
                    
                    # Chỉ số phân tán tương quan (độ lệch chuẩn của các tương quan)
                    corr_dispersion = result_df[corr_columns].std(axis=1)
                    result_df[f"{prefix}correlation_dispersion_{window}"] = corr_dispersion
                    
                    # Phân loại chế độ tương quan thị trường
                    # High: tương quan trung bình > 0.5
                    # Low: tương quan trung bình < 0.3
                    # Normal: còn lại
                    result_df[f"{prefix}high_correlation_regime_{window}"] = (avg_correlation > 0.5).astype(int)
                    result_df[f"{prefix}low_correlation_regime_{window}"] = (avg_correlation < 0.3).astype(int)
                    result_df[f"{prefix}normal_correlation_regime_{window}"] = (
                        (avg_correlation >= 0.3) & (avg_correlation <= 0.5)
                    ).astype(int)
        
        logger.debug("Đã tính đặc trưng tương quan")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tương quan: {e}")
    
    return result_df