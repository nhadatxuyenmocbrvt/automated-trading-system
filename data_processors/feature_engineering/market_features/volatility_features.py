"""
Đặc trưng về biến động.
Mô-đun này cung cấp các hàm tạo đặc trưng dựa trên biến động thị trường,
bao gồm biến động lịch sử, biến động tương đối, và các mẫu hình biến động.
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
from data_processors.feature_engineering.technical_indicators.utils import validate_price_data, true_range

# Logger
logger = setup_logger("volatility_features")

def calculate_volatility_features(
    df: pd.DataFrame, 
    price_column: str = 'close',
    windows: List[int] = [5, 10, 20, 50, 100],
    annualize: bool = True,
    trading_periods: int = 365,
    use_log_returns: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán nhiều đặc trưng biến động từ chuỗi giá.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        windows: Danh sách các kích thước cửa sổ
        annualize: Chuẩn hóa biến động theo năm
        trading_periods: Số phiên giao dịch trong một năm
        use_log_returns: Sử dụng log returns thay vì phần trăm returns
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng biến động
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Tính returns
    if use_log_returns:
        returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
    else:
        returns = result_df[price_column].pct_change()
    
    for window in windows:
        try:
            # Biến động cơ bản (độ lệch chuẩn của returns)
            volatility = returns.rolling(window=window).std()
            
            # Chuẩn hóa theo năm nếu cần
            if annualize:
                volatility = volatility * np.sqrt(trading_periods)
                result_df[f"{prefix}annualized_volatility_{window}"] = volatility
            else:
                result_df[f"{prefix}volatility_{window}"] = volatility
            
            # Biến động chuẩn hóa (biến động hiện tại / biến động trung bình dài hạn)
            long_term_volatility = returns.rolling(window=max(window*5, 50)).std()
            result_df[f"{prefix}normalized_volatility_{window}"] = volatility / long_term_volatility
            
            # Thay đổi biến động (biến động hiện tại / biến động trước đó)
            result_df[f"{prefix}volatility_change_{window}"] = volatility / volatility.shift(window//2)
            
            # Chỉ số biến động cao bất thường (z-score của biến động hiện tại)
            vol_mean = volatility.rolling(window=window*3).mean()
            vol_std = volatility.rolling(window=window*3).std()
            
            # Tránh chia cho 0
            vol_std = vol_std.replace(0, np.nan)
            result_df[f"{prefix}volatility_zscore_{window}"] = (volatility - vol_mean) / vol_std
            
            # Phân vị biến động (thứ hạng phần trăm hiện tại trong phân phối lịch sử)
            result_df[f"{prefix}volatility_rank_{window}"] = (
                volatility.rolling(window=window*3).rank(pct=True)
            )
            
            # Biến động dựa trên giá trị high-low
            if all(col in df.columns for col in ['high', 'low']):
                # Biến động Parkinson
                high_low_ratio = np.log(result_df['high'] / result_df['low'])
                parkinson_vol = np.sqrt(high_low_ratio.rolling(window=window).apply(
                    lambda x: np.sum(x**2) / (4 * np.log(2) * window)
                ))
                
                if annualize:
                    parkinson_vol = parkinson_vol * np.sqrt(trading_periods)
                
                result_df[f"{prefix}parkinson_volatility_{window}"] = parkinson_vol
            
            logger.debug(f"Đã tính đặc trưng biến động cơ bản cho cửa sổ {window}")
        
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng biến động cơ bản cho cửa sổ {window}: {e}")
    
    # Tính các đặc trưng nâng cao từ returns
    try:
        # Tính biến động không cân xứng (biến động của returns dương và âm)
        returns_pos = returns.copy()
        returns_pos[returns < 0] = np.nan
        
        returns_neg = returns.copy()
        returns_neg[returns > 0] = np.nan
        
        for window in windows:
            upside_vol = returns_pos.rolling(window=window).std()
            downside_vol = returns_neg.rolling(window=window).std()
            
            if annualize:
                upside_vol = upside_vol * np.sqrt(trading_periods)
                downside_vol = downside_vol * np.sqrt(trading_periods)
            
            result_df[f"{prefix}upside_volatility_{window}"] = upside_vol
            result_df[f"{prefix}downside_volatility_{window}"] = downside_vol
            
            # Tỷ lệ biến động up/down (>1 nếu biến động tăng mạnh hơn giảm)
            downside_vol_non_zero = downside_vol.replace(0, np.nan)
            result_df[f"{prefix}volatility_ratio_{window}"] = upside_vol / downside_vol_non_zero
        
        # Tính biến động thực hiện (realized volatility)
        squared_returns = returns ** 2
        for window in windows:
            realized_vol = np.sqrt(squared_returns.rolling(window=window).sum() / window)
            
            if annualize:
                realized_vol = realized_vol * np.sqrt(trading_periods)
            
            result_df[f"{prefix}realized_volatility_{window}"] = realized_vol
        
        logger.debug("Đã tính đặc trưng biến động nâng cao")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng biến động nâng cao: {e}")
    
    return result_df

def calculate_relative_volatility(
    df: pd.DataFrame,
    price_column: str = 'close',
    atr_windows: List[int] = [5, 10, 14, 20],
    normalize: bool = True,
    reference_window: int = 100,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng biến động tương đối dựa trên ATR.
    
    Args:
        df: DataFrame chứa dữ liệu giá OHLC
        price_column: Tên cột giá sử dụng để tính toán
        atr_windows: Danh sách các kích thước cửa sổ cho ATR
        normalize: Chuẩn hóa các giá trị biến động
        reference_window: Cửa sổ tham chiếu để chuẩn hóa
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng biến động tương đối
    """
    required_columns = ['high', 'low', 'close']
    if not validate_price_data(df, required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
        return df
    
    result_df = df.copy()
    
    # Tính True Range
    tr = true_range(result_df['high'], result_df['low'], result_df['close'])
    
    for window in atr_windows:
        try:
            # Tính ATR
            atr = tr.rolling(window=window).mean()
            
            # Chuẩn hóa ATR theo giá hiện tại (ATR %)
            price_non_zero = result_df[price_column].replace(0, np.nan)
            atr_percent = (atr / price_non_zero) * 100
            
            result_df[f"{prefix}atr_{window}"] = atr
            result_df[f"{prefix}atr_pct_{window}"] = atr_percent
            
            if normalize:
                # Chuẩn hóa ATR so với một ATR dài hạn
                atr_long = tr.rolling(window=reference_window).mean()
                atr_long_non_zero = atr_long.replace(0, np.nan)
                atr_relative = atr / atr_long_non_zero
                
                result_df[f"{prefix}relative_atr_{window}"] = atr_relative
                
                # Xếp hạng ATR hiện tại trong lịch sử
                atr_rank = atr.rolling(window=reference_window).rank(pct=True)
                result_df[f"{prefix}atr_rank_{window}"] = atr_rank
            
            # Tính sự thay đổi của ATR
            atr_change = atr / atr.shift(window//2) - 1
            result_df[f"{prefix}atr_change_{window}"] = atr_change
            
            # Phát hiện các đỉnh và đáy của ATR (biến động quá mức)
            atr_zscore = (atr - atr.rolling(window=reference_window).mean()) / atr.rolling(window=reference_window).std()
            result_df[f"{prefix}atr_zscore_{window}"] = atr_zscore
            
            # Đánh dấu các thời điểm biến động cao bất thường (ATR z-score > 2)
            result_df[f"{prefix}high_volatility_{window}"] = (atr_zscore > 2).astype(int)
            
            # Đánh dấu các thời điểm biến động thấp bất thường (ATR z-score < -1)
            result_df[f"{prefix}low_volatility_{window}"] = (atr_zscore < -1).astype(int)
            
            logger.debug(f"Đã tính đặc trưng biến động tương đối cho cửa sổ ATR {window}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng biến động tương đối cho cửa sổ ATR {window}: {e}")
    
    return result_df

def calculate_volatility_ratio(
    df: pd.DataFrame,
    price_column: str = 'close',
    short_windows: List[int] = [5, 10],
    long_windows: List[int] = [20, 50, 100],
    use_log_returns: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính tỷ lệ biến động giữa các cửa sổ thời gian khác nhau.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        short_windows: Danh sách các kích thước cửa sổ ngắn hạn
        long_windows: Danh sách các kích thước cửa sổ dài hạn
        use_log_returns: Sử dụng log returns thay vì phần trăm returns
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa tỷ lệ biến động
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Tính returns
    if use_log_returns:
        returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
    else:
        returns = result_df[price_column].pct_change()
    
    try:
        # Tính biến động cho mỗi cửa sổ
        volatilities = {}
        
        for window in short_windows + long_windows:
            volatilities[window] = returns.rolling(window=window).std()
        
        # Tính tỷ lệ biến động giữa các cửa sổ ngắn và dài
        for short_window in short_windows:
            for long_window in long_windows:
                if short_window < long_window:
                    # Tránh chia cho 0
                    long_vol_non_zero = volatilities[long_window].replace(0, np.nan)
                    
                    # Tỷ lệ biến động = Biến động ngắn hạn / Biến động dài hạn
                    vol_ratio = volatilities[short_window] / long_vol_non_zero
                    
                    result_df[f"{prefix}vol_ratio_{short_window}_{long_window}"] = vol_ratio
                    
                    # Chuẩn hóa tỷ lệ biến động
                    vol_ratio_mean = vol_ratio.rolling(window=long_window).mean()
                    vol_ratio_std = vol_ratio.rolling(window=long_window).std()
                    
                    # Tránh chia cho 0
                    vol_ratio_std_non_zero = vol_ratio_std.replace(0, np.nan)
                    
                    vol_ratio_zscore = (vol_ratio - vol_ratio_mean) / vol_ratio_std_non_zero
                    
                    result_df[f"{prefix}vol_ratio_zscore_{short_window}_{long_window}"] = vol_ratio_zscore
                    
                    # Phát hiện các trạng thái biến động đặc biệt
                    
                    # 1. Biến động tăng tốc (tỷ lệ > 1.5 và tăng so với trước đó)
                    result_df[f"{prefix}vol_acceleration_{short_window}_{long_window}"] = (
                        (vol_ratio > 1.5) & 
                        (vol_ratio > vol_ratio.shift(1))
                    ).astype(int)
                    
                    # 2. Biến động thu hẹp (tỷ lệ < 0.75 và giảm so với trước đó)
                    result_df[f"{prefix}vol_contraction_{short_window}_{long_window}"] = (
                        (vol_ratio < 0.75) & 
                        (vol_ratio < vol_ratio.shift(1))
                    ).astype(int)
                    
                    # 3. Chuyển đổi biến động (biến động thay đổi từ tăng sang giảm hoặc ngược lại)
                    result_df[f"{prefix}vol_regime_change_{short_window}_{long_window}"] = (
                        (np.sign(vol_ratio - 1) != np.sign(vol_ratio.shift(1) - 1))
                    ).astype(int)
            
        logger.debug("Đã tính đặc trưng tỷ lệ biến động")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tỷ lệ biến động: {e}")
    
    return result_df

def calculate_volatility_patterns(
    df: pd.DataFrame,
    price_column: str = 'close',
    vol_windows: List[int] = [10, 20, 50],
    lookback_periods: List[int] = [5, 10, 20],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phát hiện các mẫu hình biến động như biến động tăng dần, giảm dần hoặc phun trào.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        vol_windows: Danh sách các kích thước cửa sổ tính biến động
        lookback_periods: Danh sách các giai đoạn để phát hiện mẫu hình
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng mẫu hình biến động
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Tính log returns
    returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
    
    for window in vol_windows:
        try:
            # Tính biến động rolling
            volatility = returns.rolling(window=window).std()
            
            # Thêm vào DataFrame
            result_df[f"vol_{window}"] = volatility
            
            for period in lookback_periods:
                if period < window:
                    # 1. Phát hiện mẫu hình biến động tăng dần
                    
                    # Tính slope của biến động qua n phiên gần nhất
                    x = np.arange(period)
                    vol_slope = pd.Series(index=volatility.index)
                    
                    for i in range(period, len(volatility)):
                        y = volatility.iloc[i-period:i].values
                        slope, _ = np.polyfit(x, y, 1)
                        vol_slope.iloc[i] = slope
                    
                    # Chuẩn hóa slope
                    vol_slope = vol_slope / volatility
                    
                    # Mẫu hình biến động tăng dần (slope > 0 trong n phiên)
                    result_df[f"{prefix}increasing_vol_{window}_{period}"] = (
                        vol_slope > 0
                    ).astype(int)
                    
                    # Mẫu hình biến động giảm dần (slope < 0 trong n phiên)
                    result_df[f"{prefix}decreasing_vol_{window}_{period}"] = (
                        vol_slope < 0
                    ).astype(int)
                    
                    # 2. Phát hiện mẫu hình biến động phun trào (volatility explosion)
                    
                    # Tính sự thay đổi phần trăm của biến động
                    vol_change = (volatility / volatility.shift(period) - 1) * 100
                    
                    # Mẫu hình biến động phun trào (thay đổi > 50%)
                    result_df[f"{prefix}volatility_explosion_{window}_{period}"] = (
                        vol_change > 50
                    ).astype(int)
                    
                    # Mẫu hình biến động thu hẹp mạnh (thay đổi < -30%)
                    result_df[f"{prefix}volatility_implosion_{window}_{period}"] = (
                        vol_change < -30
                    ).astype(int)
                    
                    # 3. Phát hiện mẫu hình biến động chu kỳ
                    
                    # Tính autocorrelation của biến động
                    vol_autocorr = pd.Series(index=volatility.index)
                    
                    for i in range(period * 2, len(volatility)):
                        series = volatility.iloc[i-period*2:i]
                        # Tính autocorrelation lag=period
                        vol_autocorr.iloc[i] = series.autocorr(lag=period)
                    
                    # Mẫu hình biến động có chu kỳ (autocorrelation > 0.7)
                    result_df[f"{prefix}cyclic_volatility_{window}_{period}"] = (
                        vol_autocorr > 0.7
                    ).astype(int)
            
            logger.debug(f"Đã tính đặc trưng mẫu hình biến động cho cửa sổ {window}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính đặc trưng mẫu hình biến động cho cửa sổ {window}: {e}")
    
    # Xóa các cột trung gian
    for window in vol_windows:
        if f"vol_{window}" in result_df.columns:
            del result_df[f"vol_{window}"]
    
    return result_df

def calculate_volatility_regime(
    df: pd.DataFrame,
    price_column: str = 'close',
    window: int = 20,
    long_window: int = 100,
    num_regimes: int = 3,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phân loại chế độ biến động hiện tại (thấp, trung bình, cao).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        window: Kích thước cửa sổ tính biến động
        long_window: Kích thước cửa sổ dài hạn để phân loại
        num_regimes: Số lượng chế độ biến động cần phân loại
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng chế độ biến động
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính log returns
        returns = np.log(result_df[price_column] / result_df[price_column].shift(1))
        
        # Tính biến động
        volatility = returns.rolling(window=window).std()
        
        # Phân loại chế độ biến động
        if long_window > 0 and len(volatility) > long_window:
            # Sử dụng phân vị lịch sử để phân loại
            vol_percentile = volatility.rolling(window=long_window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # Phân loại thành num_regimes chế độ
            bins = np.linspace(0, 1, num_regimes + 1)
            labels = list(range(1, num_regimes + 1))
            
            vol_regime = pd.cut(vol_percentile, bins=bins, labels=labels, include_lowest=True)
            result_df[f"{prefix}volatility_regime_{window}"] = vol_regime
            
            # Tạo các biến nhị phân cho mỗi chế độ
            for i in range(1, num_regimes + 1):
                result_df[f"{prefix}vol_regime_{i}_{window}"] = (vol_regime == i).astype(int)
            
            # Thay đổi chế độ biến động
            regime_change = (vol_regime != vol_regime.shift(1)).astype(int)
            result_df[f"{prefix}vol_regime_change_{window}"] = regime_change
            
            # Thời gian từ lần thay đổi chế độ gần nhất
            regime_change_idx = np.where(regime_change.values)[0]
            time_since_change = np.zeros(len(result_df))
            
            for i in range(len(result_df)):
                if i in regime_change_idx:
                    time_since_change[i] = 0
                elif i > 0:
                    time_since_change[i] = time_since_change[i-1] + 1
            
            result_df[f"{prefix}time_since_vol_regime_change_{window}"] = time_since_change
        
        else:
            # Sử dụng phân loại đơn giản dựa trên z-score
            vol_mean = volatility.mean()
            vol_std = volatility.std()
            
            if vol_std == 0:
                vol_std = 1e-8  # Tránh chia cho 0
            
            vol_zscore = (volatility - vol_mean) / vol_std
            
            # Phân loại: 1=thấp (<-0.5), 2=trung bình ([-0.5,0.5]), 3=cao (>0.5)
            vol_regime = pd.Series(index=volatility.index, dtype='int')
            vol_regime[(vol_zscore <= -0.5)] = 1  # Biến động thấp
            vol_regime[(vol_zscore > -0.5) & (vol_zscore <= 0.5)] = 2  # Biến động trung bình
            vol_regime[(vol_zscore > 0.5)] = 3  # Biến động cao
            
            result_df[f"{prefix}volatility_regime_{window}"] = vol_regime
            
            # Tạo các biến nhị phân cho mỗi chế độ
            result_df[f"{prefix}vol_regime_low_{window}"] = (vol_regime == 1).astype(int)
            result_df[f"{prefix}vol_regime_medium_{window}"] = (vol_regime == 2).astype(int)
            result_df[f"{prefix}vol_regime_high_{window}"] = (vol_regime == 3).astype(int)
            
            # Thay đổi chế độ biến động
            regime_change = (vol_regime != vol_regime.shift(1)).astype(int)
            result_df[f"{prefix}vol_regime_change_{window}"] = regime_change
        
        logger.debug(f"Đã tính đặc trưng chế độ biến động cho cửa sổ {window}")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng chế độ biến động: {e}")
    
    return result_df

def calculate_garch_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    p: int = 1,
    q: int = 1,
    forecast_periods: List[int] = [1, 5, 10],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng dự báo biến động dựa trên mô hình GARCH.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_column: Tên cột giá sử dụng để tính toán
        p: Bậc của thành phần ARCH (autoregressive)
        q: Bậc của thành phần GARCH (moving average)
        forecast_periods: Danh sách các khoảng thời gian dự báo
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa dự báo biến động từ GARCH
    """
    if not validate_price_data(df, [price_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return df
    
    result_df = df.copy()
    
    # Kiểm tra xem có thư viện arch hay không
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("Không thể import module arch. Cài đặt với 'pip install arch'")
        return result_df
    
    try:
        # Tính log returns
        returns = 100 * np.log(result_df[price_column] / result_df[price_column].shift(1))
        
        # Bỏ qua các giá trị NaN
        returns = returns.dropna()
        
        if len(returns) < 100:
            logger.warning("Không đủ dữ liệu để ước lượng mô hình GARCH (cần ít nhất 100 điểm)")
            return result_df
        
        # Ước lượng mô hình GARCH
        model = arch_model(returns, mean='Zero', vol='GARCH', p=p, q=q)
        
        # Giới hạn số lượng quan sát để tránh tính toán quá lâu
        max_obs = 1000 if len(returns) > 1000 else len(returns)
        
        model_fit = model.fit(disp='off', last_obs=max_obs)
        
        # Dự báo biến động
        forecasts = model_fit.forecast(horizon=max(forecast_periods))
        
        # Lấy variance forecast
        variance_forecasts = forecasts.variance.iloc[-1]
        
        # Thêm dự báo vào DataFrame kết quả
        for period in forecast_periods:
            if period <= max(forecast_periods):
                # Volatility forecast (standard deviation)
                vol_forecast = np.sqrt(variance_forecasts.iloc[period-1])
                
                # Thêm vào DataFrame
                last_idx = result_df.index[-1]
                result_df.loc[last_idx, f"{prefix}garch_vol_forecast_{period}"] = vol_forecast
                
                # Dự báo khoảng tin cậy cho biến động
                vol_upper = vol_forecast * 1.96  # 95% CI
                vol_lower = max(0, vol_forecast * 0.04)  # Đảm bảo không âm
                
                result_df.loc[last_idx, f"{prefix}garch_vol_upper_{period}"] = vol_upper
                result_df.loc[last_idx, f"{prefix}garch_vol_lower_{period}"] = vol_lower
        
        # Tính unconditional volatility từ tham số mô hình
        omega = model_fit.params['omega']
        alpha = model_fit.params[f'alpha[1]'] if f'alpha[1]' in model_fit.params else 0
        beta = model_fit.params[f'beta[1]'] if f'beta[1]' in model_fit.params else 0
        
        if alpha + beta < 1:  # Kiểm tra tính dừng
            unconditional_var = omega / (1 - alpha - beta)
            unconditional_vol = np.sqrt(unconditional_var)
            
            # Thêm vào DataFrame
            result_df[f"{prefix}garch_long_run_vol"] = unconditional_vol
            
            # Mức độ biến động hiện tại so với dài hạn
            current_var = model_fit.conditional_volatility[-1]**2
            volatility_ratio = current_var / unconditional_var
            
            result_df.loc[last_idx, f"{prefix}garch_vol_ratio"] = volatility_ratio
            
            # Phân loại chế độ biến động dựa trên tỷ lệ
            if volatility_ratio < 0.8:
                vol_regime = 1  # Thấp
            elif volatility_ratio < 1.2:
                vol_regime = 2  # Trung bình
            else:
                vol_regime = 3  # Cao
            
            result_df.loc[last_idx, f"{prefix}garch_vol_regime"] = vol_regime
        
        logger.debug("Đã tính đặc trưng dự báo biến động GARCH")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng GARCH: {e}")
    
    return result_df