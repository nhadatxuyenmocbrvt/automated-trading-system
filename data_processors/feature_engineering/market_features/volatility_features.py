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
    prefix: str = '',
    fill_method: str = 'bfill'  # Tham số mới để chỉ định cách xử lý NaN
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
        fill_method: Phương pháp xử lý NaN ('bfill', 'ffill', 'mean', 'median', 'zero')
        
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
    
    # Điền giá trị NaN đầu tiên cho returns
    if fill_method == 'bfill' or fill_method == 'backfill':
        returns = returns.fillna(method='bfill')
    elif fill_method == 'ffill' or fill_method == 'forward':
        returns = returns.fillna(method='ffill')
    elif fill_method == 'mean':
        returns = returns.fillna(returns.mean())
    elif fill_method == 'median':
        returns = returns.fillna(returns.median())
    elif fill_method == 'zero':
        returns = returns.fillna(0)
    
    for window in windows:
        try:
            # Biến động cơ bản (độ lệch chuẩn của returns)
            volatility = returns.rolling(window=window).std()
            
            # Xử lý NaN trong volatility
            if fill_method == 'bfill' or fill_method == 'backfill':
                volatility = volatility.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                volatility = volatility.fillna(method='ffill')
            elif fill_method == 'mean':
                volatility = volatility.fillna(volatility.mean())
            elif fill_method == 'median':
                volatility = volatility.fillna(volatility.median())
            elif fill_method == 'zero':
                volatility = volatility.fillna(0)
            
            # Chuẩn hóa theo năm nếu cần
            if annualize:
                volatility = volatility * np.sqrt(trading_periods)
                result_df[f"{prefix}annualized_volatility_{window}"] = volatility
            else:
                result_df[f"{prefix}volatility_{window}"] = volatility
            
            # Biến động chuẩn hóa (biến động hiện tại / biến động trung bình dài hạn)
            long_term_volatility = returns.rolling(window=max(window*5, 50)).std()
            
            # Xử lý NaN trong long_term_volatility
            if fill_method == 'bfill' or fill_method == 'backfill':
                long_term_volatility = long_term_volatility.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                long_term_volatility = long_term_volatility.fillna(method='ffill')
            elif fill_method == 'mean':
                long_term_volatility = long_term_volatility.fillna(long_term_volatility.mean())
            elif fill_method == 'median':
                long_term_volatility = long_term_volatility.fillna(long_term_volatility.median())
            elif fill_method == 'zero':
                long_term_volatility = long_term_volatility.fillna(0)
            
            # Tránh chia cho 0
            long_term_volatility_non_zero = long_term_volatility.replace(0, np.nan)
            if fill_method == 'zero':
                # Thay thế giá trị 0 bằng giá trị nhỏ nhất khác 0 hoặc 1e-8
                min_non_zero = long_term_volatility[long_term_volatility > 0].min() if len(long_term_volatility[long_term_volatility > 0]) > 0 else 1e-8
                long_term_volatility_non_zero = long_term_volatility_non_zero.fillna(min_non_zero)
            
            normalized_volatility = volatility / long_term_volatility_non_zero
            
            # Xử lý NaN trong normalized_volatility
            if fill_method == 'bfill' or fill_method == 'backfill':
                normalized_volatility = normalized_volatility.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                normalized_volatility = normalized_volatility.fillna(method='ffill')
            elif fill_method == 'mean':
                normalized_volatility = normalized_volatility.fillna(normalized_volatility.mean())
            elif fill_method == 'median':
                normalized_volatility = normalized_volatility.fillna(normalized_volatility.median())
            elif fill_method == 'zero':
                normalized_volatility = normalized_volatility.fillna(0)
            
            result_df[f"{prefix}normalized_volatility_{window}"] = normalized_volatility
            
            # Thay đổi biến động (biến động hiện tại / biến động trước đó)
            volatility_change = volatility / volatility.shift(window//2)
            
            # Xử lý NaN trong volatility_change
            if fill_method == 'bfill' or fill_method == 'backfill':
                volatility_change = volatility_change.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                volatility_change = volatility_change.fillna(method='ffill')
            elif fill_method == 'mean':
                volatility_change = volatility_change.fillna(volatility_change.mean())
            elif fill_method == 'median':
                volatility_change = volatility_change.fillna(volatility_change.median())
            elif fill_method == 'zero':
                volatility_change = volatility_change.fillna(0)
            
            result_df[f"{prefix}volatility_change_{window}"] = volatility_change
            
            # Chỉ số biến động cao bất thường (z-score của biến động hiện tại)
            vol_mean = volatility.rolling(window=window*3).mean()
            vol_std = volatility.rolling(window=window*3).std()
            
            # Xử lý NaN trong vol_mean và vol_std
            if fill_method == 'bfill' or fill_method == 'backfill':
                vol_mean = vol_mean.fillna(method='bfill')
                vol_std = vol_std.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                vol_mean = vol_mean.fillna(method='ffill')
                vol_std = vol_std.fillna(method='ffill')
            elif fill_method == 'mean':
                vol_mean = vol_mean.fillna(vol_mean.mean())
                vol_std = vol_std.fillna(vol_std.mean())
            elif fill_method == 'median':
                vol_mean = vol_mean.fillna(vol_mean.median())
                vol_std = vol_std.fillna(vol_std.median())
            elif fill_method == 'zero':
                vol_mean = vol_mean.fillna(0)
                vol_std = vol_std.fillna(0)
            
            # Tránh chia cho 0
            vol_std_non_zero = vol_std.replace(0, np.nan)
            if fill_method == 'zero':
                # Thay thế giá trị 0 bằng giá trị nhỏ nhất khác 0 hoặc 1e-8
                min_non_zero = vol_std[vol_std > 0].min() if len(vol_std[vol_std > 0]) > 0 else 1e-8
                vol_std_non_zero = vol_std_non_zero.fillna(min_non_zero)
            
            volatility_zscore = (volatility - vol_mean) / vol_std_non_zero
            
            # Xử lý NaN trong volatility_zscore
            if fill_method == 'bfill' or fill_method == 'backfill':
                volatility_zscore = volatility_zscore.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                volatility_zscore = volatility_zscore.fillna(method='ffill')
            elif fill_method == 'mean':
                volatility_zscore = volatility_zscore.fillna(volatility_zscore.mean())
            elif fill_method == 'median':
                volatility_zscore = volatility_zscore.fillna(volatility_zscore.median())
            elif fill_method == 'zero':
                volatility_zscore = volatility_zscore.fillna(0)
            
            result_df[f"{prefix}volatility_zscore_{window}"] = volatility_zscore
            
            # Phân vị biến động (thứ hạng phần trăm hiện tại trong phân phối lịch sử)
            volatility_rank = volatility.rolling(window=window*3).rank(pct=True)
            
            # Xử lý NaN trong volatility_rank
            if fill_method == 'bfill' or fill_method == 'backfill':
                volatility_rank = volatility_rank.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                volatility_rank = volatility_rank.fillna(method='ffill')
            elif fill_method == 'mean':
                volatility_rank = volatility_rank.fillna(volatility_rank.mean())
            elif fill_method == 'median':
                volatility_rank = volatility_rank.fillna(volatility_rank.median())
            elif fill_method == 'zero':
                volatility_rank = volatility_rank.fillna(0)
            
            result_df[f"{prefix}volatility_rank_{window}"] = volatility_rank
            
            # Biến động dựa trên giá trị high-low
            if all(col in df.columns for col in ['high', 'low']):
                # Biến động Parkinson
                high_low_ratio = np.log(result_df['high'] / result_df['low'])
                parkinson_vol = np.sqrt(high_low_ratio.rolling(window=window).apply(
                    lambda x: np.sum(x**2) / (4 * np.log(2) * window)
                ))
                
                # Xử lý NaN trong parkinson_vol
                if fill_method == 'bfill' or fill_method == 'backfill':
                    parkinson_vol = parkinson_vol.fillna(method='bfill')
                elif fill_method == 'ffill' or fill_method == 'forward':
                    parkinson_vol = parkinson_vol.fillna(method='ffill')
                elif fill_method == 'mean':
                    parkinson_vol = parkinson_vol.fillna(parkinson_vol.mean())
                elif fill_method == 'median':
                    parkinson_vol = parkinson_vol.fillna(parkinson_vol.median())
                elif fill_method == 'zero':
                    parkinson_vol = parkinson_vol.fillna(0)
                
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
            
            # Xử lý NaN trong upside_vol và downside_vol
            if fill_method == 'bfill' or fill_method == 'backfill':
                upside_vol = upside_vol.fillna(method='bfill')
                downside_vol = downside_vol.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                upside_vol = upside_vol.fillna(method='ffill')
                downside_vol = downside_vol.fillna(method='ffill')
            elif fill_method == 'mean':
                upside_vol = upside_vol.fillna(upside_vol.mean())
                downside_vol = downside_vol.fillna(downside_vol.mean())
            elif fill_method == 'median':
                upside_vol = upside_vol.fillna(upside_vol.median())
                downside_vol = downside_vol.fillna(downside_vol.median())
            elif fill_method == 'zero':
                upside_vol = upside_vol.fillna(0)
                downside_vol = downside_vol.fillna(0)
            
            if annualize:
                upside_vol = upside_vol * np.sqrt(trading_periods)
                downside_vol = downside_vol * np.sqrt(trading_periods)
            
            result_df[f"{prefix}upside_volatility_{window}"] = upside_vol
            result_df[f"{prefix}downside_volatility_{window}"] = downside_vol
            
            # Tỷ lệ biến động up/down (>1 nếu biến động tăng mạnh hơn giảm)
            downside_vol_non_zero = downside_vol.replace(0, np.nan)
            if fill_method == 'zero':
                # Thay thế giá trị 0 bằng giá trị nhỏ nhất khác 0 hoặc 1e-8
                min_non_zero = downside_vol[downside_vol > 0].min() if len(downside_vol[downside_vol > 0]) > 0 else 1e-8
                downside_vol_non_zero = downside_vol_non_zero.fillna(min_non_zero)
            
            volatility_ratio = upside_vol / downside_vol_non_zero
            
            # Xử lý NaN trong volatility_ratio
            if fill_method == 'bfill' or fill_method == 'backfill':
                volatility_ratio = volatility_ratio.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                volatility_ratio = volatility_ratio.fillna(method='ffill')
            elif fill_method == 'mean':
                volatility_ratio = volatility_ratio.fillna(volatility_ratio.mean())
            elif fill_method == 'median':
                volatility_ratio = volatility_ratio.fillna(volatility_ratio.median())
            elif fill_method == 'zero':
                volatility_ratio = volatility_ratio.fillna(0)
            
            result_df[f"{prefix}volatility_ratio_{window}"] = volatility_ratio
        
        # Tính biến động thực hiện (realized volatility)
        squared_returns = returns ** 2
        for window in windows:
            realized_vol = np.sqrt(squared_returns.rolling(window=window).sum() / window)
            
            # Xử lý NaN trong realized_vol
            if fill_method == 'bfill' or fill_method == 'backfill':
                realized_vol = realized_vol.fillna(method='bfill')
            elif fill_method == 'ffill' or fill_method == 'forward':
                realized_vol = realized_vol.fillna(method='ffill')
            elif fill_method == 'mean':
                realized_vol = realized_vol.fillna(realized_vol.mean())
            elif fill_method == 'median':
                realized_vol = realized_vol.fillna(realized_vol.median())
            elif fill_method == 'zero':
                realized_vol = realized_vol.fillna(0)
            
            if annualize:
                realized_vol = realized_vol * np.sqrt(trading_periods)
            
            result_df[f"{prefix}realized_volatility_{window}"] = realized_vol
        
        logger.debug("Đã tính đặc trưng biến động nâng cao")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng biến động nâng cao: {e}")
    
    # Kiểm tra NaN cuối cùng
    nan_columns = result_df.columns[result_df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"Vẫn còn {len(nan_columns)} cột chứa NaN sau khi xử lý: {nan_columns}")
        # Điền NaN cuối cùng nếu còn
        for col in nan_columns:
            result_df[col] = result_df[col].fillna(0)
        
        logger.info("Đã điền tất cả NaN còn lại bằng 0")
    
    return result_df

def calculate_relative_volatility(
    df: pd.DataFrame,
    price_column: str = 'close',
    atr_windows: List[int] = [5, 10, 14, 20],
    normalize: bool = True,
    reference_window: int = 100,
    prefix: str = '',
    fill_method: str = 'backfill'  # Thêm tham số mới
) -> pd.DataFrame:
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
            
            # LƯU Ý: THÊM XỬ LÝ NAN Ở ĐÂY
            # Điền giá trị NaN bằng phương pháp được chỉ định
            if fill_method == 'backfill' or fill_method == 'bfill':
                atr = atr.fillna(method='bfill')
            elif fill_method == 'forward' or fill_method == 'ffill':
                atr = atr.fillna(method='ffill')
            elif fill_method == 'mean':
                # Nếu tất cả là NaN, dùng 0 thay vì mean
                mean_val = atr.mean()
                atr = atr.fillna(0 if pd.isna(mean_val) else mean_val)
            elif fill_method == 'median':
                median_val = atr.median()
                atr = atr.fillna(0 if pd.isna(median_val) else median_val)
            elif fill_method == 'zero':
                atr = atr.fillna(0)
            else:
                # Mặc định dùng backfill
                atr = atr.fillna(method='bfill')
            
            # Chuẩn hóa ATR theo giá hiện tại (ATR %)
            price_non_zero = result_df[price_column].replace(0, np.nan)
            atr_percent = (atr / price_non_zero) * 100
            
            # Xử lý NaN trong atr_percent tương tự
            if fill_method in ['backfill', 'bfill']:
                atr_percent = atr_percent.fillna(method='bfill')
            elif fill_method in ['forward', 'ffill']:
                atr_percent = atr_percent.fillna(method='ffill')
            elif fill_method == 'mean':
                mean_val = atr_percent.mean()
                atr_percent = atr_percent.fillna(0 if pd.isna(mean_val) else mean_val)
            elif fill_method == 'median':
                median_val = atr_percent.median()
                atr_percent = atr_percent.fillna(0 if pd.isna(median_val) else median_val)
            elif fill_method == 'zero':
                atr_percent = atr_percent.fillna(0)
            else:
                atr_percent = atr_percent.fillna(method='bfill')
            
            result_df[f"{prefix}atr_{window}"] = atr
            result_df[f"{prefix}atr_pct_{window}"] = atr_percent
            
            if normalize:
                # Chuẩn hóa ATR so với một ATR dài hạn
                atr_long = tr.rolling(window=reference_window).mean()
                
                # Xử lý NaN trong atr_long
                if fill_method in ['backfill', 'bfill']:
                    atr_long = atr_long.fillna(method='bfill')
                elif fill_method in ['forward', 'ffill']:
                    atr_long = atr_long.fillna(method='ffill')
                elif fill_method == 'mean':
                    mean_val = atr_long.mean()
                    atr_long = atr_long.fillna(0 if pd.isna(mean_val) else mean_val)
                elif fill_method == 'median':
                    median_val = atr_long.median()
                    atr_long = atr_long.fillna(0 if pd.isna(median_val) else median_val)
                elif fill_method == 'zero':
                    atr_long = atr_long.fillna(0)
                else:
                    atr_long = atr_long.fillna(method='bfill')
                
                atr_long_non_zero = atr_long.replace(0, np.nan)
                # Thêm giá trị nhỏ để tránh chia cho 0
                atr_long_non_zero = atr_long_non_zero.fillna(atr_long.min() if not atr_long.empty else 1e-8)
                
                atr_relative = atr / atr_long_non_zero
                
                # Xử lý NaN trong atr_relative
                if fill_method in ['backfill', 'bfill']:
                    atr_relative = atr_relative.fillna(method='bfill')
                elif fill_method in ['forward', 'ffill']:
                    atr_relative = atr_relative.fillna(method='ffill')
                elif fill_method == 'mean':
                    mean_val = atr_relative.mean()
                    atr_relative = atr_relative.fillna(1 if pd.isna(mean_val) else mean_val)
                elif fill_method == 'median':
                    median_val = atr_relative.median()
                    atr_relative = atr_relative.fillna(1 if pd.isna(median_val) else median_val)
                elif fill_method == 'zero':
                    atr_relative = atr_relative.fillna(0)
                else:
                    atr_relative = atr_relative.fillna(method='bfill')
                
                result_df[f"{prefix}relative_atr_{window}"] = atr_relative
                
                # Xếp hạng ATR hiện tại trong lịch sử
                atr_rank = atr.rolling(window=reference_window).rank(pct=True)
                
                # Xử lý NaN trong atr_rank
                if fill_method in ['backfill', 'bfill']:
                    atr_rank = atr_rank.fillna(method='bfill')
                elif fill_method in ['forward', 'ffill']:
                    atr_rank = atr_rank.fillna(method='ffill')
                elif fill_method == 'mean':
                    mean_val = atr_rank.mean()
                    atr_rank = atr_rank.fillna(0.5 if pd.isna(mean_val) else mean_val)
                elif fill_method == 'median':
                    median_val = atr_rank.median()
                    atr_rank = atr_rank.fillna(0.5 if pd.isna(median_val) else median_val)
                elif fill_method == 'zero':
                    atr_rank = atr_rank.fillna(0)
                else:
                    atr_rank = atr_rank.fillna(method='bfill')
                
                result_df[f"{prefix}atr_rank_{window}"] = atr_rank
            
            # Còn lại giữ nguyên với phần xử lý tương tự
            
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
