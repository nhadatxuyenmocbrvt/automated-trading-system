"""
Đặc trưng về thanh khoản.
Mô-đun này cung cấp các hàm tạo đặc trưng dựa trên thanh khoản thị trường,
bao gồm chỉ số thanh khoản Amihud, độ sâu thị trường, và phân tích spread.
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
logger = setup_logger("liquidity_features")

def calculate_liquidity_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: str = 'volume',
    windows: List[int] = [5, 10, 20, 50],
    value_threshold: float = 1e6,  # Ngưỡng giá trị giao dịch (USD)
    vwap_column: Optional[str] = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng thanh khoản cơ bản.
    
    Args:
        df: DataFrame chứa dữ liệu giá và khối lượng
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        windows: Danh sách các kích thước cửa sổ
        value_threshold: Ngưỡng giá trị giao dịch để đánh giá thanh khoản
        vwap_column: Tên cột VWAP (nếu có)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng thanh khoản
    """
    if not validate_price_data(df, [price_column, volume_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column} hoặc {volume_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Tính giá trị giao dịch (USD volume)
        result_df[f"{prefix}dollar_volume"] = result_df[price_column] * result_df[volume_column]
        
        # Tính chỉ số thanh khoản hàng ngày
        for window in windows:
            # 1. Khối lượng trung bình (SMA của khối lượng)
            result_df[f"{prefix}avg_volume_{window}"] = result_df[volume_column].rolling(window=window).mean()
            
            # 2. Giá trị giao dịch trung bình
            result_df[f"{prefix}avg_dollar_volume_{window}"] = result_df[f"{prefix}dollar_volume"].rolling(window=window).mean()
            
            # 3. Tỷ lệ khối lượng (so với trung bình)
            avg_vol = result_df[f"{prefix}avg_volume_{window}"]
            avg_vol_non_zero = avg_vol.replace(0, np.nan)
            result_df[f"{prefix}volume_ratio_{window}"] = result_df[volume_column] / avg_vol_non_zero
            
            # 4. Biến động khối lượng
            result_df[f"{prefix}volume_volatility_{window}"] = result_df[volume_column].rolling(window=window).std() / avg_vol_non_zero
            
            # 5. Khối lượng tích lũy (sum)
            result_df[f"{prefix}cumulative_volume_{window}"] = result_df[volume_column].rolling(window=window).sum()
            
            # 6. Chỉ số thanh khoản cao/thấp
            high_liquidity = (result_df[f"{prefix}dollar_volume"] > value_threshold).astype(int)
            result_df[f"{prefix}high_liquidity_{window}"] = high_liquidity.rolling(window=window).mean()
            
            # 7. Trend thanh khoản (thay đổi của khối lượng trung bình)
            result_df[f"{prefix}liquidity_trend_{window}"] = (
                result_df[f"{prefix}avg_volume_{window}"] / 
                result_df[f"{prefix}avg_volume_{window}"].shift(window//2) - 1
            ) * 100
        
        # Tính biến động giá theo khối lượng (nếu có VWAP)
        if vwap_column and vwap_column in result_df.columns:
            # Tính chênh lệch giá đóng cửa với VWAP (đo lực mua/bán)
            result_df[f"{prefix}close_to_vwap_pct"] = (
                (result_df[price_column] - result_df[vwap_column]) / result_df[vwap_column] * 100
            )
            
            # Tạo chỉ báo áp lực mua/bán dựa trên vị trí đóng cửa so với VWAP
            result_df[f"{prefix}buying_pressure"] = (result_df[f"{prefix}close_to_vwap_pct"] > 0).astype(int)
            result_df[f"{prefix}selling_pressure"] = (result_df[f"{prefix}close_to_vwap_pct"] < 0).astype(int)
            
            for window in windows:
                # Tính tỷ lệ ngày có áp lực mua
                result_df[f"{prefix}buying_pressure_ratio_{window}"] = (
                    result_df[f"{prefix}buying_pressure"].rolling(window=window).mean()
                )
        
        logger.debug("Đã tính đặc trưng thanh khoản cơ bản")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng thanh khoản cơ bản: {e}")
    
    return result_df

def calculate_amihud_illiquidity(
    df: pd.DataFrame,
    return_column: Optional[str] = None,
    price_column: str = 'close',
    volume_column: str = 'volume',
    windows: List[int] = [5, 10, 20],
    scaling_factor: float = 1e6,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính chỉ số thanh khoản Amihud (Amihud Illiquidity Ratio).
    
    Args:
        df: DataFrame chứa dữ liệu giá, lợi nhuận và khối lượng
        return_column: Tên cột lợi nhuận (None để tự tính từ price_column)
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        windows: Danh sách các kích thước cửa sổ
        scaling_factor: Hệ số nhân để điều chỉnh kết quả
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa chỉ số Amihud
    """
    if not validate_price_data(df, [price_column, volume_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column} hoặc {volume_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Nếu không cung cấp cột lợi nhuận, tự tính từ giá
        if not return_column or return_column not in result_df.columns:
            result_df[f"{prefix}returns"] = result_df[price_column].pct_change().abs()
            return_column = f"{prefix}returns"
        
        # Tính chỉ số Amihud: |returns| / (price * volume)
        dollar_volume = result_df[price_column] * result_df[volume_column]
        
        # Tránh chia cho 0
        dollar_volume_non_zero = dollar_volume.replace(0, np.nan)
        
        # Tính chỉ số Amihud hàng ngày
        result_df[f"{prefix}amihud_daily"] = (
            result_df[return_column].abs() / dollar_volume_non_zero * scaling_factor
        )
        
        # Tính chỉ số Amihud trung bình cho các cửa sổ khác nhau
        for window in windows:
            result_df[f"{prefix}amihud_{window}"] = (
                result_df[f"{prefix}amihud_daily"].rolling(window=window).mean()
            )
            
            # Tính log của chỉ số Amihud để giảm thiểu ảnh hưởng của các giá trị cực đoan
            amihud_window = result_df[f"{prefix}amihud_{window}"]
            amihud_window_positive = amihud_window.replace(0, np.nan)
            result_df[f"{prefix}log_amihud_{window}"] = np.log1p(amihud_window_positive)
            
            # Tính thay đổi trong chỉ số Amihud (tăng = giảm thanh khoản, giảm = tăng thanh khoản)
            result_df[f"{prefix}amihud_change_{window}"] = (
                result_df[f"{prefix}amihud_{window}"] / 
                result_df[f"{prefix}amihud_{window}"].shift(window//2) - 1
            ) * 100
            
            # Phân loại mức độ thanh khoản dựa trên phân vị
            amihud_rank = result_df[f"{prefix}amihud_{window}"].rolling(window=window*2).rank(pct=True)
            
            # Phân loại thành 3 cấp độ: cao (>0.7), trung bình (0.3-0.7), thấp (<0.3)
            # Lưu ý: Amihud cao = Thanh khoản thấp
            result_df[f"{prefix}low_liquidity_{window}"] = (amihud_rank > 0.7).astype(int)
            result_df[f"{prefix}medium_liquidity_{window}"] = (
                (amihud_rank >= 0.3) & (amihud_rank <= 0.7)
            ).astype(int)
            result_df[f"{prefix}high_liquidity_{window}"] = (amihud_rank < 0.3).astype(int)
        
        logger.debug("Đã tính chỉ số thanh khoản Amihud")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính chỉ số thanh khoản Amihud: {e}")
    
    return result_df

def calculate_market_depth(
    df: pd.DataFrame,
    orderbook_columns: bool = False,
    bid_col_prefix: str = 'bid_volume_',
    ask_col_prefix: str = 'ask_volume_',
    depth_levels: List[int] = [5, 10, 20],
    price_column: str = 'close',
    normalize: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính các đặc trưng độ sâu thị trường từ sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu sổ lệnh
        orderbook_columns: Sử dụng cột sổ lệnh có sẵn (True) hoặc dữ liệu bids/asks (False)
        bid_col_prefix: Tiền tố cho các cột khối lượng mua
        ask_col_prefix: Tiền tố cho các cột khối lượng bán
        depth_levels: Danh sách các cấp độ sâu cần tính
        price_column: Tên cột giá sử dụng để chuẩn hóa
        normalize: Chuẩn hóa độ sâu theo giá
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng độ sâu thị trường
    """
    result_df = df.copy()
    
    # Kiểm tra sự tồn tại của dữ liệu sổ lệnh
    if orderbook_columns:
        # Kiểm tra các cột level đã được định dạng sẵn (bid_volume_1, ask_volume_1, etc.)
        bid_columns = [f"{bid_col_prefix}{i}" for i in range(1, max(depth_levels) + 1)]
        ask_columns = [f"{ask_col_prefix}{i}" for i in range(1, max(depth_levels) + 1)]
        
        has_orderbook = all(col in df.columns for col in bid_columns[:1] + ask_columns[:1])
    else:
        # Kiểm tra có dữ liệu bids/asks trực tiếp không
        has_orderbook = ('bids' in df.columns or isinstance(df.get('bids', None), list)) and \
                      ('asks' in df.columns or isinstance(df.get('asks', None), list))
    
    if not has_orderbook:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin sổ lệnh")
        return result_df
    
    try:
        if orderbook_columns:
            # Tính độ sâu thị trường từ các cột có sẵn
            for level in depth_levels:
                # Tính tổng khối lượng bid và ask đến cấp độ level
                bid_cols = [f"{bid_col_prefix}{i}" for i in range(1, level + 1) if f"{bid_col_prefix}{i}" in df.columns]
                ask_cols = [f"{ask_col_prefix}{i}" for i in range(1, level + 1) if f"{ask_col_prefix}{i}" in df.columns]
                
                if bid_cols and ask_cols:
                    # Tính tổng khối lượng
                    bid_depth = result_df[bid_cols].sum(axis=1)
                    ask_depth = result_df[ask_cols].sum(axis=1)
                    
                    # Chuẩn hóa theo giá nếu cần
                    if normalize and price_column in result_df.columns:
                        price_non_zero = result_df[price_column].replace(0, np.nan)
                        bid_depth = bid_depth * result_df[price_column]
                        ask_depth = ask_depth * result_df[price_column]
                    
                    # Lưu kết quả
                    result_df[f"{prefix}bid_depth_{level}"] = bid_depth
                    result_df[f"{prefix}ask_depth_{level}"] = ask_depth
                    result_df[f"{prefix}total_depth_{level}"] = bid_depth + ask_depth
                    
                    # Tính tỷ lệ bid/ask (imbalance)
                    ask_depth_non_zero = ask_depth.replace(0, np.nan)
                    result_df[f"{prefix}depth_ratio_{level}"] = bid_depth / ask_depth_non_zero
                    
                    # Tính chỉ số mất cân bằng (-1 đến 1)
                    total_depth = bid_depth + ask_depth
                    total_depth_non_zero = total_depth.replace(0, np.nan)
                    result_df[f"{prefix}depth_imbalance_{level}"] = (bid_depth - ask_depth) / total_depth_non_zero
        else:
            # Xử lý dữ liệu bids/asks trực tiếp
            has_processed = False
            
            if isinstance(df.get('bids', None), list) and isinstance(df.get('asks', None), list):
                # Dữ liệu sổ lệnh dưới dạng thuộc tính trực tiếp
                bids = df['bids']
                asks = df['asks']
                has_processed = True
            elif 'bids' in df.columns and isinstance(df['bids'].iloc[0], list):
                # Dữ liệu sổ lệnh dưới dạng cột chứa danh sách
                for level in depth_levels:
                    bid_depth = []
                    ask_depth = []
                    
                    for idx, row in df.iterrows():
                        # Tính tổng khối lượng từ bids và asks
                        bids = row['bids'][:level] if len(row['bids']) >= level else row['bids']
                        asks = row['asks'][:level] if len(row['asks']) >= level else row['asks']
                        
                        bid_vol_sum = sum(bid[1] for bid in bids) if bids else 0
                        ask_vol_sum = sum(ask[1] for ask in asks) if asks else 0
                        
                        # Chuẩn hóa theo giá nếu cần
                        if normalize and price_column in row:
                            price = row[price_column]
                            if price > 0:
                                bid_vol_sum *= price
                                ask_vol_sum *= price
                        
                        bid_depth.append(bid_vol_sum)
                        ask_depth.append(ask_vol_sum)
                    
                    # Lưu kết quả
                    result_df[f"{prefix}bid_depth_{level}"] = bid_depth
                    result_df[f"{prefix}ask_depth_{level}"] = ask_depth
                    result_df[f"{prefix}total_depth_{level}"] = np.array(bid_depth) + np.array(ask_depth)
                    
                    # Tính tỷ lệ bid/ask (imbalance)
                    ask_depth_arr = np.array(ask_depth)
                    ask_depth_non_zero = np.where(ask_depth_arr == 0, np.nan, ask_depth_arr)
                    
                    result_df[f"{prefix}depth_ratio_{level}"] = np.array(bid_depth) / ask_depth_non_zero
                    
                    # Tính chỉ số mất cân bằng (-1 đến 1)
                    total_depth = np.array(bid_depth) + np.array(ask_depth)
                    total_depth_non_zero = np.where(total_depth == 0, np.nan, total_depth)
                    
                    result_df[f"{prefix}depth_imbalance_{level}"] = (
                        (np.array(bid_depth) - np.array(ask_depth)) / total_depth_non_zero
                    )
                
                has_processed = True
            
            if not has_processed:
                logger.warning("Không thể xử lý dữ liệu bids/asks trực tiếp")
        
        # Tính các chỉ số nâng cao về độ sâu thị trường (nếu có đủ dữ liệu)
        for level in depth_levels:
            bid_depth_col = f"{prefix}bid_depth_{level}"
            ask_depth_col = f"{prefix}ask_depth_{level}"
            
            if bid_depth_col in result_df.columns and ask_depth_col in result_df.columns:
                # Tính độ biến động của độ sâu (chỉ số thanh khoản không ổn định)
                for window in [5, 10, 20]:
                    if len(result_df) >= window:
                        # Biến động độ sâu bid
                        result_df[f"{prefix}bid_depth_volatility_{level}_{window}"] = (
                            result_df[bid_depth_col].rolling(window=window).std() / 
                            result_df[bid_depth_col].rolling(window=window).mean()
                        )
                        
                        # Biến động độ sâu ask
                        result_df[f"{prefix}ask_depth_volatility_{level}_{window}"] = (
                            result_df[ask_depth_col].rolling(window=window).std() / 
                            result_df[ask_depth_col].rolling(window=window).mean()
                        )
                        
                        # Biến động tỷ lệ bid/ask
                        depth_ratio_col = f"{prefix}depth_ratio_{level}"
                        if depth_ratio_col in result_df.columns:
                            result_df[f"{prefix}depth_ratio_volatility_{level}_{window}"] = (
                                result_df[depth_ratio_col].rolling(window=window).std()
                            )
                
                # Phát hiện các sự kiện bất thường trong độ sâu
                
                # 1. Sự kiện mất thanh khoản đột ngột (giảm độ sâu > 50%)
                result_df[f"{prefix}liquidity_crash_{level}"] = (
                    (result_df[f"{prefix}total_depth_{level}"] / 
                     result_df[f"{prefix}total_depth_{level}"].shift(1) < 0.5)
                ).astype(int)
                
                # 2. Sự kiện tăng thanh khoản đột ngột (tăng độ sâu > 100%)
                result_df[f"{prefix}liquidity_surge_{level}"] = (
                    (result_df[f"{prefix}total_depth_{level}"] / 
                     result_df[f"{prefix}total_depth_{level}"].shift(1) > 2.0)
                ).astype(int)
                
                # 3. Chuyển đổi mất cân bằng (từ thiên về mua sang thiên về bán hoặc ngược lại)
                imbalance_col = f"{prefix}depth_imbalance_{level}"
                if imbalance_col in result_df.columns:
                    result_df[f"{prefix}imbalance_shift_{level}"] = (
                        (np.sign(result_df[imbalance_col]) != 
                         np.sign(result_df[imbalance_col].shift(1)))
                    ).astype(int)
        
        logger.debug("Đã tính đặc trưng độ sâu thị trường")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng độ sâu thị trường: {e}")
    
    return result_df

def calculate_spread_analysis(
    df: pd.DataFrame,
    spread_column: Optional[str] = None,
    bid_price_column: str = 'bid_price_1',
    ask_price_column: str = 'ask_price_1',
    price_column: str = 'close',
    windows: List[int] = [5, 10, 20, 50],
    volume_column: Optional[str] = 'volume',
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phân tích spread và các đặc trưng liên quan.
    
    Args:
        df: DataFrame chứa dữ liệu spread hoặc giá mua/bán tốt nhất
        spread_column: Tên cột spread (None để tính từ bid/ask)
        bid_price_column: Tên cột giá mua tốt nhất
        ask_price_column: Tên cột giá bán tốt nhất
        price_column: Tên cột giá tham chiếu
        windows: Danh sách các kích thước cửa sổ
        volume_column: Tên cột khối lượng giao dịch (nếu có)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng phân tích spread
    """
    result_df = df.copy()
    
    # Kiểm tra sự tồn tại của dữ liệu spread hoặc bid/ask
    has_spread = spread_column and spread_column in df.columns
    has_bid_ask = bid_price_column in df.columns and ask_price_column in df.columns
    
    if not has_spread and not has_bid_ask:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin spread hoặc bid/ask")
        return result_df
    
    try:
        # Tính spread nếu chưa có
        if not has_spread and has_bid_ask:
            result_df[f"{prefix}spread"] = result_df[ask_price_column] - result_df[bid_price_column]
            spread_column = f"{prefix}spread"
        
        # Đảm bảo spread là giá trị không âm
        result_df[spread_column] = result_df[spread_column].abs()
        
        # Tính spread tương đối (%)
        if price_column in result_df.columns:
            price_non_zero = result_df[price_column].replace(0, np.nan)
            result_df[f"{prefix}relative_spread_pct"] = (result_df[spread_column] / price_non_zero) * 100
            
            # Phân tích spread
            for window in windows:
                # 1. Trung bình spread
                result_df[f"{prefix}avg_spread_{window}"] = result_df[spread_column].rolling(window=window).mean()
                
                # 2. Trung bình spread tương đối
                result_df[f"{prefix}avg_relative_spread_{window}"] = (
                    result_df[f"{prefix}relative_spread_pct"].rolling(window=window).mean()
                )
                
                # 3. Biến động spread
                result_df[f"{prefix}spread_volatility_{window}"] = (
                    result_df[spread_column].rolling(window=window).std() / 
                    result_df[f"{prefix}avg_spread_{window}"]
                )
                
                # 4. Tỷ lệ spread hiện tại so với trung bình
                avg_spread = result_df[f"{prefix}avg_spread_{window}"]
                avg_spread_non_zero = avg_spread.replace(0, np.nan)
                
                result_df[f"{prefix}spread_ratio_{window}"] = result_df[spread_column] / avg_spread_non_zero
                
                # 5. Phát hiện spread bất thường
                
                # Z-score của spread
                spread_mean = result_df[spread_column].rolling(window=window).mean()
                spread_std = result_df[spread_column].rolling(window=window).std()
                spread_std_non_zero = spread_std.replace(0, np.nan)
                
                result_df[f"{prefix}spread_zscore_{window}"] = (
                    (result_df[spread_column] - spread_mean) / spread_std_non_zero
                )
                
                # Spread cao bất thường (>2 std)
                result_df[f"{prefix}high_spread_{window}"] = (
                    result_df[f"{prefix}spread_zscore_{window}"] > 2
                ).astype(int)
                
                # Spread thấp bất thường (<-1 std)
                result_df[f"{prefix}low_spread_{window}"] = (
                    result_df[f"{prefix}spread_zscore_{window}"] < -1
                ).astype(int)
                
                # 6. Xu hướng spread
                result_df[f"{prefix}spread_trend_{window}"] = (
                    result_df[f"{prefix}avg_spread_{window}"] / 
                    result_df[f"{prefix}avg_spread_{window}"].shift(window//2) - 1
                ) * 100
        
        # Tính các chỉ số nâng cao nếu có khối lượng giao dịch
        if volume_column and volume_column in result_df.columns:
            # Tính chi phí thanh khoản trung bình (price impact per volume)
            for window in windows:
                # 1. Chi phí thanh khoản = spread * volume
                result_df[f"{prefix}liquidity_cost_{window}"] = (
                    result_df[spread_column] * result_df[volume_column]
                )
                
                # 2. Chi phí thanh khoản bình quân (trên đơn vị khối lượng)
                volume_non_zero = result_df[volume_column].replace(0, np.nan)
                result_df[f"{prefix}avg_liquidity_cost_{window}"] = (
                    result_df[f"{prefix}liquidity_cost_{window}"] / volume_non_zero
                ).rolling(window=window).mean()
                
                # 3. Thanh khoản theo spread và khối lượng
                # Liquidity score = 1 / (relative_spread * sqrt(1/volume))
                # Cao = Thanh khoản tốt, Thấp = Thanh khoản kém
                if f"{prefix}relative_spread_pct" in result_df.columns:
                    rel_spread = result_df[f"{prefix}relative_spread_pct"] / 100  # Chuyển về tỷ lệ thập phân
                    rel_spread_non_zero = rel_spread.replace(0, 0.0001)  # Tránh chia cho 0
                    
                    volume_factor = np.sqrt(1 / volume_non_zero)
                    
                    result_df[f"{prefix}liquidity_score_{window}"] = (
                        (1 / rel_spread_non_zero) * (1 / volume_factor)
                    ).rolling(window=window).mean()
                    
                    # 4. Chỉ số Market Quality (MQI)
                    # MQI = log(volume) / (relative_spread)
                    result_df[f"{prefix}market_quality_{window}"] = (
                        np.log1p(result_df[volume_column]) / rel_spread_non_zero
                    ).rolling(window=window).mean()
        
        # Tính tỷ lệ giá mid với các mốc giá quan trọng (nếu có)
        if has_bid_ask and price_column in result_df.columns:
            # Tính giá mid
            result_df[f"{prefix}mid_price"] = (result_df[bid_price_column] + result_df[ask_price_column]) / 2
            
            # Tỷ lệ giá đóng cửa / mid price
            mid_price_non_zero = result_df[f"{prefix}mid_price"].replace(0, np.nan)
            result_df[f"{prefix}close_to_mid_ratio"] = result_df[price_column] / mid_price_non_zero
            
            # Vị trí giá đóng cửa trong spread
            spread_non_zero = result_df[spread_column].replace(0, np.nan)
            result_df[f"{prefix}close_in_spread"] = (
                (result_df[price_column] - result_df[bid_price_column]) / spread_non_zero
            )
        
        logger.debug("Đã tính đặc trưng phân tích spread")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng phân tích spread: {e}")
    
    return result_df

def calculate_market_impact(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: str = 'volume',
    orderbook_data: bool = False,
    avg_trade_size: Optional[float] = None,
    windows: List[int] = [10, 20, 50],
    price_tolerance: float = 0.005,  # 0.5%
    prefix: str = ''
) -> pd.DataFrame:
    """
    Ước tính tác động thị trường (market impact) và chi phí thực thi dự kiến.
    
    Args:
        df: DataFrame chứa dữ liệu giá, khối lượng và/hoặc sổ lệnh
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        orderbook_data: Có dữ liệu sổ lệnh hay không
        avg_trade_size: Kích thước giao dịch trung bình để ước tính tác động
        windows: Danh sách các kích thước cửa sổ
        price_tolerance: Dung sai giá để ước tính chi phí thực thi (%)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa ước tính tác động thị trường
    """
    if not validate_price_data(df, [price_column, volume_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column} hoặc {volume_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Nếu không cung cấp kích thước giao dịch trung bình, ước tính từ dữ liệu
        if avg_trade_size is None:
            avg_trade_size = result_df[volume_column].mean() * 0.1  # 10% khối lượng trung bình
        
        # Tính tác động thị trường dựa trên công thức đơn giản: Impact = c * σ * (Q/V)^0.5
        # c: hằng số (thường là 0.1-1.0)
        # σ: độ biến động
        # Q: kích thước giao dịch
        # V: khối lượng
        c = 0.5  # Hằng số tác động
        
        for window in windows:
            # Tính độ biến động
            volatility = result_df[price_column].pct_change().rolling(window=window).std()
            
            # Tính tỷ lệ kích thước giao dịch / khối lượng
            volume_non_zero = result_df[volume_column].replace(0, np.nan)
            size_ratio = np.sqrt(avg_trade_size / volume_non_zero)
            
            # Ước tính tác động thị trường (%) cho kích thước giao dịch trung bình
            impact = c * volatility * size_ratio
            result_df[f"{prefix}market_impact_pct_{window}"] = impact * 100
            
            # Ước tính tác động tuyệt đối (theo giá)
            result_df[f"{prefix}market_impact_abs_{window}"] = impact * result_df[price_column]
            
            # Ước tính tác động theo các kích thước giao dịch khác nhau
            for size_mul in [0.5, 1.0, 2.0, 5.0]:
                trade_size = avg_trade_size * size_mul
                size_ratio = np.sqrt(trade_size / volume_non_zero)
                impact = c * volatility * size_ratio
                
                result_df[f"{prefix}impact_{int(size_mul*100)}pct_size_{window}"] = impact * 100
        
        # Phân tích chi tiết hơn nếu có dữ liệu sổ lệnh
        if orderbook_data:
            # Kiểm tra các cột cần thiết
            if all(col in result_df.columns for col in ['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']):
                # Tính tác động thị trường dựa trên spread
                mid_price = (result_df['bid_price_1'] + result_df['ask_price_1']) / 2
                spread_pct = (result_df['ask_price_1'] - result_df['bid_price_1']) / mid_price
                
                # Tác động thực thi tức thời (chi phí crossing spread)
                result_df[f"{prefix}immediate_impact_buy"] = spread_pct / 2 * 100
                result_df[f"{prefix}immediate_impact_sell"] = spread_pct / 2 * 100
                
                # Ước tính chi phí thực thi cho các kích thước giao dịch khác nhau
                for size_mul in [0.5, 1.0, 2.0, 5.0]:
                    trade_size = avg_trade_size * size_mul
                    
                    # Ước tính tác động bổ sung khi vượt quá khối lượng tại mức giá tốt nhất
                    buy_impact = spread_pct / 2  # Tác động cơ bản (nửa spread)
                    sell_impact = spread_pct / 2
                    
                    # Nếu kích thước giao dịch > khối lượng tại mức giá tốt nhất
                    # Thêm tác động bổ sung (giả định tuyến tính với phần vượt quá)
                    bid_vol_non_zero = result_df['bid_volume_1'].replace(0, np.nan)
                    ask_vol_non_zero = result_df['ask_volume_1'].replace(0, np.nan)
                    
                    buy_excess = np.maximum(0, 1 - ask_vol_non_zero / trade_size)
                    sell_excess = np.maximum(0, 1 - bid_vol_non_zero / trade_size)
                    
                    # Thêm tác động bổ sung (price_tolerance cho mỗi lần vượt quá 100%)
                    buy_impact += buy_excess * price_tolerance
                    sell_impact += sell_excess * price_tolerance
                    
                    result_df[f"{prefix}buy_impact_{int(size_mul*100)}pct_size"] = buy_impact * 100
                    result_df[f"{prefix}sell_impact_{int(size_mul*100)}pct_size"] = sell_impact * 100
                    
                    # Ước tính chi phí thực thi tuyệt đối ($)
                    result_df[f"{prefix}buy_cost_{int(size_mul*100)}pct_size"] = (
                        buy_impact * result_df[price_column] * trade_size
                    )
                    result_df[f"{prefix}sell_cost_{int(size_mul*100)}pct_size"] = (
                        sell_impact * result_df[price_column] * trade_size
                    )
        
        logger.debug("Đã tính đặc trưng tác động thị trường")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng tác động thị trường: {e}")
    
    return result_df

def calculate_slippage_estimation(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: str = 'volume',
    have_vwap: bool = False,
    vwap_column: str = 'vwap',
    orderbook_data: bool = False,
    trade_sizes: List[float] = [1000, 5000, 10000, 50000],  # Kích thước giao dịch (USD)
    windows: List[int] = [5, 10, 20],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Ước tính slippage dựa vào thanh khoản thị trường và kích thước giao dịch.
    
    Args:
        df: DataFrame chứa dữ liệu giá, khối lượng và/hoặc sổ lệnh
        price_column: Tên cột giá sử dụng để tính toán
        volume_column: Tên cột khối lượng giao dịch
        have_vwap: Có dữ liệu VWAP hay không
        vwap_column: Tên cột VWAP
        orderbook_data: Có dữ liệu sổ lệnh hay không
        trade_sizes: Danh sách các kích thước giao dịch (USD) để ước tính
        windows: Danh sách các kích thước cửa sổ
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa ước tính slippage
    """
    if not validate_price_data(df, [price_column, volume_column]):
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column} hoặc {volume_column}")
        return df
    
    result_df = df.copy()
    
    try:
        # Phương pháp 1: Ước tính slippage từ VWAP (nếu có)
        if have_vwap and vwap_column in result_df.columns:
            # Tính slippage thực tế = (close - vwap) / vwap
            vwap_non_zero = result_df[vwap_column].replace(0, np.nan)
            result_df[f"{prefix}realized_slippage_pct"] = (
                (result_df[price_column] - result_df[vwap_column]) / vwap_non_zero * 100
            )
            
            # Tính slippage trung bình trong các cửa sổ
            for window in windows:
                result_df[f"{prefix}avg_slippage_{window}"] = (
                    result_df[f"{prefix}realized_slippage_pct"].rolling(window=window).mean()
                )
                
                # Biến động slippage
                result_df[f"{prefix}slippage_volatility_{window}"] = (
                    result_df[f"{prefix}realized_slippage_pct"].rolling(window=window).std()
                )
        
        # Phương pháp 2: Ước tính slippage từ đặc tính thanh khoản
        # Sử dụng công thức đơn giản: Slippage = k * σ * sqrt(Q/V) * sqrt(1/ADV)
        # k: hằng số
        # σ: độ biến động
        # Q: kích thước giao dịch
        # V: khối lượng hàng ngày
        # ADV: khối lượng trung bình hàng ngày
        
        for window in windows:
            # Tính độ biến động
            volatility = result_df[price_column].pct_change().rolling(window=window).std()
            
            # Tính khối lượng trung bình hàng ngày
            adv = result_df[volume_column].rolling(window=window).mean()
            
            # Tính giá trị trung bình hàng ngày
            adv_value = (adv * result_df[price_column]).rolling(window=window).mean()
            
            # Tính hệ số thanh khoản thị trường
            adv_non_zero = adv.replace(0, np.nan)
            liquidity_factor = 1 / np.sqrt(adv_non_zero)
            
            # Ước tính slippage cho các kích thước giao dịch khác nhau
            for trade_size_usd in trade_sizes:
                # Tính kích thước giao dịch theo số lượng
                price_non_zero = result_df[price_column].replace(0, np.nan)
                trade_size_qty = trade_size_usd / price_non_zero
                
                # Tính tỷ lệ kích thước giao dịch / khối lượng
                size_ratio = np.sqrt(trade_size_qty / adv_non_zero)
                
                # Hằng số slippage (có thể điều chỉnh)
                k = 0.3
                
                # Ước tính slippage (%)
                slippage = k * volatility * size_ratio * liquidity_factor
                result_df[f"{prefix}est_slippage_{trade_size_usd}_{window}"] = slippage * 100
                
                # Ước tính chi phí slippage ($)
                result_df[f"{prefix}slippage_cost_{trade_size_usd}_{window}"] = (
                    slippage * trade_size_usd
                )
        
        # Phương pháp 3: Ước tính từ sổ lệnh (nếu có)
        if orderbook_data:
            # Kiểm tra các cột cần thiết
            ob_cols_needed = ['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']
            if all(col in result_df.columns for col in ob_cols_needed) and len(trade_sizes) > 0:
                # Tính giá mid
                mid_price = (result_df['bid_price_1'] + result_df['ask_price_1']) / 2
                
                # Tính spread tương đối (%)
                mid_price_non_zero = mid_price.replace(0, np.nan)
                spread_pct = (result_df['ask_price_1'] - result_df['bid_price_1']) / mid_price_non_zero * 100
                
                for trade_size_usd in trade_sizes:
                    # Tính kích thước giao dịch theo số lượng
                    price_non_zero = result_df[price_column].replace(0, np.nan)
                    trade_size_qty = trade_size_usd / price_non_zero
                    
                    # Tính slippage dựa trên sổ lệnh
                    
                    # Slippage mua = Tác động thực thi tăng dần dựa trên độ sâu sổ lệnh
                    buy_slippage = spread_pct / 2  # Bắt đầu với nửa spread
                    
                    # Nếu kích thước > khối lượng tại mức giá tốt nhất, thêm slippage
                    ask_vol_1 = result_df['ask_volume_1']
                    ask_vol_non_zero = ask_vol_1.replace(0, np.nan)
                    
                    # Ước tính phần vượt quá
                    buy_excess_ratio = np.maximum(0, 1 - ask_vol_non_zero / trade_size_qty)
                    
                    # Thêm slippage dựa trên phần vượt quá
                    # Giả định: Mỗi lần vượt quá 100% khối lượng mức 1, thêm 0.1% slippage
                    buy_slippage += buy_excess_ratio * 0.1 * spread_pct
                    
                    # Tương tự cho slippage bán
                    sell_slippage = spread_pct / 2
                    
                    bid_vol_1 = result_df['bid_volume_1']
                    bid_vol_non_zero = bid_vol_1.replace(0, np.nan)
                    
                    sell_excess_ratio = np.maximum(0, 1 - bid_vol_non_zero / trade_size_qty)
                    
                    sell_slippage += sell_excess_ratio * 0.1 * spread_pct
                    
                    # Lưu kết quả
                    result_df[f"{prefix}ob_buy_slippage_{trade_size_usd}"] = buy_slippage
                    result_df[f"{prefix}ob_sell_slippage_{trade_size_usd}"] = sell_slippage
                    
                    # Tính trung bình slippage mua/bán
                    result_df[f"{prefix}ob_avg_slippage_{trade_size_usd}"] = (
                        (buy_slippage + sell_slippage) / 2
                    )
                    
                    # Ước tính chi phí slippage ($)
                    result_df[f"{prefix}ob_buy_slippage_cost_{trade_size_usd}"] = (
                        buy_slippage * trade_size_usd / 100
                    )
                    result_df[f"{prefix}ob_sell_slippage_cost_{trade_size_usd}"] = (
                        sell_slippage * trade_size_usd / 100
                    )
        
        logger.debug("Đã tính đặc trưng ước tính slippage")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng ước tính slippage: {e}")
    
    return result_df