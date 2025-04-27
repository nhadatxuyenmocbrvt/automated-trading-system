"""
Đặc trưng từ sổ lệnh.
Mô-đun này cung cấp các hàm tạo đặc trưng dựa trên dữ liệu sổ lệnh (orderbook),
bao gồm phân tích áp lực thị trường, phân phối lệnh, và dự báo hỗ trợ/kháng cự.
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

# Logger
logger = setup_logger("orderbook_features")

def calculate_orderbook_imbalance(
    df: pd.DataFrame,
    levels: List[int] = [1, 5, 10, 20],
    price_buckets: Optional[List[float]] = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các chỉ số mất cân bằng (imbalance) từ sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu sổ lệnh (bids, asks)
        levels: Danh sách số lượng level sổ lệnh cần phân tích
        price_buckets: Danh sách các khoảng giá cần tính (% từ mid price)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa chỉ số mất cân bằng sổ lệnh
    """
    result_df = df.copy()
    
    # Kiểm tra cấu trúc dữ liệu sổ lệnh
    if not isinstance(df.get('bids', None), list) and 'bids' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin bids trong sổ lệnh")
        return result_df
    
    if not isinstance(df.get('asks', None), list) and 'asks' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin asks trong sổ lệnh")
        return result_df
    
    try:
        # Phân tích cấu trúc sổ lệnh
        orderbook_format = ''
        
        # Xác định định dạng dữ liệu sổ lệnh
        if 'bids' in df.columns and isinstance(df['bids'].iloc[0], list):
            # Định dạng cột chứa danh sách
            orderbook_format = 'column_list'
        elif isinstance(df.get('bids', None), list):
            # Định dạng thuộc tính trực tiếp
            orderbook_format = 'direct_property'
        elif 'bid_price_1' in df.columns:
            # Định dạng cột riêng cho từng level
            orderbook_format = 'level_columns'
        else:
            logger.error("Không thể xác định định dạng dữ liệu sổ lệnh")
            return result_df
        
        # Danh sách để lưu trữ kết quả
        results = []
        
        # Xử lý từng hàng dữ liệu
        for idx, row in df.iterrows():
            # Lấy dữ liệu sổ lệnh cho hàng hiện tại
            bids = None
            asks = None
            
            if orderbook_format == 'column_list':
                bids = row['bids']
                asks = row['asks']
            elif orderbook_format == 'direct_property':
                bids = df.at[idx, 'bids']
                asks = df.at[idx, 'asks']
            elif orderbook_format == 'level_columns':
                # Tạo danh sách bids, asks từ các cột riêng lẻ
                bids = []
                asks = []
                
                for level in range(1, max(levels) + 1):
                    if f'bid_price_{level}' in df.columns and f'bid_volume_{level}' in df.columns:
                        bid_price = row[f'bid_price_{level}']
                        bid_volume = row[f'bid_volume_{level}']
                        if not pd.isna(bid_price) and not pd.isna(bid_volume):
                            bids.append([bid_price, bid_volume])
                    
                    if f'ask_price_{level}' in df.columns and f'ask_volume_{level}' in df.columns:
                        ask_price = row[f'ask_price_{level}']
                        ask_volume = row[f'ask_volume_{level}']
                        if not pd.isna(ask_price) and not pd.isna(ask_volume):
                            asks.append([ask_price, ask_volume])
            
            # Kiểm tra xem có dữ liệu không
            if not bids or not asks:
                logger.warning(f"Không có dữ liệu sổ lệnh hợp lệ cho hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                result_row = {**{col: row[col] for col in df.columns}}
                for level in levels:
                    result_row[f"{prefix}orderbook_imbalance_{level}"] = np.nan
                    result_row[f"{prefix}bid_volume_sum_{level}"] = np.nan
                    result_row[f"{prefix}ask_volume_sum_{level}"] = np.nan
                    result_row[f"{prefix}volume_ratio_{level}"] = np.nan
                
                results.append(result_row)
                continue
            
            # Tạo dictionary cho kết quả của hàng hiện tại
            result_row = {**{col: row[col] for col in df.columns}}
            
            # Tính mid price
            best_bid_price = bids[0][0] if len(bids) > 0 else 0
            best_ask_price = asks[0][0] if len(asks) > 0 else 0
            
            if best_bid_price == 0 or best_ask_price == 0:
                logger.warning(f"Giá mua/bán tốt nhất không hợp lệ tại hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                for level in levels:
                    result_row[f"{prefix}orderbook_imbalance_{level}"] = np.nan
                    result_row[f"{prefix}bid_volume_sum_{level}"] = np.nan
                    result_row[f"{prefix}ask_volume_sum_{level}"] = np.nan
                    result_row[f"{prefix}volume_ratio_{level}"] = np.nan
                
                results.append(result_row)
                continue
            
            mid_price = (best_bid_price + best_ask_price) / 2
            result_row[f"{prefix}mid_price"] = mid_price
            
            # Tính spread
            spread = best_ask_price - best_bid_price
            result_row[f"{prefix}spread"] = spread
            result_row[f"{prefix}spread_percent"] = (spread / mid_price) * 100
            
            # Tính imbalance cho mỗi level
            for level in levels:
                # Lấy số lượng level cần thiết
                bid_levels = bids[:level] if len(bids) >= level else bids
                ask_levels = asks[:level] if len(asks) >= level else asks
                
                # Tính tổng khối lượng
                bid_volume_sum = sum(bid[1] for bid in bid_levels)
                ask_volume_sum = sum(ask[1] for ask in ask_levels)
                
                # Lưu tổng khối lượng
                result_row[f"{prefix}bid_volume_sum_{level}"] = bid_volume_sum
                result_row[f"{prefix}ask_volume_sum_{level}"] = ask_volume_sum
                
                # Tính imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
                total_volume = bid_volume_sum + ask_volume_sum
                
                if total_volume > 0:
                    imbalance = (bid_volume_sum - ask_volume_sum) / total_volume
                    result_row[f"{prefix}orderbook_imbalance_{level}"] = imbalance
                    
                    # Tính tỷ lệ khối lượng bid/ask
                    if ask_volume_sum > 0:
                        volume_ratio = bid_volume_sum / ask_volume_sum
                        result_row[f"{prefix}volume_ratio_{level}"] = volume_ratio
                    else:
                        result_row[f"{prefix}volume_ratio_{level}"] = np.inf
                else:
                    result_row[f"{prefix}orderbook_imbalance_{level}"] = 0
                    result_row[f"{prefix}volume_ratio_{level}"] = 1
            
            # Tính imbalance theo khoảng giá nếu cần
            if price_buckets:
                for bucket in price_buckets:
                    # Tính khoảng giá
                    bucket_decimal = bucket / 100
                    lower_price = mid_price * (1 - bucket_decimal)
                    upper_price = mid_price * (1 + bucket_decimal)
                    
                    # Tính tổng khối lượng trong khoảng giá
                    bid_volume_bucket = sum(bid[1] for bid in bids if bid[0] >= lower_price)
                    ask_volume_bucket = sum(ask[1] for ask in asks if ask[0] <= upper_price)
                    
                    # Lưu tổng khối lượng
                    result_row[f"{prefix}bid_volume_bucket_{bucket}"] = bid_volume_bucket
                    result_row[f"{prefix}ask_volume_bucket_{bucket}"] = ask_volume_bucket
                    
                    # Tính imbalance trong khoảng giá
                    total_volume_bucket = bid_volume_bucket + ask_volume_bucket
                    
                    if total_volume_bucket > 0:
                        imbalance_bucket = (bid_volume_bucket - ask_volume_bucket) / total_volume_bucket
                        result_row[f"{prefix}orderbook_imbalance_bucket_{bucket}"] = imbalance_bucket
                        
                        # Tính tỷ lệ khối lượng bid/ask trong khoảng giá
                        if ask_volume_bucket > 0:
                            volume_ratio_bucket = bid_volume_bucket / ask_volume_bucket
                            result_row[f"{prefix}volume_ratio_bucket_{bucket}"] = volume_ratio_bucket
                        else:
                            result_row[f"{prefix}volume_ratio_bucket_{bucket}"] = np.inf
                    else:
                        result_row[f"{prefix}orderbook_imbalance_bucket_{bucket}"] = 0
                        result_row[f"{prefix}volume_ratio_bucket_{bucket}"] = 1
            
            # Thêm vào danh sách kết quả
            results.append(result_row)
        
        # Tạo DataFrame từ danh sách kết quả
        result_df = pd.DataFrame(results)
        
        logger.debug("Đã tính đặc trưng mất cân bằng sổ lệnh")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng mất cân bằng sổ lệnh: {e}")
    
    return result_df

def calculate_orderbook_features(
    df: pd.DataFrame,
    levels: List[int] = [5, 10, 20],
    use_vwap: bool = True,
    depth_buckets: Optional[List[float]] = None,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán các đặc trưng mở rộng từ sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu sổ lệnh (bids, asks)
        levels: Danh sách số lượng level sổ lệnh cần phân tích
        use_vwap: Tính giá trung bình theo khối lượng từ sổ lệnh
        depth_buckets: Danh sách các khoảng cách tính độ sâu (% từ mid price)
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa đặc trưng sổ lệnh
    """
    result_df = df.copy()
    
    # Kiểm tra cấu trúc dữ liệu sổ lệnh
    if not isinstance(df.get('bids', None), list) and 'bids' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin bids trong sổ lệnh")
        return result_df
    
    if not isinstance(df.get('asks', None), list) and 'asks' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin asks trong sổ lệnh")
        return result_df
    
    try:
        # Phân tích cấu trúc sổ lệnh
        orderbook_format = ''
        
        # Xác định định dạng dữ liệu sổ lệnh
        if 'bids' in df.columns and isinstance(df['bids'].iloc[0], list):
            # Định dạng cột chứa danh sách
            orderbook_format = 'column_list'
        elif isinstance(df.get('bids', None), list):
            # Định dạng thuộc tính trực tiếp
            orderbook_format = 'direct_property'
        elif 'bid_price_1' in df.columns:
            # Định dạng cột riêng cho từng level
            orderbook_format = 'level_columns'
        else:
            logger.error("Không thể xác định định dạng dữ liệu sổ lệnh")
            return result_df
        
        # Danh sách để lưu trữ kết quả
        results = []
        
        # Xử lý từng hàng dữ liệu
        for idx, row in df.iterrows():
            # Lấy dữ liệu sổ lệnh cho hàng hiện tại
            bids = None
            asks = None
            
            if orderbook_format == 'column_list':
                bids = row['bids']
                asks = row['asks']
            elif orderbook_format == 'direct_property':
                bids = df.at[idx, 'bids']
                asks = df.at[idx, 'asks']
            elif orderbook_format == 'level_columns':
                # Tạo danh sách bids, asks từ các cột riêng lẻ
                bids = []
                asks = []
                
                for level in range(1, max(levels) + 1):
                    if f'bid_price_{level}' in df.columns and f'bid_volume_{level}' in df.columns:
                        bid_price = row[f'bid_price_{level}']
                        bid_volume = row[f'bid_volume_{level}']
                        if not pd.isna(bid_price) and not pd.isna(bid_volume):
                            bids.append([bid_price, bid_volume])
                    
                    if f'ask_price_{level}' in df.columns and f'ask_volume_{level}' in df.columns:
                        ask_price = row[f'ask_price_{level}']
                        ask_volume = row[f'ask_volume_{level}']
                        if not pd.isna(ask_price) and not pd.isna(ask_volume):
                            asks.append([ask_price, ask_volume])
            
            # Kiểm tra xem có dữ liệu không
            if not bids or not asks:
                logger.warning(f"Không có dữ liệu sổ lệnh hợp lệ cho hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                result_row = {**{col: row[col] for col in df.columns}}
                for level in levels:
                    result_row[f"{prefix}price_impact_buy_{level}"] = np.nan
                    result_row[f"{prefix}price_impact_sell_{level}"] = np.nan
                    result_row[f"{prefix}orderbook_slope_bid_{level}"] = np.nan
                    result_row[f"{prefix}orderbook_slope_ask_{level}"] = np.nan
                
                if use_vwap:
                    result_row[f"{prefix}vwap_bid"] = np.nan
                    result_row[f"{prefix}vwap_ask"] = np.nan
                
                results.append(result_row)
                continue
            
            # Tạo dictionary cho kết quả của hàng hiện tại
            result_row = {**{col: row[col] for col in df.columns}}
            
            # Tính mid price
            best_bid_price = bids[0][0] if len(bids) > 0 else 0
            best_ask_price = asks[0][0] if len(asks) > 0 else 0
            
            if best_bid_price == 0 or best_ask_price == 0:
                logger.warning(f"Giá mua/bán tốt nhất không hợp lệ tại hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                for level in levels:
                    result_row[f"{prefix}price_impact_buy_{level}"] = np.nan
                    result_row[f"{prefix}price_impact_sell_{level}"] = np.nan
                    result_row[f"{prefix}orderbook_slope_bid_{level}"] = np.nan
                    result_row[f"{prefix}orderbook_slope_ask_{level}"] = np.nan
                
                if use_vwap:
                    result_row[f"{prefix}vwap_bid"] = np.nan
                    result_row[f"{prefix}vwap_ask"] = np.nan
                
                results.append(result_row)
                continue
            
            mid_price = (best_bid_price + best_ask_price) / 2
            
            # Tính VWAP từ sổ lệnh nếu cần
            if use_vwap:
                # VWAP cho phía mua (bids)
                bid_volume_sum = sum(bid[1] for bid in bids)
                if bid_volume_sum > 0:
                    vwap_bid = sum(bid[0] * bid[1] for bid in bids) / bid_volume_sum
                    result_row[f"{prefix}vwap_bid"] = vwap_bid
                else:
                    result_row[f"{prefix}vwap_bid"] = best_bid_price
                
                # VWAP cho phía bán (asks)
                ask_volume_sum = sum(ask[1] for ask in asks)
                if ask_volume_sum > 0:
                    vwap_ask = sum(ask[0] * ask[1] for ask in asks) / ask_volume_sum
                    result_row[f"{prefix}vwap_ask"] = vwap_ask
                else:
                    result_row[f"{prefix}vwap_ask"] = best_ask_price
                
                # Tính hệ số bất cân xứng VWAP
                result_row[f"{prefix}vwap_ratio"] = (mid_price - result_row[f"{prefix}vwap_bid"]) / (result_row[f"{prefix}vwap_ask"] - mid_price)
            
            # Tính tác động giá (price impact) cho mỗi level
            for level in levels:
                # Lấy số lượng level cần thiết
                bid_levels = bids[:level] if len(bids) >= level else bids
                ask_levels = asks[:level] if len(asks) >= level else asks
                
                # Tính tổng khối lượng và khối lượng tích lũy tại mỗi mức giá
                bid_volume_sum = sum(bid[1] for bid in bid_levels)
                ask_volume_sum = sum(ask[1] for ask in ask_levels)
                
                # Tính khoảng cách giá (price range)
                if bid_levels:
                    lowest_bid = min(bid[0] for bid in bid_levels)
                    bid_price_range = best_bid_price - lowest_bid
                else:
                    bid_price_range = 0
                
                if ask_levels:
                    highest_ask = max(ask[0] for ask in ask_levels)
                    ask_price_range = highest_ask - best_ask_price
                else:
                    ask_price_range = 0
                
                # Tính tác động giá (price impact): % thay đổi giá khi thực hiện một lệnh có kích thước nhất định
                # Mua: Sẽ làm tăng giá từ mid_price đến mức giá asks
                # Bán: Sẽ làm giảm giá từ mid_price đến mức giá bids
                
                # Chuẩn bị dữ liệu cho price impact
                ask_prices = [ask[0] for ask in ask_levels]
                ask_volumes = [ask[1] for ask in ask_levels]
                ask_cumsum = np.cumsum(ask_volumes)
                
                bid_prices = [bid[0] for bid in bid_levels]
                bid_volumes = [bid[1] for bid in bid_levels]
                bid_cumsum = np.cumsum(bid_volumes)
                
                # Tính price impact khi mua (sẽ làm tăng giá)
                if ask_cumsum.size > 0:
                    # Giá trung bình để mua hết ask_volume_sum khối lượng
                    avg_price_buy = sum(ask_prices[i] * ask_volumes[i] for i in range(len(ask_levels))) / ask_volume_sum if ask_volume_sum > 0 else best_ask_price
                    price_impact_buy = (avg_price_buy - mid_price) / mid_price * 100
                    result_row[f"{prefix}price_impact_buy_{level}"] = price_impact_buy
                else:
                    result_row[f"{prefix}price_impact_buy_{level}"] = 0
                
                # Tính price impact khi bán (sẽ làm giảm giá)
                if bid_cumsum.size > 0:
                    # Giá trung bình để bán hết bid_volume_sum khối lượng
                    avg_price_sell = sum(bid_prices[i] * bid_volumes[i] for i in range(len(bid_levels))) / bid_volume_sum if bid_volume_sum > 0 else best_bid_price
                    price_impact_sell = (mid_price - avg_price_sell) / mid_price * 100
                    result_row[f"{prefix}price_impact_sell_{level}"] = price_impact_sell
                else:
                    result_row[f"{prefix}price_impact_sell_{level}"] = 0
                
                # Tính độ dốc của orderbook (orderbook slope)
                # Độ dốc càng thấp, thanh khoản càng tốt
                
                # Độ dốc bids: thay đổi giá / thay đổi khối lượng tích lũy
                if bid_price_range > 0 and bid_volume_sum > 0:
                    orderbook_slope_bid = bid_price_range / bid_volume_sum
                    result_row[f"{prefix}orderbook_slope_bid_{level}"] = orderbook_slope_bid
                else:
                    result_row[f"{prefix}orderbook_slope_bid_{level}"] = np.nan
                
                # Độ dốc asks: thay đổi giá / thay đổi khối lượng tích lũy
                if ask_price_range > 0 and ask_volume_sum > 0:
                    orderbook_slope_ask = ask_price_range / ask_volume_sum
                    result_row[f"{prefix}orderbook_slope_ask_{level}"] = orderbook_slope_ask
                else:
                    result_row[f"{prefix}orderbook_slope_ask_{level}"] = np.nan
            
            # Tính độ sâu (depth) tại các mức % khác nhau nếu cần
            if depth_buckets:
                for bucket in depth_buckets:
                    # Tính khoảng giá
                    bucket_decimal = bucket / 100
                    bid_price_threshold = mid_price * (1 - bucket_decimal)
                    ask_price_threshold = mid_price * (1 + bucket_decimal)
                    
                    # Tính tổng khối lượng trong khoảng
                    bid_depth_volume = sum(bid[1] for bid in bids if bid[0] >= bid_price_threshold)
                    ask_depth_volume = sum(ask[1] for ask in asks if ask[0] <= ask_price_threshold)
                    
                    # Tính giá trị thanh khoản (khối lượng * giá)
                    bid_depth_value = sum(bid[0] * bid[1] for bid in bids if bid[0] >= bid_price_threshold)
                    ask_depth_value = sum(ask[0] * ask[1] for ask in asks if ask[0] <= ask_price_threshold)
                    
                    # Lưu kết quả
                    result_row[f"{prefix}bid_depth_volume_{bucket}"] = bid_depth_volume
                    result_row[f"{prefix}ask_depth_volume_{bucket}"] = ask_depth_volume
                    result_row[f"{prefix}bid_depth_value_{bucket}"] = bid_depth_value
                    result_row[f"{prefix}ask_depth_value_{bucket}"] = ask_depth_value
                    
                    # Tính tỷ lệ bids/asks
                    if ask_depth_volume > 0:
                        result_row[f"{prefix}depth_volume_ratio_{bucket}"] = bid_depth_volume / ask_depth_volume
                    else:
                        result_row[f"{prefix}depth_volume_ratio_{bucket}"] = np.inf
                    
                    if ask_depth_value > 0:
                        result_row[f"{prefix}depth_value_ratio_{bucket}"] = bid_depth_value / ask_depth_value
                    else:
                        result_row[f"{prefix}depth_value_ratio_{bucket}"] = np.inf
            
            # Thêm vào danh sách kết quả
            results.append(result_row)
        
        # Tạo DataFrame từ danh sách kết quả
        result_df = pd.DataFrame(results)
        
        logger.debug("Đã tính đặc trưng nâng cao từ sổ lệnh")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng nâng cao từ sổ lệnh: {e}")
    
    return result_df

def calculate_market_pressure(
    df: pd.DataFrame,
    levels: List[int] = [5, 10, 20],
    time_windows: List[int] = [1, 5, 10],
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán chỉ số áp lực thị trường từ sổ lệnh theo thời gian.
    
    Args:
        df: DataFrame chứa dữ liệu sổ lệnh (bids, asks) theo thời gian
        levels: Danh sách số lượng level sổ lệnh cần phân tích
        time_windows: Danh sách cửa sổ thời gian để tính toán thay đổi
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa chỉ số áp lực thị trường
    """
    # First, calculate basic orderbook imbalance
    result_df = calculate_orderbook_imbalance(df, levels=levels, prefix=prefix)
    
    # Đảm bảo có dữ liệu cơ bản
    imbalance_cols = [f"{prefix}orderbook_imbalance_{level}" for level in levels]
    if not all(col in result_df.columns for col in imbalance_cols):
        logger.error("Không tìm thấy cột imbalance cần thiết")
        return result_df
    
    try:
        # Tính chỉ số áp lực thị trường dựa trên sự thay đổi của imbalance theo thời gian
        for level in levels:
            imbalance_col = f"{prefix}orderbook_imbalance_{level}"
            
            for window in time_windows:
                # Thay đổi imbalance trong cửa sổ thời gian
                imbalance_change = result_df[imbalance_col].diff(window)
                result_df[f"{prefix}imbalance_change_{level}_{window}"] = imbalance_change
                
                # Tốc độ thay đổi imbalance (đạo hàm)
                result_df[f"{prefix}imbalance_momentum_{level}_{window}"] = imbalance_change / window
                
                # Hướng thay đổi imbalance (1: tăng, -1: giảm, 0: không đổi)
                result_df[f"{prefix}imbalance_direction_{level}_{window}"] = np.sign(imbalance_change)
                
                # Mức độ biến động của imbalance (độ lệch chuẩn)
                result_df[f"{prefix}imbalance_volatility_{level}_{window}"] = (
                    result_df[imbalance_col].rolling(window=window).std()
                )
        
        # Tính áp lực mua/bán gộp
        for window in time_windows:
            # Áp lực mua/bán dựa trên tỷ lệ volume
            bid_volume_cols = [f"{prefix}bid_volume_sum_{level}" for level in levels]
            ask_volume_cols = [f"{prefix}ask_volume_sum_{level}" for level in levels]
            
            if all(col in result_df.columns for col in bid_volume_cols + ask_volume_cols):
                # Tính tổng khối lượng mua và bán
                result_df[f"{prefix}total_bid_volume"] = result_df[bid_volume_cols].sum(axis=1)
                result_df[f"{prefix}total_ask_volume"] = result_df[ask_volume_cols].sum(axis=1)
                
                # Tính áp lực mua/bán
                total_volume = result_df[f"{prefix}total_bid_volume"] + result_df[f"{prefix}total_ask_volume"]
                
                # Tránh chia cho 0
                total_volume_non_zero = total_volume.replace(0, np.nan)
                
                result_df[f"{prefix}buy_pressure_{window}"] = (
                    result_df[f"{prefix}total_bid_volume"].diff(window) / 
                    total_volume_non_zero * 100
                )
                
                result_df[f"{prefix}sell_pressure_{window}"] = (
                    result_df[f"{prefix}total_ask_volume"].diff(window) / 
                    total_volume_non_zero * 100
                )
                
                # Chỉ số áp lực ròng (tăng: áp lực mua, giảm: áp lực bán)
                result_df[f"{prefix}net_pressure_{window}"] = (
                    result_df[f"{prefix}buy_pressure_{window}"] - 
                    result_df[f"{prefix}sell_pressure_{window}"]
                )
        
        # Tính áp lực thị trường dựa trên giá trị thanh khoản (volume * price)
        if 'bids' in df.columns or isinstance(df.get('bids', None), list):
            for window in time_windows:
                # Chỉ số áp lực mua dựa trên giá X*Y
                bid_value_col = f"{prefix}bid_depth_value_{levels[-1]}" if f"{prefix}bid_depth_value_{levels[-1]}" in result_df.columns else None
                ask_value_col = f"{prefix}ask_depth_value_{levels[-1]}" if f"{prefix}ask_depth_value_{levels[-1]}" in result_df.columns else None
                
                if bid_value_col and ask_value_col:
                    # Tính áp lực giá trị
                    bid_value_change = result_df[bid_value_col].diff(window) / result_df[bid_value_col].shift(window) * 100
                    ask_value_change = result_df[ask_value_col].diff(window) / result_df[ask_value_col].shift(window) * 100
                    
                    result_df[f"{prefix}bid_value_pressure_{window}"] = bid_value_change
                    result_df[f"{prefix}ask_value_pressure_{window}"] = ask_value_change
                    
                    # Chỉ số áp lực giá trị ròng
                    result_df[f"{prefix}net_value_pressure_{window}"] = bid_value_change - ask_value_change
        
        logger.debug("Đã tính đặc trưng áp lực thị trường từ sổ lệnh")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng áp lực thị trường: {e}")
    
    return result_df

def calculate_order_flow(
    df: pd.DataFrame,
    trade_size_column: Optional[str] = None,
    trade_price_column: Optional[str] = None,
    trade_side_column: Optional[str] = None,
    time_windows: List[int] = [10, 20, 50],
    use_orderbook: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Tính toán chỉ số order flow từ dữ liệu giao dịch và sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu giao dịch và/hoặc sổ lệnh
        trade_size_column: Tên cột kích thước giao dịch
        trade_price_column: Tên cột giá giao dịch
        trade_side_column: Tên cột phía giao dịch ('buy' hoặc 'sell')
        time_windows: Danh sách cửa sổ thời gian để tính toán order flow
        use_orderbook: Sử dụng dữ liệu sổ lệnh nếu có
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa chỉ số order flow
    """
    result_df = df.copy()
    
    # Kiểm tra dữ liệu giao dịch
    has_trades = (
        trade_size_column is not None and trade_price_column is not None and
        trade_size_column in df.columns and trade_price_column in df.columns
    )
    
    # Kiểm tra dữ liệu sổ lệnh
    has_orderbook = (
        'bids' in df.columns or isinstance(df.get('bids', None), list) or
        ('bid_price_1' in df.columns and 'ask_price_1' in df.columns)
    )
    
    if not has_trades and (not has_orderbook or not use_orderbook):
        logger.error("Không đủ dữ liệu để tính order flow")
        return result_df
    
    try:
        # Phần 1: Tính order flow từ dữ liệu giao dịch
        if has_trades:
            # Phân loại phía giao dịch nếu có cột trade_side
            if trade_side_column and trade_side_column in df.columns:
                # Tính khối lượng giao dịch mua và bán
                buy_size = result_df.loc[result_df[trade_side_column].str.lower() == 'buy', trade_size_column]
                sell_size = result_df.loc[result_df[trade_side_column].str.lower() == 'sell', trade_size_column]
                
                # Chuyển thành Series với index giống DataFrame gốc
                buy_volume = pd.Series(0, index=result_df.index)
                buy_volume.loc[buy_size.index] = buy_size
                
                sell_volume = pd.Series(0, index=result_df.index)
                sell_volume.loc[sell_size.index] = sell_size
                
                # Thêm vào DataFrame
                result_df[f"{prefix}trade_buy_volume"] = buy_volume
                result_df[f"{prefix}trade_sell_volume"] = sell_volume
                
                # Tính order flow imbalance
                total_volume = buy_volume + sell_volume
                total_volume_non_zero = total_volume.replace(0, np.nan)
                
                order_flow = (buy_volume - sell_volume) / total_volume_non_zero
                result_df[f"{prefix}order_flow_imbalance"] = order_flow
                
                # Tính order flow cumulative
                result_df[f"{prefix}order_flow_cumulative"] = (buy_volume - sell_volume).cumsum()
                
                # Tính order flow cho mỗi cửa sổ thời gian
                for window in time_windows:
                    # Tổng khối lượng trong cửa sổ
                    buy_volume_sum = buy_volume.rolling(window=window).sum()
                    sell_volume_sum = sell_volume.rolling(window=window).sum()
                    
                    # Order flow trong cửa sổ
                    total_volume_window = buy_volume_sum + sell_volume_sum
                    total_volume_window_non_zero = total_volume_window.replace(0, np.nan)
                    
                    order_flow_window = (buy_volume_sum - sell_volume_sum) / total_volume_window_non_zero
                    
                    result_df[f"{prefix}order_flow_imbalance_{window}"] = order_flow_window
                    result_df[f"{prefix}order_flow_buy_ratio_{window}"] = buy_volume_sum / total_volume_window_non_zero
                    result_df[f"{prefix}order_flow_cumulative_{window}"] = (buy_volume - sell_volume).rolling(window=window).sum()
            else:
                # Phân loại dựa trên giá giao dịch và mid price từ sổ lệnh
                if use_orderbook and has_orderbook:
                    # Lấy mid price
                    if f"{prefix}mid_price" in result_df.columns:
                        mid_price = result_df[f"{prefix}mid_price"]
                    elif 'bid_price_1' in result_df.columns and 'ask_price_1' in result_df.columns:
                        mid_price = (result_df['bid_price_1'] + result_df['ask_price_1']) / 2
                    else:
                        logger.warning("Không thể xác định mid price từ sổ lệnh")
                        mid_price = None
                    
                    if mid_price is not None:
                        # Tính khối lượng giao dịch theo phía dựa trên giá
                        buy_mask = result_df[trade_price_column] >= mid_price
                        sell_mask = result_df[trade_price_column] < mid_price
                        
                        buy_volume = result_df.loc[buy_mask, trade_size_column]
                        sell_volume = result_df.loc[sell_mask, trade_size_column]
                        
                        # Chuyển thành Series với index giống DataFrame gốc
                        buy_volume_series = pd.Series(0, index=result_df.index)
                        buy_volume_series.loc[buy_volume.index] = buy_volume
                        
                        sell_volume_series = pd.Series(0, index=result_df.index)
                        sell_volume_series.loc[sell_volume.index] = sell_volume
                        
                        # Thêm vào DataFrame
                        result_df[f"{prefix}inferred_buy_volume"] = buy_volume_series
                        result_df[f"{prefix}inferred_sell_volume"] = sell_volume_series
                        
                        # Tính order flow imbalance
                        total_volume = buy_volume_series + sell_volume_series
                        total_volume_non_zero = total_volume.replace(0, np.nan)
                        
                        order_flow = (buy_volume_series - sell_volume_series) / total_volume_non_zero
                        result_df[f"{prefix}inferred_order_flow_imbalance"] = order_flow
                        
                        # Tính order flow cumulative
                        result_df[f"{prefix}inferred_order_flow_cumulative"] = (buy_volume_series - sell_volume_series).cumsum()
                        
                        # Tính inferred order flow cho mỗi cửa sổ thời gian
                        for window in time_windows:
                            # Tổng khối lượng trong cửa sổ
                            buy_volume_sum = buy_volume_series.rolling(window=window).sum()
                            sell_volume_sum = sell_volume_series.rolling(window=window).sum()
                            
                            # Order flow trong cửa sổ
                            total_volume_window = buy_volume_sum + sell_volume_sum
                            total_volume_window_non_zero = total_volume_window.replace(0, np.nan)
                            
                            order_flow_window = (buy_volume_sum - sell_volume_sum) / total_volume_window_non_zero
                            
                            result_df[f"{prefix}inferred_order_flow_imbalance_{window}"] = order_flow_window
                            result_df[f"{prefix}inferred_order_flow_buy_ratio_{window}"] = buy_volume_sum / total_volume_window_non_zero
                            result_df[f"{prefix}inferred_order_flow_cumulative_{window}"] = (
                                (buy_volume_series - sell_volume_series).rolling(window=window).sum()
                            )
        
        # Phần 2: Tính delta volume từ sổ lệnh (thay đổi khối lượng trong sổ lệnh theo thời gian)
        if use_orderbook and has_orderbook:
            # Lấy các cột imbalance đã tính từ hàm calculate_orderbook_imbalance
            imbalance_cols = [col for col in result_df.columns if col.startswith(f"{prefix}orderbook_imbalance_")]
            
            if imbalance_cols:
                # Tính thay đổi imbalance
                for col in imbalance_cols:
                    for window in time_windows:
                        result_df[f"{col}_change_{window}"] = result_df[col].diff(window)
                
                # Tính thay đổi khối lượng trong sổ lệnh
                bid_volume_cols = [col for col in result_df.columns if col.startswith(f"{prefix}bid_volume_sum_")]
                ask_volume_cols = [col for col in result_df.columns if col.startswith(f"{prefix}ask_volume_sum_")]
                
                for bid_col, ask_col in zip(bid_volume_cols, ask_volume_cols):
                    level = bid_col.split('_')[-1]
                    
                    for window in time_windows:
                        # Thay đổi khối lượng bid và ask
                        result_df[f"{prefix}bid_volume_delta_{level}_{window}"] = result_df[bid_col].diff(window)
                        result_df[f"{prefix}ask_volume_delta_{level}_{window}"] = result_df[ask_col].diff(window)
                        
                        # Tỷ lệ thay đổi
                        result_df[f"{prefix}bid_volume_delta_ratio_{level}_{window}"] = (
                            result_df[bid_col].diff(window) / result_df[bid_col].shift(window) * 100
                        )
                        
                        result_df[f"{prefix}ask_volume_delta_ratio_{level}_{window}"] = (
                            result_df[ask_col].diff(window) / result_df[ask_col].shift(window) * 100
                        )
                        
                        # Chỉ số delta
                        bid_delta = result_df[f"{prefix}bid_volume_delta_{level}_{window}"]
                        ask_delta = result_df[f"{prefix}ask_volume_delta_{level}_{window}"]
                        
                        delta_sum = bid_delta.abs() + ask_delta.abs()
                        delta_sum_non_zero = delta_sum.replace(0, np.nan)
                        
                        delta_index = (bid_delta - ask_delta) / delta_sum_non_zero
                        result_df[f"{prefix}delta_index_{level}_{window}"] = delta_index
                
                # Tính chỉ số order flow kết hợp (orderbook + trades nếu có)
                if has_trades and trade_side_column in df.columns:
                    # Lấy level lớn nhất có sẵn
                    max_level = max(int(col.split('_')[-1]) for col in bid_volume_cols)
                    
                    for window in time_windows:
                        # Kết hợp order flow từ giao dịch và orderbook
                        if f"{prefix}order_flow_imbalance_{window}" in result_df.columns and f"{prefix}delta_index_{max_level}_{window}" in result_df.columns:
                            result_df[f"{prefix}combined_order_flow_{window}"] = (
                                result_df[f"{prefix}order_flow_imbalance_{window}"] + 
                                result_df[f"{prefix}delta_index_{max_level}_{window}"]
                            ) / 2
        
        logger.debug("Đã tính đặc trưng order flow")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng order flow: {e}")
    
    return result_df

def calculate_liquidity_distribution(
    df: pd.DataFrame,
    num_buckets: int = 10,
    max_distance_percent: float = 5.0,
    use_log_scale: bool = True,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phân tích phân phối thanh khoản trong sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu sổ lệnh (bids, asks)
        num_buckets: Số lượng khoảng giá để phân tích
        max_distance_percent: Khoảng cách tối đa từ mid price (%)
        use_log_scale: Sử dụng thang logarithm cho các khoảng giá
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa phân phối thanh khoản
    """
    result_df = df.copy()
    
    # Kiểm tra cấu trúc dữ liệu sổ lệnh
    if not isinstance(df.get('bids', None), list) and 'bids' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin bids trong sổ lệnh")
        return result_df
    
    if not isinstance(df.get('asks', None), list) and 'asks' not in df.columns:
        logger.error("Dữ liệu không hợp lệ: thiếu thông tin asks trong sổ lệnh")
        return result_df
    
    try:
        # Phân tích cấu trúc sổ lệnh
        orderbook_format = ''
        
        # Xác định định dạng dữ liệu sổ lệnh
        if 'bids' in df.columns and isinstance(df['bids'].iloc[0], list):
            # Định dạng cột chứa danh sách
            orderbook_format = 'column_list'
        elif isinstance(df.get('bids', None), list):
            # Định dạng thuộc tính trực tiếp
            orderbook_format = 'direct_property'
        elif 'bid_price_1' in df.columns:
            # Định dạng cột riêng cho từng level
            orderbook_format = 'level_columns'
        else:
            logger.error("Không thể xác định định dạng dữ liệu sổ lệnh")
            return result_df
        
        # Danh sách để lưu trữ kết quả
        results = []
        
        # Tạo các khoảng giá
        if use_log_scale:
            # Tạo khoảng giá theo thang logarithm
            bid_distances = -np.logspace(np.log10(0.001), np.log10(max_distance_percent), num_buckets) / 100
            ask_distances = np.logspace(np.log10(0.001), np.log10(max_distance_percent), num_buckets) / 100
        else:
            # Tạo khoảng giá tuyến tính
            bid_distances = -np.linspace(0.001, max_distance_percent, num_buckets) / 100
            ask_distances = np.linspace(0.001, max_distance_percent, num_buckets) / 100
        
        # Xử lý từng hàng dữ liệu
        for idx, row in df.iterrows():
            # Lấy dữ liệu sổ lệnh cho hàng hiện tại
            bids = None
            asks = None
            
            if orderbook_format == 'column_list':
                bids = row['bids']
                asks = row['asks']
            elif orderbook_format == 'direct_property':
                bids = df.at[idx, 'bids']
                asks = df.at[idx, 'asks']
            elif orderbook_format == 'level_columns':
                # Tạo danh sách bids, asks từ các cột riêng lẻ
                bids = []
                asks = []
                
                max_level = 0
                for col in df.columns:
                    if col.startswith('bid_price_'):
                        level = int(col.split('_')[-1])
                        max_level = max(max_level, level)
                
                for level in range(1, max_level + 1):
                    if f'bid_price_{level}' in df.columns and f'bid_volume_{level}' in df.columns:
                        bid_price = row[f'bid_price_{level}']
                        bid_volume = row[f'bid_volume_{level}']
                        if not pd.isna(bid_price) and not pd.isna(bid_volume):
                            bids.append([bid_price, bid_volume])
                    
                    if f'ask_price_{level}' in df.columns and f'ask_volume_{level}' in df.columns:
                        ask_price = row[f'ask_price_{level}']
                        ask_volume = row[f'ask_volume_{level}']
                        if not pd.isna(ask_price) and not pd.isna(ask_volume):
                            asks.append([ask_price, ask_volume])
            
            # Kiểm tra xem có dữ liệu không
            if not bids or not asks:
                logger.warning(f"Không có dữ liệu sổ lệnh hợp lệ cho hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                result_row = {**{col: row[col] for col in df.columns}}
                
                # Thêm các cột phân phối
                for i in range(num_buckets):
                    result_row[f"{prefix}bid_liquidity_dist_{i+1}"] = np.nan
                    result_row[f"{prefix}ask_liquidity_dist_{i+1}"] = np.nan
                
                results.append(result_row)
                continue
            
            # Tạo dictionary cho kết quả của hàng hiện tại
            result_row = {**{col: row[col] for col in df.columns}}
            
            # Tính mid price
            best_bid_price = bids[0][0] if len(bids) > 0 else 0
            best_ask_price = asks[0][0] if len(asks) > 0 else 0
            
            if best_bid_price == 0 or best_ask_price == 0:
                logger.warning(f"Giá mua/bán tốt nhất không hợp lệ tại hàng {idx}")
                
                # Thêm giá trị NaN cho hàng này
                for i in range(num_buckets):
                    result_row[f"{prefix}bid_liquidity_dist_{i+1}"] = np.nan
                    result_row[f"{prefix}ask_liquidity_dist_{i+1}"] = np.nan
                
                results.append(result_row)
                continue
            
            mid_price = (best_bid_price + best_ask_price) / 2
            
            # Chuyển đổi khoảng giá tương đối thành giá tuyệt đối
            bid_prices = [mid_price * (1 + dist) for dist in bid_distances]
            ask_prices = [mid_price * (1 + dist) for dist in ask_distances]
            
            # Tính phân phối thanh khoản
            bid_distribution = np.zeros(num_buckets)
            ask_distribution = np.zeros(num_buckets)
            
            # Phân phối cho bids
            for i in range(num_buckets - 1):
                lower_price = bid_prices[i+1]
                upper_price = bid_prices[i]
                
                # Tính tổng khối lượng trong khoảng giá
                volume_in_range = sum(bid[1] for bid in bids if lower_price <= bid[0] < upper_price)
                bid_distribution[i] = volume_in_range
            
            # Khối lượng ở bucket cuối cùng
            bid_distribution[-1] = sum(bid[1] for bid in bids if bid[0] < bid_prices[-1])
            
            # Phân phối cho asks
            for i in range(num_buckets - 1):
                lower_price = ask_prices[i]
                upper_price = ask_prices[i+1]
                
                # Tính tổng khối lượng trong khoảng giá
                volume_in_range = sum(ask[1] for ask in asks if lower_price <= ask[0] < upper_price)
                ask_distribution[i] = volume_in_range
            
            # Khối lượng ở bucket cuối cùng
            ask_distribution[-1] = sum(ask[1] for ask in asks if ask[0] >= ask_prices[-1])
            
            # Chuẩn hóa phân phối
            total_bid_volume = sum(bid_distribution)
            total_ask_volume = sum(ask_distribution)
            
            if total_bid_volume > 0:
                bid_distribution = bid_distribution / total_bid_volume
            
            if total_ask_volume > 0:
                ask_distribution = ask_distribution / total_ask_volume
            
            # Thêm vào result_row
            for i in range(num_buckets):
                result_row[f"{prefix}bid_liquidity_dist_{i+1}"] = bid_distribution[i]
                result_row[f"{prefix}ask_liquidity_dist_{i+1}"] = ask_distribution[i]
            
            # Thêm vào danh sách kết quả
            results.append(result_row)
        
        # Tạo DataFrame từ danh sách kết quả
        result_df = pd.DataFrame(results)
        
        # Tính thêm các chỉ số mô tả phân phối
        # Entropy (độ phân tán của thanh khoản, càng cao càng phân tán đều)
        for side in ['bid', 'ask']:
            dist_cols = [f"{prefix}{side}_liquidity_dist_{i+1}" for i in range(num_buckets)]
            
            if all(col in result_df.columns for col in dist_cols):
                # Tính entropy
                entropy = np.zeros(len(result_df))
                
                for i, row in result_df.iterrows():
                    distribution = np.array([row[col] for col in dist_cols])
                    # Loại bỏ các phần tử bằng 0
                    distribution = distribution[distribution > 0]
                    
                    if len(distribution) > 0:
                        entropy[i] = -np.sum(distribution * np.log2(distribution))
                
                result_df[f"{prefix}{side}_liquidity_entropy"] = entropy
                
                # Tính chỉ số tập trung thanh khoản (tỷ lệ thanh khoản ở 20% gần mid price)
                concentration = result_df[dist_cols[:num_buckets//5]].sum(axis=1)
                result_df[f"{prefix}{side}_liquidity_concentration"] = concentration
        
        logger.debug("Đã tính đặc trưng phân phối thanh khoản")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng phân phối thanh khoản: {e}")
    
    return result_df

def calculate_support_resistance_levels(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: Optional[str] = 'volume',
    num_levels: int = 3,
    use_orderbook: bool = True,
    window: int = 100,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Phát hiện mức hỗ trợ và kháng cự từ dữ liệu giá và/hoặc sổ lệnh.
    
    Args:
        df: DataFrame chứa dữ liệu giá, khối lượng và/hoặc sổ lệnh
        price_column: Tên cột giá
        volume_column: Tên cột khối lượng
        num_levels: Số lượng mức hỗ trợ/kháng cự cần phát hiện
        use_orderbook: Sử dụng dữ liệu sổ lệnh nếu có
        window: Kích thước cửa sổ lịch sử để phát hiện mức
        prefix: Tiền tố cho tên cột kết quả
        
    Returns:
        DataFrame với các cột mới chứa mức hỗ trợ và kháng cự
    """
    result_df = df.copy()
    
    # Kiểm tra dữ liệu giá
    if price_column not in df.columns:
        logger.error(f"Dữ liệu không hợp lệ: thiếu cột {price_column}")
        return result_df
    
    try:
        # 1. Phát hiện mức hỗ trợ và kháng cự từ giá lịch sử
        for i in range(len(result_df)):
            # Lấy cửa sổ dữ liệu lịch sử
            start_idx = max(0, i - window)
            historical_data = result_df.iloc[start_idx:i+1]
            
            if len(historical_data) < 10:  # Cần ít nhất 10 điểm dữ liệu
                continue
            
            current_price = result_df.iloc[i][price_column]
            
            # Tìm các đỉnh và đáy cục bộ
            price_series = historical_data[price_column]
            
            # Tìm đỉnh cục bộ (yêu cầu ít nhất 2 điểm ở mỗi bên)
            peaks = (price_series > price_series.shift(2)) & (price_series > price_series.shift(1)) & \
                    (price_series > price_series.shift(-1)) & (price_series > price_series.shift(-2))
            
            # Tìm đáy cục bộ (yêu cầu ít nhất 2 điểm ở mỗi bên)
            troughs = (price_series < price_series.shift(2)) & (price_series < price_series.shift(1)) & \
                     (price_series < price_series.shift(-1)) & (price_series < price_series.shift(-2))
            
            # Lấy giá tại các đỉnh và đáy
            peak_prices = price_series[peaks].values
            trough_prices = price_series[troughs].values
            
            # Nếu có dữ liệu khối lượng, tính trọng số dựa trên khối lượng
            if volume_column in df.columns:
                volume_series = historical_data[volume_column]
                peak_volumes = volume_series[peaks].values
                trough_volumes = volume_series[troughs].values
                
                # Chuẩn hóa khối lượng
                if len(peak_volumes) > 0:
                    peak_volumes = peak_volumes / peak_volumes.max()
                
                if len(trough_volumes) > 0:
                    trough_volumes = trough_volumes / trough_volumes.max()
            else:
                # Nếu không có khối lượng, sử dụng trọng số đều
                peak_volumes = np.ones_like(peak_prices) if len(peak_prices) > 0 else np.array([])
                trough_volumes = np.ones_like(trough_prices) if len(trough_prices) > 0 else np.array([])
            
            # Phát hiện mức bằng cách nhóm giá trị gần nhau
            def cluster_levels(prices, volumes, tolerance=0.02):
                if len(prices) == 0:
                    return []
                
                # Nhóm các giá trị trong khoảng tolerance
                clusters = []
                current_cluster = {'prices': [prices[0]], 'volumes': [volumes[0]]}
                
                for j in range(1, len(prices)):
                    # Tính giá trung bình của cluster hiện tại
                    cluster_avg = np.average(current_cluster['prices'], 
                                            weights=current_cluster['volumes'])
                    
                    # Nếu giá hiện tại nằm trong khoảng tolerance
                    if abs(prices[j] - cluster_avg) / cluster_avg <= tolerance:
                        current_cluster['prices'].append(prices[j])
                        current_cluster['volumes'].append(volumes[j])
                    else:
                        # Tính giá trung bình có trọng số cho cluster hiện tại
                        weighted_avg = np.average(current_cluster['prices'], 
                                                weights=current_cluster['volumes'])
                        
                        # Thêm cluster vào danh sách
                        clusters.append({
                            'price': weighted_avg,
                            'strength': sum(current_cluster['volumes']),
                            'count': len(current_cluster['prices'])
                        })
                        
                        # Bắt đầu cluster mới
                        current_cluster = {'prices': [prices[j]], 'volumes': [volumes[j]]}
                
                # Thêm cluster cuối cùng
                if current_cluster['prices']:
                    weighted_avg = np.average(current_cluster['prices'], 
                                            weights=current_cluster['volumes'])
                    
                    clusters.append({
                        'price': weighted_avg,
                        'strength': sum(current_cluster['volumes']),
                        'count': len(current_cluster['prices'])
                    })
                
                # Sắp xếp theo độ mạnh giảm dần
                return sorted(clusters, key=lambda x: (x['strength'], x['count']), reverse=True)
            
            # Phát hiện mức kháng cự (giá lớn hơn giá hiện tại)
            resistance_candidates = peak_prices[peak_prices > current_price]
            resistance_volumes = peak_volumes[peak_prices > current_price] if len(peak_volumes) > 0 else np.array([])
            
            resistance_clusters = cluster_levels(resistance_candidates, resistance_volumes)
            
            # Phát hiện mức hỗ trợ (giá nhỏ hơn giá hiện tại)
            support_candidates = trough_prices[trough_prices < current_price]
            support_volumes = trough_volumes[trough_prices < current_price] if len(trough_volumes) > 0 else np.array([])
            
            support_clusters = cluster_levels(support_candidates, support_volumes)
            
            # Lấy top num_levels mức kháng cự và hỗ trợ
            resistance_levels = [cluster['price'] for cluster in resistance_clusters[:num_levels]]
            support_levels = [cluster['price'] for cluster in support_clusters[:num_levels]]
            
            # Đảm bảo đủ số lượng mức bằng cách thêm NaN
            resistance_levels += [np.nan] * (num_levels - len(resistance_levels))
            support_levels += [np.nan] * (num_levels - len(support_levels))
            
            # Thêm vào DataFrame
            for j in range(num_levels):
                result_df.loc[result_df.index[i], f"{prefix}resistance_level_{j+1}"] = resistance_levels[j]
                result_df.loc[result_df.index[i], f"{prefix}support_level_{j+1}"] = support_levels[j]
        
        # 2. Phát hiện mức từ sổ lệnh nếu có
        if use_orderbook:
            has_orderbook = (
                'bids' in df.columns or isinstance(df.get('bids', None), list) or
                ('bid_price_1' in df.columns and 'ask_price_1' in df.columns)
            )
            
            if has_orderbook:
                # Tính mức hỗ trợ/kháng cự từ tập trung khối lượng trong sổ lệnh
                for i in range(len(result_df)):
                    try:
                        # Kiểm tra có thông tin sổ lệnh không
                        bids = None
                        asks = None
                        
                        if 'bids' in result_df.columns and isinstance(result_df['bids'].iloc[i], list):
                            bids = result_df['bids'].iloc[i]
                            asks = result_df['asks'].iloc[i]
                        elif isinstance(df.get('bids', None), list):
                            bids = df.at[result_df.index[i], 'bids']
                            asks = df.at[result_df.index[i], 'asks']
                        elif 'bid_price_1' in result_df.columns:
                            # Tạo danh sách bids, asks từ các cột riêng lẻ
                            bids = []
                            asks = []
                            
                            max_level = 0
                            for col in result_df.columns:
                                if col.startswith('bid_price_'):
                                    level = int(col.split('_')[-1])
                                    max_level = max(max_level, level)
                            
                            for level in range(1, max_level + 1):
                                bid_col = f'bid_price_{level}'
                                bid_vol_col = f'bid_volume_{level}'
                                ask_col = f'ask_price_{level}'
                                ask_vol_col = f'ask_volume_{level}'
                                
                                if all(col in result_df.columns for col in [bid_col, bid_vol_col]):
                                    bid_price = result_df.iloc[i][bid_col]
                                    bid_volume = result_df.iloc[i][bid_vol_col]
                                    if not pd.isna(bid_price) and not pd.isna(bid_volume):
                                        bids.append([bid_price, bid_volume])
                                
                                if all(col in result_df.columns for col in [ask_col, ask_vol_col]):
                                    ask_price = result_df.iloc[i][ask_col]
                                    ask_volume = result_df.iloc[i][ask_vol_col]
                                    if not pd.isna(ask_price) and not pd.isna(ask_volume):
                                        asks.append([ask_price, ask_volume])
                        
                        if not bids or not asks:
                            continue
                        
                        # Tìm mức tập trung khối lượng
                        def find_volume_clusters(orders, threshold=0.05):
                            if not orders:
                                return []
                            
                            # Tính tổng khối lượng
                            total_volume = sum(order[1] for order in orders)
                            
                            # Nhóm các mức giá theo khoảng giá
                            prices = np.array([order[0] for order in orders])
                            volumes = np.array([order[1] for order in orders])
                            
                            # Nhóm các giá trị trong khoảng threshold
                            clusters = []
                            current_cluster = {'prices': [prices[0]], 'volumes': [volumes[0]]}
                            
                            for j in range(1, len(prices)):
                                # Tính giá trung bình của cluster hiện tại
                                cluster_avg = np.average(current_cluster['prices'], 
                                                        weights=current_cluster['volumes'])
                                
                                # Nếu giá hiện tại nằm trong khoảng threshold
                                if abs(prices[j] - cluster_avg) / cluster_avg <= threshold:
                                    current_cluster['prices'].append(prices[j])
                                    current_cluster['volumes'].append(volumes[j])
                                else:
                                    # Tính giá trung bình có trọng số cho cluster hiện tại
                                    weighted_avg = np.average(current_cluster['prices'], 
                                                            weights=current_cluster['volumes'])
                                    
                                    # Tính % khối lượng
                                    volume_percent = sum(current_cluster['volumes']) / total_volume
                                    
                                    # Thêm cluster nếu có > 5% tổng khối lượng
                                    if volume_percent >= 0.05:
                                        clusters.append({
                                            'price': weighted_avg,
                                            'volume': sum(current_cluster['volumes']),
                                            'volume_percent': volume_percent
                                        })
                                    
                                    # Bắt đầu cluster mới
                                    current_cluster = {'prices': [prices[j]], 'volumes': [volumes[j]]}
                            
                            # Thêm cluster cuối cùng
                            if current_cluster['prices']:
                                weighted_avg = np.average(current_cluster['prices'], 
                                                        weights=current_cluster['volumes'])
                                
                                volume_percent = sum(current_cluster['volumes']) / total_volume
                                
                                if volume_percent >= 0.05:
                                    clusters.append({
                                        'price': weighted_avg,
                                        'volume': sum(current_cluster['volumes']),
                                        'volume_percent': volume_percent
                                    })
                            
                            # Sắp xếp theo % khối lượng giảm dần
                            return sorted(clusters, key=lambda x: x['volume_percent'], reverse=True)
                        
                        # Tìm mức hỗ trợ từ bids
                        bid_clusters = find_volume_clusters(bids)
                        
                        # Tìm mức kháng cự từ asks
                        ask_clusters = find_volume_clusters(asks)
                        
                        # Thêm vào DataFrame
                        for j in range(min(num_levels, len(bid_clusters))):
                            col_name = f"{prefix}ob_support_level_{j+1}"
                            if col_name not in result_df.columns:
                                result_df[col_name] = np.nan
                            
                            result_df.loc[result_df.index[i], col_name] = bid_clusters[j]['price']
                        
                        for j in range(min(num_levels, len(ask_clusters))):
                            col_name = f"{prefix}ob_resistance_level_{j+1}"
                            if col_name not in result_df.columns:
                                result_df[col_name] = np.nan
                            
                            result_df.loc[result_df.index[i], col_name] = ask_clusters[j]['price']
                    
                    except Exception as e:
                        logger.warning(f"Lỗi khi tính mức hỗ trợ/kháng cự từ sổ lệnh tại hàng {i}: {e}")
                        continue
        
        # 3. Tính khoảng cách đến các mức hỗ trợ/kháng cự
        current_prices = result_df[price_column]
        
        # Tính khoảng cách đến mức từ giá lịch sử
        for j in range(num_levels):
            if f"{prefix}resistance_level_{j+1}" in result_df.columns:
                resistance_col = f"{prefix}resistance_level_{j+1}"
                result_df[f"{prefix}distance_to_resistance_{j+1}"] = (
                    (result_df[resistance_col] - current_prices) / current_prices * 100
                )
            
            if f"{prefix}support_level_{j+1}" in result_df.columns:
                support_col = f"{prefix}support_level_{j+1}"
                result_df[f"{prefix}distance_to_support_{j+1}"] = (
                    (current_prices - result_df[support_col]) / current_prices * 100
                )
        
        # Tính khoảng cách đến mức từ sổ lệnh
        if use_orderbook:
            for j in range(num_levels):
                if f"{prefix}ob_resistance_level_{j+1}" in result_df.columns:
                    resistance_col = f"{prefix}ob_resistance_level_{j+1}"
                    result_df[f"{prefix}distance_to_ob_resistance_{j+1}"] = (
                        (result_df[resistance_col] - current_prices) / current_prices * 100
                    )
                
                if f"{prefix}ob_support_level_{j+1}" in result_df.columns:
                    support_col = f"{prefix}ob_support_level_{j+1}"
                    result_df[f"{prefix}distance_to_ob_support_{j+1}"] = (
                        (current_prices - result_df[support_col]) / current_prices * 100
                    )
        
        logger.debug("Đã tính đặc trưng mức hỗ trợ và kháng cự")
        
    except Exception as e:
        logger.error(f"Lỗi khi tính đặc trưng mức hỗ trợ và kháng cự: {e}")
    
    return result_df