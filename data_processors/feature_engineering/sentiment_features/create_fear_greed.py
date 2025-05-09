"""
Tạo chỉ số Fear & Greed dựa trên hành động giá.
File này cung cấp các hàm để tính toán và tạo chỉ số Fear & Greed
dựa trên dữ liệu về giá, khối lượng và các chỉ báo kỹ thuật.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import math

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
    relative_strength_index, commodity_channel_index, money_flow_index, rate_of_change
)
from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
    average_true_range, bollinger_bandwidth, ulcer_index, standard_deviation, historical_volatility
)
from data_processors.feature_engineering.technical_indicators.utils import (
    prepare_price_data, validate_price_data, normalize_indicator
)

# Thiết lập logger
logger = get_logger("sentiment_features")

class FearGreedIndex:
    """
    Lớp tính toán chỉ số Fear & Greed dựa trên hành động giá.
    Chỉ số này đo lường tâm lý thị trường từ "Extreme Fear" (0) đến "Extreme Greed" (100).
    """
    
    def __init__(self, lookback_period: int = 90):
        """
        Khởi tạo đối tượng FearGreedIndex.
        
        Args:
            lookback_period: Số ngày nhìn lại để chuẩn hóa các thành phần
        """
        self.lookback_period = lookback_period
        
        # Trọng số các thành phần trong chỉ số Fear & Greed
        self.component_weights = {
            'rsi': 0.20,             # Chỉ số sức mạnh tương đối
            'volatility': 0.15,      # Biến động giá
            'momentum': 0.15,        # Xung lượng giá
            'market_trend': 0.15,    # Xu hướng thị trường
            'volume': 0.10,          # Khối lượng giao dịch
            'put_call_ratio': 0.15,  # Tỉ lệ đặt lệnh bán/mua (mô phỏng)
            'price_strength': 0.10   # Sức mạnh giá
        }
        
        # Bảng phân loại giá trị Fear & Greed
        self.fear_greed_classification = {
            (0, 25): "Extreme Fear",
            (25, 45): "Fear",
            (45, 55): "Neutral",
            (55, 75): "Greed",
            (75, 101): "Extreme Greed"
        }
    
    def calculate_rsi_component(
        self, 
        df: pd.DataFrame, 
        column: str = 'close',
        window: int = 14
    ) -> pd.Series:
        """
        Tính toán thành phần RSI của chỉ số.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            column: Tên cột dùng để tính RSI
            window: Kích thước cửa sổ cho RSI
            
        Returns:
            Series chứa giá trị thành phần RSI (0-100)
        """
        if not validate_price_data(df, [column]):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        # Tính RSI
        result_df = relative_strength_index(
            df, 
            column=column, 
            window=window,
            normalize=False
        )
        
        rsi_col = f"rsi_{window}"
        if rsi_col not in result_df.columns:
            raise ValueError(f"Không tìm thấy cột {rsi_col} trong kết quả RSI")
        
        # RSI đã trong khoảng 0-100, không cần chuẩn hóa thêm
        # RSI < 30: extreme fear, RSI > 70: extreme greed
        return result_df[rsi_col]
    
    def calculate_volatility_component(
        self, 
        df: pd.DataFrame,
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Tính toán thành phần biến động của chỉ số.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần biến động (0-100)
        """
        required_columns = ['high', 'low', 'close']
        if not validate_price_data(df, required_columns):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
        
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        try:
            # Tạo bản sao DataFrame để tránh thay đổi dữ liệu gốc
            result_df = df.copy()
            
            # Tính ATR
            result_df = average_true_range(
                result_df,
                window=14,
                normalize_by_price=True
            )
            
            # Tính độ lệch chuẩn
            result_df = standard_deviation(
                result_df,
                window=20,
                trading_periods=252
            )
            
            # Tính Bollinger Bandwidth
            try:
                result_df = bollinger_bandwidth(
                    result_df,
                    window=20,
                    std_dev=2.0
                )
                has_bbw = 'bbw_20' in result_df.columns
            except Exception as e:
                logger.warning(f"Không thể tính Bollinger Bandwidth: {str(e)}")
                has_bbw = False
                
                # Tự tính Bollinger Bandwidth nếu hàm bollinger_bandwidth() thất bại
                try:
                    if 'sma_20' not in result_df.columns:
                        result_df['sma_20'] = result_df['close'].rolling(window=20).mean()
                    
                    if 'stddev_20' in result_df.columns:
                        upper_band = result_df['sma_20'] + (result_df['stddev_20'] * 2)
                        lower_band = result_df['sma_20'] - (result_df['stddev_20'] * 2)
                        result_df['bbw_20'] = (upper_band - lower_band) / result_df['sma_20']
                        has_bbw = True
                except Exception:
                    logger.warning("Không thể tự tính Bollinger Bandwidth")
                    has_bbw = False
            
            # Kết hợp các thành phần biến động
            # 1. Chuẩn hóa ATR
            atr_col = 'atr_pct_14'
            if atr_col in result_df.columns:
                atr_pct = result_df[atr_col]
                atr_min = atr_pct.rolling(lookback_period).min()
                atr_max = atr_pct.rolling(lookback_period).max()
                atr_range = atr_max - atr_min
                atr_normalized = ((atr_pct - atr_min) / atr_range.replace(0, 1)).clip(0, 1)
            else:
                logger.warning(f"Không tìm thấy cột {atr_col}, sử dụng giá trị mặc định")
                atr_normalized = pd.Series(0.5, index=df.index)
            
            # 2. Chuẩn hóa độ lệch chuẩn
            stddev_col = 'stddev_20'
            if stddev_col in result_df.columns:
                stddev = result_df[stddev_col]
                stddev_min = stddev.rolling(lookback_period).min()
                stddev_max = stddev.rolling(lookback_period).max()
                stddev_range = stddev_max - stddev_min
                stddev_normalized = ((stddev - stddev_min) / stddev_range.replace(0, 1)).clip(0, 1)
            else:
                logger.warning(f"Không tìm thấy cột {stddev_col}, sử dụng giá trị mặc định")
                stddev_normalized = pd.Series(0.5, index=df.index)
            
            # 3. Chuẩn hóa Bollinger Bandwidth
            bbw_col = 'bbw_20'
            if has_bbw and bbw_col in result_df.columns:
                bbw = result_df[bbw_col]
                bbw_min = bbw.rolling(lookback_period).min()
                bbw_max = bbw.rolling(lookback_period).max()
                bbw_range = bbw_max - bbw_min
                bbw_normalized = ((bbw - bbw_min) / bbw_range.replace(0, 1)).clip(0, 1)
            else:
                logger.warning(f"Không tìm thấy cột {bbw_col}, sử dụng giá trị mặc định")
                bbw_normalized = pd.Series(0.5, index=df.index)
            
            # Xử lý trường hợp tất cả các thành phần đều thiếu
            if atr_normalized.isna().all() and stddev_normalized.isna().all() and bbw_normalized.isna().all():
                logger.warning("Tất cả các thành phần volatility đều thiếu, trả về 50 cho tất cả")
                return pd.Series(50, index=df.index)
            
            # Chuyển đổi: biến động cao = extreme fear (0), biến động thấp = extreme greed (100)
            # Điều chỉnh trọng số khi thiếu thành phần
            weights = {'atr': 0.4, 'stddev': 0.3, 'bbw': 0.3}
            total_weight = sum(weights.values())
            
            volatility_component = 100 * (1 - (
                (atr_normalized * weights['atr'] + 
                stddev_normalized * weights['stddev'] + 
                bbw_normalized * weights['bbw']) / total_weight
            ))
            
            # Điền các giá trị NaN với giá trị trung bình
            volatility_component = volatility_component.fillna(50)
            
            return volatility_component
        
        except Exception as e:
            logger.error(f"Lỗi trong calculate_volatility_component: {str(e)}")
            # Trả về giá trị trung lập nếu có lỗi
            return pd.Series(50, index=df.index)
    
    def calculate_momentum_component(
        self, 
        df: pd.DataFrame,
        column: str = 'close',
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Tính toán thành phần xung lượng của chỉ số.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            column: Tên cột dùng để tính xung lượng
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần xung lượng (0-100)
        """
        try:
            if not validate_price_data(df, [column]):
                raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
            
            if lookback_period is None:
                lookback_period = self.lookback_period
            
            # Tính Rate of Change cho nhiều khung thời gian
            result_df = df.copy()
            
            # Tính ROC-1, ROC-5, ROC-10, ROC-20
            for window in [1, 5, 10, 20]:
                try:
                    result_df = rate_of_change(
                        result_df,
                        column=column,
                        window=window,
                        percentage=True
                    )
                except Exception as e:
                    logger.warning(f"Lỗi khi tính ROC-{window}: {str(e)}")
                    # Tạo cột ROC giả
                    result_df[f'roc_{window}'] = np.zeros(len(result_df))
            
            # Tính Money Flow Index
            has_mfi = False
            if 'volume' in df.columns:
                try:
                    result_df = money_flow_index(
                        result_df,
                        window=14
                    )
                    has_mfi = 'mfi_14' in result_df.columns
                except Exception as e:
                    logger.warning(f"Lỗi khi tính Money Flow Index: {str(e)}")
                    has_mfi = False
            
            # Kết hợp các thành phần xung lượng
            momentum_weights = {
                'roc_1': 0.1,
                'roc_5': 0.2,
                'roc_10': 0.3,
                'roc_20': 0.2,
                'mfi': 0.2
            }
            
            # Chuẩn hóa các thành phần và xử lý ngoại lệ
            components = {}
            total_weight = 0
            
            # 1. ROC-1
            if 'roc_1' in result_df.columns:
                try:
                    roc_1 = result_df['roc_1']
                    roc_1_min = roc_1.rolling(lookback_period).min()
                    roc_1_max = roc_1.rolling(lookback_period).max()
                    roc_1_range = roc_1_max - roc_1_min
                    components['roc_1'] = ((roc_1 - roc_1_min) / roc_1_range.replace(0, 1)).clip(0, 1)
                    total_weight += momentum_weights['roc_1']
                except Exception as e:
                    logger.warning(f"Lỗi khi chuẩn hóa ROC-1: {str(e)}")
                    components['roc_1'] = pd.Series(0.5, index=df.index)
                    total_weight += momentum_weights['roc_1']
            else:
                logger.warning("Không tìm thấy cột roc_1, sử dụng giá trị mặc định")
                components['roc_1'] = pd.Series(0.5, index=df.index)
                total_weight += momentum_weights['roc_1']
            
            # 2. ROC-5
            if 'roc_5' in result_df.columns:
                try:
                    roc_5 = result_df['roc_5']
                    roc_5_min = roc_5.rolling(lookback_period).min()
                    roc_5_max = roc_5.rolling(lookback_period).max()
                    roc_5_range = roc_5_max - roc_5_min
                    components['roc_5'] = ((roc_5 - roc_5_min) / roc_5_range.replace(0, 1)).clip(0, 1)
                    total_weight += momentum_weights['roc_5']
                except Exception as e:
                    logger.warning(f"Lỗi khi chuẩn hóa ROC-5: {str(e)}")
                    components['roc_5'] = pd.Series(0.5, index=df.index)
                    total_weight += momentum_weights['roc_5']
            else:
                logger.warning("Không tìm thấy cột roc_5, sử dụng giá trị mặc định")
                components['roc_5'] = pd.Series(0.5, index=df.index)
                total_weight += momentum_weights['roc_5']
            
            # 3. ROC-10
            if 'roc_10' in result_df.columns:
                try:
                    roc_10 = result_df['roc_10']
                    roc_10_min = roc_10.rolling(lookback_period).min()
                    roc_10_max = roc_10.rolling(lookback_period).max()
                    roc_10_range = roc_10_max - roc_10_min
                    components['roc_10'] = ((roc_10 - roc_10_min) / roc_10_range.replace(0, 1)).clip(0, 1)
                    total_weight += momentum_weights['roc_10']
                except Exception as e:
                    logger.warning(f"Lỗi khi chuẩn hóa ROC-10: {str(e)}")
                    components['roc_10'] = pd.Series(0.5, index=df.index)
                    total_weight += momentum_weights['roc_10']
            else:
                logger.warning("Không tìm thấy cột roc_10, sử dụng giá trị mặc định")
                components['roc_10'] = pd.Series(0.5, index=df.index)
                total_weight += momentum_weights['roc_10']
            
            # 4. ROC-20
            if 'roc_20' in result_df.columns:
                try:
                    roc_20 = result_df['roc_20']
                    roc_20_min = roc_20.rolling(lookback_period).min()
                    roc_20_max = roc_20.rolling(lookback_period).max()
                    roc_20_range = roc_20_max - roc_20_min
                    components['roc_20'] = ((roc_20 - roc_20_min) / roc_20_range.replace(0, 1)).clip(0, 1)
                    total_weight += momentum_weights['roc_20']
                except Exception as e:
                    logger.warning(f"Lỗi khi chuẩn hóa ROC-20: {str(e)}")
                    components['roc_20'] = pd.Series(0.5, index=df.index)
                    total_weight += momentum_weights['roc_20']
            else:
                logger.warning("Không tìm thấy cột roc_20, sử dụng giá trị mặc định")
                components['roc_20'] = pd.Series(0.5, index=df.index)
                total_weight += momentum_weights['roc_20']
            
            # 5. MFI (nếu có)
            if has_mfi and 'mfi_14' in result_df.columns:
                try:
                    mfi = result_df['mfi_14']
                    # MFI đã trong khoảng 0-100
                    components['mfi'] = mfi / 100
                    total_weight += momentum_weights['mfi']
                except Exception as e:
                    logger.warning(f"Lỗi khi chuẩn hóa MFI: {str(e)}")
                    components['mfi'] = pd.Series(0.5, index=df.index)
                    total_weight += momentum_weights['mfi']
            else:
                logger.warning("Không tìm thấy cột mfi_14, sử dụng giá trị mặc định")
                components['mfi'] = pd.Series(0.5, index=df.index)
                total_weight += momentum_weights['mfi']
            
            # Phòng trường hợp không có thành phần nào
            if total_weight == 0:
                logger.warning("Không có thành phần momentum nào, trả về giá trị mặc định")
                return pd.Series(50, index=df.index)
            
            # Tính toán momentum component với trọng số tương ứng
            momentum_component = pd.Series(0.0, index=df.index)
            for comp_name, comp_series in components.items():
                if comp_name in momentum_weights:
                    momentum_component += (momentum_weights[comp_name] / total_weight) * comp_series * 100
            
            # Điền các giá trị NaN với giá trị trung bình
            momentum_component = momentum_component.fillna(50)
            
            return momentum_component
            
        except Exception as e:
            logger.error(f"Lỗi trong calculate_momentum_component: {str(e)}")
            # Trả về giá trị trung lập nếu có lỗi
            return pd.Series(50, index=df.index)
    
    def calculate_market_trend_component(
        self, 
        df: pd.DataFrame,
        column: str = 'close',
        short_ma: int = 10,
        long_ma: int = 50,
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Tính toán thành phần xu hướng thị trường của chỉ số.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            column: Tên cột dùng để tính xu hướng
            short_ma: Kích thước cửa sổ cho MA ngắn hạn
            long_ma: Kích thước cửa sổ cho MA dài hạn
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần xu hướng thị trường (0-100)
        """
        if not validate_price_data(df, [column]):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        # Tính các đường trung bình động
        result_df = df.copy()
        
        # MA ngắn hạn
        result_df[f'sma_{short_ma}'] = result_df[column].rolling(window=short_ma).mean()
        
        # MA dài hạn
        result_df[f'sma_{long_ma}'] = result_df[column].rolling(window=long_ma).mean()
        
        # Tính khoảng cách giữa giá và các đường MA
        result_df['price_to_short_ma'] = (result_df[column] / result_df[f'sma_{short_ma}'] - 1) * 100
        result_df['price_to_long_ma'] = (result_df[column] / result_df[f'sma_{long_ma}'] - 1) * 100
        
        # Tính khoảng cách giữa MA ngắn hạn và MA dài hạn
        result_df['short_to_long_ma'] = (result_df[f'sma_{short_ma}'] / result_df[f'sma_{long_ma}'] - 1) * 100
        
        # Chuẩn hóa các thành phần xu hướng
        # 1. Price to short MA
        p2s = result_df['price_to_short_ma']
        p2s_min = p2s.rolling(lookback_period).min()
        p2s_max = p2s.rolling(lookback_period).max()
        p2s_range = p2s_max - p2s_min
        p2s_normalized = ((p2s - p2s_min) / p2s_range).clip(0, 1)
        
        # 2. Price to long MA
        p2l = result_df['price_to_long_ma']
        p2l_min = p2l.rolling(lookback_period).min()
        p2l_max = p2l.rolling(lookback_period).max()
        p2l_range = p2l_max - p2l_min
        p2l_normalized = ((p2l - p2l_min) / p2l_range).clip(0, 1)
        
        # 3. Short MA to long MA
        s2l = result_df['short_to_long_ma']
        s2l_min = s2l.rolling(lookback_period).min()
        s2l_max = s2l.rolling(lookback_period).max()
        s2l_range = s2l_max - s2l_min
        s2l_normalized = ((s2l - s2l_min) / s2l_range).clip(0, 1)
        
        # Tính toán market trend component
        # Xu hướng tăng mạnh = extreme greed (100), xu hướng giảm mạnh = extreme fear (0)
        market_trend_weights = {
            'price_to_short_ma': 0.35,
            'price_to_long_ma': 0.35,
            'short_to_long_ma': 0.30
        }
        
        market_trend_component = 100 * (
            market_trend_weights['price_to_short_ma'] * p2s_normalized +
            market_trend_weights['price_to_long_ma'] * p2l_normalized +
            market_trend_weights['short_to_long_ma'] * s2l_normalized
        )
        
        # Điền các giá trị NaN với giá trị trung bình
        market_trend_component = market_trend_component.fillna(50)
        
        return market_trend_component
    
    def calculate_volume_component(
        self, 
        df: pd.DataFrame,
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Tính toán thành phần khối lượng giao dịch của chỉ số.
        
        Args:
            df: DataFrame chứa dữ liệu giá và khối lượng
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần khối lượng (0-100)
        """
        required_columns = ['close', 'volume']
        if not validate_price_data(df, required_columns):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
        
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        result_df = df.copy()
        
        # Tính khối lượng tương đối
        result_df['rel_volume'] = result_df['volume'] / result_df['volume'].rolling(window=20).mean()
        
        # Tính khối lượng theo xu hướng giá
        result_df['price_change'] = result_df['close'].pct_change()
        result_df['volume_trend'] = result_df['volume'] * np.sign(result_df['price_change'])
        result_df['volume_trend_ma'] = result_df['volume_trend'].rolling(window=5).mean()
        
        # Chuẩn hóa các thành phần khối lượng
        # 1. Khối lượng tương đối
        rel_vol = result_df['rel_volume']
        rel_vol_min = rel_vol.rolling(lookback_period).min()
        rel_vol_max = rel_vol.rolling(lookback_period).max()
        rel_vol_range = rel_vol_max - rel_vol_min
        rel_vol_normalized = ((rel_vol - rel_vol_min) / rel_vol_range).clip(0, 1)
        
        # 2. Xu hướng khối lượng
        vol_trend = result_df['volume_trend_ma']
        vol_trend_min = vol_trend.rolling(lookback_period).min()
        vol_trend_max = vol_trend.rolling(lookback_period).max()
        vol_trend_range = vol_trend_max - vol_trend_min
        vol_trend_normalized = ((vol_trend - vol_trend_min) / vol_trend_range).clip(0, 1)
        
        # Tính toán volume component
        # Khối lượng cao theo xu hướng tăng = extreme greed (100)
        # Khối lượng cao theo xu hướng giảm = extreme fear (0)
        volume_component = 100 * (0.5 * rel_vol_normalized + 0.5 * vol_trend_normalized)
        
        # Điền các giá trị NaN với giá trị trung bình
        volume_component = volume_component.fillna(50)
        
        return volume_component
    
    def calculate_put_call_ratio(
        self, 
        df: pd.DataFrame,
        column: str = 'close',
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Mô phỏng tỉ lệ put/call dựa trên hành động giá.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            column: Tên cột dùng để tính
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần put/call ratio (0-100)
        """
        if not validate_price_data(df, [column]):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        if lookback_period is None:
            lookback_period = self.lookback_period
            
        result_df = df.copy()
        
        # Tính biến động ngắn hạn
        result_df['short_volatility'] = result_df[column].pct_change().rolling(window=5).std() * 100
        
        # Tính RSI
        result_df = relative_strength_index(result_df, column=column, window=14)
        
        # Tính hướng giá ngắn hạn
        result_df['price_direction'] = np.sign(result_df[column].pct_change(periods=3))
        
        # Tính put/call ratio mô phỏng
        # Khi RSI thấp và biến động cao => put/call ratio cao (bán > mua)
        # Khi RSI cao và biến động thấp => put/call ratio thấp (mua > bán)
        result_df['pseudo_put_call'] = (100 - result_df['rsi_14']) * result_df['short_volatility'] / 100
        
        # Chuẩn hóa put/call ratio
        pc_ratio = result_df['pseudo_put_call']
        pc_ratio_min = pc_ratio.rolling(lookback_period).min()
        pc_ratio_max = pc_ratio.rolling(lookback_period).max()
        pc_ratio_range = pc_ratio_max - pc_ratio_min
        pc_ratio_normalized = ((pc_ratio - pc_ratio_min) / pc_ratio_range).clip(0, 1)
        
        # Chuyển đổi: put/call ratio cao = extreme fear (0), put/call ratio thấp = extreme greed (100)
        put_call_component = 100 * (1 - pc_ratio_normalized)
        
        # Điền các giá trị NaN với giá trị trung bình
        put_call_component = put_call_component.fillna(50)
        
        return put_call_component
    
    def calculate_price_strength(
        self, 
        df: pd.DataFrame,
        column: str = 'close',
        price_levels: Optional[List[int]] = None,
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Tính toán sức mạnh giá dựa trên mức giá tâm lý quan trọng.
        
        Args:
            df: DataFrame chứa dữ liệu giá
            column: Tên cột dùng để tính
            price_levels: Danh sách số ngày để tính các mức giá quan trọng
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            Series chứa giá trị thành phần sức mạnh giá (0-100)
        """
        if not validate_price_data(df, [column]):
            raise ValueError(f"Dữ liệu không hợp lệ: thiếu cột {column}")
        
        if price_levels is None:
            price_levels = [10, 20, 50, 100, 200]
            
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        result_df = df.copy()
        
        # Tính các mức giá quan trọng
        for level in price_levels:
            # Giá cao nhất trong n ngày
            result_df[f'high_{level}d'] = result_df[column].rolling(window=level).max()
            
            # Giá thấp nhất trong n ngày
            result_df[f'low_{level}d'] = result_df[column].rolling(window=level).min()
            
            # Khoảng cách từ giá hiện tại đến các mức cao/thấp
            result_df[f'dist_to_high_{level}d'] = (result_df[column] / result_df[f'high_{level}d'] - 1) * 100
            result_df[f'dist_to_low_{level}d'] = (result_df[column] / result_df[f'low_{level}d'] - 1) * 100
        
        # Tính vị trí tương đối trong khoảng giá
        for level in price_levels:
            high_col = f'high_{level}d'
            low_col = f'low_{level}d'
            
            # Vị trí tương đối: 0 = thấp nhất, 1 = cao nhất
            range_col = f'price_range_{level}d'
            position_col = f'price_position_{level}d'
            
            result_df[range_col] = result_df[high_col] - result_df[low_col]
            result_df[position_col] = (result_df[column] - result_df[low_col]) / result_df[range_col]
        
        # Tính sức mạnh giá dựa trên vị trí trong các khoảng giá
        position_cols = [f'price_position_{level}d' for level in price_levels]
        
        # Gán trọng số cho các khoảng giá khác nhau
        weights = {
            'price_position_10d': 0.05,   # Ngắn hạn
            'price_position_20d': 0.15, 
            'price_position_50d': 0.25,   # Trung hạn
            'price_position_100d': 0.25,
            'price_position_200d': 0.30   # Dài hạn
        }
        
        # Tính toán chỉ số sức mạnh giá
        result_df['price_strength'] = 0
        
        for col, weight in weights.items():
            if col in result_df.columns:
                result_df['price_strength'] += result_df[col] * weight
        
        # Chuyển đổi thành thang đo 0-100
        price_strength_component = result_df['price_strength'] * 100
        
        # Chuẩn hóa trong khoảng nhìn lại
        ps_min = price_strength_component.rolling(lookback_period).min()
        ps_max = price_strength_component.rolling(lookback_period).max()
        ps_range = ps_max - ps_min
        price_strength_component = ((price_strength_component - ps_min) / ps_range * 100).clip(0, 100)
        
        # Điền các giá trị NaN với giá trị trung bình
        price_strength_component = price_strength_component.fillna(50)
        
        return price_strength_component
    
    def calculate_fear_greed_index(
        self, 
        df: pd.DataFrame,
        column: str = 'close',
        lookback_period: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Tính toán chỉ số Fear & Greed tổng hợp.
        
        Args:
            df: DataFrame chứa dữ liệu giá và khối lượng
            column: Tên cột dùng để tính
            lookback_period: Số ngày nhìn lại để chuẩn hóa
            
        Returns:
            DataFrame với các thành phần và chỉ số Fear & Greed tổng hợp
        """
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        result_df = df.copy()
        
        # Tính các thành phần với xử lý lỗi
        components = {}
        
        try:
            # 1. RSI Component
            try:
                components['fear_greed_rsi'] = self.calculate_rsi_component(result_df, column)
            except Exception as e:
                logger.warning(f"Lỗi khi tính RSI component: {str(e)}")
                components['fear_greed_rsi'] = pd.Series(50, index=df.index)  # Giá trị trung lập
            
            # 2. Volatility Component
            try:
                components['fear_greed_volatility'] = self.calculate_volatility_component(result_df, lookback_period)
            except Exception as e:
                logger.warning(f"Lỗi khi tính Volatility component: {str(e)}")
                components['fear_greed_volatility'] = pd.Series(50, index=df.index)
            
            # 3. Momentum Component
            try:
                components['fear_greed_momentum'] = self.calculate_momentum_component(result_df, column, lookback_period)
            except Exception as e:
                logger.warning(f"Lỗi khi tính Momentum component: {str(e)}")
                components['fear_greed_momentum'] = pd.Series(50, index=df.index)
            
            # 4. Market Trend Component
            try:
                components['fear_greed_market_trend'] = self.calculate_market_trend_component(result_df, column, lookback_period=lookback_period)
            except Exception as e:
                logger.warning(f"Lỗi khi tính Market Trend component: {str(e)}")
                components['fear_greed_market_trend'] = pd.Series(50, index=df.index)
            
            # 5. Volume Component (nếu có dữ liệu khối lượng)
            try:
                if 'volume' in result_df.columns:
                    components['fear_greed_volume'] = self.calculate_volume_component(result_df, lookback_period)
                else:
                    components['fear_greed_volume'] = pd.Series(50, index=df.index)  # Giá trị trung bình nếu không có dữ liệu khối lượng
            except Exception as e:
                logger.warning(f"Lỗi khi tính Volume component: {str(e)}")
                components['fear_greed_volume'] = pd.Series(50, index=df.index)
            
            # 6. Put/Call Ratio Component
            try:
                components['fear_greed_put_call'] = self.calculate_put_call_ratio(result_df, column, lookback_period)
            except Exception as e:
                logger.warning(f"Lỗi khi tính Put/Call Ratio component: {str(e)}")
                components['fear_greed_put_call'] = pd.Series(50, index=df.index)
            
            # 7. Price Strength Component
            try:
                components['fear_greed_price_strength'] = self.calculate_price_strength(result_df, column, lookback_period=lookback_period)
            except Exception as e:
                logger.warning(f"Lỗi khi tính Price Strength component: {str(e)}")
                components['fear_greed_price_strength'] = pd.Series(50, index=df.index)
            
            # Thêm các thành phần vào DataFrame kết quả
            for name, series in components.items():
                result_df[name] = series
            
            # Tính chỉ số Fear & Greed tổng hợp
            fear_greed_index = pd.Series(0.0, index=df.index)
            
            for component_name, weight in self.component_weights.items():
                component_col = f'fear_greed_{component_name}'
                if component_col in result_df.columns:
                    fear_greed_index += weight * result_df[component_col]
            
            result_df['fear_greed_index'] = fear_greed_index
            
            # Thêm phân loại Fear & Greed
            result_df['fear_greed_classification'] = result_df['fear_greed_index'].apply(self.get_fear_greed_label)
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số Fear & Greed: {str(e)}")
            raise
        
        return result_df
    
    def get_fear_greed_label(self, value: float) -> str:
        """
        Phân loại giá trị Fear & Greed.
        
        Args:
            value: Giá trị Fear & Greed (0-100)
            
        Returns:
            Nhãn phân loại
        """
        for (low, high), label in self.fear_greed_classification.items():
            if low <= value < high:
                return label
        
        # Giá trị mặc định
        return "Neutral"


# Các hàm tiện ích (được di chuyển ra khỏi lớp FearGreedIndex)
def calculate_market_sentiment_features(df: pd.DataFrame, column: str = 'close', lookback_period: int = 90, prefix: str = 'sentiment_') -> pd.DataFrame:
    """
    Tính toán đặc trưng tâm lý thị trường từ dữ liệu giá.
    
    Args:
        df: DataFrame với dữ liệu giá
        column: Tên cột giá sử dụng làm cơ sở
        lookback_period: Số ngày nhìn lại để tính chỉ số
        prefix: Tiền tố cho tên cột mới
        
    Returns:
        DataFrame với các đặc trưng tâm lý thị trường
    """
    try:
        # Kiểm tra DataFrame
        if df is None or df.empty:
            logger.error("DataFrame trống hoặc None")
            return df
        
        # Đảm bảo cột giá tồn tại
        if column not in df.columns:
            logger.error(f"Không tìm thấy cột {column} trong DataFrame")
            return df
        
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        result_df = df.copy()
        
        # Tự tính các chỉ báo kỹ thuật cần thiết trước
        logger.info("Tự tính toán các chỉ báo kỹ thuật cần thiết...")
        
        # 1. Tính SMA và Bollinger Bandwidth
        if 'bbw_20' not in result_df.columns:
            logger.info("Tự tính toán Bollinger Bandwidth...")
            # Tính SMA20
            result_df['sma_20'] = result_df[column].rolling(window=20).mean()
            # Tính độ lệch chuẩn
            result_df['stddev_20'] = result_df[column].rolling(window=20).std()
            # Tính Bollinger Bands
            result_df['upper_band'] = result_df['sma_20'] + (result_df['stddev_20'] * 2)
            result_df['lower_band'] = result_df['sma_20'] - (result_df['stddev_20'] * 2)
            # Tính Bollinger Bandwidth
            result_df['bbw_20'] = (result_df['upper_band'] - result_df['lower_band']) / result_df['sma_20']
        
        # 2. Tính Rate of Change (ROC) cho các khung thời gian
        for window in [1, 5, 10, 20]:
            roc_col = f'roc_{window}'
            if roc_col not in result_df.columns:
                logger.info(f"Tự tính toán {roc_col}...")
                result_df[roc_col] = result_df[column].pct_change(periods=window) * 100
        
        # Tạo instance của lớp FearGreedIndex
        fear_greed = FearGreedIndex(lookback_period=lookback_period)
        
        # Tính các thành phần tâm lý sử dụng các phương thức của lớp FearGreedIndex
        volatility_sentiment = fear_greed.calculate_volatility_component(result_df, lookback_period)
        momentum_sentiment = fear_greed.calculate_momentum_component(result_df, column, lookback_period)
        trend_sentiment = fear_greed.calculate_market_trend_component(result_df, column, lookback_period=lookback_period)
        
        # Kiểm tra cột volume trước khi tính
        if 'volume' in result_df.columns:
            volume_sentiment = fear_greed.calculate_volume_component(result_df, lookback_period)
        else:
            logger.warning("Không tìm thấy cột 'volume', sử dụng giá trị mặc định 50 cho thành phần khối lượng")
            volume_sentiment = pd.Series(50, index=result_df.index)
        
        # Thêm vào DataFrame
        result_df[f'{prefix}volatility'] = volatility_sentiment
        result_df[f'{prefix}momentum'] = momentum_sentiment
        result_df[f'{prefix}trend'] = trend_sentiment
        result_df[f'{prefix}volume'] = volume_sentiment
        
        # Tính chỉ số Fear & Greed tổng hợp
        # Trọng số: Biến động (30%), Động lượng (25%), Xu hướng (25%), Khối lượng (20%)
        result_df[f'{prefix}fear_greed_index'] = (
            0.3 * volatility_sentiment +
            0.25 * momentum_sentiment +
            0.25 * trend_sentiment +
            0.2 * volume_sentiment
        )
        
        # Thêm phân loại
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
        
        result_df[f'{prefix}fear_greed_label'] = pd.cut(
            result_df[f'{prefix}fear_greed_index'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Tính đạo hàm (thay đổi) của chỉ số
        result_df[f'{prefix}fear_greed_change'] = result_df[f'{prefix}fear_greed_index'].diff()
        
        # Tính MA của chỉ số Fear & Greed
        result_df[f'{prefix}fear_greed_ma7'] = result_df[f'{prefix}fear_greed_index'].rolling(window=7).mean()
        result_df[f'{prefix}fear_greed_ma14'] = result_df[f'{prefix}fear_greed_index'].rolling(window=14).mean()
        
        # Tính chênh lệch giữa chỉ số hiện tại và MA
        result_df[f'{prefix}fear_greed_ma7_diff'] = result_df[f'{prefix}fear_greed_index'] - result_df[f'{prefix}fear_greed_ma7']
        result_df[f'{prefix}fear_greed_ma14_diff'] = result_df[f'{prefix}fear_greed_index'] - result_df[f'{prefix}fear_greed_ma14']
        
        logger.info("Đã tạo thành công đặc trưng tâm lý thị trường từ hành động giá")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi tính toán đặc trưng tâm lý thị trường: {str(e)}")
        return df  # Trả về DataFrame gốc nếu có lỗi

def generate_fear_greed_history(
    df: pd.DataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    column: str = 'close',
    resample: bool = True,
    resample_rule: str = 'D'
) -> pd.DataFrame:
    """
    Tạo lịch sử chỉ số Fear & Greed từ dữ liệu giá.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        start_date: Ngày bắt đầu (None = toàn bộ dữ liệu)
        end_date: Ngày kết thúc (None = toàn bộ dữ liệu)
        column: Tên cột dùng để tính
        resample: Có lấy mẫu lại theo thời gian không
        resample_rule: Quy tắc lấy mẫu lại ('D' = ngày, 'W' = tuần, 'M' = tháng)
        
    Returns:
        DataFrame chứa lịch sử chỉ số Fear & Greed
    """
    # Kiểm tra dữ liệu đầu vào
    required_columns = [column, 'timestamp']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Dữ liệu không hợp lệ: thiếu các cột {required_columns}")
        return pd.DataFrame()
    
    # Tạo bản sao của DataFrame
    data = df.copy()
    
    # Đảm bảo timestamp là datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Lọc dữ liệu theo ngày
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        data = data[data['timestamp'] >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        data = data[data['timestamp'] <= end_date]
    
    # Tính toán chỉ số Fear & Greed
    fear_greed = FearGreedIndex(lookback_period=min(90, len(data)))
    result_df = fear_greed.calculate_fear_greed_index(data, column)
    
    # Lấy mẫu lại theo thời gian nếu cần
    if resample and len(result_df) > 0:
        # Đặt timestamp làm index
        result_df = result_df.set_index('timestamp')
        
        # Lấy mẫu lại
        result_df = result_df.resample(resample_rule).last()
        
        # Reset index
        result_df = result_df.reset_index()
    
    # Chọn các cột liên quan đến Fear & Greed
    fear_greed_cols = ['timestamp'] + [col for col in result_df.columns if col.startswith('fear_greed_')]
    result_df = result_df[fear_greed_cols]
    
    return result_df


def export_fear_greed_to_csv(
    df: pd.DataFrame,
    output_path: str,
    include_components: bool = True
) -> bool:
    """
    Xuất dữ liệu chỉ số Fear & Greed ra file CSV.
    
    Args:
        df: DataFrame chứa chỉ số Fear & Greed
        output_path: Đường dẫn file CSV đầu ra
        include_components: Có đưa các thành phần vào không
        
    Returns:
        True nếu xuất thành công, False nếu có lỗi
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        required_columns = ['timestamp', 'fear_greed_index', 'fear_greed_classification']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Dữ liệu không hợp lệ: thiếu các cột {missing_cols}")
            return False
        
        # Tạo bản sao của DataFrame
        export_df = df.copy()
        
        # Chọn các cột cần xuất
        if include_components:
            export_cols = ['timestamp', 'fear_greed_index', 'fear_greed_classification'] + \
                        [col for col in df.columns if col.startswith('fear_greed_') and 
                        col not in ['fear_greed_index', 'fear_greed_classification']]
        else:
            export_cols = ['timestamp', 'fear_greed_index', 'fear_greed_classification']
        
        # Chỉ giữ lại các cột tồn tại trong DataFrame
        export_cols = [col for col in export_cols if col in export_df.columns]
        
        # Xuất ra CSV
        export_df[export_cols].to_csv(output_path, index=False)
        
        logger.info(f"Đã xuất chỉ số Fear & Greed thành công vào {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi xuất chỉ số Fear & Greed ra CSV: {str(e)}")
        return False


def plot_fear_greed_chart(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    chart_title: str = "Crypto Fear & Greed Index",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[Any]:
    """
    Vẽ biểu đồ chỉ số Fear & Greed.
    
    Args:
        df: DataFrame chứa chỉ số Fear & Greed
        output_path: Đường dẫn file ảnh đầu ra (None = không lưu)
        chart_title: Tiêu đề biểu đồ
        figsize: Kích thước biểu đồ (width, height)
        
    Returns:
        Đối tượng biểu đồ hoặc None nếu có lỗi
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        required_columns = ['timestamp', 'fear_greed_index', 'fear_greed_classification']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Dữ liệu không hợp lệ: thiếu các cột {missing_cols}")
            return None
        
        # Import matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.colors as mcolors
        except ImportError:
            logger.error("Không thể import matplotlib. Hãy cài đặt matplotlib để vẽ biểu đồ.")
            return None
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=figsize)
        
        # Dữ liệu
        x = df['timestamp']
        y = df['fear_greed_index']
        
        # Tạo colormap
        colors = [(0.8, 0.0, 0.0), (1.0, 0.5, 0.0), (0.9, 0.9, 0.2), (0.2, 0.8, 0.2), (0.0, 0.6, 0.0)]
        cmap = mcolors.LinearSegmentedColormap.from_list('fear_greed', colors, N=100)
        
        # Vẽ biểu đồ cột
        bars = ax.bar(x, y, width=0.8, color=cmap(y/100))
        
        # Tùy chỉnh trục X
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        # Tùy chỉnh trục Y
        ax.set_ylim(0, 100)
        ax.set_yticks([12.5, 35, 50, 65, 87.5])
        ax.set_yticklabels(['Extreme\nFear', 'Fear', 'Neutral', 'Greed', 'Extreme\nGreed'])
        
        # Thêm đường grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tiêu đề và nhãn
        ax.set_title(chart_title, fontsize=16, pad=20)
        ax.set_ylabel('Fear & Greed Index', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Thêm giá trị vào thanh cuối cùng
        last_bar = bars[-1]
        ax.text(last_bar.get_x() + last_bar.get_width()/2, last_bar.get_height() + 5,
                f"{y.iloc[-1]:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Thêm chú thích
        last_class = df['fear_greed_classification'].iloc[-1]
        last_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        ax.text(0.02, 0.02, f"Last updated: {last_date}\nCurrent: {last_class} ({y.iloc[-1]:.0f})", 
                transform=ax.transAxes, fontsize=12, va='bottom')
        
        # Làm chặt layout
        plt.tight_layout()
        
        # Lưu biểu đồ nếu có đường dẫn
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ vào {output_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ Fear & Greed: {str(e)}")
        return None


if __name__ == "__main__":
    # Mã thực thi khi file được chạy trực tiếp
    import argparse
    import pandas as pd
    import os
    from pathlib import Path
    
    # Tạo parser cho các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Tạo và tính toán chỉ số Fear & Greed cho dữ liệu giá crypto.")
    parser.add_argument("--input", "-i", type=str, help="Đường dẫn file dữ liệu đầu vào (CSV, Parquet). Nếu không chỉ định, sẽ tìm trong thư mục data/processed")
    parser.add_argument("--output", "-o", type=str, help="Đường dẫn thư mục đầu ra. Mặc định là data/sentiment")
    parser.add_argument("--symbol", "-s", type=str, help="Symbol cần tính toán (nếu không chỉ định, sẽ xử lý tất cả các file)")
    parser.add_argument("--plot", "-p", action="store_true", help="Tạo biểu đồ Fear & Greed")
    parser.add_argument("--all", "-a", action="store_true", help="Xử lý tất cả các file trong thư mục data/processed")
    
    args = parser.parse_args()
    
    # Xác định thư mục dữ liệu và đầu ra
    base_dir = Path(__file__).parents[3]  # Thư mục gốc của dự án
    
    # Thư mục dữ liệu đầu ra
    if args.output:
        output_dir = Path(args.output)
    else:
        # Sử dụng đường dẫn chỉ định
        output_dir = Path("E:/AI_AGENT/automated-trading-system/data/sentiment")
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh sách các file cần xử lý
    input_files = []
    
    if args.input:
        # Nếu chỉ định file đầu vào cụ thể
        input_files = [Path(args.input)]
    elif args.symbol:
        # Nếu chỉ định symbol cụ thể
        symbol_lower = args.symbol.lower().replace("/", "_")
        processed_dir = base_dir / "data" / "processed"
        pattern = f"{symbol_lower}_*.parquet"
        
        # Tìm các file khớp với pattern
        matching_files = list(processed_dir.glob(pattern))
        
        if not matching_files:
            print(f"Không tìm thấy file dữ liệu nào khớp với {pattern} trong {processed_dir}")
            exit(1)
        
        # Sắp xếp theo thời gian sửa đổi (mới nhất cuối cùng)
        matching_files.sort(key=lambda x: os.path.getmtime(x))
        input_files = [matching_files[-1]]
    else:
        # Xử lý tất cả các file parquet trong thư mục
        processed_dir = base_dir / "data" / "processed"
        input_files = list(processed_dir.glob("*.parquet"))
        
        if not input_files:
            print(f"Không tìm thấy file dữ liệu nào trong {processed_dir}")
            exit(1)
    
    print(f"Tìm thấy {len(input_files)} file để xử lý")
    
    # Xử lý từng file dữ liệu
    for input_file in input_files:
        try:
            print(f"\nĐang xử lý file: {input_file}")
            # Đọc dữ liệu
            if str(input_file).endswith('.csv'):
                df = pd.read_csv(input_file)
            else:  # Giả sử định dạng Parquet
                df = pd.read_parquet(input_file)
            
            # Phân tích tên file để xác định symbol
            file_name = input_file.stem
            symbol = file_name.split('_')[0]
            
            # Kiểm tra dữ liệu đầu vào
            print(f"Số lượng dòng: {len(df)}")
            print(f"Các cột: {df.columns.tolist()}")
            
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Thiếu các cột cần thiết: {missing_columns}")
                
                # Thử dự đoán tên cột
                for col in missing_columns:
                    if col == 'open' and 'Open' in df.columns:
                        df['open'] = df['Open']
                    elif col == 'high' and 'High' in df.columns:
                        df['high'] = df['High']
                    elif col == 'low' and 'Low' in df.columns:
                        df['low'] = df['Low']
                    elif col == 'close' and 'Close' in df.columns:
                        df['close'] = df['Close']
                    else:
                        # Tạo cột giả
                        print(f"Tạo cột giả cho {col}")
                        df[col] = df['close'] if 'close' in df.columns else np.ones(len(df))    
            
            # Kiểm tra xem df có cột timestamp không, nếu không thì tạo
            if 'timestamp' not in df.columns:
                print("Warning: Không tìm thấy cột 'timestamp', sẽ tạo dựa trên index")
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df))
            
            # Tính toán chỉ số Fear & Greed
            print(f"Tính toán chỉ số Fear & Greed cho {symbol}...")
            
            result_df = calculate_market_sentiment_features(
                df,
                column='close',
                lookback_period=90,
                prefix='sentiment_'
            )
            
            # Xác định tên file đầu ra
            output_file = output_dir / f"{symbol}_fear_greed.parquet"
            
            # Lưu kết quả
            result_df.to_parquet(output_file)
            print(f"Đã lưu kết quả vào: {output_file}")
            
            # Lưu file CSV cho dễ xem
            csv_output = output_dir / f"{symbol}_fear_greed.csv"
            result_df.to_csv(csv_output)
            print(f"Đã lưu kết quả vào: {csv_output}")
            
            # Tạo biểu đồ nếu được yêu cầu
            if args.plot:
                plot_path = output_dir / f"{symbol}_fear_greed.png"
                
                # Chuẩn bị dữ liệu cho biểu đồ
                plot_df = result_df.copy()
                sentiment_cols = [col for col in plot_df.columns if col.startswith('sentiment_')]
                
                # Đổi tên cột
                rename_dict = {}
                for col in sentiment_cols:
                    if col == 'sentiment_fear_greed_index':
                        rename_dict[col] = 'fear_greed_index'
                    elif col == 'sentiment_fear_greed_label':
                        rename_dict[col] = 'fear_greed_classification'
                
                if rename_dict:
                    plot_df = plot_df.rename(columns=rename_dict)
                    # Vẽ biểu đồ
                    try:
                        plot_fear_greed_chart(plot_df, str(plot_path))
                        print(f"Đã lưu biểu đồ vào: {plot_path}")
                    except Exception as e:
                        print(f"Lỗi khi tạo biểu đồ: {str(e)}")
        
        except Exception as e:
            print(f"Lỗi khi xử lý file {input_file}: {str(e)}")