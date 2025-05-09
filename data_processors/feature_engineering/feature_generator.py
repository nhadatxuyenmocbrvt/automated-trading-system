"""
Tạo đặc trưng cho mô hình.
File này cung cấp lớp FeatureGenerator để tạo, quản lý và áp dụng các đặc trưng
từ các module con, hỗ trợ tiền xử lý, tạo pipeline và biến đổi dữ liệu.
"""

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import warnings
import concurrent.futures
from functools import partial

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logger
from config.constants import Indicator, DEFAULT_INDICATOR_PARAMS

# Import các module feature engineering
from data_processors.feature_engineering.technical_indicators.trend_indicators import (
    simple_moving_average, exponential_moving_average, bollinger_bands,
    moving_average_convergence_divergence, average_directional_index,
    parabolic_sar, ichimoku_cloud, supertrend
)
from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
    relative_strength_index, stochastic_oscillator, williams_r,
    commodity_channel_index, rate_of_change, money_flow_index,
    true_strength_index
)
from data_processors.feature_engineering.technical_indicators.volume_indicators import (
    on_balance_volume, accumulation_distribution_line, chaikin_money_flow,
    volume_weighted_average_price, ease_of_movement, volume_oscillator,
    price_volume_trend, log_transform_volume
)
from data_processors.feature_engineering.market_features.price_features import (
    calculate_returns, calculate_log_returns, calculate_rsi_features,
    calculate_price_momentum, calculate_price_ratios, calculate_price_channels,
    calculate_price_crossovers, calculate_price_divergence
)
from data_processors.feature_engineering.market_features.volatility_features import (
    calculate_volatility_features, calculate_relative_volatility,
    calculate_volatility_ratio, calculate_volatility_patterns,
    calculate_volatility_regime, calculate_garch_features
)

# Import các module tiện ích
from data_processors.utils.preprocessing import (
    normalize_features, standardize_features, min_max_scale,
    handle_extreme_values, log_transform, winsorize,
    clean_technical_features, normalize_technical_indicators,
    standardize_technical_indicators, detect_and_fix_indicator_outliers
)

class FeatureGenerator:
    """
    Lớp chính để tạo, quản lý và áp dụng các đặc trưng từ các module con.
    Hỗ trợ tiền xử lý dữ liệu, xây dựng pipeline và biến đổi dữ liệu.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_workers: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo FeatureGenerator.
        
        Args:
            data_dir: Thư mục dữ liệu để lưu/tải các pipeline đặc trưng
            max_workers: Số luồng tối đa cho việc xử lý song song
            logger: Logger hiện có (nếu có)
        """
        # Thiết lập logger
        if logger is None:
            self.logger = setup_logger("feature_generator")
        else:
            self.logger = logger
        
        # Thiết lập thư mục dữ liệu
        if data_dir is None:
            # Sử dụng thư mục hiện tại
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "feature_pipelines"
        else:
            self.data_dir = data_dir
        
        # Tạo thư mục nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập số luồng tối đa
        self.max_workers = max_workers
        
        # Khởi tạo các thuộc tính quản lý đặc trưng
        self.registered_features = {}
        self.feature_pipelines = {}
        self.fitted_params = {}
        
        # Danh sách đặc trưng mặc định
        self.default_features = {
            'trend': [
                'sma', 'ema', 'bollinger_bands', 'macd', 'adx',
                'parabolic_sar', 'supertrend'
            ],
            'momentum': [
                'rsi', 'stochastic_oscillator', 'williams_r', 'cci',
                'roc', 'mfi', 'tsi'
            ],
            'volume': [
                'obv', 'ad_line', 'cmf', 'vwap', 'eom', 'volume_oscillator',
                'pvt', 'volume_log'
            ],
            'volatility': [
                'atr', 'volatility_features', 'volatility_ratio',
                'volatility_patterns', 'volatility_regime'
            ],
            'price': [
                'returns', 'log_returns', 'price_momentum', 'price_ratios',
                'price_channels', 'price_crossovers'
            ]
        }
        
        # Đăng ký các hàm xử lý đặc trưng
        self.preprocessors = {
            'normalize': normalize_features,
            'standardize': standardize_features,
            'minmax': min_max_scale,
            'handle_extremes': handle_extreme_values,
            'log_transform': log_transform,
            'winsorize': winsorize,
            'clean_technical': clean_technical_features,
            'normalize_indicators': normalize_technical_indicators,
            'standardize_indicators': standardize_technical_indicators,
            'fix_outliers': detect_and_fix_indicator_outliers
        }
        
        # Đăng ký các thành phần chuyển đổi
        self.transformers = {
            'normalize_technical': normalize_technical_indicators,
            'standardize_technical': standardize_technical_indicators
        }
        
        # Đăng ký các bộ chọn lọc đặc trưng
        self.feature_selectors = {
            'correlation': self._correlation_selector,
            'variance': self._variance_selector,
            'statistical_correlation': self._statistical_correlation_selector
        }
        
        self.logger.info("Đã khởi tạo FeatureGenerator")
    
    def register_feature(
        self,
        feature_name: str,
        feature_func: Callable,
        feature_params: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> None:
        """
        Đăng ký một đặc trưng mới.
        
        Args:
            feature_name: Tên đặc trưng
            feature_func: Hàm tính toán đặc trưng
            feature_params: Tham số cho hàm đặc trưng
            category: Danh mục đặc trưng (trend, momentum, volume, volatility, price)
        """
        if feature_params is None:
            feature_params = {}
        
        if category is None:
            # Tự động phân loại đặc trưng
            if any(keyword in feature_name for keyword in ["sma", "ema", "macd", "bollinger", "supertrend"]):
                category = "trend"
            elif any(keyword in feature_name for keyword in ["rsi", "stoch", "cci", "roc", "mfi"]):
                category = "momentum"
            elif any(keyword in feature_name for keyword in ["obv", "volume", "vwap", "ad_line"]):
                category = "volume"
            elif any(keyword in feature_name for keyword in ["atr", "volatility"]):
                category = "volatility"
            elif any(keyword in feature_name for keyword in ["returns", "price"]):
                category = "price"
            else:
                category = "other"
        
        self.registered_features[feature_name] = {
            "func": feature_func,
            "params": feature_params,
            "category": category
        }
        
        self.logger.debug(f"Đã đăng ký đặc trưng '{feature_name}' trong danh mục '{category}'")
    
    def register_default_features(
        self,
        all_indicators: bool = False,
        categories: Optional[List[str]] = None
    ) -> None:
        """
        Đăng ký các đặc trưng mặc định.
        
        Args:
            all_indicators: Đăng ký tất cả các chỉ báo hay không
            categories: Danh sách các danh mục đặc trưng cần đăng ký
        """
        if categories is None:
            # Nếu không chỉ định danh mục, đăng ký tất cả hoặc nhóm mặc định
            if all_indicators:
                categories = list(self.default_features.keys())
            else:
                # Một tập con chính của các đặc trưng
                categories = ["trend", "momentum", "volume"]
        
        # Đăng ký các đặc trưng xu hướng
        if "trend" in categories:
            # SMA
            for window in [5, 10, 20, 50, 100, 200]:
                self.register_feature(
                    f"sma_{window}",
                    simple_moving_average,
                    {"window": window},
                    "trend"
                )
            
            # EMA
            for window in [5, 10, 20, 50, 100, 200]:
                self.register_feature(
                    f"ema_{window}",
                    exponential_moving_average,
                    {"window": window},
                    "trend"
                )
            
            # Bollinger Bands
            for window in [20, 50]:
                for std_dev in [2.0, 3.0]:
                    self.register_feature(
                        f"bollinger_bands_{window}_{int(std_dev)}",
                        bollinger_bands,
                        {"window": window, "std_dev": std_dev},
                        "trend"
                    )
            
            # MACD
            self.register_feature(
                "macd",
                moving_average_convergence_divergence,
                {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "trend"
            )
            
            # ADX
            for window in [14, 21]:
                self.register_feature(
                    f"adx_{window}",
                    average_directional_index,
                    {"window": window},
                    "trend"
                )
            
            # Parabolic SAR
            self.register_feature(
                "parabolic_sar",
                parabolic_sar,
                {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2},
                "trend"
            )
            
            # Supertrend
            for period in [10, 20]:
                for multiplier in [2.0, 3.0]:
                    self.register_feature(
                        f"supertrend_{period}_{multiplier}",
                        supertrend,
                        {"period": period, "multiplier": multiplier},
                        "trend"
                    )
        
        # Đăng ký các đặc trưng động lượng
        if "momentum" in categories:
            # RSI
            for window in [7, 14, 21]:
                self.register_feature(
                    f"rsi_{window}",
                    relative_strength_index,
                    {"window": window},
                    "momentum"
                )
            
            # Stochastic Oscillator
            for k_period in [14, 21]:
                for d_period in [3, 5]:
                    self.register_feature(
                        f"stochastic_{k_period}_{d_period}",
                        stochastic_oscillator,
                        {"k_period": k_period, "d_period": d_period},
                        "momentum"
                    )
            
            # Williams %R
            for window in [14, 21]:
                self.register_feature(
                    f"williams_r_{window}",
                    williams_r,
                    {"window": window},
                    "momentum"
                )
            
            # CCI
            for window in [20, 40]:
                self.register_feature(
                    f"cci_{window}",
                    commodity_channel_index,
                    {"window": window},
                    "momentum"
                )
            
            # Rate of Change
            for window in [9, 14, 21]:
                self.register_feature(
                    f"roc_{window}",
                    rate_of_change,
                    {"window": window, "percentage": True},
                    "momentum"
                )
            
            # Money Flow Index
            for window in [14, 21]:
                self.register_feature(
                    f"mfi_{window}",
                    money_flow_index,
                    {"window": window},
                    "momentum"
                )
            
            # True Strength Index
            self.register_feature(
                "tsi",
                true_strength_index,
                {"long_window": 25, "short_window": 13, "signal_window": 7},
                "momentum"
            )
        
        # Đăng ký các đặc trưng khối lượng
        if "volume" in categories:
            # On-Balance Volume
            self.register_feature(
                "obv",
                on_balance_volume,
                {},
                "volume"
            )
            
            # Accumulation/Distribution Line
            self.register_feature(
                "ad_line",
                accumulation_distribution_line,
                {},
                "volume"
            )
            
            # Chaikin Money Flow
            for window in [20, 50]:
                self.register_feature(
                    f"cmf_{window}",
                    chaikin_money_flow,
                    {"window": window},
                    "volume"
                )
            
            # Volume-Weighted Average Price
            for window in [14, 30]:
                self.register_feature(
                    f"vwap_{window}",
                    volume_weighted_average_price,
                    {"window": window},
                    "volume"
                )
            
            # Ease of Movement
            for window in [14, 21]:
                self.register_feature(
                    f"eom_{window}",
                    ease_of_movement,
                    {"window": window},
                    "volume"
                )
            
            # Volume Oscillator
            for short_window, long_window in [(5, 10), (10, 20)]:
                self.register_feature(
                    f"volume_oscillator_{short_window}_{long_window}",
                    volume_oscillator,
                    {"short_window": short_window, "long_window": long_window},
                    "volume"
                )
            
            # Price Volume Trend
            self.register_feature(
                "pvt",
                price_volume_trend,
                {},
                "volume"
            )
            
            # Log Transform Volume
            self.register_feature(
                "volume_log",
                log_transform_volume,
                {},
                "volume"
            )
        
        # Đăng ký các đặc trưng biến động
        if "volatility" in categories:
            # Biến động cơ bản
            for window in [5, 10, 20, 50, 100]:
                self.register_feature(
                    f"volatility_{window}",
                    calculate_volatility_features,
                    {"windows": [window]},
                    "volatility"
                )
            
            # Biến động tương đối
            for window in [5, 10, 14, 20]:
                self.register_feature(
                    f"relative_volatility_{window}",
                    calculate_relative_volatility,
                    {"atr_windows": [window]},
                    "volatility"
                )
            
            # Tỷ lệ biến động
            self.register_feature(
                "volatility_ratio",
                calculate_volatility_ratio,
                {"short_windows": [5, 10], "long_windows": [20, 50]},
                "volatility"
            )
            
            # Mẫu hình biến động
            self.register_feature(
                "volatility_patterns",
                calculate_volatility_patterns,
                {"vol_windows": [10, 20, 50], "lookback_periods": [5, 10, 20]},
                "volatility"
            )
            
            # Chế độ biến động
            for window in [20, 50]:
                self.register_feature(
                    f"volatility_regime_{window}",
                    calculate_volatility_regime,
                    {"window": window, "long_window": 100},
                    "volatility"
                )
        
        # Đăng ký các đặc trưng giá
        if "price" in categories:
            # Returns
            self.register_feature(
                "returns",
                calculate_returns,
                {"periods": [1, 2, 3, 5, 10, 20], "percentage": True},
                "price"
            )
            
            # Log Returns
            self.register_feature(
                "log_returns",
                calculate_log_returns,
                {"periods": [1, 2, 3, 5, 10, 20]},
                "price"
            )
            
            # Price Momentum
            self.register_feature(
                "price_momentum",
                calculate_price_momentum,
                {"windows": [5, 10, 20, 50, 100], "normalize": True},
                "price"
            )
            
            # Price Ratios
            self.register_feature(
                "price_ratios",
                calculate_price_ratios,
                {"windows": [5, 10, 20, 50]},
                "price"
            )
            
            # Price Channels
            self.register_feature(
                "price_channels",
                calculate_price_channels,
                {"windows": [10, 20, 50]},
                "price"
            )
            
            # Price Crossovers
            self.register_feature(
                "price_crossovers",
                calculate_price_crossovers,
                {"ma_windows": [5, 10, 20, 50, 100, 200]},
                "price"
            )
        
        self.logger.info(f"Đã đăng ký tổng cộng {len(self.registered_features)} đặc trưng mặc định")
    
    def register_custom_feature(
        self,
        feature_name: str,
        feature_func: Callable,
        feature_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Đăng ký một đặc trưng tùy chỉnh.
        
        Args:
            feature_name: Tên đặc trưng
            feature_func: Hàm tính toán đặc trưng
            feature_params: Tham số cho hàm đặc trưng
        """
        self.register_feature(feature_name, feature_func, feature_params, "custom")
    
    def compute_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        feature_func: Callable,
        feature_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Tính toán một đặc trưng cho DataFrame.
        
        Args:
            df: DataFrame đầu vào
            feature_name: Tên đặc trưng
            feature_func: Hàm tính toán đặc trưng
            feature_params: Tham số cho hàm đặc trưng
            
        Returns:
            DataFrame với đặc trưng đã tính toán
        """
        if feature_params is None:
            feature_params = {}
        
        try:
            start_time = time.time()
            
            # Kiểm tra đầu vào
            if df.empty:
                self.logger.warning(f"DataFrame rỗng, không thể tính toán đặc trưng '{feature_name}'")
                return df
            
            # Tính toán đặc trưng
            result_df = feature_func(df, **feature_params)
            
            # Tính toán thời gian xử lý
            process_time = time.time() - start_time
            
            if process_time > 1.0:  # Nếu xử lý mất hơn 1 giây
                self.logger.info(f"Tính toán đặc trưng '{feature_name}' hoàn thành trong {process_time:.2f}s")
            else:
                self.logger.debug(f"Tính toán đặc trưng '{feature_name}' hoàn thành trong {process_time*1000:.2f}ms")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán đặc trưng '{feature_name}': {str(e)}")
            # Trả về DataFrame gốc nếu có lỗi
            return df
    
    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Tính toán nhiều đặc trưng cho DataFrame.
        
        Args:
            df: DataFrame đầu vào
            feature_names: Danh sách tên đặc trưng cần tính toán (None cho tất cả)
            parallel: Xử lý song song hay không
            
        Returns:
            DataFrame với các đặc trưng đã tính toán
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không thể tính toán đặc trưng")
            return df
        
        # Tạo bản sao của DataFrame đầu vào
        result_df = df.copy()
        
        # Nếu không chỉ định feature_names, sử dụng tất cả đặc trưng đã đăng ký
        if feature_names is None:
            feature_names = list(self.registered_features.keys())
        
        # Lọc các đặc trưng không tồn tại
        valid_features = [name for name in feature_names if name in self.registered_features]
        
        if not valid_features:
            self.logger.warning("Không có đặc trưng hợp lệ để tính toán")
            return result_df
        
        self.logger.info(f"Bắt đầu tính toán {len(valid_features)} đặc trưng")
        
        # Xử lý tuần tự
        if not parallel or len(valid_features) <= 1 or self.max_workers <= 1:
            for feature_name in valid_features:
                feature_info = self.registered_features[feature_name]
                result_df = self.compute_feature(
                    result_df,
                    feature_name,
                    feature_info["func"],
                    feature_info["params"]
                )
        else:
            # Xử lý song song
            try:
                # Tạo các tác vụ cho các đặc trưng
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(valid_features))) as executor:
                    # Map từng đặc trưng vào một luồng riêng
                    futures = {}
                    
                    for feature_name in valid_features:
                        feature_info = self.registered_features[feature_name]
                        futures[feature_name] = executor.submit(
                            self.compute_feature,
                            result_df.copy(),  # Truyền bản sao để tránh xung đột
                            feature_name,
                            feature_info["func"],
                            feature_info["params"]
                        )
                    
                    # Tổng hợp kết quả
                    for feature_name, future in futures.items():
                        try:
                            # Lấy kết quả từ future
                            feature_df = future.result(timeout=60)  # Timeout 60s
                            
                            # Tìm các cột mới được thêm vào
                            original_cols = set(result_df.columns)
                            feature_cols = set(feature_df.columns)
                            new_cols = feature_cols - original_cols
                            
                            # Thêm các cột mới vào result_df
                            for col in new_cols:
                                result_df[col] = feature_df[col]
                                
                        except concurrent.futures.TimeoutError:
                            self.logger.warning(f"Tính toán đặc trưng '{feature_name}' vượt quá thời gian chờ")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi tính toán đặc trưng '{feature_name}' song song: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi xử lý song song: {str(e)}")
                # Fallback: xử lý tuần tự nếu có lỗi song song
                for feature_name in valid_features:
                    feature_info = self.registered_features[feature_name]
                    result_df = self.compute_feature(
                        result_df,
                        feature_name,
                        feature_info["func"],
                        feature_info["params"]
                    )
        
        # Tính toán số cột mới được thêm vào
        original_col_count = len(df.columns)
        final_col_count = len(result_df.columns)
        added_col_count = final_col_count - original_col_count
        
        self.logger.info(f"Đã hoàn thành tính toán đặc trưng: {added_col_count} cột mới được thêm vào")
        
        return result_df
    
    def create_feature_pipeline(
        self,
        feature_names: Optional[List[str]] = None,
        preprocessor_names: Optional[List[str]] = None,
        transformer_names: Optional[List[str]] = None,
        feature_selector: Optional[str] = None,
        save_pipeline: bool = True,
        pipeline_name: Optional[str] = None
    ) -> str:
        """
        Tạo một pipeline đặc trưng.
        
        Args:
            feature_names: Danh sách tên đặc trưng cần tính toán
            preprocessor_names: Danh sách tên tiền xử lý
            transformer_names: Danh sách tên biến đổi
            feature_selector: Tên bộ chọn lọc đặc trưng
            save_pipeline: Lưu pipeline hay không
            pipeline_name: Tên pipeline (tạo ngẫu nhiên nếu None)
            
        Returns:
            Tên pipeline đã tạo
        """
        # Tạo tên pipeline nếu không được cung cấp
        if pipeline_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pipeline_name = f"pipeline_{timestamp}"
        
        # Tạo pipeline mới
        pipeline = {
            "name": pipeline_name,
            "created_at": datetime.now().isoformat(),
            "feature_names": feature_names,
            "preprocessor_names": preprocessor_names,
            "transformer_names": transformer_names,
            "feature_selector": feature_selector,
            "fitted": False,
            "fitted_params": {}
        }
        
        # Lưu pipeline
        self.feature_pipelines[pipeline_name] = pipeline
        
        if save_pipeline:
            self._save_pipeline(pipeline_name)
        
        self.logger.info(f"Đã tạo pipeline '{pipeline_name}'")
        
        return pipeline_name
    
    def _save_pipeline(self, pipeline_name: str) -> bool:
        """
        Lưu pipeline vào file.
        
        Args:
            pipeline_name: Tên pipeline cần lưu
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if pipeline_name not in self.feature_pipelines:
            self.logger.warning(f"Pipeline '{pipeline_name}' không tồn tại")
            return False
        
        safe_pipeline_name = pipeline_name.replace('/', '_').replace('\\', '_')
        pipeline_path = self.data_dir / f"{safe_pipeline_name}.joblib"
        
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.feature_pipelines[pipeline_name], pipeline_path)
            self.logger.debug(f"Đã lưu pipeline '{pipeline_name}' vào {pipeline_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu pipeline '{pipeline_name}': {str(e)}")
            return False
    
    def load_pipeline(self, pipeline_name: str) -> bool:
        """
        Tải pipeline từ file.
        
        Args:
            pipeline_name: Tên pipeline cần tải
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        safe_pipeline_name = pipeline_name.replace('/', '_').replace('\\', '_')
        pipeline_path = self.data_dir / f"{safe_pipeline_name}.joblib"
        
        if not pipeline_path.exists():
            self.logger.warning(f"File pipeline '{pipeline_path}' không tồn tại")
            return False
        
        try:
            pipeline = joblib.load(pipeline_path)
            self.feature_pipelines[pipeline_name] = pipeline
            
            # Tải các tham số đã fit nếu có
            if pipeline.get("fitted", False) and "fitted_params" in pipeline:
                self.fitted_params[pipeline_name] = pipeline["fitted_params"]
            
            self.logger.info(f"Đã tải pipeline '{pipeline_name}' từ {pipeline_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải pipeline '{pipeline_name}': {str(e)}")
            return False
    
    def fit_transform_data(
        self,
        df: pd.DataFrame,
        pipeline_name: str,
        save_fitted_params: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng pipeline trên dữ liệu, bao gồm học và biến đổi.
        
        Args:
            df: DataFrame đầu vào
            pipeline_name: Tên pipeline cần áp dụng
            save_fitted_params: Lưu các tham số đã học hay không
            
        Returns:
            DataFrame đã biến đổi
        """
        if pipeline_name not in self.feature_pipelines:
            self.logger.warning(f"Pipeline '{pipeline_name}' không tồn tại")
            return df
        
        pipeline = self.feature_pipelines[pipeline_name]
        
        # Tạo bản sao của DataFrame đầu vào
        result_df = df.copy()
        
        # Bước 1: Tính toán các đặc trưng
        if pipeline.get("feature_names"):
            result_df = self.compute_features(
                result_df,
                feature_names=pipeline["feature_names"],
                parallel=True
            )
        
        # Bước 2: Áp dụng các bước tiền xử lý
        fitted_params = {}
        
        if pipeline.get("preprocessor_names"):
            for preprocessor_name in pipeline["preprocessor_names"]:
                if preprocessor_name in self.preprocessors:
                    preprocessor_func = self.preprocessors[preprocessor_name]
                    try:
                        # Áp dụng preprocessor với fit=True
                        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                        processed_df, params = preprocessor_func(
                            result_df,
                            columns=numeric_cols,
                            fit=True
                        )
                        
                        # Cập nhật DataFrame và fitted_params
                        result_df = processed_df
                        fitted_params[preprocessor_name] = params
                        
                        self.logger.debug(f"Đã áp dụng tiền xử lý '{preprocessor_name}'")
                        
                    except Exception as e:
                        self.logger.error(f"Lỗi khi áp dụng tiền xử lý '{preprocessor_name}': {str(e)}")
        
        # Bước 3: Áp dụng các bước biến đổi
        if pipeline.get("transformer_names"):
            for transformer_name in pipeline["transformer_names"]:
                if transformer_name in self.transformers:
                    transformer_func = self.transformers[transformer_name]
                    try:
                        # Áp dụng transformer
                        result_df = transformer_func(result_df)
                        
                        self.logger.debug(f"Đã áp dụng biến đổi '{transformer_name}'")
                        
                    except Exception as e:
                        self.logger.error(f"Lỗi khi áp dụng biến đổi '{transformer_name}': {str(e)}")
        
        # Bước 4: Áp dụng bộ chọn lọc đặc trưng
        if pipeline.get("feature_selector") and pipeline["feature_selector"] in self.feature_selectors:
            selector_func = self.feature_selectors[pipeline["feature_selector"]]
            try:
                # Áp dụng bộ chọn lọc
                result_df, selected_features = selector_func(result_df)
                
                # Lưu danh sách đặc trưng đã chọn
                fitted_params["selected_features"] = selected_features
                
                self.logger.debug(f"Đã áp dụng bộ chọn lọc '{pipeline['feature_selector']}', giữ lại {len(selected_features)} đặc trưng")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi áp dụng bộ chọn lọc '{pipeline['feature_selector']}': {str(e)}")
        
        # Cập nhật trạng thái của pipeline
        pipeline["fitted"] = True
        pipeline["fitted_params"] = fitted_params
        pipeline["last_fitted"] = datetime.now().isoformat()
        
        # Lưu lại các tham số đã học
        if save_fitted_params:
            self.fitted_params[pipeline_name] = fitted_params
            self._save_pipeline(pipeline_name)
        
        return result_df
    
    def transform_data(
        self,
        df: pd.DataFrame,
        pipeline_name: str,
        fit: bool = False,
        preserve_timestamp: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng pipeline đã học trên dữ liệu mới.
        
        Args:
            df: DataFrame đầu vào
            pipeline_name: Tên pipeline cần áp dụng
            fit: Học lại pipeline hay không
            preserve_timestamp: Giữ lại cột timestamp hay không
            
        Returns:
            DataFrame đã biến đổi
        """
        if pipeline_name not in self.feature_pipelines:
            self.logger.warning(f"Pipeline '{pipeline_name}' không tồn tại")
            return df
        
        pipeline = self.feature_pipelines[pipeline_name]
        
        # Áp dụng fit_transform nếu được yêu cầu hoặc pipeline chưa được fit
        if fit or not pipeline.get("fitted", False):
            return self.fit_transform_data(df, pipeline_name, save_fitted_params=True)
        
        # Tạo bản sao của DataFrame đầu vào
        result_df = df.copy()
        
        # Lưu cột timestamp nếu cần
        timestamp_col = None
        if preserve_timestamp and 'timestamp' in result_df.columns:
            timestamp_col = result_df['timestamp'].copy()
        
        # Bước 1: Tính toán các đặc trưng
        if pipeline.get("feature_names"):
            result_df = self.compute_features(
                result_df,
                feature_names=pipeline["feature_names"],
                parallel=True
            )
        
        # Bước 2: Áp dụng các bước tiền xử lý với tham số đã học
        if pipeline.get("preprocessor_names"):
            for preprocessor_name in pipeline["preprocessor_names"]:
                if preprocessor_name in self.preprocessors:
                    preprocessor_func = self.preprocessors[preprocessor_name]
                    try:
                        # Kiểm tra xem có tham số đã học cho preprocessor này không
                        if (pipeline_name in self.fitted_params and 
                            preprocessor_name in self.fitted_params[pipeline_name]):
                            
                            fitted_params = self.fitted_params[pipeline_name][preprocessor_name]
                            
                            # Áp dụng preprocessor với fit=False
                            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                            result_df = preprocessor_func(
                                result_df,
                                columns=numeric_cols,
                                fit=False,
                                fitted_params=fitted_params
                            )
                            
                            self.logger.debug(f"Đã áp dụng tiền xử lý '{preprocessor_name}' với tham số đã học")
                            
                        else:
                            # Nếu không có tham số đã học, áp dụng với fit=True
                            self.logger.warning(f"Không tìm thấy tham số đã học cho '{preprocessor_name}', thực hiện fit_transform")
                            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
                            processed_df, _ = preprocessor_func(
                                result_df,
                                columns=numeric_cols,
                                fit=True
                            )
                            result_df = processed_df
                            
                    except Exception as e:
                        self.logger.error(f"Lỗi khi áp dụng tiền xử lý '{preprocessor_name}': {str(e)}")
        
        # Bước 3: Áp dụng các bước biến đổi
        if pipeline.get("transformer_names"):
            for transformer_name in pipeline["transformer_names"]:
                if transformer_name in self.transformers:
                    transformer_func = self.transformers[transformer_name]
                    try:
                        # Áp dụng transformer
                        result_df = transformer_func(result_df)
                        
                        self.logger.debug(f"Đã áp dụng biến đổi '{transformer_name}'")
                        
                    except Exception as e:
                        self.logger.error(f"Lỗi khi áp dụng biến đổi '{transformer_name}': {str(e)}")
        
        # Bước 4: Áp dụng bộ chọn lọc đặc trưng
        if (pipeline.get("feature_selector") and 
            pipeline["feature_selector"] in self.feature_selectors and
            pipeline_name in self.fitted_params and 
            "selected_features" in self.fitted_params[pipeline_name]):
            
            # Lấy danh sách đặc trưng đã chọn
            selected_features = self.fitted_params[pipeline_name]["selected_features"]
            
            # Giữ lại các cột cần thiết
            essential_cols = []
            if 'timestamp' in result_df.columns:
                essential_cols.append('timestamp')
            if 'open' in result_df.columns:
                essential_cols.append('open')
            if 'high' in result_df.columns:
                essential_cols.append('high')
            if 'low' in result_df.columns:
                essential_cols.append('low')
            if 'close' in result_df.columns:
                essential_cols.append('close')
            if 'volume' in result_df.columns:
                essential_cols.append('volume')
            
            # Lọc các đặc trưng đã chọn cùng với các cột cần thiết
            all_cols = essential_cols + [col for col in selected_features if col in result_df.columns and col not in essential_cols]
            
            # Lọc DataFrame
            result_df = result_df[all_cols]
            
            self.logger.debug(f"Đã áp dụng bộ chọn lọc, giữ lại {len(all_cols)} cột")
        
        # Khôi phục cột timestamp nếu cần
        if timestamp_col is not None:
            result_df['timestamp'] = timestamp_col
        
        return result_df
    
    def _correlation_selector(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chọn lọc đặc trưng dựa trên tương quan.
        
        Args:
            df: DataFrame đầu vào
            threshold: Ngưỡng tương quan để loại bỏ đặc trưng
            target_column: Cột mục tiêu (nếu có)
            
        Returns:
            Tuple (DataFrame đã lọc, danh sách đặc trưng đã chọn)
        """
        # Tạo bản sao của DataFrame đầu vào
        result_df = df.copy()
        
        # Lấy các cột số
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ cột mục tiêu khỏi danh sách đặc trưng nếu có
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Loại bỏ các cột OHLCV khỏi quá trình chọn lọc
        essential_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        features = [col for col in numeric_cols if col not in essential_cols]
        
        # Nếu không có đặc trưng để chọn lọc
        if not features:
            return result_df, numeric_cols
        
        # Tính ma trận tương quan
        corr_matrix = result_df[features].corr().abs()
        
        # Tạo ma trận tam giác trên
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Tìm các cột có tương quan cao
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Giữ lại các cột không bị loại bỏ
        selected_features = [col for col in numeric_cols if col not in to_drop]
        
        # Lọc DataFrame
        all_cols = selected_features + [col for col in result_df.columns if col not in numeric_cols]
        result_df = result_df[all_cols]
        
        self.logger.info(f"Đã loại bỏ {len(to_drop)} đặc trưng có tương quan > {threshold}")
        
        return result_df, selected_features
    
    def _variance_selector(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chọn lọc đặc trưng dựa trên phương sai.
        
        Args:
            df: DataFrame đầu vào
            threshold: Ngưỡng phương sai để giữ lại đặc trưng
            
        Returns:
            Tuple (DataFrame đã lọc, danh sách đặc trưng đã chọn)
        """
        # Tạo bản sao của DataFrame đầu vào
        result_df = df.copy()
        
        # Lấy các cột số
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Loại bỏ các cột OHLCV khỏi quá trình chọn lọc
        essential_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        features = [col for col in numeric_cols if col not in essential_cols]
        
        # Nếu không có đặc trưng để chọn lọc
        if not features:
            return result_df, numeric_cols
        
        # Tính phương sai của mỗi đặc trưng
        variances = result_df[features].var()
        
        # Chọn lọc các đặc trưng có phương sai lớn hơn ngưỡng
        selected_features = [col for col in features if variances[col] > threshold]
        selected_features += [col for col in numeric_cols if col not in features]
        
        # Lọc DataFrame
        all_cols = selected_features + [col for col in result_df.columns if col not in numeric_cols]
        result_df = result_df[all_cols]
        
        self.logger.info(f"Đã loại bỏ {len(features) - len(selected_features) + len(numeric_cols) - len(features)} đặc trưng có phương sai < {threshold}")
        
        return result_df, selected_features
    
    def _statistical_correlation_selector(
        self,
        df: pd.DataFrame,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.001,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chọn lọc đặc trưng dựa trên phân tích thống kê.
        
        Args:
            df: DataFrame đầu vào
            correlation_threshold: Ngưỡng tương quan để loại bỏ đặc trưng
            variance_threshold: Ngưỡng phương sai để giữ lại đặc trưng
            target_column: Cột mục tiêu (nếu có)
            
        Returns:
            Tuple (DataFrame đã lọc, danh sách đặc trưng đã chọn)
        """
        # Bước 1: Loại bỏ các đặc trưng có phương sai thấp
        variance_filtered_df, variance_features = self._variance_selector(df, variance_threshold)
        
        # Bước 2: Loại bỏ các đặc trưng có tương quan cao
        final_df, selected_features = self._correlation_selector(
            variance_filtered_df, 
            correlation_threshold,
            target_column
        )
        
        return final_df, selected_features
    
    def save_state(self) -> bool:
        """
        Lưu trạng thái hiện tại của FeatureGenerator.
        
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            state_file = self.data_dir / "feature_generator_state.joblib"
            
            state = {
                "registered_features": self.registered_features,
                "feature_pipelines": self.feature_pipelines,
                "fitted_params": self.fitted_params,
                "timestamp": datetime.now().isoformat()
            }
            
            joblib.dump(state, state_file)
            self.logger.info(f"Đã lưu trạng thái FeatureGenerator vào {state_file}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái: {str(e)}")
            return False
    
    def load_state(self) -> bool:
        """
        Tải trạng thái của FeatureGenerator từ file.
        
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            state_file = self.data_dir / "feature_generator_state.joblib"
            
            if not state_file.exists():
                self.logger.warning(f"File trạng thái {state_file} không tồn tại")
                return False
            
            state = joblib.load(state_file)
            
            self.registered_features = state.get("registered_features", {})
            self.feature_pipelines = state.get("feature_pipelines", {})
            self.fitted_params = state.get("fitted_params", {})
            
            self.logger.info(f"Đã tải trạng thái FeatureGenerator từ {state_file}")
            
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái: {str(e)}")
            return False
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """
        Liệt kê tất cả các pipeline đã đăng ký.
        
        Returns:
            Danh sách các pipeline
        """
        return [
            {
                "name": name,
                "created_at": pipeline.get("created_at", "Unknown"),
                "feature_count": len(pipeline.get("feature_names", [])),
                "fitted": pipeline.get("fitted", False),
                "last_fitted": pipeline.get("last_fitted", "Never")
            }
            for name, pipeline in self.feature_pipelines.items()
        ]
    
    def list_features(self, category: Optional[str] = None) -> List[str]:
        """
        Liệt kê tất cả các đặc trưng đã đăng ký.
        
        Args:
            category: Danh mục đặc trưng (None để liệt kê tất cả)
            
        Returns:
            Danh sách tên đặc trưng
        """
        if category:
            return [
                name for name, info in self.registered_features.items()
                if info.get("category") == category
            ]
        else:
            return list(self.registered_features.keys())