"""
Lớp chính để backtest chiến lược giao dịch.
File này cung cấp lớp Backtester làm điểm truy cập chính cho các chức năng kiểm tra 
chiến lược giao dịch và đánh giá hiệu suất, tích hợp với các module khác trong hệ thống.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import concurrent.futures
from functools import partial
import pickle
import warnings

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderType, OrderStatus, PositionStatus
from config.constants import Timeframe, ErrorCode, BacktestMetric
from config.system_config import get_system_config, DATA_DIR, MODEL_DIR, BACKTEST_DIR

# Import các module backtesting
from backtesting.strategy_tester import StrategyTester
try:
    from backtesting.performance_metrics import PerformanceMetrics
    PERFORMANCE_METRICS_AVAILABLE = True
except ImportError:
    PERFORMANCE_METRICS_AVAILABLE = False
    
try:
    from backtesting.historical_simulator import HistoricalSimulator
    HISTORICAL_SIMULATOR_AVAILABLE = True
except ImportError:
    HISTORICAL_SIMULATOR_AVAILABLE = False

# Import các module xử lý dữ liệu
try:
    from data_processors.data_pipeline import DataPipeline
    DATA_PIPELINE_AVAILABLE = True
except ImportError:
    DATA_PIPELINE_AVAILABLE = False


class Backtester:
    """
    Lớp chính để backtest các chiến lược giao dịch.
    
    Cung cấp giao diện thống nhất để:
    1. Tải và tiền xử lý dữ liệu
    2. Định nghĩa và đăng ký chiến lược
    3. Chạy backtest trên nhiều khung thời gian và cặp tiền
    4. Phân tích và đánh giá hiệu suất
    5. Trực quan hóa kết quả
    6. Tối ưu hóa tham số chiến lược
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        use_data_pipeline: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo backtester.
        
        Args:
            data_dir: Thư mục dữ liệu đầu vào
            output_dir: Thư mục đầu ra kết quả
            config: Cấu hình backtester
            use_data_pipeline: Sử dụng DataPipeline cho xử lý dữ liệu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("backtester")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thư mục dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR
        else:
            self.data_dir = Path(data_dir)
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = BACKTEST_DIR / f"backtest_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình mặc định
        self.default_config = {
            "backtest": {
                "initial_balance": 10000.0,
                "fee_rate": 0.001,
                "slippage": 0.0005,
                "leverage": 1.0,
                "risk_per_trade": 0.02,
                "allow_short": True,
                "max_positions": 1,
                "start_date": None,
                "end_date": None,
                "use_target_column": False,
                "target_columns": ["label", "future_return_10", "direction_10"],
                "data_frequency": "1h",
                "rebalance_frequency": "1d",
                "execution_delay": 0,
                "commission_model": "percentage",
                "slippage_model": "percentage"
            },
            "strategy_defaults": {
                "lookback_window": 20,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
                "rsi_period": 14,
                "atr_period": 14,
                "volatility_period": 20,
                "signal_threshold": 0.0,
                "overbought_threshold": 70,
                "oversold_threshold": 30,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.1,
                "trailing_stop_pct": 0.03,
                "ignored_features": ["timestamp", "volume_log"]
            },
            "optimization": {
                "max_workers": 8,
                "metric_to_optimize": BacktestMetric.SHARPE_RATIO.value,
                "use_walk_forward": True,
                "walk_forward_window": 90,
                "walk_forward_step": 30,
                "retrain_on_each_window": True
            },
            "visualization": {
                "plot_equity_curve": True,
                "plot_drawdowns": True,
                "plot_trade_distribution": True,
                "plot_monthly_returns": True,
                "plot_rolling_metrics": True,
                "rolling_window": 30,
                "save_plots": True,
                "plot_format": "png",
                "plot_dpi": 150,
                "plot_figsize": (12, 8)
            },
            "data_processing": {
                "use_pipeline": use_data_pipeline and DATA_PIPELINE_AVAILABLE,
                "clean_data": True,
                "generate_features": True,
                "remove_outliers": True,
                "feature_selection": True,
                "normalization": True,
                "impute_missing": True,
                "target_generation": True,
                "test_size": 0.2
            }
        }
        
        # Cập nhật cấu hình với cấu hình được cung cấp
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
        
        # Khởi tạo DataPipeline nếu sử dụng
        self.data_pipeline = None
        if self.config["data_processing"]["use_pipeline"]:
            try:
                self.data_pipeline = DataPipeline(data_dir=self.data_dir, output_dir=self.output_dir / "data")
                self.logger.info("Đã khởi tạo DataPipeline")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo DataPipeline: {e}")
                self.config["data_processing"]["use_pipeline"] = False
        
        # Khởi tạo dictionary lưu trữ chiến lược
        self.strategies = {}
        
        # Khởi tạo dictionary lưu trữ StrategyTester
        self.testers = {}
        
        # Khởi tạo dictionary lưu trữ dữ liệu đã xử lý
        self.processed_data = {}
        
        # Khởi tạo danh sách kết quả
        self.results = {}
        
        # Khởi tạo PerformanceMetrics nếu có
        self.performance_metrics = None
        if PERFORMANCE_METRICS_AVAILABLE:
            try:
                self.performance_metrics = PerformanceMetrics(logger=self.logger)
                self.logger.info("Đã khởi tạo PerformanceMetrics")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo PerformanceMetrics: {e}")
        
        # Khởi tạo HistoricalSimulator nếu có
        self.historical_simulator = None
        if HISTORICAL_SIMULATOR_AVAILABLE:
            try:
                self.historical_simulator = HistoricalSimulator(logger=self.logger)
                self.logger.info("Đã khởi tạo HistoricalSimulator")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo HistoricalSimulator: {e}")
        
        self.logger.info(f"Đã khởi tạo Backtester với output_dir={self.output_dir}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Cập nhật từ điển lồng nhau.
        
        Args:
            d: Từ điển cần cập nhật
            u: Từ điển chứa các giá trị mới
            
        Returns:
            Từ điển đã cập nhật
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def register_strategy(
        self,
        strategy_func: Callable,
        strategy_name: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Đăng ký chiến lược giao dịch.
        
        Args:
            strategy_func: Hàm chiến lược
            strategy_name: Tên chiến lược
            strategy_params: Tham số chiến lược
        """
        # Thiết lập tham số mặc định
        if strategy_params is None:
            strategy_params = {}
        
        # Kết hợp tham số mặc định và tham số tùy chỉnh
        default_params = self.config["strategy_defaults"].copy()
        params = {**default_params, **strategy_params}
        
        # Đăng ký chiến lược
        self.strategies[strategy_name] = {
            "function": strategy_func,
            "params": params,
            "registered_at": datetime.now().isoformat()
        }
        
        # Tạo thư mục đầu ra cho chiến lược
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Đã đăng ký chiến lược '{strategy_name}' với {len(params)} tham số")
    
    def load_data(
        self,
        symbols: Union[str, List[str]],
        timeframe: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        file_paths: Optional[Dict[str, Path]] = None,
        data_format: str = "parquet",
        clean_data: bool = None,
        generate_features: bool = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Tải dữ liệu cho backtest.
        
        Args:
            symbols: Danh sách ký hiệu tài sản
            timeframe: Khung thời gian dữ liệu
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            file_paths: Dict ánh xạ symbol -> đường dẫn file
            data_format: Định dạng dữ liệu
            clean_data: Làm sạch dữ liệu
            generate_features: Tạo đặc trưng
            
        Returns:
            Dict ánh xạ symbol -> DataFrame dữ liệu
        """
        # Chuyển đổi symbols thành danh sách
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Chuyển đổi start_date và end_date thành datetime
        if start_date is None:
            start_date = self.config["backtest"]["start_date"]
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        if end_date is None:
            end_date = self.config["backtest"]["end_date"]
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Thiết lập clean_data và generate_features
        if clean_data is None:
            clean_data = self.config["data_processing"]["clean_data"]
        if generate_features is None:
            generate_features = self.config["data_processing"]["generate_features"]
        
        # Tạo dictionary lưu trữ dữ liệu
        data_dict = {}
        
        # Nếu sử dụng DataPipeline
        if self.config["data_processing"]["use_pipeline"] and self.data_pipeline:
            # Sử dụng DataPipeline để tải và xử lý dữ liệu
            try:
                # Thiết lập đường dẫn file nếu có
                input_files = []
                if file_paths:
                    for symbol in symbols:
                        if symbol in file_paths:
                            input_files.append(file_paths[symbol])
                
                # Thiết lập pipeline và chạy
                pipeline_params = {
                    "clean_data": {
                        "enabled": clean_data,
                        "handle_leading_nan": self.config["data_processing"]["impute_missing"],
                        "remove_outliers": self.config["data_processing"]["remove_outliers"]
                    },
                    "generate_features": {
                        "enabled": generate_features,
                        "all_indicators": True
                    }
                }
                
                # Chạy pipeline
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result_data = loop.run_until_complete(
                    self.data_pipeline.run_pipeline(
                        input_files=input_files,
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=start_date,
                        end_time=end_date,
                        save_results=False
                    )
                )
                
                # Kiểm tra kết quả
                if result_data:
                    # Lọc theo start_date và end_date nếu chưa được lọc
                    for symbol, df in result_data.items():
                        if 'timestamp' in df.columns:
                            if start_date:
                                df = df[df['timestamp'] >= start_date]
                            if end_date:
                                df = df[df['timestamp'] <= end_date]
                            data_dict[symbol] = df
                
                if not data_dict:
                    self.logger.warning("DataPipeline không trả về dữ liệu. Chuyển sang phương pháp tải trực tiếp.")
                else:
                    self.logger.info(f"Đã tải và xử lý dữ liệu cho {len(data_dict)} symbols sử dụng DataPipeline")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng DataPipeline: {e}")
                self.logger.warning("Chuyển sang phương pháp tải trực tiếp")
        
        # Nếu không sử dụng DataPipeline hoặc DataPipeline không trả về dữ liệu
        if not data_dict:
            # Tải dữ liệu trực tiếp
            for symbol in symbols:
                file_path = None
                if file_paths and symbol in file_paths:
                    file_path = file_paths[symbol]
                
                # Tạo StrategyTester tạm thời để tải dữ liệu
                temp_tester = StrategyTester(
                    strategy_name=f"temp_{symbol}",
                    logger=self.logger
                )
                
                # Tải dữ liệu
                df = temp_tester.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    file_path=file_path,
                    data_format=data_format
                )
                
                if not df.empty:
                    data_dict[symbol] = df
                    self.logger.info(f"Đã tải {len(df)} dòng dữ liệu cho {symbol}")
                else:
                    self.logger.warning(f"Không tìm thấy dữ liệu cho {symbol}")
        
        # Lưu dữ liệu đã xử lý
        self.processed_data.update(data_dict)
        
        return data_dict
    
    def prepare_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target_column: Optional[str] = None,
        test_size: Optional[float] = None,
        generate_features: bool = None,
        normalize: bool = None,
        feature_selection: bool = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Chuẩn bị dữ liệu cho backtest.
        
        Args:
            data_dict: Dict ánh xạ symbol -> DataFrame dữ liệu
            target_column: Tên cột mục tiêu
            test_size: Tỷ lệ dữ liệu kiểm tra
            generate_features: Tạo đặc trưng
            normalize: Chuẩn hóa dữ liệu
            feature_selection: Chọn lọc đặc trưng
            
        Returns:
            Dict ánh xạ symbol -> Dict chứa 'train' và 'test' DataFrames
        """
        if not data_dict:
            self.logger.error("Không có dữ liệu để chuẩn bị")
            return {}
        
        # Thiết lập tham số
        if test_size is None:
            test_size = self.config["data_processing"]["test_size"]
        
        if generate_features is None:
            generate_features = self.config["data_processing"]["generate_features"]
        
        if normalize is None:
            normalize = self.config["data_processing"]["normalization"]
        
        if feature_selection is None:
            feature_selection = self.config["data_processing"]["feature_selection"]
        
        # Sử dụng target_column từ cấu hình nếu không được chỉ định
        if target_column is None and self.config["backtest"]["use_target_column"] and self.config["backtest"]["target_columns"]:
            target_columns = self.config["backtest"]["target_columns"]
            # Tìm cột target đầu tiên có trong dữ liệu
            for symbol, df in data_dict.items():
                for col in target_columns:
                    if col in df.columns:
                        target_column = col
                        break
                if target_column:
                    break
        
        # Kết quả
        prepared_data = {}
        
        # Chuẩn bị dữ liệu cho mỗi symbol
        for symbol, df in data_dict.items():
            self.logger.info(f"Chuẩn bị dữ liệu cho {symbol}")
            
            # Xử lý dữ liệu thiếu nếu có
            if df.isna().any().any() and self.config["data_processing"]["impute_missing"]:
                self.logger.info(f"Phát hiện dữ liệu thiếu trong {symbol}, tiến hành xử lý")
                # Xử lý NaN ở cột timestamp
                if 'timestamp' in df.columns and df['timestamp'].isna().any():
                    df = df.dropna(subset=['timestamp'])
                
                # Xử lý NaN ở các cột khác
                for col in df.columns:
                    if col == 'timestamp':
                        continue
                    
                    # Đếm số giá trị NaN
                    na_count = df[col].isna().sum()
                    if na_count > 0:
                        self.logger.debug(f"Cột {col} có {na_count} giá trị NaN")
                        # Sử dụng phương pháp nội suy
                        df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # Chuẩn hóa dữ liệu nếu cần
            if normalize:
                try:
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    # Loại bỏ cột target và timestamp nếu có
                    if target_column and target_column in numeric_cols:
                        numeric_cols.remove(target_column)
                    if 'timestamp' in numeric_cols:
                        numeric_cols.remove('timestamp')
                    
                    # Loại bỏ các cột bị bỏ qua
                    for col in self.config["strategy_defaults"]["ignored_features"]:
                        if col in numeric_cols:
                            numeric_cols.remove(col)
                    
                    # Chuẩn hóa Z-score
                    for col in numeric_cols:
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[f"{col}_norm"] = (df[col] - mean) / std
                    
                    self.logger.info(f"Đã chuẩn hóa {len(numeric_cols)} cột số cho {symbol}")
                
                except Exception as e:
                    self.logger.error(f"Lỗi khi chuẩn hóa dữ liệu cho {symbol}: {e}")
            
            # Chọn lọc đặc trưng nếu cần
            if feature_selection and len(df.columns) > 20:
                try:
                    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
                    
                    # Chọn các cột số
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    # Loại bỏ cột target và timestamp nếu có
                    if target_column and target_column in numeric_cols:
                        numeric_cols.remove(target_column)
                    if 'timestamp' in numeric_cols:
                        numeric_cols.remove('timestamp')
                    
                    # Loại bỏ các cột bị bỏ qua
                    for col in self.config["strategy_defaults"]["ignored_features"]:
                        if col in numeric_cols:
                            numeric_cols.remove(col)
                    
                    # Tạo bản sao DataFrame chỉ với các cột số
                    X = df[numeric_cols].copy()
                    
                    # Loại bỏ các đặc trưng có phương sai thấp
                    var_threshold = VarianceThreshold(threshold=0.01)
                    X_var = var_threshold.fit_transform(X)
                    selected_var_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if var_threshold.get_support()[i]]
                    
                    # Chọn K đặc trưng tốt nhất nếu có target_column
                    if target_column and target_column in df.columns:
                        y = df[target_column].copy()
                        
                        # Loại bỏ các mẫu có giá trị NaN trong target
                        valid_idx = ~y.isna()
                        X_valid = X.loc[valid_idx]
                        y_valid = y.loc[valid_idx]
                        
                        if len(X_valid) > 0:
                            # Chọn k đặc trưng tốt nhất
                            k = min(30, X_valid.shape[1])
                            selector = SelectKBest(f_regression, k=k)
                            selector.fit(X_valid, y_valid)
                            
                            selected_kb_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if selector.get_support()[i]]
                            
                            # Kết hợp các đặc trưng đã chọn
                            selected_cols = list(set(selected_var_cols).intersection(set(selected_kb_cols)))
                            
                            self.logger.info(f"Đã chọn {len(selected_cols)} đặc trưng tốt nhất cho {symbol} dựa trên target {target_column}")
                        else:
                            selected_cols = selected_var_cols
                            self.logger.warning(f"Không có dữ liệu hợp lệ để chọn lọc đặc trưng dựa trên target cho {symbol}")
                    else:
                        selected_cols = selected_var_cols
                        self.logger.info(f"Đã loại bỏ các đặc trưng có phương sai thấp, còn lại {len(selected_cols)} đặc trưng cho {symbol}")
                    
                    # Tạo DataFrame mới với các cột đã chọn
                    selected_df = df[['timestamp'] + selected_cols].copy()
                    if target_column and target_column in df.columns:
                        selected_df[target_column] = df[target_column]
                    
                    # Thêm các cột OHLCV cơ bản
                    basic_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in basic_cols:
                        if col in df.columns and col not in selected_df.columns:
                            selected_df[col] = df[col]
                    
                    # Cập nhật DataFrame
                    df = selected_df
                    self.logger.info(f"DataFrame sau khi chọn lọc đặc trưng: {df.shape[0]} dòng, {df.shape[1]} cột")
                
                except Exception as e:
                    self.logger.error(f"Lỗi khi chọn lọc đặc trưng cho {symbol}: {e}")
            
            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            if test_size > 0:
                # Tính chỉ số phân chia
                split_idx = int(len(df) * (1 - test_size))
                
                # Chia dữ liệu
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()
                
                prepared_data[symbol] = {
                    'train': train_df,
                    'test': test_df,
                    'all': df
                }
                
                self.logger.info(f"Đã chia dữ liệu cho {symbol}: {len(train_df)} mẫu huấn luyện, {len(test_df)} mẫu kiểm tra")
            else:
                # Sử dụng toàn bộ dữ liệu
                prepared_data[symbol] = {
                    'train': df.copy(),
                    'test': pd.DataFrame(),
                    'all': df.copy()
                }
                
                self.logger.info(f"Sử dụng toàn bộ {len(df)} mẫu dữ liệu cho {symbol}")
        
        return prepared_data
    
    def run_backtest(
        self,
        strategy_name: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy_params: Optional[Dict[str, Any]] = None,
        backtest_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Chạy backtest cho một chiến lược.
        
        Args:
            strategy_name: Tên chiến lược
            data: DataFrame hoặc Dict chứa dữ liệu
            strategy_params: Tham số chiến lược
            backtest_params: Tham số backtest
            verbose: In thông tin chi tiết
            
        Returns:
            Dict kết quả backtest
        """
        # Kiểm tra chiến lược
        if strategy_name not in self.strategies:
            self.logger.error(f"Chiến lược '{strategy_name}' chưa được đăng ký")
            return {"status": "error", "message": f"Chiến lược '{strategy_name}' chưa được đăng ký"}
        
        # Lấy thông tin chiến lược
        strategy_info = self.strategies[strategy_name]
        strategy_func = strategy_info["function"]
        
        # Kết hợp tham số chiến lược
        combined_strategy_params = strategy_info["params"].copy()
        if strategy_params:
            combined_strategy_params.update(strategy_params)
        
        # Thiết lập tham số backtest
        bt_params = self.config["backtest"].copy()
        if backtest_params:
            bt_params.update(backtest_params)
        
        # Thư mục đầu ra cho chiến lược
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # Kết quả backtest
        backtest_results = {
            "strategy_name": strategy_name,
            "strategy_params": combined_strategy_params,
            "backtest_params": bt_params,
            "symbol_results": {},
            "combined_result": None,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        # Tiền xử lý dữ liệu
        if isinstance(data, pd.DataFrame):
            # Nếu data là DataFrame, sử dụng nó cho symbol mặc định
            test_data = {"default": data}
        elif isinstance(data, dict):
            # Nếu data là dict, sử dụng các key là symbol
            test_data = data
        else:
            self.logger.error(f"Định dạng dữ liệu không được hỗ trợ: {type(data)}")
            return {"status": "error", "message": f"Định dạng dữ liệu không được hỗ trợ: {type(data)}"}
        
        # Chạy backtest cho mỗi symbol
        for symbol, symbol_data in test_data.items():
            self.logger.info(f"Chạy backtest chiến lược '{strategy_name}' cho {symbol}")
            
            # Tạo StrategyTester
            tester = StrategyTester(
                strategy_func=strategy_func,
                strategy_name=f"{strategy_name}_{symbol}",
                initial_balance=bt_params["initial_balance"],
                fee_rate=bt_params["fee_rate"],
                slippage=bt_params["slippage"],
                leverage=bt_params["leverage"],
                risk_per_trade=bt_params["risk_per_trade"],
                allow_short=bt_params["allow_short"],
                max_positions=bt_params["max_positions"],
                logger=self.logger
            )
            
            # Lưu tester
            self.testers[f"{strategy_name}_{symbol}"] = tester
            
            # Chạy backtest
            try:
                # Chuẩn bị strategy_args
                strategy_args = {
                    "params": combined_strategy_params,
                    "symbol": symbol
                }
                
                # Chạy test
                test_result = tester.run_test(
                    data=symbol_data,
                    strategy_args=strategy_args,
                    verbose=verbose
                )
                
                # Lưu kết quả
                backtest_results["symbol_results"][symbol] = test_result
                
                # Lưu kết quả vào file
                result_file = strategy_dir / f"{symbol}_result.json"
                tester.save_results(test_result, file_path=result_file)
                
                self.logger.info(f"Kết quả backtest cho {symbol}: ROI {test_result['roi']:.2%}, Sharpe {test_result['metrics'].get('sharpe_ratio', 0):.2f}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi chạy backtest cho {symbol}: {e}")
                backtest_results["symbol_results"][symbol] = {"status": "error", "message": str(e)}
        
        # Tính kết quả tổng hợp
        if backtest_results["symbol_results"]:
            try:
                # Tạo danh sách kết quả
                results_list = [r for r in backtest_results["symbol_results"].values() if r.get("status") == "success"]
                
                if results_list:
                    # Tính các chỉ số trung bình
                    metrics_keys = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'win_rate']
                    combined_metrics = {}
                    
                    for key in metrics_keys:
                        values = [r.get("metrics", {}).get(key, 0) for r in results_list]
                        if values:
                            combined_metrics[key] = sum(values) / len(values)
                    
                    # Tính tổng ROI
                    total_initial_balance = sum(r.get("initial_balance", bt_params["initial_balance"]) for r in results_list)
                    total_final_balance = sum(r.get("final_balance", 0) for r in results_list)
                    
                    if total_initial_balance > 0:
                        combined_roi = (total_final_balance - total_initial_balance) / total_initial_balance
                    else:
                        combined_roi = 0
                    
                    # Tạo kết quả tổng hợp
                    backtest_results["combined_result"] = {
                        "metrics": combined_metrics,
                        "initial_balance": total_initial_balance,
                        "final_balance": total_final_balance,
                        "roi": combined_roi,
                        "symbols_count": len(results_list)
                    }
                    
                    self.logger.info(f"Kết quả tổng hợp: ROI {combined_roi:.2%}, Sharpe {combined_metrics.get('sharpe_ratio', 0):.2f}")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi tính kết quả tổng hợp: {e}")
        
        # Cập nhật trạng thái
        backtest_results["end_time"] = datetime.now().isoformat()
        backtest_results["status"] = "completed"
        
        # Lưu kết quả
        self.results[strategy_name] = backtest_results
        
        # Lưu kết quả tổng hợp vào file
        combined_file = strategy_dir / "combined_result.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(backtest_results, f, indent=4, ensure_ascii=False)
        
        return backtest_results
    
    def run_multiple_backtests(
        self,
        strategies: Union[str, List[str]],
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
        backtest_params: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        max_workers: int = None,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Chạy backtest cho nhiều chiến lược.
        
        Args:
            strategies: Tên chiến lược hoặc danh sách tên chiến lược
            data: DataFrame hoặc Dict chứa dữ liệu
            strategy_params: Dict ánh xạ tên chiến lược -> tham số
            backtest_params: Tham số backtest chung
            parallel: Chạy song song hay tuần tự
            max_workers: Số luồng tối đa
            verbose: In thông tin chi tiết
            
        Returns:
            Dict ánh xạ tên chiến lược -> kết quả backtest
        """
        # Chuyển đổi strategies thành danh sách
        if isinstance(strategies, str):
            strategies = [strategies]
        
        # Kiểm tra chiến lược
        valid_strategies = [s for s in strategies if s in self.strategies]
        if not valid_strategies:
            self.logger.error("Không có chiến lược hợp lệ để chạy backtest")
            return {}
        
        # Thiết lập strategy_params
        if strategy_params is None:
            strategy_params = {}
        
        # Thiết lập max_workers
        if max_workers is None:
            max_workers = self.config["optimization"]["max_workers"]
        
        # Kết quả
        all_results = {}
        
        if parallel and len(valid_strategies) > 1 and max_workers > 1:
            # Chạy song song
            self.logger.info(f"Chạy song song {len(valid_strategies)} chiến lược với {max_workers} workers")
            
            # Hàm chạy một backtest
            def run_one_backtest(strategy):
                params = strategy_params.get(strategy, {})
                return strategy, self.run_backtest(
                    strategy_name=strategy,
                    data=data,
                    strategy_params=params,
                    backtest_params=backtest_params,
                    verbose=verbose
                )
            
            # Chạy song song
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for strategy, result in executor.map(run_one_backtest, valid_strategies):
                    all_results[strategy] = result
        
        else:
            # Chạy tuần tự
            for strategy in valid_strategies:
                self.logger.info(f"Chạy backtest cho chiến lược '{strategy}'")
                
                params = strategy_params.get(strategy, {})
                result = self.run_backtest(
                    strategy_name=strategy,
                    data=data,
                    strategy_params=params,
                    backtest_params=backtest_params,
                    verbose=verbose
                )
                
                all_results[strategy] = result
        
        # Lưu tất cả kết quả
        with open(self.output_dir / "all_results.json", "w", encoding="utf-8") as f:
            # Tạo dict có thể serialize
            serializable_results = {}
            for strategy, result in all_results.items():
                # Loại bỏ các phần không thể serialize
                clean_result = {}
                for key, value in result.items():
                    if key != "symbol_results":
                        clean_result[key] = value
                    else:
                        # Chỉ lấy các thông tin cần thiết từ symbol_results
                        clean_symbols = {}
                        for symbol, symbol_result in value.items():
                            if isinstance(symbol_result, dict):
                                clean_symbol_result = {
                                    "roi": symbol_result.get("roi", 0),
                                    "metrics": symbol_result.get("metrics", {}),
                                    "initial_balance": symbol_result.get("initial_balance", 0),
                                    "final_balance": symbol_result.get("final_balance", 0),
                                    "status": symbol_result.get("status", "unknown")
                                }
                                clean_symbols[symbol] = clean_symbol_result
                        clean_result["symbol_results"] = clean_symbols
                
                serializable_results[strategy] = clean_result
            
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        
        return all_results
    
    def compare_strategies(
        self,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        show_plot: bool = True,
        save_plot: bool = True
    ) -> pd.DataFrame:
        """
        So sánh hiệu suất của nhiều chiến lược.
        
        Args:
            results: Dict ánh xạ tên chiến lược -> kết quả backtest
            metrics: Danh sách chỉ số cần so sánh
            symbols: Danh sách ký hiệu tài sản cần so sánh
            show_plot: Hiển thị biểu đồ
            save_plot: Lưu biểu đồ
            
        Returns:
            DataFrame so sánh các chiến lược
        """
        # Sử dụng kết quả có sẵn nếu không được cung cấp
        if results is None:
            results = self.results
        
        if not results:
            self.logger.error("Không có kết quả để so sánh")
            return pd.DataFrame()
        
        # Thiết lập metrics mặc định
        if metrics is None:
            metrics = [
                BacktestMetric.TOTAL_RETURN.value,
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value,
                "roi"
            ]
        
        # Dữ liệu so sánh
        comparison_data = []
        
        # Lặp qua từng chiến lược
        for strategy_name, strategy_result in results.items():
            # Lấy kết quả tổng hợp
            if strategy_result.get("combined_result"):
                combined = strategy_result["combined_result"]
                
                row = {
                    "strategy": strategy_name,
                    "symbols_count": combined.get("symbols_count", 0),
                    "initial_balance": combined.get("initial_balance", 0),
                    "final_balance": combined.get("final_balance", 0),
                    "roi": combined.get("roi", 0)
                }
                
                # Thêm các metrics
                for metric in metrics:
                    if metric in combined.get("metrics", {}):
                        row[metric] = combined["metrics"][metric]
                
                comparison_data.append(row)
            
            # Lặp qua từng symbol nếu cần
            if symbols:
                for symbol in symbols:
                    if symbol in strategy_result.get("symbol_results", {}):
                        symbol_result = strategy_result["symbol_results"][symbol]
                        
                        if not isinstance(symbol_result, dict) or symbol_result.get("status") != "success":
                            continue
                        
                        row = {
                            "strategy": f"{strategy_name}_{symbol}",
                            "symbol": symbol,
                            "initial_balance": symbol_result.get("initial_balance", 0),
                            "final_balance": symbol_result.get("final_balance", 0),
                            "roi": symbol_result.get("roi", 0)
                        }
                        
                        # Thêm các metrics
                        for metric in metrics:
                            if metric in symbol_result.get("metrics", {}):
                                row[metric] = symbol_result["metrics"][metric]
                        
                        comparison_data.append(row)
        
        # Tạo DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # In bảng so sánh
        if not comparison_df.empty:
            # Định dạng cột ROI và Max Drawdown thành phần trăm
            for col in ['roi', BacktestMetric.MAX_DRAWDOWN.value]:
                if col in comparison_df.columns:
                    comparison_df[f"{col}_pct"] = comparison_df[col].apply(lambda x: f"{x:.2%}")
            
            self.logger.info("\nSo sánh hiệu suất chiến lược:")
            self.logger.info("-" * 80)
            self.logger.info(comparison_df.to_string(index=False))
            self.logger.info("-" * 80)
            
            # Lưu bảng so sánh
            comparison_df.to_csv(self.output_dir / "strategy_comparison.csv", index=False)
            
            # Vẽ biểu đồ so sánh
            if show_plot or save_plot:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Bar chart ROI
                    plt.subplot(2, 2, 1)
                    sns.barplot(x='strategy', y='roi', data=comparison_df)
                    plt.title('ROI by Strategy')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Bar chart Sharpe Ratio
                    plt.subplot(2, 2, 2)
                    if BacktestMetric.SHARPE_RATIO.value in comparison_df.columns:
                        sns.barplot(x='strategy', y=BacktestMetric.SHARPE_RATIO.value, data=comparison_df)
                        plt.title('Sharpe Ratio by Strategy')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Bar chart Max Drawdown
                    plt.subplot(2, 2, 3)
                    if BacktestMetric.MAX_DRAWDOWN.value in comparison_df.columns:
                        sns.barplot(x='strategy', y=BacktestMetric.MAX_DRAWDOWN.value, data=comparison_df)
                        plt.title('Max Drawdown by Strategy')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Bar chart Win Rate
                    plt.subplot(2, 2, 4)
                    if BacktestMetric.WIN_RATE.value in comparison_df.columns:
                        sns.barplot(x='strategy', y=BacktestMetric.WIN_RATE.value, data=comparison_df)
                        plt.title('Win Rate by Strategy')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    plt.tight_layout()
                    
                    # Lưu biểu đồ
                    if save_plot:
                        plt.savefig(self.output_dir / "strategy_comparison.png", dpi=150, bbox_inches='tight')
                    
                    # Hiển thị biểu đồ
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()
                
                except Exception as e:
                    self.logger.error(f"Lỗi khi vẽ biểu đồ so sánh: {e}")
        
        return comparison_df
    
    def optimize_strategy(
        self,
        strategy_name: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        param_grid: Dict[str, List[Any]],
        metric_to_optimize: str = None,
        max_workers: int = None,
        use_walk_forward: bool = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa tham số chiến lược.
        
        Args:
            strategy_name: Tên chiến lược
            data: DataFrame hoặc Dict chứa dữ liệu
            param_grid: Grid tham số cần tối ưu hóa
            metric_to_optimize: Chỉ số cần tối ưu hóa
            max_workers: Số luồng tối đa
            use_walk_forward: Sử dụng phương pháp walk-forward
            verbose: In thông tin chi tiết
            
        Returns:
            Dict kết quả tối ưu hóa
        """
        # Kiểm tra chiến lược
        if strategy_name not in self.strategies:
            self.logger.error(f"Chiến lược '{strategy_name}' chưa được đăng ký")
            return {"status": "error", "message": f"Chiến lược '{strategy_name}' chưa được đăng ký"}
        
        # Thiết lập metric_to_optimize
        if metric_to_optimize is None:
            metric_to_optimize = self.config["optimization"]["metric_to_optimize"]
        
        # Thiết lập max_workers
        if max_workers is None:
            max_workers = self.config["optimization"]["max_workers"]
        
        # Thiết lập use_walk_forward
        if use_walk_forward is None:
            use_walk_forward = self.config["optimization"]["use_walk_forward"]
        
        # Thư mục đầu ra cho tối ưu hóa
        optim_dir = self.output_dir / strategy_name / "optimization"
        optim_dir.mkdir(exist_ok=True, parents=True)
        
        # Kết quả tối ưu hóa
        optim_results = {
            "strategy_name": strategy_name,
            "param_grid": param_grid,
            "metric_to_optimize": metric_to_optimize,
            "use_walk_forward": use_walk_forward,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "best_params": None,
            "best_metric_value": None,
            "symbol_results": {}
        }
        
        # Tiền xử lý dữ liệu
        if isinstance(data, pd.DataFrame):
            # Nếu data là DataFrame, sử dụng nó cho symbol mặc định
            test_data = {"default": data}
        elif isinstance(data, dict):
            # Nếu data là dict, sử dụng các key là symbol
            test_data = data
        else:
            self.logger.error(f"Định dạng dữ liệu không được hỗ trợ: {type(data)}")
            return {"status": "error", "message": f"Định dạng dữ liệu không được hỗ trợ: {type(data)}"}
        
        # Chạy tối ưu hóa cho mỗi symbol
        all_symbol_results = {}
        
        for symbol, symbol_data in test_data.items():
            self.logger.info(f"Tối ưu hóa chiến lược '{strategy_name}' cho {symbol}")
            
            # Lấy thông tin chiến lược
            strategy_info = self.strategies[strategy_name]
            strategy_func = strategy_info["function"]
            
            # Tạo StrategyTester
            tester = StrategyTester(
                strategy_func=strategy_func,
                strategy_name=f"{strategy_name}_{symbol}",
                initial_balance=self.config["backtest"]["initial_balance"],
                fee_rate=self.config["backtest"]["fee_rate"],
                slippage=self.config["backtest"]["slippage"],
                leverage=self.config["backtest"]["leverage"],
                risk_per_trade=self.config["backtest"]["risk_per_trade"],
                allow_short=self.config["backtest"]["allow_short"],
                max_positions=self.config["backtest"]["max_positions"],
                logger=self.logger
            )
            
            # Chạy tối ưu hóa
            try:
                if use_walk_forward:
                    # Sử dụng phương pháp walk-forward
                    window_size = self.config["optimization"]["walk_forward_window"]
                    step_size = self.config["optimization"]["walk_forward_step"]
                    retrain = self.config["optimization"]["retrain_on_each_window"]
                    
                    # Định nghĩa hàm huấn luyện lại
                    def retrain_function(train_data, window_start, train_end):
                        # Tối ưu hóa tham số trên tập huấn luyện
                        optim_result = tester.optimize_parameters(
                            data=train_data,
                            param_grid=param_grid,
                            metric_to_optimize=metric_to_optimize,
                            max_workers=max_workers
                        )
                        
                        # Trả về tham số tối ưu
                        if optim_result["status"] == "success":
                            return {"params": optim_result["best_params"]}
                        else:
                            return {"params": strategy_info["params"]}
                    
                    # Chạy walk-forward test
                    walk_forward_result = tester.run_walk_forward_test(
                        data=symbol_data,
                        window_size=window_size,
                        step_size=step_size,
                        retrain_function=retrain_function if retrain else None,
                        verbose=verbose
                    )
                    
                    # Lưu kết quả
                    all_symbol_results[symbol] = walk_forward_result
                    
                else:
                    # Sử dụng tối ưu hóa thông thường
                    optim_result = tester.optimize_parameters(
                        data=symbol_data,
                        param_grid=param_grid,
                        metric_to_optimize=metric_to_optimize,
                        max_workers=max_workers
                    )
                    
                    # Lưu kết quả
                    all_symbol_results[symbol] = optim_result
                
                # Lưu kết quả vào file
                result_file = optim_dir / f"{symbol}_optimization.pickle"
                with open(result_file, "wb") as f:
                    pickle.dump(all_symbol_results[symbol], f)
                
                # Lưu kết quả dạng JSON nếu có thể
                try:
                    json_file = optim_dir / f"{symbol}_optimization.json"
                    
                    # Tạo dict có thể serialize
                    json_result = {
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "metric_to_optimize": metric_to_optimize,
                        "use_walk_forward": use_walk_forward,
                        "best_params": all_symbol_results[symbol].get("best_params"),
                        "best_metric_value": all_symbol_results[symbol].get("best_metric_value")
                    }
                    
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(json_result, f, indent=4, ensure_ascii=False)
                
                except Exception as e:
                    self.logger.warning(f"Không thể lưu kết quả dạng JSON cho {symbol}: {e}")
                
                self.logger.info(f"Kết quả tối ưu hóa cho {symbol}: {all_symbol_results[symbol].get('best_params')}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tối ưu hóa cho {symbol}: {e}")
                all_symbol_results[symbol] = {"status": "error", "message": str(e)}
        
        # Cập nhật kết quả tối ưu hóa
        optim_results["symbol_results"] = all_symbol_results
        optim_results["end_time"] = datetime.now().isoformat()
        optim_results["status"] = "completed"
        
        # Tìm tham số tốt nhất tổng thể
        best_metric_value = None
        best_params = None
        best_symbol = None
        
        for symbol, result in all_symbol_results.items():
            if result.get("status") == "success" and result.get("best_metric_value") is not None:
                metric_value = result["best_metric_value"]
                
                # Kiểm tra metrics nào là càng nhỏ càng tốt
                if metric_to_optimize in [BacktestMetric.MAX_DRAWDOWN.value]:
                    # Càng nhỏ càng tốt
                    if best_metric_value is None or metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_params = result["best_params"]
                        best_symbol = symbol
                else:
                    # Càng lớn càng tốt
                    if best_metric_value is None or metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_params = result["best_params"]
                        best_symbol = symbol
        
        optim_results["best_params"] = best_params
        optim_results["best_metric_value"] = best_metric_value
        optim_results["best_symbol"] = best_symbol
        
        # Lưu kết quả tổng hợp
        with open(optim_dir / "optimization_results.json", "w", encoding="utf-8") as f:
            # Tạo dict có thể serialize
            json_result = {
                "strategy_name": strategy_name,
                "param_grid": param_grid,
                "metric_to_optimize": metric_to_optimize,
                "use_walk_forward": use_walk_forward,
                "start_time": optim_results["start_time"],
                "end_time": optim_results["end_time"],
                "status": optim_results["status"],
                "best_params": best_params,
                "best_metric_value": best_metric_value,
                "best_symbol": best_symbol
            }
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        # In kết quả tối ưu hóa
        if best_params:
            self.logger.info(f"\nKết quả tối ưu hóa cho chiến lược '{strategy_name}':")
            self.logger.info(f"Tham số tốt nhất: {best_params}")
            self.logger.info(f"Giá trị {metric_to_optimize}: {best_metric_value}")
            self.logger.info(f"Symbol tốt nhất: {best_symbol}")
            
            # Cập nhật tham số tốt nhất cho chiến lược
            self.strategies[strategy_name]["params"].update(best_params)
            self.logger.info(f"Đã cập nhật tham số tốt nhất cho chiến lược '{strategy_name}'")
        
        return optim_results
    
    def plot_equity_curve(
        self,
        results: Dict[str, Any],
        show_plot: bool = True,
        save_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> None:
        """
        Vẽ đường cong equity.
        
        Args:
            results: Kết quả backtest
            show_plot: Hiển thị biểu đồ
            save_plot: Lưu biểu đồ
            figsize: Kích thước biểu đồ
        """
        if not results or results.get("status") != "success":
            self.logger.error("Không có kết quả hợp lệ để vẽ đường cong equity")
            return
        
        # Thiết lập kích thước biểu đồ
        if figsize is None:
            figsize = self.config["visualization"]["plot_figsize"]
        
        # Lấy thông tin cơ bản
        strategy_name = results.get("strategy_name", "Unnamed Strategy")
        
        # Lấy lịch sử equity
        balance_history = results.get("balance_history", {})
        
        if not balance_history or "timestamp" not in balance_history or "equity" not in balance_history:
            self.logger.error("Không có dữ liệu lịch sử equity để vẽ biểu đồ")