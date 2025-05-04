"""
Pipeline xử lý dữ liệu thị trường.
File này cung cấp lớp DataPipeline để quản lý toàn bộ quy trình xử lý dữ liệu,
từ thu thập, làm sạch, đến tạo đặc trưng và chuẩn bị cho huấn luyện.
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
from datetime import datetime, timedelta
import joblib
import concurrent.futures
from functools import partial
import asyncio

# Import các module từ hệ thống
from config.system_config import get_system_config, DATA_DIR, MODEL_DIR
from config.logging_config import setup_logger
from config.constants import Timeframe, Exchange

# Import các module xử lý dữ liệu
from data_processors.cleaners.data_cleaner import DataCleaner
from data_processors.cleaners.outlier_detector import OutlierDetector, OutlierDetectionMethod
from data_processors.cleaners.missing_data_handler import MissingDataHandler, MissingValueMethod

# Import module tạo đặc trưng
from data_processors.feature_engineering.feature_generator import FeatureGenerator

# Import module thu thập dữ liệu nếu cần
try:
    from data_collectors.market_data.historical_data_collector import HistoricalDataCollector, create_data_collector
    from data_collectors.news_collector.sentiment_collector import SentimentCollector
    DATA_COLLECTORS_AVAILABLE = True
except ImportError:
    DATA_COLLECTORS_AVAILABLE = False


class DataPipeline:
    """
    Lớp quản lý quy trình xử lý dữ liệu từ đầu đến cuối.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 4
    ):
        """
        Khởi tạo pipeline xử lý dữ liệu.
        
        Args:
            config: Cấu hình pipeline
            data_dir: Thư mục dữ liệu đầu vào
            output_dir: Thư mục đầu ra kết quả xử lý
            logger: Logger hiện có (nếu có)
            max_workers: Số luồng tối đa cho xử lý song song
        """
        # Thiết lập logger
        if logger is None:
            self.logger = setup_logger("data_pipeline")
        else:
            self.logger = logger
        
        # Thiết lập hệ thống cấu hình
        self.system_config = get_system_config()
        
        # Thiết lập cấu hình mặc định nếu không được cung cấp
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Thiết lập thư mục dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR
        else:
            self.data_dir = data_dir
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = self.data_dir / "processed"
        else:
            self.output_dir = output_dir
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập số luồng tối đa
        self.max_workers = max_workers
        
        # Khởi tạo các thành phần xử lý
        self._init_components()
        
        # Lưu trạng thái dữ liệu
        self.data_state = {}
        
        # Lưu các pipeline đã đăng ký
        self.registered_pipelines = {}
        
        self.logger.info("Đã khởi tạo DataPipeline")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình mặc định cho pipeline.
        
        Returns:
            Cấu hình mặc định
        """
        return {
            "data_cleaning": {
                "enabled": True,
                "remove_outliers": False,
                "handle_missing_values": True,
                "handle_leading_nan": True,
                "leading_nan_method": "backfill",
                "min_periods": 5,
                "remove_duplicates": True,
                "outlier_method": OutlierDetectionMethod.Z_SCORE,
                "missing_value_method": MissingValueMethod.INTERPOLATE
            },
            "feature_engineering": {
                "enabled": True,
                "normalize": True,
                "normalize_method": "z-score",
                "feature_selection": True,
                "remove_redundant": True,
                "correlation_threshold": 0.95,
                "max_features": 100
            },
            "target_generation": {
                "enabled": True,
                "price_column": "close",
                "target_types": ["direction", "return", "volatility"],
                "horizons": [1, 3, 5, 10],
                "threshold": 0.001
            },
            "collectors": {
                "enabled": DATA_COLLECTORS_AVAILABLE,
                "default_exchange": self.system_config.get("default_exchange", "binance"),
                "default_symbols": self.system_config.get("trading.default_symbol", "BTC/USDT").split(","),
                "default_timeframe": self.system_config.get("trading.default_timeframe", "1h"),
                "include_sentiment": True
            },
            "output": {
                "format": "parquet",
                "keep_intermediate": False,
                "include_metadata": True
            }
        }
    
    def _init_components(self) -> None:
        """
        Khởi tạo các thành phần xử lý dữ liệu.
        """
        # Khởi tạo DataCleaner
        cleaning_config = self.config.get("data_cleaning", {})

        # Khởi tạo MissingDataHandler để sử dụng trực tiếp
        missing_data_handler_kwargs = cleaning_config.get("missing_data_handler_kwargs", {})
        self.missing_data_handler = MissingDataHandler(
            method=cleaning_config.get("missing_value_method", MissingValueMethod.INTERPOLATE),
            **missing_data_handler_kwargs
        )

        # Khởi tạo DataCleaner
        self.data_cleaner = DataCleaner(
            use_outlier_detection=cleaning_config.get("remove_outliers", True),
            use_missing_data_handler=cleaning_config.get("handle_missing_values", True),
            normalize_method=self.config.get("feature_engineering", {}).get("normalize_method", "z-score"),
            outlier_detector_kwargs={
                "method": cleaning_config.get("outlier_method", OutlierDetectionMethod.Z_SCORE),
                "threshold": cleaning_config.get("outlier_threshold", 10.0)
            },
            missing_data_handler_kwargs=missing_data_handler_kwargs
        )
        
        # Khởi tạo FeatureGenerator
        feature_config = self.config.get("feature_engineering", {})
        self.feature_generator = FeatureGenerator(
            data_dir=self.output_dir / "feature_engineering",
            max_workers=self.max_workers,
            logger=self.logger
        )
        
        # Đăng ký các đặc trưng mặc định
        self.feature_generator.register_default_features(all_indicators=True)
        
        # Khởi tạo các collector nếu được kích hoạt
        self.collectors = {}
        if self.config.get("collectors", {}).get("enabled", False) and DATA_COLLECTORS_AVAILABLE:
            self._init_collectors()
    
    def _init_collectors(self) -> None:
        """
        Khởi tạo các collector dữ liệu nếu cần.
        """
        collector_config = self.config.get("collectors", {})
        default_exchange = collector_config.get("default_exchange", "binance")
        
        try:
            # Khởi tạo collector cho dữ liệu lịch sử
            self.collectors["historical"] = None  # Sẽ được khởi tạo khi cần trong collect_data
            
            # Khởi tạo collector cho dữ liệu tâm lý
            if collector_config.get("include_sentiment", True):
                self.collectors["sentiment"] = SentimentCollector(
                    data_dir=self.data_dir / "sentiment"
                )
            
            self.logger.info(f"Đã khởi tạo các collector dữ liệu cho {default_exchange}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo collector: {str(e)}")
            self.config["collectors"]["enabled"] = False
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Lưu cấu hình pipeline vào file.
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        if config_path is None:
            config_path = self.output_dir / "pipeline_config.json"
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu cấu hình pipeline vào {config_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cấu hình: {str(e)}")
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Tải cấu hình pipeline từ file.
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        config_path = Path(config_path)
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            
            # Khởi tạo lại các thành phần với cấu hình mới
            self._init_components()
            
            self.logger.info(f"Đã tải cấu hình pipeline từ {config_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
    
    async def collect_data(
        self,
        exchange_id: str = None,
        symbols: List[str] = None,
        timeframe: str = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_sentiment: Optional[bool] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        is_futures: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu từ các nguồn.
        
        Args:
            exchange_id: ID sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            include_sentiment: Bao gồm dữ liệu tâm lý
            api_key: API key (nếu cần)
            api_secret: API secret (nếu cần)
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        if not self.config.get("collectors", {}).get("enabled", False):
            self.logger.warning("Collector không được kích hoạt. Không thể thu thập dữ liệu.")
            return {}
        
        # Lấy giá trị mặc định từ cấu hình nếu không được cung cấp
        collector_config = self.config.get("collectors", {})
        
        if exchange_id is None:
            exchange_id = collector_config.get("default_exchange", "binance")
        
        if symbols is None:
            symbols = collector_config.get("default_symbols", ["BTC/USDT"])
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]
        
        if timeframe is None:
            timeframe = collector_config.get("default_timeframe", "1h")
        
        if include_sentiment is None:
            include_sentiment = collector_config.get("include_sentiment", True)
        
        # Thiết lập thời gian mặc định nếu không được chỉ định
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày
            lookback_days = collector_config.get("lookback_days", 30)
            start_time = end_time - timedelta(days=lookback_days)
        
        self.logger.info(f"Thu thập dữ liệu từ {exchange_id} cho {len(symbols)} cặp giao dịch, khung thời gian {timeframe}")
        
        results = {}
        
        try:
            # Khởi tạo collector cho dữ liệu lịch sử nếu chưa có
            if self.collectors.get("historical") is None:
                historical_collector = await create_data_collector(
                    exchange_id=exchange_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=False,
                    max_workers=self.max_workers,
                    is_futures=is_futures
                )
                self.collectors["historical"] = historical_collector
            else:
                historical_collector = self.collectors["historical"]
            
            # Thu thập dữ liệu OHLCV
            market_data = await historical_collector.collect_all_symbols_ohlcv(
                symbols=symbols,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                concurrency=min(len(symbols), self.max_workers)
            )
            
            # Lưu kết quả
            for symbol, df in market_data.items():
                if not df.empty:
                    results[symbol] = df
                    self.logger.info(f"Đã thu thập {len(df)} dòng dữ liệu cho {symbol}")
                else:
                    self.logger.warning(f"Không có dữ liệu cho {symbol}")
            
            # Thu thập dữ liệu tâm lý nếu cần
            if include_sentiment and "sentiment" in self.collectors:
                sentiment_collector = self.collectors["sentiment"]
                
                # Tách mã tài sản từ symbol đầu tiên (ví dụ: "BTC/USDT" -> "BTC")
                if symbols:
                    asset = symbols[0].split('/')[0]
                    
                    # Thu thập dữ liệu tâm lý
                    sentiment_data = await sentiment_collector.collect_from_all_sources(asset=asset)
                    
                    # Chuyển sang DataFrame và lưu
                    if any(sentiment_data.values()):
                        all_sentiment = []
                        for source_name, source_data in sentiment_data.items():
                            if source_data:
                                all_sentiment.extend(source_data)
                        
                        if all_sentiment:
                            sentiment_df = sentiment_collector.convert_to_dataframe(all_sentiment)
                            results["sentiment"] = sentiment_df
                            self.logger.info(f"Đã thu thập {len(sentiment_df)} dòng dữ liệu tâm lý")
            
            # Cập nhật trạng thái dữ liệu
            self.data_state = {
                "exchange": exchange_id,
                "symbols": symbols,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "record_counts": {symbol: len(df) for symbol, df in results.items()}
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}")
            return {}
        finally:
            # Đóng các collector nếu cần
            try:
                if historical_collector:
                    await historical_collector.exchange_connector.close()
            except:
                pass
    
    def load_data(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        file_format: Optional[str] = None,
        symbol_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Tải dữ liệu từ file.
        
        Args:
            file_paths: Đường dẫn file hoặc danh sách đường dẫn
            file_format: Định dạng file ('csv', 'parquet', 'json')
            symbol_mapping: Ánh xạ từ tên file sang symbol
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        results = {}
        
        # Chuyển đổi sang danh sách
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        # Chuyển đổi sang Path
        file_paths = [Path(path) for path in file_paths]
        
        for file_path in file_paths:
            try:
                # Xác định định dạng file nếu không được cung cấp
                if file_format is None:
                    suffix = file_path.suffix.lower()
                    if suffix == '.csv':
                        file_format = 'csv'
                    elif suffix == '.parquet':
                        file_format = 'parquet'
                    elif suffix == '.json':
                        file_format = 'json'
                    else:
                        self.logger.error(f"Không thể xác định định dạng file: {file_path}")
                        continue
                
                # Tải dữ liệu
                if file_format == 'csv':
                    df = pd.read_csv(file_path)
                    # Chuyển cột timestamp sang datetime nếu có
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                elif file_format == 'parquet':
                    df = pd.read_parquet(file_path)
                elif file_format == 'json':
                    df = pd.read_json(file_path)
                else:
                    self.logger.error(f"Định dạng file không được hỗ trợ: {file_format}")
                    continue
                
                # Xác định symbol
                symbol = None
                
                # Sử dụng symbol_mapping nếu có
                if symbol_mapping and file_path.stem in symbol_mapping:
                    symbol = symbol_mapping[file_path.stem]
                else:
                    # Thử tìm symbol từ tên file
                    parts = file_path.stem.split('_')
                    if len(parts) > 0:
                        # Ưu tiên tìm kiếm các cặp phổ biến
                        common_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
                        for pair in common_pairs:
                            base, quote = pair.split('/')
                            if base.lower() in parts[0].lower():
                                symbol = pair
                                break
                        
                        # Nếu không tìm thấy, sử dụng phần đầu tiên
                        if symbol is None:
                            symbol = parts[0].upper()
                
                # Nếu vẫn không xác định được, sử dụng tên file
                if symbol is None:
                    symbol = file_path.stem
                
                results[symbol] = df
                self.logger.info(f"Đã tải {len(df)} dòng dữ liệu từ {file_path} cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tải file {file_path}: {str(e)}")
        
        return results
    
    def clean_data(
        self,
        data: Dict[str, pd.DataFrame],
        clean_ohlcv: bool = True,
        clean_orderbook: bool = False,
        clean_trades: bool = False,
        clean_sentiment: bool = True,
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Thêm tham số mới
        handle_leading_nan: bool = True,
        leading_nan_method: str = 'backfill',
        min_periods: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Làm sạch dữ liệu.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            clean_ohlcv: Làm sạch dữ liệu OHLCV
            clean_orderbook: Làm sạch dữ liệu orderbook
            clean_trades: Làm sạch dữ liệu giao dịch
            clean_sentiment: Làm sạch dữ liệu tâm lý
            configs: Cấu hình tùy chỉnh cho mỗi loại dữ liệu
            handle_leading_nan: Xử lý NaN ở đầu dữ liệu
            leading_nan_method: Phương pháp xử lý NaN ở đầu ('backfill', 'zero', 'mean', 'median')
            min_periods: Số lượng giá trị tối thiểu để tính giá trị thay thế
            
        Returns:
            Dict với key là symbol và value là DataFrame đã làm sạch
        """
        if not self.config.get("data_cleaning", {}).get("enabled", True):
            self.logger.info("Bỏ qua bước làm sạch dữ liệu (đã bị tắt trong cấu hình)")
            return data
        
        # Cấu hình mặc định
        default_configs = {
            "ohlcv": {
                "handle_gaps": True,
                "handle_negative_values": True,
                "verify_high_low": True,
                "verify_open_close": True,
                "flag_outliers_only": False
            },
            "orderbook": {
                "verify_price_levels": True,
                "verify_quantities": True,
                "flag_outliers_only": False,
                "min_quantity_threshold": 0.0
            },
            "trades": {
                "verify_price": True,
                "verify_amount": True,
                "flag_outliers_only": False,
                "min_quantity_threshold": 0.0
            },
            "sentiment": {
                "remove_outliers": True,
                "handle_missing_values": True,
                "normalize": True
            }
        }
        
        # Cập nhật với cấu hình tùy chỉnh nếu có
        if configs:
            for key, config in configs.items():
                if key in default_configs:
                    default_configs[key].update(config)
        
        results = {}
        
        for symbol, df in data.items():
            try:
                # Trước tiên xử lý giá trị NaN ở đầu nếu được yêu cầu
                if handle_leading_nan:
                    df = self.missing_data_handler.handle_leading_nan(
                        df,
                        columns=None,
                        method=leading_nan_method,
                        min_periods=min_periods
                    )
                    self.logger.info(f"Đã xử lý giá trị NaN ở đầu cho {symbol} bằng phương pháp {leading_nan_method}")

                # Xác định loại dữ liệu
                if symbol.lower() == "sentiment" and clean_sentiment:
                    # Làm sạch dữ liệu tâm lý
                    config = default_configs["sentiment"]
                    cleaned_df = self.data_cleaner.clean_dataframe(
                        df,
                        drop_na=config.get("drop_na", False),
                        normalize=config.get("normalize", True),
                        remove_outliers=config.get("remove_outliers", True),
                        handle_missing=config.get("handle_missing_values", True),
                        remove_duplicates=True
                    )
                    
                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu tâm lý")
                    
                elif all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']) and clean_ohlcv:
                    # Làm sạch dữ liệu OHLCV
                    config = default_configs["ohlcv"]
                    cleaned_df = self.data_cleaner.clean_ohlcv_data(
                        df,
                        handle_gaps=config.get("handle_gaps", True),
                        handle_negative_values=config.get("handle_negative_values", True),
                        verify_high_low=config.get("verify_high_low", True),
                        verify_open_close=config.get("verify_open_close", True),
                        flag_outliers_only=config.get("flag_outliers_only", False)
                    )
                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu OHLCV cho {symbol}")
                    
                elif all(col in df.columns for col in ['price', 'amount', 'side']) and clean_orderbook:
                    # Làm sạch dữ liệu orderbook
                    config = default_configs["orderbook"]
                    cleaned_df = self.data_cleaner.clean_orderbook_data(
                        df,
                        verify_price_levels=config.get("verify_price_levels", True),
                        verify_quantities=config.get("verify_quantities", True),
                        flag_outliers_only=config.get("flag_outliers_only", False),
                        min_quantity_threshold=config.get("min_quantity_threshold", 0.0)
                    )
                    
                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu orderbook cho {symbol}")
                    
                elif all(col in df.columns for col in ['price', 'amount']) and clean_trades:
                    # Làm sạch dữ liệu giao dịch
                    config = default_configs["trades"]
                    cleaned_df = self.data_cleaner.clean_trade_data(
                        df,
                        verify_price=config.get("verify_price", True),
                        verify_amount=config.get("verify_amount", True),
                        flag_outliers_only=config.get("flag_outliers_only", False),
                        min_quantity_threshold=config.get("min_quantity_threshold", 0.0)
                    )
                    
                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu giao dịch cho {symbol}")
                    
                else:
                    # Làm sạch dữ liệu chung
                    cleaned_df = self.data_cleaner.clean_dataframe(
                        df,
                        drop_na=False,
                        normalize=False,
                        remove_outliers=self.config.get("data_cleaning", {}).get("remove_outliers", True),
                        handle_missing=self.config.get("data_cleaning", {}).get("handle_missing_values", True),
                        remove_duplicates=True
                    )
                    
                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu chung cho {symbol}")
                
                results[symbol] = cleaned_df
                
            except Exception as e:
                self.logger.error(f"Lỗi khi làm sạch dữ liệu cho {symbol}: {str(e)}")
                # Trả về dữ liệu gốc nếu có lỗi
                results[symbol] = df
    
        return results
    
    def generate_features(
        self,
        data: Dict[str, pd.DataFrame],
        feature_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_pipeline: Optional[str] = None,
        fit_pipeline: bool = True,
        all_indicators: bool = False  # Thêm tham số này
    ) -> Dict[str, pd.DataFrame]:
        """
        Tạo đặc trưng cho dữ liệu.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            feature_configs: Cấu hình đặc trưng cho mỗi symbol
            use_pipeline: Tên pipeline có sẵn để sử dụng
            fit_pipeline: Học pipeline mới hay không
            all_indicators: Sử dụng tất cả các chỉ báo kỹ thuật có sẵn  # Thêm mô tả này
            
        Returns:
            Dict với key là symbol và value là DataFrame có đặc trưng
        """
        if not self.config.get("feature_engineering", {}).get("enabled", True):
            self.logger.info("Bỏ qua bước tạo đặc trưng (đã bị tắt trong cấu hình)")
            return data
        
        results = {}
        
        # Đảm bảo đăng ký chỉ báo dựa trên all_indicators
        if all_indicators:
            self.feature_generator.register_default_features(all_indicators=True)
        else:
            # Nếu không có yêu cầu all_indicators, 
            # không đăng ký lại các chỉ báo kỹ thuật vì đã được đăng ký lúc khởi tạo
            pass

        # Nếu sử dụng pipeline có sẵn
        if use_pipeline and use_pipeline in self.registered_pipelines:
            self.logger.info(f"Sử dụng pipeline '{use_pipeline}' để tạo đặc trưng")
            pipeline_config = self.registered_pipelines[use_pipeline]
            
            # Tạo đặc trưng cho từng DataFrame
            for symbol, df in data.items():
                try:
                    # Bỏ qua dữ liệu tâm lý
                    if symbol.lower() == "sentiment":
                        results[symbol] = df
                        continue
                    
                    # Kiểm tra yêu cầu tối thiểu
                    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        self.logger.warning(f"Bỏ qua tạo đặc trưng cho {symbol}: Thiếu các cột OHLC cần thiết")
                        results[symbol] = df
                        continue
                    
                    # Áp dụng pipeline
                    features_df = self.feature_generator.transform_data(
                        df, 
                        pipeline_name=pipeline_config.get("pipeline_name"),
                        fit=fit_pipeline
                    )
                    
                    results[symbol] = features_df
                    self.logger.info(f"Đã tạo đặc trưng cho {symbol}: {df.shape[1]} -> {features_df.shape[1]} cột")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi tạo đặc trưng cho {symbol}: {str(e)}")
                    results[symbol] = df
            
            return results
        
        # Tạo đặc trưng với cấu hình riêng cho mỗi symbol
        for symbol, df in data.items():
            try:
                # Bỏ qua dữ liệu tâm lý
                if symbol.lower() == "sentiment":
                    results[symbol] = df
                    continue
                
                # Lấy cấu hình cho symbol này
                symbol_config = feature_configs.get(symbol, {}) if feature_configs else {}
                
                # Kiểm tra yêu cầu tối thiểu
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    self.logger.warning(f"Bỏ qua tạo đặc trưng cho {symbol}: Thiếu các cột OHLC cần thiết")
                    results[symbol] = df
                    continue
                
                # Tạo danh sách đặc trưng cần tính
                feature_names = symbol_config.get("feature_names")
                
                # Lấy danh sách bộ tiền xử lý
                preprocessor_names = symbol_config.get("preprocessor_names", [])
                if not preprocessor_names and self.config.get("feature_engineering", {}).get("normalize", True):
                    # Thêm normalize làm mặc định nếu được kích hoạt
                    preprocessor_names = ["normalize"]
                
                # Lấy danh sách biến đổi
                transformer_names = symbol_config.get("transformer_names")
                
                # Lấy tên bộ chọn lọc đặc trưng
                feature_selector = symbol_config.get("feature_selector")
                if feature_selector is None and self.config.get("feature_engineering", {}).get("feature_selection", True):
                    # Thêm bộ chọn lọc mặc định nếu được kích hoạt
                    feature_selector = "statistical_correlation"
                
                # Tạo pipeline và tính toán đặc trưng
                pipeline_name = f"{symbol}_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.feature_generator.create_feature_pipeline(
                    feature_names=feature_names,
                    preprocessor_names=preprocessor_names,
                    transformer_names=transformer_names,
                    feature_selector=feature_selector,
                    save_pipeline=True,
                    pipeline_name=pipeline_name
                )
                
                # Áp dụng pipeline
                features_df = self.feature_generator.transform_data(df, pipeline_name=pipeline_name, fit=fit_pipeline)
                
                # Lưu pipeline đã đăng ký
                self.registered_pipelines[pipeline_name] = {
                    "pipeline_name": pipeline_name,
                    "symbol": symbol,
                    "created_at": datetime.now().isoformat(),
                    "feature_count": features_df.shape[1],
                    "config": {
                        "feature_names": feature_names,
                        "preprocessor_names": preprocessor_names,
                        "transformer_names": transformer_names,
                        "feature_selector": feature_selector
                    }
                }
                
                results[symbol] = features_df
                self.logger.info(f"Đã tạo đặc trưng cho {symbol}: {df.shape[1]} -> {features_df.shape[1]} cột")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo đặc trưng cho {symbol}: {str(e)}")
                results[symbol] = df
        
        return results
    
    def merge_sentiment_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[pd.DataFrame] = None,
        method: str = 'last_value',
        window: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """
        Kết hợp dữ liệu thị trường với dữ liệu tâm lý.
        
        Args:
            market_data: Dict với key là symbol và value là DataFrame thị trường
            sentiment_data: DataFrame dữ liệu tâm lý
            method: Phương pháp kết hợp ('last_value', 'mean', 'weighted_mean')
            window: Kích thước cửa sổ thời gian ('1D', '12H', '4H', ...)
            
        Returns:
            Dict với key là symbol và value là DataFrame đã kết hợp
        """
        # Nếu không có dữ liệu tâm lý
        if sentiment_data is None:
            if "sentiment" in market_data:
                sentiment_data = market_data["sentiment"]
                # Xóa khỏi market_data
                del market_data["sentiment"]
            else:
                return market_data
        
        # Nếu vẫn không có dữ liệu tâm lý
        if sentiment_data is None or sentiment_data.empty:
            return market_data
        
        results = {}
        
        # Chuyển timestamp sang datetime nếu cần
        if 'timestamp' in sentiment_data.columns and not pd.api.types.is_datetime64_any_dtype(sentiment_data['timestamp']):
            sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
        
        for symbol, df in market_data.items():
            try:
                # Chuyển timestamp sang datetime nếu cần
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Tạo bản sao để tránh thay đổi dữ liệu gốc
                merged_df = df.copy()
                
                # Lọc dữ liệu tâm lý cho asset tương ứng
                asset = symbol.split('/')[0] if '/' in symbol else symbol
                filtered_sentiment = sentiment_data[
                    (sentiment_data['asset'] == asset) | 
                    (sentiment_data['asset'].isna())
                ].copy() if 'asset' in sentiment_data.columns else sentiment_data.copy()
                
                if filtered_sentiment.empty:
                    self.logger.warning(f"Không có dữ liệu tâm lý cho {asset}")
                    results[symbol] = merged_df
                    continue
                
                # Đặt timestamp làm index
                if 'timestamp' in merged_df.columns:
                    merged_df.set_index('timestamp', inplace=True)
                
                if 'timestamp' in filtered_sentiment.columns:
                    filtered_sentiment.set_index('timestamp', inplace=True)
                
                # Tính giá trị tâm lý theo phương pháp đã chọn
                if method == 'last_value':
                    # Lấy giá trị cuối cùng trong cửa sổ
                    resampled_sentiment = filtered_sentiment['value'].resample(window).last()
                elif method == 'mean':
                    # Lấy giá trị trung bình trong cửa sổ
                    resampled_sentiment = filtered_sentiment['value'].resample(window).mean()
                elif method == 'weighted_mean':
                    # Lấy giá trị trung bình có trọng số (mới hơn có trọng số cao hơn)
                    # Giả sử có cột 'weight' hoặc tính toán trọng số dựa trên thời gian
                    if 'weight' in filtered_sentiment.columns:
                        weights = filtered_sentiment['weight']
                    else:
                        # Tính trọng số dựa trên thời gian (mới hơn có trọng số cao hơn)
                        max_time = filtered_sentiment.index.max()
                        filtered_sentiment['weight'] = (filtered_sentiment.index - filtered_sentiment.index.min()).total_seconds() / (max_time - filtered_sentiment.index.min()).total_seconds()
                        weights = filtered_sentiment['weight']
                    
                    # Tính trung bình có trọng số
                    weighted_values = filtered_sentiment['value'] * weights
                    resampled_sentiment = (weighted_values.resample(window).sum() / weights.resample(window).sum()).fillna(method='ffill')
                else:
                    self.logger.warning(f"Phương pháp không hợp lệ: {method}, sử dụng 'last_value'")
                    resampled_sentiment = filtered_sentiment['value'].resample(window).last()
                
                # Đổi tên để tránh xung đột
                resampled_sentiment.name = 'sentiment_value'
                
                # Kết hợp vào dữ liệu thị trường
                merged_df = merged_df.join(resampled_sentiment, how='left')
                
                # Điền các giá trị thiếu bằng forward fill và backward fill
                merged_df['sentiment_value'] = merged_df['sentiment_value'].fillna(method='ffill').fillna(method='bfill')
                
                # Đặt lại index thành cột
                merged_df.reset_index(inplace=True)
                
                results[symbol] = merged_df
                self.logger.info(f"Đã kết hợp dữ liệu tâm lý cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi kết hợp dữ liệu tâm lý cho {symbol}: {str(e)}")
                results[symbol] = df
        
        return results
    
    def remove_redundant_indicators(
        self,
        data: Dict[str, pd.DataFrame],
        correlation_threshold: float = 0.95,
        redundant_groups: Optional[List[List[str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Loại bỏ các chỉ báo kỹ thuật trùng lặp thông tin.
    
        Args:
            data: Dict với key là symbol và value là DataFrame
            correlation_threshold: Ngưỡng tương quan để xác định đặc trưng trùng lặp
            redundant_groups: Danh sách các nhóm đặc trưng đã biết là trùng lặp
        
        Returns:
            Dict với key là symbol và value là DataFrame đã loại bỏ chỉ báo trùng lặp
        """
        results = {}
    
        # Thiết lập nhóm mặc định nếu không cung cấp
        if redundant_groups is None:
            redundant_groups = [
                ['macd_line', 'macd_signal', 'macd_histogram'],
                ['atr_14', 'atr_pct_14', 'atr_norm_14'],
                ['bb_middle_20', 'sma_20']  # bb_middle_20 thường là SMA của close
            ]
    
        for symbol, df in data.items():
            self.logger.info(f"Loại bỏ các chỉ báo trùng lặp cho {symbol}")
            try:
                # Kiểm tra xem MissingDataHandler đã được khởi tạo chưa
                if hasattr(self, 'missing_data_handler'):
                    # Sử dụng phương thức từ MissingDataHandler
                    results[symbol] = self.missing_data_handler.remove_redundant_features(
                        df, 
                        correlation_threshold=correlation_threshold,
                        redundant_groups=redundant_groups
                    )
                
                    # Ghi log số cột đã loại bỏ
                    removed_count = len(df.columns) - len(results[symbol].columns)
                    if removed_count > 0:
                        self.logger.info(f"Đã loại bỏ {removed_count} chỉ báo trùng lặp cho {symbol}")
                    else:
                        self.logger.info(f"Không tìm thấy chỉ báo trùng lặp cho {symbol}")
                else:
                    # Nếu không có MissingDataHandler, trả về dữ liệu gốc
                    results[symbol] = df
                    self.logger.warning("MissingDataHandler không được khởi tạo, không thể loại bỏ chỉ báo trùng lặp")
            except Exception as e:
                self.logger.error(f"Lỗi khi loại bỏ chỉ báo trùng lặp cho {symbol}: {str(e)}")
                results[symbol] = df
    
        return results
    
    def create_target_features(
        self,
        data: Dict[str, pd.DataFrame],
        price_column: str = 'close',
        target_types: List[str] = ['direction', 'return'],
        horizons: List[int] = [1, 3, 5],
        threshold: float = 0.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Tạo các cột mục tiêu cho huấn luyện có giám sát.
    
        Args:
            data: Dict với key là symbol và value là DataFrame
            price_column: Tên cột giá sử dụng làm cơ sở
            target_types: Loại mục tiêu ('direction', 'return', 'volatility')
            horizons: Các khung thời gian tương lai
            threshold: Ngưỡng cho target direction
        
        Returns:
            Dict với key là symbol và value là DataFrame có cột mục tiêu
        """
        results = {}
    
        for symbol, df in data.items():
            self.logger.info(f"Tạo cột mục tiêu cho {symbol}")
            try:
                # Bỏ qua dữ liệu tâm lý
                if symbol.lower() == "sentiment":
                    results[symbol] = df
                    continue
                
                # Kiểm tra xem cột giá có trong DataFrame không
                if price_column not in df.columns:
                    self.logger.warning(f"Không tìm thấy cột giá {price_column} cho {symbol}, bỏ qua tạo mục tiêu")
                    results[symbol] = df
                    continue
                
                # Kiểm tra xem MissingDataHandler đã được khởi tạo chưa
                if hasattr(self, 'missing_data_handler'):
                    # Sử dụng phương thức từ MissingDataHandler
                    results[symbol] = self.missing_data_handler.create_target_columns(
                        df,
                        price_column=price_column,
                        target_types=target_types,
                        horizons=horizons,
                        threshold=threshold
                    )
                
                    # Ghi log số cột mục tiêu đã tạo
                    new_cols = set(results[symbol].columns) - set(df.columns)
                    self.logger.info(f"Đã tạo {len(new_cols)} cột mục tiêu cho {symbol}: {', '.join(new_cols)}")
                else:
                    # Nếu không có MissingDataHandler, trả về dữ liệu gốc
                    results[symbol] = df
                    self.logger.warning("MissingDataHandler không được khởi tạo, không thể tạo cột mục tiêu")
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo cột mục tiêu cho {symbol}: {str(e)}")
                results[symbol] = df
    
        return results

    def prepare_training_data(
        self,
        data: Dict[str, pd.DataFrame],
        target_column: str = 'close',
        price_delta_periods: int = 1,
        normalize_targets: bool = True,
        train_test_split: float = 0.8,
        include_timestamp: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Chuẩn bị dữ liệu cho huấn luyện.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            target_column: Cột mục tiêu
            price_delta_periods: Số kỳ cho delta giá
            normalize_targets: Chuẩn hóa giá trị mục tiêu
            train_test_split: Tỷ lệ chia tập train/test
            include_timestamp: Bao gồm timestamp trong đầu ra
            
        Returns:
            Dict với key là symbol và value là Dict với 'train' và 'test' DataFrames
        """
        results = {}
        
        for symbol, df in data.items():
            try:
                # Kiểm tra cột mục tiêu
                if target_column not in df.columns:
                    self.logger.warning(f"Bỏ qua chuẩn bị dữ liệu cho {symbol}: Không tìm thấy cột mục tiêu {target_column}")
                    continue
                
                # Tạo bản sao
                prepared_df = df.copy()
                
                # Tạo cột mục tiêu - delta giá
                prepared_df[f'{target_column}_delta'] = prepared_df[target_column].pct_change(periods=price_delta_periods)
                
                # Tạo cột hướng giá (1: tăng, 0: giảm/không đổi)
                prepared_df[f'{target_column}_direction'] = (prepared_df[f'{target_column}_delta'] > 0).astype(int)
                
                # Loại bỏ các dòng có NA do tính delta
                prepared_df = prepared_df.dropna(subset=[f'{target_column}_delta'])
                
                # Loại bỏ cột timestamp nếu không cần
                if not include_timestamp and 'timestamp' in prepared_df.columns:
                    timestamp_values = prepared_df['timestamp']
                    prepared_df = prepared_df.drop('timestamp', axis=1)
                
                # Chuẩn hóa giá trị mục tiêu nếu cần
                if normalize_targets:
                    mean = prepared_df[f'{target_column}_delta'].mean()
                    std = prepared_df[f'{target_column}_delta'].std()
                    
                    if std > 0:
                        prepared_df[f'{target_column}_delta_norm'] = (prepared_df[f'{target_column}_delta'] - mean) / std
                    else:
                        prepared_df[f'{target_column}_delta_norm'] = prepared_df[f'{target_column}_delta'] - mean
                
                # Chia tập huấn luyện/kiểm thử
                train_size = int(len(prepared_df) * train_test_split)
                train_df = prepared_df.iloc[:train_size]
                test_df = prepared_df.iloc[train_size:]
                
                # Thêm lại timestamp nếu đã loại bỏ
                if not include_timestamp and 'timestamp' in df.columns:
                    train_df.insert(0, 'timestamp', timestamp_values[:train_size].values)
                    test_df.insert(0, 'timestamp', timestamp_values[train_size:].values)
                
                results[symbol] = {
                    'train': train_df,
                    'test': test_df,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'feature_count': train_df.shape[1]
                }
                
                self.logger.info(f"Đã chuẩn bị dữ liệu huấn luyện cho {symbol}: {len(train_df)} mẫu train, {len(test_df)} mẫu test")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi chuẩn bị dữ liệu huấn luyện cho {symbol}: {str(e)}")
        
        return results
    
    def save_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
        file_format: str = 'parquet',
        include_metadata: bool = True
    ) -> Dict[str, str]:
        """
        Lưu dữ liệu đã xử lý.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            output_dir: Thư mục đầu ra
            file_format: Định dạng file ('csv', 'parquet', 'json')
            include_metadata: Bao gồm metadata
            
        Returns:
            Dict với key là symbol và value là đường dẫn file
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lấy định dạng từ cấu hình nếu không được chỉ định
        if file_format is None:
            file_format = self.config.get("output", {}).get("format", "parquet")
        
        # Kiểm tra định dạng
        if file_format not in ['csv', 'parquet', 'json']:
            self.logger.warning(f"Định dạng không được hỗ trợ: {file_format}, sử dụng 'parquet'")
            file_format = 'parquet'
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for symbol, df in data.items():
            try:
                # Chuyển đổi các cột string thành float nếu có thể
                df_prepared = df.copy()
                for col in df_prepared.columns:
                    if df_prepared[col].dtype == 'object' or df_prepared[col].dtype == 'string':
                        try:
                            # Thử chuyển đổi sang float
                            df_prepared[col] = df_prepared[col].astype(float)
                            self.logger.info(f"Đã chuyển đổi cột {col} từ {df[col].dtype} sang float")
                        except (ValueError, TypeError):
                            # Giữ nguyên nếu không thể chuyển đổi
                            self.logger.debug(f"Không thể chuyển đổi cột {col} sang float")

                # Tạo tên file
                filename = f"{symbol.replace('/', '_').lower()}_{timestamp}"
                file_path = output_dir / f"{filename}.{file_format}"
                
                # Lưu dữ liệu
                if file_format == 'csv':
                    df_prepared.to_csv(file_path, index=False)
                elif file_format == 'parquet':
                    df_prepared.to_parquet(file_path, index=False)
                elif file_format == 'json':
                    df_prepared.to_json(file_path, orient='records', date_format='iso')
                
                results[symbol] = str(file_path)
                self.logger.info(f"Đã lưu {len(df)} dòng dữ liệu cho {symbol} vào {file_path}")
                
                # Lưu metadata nếu cần
                if include_metadata:
                    metadata = {
                        "symbol": symbol,
                        "rows": len(df),
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(df[col].dtype) for col in df.columns},
                        "saved_at": datetime.now().isoformat(),
                        "file_format": file_format,
                        "file_path": str(file_path),
                        "data_stats": {
                            col: {
                                "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                                "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                                "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                                "null_count": int(df[col].isna().sum())
                            } for col in df.select_dtypes(include=[np.number]).columns
                        }
                    }
                    
                    metadata_path = output_dir / f"{filename}_metadata.json"
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    
                    self.logger.debug(f"Đã lưu metadata cho {symbol} vào {metadata_path}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu dữ liệu cho {symbol}: {str(e)}")
        
        return results
    
    def save_pipeline_state(self, file_path: Optional[Path] = None) -> None:
        """
        Lưu trạng thái của pipeline.
        
        Args:
            file_path: Đường dẫn file trạng thái
        """
        if file_path is None:
            file_path = self.output_dir / "pipeline_state.joblib"
        
        # Lưu trạng thái của FeatureGenerator
        self.feature_generator.save_state()
        
        # Lưu trạng thái của pipeline
        state = {
            "data_state": self.data_state,
            "registered_pipelines": self.registered_pipelines,
            "config": self.config,
            "last_saved": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        try:
            joblib.dump(state, file_path)
            self.logger.info(f"Đã lưu trạng thái pipeline vào {file_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái pipeline: {str(e)}")
    
    def load_pipeline_state(self, file_path: Optional[Path] = None) -> bool:
        """
        Tải trạng thái của pipeline.
        
        Args:
            file_path: Đường dẫn file trạng thái
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if file_path is None:
            file_path = self.output_dir / "pipeline_state.joblib"
        
        if not Path(file_path).exists():
            self.logger.warning(f"File trạng thái không tồn tại: {file_path}")
            return False
        
        try:
            state = joblib.load(file_path)
            
            # Khôi phục trạng thái
            self.data_state = state.get("data_state", {})
            self.registered_pipelines = state.get("registered_pipelines", {})
            self.config = state.get("config", self.config)
            
            # Khôi phục trạng thái của FeatureGenerator
            self.feature_generator.load_state()
            
            self.logger.info(f"Đã tải trạng thái pipeline từ {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái pipeline: {str(e)}")
            return False
    
    def register_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: str = ""
    ) -> None:
        """
        Đăng ký pipeline tùy chỉnh.
        
        Args:
            name: Tên pipeline
            steps: Danh sách các bước xử lý
            description: Mô tả pipeline
        """
        self.registered_pipelines[name] = {
            "pipeline_name": name,
            "steps": steps,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        self.logger.info(f"Đã đăng ký pipeline '{name}' với {len(steps)} bước")
    
    async def run_pipeline(
        self,
        pipeline_name: Optional[str] = None,
        input_data: Optional[Dict[str, pd.DataFrame]] = None,
        input_files: Optional[List[Union[str, Path]]] = None,
        exchange_id: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_dir: Optional[Path] = None,
        save_results: bool = True,
        is_futures: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Chạy pipeline xử lý dữ liệu đầy đủ.
        
        Args:
            pipeline_name: Tên pipeline đã đăng ký
            input_data: Dữ liệu đầu vào
            input_files: Danh sách file đầu vào
            exchange_id: ID sàn giao dịch (dùng để thu thập dữ liệu)
            symbols: Danh sách cặp giao dịch (dùng để thu thập dữ liệu)
            timeframe: Khung thời gian (dùng để thu thập dữ liệu)
            start_time: Thời gian bắt đầu (dùng để thu thập dữ liệu)
            end_time: Thời gian kết thúc (dùng để thu thập dữ liệu)
            output_dir: Thư mục đầu ra
            save_results: Lưu kết quả hay không
            is_futures: Là thị trường hợp đồng tương lai hay không
            
        Returns:
            Dict với key là symbol và value là DataFrame kết quả
        """
        # Lấy pipeline nếu đã đăng ký
        pipeline_steps = []
        if pipeline_name and pipeline_name in self.registered_pipelines:
            pipeline_config = self.registered_pipelines[pipeline_name]
            pipeline_steps = pipeline_config.get("steps", [])
            
            self.logger.info(f"Chạy pipeline '{pipeline_name}' với {len(pipeline_steps)} bước")
        else:
            # Sử dụng các bước mặc định
            pipeline_steps = [
                {"name": "collect_data", "enabled": exchange_id is not None},
                {"name": "load_data", "enabled": input_files is not None},
                {"name": "clean_data", "enabled": True, "params": {
                    "handle_leading_nan": True,
                    "leading_nan_method": "backfill",
                    "min_periods": 5
                }},
                {"name": "generate_features", "enabled": self.config.get("feature_engineering", {}).get("enabled", True)},
                {"name": "remove_redundant_indicators", "enabled": True, "params": {
                    "correlation_threshold": 0.95,
                    "redundant_groups": [
                        ['macd_line', 'macd_signal', 'macd_histogram'],
                        ['atr_14', 'atr_pct_14', 'atr_norm_14'],
                        ['bb_middle_20', 'sma_20', 'bb_upper_20', 'bb_lower_20', 'bb_percent_b_20'],
                        ['plus_di_14', 'minus_di_14', 'adx_14']
                    ]
                }},
                {"name": "create_target_features", "enabled": True, "params": {
                    "price_column": "close",
                    "target_types": ["direction", "return", "volatility"],
                    "horizons": [1, 3, 5, 10],
                    "threshold": 0.001
                }},
                {"name": "merge_sentiment", "enabled": self.config.get("collectors", {}).get("include_sentiment", True)},
                {"name": "prepare_training", "enabled": False},  # Mặc định không chuẩn bị dữ liệu huấn luyện
                {"name": "save_data", "enabled": save_results}
            ]
            
            self.logger.info(f"Chạy pipeline mặc định với {len(pipeline_steps)} bước")
        
        # Kết quả của mỗi bước
        step_results = {}
        
        # Dữ liệu hiện tại
        current_data = input_data or {}
        
        # Thực hiện các bước pipeline
        for step in pipeline_steps:
            step_name = step.get("name", "")
            enabled = step.get("enabled", True)
            step_params = step.get("params", {})
            
            if not enabled:
                self.logger.info(f"Bỏ qua bước '{step_name}' (đã bị tắt)")
                continue
            
            self.logger.info(f"Thực hiện bước '{step_name}'")
            
            try:
                if step_name == "collect_data":
                    # Thu thập dữ liệu
                    collected_data = await self.collect_data(
                        exchange_id=exchange_id,
                        symbols=symbols,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time,
                        is_futures=is_futures,
                        **step_params
                    )
                    
                    if collected_data:
                        current_data = collected_data
                        step_results["collect_data"] = collected_data
                    else:
                        self.logger.warning("Không thu thập được dữ liệu")
                
                elif step_name == "load_data":
                    # Tải dữ liệu từ file
                    loaded_data = self.load_data(
                        file_paths=input_files,
                        **step_params
                    )
                    
                    if loaded_data:
                        current_data = loaded_data
                        step_results["load_data"] = loaded_data
                    else:
                        self.logger.warning("Không tải được dữ liệu từ file")
                
                elif step_name == "clean_data":
                    # Làm sạch dữ liệu
                    cleaned_data = self.clean_data(
                        current_data,
                        handle_leading_nan=step_params.get("handle_leading_nan", True),
                        leading_nan_method=step_params.get("leading_nan_method", "backfill"),
                        min_periods=step_params.get("min_periods", 5),
                        **{k: v for k, v in step_params.items() if k not in ["handle_leading_nan", "leading_nan_method", "min_periods"]}
                    )
                    current_data = cleaned_data
                    step_results["clean_data"] = cleaned_data
                
                elif step_name == "generate_features":
                    # Tạo đặc trưng
                    featured_data = self.generate_features(current_data, **step_params)
                    current_data = featured_data
                    step_results["generate_features"] = featured_data

                elif step_name == "remove_redundant_indicators":
                    # Loại bỏ các chỉ báo trùng lặp
                    pruned_data = self.remove_redundant_indicators(
                        current_data,
                        correlation_threshold=step_params.get("correlation_threshold", 0.95),
                        redundant_groups=step_params.get("redundant_groups", None)
                    )
                    current_data = pruned_data
                    step_results["remove_redundant_indicators"] = pruned_data

                elif step_name == "create_target_features":
                    # Tạo cột mục tiêu cho huấn luyện có giám sát
                    data_with_targets = self.create_target_features(
                        current_data,
                        price_column=step_params.get("price_column", "close"),
                        target_types=step_params.get("target_types", ["direction", "return"]),
                        horizons=step_params.get("horizons", [1, 3, 5]),
                        threshold=step_params.get("threshold", 0.0)
                    )
                    current_data = data_with_targets
                    step_results["create_target_features"] = data_with_targets
                                    
                elif step_name == "merge_sentiment":
                    # Kết hợp dữ liệu tâm lý
                    sentiment_data = step_params.get("sentiment_data", None)
                    if sentiment_data is None and "sentiment" in current_data:
                        sentiment_data = current_data["sentiment"]
                        
                    if sentiment_data is not None:
                        merged_data = self.merge_sentiment_data(
                            current_data,
                            sentiment_data=sentiment_data,
                            **{k: v for k, v in step_params.items() if k != "sentiment_data"}
                        )
                        current_data = merged_data
                        step_results["merge_sentiment"] = merged_data
                    else:
                        self.logger.info("Không có dữ liệu tâm lý để kết hợp")
                
                elif step_name == "prepare_training":
                    # Chuẩn bị dữ liệu huấn luyện
                    training_data = self.prepare_training_data(
                        current_data,
                        **step_params
                    )
                    step_results["prepare_training"] = training_data
                    
                    # Chuyển đổi định dạng dữ liệu sang Dict[str, pd.DataFrame]
                    training_flat = {}
                    for symbol, data_dict in training_data.items():
                        training_flat[f"{symbol}_train"] = data_dict['train']
                        training_flat[f"{symbol}_test"] = data_dict['test']
                    
                    current_data = training_flat
                
                elif step_name == "save_data":
                    # Lưu dữ liệu
                    if output_dir is not None:
                        step_params["output_dir"] = output_dir
                        
                    saved_paths = self.save_data(
                        current_data,
                        **step_params
                    )
                    step_results["save_data"] = saved_paths
                
                else:
                    self.logger.warning(f"Không nhận dạng được bước '{step_name}'")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thực hiện bước '{step_name}': {str(e)}")
        
        # Lưu trạng thái pipeline nếu cần
        if save_results:
            self.save_pipeline_state()
        
        return current_data


async def run_test():
    """
    Hàm chạy thử nghiệm với các chức năng của DataPipeline.
    """
    # Khởi tạo pipeline
    pipeline = DataPipeline()
    
    # Tạo dữ liệu kiểm thử
    test_data = {
        "BTC/USDT": pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1h"),
            "open": np.random.normal(20000, 1000, 100),
            "high": np.random.normal(20500, 1000, 100),
            "low": np.random.normal(19500, 1000, 100),
            "close": np.random.normal(20200, 1000, 100),
            "volume": np.random.normal(100, 30, 100)
        })
    }
        
    # Thêm dữ liệu tâm lý
    test_data["sentiment"] = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="2h"),
        "value": np.random.normal(0, 0.5, 50),
        "label": ["Neutral"] * 50,
        "source": ["Test Source"] * 50,
        "asset": ["BTC"] * 50
    })
    
    # Đăng ký pipeline tùy chỉnh
    pipeline.register_pipeline(
        name="test_pipeline",
        steps=[
            {"name": "clean_data", "enabled": True, "params": {"clean_ohlcv": True}},
            {"name": "generate_features", "enabled": True, "params": {"feature_configs": {
                "BTC/USDT": {
                    "feature_names": ["trend_sma", "trend_ema", "momentum_rsi", "volatility_bbands"]
                }
            }}},
            {"name": "merge_sentiment", "enabled": True, "params": {"method": "last_value"}},
            {"name": "save_data", "enabled": True, "params": {"file_format": "parquet"}}
        ],
        description="Pipeline kiểm thử với các chỉ báo kỹ thuật và dữ liệu tâm lý"
    )
    
    # Chạy pipeline
    print("Đang chạy pipeline kiểm thử...")
    result_data = await pipeline.run_pipeline(
        pipeline_name="test_pipeline",
        input_data=test_data,
        save_results=True
    )
    
    # In thông tin kết quả
    for symbol, df in result_data.items():
        print(f"Kết quả cho {symbol}: {df.shape[0]} dòng, {df.shape[1]} cột")
        print(f"Các cột: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    print("Hoàn thành kiểm thử!")


def main():
    """
    Hàm main để chạy thử nghiệm.
    """
    asyncio.run(run_test())

if __name__ == "__main__":
    # Chạy hàm main
    main()