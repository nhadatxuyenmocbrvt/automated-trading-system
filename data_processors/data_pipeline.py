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
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
import joblib
import concurrent.futures
from functools import partial
import asyncio
import random
import traceback
import inspect

# Import các module từ hệ thống
from config.system_config import get_system_config, DATA_DIR, MODEL_DIR
from config.logging_config import setup_logger
from config.constants import Timeframe, Exchange

# Import các module xử lý dữ liệu
from data_processors.cleaners.data_cleaner import DataCleaner
from data_processors.cleaners.outlier_detector import OutlierDetector, OutlierDetectionMethod
from data_processors.cleaners.missing_data_handler import MissingDataHandler, MissingValueMethod
from data_processors.utils.preprocessing import fill_nan_values, handle_leading_nans

# Import module tạo đặc trưng
from data_processors.feature_engineering.feature_generator import FeatureGenerator
from data_processors.utils.preprocessing import ensure_valid_price_data

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
            aggressive_cleaning=cleaning_config.get("aggressive_nan_handling", True),
            ensure_no_nan=cleaning_config.get("fill_all_nan", True)
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

    def _convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi cột timestamp sang định dạng datetime64[ns] chuẩn.
        
        Args:
            df: DataFrame chứa cột timestamp cần chuyển đổi
            
        Returns:
            DataFrame với cột timestamp đã chuyển đổi
        """
        if 'timestamp' not in df.columns:
            return df
            
        df_copy = df.copy()
        
        # Nếu timestamp đã ở dạng datetime64, không cần xử lý thêm
        if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            return df_copy
            
        # Lưu lại loại dữ liệu ban đầu để debug
        original_type = str(df_copy['timestamp'].dtype)
        original_sample = str(df_copy['timestamp'].iloc[0]) if not df_copy.empty else "N/A"
        
        try:
            # Kiểm tra xem timestamp có phải số không
            if pd.api.types.is_numeric_dtype(df_copy['timestamp']):
                # Thử với đơn vị khác nhau
                timestamp_values = df_copy['timestamp'].astype(float).values
                
                # Kiểm tra phạm vi để xác định đơn vị
                max_ts = np.max(timestamp_values)
                
                # Xác định đơn vị thời gian dựa trên giá trị tối đa
                if max_ts > 1e18:  # nanoseconds
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ns')
                elif max_ts > 1e15:  # microseconds
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='us')
                elif max_ts > 1e12:  # milliseconds
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
                else:  # seconds (giá trị thông thường là ~1.6e9 cho năm 2020)
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='s')
                
                # Kiểm tra kết quả chuyển đổi
                first_year = df_copy['timestamp'].dt.year.iloc[0] if not df_copy.empty else None
                
                # Nếu năm không hợp lệ, thử lại với đơn vị khác
                if first_year is not None and (first_year < 1900 or first_year > 2100):
                    if max_ts > 1e12:  # Đổi từ ms sang s
                        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'].astype(float) / 1000, unit='s')
                    else:  # Đổi từ s sang ms
                        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'].astype(float) * 1000, unit='ms')
            else:
                # Nếu là chuỗi, thử parse trực tiếp
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
                
                # Nếu parse thất bại và có chứa dấu phẩy nghìn
                if df_copy['timestamp'].isna().all() and isinstance(df_copy['timestamp'].iloc[0], str):
                    # Loại bỏ dấu phẩy nghìn nếu là số được format
                    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'].str.replace(',', ''), errors='coerce')
            
            # Kiểm tra lỗi chuyển đổi
            na_count = df_copy['timestamp'].isna().sum()
            if na_count > 0:
                self.logger.warning(f"Có {na_count} giá trị timestamp không thể chuyển đổi. Type ban đầu: {original_type}, Mẫu: {original_sample}")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi timestamp: {str(e)}")
            # Trả về DataFrame gốc nếu có lỗi
            return df

    async def collect_data_improved(
        self,
        exchange_id: str = None,
        symbols: List[str] = None,
        timeframe: str = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_sentiment: Optional[bool] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        is_futures: bool = False,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu từ các nguồn với xử lý lỗi và kiểm soát tốc độ cải tiến.
        
        Args:
            exchange_id: ID sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            include_sentiment: Bao gồm dữ liệu tâm lý
            api_key: API key (nếu cần)
            api_secret: API secret (nếu cần)
            is_futures: Là thị trường hợp đồng tương lai hay không
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ ban đầu giữa các lần thử lại (giây)
            
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
        
        # Tối ưu số lượng workers dựa trên số lượng symbols
        optimal_workers = min(self.max_workers, len(symbols))
        
        # Tạo semaphore để kiểm soát số lượng requests đồng thời
        semaphore = asyncio.Semaphore(optimal_workers)
        
        async def fetch_with_retry(symbol):
            """Hàm thu thập dữ liệu với cơ chế retry thông minh"""
            async with semaphore:  # Đảm bảo không quá nhiều requests đồng thời
                # Thêm delay ngẫu nhiên để tránh đồng thời quá nhiều request
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                attempt = 0
                delay = retry_delay
                days_diff = 0  # Định nghĩa biến days_diff ở đây
                
                while attempt < max_retries:
                    try:
                        if self.collectors.get("historical") is None:
                            # Khởi tạo collector nếu chưa có
                            self.collectors["historical"] = await create_data_collector(
                                exchange_id=exchange_id,
                                api_key=api_key,
                                api_secret=api_secret,
                                testnet=False,
                                max_workers=self.max_workers,
                                is_futures=is_futures
                            )
                        
                        historical_collector = self.collectors["historical"]
                        
                        # Kiểm tra khoảng thời gian thu thập
                        time_range_large = False
                        if start_time and end_time:
                            days_diff = (end_time - start_time).days
                            if days_diff > 30:  # Nếu khoảng thời gian lớn hơn 30 ngày
                                time_range_large = True
                        
                        # Sử dụng fetch_historical_klines nếu khoảng thời gian lớn và là Binance
                        if time_range_large and 'binance' in exchange_id.lower() and hasattr(historical_collector.exchange_connector, 'fetch_historical_klines'):
                            self.logger.info(f"Khoảng thời gian lớn ({days_diff} ngày), sử dụng fetch_historical_klines cho {symbol}")
                            
                            # Chuyển đổi thời gian sang định dạng timestamp (milliseconds)
                            since_ms = int(start_time.timestamp() * 1000) if isinstance(start_time, datetime) else start_time
                            until_ms = int(end_time.timestamp() * 1000) if isinstance(end_time, datetime) else end_time
                            
                            # Thu thập dữ liệu với phân trang KHÔNG SỬ DỤNG AWAIT vì không phải coroutine
                            ohlcv_data = historical_collector.exchange_connector.fetch_historical_klines(
                                symbol=symbol,
                                interval=timeframe,
                                start_time=since_ms,
                                end_time=until_ms,
                                limit=1000  # Binance cho phép tối đa 1000 candles mỗi request
                            )
                        else:
                            # Thu thập OHLCV như cũ nếu khoảng thời gian nhỏ
                            fetch_method = historical_collector.exchange_connector.fetch_ohlcv
                            if inspect.iscoroutinefunction(fetch_method):
                                # Nếu là coroutine function, sử dụng await
                                ohlcv_data = await fetch_method(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    since=int(start_time.timestamp() * 1000) if isinstance(start_time, datetime) else start_time,
                                    limit=None
                                )
                            else:
                                # Nếu không phải coroutine function, gọi trực tiếp
                                ohlcv_data = fetch_method(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    since=int(start_time.timestamp() * 1000) if isinstance(start_time, datetime) else start_time,
                                    limit=None
                                )
                        
                        if not ohlcv_data:
                            self.logger.warning(f"Không có dữ liệu OHLCV cho {symbol}")
                            return symbol, None
                        
                        # Chuyển thành DataFrame
                        df = pd.DataFrame(
                            ohlcv_data,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        
                        # Chuyển timestamp sang datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Đảm bảo tất cả các cột số là float
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = df[col].astype(float)
                        
                        # Thêm thông tin symbol
                        df['symbol'] = symbol
                        
                        # Kiểm tra dữ liệu và log
                        self.logger.info(f"Đã thu thập {len(df)} dòng dữ liệu OHLCV cho {symbol}, "
                                        f"từ {df['timestamp'].min()} đến {df['timestamp'].max()}")
                        
                        return symbol, df
                        
                    except Exception as e:
                        attempt += 1
                        error_msg = str(e)
                        self.logger.warning(f"Lỗi khi thu thập dữ liệu cho {symbol} (lần {attempt}/{max_retries}): {error_msg}")
                        
                        if attempt < max_retries:
                            # Thực hiện retry với backoff
                            await asyncio.sleep(delay)
                            delay *= 2  # Tăng gấp đôi thời gian chờ
                        else:
                            self.logger.error(f"Đã thử {max_retries} lần nhưng không thể thu thập dữ liệu cho {symbol}")
                            return symbol, None
        
        # Thu thập dữ liệu thị trường đồng thời
        tasks = [fetch_with_retry(symbol) for symbol in symbols]
        results = {}
        
        for task in asyncio.as_completed(tasks):
            symbol, df = await task
            if df is not None and not df.empty:
                results[symbol] = df
        
        # Thu thập dữ liệu tâm lý nếu cần
        if include_sentiment and self.collectors.get("sentiment") is not None:
            for symbol in symbols:
                # Trích xuất asset từ symbol (VD: BTC/USDT -> BTC)
                asset = symbol.split('/')[0] if '/' in symbol else symbol
                self.logger.info(f"Thu thập dữ liệu tâm lý cho asset {asset}")
                
                sentiment_data = await self._collect_sentiment_data([symbol], start_time, end_time)
                
                if sentiment_data:
                    # Thêm dữ liệu tâm lý vào kết quả
                    results.update(sentiment_data)
        
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
    
    async def _collect_sentiment_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu tâm lý cho các symbols.
        
        Args:
            symbols: Danh sách cặp giao dịch
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            
        Returns:
            Dict với key là symbol_sentiment và value là DataFrame dữ liệu tâm lý
        """
        sentiment_results = {}
        
        try:
            if not symbols:
                return {}
            
            # Thu thập cho mỗi symbol
            for symbol in symbols:
                # Lấy asset từ symbol (VD: "BTC/USDT" -> "BTC")
                asset = symbol.split('/')[0] if '/' in symbol else symbol
                
                self.logger.info(f"Thu thập dữ liệu tâm lý cho asset {asset}")
                
                # Thu thập dữ liệu tâm lý từ Binance
                from cli.commands.collect_commands import CollectCommands
                
                # Khởi tạo CollectCommands
                cmd = CollectCommands()
                
                # Chuyển datetime sang string format YYYY-MM-DD cho start_date và end_date
                start_date = start_time.strftime("%Y-%m-%d") if start_time else None
                end_date = end_time.strftime("%Y-%m-%d") if end_time else None
                
                # Thu thập dữ liệu tâm lý từ Binance
                binance_sentiment = await cmd.collect_binance_sentiment(
                    exchange_id="binance",
                    symbols=[symbol],  # Chỉ một symbol tại một thời điểm
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=self.data_dir / "sentiment" / "binance",
                    save_format='parquet'
                )
                
                if binance_sentiment and any(binance_sentiment.values()):
                    # Tạo DataFrame từ kết quả
                    for s, file_path in binance_sentiment.items():
                        try:
                            # Đọc file parquet đã lưu
                            sentiment_df = pd.read_parquet(file_path)
                            sentiment_key = f"{asset}_sentiment"
                            sentiment_results[sentiment_key] = sentiment_df
                            self.logger.info(f"Đã thu thập {len(sentiment_df)} dòng dữ liệu tâm lý từ Binance cho {asset}")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi đọc file dữ liệu tâm lý: {str(e)}")
                else:
                    self.logger.warning(f"Không thể thu thập dữ liệu tâm lý từ Binance cho {asset}")
        
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý: {str(e)}")
        
        return sentiment_results
    
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
        is_futures: bool = False,
        preserve_timestamp: bool = True  # Thêm tham số này
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
        # Sử dụng phương thức cải tiến
        return await self.collect_data_improved(
            exchange_id=exchange_id,
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            include_sentiment=include_sentiment,
            api_key=api_key,
            api_secret=api_secret,
            is_futures=is_futures
        )

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
                    # Sử dụng parse_dates để chuyển đổi timestamp ngay khi đọc file
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])
                    
                    # Kiểm tra và xử lý thêm nếu timestamp vẫn là object
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        self.logger.info(f"Cột timestamp vẫn không phải datetime64, thử chuyển đổi thủ công")
                        df = self._convert_timestamp(df)
                    
                elif file_format == 'parquet':
                    df = pd.read_parquet(file_path)
                    # Chuyển đổi timestamp nếu cần
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
                        
                elif file_format == 'json':
                    df = pd.read_json(file_path)
                    # Chuyển đổi timestamp nếu cần
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
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
                            # Xác định nếu là dữ liệu tâm lý
                            if 'sentiment' in file_path.stem.lower():
                                symbol = f"{symbol}_sentiment"
                
                # Nếu vẫn không xác định được, sử dụng tên file
                if symbol is None:
                    symbol = file_path.stem
                
                # Kiểm tra chất lượng dữ liệu
                if df.empty:
                    self.logger.warning(f"DataFrame trống cho {symbol} từ file {file_path}")
                    continue
                
                # Log thông tin về timestamp để debug
                if 'timestamp' in df.columns:
                    self.logger.info(f"Timestamp cho {symbol}: dtype={df['timestamp'].dtype}, "
                                    f"min={df['timestamp'].min()}, max={df['timestamp'].max()}")
                
                results[symbol] = df
                self.logger.info(f"Đã tải {len(df)} dòng dữ liệu từ {file_path} cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tải file {file_path}: {str(e)}")
                traceback.print_exc()
        
        return results
    
    def clean_sentiment_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu tâm lý.
        
        Args:
            sentiment_df: DataFrame dữ liệu tâm lý
            
        Returns:
            DataFrame dữ liệu tâm lý đã làm sạch
        """
        if sentiment_df is None or sentiment_df.empty:
            self.logger.warning("Không có dữ liệu tâm lý để làm sạch")
            return sentiment_df
        
        self.logger.info(f"Bắt đầu làm sạch dữ liệu tâm lý: {len(sentiment_df)} dòng")
        
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df = sentiment_df.copy()
        
        # 1. Xử lý timestamp
        if 'timestamp' in df.columns:
            # Chuyển đổi timestamp sang datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = self._convert_timestamp(df)
                
                # Loại bỏ các dòng có timestamp không hợp lệ
                invalid_timestamp = df['timestamp'].isna().sum()
                if invalid_timestamp > 0:
                    self.logger.warning(f"Loại bỏ {invalid_timestamp} dòng có timestamp không hợp lệ")
                    df = df.dropna(subset=['timestamp'])
        
        # 2. Xác định cột chứa giá trị tâm lý
        value_column = None
        for col in ['value', 'sentiment_value', 'sentiment', 'score', 'fear_greed_value']:
            if col in df.columns:
                value_column = col
                break
        
        if value_column is None:
            self.logger.error("Không tìm thấy cột chứa giá trị tâm lý")
            return df
        
        # 3. Chuyển đổi giá trị sang số
        try:
            # Nếu cột giá trị có định dạng không phải số
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                # Thử chuyển đổi sang số
                df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
                self.logger.info(f"Đã chuyển đổi cột {value_column} sang kiểu số")
        except Exception as e:
            self.logger.warning(f"Không thể chuyển đổi cột {value_column} sang kiểu số: {str(e)}")
        
        from data_processors.utils.preprocessing import handle_extreme_values, min_max_scale
        # 4. Xử lý giá trị ngoại lệ (outliers)
        if pd.api.types.is_numeric_dtype(df[value_column]):
            # Tính Q1, Q3 và IQR
            Q1 = df[value_column].quantile(0.25)
            Q3 = df[value_column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Xác định ngưỡng outlier
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Đếm số lượng outlier
            outliers = ((df[value_column] < lower_bound) | (df[value_column] > upper_bound)).sum()
            
            if outliers > 0:
                self.logger.info(f"Phát hiện {outliers} giá trị ngoại lệ trong cột {value_column}")
                
                # Xử lý outlier bằng cách winsorize (cắt giới hạn)
                df.loc[df[value_column] < lower_bound, value_column] = lower_bound
                df.loc[df[value_column] > upper_bound, value_column] = upper_bound
                self.logger.info(f"Đã xử lý giá trị ngoại lệ bằng phương pháp winsorize")
        
        # 6. Chuẩn hóa giá trị nếu cần
        if pd.api.types.is_numeric_dtype(df[value_column]):
            # Chuẩn hóa về khoảng [0, 1] hoặc [-1, 1] tùy theo loại dữ liệu
            min_val = df[value_column].min()
            max_val = df[value_column].max()

            if min_val < 0 or max_val > 1:
                # Tạo cột mới cho giá trị chuẩn hóa
                norm_col = f"{value_column}_normalized"
                
                # Chuẩn hóa MinMax nhưng bảo toàn biến thiên
                df[norm_col] = (df[value_column] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                self.logger.info(f"Đã chuẩn hóa giá trị tâm lý về khoảng [0, 1] trong cột {norm_col}")
                
                # Giữ lại phiên bản được đánh dấu biến thiên bằng cách thêm một cột marker
                df['sentiment_has_change'] = df[value_column].diff().abs() > 0.001
                
                # Thêm cột để dánh giá mức độ biến thiên
                # Calculate rolling volatility of sentiment
                df['sentiment_volatility'] = df[value_column].rolling(window=5).std()
        
        # 7. Sắp xếp theo timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        self.logger.info(f"Hoàn thành làm sạch dữ liệu tâm lý: {len(df)} dòng")
        
        return df

    def handle_missing_values_in_dataframe(self, df, method='bfill', handle_leading=True):
        """
        Xử lý tất cả các giá trị NaN trong DataFrame.
        
        Args:
            df: DataFrame cần xử lý
            method: Phương pháp điền ('bfill', 'ffill', 'mean', 'median', 'zero')
            handle_leading: Có xử lý riêng NaN ở đầu không
            
        Returns:
            DataFrame đã xử lý NaN
        """
        from data_processors.utils.preprocessing import handle_leading_nans, fill_nan_values
        
        result_df = df.copy()
        # Chỉ áp dụng với các cột số
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if result_df[col].isna().any():
                # Xử lý NaN ở đầu nếu cần
                if handle_leading:
                    result_df[col] = handle_leading_nans(result_df[col])
                    
                # Xử lý NaN còn lại
                if result_df[col].isna().any():
                    result_df[col] = fill_nan_values(result_df[col], method=method)
        
        return result_df
    
    def clean_data(
        self,
        data: Dict[str, pd.DataFrame],
        clean_ohlcv: bool = True,
        clean_orderbook: bool = False,
        clean_trades: bool = False,
        clean_sentiment: bool = True,
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Thay đổi mặc định của các tham số xử lý NaN
        handle_leading_nan: bool = True,
        leading_nan_method: str = 'backfill',
        min_periods: int = 5,
        handle_extreme_volume: bool = True,
        preserve_timestamp: bool = True,
        # Đổi giá trị mặc định 
        aggressive_nan_handling: bool = True,
        fill_all_nan: bool = True, 
        fill_method: str = 'interpolate'
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
            handle_extreme_volume: Xử lý giá trị cực đại của khối lượng
            preserve_timestamp: Giữ nguyên timestamp
            aggressive_nan_handling: Xử lý triệt để giá trị NaN
            fill_all_nan: Đảm bảo không còn NaN nào
            fill_method: Phương pháp điền các giá trị NaN
            
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
        
        # Phân loại dữ liệu
        market_data = {}
        sentiment_data = {}
        
        # Phân loại dữ liệu trước khi xử lý
        for symbol, df in data.items():
            if symbol.lower().endswith('_sentiment') or symbol.lower() == 'sentiment':
                sentiment_data[symbol] = df
            else:
                market_data[symbol] = df
        
        # Làm sạch dữ liệu tâm lý
        if clean_sentiment and sentiment_data:
            self.logger.info(f"Làm sạch {len(sentiment_data)} bộ dữ liệu tâm lý")
            for symbol, df in sentiment_data.items():
                try:
                    # Sử dụng hàm làm sạch tâm lý chuyên biệt
                    cleaned_sentiment = self.clean_sentiment_data(df)
                    results[symbol] = cleaned_sentiment
                    
                    self.logger.info(f"Đã làm sạch dữ liệu tâm lý cho {symbol}: {len(df)} -> {len(cleaned_sentiment)} dòng")
                except Exception as e:
                    self.logger.error(f"Lỗi khi làm sạch dữ liệu tâm lý cho {symbol}: {str(e)}")
                    results[symbol] = df
        
        # Làm sạch dữ liệu thị trường
        for symbol, df in market_data.items():
            try:
                # Lưu bản sao của các cột giá gốc
                original_ohlc = None
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    original_ohlc = {
                        'open': df['open'].copy(),
                        'high': df['high'].copy(),
                        'low': df['low'].copy(),
                        'close': df['close'].copy()
                    }
                    
                    # Sửa high < low và giá trị âm trước khi xử lý
                    df = ensure_valid_price_data(df, fix_high_low=True, ensure_positive=True)                
                # Lưu cột timestamp nếu có
                timestamp_col = None
                if preserve_timestamp and 'timestamp' in df.columns:
                    # Đảm bảo timestamp là datetime trước khi lưu
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
                    timestamp_col = df['timestamp'].copy()

                if handle_leading_nan:
                    # Sử dụng trực tiếp từ preprocessing.py
                    from data_processors.utils.preprocessing import handle_leading_nans
                    
                    # Xử lý các cột số
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        if df[col].isna().any():
                            # Sử dụng giá trị fill_value dựa trên leading_nan_method
                            if leading_nan_method == 'zero':
                                fill_value = 0
                            elif leading_nan_method == 'mean' and len(df[col].dropna()) > min_periods:
                                fill_value = df[col].mean()
                            elif leading_nan_method == 'median' and len(df[col].dropna()) > min_periods:
                                fill_value = df[col].median()
                            else:
                                # Mặc định để None sẽ sử dụng giá trị hợp lệ đầu tiên
                                fill_value = None
                                
                            df[col] = handle_leading_nans(df[col], fill_value=fill_value)
                    
                    self.logger.info(f"Đã xử lý giá trị NaN ở đầu cho {symbol} bằng phương pháp {leading_nan_method}")

                if aggressive_nan_handling:
                    # Xử lý NaN còn lại sau khi đã xử lý NaN ở đầu
                    df = self.handle_missing_values_in_dataframe(
                        df, 
                        method=fill_method if fill_method else 'interpolate',
                        handle_leading=False # Đã xử lý ở trên rồi
                    )

                # Xác định loại dữ liệu và làm sạch
                if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']) and clean_ohlcv:
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
                    
                    # THÊM ĐOẠN CODE NÀY: Làm sạch và chuẩn hóa các chỉ báo kỹ thuật
                    if handle_extreme_volume or config.get("handle_technical_indicators", True):
                        # Import các hàm cần thiết
                        from data_processors.utils.preprocessing import clean_technical_features

                        # Cấu hình cho việc làm sạch và chuẩn hóa chỉ báo kỹ thuật
                        tech_indicator_config = {
                            'fix_outliers': True,
                            'outlier_method': 'zscore',
                            'outlier_threshold': config.get("outlier_threshold", 3.0),
                            'normalize_indicators': True,
                            'standardize_indicators': True,
                            'add_trend_strength': True,
                            'add_volatility_zscore': True,
                            'generate_labels': config.get("generate_labels", False),
                            'label_window': config.get("label_window", 10),
                            'label_threshold': config.get("label_threshold", 0.01),
                            'price_col': 'close',
                            'high_col': 'high',
                            'low_col': 'low',
                            'close_col': 'close'
                        }

                        # Áp dụng clean_technical_features
                        cleaned_df = clean_technical_features(cleaned_df, tech_indicator_config)
                        self.logger.info(f"Đã làm sạch và chuẩn hóa các chỉ báo kỹ thuật cho {symbol}")

                    self.logger.info(f"Đã làm sạch {len(df)} -> {len(cleaned_df)} dòng dữ liệu OHLCV cho {symbol}")
                    
                # Xử lý các loại dữ liệu khác (orderbook, trades, ...)
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
                
                # Khôi phục cột timestamp nếu đã lưu
                if timestamp_col is not None:
                    cleaned_df['timestamp'] = timestamp_col
                elif 'timestamp' in cleaned_df.columns and not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
                    # Đảm bảo timestamp là datetime
                    cleaned_df = self._convert_timestamp(cleaned_df)

                results[symbol] = cleaned_df
                # Kiểm tra lại và sửa một lần nữa sau khi xử lý
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    df = ensure_valid_price_data(df, fix_high_low=True, ensure_positive=True)
            except Exception as e:
                self.logger.error(f"Lỗi khi làm sạch dữ liệu cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                # Trả về dữ liệu gốc nếu có lỗi
                results[symbol] = df
    
        return results
    
    def fillna_smart(self, df, columns=None, methods=['ffill', 'bfill', 'zero']):
        """
        Điền giá trị NaN bằng phương pháp thông minh kết hợp nhiều phương pháp.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách các cột cần xử lý (None để xử lý tất cả)
            methods: Thứ tự các phương pháp điền NaN
            
        Returns:
            DataFrame đã xử lý
        """
        result_df = df.copy()
        
        # Xác định các cột cần xử lý
        if columns is None:
            columns = result_df.select_dtypes(include=['number']).columns
        
        # Xử lý từng cột
        for col in columns:
            if col not in result_df.columns:
                continue
                
            # Kiểm tra xem có NaN không
            if not result_df[col].isna().any():
                continue
                
            # Số lượng NaN ban đầu
            nan_count = result_df[col].isna().sum()
            
            # Áp dụng lần lượt các phương pháp
            for method in methods:
                if method == 'ffill':
                    result_df[col] = result_df[col].fillna(method='ffill')
                elif method == 'bfill':
                    result_df[col] = result_df[col].fillna(method='bfill')
                elif method == 'zero':
                    result_df[col] = result_df[col].fillna(0)
                elif method == 'mean':
                    mean_val = result_df[col].mean()
                    result_df[col] = result_df[col].fillna(mean_val)
                elif method == 'median':
                    median_val = result_df[col].median()
                    result_df[col] = result_df[col].fillna(median_val)
                elif method == 'interpolate':
                    result_df[col] = result_df[col].interpolate(method='linear')
                
                # Kiểm tra xem đã điền hết NaN chưa
                if not result_df[col].isna().any():
                    break
            
            # Kiểm tra nếu vẫn còn NaN, điền bằng 0
            if result_df[col].isna().any():
                result_df[col] = result_df[col].fillna(0)
                
            # Số lượng NaN đã điền
            filled_count = nan_count - result_df[col].isna().sum()
            if filled_count > 0:
                self.logger.debug(f"Đã điền {filled_count} giá trị NaN trong cột {col}")
        
        return result_df

    def generate_features(
        self,
        data: Dict[str, pd.DataFrame],
        feature_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_pipeline: Optional[str] = None,
        fit_pipeline: bool = True,
        all_indicators: bool = False,  # Thêm tham số này
        clean_indicators: bool = True,
        preserve_timestamp: bool = True  # Thêm tham số này
    ) -> Dict[str, pd.DataFrame]:
        """
        Tạo đặc trưng cho dữ liệu.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            feature_configs: Cấu hình đặc trưng cho mỗi symbol
            use_pipeline: Tên pipeline có sẵn để sử dụng
            fit_pipeline: Học pipeline mới hay không
            all_indicators: Sử dụng tất cả các chỉ báo kỹ thuật có sẵn  # Thêm mô tả này
            clean_indicators: Làm sạch các chỉ báo kỹ thuật
            preserve_timestamp: Giữ nguyên timestamp
            
        Returns:
            Dict với key là symbol và value là DataFrame có đặc trưng
        """
        if not self.config.get("feature_engineering", {}).get("enabled", True):
            self.logger.info("Bỏ qua bước tạo đặc trưng (đã bị tắt trong cấu hình)")
            return data
        
        results = {}
        
        # Phân loại dữ liệu
        market_data = {}
        sentiment_data = {}
        
        # Phân loại dữ liệu trước khi xử lý
        for symbol, df in data.items():
            if symbol.lower().endswith('_sentiment') or symbol.lower() == 'sentiment':
                sentiment_data[symbol] = df
            else:
                market_data[symbol] = df
        
        # Thêm các sentiment data vào kết quả mà không xử lý
        for symbol, df in sentiment_data.items():
            results[symbol] = df
        
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
            for symbol, df in market_data.items():
                try:
                    # Kiểm tra yêu cầu tối thiểu
                    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        self.logger.warning(f"Bỏ qua tạo đặc trưng cho {symbol}: Thiếu các cột OHLC cần thiết")
                        results[symbol] = df
                        continue
                    
                    # Lưu timestamp nếu cần
                    timestamp_col = None
                    if preserve_timestamp and 'timestamp' in df.columns:
                        # Đảm bảo timestamp là datetime
                        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                            df = self._convert_timestamp(df)
                        timestamp_col = df['timestamp'].copy()
                        # Loại bỏ timestamp tạm thời để tạo đặc trưng
                        df = df.drop('timestamp', axis=1)
                    
                    # Áp dụng pipeline
                    features_df = self.feature_generator.transform_data(
                        df, 
                        pipeline_name=pipeline_config.get("pipeline_name"),
                        fit=fit_pipeline
                    )
                    
                    # Khôi phục timestamp nếu đã lưu
                    if timestamp_col is not None:
                        features_df.insert(0, 'timestamp', timestamp_col)
                    
                    results[symbol] = features_df
                    self.logger.info(f"Đã tạo đặc trưng cho {symbol}: {df.shape[1]} -> {features_df.shape[1]} cột")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi tạo đặc trưng cho {symbol}: {str(e)}")
                    results[symbol] = df
            
            return results
        
        # Tạo đặc trưng với cấu hình riêng cho mỗi symbol
        # Tạo đặc trưng với cấu hình riêng cho mỗi symbol
        for symbol, df in market_data.items():
            try:
                # Lấy cấu hình cho symbol này
                if feature_configs is None:
                    feature_configs = {}
                
                # Sử dụng cấu hình chung cho "_default_" nếu không có cấu hình riêng
                if symbol not in feature_configs and "_default_" in feature_configs:
                    symbol_config = feature_configs["_default_"].copy()
                else:
                    symbol_config = feature_configs.get(symbol, {})
                
                # Kiểm tra yêu cầu tối thiểu
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    self.logger.warning(f"Bỏ qua tạo đặc trưng cho {symbol}: Thiếu các cột OHLC cần thiết")
                    results[symbol] = df
                    continue
                
                # THÊM: Lưu bản sao dữ liệu giá gốc
                original_price_data = {
                    'open': df['open'].copy(),
                    'high': df['high'].copy(),
                    'low': df['low'].copy(),
                    'close': df['close'].copy()
                }
                if 'volume' in df.columns:
                    original_price_data['volume'] = df['volume'].copy()
                    
                # Sửa lỗi dữ liệu trước khi tạo đặc trưng
                df = ensure_valid_price_data(df, fix_high_low=True, ensure_positive=True)
                
                # Lưu timestamp nếu cần
                timestamp_col = None
                timestamp_index = False
                if preserve_timestamp and 'timestamp' in df.columns:
                    # Đảm bảo timestamp là datetime
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
                    timestamp_col = df['timestamp'].copy()
                    # Loại bỏ timestamp tạm thời để tạo đặc trưng
                    df = df.drop('timestamp', axis=1)
                    
                # ... [code tạo đặc trưng không thay đổi] ...
                    
                # Khôi phục timestamp nếu đã lưu
                if timestamp_col is not None:
                    features_df.insert(0, 'timestamp', timestamp_col)
                    
                # THÊM: Khôi phục dữ liệu giá gốc
                for col, data in original_price_data.items():
                    if col in features_df.columns:
                        features_df[col] = data
                        
                # THÊM: Đảm bảo dữ liệu giá hợp lệ sau khi khôi phục
                features_df = ensure_valid_price_data(features_df, fix_high_low=True, ensure_positive=True)
                
                # Kiểm tra sau khi sửa - Log để debug
                high_low_issue = (features_df['high'] < features_df['low']).sum() if all(col in features_df.columns for col in ['high', 'low']) else 0
                neg_values = {col: (features_df[col] < 0).sum() for col in ['open', 'high', 'low', 'close'] if col in features_df.columns}
                
                if high_low_issue > 0 or sum(neg_values.values()) > 0:
                    self.logger.warning(f"Vẫn còn vấn đề sau khi khôi phục dữ liệu giá cho {symbol}: {high_low_issue} high<low, {neg_values}")
                else:
                    self.logger.debug(f"Đã khôi phục và sửa thành công dữ liệu giá cho {symbol}")
                    
                results[symbol] = features_df
                self.logger.info(f"Đã tạo đặc trưng cho {symbol}: {df.shape[1]} -> {features_df.shape[1]} cột")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo đặc trưng cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                results[symbol] = df
        
    def find_sentiment_files(self, 
                            sentiment_dir: Path, 
                            asset: str,
                            include_subdirs: bool = True) -> List[Path]:
        """
        Tìm kiếm file tâm lý trong thư mục và các thư mục con.
        
        Args:
            sentiment_dir: Thư mục chứa dữ liệu tâm lý
            asset: Mã tài sản (BTC, ETH, ...)
            include_subdirs: Có tìm kiếm trong thư mục con hay không
            
        Returns:
            Danh sách đường dẫn tới các file tâm lý, sắp xếp theo thời gian gần nhất
        """
        self.logger.info(f"Tìm kiếm dữ liệu tâm lý cho {asset} trong {sentiment_dir}")
        
        if not sentiment_dir.exists():
            self.logger.warning(f"Thư mục {sentiment_dir} không tồn tại")
            sentiment_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Đã tạo thư mục {sentiment_dir}")
            return []
        
        asset_lower = asset.lower()
        
        # Pattern tìm kiếm các file tâm lý
        patterns = [
            f"*{asset_lower}*sentiment*.parquet",  # BTC_sentiment.parquet
            f"*{asset_lower}*sentiment*.csv",      # BTC_sentiment.csv
            f"*sentiment*{asset_lower}*.parquet",  # sentiment_BTC.parquet
            f"*sentiment*{asset_lower}*.csv",      # sentiment_BTC.csv
            f"*{asset_lower}*fear*greed*.parquet", # BTC_fear_greed.parquet
            f"*{asset_lower}*fear*greed*.csv",     # BTC_fear_greed.csv
            "*fear*greed*.parquet",                # fear_greed_index.parquet (general)
            "*fear*greed*.csv"                     # fear_greed_index.csv (general)
        ]
        
        # Tìm kiếm trong thư mục hiện tại
        files = []
        for pattern in patterns:
            files.extend(list(sentiment_dir.glob(pattern)))
            
            # Nếu đã tìm thấy file cho asset cụ thể, không cần tìm file chung
            if files and asset_lower in patterns[0]:
                break
        
        # Nếu không tìm thấy và include_subdirs=True, tìm trong thư mục con
        if not files and include_subdirs:
            subdirs = [d for d in sentiment_dir.iterdir() if d.is_dir()]
            for subdir in subdirs:
                for pattern in patterns:
                    subdir_files = list(subdir.glob(pattern))
                    if subdir_files:
                        files.extend(subdir_files)
                        # Nếu đã tìm thấy file cho asset cụ thể, không cần tìm file chung
                        if asset_lower in patterns[0]:
                            break
        
        # Sắp xếp theo thời gian sửa đổi mới nhất
        if files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.logger.info(f"Tìm thấy {len(files)} file tâm lý cho {asset}, sử dụng file mới nhất: {files[0]}")
        else:
            self.logger.warning(f"Không tìm thấy file tâm lý nào cho {asset} trong {sentiment_dir}")
            self.logger.warning(f"Bạn có thể tạo file tâm lý bằng lệnh:")
            self.logger.warning(f"python main.py collect sentiment --asset {asset} --output-dir data/sentiment")
            self.logger.warning(f"hoặc:")
            self.logger.warning(f"python main.py collect fear_greed --output-dir data/sentiment")
        
        return files

    def merge_sentiment_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], str, Path]] = None,
        sentiment_dir: Optional[Union[str, Path]] = None,
        method: str = 'last_value',
        window: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """
        Kết hợp dữ liệu thị trường với dữ liệu tâm lý - phiên bản đơn giản hóa
        """
        results = {}
        sentiment_by_symbol = {}
        
        if market_data is None or not market_data:
            self.logger.warning("Không có dữ liệu thị trường để kết hợp")
            return {}
        
        # Xử lý các trường hợp cung cấp dữ liệu tâm lý
        if sentiment_data is not None:
            if isinstance(sentiment_data, pd.DataFrame):
                # Trường hợp 1: Dữ liệu tâm lý là một DataFrame duy nhất
                self.logger.info("Sử dụng DataFrame tâm lý được cung cấp trực tiếp")
                common_sentiment_data = sentiment_data
                # KHÔNG GỌI clean_sentiment_data() ở đây
            elif isinstance(sentiment_data, dict):
                # Trường hợp 2: Dữ liệu tâm lý là Dictionary chứa DataFrame theo symbol
                self.logger.info("Sử dụng Dictionary dữ liệu tâm lý theo symbol")
                sentiment_by_symbol = sentiment_data
                common_sentiment_data = None
            elif isinstance(sentiment_data, (str, Path)):
                # Trường hợp 3: Dữ liệu tâm lý là đường dẫn file
                self.logger.info(f"Tải dữ liệu tâm lý từ file: {sentiment_data}")
                try:
                    file_path = Path(sentiment_data)
                    if file_path.suffix.lower() == '.csv':
                        common_sentiment_data = pd.read_csv(file_path, parse_dates=['timestamp'])
                    elif file_path.suffix.lower() == '.parquet':
                        common_sentiment_data = pd.read_parquet(file_path)
                    else:
                        self.logger.warning(f"Định dạng file không được hỗ trợ: {file_path.suffix}")
                        common_sentiment_data = None
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải file dữ liệu tâm lý: {str(e)}")
                    common_sentiment_data = None
            else:
                self.logger.warning("Định dạng dữ liệu tâm lý không được hỗ trợ")
                common_sentiment_data = None
        else:
            # Tìm kiếm trong market_data nếu có dữ liệu tâm lý
            sentiment_keys = [k for k in market_data.keys() if k.lower().endswith('_sentiment') or k.lower() == 'sentiment']
            
            if sentiment_keys:
                # Sử dụng dữ liệu tâm lý từ market_data
                for key in sentiment_keys:
                    sentiment_by_symbol[key] = market_data[key]
                    # Loại bỏ khỏi market_data
                    market_data = {k: v for k, v in market_data.items() if k != key}
                
                common_sentiment_data = None
            else:
                common_sentiment_data = None
        
        # Tìm kiếm file dữ liệu tâm lý riêng cho từng symbol nếu cần
        if sentiment_dir is not None and not sentiment_by_symbol:
            # ...giữ nguyên phần này...
            sentiment_dir_path = Path(sentiment_dir)
            # Đảm bảo thư mục tồn tại
            if not sentiment_dir_path.exists():
                sentiment_dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Tìm kiếm dữ liệu tâm lý từ thư mục: {sentiment_dir_path}")
            
            # Tìm file cho từng symbol
            for symbol in market_data.keys():
                asset = symbol.split('/')[0] if '/' in symbol else symbol
                sentiment_files = self.find_sentiment_files(sentiment_dir_path, asset)
                
                if sentiment_files:
                    # Đọc file mới nhất
                    file_path = sentiment_files[0]
                    try:
                        if file_path.suffix.lower() == '.csv':
                            df = pd.read_csv(file_path)
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        elif file_path.suffix.lower() == '.parquet':
                            df = pd.read_parquet(file_path)
                        
                        if df is not None and not df.empty:
                            # KHÔNG gọi clean_sentiment_data ở đây
                            sentiment_by_symbol[symbol] = df
                    except Exception as e:
                        self.logger.error(f"Lỗi khi đọc file tâm lý {file_path}: {str(e)}")
        
        # Xử lý từng symbol
        for symbol, df in market_data.items():
            try:
                # Đảm bảo timestamp là datetime
                if 'timestamp' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
                    
                    self.logger.info(f"Timestamp cho {symbol}: {df['timestamp'].dtype}, "
                                    f"min={df['timestamp'].min()}, max={df['timestamp'].max()}")
                else:
                    self.logger.warning(f"Không tìm thấy cột timestamp trong dữ liệu thị trường cho {symbol}")
                    results[symbol] = df
                    continue
                
                # Tạo bản sao để tránh thay đổi dữ liệu gốc
                merged_df = df.copy()
                
                # Xác định dữ liệu tâm lý cho symbol này
                symbol_sentiment = None
                
                # Ưu tiên sử dụng dữ liệu tâm lý riêng cho symbol
                if symbol in sentiment_by_symbol:
                    symbol_sentiment = sentiment_by_symbol[symbol]
                elif f"{symbol}_sentiment" in sentiment_by_symbol:
                    symbol_sentiment = sentiment_by_symbol[f"{symbol}_sentiment"]
                elif '/' in symbol:
                    asset = symbol.split('/')[0]
                    asset_key = f"{asset}_sentiment"
                    if asset_key in sentiment_by_symbol:
                        symbol_sentiment = sentiment_by_symbol[asset_key]
                elif common_sentiment_data is not None:
                    # Chọn dữ liệu tâm lý phù hợp từ common_sentiment_data
                    asset = symbol.split('/')[0] if '/' in symbol else symbol
                    if 'asset' in common_sentiment_data.columns:
                        asset_mask = (
                            (common_sentiment_data['asset'] == asset) | 
                            (common_sentiment_data['asset'].str.lower() == asset.lower() if isinstance(asset, str) else False)
                        )
                        symbol_sentiment = common_sentiment_data[asset_mask].copy()
                    else:
                        symbol_sentiment = common_sentiment_data.copy()
                
                # Nếu không có dữ liệu tâm lý
                if symbol_sentiment is None or symbol_sentiment.empty:
                    self.logger.warning(f"Không có dữ liệu tâm lý cho {symbol}")
                    results[symbol] = merged_df
                    continue
                
                # Xác định cột giá trị tâm lý
                value_column = None
                for col in ['sentiment_value', 'sentiment_value_normalized', 'value', 'sentiment', 'score', 'fear_greed_value']:
                    if col in symbol_sentiment.columns:
                        value_column = col
                        self.logger.info(f"Sử dụng cột {value_column} cho giá trị tâm lý của {symbol}")
                        break
                
                if value_column is None:
                    self.logger.warning(f"Không tìm thấy cột giá trị tâm lý cho {symbol}")
                    results[symbol] = merged_df
                    continue
                
                # Đảm bảo timestamp trong dữ liệu tâm lý là datetime
                if 'timestamp' in symbol_sentiment.columns:
                    if not pd.api.types.is_datetime64_any_dtype(symbol_sentiment['timestamp']):
                        symbol_sentiment = self._convert_timestamp(symbol_sentiment)
                    
                    self.logger.info(f"Timestamp tâm lý cho {symbol}: {symbol_sentiment['timestamp'].dtype}, "
                                    f"min={symbol_sentiment['timestamp'].min()}, max={symbol_sentiment['timestamp'].max()}")
                else:
                    self.logger.warning(f"Không tìm thấy cột timestamp trong dữ liệu tâm lý cho {symbol}")
                    results[symbol] = merged_df
                    continue
                
                # Đặt timestamp làm index để kết hợp dữ liệu
                market_df = merged_df.set_index('timestamp')
                sentiment_df = symbol_sentiment.set_index('timestamp')
                
                # Chuẩn bị dữ liệu tâm lý - chỉ giữ lại cột value_column cho đơn giản
                sentiment_series = sentiment_df[value_column]
                
                # ĐƠN GIẢN HÓA: Tính giá trị tâm lý theo phương pháp đã chọn
                if method == 'last_value':
                    resampled_sentiment = sentiment_series.resample(window).last()
                elif method == 'mean':
                    resampled_sentiment = sentiment_series.resample(window).mean()
                elif method == 'interpolate':
                    # Phương pháp nội suy thay vì lấy last_value
                    resampled_sentiment = sentiment_series.resample(window).mean()
                    resampled_sentiment = resampled_sentiment.interpolate(method='linear')
                else:
                    self.logger.warning(f"Phương pháp không hợp lệ: {method}, sử dụng 'last_value'")
                    resampled_sentiment = sentiment_series.resample(window).last()
                
                # Kết quả kết hợp
                merged_result = pd.DataFrame(index=market_df.index)
                
                # Thêm dữ liệu thị trường
                for col in market_df.columns:
                    merged_result[col] = market_df[col]
                
                # ĐƠNG GIẢN HÓA: Chỉ thêm cột sentiment_value gốc
                merged_result['sentiment_value'] = np.nan
                
                # Tìm các ngày chung giữa dữ liệu tâm lý và dữ liệu thị trường
                common_dates = merged_result.index.intersection(resampled_sentiment.index)
                self.logger.info(f"Tìm thấy {len(common_dates)} ngày chung giữa dữ liệu thị trường và tâm lý")
                
                if len(common_dates) > 0:
                    # Gán giá trị tâm lý cho các ngày chung
                    merged_result.loc[common_dates, 'sentiment_value'] = resampled_sentiment.loc[common_dates]
                    
                    # Điền các giá trị thiếu - sử dụng nội suy tuyến tính thay vì fillna
                    if merged_result['sentiment_value'].isna().any():
                        merged_result['sentiment_value'] = merged_result['sentiment_value'].interpolate(method='linear')
                        # Xử lý NaN ở đầu và cuối (không thể nội suy)
                        merged_result['sentiment_value'] = merged_result['sentiment_value'].fillna(method='ffill').fillna(method='bfill')
                    
                    # ĐƠN GIẢN HÓA: Chỉ tính thêm sentiment_change
                    merged_result['sentiment_change'] = merged_result['sentiment_value'].diff()
                    
                    # Kiểm tra tỷ lệ giá trị bằng 0
                    zero_ratio = (merged_result['sentiment_change'] == 0).mean()
                    if zero_ratio > 0.5:  # Nếu hơn 50% là 0
                        self.logger.warning(f"Phát hiện {zero_ratio:.2%} giá trị sentiment_change = 0, áp dụng nhiễu")
                        
                        # Tăng biên độ nhiễu lên đáng kể
                        np.random.seed(42)
                        noise_scale = merged_result['sentiment_value'].std() * 0.1  # Tăng lên 10% độ lệch chuẩn
                        noise = np.random.normal(0, noise_scale, size=len(merged_result))
                        
                        # Chỉ thêm nhiễu vào sentiment_value
                        merged_result['sentiment_value_noisy'] = merged_result['sentiment_value'] * (1 + noise)
                        
                        # Cập nhật sentiment_change
                        merged_result['sentiment_change'] = merged_result['sentiment_value_noisy'].diff()
                    
                    # Đếm số lượng giá trị hợp lệ
                    valid_sentiment = merged_result['sentiment_value'].notna().sum()
                    self.logger.info(f"Số giá trị tâm lý hợp lệ: {valid_sentiment}/{len(merged_result)} ({valid_sentiment/len(merged_result)*100:.2f}%)")
                else:
                    self.logger.warning(f"Không tìm thấy ngày chung giữa dữ liệu thị trường và tâm lý cho {symbol}")
                
                # Reset index để đưa timestamp trở lại thành cột
                merged_result = merged_result.reset_index()
                
                results[symbol] = merged_result
                self.logger.info(f"Đã kết hợp dữ liệu tâm lý vào dữ liệu thị trường cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi kết hợp dữ liệu tâm lý cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                # Trả về dữ liệu gốc nếu có lỗi
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
    
    def clean_and_standardize_indicators(
        self,
        data: Dict[str, pd.DataFrame],
        fix_outliers: bool = True,
        normalize_indicators: bool = True,
        standardize_indicators: bool = True,
        add_trend_strength: bool = True,
        add_volatility_zscore: bool = True,
        generate_labels: bool = False,
        label_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Làm sạch và chuẩn hóa các chỉ báo kỹ thuật trong dữ liệu.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            fix_outliers: Sửa các giá trị ngoại lệ
            normalize_indicators: Chuẩn hóa các chỉ báo về khoảng [0, 1]
            standardize_indicators: Chuẩn hóa các chỉ báo về trung bình 0, độ lệch chuẩn 1
            add_trend_strength: Thêm chỉ báo độ mạnh xu hướng
            add_volatility_zscore: Thêm chỉ báo Z-score của biến động
            generate_labels: Tạo nhãn cho huấn luyện
            label_config: Cấu hình cho việc tạo nhãn
        
        Returns:
            Dict với key là symbol và value là DataFrame đã xử lý
        """
        from data_processors.utils.preprocessing import clean_technical_features, handle_leading_nans, fill_nan_values
        
        if label_config is None:
            label_config = {
                'label_window': 10,
                'label_threshold': 0.01
            }
        
        results = {}
        
        for symbol, df in data.items():
            self.logger.info(f"Làm sạch và chuẩn hóa các chỉ báo kỹ thuật cho {symbol}")
            try:
                # Bỏ qua dữ liệu tâm lý
                if symbol.lower().endswith('_sentiment') or symbol.lower() == 'sentiment':
                    results[symbol] = df
                    continue
                
                # Kiểm tra xem có đủ cột OHLC không
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Bỏ qua {symbol}: Thiếu các cột OHLC cần thiết: {[col for col in required_cols if col not in df.columns]}")
                    results[symbol] = df
                    continue
                
                # Tạo cấu hình cho việc làm sạch và chuẩn hóa
                config = {
                    'fix_outliers': fix_outliers,
                    'outlier_method': 'zscore',
                    'outlier_threshold': 3.0,
                    'normalize_indicators': normalize_indicators,
                    'standardize_indicators': standardize_indicators,
                    'add_trend_strength': add_trend_strength,
                    'add_volatility_zscore': add_volatility_zscore,
                    'generate_labels': generate_labels,
                    'label_window': label_config.get('label_window', 10),
                    'label_threshold': label_config.get('label_threshold', 0.01)
                }
                
                # Áp dụng clean_technical_features
                processed_df = clean_technical_features(df, config)
                
                # Xử lý NaN còn sót lại sau khi chuẩn hóa
                for indicator_type in config:
                    for suffix in ['_std', '_norm', '_scaled', '_log']:
                        cols = [col for col in processed_df.columns if col.endswith(suffix)]
                        for col in cols:
                            if processed_df[col].isna().any():
                                # Xử lý NaN
                                processed_df[col] = handle_leading_nans(processed_df[col])
                                if processed_df[col].isna().any():
                                    processed_df[col] = fill_nan_values(processed_df[col], method='ffill')
                
                # Ghi log các cột mới được thêm vào
                new_cols = set(processed_df.columns) - set(df.columns)
                if new_cols:
                    self.logger.info(f"Đã thêm {len(new_cols)} cột mới cho {symbol}: {', '.join(new_cols)}")
                
                results[symbol] = processed_df
                
            except Exception as e:
                self.logger.error(f"Lỗi khi làm sạch và chuẩn hóa các chỉ báo kỹ thuật cho {symbol}: {str(e)}")
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
                if symbol.lower().endswith('_sentiment') or symbol.lower() == 'sentiment':
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
                    if new_cols:
                        self.logger.info(f"Đã tạo {len(new_cols)} cột mục tiêu cho {symbol}: {', '.join(new_cols)}")
                    else:
                        self.logger.warning(f"Không có cột mục tiêu mới nào được tạo cho {symbol}")
                else:
                    # Nếu không có MissingDataHandler, trả về dữ liệu gốc
                    results[symbol] = df
                    self.logger.warning("MissingDataHandler không được khởi tạo, không thể tạo cột mục tiêu")
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo cột mục tiêu cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                results[symbol] = df
    
        return results
    

    
    def check_data_consistency(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Kiểm tra tính nhất quán của dữ liệu trước khi lưu.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            
        Returns:
            Dict với key là symbol và value là báo cáo kiểm tra
        """
        results = {}
        
        # Import hàm clean_sentiment_features
        from data_processors.utils.preprocessing import clean_sentiment_features

        for symbol, df in data.items():
            self.logger.info(f"Kiểm tra tính nhất quán của dữ liệu cho {symbol}")
            
            # Làm sạch dữ liệu tâm lý trước khi kiểm tra - không làm mất thông tin biến thiên
            df_clean = df.copy()
            if any('sentiment_' in col for col in df_clean.columns):
                try:
                    # Thêm biến để theo dõi nếu dữ liệu đã được làm sạch
                    _already_cleaned = True
                    
                    # Kiểm tra cột sentiment_change
                    if 'sentiment_change' in df_clean.columns:
                        zero_ratio = (df_clean['sentiment_change'] == 0).mean()
                        if zero_ratio > 0.7:  # Nếu hơn 70% giá trị là 0
                            _already_cleaned = False
                            self.logger.warning(f"Phát hiện {zero_ratio:.2%} giá trị sentiment_change = 0, cần tái tính toán")
                    
                    # Chỉ làm sạch nếu thực sự cần thiết
                    if not _already_cleaned:
                        sentiment_cols = [col for col in df_clean.columns if 'sentiment_' in col]
                        
                        # Tính lại sentiment_change từ sentiment_value nếu có thể
                        if 'sentiment_value' in df_clean.columns:
                            # Làm sạch sentiment_value bằng nội suy trước
                            if df_clean['sentiment_value'].isna().any():
                                df_clean['sentiment_value'] = df_clean['sentiment_value'].interpolate(method='linear')
                                df_clean['sentiment_value'] = df_clean['sentiment_value'].fillna(method='ffill').fillna(method='bfill')
                            
                            # Tính lại sentiment_change
                            df_clean['sentiment_change'] = df_clean['sentiment_value'].diff()

                            # Điền giá trị NaN ở dòng đầu tiên
                            if df_clean['sentiment_change'].isna().any():
                                # Sử dụng giá trị 0 hoặc giá trị nhỏ
                                first_valid_change = df_clean['sentiment_change'].dropna().iloc[0] if not df_clean['sentiment_change'].dropna().empty else 0
                                df_clean['sentiment_change'] = df_clean['sentiment_change'].fillna(first_valid_change)

                            # Thêm nhiễu nhỏ để tránh giá trị 0 liên tiếp (khi cần)
                            zero_pct = (df_clean['sentiment_change'] == 0).mean()
                            if zero_pct > 0.5:
                                self.logger.warning(f"Áp dụng jittering cho sentiment_change với {zero_pct:.2%} giá trị bằng 0")
                                # Thêm nhiễu nhỏ hơn nhưng đủ để phân biệt
                                np.random.seed(42)  # Để kết quả nhất quán
                                small_jitter = np.random.normal(0, df_clean['sentiment_value'].std() * 0.005, size=len(df_clean))
                                df_clean['sentiment_value_enhanced'] = df_clean['sentiment_value'] * (1 + small_jitter)
                                df_clean['sentiment_change'] = df_clean['sentiment_value_enhanced'].diff().fillna(0)
                                
                                # Đảm bảo không còn NaN
                                df_clean['sentiment_change'] = df_clean['sentiment_change'].fillna(0)
                                
                                # Xóa cột phụ trợ
                                df_clean.drop('sentiment_value_enhanced', axis=1, inplace=True)
                            
                        # Xử lý các cột còn lại
                        for col in sentiment_cols:
                            if col == 'sentiment_change':
                                continue  # Đã xử lý ở trên
                                
                            if df_clean[col].isna().any():
                                # Sử dụng nội suy tuyến tính trước
                                df_clean[col] = df_clean[col].interpolate(method='linear')
                                
                                # Sau đó mới dùng ffill/bfill cho các điểm không thể nội suy
                                if df_clean[col].isna().any():
                                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                                
                                # Điền 0 cho các NaN còn lại (nếu có)
                                if df_clean[col].isna().any():
                                    df_clean[col] = df_clean[col].fillna(0)
                        
                        self.logger.info(f"Đã làm sạch các cột tâm lý cho {symbol} trước khi kiểm tra tính nhất quán")
                        
                        # Cập nhật lại data với dữ liệu đã làm sạch
                        data[symbol] = df_clean
                        df = df_clean
                except Exception as e:
                    self.logger.error(f"Lỗi khi làm sạch dữ liệu tâm lý: {str(e)}")
            
            report = {
                "rows": len(df),
                "columns": len(df.columns),
                "issues": [],
                "warnings": []
            }
            
            try:
                # 1. Kiểm tra timestamp
                if 'timestamp' in df.columns:
                    # Kiểm tra kiểu dữ liệu timestamp
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        report["issues"].append("timestamp không phải kiểu datetime64")
                        
                        # Thử chuyển đổi timestamp
                        try:
                            df = self._convert_timestamp(df)
                            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                                report["warnings"].append("timestamp đã được chuyển đổi tự động sang datetime64")
                            else:
                                report["issues"].append("không thể chuyển đổi timestamp sang datetime64")
                        except Exception as e:
                            report["issues"].append(f"lỗi khi chuyển đổi timestamp: {str(e)}")
                    
                    # Kiểm tra khoảng thời gian
                    if len(df) > 1:
                        time_diffs = pd.Series(df['timestamp'].diff().dropna())
                        unique_diffs = time_diffs.unique()
                        
                        if len(unique_diffs) > 1:
                            report["warnings"].append(f"khoảng cách timestamp không đều ({len(unique_diffs)} giá trị khác nhau)")
                            
                            # Tính thống kê khoảng cách
                            report["timestamp_stats"] = {
                                "min_diff": str(time_diffs.min()),
                                "max_diff": str(time_diffs.max()),
                                "mean_diff": str(time_diffs.mean()),
                                "unique_diffs": len(unique_diffs)
                            }
                        
                        # Kiểm tra dữ liệu trùng lặp
                        duplicates = df['timestamp'].duplicated().sum()
                        if duplicates > 0:
                            report["issues"].append(f"có {duplicates} timestamp trùng lặp")
                else:
                    report["issues"].append("thiếu cột timestamp")
                
                # 2. Kiểm tra dữ liệu thiếu (NaN)
                missing_values = df.isna().sum()
                missing_columns = missing_values[missing_values > 0].to_dict()
                
                if missing_columns:
                    null_columns = df.columns[df.isna().any()].tolist()
                    self.logger.warning(f"Các cột chứa NaN: {null_columns}")
                    self.logger.warning(f"Số lượng NaN trong mỗi cột: {df[null_columns].isna().sum()}")

                    report["warnings"].append(f"có {len(missing_columns)} cột chứa giá trị NaN")
                    report["missing_values"] = missing_columns
                
                # 3. Kiểm tra dữ liệu tâm lý
                if 'sentiment_value' in df.columns:
                    sentiment_na = df['sentiment_value'].isna().sum()
                    sentiment_coverage = (len(df) - sentiment_na) / len(df) * 100
                    
                    report["sentiment_stats"] = {
                        "missing_values": sentiment_na,
                        "coverage_percentage": sentiment_coverage
                    }
                    
                    if sentiment_coverage < 50:
                        report["issues"].append(f"dữ liệu tâm lý có tỷ lệ phủ thấp ({sentiment_coverage:.2f}%)")
                    elif sentiment_coverage < 90:
                        report["warnings"].append(f"dữ liệu tâm lý chưa đầy đủ ({sentiment_coverage:.2f}%)")
                    
                    # Kiểm tra kiểu dữ liệu
                    if not pd.api.types.is_numeric_dtype(df['sentiment_value']):
                        report["issues"].append("sentiment_value không phải kiểu số")
                
                # 4. Kiểm tra dữ liệu OHLCV
                ohlc_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in ohlc_cols):
                    # Kiểm tra logic OHLC
                    high_low_issue = (df['high'] < df['low']).sum()
                    if high_low_issue > 0:
                        report["issues"].append(f"có {high_low_issue} candle với high < low")
                    
                    # Kiểm tra giá trị âm
                    negative_values = {}
                    for col in ohlc_cols:
                        neg_count = (df[col] < 0).sum()
                        if neg_count > 0:
                            negative_values[col] = neg_count
                    
                    if negative_values:
                        report["issues"].append(f"có giá trị âm trong cột OHLC: {negative_values}")
                
                # 5. Kiểm tra cột mục tiêu
                target_cols = [col for col in df.columns if col.startswith('target_')]
                if target_cols:
                    missing_target = {}
                    for col in target_cols:
                        na_count = df[col].isna().sum()
                        if na_count > 0:
                            missing_target[col] = na_count
                    
                    if missing_target:
                        report["warnings"].append(f"có giá trị NaN trong {len(missing_target)} cột mục tiêu")
                        report["missing_target"] = missing_target
                
                # 6. Đánh giá tổng thể
                if report["issues"]:
                    report["overall_status"] = "failed" if len(report["issues"]) > 2 else "warning"
                elif report["warnings"]:
                    report["overall_status"] = "warning"
                else:
                    report["overall_status"] = "passed"
                
                results[symbol] = report
                
                # Log kết quả kiểm tra
                if report["overall_status"] == "warning":
                    if report["warnings"]:
                        # Hiển thị chi tiết các cảnh báo
                        warning_details = ', '.join(report['warnings'])
                        missing_info = ""
                        if "missing_values" in report and report["missing_values"]:
                            cols_with_missing = sorted(report["missing_values"].items(), key=lambda x: x[1], reverse=True)
                            top_missing = cols_with_missing[:5]  # Top 5 cột có nhiều giá trị NaN nhất
                            missing_info = f", các cột có nhiều NaN nhất: {', '.join([f'{k}({v})' for k,v in top_missing])}"
                            if len(cols_with_missing) > 5:
                                missing_info += f" và {len(cols_with_missing)-5} cột khác"
                        
                        self.logger.warning(f"Dữ liệu {symbol} có cảnh báo: {warning_details}{missing_info}")
                    else:
                        # Kiểm tra chi tiết các vấn đề tiềm ẩn
                        warnings = []
                        
                        # 1. Kiểm tra vấn đề về timestamp
                        if "timestamp" in df.columns:
                            if df['timestamp'].duplicated().sum() > 0:
                                warnings.append(f"Có {df['timestamp'].duplicated().sum()} timestamp trùng lặp")
                                
                            if len(df) > 1:
                                time_diffs = pd.Series(df['timestamp'].diff().dropna().astype('timedelta64[s]') / 3600)
                                unique_diffs = time_diffs.unique()
                                
                                if len(unique_diffs) > 1:
                                    warnings.append(f"Khoảng cách timestamp không đều ({len(unique_diffs)} giá trị khác nhau)")
                        
                        # 2. Kiểm tra vấn đề dữ liệu tâm lý
                        sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
                        if sentiment_cols:
                            if 'sentiment_change' in df.columns:
                                zero_change_ratio = (df['sentiment_change'] == 0).mean()
                                if zero_change_ratio > 0.5:
                                    warnings.append(f"{zero_change_ratio:.1%} giá trị sentiment_change bằng 0")
                                    
                            missing_sentiment = sum(df[col].isna().sum() for col in sentiment_cols)
                            if missing_sentiment > 0:
                                warnings.append(f"Có {missing_sentiment} giá trị NaN trong các cột sentiment")
                        
                        # 3. Kiểm tra các vấn đề với OHLCV
                        ohlc_cols = ['open', 'high', 'low', 'close']
                        if all(col in df.columns for col in ohlc_cols):
                            # Kiểm tra logic high/low
                            high_low_issues = (df['high'] < df['low']).sum()
                            if high_low_issues > 0:
                                warnings.append(f"Có {high_low_issues} candle với high < low")
                                
                            # Kiểm tra giá trị âm
                            negative_values = {col: (df[col] < 0).sum() for col in ohlc_cols if (df[col] < 0).any()}
                            if negative_values:
                                warnings.append(f"Có giá trị âm trong: {', '.join([f'{k}({v})' for k,v in negative_values.items()])}")
                        
                        # 4. Kiểm tra vấn đề với mức độ NaN nói chung
                        nan_cols = df.columns[df.isna().any()].tolist()
                        if nan_cols:
                            total_rows = len(df)
                            high_nan_cols = [col for col in nan_cols if df[col].isna().sum() / total_rows > 0.1]  # >10% giá trị NaN
                            if high_nan_cols:
                                warnings.append(f"Các cột có >10% giá trị NaN: {', '.join(high_nan_cols)}")
                        
                        # Hiển thị cảnh báo chi tiết nếu tìm thấy
                        if warnings:
                            self.logger.warning(f"Dữ liệu {symbol} có cảnh báo tiềm ẩn: {'; '.join(warnings)}")
                        else:
                            # Nếu không tìm thấy vấn đề cụ thể, hiển thị thông tin về dữ liệu
                            columns_info = f"gồm {len(df.columns)} cột và {len(df)} dòng"
                            dtypes_count = df.dtypes.value_counts().to_dict()
                            dtypes_info = ', '.join([f"{v} cột {k}" for k, v in dtypes_count.items()])
                            
                            self.logger.warning(f"Dữ liệu {symbol} có cảnh báo tiềm ẩn không rõ nguyên nhân ({columns_info}, {dtypes_info})")
                elif report["overall_status"] == "failed":     # <-- SỬA: Kiểm tra trạng thái "failed" thay vì "warning"
                    self.logger.error(f"Dữ liệu {symbol} có vấn đề nghiêm trọng: {', '.join(report['issues'])}")
                else:
                    self.logger.info(f"Dữ liệu {symbol} đạt yêu cầu nhất quán")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi kiểm tra tính nhất quán cho {symbol}: {str(e)}")
                results[symbol] = {"overall_status": "error", "error": str(e)}
        
        return results
    
    def save_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
        file_format: str = 'parquet',
        include_metadata: bool = True,
        preserve_timestamp: bool = True
    ) -> Dict[str, str]:
        """
        Lưu dữ liệu đã xử lý.
        
        Args:
            data: Dict với key là symbol và value là DataFrame
            output_dir: Thư mục đầu ra
            file_format: Định dạng file ('csv', 'parquet', 'json')
            include_metadata: Bao gồm metadata
            preserve_timestamp: Giữ nguyên timestamp
            
        Returns:
            Dict với key là symbol và value là đường dẫn file
        """       
        
        # Thêm class NumpyEncoder trong phương thức
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                return super(NumpyEncoder, self).default(obj)        
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
        
        # Kiểm tra tính nhất quán dữ liệu trước khi lưu
        consistency_reports = self.check_data_consistency(data)
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Bắt đầu lưu {len(data)} DataFrame vào {output_dir}")
        
        for symbol, df in data.items():
            try:
                # Kiểm tra báo cáo nhất quán
                report = consistency_reports.get(symbol, {})
                status = report.get("overall_status", "unknown")
                
                if status == "failed":
                    self.logger.warning(f"Dữ liệu {symbol} có vấn đề nghiêm trọng, lưu có thể không chính xác")
                
                # Thực hiện các bước cuối cùng để chuẩn bị dữ liệu
                df_to_save = df.copy()
                
                if all(col in df_to_save.columns for col in ['open', 'high', 'low', 'close']):
                    # Sửa dữ liệu bất thường trước khi lưu
                    df_to_save = ensure_valid_price_data(df_to_save, fix_high_low=True, ensure_positive=True)
                    
                    # Kiểm tra sau khi sửa - Log để debug
                    high_low_issue = (df_to_save['high'] < df_to_save['low']).sum()
                    neg_values = {col: (df_to_save[col] < 0).sum() for col in ['open', 'high', 'low', 'close']}
                    
                    if high_low_issue > 0 or sum(neg_values.values()) > 0:
                        self.logger.warning(f"Vẫn còn vấn đề sau khi sửa cho {symbol}: {high_low_issue} high<low, {neg_values}")
                    else:
                        self.logger.info(f"Đã sửa thành công dữ liệu giá cho {symbol} trước khi lưu")

                # Đảm bảo timestamp là datetime64[ns]
                if 'timestamp' in df_to_save.columns and preserve_timestamp:
                    if not pd.api.types.is_datetime64_any_dtype(df_to_save['timestamp']):
                        self.logger.info(f"Chuyển đổi timestamp cho {symbol} trước khi lưu")
                        df_to_save = self._convert_timestamp(df_to_save)
                
                # Tạo tên file
                filename = f"{symbol.replace('/', '_').lower()}_{timestamp}"
                file_path = output_dir / f"{filename}.{file_format}"
                
                self.logger.info(f"Đang lưu {symbol} vào {file_path}")
                
                # Lưu dữ liệu
                try:
                    if file_format == 'csv':
                        df_to_save.to_csv(file_path, index=False)
                        self.logger.info(f"Đã lưu CSV cho {symbol}")
                    elif file_format == 'parquet':
                        df_to_save.to_parquet(file_path, index=False)
                        self.logger.info(f"Đã lưu Parquet cho {symbol}")
                    elif file_format == 'json':
                        df_to_save.to_json(file_path, orient='records', date_format='iso')
                        self.logger.info(f"Đã lưu JSON cho {symbol}")
                    
                    # Lưu đường dẫn vào kết quả
                    results[symbol] = str(file_path)
                    
                    # Lưu metadata nếu cần
                    if include_metadata:
                        metadata = {
                            "symbol": symbol,
                            "rows": len(df_to_save),
                            "columns": df_to_save.columns.tolist(),
                            "saved_at": datetime.now().isoformat(),
                            "file_format": file_format,
                            "file_path": str(file_path),
                            "consistency_report": report
                        }
                        
                        # Thêm thông tin thời gian nếu có
                        if 'timestamp' in df_to_save.columns:
                            metadata["start_date"] = df_to_save['timestamp'].min().isoformat()
                            metadata["end_date"] = df_to_save['timestamp'].max().isoformat()
                        
                        metadata_path = output_dir / f"{filename}_metadata.json"
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
                        
                        self.logger.info(f"Đã lưu metadata cho {symbol}")
                
                except Exception as e:
                    self.logger.error(f"Lỗi khi lưu file cho {symbol}: {str(e)}")
                    traceback.print_exc()
                    
            except Exception as e:
                self.logger.error(f"Lỗi khi xử lý dữ liệu cho {symbol}: {str(e)}")
                traceback.print_exc()
        
        # Lưu báo cáo tổng hợp
        summary = {
            "timestamp": timestamp,
            "saved_symbols": len(results),
            "total_symbols": len(data),
            "output_dir": str(output_dir),
            "file_format": file_format,
            "consistency_summary": {
                symbol: report.get("overall_status", "unknown") 
                for symbol, report in consistency_reports.items()
            }
        }
        
        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu tổng cộng {len(results)}/{len(data)} file dữ liệu")
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
        is_futures: bool = False,
        preserve_timestamp: bool = True,
        sentiment_dir: Optional[Union[str, Path]] = None,
        include_sentiment: bool = None
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
        # Đảm bảo thư mục sentiment tồn tại
        sentiment_dir_path = self.data_dir / "sentiment"
        if not sentiment_dir_path.exists():
            sentiment_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Đã tạo thư mục dữ liệu tâm lý: {sentiment_dir_path}")
        else:
            sentiment_files = list(sentiment_dir_path.glob("*.csv")) + list(sentiment_dir_path.glob("*.parquet"))
            if not sentiment_files:
                self.logger.warning("Thư mục sentiment không có file dữ liệu nào.")
                self.logger.warning("Bạn có thể tạo dữ liệu tâm lý bằng lệnh:")
                self.logger.warning("python main.py collect fear_greed --output-dir data/sentiment")        
        
        # Cập nhật cấu hình nếu có tham số include_sentiment
        if include_sentiment is not None:
            if "collectors" not in self.config:
                self.config["collectors"] = {}
            self.config["collectors"]["include_sentiment"] = include_sentiment
            self.logger.info(f"Đã cập nhật cấu hình: include_sentiment={include_sentiment}")

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
                {"name": "collect_binance_sentiment", "enabled": exchange_id is not None, "params": {
                    "output_dir": str(self.data_dir / "sentiment" / "binance"),
                    "save_format": "parquet"
                }},                
                {"name": "load_data", "enabled": input_files is not None},
                {"name": "clean_technical_features", "enabled": True, "params": {
                    "fix_outliers": True,
                    "outlier_method": "zscore",
                    "outlier_threshold": 3.0,
                    "normalize_indicators": True,
                    "standardize_indicators": True,
                    "add_trend_strength": True,
                    "add_volatility_zscore": True
                }},
                {"name": "clean_and_standardize_indicators", "enabled": True, "params": {
                    "fix_outliers": True,
                    "normalize_indicators": True,
                    "standardize_indicators": True,
                    "add_trend_strength": True,
                    "add_volatility_zscore": True,
                    "generate_labels": True,
                    "label_config": {
                        "label_window": 10,
                        "label_threshold": 0.01
                    }
                }},
                {"name": "generate_features", "enabled": self.config.get("feature_engineering", {}).get("enabled", True), "params": {
                    "all_indicators": True,
                    "clean_indicators": True,
                    "feature_configs": {
                        "_default_": {
                            "generate_labels": True,
                            "label_window": 10,
                            "label_threshold": 0.01
                        }
                    }
                }},
                {"name": "calculate_fear_greed", "enabled": True},
                {"name": "remove_redundant_indicators", "enabled": True, "params": {
                    "correlation_threshold": 0.95,
                    "redundant_groups": [
                        ['macd_line', 'macd_signal', 'macd_histogram'],
                        ['atr_14', 'atr_pct_14', 'atr_norm_14', 'atr_norm_14_std'],
                        ['bb_middle_20', 'sma_20', 'bb_upper_20', 'bb_lower_20', 'bb_percent_b_20'],
                        ['plus_di_14', 'minus_di_14', 'adx_14'],
                        ['volume', 'volume_log'],  # Thêm cặp volume và volume_log
                        ['rsi_14', 'rsi_14_norm']  # Thêm cặp rsi và rsi chuẩn hóa
                    ]
                }},
                {"name": "create_target_features", "enabled": True, "params": {
                    "price_column": "close",
                    "target_types": ["direction", "return", "volatility"],
                    "horizons": [1, 3, 5, 10],
                    "threshold": 0.001
                }},
                {"name": "merge_sentiment", "enabled": self.config.get("collectors", {}).get("include_sentiment", True), "params": {
                    "sentiment_dir": sentiment_dir if sentiment_dir else str(self.data_dir / "sentiment"),
                    "method": "last_value",
                    "window": "1D"
                }},
                {"name": "check_consistency", "enabled": True},
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
                    # Thu thập dữ liệu với phương thức cải tiến
                    collected_data = await self.collect_data_improved(
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
                        handle_extreme_volume=step_params.get("handle_extreme_volume", True),
                        preserve_timestamp=preserve_timestamp,  # Truyền tham số
                        **{k: v for k, v in step_params.items() if k not in ["handle_leading_nan", "leading_nan_method", "min_periods"]}
                    )
                    current_data = cleaned_data
                    step_results["clean_data"] = cleaned_data
                
                elif step_name == "clean_and_standardize_indicators":
                    # Làm sạch và chuẩn hóa các chỉ báo kỹ thuật
                    cleaned_indicators = self.clean_and_standardize_indicators(
                        current_data,
                        **step_params
                    )
                    current_data = cleaned_indicators
                    step_results["clean_and_standardize_indicators"] = cleaned_indicators

                elif step_name == "generate_features":
                    # Tạo đặc trưng
                    featured_data = self.generate_features(
                        current_data,
                        feature_configs=step_params.get("feature_configs", None),
                        use_pipeline=step_params.get("use_pipeline", None),
                        fit_pipeline=step_params.get("fit_pipeline", True),
                        all_indicators=step_params.get("all_indicators", True),
                        clean_indicators=step_params.get("clean_indicators", True),
                        preserve_timestamp=preserve_timestamp  # Truyền tham số
                    )
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
                    
                    # Tìm kiếm dữ liệu tâm lý trong current_data
                    sentiment_keys = [k for k in current_data.keys() if k.lower().endswith('_sentiment') or k.lower() == 'sentiment']
                    
                    if sentiment_keys:
                        # Tách dữ liệu tâm lý và dữ liệu thị trường
                        market_data = {k: v for k, v in current_data.items() if k not in sentiment_keys}
                        isolated_sentiment = {k: current_data[k] for k in sentiment_keys}
                        
                        # Kết hợp dữ liệu
                        merged_data = self.merge_sentiment_data(
                            market_data=market_data,
                            sentiment_data=isolated_sentiment,
                            **{k: v for k, v in step_params.items() if k != "sentiment_data"}
                        )
                        current_data = merged_data
                        step_results["merge_sentiment"] = merged_data

                        # Kiểm tra chất lượng dữ liệu tâm lý
                        for symbol, df in current_data.items():
                            if 'sentiment_value' in df.columns and 'sentiment_change' in df.columns:
                                quality_metrics = self.check_sentiment_quality(df, 'sentiment_value')
                                
                                # Log thông tin chất lượng
                                self.logger.info(f"Chất lượng dữ liệu tâm lý cho {symbol}: {quality_metrics}")
                                
                                # Xử lý lại nếu chất lượng kém
                                if quality_metrics.get('quality') == 'poor':
                                    self.logger.warning(f"Phát hiện dữ liệu tâm lý chất lượng kém cho {symbol}, đang tái xử lý...")
                                    
                                    # Tính lại sentiment_change
                                    df['sentiment_change'] = df['sentiment_value'].diff()
                                    
                                    # Thêm tỷ lệ % thay đổi
                                    df['sentiment_pct_change'] = df['sentiment_value'].pct_change() * 100
                                    
                                    # Thêm chỉ báo mức biến thiên tâm lý
                                    df['sentiment_volatility'] = df['sentiment_value'].rolling(window=5).std()
                                    
                                    # Nếu vẫn còn quá nhiều giá trị 0, thêm nhiễu nhỏ
                                    if (df['sentiment_change'] == 0).mean() > 0.7:
                                        self.logger.warning(f"Áp dụng kỹ thuật jittering cho {symbol}")
                                        np.random.seed(42)  # Đảm bảo khả năng tái tạo
                                        small_noise = np.random.normal(0, df['sentiment_value'].std() * 0.001, size=len(df))
                                        df['sentiment_value_enhanced'] = df['sentiment_value'] + small_noise
                                        df['sentiment_change'] = df['sentiment_value_enhanced'].diff()
                                        
                                        # Kiểm tra lại
                                        zero_ratio = (df['sentiment_change'] == 0).mean()
                                        self.logger.info(f"Sau khi tái xử lý: {zero_ratio:.2%} giá trị sentiment_change = 0")

                    elif sentiment_data is not None:
                        # Sử dụng dữ liệu tâm lý được cung cấp
                        merged_data = self.merge_sentiment_data(
                            market_data=current_data,
                            sentiment_data=sentiment_data,
                            **{k: v for k, v in step_params.items() if k != "sentiment_data"}
                        )
                        current_data = merged_data
                        step_results["merge_sentiment"] = merged_data
                    else:
                        # Sử dụng thư mục dữ liệu tâm lý nếu được cung cấp
                        sentiment_dir_param = step_params.get("sentiment_dir")
                        if sentiment_dir_param:
                            merged_data = self.merge_sentiment_data(
                                market_data=current_data,
                                sentiment_dir=sentiment_dir_param,
                                **{k: v for k, v in step_params.items() if k != "sentiment_dir"}
                            )
                            current_data = merged_data
                            step_results["merge_sentiment"] = merged_data
                        else:
                            self.logger.info("Không có dữ liệu tâm lý để kết hợp")
               
                elif step_name == "check_consistency":
                    # Kiểm tra tính nhất quán dữ liệu
                    consistency_report = self.check_data_consistency(current_data)
                    step_results["check_consistency"] = consistency_report
                    
                    # Log kết quả kiểm tra tính nhất quán
                    failed_symbols = [symbol for symbol, report in consistency_report.items() 
                                    if report.get('overall_status') == 'failed']
                    if failed_symbols:
                        self.logger.warning(f"Có {len(failed_symbols)} symbol không đạt yêu cầu nhất quán: {failed_symbols}")
                    
                    # Không thay đổi dữ liệu ở bước này
                
                elif step_name == "save_data":
                    # Lưu dữ liệu
                    if output_dir is not None:
                        step_params["output_dir"] = output_dir
                        
                    saved_paths = self.save_data(
                        current_data,
                        preserve_timestamp=preserve_timestamp,  # Truyền tham số
                        **step_params
                    )
                    step_results["save_data"] = saved_paths
                
                elif step_name == "collect_binance_sentiment":
                    # Thu thập dữ liệu tâm lý từ Binance
                    asset = symbols[0].split('/')[0] if '/' in symbols[0] else symbols[0]
                    self.logger.info(f"Bước thu thập dữ liệu tâm lý từ Binance cho {asset}")
                    
                    from cli.commands.collect_commands import CollectCommands
                    cmd = CollectCommands()

                elif step_name == "calculate_fear_greed":
                    # Tính toán đặc trưng tâm lý thị trường dựa trên hành động giá (Fear & Greed)
                    fear_greed_data = self.calculate_fear_greed_features(current_data)
                    current_data = fear_greed_data
                    step_results["calculate_fear_greed"] = fear_greed_data
                    self.logger.info("Đã tính toán đặc trưng tâm lý thị trường (Fear & Greed)")                

                    # Chuyển datetime sang string format YYYY-MM-DD
                    start_date = start_time.strftime("%Y-%m-%d") if isinstance(start_time, datetime) else start_time
                    end_date = end_time.strftime("%Y-%m-%d") if isinstance(end_time, datetime) else end_time
                    
                    # Thu thập dữ liệu tâm lý
                    binance_sentiment_results = await cmd.collect_binance_sentiment(
                        exchange_id=exchange_id,
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        output_dir=step_params.get("output_dir", self.data_dir / "sentiment" / "binance"),
                        save_format=step_params.get("save_format", 'parquet')
                    )
                    
                    if binance_sentiment_results:
                        # Đọc các file đã lưu và thêm vào dữ liệu hiện tại
                        sentiment_data = {}
                        for symbol, file_path in binance_sentiment_results.items():
                            try:
                                sentiment_df = pd.read_parquet(file_path)
                                asset_name = symbol.split('/')[0] if '/' in symbol else symbol
                                sentiment_key = f"{asset_name}_sentiment"
                                sentiment_data[sentiment_key] = sentiment_df
                                self.logger.info(f"Đã đọc {len(sentiment_df)} dòng dữ liệu tâm lý cho {asset_name}")
                            except Exception as e:
                                self.logger.error(f"Lỗi khi đọc file dữ liệu tâm lý: {str(e)}")
                        
                        # Thêm dữ liệu tâm lý vào kết quả
                        if sentiment_data:
                            current_data.update(sentiment_data)
                            step_results["collect_binance_sentiment"] = sentiment_data
                    else:
                        self.logger.warning("Không thu thập được dữ liệu tâm lý từ Binance")

            except Exception as e:
                self.logger.error(f"Lỗi khi thực hiện bước '{step_name}': {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Lưu trạng thái pipeline nếu cần
        if save_results:
            self.save_pipeline_state()
        
        return current_data

    def calculate_fear_greed_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Tính toán đặc trưng tâm lý thị trường dựa trên hành động giá (Fear & Greed).
        
        Args:
            data: Dict với key là symbol và value là DataFrame
                
        Returns:
            Dict với key là symbol và value là DataFrame đã tính toán đặc trưng tâm lý
        """
        try:
            # Import module Fear & Greed
            from data_processors.feature_engineering.sentiment_features.create_fear_greed import calculate_market_sentiment_features
            
            results = {}
            for symbol, df in data.items():
                # Bỏ qua dữ liệu tâm lý
                if symbol.lower().endswith('_sentiment') or symbol.lower() == 'sentiment':
                    results[symbol] = df
                    continue
                
                # Tính toán đặc trưng tâm lý
                self.logger.info(f"Đang tính toán đặc trưng tâm lý thị trường cho {symbol}...")
                
                # Xử lý trước dữ liệu để tính các chỉ báo cần thiết nếu chưa có
                enhanced_df = df.copy()
                
                # 1. Tính Bollinger Bands Width nếu chưa có
                if 'bbw_20' not in enhanced_df.columns:
                    if all(col in enhanced_df.columns for col in ['bb_upper_20', 'bb_lower_20', 'bb_middle_20']):
                        enhanced_df['bbw_20'] = (enhanced_df['bb_upper_20'] - enhanced_df['bb_lower_20']) / enhanced_df['bb_middle_20']
                    else:
                        try:
                            from data_processors.feature_engineering.technical_indicator.trend_indicators import calculate_bollinger_bands
                            bbands_df = calculate_bollinger_bands(enhanced_df, timeperiod=20)
                            if bbands_df is not None:
                                enhanced_df['bbw_20'] = (bbands_df['bb_upper_20'] - bbands_df['bb_lower_20']) / bbands_df['bb_middle_20']
                        except Exception as e:
                            self.logger.warning(f"Không thể tính Bollinger Bands Width: {str(e)}")
                
                # 2. Tính Rate of Change (ROC) nếu chưa có
                roc_periods = [1, 5, 10, 20]
                for period in roc_periods:
                    roc_col = f'roc_{period}'
                    if roc_col not in enhanced_df.columns and 'close' in enhanced_df.columns:
                        try:
                            from data_processors.feature_engineering.technical_indicator.momentum_indicators import calculate_rate_of_change
                            roc_df = calculate_rate_of_change(enhanced_df, timeperiod=period)
                            if roc_df is not None and roc_col in roc_df.columns:
                                enhanced_df[roc_col] = roc_df[roc_col]
                            else:
                                # Tính thủ công nếu hàm không hoạt động
                                enhanced_df[roc_col] = enhanced_df['close'].pct_change(period) * 100
                        except Exception as e:
                            self.logger.warning(f"Không thể tính ROC-{period}: {str(e)}")
                            # Tính thủ công
                            enhanced_df[roc_col] = enhanced_df['close'].pct_change(period) * 100
                
                # Phát hiện timeframe từ khoảng cách thời gian nếu có timestamp
                lookback_period = 90  # Mặc định
                if 'timestamp' in enhanced_df.columns and len(enhanced_df) > 1:
                    time_diff = pd.to_datetime(enhanced_df['timestamp'].iloc[1]) - pd.to_datetime(enhanced_df['timestamp'].iloc[0])
                    time_diff_seconds = time_diff.total_seconds()
                    if time_diff_seconds <= 3600:  # 1 giờ
                        lookback_period = 90 * 24  # 90 ngày * 24 giờ
                    elif time_diff_seconds <= 86400:  # 1 ngày
                        lookback_period = 90
                    else:
                        lookback_period = 30
                
                # Tính toán đặc trưng
                final_df = calculate_market_sentiment_features(
                    df=enhanced_df,
                    column='close',
                    lookback_period=lookback_period,
                    prefix='sentiment_'
                )
                
                results[symbol] = final_df
                
                # Log số lượng đặc trưng được thêm vào
                added_cols = set(final_df.columns) - set(df.columns)
                if added_cols:
                    self.logger.info(f"Đã thêm {len(added_cols)} đặc trưng tâm lý cho {symbol}: {', '.join(added_cols)}")
                
            return results
                
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán đặc trưng tâm lý: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Trả về dữ liệu gốc nếu có lỗi
            return data        

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

def check_sentiment_quality(self, df, column='sentiment_value'):
    """
    Kiểm tra chất lượng dữ liệu tâm lý.
    
    Args:
        df: DataFrame chứa dữ liệu tâm lý
        column: Tên cột tâm lý cần kiểm tra
        
    Returns:
        Dict chứa các chỉ số chất lượng
    """
    if column not in df.columns:
        self.logger.warning(f"Không tìm thấy cột {column} trong DataFrame")
        return {}
    
    quality_metrics = {}
    
    # Tính tỷ lệ giá trị không thay đổi
    if len(df) > 1:
        diff_series = df[column].diff()
        zero_diff_ratio = (diff_series == 0).mean()
        quality_metrics['zero_diff_ratio'] = zero_diff_ratio
        
        # Tính số lượng giá trị 0 liên tiếp tối đa
        zero_streaks = []
        current_streak = 0
        
        for val in diff_series:
            if val == 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    zero_streaks.append(current_streak)
                current_streak = 0
                
        if current_streak > 0:
            zero_streaks.append(current_streak)
            
        max_zero_streak = max(zero_streaks) if zero_streaks else 0
        quality_metrics['max_zero_streak'] = max_zero_streak
        
        # Độ biến thiên
        std = df[column].std()
        mean = df[column].mean()
        
        if mean != 0:
            variation_coef = std / abs(mean)
            quality_metrics['variation_coef'] = variation_coef
        
        # Đánh giá
        if zero_diff_ratio > 0.7:
            quality_metrics['quality'] = 'poor'
            self.logger.warning(f"Chất lượng dữ liệu tâm lý kém: {zero_diff_ratio:.2%} giá trị không thay đổi")
            
            if max_zero_streak > 10:
                self.logger.warning(f"Phát hiện {max_zero_streak} giá trị 0 liên tiếp trong sentiment_change")
        elif zero_diff_ratio > 0.5:
            quality_metrics['quality'] = 'moderate'
            self.logger.info(f"Chất lượng dữ liệu tâm lý trung bình: {zero_diff_ratio:.2%} giá trị không thay đổi")
        else:
            quality_metrics['quality'] = 'good'
            self.logger.info(f"Chất lượng dữ liệu tâm lý tốt: chỉ {zero_diff_ratio:.2%} giá trị không thay đổi")
    
    return quality_metrics

def main():
    """
    Hàm main để chạy thử nghiệm.
    """
    asyncio.run(run_test())

if __name__ == "__main__":
    # Chạy hàm main
    main()