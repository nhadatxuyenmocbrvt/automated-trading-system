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
                        
                        # Thu thập OHLCV
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
            sentiment_results = await self._collect_sentiment_data(symbols, start_time, end_time)
            
            if sentiment_results:
                # Thêm dữ liệu tâm lý vào kết quả
                results.update(sentiment_results)
        
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
            
            # Lấy asset từ symbol đầu tiên (VD: "BTC/USDT" -> "BTC")
            asset = symbols[0].split('/')[0] if '/' in symbols[0] else symbols[0]
            
            self.logger.info(f"Thu thập dữ liệu tâm lý cho asset {asset}")
            
            # SỬA CHỮA: Thay vì thu thập từ Fear & Greed Index, thu thập từ Binance
            from cli.commands.collect_commands import CollectCommands
            
            # Khởi tạo CollectCommands
            cmd = CollectCommands()
            
            # Thu thập dữ liệu tâm lý từ Binance
            self.logger.info(f"Thu thập dữ liệu tâm lý từ Binance cho {asset}")
            
            # Chuyển datetime sang string format YYYY-MM-DD cho start_date và end_date
            start_date = start_time.strftime("%Y-%m-%d") if start_time else None
            end_date = end_time.strftime("%Y-%m-%d") if end_time else None
            
            # Thu thập dữ liệu tâm lý từ Binance
            binance_sentiment = await cmd.collect_binance_sentiment(
                exchange_id="binance",
                symbols=[symbol for symbol in symbols if asset in symbol],
                start_date=start_date,
                end_date=end_date,
                output_dir=self.data_dir / "sentiment" / "binance",
                save_format='parquet'
            )
            
            if binance_sentiment and any(binance_sentiment.values()):
                # Tạo DataFrame từ kết quả
                for symbol, file_path in binance_sentiment.items():
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
                
                # Vẫn thử thu thập từ các nguồn khác nếu có (giữ lại code cũ)
                sentiment_collector = self.collectors.get("sentiment")
                if sentiment_collector is not None:
                    try:
                        sentiment_data = await sentiment_collector.collect_from_all_sources(asset=asset)
                        
                        if any(sentiment_data.values()):
                            self.logger.info(f"Đã thu thập dữ liệu tâm lý từ các nguồn khác cho {asset}")
                            # Xử lý dữ liệu như code hiện tại
                            all_sentiment = []
                            for source_name, source_data in sentiment_data.items():
                                if source_data:
                                    self.logger.info(f"Đã thu thập {len(source_data)} mục dữ liệu tâm lý từ nguồn {source_name}")
                                    all_sentiment.extend(source_data)
                            
                            if all_sentiment:
                                sentiment_df = sentiment_collector.convert_to_dataframe(all_sentiment)
                                sentiment_key = f"{asset}_sentiment"
                                sentiment_results[sentiment_key] = sentiment_df
                    except Exception as e:
                        self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý từ các nguồn khác: {str(e)}")
        
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
        
        # 5. Xử lý giá trị thiếu (NaN)
        missing_values = df[value_column].isna().sum()
        if missing_values > 0:
            self.logger.info(f"Phát hiện {missing_values} giá trị thiếu trong cột {value_column}")
            
            # Interpolate (nội suy) các giá trị thiếu
            df[value_column] = df[value_column].interpolate(method='time')
            
            # Nếu còn giá trị NaN ở đầu hoặc cuối, sử dụng forward/backward fill
            df[value_column] = df[value_column].fillna(method='ffill').fillna(method='bfill')
            
            remaining_na = df[value_column].isna().sum()
            if remaining_na > 0:
                self.logger.warning(f"Vẫn còn {remaining_na} giá trị thiếu sau khi xử lý")
        
        # 6. Chuẩn hóa giá trị nếu cần
        if pd.api.types.is_numeric_dtype(df[value_column]):
            # Chuẩn hóa về khoảng [0, 1] hoặc [-1, 1] tùy theo loại dữ liệu
            min_val = df[value_column].min()
            max_val = df[value_column].max()
            
            if min_val < 0 or max_val > 1:
                # Tạo cột mới cho giá trị chuẩn hóa
                norm_col = f"{value_column}_normalized"
                
                # Chuẩn hóa MinMax
                df[norm_col] = (df[value_column] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                self.logger.info(f"Đã chuẩn hóa giá trị tâm lý về khoảng [0, 1] trong cột {norm_col}")
        
        # 7. Sắp xếp theo timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        self.logger.info(f"Hoàn thành làm sạch dữ liệu tâm lý: {len(df)} dòng")
        
        return df
    
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
                # Lưu cột timestamp nếu có
                timestamp_col = None
                if preserve_timestamp and 'timestamp' in df.columns:
                    # Đảm bảo timestamp là datetime trước khi lưu
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df = self._convert_timestamp(df)
                    timestamp_col = df['timestamp'].copy()

                if handle_leading_nan:
                    df = self.missing_data_handler.handle_leading_nan(
                        df,
                        columns=None,
                        method=leading_nan_method,
                        min_periods=min_periods
                    )
                    self.logger.info(f"Đã xử lý giá trị NaN ở đầu cho {symbol} bằng phương pháp {leading_nan_method}")


                if aggressive_nan_handling:
                    nan_count_before = df.isna().sum().sum()
                    
                    if nan_count_before > 0:
                        self.logger.info(f"Còn {nan_count_before} giá trị NaN sau khi xử lý ban đầu cho {symbol}, tiến hành xử lý triệt để")
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if fill_method == 'ffill+bfill':
                            # Kết hợp forward fill và backward fill
                            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
                        elif fill_method == 'interpolate':
                            # Sử dụng nội suy tuyến tính và sau đó ffill/bfill cho các giá trị còn thiếu
                            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
                        elif fill_method == 'mean':
                            # Sử dụng giá trị trung bình của cột
                            for col in numeric_cols:
                                df[col] = df[col].fillna(df[col].mean())
                        
                        # Nếu vẫn còn NaN (ít xảy ra) nhưng cần đảm bảo không còn NaN nào
                        if fill_all_nan and df[numeric_cols].isna().sum().sum() > 0:
                            # Sử dụng zero hoặc một giá trị hợp lý cho cột tương ứng
                            for col in numeric_cols:
                                if df[col].isna().sum() > 0:
                                    # Với giá, sử dụng giá đóng cửa gần nhất
                                    if col in ['open', 'high', 'low', 'close']:
                                        if 'close' in df.columns and not df['close'].isna().all():
                                            last_valid = df['close'].dropna().iloc[-1]
                                            df[col] = df[col].fillna(last_valid)
                                        else:
                                            # Nếu không có giá đóng cửa, sử dụng trung bình
                                            col_mean = df[col].mean()
                                            if np.isnan(col_mean):  # Nếu trung bình là NaN
                                                col_mean = 0.0
                                            df[col] = df[col].fillna(col_mean)
                                    # Với volume, sử dụng 0
                                    elif col == 'volume':
                                        df[col] = df[col].fillna(0)
                                    # Với các cột khác, sử dụng trung bình
                                    else:
                                        col_mean = df[col].mean()
                                        if np.isnan(col_mean):  # Nếu trung bình là NaN
                                            col_mean = 0.0
                                        df[col] = df[col].fillna(col_mean)
                        
                        # Áp dụng MissingDataHandler để xử lý triệt để NaN
                        df = self.missing_data_handler.handle_missing_values(
                            df,
                            columns=df.columns
                        )

                        # Kiểm tra lại số lượng NaN sau khi xử lý
                        nan_count_after = df.isna().sum().sum()
                        self.logger.info(f"Đã xử lý {nan_count_before - nan_count_after} giá trị NaN cho {symbol}, còn lại {nan_count_after} giá trị NaN")
                        
                        nan_columns = df.columns[df.isna().any()].tolist()
                        if len(nan_columns) > 0:
                            self.logger.warning(f"Dữ liệu {symbol} có cảnh báo: {len(nan_columns)} cột chứa giá trị NaN: {nan_columns}")

                        # Kiểm tra và xử lý triệt để các giá trị NaN còn lại
                        nan_columns = df.columns[df.isna().any()].tolist()
                        if nan_columns:
                            self.logger.warning(f"Vẫn còn {len(nan_columns)} cột chứa NaN sau khi xử lý: {nan_columns}")
                            
                            # Liệt kê số lượng NaN trong mỗi cột
                            nan_counts = df[nan_columns].isna().sum()
                            self.logger.warning(f"Số lượng NaN trong mỗi cột: {nan_counts}")
                            
                            # Xử lý NaN cho tất cả cột
                            for col in nan_columns:
                                # Chọn phương pháp phù hợp tùy theo loại cột
                                if 'rank' in col or 'percentile' in col:
                                    # Với cột xếp hạng, thường nên sử dụng backfill
                                    df[col] = df[col].fillna(method='bfill').fillna(0)
                                elif 'atr' in col or 'volatility' in col:
                                    # Với cột biến động, có thể sử dụng giá trị trung vị
                                    median_val = df[col].median()
                                    if pd.isna(median_val):  # Nếu median là NaN
                                        df[col] = df[col].fillna(0)
                                    else:
                                        df[col] = df[col].fillna(median_val)
                                else:
                                    # Xử lý các cột khác
                                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill').fillna(0)
                            
                            # Kiểm tra lại sau khi xử lý
                            remaining_nan = df.isna().sum().sum()
                            if remaining_nan > 0:
                                self.logger.warning(f"Vẫn còn {remaining_nan} giá trị NaN sau khi xử lý. Điền tất cả bằng 0.")
                                df = df.fillna(0)  # Điền tất cả NaN còn lại bằng 0
                            else:
                                self.logger.info("Đã xử lý thành công tất cả giá trị NaN.")                        
                
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

            except Exception as e:
                self.logger.error(f"Lỗi khi làm sạch dữ liệu cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                # Trả về dữ liệu gốc nếu có lỗi
                results[symbol] = df
    
        return results
    
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
                
                # THÊM ĐOẠN CODE NÀY: Làm sạch và chuẩn hóa các chỉ báo kỹ thuật trước khi tạo đặc trưng
                if clean_indicators:
                    # Import các hàm cần thiết
                    from data_processors.utils.preprocessing import clean_technical_features
                    
                    # Cấu hình làm sạch chỉ báo
                    clean_config = {
                        'fix_outliers': True,
                        'outlier_method': 'zscore',
                        'outlier_threshold': 3.0,
                        'normalize_indicators': True,
                        'standardize_indicators': True,
                        'add_trend_strength': True,
                        'add_volatility_zscore': True,
                        'generate_labels': symbol_config.get("generate_labels", False),
                        'label_window': symbol_config.get("label_window", 10),
                        'label_threshold': symbol_config.get("label_threshold", 0.01)
                    }

                    # Áp dụng clean_technical_features
                    df = clean_technical_features(df, clean_config)
                    self.logger.info(f"Đã làm sạch và chuẩn hóa các chỉ báo kỹ thuật cho {symbol} trước khi tạo đặc trưng")                    

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
                
                # Khôi phục timestamp nếu đã lưu
                if timestamp_col is not None:
                    features_df.insert(0, 'timestamp', timestamp_col)
                
                results[symbol] = features_df
                self.logger.info(f"Đã tạo đặc trưng cho {symbol}: {df.shape[1]} -> {features_df.shape[1]} cột")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo đặc trưng cho {symbol}: {str(e)}")
                self.logger.error(traceback.format_exc())
                results[symbol] = df
        
        return results
    
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
        Kết hợp dữ liệu thị trường với dữ liệu tâm lý.
        
        Args:
            market_data: Dict với key là symbol và value là DataFrame thị trường
            sentiment_data: DataFrame dữ liệu tâm lý hoặc Dict chứa dữ liệu tâm lý theo symbol
                            hoặc đường dẫn đến file dữ liệu tâm lý chung
            sentiment_dir: Thư mục chứa các file dữ liệu tâm lý riêng cho từng symbol
            method: Phương pháp kết hợp ('last_value', 'mean', 'weighted_mean')
            window: Kích thước cửa sổ thời gian ('1D', '12H', '4H', ...)
            
        Returns:
            Dict với key là symbol và value là DataFrame đã kết hợp
        """
        results = {}
        sentiment_by_symbol = {}
        
        if market_data is None or not market_data:
            self.logger.warning("Không có dữ liệu thị trường để kết hợp")
            return {}
        
        # Xử lý các trường hợp cung cấp dữ liệu tâm lý
        if sentiment_data is not None:
            if isinstance(sentiment_data, pd.DataFrame):
                # Trường hợp 1: sentiment_data là một DataFrame duy nhất
                self.logger.info("Sử dụng DataFrame tâm lý được cung cấp trực tiếp")
                sentiment_data = self.clean_sentiment_data(sentiment_data)  # Làm sạch dữ liệu tâm lý
                common_sentiment_data = sentiment_data
            elif isinstance(sentiment_data, dict):
                # Trường hợp 2: sentiment_data là Dictionary chứa DataFrame theo symbol
                self.logger.info("Sử dụng Dictionary dữ liệu tâm lý theo symbol")
                # Làm sạch dữ liệu tâm lý cho từng symbol
                sentiment_by_symbol = {k: self.clean_sentiment_data(v) for k, v in sentiment_data.items()}
                common_sentiment_data = None
            elif isinstance(sentiment_data, (str, Path)):
                # Trường hợp 3: sentiment_data là đường dẫn file
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
                    
                    # Làm sạch dữ liệu tâm lý
                    if common_sentiment_data is not None:
                        common_sentiment_data = self.clean_sentiment_data(common_sentiment_data)
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
        
        # Tìm kiếm file dữ liệu tâm lý riêng cho từng symbol
        if sentiment_dir is not None:
            sentiment_dir_path = Path(sentiment_dir)
            
            # Đảm bảo thư mục tồn tại
            if not sentiment_dir_path.exists():
                sentiment_dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Đã tạo thư mục dữ liệu tâm lý: {sentiment_dir_path}")
            
            self.logger.info(f"Tìm kiếm dữ liệu tâm lý từ thư mục: {sentiment_dir_path}")
            
            # Lấy danh sách symbols
            for symbol in market_data.keys():
                # Chuẩn bị tên file dựa trên symbol
                asset = symbol.split('/')[0] if '/' in symbol else symbol
                
                # Sử dụng hàm find_sentiment_files để tìm kiếm file
                sentiment_files = self.find_sentiment_files(sentiment_dir_path, asset)
                
                if sentiment_files:
                    # Sử dụng file mới nhất
                    file_path = sentiment_files[0]
                    self.logger.info(f"Sử dụng file tâm lý: {file_path}")
                    
                    try:
                        # Đọc dữ liệu tâm lý
                        if file_path.suffix.lower() == '.csv':
                            df = pd.read_csv(file_path)
                            # Xử lý timestamp nếu có
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        elif file_path.suffix.lower() == '.parquet':
                            df = pd.read_parquet(file_path)
                        
                        # Kiểm tra dữ liệu đọc được
                        if df is not None and not df.empty:
                            # Làm sạch dữ liệu tâm lý
                            sentiment_by_symbol[symbol] = self.clean_sentiment_data(df)
                            self.logger.info(f"Đã đọc {len(df)} dòng dữ liệu tâm lý cho {symbol}")
                        else:
                            self.logger.warning(f"File tâm lý {file_path} rỗng hoặc không đọc được")
                    except Exception as e:
                        self.logger.error(f"Lỗi khi đọc file tâm lý {file_path}: {str(e)}")
                        traceback.print_exc()
                else:
                    # Nếu không tìm thấy file riêng, thử tìm file tâm lý chung
                    self.logger.info(f"Không tìm thấy file tâm lý riêng cho {symbol}, tìm file tâm lý chung...")
                    general_files = self.find_sentiment_files(sentiment_dir_path, "", include_subdirs=True)
                    
                    if general_files:
                        file_path = general_files[0]
                        self.logger.info(f"Sử dụng file tâm lý chung: {file_path}")
                        
                        try:
                            # Đọc dữ liệu tâm lý chung
                            if file_path.suffix.lower() == '.csv':
                                df = pd.read_csv(file_path)
                                if 'timestamp' in df.columns:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                            elif file_path.suffix.lower() == '.parquet':
                                df = pd.read_parquet(file_path)
                            
                            # Thêm cột asset nếu chưa có
                            if 'asset' not in df.columns:
                                df['asset'] = 'GENERAL'
                            
                            # Làm sạch dữ liệu tâm lý chung
                            sentiment_by_symbol[symbol] = self.clean_sentiment_data(df)
                            self.logger.info(f"Đã đọc {len(df)} dòng dữ liệu tâm lý chung cho {symbol}")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi đọc file tâm lý chung {file_path}: {str(e)}")
        
        # Xử lý từng symbol
        for symbol, df in market_data.items():
            try:
                # Đảm bảo timestamp là datetime
                if 'timestamp' in df.columns:
                    # Chuyển timestamp sang datetime nếu cần
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
                    self.logger.info(f"Sử dụng dữ liệu tâm lý riêng cho {symbol}")
                # Nếu không có, thử với symbol_sentiment
                elif f"{symbol}_sentiment" in sentiment_by_symbol:
                    symbol_sentiment = sentiment_by_symbol[f"{symbol}_sentiment"]
                    self.logger.info(f"Sử dụng dữ liệu tâm lý {symbol}_sentiment")
                # Nếu không có, thử tìm dữ liệu tâm lý cho asset
                elif '/' in symbol:
                    asset = symbol.split('/')[0]
                    asset_key = f"{asset}_sentiment"
                    if asset_key in sentiment_by_symbol:
                        symbol_sentiment = sentiment_by_symbol[asset_key]
                        self.logger.info(f"Sử dụng dữ liệu tâm lý asset {asset_key}")
                # Nếu không có, sử dụng dữ liệu chung
                elif common_sentiment_data is not None:
                    # Lọc dữ liệu tâm lý cho asset tương ứng
                    asset = symbol.split('/')[0] if '/' in symbol else symbol
                    
                    if 'asset' in common_sentiment_data.columns:
                        asset_mask = (
                            (common_sentiment_data['asset'] == asset) | 
                            (common_sentiment_data['asset'].str.lower() == asset.lower() if isinstance(asset, str) else False) |
                            (common_sentiment_data['asset'].isna())
                        )
                        symbol_sentiment = common_sentiment_data[asset_mask].copy()
                        self.logger.info(f"Lọc dữ liệu tâm lý chung cho {asset}: {len(symbol_sentiment)} dòng")
                    else:
                        # Nếu không có cột asset, sử dụng tất cả dữ liệu tâm lý
                        symbol_sentiment = common_sentiment_data.copy()
                        self.logger.info(f"Sử dụng dữ liệu tâm lý chung cho {symbol} (không có thông tin asset)")
                
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
                
                # Tính giá trị tâm lý theo phương pháp đã chọn
                if method == 'last_value':
                    # Lấy giá trị cuối cùng trong cửa sổ
                    resampled_sentiment = sentiment_df[value_column].resample(window).last()
                elif method == 'mean':
                    # Lấy giá trị trung bình trong cửa sổ
                    resampled_sentiment = sentiment_df[value_column].resample(window).mean()
                elif method == 'weighted_mean':
                    # Lấy giá trị trung bình có trọng số
                    if 'weight' in sentiment_df.columns:
                        weights = sentiment_df['weight']
                    else:
                        # Tính trọng số dựa trên thời gian
                        max_time = sentiment_df.index.max()
                        min_time = sentiment_df.index.min()
                        time_diff = (sentiment_df.index - min_time).total_seconds()
                        max_diff = (max_time - min_time).total_seconds() or 1  # Tránh chia cho 0
                        weights = time_diff / max_diff
                    
                    weighted_values = sentiment_df[value_column] * weights
                    resampled_sentiment = (weighted_values.resample(window).sum() / weights.resample(window).sum()).fillna(method='ffill')
                else:
                    self.logger.warning(f"Phương pháp không hợp lệ: {method}, sử dụng 'last_value'")
                    resampled_sentiment = sentiment_df[value_column].resample(window).last()
                
                # Kết quả kết hợp
                merged_result = pd.DataFrame(index=market_df.index)
                
                # Thêm dữ liệu thị trường
                for col in market_df.columns:
                    merged_result[col] = market_df[col]
                
                # Thêm giá trị tâm lý
                merged_result['sentiment_value'] = np.nan  # Khởi tạo cột trống
                
                # Tìm các ngày chung giữa dữ liệu tâm lý và dữ liệu thị trường
                # Cách này hiệu quả hơn pd.merge()
                common_dates = merged_result.index.intersection(resampled_sentiment.index)
                self.logger.info(f"Tìm thấy {len(common_dates)} ngày chung giữa dữ liệu thị trường và tâm lý")
                
                if len(common_dates) > 0:
                    # Gán giá trị tâm lý cho các ngày chung
                    merged_result.loc[common_dates, 'sentiment_value'] = resampled_sentiment.loc[common_dates]
                    
                    # Điền các giá trị thiếu bằng forward fill và backward fill
                    merged_result['sentiment_value'] = merged_result['sentiment_value'].fillna(method='ffill').fillna(method='bfill')
                    
                    # Đếm số lượng giá trị hợp lệ
                    valid_sentiment = merged_result['sentiment_value'].notna().sum()
                    self.logger.info(f"Số giá trị tâm lý hợp lệ: {valid_sentiment}/{len(merged_result)} ({valid_sentiment/len(merged_result)*100:.2f}%)")
                    
                    # Thêm các biến phái sinh tâm lý
                    # Lag values
                    for lag in [1, 3, 5]:
                        merged_result[f'sentiment_lag_{lag}'] = merged_result['sentiment_value'].shift(lag)
                    
                    # Thay đổi so với kỳ trước
                    merged_result['sentiment_change'] = merged_result['sentiment_value'].diff()
                    
                    # Trung bình động của tâm lý
                    for window_size in [5, 10, 20]:
                        merged_result[f'sentiment_ma_{window_size}'] = merged_result['sentiment_value'].rolling(window=window_size).mean()
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
        from data_processors.utils.preprocessing import clean_technical_features
        
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
        
        for symbol, df in data.items():
            self.logger.info(f"Kiểm tra tính nhất quán của dữ liệu cho {symbol}")
            
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
                if report["overall_status"] == "failed":
                    self.logger.error(f"Dữ liệu {symbol} có vấn đề nghiêm trọng: {', '.join(report['issues'])}")
                elif report["overall_status"] == "warning":
                    self.logger.warning(f"Dữ liệu {symbol} có cảnh báo: {', '.join(report['warnings'])}")
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
        sentiment_dir: Optional[Union[str, Path]] = None
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
                {"name": "clean_data", "enabled": True, "params": {
                    "handle_leading_nan": True,
                    "leading_nan_method": "backfill",
                    "min_periods": 5,
                    "handle_extreme_volume": True,
                    "configs": {
                        "ohlcv": {
                            "handle_gaps": True,
                            "handle_negative_values": True,
                            "verify_high_low": True,
                            "verify_open_close": True,
                            "handle_technical_indicators": True,
                            "generate_labels": True,
                            "label_window": 10,
                            "label_threshold": 0.01,
                            "outlier_threshold": 3.0
                        }
                    }
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
                self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý từ Binance: {str(e)}")
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


def main():
    """
    Hàm main để chạy thử nghiệm.
    """
    asyncio.run(run_test())

if __name__ == "__main__":
    # Chạy hàm main
    main()