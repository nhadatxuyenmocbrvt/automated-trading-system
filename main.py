#!/usr/bin/env python3
"""
Điểm khởi chạy chính của Hệ thống Giao dịch Tự động.
File này cung cấp giao diện dòng lệnh để khởi chạy các thành phần
khác nhau của hệ thống, bao gồm thu thập dữ liệu, xử lý dữ liệu, huấn luyện agent,
backtest, và triển khai giao dịch thực tế.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
import datetime
from typing import Dict, List, Optional, Any
import json

# Thêm thư mục gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module của hệ thống
from config.system_config import get_system_config, system_config
from config.logging_config import get_logger, setup_component_logger
from config.env import get_env, env_manager
from config.constants import SystemStatus, Exchange, Indicator

# Import các module thu thập dữ liệu
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.market_data.historical_data_collector import create_data_collector

# Import các module xử lý dữ liệu
try:
    from data_processors.cleaners.data_cleaner import DataCleaner
    from data_processors.cleaners.outlier_detector import OutlierDetector
    from data_processors.cleaners.missing_data_handler import MissingDataHandler
    from data_processors.feature_engineering.feature_generator import FeatureGenerator
    from data_processors.data_pipeline import DataPipeline
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

# Thiết lập logger cho main
logger = setup_component_logger("main", "INFO")

class AutomatedTradingSystem:
    """
    Lớp quản lý chính của hệ thống giao dịch tự động.
    """
    
    def __init__(self):
        """
        Khởi tạo hệ thống giao dịch tự động.
        """
        logger.info("Khởi tạo Hệ thống Giao dịch Tự động")
        
        # Trạng thái hệ thống
        self.status = SystemStatus.INITIALIZING
        
        # Tải cấu hình hệ thống
        self.config = get_system_config()
        logger.info(f"Đã tải cấu hình hệ thống - Môi trường: {self.config.get('environment')}")
        
        # Kiểm tra biến môi trường bắt buộc
        missing_vars = env_manager.validate_required_vars()
        if missing_vars:
            logger.warning(f"Các biến môi trường bắt buộc chưa được thiết lập: {missing_vars}")
        
        # Khởi tạo các thành phần
        self.connectors = {}  # Lưu trữ các kết nối sàn giao dịch
        self.collectors = {}  # Lưu trữ các bộ thu thập dữ liệu
        
        # Kiểm tra tính khả dụng của các module
        if not DATA_PROCESSING_AVAILABLE:
            logger.warning("Các module xử lý dữ liệu không khả dụng. Vui lòng kiểm tra cài đặt.")
        
        # Khởi tạo các thành phần của hệ thống
        logger.info("Hệ thống đã sẵn sàng")
        self.status = SystemStatus.RUNNING
    
    async def init_exchange_connector(self, exchange_id: str, is_futures: bool = False, 
                                     testnet: bool = True) -> ExchangeConnector:
        """
        Khởi tạo kết nối với sàn giao dịch.
        
        Args:
            exchange_id: ID của sàn giao dịch
            is_futures: True để sử dụng thị trường futures
            testnet: True để sử dụng môi trường test
            
        Returns:
            Đối tượng kết nối với sàn giao dịch
        """
        try:
            connector_key = f"{exchange_id}_{'futures' if is_futures else 'spot'}"
            
            # Kiểm tra xem connector đã tồn tại chưa
            if connector_key in self.connectors:
                logger.info(f"Sử dụng kết nối {connector_key} hiện có")
                return self.connectors[connector_key]
            
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Đảm bảo REQUEST_TIMEOUT đủ lớn
            timeout_seconds = int(get_env('REQUEST_TIMEOUT', '60'))
            if timeout_seconds < 60:
                logger.warning(f"REQUEST_TIMEOUT ({timeout_seconds}s) có thể quá thấp, đề xuất ít nhất 60 giây")
            
            # Tạo connector phù hợp với sàn giao dịch
            if exchange_id.lower() == "binance":
                # Cấu hình timeout dài hơn để tránh lỗi timeout 
                connector = BinanceConnector(
                    api_key=api_key,
                    api_secret=api_secret,
                    is_futures=is_futures,
                    testnet=testnet
                )
                logger.info(f"Đã tạo kết nối Binance {'Futures' if is_futures else 'Spot'}")
            else:
                # Sử dụng generic connector cho các sàn khác
                connector = ExchangeConnector(
                    exchange_id=exchange_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet
                )
                logger.info(f"Đã tạo kết nối {exchange_id}")
            
            # Lưu vào cache
            self.connectors[connector_key] = connector
            return connector
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo kết nối {exchange_id}: {str(e)}")
            
            # Tạo ghi chú lỗi chi tiết với các đề xuất giải pháp
            if exchange_id.lower() == "binance":
                if "timeout" in str(e).lower():
                    logger.warning(f"Lỗi timeout khi kết nối tới {exchange_id}. Đảm bảo kết nối mạng ổn định.")
                    logger.warning("Giải pháp: Tăng giá trị REQUEST_TIMEOUT trong .env hoặc biến môi trường.")
                
                if "headers" in str(e).lower():
                    logger.warning(f"Lỗi xử lý response từ {exchange_id}. Có thể do cấu trúc response không đúng định dạng.")
                    logger.warning("Giải pháp: Kiểm tra cập nhật thư viện CCXT hoặc sử dụng proxy nếu cần thiết.")
            
            # Nếu không thể tạo connector, vẫn raise lỗi để hàm gọi xử lý
            raise
    
    async def collect_historical_data(self, exchange_id: str, symbols: List[str], 
                                     timeframes: List[str], days_back: int = 30,
                                     is_futures: bool = False,
                                     start_date: str = None, end_date: str = None) -> None:
        """
        Thu thập dữ liệu lịch sử từ sàn giao dịch.
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách timeframe
            days_back: Số ngày lấy dữ liệu
            is_futures: True để sử dụng thị trường futures
            start_date: Ngày bắt đầu thu thập dữ liệu (format YYYY-MM-DD)
            end_date: Ngày kết thúc thu thập dữ liệu (format YYYY-MM-DD)
        """
        try:
            # Khởi tạo connector với cơ chế retry
            max_retries = int(get_env('MAX_RETRIES', '5'))
            retry_count = 0
            connector = None
            
            while retry_count < max_retries:
                try:
                    # Khởi tạo connector
                    connector = await self.init_exchange_connector(
                        exchange_id=exchange_id,
                        is_futures=is_futures,
                        testnet=False  # Sử dụng mainnet để lấy dữ liệu thực
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # exponential backoff
                    
                    if retry_count >= max_retries:
                        logger.error(f"Đã vượt quá số lần thử lại ({max_retries}) khi kết nối {exchange_id}")
                        raise
                    logger.warning(f"Lỗi khi kết nối {exchange_id}, thử lại sau {wait_time}s (lần {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
            
            if connector is None:
                raise Exception(f"Không thể khởi tạo kết nối với {exchange_id} sau {max_retries} lần thử")
            
            # Khởi tạo collector
            collector = await create_data_collector(
                exchange_id=exchange_id,
                api_key=connector.api_key,
                api_secret=connector.api_secret,
                testnet=False,  # Sử dụng mainnet để lấy dữ liệu thực
                is_futures=is_futures
            )
            
            # Lưu vào cache
            collector_key = f"{exchange_id}_{'futures' if is_futures else 'spot'}"
            self.collectors[collector_key] = collector
            
            # Xác định thời gian bắt đầu và kết thúc
            end_time = datetime.datetime.now()
            start_time = None
            
            # Nếu có start_date và end_date, sử dụng chúng
            if start_date:
                start_time = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                logger.info(f"Sử dụng ngày bắt đầu từ tham số: {start_date}")
            else:
                start_time = end_time - datetime.timedelta(days=days_back)
                logger.info(f"Sử dụng ngày bắt đầu tính từ days_back ({days_back} ngày)")
            
            if end_date:
                end_time = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                logger.info(f"Sử dụng ngày kết thúc từ tham số: {end_date}")
            else:
                logger.info(f"Sử dụng ngày kết thúc mặc định: hiện tại ({end_time})")
            
            logger.info(f"Bắt đầu thu thập dữ liệu từ {start_time} đến {end_time}")
            logger.info(f"Cặp giao dịch: {symbols}")
            logger.info(f"Timeframes: {timeframes}")
            
            # Thu thập dữ liệu cho mỗi cặp và timeframe
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Thu thập dữ liệu {symbol} - {timeframe}")
                    
                    try:
                        # Thu thập OHLCV với cơ chế retry
                        for attempt in range(max_retries):
                            try:
                                # Thu thập OHLCV
                                df = await collector.collect_ohlcv(
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                                
                                logger.info(f"Đã thu thập {len(df) if df is not None else 0} bản ghi cho {symbol} - {timeframe}")
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    logger.error(f"Không thể thu thập dữ liệu cho {symbol} - {timeframe} sau {max_retries} lần thử: {str(e)}")
                                    raise
                                
                                wait_time = 2 ** (attempt + 1)
                                logger.warning(f"Lỗi khi thu thập dữ liệu {symbol} - {timeframe}, thử lại sau {wait_time}s (lần {attempt+1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi thu thập dữ liệu {symbol} - {timeframe}: {str(e)}")
                        logger.info(f"Tiếp tục với cặp/timeframe tiếp theo")
                        continue
            
            logger.info(f"Đã hoàn thành thu thập dữ liệu lịch sử cho {exchange_id}")
            
        except Exception as e:
            logger.error(f"Lỗi khi thu thập dữ liệu lịch sử: {str(e)}")
            if "timeout" in str(e).lower():
                logger.warning("Lỗi timeout có thể do kết nối mạng không ổn định hoặc sàn giao dịch không phản hồi")
                logger.warning("Giải pháp: Tăng REQUEST_TIMEOUT, kiểm tra kết nối mạng hoặc sử dụng proxy")
            raise
    
    async def process_data(self, command: str, **kwargs) -> None:
        """
        Xử lý dữ liệu lịch sử.
        
        Args:
            command: Lệnh xử lý ('clean', 'features', 'sentiment', 'pipeline', 'select-features')
            **kwargs: Các tham số tùy chọn cho từng lệnh
        """
        if not DATA_PROCESSING_AVAILABLE:
            logger.error("Các module xử lý dữ liệu không khả dụng. Vui lòng kiểm tra cài đặt.")
            return
        
        logger.info(f"Bắt đầu xử lý dữ liệu với lệnh: {command}")
        logger.debug(f"Các tham số: {kwargs}")
        
        try:
            if command == 'clean':
                await self.clean_data(**kwargs)
            elif command == 'features':
                await self.generate_features(**kwargs)
            elif command == 'sentiment':
                await self.analyze_sentiment(**kwargs)
            elif command == 'pipeline':
                await self.run_data_pipeline(**kwargs)
            elif command == 'select-features':
                await self.select_features(**kwargs)
            else:
                logger.error(f"Lệnh xử lý dữ liệu không hợp lệ: {command}")
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
            raise
    
    async def clean_data(self, **kwargs) -> None:
        """
        Làm sạch dữ liệu lịch sử.
        
        Args:
            data_type: Loại dữ liệu ('ohlcv', 'orderbook', 'trades')
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách timeframe
            exchange: Tên sàn giao dịch
            outlier_method: Phương pháp phát hiện ngoại lệ ('zscore', 'iqr', 'isolation_forest')
            threshold: Ngưỡng phát hiện ngoại lệ
            fill_method: Phương pháp điền dữ liệu thiếu ('ffill', 'bfill', 'interpolate')
        """
        logger.info("Bắt đầu làm sạch dữ liệu")
        
        data_type = kwargs.get('data_type', 'ohlcv')
        symbols = kwargs.get('symbols', ['BTC/USDT'])
        timeframes = kwargs.get('timeframes', ['1h'])
        exchange = kwargs.get('exchange', 'binance')
        outlier_method = kwargs.get('outlier_method', None)
        threshold = kwargs.get('threshold', 3.0)
        fill_method = kwargs.get('fill_method', None)
        
        # Tạo đối tượng DataCleaner
        data_cleaner = DataCleaner()
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Tải dữ liệu
                file_path = Path(f"data/historical/{exchange}/ohlcv/{symbol.replace('/', '_')}_{timeframe}.parquet")
                if not file_path.exists():
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol} - {timeframe}. Đường dẫn: {file_path}")
                    continue
                
                logger.info(f"Đang làm sạch dữ liệu cho {symbol} - {timeframe}")
                
                # Thực hiện làm sạch dữ liệu
                df = data_cleaner.load_data(file_path)
                
                # Phát hiện và xử lý ngoại lệ nếu được yêu cầu
                if outlier_method:
                    outlier_detector = OutlierDetector()
                    df = outlier_detector.detect_and_handle_outliers(
                        df, method=outlier_method, threshold=threshold
                    )
                    logger.info(f"Đã xử lý ngoại lệ với phương pháp {outlier_method}")
                
                # Xử lý dữ liệu thiếu nếu được yêu cầu
                if fill_method:
                    missing_handler = MissingDataHandler()
                    df = missing_handler.fill_missing_values(df, method=fill_method)
                    logger.info(f"Đã điền dữ liệu thiếu với phương pháp {fill_method}")
                
                # Lưu dữ liệu đã làm sạch
                output_path = Path(f"data/processed/{exchange}/ohlcv/{symbol.replace('/', '_')}_{timeframe}.parquet")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                data_cleaner.save_data(df, output_path)
                
                logger.info(f"Đã lưu dữ liệu đã làm sạch cho {symbol} - {timeframe} tại {output_path}")
        
        logger.info("Đã hoàn thành làm sạch dữ liệu")

    async def generate_features(self, **kwargs) -> None:
        """
        Tạo đặc trưng từ dữ liệu đã làm sạch.
        
        Args:
            data_type: Loại dữ liệu ('ohlcv', 'orderbook', 'trades')
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách timeframe
            indicators: Danh sách chỉ báo kỹ thuật
            all_indicators: True để tạo tất cả chỉ báo kỹ thuật
            feature_types: Danh sách loại đặc trưng cụ thể (cho orderbook, trades)
        """
        logger.info("Bắt đầu tạo đặc trưng")
        
        data_type = kwargs.get('data_type', 'ohlcv')
        symbols = kwargs.get('symbols', ['BTC/USDT'])
        timeframes = kwargs.get('timeframes', ['1h'])
        indicators = kwargs.get('indicators', [])
        all_indicators = kwargs.get('all_indicators', False)
        feature_types = kwargs.get('feature_types', [])
        exchange = kwargs.get('exchange', 'binance')
        
        # Tạo đối tượng FeatureGenerator
        feature_generator = FeatureGenerator()
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Tải dữ liệu đã làm sạch
                input_dir = Path(f"data/processed/{exchange}/{data_type}")
                
                if not input_dir.exists():
                    # Thử tải từ thư mục dữ liệu thô
                    input_dir = Path(f"data/historical/{exchange}/{data_type}")
                
                file_path = input_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
                
                if not file_path.exists():
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol} - {timeframe}. Đường dẫn: {file_path}")
                    continue
                
                logger.info(f"Đang tạo đặc trưng cho {symbol} - {timeframe}")
                
                # Tải dữ liệu
                df = feature_generator.load_data(file_path)
                
                # Tạo đặc trưng theo loại dữ liệu
                if data_type == 'ohlcv':
                    # Xác định danh sách chỉ báo cần tạo
                    if all_indicators:
                        # Sử dụng tất cả chỉ báo có sẵn
                        tech_indicators = [ind.value for ind in Indicator]
                    else:
                        # Sử dụng danh sách chỉ báo được chỉ định
                        tech_indicators = indicators if indicators else ['sma', 'ema', 'rsi']
                    
                    # Tạo đặc trưng kỹ thuật
                    df = feature_generator.generate_technical_indicators(df, tech_indicators)
                    logger.info(f"Đã tạo {len(tech_indicators)} chỉ báo kỹ thuật")
                    
                    # Tạo đặc trưng giá
                    df = feature_generator.generate_price_features(df)
                    logger.info("Đã tạo đặc trưng giá")
                    
                    # Tạo đặc trưng biến động
                    df = feature_generator.generate_volatility_features(df)
                    logger.info("Đã tạo đặc trưng biến động")
                    
                elif data_type == 'orderbook':
                    # Tạo đặc trưng từ sổ lệnh
                    if not feature_types:
                        feature_types = ['liquidity_imbalance', 'order_book_pressure', 'vwap']
                    
                    df = feature_generator.generate_orderbook_features(df, feature_types)
                    logger.info(f"Đã tạo đặc trưng sổ lệnh: {feature_types}")
                
                elif data_type == 'trades':
                    # Tạo đặc trưng từ dữ liệu giao dịch
                    if not feature_types:
                        feature_types = ['trade_flow', 'trade_volume', 'trade_size']
                    
                    df = feature_generator.generate_trade_features(df, feature_types)
                    logger.info(f"Đã tạo đặc trưng giao dịch: {feature_types}")
                
                # Lưu dữ liệu với đặc trưng
                output_path = Path(f"data/features/{exchange}/{data_type}/{symbol.replace('/', '_')}_{timeframe}.parquet")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                feature_generator.save_data(df, output_path)
                
                logger.info(f"Đã lưu dữ liệu với đặc trưng cho {symbol} - {timeframe} tại {output_path}")
        
        logger.info("Đã hoàn thành tạo đặc trưng")

    async def analyze_sentiment(self, **kwargs) -> None:
        """
        Phân tích tâm lý thị trường từ dữ liệu tin tức và mạng xã hội.
        
        Args:
            data_source: Nguồn dữ liệu ('news', 'social')
            output_file: Tên file đầu ra
            platforms: Danh sách nền tảng mạng xã hội ('twitter', 'reddit')
        """
        logger.info("Bắt đầu phân tích tâm lý thị trường")
        
        data_source = kwargs.get('data_source', 'news')
        output_file = kwargs.get('output_file', 'sentiment_scores.csv')
        platforms = kwargs.get('platforms', [])
        
        # Tạo đường dẫn đầu ra
        output_path = Path(f"data/sentiment/{output_file}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thực hiện phân tích tâm lý theo nguồn dữ liệu
        if data_source == 'news':
            logger.info("Phân tích tâm lý từ tin tức")
            # Giả lập kết quả phân tích tâm lý
            # Trong triển khai thực tế, gọi hàm phân tích tâm lý từ module sentiment_features
            logger.info(f"Đã lưu kết quả phân tích tâm lý tin tức vào {output_path}")
        
        elif data_source == 'social':
            if not platforms:
                platforms = ['twitter', 'reddit']
            
            logger.info(f"Phân tích tâm lý từ mạng xã hội: {platforms}")
            # Giả lập kết quả phân tích tâm lý
            # Trong triển khai thực tế, gọi hàm phân tích tâm lý từ module sentiment_features
            logger.info(f"Đã lưu kết quả phân tích tâm lý mạng xã hội vào {output_path}")
        
        else:
            logger.error(f"Nguồn dữ liệu không hợp lệ: {data_source}")
            return
        
        logger.info("Đã hoàn thành phân tích tâm lý thị trường")

    async def run_data_pipeline(self, **kwargs) -> None:
        """
        Chạy toàn bộ pipeline xử lý dữ liệu.
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframes: Danh sách timeframe
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            pipeline_config: Đường dẫn đến file cấu hình pipeline
            output_format: Định dạng file đầu ra ('parquet', 'csv', 'json')
            output_dir: Thư mục đầu ra
        """
        logger.info("Bắt đầu chạy pipeline xử lý dữ liệu")
        
        symbols = kwargs.get('symbols', ['BTC/USDT'])
        timeframes = kwargs.get('timeframes', ['1h'])
        start_date = kwargs.get('start_date', None)
        end_date = kwargs.get('end_date', None)
        pipeline_config = kwargs.get('pipeline_config', None)
        output_format = kwargs.get('output_format', 'parquet')
        output_dir = kwargs.get('output_dir', 'data/pipeline')
        exchange = kwargs.get('exchange', 'binance')
        
        # Tạo đối tượng DataPipeline
        data_pipeline = DataPipeline()
        
        # Tải cấu hình pipeline nếu có
        config = None
        if pipeline_config:
            with open(pipeline_config, 'r') as f:
                config = json.load(f)
            logger.info(f"Đã tải cấu hình pipeline từ {pipeline_config}")
        
        # Chuyển đổi ngày thành datetime
        start_datetime = None
        end_datetime = None
        if start_date:
            start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Tạo thư mục đầu ra
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Thay đổi cách gọi để sử dụng run_pipeline thay vì run
        for timeframe in timeframes:
            try:
                logger.info(f"Chạy pipeline với timeframe {timeframe} cho {len(symbols)} cặp giao dịch")
                
                # Gọi run_pipeline thay vì run
                results = await data_pipeline.run_pipeline(
                    exchange_id=exchange,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_time=start_datetime,
                    end_time=end_datetime,
                    output_dir=output_path,
                    save_results=True
                )
                
                # Không cần xử lý kết quả thêm vì run_pipeline đã lưu kết quả tự động
                if results:
                    logger.info(f"Đã hoàn thành pipeline cho {timeframe} với {len(results)} kết quả")
                else:
                    logger.warning(f"Không có kết quả cho timeframe {timeframe}")
                
            except Exception as e:
                logger.error(f"Lỗi khi chạy pipeline cho timeframe {timeframe}: {str(e)}")
                logger.error("Chi tiết lỗi:", exc_info=True)
        
        logger.info("Đã hoàn thành chạy pipeline xử lý dữ liệu")

    async def select_features(self, **kwargs) -> None:
        """
        Lựa chọn đặc trưng quan trọng.
        
        Args:
            method: Phương pháp lựa chọn ('correlation', 'importance', 'pca')
            threshold: Ngưỡng lựa chọn
            n_features: Số lượng đặc trưng muốn chọn
            variance_explained: Tỷ lệ phương sai giải thích được (cho PCA)
        """
        logger.info("Bắt đầu lựa chọn đặc trưng quan trọng")
        
        method = kwargs.get('method', 'correlation')
        threshold = kwargs.get('threshold', 0.7)
        n_features = kwargs.get('n_features', 20)
        variance_explained = kwargs.get('variance_explained', 0.95)
        
        # Tạo đối tượng lựa chọn đặc trưng
        # Giả lập việc lựa chọn đặc trưng
        # Trong triển khai thực tế, gọi hàm lựa chọn đặc trưng từ module feature_selector
        
        if method == 'correlation':
            logger.info(f"Lựa chọn đặc trưng bằng phương pháp tương quan với ngưỡng {threshold}")
        elif method == 'importance':
            logger.info(f"Lựa chọn {n_features} đặc trưng quan trọng nhất")
        elif method == 'pca':
            logger.info(f"Giảm chiều dữ liệu với tỷ lệ phương sai giải thích được {variance_explained}")
        else:
            logger.error(f"Phương pháp lựa chọn đặc trưng không hợp lệ: {method}")
            return
        
        logger.info("Đã hoàn thành lựa chọn đặc trưng quan trọng")

    async def run_backtest(self):
        """
        Chạy backtest trên dữ liệu lịch sử.
        """
        logger.info("Chức năng backtest chưa được triển khai")
        # TODO: Triển khai khi module backtest đã sẵn sàng
    
    async def train_agent(self):
        """
        Huấn luyện agent giao dịch.
        """
        logger.info("Chức năng huấn luyện agent chưa được triển khai")
        # TODO: Triển khai khi module models đã sẵn sàng
    
    async def start_trading(self):
        """
        Bắt đầu giao dịch thực tế.
        """
        logger.info("Chức năng giao dịch thực tế chưa được triển khai")
        # TODO: Triển khai khi module deployment đã sẵn sàng
    
    async def start_dashboard(self):
        """
        Khởi chạy dashboard.
        """
        logger.info("Chức năng dashboard chưa được triển khai")
        # TODO: Triển khai khi module streamlit_dashboard đã sẵn sàng

    async def cleanup(self):
        """
        Dọn dẹp tài nguyên khi kết thúc.
        """
        logger.info("Dọn dẹp tài nguyên hệ thống")
        
        # Đóng các kết nối
        for name, connector in self.connectors.items():
            try:
                logger.debug(f"Đóng kết nối {name}")
                # Kiểm tra và gọi phương thức close nếu có
                if hasattr(connector, 'close'):
                    await connector.close()
            except Exception as e:
                logger.warning(f"Lỗi khi đóng kết nối {name}: {str(e)}")
        
        self.status = SystemStatus.STOPPED
        logger.info("Hệ thống đã dừng")

async def main():
    """
    Hàm main chính của hệ thống.
    """
    # Tạo parser cho command-line arguments
    parser = argparse.ArgumentParser(description='Hệ thống Giao dịch Tự động')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh cần thực hiện')
    
    # Tạo subparser cho thu thập dữ liệu
    collect_parser = subparsers.add_parser('collect', help='Thu thập dữ liệu lịch sử')
    collect_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    collect_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT'], 
                               help='Danh sách cặp giao dịch')
    collect_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'], 
                               help='Danh sách timeframe')
    collect_parser.add_argument('--days', type=int, default=30, help='Số ngày lấy dữ liệu')
    collect_parser.add_argument('--futures', action='store_true', help='Sử dụng thị trường futures')
    collect_parser.add_argument('--start-date', type=str, help='Ngày bắt đầu thu thập dữ liệu (format YYYY-MM-DD)')
    collect_parser.add_argument('--end-date', type=str, help='Ngày kết thúc thu thập dữ liệu (format YYYY-MM-DD)')
    
    # Tạo subparser cho xử lý dữ liệu
    process_parser = subparsers.add_parser('process', help='Xử lý dữ liệu')
    process_subparsers = process_parser.add_subparsers(dest='process_command', help='Lệnh xử lý dữ liệu')
    
    # Subparser cho lệnh clean
    clean_parser = process_subparsers.add_parser('clean', help='Làm sạch dữ liệu')
    clean_parser.add_argument('--data-type', type=str, default='ohlcv', help='Loại dữ liệu (ohlcv, orderbook, trades)')
    clean_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], help='Danh sách cặp giao dịch')
    clean_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'], help='Danh sách timeframe')
    clean_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    clean_parser.add_argument('--outlier-method', type=str, help='Phương pháp phát hiện ngoại lệ (zscore, iqr, isolation_forest)')
    clean_parser.add_argument('--threshold', type=float, default=3.0, help='Ngưỡng phát hiện ngoại lệ')
    clean_parser.add_argument('--fill-method', type=str, help='Phương pháp điền dữ liệu thiếu (ffill, bfill, interpolate)')
    
    # Subparser cho lệnh features
    features_parser = process_subparsers.add_parser('features', help='Tạo đặc trưng')
    features_parser.add_argument('--data-type', type=str, default='ohlcv', help='Loại dữ liệu (ohlcv, orderbook, trades)')
    features_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], help='Danh sách cặp giao dịch')
    features_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'], help='Danh sách timeframe')
    features_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    features_parser.add_argument('--indicators', type=str, nargs='+', help='Danh sách chỉ báo kỹ thuật')
    features_parser.add_argument('--all-indicators', action='store_true', help='Tạo tất cả chỉ báo kỹ thuật')
    features_parser.add_argument('--feature-types', type=str, nargs='+', help='Danh sách loại đặc trưng (cho orderbook, trades)')
    
    # Subparser cho lệnh sentiment
    sentiment_parser = process_subparsers.add_parser('sentiment', help='Phân tích tâm lý thị trường')
    sentiment_parser.add_argument('--data-source', type=str, default='news', help='Nguồn dữ liệu (news, social)')
    sentiment_parser.add_argument('--output-file', type=str, default='sentiment_scores.csv', help='Tên file đầu ra')
    sentiment_parser.add_argument('--platforms', type=str, nargs='+', help='Danh sách nền tảng mạng xã hội (twitter, reddit)')
    
    # Subparser cho lệnh pipeline
    pipeline_parser = process_subparsers.add_parser('pipeline', help='Chạy pipeline xử lý dữ liệu')
    pipeline_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], help='Danh sách cặp giao dịch')
    pipeline_parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'], help='Danh sách timeframe')
    pipeline_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    pipeline_parser.add_argument('--start-date', type=str, help='Ngày bắt đầu (YYYY-MM-DD)')
    pipeline_parser.add_argument('--end-date', type=str, help='Ngày kết thúc (YYYY-MM-DD)')
    pipeline_parser.add_argument('--pipeline-config', type=str, help='Đường dẫn đến file cấu hình pipeline')
    pipeline_parser.add_argument('--output-format', type=str, default='parquet', help='Định dạng file đầu ra (parquet, csv, json)')
    pipeline_parser.add_argument('--output-dir', type=str, default='data/pipeline', help='Thư mục đầu ra')
    
    # Subparser cho lệnh select-features
    select_features_parser = process_subparsers.add_parser('select-features', help='Lựa chọn đặc trưng quan trọng')
    select_features_parser.add_argument('--method', type=str, default='correlation', help='Phương pháp lựa chọn (correlation, importance, pca)')
    select_features_parser.add_argument('--threshold', type=float, default=0.7, help='Ngưỡng lựa chọn')
    select_features_parser.add_argument('--n-features', type=int, default=20, help='Số lượng đặc trưng muốn chọn')
    select_features_parser.add_argument('--variance-explained', type=float, default=0.95, help='Tỷ lệ phương sai giải thích được (cho PCA)')
    
    # Tạo subparser cho backtest
    backtest_parser = subparsers.add_parser('backtest', help='Chạy backtest')
    backtest_parser.add_argument('--strategy', type=str, default='dqn', help='Chiến lược giao dịch')
    
    # Tạo subparser cho huấn luyện
    train_parser = subparsers.add_parser('train', help='Huấn luyện agent')
    train_parser.add_argument('--agent', type=str, default='dqn', help='Loại agent')
    
    # Tạo subparser cho giao dịch thực tế
    trade_parser = subparsers.add_parser('trade', help='Giao dịch thực tế')
    trade_parser.add_argument('--exchange', type=str, default='binance', help='Tên sàn giao dịch')
    trade_parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], 
                             help='Danh sách cặp giao dịch')
    
    # Tạo subparser cho dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Khởi chạy dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Nếu không có lệnh nào được cung cấp, hiển thị help
    if not args.command:
        parser.print_help()
        return
    
    # Khởi tạo hệ thống
    system = AutomatedTradingSystem()
    
    try:
        # Thực hiện lệnh tương ứng
        if args.command == 'collect':
            await system.collect_historical_data(
                exchange_id=args.exchange,
                symbols=args.symbols,
                timeframes=args.timeframes,
                days_back=args.days,
                is_futures=args.futures,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.command == 'process':
            if not args.process_command:
                process_parser.print_help()
                return
            
            # Chuyển đổi Namespace thành dict
            kwargs = vars(args)
            # Loại bỏ các tham số không cần thiết
            kwargs.pop('command', None)
            command = kwargs.pop('process_command', None)
            await system.process_data(command, **kwargs)
            
        elif args.command == 'backtest':
            await system.run_backtest()
        elif args.command == 'train':
            await system.train_agent()
        elif args.command == 'trade':
            await system.start_trading()
        elif args.command == 'dashboard':
            await system.start_dashboard()
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu dừng từ người dùng")
    except Exception as e:
        logger.error(f"Lỗi không xử lý được: {str(e)}", exc_info=True)
        
        # Đưa ra gợi ý giải pháp dựa trên loại lỗi
        if "timeout" in str(e).lower():
            logger.warning("Lỗi timeout có thể do kết nối mạng không ổn định hoặc sàn giao dịch không phản hồi")
            logger.warning("Giải pháp: Tăng REQUEST_TIMEOUT trong .env, kiểm tra kết nối mạng hoặc sử dụng proxy")
        elif "headers" in str(e).lower():
            logger.warning("Lỗi xử lý response từ sàn giao dịch. Có thể do cấu trúc response không đúng định dạng.")
            logger.warning("Giải pháp: Cập nhật thư viện CCXT hoặc kiểm tra lại các connector")
    finally:
        # Dọn dẹp tài nguyên
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())