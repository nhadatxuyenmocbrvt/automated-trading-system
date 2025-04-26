"""
Thu thập dữ liệu lịch sử từ các sàn giao dịch.
File này cung cấp các lớp và phương thức để tải và lưu trữ dữ liệu lịch sử
từ các sàn giao dịch tiền điện tử, hỗ trợ nhiều loại dữ liệu khác nhau như
OHLCV, orderbook snapshot, và giao dịch lịch sử.
"""

import os
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import logging
import json
from pathlib import Path
import concurrent.futures
from functools import partial
from enum import Enum, auto

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collectors.exchange_api.generic_connector import GenericExchangeConnector
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector
from config.logging_config import setup_logger
from config.constants import Timeframe, TIMEFRAME_TO_SECONDS, Exchange, ErrorCode
from config.env import get_env
from config.system_config import DATA_DIR, BASE_DIR


class DataType(Enum):
    """Loại dữ liệu hỗ trợ."""
    OHLCV = auto()
    ORDERBOOK = auto()
    TRADES = auto()
    FUNDING = auto()


class FileFormat(Enum):
    """Định dạng file lưu trữ."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"


class HistoricalDataCollector:
    """
    Lớp chính để thu thập dữ liệu lịch sử từ các sàn giao dịch.
    Hỗ trợ thu thập đa dạng loại dữ liệu và lưu trữ dưới nhiều định dạng.
    """
    
    def __init__(
        self,
        exchange_connector: GenericExchangeConnector,
        data_dir: Path = None,
        max_workers: int = 4,
        rate_limit_factor: float = 0.8
    ):
        """
        Khởi tạo bộ thu thập dữ liệu lịch sử.
        
        Args:
            exchange_connector: Kết nối với sàn giao dịch
            data_dir: Thư mục lưu trữ dữ liệu (mặc định là DATA_DIR từ config)
            max_workers: Số luồng tối đa cho việc thu thập song song
            rate_limit_factor: Hệ số để giảm tốc độ gọi API (0.0 - 1.0)
        """
        self.exchange_connector = exchange_connector
        self.exchange_id = exchange_connector.exchange_id
        self.logger = setup_logger(f"historical_data_collector_{self.exchange_id}")
        
        # Thiết lập thư mục lưu trữ dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR / 'historical' / self.exchange_id
        else:
            self.data_dir = data_dir / 'historical' / self.exchange_id
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thư mục cho từng loại dữ liệu
        self.data_type_dirs = {
            DataType.OHLCV: self.data_dir / 'ohlcv',
            DataType.ORDERBOOK: self.data_dir / 'orderbook',
            DataType.TRADES: self.data_dir / 'trades',
            DataType.FUNDING: self.data_dir / 'funding',
        }
        
        # Tạo các thư mục con
        for dir_path in self.data_type_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Cấu hình rate limit
        self.rate_limit = exchange_connector.exchange.rateLimit / 1000  # Chuyển ms thành giây
        self.rate_limit_sleep = self.rate_limit * rate_limit_factor
        
        # Cấu hình cho việc thu thập song song
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        
        self.logger.info(f"Đã khởi tạo HistoricalDataCollector cho {self.exchange_id}")
    
    async def collect_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập dữ liệu OHLCV (Open, High, Low, Close, Volume) cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            timeframe: Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)
            start_time: Thời gian bắt đầu (mặc định là 30 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            limit: Số lượng nến tối đa mỗi lần gọi API
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        # Chuẩn bị tham số
        filename = f"{symbol.replace('/', '_')}_{timeframe}".lower()
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)
        
        self.logger.info(f"Thu thập OHLCV cho {symbol} ({timeframe}) từ {start_time} đến {end_time}")
        
        # Kiểm tra xem cặp giao dịch có hiệu lực không
        await self._validate_symbol(symbol)
        
        # Xác định đường dẫn file và kiểm tra dữ liệu hiện có
        file_path = self._get_file_path(DataType.OHLCV, filename, save_format)
        
        # Đọc dữ liệu hiện có nếu cần
        existing_data = None
        last_timestamp = None
        
        if update_existing and file_path.exists():
            existing_data = self._load_dataframe(file_path, save_format)
            
            if existing_data is not None and not existing_data.empty:
                existing_data = existing_data.sort_values('timestamp')
                if 'timestamp' in existing_data.columns:
                    last_timestamp = existing_data['timestamp'].max()
                    
                    # Cập nhật thời gian bắt đầu nếu có dữ liệu hiện có
                    if last_timestamp:
                        actual_last_time = pd.to_datetime(last_timestamp).to_pydatetime()
                        # Trừ đi một khoảng thời gian của timeframe để chắc chắn không bỏ lỡ dữ liệu
                        tf_seconds = TIMEFRAME_TO_SECONDS.get(timeframe, 3600)
                        start_time = max(start_time, actual_last_time - timedelta(seconds=tf_seconds))
                        self.logger.info(f"Cập nhật dữ liệu từ {start_time} (dữ liệu cuối {actual_last_time})")
        
        # Thu thập dữ liệu
        raw_data = await self._fetch_time_series_data(
            fetch_method=self.exchange_connector.fetch_ohlcv,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Chuyển đổi dữ liệu thành DataFrame
        if raw_data:
            df = pd.DataFrame(raw_data, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Loại bỏ các bản ghi trùng lặp
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Ghép với dữ liệu hiện có nếu có
            if existing_data is not None and not existing_data.empty:
                df = self._merge_dataframes(existing_data, df, 'timestamp')
            
            # Lưu dữ liệu
            self._save_dataframe(df, file_path, save_format)
            
            return df
        else:
            self.logger.warning(f"Không có dữ liệu mới cho {symbol}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_orderbook_snapshots(
        self,
        symbol: str,
        interval: int = 3600,  # 1 giờ
        depth: int = 20,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_format: str = 'parquet'
    ) -> List[Dict]:
        """
        Thu thập snapshot của orderbook theo định kỳ.
        
        Args:
            symbol: Cặp giao dịch
            interval: Khoảng thời gian giữa các snapshot (giây)
            depth: Độ sâu của orderbook
            start_time: Thời gian bắt đầu (mặc định là 7 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            
        Returns:
            Danh sách snapshot orderbook
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 7 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=7)
        
        self.logger.info(f"Thu thập orderbook snapshot cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_depth{depth}_interval{interval}s".lower()
        file_path = self._get_file_path(DataType.ORDERBOOK, filename, save_format)
        
        # Tạo danh sách các mốc thời gian cần lấy snapshot
        time_points = []
        current_time = start_time
        while current_time <= end_time:
            time_points.append(current_time)
            current_time += timedelta(seconds=interval)
        
        if not time_points:
            self.logger.warning("Không có điểm thời gian nào để thu thập")
            return []
        
        # Đọc dữ liệu hiện có
        existing_snapshots, existing_timestamps = self._load_snapshots(file_path, save_format)
        
        # Thu thập snapshots mới
        new_snapshots = await self._fetch_orderbook_snapshots(symbol, time_points, depth, existing_timestamps)
        
        # Kết hợp với dữ liệu hiện có
        all_snapshots = existing_snapshots + new_snapshots
        
        # Sắp xếp theo thời gian
        all_snapshots.sort(key=lambda x: x['timestamp'])
        
        # Lưu dữ liệu
        if all_snapshots:
            self._save_snapshots(all_snapshots, file_path, save_format)
        
        return all_snapshots
    
    async def collect_historical_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập lịch sử giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu (mặc định là 1 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            limit: Số lượng giao dịch tối đa mỗi lần gọi API
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa lịch sử giao dịch
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 1 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
        
        self.logger.info(f"Thu thập lịch sử giao dịch cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_trades".lower()
        file_path = self._get_file_path(DataType.TRADES, filename, save_format)
        
        # Đọc dữ liệu hiện có nếu cần
        existing_data = None
        last_timestamp = None
        
        if update_existing and file_path.exists():
            existing_data = self._load_dataframe(file_path, save_format)
            
            if existing_data is not None and not existing_data.empty:
                existing_data = existing_data.sort_values('timestamp')
                if 'timestamp' in existing_data.columns:
                    last_timestamp = existing_data['timestamp'].max()
                    
                    # Cập nhật thời gian bắt đầu nếu có dữ liệu hiện có
                    if last_timestamp:
                        actual_last_time = pd.to_datetime(last_timestamp).to_pydatetime()
                        # Bắt đầu từ thời điểm cuối cùng có dữ liệu
                        start_time = max(start_time, actual_last_time)
                        self.logger.info(f"Cập nhật dữ liệu từ {start_time} (dữ liệu cuối {actual_last_time})")
        
        # Thu thập dữ liệu
        raw_trades = await self._fetch_time_series_data(
            fetch_method=self.exchange_connector.fetch_trades,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        # Chuyển đổi dữ liệu thành DataFrame
        if raw_trades:
            # Chuẩn hóa dữ liệu
            normalized_trades = self._normalize_trades(raw_trades, symbol)
            
            df = pd.DataFrame(normalized_trades)
            
            # Chuyển đổi timestamp sang datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Loại bỏ các bản ghi trùng lặp
            df = self._deduplicate_trades(df)
            
            # Ghép với dữ liệu hiện có nếu có
            if existing_data is not None and not existing_data.empty:
                df = self._merge_dataframes(existing_data, df, ['id'] if 'id' in df.columns else ['timestamp', 'price', 'amount'])
            
            # Lưu dữ liệu
            self._save_dataframe(df, file_path, save_format)
            
            return df
        else:
            self.logger.warning(f"Không có dữ liệu giao dịch mới cho {symbol}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_funding_rates(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập tỷ lệ tài trợ cho hợp đồng tương lai.
        
        Args:
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu (mặc định là 30 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa dữ liệu tỷ lệ tài trợ
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)
        
        self.logger.info(f"Thu thập tỷ lệ tài trợ cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_funding".lower()
        file_path = self._get_file_path(DataType.FUNDING, filename, save_format)
        
        # Tỷ lệ tài trợ chỉ khả dụng cho hợp đồng tương lai
        if not hasattr(self.exchange_connector, 'fetch_funding_rate'):
            self.logger.error(f"Sàn giao dịch {self.exchange_id} không hỗ trợ thu thập tỷ lệ tài trợ")
            return pd.DataFrame()
        
        # Đọc dữ liệu hiện có
        existing_data = None
        
        if update_existing and file_path.exists():
            existing_data = self._load_dataframe(file_path, save_format)
        
        # Thu thập dữ liệu mới
        try:
            # Gọi API để lấy tỷ lệ tài trợ hiện tại
            async with self.semaphore:
                funding_rate = await self.exchange_connector.fetch_funding_rate(symbol)
                await asyncio.sleep(self.rate_limit_sleep)
            
            if not funding_rate:
                self.logger.warning(f"Không có dữ liệu tỷ lệ tài trợ cho {symbol}")
                return pd.DataFrame() if existing_data is None else existing_data
            
            # Tạo DataFrame
            new_data = pd.DataFrame([{
                'symbol': symbol,
                'timestamp': pd.to_datetime(funding_rate.get('timestamp', datetime.now().timestamp() * 1000), unit='ms'),
                'fundingRate': funding_rate.get('fundingRate', None),
                'fundingTime': pd.to_datetime(funding_rate.get('fundingTime', None), unit='ms') if funding_rate.get('fundingTime') else None,
                'datetime': funding_rate.get('datetime', None)
            }])
            
            # Ghép với dữ liệu hiện có
            if existing_data is not None and not existing_data.empty:
                new_data = self._merge_dataframes(existing_data, new_data, 'timestamp')
            
            # Lưu dữ liệu
            self._save_dataframe(new_data, file_path, save_format)
            
            return new_data
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ cho {symbol}: {e}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_all_symbols_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        concurrency: int = 3,
        save_format: str = 'parquet'
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu OHLCV cho nhiều cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            concurrency: Số lượng cặp giao dịch thu thập đồng thời
            save_format: Định dạng lưu trữ
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        self.logger.info(f"Thu thập OHLCV cho {len(symbols)} cặp giao dịch với {timeframe}")
        
        # Giới hạn số lượng concurrency để tránh rate limit
        max_concurrent = min(concurrency, self.max_workers)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_symbol_data(symbol):
            async with semaphore:
                self.logger.debug(f"Bắt đầu thu thập dữ liệu cho {symbol}")
                try:
                    df = await self.collect_ohlcv(
                        symbol, timeframe, start_time, end_time, save_format=save_format
                    )
                    return symbol, df
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {e}")
                    return symbol, pd.DataFrame()
        
        # Tạo tasks và thực hiện
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        data_dict = {}
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Lỗi khi thu thập dữ liệu: {res}")
            else:
                symbol, df = res
                data_dict[symbol] = df
        
        self.logger.info(f"Đã thu thập dữ liệu cho {len(data_dict)} cặp giao dịch")
        return data_dict
    
    async def collect_all_funding_rates(
        self,
        symbols: List[str],
        save_format: str = 'parquet'
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập tỷ lệ tài trợ cho nhiều cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            save_format: Định dạng lưu trữ
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        self.logger.info(f"Thu thập tỷ lệ tài trợ cho {len(symbols)} cặp giao dịch")
        
        # Xác định số lượng concurrency tối đa
        max_concurrent = min(5, self.max_workers)  # Giới hạn thấp hơn cho funding rate
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_funding_data(symbol):
            async with semaphore:
                self.logger.debug(f"Bắt đầu thu thập tỷ lệ tài trợ cho {symbol}")
                try:
                    df = await self.collect_funding_rates(
                        symbol, save_format=save_format
                    )
                    return symbol, df
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ cho {symbol}: {e}")
                    return symbol, pd.DataFrame()
        
        # Tạo tasks và thực hiện
        tasks = [fetch_funding_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        data_dict = {}
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ: {res}")
            else:
                symbol, df = res
                data_dict[symbol] = df
        
        self.logger.info(f"Đã thu thập tỷ lệ tài trợ cho {len(data_dict)} cặp giao dịch")
        return data_dict
    
    @staticmethod
    def get_available_timeframes() -> Dict[str, int]:
        """
        Lấy danh sách khung thời gian có sẵn.
        
        Returns:
            Dict với key là tên timeframe và value là số giây
        """
        return TIMEFRAME_TO_SECONDS
    
    def get_local_data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã được lưu trữ cục bộ.
        
        Returns:
            Dict chứa thông tin về dữ liệu
        """
        info = {
            "exchange": self.exchange_id,
            "ohlcv": {},
            "orderbook": {},
            "trades": {},
            "funding": {}
        }
        
        # Thu thập thông tin cho từng loại dữ liệu
        data_types_map = {
            DataType.OHLCV: "ohlcv",
            DataType.ORDERBOOK: "orderbook",
            DataType.TRADES: "trades",
            DataType.FUNDING: "funding"
        }
        
        for data_type, info_key in data_types_map.items():
            dir_path = self.data_type_dirs[data_type]
            files = []
            for format_ext in ['parquet', 'csv', 'json']:
                files.extend(list(dir_path.glob(f"*.{format_ext}")))
            
            for file in files:
                try:
                    file_info = self._get_file_info(file, data_type)
                    if file_info:
                        info[info_key][file.stem] = file_info
                except Exception as e:
                    self.logger.warning(f"Không thể đọc file {file}: {e}")
        
        return info
    
    # === PRIVATE HELPER METHODS ===
    
    async def _validate_symbol(self, symbol: str) -> bool:
        """
        Kiểm tra xem cặp giao dịch có hiệu lực không.
        
        Args:
            symbol: Cặp giao dịch cần kiểm tra
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        try:
            if symbol not in self.exchange_connector.markets:
                await self.exchange_connector.load_markets(reload=True)
                if symbol not in self.exchange_connector.markets:
                    self.logger.error(f"Cặp giao dịch {symbol} không hợp lệ cho {self.exchange_id}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra cặp giao dịch {symbol}: {e}")
            return False
    
    def _get_file_path(self, data_type: DataType, filename: str, save_format: str) -> Path:
        """
        Tạo đường dẫn file dựa trên loại dữ liệu và định dạng.
        
        Args:
            data_type: Loại dữ liệu
            filename: Tên file (không có phần mở rộng)
            save_format: Định dạng lưu trữ
            
        Returns:
            Đường dẫn file
        """
        if save_format not in ['parquet', 'csv', 'json']:
            self.logger.warning(f"Định dạng {save_format} không được hỗ trợ, sử dụng parquet")
            save_format = 'parquet'
        
        dir_path = self.data_type_dirs[data_type]
        return dir_path / f"{filename}.{save_format}"
    
    def _load_dataframe(self, file_path: Path, file_format: str) -> Optional[pd.DataFrame]:
        """
        Đọc DataFrame từ file.
        
        Args:
            file_path: Đường dẫn file
            file_format: Định dạng file
            
        Returns:
            DataFrame hoặc None nếu có lỗi
        """
        try:
            if file_format == 'parquet':
                return pd.read_parquet(file_path)
            elif file_format == 'csv':
                return pd.read_csv(file_path, parse_dates=['timestamp'])
            elif file_format == 'json':
                return pd.read_json(file_path, orient='records')
            else:
                self.logger.warning(f"Định dạng không được hỗ trợ: {file_format}")
                return None
        except Exception as e:
            self.logger.warning(f"Không thể đọc file {file_path}: {e}")
            return None
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: Path, file_format: str) -> bool:
        """
        Lưu DataFrame vào file.
        
        Args:
            df: DataFrame cần lưu
            file_path: Đường dẫn file
            file_format: Định dạng file
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            if file_format == 'parquet':
                df.to_parquet(file_path, index=False)
            elif file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            else:
                self.logger.warning(f"Định dạng không được hỗ trợ: {file_format}")
                return False
            
            self.logger.info(f"Đã lưu {len(df)} bản ghi vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu vào {file_path}: {e}")
            return False
    
    def _load_snapshots(self, file_path: Path, file_format: str) -> Tuple[List[Dict], Set[int]]:
        """
        Đọc snapshots từ file.
        
        Args:
            file_path: Đường dẫn file
            file_format: Định dạng file
            
        Returns:
            Tuple (danh sách snapshots, tập hợp timestamps)
        """
        existing_snapshots = []
        existing_timestamps = set()
        
        if file_path.exists():
            try:
                if file_format == 'parquet':
                    df = pd.read_parquet(file_path)
                    existing_snapshots = df.to_dict('records')
                elif file_format == 'csv':
                    df = pd.read_csv(file_path)
                    # Chuyển timestamp thành datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    existing_snapshots = df.to_dict('records')
                elif file_format == 'json':
                    with open(file_path, 'r') as f:
                        existing_snapshots = json.load(f)
                
                # Lấy timestamps từ các snapshots hiện có
                for snapshot in existing_snapshots:
                    if 'timestamp' in snapshot:
                        if isinstance(snapshot['timestamp'], str):
                            dt = datetime.fromisoformat(snapshot['timestamp'].replace('Z', '+00:00'))
                            existing_timestamps.add(int(dt.timestamp() * 1000))
                        else:
                            existing_timestamps.add(snapshot['timestamp'])
                
                self.logger.info(f"Đã tải {len(existing_snapshots)} snapshot hiện có")
            except Exception as e:
                self.logger.warning(f"Không thể đọc dữ liệu hiện có từ {file_path}: {e}")
        
        return existing_snapshots, existing_timestamps
    
    def _save_snapshots(self, snapshots: List[Dict], file_path: Path, file_format: str) -> bool:
        """
        Lưu snapshots vào file.
        
        Args:
            snapshots: Danh sách snapshots
            file_path: Đường dẫn file
            file_format: Định dạng file
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            if file_format == 'parquet':
                pd.DataFrame(snapshots).to_parquet(file_path, index=False)
            elif file_format == 'csv':
                pd.DataFrame(snapshots).to_csv(file_path, index=False)
            elif file_format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(snapshots, f, indent=2)
            else:
                self.logger.warning(f"Định dạng không được hỗ trợ: {file_format}")
                return False
            
            self.logger.info(f"Đã lưu {len(snapshots)} snapshot vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu snapshots vào {file_path}: {e}")
            return False
    
    def _merge_dataframes(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, unique_keys) -> pd.DataFrame:
        """
        Kết hợp dữ liệu mới với dữ liệu hiện có, loại bỏ trùng lặp.
        
        Args:
            existing_df: DataFrame hiện có
            new_df: DataFrame mới
            unique_keys: Keys để xác định trùng lặp (str hoặc List[str])
            
        Returns:
            DataFrame đã kết hợp
        """
        if existing_df.empty:
            return new_df
        if new_df.empty:
            return existing_df
            
        # Kết hợp dữ liệu cũ và mới
        combined_df = pd.concat([existing_df, new_df])
        
        # Loại bỏ trùng lặp
        combined_df = combined_df.drop_duplicates(subset=unique_keys)
        
        # Sắp xếp theo thời gian
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
        
        return combined_df
    
    def _normalize_trades(self, trades: List[Dict], symbol: str) -> List[Dict]:
        """
        Chuẩn hóa dữ liệu giao dịch.
        
        Args:
            trades: Danh sách giao dịch thô
            symbol: Cặp giao dịch
            
        Returns:
            Danh sách giao dịch đã chuẩn hóa
        """
        normalized_trades = []
        
        for trade in trades:
            normalized_trade = {
                'id': trade.get('id', None),
                'timestamp': trade.get('timestamp', None),
                'datetime': trade.get('datetime', None),
                'symbol': trade.get('symbol', symbol),
                'side': trade.get('side', None),
                'price': trade.get('price', None),
                'amount': trade.get('amount', None),
                'cost': trade.get('cost', None),
                'fee': json.dumps(trade.get('fee', {})) if trade.get('fee') else None,
                'fee_currency': trade.get('feeCurrency', None) if 'feeCurrency' in trade else None,
                'type': trade.get('type', None),
                'takerOrMaker': trade.get('takerOrMaker', None),
            }
            normalized_trades.append(normalized_trade)
        
        return normalized_trades
    
    def _deduplicate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ giao dịch trùng lặp.
        
        Args:
            df: DataFrame giao dịch
            
        Returns:
            DataFrame đã loại bỏ trùng lặp
        """
        if 'id' in df.columns:
            return df.drop_duplicates(subset=['id'])
        else:
            # Nếu không có id, dùng timestamp, price, và amount
            return df.drop_duplicates(subset=['timestamp', 'price', 'amount'])
    
    def _get_file_info(self, file_path: Path, data_type: DataType) -> Optional[Dict]:
        """
        Lấy thông tin về một file dữ liệu.
        
        Args:
            file_path: Đường dẫn file
            data_type: Loại dữ liệu
            
        Returns:
            Dict chứa thông tin hoặc None nếu có lỗi
        """
        try:
            # Đọc file dựa vào phần mở rộng
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                if data_type == DataType.ORDERBOOK:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Tạo DataFrame từ dữ liệu
                    df = pd.DataFrame({
                        'timestamp': [item.get('timestamp') for item in data],
                        'symbol': [item.get('symbol') for item in data]
                    })
                else:
                    df = pd.read_json(file_path)
            else:
                return None
            
            # Phân tích tên file để lấy symbol và thông tin khác
            parts = file_path.stem.split('_')
            symbol = parts[0] if len(parts) > 0 else 'unknown'
            timeframe = parts[1] if len(parts) > 1 and data_type == DataType.OHLCV else 'unknown'
            
            # Chuẩn bị thông tin chung
            info = {
                "symbol": symbol,
                "rows": len(df),
                "file": str(file_path)
            }
            
            # Thêm thông tin dựa vào loại dữ liệu
            if 'timestamp' in df.columns:
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                info["start_date"] = df['timestamp'].min().strftime('%Y-%m-%d')
                info["end_date"] = df['timestamp'].max().strftime('%Y-%m-%d')
            
            if data_type == DataType.OHLCV:
                info["timeframe"] = timeframe
            
            return info
        except Exception as e:
            self.logger.warning(f"Không thể đọc thông tin từ file {file_path}: {e}")
            return None
    
    async def _fetch_time_series_data(
        self,
        fetch_method: Callable,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
        timeframe: Optional[str] = None
    ) -> List[Any]:
        """
        Phương thức tổng quát để thu thập dữ liệu theo chuỗi thời gian.
        
        Args:
            fetch_method: Phương thức để gọi API (fetch_ohlcv, fetch_trades,...)
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            limit: Số lượng bản ghi tối đa mỗi lần gọi
            timeframe: Khung thời gian (chỉ cần cho OHLCV)
            
        Returns:
            Danh sách dữ liệu đã thu thập
        """
        # Chuyển đổi datetime sang timestamp (ms)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                # Chuẩn bị tham số cho phương thức fetch
                if timeframe:
                    args = [symbol, timeframe, current_ts, limit]
                else:
                    args = [symbol, current_ts, limit]
                
                # Gọi API với giới hạn rate
                async with self.semaphore:
                    data = await fetch_method(*args)
                    await asyncio.sleep(self.rate_limit_sleep)
                
                if not data or len(data) == 0:
                    self.logger.debug(f"Không có dữ liệu mới từ {datetime.fromtimestamp(current_ts/1000)}")
                    break
                
                all_data.extend(data)
                self.logger.debug(f"Đã lấy {len(data)} bản ghi từ {datetime.fromtimestamp(current_ts/1000)}")
                
                # Cập nhật timestamp cho lần gọi tiếp theo
                if isinstance(data[-1], dict) and 'timestamp' in data[-1]:
                    # Format cho trades và các loại dữ liệu dạng dict
                    last_timestamp = data[-1]['timestamp']
                else:
                    # Format cho OHLCV (dạng list)
                    last_timestamp = data[-1][0] if isinstance(data[-1], list) and len(data[-1]) > 0 else None
                
                if last_timestamp is None or last_timestamp <= current_ts:
                    # Tránh lặp vô hạn nếu không tăng timestamp
                    self.logger.warning("Timestamp không tăng, dừng thu thập")
                    break
                
                current_ts = last_timestamp + 1
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {e}")
                # Tạm dừng dài hơn khi có lỗi
                await asyncio.sleep(self.rate_limit_sleep * 2)
                break
            
            # Kiểm tra đã đạt đến end_time chưa
            if current_ts >= end_ts:
                break
        
        return all_data
    
    async def _fetch_orderbook_snapshots(
        self, 
        symbol: str, 
        time_points: List[datetime],
        depth: int,
        existing_timestamps: Set[int]
    ) -> List[Dict]:
        """
        Thu thập orderbook snapshots cho các thời điểm cụ thể.
        
        Args:
            symbol: Cặp giao dịch
            time_points: Danh sách thời điểm cần lấy snapshot
            depth: Độ sâu của orderbook
            existing_timestamps: Set các timestamps đã có dữ liệu
            
        Returns:
            Danh sách snapshots mới
        """
        snapshots = []
        
        for time_point in time_points:
            # Bỏ qua các điểm thời gian đã có dữ liệu
            if int(time_point.timestamp() * 1000) in existing_timestamps:
                continue
            
            try:
                # Lấy snapshot orderbook
                async with self.semaphore:
                    orderbook = await self.exchange_connector.fetch_order_book(
                        symbol, depth
                    )
                    await asyncio.sleep(self.rate_limit_sleep)
                
                if orderbook:
                    # Tạo snapshot với thông tin timestamp của time_point
                    snapshot = {
                        "symbol": symbol,
                        "timestamp": time_point.isoformat(),
                        "bids": orderbook['bids'],
                        "asks": orderbook['asks'],
                        "datetime": orderbook.get('datetime', None),
                        "nonce": orderbook.get('nonce', None)
                    }
                    snapshots.append(snapshot)
                    self.logger.debug(f"Đã lấy snapshot cho {symbol} tại {time_point}")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi lấy orderbook cho {symbol} tại {time_point}: {e}")
        
        return snapshots


# Factory function
async def create_data_collector(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True,
    is_futures: bool = False,
    max_workers: int = 4
) -> HistoricalDataCollector:
    """
    Tạo một instance của HistoricalDataCollector cho sàn giao dịch cụ thể.
    
    Args:
        exchange_id: ID của sàn giao dịch
        api_key: Khóa API
        api_secret: Mật khẩu API
        sandbox: Sử dụng môi trường testnet
        is_futures: Sử dụng tài khoản futures
        max_workers: Số luồng tối đa cho việc thu thập song song
        
    Returns:
        Instance của HistoricalDataCollector
    """
    # Tạo connector cho sàn giao dịch
    exchange_connector = None
    
    if exchange_id.lower() == 'binance':
        exchange_connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            futures=is_futures
        )
    elif exchange_id.lower() == 'bybit':
        exchange_connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            category='linear' if is_futures else 'spot'
        )
    else:
        # Sử dụng GenericExchangeConnector cho các sàn khác
        exchange_connector = GenericExchangeConnector(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )
    
    # Khởi tạo connector
    await exchange_connector.initialize()
    
    # Tạo collector
    collector = HistoricalDataCollector(
        exchange_connector=exchange_connector,
        max_workers=max_workers
    )
    
    return collector

async def main():
    """
    Hàm chính để chạy collector.
    """
    # Đọc thông tin cấu hình từ biến môi trường
    exchange_id = get_env('DEFAULT_EXCHANGE', 'binance')
    api_key = get_env(f'{exchange_id.upper()}_API_KEY', '')
    api_secret = get_env(f'{exchange_id.upper()}_API_SECRET', '')
    
    # Khởi tạo collector
    collector = await create_data_collector(
        exchange_id=exchange_id,
        api_key=api_key,
        api_secret=api_secret,
        sandbox=True
    )
    
    # Lấy danh sách cặp giao dịch phổ biến
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Thu thập dữ liệu OHLCV
    for symbol in symbols:
        await collector.collect_ohlcv(
            symbol=symbol,
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now()
        )
    
    # Thu thập orderbook snapshot
    for symbol in symbols:
        await collector.collect_orderbook_snapshots(
            symbol=symbol,
            interval=3600,  # 1 giờ
            depth=20,
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
    
    # Đóng kết nối
    await collector.exchange_connector.close()

if __name__ == "__main__":
    asyncio.run(main())