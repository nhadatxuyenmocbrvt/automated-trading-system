"""
Thu thập dữ liệu thời gian thực từ các sàn giao dịch.
File này cung cấp các lớp và phương thức để theo dõi và xử lý dữ liệu thị trường
theo thời gian thực thông qua các kết nối websocket, hỗ trợ nhiều loại dữ liệu
như ticker, orderbook, kline, và giao dịch.
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
import signal
import queue
from concurrent.futures import ThreadPoolExecutor
import threading

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector
from config.logging_config import setup_logger
from config.constants import Timeframe, TIMEFRAME_TO_SECONDS, Exchange, ErrorCode
from config.env import get_env
from config.system_config import DATA_DIR, BASE_DIR


class DataHandler:
    """
    Lớp cơ sở để xử lý dữ liệu thời gian thực.
    Các lớp con cần triển khai phương thức process_data.
    """
    
    def __init__(self, name: str = "base_handler"):
        """
        Khởi tạo handler dữ liệu.
        
        Args:
            name: Tên của handler
        """
        self.name = name
        self.logger = setup_logger(f"data_handler_{name}")
    
    async def process_data(self, data: Dict) -> None:
        """
        Xử lý dữ liệu nhận được.
        
        Args:
            data: Dữ liệu cần xử lý
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")


class ConsoleOutputHandler(DataHandler):
    """
    Handler đơn giản để in dữ liệu ra console.
    """
    
    def __init__(self, name: str = "console", log_level: str = "INFO"):
        """
        Khởi tạo handler console.
        
        Args:
            name: Tên của handler
            log_level: Mức độ log ('DEBUG', 'INFO', 'WARNING', ...)
        """
        super().__init__(name)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    async def process_data(self, data: Dict) -> None:
        """
        In dữ liệu ra console.
        
        Args:
            data: Dữ liệu cần xử lý
        """
        data_type = data.get('type', 'unknown')
        symbol = data.get('symbol', 'unknown')
        
        if data_type == 'ticker':
            if self.log_level <= logging.INFO:
                self.logger.info(f"Ticker {symbol}: Last: {data.get('last')}, Bid: {data.get('bid')}, Ask: {data.get('ask')}")
        elif data_type == 'kline':
            if self.log_level <= logging.INFO:
                kline = data.get('data', {})
                self.logger.info(f"Kline {symbol} {data.get('interval')}: Open: {kline.get('open')}, Close: {kline.get('close')}")
        elif data_type == 'orderbook':
            if self.log_level <= logging.DEBUG:
                top_bid = data.get('bids', [[0, 0]])[0]
                top_ask = data.get('asks', [[0, 0]])[0]
                self.logger.debug(f"Orderbook {symbol}: Top Bid: {top_bid}, Top Ask: {top_ask}")
        elif data_type == 'trade':
            if self.log_level <= logging.DEBUG:
                self.logger.debug(f"Trade {symbol}: Price: {data.get('price')}, Size: {data.get('amount')}, Side: {data.get('side')}")
        else:
            if self.log_level <= logging.DEBUG:
                self.logger.debug(f"Data {data_type} for {symbol}: {json.dumps(data, default=str)[:100]}...")


class CSVStorageHandler(DataHandler):
    """
    Handler để lưu trữ dữ liệu vào file CSV.
    """
    
    def __init__(
        self, 
        name: str = "csv_storage",
        data_dir: Path = None,
        flush_interval: int = 60,  # Lưu dữ liệu xuống đĩa mỗi 60 giây
        max_rows_in_memory: int = 10000,  # Lưu dữ liệu xuống đĩa sau khi đạt 10000 bản ghi
        max_rows_per_file: int = 1000000  # Tối đa 1 triệu bản ghi mỗi file
    ):
        """
        Khởi tạo handler CSV.
        
        Args:
            name: Tên của handler
            data_dir: Thư mục lưu trữ dữ liệu
            flush_interval: Khoảng thời gian giữa các lần lưu dữ liệu (giây)
            max_rows_in_memory: Số bản ghi tối đa lưu trong bộ nhớ
            max_rows_per_file: Số bản ghi tối đa trong một file
        """
        super().__init__(name)
        
        if data_dir is None:
            self.data_dir = DATA_DIR / 'realtime'
        else:
            self.data_dir = data_dir / 'realtime'
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thư mục cho từng loại dữ liệu
        self.ticker_dir = self.data_dir / 'ticker'
        self.kline_dir = self.data_dir / 'kline'
        self.orderbook_dir = self.data_dir / 'orderbook'
        self.trade_dir = self.data_dir / 'trade'
        
        for dir_path in [self.ticker_dir, self.kline_dir, self.orderbook_dir, self.trade_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.flush_interval = flush_interval
        self.max_rows_in_memory = max_rows_in_memory
        self.max_rows_per_file = max_rows_per_file
        
        # Buffer lưu trữ dữ liệu trong bộ nhớ
        self.data_buffers = {
            'ticker': {},
            'kline': {},
            'orderbook': {},
            'trade': {}
        }
        
        # Thời gian flush cuối cùng
        self.last_flush_time = time.time()
        
        # Khởi động task flush định kỳ
        self.flush_task = asyncio.create_task(self._periodic_flush())
        
        self.logger.info(f"Đã khởi tạo CSV storage handler: {self.data_dir}")
    
    async def process_data(self, data: Dict) -> None:
        """
        Xử lý và lưu trữ dữ liệu.
        
        Args:
            data: Dữ liệu cần xử lý
        """
        data_type = data.get('type', 'unknown')
        symbol = data.get('symbol', 'unknown')
        
        if data_type not in self.data_buffers:
            self.logger.warning(f"Loại dữ liệu không được hỗ trợ: {data_type}")
            return
        
        # Tạo key cho buffer
        buffer_key = f"{symbol}_{data_type}"
        
        # Tạo buffer nếu chưa tồn tại
        if buffer_key not in self.data_buffers[data_type]:
            self.data_buffers[data_type][buffer_key] = []
        
        # Chuẩn hóa dữ liệu tùy thuộc vào loại
        normalized_data = self._normalize_data(data)
        
        # Thêm dữ liệu vào buffer
        self.data_buffers[data_type][buffer_key].append(normalized_data)
        
        # Kiểm tra xem có cần flush không
        if len(self.data_buffers[data_type][buffer_key]) >= self.max_rows_in_memory:
            await self._flush_buffer(data_type, buffer_key)
    
    def _normalize_data(self, data: Dict) -> Dict:
        """
        Chuẩn hóa dữ liệu để lưu trữ.
        
        Args:
            data: Dữ liệu cần chuẩn hóa
            
        Returns:
            Dữ liệu đã chuẩn hóa
        """
        data_type = data.get('type', 'unknown')
        timestamp = data.get('timestamp', datetime.now().timestamp() * 1000)
        
        # Chuyển đổi timestamp nếu cần
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp() * 1000
        
        normalized = {
            'timestamp': timestamp,
            'datetime': data.get('datetime', datetime.fromtimestamp(timestamp / 1000).isoformat()),
            'symbol': data.get('symbol', 'unknown')
        }
        
        # Chuẩn hóa dữ liệu dựa trên loại
        if data_type == 'ticker':
            normalized.update({
                'last': data.get('last', None),
                'bid': data.get('bid', None),
                'ask': data.get('ask', None),
                'volume': data.get('volume', None),
                'high': data.get('high', None),
                'low': data.get('low', None),
                'change': data.get('change', None),
                'percentage': data.get('percentage', None),
                'base_volume': data.get('baseVolume', None),
                'quote_volume': data.get('quoteVolume', None)
            })
        elif data_type == 'kline':
            kline_data = data.get('data', {})
            normalized.update({
                'interval': data.get('interval', '1m'),
                'open': kline_data.get('open', None),
                'high': kline_data.get('high', None),
                'low': kline_data.get('low', None),
                'close': kline_data.get('close', None),
                'volume': kline_data.get('volume', None),
                'is_closed': kline_data.get('isClosed', False),
                'quote_volume': kline_data.get('quoteVolume', None),
                'trades': kline_data.get('trades', None)
            })
        elif data_type == 'orderbook':
            # Lưu trữ dữ liệu sổ lệnh theo định dạng đơn giản hơn
            normalized.update({
                'bids': json.dumps(data.get('bids', [])),
                'asks': json.dumps(data.get('asks', [])),
                'nonce': data.get('nonce', None)
            })
        elif data_type == 'trade':
            normalized.update({
                'id': data.get('id', None),
                'price': data.get('price', None),
                'amount': data.get('amount', None),
                'cost': data.get('cost', None),
                'side': data.get('side', None),
                'taker_or_maker': data.get('takerOrMaker', None),
                'fee': json.dumps(data.get('fee', {})) if data.get('fee') else None
            })
        
        return normalized
    
    async def _periodic_flush(self) -> None:
        """
        Định kỳ lưu dữ liệu từ bộ nhớ xuống đĩa.
        """
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all_buffers()
        except asyncio.CancelledError:
            # Flush tất cả dữ liệu khi task bị hủy
            await self._flush_all_buffers()
            self.logger.info("Periodic flush task đã bị hủy, đã lưu tất cả dữ liệu")
        except Exception as e:
            self.logger.error(f"Lỗi trong periodic flush task: {e}")
    
    async def _flush_all_buffers(self) -> None:
        """
        Lưu tất cả dữ liệu trong bộ nhớ xuống đĩa.
        """
        current_time = time.time()
        
        # Kiểm tra xem đã đến thời gian flush chưa
        if current_time - self.last_flush_time < self.flush_interval:
            return
        
        for data_type in self.data_buffers:
            for buffer_key in list(self.data_buffers[data_type].keys()):
                if self.data_buffers[data_type][buffer_key]:
                    await self._flush_buffer(data_type, buffer_key)
        
        self.last_flush_time = current_time
    
    async def _flush_buffer(self, data_type: str, buffer_key: str) -> None:
        """
        Lưu dữ liệu từ buffer xuống đĩa.
        
        Args:
            data_type: Loại dữ liệu
            buffer_key: Khóa của buffer
        """
        # Kiểm tra xem buffer có dữ liệu không
        if not self.data_buffers[data_type][buffer_key]:
            return
        
        # Lấy dữ liệu từ buffer
        buffer_data = self.data_buffers[data_type][buffer_key]
        self.data_buffers[data_type][buffer_key] = []  # Reset buffer
        
        # Xác định thư mục lưu trữ
        if data_type == 'ticker':
            dir_path = self.ticker_dir
        elif data_type == 'kline':
            dir_path = self.kline_dir
        elif data_type == 'orderbook':
            dir_path = self.orderbook_dir
        elif data_type == 'trade':
            dir_path = self.trade_dir
        else:
            self.logger.warning(f"Loại dữ liệu không được hỗ trợ: {data_type}")
            return
        
        # Tạo tên file
        symbol = buffer_key.split('_')[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{data_type}_{timestamp}.csv"
        file_path = dir_path / filename
        
        try:
            # Tạo DataFrame từ dữ liệu
            df = pd.DataFrame(buffer_data)
            
            # Kiểm tra file hiện có
            file_exists = file_path.exists()
            
            # Ghi dữ liệu
            if file_exists:
                # Thêm vào file hiện có
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                # Tạo file mới
                df.to_csv(file_path, header=True, index=False)
            
            self.logger.debug(f"Đã lưu {len(buffer_data)} bản ghi {data_type} vào {file_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu {data_type} xuống đĩa: {e}")
            # Lưu lại dữ liệu vào buffer để thử lại sau
            self.data_buffers[data_type][buffer_key].extend(buffer_data)
    
    async def close(self) -> None:
        """
        Đóng handler và lưu tất cả dữ liệu còn lại.
        """
        # Hủy task flush định kỳ
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush tất cả dữ liệu
        await self._flush_all_buffers()
        self.logger.info("Đã đóng CSV storage handler và lưu tất cả dữ liệu")


class DatabaseStorageHandler(DataHandler):
    """
    Handler để lưu trữ dữ liệu vào cơ sở dữ liệu.
    Đây là lớp trừu tượng, các lớp con cần triển khai phương thức _save_to_database.
    """
    
    def __init__(
        self, 
        name: str = "db_storage",
        batch_size: int = 100,
        flush_interval: int = 10
    ):
        """
        Khởi tạo handler database.
        
        Args:
            name: Tên của handler
            batch_size: Số bản ghi tối đa trong một batch
            flush_interval: Khoảng thời gian giữa các lần lưu dữ liệu (giây)
        """
        super().__init__(name)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Buffer lưu trữ dữ liệu trong bộ nhớ
        self.data_buffers = {
            'ticker': [],
            'kline': [],
            'orderbook': [],
            'trade': []
        }
        
        # Thời gian flush cuối cùng
        self.last_flush_time = time.time()
        
        # Khởi động task flush định kỳ
        self.flush_task = asyncio.create_task(self._periodic_flush())
        
        self.logger.info(f"Đã khởi tạo database storage handler")
    
    async def process_data(self, data: Dict) -> None:
        """
        Xử lý và lưu trữ dữ liệu.
        
        Args:
            data: Dữ liệu cần xử lý
        """
        data_type = data.get('type', 'unknown')
        
        if data_type not in self.data_buffers:
            self.logger.warning(f"Loại dữ liệu không được hỗ trợ: {data_type}")
            return
        
        # Thêm dữ liệu vào buffer
        self.data_buffers[data_type].append(data)
        
        # Kiểm tra xem có cần flush không
        if len(self.data_buffers[data_type]) >= self.batch_size:
            await self._flush_buffer(data_type)
    
    async def _periodic_flush(self) -> None:
        """
        Định kỳ lưu dữ liệu từ bộ nhớ xuống cơ sở dữ liệu.
        """
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all_buffers()
        except asyncio.CancelledError:
            # Flush tất cả dữ liệu khi task bị hủy
            await self._flush_all_buffers()
            self.logger.info("Periodic flush task đã bị hủy, đã lưu tất cả dữ liệu")
        except Exception as e:
            self.logger.error(f"Lỗi trong periodic flush task: {e}")
    
    async def _flush_all_buffers(self) -> None:
        """
        Lưu tất cả dữ liệu trong bộ nhớ xuống cơ sở dữ liệu.
        """
        current_time = time.time()
        
        # Kiểm tra xem đã đến thời gian flush chưa
        if current_time - self.last_flush_time < self.flush_interval:
            return
        
        for data_type in self.data_buffers:
            if self.data_buffers[data_type]:
                await self._flush_buffer(data_type)
        
        self.last_flush_time = current_time
    
    async def _flush_buffer(self, data_type: str) -> None:
        """
        Lưu dữ liệu từ buffer xuống cơ sở dữ liệu.
        
        Args:
            data_type: Loại dữ liệu
        """
        # Kiểm tra xem buffer có dữ liệu không
        if not self.data_buffers[data_type]:
            return
        
        # Lấy dữ liệu từ buffer
        buffer_data = self.data_buffers[data_type]
        self.data_buffers[data_type] = []  # Reset buffer
        
        try:
            # Lưu dữ liệu xuống cơ sở dữ liệu
            await self._save_to_database(data_type, buffer_data)
            self.logger.debug(f"Đã lưu {len(buffer_data)} bản ghi {data_type} vào cơ sở dữ liệu")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu {data_type} xuống cơ sở dữ liệu: {e}")
            # Lưu lại dữ liệu vào buffer để thử lại sau
            self.data_buffers[data_type].extend(buffer_data)
    
    async def _save_to_database(self, data_type: str, data: List[Dict]) -> None:
        """
        Lưu dữ liệu xuống cơ sở dữ liệu.
        Lớp con cần triển khai phương thức này.
        
        Args:
            data_type: Loại dữ liệu
            data: Dữ liệu cần lưu
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")
    
    async def close(self) -> None:
        """
        Đóng handler và lưu tất cả dữ liệu còn lại.
        """
        # Hủy task flush định kỳ
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush tất cả dữ liệu
        await self._flush_all_buffers()
        self.logger.info("Đã đóng database storage handler và lưu tất cả dữ liệu")


class CustomProcessingHandler(DataHandler):
    """
    Handler cho phép xử lý tùy chỉnh thông qua một hàm callback.
    """
    
    def __init__(
        self, 
        callback: Callable[[Dict], None],
        name: str = "custom_processor",
        filter_func: Optional[Callable[[Dict], bool]] = None
    ):
        """
        Khởi tạo custom handler.
        
        Args:
            callback: Hàm callback để xử lý dữ liệu
            name: Tên của handler
            filter_func: Hàm lọc dữ liệu (trả về True nếu cần xử lý)
        """
        super().__init__(name)
        self.callback = callback
        self.filter_func = filter_func
        self.logger.info(f"Đã khởi tạo custom processing handler")
    
    async def process_data(self, data: Dict) -> None:
        """
        Xử lý dữ liệu bằng hàm callback.
        
        Args:
            data: Dữ liệu cần xử lý
        """
        # Kiểm tra filter nếu có
        if self.filter_func is not None and not self.filter_func(data):
            return
        
        try:
            # Chạy callback
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(data)
            else:
                self.callback(data)
        except Exception as e:
            self.logger.error(f"Lỗi khi chạy callback: {e}")


class RealtimeDataStream:
    """
    Lớp chính để thu thập và xử lý dữ liệu thời gian thực.
    """
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        data_handlers: Optional[List[DataHandler]] = None,
        websocket_reconnect_interval: int = 30,
        heartbeat_interval: int = 15
    ):
        """
        Khởi tạo stream dữ liệu thời gian thực.
        
        Args:
            exchange_connector: Kết nối với sàn giao dịch
            data_handlers: Danh sách các handler để xử lý dữ liệu
            websocket_reconnect_interval: Thời gian giữa các lần thử kết nối lại (giây)
            heartbeat_interval: Thời gian giữa các lần kiểm tra kết nối (giây)
        """
        self.exchange_connector = exchange_connector
        self.exchange_id = exchange_connector.exchange_id
        self.logger = setup_logger(f"realtime_data_stream_{self.exchange_id}")
        
        # Handlers
        self.data_handlers = data_handlers or []
        
        # Cấu hình kết nối
        self.websocket_reconnect_interval = websocket_reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        
        # Trạng thái
        self.is_running = False
        self.subscription_symbols = set()
        self.subscription_channels = {}  # {symbol: [channels]}
        self.subscription_tasks = {}  # {symbol_channel: task}
        
        # Queue dữ liệu
        self.data_queue = asyncio.Queue()
        
        # Lock để đồng bộ hóa các thao tác
        self.lock = asyncio.Lock()
        
        # Tasks
        self.processor_task = None
        self.heartbeat_task = None
        
        # Callbacks
        self._message_callbacks = []
        
        self.logger.info(f"Đã khởi tạo RealtimeDataStream cho {self.exchange_id}")
    
    def add_data_handler(self, handler: DataHandler) -> None:
        """
        Thêm handler dữ liệu.
        
        Args:
            handler: Handler cần thêm
        """
        self.data_handlers.append(handler)
        self.logger.info(f"Đã thêm handler: {handler.name}")
    
    def add_message_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Thêm callback để xử lý thông điệp raw.
        
        Args:
            callback: Hàm callback
        """
        self._message_callbacks.append(callback)
    
    async def _process_message(self, message: Dict) -> None:
        """
        Xử lý thông điệp.
        
        Args:
            message: Thông điệp cần xử lý
        """
        # Thêm timestamp nếu chưa có
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().timestamp() * 1000
        
        # Thêm thông điệp vào queue
        await self.data_queue.put(message)
        
        # Gọi các message callbacks
        for callback in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"Lỗi khi gọi message callback: {e}")
    
    async def _websocket_message_handler(self, message: Dict) -> None:
        """
        Xử lý thông điệp từ websocket.
        
        Args:
            message: Thông điệp từ websocket
        """
        try:
            # Chuẩn hóa thông điệp
            normalized_message = self._normalize_ws_message(message)
            
            if normalized_message:
                await self._process_message(normalized_message)
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý thông điệp websocket: {e}")
    
    def _normalize_ws_message(self, message: Dict) -> Optional[Dict]:
        """
        Chuẩn hóa thông điệp websocket theo định dạng chung.
        
        Args:
            message: Thông điệp gốc
            
        Returns:
            Thông điệp đã chuẩn hóa hoặc None nếu không hỗ trợ
        """
        try:
            # Thông điệp từ các sàn khác nhau có cấu trúc khác nhau
            # Cần chuẩn hóa để có định dạng chung
            
            # Xác định sàn giao dịch
            if self.exchange_id == 'binance':
                return self._normalize_binance_message(message)
            elif self.exchange_id == 'bybit':
                return self._normalize_bybit_message(message)
            else:
                # Xử lý chung cho các sàn khác
                return self._normalize_generic_message(message)
        except Exception as e:
            self.logger.error(f"Lỗi khi chuẩn hóa thông điệp: {e}")
            return None
    
    def _normalize_binance_message(self, message: Dict) -> Optional[Dict]:
        """
        Chuẩn hóa thông điệp từ Binance.
        
        Args:
            message: Thông điệp gốc
            
        Returns:
            Thông điệp đã chuẩn hóa
        """
        if 'stream' not in message or 'data' not in message:
            return None
        
        stream = message['stream']
        data = message['data']
        
        # Xác định loại dữ liệu và symbol
        data_type = None
        symbol = None
        
        if '@ticker' in stream:
            data_type = 'ticker'
            symbol = data.get('s', '').replace('USDT', '/USDT')
            
            return {
                'type': data_type,
                'symbol': symbol,
                'timestamp': data.get('E', int(time.time() * 1000)),
                'datetime': datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000).isoformat(),
                'last': float(data.get('c', 0)),
                'open': float(data.get('o', 0)),
                'high': float(data.get('h', 0)),
                'low': float(data.get('l', 0)),
                'bid': float(data.get('b', 0)),
                'ask': float(data.get('a', 0)),
                'volume': float(data.get('v', 0)),
                'change': float(data.get('p', 0)),
                'percentage': float(data.get('P', 0)),
                'baseVolume': float(data.get('v', 0)),
                'quoteVolume': float(data.get('q', 0))
            }
            
        elif '@kline' in stream:
            data_type = 'kline'
            parts = stream.split('_')
            symbol = parts[0].replace('USDT', '/USDT')
            interval = parts[1].split('@')[0] if len(parts) > 1 else '1m'
            
            k = data.get('k', {})
            
            return {
                'type': data_type,
                'symbol': symbol,
                'interval': interval,
                'timestamp': data.get('E', int(time.time() * 1000)),
                'datetime': datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000).isoformat(),
                'data': {
                    'open': float(k.get('o', 0)),
                    'high': float(k.get('h', 0)),
                    'low': float(k.get('l', 0)),
                    'close': float(k.get('c', 0)),
                    'volume': float(k.get('v', 0)),
                    'isClosed': k.get('x', False),
                    'quoteVolume': float(k.get('q', 0)),
                    'trades': int(k.get('n', 0))
                }
            }
            
        elif '@depth' in stream:
            data_type = 'orderbook'
            parts = stream.split('@')
            symbol = parts[0].replace('USDT', '/USDT')
            
            return {
                'type': data_type,
                'symbol': symbol,
                'timestamp': data.get('E', int(time.time() * 1000)),
                'datetime': datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000).isoformat(),
                'bids': [[float(price), float(amount)] for price, amount in data.get('b', [])],
                'asks': [[float(price), float(amount)] for price, amount in data.get('a', [])],
                'nonce': data.get('u', None)
            }
            
        elif '@trade' in stream:
            data_type = 'trade'
            parts = stream.split('@')
            symbol = parts[0].replace('USDT', '/USDT')
            
            return {
                'type': data_type,
                'symbol': symbol,
                'id': str(data.get('t', '')),
                'timestamp': data.get('E', int(time.time() * 1000)),
                'datetime': datetime.fromtimestamp(data.get('E', time.time() * 1000) / 1000).isoformat(),
                'side': 'buy' if data.get('m', False) else 'sell',
                'price': float(data.get('p', 0)),
                'amount': float(data.get('q', 0)),
                'cost': float(data.get('p', 0)) * float(data.get('q', 0))
            }
        
        return None
    
    def _normalize_bybit_message(self, message: Dict) -> Optional[Dict]:
        """
        Chuẩn hóa thông điệp từ Bybit.
        
        Args:
            message: Thông điệp gốc
            
        Returns:
            Thông điệp đã chuẩn hóa
        """
        if 'topic' not in message or 'data' not in message:
            return None
        
        topic = message['topic']
        data = message['data']
        
        # Xác định loại dữ liệu và symbol
        data_type = None
        symbol = None
        
        if 'tickers' in topic:
            data_type = 'ticker'
            parts = topic.split('.')
            symbol = parts[1] if len(parts) > 1 else None
            symbol = symbol.replace('USDT', '/USDT') if symbol else None
            
            if isinstance(data, dict):
                return {
                    'type': data_type,
                    'symbol': symbol,
                    'timestamp': int(data.get('timestamp', time.time() * 1000)),
                    'datetime': datetime.fromtimestamp(int(data.get('timestamp', time.time() * 1000)) / 1000).isoformat(),
                    'last': float(data.get('lastPrice', 0)),
                    'open': float(data.get('openPrice', 0)),
                    'high': float(data.get('highPrice24h', 0)),
                    'low': float(data.get('lowPrice24h', 0)),
                    'bid': float(data.get('bid1Price', 0)),
                    'ask': float(data.get('ask1Price', 0)),
                    'volume': float(data.get('volume24h', 0)),
                    'change': float(data.get('price24hPcnt', 0)) * 100,
                    'percentage': float(data.get('price24hPcnt', 0)) * 100,
                    'baseVolume': float(data.get('volume24h', 0)),
                    'quoteVolume': float(data.get('turnover24h', 0))
                }
            
        elif 'kline' in topic:
            data_type = 'kline'
            parts = topic.split('.')
            if len(parts) > 2:
                interval = parts[1]
                symbol = parts[2].replace('USDT', '/USDT')
            else:
                interval = '1m'
                symbol = None
            
            if isinstance(data, list) and len(data) > 0:
                kline = data[0]
                return {
                    'type': data_type,
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': int(kline.get('timestamp', time.time() * 1000)),
                    'datetime': datetime.fromtimestamp(int(kline.get('timestamp', time.time() * 1000)) / 1000).isoformat(),
                    'data': {
                        'open': float(kline.get('open', 0)),
                        'high': float(kline.get('high', 0)),
                        'low': float(kline.get('low', 0)),
                        'close': float(kline.get('close', 0)),
                        'volume': float(kline.get('volume', 0)),
                        'isClosed': True,
                        'quoteVolume': float(kline.get('turnover', 0)),
                        'trades': int(kline.get('confirm', 0))
                    }
                }
            
        elif 'orderbook' in topic:
            data_type = 'orderbook'
            parts = topic.split('.')
            symbol = parts[1] if len(parts) > 1 else None
            symbol = symbol.replace('USDT', '/USDT') if symbol else None
            
            return {
                'type': data_type,
                'symbol': symbol,
                'timestamp': int(data.get('timestamp', time.time() * 1000)),
                'datetime': datetime.fromtimestamp(int(data.get('timestamp', time.time() * 1000)) / 1000).isoformat(),
                'bids': [[float(item[0]), float(item[1])] for item in data.get('b', [])],
                'asks': [[float(item[0]), float(item[1])] for item in data.get('a', [])],
                'nonce': data.get('u', None)
            }
            
        elif 'publicTrade' in topic:
            data_type = 'trade'
            parts = topic.split('.')
            symbol = parts[1] if len(parts) > 1 else None
            symbol = symbol.replace('USDT', '/USDT') if symbol else None
            
            if isinstance(data, list) and len(data) > 0:
                trade = data[0]
                return {
                    'type': data_type,
                    'symbol': symbol,
                    'id': str(trade.get('i', '')),
                    'timestamp': int(trade.get('T', time.time() * 1000)),
                    'datetime': datetime.fromtimestamp(int(trade.get('T', time.time() * 1000)) / 1000).isoformat(),
                    'side': trade.get('S', '').lower(),
                    'price': float(trade.get('p', 0)),
                    'amount': float(trade.get('v', 0)),
                    'cost': float(trade.get('p', 0)) * float(trade.get('v', 0))
                }
        
        return None
    
    def _normalize_generic_message(self, message: Dict) -> Optional[Dict]:
        """
        Chuẩn hóa thông điệp từ các sàn khác.
        
        Args:
            message: Thông điệp gốc
            
        Returns:
            Thông điệp đã chuẩn hóa
        """
        # Xử lý chung cho các sàn khác
        # Đây là phần khó vì mỗi sàn có cấu trúc riêng
        # Chúng ta sẽ cố gắng trích xuất thông tin quan trọng nhất
        
        current_time = int(time.time() * 1000)
        
        # Thử xác định loại dữ liệu
        if 'ticker' in message:
            return {
                'type': 'ticker',
                'symbol': message.get('symbol', 'unknown'),
                'timestamp': message.get('timestamp', current_time),
                'datetime': datetime.fromtimestamp(message.get('timestamp', current_time) / 1000).isoformat(),
                'last': message.get('last', None),
                'bid': message.get('bid', None),
                'ask': message.get('ask', None),
                'high': message.get('high', None),
                'low': message.get('low', None),
                'volume': message.get('volume', None),
                'change': message.get('change', None),
                'percentage': message.get('percentage', None)
            }
        elif 'kline' in message or 'candle' in message or 'ohlc' in message:
            kline_data = message.get('kline', message.get('candle', message.get('ohlc', {})))
            
            return {
                'type': 'kline',
                'symbol': message.get('symbol', 'unknown'),
                'interval': message.get('interval', '1m'),
                'timestamp': message.get('timestamp', current_time),
                'datetime': datetime.fromtimestamp(message.get('timestamp', current_time) / 1000).isoformat(),
                'data': {
                    'open': kline_data.get('open', None),
                    'high': kline_data.get('high', None),
                    'low': kline_data.get('low', None),
                    'close': kline_data.get('close', None),
                    'volume': kline_data.get('volume', None),
                    'isClosed': kline_data.get('isClosed', False),
                }
            }
        elif 'orderbook' in message or 'depth' in message:
            return {
                'type': 'orderbook',
                'symbol': message.get('symbol', 'unknown'),
                'timestamp': message.get('timestamp', current_time),
                'datetime': datetime.fromtimestamp(message.get('timestamp', current_time) / 1000).isoformat(),
                'bids': message.get('bids', []),
                'asks': message.get('asks', []),
                'nonce': message.get('nonce', None)
            }
        elif 'trade' in message or 'execution' in message:
            return {
                'type': 'trade',
                'symbol': message.get('symbol', 'unknown'),
                'id': message.get('id', None),
                'timestamp': message.get('timestamp', current_time),
                'datetime': datetime.fromtimestamp(message.get('timestamp', current_time) / 1000).isoformat(),
                'side': message.get('side', None),
                'price': message.get('price', None),
                'amount': message.get('amount', None),
                'cost': message.get('cost', None)
            }
        
        return None
    
    async def _processor_loop(self) -> None:
        """
        Vòng lặp chính để xử lý dữ liệu từ queue.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý dữ liệu")
        
        try:
            while self.is_running:
                # Lấy dữ liệu từ queue
                message = await self.data_queue.get()
                
                # Xử lý dữ liệu với tất cả các handlers
                for handler in self.data_handlers:
                    try:
                        await handler.process_data(message)
                    except Exception as e:
                        self.logger.error(f"Lỗi khi xử lý dữ liệu với handler {handler.name}: {e}")
                
                # Đánh dấu task đã hoàn thành
                self.data_queue.task_done()
        
        except asyncio.CancelledError:
            self.logger.info("Vòng lặp xử lý dữ liệu đã bị hủy")
        except Exception as e:
            self.logger.error(f"Lỗi trong vòng lặp xử lý dữ liệu: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """
        Vòng lặp kiểm tra kết nối định kỳ.
        """
        self.logger.info("Bắt đầu vòng lặp kiểm tra kết nối")
        
        try:
            while self.is_running:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Kiểm tra các kết nối websocket
                await self._check_subscriptions()
        
        except asyncio.CancelledError:
            self.logger.info("Vòng lặp kiểm tra kết nối đã bị hủy")
        except Exception as e:
            self.logger.error(f"Lỗi trong vòng lặp kiểm tra kết nối: {e}")
    
    async def _check_subscriptions(self) -> None:
        """
        Kiểm tra và khôi phục các kết nối websocket nếu cần.
        """
        async with self.lock:
            # Kiểm tra trạng thái của tất cả các subscription tasks
            for key, task in list(self.subscription_tasks.items()):
                if task.done():
                    exception = task.exception()
                    if exception:
                        self.logger.warning(f"Subscription task {key} đã kết thúc với lỗi: {exception}")
                    else:
                        self.logger.warning(f"Subscription task {key} đã kết thúc mà không có lỗi")
                    
                    # Lấy thông tin subscription
                    parts = key.split('_')
                    symbol = parts[0]
                    channels = self.subscription_channels.get(symbol, [])
                    
                    # Tạo lại subscription
                    self.logger.info(f"Khôi phục subscription cho {symbol} với kênh {channels}")
                    await self._subscribe_symbol(symbol, channels)
    
    async def _subscribe_symbol(self, symbol: str, channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            channels: Danh sách kênh
        """
        # Chuẩn hóa channels nếu cần
        normalized_channels = self._normalize_channels(channels)
        
        # Tạo task đăng ký
        task_key = f"{symbol}_{'_'.join(normalized_channels)}"
        
        # Hủy task cũ nếu có
        if task_key in self.subscription_tasks:
            old_task = self.subscription_tasks[task_key]
            if not old_task.done():
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass
        
        # Tạo closure để đăng ký và xử lý kết nối
        async def subscribe_and_process():
            while self.is_running:
                try:
                    self.logger.info(f"Đăng ký {symbol} với kênh {normalized_channels}")
                    
                    # Nếu exchange_connector có phương thức _process_websocket_message_raw, sử dụng callback trực tiếp
                    # Nếu không, sử dụng callback gián tiếp
                    if hasattr(self.exchange_connector, '_process_websocket_message_raw'):
                        # Đăng ký callback trực tiếp
                        # Sử dụng bind để tạo hàm bound với đối tượng exchange_connector
                        original_callback = self.exchange_connector._process_websocket_message_raw
                        
                        # Tạo wrapper để chuyển tiếp thông điệp
                        async def message_forwarder(message):
                            # Xử lý bởi exchange_connector
                            await original_callback(message)
                            # Rồi chuyển tiếp cho chúng ta
                            await self._websocket_message_handler(json.loads(message) if isinstance(message, str) else message)
                        
                        # Thay thế callback
                        self.exchange_connector._process_websocket_message_raw = message_forwarder
                        
                        # Đăng ký subscription
                        await self.exchange_connector.subscribe_to_websocket(symbol, normalized_channels)
                        
                    else:
                        # Nếu không hỗ trợ callback trực tiếp, sử dụng phương thức chung
                        # Đây là một cách tiếp cận đơn giản nhưng có thể không hoạt động cho tất cả các sàn
                        self.logger.warning(f"Không tìm thấy phương thức _process_websocket_message_raw, sử dụng phương thức chung")
                        
                        # Thử sử dụng phương thức subscribe_to_websocket
                        await self.exchange_connector.subscribe_to_websocket(symbol, normalized_channels)
                    
                    # Lưu thông tin subscription
                    self.subscription_symbols.add(symbol)
                    self.subscription_channels[symbol] = normalized_channels
                    
                    # Đợi cho đến khi task bị hủy
                    while True:
                        await asyncio.sleep(60)  # Kiểm tra định kỳ
                
                except asyncio.CancelledError:
                    self.logger.info(f"Task subscription cho {symbol} đã bị hủy")
                    raise
                except Exception as e:
                    self.logger.error(f"Lỗi khi đăng ký {symbol}: {e}")
                    await asyncio.sleep(self.websocket_reconnect_interval)
        
        # Tạo và lưu task
        task = asyncio.create_task(subscribe_and_process())
        self.subscription_tasks[task_key] = task
    
    def _normalize_channels(self, channels: List[str]) -> List[str]:
        """
        Chuẩn hóa danh sách kênh dựa trên sàn giao dịch.
        
        Args:
            channels: Danh sách kênh gốc
            
        Returns:
            Danh sách kênh đã chuẩn hóa
        """
        normalized = []
        
        for channel in channels:
            if self.exchange_id == 'binance':
                # Binance sử dụng ticker, kline_1m, depth, trade
                if channel in ['ticker', 'depth', 'trade']:
                    normalized.append(channel)
                elif channel.startswith('kline_'):
                    normalized.append(channel)
                elif channel in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    normalized.append(f"kline_{channel}")
            elif self.exchange_id == 'bybit':
                # Bybit sử dụng ticker, kline.1, orderbook, trade
                if channel == 'ticker':
                    normalized.append('ticker')
                elif channel == 'depth' or channel == 'orderbook':
                    normalized.append('orderbook')
                elif channel == 'trade':
                    normalized.append('trade')
                elif channel.startswith('kline_'):
                    interval = channel.split('_')[1]
                    normalized.append(f"kline.{interval}")
                elif channel in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                    normalized.append(f"kline.{channel}")
            else:
                # Sàn khác, sử dụng channels gốc
                normalized.append(channel)
        
        return normalized
    
    async def subscribe(self, symbol: str, channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            channels: Danh sách kênh
        """
        async with self.lock:
            await self._subscribe_symbol(symbol, channels)
    
    async def subscribe_multiple(self, symbols: List[str], channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu cho nhiều cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            channels: Danh sách kênh
        """
        for symbol in symbols:
            await self.subscribe(symbol, channels)
    
    async def unsubscribe(self, symbol: str, channels: Optional[List[str]] = None) -> None:
        """
        Hủy đăng ký nhận dữ liệu.
        
        Args:
            symbol: Cặp giao dịch
            channels: Danh sách kênh (None để hủy tất cả)
        """
        async with self.lock:
            # Nếu channels là None, hủy tất cả các kênh cho symbol
            if channels is None:
                for key, task in list(self.subscription_tasks.items()):
                    if key.startswith(f"{symbol}_"):
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        del self.subscription_tasks[key]
                
                if symbol in self.subscription_channels:
                    del self.subscription_channels[symbol]
                
                self.subscription_symbols.discard(symbol)
                self.logger.info(f"Đã hủy đăng ký tất cả kênh cho {symbol}")
            else:
                # Hủy các kênh cụ thể
                normalized_channels = self._normalize_channels(channels)
                
                # Lấy danh sách kênh hiện tại
                current_channels = self.subscription_channels.get(symbol, [])
                
                # Loại bỏ các kênh cần hủy
                new_channels = [ch for ch in current_channels if ch not in normalized_channels]
                
                # Nếu không còn kênh nào, hủy đăng ký symbol
                if not new_channels:
                    await self.unsubscribe(symbol)
                else:
                    # Hủy đăng ký hiện tại
                    await self.unsubscribe(symbol)
                    
                    # Đăng ký lại với các kênh còn lại
                    await self.subscribe(symbol, new_channels)
                    
                    self.logger.info(f"Đã hủy đăng ký {channels} cho {symbol}")
    
    async def start(self) -> None:
        """
        Bắt đầu stream dữ liệu.
        """
        async with self.lock:
            if self.is_running:
                self.logger.warning("RealtimeDataStream đã đang chạy")
                return
            
            self.is_running = True
            
            # Khởi động các tasks
            self.processor_task = asyncio.create_task(self._processor_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.logger.info("Đã bắt đầu RealtimeDataStream")
    
    async def stop(self) -> None:
        """
        Dừng stream dữ liệu.
        """
        async with self.lock:
            if not self.is_running:
                self.logger.warning("RealtimeDataStream không chạy")
                return
            
            self.is_running = False
            
            # Hủy tất cả các subscription
            for key, task in list(self.subscription_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.subscription_tasks.clear()
            self.subscription_channels.clear()
            self.subscription_symbols.clear()
            
            # Hủy các tasks
            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                try:
                    await self.processor_task
                except asyncio.CancelledError:
                    pass
            
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Đóng tất cả các handlers
            for handler in self.data_handlers:
                if hasattr(handler, 'close'):
                    await handler.close()
            
            # Đảm bảo tất cả dữ liệu trong queue đã được xử lý
            if not self.data_queue.empty():
                self.logger.info("Đang xử lý các dữ liệu còn lại trong queue")
                await self.data_queue.join()
            
            self.logger.info("Đã dừng RealtimeDataStream")
    
    def get_subscription_info(self) -> Dict:
        """
        Lấy thông tin về các subscription hiện tại.
        
        Returns:
            Thông tin subscription
        """
        return {
            "exchange": self.exchange_id,
            "symbols": list(self.subscription_symbols),
            "channels": self.subscription_channels,
            "is_running": self.is_running,
            "queue_size": self.data_queue.qsize(),
            "handlers": [handler.name for handler in self.data_handlers]
        }


# Factory function
async def create_realtime_stream(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True,
    is_futures: bool = False,
    data_handlers: Optional[List[DataHandler]] = None
) -> RealtimeDataStream:
    """
    Tạo một instance của RealtimeDataStream cho sàn giao dịch cụ thể.
    
    Args:
        exchange_id: ID của sàn giao dịch
        api_key: Khóa API
        api_secret: Mật khẩu API
        sandbox: Sử dụng môi trường testnet
        is_futures: Sử dụng tài khoản futures
        data_handlers: Danh sách các handler để xử lý dữ liệu
        
    Returns:
        Instance của RealtimeDataStream
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
        # Sử dụng ExchangeConnector cho các sàn khác
        exchange_connector = ExchangeConnector(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )
    
    # Khởi tạo connector
    await exchange_connector.initialize()
    
    # Tạo data handlers mặc định nếu cần
    if data_handlers is None:
        # Tạo console handler để in dữ liệu
        console_handler = ConsoleOutputHandler(name=f"{exchange_id}_console", log_level="INFO")
        
        # Tạo CSV handler để lưu dữ liệu
        csv_handler = CSVStorageHandler(name=f"{exchange_id}_csv_storage")
        
        data_handlers = [console_handler, csv_handler]
    
    # Tạo stream
    stream = RealtimeDataStream(
        exchange_connector=exchange_connector,
        data_handlers=data_handlers
    )
    
    return stream

async def main():
    """
    Hàm chính để chạy stream.
    """
    # Đọc thông tin cấu hình từ biến môi trường
    exchange_id = get_env('DEFAULT_EXCHANGE', 'binance')
    api_key = get_env(f'{exchange_id.upper()}_API_KEY', '')
    api_secret = get_env(f'{exchange_id.upper()}_API_SECRET', '')
    
    # Tạo stream
    stream = await create_realtime_stream(
        exchange_id=exchange_id,
        api_key=api_key,
        api_secret=api_secret,
        sandbox=True
    )
    
    # Bắt đầu stream
    await stream.start()
    
    try:
        # Đăng ký nhận dữ liệu
        symbols = ['BTC/USDT', 'ETH/USDT']
        channels = ['ticker', 'kline_1m']
        
        for symbol in symbols:
            await stream.subscribe(symbol, channels)
        
        # Đợi một thời gian
        print(f"Đang thu thập dữ liệu từ {exchange_id} cho {symbols}...")
        await asyncio.sleep(300)  # Chạy trong 5 phút
        
    except KeyboardInterrupt:
        print("Đã nhận Ctrl+C, đang dừng...")
    finally:
        # Dừng stream
        await stream.stop()
        
        # Đóng kết nối
        await stream.exchange_connector.close()
        
        print("Đã dừng stream")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Đã nhận Ctrl+C, thoát...")