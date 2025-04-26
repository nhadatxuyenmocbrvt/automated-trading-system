"""
Generic Exchange Connector - Lớp trừu tượng cơ sở để kết nối với các sàn giao dịch tiền điện tử.
"""

import os
import time
import logging
import hmac
import hashlib
import json
import random
import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import asyncio
import ccxt
import ccxt.async_support as ccxt_async
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import các module cấu hình
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from config.security_config import SecretManager
from config.system_config import SystemConfig


class WebSocketManager:
    """
    Quản lý các kết nối websocket và tự động reconnect khi cần thiết.
    """
    
    def __init__(self, logger):
        self.connections = {}  # {connection_key: {websocket, task, symbols, channels}}
        self.logger = logger
        self.connection_count = 0
        self.max_connections = 10  # Số lượng kết nối tối đa
        self.symbols_per_connection = 20  # Số lượng symbols tối đa cho mỗi kết nối
        self.reconnect_delays = [1, 2, 5, 10, 30, 60]  # Backoff tăng dần (giây)
        self.processing_callbacks = {}  # {connection_key: callback_function}
    
    async def add_connection(self, ws_url, symbols, channels, connect_func, message_callback):
        """
        Thêm một kết nối websocket mới hoặc gộp vào kết nối hiện tại nếu có thể.
        
        Args:
            ws_url (str): URL websocket
            symbols (List[str]): Danh sách symbols
            channels (List[str]): Danh sách kênh
            connect_func (Callable): Hàm kết nối websocket
            message_callback (Callable): Hàm xử lý dữ liệu nhận được
        """
        # Tìm kiếm kết nối hiện có có thể thêm symbols vào
        for key, conn_info in self.connections.items():
            if len(conn_info['symbols']) < self.symbols_per_connection and ws_url == conn_info['ws_url']:
                # Thêm symbols vào kết nối hiện có
                new_symbols = [s for s in symbols if s not in conn_info['symbols']]
                if new_symbols:
                    conn_info['symbols'].extend(new_symbols)
                    self.logger.info(f"Đã thêm {len(new_symbols)} symbols vào kết nối websocket hiện có: {key}")
                    
                    # Cập nhật kết nối với symbols mới
                    await connect_func(conn_info['websocket'], new_symbols, channels)
                    return
        
        # Nếu không tìm thấy kết nối phù hợp hoặc đã đủ symbols, tạo kết nối mới
        if len(self.connections) >= self.max_connections:
            self.logger.warning(f"Đã đạt giới hạn số lượng kết nối websocket ({self.max_connections}). Hãy xem xét tối ưu hóa.")
        
        connection_key = f"conn_{self.connection_count}"
        self.connection_count += 1
        
        # Lưu thông tin callback xử lý
        self.processing_callbacks[connection_key] = message_callback
        
        # Tạo task cho kết nối mới
        connection_task = asyncio.create_task(
            self._maintain_connection(connection_key, ws_url, symbols, channels, connect_func)
        )
        
        # Lưu thông tin kết nối (task sẽ tự điền websocket sau)
        self.connections[connection_key] = {
            'task': connection_task,
            'symbols': symbols.copy(),
            'channels': channels.copy(),
            'ws_url': ws_url,
            'websocket': None,
            'reconnect_attempt': 0
        }
        
        self.logger.info(f"Đã tạo kết nối websocket mới: {connection_key} với {len(symbols)} symbols")
    
    async def _maintain_connection(self, connection_key, ws_url, symbols, channels, connect_func):
        """
        Duy trì kết nối websocket với cơ chế reconnect tự động.
        
        Args:
            connection_key (str): Khóa kết nối
            ws_url (str): URL websocket
            symbols (List[str]): Danh sách symbols
            channels (List[str]): Danh sách kênh
            connect_func (Callable): Hàm kết nối websocket
        """
        conn_info = self.connections[connection_key]
        last_disconnect_time = 0
        
        while connection_key in self.connections:
            try:
                # Kết nối websocket
                self.logger.info(f"Đang kết nối websocket {connection_key} đến {ws_url}")
                websocket = await connect_func(None, symbols, channels)
                
                # Lưu websocket
                conn_info['websocket'] = websocket
                conn_info['reconnect_attempt'] = 0  # Reset số lần thử kết nối lại
                
                # Xử lý dữ liệu nhận được
                while True:
                    message = await websocket.recv()
                    if connection_key in self.processing_callbacks:
                        await self.processing_callbacks[connection_key](message)
            
            except asyncio.CancelledError:
                self.logger.info(f"Kết nối websocket {connection_key} bị hủy")
                break
                
            except Exception as e:
                current_time = time.time()
                conn_info['websocket'] = None
                
                # Tăng số lần thử kết nối lại
                conn_info['reconnect_attempt'] += 1
                
                # Tính toán thời gian chờ trước khi kết nối lại
                retry_index = min(conn_info['reconnect_attempt'] - 1, len(self.reconnect_delays) - 1)
                delay = self.reconnect_delays[retry_index]
                
                # Thêm jitter để tránh thundering herd
                jitter = random.uniform(0, 0.3 * delay)
                total_delay = delay + jitter
                
                self.logger.warning(
                    f"Kết nối websocket {connection_key} bị ngắt: {str(e)}. "
                    f"Thử kết nối lại sau {total_delay:.2f}s (lần thứ {conn_info['reconnect_attempt']})"
                )
                
                # Chờ trước khi kết nối lại
                await asyncio.sleep(total_delay)
    
    async def close_all(self):
        """Đóng tất cả các kết nối websocket."""
        for key, conn_info in list(self.connections.items()):
            try:
                # Hủy task
                if conn_info['task'] and not conn_info['task'].done():
                    conn_info['task'].cancel()
                
                # Đóng websocket
                if conn_info['websocket']:
                    await conn_info['websocket'].close()
                
                self.logger.info(f"Đã đóng kết nối websocket: {key}")
            except Exception as e:
                self.logger.error(f"Lỗi khi đóng kết nối websocket {key}: {e}")
        
        self.connections = {}
        self.processing_callbacks = {}


class GenericExchangeConnector(ABC):
    """
    Lớp trừu tượng cơ sở định nghĩa giao diện chung cho tất cả các kết nối sàn giao dịch.
    Lớp này cung cấp các phương thức chung và khung cơ bản cho các lớp con cụ thể.
    """

    def __init__(
        self, 
        exchange_id: str,
        api_key: str = None, 
        api_secret: str = None,
        api_passphrase: str = None,
        sandbox: bool = True,
        rate_limit: bool = True,
        config: SystemConfig = None,
        secret_manager: SecretManager = None,
        max_retry_attempts: int = 3
    ):
        """
        Khởi tạo connector sàn giao dịch chung.
        
        Args:
            exchange_id (str): ID của sàn giao dịch (vd: 'binance', 'bybit')
            api_key (str, optional): Khóa API. Mặc định lấy từ biến môi trường.
            api_secret (str, optional): Mật khẩu API. Mặc định lấy từ biến môi trường.
            api_passphrase (str, optional): Passphrase cho một số sàn. Mặc định là None.
            sandbox (bool, optional): Sử dụng môi trường sandbox/testnet. Mặc định là True.
            rate_limit (bool, optional): Kích hoạt giới hạn tốc độ gọi API. Mặc định là True.
            config (SystemConfig, optional): Cấu hình hệ thống. Mặc định là None.
            secret_manager (SecretManager, optional): Trình quản lý bí mật. Mặc định là None.
            max_retry_attempts (int, optional): Số lần thử lại tối đa cho API calls. Mặc định là 3.
        """
        # Tham số thêm cho khả năng phục hồi
        self.max_retry_attempts = max_retry_attempts
        self.exchange_id = exchange_id
        self.logger = setup_logger(f"{exchange_id}_connector")
        self.config = config if config else SystemConfig()
        self.secret_manager = secret_manager if secret_manager else SecretManager()
        
        # Lấy thông tin xác thực từ biến môi trường nếu không được cung cấp
        self.api_key = api_key or os.environ.get(f"{exchange_id.upper()}_API_KEY")
        self.api_secret = api_secret or os.environ.get(f"{exchange_id.upper()}_API_SECRET")
        self.api_passphrase = api_passphrase or os.environ.get(f"{exchange_id.upper()}_API_PASSPHRASE")
        
        # Xác minh thông tin xác thực
        if not all([self.api_key, self.api_secret]):
            self.logger.warning(f"API key hoặc secret không được cung cấp cho {exchange_id}. Một số chức năng có thể không khả dụng.")
        
        # Cấu hình nâng cao cho request timeout và tùy chọn mạng
        self.request_timeout = 30000  # 30 giây
        self.recv_window = 60000  # 60 giây
        
        # Cấu hình CCXT
        self.ccxt_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.api_passphrase,
            'enableRateLimit': rate_limit,
            'timeout': self.request_timeout,
            'options': {
                'defaultType': 'future',  # 'spot', 'future', 'margin'
                'recvWindow': self.recv_window,
                'warnOnFetchOpenOrdersWithoutSymbol': False,
                'fetchOrderBookLimit': 1000,  # Mặc định giới hạn sổ lệnh cao hơn
            },
            # Cải thiện xử lý HTTP khi mở rộng quy mô
            'headers': {
                'User-Agent': f'AutomatedTradingSystem/{exchange_id}'
            }
        }
        
        if sandbox:
            self.ccxt_config['sandbox'] = True
        
        # Khởi tạo WebSocketManager
        self.ws_manager = WebSocketManager(self.logger)
        
        # Khởi tạo bộ đệm dữ liệu
        self.data_buffer = {
            'tickers': {},
            'orderbooks': {},
            'trades': {},
            'klines': {}
        }
        
        # Cờ chạy cho các background tasks
        self.running = True
        self.background_tasks = []
        
        try:
            # Khởi tạo các đối tượng CCXT
            self.exchange = getattr(ccxt, exchange_id)(self.ccxt_config)
            self.async_exchange = getattr(ccxt_async, exchange_id)(self.ccxt_config)
            
            if sandbox:
                if hasattr(self.exchange, 'set_sandbox_mode'):
                    self.exchange.set_sandbox_mode(True)
                if hasattr(self.async_exchange, 'set_sandbox_mode'):
                    self.async_exchange.set_sandbox_mode(True)
                self.logger.info(f"Đã kích hoạt chế độ sandbox cho {exchange_id}")
            
            self.markets = {}
            self.logger.info(f"Đã khởi tạo connector {exchange_id}")
        except Exception as e:
            self.logger.error(f"Không thể khởi tạo connector {exchange_id}: {e}")
            raise
    
    def api_retry_decorator(self, retry_attempts: int = None):
        """
        Tạo decorator để tự động thử lại các cuộc gọi API với backoff theo cấp số nhân.
        
        Args:
            retry_attempts (int, optional): Số lần thử lại. Mặc định sử dụng max_retry_attempts.
            
        Returns:
            Callable: Decorator
        """
        attempts = retry_attempts or self.max_retry_attempts
        
        def decorator(func):
            @functools.wraps(func)
            @retry(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type((
                    ccxt.NetworkError, 
                    ccxt.ExchangeNotAvailable, 
                    ccxt.RequestTimeout,
                    ccxt.DDoSProtection
                )),
                reraise=True
            )
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
                    self.logger.warning(f"Tự động thử lại {func.__name__} do lỗi: {str(e)}")
                    raise
                except Exception as e:
                    self.logger.error(f"Lỗi {func.__name__}: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    async def initialize(self) -> None:
        """Khởi tạo connector, tải thị trường và thiết lập bất kỳ cài đặt cần thiết nào."""
        try:
            # Tải thông tin thị trường với retry
            load_markets = self.api_retry_decorator()(self.async_exchange.load_markets)
            self.markets = await load_markets(reload=True)
            self.logger.info(f"Đã tải {len(self.markets)} thị trường từ {self.exchange_id}")
            
            # Khởi động các background tasks
            self._start_background_tasks()
            
            # Kiểm tra trạng thái API
            await self.check_connection()
        except Exception as e:
            self.logger.error(f"Không thể khởi tạo {self.exchange_id} connector: {e}")
            raise
    
    def _start_background_tasks(self):
        """Khởi động các background tasks."""
        # Task làm sạch bộ đệm dữ liệu định kỳ
        cleanup_task = asyncio.create_task(self._cleanup_data_buffer())
        self.background_tasks.append(cleanup_task)
        
        # Task định kỳ kiểm tra kết nối
        heartbeat_task = asyncio.create_task(self._connection_heartbeat())
        self.background_tasks.append(heartbeat_task)
        
        self.logger.info(f"Đã khởi động {len(self.background_tasks)} background tasks")
    
    async def _cleanup_data_buffer(self):
        """Làm sạch bộ đệm dữ liệu định kỳ để tránh rò rỉ bộ nhớ."""
        cleanup_interval = 300  # 5 phút
        buffer_ttl = 3600  # 1 giờ
        
        while self.running:
            try:
                current_time = time.time()
                
                # Làm sạch dữ liệu cũ trong buffer
                for data_type in self.data_buffer:
                    for key in list(self.data_buffer[data_type].keys()):
                        entry = self.data_buffer[data_type][key]
                        if 'timestamp' in entry and (current_time - entry['timestamp'] > buffer_ttl):
                            del self.data_buffer[data_type][key]
                
                self.logger.debug(f"Đã làm sạch bộ đệm dữ liệu, kích thước hiện tại: ticker={len(self.data_buffer['tickers'])}, orderbook={len(self.data_buffer['orderbooks'])}")
            except Exception as e:
                self.logger.error(f"Lỗi khi làm sạch bộ đệm dữ liệu: {e}")
            
            await asyncio.sleep(cleanup_interval)
    
    async def _connection_heartbeat(self):
        """Kiểm tra kết nối định kỳ và thử kết nối lại nếu cần."""
        heartbeat_interval = 60  # 1 phút
        
        while self.running:
            try:
                # Kiểm tra kết nối
                is_connected = await self.check_connection()
                
                if not is_connected:
                    self.logger.warning(f"Kết nối với {self.exchange_id} không khả dụng, thử kết nối lại...")
                    try:
                        # Tạo lại đối tượng exchange nếu cần
                        await self.async_exchange.close()
                        self.async_exchange = getattr(ccxt.async_support, self.exchange_id)(self.ccxt_config)
                        
                        # Tải lại thị trường
                        load_markets = self.api_retry_decorator()(self.async_exchange.load_markets)
                        self.markets = await load_markets(reload=True)
                        
                        self.logger.info(f"Đã kết nối lại thành công với {self.exchange_id}")
                    except Exception as e:
                        self.logger.error(f"Không thể kết nối lại với {self.exchange_id}: {e}")
            except Exception as e:
                self.logger.error(f"Lỗi trong heartbeat: {e}")
            
            await asyncio.sleep(heartbeat_interval)
    
    async def check_connection(self) -> bool:
        """Kiểm tra kết nối với sàn giao dịch có hoạt động không."""
        try:
            # Cách đơn giản để kiểm tra kết nối là lấy thời gian máy chủ
            fetch_time = self.api_retry_decorator(1)(self.async_exchange.fetch_time)
            server_time = await fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            if time_diff > 10000:  # Chênh lệch > 10 giây
                self.logger.warning(f"Chênh lệch thời gian lớn với {self.exchange_id}: {time_diff}ms")
            
            self.logger.debug(f"Kết nối với {self.exchange_id} hoạt động tốt. Chênh lệch thời gian: {time_diff}ms")
            return True
        except Exception as e:
            self.logger.error(f"Kiểm tra kết nối với {self.exchange_id} thất bại: {e}")
            return False
    
    async def close(self) -> None:
        """Đóng tất cả các kết nối và dừng background tasks."""
        self.running = False
        
        # Dừng các background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Đóng tất cả kết nối websocket
        await self.ws_manager.close_all()
        
        # Đóng kết nối CCXT
        try:
            await self.async_exchange.close()
            self.logger.info(f"Đã đóng kết nối với {self.exchange_id}")
        except Exception as e:
            self.logger.error(f"Lỗi khi đóng kết nối với {self.exchange_id}: {e}")
    
    async def batch_fetch_tickers(self, symbols: List[str]) -> Dict:
        """
        Lấy ticker cho nhiều symbol cùng lúc, tối ưu cho trường hợp >100 symbols.
        
        Args:
            symbols (List[str]): Danh sách symbols
            
        Returns:
            Dict: Dữ liệu ticker
        """
        batch_size = 100  # Kích thước tối đa cho mỗi request
        results = {}
        
        # Chia nhỏ danh sách symbols thành các batch
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            
            try:
                fetch_tickers = self.api_retry_decorator()(self.async_exchange.fetch_tickers)
                batch_results = await fetch_tickers(batch_symbols)
                results.update(batch_results)
                
                # Tránh rate limit
                if i + batch_size < len(symbols):
                    await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Lỗi khi lấy batch tickers {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {e}")
        
        return results
    
    # === PHƯƠNG THỨC THÔNG TIN THỊ TRƯỜNG ===
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """
        Lấy thông tin ticker cho một cặp giao dịch.
        
        Args:
            symbol (str): Cặp giao dịch (vd: 'BTC/USDT')
            
        Returns:
            Dict: Thông tin ticker
        """
        # Kiểm tra xem có trong bộ đệm không
        cached_ticker = self.data_buffer['tickers'].get(symbol)
        if cached_ticker and time.time() - cached_ticker.get('timestamp', 0) < 5:  # Cache 5 giây
            return cached_ticker['data']
        
        try:
            fetch_ticker = self.api_retry_decorator()(self.async_exchange.fetch_ticker)
            ticker = await fetch_ticker(symbol)
            
            # Lưu vào bộ đệm
            self.data_buffer['tickers'][symbol] = {
                'data': ticker,
                'timestamp': time.time()
            }
            
            return ticker
        except Exception as e:
            self.logger.error(f"Không thể lấy ticker cho {symbol} từ {self.exchange_id}: {e}")
            raise
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Lấy dữ liệu sổ lệnh.
        
        Args:
            symbol (str): Cặp giao dịch
            limit (int, optional): Số lượng lệnh tối đa. Mặc định là 20.
            
        Returns:
            Dict: Dữ liệu sổ lệnh
        """
        # Kiểm tra xem có trong bộ đệm không
        cache_key = f"{symbol}_{limit}"
        cached_orderbook = self.data_buffer['orderbooks'].get(cache_key)
        if cached_orderbook and time.time() - cached_orderbook.get('timestamp', 0) < 2:  # Cache 2 giây
            return cached_orderbook['data']
        
        try:
            fetch_order_book = self.api_retry_decorator()(self.async_exchange.fetch_order_book)
            orderbook = await fetch_order_book(symbol, limit)
            
            # Lưu vào bộ đệm
            self.data_buffer['orderbooks'][cache_key] = {
                'data': orderbook,
                'timestamp': time.time()
            }
            
            return orderbook
        except Exception as e:
            self.logger.error(f"Không thể lấy sổ lệnh cho {symbol} từ {self.exchange_id}: {e}")
            raise

    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        since: Optional[int] = None, 
        limit: Optional[int] = 100
    ) -> List:
        """
        Lấy dữ liệu OHLCV (nến).
        
        Args:
            symbol (str): Cặp giao dịch
            timeframe (str, optional): Khung thời gian. Mặc định là '1h'.
            since (int, optional): Thời gian bắt đầu theo timestamp (ms). Mặc định là None.
            limit (int, optional): Số lượng nến tối đa. Mặc định là 100.
            
        Returns:
            List: Dữ liệu OHLCV
        """
        # Kiểm tra xem có trong bộ đệm không cho trường hợp không có since (dữ liệu hiện tại)
        if since is None:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            cached_candles = self.data_buffer['klines'].get(cache_key)
            if cached_candles and time.time() - cached_candles.get('timestamp', 0) < 60:  # Cache 60 giây
                return cached_candles['data']
        
        try:
            fetch_ohlcv = self.api_retry_decorator()(self.async_exchange.fetch_ohlcv)
            candles = await fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Lưu vào bộ đệm cho trường hợp không có since
            if since is None:
                cache_key = f"{symbol}_{timeframe}_{limit}"
                self.data_buffer['klines'][cache_key] = {
                    'data': candles,
                    'timestamp': time.time()
                }
            
            return candles
        except Exception as e:
            self.logger.error(f"Không thể lấy OHLCV cho {symbol} từ {self.exchange_id}: {e}")
            raise
    
    async def fetch_trades(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = 100) -> List:
        """
        Lấy danh sách giao dịch gần đây.
        
        Args:
            symbol (str): Cặp giao dịch
            since (int, optional): Thời gian bắt đầu theo timestamp (ms). Mặc định là None.
            limit (int, optional): Số lượng giao dịch tối đa. Mặc định là 100.
            
        Returns:
            List: Danh sách giao dịch
        """
        # Kiểm tra xem có trong bộ đệm không cho trường hợp không có since (dữ liệu hiện tại)
        if since is None:
            cache_key = f"{symbol}_{limit}"
            cached_trades = self.data_buffer['trades'].get(cache_key)
            if cached_trades and time.time() - cached_trades.get('timestamp', 0) < 5:  # Cache 5 giây
                return cached_trades['data']
        
        try:
            fetch_trades = self.api_retry_decorator()(self.async_exchange.fetch_trades)
            trades = await fetch_trades(symbol, since, limit)
            
            # Lưu vào bộ đệm cho trường hợp không có since
            if since is None:
                cache_key = f"{symbol}_{limit}"
                self.data_buffer['trades'][cache_key] = {
                    'data': trades,
                    'timestamp': time.time()
                }
            
            return trades
        except Exception as e:
            self.logger.error(f"Không thể lấy danh sách giao dịch cho {symbol} từ {self.exchange_id}: {e}")
            raise
    
    async def batch_fetch_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: Optional[int] = 100
    ) -> Dict[str, List]:
        """
        Lấy dữ liệu OHLCV cho nhiều cặp giao dịch đồng thời.
        
        Args:
            symbols (List[str]): Danh sách cặp giao dịch
            timeframe (str, optional): Khung thời gian. Mặc định là '1h'.
            since (int, optional): Thời gian bắt đầu theo timestamp (ms). Mặc định là None.
            limit (int, optional): Số lượng nến tối đa. Mặc định là 100.
            
        Returns:
            Dict[str, List]: Dữ liệu OHLCV theo cặp giao dịch
        """
        # Chia nhỏ danh sách symbol để tối ưu hiệu suất
        max_concurrent = 10  # Số lượng request đồng thời tối đa
        
        results = {}
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i+max_concurrent]
            tasks = []
            
            for symbol in batch:
                task = asyncio.create_task(self.fetch_ohlcv(symbol, timeframe, since, limit))
                tasks.append((symbol, task))
            
            for symbol, task in tasks:
                try:
                    results[symbol] = await task
                except Exception as e:
                    self.logger.error(f"Không thể lấy OHLCV cho {symbol}: {e}")
                    results[symbol] = []
            
            # Tránh rate limit với batch lớn
            if i + max_concurrent < len(symbols):
                await asyncio.sleep(0.5)
        
        return results
    
    # === PHƯƠNG THỨC GIAO DỊCH ===
    
    async def create_order(
        self, 
        symbol: str, 
        order_type: str, 
        side: str, 
        amount: float, 
        price: Optional[float] = None, 
        params: Dict = {}
    ) -> Dict:
        """
        Tạo một lệnh giao dịch mới.
        
        Args:
            symbol (str): Cặp giao dịch
            order_type (str): Loại lệnh ('limit', 'market')
            side (str): Phía giao dịch ('buy', 'sell')
            amount (float): Số lượng
            price (float, optional): Giá (cần thiết cho lệnh limit). Mặc định là None.
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            Dict: Thông tin lệnh đã tạo
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để tạo lệnh")
        
        try:
            # Xác nhận lại thị trường trước khi tạo lệnh
            if symbol not in self.markets:
                self.logger.warning(f"Symbol {symbol} không tìm thấy trong thị trường đã tải. Tải lại thị trường.")
                await self.async_exchange.load_markets(reload=True)
                
                if symbol not in self.markets:
                    raise ValueError(f"Symbol {symbol} không hợp lệ")
            
            # Kiểm tra kết nối trước khi tạo lệnh
            if not await self.check_connection():
                self.logger.warning("Kết nối không ổn định, thử kết nối lại trước khi tạo lệnh")
                await self.check_connection()
            
            # Tạo lệnh với retry
            create_order = self.api_retry_decorator()(self.async_exchange.create_order)
            order = await create_order(symbol, order_type, side, amount, price, params)
            
            self.logger.info(f"Đã tạo lệnh {order_type} {side} cho {symbol}: {order['id']}")
            return order
        except Exception as e:
            self.logger.error(f"Không thể tạo lệnh cho {symbol} trên {self.exchange_id}: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str, params: Dict = {}) -> Dict:
        """
        Hủy một lệnh giao dịch.
        
        Args:
            order_id (str): ID của lệnh
            symbol (str): Cặp giao dịch
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            Dict: Thông tin lệnh đã hủy
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để hủy lệnh")
        
        try:
            # Hủy lệnh với retry
            cancel_order = self.api_retry_decorator()(self.async_exchange.cancel_order)
            result = await cancel_order(order_id, symbol, params)
            
            self.logger.info(f"Đã hủy lệnh {order_id} cho {symbol}")
            return result
        except ccxt.OrderNotFound as e:
            self.logger.warning(f"Lệnh {order_id} không tìm thấy hoặc đã thực thi: {e}")
            # Kiểm tra lại trạng thái lệnh
            try:
                order_status = await self.fetch_order(order_id, symbol, params)
                return order_status
            except:
                raise e
        except Exception as e:
            self.logger.error(f"Không thể hủy lệnh {order_id} trên {self.exchange_id}: {e}")
            raise
    
    async def cancel_all_orders(self, symbol: Optional[str] = None, params: Dict = {}) -> List:
        """
        Hủy tất cả các lệnh đang mở.
        
        Args:
            symbol (str, optional): Cặp giao dịch cụ thể. Mặc định là None (tất cả).
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            List: Kết quả hủy lệnh
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để hủy lệnh")
        
        try:
            # Kiểm tra xem exchange có hỗ trợ hủy tất cả lệnh không
            if hasattr(self.async_exchange, 'cancel_all_orders'):
                cancel_all = self.api_retry_decorator()(self.async_exchange.cancel_all_orders)
                result = await cancel_all(symbol, params)
                self.logger.info(f"Đã hủy tất cả lệnh cho {symbol if symbol else 'tất cả symbols'}")
                return result
            else:
                # Thực hiện thủ công nếu không hỗ trợ
                open_orders = await self.fetch_open_orders(symbol)
                results = []
                
                for order in open_orders:
                    try:
                        result = await self.cancel_order(order['id'], order['symbol'], params)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Không thể hủy lệnh {order['id']}: {e}")
                
                return results
        except Exception as e:
            self.logger.error(f"Không thể hủy tất cả lệnh: {e}")
            raise
    
    async def fetch_order(self, order_id: str, symbol: str, params: Dict = {}) -> Dict:
        """
        Lấy thông tin về một lệnh cụ thể.
        
        Args:
            order_id (str): ID của lệnh
            symbol (str): Cặp giao dịch
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            Dict: Thông tin lệnh
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy thông tin lệnh")
        
        try:
            fetch_order = self.api_retry_decorator()(self.async_exchange.fetch_order)
            order = await fetch_order(order_id, symbol, params)
            return order
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin lệnh {order_id} trên {self.exchange_id}: {e}")
            raise
    
    async def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, limit: Optional[int] = None, params: Dict = {}) -> List:
        """
        Lấy danh sách các lệnh đang mở.
        
        Args:
            symbol (str, optional): Cặp giao dịch. Mặc định là None (tất cả các cặp).
            since (int, optional): Thời gian bắt đầu theo timestamp (ms). Mặc định là None.
            limit (int, optional): Số lượng lệnh tối đa. Mặc định là None.
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            List: Danh sách lệnh đang mở
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy danh sách lệnh đang mở")
        
        try:
            fetch_open_orders = self.api_retry_decorator()(self.async_exchange.fetch_open_orders)
            open_orders = await fetch_open_orders(symbol, since, limit, params)
            return open_orders
        except Exception as e:
            self.logger.error(f"Không thể lấy danh sách lệnh đang mở trên {self.exchange_id}: {e}")
            raise
            
    async def batch_create_orders(self, orders: List[Dict]) -> List[Dict]:
        """
        Tạo nhiều lệnh giao dịch cùng lúc.
        
        Args:
            orders (List[Dict]): Danh sách lệnh cần tạo, mỗi lệnh là một Dict với các khóa:
                                'symbol', 'type', 'side', 'amount', 'price', 'params'
            
        Returns:
            List[Dict]: Kết quả tạo lệnh
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để tạo lệnh")
        
        results = []
        
        # Kiểm tra xem exchange có hỗ trợ create_orders không
        if hasattr(self.async_exchange, 'create_orders'):
            try:
                create_orders = self.api_retry_decorator()(self.async_exchange.create_orders)
                results = await create_orders(orders)
                self.logger.info(f"Đã tạo {len(results)} lệnh hàng loạt")
                return results
            except Exception as e:
                self.logger.error(f"Không thể tạo lệnh hàng loạt: {e}")
                # Tiếp tục với phương pháp thủ công
        
        # Nếu không hỗ trợ, thực hiện từng lệnh một
        max_concurrent = 5  # Số lượng lệnh tạo đồng thời tối đa
        
        for i in range(0, len(orders), max_concurrent):
            batch = orders[i:i+max_concurrent]
            tasks = []
            
            for order in batch:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                amount = order.get('amount')
                price = order.get('price')
                params = order.get('params', {})
                
                task = asyncio.create_task(
                    self.create_order(symbol, order_type, side, amount, price, params)
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Lỗi khi tạo lệnh trong batch: {result}")
                    results.append({'success': False, 'error': str(result)})
                else:
                    results.append(result)
            
            # Tránh rate limit
            if i + max_concurrent < len(orders):
                await asyncio.sleep(1)
        
        return results
    
    # === PHƯƠNG THỨC TÀI KHOẢN ===
    
    async def fetch_balance(self, params: Dict = {}) -> Dict:
        """
        Lấy số dư tài khoản.
        
        Args:
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            Dict: Thông tin số dư
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy số dư tài khoản")
        
        try:
            balance = await self.async_exchange.fetch_balance(params)
            return balance
        except Exception as e:
            self.logger.error(f"Không thể lấy số dư tài khoản trên {self.exchange_id}: {e}")
            raise
    
    async def fetch_positions(self, symbols: List[str] = None, params: Dict = {}) -> List:
        """
        Lấy danh sách vị thế hiện tại (cho tài khoản margin/futures).
        
        Args:
            symbols (List[str], optional): Danh sách cặp giao dịch. Mặc định là None (tất cả).
            params (Dict, optional): Tham số bổ sung. Mặc định là {}.
            
        Returns:
            List: Danh sách vị thế
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy vị thế")
        
        try:
            # Kiểm tra xem sàn có hỗ trợ fetch_positions không
            if hasattr(self.async_exchange, 'fetch_positions'):
                positions = await self.async_exchange.fetch_positions(symbols, params)
                return positions
            else:
                self.logger.warning(f"{self.exchange_id} không hỗ trợ trực tiếp fetch_positions")
                # Thực hiện xử lý thay thế
                return []
        except Exception as e:
            self.logger.error(f"Không thể lấy vị thế trên {self.exchange_id}: {e}")
            raise
    
    # === PHƯƠNG THỨC TÙY CHỈNH ===
    
    @abstractmethod
    async def get_exchange_info(self) -> Dict:
        """
        Lấy thông tin chi tiết về sàn giao dịch.
        
        Returns:
            Dict: Thông tin sàn giao dịch
        """
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict:
        """
        Lấy thông tin phí giao dịch.
        
        Args:
            symbol (str, optional): Cặp giao dịch. Mặc định là None (tất cả các cặp).
            
        Returns:
            Dict: Thông tin phí giao dịch
        """
        pass
    
    @abstractmethod
    async def subscribe_to_websocket(self, symbol: str, channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu từ websocket.
        
        Args:
            symbol (str): Cặp giao dịch
            channels (List[str]): Các kênh cần đăng ký
        """
        pass
    
    # === PHƯƠNG THỨC TIỆN ÍCH ===
    
    def symbol_formatter(self, base_currency: str, quote_currency: str) -> str:
        """
        Định dạng cặp giao dịch theo định dạng của sàn.
        
        Args:
            base_currency (str): Tiền tệ cơ sở
            quote_currency (str): Tiền tệ báo giá
            
        Returns:
            str: Chuỗi định dạng cặp giao dịch
        """
        return f"{base_currency.upper()}/{quote_currency.upper()}"
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List:
        """
        Lấy dữ liệu lịch sử cho một khoảng thời gian.
        
        Args:
            symbol (str): Cặp giao dịch
            timeframe (str, optional): Khung thời gian. Mặc định là '1h'.
            start_time (datetime, optional): Thời gian bắt đầu. Mặc định là None.
            end_time (datetime, optional): Thời gian kết thúc. Mặc định là None.
            limit (int, optional): Số lượng nến tối đa mỗi lần gọi. Mặc định là 1000.
            
        Returns:
            List: Danh sách dữ liệu OHLCV
        """
        # Xác định thời gian bắt đầu, kết thúc nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày gần nhất
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)
        
        # Chuyển đổi datetime sang timestamp (ms)
        since = int(start_time.timestamp() * 1000)
        until = int(end_time.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        # Lặp cho đến khi lấy được tất cả dữ liệu
        while current_since < until:
            try:
                candles = await self.fetch_ohlcv(symbol, timeframe, current_since, limit)
                
                if not candles or len(candles) == 0:
                    break
                
                all_candles.extend(candles)
                
                # Cập nhật thời gian bắt đầu cho lần lấy tiếp theo
                last_candle_time = candles[-1][0]
                current_since = last_candle_time + 1
                
                # Tránh gọi API quá nhanh
                await asyncio.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lấy dữ liệu lịch sử cho {symbol}: {e}")
                break
        
        # Lọc dữ liệu trong khoảng thời gian yêu cầu
        filtered_candles = [candle for candle in all_candles if candle[0] >= since and candle[0] <= until]
        
        # Sắp xếp theo thời gian
        filtered_candles.sort(key=lambda x: x[0])
        
        return filtered_candles