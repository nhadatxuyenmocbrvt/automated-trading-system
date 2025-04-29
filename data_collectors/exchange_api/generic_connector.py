"""
Lớp kết nối sàn giao dịch chung.
File này định nghĩa một lớp trừu tượng làm cơ sở cho kết nối với mọi sàn giao dịch,
cung cấp các API chung và xử lý lỗi thống nhất.
"""

import time
import json
import hmac
import hashlib
import urllib.parse
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import ccxt  # Thư viện kết nối sàn giao dịch đa năng

from config.system_config import get_system_config
from config.logging_config import setup_logger
from config.constants import Timeframe, OrderType, TimeInForce, OrderStatus, ErrorCode
from config.env import get_env
from config.security_config import get_security_config
from config.utils.encryption import decrypt_api_credentials

class APIError(Exception):
    """Custom exception cho các lỗi liên quan đến API sàn giao dịch."""
    
    def __init__(self, error_code: ErrorCode, message: str, exchange: str = "", 
                 status_code: Optional[int] = None, response: Optional[Dict] = None):
        self.error_code = error_code
        self.exchange = exchange
        self.status_code = status_code
        self.response = response
        super().__init__(f"[{exchange}] {message}")


class ExchangeConnector(ABC):
    """
    Lớp cơ sở trừu tượng cho kết nối sàn giao dịch.
    Định nghĩa các phương thức chung mà mọi sàn giao dịch cần triển khai.
    """
    
    def __init__(self, exchange_id: str, api_key: str = '', api_secret: str = '', 
                 testnet: bool = False, use_proxy: bool = False):
        """
        Khởi tạo kết nối sàn giao dịch.
        
        Args:
            exchange_id: ID của sàn giao dịch (binance, bybit, ...)
            api_key: API key (tùy chọn, có thể lấy từ cấu hình)
            api_secret: API secret (tùy chọn, có thể lấy từ cấu hình)
            testnet: True để sử dụng testnet thay vì mainnet
            use_proxy: True để sử dụng proxy nếu đã cấu hình
        """
        self.exchange_id = exchange_id.lower()
        self.logger = setup_logger(f"{self.exchange_id}_connector")
        self.system_config = get_system_config()
        self.security_config = get_security_config()
        
        # Thiết lập các giá trị mặc định
        self.testnet = testnet
        self.use_proxy = use_proxy
        self.proxy = get_env('HTTP_PROXY', '') if use_proxy else ''
        self.timeout = get_env('REQUEST_TIMEOUT', 30000)  # milliseconds
        self.max_retries = get_env('MAX_RETRIES', 3)
        
        # Lấy API key từ tham số nếu cung cấp, nếu không thì lấy từ cấu hình
        if not api_key or not api_secret:
            api_key, api_secret = self._get_api_credentials()
        
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Khởi tạo kết nối ccxt
        self.exchange = self._init_ccxt()
        
        # Lưu trữ các định dạng chuẩn cho mỗi sàn
        self._timeframe_map = {}  # Ánh xạ timeframe giữa chuẩn chung và sàn cụ thể
        self._order_type_map = {}  # Ánh xạ order type giữa chuẩn chung và sàn cụ thể
        self._time_in_force_map = {}  # Ánh xạ time in force giữa chuẩn chung và sàn cụ thể
        
        # Cache cho dữ liệu thị trường để tránh gọi API quá nhiều
        self._market_cache = {}
        self._last_market_update = datetime.now() - timedelta(hours=1)  # Đảm bảo cập nhật lần đầu
        self._ticker_cache = {}
        self._last_ticker_update = {}
        
        # Khởi tạo ánh xạ các timeframe và order type
        self._init_mapping()
        
        self.logger.info(f"Đã khởi tạo kết nối {self.exchange_id}" + 
                         f" {'testnet' if self.testnet else 'mainnet'}")
    
    def _get_api_credentials(self) -> Tuple[str, str]:
        """
        Lấy API key và secret từ cấu hình.
        
        Returns:
            Tuple (api_key, api_secret)
        """
        # Lấy danh sách tất cả các key cho sàn hiện tại từ security_config
        exchange_keys = self.security_config.get_exchange_keys(self.exchange_id)
        
        if not exchange_keys:
            # Không tìm thấy key, thử lấy từ biến môi trường
            env_key = get_env(f"{self.exchange_id.upper()}_API_KEY", '')
            env_secret = get_env(f"{self.exchange_id.upper()}_API_SECRET", '')
            
            if env_key and env_secret:
                self.logger.info(f"Sử dụng API credentials từ biến môi trường cho {self.exchange_id}")
                return env_key, env_secret
            else:
                self.logger.warning(f"Không tìm thấy API credentials cho {self.exchange_id}")
                return '', ''
        
        # Lấy key đầu tiên từ danh sách (thường là active key)
        key_info = exchange_keys[0]
        
        self.logger.info(f"Sử dụng API key '{key_info['id']}' cho {self.exchange_id}")
        return key_info['api_key'], key_info['api_secret']
    
    @abstractmethod
    def _init_ccxt(self) -> ccxt.Exchange:
        """
        Khởi tạo đối tượng ccxt Exchange.
        Cần được triển khai bởi từng lớp con với các tùy chọn cụ thể.
        
        Returns:
            Đối tượng ccxt Exchange đã được cấu hình
        """
        pass
    
    @abstractmethod
    def _init_mapping(self) -> None:
        """
        Khởi tạo ánh xạ giữa các định dạng chuẩn và định dạng sàn cụ thể.
        Bao gồm timeframe, order type, và time in force.
        """
        pass
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """
        Chuyển đổi từ timeframe chuẩn sang định dạng của sàn cụ thể.
        
        Args:
            timeframe: Timeframe chuẩn (1m, 5m, 1h, 1d, ...)
            
        Returns:
            Timeframe định dạng sàn
        """
        return self._timeframe_map.get(timeframe, timeframe)
    
    def _convert_order_type(self, order_type: str) -> str:
        """
        Chuyển đổi từ order type chuẩn sang định dạng của sàn cụ thể.
        
        Args:
            order_type: Order type chuẩn (market, limit, stop_loss, ...)
            
        Returns:
            Order type định dạng sàn
        """
        return self._order_type_map.get(order_type, order_type)
    
    def _convert_time_in_force(self, time_in_force: str) -> str:
        """
        Chuyển đổi từ time in force chuẩn sang định dạng của sàn cụ thể.
        
        Args:
            time_in_force: Time in force chuẩn (gtc, ioc, fok, ...)
            
        Returns:
            Time in force định dạng sàn
        """
        return self._time_in_force_map.get(time_in_force, time_in_force)
    
    def _handle_error(self, e: Exception, method_name: str) -> None:
        """
        Xử lý lỗi từ API sàn giao dịch.
        
        Args:
            e: Exception đã bắt được
            method_name: Tên của phương thức gặp lỗi
        
        Raises:
            APIError: Exception với thông tin chi tiết về lỗi
        """
        error_code = ErrorCode.API_ERROR
        status_code = None
        response = None
        
        if isinstance(e, ccxt.NetworkError):
            error_code = ErrorCode.CONNECTION_ERROR
            message = f"Lỗi kết nối trong {method_name}: {str(e)}"
        elif isinstance(e, ccxt.ExchangeError):
            message = f"Lỗi sàn giao dịch trong {method_name}: {str(e)}"
        elif isinstance(e, ccxt.AuthenticationError):
            error_code = ErrorCode.AUTHENTICATION_FAILED
            message = f"Lỗi xác thực trong {method_name}: {str(e)}"
        elif isinstance(e, ccxt.DDoSProtection) or isinstance(e, ccxt.RateLimitExceeded):
            error_code = ErrorCode.RATE_LIMIT_EXCEEDED
            message = f"Vượt quá giới hạn API trong {method_name}: {str(e)}"
        elif isinstance(e, ccxt.InvalidOrder):
            error_code = ErrorCode.INVALID_ORDER_PARAMS
            message = f"Lệnh không hợp lệ trong {method_name}: {str(e)}"
        else:
            message = f"Lỗi không xác định trong {method_name}: {str(e)}"
        
        # Cố gắng trích xuất thêm thông tin chi tiết
        if hasattr(e, 'status_code'):
            status_code = e.status_code
        if hasattr(e, 'response'):
            response = e.response
            
        # Log lỗi
        self.logger.error(message)
        
        # Tạo và ném exception
        raise APIError(
            error_code=error_code,
            message=message,
            exchange=self.exchange_id,
            status_code=status_code,
            response=response
        )
    
    def _retry_api_call(self, method: callable, *args, **kwargs) -> Any:
        """
        Gọi phương thức API với cơ chế retry.
        
        Args:
            method: Phương thức API cần gọi
            *args: Các tham số cho phương thức
            **kwargs: Các tham số từ khóa cho phương thức
            
        Returns:
            Kết quả từ phương thức API
            
        Raises:
            APIError: Nếu tất cả các lần retry đều thất bại
        """
        method_name = method.__name__ if hasattr(method, '__name__') else "unknown_method"
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                return method(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                
                # Kiểm tra nếu là lỗi timeout, tăng thêm thời gian chờ
                is_timeout = isinstance(e, ccxt.RequestTimeout) or "timeout" in str(e).lower()
                
                # Log chi tiết lỗi
                error_msg = str(e)
                if hasattr(e, '__traceback__'):
                    error_details = traceback.format_exception(type(e), e, e.__traceback__)
                    if len(error_details) > 0:
                        error_msg = f"{error_msg}\n{''.join(error_details[-3:])}"  # Chỉ lấy 3 dòng cuối của traceback
                
                if retry_count >= self.max_retries:
                    self.logger.error(f"Đã vượt quá số lần thử lại. Lỗi cuối cùng: {error_msg}")
                    self._handle_error(e, method_name)
                
                # Tăng thời gian chờ theo cấp số nhân, và thêm thời gian nếu là lỗi timeout
                wait_time_base = 0.5 * (2 ** retry_count)
                wait_time = wait_time_base * 2 if is_timeout else wait_time_base
                
                self.logger.warning(
                    f"Lỗi khi gọi {method_name}, thử lại sau {wait_time:.1f}s "
                    f"(lần thử {retry_count}/{self.max_retries}). Lỗi: {error_msg[:200]}..."
                )
                time.sleep(wait_time)
    
    def fetch_markets(self, force_update: bool = False) -> List[Dict]:
        """
        Lấy thông tin về các thị trường có sẵn.
        
        Args:
            force_update: True để bỏ qua cache và cập nhật mới
            
        Returns:
            Danh sách các thị trường
        """
        # Kiểm tra cache
        cache_expired = (datetime.now() - self._last_market_update).total_seconds() > 3600  # 1 giờ
        
        if force_update or cache_expired or not self._market_cache:
            try:
                # Kiểm tra xem thị trường đã được tải trong đối tượng exchange chưa
                if hasattr(self.exchange, 'markets') and self.exchange.markets and not force_update:
                    markets = list(self.exchange.markets.values())
                    self._market_cache = markets
                    self._last_market_update = datetime.now()
                    self.logger.info(f"Sử dụng thông tin {len(markets)} thị trường từ đối tượng exchange")
                else:
                    # Tìm cách tải thị trường an toàn hơn
                    try:
                        # Thử fetch_markets trước
                        markets_data = self._retry_api_call(self.exchange.fetch_markets)
                        self._market_cache = markets_data
                    except Exception as e:
                        # Nếu có lỗi, thử lấy từ thuộc tính markets của exchange
                        if hasattr(self.exchange, 'markets') and self.exchange.markets:
                            self._market_cache = list(self.exchange.markets.values())
                            self.logger.warning(f"Sử dụng thông tin thị trường sẵn có sau khi gặp lỗi: {str(e)}")
                        else:
                            # Nếu không có cách nào lấy được thị trường, khởi tạo cache trống
                            self._market_cache = []
                            self.logger.error(f"Không thể lấy thông tin thị trường: {str(e)}")
                        
                    self._last_market_update = datetime.now()
                    self.logger.info(f"Đã cập nhật thông tin {len(self._market_cache)} thị trường")
            except Exception as e:
                # Xử lý lỗi cuối cùng nếu mọi cách đều thất bại
                self.logger.error(f"Lỗi nghiêm trọng khi lấy thông tin thị trường: {str(e)}")
                if not self._market_cache:
                    self._market_cache = []  # Đảm bảo cache không bị None
        
        return self._market_cache
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Lấy thông tin ticker của một symbol.
        
        Args:
            symbol: Symbol cần lấy thông tin (ví dụ: 'BTC/USDT')
            
        Returns:
            Thông tin ticker
        """
        # Kiểm tra cache
        cache_expired = symbol not in self._ticker_cache or \
                        (datetime.now() - self._last_ticker_update.get(symbol, datetime(1970, 1, 1))).total_seconds() > 60  # 1 phút
        
        if cache_expired:
            try:
                self._ticker_cache[symbol] = self._retry_api_call(self.exchange.fetch_ticker, symbol)
                self._last_ticker_update[symbol] = datetime.now()
            except Exception as e:
                self._handle_error(e, f"fetch_ticker({symbol})")
        
        return self._ticker_cache[symbol]
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                   since: Optional[int] = None, limit: Optional[int] = None,
                   params: Dict = {}) -> List[List]:
        """
        Lấy dữ liệu OHLCV (giá mở, cao, thấp, đóng, khối lượng).
        
        Args:
            symbol: Symbol cần lấy dữ liệu (ví dụ: 'BTC/USDT')
            timeframe: Khung thời gian (ví dụ: '1m', '5m', '1h', '1d')
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng candle tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Dữ liệu OHLCV dưới dạng [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            # Chuyển đổi timeframe sang định dạng của sàn
            tf = self._convert_timeframe(timeframe)
            
            # Gọi API để lấy dữ liệu
            ohlcv = self._retry_api_call(
                self.exchange.fetch_ohlcv,
                symbol, tf, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(ohlcv)} candles {timeframe} cho {symbol}")
            return ohlcv
        except Exception as e:
            self._handle_error(e, f"fetch_ohlcv({symbol}, {timeframe})")
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict:
        """
        Lấy dữ liệu sổ lệnh (order book).
        
        Args:
            symbol: Symbol cần lấy dữ liệu (ví dụ: 'BTC/USDT')
            limit: Số lượng mức giá tối đa (tùy chọn)
            
        Returns:
            Dữ liệu order book
        """
        try:
            order_book = self._retry_api_call(
                self.exchange.fetch_order_book,
                symbol, limit
            )
            
            self.logger.info(f"Đã lấy order book cho {symbol} với {limit} mức giá")
            return order_book
        except Exception as e:
            self._handle_error(e, f"fetch_order_book({symbol})")
    
    def fetch_balance(self) -> Dict:
        """
        Lấy thông tin số dư tài khoản.
        
        Returns:
            Thông tin số dư
        """
        try:
            balance = self._retry_api_call(self.exchange.fetch_balance)
            self.logger.info(f"Đã lấy thông tin số dư tài khoản")
            return balance
        except Exception as e:
            self._handle_error(e, "fetch_balance")
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float,
                    price: Optional[float] = None, params: Dict = {}) -> Dict:
        """
        Tạo lệnh giao dịch.
        
        Args:
            symbol: Symbol giao dịch (ví dụ: 'BTC/USDT')
            order_type: Loại lệnh ('market', 'limit', ...)
            side: Phía giao dịch ('buy' hoặc 'sell')
            amount: Số lượng giao dịch
            price: Giá (bắt buộc đối với limit orders)
            params: Tham số bổ sung cho API
            
        Returns:
            Thông tin lệnh đã tạo
        """
        try:
            # Chuyển đổi order_type sang định dạng của sàn
            ot = self._convert_order_type(order_type)
            
            # Đảm bảo có price cho limit orders
            if order_type.lower() == 'limit' and price is None:
                raise ValueError("Price is required for limit orders")
            
            order = self._retry_api_call(
                self.exchange.create_order,
                symbol, ot, side, amount, price, params
            )
            
            self.logger.info(
                f"Đã tạo lệnh {order_type} {side} {amount} {symbol}" +
                (f" @ {price}" if price else "")
            )
            return order
        except Exception as e:
            self._handle_error(e, f"create_order({symbol}, {order_type}, {side})")
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None, params: Dict = {}) -> Dict:
        """
        Hủy lệnh giao dịch.
        
        Args:
            order_id: ID của lệnh cần hủy
            symbol: Symbol giao dịch (bắt buộc với một số sàn)
            params: Tham số bổ sung cho API
            
        Returns:
            Thông tin lệnh đã hủy
        """
        try:
            result = self._retry_api_call(
                self.exchange.cancel_order,
                order_id, symbol, params
            )
            
            self.logger.info(f"Đã hủy lệnh {order_id}" + (f" cho {symbol}" if symbol else ""))
            return result
        except Exception as e:
            self._handle_error(e, f"cancel_order({order_id})")
    
    def fetch_order(self, order_id: str, symbol: Optional[str] = None, params: Dict = {}) -> Dict:
        """
        Lấy thông tin lệnh giao dịch.
        
        Args:
            order_id: ID của lệnh
            symbol: Symbol giao dịch (bắt buộc với một số sàn)
            params: Tham số bổ sung cho API
            
        Returns:
            Thông tin lệnh
        """
        try:
            order = self._retry_api_call(
                self.exchange.fetch_order,
                order_id, symbol, params
            )
            
            return order
        except Exception as e:
            self._handle_error(e, f"fetch_order({order_id})")
    
    def fetch_orders(self, symbol: Optional[str] = None, since: Optional[int] = None,
                    limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy danh sách lệnh giao dịch.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng lệnh tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách lệnh
        """
        try:
            orders = self._retry_api_call(
                self.exchange.fetch_orders,
                symbol, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(orders)} lệnh" + (f" cho {symbol}" if symbol else ""))
            return orders
        except Exception as e:
            self._handle_error(e, "fetch_orders")
    
    def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[int] = None,
                         limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy danh sách lệnh đang mở.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng lệnh tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách lệnh đang mở
        """
        try:
            orders = self._retry_api_call(
                self.exchange.fetch_open_orders,
                symbol, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(orders)} lệnh đang mở" + (f" cho {symbol}" if symbol else ""))
            return orders
        except Exception as e:
            self._handle_error(e, "fetch_open_orders")
    
    def fetch_closed_orders(self, symbol: Optional[str] = None, since: Optional[int] = None,
                           limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy danh sách lệnh đã đóng.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng lệnh tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách lệnh đã đóng
        """
        try:
            orders = self._retry_api_call(
                self.exchange.fetch_closed_orders,
                symbol, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(orders)} lệnh đã đóng" + (f" cho {symbol}" if symbol else ""))
            return orders
        except Exception as e:
            self._handle_error(e, "fetch_closed_orders")
    
    def fetch_my_trades(self, symbol: Optional[str] = None, since: Optional[int] = None,
                       limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy danh sách giao dịch của tài khoản.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng giao dịch tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách giao dịch
        """
        try:
            trades = self._retry_api_call(
                self.exchange.fetch_my_trades,
                symbol, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(trades)} giao dịch" + (f" cho {symbol}" if symbol else ""))
            return trades
        except Exception as e:
            self._handle_error(e, "fetch_my_trades")
    
    def get_supported_timeframes(self) -> Dict[str, str]:
        """
        Lấy danh sách các timeframe được hỗ trợ.
        
        Returns:
            Dict ánh xạ từ timeframe chuẩn sang timeframe của sàn
        """
        return self._timeframe_map
    
    def get_market_precision(self, symbol: str) -> Dict:
        """
        Lấy thông tin về độ chính xác (số chữ số thập phân) của giá và số lượng.
        
        Args:
            symbol: Symbol cần lấy thông tin
            
        Returns:
            Dict chứa thông tin precision
        """
        markets = self.fetch_markets()
        
        for market in markets:
            if market['symbol'] == symbol:
                return market['precision']
        
        raise APIError(
            error_code=ErrorCode.DATA_NOT_FOUND,
            message=f"Không tìm thấy thông tin cho symbol {symbol}",
            exchange=self.exchange_id
        )
    
    def get_market_limits(self, symbol: str) -> Dict:
        """
        Lấy thông tin về giới hạn giá và số lượng.
        
        Args:
            symbol: Symbol cần lấy thông tin
            
        Returns:
            Dict chứa thông tin limits
        """
        markets = self.fetch_markets()
        
        for market in markets:
            if market['symbol'] == symbol:
                return market['limits']
        
        raise APIError(
            error_code=ErrorCode.DATA_NOT_FOUND,
            message=f"Không tìm thấy thông tin cho symbol {symbol}",
            exchange=self.exchange_id
        )
    
    def test_connection(self) -> bool:
        """
        Kiểm tra kết nối với sàn giao dịch.
        
        Returns:
            True nếu kết nối thành công, False nếu thất bại
        """
        try:
            # Sử dụng phương thức load_markets với timeout phù hợp
            self.exchange.load_markets()
            return True
        except Exception as e:
            self.logger.error(f"Kiểm tra kết nối thất bại: {str(e)}")
            return False
            
    async def initialize(self) -> bool:
        """
        Khởi tạo kết nối và tải thông tin cần thiết.
        Hàm này nên được gọi sau khi tạo đối tượng connector.
        
        Returns:
            True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            # Thử tải thị trường để kiểm tra kết nối
            _ = self.fetch_markets()
            self.logger.info(f"Đã khởi tạo thành công connector {self.exchange_id}")
            return True
        except Exception as e:
            self.logger.error(f"Khởi tạo connector {self.exchange_id} thất bại: {str(e)}")
            return False
    
    async def close(self):
        """
        Đóng kết nối với sàn giao dịch.
        """
        # Không cần implement chi tiết vì CCXT tự đóng kết nối
        self.logger.info(f"Đóng kết nối với {self.exchange_id}")
        
        # Xóa các tài nguyên nếu cần
        self._market_cache.clear()
        self._ticker_cache.clear()
        self._last_ticker_update.clear()