"""
Kết nối API Binance.
File này triển khai lớp kết nối với sàn giao dịch Binance,
hỗ trợ cả Spot và Futures markets.
"""

import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import ccxt

from data_collectors.exchange_api.generic_connector import ExchangeConnector, APIError
from config.constants import OrderType, TimeInForce, ErrorCode
from config.env import get_env

class BinanceConnector(ExchangeConnector):
    """
    Lớp kết nối với sàn giao dịch Binance.
    Hỗ trợ Binance Spot và Binance Futures (USDM & COINM).
    """
    
    def __init__(self, api_key: str = '', api_secret: str = '', 
                 is_futures: bool = False, testnet: bool = False, 
                 use_proxy: bool = False):
        """
        Khởi tạo kết nối Binance.
        
        Args:
            api_key: API key Binance
            api_secret: API secret Binance
            is_futures: True để sử dụng Binance Futures thay vì Spot
            testnet: True để sử dụng testnet
            use_proxy: True để sử dụng proxy nếu đã cấu hình
        """
        self.is_futures = is_futures
        
        # Xác định ID sàn giao dịch dựa vào loại thị trường
        exchange_id = "binance"
        if is_futures:
            exchange_id = "binanceusdm"  # ID cho thị trường USDⓈ-M Futures
        
        super().__init__(exchange_id, api_key, api_secret, testnet, use_proxy)
        
        # Cache cho funding rate và giới hạn giao dịch
        self._funding_rate_cache = {}
        self._trading_limits_cache = {}
        
        # Thêm biến để theo dõi lỗi tải market
        self._market_loading_error_logged = False
        
        self.logger.info(f"Đã khởi tạo kết nối Binance {'Futures' if is_futures else 'Spot'}")
    
    def _init_ccxt(self) -> ccxt.Exchange:
        """
        Khởi tạo đối tượng ccxt Exchange cho Binance.
        
        Returns:
            Đối tượng ccxt.binance đã được cấu hình
        """
        options = {}
        
        # Cấu hình cho Futures
        if self.is_futures:
            options['defaultType'] = 'future'
            
        # Thêm các tùy chọn giúp giảm số lượng yêu cầu đến API
        options['fetchCurrencies'] = False  # Tắt tính năng lấy danh sách tiền tệ
        
        # Đảm bảo timeout là đủ lớn
        timeout = int(get_env('REQUEST_TIMEOUT', '30')) * 1000  # Chuyển từ giây sang millisecond
        
        # Tạo đối tượng Binance với các tùy chọn
        params = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'timeout': timeout,
            'enableRateLimit': True,
        }
        exchange = ccxt.binanceusdm(params)  # futures
        # hoặc
        exchange = ccxt.binance(params)      # spot
        
        # Thêm proxy nếu có
        if self.use_proxy and self.proxy:
            params['proxies'] = {
                'http': self.proxy,
                'https': self.proxy
            }
        
        # Sử dụng testnet nếu yêu cầu
        if self.testnet:
            if self.is_futures:
                params['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
            else:
                params['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api/v3',
                        'private': 'https://testnet.binance.vision/api/v3',
                    }
                }
        
        # Khởi tạo đối tượng Binance phù hợp
        if self.is_futures:
            exchange = ccxt.binanceusdm(params)
        else:
            exchange = ccxt.binance(params)
        
        # Tải thông tin thị trường một cách tối giản
        # Sử dụng quá trình tùy chỉnh để tránh API sapiGetCapitalConfigGetall
        try:
            # Tăng giá trị recvWindow (thời gian nhận) cho các yêu cầu API
            exchange.options['recvWindow'] = 60000  # 60 giây
            
            # Tải thị trường với số lần thử lại và xử lý lỗi cải tiến
            max_retries = 5  # Tăng số lần thử lại
            wait_time_base = 3  # Thời gian chờ cơ bản (giây)
            
            for i in range(max_retries):
                try:
                    # Gọi phương thức fetch_markets trực tiếp thay vì load_markets để có thể kiểm soát tốt hơn
                    markets_data = exchange.fetch_markets()
                    # Lưu trữ thông tin thị trường
                    exchange.markets = {}
                    exchange.markets_by_id = {}
                    
                    # Xử lý dữ liệu thị trường
                    for market in markets_data:
                        exchange.markets[market['symbol']] = market
                        # Sử dụng setdefault để tránh KeyError
                        if market['id'] not in exchange.markets_by_id:
                            exchange.markets_by_id[market['id']] = []
                        exchange.markets_by_id[market['id']].append(market)
                    
                    exchange.marketsLoaded = True
                    self.logger.info(f"Đã tải thành công {len(markets_data)} thị trường")
                    break
                    
                except ccxt.RequestTimeout as e:
                    wait_time = wait_time_base * (i + 1)  # Tăng thời gian chờ theo cấp số cộng
                    self.logger.warning(f"Timeout khi tải thị trường, thử lại ({i+1}/{max_retries}) sau {wait_time}s")
                    
                    if i == max_retries - 1:
                        self.logger.error(f"Đã vượt quá số lần thử lại ({max_retries}). Không thể tải thị trường.")
                        # Khởi tạo danh sách thị trường trống để tránh lỗi
                        exchange.markets = {}
                        exchange.markets_by_id = {}
                        exchange.marketsLoaded = True
                        # Không raise lỗi, tiếp tục với danh sách trống
                        self.logger.warning("Tiếp tục với danh sách thị trường trống. Dữ liệu thị trường sẽ được tải sau khi cần.")
                    
                    time.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải thị trường: {str(e)}")
                    if i == max_retries - 1:
                        # Khởi tạo danh sách thị trường trống để tránh lỗi
                        exchange.markets = {}
                        exchange.markets_by_id = {}
                        exchange.marketsLoaded = True
                        self.logger.warning("Tiếp tục với danh sách thị trường trống. Dữ liệu thị trường sẽ được tải sau khi cần.")
                    time.sleep(wait_time_base * (i + 1))
            
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo thị trường: {str(e)}")
            # Khởi tạo danh sách thị trường trống để tránh lỗi
            exchange.markets = {}
            exchange.markets_by_id = {}
            exchange.marketsLoaded = True
            self.logger.warning("Tiếp tục với danh sách thị trường trống. Dữ liệu thị trường sẽ được tải sau khi cần.")
        
        return exchange
    
    def _init_mapping(self) -> None:
        """
        Khởi tạo ánh xạ giữa các định dạng chuẩn và định dạng Binance.
        """
        # Ánh xạ timeframe
        self._timeframe_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M',
        }
        
        # Ánh xạ order type
        self._order_type_map = {
            OrderType.MARKET.value: 'market',
            OrderType.LIMIT.value: 'limit',
            OrderType.STOP_LOSS.value: 'stop_loss',
            OrderType.TAKE_PROFIT.value: 'take_profit',
            OrderType.STOP_LIMIT.value: 'stop_limit',
            OrderType.TRAILING_STOP.value: 'trailing_stop',
        }
        
        # Ánh xạ time in force
        self._time_in_force_map = {
            TimeInForce.GTC.value: 'GTC',  # Good Till Cancel
            TimeInForce.IOC.value: 'IOC',  # Immediate Or Cancel
            TimeInForce.FOK.value: 'FOK',  # Fill Or Kill
            TimeInForce.GTX.value: 'GTX',  # Good Till Crossing
        }
    
    def fetch_markets(self, force_update: bool = False) -> List[Dict]:
        """
        Lấy thông tin về các thị trường có sẵn.
        Ghi đè phương thức từ lớp cơ sở để xử lý lỗi tốt hơn.
        
        Args:
            force_update: True để bỏ qua cache và cập nhật mới
            
        Returns:
            Danh sách các thị trường
        """
        if force_update or not self._market_cache:
            try:
                # Sử dụng phương thức tải thị trường an toàn
                if hasattr(self.exchange, 'markets') and self.exchange.markets:
                    markets = list(self.exchange.markets.values())
                    
                    if markets and not force_update:
                        self._market_cache = markets
                        self.logger.info(f"Đã sử dụng thông tin {len(markets)} thị trường từ cache")
                        return self._market_cache
                
                # Tải lại thị trường
                self.logger.info("Đang tải lại thông tin thị trường...")
                max_retries = 3
                wait_time_base = 2
                
                for i in range(max_retries):
                    try:
                        # Tải thị trường với số lần thử lại
                        markets_data = self._retry_api_call(self.exchange.fetch_markets)
                        self._market_cache = markets_data
                        self.logger.info(f"Đã cập nhật thông tin {len(markets_data)} thị trường")
                        break
                    except Exception as e:
                        if i == max_retries - 1:
                            raise
                        wait_time = wait_time_base * (2 ** i)
                        self.logger.warning(f"Lỗi khi tải thị trường ({i+1}/{max_retries}), thử lại sau {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi tải thị trường: {str(e)}")
                # Trả về list trống thay vì raise lỗi để tránh làm crash chương trình
                if not self._market_cache:
                    self._market_cache = []
        
        return self._market_cache
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Lấy thông tin tỷ lệ funding (chỉ dành cho Futures).
        
        Args:
            symbol: Symbol cần lấy thông tin funding rate
            
        Returns:
            Thông tin funding rate
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Funding rate chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        try:
            funding_rate = self._retry_api_call(
                self.exchange.fetch_funding_rate,
                symbol
            )
            
            self._funding_rate_cache[symbol] = funding_rate
            return funding_rate
        except Exception as e:
            self._handle_error(e, f"fetch_funding_rate({symbol})")
    
    def fetch_funding_history(self, symbol: str, since: Optional[int] = None,
                             limit: Optional[int] = None) -> List[Dict]:
        """
        Lấy lịch sử funding payment (chỉ dành cho Futures).
        
        Args:
            symbol: Symbol cần lấy lịch sử funding
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            
        Returns:
            Lịch sử funding payment
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Funding history chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        try:
            funding_history = self._retry_api_call(
                self.exchange.fetch_funding_history,
                symbol, since, limit
            )
            
            self.logger.info(f"Đã lấy {len(funding_history)} funding payments cho {symbol}")
            return funding_history
        except Exception as e:
            self._handle_error(e, f"fetch_funding_history({symbol})")
    
    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Lấy thông tin vị thế mở (chỉ dành cho Futures).
        
        Args:
            symbols: Danh sách symbols cần lấy vị thế
            
        Returns:
            Danh sách các vị thế đang mở
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Positions chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        try:
            positions = self._retry_api_call(
                self.exchange.fetch_positions,
                symbols
            )
            
            # Lọc các vị thế có số lượng khác 0
            active_positions = [p for p in positions if float(p['contracts']) > 0]
            
            self.logger.info(f"Đã lấy {len(active_positions)} vị thế đang mở")
            return active_positions
        except Exception as e:
            self._handle_error(e, "fetch_positions")
    
    def fetch_position(self, symbol: str) -> Dict:
        """
        Lấy thông tin vị thế cho một symbol cụ thể (chỉ dành cho Futures).
        
        Args:
            symbol: Symbol cần lấy thông tin vị thế
            
        Returns:
            Thông tin vị thế
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Positions chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        try:
            position = self._retry_api_call(
                self.exchange.fetch_position,
                symbol
            )
            
            return position
        except Exception as e:
            self._handle_error(e, f"fetch_position({symbol})")
    
    def set_leverage(self, leverage: int, symbol: str) -> Dict:
        """
        Thiết lập đòn bẩy cho một symbol (chỉ dành cho Futures).
        
        Args:
            leverage: Giá trị đòn bẩy (1-125)
            symbol: Symbol cần thiết lập đòn bẩy
            
        Returns:
            Kết quả từ API
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Leverage chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        try:
            result = self._retry_api_call(
                self.exchange.set_leverage,
                leverage, symbol
            )
            
            self.logger.info(f"Đã thiết lập đòn bẩy {leverage}x cho {symbol}")
            return result
        except Exception as e:
            self._handle_error(e, f"set_leverage({leverage}, {symbol})")
    
    def set_margin_mode(self, margin_mode: str, symbol: str) -> Dict:
        """
        Thiết lập chế độ margin (ISOLATED hoặc CROSSED) cho một symbol (chỉ dành cho Futures).
        
        Args:
            margin_mode: 'ISOLATED' hoặc 'CROSSED'
            symbol: Symbol cần thiết lập chế độ margin
            
        Returns:
            Kết quả từ API
            
        Raises:
            APIError: Nếu không phải thị trường futures
        """
        if not self.is_futures:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Margin mode chỉ có sẵn cho thị trường futures",
                exchange=self.exchange_id
            )
        
        # Chuyển đổi margin_mode sang định dạng ccxt
        ccxt_margin_mode = margin_mode.lower()
        
        try:
            result = self._retry_api_call(
                self.exchange.set_margin_mode,
                ccxt_margin_mode, symbol
            )
            
            self.logger.info(f"Đã thiết lập chế độ margin {margin_mode} cho {symbol}")
            return result
        except Exception as e:
            # Binance sẽ trả về lỗi nếu margin mode đã được thiết lập
            # Chúng ta sẽ bỏ qua lỗi này
            if "already" in str(e).lower():
                self.logger.info(f"Chế độ margin {margin_mode} đã được thiết lập cho {symbol}")
                return {"info": f"Margin mode {margin_mode} already set for {symbol}"}
            
            self._handle_error(e, f"set_margin_mode({margin_mode}, {symbol})")
    
    def fetch_deposit_address(self, code: str, params: Dict = {}) -> Dict:
        """
        Lấy địa chỉ nạp tiền cho một đồng coin.
        
        Args:
            code: Mã coin (ví dụ: 'BTC', 'ETH', 'USDT')
            params: Tham số bổ sung cho API
            
        Returns:
            Thông tin địa chỉ nạp tiền
        """
        try:
            address = self._retry_api_call(
                self.exchange.fetch_deposit_address,
                code, params
            )
            
            self.logger.info(f"Đã lấy địa chỉ nạp tiền cho {code}")
            return address
        except Exception as e:
            self._handle_error(e, f"fetch_deposit_address({code})")
    
    def fetch_deposits(self, code: Optional[str] = None, since: Optional[int] = None,
                      limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy lịch sử nạp tiền.
        
        Args:
            code: Mã coin (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng kết quả tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Lịch sử nạp tiền
        """
        try:
            deposits = self._retry_api_call(
                self.exchange.fetch_deposits,
                code, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(deposits)} lịch sử nạp tiền" + 
                             (f" cho {code}" if code else ""))
            return deposits
        except Exception as e:
            self._handle_error(e, "fetch_deposits")
    
    def fetch_withdrawals(self, code: Optional[str] = None, since: Optional[int] = None,
                         limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy lịch sử rút tiền.
        
        Args:
            code: Mã coin (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch (tùy chọn)
            limit: Số lượng kết quả tối đa (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Lịch sử rút tiền
        """
        try:
            withdrawals = self._retry_api_call(
                self.exchange.fetch_withdrawals,
                code, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(withdrawals)} lịch sử rút tiền" + 
                             (f" cho {code}" if code else ""))
            return withdrawals
        except Exception as e:
            self._handle_error(e, "fetch_withdrawals")
    
    def fetch_trading_fees(self) -> Dict:
        """
        Lấy thông tin phí giao dịch.
        
        Returns:
            Thông tin phí giao dịch
        """
        try:
            fees = self._retry_api_call(self.exchange.fetch_trading_fees)
            self.logger.info(f"Đã lấy thông tin phí giao dịch")
            return fees
        except Exception as e:
            self._handle_error(e, "fetch_trading_fees")
    
    def fetch_klines(self, symbol: str, interval: str, start_time: Optional[int] = None,
                   end_time: Optional[int] = None, limit: int = 500) -> List[List]:
        """
        Phương thức đặc biệt cho Binance để lấy dữ liệu K-lines (candlestick).
        Hỗ trợ xác định thời gian bắt đầu và kết thúc chính xác.
        
        Args:
            symbol: Symbol cần lấy dữ liệu
            interval: Khoảng thời gian (1m, 5m, 1h, 1d, ...)
            start_time: Thời gian bắt đầu tính từ millisecond epoch
            end_time: Thời gian kết thúc tính từ millisecond epoch
            limit: Số lượng candles tối đa (max 1000)
            
        Returns:
            Dữ liệu candlestick
        """
        params = {}
        
        if start_time is not None:
            params['startTime'] = start_time
        
        if end_time is not None:
            params['endTime'] = end_time
        
        params['limit'] = min(limit, 1000)  # Binance giới hạn tối đa 1000 candles
        
        try:
            # Sử dụng phương thức API riêng của Binance
            klines = self._retry_api_call(
                self.exchange.fetch_ohlcv,
                symbol, interval, None, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(klines)} klines {interval} cho {symbol}")
            return klines
        except Exception as e:
            self._handle_error(e, f"fetch_klines({symbol}, {interval})")
    
    def fetch_historical_klines(self, symbol: str, interval: str, 
                               start_time: int, end_time: Optional[int] = None, 
                               limit: int = 1000) -> List[List]:
        """
        Lấy lượng lớn dữ liệu lịch sử K-lines bằng cách gọi nhiều lần API.
        
        Args:
            symbol: Symbol cần lấy dữ liệu
            interval: Khoảng thời gian (1m, 5m, 1h, 1d, ...)
            start_time: Thời gian bắt đầu tính từ millisecond epoch
            end_time: Thời gian kết thúc tính từ millisecond epoch (mặc định là hiện tại)
            limit: Số lượng candles cho mỗi request
            
        Returns:
            Dữ liệu candlestick
        """
        if end_time is None:
            end_time = int(time.time() * 1000)  # Hiện tại
            
        # Binance giới hạn tối đa 1000 candles mỗi request
        limit = min(limit, 1000)
        
        # Danh sách kết quả
        all_klines = []
        current_start = start_time
        
        self.logger.info(f"Bắt đầu lấy dữ liệu lịch sử cho {symbol} từ {datetime.fromtimestamp(start_time/1000)}")
        
        while current_start < end_time:
            # Lấy dữ liệu
            klines = self.fetch_klines(
                symbol, interval, current_start, end_time, limit
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Cập nhật thời gian bắt đầu cho request tiếp theo
            # Thời gian của candle cuối cùng + 1ms
            current_start = klines[-1][0] + 1
            
            # Thêm delay để tránh rate limit
            time.sleep(0.2)
        
        self.logger.info(f"Đã lấy tổng cộng {len(all_klines)} klines cho {symbol}")
        return all_klines