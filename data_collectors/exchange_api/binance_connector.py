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
            
        # Tùy chọn giúp tăng tính ổn định khi kết nối
        options['fetchCurrencies'] = False  # Tắt tính năng lấy danh sách tiền tệ
        options['recvWindow'] = 60000  # Tăng thời gian chờ cho API private
        options['adjustForTimeDifference'] = True  # Tự động điều chỉnh timestamp
        options['verbose'] = False  # Tắt chế độ verbose để giảm log
        
        # Tạo đối tượng Binance với các tùy chọn
        params = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'timeout': self.timeout,  # Sử dụng timeout từ lớp cha
            'enableRateLimit': True,
            'options': options,
            # Giữ kết nối sống
            'keepAlive': True,
            # Tăng thời gian cho phép không hoạt động
            'session': {'timeout': 60},  # 60 giây
        }
        
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
        exchange = None
        if self.is_futures:
            exchange = ccxt.binanceusdm(params)
        else:
            exchange = ccxt.binance(params)
        
        # Khởi tạo thị trường an toàn để tránh lỗi
        try:
            # Tải thông tin thị trường với retry
            self._safe_load_markets(exchange)
        except Exception as e:
            self.logger.error(f"Lỗi khi load_markets: {str(e)}")
            # Khởi tạo danh sách thị trường trống
            exchange.markets = {}
            exchange.markets_by_id = {}
            exchange.marketsLoaded = True
            self.logger.warning("Tiếp tục với danh sách thị trường trống. Sẽ tải lại khi cần.")
        
        return exchange
    
    def _safe_load_markets(self, exchange: ccxt.Exchange) -> None:
        """
        Tải thị trường với cơ chế an toàn và retry.
        
        Args:
            exchange: Đối tượng exchange cần tải thị trường
        """
        max_retries = 3
        retry = 0
        wait_time_base = 2
        
        while retry < max_retries:
            try:
                # Tùy chỉnh tải thị trường để tránh các API không cần thiết
                markets_response = exchange.publicGetExchangeInfo()
                
                if not markets_response or 'symbols' not in markets_response:
                    raise ValueError("Invalid response from exchange info API")
                
                markets = []
                for market in markets_response['symbols']:
                    # Chỉ xử lý các thị trường đang giao dịch
                    if market['status'] == 'TRADING':
                        base_currency = market['baseAsset']
                        quote_currency = market['quoteAsset']
                        symbol = f"{base_currency}/{quote_currency}"
                        
                        precision = {
                            'amount': None,
                            'price': None
                        }
                        
                        # Tìm độ chính xác từ filters
                        for filter_item in market.get('filters', []):
                            if filter_item['filterType'] == 'LOT_SIZE':
                                step_size = filter_item.get('stepSize', '0.00000001')
                                precision['amount'] = self._calculate_precision(step_size)
                            
                            if filter_item['filterType'] == 'PRICE_FILTER':
                                tick_size = filter_item.get('tickSize', '0.00000001')
                                precision['price'] = self._calculate_precision(tick_size)
                        
                        markets.append({
                            'id': market['symbol'],
                            'symbol': symbol,
                            'base': base_currency,
                            'quote': quote_currency,
                            'active': True,
                            'precision': precision,
                            'limits': {
                                'amount': {
                                    'min': None,
                                    'max': None
                                },
                                'price': {
                                    'min': None,
                                    'max': None
                                }
                            },
                            'info': market,
                        })
                
                # Cập nhật thị trường vào exchange
                exchange.markets = {}
                exchange.markets_by_id = {}
                
                for market in markets:
                    exchange.markets[market['symbol']] = market
                    if market['id'] not in exchange.markets_by_id:
                        exchange.markets_by_id[market['id']] = []
                    exchange.markets_by_id[market['id']].append(market)
                
                exchange.marketsLoaded = True
                self.logger.info(f"Đã tải thành công {len(markets)} thị trường")
                break
                
            except Exception as e:
                retry += 1
                wait_time = wait_time_base * (2 ** retry)
                self.logger.warning(f"Lỗi khi tải thị trường, thử lại {retry}/{max_retries}: {str(e)}")
                
                if retry >= max_retries:
                    # Thử phương pháp thay thế khi không thể tải thị trường
                    self.logger.warning("Đang thử phương pháp thay thế để tải thị trường...")
                    try:
                        # Thử phương thức load_markets() của ccxt
                        exchange.load_markets()
                        self.logger.info(f"Đã tải {len(exchange.markets)} thị trường bằng phương pháp thay thế")
                        break
                    except Exception as fallback_error:
                        self.logger.error(f"Không thể tải thị trường: {str(fallback_error)}")
                        raise
                
                time.sleep(wait_time)
    
    def _calculate_precision(self, step_size_str: str) -> int:
        """
        Tính toán số chữ số thập phân từ step size.
        
        Args:
            step_size_str: String biểu diễn step size (ví dụ: '0.00100000')
            
        Returns:
            Số chữ số thập phân
        """
        try:
            step_size = float(step_size_str)
            if step_size == 0:
                return 0
                
            precision = 0
            step_size_str = str(step_size)
            
            if '.' in step_size_str:
                precision = len(step_size_str.split('.')[1].rstrip('0'))
            
            return precision
        except (ValueError, TypeError):
            return 8  # Giá trị mặc định
    
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
        # Nếu danh sách thị trường trống hoặc cần cập nhật
        if force_update or not self.exchange.markets:
            try:
                self._safe_load_markets(self.exchange)
                self._market_cache = list(self.exchange.markets.values())
                self._last_market_update = datetime.now()
                
                self.logger.info(f"Đã cập nhật thông tin {len(self._market_cache)} thị trường")
            except Exception as e:
                self.logger.error(f"Lỗi khi tải thị trường: {str(e)}")
                # Trả về cache hiện tại nếu có lỗi
                if self._market_cache:
                    return self._market_cache
                return []
        elif not self._market_cache:
            self._market_cache = list(self.exchange.markets.values())
        
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
            active_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
            
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