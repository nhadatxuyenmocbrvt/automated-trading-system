"""
Kết nối API ByBit.
File này triển khai lớp kết nối với sàn giao dịch ByBit,
hỗ trợ cả Spot và Derivatives markets.
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

class BybitConnector(ExchangeConnector):
    """
    Lớp kết nối với sàn giao dịch ByBit.
    Hỗ trợ ByBit Spot, Inverse Perpetual, USDT Perpetual và Options.
    """
    
    def __init__(self, api_key: str = '', api_secret: str = '', 
                 market_type: str = 'spot', testnet: bool = False, 
                 use_proxy: bool = False):
        """
        Khởi tạo kết nối ByBit.
        
        Args:
            api_key: API key ByBit
            api_secret: API secret ByBit
            market_type: Loại thị trường ('spot', 'linear', 'inverse', 'option')
            testnet: True để sử dụng testnet
            use_proxy: True để sử dụng proxy nếu đã cấu hình
        """
        self.market_type = market_type.lower()
        
        # Xác định ID sàn giao dịch dựa vào loại thị trường
        exchange_id = "bybit"
        
        super().__init__(exchange_id, api_key, api_secret, testnet, use_proxy)
        
        # Cache cho các thông tin
        self._tickers_cache = {}
        self._insurance_fund_cache = {}
        
        self.logger.info(f"Đã khởi tạo kết nối ByBit market type: {market_type}")
    
    def _init_ccxt(self) -> ccxt.Exchange:
        """
        Khởi tạo đối tượng ccxt Exchange cho ByBit.
        
        Returns:
            Đối tượng ccxt.bybit đã được cấu hình
        """
        # Thiết lập options dựa trên market_type
        options = {
            'defaultType': 'spot'  # Mặc định là spot
        }
        
        if self.market_type == 'linear':
            options['defaultType'] = 'swap'
            options['defaultSubType'] = 'linear'
        elif self.market_type == 'inverse':
            options['defaultType'] = 'swap'
            options['defaultSubType'] = 'inverse'
        elif self.market_type == 'option':
            options['defaultType'] = 'option'
        
        # Tạo đối tượng ByBit với các tùy chọn
        params = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'timeout': self.timeout,
            'enableRateLimit': True,
            'options': options,
        }
        
        # Thêm proxy nếu có
        if self.use_proxy and self.proxy:
            params['proxies'] = {
                'http': self.proxy,
                'https': self.proxy
            }
        
        # Sử dụng testnet nếu yêu cầu
        if self.testnet:
            params['urls'] = {
                'api': {
                    'public': 'https://api-testnet.bybit.com',
                    'private': 'https://api-testnet.bybit.com',
                }
            }
        
        # Khởi tạo đối tượng ByBit
        exchange = ccxt.bybit(params)
        
        # Tải thông tin thị trường
        exchange.load_markets()
        
        return exchange
    
    def _init_mapping(self) -> None:
        """
        Khởi tạo ánh xạ giữa các định dạng chuẩn và định dạng ByBit.
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
            '12h': '12h',
            '1d': '1d',
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
            OrderType.TRAILING_STOP.value: 'trailing_stop_market',
        }
        
        # Ánh xạ time in force
        self._time_in_force_map = {
            TimeInForce.GTC.value: 'GTC',  # Good Till Cancel
            TimeInForce.IOC.value: 'IOC',  # Immediate Or Cancel
            TimeInForce.FOK.value: 'FOK',  # Fill Or Kill
            TimeInForce.GTX.value: 'PostOnly',  # ByBit dùng PostOnly thay vì GTX
        }
    
    def is_derivatives(self) -> bool:
        """
        Kiểm tra xem đây có phải là kết nối thị trường derivatives (linear, inverse, option).
        
        Returns:
            True nếu là thị trường derivatives, False nếu là spot
        """
        return self.market_type in ['linear', 'inverse', 'option']
    
    def fetch_tickers(self, symbols: Optional[List[str]] = None, params: Dict = {}) -> Dict[str, Dict]:
        """
        Lấy thông tin ticker của nhiều symbol cùng lúc.
        
        Args:
            symbols: Danh sách các symbol cần lấy thông tin
            params: Tham số bổ sung cho API
            
        Returns:
            Dict với key là symbol và value là thông tin ticker
        """
        try:
            tickers = self._retry_api_call(
                self.exchange.fetch_tickers,
                symbols, params
            )
            
            # Cập nhật cache
            for symbol, ticker in tickers.items():
                self._tickers_cache[symbol] = ticker
            
            self.logger.info(f"Đã lấy thông tin tickers cho {len(tickers)} symbols")
            return tickers
        except Exception as e:
            self._handle_error(e, "fetch_tickers")
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Lấy thông tin tỷ lệ funding hiện tại (chỉ dành cho Perpetual).
        
        Args:
            symbol: Symbol cần lấy thông tin funding rate
            
        Returns:
            Thông tin funding rate
            
        Raises:
            APIError: Nếu không phải thị trường perpetual
        """
        if not self.is_derivatives() or self.market_type == 'option':
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Funding rate chỉ có sẵn cho thị trường perpetual (linear/inverse)",
                exchange=self.exchange_id
            )
        
        try:
            funding_rate = self._retry_api_call(
                self.exchange.fetch_funding_rate,
                symbol
            )
            
            return funding_rate
        except Exception as e:
            self._handle_error(e, f"fetch_funding_rate({symbol})")
    
    def fetch_funding_history(self, symbol: str, since: Optional[int] = None,
                             limit: Optional[int] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy lịch sử funding payment (chỉ dành cho Perpetual).
        
        Args:
            symbol: Symbol cần lấy lịch sử funding
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            params: Tham số bổ sung cho API
            
        Returns:
            Lịch sử funding payment
            
        Raises:
            APIError: Nếu không phải thị trường perpetual
        """
        if not self.is_derivatives() or self.market_type == 'option':
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Funding history chỉ có sẵn cho thị trường perpetual (linear/inverse)",
                exchange=self.exchange_id
            )
        
        try:
            funding_history = self._retry_api_call(
                self.exchange.fetch_funding_history,
                symbol, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(funding_history)} funding payments cho {symbol}")
            return funding_history
        except Exception as e:
            self._handle_error(e, f"fetch_funding_history({symbol})")
    
    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Dict = {}) -> List[Dict]:
        """
        Lấy thông tin vị thế (chỉ dành cho Derivatives).
        
        Args:
            symbols: Danh sách symbols cần lấy vị thế
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách các vị thế
            
        Raises:
            APIError: Nếu không phải thị trường derivatives
        """
        if not self.is_derivatives():
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Positions chỉ có sẵn cho thị trường derivatives",
                exchange=self.exchange_id
            )
        
        try:
            positions = self._retry_api_call(
                self.exchange.fetch_positions,
                symbols, params
            )
            
            # Lọc các vị thế có số lượng khác 0
            active_positions = [p for p in positions if float(p['contracts'] or 0) > 0]
            
            self.logger.info(f"Đã lấy {len(active_positions)} vị thế đang mở")
            return active_positions
        except Exception as e:
            self._handle_error(e, "fetch_positions")
    
    def set_leverage(self, leverage: int, symbol: str, params: Dict = {}) -> Dict:
        """
        Thiết lập đòn bẩy cho một symbol (chỉ dành cho Derivatives).
        
        Args:
            leverage: Giá trị đòn bẩy
            symbol: Symbol cần thiết lập đòn bẩy
            params: Tham số bổ sung cho API
            
        Returns:
            Kết quả từ API
            
        Raises:
            APIError: Nếu không phải thị trường derivatives
        """
        if not self.is_derivatives():
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Leverage chỉ có sẵn cho thị trường derivatives",
                exchange=self.exchange_id
            )
        
        try:
            result = self._retry_api_call(
                self.exchange.set_leverage,
                leverage, symbol, params
            )
            
            self.logger.info(f"Đã thiết lập đòn bẩy {leverage}x cho {symbol}")
            return result
        except Exception as e:
            self._handle_error(e, f"set_leverage({leverage}, {symbol})")
    
    def set_margin_mode(self, margin_mode: str, symbol: str, params: Dict = {}) -> Dict:
        """
        Thiết lập chế độ margin (isolated hoặc cross) cho một symbol (chỉ dành cho Derivatives).
        
        Args:
            margin_mode: 'isolated' hoặc 'cross'
            symbol: Symbol cần thiết lập chế độ margin
            params: Tham số bổ sung cho API
            
        Returns:
            Kết quả từ API
            
        Raises:
            APIError: Nếu không phải thị trường derivatives
        """
        if not self.is_derivatives():
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Margin mode chỉ có sẵn cho thị trường derivatives",
                exchange=self.exchange_id
            )
        
        # ByBit yêu cầu margin_mode được chuẩn hóa
        margin_mode = margin_mode.lower()
        if margin_mode not in ['isolated', 'cross']:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Margin mode phải là 'isolated' hoặc 'cross'",
                exchange=self.exchange_id
            )
        
        try:
            result = self._retry_api_call(
                self.exchange.set_margin_mode,
                margin_mode, symbol, params
            )
            
            self.logger.info(f"Đã thiết lập chế độ margin {margin_mode} cho {symbol}")
            return result
        except Exception as e:
            # Bybit có thể trả về lỗi nếu margin mode đã được thiết lập
            if "already" in str(e).lower():
                self.logger.info(f"Chế độ margin {margin_mode} đã được thiết lập cho {symbol}")
                return {"info": f"Margin mode {margin_mode} already set for {symbol}"}
            
            self._handle_error(e, f"set_margin_mode({margin_mode}, {symbol})")
    
    def fetch_insurance_fund(self, symbol: Optional[str] = None) -> Dict:
        """
        Lấy thông tin quỹ bảo hiểm (chỉ dành cho Derivatives).
        
        Args:
            symbol: Symbol cần lấy thông tin bảo hiểm (tùy chọn)
            
        Returns:
            Thông tin quỹ bảo hiểm
            
        Raises:
            APIError: Nếu không phải thị trường derivatives
        """
        if not self.is_derivatives():
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Insurance fund chỉ có sẵn cho thị trường derivatives",
                exchange=self.exchange_id
            )
        
        # ByBit không hỗ trợ trực tiếp qua ccxt, sử dụng API request tùy chỉnh
        # Xây dựng tham số
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        try:
            # Đối với ByBit, ta cần thực hiện gọi REST API trực tiếp
            # ở đây sử dụng phương thức chung đối với tất cả request không được hỗ trợ bởi ccxt
            response = self._execute_rest_request('GET', '/v2/public/insurance', params)
            
            # Cập nhật cache
            if symbol:
                self._insurance_fund_cache[symbol] = response
            else:
                self._insurance_fund_cache['all'] = response
            
            return response
        except Exception as e:
            self._handle_error(e, "fetch_insurance_fund")
    
    def _execute_rest_request(self, method: str, endpoint: str, params: Dict = {}) -> Dict:
        """
        Thực hiện request REST API tùy chỉnh cho các tính năng không được hỗ trợ bởi ccxt.
        
        Args:
            method: Phương thức HTTP (GET, POST, etc.)
            endpoint: Endpoint API
            params: Tham số request
            
        Returns:
            Kết quả từ API
        """
        # Xác định base URL dựa trên testnet
        base_url = 'https://api-testnet.bybit.com' if self.testnet else 'https://api.bybit.com'
        
        # Tạo timestamp và signature cho API bảo mật nếu cần
        is_private = not endpoint.startswith('/v2/public/')
        
        if is_private:
            timestamp = int(time.time() * 1000)
            params['api_key'] = self.api_key
            params['timestamp'] = timestamp
            
            # Tạo query string được sắp xếp theo bảng chữ cái
            query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
            
            # Tạo signature
            signature = hmac.new(
                self.api_secret.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            params['sign'] = signature
        
        # Gọi API thông qua ccxt
        url = base_url + endpoint
        
        if method == 'GET':
            response = self.exchange.fetch(url, params=params)
        else:  # POST, DELETE, etc.
            response = self.exchange.fetch(url, method=method, params=params)
        
        # Xử lý kết quả
        if 'ret_code' in response and response['ret_code'] != 0:
            raise APIError(
                error_code=ErrorCode.API_ERROR,
                message=f"ByBit API error: {response.get('ret_msg', 'Unknown error')}",
                exchange=self.exchange_id,
                response=response
            )
        
        return response.get('result', {})
    
    def fetch_wallet_balance(self) -> Dict:
        """
        Lấy số dư ví (khác với account balance thông thường).
        Phương thức này hữu ích cho các thị trường derivatives.
        
        Returns:
            Thông tin số dư ví
        """
        try:
            if self.is_derivatives():
                # Đối với derivatives, sử dụng fetch_balance với params tùy chỉnh
                params = {'type': self.market_type}
                balance = self._retry_api_call(
                    self.exchange.fetch_balance,
                    params
                )
                return balance
            else:
                # Đối với spot, sử dụng fetch_balance thông thường
                return self._retry_api_call(self.exchange.fetch_balance)
        except Exception as e:
            self._handle_error(e, "fetch_wallet_balance")
    
    def fetch_deposit_address(self, code: str, network: Optional[str] = None) -> Dict:
        """
        Lấy địa chỉ nạp tiền cho một đồng coin.
        
        Args:
            code: Mã coin (ví dụ: 'BTC', 'ETH', 'USDT')
            network: Mạng blockchain (tùy chọn, ví dụ: 'ERC20', 'TRC20')
            
        Returns:
            Thông tin địa chỉ nạp tiền
        """
        params = {}
        if network:
            params['network'] = network
        
        try:
            address = self._retry_api_call(
                self.exchange.fetch_deposit_address,
                code, params
            )
            
            self.logger.info(f"Đã lấy địa chỉ nạp tiền cho {code}" + 
                             (f" trên mạng {network}" if network else ""))
            return address
        except Exception as e:
            self._handle_error(e, f"fetch_deposit_address({code})")
    
    def fetch_klines(self, symbol: str, interval: str, since: Optional[int] = None,
                   limit: Optional[int] = None, params: Dict = {}) -> List[List]:
        """
        Lấy dữ liệu candlestick (OHLCV) từ ByBit với các tham số tùy chỉnh.
        
        Args:
            symbol: Symbol cần lấy dữ liệu
            interval: Khoảng thời gian (1m, 5m, 1h, 1d, ...)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng candles tối đa (max 200 cho ByBit)
            params: Tham số bổ sung cho API
            
        Returns:
            Dữ liệu OHLCV dưới dạng [[timestamp, open, high, low, close, volume], ...]
        """
        # Chuyển đổi interval sang định dạng ByBit nếu cần
        interval = self._convert_timeframe(interval)
        
        # ByBit giới hạn 200 candles mỗi request
        if limit is None or limit > 200:
            limit = 200
        
        # Gọi API để lấy dữ liệu
        try:
            ohlcv = self._retry_api_call(
                self.exchange.fetch_ohlcv,
                symbol, interval, since, limit, params
            )
            
            self.logger.info(f"Đã lấy {len(ohlcv)} candles {interval} cho {symbol}")
            return ohlcv
        except Exception as e:
            self._handle_error(e, f"fetch_klines({symbol}, {interval})")
    
    def fetch_historical_klines(self, symbol: str, interval: str, 
                               start_time: int, end_time: Optional[int] = None, 
                               limit: int = 200) -> List[List]:
        """
        Lấy lượng lớn dữ liệu lịch sử candlestick bằng cách gọi nhiều lần API.
        
        Args:
            symbol: Symbol cần lấy dữ liệu
            interval: Khoảng thời gian (1m, 5m, 1h, 1d, ...)
            start_time: Thời gian bắt đầu tính từ millisecond epoch
            end_time: Thời gian kết thúc tính từ millisecond epoch (mặc định là hiện tại)
            limit: Số lượng candles cho mỗi request (max 200)
            
        Returns:
            Dữ liệu OHLCV
        """
        if end_time is None:
            end_time = int(time.time() * 1000)  # Hiện tại
            
        # ByBit giới hạn tối đa 200 candles mỗi request
        limit = min(limit, 200)
        
        # Danh sách kết quả
        all_klines = []
        current_start = start_time
        
        self.logger.info(f"Bắt đầu lấy dữ liệu lịch sử cho {symbol} từ {datetime.fromtimestamp(start_time/1000)}")
        
        while current_start < end_time:
            # Lấy dữ liệu
            klines = self.fetch_klines(
                symbol, interval, current_start, limit
            )
            
            # Không có dữ liệu, thoát vòng lặp
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Cập nhật thời gian bắt đầu cho request tiếp theo
            # Thời gian của candle cuối cùng + 1ms
            current_start = klines[-1][0] + 1
            
            # Thêm delay để tránh rate limit
            time.sleep(0.5)
        
        self.logger.info(f"Đã lấy tổng cộng {len(all_klines)} klines cho {symbol}")
        return all_klines
    
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
    
    def create_conditional_order(self, symbol: str, order_type: str, side: str, 
                                amount: float, trigger_price: float, 
                                price: Optional[float] = None, params: Dict = {}) -> Dict:
        """
        Tạo lệnh có điều kiện (conditional order).
        
        Args:
            symbol: Symbol giao dịch
            order_type: Loại lệnh ('limit' hoặc 'market')
            side: Phía giao dịch ('buy' hoặc 'sell')
            amount: Số lượng giao dịch
            trigger_price: Giá kích hoạt
            price: Giá đặt lệnh (bắt buộc cho limit orders)
            params: Tham số bổ sung cho API
            
        Returns:
            Thông tin lệnh đã tạo
        """
        # ByBit có API riêng cho conditional orders
        # Chuẩn bị tham số cho API
        api_params = params.copy()
        api_params['stop_px'] = trigger_price
        
        if order_type.lower() == 'limit' and price is None:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Price is required for limit conditional orders",
                exchange=self.exchange_id
            )
        
        try:
            # Tạo stop order qua ccxt
            order = self._retry_api_call(
                self.exchange.create_order,
                symbol, order_type, side, amount, price, api_params
            )
            
            self.logger.info(
                f"Đã tạo lệnh có điều kiện {order_type} {side} {amount} {symbol}" +
                f" với giá kích hoạt {trigger_price}" +
                (f" và giá đặt lệnh {price}" if price else "")
            )
            return order
        except Exception as e:
            self._handle_error(e, "create_conditional_order")
    
    def cancel_all_orders(self, symbol: Optional[str] = None, params: Dict = {}) -> List[Dict]:
        """
        Hủy tất cả các lệnh đang mở.
        
        Args:
            symbol: Symbol cần hủy lệnh (tùy chọn)
            params: Tham số bổ sung cho API
            
        Returns:
            Danh sách các lệnh đã hủy
        """
        try:
            result = self._retry_api_call(
                self.exchange.cancel_all_orders,
                symbol, params
            )
            
            self.logger.info(f"Đã hủy tất cả các lệnh" + (f" cho {symbol}" if symbol else ""))
            return result
        except Exception as e:
            self._handle_error(e, "cancel_all_orders")