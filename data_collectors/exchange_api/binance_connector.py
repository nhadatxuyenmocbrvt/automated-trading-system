"""
Binance Exchange Connector - Triển khai kết nối cụ thể cho sàn giao dịch Binance.
"""

import os
import time
import json
import hmac
import hashlib
import logging
import asyncio
import websockets
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import urllib.parse

# Import connector chung
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collectors.exchange_api.generic_connector import GenericExchangeConnector
from config.logging_config import setup_logger
from config.security_config import SecretManager
from config.system_config import SystemConfig


class BinanceConnector(GenericExchangeConnector):
    """
    Connector cho sàn giao dịch Binance, triển khai các phương thức cụ thể
    theo API của Binance.
    """
    
    # Các URL endpoint của Binance
    SPOT_API_URL = 'https://api.binance.com'
    SPOT_API_TESTNET_URL = 'https://testnet.binance.vision'
    FUTURES_API_URL = 'https://fapi.binance.com'
    FUTURES_API_TESTNET_URL = 'https://testnet.binancefuture.com'
    
    # Các URL websocket
    SPOT_WS_URL = 'wss://stream.binance.com:9443/ws'
    SPOT_WS_TESTNET_URL = 'wss://testnet.binance.vision/ws'
    FUTURES_WS_URL = 'wss://fstream.binance.com/ws'
    FUTURES_WS_TESTNET_URL = 'wss://stream.binancefuture.com/ws'
    
    def __init__(
        self, 
        api_key: str = None, 
        api_secret: str = None,
        sandbox: bool = True,
        rate_limit: bool = True,
        futures: bool = True,
        config: SystemConfig = None,
        secret_manager: SecretManager = None
    ):
        """
        Khởi tạo connector Binance.
        
        Args:
            api_key (str, optional): Khóa API. Mặc định lấy từ biến môi trường.
            api_secret (str, optional): Mật khẩu API. Mặc định lấy từ biến môi trường.
            sandbox (bool, optional): Sử dụng testnet. Mặc định là True.
            rate_limit (bool, optional): Kích hoạt giới hạn tốc độ gọi API. Mặc định là True.
            futures (bool, optional): Sử dụng tài khoản futures. Mặc định là True.
            config (SystemConfig, optional): Cấu hình hệ thống. Mặc định là None.
            secret_manager (SecretManager, optional): Trình quản lý bí mật. Mặc định là None.
        """
        # Cấu hình bổ sung cho Binance
        self.futures = futures
        self.logger = setup_logger("binance_connector")
        
        # Gọi constructor của lớp cha
        super().__init__(
            exchange_id='binance',
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit,
            config=config,
            secret_manager=secret_manager
        )
        
        # Thiết lập các options đặc biệt cho Binance
        if futures:
            self.exchange.options['defaultType'] = 'future'
            self.async_exchange.options['defaultType'] = 'future'
            self.logger.info("Đã thiết lập chế độ futures cho Binance")
        else:
            self.exchange.options['defaultType'] = 'spot'
            self.async_exchange.options['defaultType'] = 'spot'
            self.logger.info("Đã thiết lập chế độ spot cho Binance")
        
        # Lưu các websocket connection
        self.ws_connections = {}
        
        # Lưu trữ thời gian cập nhật cuối cùng cho các thông tin
        self.last_exchange_info_update = 0
        self.exchange_info = {}
        self.trading_fees = {}
        
        self.logger.info("Đã khởi tạo Binance connector")
    
    async def initialize(self) -> None:
        """Khởi tạo connector Binance, tải thông tin thị trường và thiết lập cấu hình cần thiết."""
        await super().initialize()
        
        # Lấy thông tin bổ sung đặc biệt cho Binance
        await self.get_exchange_info()
        
        # Kiểm tra giới hạn API và trạng thái tài khoản nếu có khóa API
        if self.api_key and self.api_secret:
            try:
                # Lấy thông tin phí giao dịch
                await self.get_trading_fees()
                
                # Kiểm tra quyền tài khoản
                if self.futures:
                    account_info = await self.async_exchange.fapiPrivateGetAccount()
                else:
                    account_info = await self.async_exchange.privateGetAccount()
                
                self.logger.info(f"Đã xác thực tài khoản Binance thành công: {account_info.get('accountType', 'unknown')}")
            except Exception as e:
                self.logger.warning(f"Không thể lấy thông tin tài khoản Binance: {e}")
    
    async def get_exchange_info(self) -> Dict:
        """
        Lấy thông tin chi tiết về sàn giao dịch Binance.
        
        Returns:
            Dict: Thông tin sàn giao dịch
        """
        # Kiểm tra xem đã lấy thông tin gần đây chưa để tránh gọi API quá nhiều
        current_time = time.time()
        if self.exchange_info and (current_time - self.last_exchange_info_update < 3600):  # 1 giờ
            return self.exchange_info
        
        try:
            if self.futures:
                # Gọi API cho futures
                exchange_info = await self.async_exchange.fapiPublicGetExchangeInfo()
            else:
                # Gọi API cho spot
                exchange_info = await self.async_exchange.publicGetExchangeInfo()
            
            self.exchange_info = exchange_info
            self.last_exchange_info_update = current_time
            
            # Ghi log một số thông tin hữu ích
            symbols_count = len(exchange_info.get('symbols', []))
            self.logger.info(f"Đã lấy thông tin sàn Binance: {symbols_count} cặp giao dịch")
            
            return exchange_info
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin sàn Binance: {e}")
            raise
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict:
        """
        Lấy thông tin phí giao dịch của Binance.
        
        Args:
            symbol (str, optional): Cặp giao dịch cụ thể. Mặc định là None (tất cả).
            
        Returns:
            Dict: Thông tin phí giao dịch
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy thông tin phí giao dịch")
        
        try:
            if self.futures:
                # Cách lấy phí cho futures
                fees = await self.async_exchange.fapiPrivateGetCommissionRate()
                if symbol:
                    self.trading_fees[symbol] = fees
                else:
                    # Phí futures áp dụng cho tất cả các cặp
                    self.trading_fees = {'all': fees}
            else:
                # Phí giao dịch spot
                if symbol:
                    params = {'symbol': symbol}
                    fees = await self.async_exchange.privateGetTradeFee(params)
                else:
                    fees = await self.async_exchange.privateGetTradeFee()
                
                # Chuyển đổi định dạng
                result = {}
                for fee_info in fees.get('tradeFee', []):
                    result[fee_info['symbol']] = {
                        'maker': float(fee_info['maker']),
                        'taker': float(fee_info['taker'])
                    }
                self.trading_fees = result
            
            return self.trading_fees
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin phí giao dịch Binance: {e}")
            raise
    
    async def subscribe_to_websocket(self, symbol: str, channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu từ websocket Binance.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            channels (List[str]): Các kênh cần đăng ký (ví dụ: ['ticker', 'kline_1m', 'depth'])
        """
        # Chuẩn hóa symbol cho Binance (loại bỏ ký hiệu '/' và chuyển thành chữ thường)
        formatted_symbol = symbol.replace('/', '').lower()
        
        # Xác định websocket URL
        if self.futures:
            ws_url = self.FUTURES_WS_TESTNET_URL if self.exchange.sandbox else self.FUTURES_WS_URL
        else:
            ws_url = self.SPOT_WS_TESTNET_URL if self.exchange.sandbox else self.SPOT_WS_URL
        
        # Tạo kênh streams theo định dạng Binance
        streams = []
        for channel in channels:
            if channel == 'ticker':
                streams.append(f"{formatted_symbol}@ticker")
            elif channel.startswith('kline_'):
                interval = channel.split('_')[1]
                streams.append(f"{formatted_symbol}@kline_{interval}")
            elif channel == 'depth':
                streams.append(f"{formatted_symbol}@depth20")
            elif channel == 'trades':
                streams.append(f"{formatted_symbol}@trade")
            else:
                self.logger.warning(f"Kênh không được hỗ trợ: {channel}")
        
        if not streams:
            self.logger.error("Không có kênh hợp lệ để đăng ký")
            return
        
        # Sử dụng WebSocketManager để quản lý kết nối
        await self.ws_manager.add_connection(
            ws_url,
            [formatted_symbol],
            streams,
            self._connect_binance_websocket,
            self._process_websocket_message_raw
        )
    
    async def subscribe_multiple_symbols(self, symbols: List[str], channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu từ websocket Binance cho nhiều symbols cùng lúc.
        
        Args:
            symbols (List[str]): Danh sách cặp giao dịch (ví dụ: ['BTC/USDT', 'ETH/USDT'])
            channels (List[str]): Các kênh cần đăng ký (ví dụ: ['ticker', 'kline_1m'])
        """
        if not symbols:
            self.logger.error("Danh sách symbols trống")
            return
        
        # Phân nhóm symbols để tối ưu hóa kết nối
        # Binance cho phép gộp nhiều streams trong một kết nối websocket
        max_symbols_per_connection = self.ws_manager.symbols_per_connection
        
        for i in range(0, len(symbols), max_symbols_per_connection):
            batch_symbols = symbols[i:i+max_symbols_per_connection]
            
            # Chuẩn hóa symbols
            formatted_symbols = [symbol.replace('/', '').lower() for symbol in batch_symbols]
            
            # Xác định websocket URL
            if self.futures:
                ws_url = self.FUTURES_WS_TESTNET_URL if self.exchange.sandbox else self.FUTURES_WS_URL
            else:
                ws_url = self.SPOT_WS_TESTNET_URL if self.exchange.sandbox else self.SPOT_WS_URL
            
            # Sử dụng WebSocketManager để quản lý kết nối
            await self.ws_manager.add_connection(
                ws_url,
                formatted_symbols,
                channels,
                self._connect_binance_websocket,
                self._process_websocket_message_raw
            )
    
    async def _connect_binance_websocket(self, existing_ws, symbols: List[str], channels: List[str]):
        """
        Kết nối tới websocket Binance và đăng ký các streams.
        
        Args:
            existing_ws: Websocket hiện có (None nếu là kết nối mới)
            symbols (List[str]): Danh sách symbols đã định dạng
            channels (List[str]): Loại kênh dữ liệu
            
        Returns:
            websocket: Kết nối websocket đã thiết lập
        """
        # Tạo danh sách streams
        all_streams = []
        for symbol in symbols:
            for channel in channels:
                if channel == 'ticker':
                    all_streams.append(f"{symbol}@ticker")
                elif channel.startswith('kline_'):
                    interval = channel.split('_')[1]
                    all_streams.append(f"{symbol}@kline_{interval}")
                elif channel == 'depth':
                    all_streams.append(f"{symbol}@depth20")
                elif channel == 'trades':
                    all_streams.append(f"{symbol}@trade")
        
        # Xác định websocket URL
        if self.futures:
            ws_url = self.FUTURES_WS_TESTNET_URL if self.exchange.sandbox else self.FUTURES_WS_URL
        else:
            ws_url = self.SPOT_WS_TESTNET_URL if self.exchange.sandbox else self.SPOT_WS_URL
        
        # Tạo URL kết nối với streams
        combined_streams_url = f"{ws_url}/stream?streams={'/'.join(all_streams)}"
        
        # Nếu đã có kết nối, đóng và tạo lại
        if existing_ws:
            try:
                await existing_ws.close()
            except:
                pass
        
        # Thiết lập kết nối mới
        try:
            websocket = await websockets.connect(
                combined_streams_url,
                ping_interval=30,
                ping_timeout=10,
                max_size=2**24,  # 16MB max message size
                close_timeout=10
            )
            
            self.logger.info(f"Đã kết nối websocket Binance cho {len(symbols)} symbols với {len(channels)} kênh")
            return websocket
        except Exception as e:
            self.logger.error(f"Không thể kết nối websocket Binance: {e}")
            raise
    
    async def _process_websocket_message_raw(self, message: str) -> None:
        """
        Xử lý dữ liệu raw từ websocket.
        
        Args:
            message (str): Dữ liệu dạng JSON string từ websocket
        """
        try:
            data = json.loads(message)
            await self._process_websocket_message(data)
        except json.JSONDecodeError:
            self.logger.error(f"Không thể phân tích dữ liệu websocket: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Lỗi xử lý dữ liệu websocket: {e}")
            
    async def _process_websocket_batch_updates(self, batch_size=100, max_delay=1.0):
        """
        Xử lý hàng đợi cập nhật từ websocket theo batch để tối ưu hiệu suất.
        
        Args:
            batch_size (int): Số lượng cập nhật tối đa trong một batch
            max_delay (float): Thời gian chờ tối đa (giây)
        """
        if not hasattr(self, '_update_queue'):
            self._update_queue = asyncio.Queue()
            
        # Lấy nhiều cập nhật cùng lúc
        updates = []
        start_time = time.time()
        
        try:
            # Lấy phần tử đầu tiên (có thể chờ)
            first_update = await self._update_queue.get()
            updates.append(first_update)
            
            # Lấy thêm các phần tử khác (không chờ)
            while len(updates) < batch_size and time.time() - start_time < max_delay:
                try:
                    update = self._update_queue.get_nowait()
                    updates.append(update)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.01)  # Chờ một chút
        
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý batch updates: {e}")
            
        # Xử lý tất cả cập nhật trong batch
        for update in updates:
            try:
                # TODO: Xử lý cập nhật theo loại
                pass
            except Exception as e:
                self.logger.error(f"Lỗi xử lý cập nhật trong batch: {e}")
            finally:
                self._update_queue.task_done()
    
    async def _process_websocket_message(self, data: Dict) -> None:
        """
        Xử lý dữ liệu nhận được từ websocket.
        
        Args:
            data (Dict): Dữ liệu từ websocket
        """
        # Trích xuất thông tin từ dữ liệu
        if 'stream' in data and 'data' in data:
            stream = data['stream']
            stream_data = data['data']
            
            # Xác định loại dữ liệu
            if 'ticker' in stream:
                self.logger.debug(f"Nhận ticker: {stream_data['s']} - Giá: {stream_data['c']}")
                # Xử lý dữ liệu ticker
                # TODO: Thêm logic để xử lý dữ liệu ticker (lưu vào cơ sở dữ liệu, phân tích, v.v.)
                
            elif 'kline' in stream:
                kline = stream_data['k']
                self.logger.debug(f"Nhận kline: {stream_data['s']} - Interval: {kline['i']} - Đóng: {kline['c']}")
                # Xử lý dữ liệu nến
                # TODO: Thêm logic để xử lý dữ liệu kline
                
            elif 'depth' in stream:
                # Xử lý dữ liệu sổ lệnh
                self.logger.debug(f"Nhận depth: {stream_data['s']} - Bids: {len(stream_data['b'])} - Asks: {len(stream_data['a'])}")
                # TODO: Thêm logic để xử lý dữ liệu depth
                
            elif 'trade' in stream:
                # Xử lý dữ liệu giao dịch
                self.logger.debug(f"Nhận trade: {stream_data['s']} - Giá: {stream_data['p']} - Khối lượng: {stream_data['q']}")
                # TODO: Thêm logic để xử lý dữ liệu trade
                
            else:
                self.logger.debug(f"Nhận dữ liệu không xác định: {stream}")
        else:
            self.logger.warning(f"Định dạng dữ liệu không đúng: {data}")
    
    async def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Lấy tỷ lệ tài trợ hiện tại cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            
        Returns:
            Dict: Thông tin tỷ lệ tài trợ
        """
        if not self.futures:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            # Gọi API Binance để lấy tỷ lệ tài trợ
            params = {'symbol': formatted_symbol}
            funding_rate = await self.async_exchange.fapiPublicGetFundingRate(params)
            
            if isinstance(funding_rate, list) and len(funding_rate) > 0:
                return {
                    'symbol': symbol,
                    'lastFundingRate': float(funding_rate[0]['lastFundingRate']),
                    'nextFundingTime': funding_rate[0]['nextFundingTime'],
                    'timestamp': funding_rate[0]['time']
                }
            return {}
        except Exception as e:
            self.logger.error(f"Không thể lấy tỷ lệ tài trợ cho {symbol}: {e}")
            raise
    
    async def fetch_funding_history(self, symbol: str, limit: int = 100) -> List:
        """
        Lấy lịch sử tỷ lệ tài trợ cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            limit (int, optional): Số lượng bản ghi tối đa. Mặc định là 100.
            
        Returns:
            List: Lịch sử tỷ lệ tài trợ
        """
        if not self.futures:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy lịch sử tỷ lệ tài trợ")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            # Gọi API Binance để lấy lịch sử tỷ lệ tài trợ
            params = {
                'symbol': formatted_symbol,
                'limit': limit
            }
            funding_history = await self.async_exchange.fapiPrivateGetFundingRate(params)
            return funding_history
        except Exception as e:
            self.logger.error(f"Không thể lấy lịch sử tỷ lệ tài trợ cho {symbol}: {e}")
            raise
    
    async def fetch_leverage_brackets(self, symbol: Optional[str] = None) -> Dict:
        """
        Lấy thông tin về các mức đòn bẩy có sẵn.
        
        Args:
            symbol (str, optional): Cặp giao dịch cụ thể. Mặc định là None (tất cả).
            
        Returns:
            Dict: Thông tin mức đòn bẩy
        """
        if not self.futures:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy thông tin mức đòn bẩy")
        
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.replace('/', '')
            
            leverage_brackets = await self.async_exchange.fapiPrivateGetLeverageBracket(params)
            return leverage_brackets
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin mức đòn bẩy: {e}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Thiết lập đòn bẩy cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch
            leverage (int): Mức đòn bẩy (1-125)
            
        Returns:
            Dict: Kết quả thiết lập
        """
        if not self.futures:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để thiết lập đòn bẩy")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': formatted_symbol,
                'leverage': leverage
            }
            result = await self.async_exchange.fapiPrivatePostLeverage(params)
            self.logger.info(f"Đã thiết lập đòn bẩy {leverage}x cho {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Không thể thiết lập đòn bẩy cho {symbol}: {e}")
            raise
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Thiết lập loại margin cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch
            margin_type (str): Loại margin ('ISOLATED' hoặc 'CROSSED')
            
        Returns:
            Dict: Kết quả thiết lập
        """
        if not self.futures:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để thiết lập loại margin")
        
        if margin_type not in ['ISOLATED', 'CROSSED']:
            raise ValueError("margin_type phải là 'ISOLATED' hoặc 'CROSSED'")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': formatted_symbol,
                'marginType': margin_type
            }
            result = await self.async_exchange.fapiPrivatePostMarginType(params)
            self.logger.info(f"Đã thiết lập margin type {margin_type} cho {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Không thể thiết lập margin type cho {symbol}: {e}")
            raise
    
    async def close_all_connections(self) -> None:
        """Đóng tất cả các kết nối websocket."""
        for key, ws in list(self.ws_connections.items()):
            try:
                await ws.close()
                self.logger.info(f"Đã đóng kết nối websocket: {key}")
            except Exception as e:
                self.logger.error(f"Lỗi khi đóng kết nối websocket {key}: {e}")
        
        self.ws_connections = {}
        
        # Đóng kết nối CCXT
        await super().close()
    
    # === PHƯƠNG THỨC TIỆN ÍCH BINANCE ===
    
    def symbol_formatter(self, base_currency: str, quote_currency: str) -> str:
        """
        Định dạng cặp giao dịch theo cách Binance yêu cầu.
        
        Args:
            base_currency (str): Tiền tệ cơ sở
            quote_currency (str): Tiền tệ báo giá
            
        Returns:
            str: Chuỗi định dạng cặp giao dịch
        """
        # Binance sử dụng định dạng với dấu '/'
        return f"{base_currency.upper()}/{quote_currency.upper()}"
    
    async def get_all_symbols(self) -> List[str]:
        """
        Lấy tất cả các cặp giao dịch có sẵn.
        
        Returns:
            List[str]: Danh sách các cặp giao dịch
        """
        exchange_info = await self.get_exchange_info()
        symbols = [symbol['symbol'] for symbol in exchange_info.get('symbols', [])]
        return symbols
    
    async def get_server_time(self) -> int:
        """
        Lấy thời gian của máy chủ Binance.
        
        Returns:
            int: Thời gian máy chủ (timestamp)
        """
        try:
            if self.futures:
                time_info = await self.async_exchange.fapiPublicGetTime()
            else:
                time_info = await self.async_exchange.publicGetTime()
            
            return time_info['serverTime']
        except Exception as e:
            self.logger.error(f"Không thể lấy thời gian máy chủ Binance: {e}")
            raise
    
    async def get_latest_market_data(self, symbols: List[str] = None) -> Dict:
        """
        Lấy dữ liệu thị trường mới nhất cho nhiều cặp giao dịch.
        
        Args:
            symbols (List[str], optional): Danh sách cặp giao dịch. Mặc định là None (tất cả).
            
        Returns:
            Dict: Dữ liệu thị trường
        """
        try:
            # Nếu không có symbols, lấy tất cả
            if not symbols:
                all_tickers = await self.async_exchange.fetch_tickers()
                return all_tickers
            
            # Lấy ticker cho từng symbol
            result = {}
            for symbol in symbols:
                ticker = await self.fetch_ticker(symbol)
                result[symbol] = ticker
            
            return result
        except Exception as e:
            self.logger.error(f"Không thể lấy dữ liệu thị trường mới nhất: {e}")
            raise