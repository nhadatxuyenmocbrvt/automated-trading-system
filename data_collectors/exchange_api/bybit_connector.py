"""
ByBit Exchange Connector - Triển khai kết nối cụ thể cho sàn giao dịch ByBit.
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


class BybitConnector(GenericExchangeConnector):
    """
    Connector cho sàn giao dịch ByBit, triển khai các phương thức cụ thể
    theo API của ByBit.
    """
    
    # Các URL endpoint của ByBit
    V5_API_URL = 'https://api.bybit.com'
    V5_API_TESTNET_URL = 'https://api-testnet.bybit.com'
    
    # Các URL websocket
    V5_PUBLIC_WS_URL = 'wss://stream.bybit.com/v5/public'
    V5_PRIVATE_WS_URL = 'wss://stream.bybit.com/v5/private'
    V5_PUBLIC_WS_TESTNET_URL = 'wss://stream-testnet.bybit.com/v5/public'
    V5_PRIVATE_WS_TESTNET_URL = 'wss://stream-testnet.bybit.com/v5/private'
    
    def __init__(
        self, 
        api_key: str = None, 
        api_secret: str = None,
        sandbox: bool = True,
        rate_limit: bool = True,
        category: str = 'linear',
        config: SystemConfig = None,
        secret_manager: SecretManager = None
    ):
        """
        Khởi tạo connector ByBit.
        
        Args:
            api_key (str, optional): Khóa API. Mặc định lấy từ biến môi trường.
            api_secret (str, optional): Mật khẩu API. Mặc định lấy từ biến môi trường.
            sandbox (bool, optional): Sử dụng testnet. Mặc định là True.
            rate_limit (bool, optional): Kích hoạt giới hạn tốc độ gọi API. Mặc định là True.
            category (str, optional): Loại giao dịch ('spot', 'linear', 'inverse', 'option'). Mặc định là 'linear'.
            config (SystemConfig, optional): Cấu hình hệ thống. Mặc định là None.
            secret_manager (SecretManager, optional): Trình quản lý bí mật. Mặc định là None.
        """
        # Cấu hình bổ sung cho ByBit
        self.category = category
        self.logger = setup_logger("bybit_connector")
        
        # Xác thực loại giao dịch
        if category not in ['spot', 'linear', 'inverse', 'option']:
            self.logger.warning(f"Loại giao dịch không hợp lệ: {category}. Sử dụng 'linear' mặc định.")
            self.category = 'linear'
        
        # Gọi constructor của lớp cha
        super().__init__(
            exchange_id='bybit',
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit,
            config=config,
            secret_manager=secret_manager
        )
        
        # Thiết lập các options đặc biệt cho ByBit
        if category in ['linear', 'inverse']:
            self.exchange.options['defaultType'] = 'future'
            self.async_exchange.options['defaultType'] = 'future'
            self.logger.info(f"Đã thiết lập chế độ futures {category} cho ByBit")
        else:
            self.exchange.options['defaultType'] = 'spot'
            self.async_exchange.options['defaultType'] = 'spot'
            self.logger.info(f"Đã thiết lập chế độ {category} cho ByBit")
        
        # ByBit API V5 settings
        self.exchange.options['versions'] = {
            'public': {'get': {'tradingRules': 'v5'}},
            'private': {'get': {'apiKey': 'v5'}}
        }
        self.async_exchange.options['versions'] = {
            'public': {'get': {'tradingRules': 'v5'}},
            'private': {'get': {'apiKey': 'v5'}}
        }
        
        # Lưu các websocket connection
        self.ws_connections = {}
        
        # Lưu trữ thời gian cập nhật cuối cùng cho các thông tin
        self.last_exchange_info_update = 0
        self.exchange_info = {}
        self.trading_fees = {}
        
        self.logger.info("Đã khởi tạo ByBit connector")
    
    async def initialize(self) -> None:
        """Khởi tạo connector ByBit, tải thông tin thị trường và thiết lập cấu hình cần thiết."""
        await super().initialize()
        
        # Lấy thông tin bổ sung đặc biệt cho ByBit
        await self.get_exchange_info()
        
        # Kiểm tra giới hạn API và trạng thái tài khoản nếu có khóa API
        if self.api_key and self.api_secret:
            try:
                # Lấy thông tin phí giao dịch
                await self.get_trading_fees()
                
                # Kiểm tra quyền tài khoản
                account_info = await self.async_exchange.privateGetV5AccountWalletBalances({
                    'accountType': 'UNIFIED'
                })
                
                if account_info.get('retCode') == 0:
                    self.logger.info(f"Đã xác thực tài khoản ByBit thành công")
                else:
                    self.logger.warning(f"Không thể xác thực tài khoản ByBit: {account_info.get('retMsg')}")
            except Exception as e:
                self.logger.warning(f"Không thể lấy thông tin tài khoản ByBit: {e}")
    
    async def get_exchange_info(self) -> Dict:
        """
        Lấy thông tin chi tiết về sàn giao dịch ByBit.
        
        Returns:
            Dict: Thông tin sàn giao dịch
        """
        # Kiểm tra xem đã lấy thông tin gần đây chưa để tránh gọi API quá nhiều
        current_time = time.time()
        if self.exchange_info and (current_time - self.last_exchange_info_update < 3600):  # 1 giờ
            return self.exchange_info
        
        try:
            # Sử dụng API v5 để lấy thông tin
            params = {'category': self.category}
            
            # ByBit sử dụng API V5 để lấy thông tin công cụ (instruments)
            instruments = await self.async_exchange.publicGetV5MarketInstrumentsInfo(params)
            
            if instruments.get('retCode') == 0:
                result = instruments.get('result', {})
                self.exchange_info = result
                self.last_exchange_info_update = current_time
                
                # Ghi log một số thông tin hữu ích
                list_key = 'list'
                if list_key in result:
                    symbols_count = len(result[list_key])
                    self.logger.info(f"Đã lấy thông tin sàn ByBit: {symbols_count} cặp giao dịch {self.category}")
                
                return result
            else:
                error_msg = instruments.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy thông tin sàn ByBit: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin sàn ByBit: {e}")
            raise
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict:
        """
        Lấy thông tin phí giao dịch của ByBit.
        
        Args:
            symbol (str, optional): Cặp giao dịch cụ thể. Mặc định là None (tất cả).
            
        Returns:
            Dict: Thông tin phí giao dịch
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy thông tin phí giao dịch")
        
        try:
            params = {'category': self.category}
            if symbol:
                params['symbol'] = symbol.replace('/', '')
            
            fee_rate = await self.async_exchange.privateGetV5AccountFeeRate(params)
            
            if fee_rate.get('retCode') == 0:
                result = fee_rate.get('result', {})
                
                # Chuyển đổi định dạng
                fee_info = {}
                list_data = result.get('list', [])
                
                if symbol:
                    # Phí cho một symbol cụ thể
                    if list_data and len(list_data) > 0:
                        fee_info[symbol] = {
                            'maker': float(list_data[0].get('makerFeeRate', 0)),
                            'taker': float(list_data[0].get('takerFeeRate', 0))
                        }
                else:
                    # Phí cho tất cả các symbol
                    for item in list_data:
                        sym = item.get('symbol', '')
                        if sym:
                            fee_info[sym] = {
                                'maker': float(item.get('makerFeeRate', 0)),
                                'taker': float(item.get('takerFeeRate', 0))
                            }
                
                self.trading_fees = fee_info
                return fee_info
            else:
                error_msg = fee_rate.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy thông tin phí giao dịch ByBit: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin phí giao dịch ByBit: {e}")
            raise
    
    async def subscribe_to_websocket(self, symbol: str, channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu từ websocket ByBit.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            channels (List[str]): Các kênh cần đăng ký (ví dụ: ['ticker', 'kline.1', 'orderbook', 'trade'])
        """
        # Chuẩn hóa symbol cho ByBit (loại bỏ ký hiệu '/')
        formatted_symbol = symbol.replace('/', '')
        
        # Xác định websocket URL
        ws_url = self.V5_PUBLIC_WS_TESTNET_URL if self.exchange.sandbox else self.V5_PUBLIC_WS_URL
        
        # Tạo danh sách topics theo định dạng ByBit V5
        topics = self._generate_topics([formatted_symbol], channels)
        
        if not topics:
            self.logger.error("Không có kênh hợp lệ để đăng ký")
            return
        
        # Sử dụng WebSocketManager để quản lý kết nối
        await self.ws_manager.add_connection(
            ws_url,
            [formatted_symbol],
            channels,
            self._connect_bybit_websocket,
            self._process_websocket_message_raw
        )
        
    def _generate_topics(self, symbols: List[str], channels: List[str]) -> List[str]:
        """
        Tạo danh sách topics cho ByBit V5 API.
        
        Args:
            symbols (List[str]): Danh sách symbol đã được định dạng
            channels (List[str]): Danh sách kênh
            
        Returns:
            List[str]: Danh sách topics
        """
        topics = []
        for symbol in symbols:
            for channel in channels:
                if channel == 'ticker':
                    topics.append(f"tickers.{symbol}")
                elif channel.startswith('kline'):
                    interval = channel.split('.')[1] if '.' in channel else '1'
                    topics.append(f"kline.{interval}.{symbol}")
                elif channel == 'orderbook':
                    topics.append(f"orderbook.50.{symbol}")
                elif channel == 'trade':
                    topics.append(f"publicTrade.{symbol}")
                else:
                    self.logger.warning(f"Kênh không được hỗ trợ: {channel}")
        return topics
    
    async def subscribe_multiple_symbols(self, symbols: List[str], channels: List[str]) -> None:
        """
        Đăng ký nhận dữ liệu từ websocket ByBit cho nhiều symbols cùng lúc.
        
        Args:
            symbols (List[str]): Danh sách cặp giao dịch (ví dụ: ['BTC/USDT', 'ETH/USDT'])
            channels (List[str]): Các kênh cần đăng ký (ví dụ: ['ticker', 'kline.1'])
        """
        if not symbols:
            self.logger.error("Danh sách symbols trống")
            return
        
        # Phân nhóm symbols để tối ưu hóa kết nối
        max_symbols_per_connection = min(self.ws_manager.symbols_per_connection, 20)  # ByBit giới hạn 20 topics/kết nối
        
        for i in range(0, len(symbols), max_symbols_per_connection):
            batch_symbols = symbols[i:i+max_symbols_per_connection]
            
            # Chuẩn hóa symbols
            formatted_symbols = [symbol.replace('/', '') for symbol in batch_symbols]
            
            # Xác định websocket URL
            ws_url = self.V5_PUBLIC_WS_TESTNET_URL if self.exchange.sandbox else self.V5_PUBLIC_WS_URL
            
            # Sử dụng WebSocketManager để quản lý kết nối
            await self.ws_manager.add_connection(
                ws_url,
                formatted_symbols,
                channels,
                self._connect_bybit_websocket,
                self._process_websocket_message_raw
            )
    
    async def _connect_bybit_websocket(self, existing_ws, symbols: List[str], channels: List[str]):
        """
        Kết nối tới websocket ByBit và đăng ký các topics.
        
        Args:
            existing_ws: Websocket hiện có (None nếu là kết nối mới)
            symbols (List[str]): Danh sách symbols đã định dạng
            channels (List[str]): Loại kênh dữ liệu
            
        Returns:
            websocket: Kết nối websocket đã thiết lập
        """
        # Tạo danh sách topics
        topics = self._generate_topics(symbols, channels)
        
        # Xác định websocket URL
        ws_url = self.V5_PUBLIC_WS_TESTNET_URL if self.exchange.sandbox else self.V5_PUBLIC_WS_URL
        
        # Nếu đã có kết nối, đóng và tạo lại
        if existing_ws:
            try:
                # Gửi unsubscribe nếu có thể
                try:
                    unsubscribe_message = {
                        "op": "unsubscribe",
                        "args": topics
                    }
                    await existing_ws.send(json.dumps(unsubscribe_message))
                except:
                    pass
                
                await existing_ws.close()
            except:
                pass
        
        # Thiết lập kết nối mới
        try:
            # Sử dụng các tham số kết nối tốt hơn
            websocket = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                max_size=2**24,  # 16MB max message size
                close_timeout=10
            )
            
            # Gửi yêu cầu đăng ký
            subscription_message = {
                "op": "subscribe",
                "args": topics
            }
            await websocket.send(json.dumps(subscription_message))
            
            # Đợi phản hồi đăng ký
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                if 'op' in response_data and response_data['op'] == 'subscribe':
                    if response_data.get('success'):
                        self.logger.info(f"Đăng ký thành công: {len(topics)} topics")
                    else:
                        self.logger.warning(f"Đăng ký không thành công: {response_data.get('ret_msg')}")
            except asyncio.TimeoutError:
                self.logger.warning("Không nhận được phản hồi đăng ký, giả định thành công")
            
            self.logger.info(f"Đã kết nối websocket ByBit cho {len(symbols)} symbols với {len(channels)} kênh")
            return websocket
            
        except Exception as e:
            self.logger.error(f"Không thể kết nối websocket ByBit: {e}")
            raise
    
    async def _process_websocket_message_raw(self, message: str) -> None:
        """
        Xử lý dữ liệu raw từ websocket.
        
        Args:
            message (str): Dữ liệu dạng JSON string từ websocket
        """
        try:
            data = json.loads(message)
            
            # Xử lý các tin nhắn heartbeat và status
            if 'op' in data:
                if data['op'] == 'ping':
                    # Không cần xử lý, thư viện websockets tự động phản hồi pong
                    return
                elif data['op'] == 'subscribe':
                    # Đã xử lý trong _connect_bybit_websocket
                    return
                elif data['op'] == 'pong':
                    # Phản hồi ping
                    return
            
            # Xử lý dữ liệu thực tế
            await self._process_websocket_message(data)
            
        except json.JSONDecodeError:
            self.logger.error(f"Không thể phân tích dữ liệu websocket: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Lỗi xử lý dữ liệu websocket: {e}")
    
    # Thêm phương thức ping websocket định kỳ
    async def _ping_websocket(self, websocket, interval=15):
        """
        Gửi ping định kỳ để giữ kết nối websocket ByBit.
        
        Args:
            websocket: Kết nối websocket
            interval (int): Khoảng thời gian giữa các ping (giây)
        """
        while True:
            try:
                ping_message = {"op": "ping"}
                await websocket.send(json.dumps(ping_message))
                self.logger.debug("Đã gửi ping tới ByBit websocket")
            except Exception as e:
                self.logger.error(f"Lỗi khi gửi ping: {e}")
                break
                
            await asyncio.sleep(interval)
            
    def _parse_bybit_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        Phân tích symbol ByBit để lấy base và quote currency.
        
        Args:
            symbol (str): Symbol của ByBit (ví dụ: 'BTCUSDT')
            
        Returns:
            Tuple[str, str]: (base_currency, quote_currency)
        """
        # Danh sách các quote currency phổ biến
        quote_currencies = ['USDT', 'USD', 'USDC', 'BTC', 'ETH']
        
        # Tìm quote currency
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        
        # Nếu không tìm thấy, thử cách khác
        if len(symbol) >= 6:
            # Giả định 3-4 ký tự cuối là quote currency
            base = symbol[:-4]
            quote = symbol[-4:]
            return base, quote
        
        # Trường hợp mặc định
        return symbol, ""
    
    async def _process_websocket_message(self, data: Dict) -> None:
        """
        Xử lý dữ liệu nhận được từ websocket.
        
        Args:
            data (Dict): Dữ liệu từ websocket
        """
        # Kiểm tra tin nhắn ping/pong
        if 'op' in data and data['op'] == 'ping':
            # Không cần xử lý, ByBit websocket client tự động phản hồi pong
            return
        
        # Kiểm tra và xử lý dữ liệu
        if 'topic' in data and 'data' in data:
            topic = data['topic']
            topic_data = data['data']
            
            # Xác định loại dữ liệu dựa trên topic
            if topic.startswith('tickers'):
                # Xử lý dữ liệu ticker
                self.logger.debug(f"Nhận ticker: {topic} - Dữ liệu: {topic_data}")
                # TODO: Thêm logic để xử lý dữ liệu ticker
                
            elif topic.startswith('kline'):
                # Xử lý dữ liệu nến
                self.logger.debug(f"Nhận kline: {topic}")
                # TODO: Thêm logic để xử lý dữ liệu kline
                
            elif topic.startswith('orderbook'):
                # Xử lý dữ liệu sổ lệnh
                self.logger.debug(f"Nhận orderbook: {topic}")
                # TODO: Thêm logic để xử lý dữ liệu orderbook
                
            elif topic.startswith('publicTrade'):
                # Xử lý dữ liệu giao dịch
                self.logger.debug(f"Nhận trades: {topic}")
                # TODO: Thêm logic để xử lý dữ liệu trade
                
            else:
                self.logger.debug(f"Nhận topic không xác định: {topic}")
        else:
            self.logger.warning(f"Định dạng dữ liệu không được hỗ trợ: {data}")
    
    async def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        Lấy tỷ lệ tài trợ hiện tại cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            
        Returns:
            Dict: Thông tin tỷ lệ tài trợ
        """
        if self.category not in ['linear', 'inverse']:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures (linear hoặc inverse)")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            params = {
                'category': self.category,
                'symbol': formatted_symbol
            }
            
            # Gọi API ByBit để lấy tỷ lệ tài trợ
            funding_info = await self.async_exchange.publicGetV5MarketFundingPrevrate(params)
            
            if funding_info.get('retCode') == 0:
                result = funding_info.get('result', {})
                list_data = result.get('list', [])
                
                if list_data and len(list_data) > 0:
                    return {
                        'symbol': symbol,
                        'fundingRate': float(list_data[0].get('fundingRate', 0)),
                        'fundingTime': list_data[0].get('fundingRateTimestamp', 0)
                    }
                return {}
            else:
                error_msg = funding_info.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy tỷ lệ tài trợ cho {symbol}: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy tỷ lệ tài trợ cho {symbol}: {e}")
            raise
    
    async def fetch_leverage(self, symbol: str) -> Dict:
        """
        Lấy thông tin đòn bẩy hiện tại cho một cặp giao dịch.
        
        Args:
            symbol (str): Cặp giao dịch (ví dụ: 'BTC/USDT')
            
        Returns:
            Dict: Thông tin đòn bẩy
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy thông tin đòn bẩy")
        
        if self.category not in ['linear', 'inverse']:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures (linear hoặc inverse)")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            params = {
                'category': self.category,
                'symbol': formatted_symbol
            }
            
            # Gọi API ByBit để lấy thông tin đòn bẩy
            position_info = await self.async_exchange.privateGetV5PositionList(params)
            
            if position_info.get('retCode') == 0:
                result = position_info.get('result', {})
                list_data = result.get('list', [])
                
                if list_data and len(list_data) > 0:
                    return {
                        'symbol': symbol,
                        'leverage': float(list_data[0].get('leverage', 0)),
                        'marginType': list_data[0].get('tradeMode', 0)  # 0: cross margin, 1: isolated margin
                    }
                return {
                    'symbol': symbol,
                    'leverage': 0,
                    'marginType': 0
                }
            else:
                error_msg = position_info.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy thông tin đòn bẩy cho {symbol}: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin đòn bẩy cho {symbol}: {e}")
            raise
    
    async def set_leverage(self, symbol: str, leverage: int, margin_mode: str = "ISOLATED") -> Dict:
        """
        Thiết lập đòn bẩy cho một cặp giao dịch futures.
        
        Args:
            symbol (str): Cặp giao dịch
            leverage (int): Mức đòn bẩy (1-100 tùy thuộc vào symbol)
            margin_mode (str, optional): Chế độ margin ('ISOLATED' hoặc 'CROSS'). Mặc định là 'ISOLATED'.
            
        Returns:
            Dict: Kết quả thiết lập
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để thiết lập đòn bẩy")
        
        if self.category not in ['linear', 'inverse']:
            raise ValueError("Phương thức này chỉ khả dụng trong chế độ futures (linear hoặc inverse)")
        
        if margin_mode not in ['ISOLATED', 'CROSS']:
            raise ValueError("margin_mode phải là 'ISOLATED' hoặc 'CROSS'")
        
        try:
            # Định dạng lại symbol nếu cần
            formatted_symbol = symbol.replace('/', '')
            
            # Chuyển đổi margin_mode sang tradeMode của ByBit
            trade_mode = 1 if margin_mode == 'ISOLATED' else 0
            
            params = {
                'category': self.category,
                'symbol': formatted_symbol,
                'leverage': leverage,
                'tradeMode': trade_mode
            }
            
            # Gọi API ByBit để thiết lập đòn bẩy
            result = await self.async_exchange.privatePostV5PositionLeverageSave(params)
            
            if result.get('retCode') == 0:
                self.logger.info(f"Đã thiết lập đòn bẩy {leverage}x và margin mode {margin_mode} cho {symbol}")
                return {
                    'success': True,
                    'symbol': symbol,
                    'leverage': leverage,
                    'marginMode': margin_mode
                }
            else:
                error_msg = result.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể thiết lập đòn bẩy cho {symbol}: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể thiết lập đòn bẩy cho {symbol}: {e}")
            raise
    
    async def fetch_tickers(self, symbols: List[str] = None) -> Dict:
        """
        Lấy thông tin ticker cho nhiều cặp giao dịch.
        
        Args:
            symbols (List[str], optional): Danh sách cặp giao dịch. Mặc định là None (tất cả).
            
        Returns:
            Dict: Thông tin ticker
        """
        try:
            params = {'category': self.category}
            
            if symbols and len(symbols) > 0:
                # Lấy ticker cho từng symbol riêng lẻ
                result = {}
                for symbol in symbols:
                    formatted_symbol = symbol.replace('/', '')
                    params['symbol'] = formatted_symbol
                    
                    ticker_data = await self.async_exchange.publicGetV5MarketTickers(params)
                    
                    if ticker_data.get('retCode') == 0:
                        ticker_list = ticker_data.get('result', {}).get('list', [])
                        if ticker_list and len(ticker_list) > 0:
                            result[symbol] = ticker_list[0]
                
                return result
            else:
                # Lấy tất cả ticker
                ticker_data = await self.async_exchange.publicGetV5MarketTickers(params)
                
                if ticker_data.get('retCode') == 0:
                    ticker_list = ticker_data.get('result', {}).get('list', [])
                    
                    # Chuyển đổi định dạng
                    result = {}
                    for ticker in ticker_list:
                        symbol = ticker.get('symbol', '')
                        if symbol:
                            # Định dạng lại symbol với dấu '/'
                            for quote in ['USDT', 'USD', 'BTC', 'ETH']:
                                if symbol.endswith(quote):
                                    base = symbol[:-len(quote)]
                                    formatted_symbol = f"{base}/{quote}"
                                    result[formatted_symbol] = ticker
                                    break
                            else:
                                # Nếu không thể định dạng, sử dụng symbol gốc
                                result[symbol] = ticker
                    
                    return result
                else:
                    error_msg = ticker_data.get('retMsg', 'Unknown error')
                    self.logger.error(f"Không thể lấy thông tin ticker: {error_msg}")
                    raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy thông tin ticker: {e}")
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
    
    # === PHƯƠNG THỨC TIỆN ÍCH BYBIT ===
    
    def symbol_formatter(self, base_currency: str, quote_currency: str) -> str:
        """
        Định dạng cặp giao dịch theo cách ByBit yêu cầu.
        
        Args:
            base_currency (str): Tiền tệ cơ sở
            quote_currency (str): Tiền tệ báo giá
            
        Returns:
            str: Chuỗi định dạng cặp giao dịch
        """
        # ByBit V5 API sử dụng định dạng với dấu '/'
        return f"{base_currency.upper()}/{quote_currency.upper()}"
    
    async def get_all_symbols(self) -> List[str]:
        """
        Lấy tất cả các cặp giao dịch có sẵn.
        
        Returns:
            List[str]: Danh sách các cặp giao dịch
        """
        exchange_info = await self.get_exchange_info()
        
        symbols = []
        list_data = exchange_info.get('list', [])
        
        for item in list_data:
            symbol = item.get('symbol', '')
            if symbol:
                # Định dạng lại symbol với dấu '/'
                for quote in ['USDT', 'USD', 'BTC', 'ETH']:
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        formatted_symbol = f"{base}/{quote}"
                        symbols.append(formatted_symbol)
                        break
                else:
                    # Nếu không thể định dạng, sử dụng symbol gốc
                    symbols.append(symbol)
        
        return symbols
    
    async def get_server_time(self) -> int:
        """
        Lấy thời gian của máy chủ ByBit.
        
        Returns:
            int: Thời gian máy chủ (timestamp)
        """
        try:
            time_info = await self.async_exchange.publicGetV5MarketTime()
            
            if time_info.get('retCode') == 0:
                return int(time_info.get('result', {}).get('timeSecond', 0)) * 1000
            else:
                error_msg = time_info.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy thời gian máy chủ ByBit: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy thời gian máy chủ ByBit: {e}")
            raise
    
    async def get_wallet_balance(self, coin: Optional[str] = None) -> Dict:
        """
        Lấy số dư ví ByBit.
        
        Args:
            coin (str, optional): Mã tiền tệ. Mặc định là None (tất cả).
            
        Returns:
            Dict: Thông tin số dư ví
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key và secret cần thiết để lấy số dư ví")
        
        try:
            params = {'accountType': 'UNIFIED'}
            if coin:
                params['coin'] = coin.upper()
            
            balance_info = await self.async_exchange.privateGetV5AccountWalletBalance(params)
            
            if balance_info.get('retCode') == 0:
                return balance_info.get('result', {})
            else:
                error_msg = balance_info.get('retMsg', 'Unknown error')
                self.logger.error(f"Không thể lấy số dư ví ByBit: {error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Không thể lấy số dư ví ByBit: {e}")
            raise