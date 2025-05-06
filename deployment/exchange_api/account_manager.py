"""
Quản lý tài khoản sàn giao dịch.
File này cung cấp các chức năng để quản lý tài khoản giao dịch,
bao gồm theo dõi số dư, lịch sử giao dịch, và các thông tin tài khoản khác.
"""

import time
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import threading
import json

from config.constants import ErrorCode
from config.utils.validators import is_valid_number
from config.env import get_env
from config.logging_config import get_logger
from data_collectors.exchange_api.generic_connector import APIError

class AccountManager:
    """
    Lớp quản lý tài khoản giao dịch.
    Cung cấp các phương thức để theo dõi số dư, lịch sử giao dịch, 
    và thông tin tài khoản trên các sàn giao dịch.
    """
    
    def __init__(self, exchange_connector):
        """
        Khởi tạo đối tượng AccountManager.
        
        Args:
            exchange_connector: Đối tượng kết nối với sàn giao dịch
        """
        self.exchange = exchange_connector
        self.logger = get_logger("account_manager")
        
        # Cache cho dữ liệu tài khoản
        self._balance_cache = {}
        self._balance_cache_time = 0
        self._balance_cache_lock = threading.Lock()
        
        # Cache cho lịch sử giao dịch
        self._trade_history_cache = {}
        self._trade_history_cache_time = 0
        self._trade_history_cache_lock = threading.Lock()
        
        # Cấu hình thời gian hết hạn cache (giây)
        self.balance_cache_expiry = int(get_env("BALANCE_CACHE_EXPIRY", "30"))
        self.trade_history_cache_expiry = int(get_env("TRADE_HISTORY_CACHE_EXPIRY", "300"))
        
        # Cấu hình số lượng lần thử lại
        self.max_retries = int(get_env("MAX_RETRIES", "3"))
        
        self.logger.info(f"Khởi tạo AccountManager cho sàn {self.exchange.exchange_id}")
    
    def get_balance(self, force_update: bool = False, retries: int = None) -> Dict[str, Dict[str, float]]:
        """
        Lấy số dư tài khoản.
        
        Args:
            force_update: Bỏ qua cache và lấy dữ liệu mới
            retries: Số lần thử lại (None để sử dụng giá trị mặc định)
            
        Returns:
            Dict chứa số dư của các đồng tiền
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        # Dùng số lần thử lại mặc định nếu không được chỉ định
        if retries is None:
            retries = self.max_retries
        
        current_time = time.time()
        
        # Kiểm tra cache nếu không yêu cầu cập nhật mới
        if not force_update:
            with self._balance_cache_lock:
                if (current_time - self._balance_cache_time < self.balance_cache_expiry and 
                    self._balance_cache):
                    self.logger.debug("Trả về số dư từ cache")
                    return self._balance_cache
        
        try:
            # Lấy số dư tài khoản từ sàn giao dịch
            balance = self.exchange.fetch_balance()
            
            # Cập nhật cache
            with self._balance_cache_lock:
                self._balance_cache = balance
                self._balance_cache_time = current_time
            
            self.logger.info(f"Đã cập nhật số dư tài khoản trên {self.exchange.exchange_id}")
            return balance
            
        except Exception as e:
            # Thử lại nếu còn lần thử
            if retries > 0:
                self.logger.warning(f"Lỗi khi lấy số dư, thử lại ({retries}): {str(e)}")
                time.sleep(1)  # Chờ 1 giây trước khi thử lại
                return self.get_balance(force_update=True, retries=retries-1)
            
            # Nếu hết lần thử, trả về cache nếu có
            if self._balance_cache:
                self.logger.warning(f"Sử dụng cache cũ sau khi không thể lấy số dư mới: {str(e)}")
                return self._balance_cache
            
            # Nếu không có cache, báo lỗi
            self.logger.error(f"Không thể lấy số dư tài khoản: {str(e)}")
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy số dư tài khoản: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_available_balance(self, symbol: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Lấy số dư khả dụng cho giao dịch.
        
        Args:
            symbol: Mã tiền tệ (ví dụ: 'USDT', 'BTC'). Nếu None, trả về tất cả
            
        Returns:
            Số dư khả dụng của symbol hoặc dict tất cả số dư khả dụng
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        balance = self.get_balance()
        
        # Trích xuất số dư khả dụng
        available_balance = {}
        for currency, details in balance.items():
            if 'free' in details:
                available_balance[currency] = details['free']
        
        # Trả về số dư của một symbol cụ thể
        if symbol:
            symbol = symbol.upper()
            return available_balance.get(symbol, 0.0)
        
        # Trả về tất cả số dư
        return available_balance
    
    def get_total_balance(self, symbol: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Lấy tổng số dư (bao gồm cả số dư trong lệnh và vị thế).
        
        Args:
            symbol: Mã tiền tệ (ví dụ: 'USDT', 'BTC'). Nếu None, trả về tất cả
            
        Returns:
            Tổng số dư của symbol hoặc dict tất cả tổng số dư
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        balance = self.get_balance()
        
        # Trích xuất tổng số dư
        total_balance = {}
        for currency, details in balance.items():
            if 'total' in details:
                total_balance[currency] = details['total']
        
        # Trả về số dư của một symbol cụ thể
        if symbol:
            symbol = symbol.upper()
            return total_balance.get(symbol, 0.0)
        
        # Trả về tất cả số dư
        return total_balance
    
    def get_locked_balance(self, symbol: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Lấy số dư đang bị khóa trong các lệnh.
        
        Args:
            symbol: Mã tiền tệ (ví dụ: 'USDT', 'BTC'). Nếu None, trả về tất cả
            
        Returns:
            Số dư bị khóa của symbol hoặc dict tất cả số dư bị khóa
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        balance = self.get_balance()
        
        # Trích xuất số dư bị khóa
        locked_balance = {}
        for currency, details in balance.items():
            if 'used' in details:
                locked_balance[currency] = details['used']
        
        # Trả về số dư bị khóa của một symbol cụ thể
        if symbol:
            symbol = symbol.upper()
            return locked_balance.get(symbol, 0.0)
        
        # Trả về tất cả số dư bị khóa
        return locked_balance
    
    def get_btc_value(self) -> float:
        """
        Lấy tổng giá trị tài khoản quy đổi ra BTC.
        
        Returns:
            Tổng giá trị tài khoản theo BTC
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            balance = self.get_balance()
            
            # Kiểm tra xem sàn có cung cấp giá trị BTC không
            if 'info' in balance and 'totalBtcValue' in balance['info']:
                return float(balance['info']['totalBtcValue'])
            
            # Nếu không, tính toán dựa trên tỷ giá hiện tại
            btc_value = 0.0
            
            # Lấy tất cả các cặp giao dịch với BTC
            btc_pairs = {}
            markets = self.exchange.fetch_markets()
            
            for market in markets:
                base = market['base']
                quote = market['quote']
                
                if quote == 'BTC':
                    symbol = f"{base}/{quote}"
                    try:
                        # Lấy giá hiện tại
                        ticker = self.exchange.fetch_ticker(symbol)
                        btc_pairs[base] = ticker['last']
                    except Exception as e:
                        self.logger.warning(f"Không thể lấy giá {symbol}: {str(e)}")
            
            # Tính tổng giá trị BTC
            for currency, details in balance.items():
                if currency == 'BTC':
                    btc_value += details.get('total', 0.0)
                    continue
                
                if currency in btc_pairs:
                    btc_value += details.get('total', 0.0) * btc_pairs[currency]
            
            return btc_value
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính giá trị BTC: {str(e)}")
            return 0.0
    
    def get_equivalent_usdt_value(self) -> float:
        """
        Lấy tổng giá trị tài khoản quy đổi ra USDT.
        
        Returns:
            Tổng giá trị tài khoản theo USDT
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            balance = self.get_balance()
            
            # Kiểm tra xem sàn có cung cấp giá trị USDT không
            if 'info' in balance and 'totalUsdtValue' in balance['info']:
                return float(balance['info']['totalUsdtValue'])
            
            # Nếu không, tính toán dựa trên tỷ giá hiện tại
            usdt_value = 0.0
            
            # Lấy tất cả các cặp giao dịch với USDT
            usdt_pairs = {}
            markets = self.exchange.fetch_markets()
            
            for market in markets:
                base = market['base']
                quote = market['quote']
                
                if quote == 'USDT':
                    symbol = f"{base}/{quote}"
                    try:
                        # Lấy giá hiện tại
                        ticker = self.exchange.fetch_ticker(symbol)
                        usdt_pairs[base] = ticker['last']
                    except Exception as e:
                        self.logger.warning(f"Không thể lấy giá {symbol}: {str(e)}")
            
            # Lấy giá BTC/USDT nếu cần
            btc_usdt_price = 0.0
            try:
                btc_ticker = self.exchange.fetch_ticker('BTC/USDT')
                btc_usdt_price = btc_ticker['last']
            except Exception as e:
                self.logger.warning(f"Không thể lấy giá BTC/USDT: {str(e)}")
            
            # Tính tổng giá trị USDT
            for currency, details in balance.items():
                total = details.get('total', 0.0)
                if total <= 0:
                    continue
                
                if currency == 'USDT':
                    usdt_value += total
                    continue
                
                if currency in usdt_pairs:
                    usdt_value += total * usdt_pairs[currency]
                elif btc_usdt_price > 0 and currency == 'BTC':
                    usdt_value += total * btc_usdt_price
            
            return usdt_value
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính giá trị USDT: {str(e)}")
            return 0.0
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         since: Optional[int] = None, 
                         limit: Optional[int] = 100,
                         force_update: bool = False) -> List[Dict]:
        """
        Lấy lịch sử giao dịch.
        
        Args:
            symbol: Symbol cần lấy lịch sử (None để lấy tất cả)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            force_update: Bỏ qua cache và lấy dữ liệu mới
            
        Returns:
            Danh sách lịch sử giao dịch
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        cache_key = f"{symbol}_{since}_{limit}"
        current_time = time.time()
        
        # Kiểm tra cache nếu không yêu cầu cập nhật mới
        if not force_update:
            with self._trade_history_cache_lock:
                if (cache_key in self._trade_history_cache and
                    current_time - self._trade_history_cache_time < self.trade_history_cache_expiry):
                    self.logger.debug(f"Trả về lịch sử giao dịch từ cache cho {symbol}")
                    return self._trade_history_cache[cache_key]
        
        try:
            # Lấy lịch sử giao dịch từ sàn
            trades = self.exchange.fetch_my_trades(symbol, since, limit)
            
            # Cập nhật cache
            with self._trade_history_cache_lock:
                self._trade_history_cache[cache_key] = trades
                self._trade_history_cache_time = current_time
            
            self.logger.info(f"Đã cập nhật lịch sử giao dịch cho {symbol or 'tất cả'}")
            return trades
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy lịch sử giao dịch: {str(e)}")
            
            # Trả về cache nếu có
            if cache_key in self._trade_history_cache:
                self.logger.warning(f"Sử dụng cache cũ cho lịch sử giao dịch: {str(e)}")
                return self._trade_history_cache[cache_key]
            
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy lịch sử giao dịch: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_deposits(self, currency: Optional[str] = None, 
                    since: Optional[int] = None, 
                    limit: Optional[int] = 100) -> List[Dict]:
        """
        Lấy lịch sử nạp tiền.
        
        Args:
            currency: Mã tiền tệ (None để lấy tất cả)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            
        Returns:
            Danh sách lịch sử nạp tiền
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            deposits = self.exchange.fetch_deposits(currency, since, limit)
            self.logger.info(f"Đã lấy {len(deposits)} lịch sử nạp tiền" +
                            (f" cho {currency}" if currency else ""))
            return deposits
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy lịch sử nạp tiền: {str(e)}")
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy lịch sử nạp tiền: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_withdrawals(self, currency: Optional[str] = None, 
                       since: Optional[int] = None, 
                       limit: Optional[int] = 100) -> List[Dict]:
        """
        Lấy lịch sử rút tiền.
        
        Args:
            currency: Mã tiền tệ (None để lấy tất cả)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            
        Returns:
            Danh sách lịch sử rút tiền
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            withdrawals = self.exchange.fetch_withdrawals(currency, since, limit)
            self.logger.info(f"Đã lấy {len(withdrawals)} lịch sử rút tiền" +
                            (f" cho {currency}" if currency else ""))
            return withdrawals
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy lịch sử rút tiền: {str(e)}")
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy lịch sử rút tiền: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_deposit_address(self, currency: str, network: Optional[str] = None) -> Dict:
        """
        Lấy địa chỉ nạp tiền cho một đồng coin.
        
        Args:
            currency: Mã tiền tệ (ví dụ: 'BTC', 'ETH')
            network: Mạng lưới (tùy chọn, ví dụ: 'ERC20', 'TRC20')
            
        Returns:
            Thông tin địa chỉ nạp tiền
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        params = {}
        if network:
            params['network'] = network
            
        try:
            address = self.exchange.fetch_deposit_address(currency, params)
            self.logger.info(f"Đã lấy địa chỉ nạp tiền cho {currency}" +
                            (f" trên mạng {network}" if network else ""))
            return address
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy địa chỉ nạp tiền: {str(e)}")
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy địa chỉ nạp tiền: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_trade_volume(self, timeframe: str = '30d') -> Dict:
        """
        Lấy khối lượng giao dịch trong một khoảng thời gian.
        
        Args:
            timeframe: Khoảng thời gian ('24h', '7d', '30d')
            
        Returns:
            Thông tin khối lượng giao dịch
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        # Tính thời gian bắt đầu dựa trên timeframe
        now = datetime.now()
        
        if timeframe == '24h':
            since = int((now - timedelta(days=1)).timestamp() * 1000)
        elif timeframe == '7d':
            since = int((now - timedelta(days=7)).timestamp() * 1000)
        elif timeframe == '30d':
            since = int((now - timedelta(days=30)).timestamp() * 1000)
        else:
            raise ValueError(f"Timeframe không hợp lệ: {timeframe}")
        
        try:
            # Lấy lịch sử giao dịch
            trades = self.get_trade_history(since=since, limit=1000)
            
            # Tính tổng khối lượng
            total_volume = 0.0
            total_cost = 0.0
            
            for trade in trades:
                # Số lượng và giá
                amount = trade.get('amount', 0.0)
                price = trade.get('price', 0.0)
                cost = trade.get('cost', amount * price)
                
                total_volume += amount
                total_cost += cost
            
            # Phân tích theo symbol
            volume_by_symbol = {}
            
            for trade in trades:
                symbol = trade.get('symbol', '')
                amount = trade.get('amount', 0.0)
                cost = trade.get('cost', 0.0)
                
                if symbol not in volume_by_symbol:
                    volume_by_symbol[symbol] = {'volume': 0.0, 'cost': 0.0}
                
                volume_by_symbol[symbol]['volume'] += amount
                volume_by_symbol[symbol]['cost'] += cost
            
            # Kết quả
            result = {
                'total_volume': total_volume,
                'total_cost': total_cost,
                'by_symbol': volume_by_symbol,
                'timeframe': timeframe,
                'start_time': since,
                'end_time': int(now.timestamp() * 1000),
                'trade_count': len(trades)
            }
            
            self.logger.info(f"Đã tính toán khối lượng giao dịch cho {timeframe}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính khối lượng giao dịch: {str(e)}")
            raise APIError(
                error_code=ErrorCode.DATA_NOT_FOUND,
                message=f"Không thể tính khối lượng giao dịch: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def get_fee_tier(self) -> Dict:
        """
        Lấy thông tin mức phí giao dịch hiện tại.
        
        Returns:
            Thông tin mức phí
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            # Lấy thông tin phí
            fees = self.exchange.fetch_trading_fees()
            
            # Thêm khối lượng giao dịch 30 ngày
            try:
                volume = self.get_trade_volume('30d')
                return {
                    'fees': fees,
                    'volume_30d': volume.get('total_cost', 0.0),
                    'trade_count_30d': volume.get('trade_count', 0)
                }
            except Exception as e:
                self.logger.warning(f"Không thể lấy khối lượng giao dịch: {str(e)}")
                return {'fees': fees}
                
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy thông tin mức phí: {str(e)}")
            raise APIError(
                error_code=ErrorCode.API_ERROR,
                message=f"Không thể lấy thông tin mức phí: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def analyze_trade_history(self, symbol: Optional[str] = None, 
                            days: int = 30) -> Dict:
        """
        Phân tích lịch sử giao dịch.
        
        Args:
            symbol: Symbol cần phân tích (None để phân tích tất cả)
            days: Số ngày cần phân tích
            
        Returns:
            Kết quả phân tích
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        try:
            # Tính thời gian bắt đầu
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Lấy lịch sử giao dịch
            trades = self.get_trade_history(symbol, since, 1000)
            
            if not trades:
                return {
                    'message': f"Không có giao dịch nào trong {days} ngày qua",
                    'trade_count': 0
                }
            
            # Chuyển đổi thành DataFrame để dễ phân tích
            df = pd.DataFrame(trades)
            
            # Thêm cột side_num để tính P&L
            df['side_num'] = df['side'].apply(lambda x: 1 if x == 'buy' else -1)
            
            # Tính P&L
            results = {
                'trade_count': len(df),
                'symbols': len(df['symbol'].unique()),
                'volume': df['amount'].sum(),
                'cost': df['cost'].sum(),
                'fee': df['fee'].apply(lambda x: x.get('cost', 0) if isinstance(x, dict) else 0).sum()
            }
            
            # Phân tích theo symbol
            by_symbol = {}
            
            for sym, group in df.groupby('symbol'):
                buy_trades = group[group['side'] == 'buy']
                sell_trades = group[group['side'] == 'sell']
                
                buy_volume = buy_trades['amount'].sum()
                sell_volume = sell_trades['amount'].sum()
                buy_cost = buy_trades['cost'].sum()
                sell_cost = sell_trades['cost'].sum()
                
                # Ước tính P&L
                realized_pnl = sell_cost - buy_cost * (sell_volume / buy_volume) if buy_volume > 0 else 0
                
                by_symbol[sym] = {
                    'trade_count': len(group),
                    'buy_count': len(buy_trades),
                    'sell_count': len(sell_trades),
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_cost': buy_cost,
                    'sell_cost': sell_cost,
                    'estimated_pnl': realized_pnl
                }
            
            results['by_symbol'] = by_symbol
            
            # Phân tích theo ngày
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['datetime'].dt.date
            
            by_date = {}
            
            for date, group in df.groupby('date'):
                date_str = date.strftime('%Y-%m-%d')
                by_date[date_str] = {
                    'trade_count': len(group),
                    'volume': group['amount'].sum(),
                    'cost': group['cost'].sum()
                }
            
            results['by_date'] = by_date
            results['days'] = days
            results['start_date'] = datetime.fromtimestamp(since/1000).strftime('%Y-%m-%d')
            results['end_date'] = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"Đã phân tích {len(trades)} giao dịch trong {days} ngày qua")
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích lịch sử giao dịch: {str(e)}")
            raise APIError(
                error_code=ErrorCode.DATA_CORRUPTED,
                message=f"Không thể phân tích lịch sử giao dịch: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def validate_balance_for_order(self, symbol: str, side: str, 
                                 amount: float, price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Kiểm tra xem số dư có đủ để đặt lệnh không.
        
        Args:
            symbol: Symbol cần đặt lệnh (ví dụ: 'BTC/USDT')
            side: Phía đặt lệnh ('buy' hoặc 'sell')
            amount: Số lượng đặt lệnh
            price: Giá đặt lệnh (không cần thiết cho lệnh market)
            
        Returns:
            Tuple (is_valid, message)
        """
        try:
            # Kiểm tra tham số đầu vào
            if not is_valid_number(amount, min_value=0):
                return False, "Số lượng không hợp lệ"
                
            if price is not None and not is_valid_number(price, min_value=0):
                return False, "Giá không hợp lệ"
                
            if side not in ['buy', 'sell']:
                return False, "Side không hợp lệ, phải là 'buy' hoặc 'sell'"
            
            # Phân tách base và quote từ symbol
            parts = symbol.split('/')
            if len(parts) != 2:
                return False, f"Symbol không hợp lệ: {symbol}"
                
            base_currency = parts[0]
            quote_currency = parts[1]
            
            # Lấy số dư khả dụng
            balances = self.get_available_balance()
            
            # Kiểm tra đủ số dư
            if side == 'buy':
                # Đối với lệnh mua, cần đủ quote currency
                required_quote = amount * (price or 0)
                if price is None:
                    # Đối với lệnh market, cần lấy giá hiện tại
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        required_quote = amount * current_price
                    except Exception as e:
                        self.logger.error(f"Không thể lấy giá hiện tại: {str(e)}")
                        return False, f"Không thể xác định giá hiện tại cho {symbol}"
                
                # Thêm buffer 1% cho phí và biến động giá
                required_quote *= 1.01
                
                available_quote = balances.get(quote_currency, 0.0)
                
                if available_quote < required_quote:
                    return False, f"Số dư {quote_currency} không đủ: có {available_quote}, cần {required_quote}"
                    
            else:  # side == 'sell'
                # Đối với lệnh bán, cần đủ base currency
                available_base = balances.get(base_currency, 0.0)
                
                if available_base < amount:
                    return False, f"Số dư {base_currency} không đủ: có {available_base}, cần {amount}"
            
            return True, "Số dư đủ để đặt lệnh"
            
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra số dư: {str(e)}")
            return False, f"Lỗi khi kiểm tra số dư: {str(e)}"
    
    def transfer_between_accounts(self, currency: str, amount: float, 
                                from_account: str, to_account: str) -> Dict:
        """
        Chuyển tiền giữa các tài khoản (ví dụ: từ spot sang futures).
        
        Args:
            currency: Mã tiền tệ cần chuyển
            amount: Số lượng cần chuyển
            from_account: Tài khoản nguồn ('spot', 'margin', 'futures')
            to_account: Tài khoản đích ('spot', 'margin', 'futures')
            
        Returns:
            Kết quả chuyển tiền
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        # Kiểm tra tham số đầu vào
        if not is_valid_number(amount, min_value=0):
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Số lượng không hợp lệ",
                exchange=self.exchange.exchange_id
            )
        
        valid_accounts = ['spot', 'margin', 'futures', 'funding']
        if from_account not in valid_accounts or to_account not in valid_accounts:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message=f"Tài khoản không hợp lệ, phải là một trong: {', '.join(valid_accounts)}",
                exchange=self.exchange.exchange_id
            )
        
        if from_account == to_account:
            raise APIError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message="Tài khoản nguồn và đích không thể giống nhau",
                exchange=self.exchange.exchange_id
            )
        
        try:
            # Kiểm tra số dư
            balance = self.get_available_balance(currency)
            if balance < amount:
                raise APIError(
                    error_code=ErrorCode.INSUFFICIENT_BALANCE,
                    message=f"Số dư {currency} không đủ: có {balance}, cần {amount}",
                    exchange=self.exchange.exchange_id
                )
            
            # Gọi API chuyển tiền (tùy thuộc vào sàn giao dịch)
            params = {
                'type': f"{from_account.upper()}_TO_{to_account.upper()}"
            }
            
            # Kiểm tra phương thức API cụ thể của sàn
            if hasattr(self.exchange, 'transfer'):
                result = self.exchange.transfer(currency, amount, from_account, to_account, params)
            elif hasattr(self.exchange, 'sapi_post_futures_transfer'):
                # Đặc biệt cho Binance
                result = self.exchange.sapi_post_futures_transfer({
                    'asset': currency,
                    'amount': amount,
                    'type': 1 if from_account == 'spot' and to_account == 'futures' else 2
                })
            else:
                raise APIError(
                    error_code=ErrorCode.API_ERROR,
                    message=f"Sàn {self.exchange.exchange_id} không hỗ trợ chuyển tiền giữa các tài khoản",
                    exchange=self.exchange.exchange_id
                )
            
            self.logger.info(f"Đã chuyển {amount} {currency} từ {from_account} sang {to_account}")
            
            # Làm mới cache số dư
            self.get_balance(force_update=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển tiền: {str(e)}")
            raise APIError(
                error_code=ErrorCode.API_ERROR,
                message=f"Không thể chuyển tiền: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def save_balance_snapshot(self, filename: Optional[str] = None) -> str:
        """
        Lưu snapshot số dư hiện tại vào file.
        
        Args:
            filename: Tên file (mặc định: balance_snapshot_{timestamp}.json)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"balance_snapshot_{timestamp}.json"
        
        # Lấy số dư
        balance = self.get_balance(force_update=True)
        
        # Thêm metadata
        snapshot = {
            'timestamp': int(time.time()),
            'datetime': datetime.now().isoformat(),
            'exchange': self.exchange.exchange_id,
            'balance': balance,
            'btc_value': self.get_btc_value(),
            'usdt_value': self.get_equivalent_usdt_value()
        }
        
        # Lưu file
        try:
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=4)
            
            self.logger.info(f"Đã lưu snapshot số dư vào {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu snapshot số dư: {str(e)}")
            return ""
    
    def compare_with_snapshot(self, snapshot_file: str) -> Dict:
        """
        So sánh số dư hiện tại với snapshot đã lưu.
        
        Args:
            snapshot_file: Đường dẫn đến file snapshot
            
        Returns:
            Kết quả so sánh
        """
        try:
            # Đọc snapshot
            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
            
            # Lấy số dư hiện tại
            current_balance = self.get_balance(force_update=True)
            current_btc_value = self.get_btc_value()
            current_usdt_value = self.get_equivalent_usdt_value()
            
            # So sánh
            old_balances = snapshot.get('balance', {})
            changes = {}
            
            # Tính toán sự thay đổi
            all_currencies = set(list(current_balance.keys()) + list(old_balances.keys()))
            
            for currency in all_currencies:
                old_total = old_balances.get(currency, {}).get('total', 0.0)
                current_total = current_balance.get(currency, {}).get('total', 0.0)
                
                if old_total != current_total:
                    changes[currency] = {
                        'old': old_total,
                        'current': current_total,
                        'change': current_total - old_total,
                        'percent': (current_total - old_total) / old_total * 100 if old_total else float('inf')
                    }
            
            # Tính toán sự thay đổi giá trị
            old_btc_value = snapshot.get('btc_value', 0.0)
            old_usdt_value = snapshot.get('usdt_value', 0.0)
            
            value_change = {
                'btc': {
                    'old': old_btc_value,
                    'current': current_btc_value,
                    'change': current_btc_value - old_btc_value,
                    'percent': (current_btc_value - old_btc_value) / old_btc_value * 100 if old_btc_value else float('inf')
                },
                'usdt': {
                    'old': old_usdt_value,
                    'current': current_usdt_value,
                    'change': current_usdt_value - old_usdt_value,
                    'percent': (current_usdt_value - old_usdt_value) / old_usdt_value * 100 if old_usdt_value else float('inf')
                }
            }
            
            # Kết quả
            result = {
                'snapshot_time': snapshot.get('datetime', ''),
                'current_time': datetime.now().isoformat(),
                'time_diff_hours': (time.time() - snapshot.get('timestamp', 0)) / 3600,
                'changes': changes,
                'value_change': value_change
            }
            
            self.logger.info(f"Đã so sánh số dư hiện tại với snapshot {snapshot_file}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi so sánh snapshot: {str(e)}")
            return {'error': str(e)}

    def get_account_info(self) -> Dict:
        """
        Lấy thông tin chi tiết về tài khoản.
        
        Returns:
            Thông tin chi tiết tài khoản
        """
        try:
            # Lấy thông tin cơ bản
            balance = self.get_balance(force_update=True)
            btc_value = self.get_btc_value()
            usdt_value = self.get_equivalent_usdt_value()
            
            # Lấy thông tin phí
            fees = {}
            try:
                fees = self.exchange.fetch_trading_fees()
            except Exception as e:
                self.logger.warning(f"Không thể lấy thông tin phí: {str(e)}")
            
            # Lấy thông tin khối lượng giao dịch
            volume = {}
            try:
                volume = self.get_trade_volume('30d')
            except Exception as e:
                self.logger.warning(f"Không thể lấy khối lượng giao dịch: {str(e)}")
            
            # Thông tin tài khoản
            result = {
                'exchange': self.exchange.exchange_id,
                'timestamp': int(time.time()),
                'datetime': datetime.now().isoformat(),
                'btc_value': btc_value,
                'usdt_value': usdt_value,
                'fees': fees,
                'trade_volume_30d': volume.get('total_cost', 0.0),
                'trade_count_30d': volume.get('trade_count', 0),
                'balance_count': len([c for c in balance if balance[c].get('total', 0.0) > 0]),
                'balance': balance
            }
            
            # Thêm thông tin vị thế nếu là tài khoản futures
            if hasattr(self.exchange, 'fetch_positions'):
                try:
                    positions = self.exchange.fetch_positions()
                    active_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
                    result['positions'] = {
                        'total': len(active_positions),
                        'details': active_positions
                    }
                except Exception as e:
                    self.logger.warning(f"Không thể lấy thông tin vị thế: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy thông tin tài khoản: {str(e)}")
            return {'error': str(e)}