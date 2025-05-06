"""
Theo dõi vị thế giao dịch.
File này cung cấp các chức năng theo dõi và quản lý vị thế giao dịch theo thời gian thực,
bao gồm theo dõi trạng thái, lợi nhuận, và các chỉ số liên quan đến vị thế.
"""

import time
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json

from config.constants import PositionSide, PositionStatus, ErrorCode
from config.utils.validators import is_valid_number
from config.env import get_env
from config.logging_config import get_logger
from data_collectors.exchange_api.generic_connector import APIError
from risk_management.stop_loss import StopLossManager
from risk_management.take_profit import TakeProfitManager

class PositionTracker:
    """
    Lớp theo dõi và quản lý vị thế giao dịch.
    Cung cấp các phương thức để theo dõi trạng thái vị thế,
    tính toán lợi nhuận, và quản lý các tham số liên quan đến vị thế.
    """
    
    def __init__(self, exchange_connector, 
                stop_loss_manager: Optional[StopLossManager] = None,
                take_profit_manager: Optional[TakeProfitManager] = None):
        """
        Khởi tạo đối tượng PositionTracker.
        
        Args:
            exchange_connector: Đối tượng kết nối với sàn giao dịch
            stop_loss_manager: Đối tượng quản lý dừng lỗ (tùy chọn)
            take_profit_manager: Đối tượng quản lý chốt lời (tùy chọn)
        """
        self.exchange = exchange_connector
        self.logger = get_logger("position_tracker")
        
        # Khởi tạo quản lý dừng lỗ và chốt lời nếu không được cung cấp
        self.stop_loss_manager = stop_loss_manager or StopLossManager()
        self.take_profit_manager = take_profit_manager or TakeProfitManager()
        
        # Cache cho dữ liệu vị thế
        self._positions_cache = {}
        self._positions_cache_time = 0
        self._positions_cache_lock = threading.Lock()
        
        # Lịch sử vị thế
        self.position_history = []
        
        # Thời gian cập nhật vị thế
        self.last_update_time = {}
        
        # Cấu hình thời gian hết hạn cache (giây)
        self.positions_cache_expiry = int(get_env("POSITIONS_CACHE_EXPIRY", "10"))
        
        # Cấu hình số lượng lần thử lại
        self.max_retries = int(get_env("MAX_RETRIES", "3"))
        
        # Theo dõi giá PnL thực thi
        self.trade_execution_prices = {}
        
        # Biến theo dõi cho threading
        self._stop_tracker = False
        self._tracker_thread = None
        
        self.logger.info(f"Khởi tạo PositionTracker cho sàn {self.exchange.exchange_id}")
    
    def start_tracking(self, interval: int = 5) -> None:
        """
        Bắt đầu theo dõi vị thế trong một thread riêng biệt.
        
        Args:
            interval: Khoảng thời gian giữa các lần cập nhật (giây)
        """
        if self._tracker_thread is not None and self._tracker_thread.is_alive():
            self.logger.warning("Thread theo dõi vị thế đã đang chạy")
            return
        
        self._stop_tracker = False
        self._tracker_thread = threading.Thread(
            target=self._tracking_loop,
            args=(interval,),
            daemon=True
        )
        self._tracker_thread.start()
        self.logger.info(f"Đã bắt đầu theo dõi vị thế với interval={interval}s")
    
    def stop_tracking(self) -> None:
        """Dừng theo dõi vị thế."""
        if self._tracker_thread is None or not self._tracker_thread.is_alive():
            self.logger.warning("Thread theo dõi vị thế không đang chạy")
            return
        
        self._stop_tracker = True
        self._tracker_thread.join(timeout=10)
        if self._tracker_thread.is_alive():
            self.logger.warning("Không thể dừng thread theo dõi vị thế")
        else:
            self.logger.info("Đã dừng theo dõi vị thế")
            self._tracker_thread = None
    
    def _tracking_loop(self, interval: int) -> None:
        """
        Vòng lặp theo dõi vị thế trong thread riêng biệt.
        
        Args:
            interval: Khoảng thời gian giữa các lần cập nhật (giây)
        """
        self.logger.info(f"Bắt đầu vòng lặp theo dõi vị thế mỗi {interval} giây")
        
        while not self._stop_tracker:
            try:
                # Cập nhật thông tin vị thế
                current_positions = self.get_positions(force_update=True)
                
                # Kiểm tra dừng lỗ và chốt lời cho mỗi vị thế
                for symbol, position in current_positions.items():
                    # Chỉ xử lý các vị thế mở
                    if position.get('status') != PositionStatus.OPEN.value:
                        continue
                    
                    current_price = position.get('current_price', 0)
                    if current_price <= 0:
                        # Lấy giá hiện tại nếu không có
                        try:
                            ticker = self.exchange.fetch_ticker(symbol)
                            current_price = ticker['last']
                            position['current_price'] = current_price
                        except Exception as e:
                            self.logger.warning(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                            continue
                    
                    # Kiểm tra dừng lỗ
                    if self.stop_loss_manager.check_stop_loss(position, current_price):
                        self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                        try:
                            # Đóng vị thế bằng lệnh market
                            self.close_position(symbol, reason="stop_loss")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi đóng vị thế dừng lỗ {symbol}: {str(e)}")
                    
                    # Kiểm tra chốt lời
                    elif self.take_profit_manager.check_take_profit(position, current_price):
                        self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                        try:
                            # Đóng vị thế bằng lệnh market
                            self.close_position(symbol, reason="take_profit")
                        except Exception as e:
                            self.logger.error(f"Lỗi khi đóng vị thế chốt lời {symbol}: {str(e)}")
                    
                    # Cập nhật trailing stop nếu được kích hoạt
                    elif 'trailing_stop_enabled' in position and position['trailing_stop_enabled']:
                        old_stop = position.get('trailing_stop', 0)
                        updated = self.stop_loss_manager.update_trailing_stop(position, current_price)
                        
                        if updated and position.get('trailing_stop', 0) != old_stop:
                            self.logger.info(f"Đã cập nhật trailing stop cho {symbol} từ {old_stop} thành {position['trailing_stop']}")
                            
                            # Cập nhật trailing stop trên sàn nếu hỗ trợ
                            try:
                                if hasattr(self.exchange, 'update_stop_loss') and 'order_id' in position:
                                    self.exchange.update_stop_loss(symbol, position['order_id'], position['trailing_stop'])
                            except Exception as e:
                                self.logger.warning(f"Không thể cập nhật trailing stop trên sàn cho {symbol}: {str(e)}")
                
                # Cập nhật thống kê
                self.update_position_statistics()
                
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp theo dõi vị thế: {str(e)}")
            
            # Ngủ trong khoảng thời gian interval
            for _ in range(interval):
                if self._stop_tracker:
                    break
                time.sleep(1)
    
    def get_positions(self, force_update: bool = False, retries: int = None) -> Dict[str, Dict[str, Any]]:
        """
        Lấy danh sách vị thế hiện tại.
        
        Args:
            force_update: Bỏ qua cache và lấy dữ liệu mới
            retries: Số lần thử lại (None để sử dụng giá trị mặc định)
            
        Returns:
            Dict các vị thế hiện tại, với key là symbol và value là thông tin vị thế
            
        Raises:
            APIError: Nếu có lỗi khi gọi API
        """
        # Dùng số lần thử lại mặc định nếu không được chỉ định
        if retries is None:
            retries = self.max_retries
        
        current_time = time.time()
        
        # Kiểm tra cache nếu không yêu cầu cập nhật mới
        if not force_update:
            with self._positions_cache_lock:
                if (current_time - self._positions_cache_time < self.positions_cache_expiry and 
                    self._positions_cache):
                    self.logger.debug("Trả về vị thế từ cache")
                    return self._positions_cache
        
        # Kiểm tra xem sàn giao dịch có hỗ trợ fetch_positions không
        if not hasattr(self.exchange, 'fetch_positions'):
            # Nếu không hỗ trợ, thử dùng fetch_open_orders để ước tính vị thế
            self.logger.warning(f"Sàn {self.exchange.exchange_id} không hỗ trợ fetch_positions")
            return self._get_positions_from_orders()
        
        try:
            # Lấy danh sách vị thế từ sàn giao dịch
            positions_list = self.exchange.fetch_positions()
            
            # Chuyển đổi danh sách thành dict với key là symbol
            positions_dict = {}
            
            for position in positions_list:
                # Bỏ qua các vị thế có size = 0
                contracts = float(position.get('contracts', 0))
                if contracts <= 0:
                    continue
                
                symbol = position.get('symbol', '')
                
                # Chuẩn hóa thông tin vị thế
                side = 'long' if position.get('side', '').lower() == 'long' else 'short'
                entry_price = float(position.get('entryPrice', position.get('entry_price', 0)))
                current_price = float(position.get('markPrice', position.get('mark_price', 0)))
                size = float(position.get('contracts', position.get('size', 0)))
                leverage = float(position.get('leverage', 1))
                
                # Tính lợi nhuận
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                
                # Tính phần trăm lợi nhuận
                if entry_price > 0:
                    unrealized_pnl_percent = unrealized_pnl / (entry_price * size) * 100
                else:
                    unrealized_pnl_percent = 0
                
                # Cập nhật thời gian
                self.last_update_time[symbol] = datetime.now()
                
                # Lấy thông tin stop loss và take profit
                stop_loss = position.get('stopLoss', position.get('stop_loss', 0))
                take_profit = position.get('takeProfit', position.get('take_profit', 0))
                
                # Tạo dict thông tin vị thế
                position_info = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'size': size,
                    'leverage': leverage,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl_percent,
                    'status': PositionStatus.OPEN.value,
                    'last_update_time': self.last_update_time[symbol].isoformat(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'liquidation_price': position.get('liquidationPrice', position.get('liquidation_price', 0)),
                    'margin_mode': position.get('marginMode', position.get('margin_mode', 'cross')),
                    'position_value': entry_price * size,
                    'notional_value': current_price * size,
                    'creation_time': position.get('creationTime', position.get('creation_time', self.last_update_time[symbol].isoformat())),
                }
                
                # Thêm thông tin trailing stop nếu có
                if 'trailing_stop' in position:
                    position_info['trailing_stop'] = position['trailing_stop']
                    position_info['trailing_stop_enabled'] = True
                
                # Thêm vào dict kết quả
                positions_dict[symbol] = position_info
            
            # Cập nhật cache
            with self._positions_cache_lock:
                self._positions_cache = positions_dict
                self._positions_cache_time = current_time
            
            self.logger.info(f"Đã cập nhật {len(positions_dict)} vị thế trên {self.exchange.exchange_id}")
            return positions_dict
            
        except Exception as e:
            # Thử lại nếu còn lần thử
            if retries > 0:
                self.logger.warning(f"Lỗi khi lấy vị thế, thử lại ({retries}): {str(e)}")
                time.sleep(1)  # Chờ 1 giây trước khi thử lại
                return self.get_positions(force_update=True, retries=retries-1)
            
            # Nếu hết lần thử, trả về cache nếu có
            if self._positions_cache:
                self.logger.warning(f"Sử dụng cache cũ sau khi không thể lấy vị thế mới: {str(e)}")
                return self._positions_cache
            
            # Nếu không có cache, báo lỗi
            self.logger.error(f"Không thể lấy vị thế: {str(e)}")
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Không thể lấy vị thế: {str(e)}",
                exchange=self.exchange.exchange_id
            )
    
    def _get_positions_from_orders(self) -> Dict[str, Dict[str, Any]]:
        """
        Ước tính vị thế dựa trên lệnh mở.
        Sử dụng khi sàn không hỗ trợ trực tiếp fetch_positions.
        
        Returns:
            Dict các vị thế ước tính, với key là symbol và value là thông tin vị thế
        """
        try:
            # Lấy lệnh mở
            open_orders = self.exchange.fetch_open_orders()
            
            # Lấy danh sách giao dịch gần đây
            trades = {}
            try:
                recent_trades = self.exchange.fetch_my_trades(limit=100)
                # Nhóm theo symbol
                for trade in recent_trades:
                    symbol = trade.get('symbol', '')
                    if symbol not in trades:
                        trades[symbol] = []
                    trades[symbol].append(trade)
            except Exception as e:
                self.logger.warning(f"Không thể lấy lịch sử giao dịch để ước tính vị thế: {str(e)}")
            
            # Tạo dict vị thế ước tính
            positions_dict = {}
            
            # Tạo vị thế từ lệnh stop loss và take profit
            for order in open_orders:
                order_type = order.get('type', '').lower()
                if order_type in ['stop_loss', 'take_profit', 'stop', 'limit']:
                    symbol = order.get('symbol', '')
                    side = order.get('side', '').lower()
                    
                    # Xác định phía vị thế dựa trên loại lệnh và phía
                    position_side = None
                    if order_type in ['stop_loss', 'stop']:
                        position_side = 'long' if side == 'sell' else 'short'
                    elif order_type == 'take_profit':
                        position_side = 'long' if side == 'sell' else 'short'
                    elif order_type == 'limit':
                        # Lệnh limit có thể là mở hoặc đóng vị thế, cần phân tích thêm
                        if symbol in trades and trades[symbol]:
                            # Ước tính dựa trên giao dịch gần nhất
                            last_trade = trades[symbol][0]
                            last_side = last_trade.get('side', '').lower()
                            if last_side != side:
                                # Nếu phía khác với giao dịch gần nhất, có thể là lệnh đóng
                                position_side = 'long' if last_side == 'buy' else 'short'
                    
                    if position_side and symbol not in positions_dict:
                        # Ước tính giá vào và kích thước
                        entry_price = 0
                        size = 0
                        
                        if symbol in trades and trades[symbol]:
                            # Ước tính từ giao dịch gần đây
                            for trade in trades[symbol]:
                                trade_side = trade.get('side', '').lower()
                                if trade_side == ('buy' if position_side == 'long' else 'sell'):
                                    trade_price = float(trade.get('price', 0))
                                    trade_amount = float(trade.get('amount', 0))
                                    
                                    if entry_price == 0:
                                        entry_price = trade_price
                                    else:
                                        # Tính giá trung bình
                                        entry_price = (entry_price * size + trade_price * trade_amount) / (size + trade_amount)
                                    
                                    size += trade_amount
                        
                        # Lấy giá hiện tại
                        current_price = 0
                        try:
                            ticker = self.exchange.fetch_ticker(symbol)
                            current_price = ticker['last']
                        except Exception as e:
                            self.logger.warning(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                        
                        # Tính lợi nhuận ước tính
                        unrealized_pnl = 0
                        unrealized_pnl_percent = 0
                        
                        if entry_price > 0 and current_price > 0 and size > 0:
                            if position_side == 'long':
                                unrealized_pnl = (current_price - entry_price) * size
                            else:  # short
                                unrealized_pnl = (entry_price - current_price) * size
                            
                            unrealized_pnl_percent = unrealized_pnl / (entry_price * size) * 100
                        
                        # Tạo thông tin vị thế
                        position_info = {
                            'symbol': symbol,
                            'side': position_side,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'size': size,
                            'leverage': 1,  # Mặc định, không thể xác định chính xác
                            'unrealized_pnl': unrealized_pnl,
                            'unrealized_pnl_percent': unrealized_pnl_percent,
                            'status': PositionStatus.OPEN.value,
                            'last_update_time': datetime.now().isoformat(),
                            'stop_loss': order.get('price', 0) if order_type in ['stop_loss', 'stop'] else 0,
                            'take_profit': order.get('price', 0) if order_type == 'take_profit' else 0,
                            'position_value': entry_price * size,
                            'notional_value': current_price * size,
                            'creation_time': datetime.now().isoformat(),
                            'is_estimated': True  # Đánh dấu là vị thế ước tính
                        }
                        
                        positions_dict[symbol] = position_info
            
            # Cập nhật cache
            with self._positions_cache_lock:
                self._positions_cache = positions_dict
                self._positions_cache_time = time.time()
            
            self.logger.info(f"Đã ước tính {len(positions_dict)} vị thế từ lệnh mở trên {self.exchange.exchange_id}")
            return positions_dict
            
        except Exception as e:
            self.logger.error(f"Lỗi khi ước tính vị thế từ lệnh: {str(e)}")
            
            # Trả về cache nếu có
            if self._positions_cache:
                self.logger.warning(f"Sử dụng cache vị thế cũ: {str(e)}")
                return self._positions_cache
            
            # Trả về dict trống nếu không có cache
            return {}
    
    def get_position(self, symbol: str, force_update: bool = False) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin vị thế cho một symbol cụ thể.
        
        Args:
            symbol: Symbol cần lấy thông tin
            force_update: Bỏ qua cache và lấy dữ liệu mới
            
        Returns:
            Thông tin vị thế hoặc None nếu không tìm thấy
        """
        positions = self.get_positions(force_update=force_update)
        return positions.get(symbol)
    
    def open_position(self, symbol: str, side: str, size: float, 
                    entry_price: Optional[float] = None,
                    leverage: float = 1.0,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    trailing_stop: bool = False,
                    trailing_stop_percent: float = 0.01,
                    order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Mở vị thế mới hoặc cập nhật vị thế hiện có.
        
        Args:
            symbol: Symbol cần mở vị thế
            side: Phía vị thế ('long' hoặc 'short')
            size: Kích thước vị thế
            entry_price: Giá vào (None để sử dụng giá thị trường)
            leverage: Đòn bẩy
            stop_loss: Giá dừng lỗ (None để tự động tính)
            take_profit: Giá chốt lời (None để tự động tính)
            trailing_stop: Bật/tắt trailing stop
            trailing_stop_percent: Phần trăm trailing stop
            order_id: ID lệnh (nếu có)
            
        Returns:
            Dict thông tin vị thế
            
        Raises:
            ValueError: Nếu tham số không hợp lệ
        """
        # Kiểm tra tham số đầu vào
        if not is_valid_number(size, min_value=0):
            raise ValueError("Kích thước vị thế không hợp lệ")
        
        if not is_valid_number(leverage, min_value=1):
            raise ValueError("Đòn bẩy không hợp lệ")
        
        if side not in ['long', 'short']:
            raise ValueError("Phía vị thế không hợp lệ, phải là 'long' hoặc 'short'")
        
        # Lấy giá hiện tại nếu không cung cấp entry_price
        if entry_price is None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                entry_price = ticker['last']
            except Exception as e:
                self.logger.error(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                raise ValueError(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
        
        # Kiểm tra vị thế hiện có
        current_position = self.get_position(symbol)
        
        # Tính stop loss nếu không được cung cấp
        if stop_loss is None:
            if side == 'long':
                stop_loss = entry_price * (1 - 0.05)  # Mặc định 5% dưới giá vào
            else:
                stop_loss = entry_price * (1 + 0.05)  # Mặc định 5% trên giá vào
        
        # Tính take profit nếu không được cung cấp
        if take_profit is None:
            if side == 'long':
                take_profit = entry_price * (1 + 0.1)  # Mặc định 10% trên giá vào
            else:
                take_profit = entry_price * (1 - 0.1)  # Mặc định 10% dưới giá vào
        
        # Thời gian hiện tại
        current_time = datetime.now()
        
        # Tạo thông tin vị thế mới
        position_info = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': size,
            'leverage': leverage,
            'unrealized_pnl': 0,  # Ban đầu là 0
            'unrealized_pnl_percent': 0,  # Ban đầu là 0
            'status': PositionStatus.OPEN.value,
            'last_update_time': current_time.isoformat(),
            'creation_time': current_time.isoformat(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_value': entry_price * size,
            'notional_value': entry_price * size,
            'trailing_stop_enabled': trailing_stop,
            'trailing_stop_percent': trailing_stop_percent,
            'trailing_stop': stop_loss,
            'initial_stop_loss': stop_loss
        }
        
        # Thêm order_id nếu có
        if order_id:
            position_info['order_id'] = order_id
        
        # Cập nhật cache
        with self._positions_cache_lock:
            self._positions_cache[symbol] = position_info
            self._positions_cache_time = time.time()
        
        # Cập nhật thời gian cập nhật
        self.last_update_time[symbol] = current_time
        
        self.logger.info(f"Đã mở vị thế {side} cho {symbol}: {size} @ {entry_price} (SL: {stop_loss}, TP: {take_profit})")
        return position_info
    
    def update_position(self, symbol: str, current_price: Optional[float] = None, 
                      stop_loss: Optional[float] = None, 
                      take_profit: Optional[float] = None,
                      trailing_stop_enabled: Optional[bool] = None) -> Dict[str, Any]:
        """
        Cập nhật thông tin vị thế.
        
        Args:
            symbol: Symbol cần cập nhật
            current_price: Giá hiện tại mới
            stop_loss: Giá dừng lỗ mới
            take_profit: Giá chốt lời mới
            trailing_stop_enabled: Bật/tắt trailing stop
            
        Returns:
            Dict thông tin vị thế đã cập nhật
            
        Raises:
            ValueError: Nếu vị thế không tồn tại
        """
        # Lấy thông tin vị thế hiện tại
        position = self.get_position(symbol)
        
        if not position:
            raise ValueError(f"Không tìm thấy vị thế cho {symbol}")
        
        # Lấy giá hiện tại nếu không được cung cấp
        if current_price is None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
            except Exception as e:
                self.logger.warning(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                # Sử dụng giá hiện tại cũ
                current_price = position.get('current_price', 0)
        
        # Cập nhật giá hiện tại
        if current_price > 0:
            position['current_price'] = current_price
            
            # Tính lại lợi nhuận
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            side = position.get('side', 'long')
            
            if entry_price > 0 and size > 0:
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = unrealized_pnl / (entry_price * size) * 100 if entry_price > 0 else 0
                position['notional_value'] = current_price * size
        
        # Cập nhật stop loss nếu được cung cấp
        if stop_loss is not None:
            position['stop_loss'] = stop_loss
            
            # Cập nhật trailing stop nếu được bật
            if position.get('trailing_stop_enabled', False):
                position['trailing_stop'] = stop_loss
        
        # Cập nhật take profit nếu được cung cấp
        if take_profit is not None:
            position['take_profit'] = take_profit
        
        # Cập nhật trạng thái trailing stop nếu được chỉ định
        if trailing_stop_enabled is not None:
            position['trailing_stop_enabled'] = trailing_stop_enabled
            
            # Nếu bật trailing stop, khởi tạo giá trị ban đầu
            if trailing_stop_enabled and 'trailing_stop' not in position:
                position['trailing_stop'] = position.get('stop_loss', 0)
                position['initial_stop_loss'] = position.get('stop_loss', 0)
        
        # Cập nhật thời gian cập nhật
        position['last_update_time'] = datetime.now().isoformat()
        self.last_update_time[symbol] = datetime.now()
        
        # Cập nhật cache
        with self._positions_cache_lock:
            self._positions_cache[symbol] = position
            self._positions_cache_time = time.time()
        
        self.logger.info(f"Đã cập nhật vị thế cho {symbol}: Giá hiện tại={current_price}, PnL={position['unrealized_pnl']:.2f}")
        return position
    
    def close_position(self, symbol: str, exit_price: Optional[float] = None, 
                      partial_size: Optional[float] = None,
                      reason: str = "manual") -> Dict[str, Any]:
        """
        Đóng vị thế hoặc một phần vị thế.
        
        Args:
            symbol: Symbol cần đóng vị thế
            exit_price: Giá thoát (None để sử dụng giá thị trường)
            partial_size: Kích thước cần đóng (None để đóng toàn bộ)
            reason: Lý do đóng vị thế
            
        Returns:
            Dict thông tin kết quả
            
        Raises:
            ValueError: Nếu vị thế không tồn tại
        """
        # Lấy thông tin vị thế hiện tại
        position = self.get_position(symbol)
        
        if not position:
            raise ValueError(f"Không tìm thấy vị thế cho {symbol}")
        
        # Lấy giá thoát nếu không được cung cấp
        if exit_price is None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                exit_price = ticker['last']
            except Exception as e:
                self.logger.error(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                raise ValueError(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
        
        # Lấy thông tin vị thế
        entry_price = position.get('entry_price', 0)
        total_size = position.get('size', 0)
        side = position.get('side', 'long')
        
        # Sử dụng toàn bộ size nếu partial_size không được chỉ định
        if partial_size is None or partial_size >= total_size:
            size_to_close = total_size
            is_partial = False
        else:
            size_to_close = partial_size
            is_partial = True
        
        # Tính lợi nhuận đã thực hiện
        if side == 'long':
            realized_pnl = (exit_price - entry_price) * size_to_close
        else:  # short
            realized_pnl = (entry_price - exit_price) * size_to_close
        
        # Tính phần trăm lợi nhuận
        if entry_price > 0:
            realized_pnl_percent = realized_pnl / (entry_price * size_to_close) * 100
        else:
            realized_pnl_percent = 0
        
        # Thêm vào lịch sử vị thế
        position_history_entry = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size_to_close,
            'leverage': position.get('leverage', 1),
            'realized_pnl': realized_pnl,
            'realized_pnl_percent': realized_pnl_percent,
            'entry_time': position.get('creation_time', ''),
            'exit_time': datetime.now().isoformat(),
            'duration_hours': (datetime.now() - datetime.fromisoformat(position.get('creation_time', datetime.now().isoformat()))).total_seconds() / 3600,
            'reason': reason,
            'is_partial': is_partial
        }
        
        # Thêm vào lịch sử
        self.position_history.append(position_history_entry)
        
        # Cập nhật hoặc xóa vị thế khỏi cache
        with self._positions_cache_lock:
            if is_partial:
                # Cập nhật size mới
                remaining_size = total_size - size_to_close
                position['size'] = remaining_size
                position['position_value'] = entry_price * remaining_size
                position['notional_value'] = exit_price * remaining_size
                
                # Cập nhật lợi nhuận
                if side == 'long':
                    unrealized_pnl = (exit_price - entry_price) * remaining_size
                else:  # short
                    unrealized_pnl = (entry_price - exit_price) * remaining_size
                
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = unrealized_pnl / (entry_price * remaining_size) * 100 if entry_price > 0 else 0
                
                # Cập nhật thời gian
                position['last_update_time'] = datetime.now().isoformat()
                self._positions_cache[symbol] = position
                
                self.logger.info(f"Đã đóng một phần vị thế {symbol} ({size_to_close}/{total_size}): Giá={exit_price}, PnL={realized_pnl:.2f} ({realized_pnl_percent:.2f}%), Lý do={reason}")
            else:
                # Xóa vị thế khỏi cache nếu đóng toàn bộ
                if symbol in self._positions_cache:
                    del self._positions_cache[symbol]
                
                # Xóa khỏi thời gian cập nhật
                if symbol in self.last_update_time:
                    del self.last_update_time[symbol]
                
                self.logger.info(f"Đã đóng toàn bộ vị thế {symbol}: Giá={exit_price}, PnL={realized_pnl:.2f} ({realized_pnl_percent:.2f}%), Lý do={reason}")
            
            # Cập nhật thời gian cache
            self._positions_cache_time = time.time()
        
        # Kết quả
        result = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size_to_close,
            'realized_pnl': realized_pnl,
            'realized_pnl_percent': realized_pnl_percent,
            'is_partial': is_partial,
            'reason': reason,
            'remaining_size': position.get('size', 0) if is_partial else 0
        }
        
        return result
    
    def update_position_statistics(self) -> Dict[str, Any]:
        """
        Cập nhật và trả về thống kê về các vị thế.
        
        Returns:
            Dict chứa thống kê vị thế
        """
        # Lấy danh sách vị thế hiện tại
        current_positions = self.get_positions()
        
        # Khởi tạo thống kê
        stats = {
            'total_positions': len(current_positions),
            'total_long_positions': 0,
            'total_short_positions': 0,
            'total_unrealized_pnl': 0,
            'total_position_value': 0,
            'total_notional_value': 0,
            'average_leverage': 0,
            'profitable_positions': 0,
            'losing_positions': 0,
            'largest_profit_symbol': '',
            'largest_profit_amount': 0,
            'largest_loss_symbol': '',
            'largest_loss_amount': 0,
            'position_symbols': list(current_positions.keys()),
            'position_distribution': {},
            'realized_pnl_total': sum(pos.get('realized_pnl', 0) for pos in self.position_history),
            'win_count': sum(1 for pos in self.position_history if pos.get('realized_pnl', 0) > 0),
            'loss_count': sum(1 for pos in self.position_history if pos.get('realized_pnl', 0) <= 0),
            'win_rate': 0
        }
        
        # Tính win rate
        total_closed = stats['win_count'] + stats['loss_count']
        if total_closed > 0:
            stats['win_rate'] = stats['win_count'] / total_closed * 100
        
        # Nếu không có vị thế, trả về thống kê cơ bản
        if not current_positions:
            return stats
        
        # Tính thống kê chi tiết
        for symbol, position in current_positions.items():
            # Phân loại vị thế
            side = position.get('side', 'long')
            
            if side == 'long':
                stats['total_long_positions'] += 1
            else:  # short
                stats['total_short_positions'] += 1
            
            # Cộng dồn PnL và giá trị
            unrealized_pnl = position.get('unrealized_pnl', 0)
            stats['total_unrealized_pnl'] += unrealized_pnl
            stats['total_position_value'] += position.get('position_value', 0)
            stats['total_notional_value'] += position.get('notional_value', 0)
            
            # Cộng dồn đòn bẩy
            stats['average_leverage'] += position.get('leverage', 1)
            
            # Phân loại lợi nhuận/lỗ
            if unrealized_pnl > 0:
                stats['profitable_positions'] += 1
                
                # Kiểm tra lợi nhuận lớn nhất
                if unrealized_pnl > stats['largest_profit_amount']:
                    stats['largest_profit_amount'] = unrealized_pnl
                    stats['largest_profit_symbol'] = symbol
            else:
                stats['losing_positions'] += 1
                
                # Kiểm tra lỗ lớn nhất
                if unrealized_pnl < stats['largest_loss_amount']:
                    stats['largest_loss_amount'] = unrealized_pnl
                    stats['largest_loss_symbol'] = symbol
        
        # Tính trung bình đòn bẩy
        if stats['total_positions'] > 0:
            stats['average_leverage'] /= stats['total_positions']
        
        # Tính phân bố vị thế
        for symbol, position in current_positions.items():
            notional_value = position.get('notional_value', 0)
            if stats['total_notional_value'] > 0:
                stats['position_distribution'][symbol] = notional_value / stats['total_notional_value'] * 100
        
        return stats
    
    def get_position_history(self, symbol: Optional[str] = None, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           include_open: bool = False) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử vị thế theo các điều kiện lọc.
        
        Args:
            symbol: Symbol cần lọc (None để lấy tất cả)
            start_time: Thời gian bắt đầu (None để không giới hạn)
            end_time: Thời gian kết thúc (None để sử dụng thời gian hiện tại)
            include_open: Có bao gồm các vị thế đang mở không
            
        Returns:
            List các mục trong lịch sử vị thế
        """
        # Mặc định end_time là thời gian hiện tại
        if end_time is None:
            end_time = datetime.now()
        
        # Lọc lịch sử vị thế
        filtered_history = []
        
        for position in self.position_history:
            # Lọc theo symbol
            if symbol is not None and position.get('symbol') != symbol:
                continue
            
            # Lọc theo thời gian
            exit_time = datetime.fromisoformat(position.get('exit_time', datetime.now().isoformat()))
            
            if start_time is not None and exit_time < start_time:
                continue
                
            if exit_time > end_time:
                continue
            
            filtered_history.append(position)
        
        # Thêm vị thế đang mở nếu cần
        if include_open:
            current_positions = self.get_positions()
            
            for symbol, position in current_positions.items():
                # Lọc theo symbol
                if symbol is not None and symbol != symbol:
                    continue
                
                # Chuyển đổi thành định dạng giống lịch sử
                creation_time = datetime.fromisoformat(position.get('creation_time', datetime.now().isoformat()))
                
                # Lọc theo thời gian
                if start_time is not None and creation_time < start_time:
                    continue
                
                # Tạo entry cho lịch sử
                current_price = position.get('current_price', 0)
                entry_price = position.get('entry_price', 0)
                size = position.get('size', 0)
                side = position.get('side', 'long')
                
                # Tính unrealized PnL
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                
                # Tính phần trăm
                unrealized_pnl_percent = unrealized_pnl / (entry_price * size) * 100 if entry_price > 0 else 0
                
                history_entry = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'size': size,
                    'leverage': position.get('leverage', 1),
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl_percent,
                    'entry_time': position.get('creation_time', ''),
                    'duration_hours': (datetime.now() - creation_time).total_seconds() / 3600,
                    'status': 'open',
                    'is_open': True
                }
                
                filtered_history.append(history_entry)
        
        return filtered_history
    
    def analyze_position_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Phân tích lịch sử vị thế.
        
        Args:
            days: Số ngày cần phân tích
            
        Returns:
            Dict chứa kết quả phân tích
        """
        # Tính thời gian bắt đầu
        start_time = datetime.now() - timedelta(days=days)
        
        # Lấy lịch sử vị thế
        history = self.get_position_history(start_time=start_time, include_open=True)
        
        if not history:
            return {
                'message': 'Không có dữ liệu vị thế trong khoảng thời gian đã chọn',
                'total_positions': 0
            }
        
        # Chuyển thành DataFrame để phân tích
        try:
            df = pd.DataFrame(history)
            
            # Lọc vị thế đã đóng
            closed_positions = df[~df.get('is_open', False)]
            open_positions = df[df.get('is_open', False)]
            
            # Tính thống kê
            total_realized_pnl = closed_positions.get('realized_pnl', 0).sum()
            total_unrealized_pnl = open_positions.get('unrealized_pnl', 0).sum()
            
            win_count = len(closed_positions[closed_positions.get('realized_pnl', 0) > 0])
            loss_count = len(closed_positions[closed_positions.get('realized_pnl', 0) <= 0])
            total_closed = len(closed_positions)
            
            win_rate = win_count / total_closed * 100 if total_closed > 0 else 0
            
            # Tính thống kê theo symbol
            by_symbol = {}
            
            for symbol, group in df.groupby('symbol'):
                closed_group = group[~group.get('is_open', False)]
                open_group = group[group.get('is_open', False)]
                
                symbol_closed_count = len(closed_group)
                symbol_win_count = len(closed_group[closed_group.get('realized_pnl', 0) > 0])
                
                by_symbol[symbol] = {
                    'total_positions': len(group),
                    'closed_positions': symbol_closed_count,
                    'win_count': symbol_win_count,
                    'loss_count': symbol_closed_count - symbol_win_count,
                    'win_rate': symbol_win_count / symbol_closed_count * 100 if symbol_closed_count > 0 else 0,
                    'realized_pnl': closed_group.get('realized_pnl', 0).sum(),
                    'unrealized_pnl': open_group.get('unrealized_pnl', 0).sum(),
                    'total_pnl': closed_group.get('realized_pnl', 0).sum() + open_group.get('unrealized_pnl', 0).sum(),
                    'average_duration_hours': closed_group.get('duration_hours', 0).mean() if not closed_group.empty else 0
                }
            
            # Tính thống kê theo phía
            by_side = {}
            
            for side, group in df.groupby('side'):
                closed_group = group[~group.get('is_open', False)]
                open_group = group[group.get('is_open', False)]
                
                side_closed_count = len(closed_group)
                side_win_count = len(closed_group[closed_group.get('realized_pnl', 0) > 0])
                
                by_side[side] = {
                    'total_positions': len(group),
                    'closed_positions': side_closed_count,
                    'win_count': side_win_count,
                    'loss_count': side_closed_count - side_win_count,
                    'win_rate': side_win_count / side_closed_count * 100 if side_closed_count > 0 else 0,
                    'realized_pnl': closed_group.get('realized_pnl', 0).sum(),
                    'unrealized_pnl': open_group.get('unrealized_pnl', 0).sum(),
                    'total_pnl': closed_group.get('realized_pnl', 0).sum() + open_group.get('unrealized_pnl', 0).sum(),
                    'average_duration_hours': closed_group.get('duration_hours', 0).mean() if not closed_group.empty else 0
                }
            
            # Tính thống kê theo lý do đóng vị thế
            by_reason = {}
            
            if 'reason' in closed_positions.columns:
                for reason, group in closed_positions.groupby('reason'):
                    reason_count = len(group)
                    reason_win_count = len(group[group.get('realized_pnl', 0) > 0])
                    
                    by_reason[reason] = {
                        'total_positions': reason_count,
                        'win_count': reason_win_count,
                        'loss_count': reason_count - reason_win_count,
                        'win_rate': reason_win_count / reason_count * 100 if reason_count > 0 else 0,
                        'realized_pnl': group.get('realized_pnl', 0).sum(),
                        'average_pnl': group.get('realized_pnl', 0).mean(),
                        'average_duration_hours': group.get('duration_hours', 0).mean()
                    }
            
            # Tính thống kê theo ngày
            df['date'] = pd.to_datetime(df.get('exit_time', datetime.now()))
            daily_stats = {}
            
            for date, group in df.groupby(df['date'].dt.date):
                closed_group = group[~group.get('is_open', False)]
                
                date_str = date.strftime('%Y-%m-%d')
                daily_stats[date_str] = {
                    'total_positions': len(group),
                    'closed_positions': len(closed_group),
                    'realized_pnl': closed_group.get('realized_pnl', 0).sum()
                }
            
            # Kết quả phân tích
            result = {
                'total_positions': len(df),
                'closed_positions': total_closed,
                'open_positions': len(open_positions),
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'total_realized_pnl': total_realized_pnl,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_pnl': total_realized_pnl + total_unrealized_pnl,
                'average_realized_pnl': closed_positions.get('realized_pnl', 0).mean() if not closed_positions.empty else 0,
                'average_duration_hours': closed_positions.get('duration_hours', 0).mean() if not closed_positions.empty else 0,
                'max_profit': closed_positions.get('realized_pnl', 0).max() if not closed_positions.empty else 0,
                'max_loss': closed_positions.get('realized_pnl', 0).min() if not closed_positions.empty else 0,
                'by_symbol': by_symbol,
                'by_side': by_side,
                'by_reason': by_reason,
                'daily_stats': daily_stats,
                'days': days,
                'start_date': start_time.strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích lịch sử vị thế: {str(e)}")
            return {
                'error': str(e),
                'message': 'Lỗi khi phân tích lịch sử vị thế',
                'total_positions': len(history)
            }
    
    def get_unrealized_pnl(self, symbol: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Lấy lợi nhuận chưa thực hiện.
        
        Args:
            symbol: Symbol cần lấy P&L (None để lấy tất cả)
            
        Returns:
            Float P&L nếu chỉ định symbol, Dict P&L theo symbol nếu không
        """
        positions = self.get_positions()
        
        if symbol:
            position = positions.get(symbol)
            return position.get('unrealized_pnl', 0) if position else 0
        
        # Tính tổng P&L cho tất cả vị thế
        pnl_by_symbol = {sym: pos.get('unrealized_pnl', 0) for sym, pos in positions.items()}
        return pnl_by_symbol
    
    def get_total_position_value(self) -> float:
        """
        Lấy tổng giá trị vị thế.
        
        Returns:
            Tổng giá trị vị thế
        """
        positions = self.get_positions()
        return sum(pos.get('notional_value', 0) for pos in positions.values())
    
    def get_position_distribution(self) -> Dict[str, float]:
        """
        Lấy phân bố vị thế theo tỷ lệ phần trăm.
        
        Returns:
            Dict với key là symbol và value là phần trăm
        """
        positions = self.get_positions()
        total_value = sum(pos.get('notional_value', 0) for pos in positions.values())
        
        if total_value <= 0:
            return {}
        
        distribution = {
            symbol: (position.get('notional_value', 0) / total_value * 100)
            for symbol, position in positions.items()
        }
        
        return distribution
    
    def check_margin_level(self, positions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Kiểm tra mức ký quỹ và cảnh báo nếu gần bị thanh lý.
        
        Args:
            positions: Dict các vị thế (None để lấy tự động)
            
        Returns:
            Dict thông tin mức ký quỹ
        """
        if positions is None:
            positions = self.get_positions()
        
        # Tính tổng giá trị vị thế và tổng lợi nhuận
        total_position_value = sum(pos.get('notional_value', 0) for pos in positions.values())
        total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
        
        # Lấy số dư khả dụng
        available_balance = 0
        
        try:
            # Sàn futures thường có balance riêng
            if hasattr(self.exchange, 'fetch_balance'):
                balance = self.exchange.fetch_balance()
                
                # Tìm số dư USDT hoặc tương tự
                for currency in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD']:
                    if currency in balance:
                        available_balance = balance[currency].get('free', 0)
                        break
        except Exception as e:
            self.logger.warning(f"Không thể lấy số dư: {str(e)}")
        
        # Tính mức ký quỹ
        # Margin level = (Balance + Unrealized PnL) / Used Margin
        # Used Margin = Position Value / Leverage
        
        used_margin = 0
        for position in positions.values():
            position_value = position.get('notional_value', 0)
            leverage = position.get('leverage', 1)
            
            if leverage > 0:
                used_margin += position_value / leverage
        
        margin_level = float('inf')  # Mặc định nếu không có vị thế
        
        if used_margin > 0:
            margin_level = (available_balance + total_unrealized_pnl) / used_margin
        
        # Xác định mức cảnh báo
        warning_level = "safe"
        liquidation_risk = 0
        
        if margin_level < 1.1:
            warning_level = "extreme"
            liquidation_risk = 90
        elif margin_level < 1.3:
            warning_level = "high"
            liquidation_risk = 70
        elif margin_level < 1.5:
            warning_level = "medium"
            liquidation_risk = 50
        elif margin_level < 2.0:
            warning_level = "low"
            liquidation_risk = 30
        
        # Kiểm tra từng vị thế có gần bị thanh lý không
        positions_at_risk = {}
        
        for symbol, position in positions.items():
            liquidation_price = position.get('liquidation_price', 0)
            current_price = position.get('current_price', 0)
            side = position.get('side', 'long')
            
            if liquidation_price <= 0 or current_price <= 0:
                continue
            
            # Tính khoảng cách đến thanh lý
            if side == 'long':
                distance_to_liquidation = (current_price - liquidation_price) / current_price * 100
            else:  # short
                distance_to_liquidation = (liquidation_price - current_price) / current_price * 100
            
            # Xác định mức cảnh báo
            position_warning = "safe"
            
            if distance_to_liquidation < 5:
                position_warning = "extreme"
            elif distance_to_liquidation < 10:
                position_warning = "high"
            elif distance_to_liquidation < 20:
                position_warning = "medium"
            elif distance_to_liquidation < 30:
                position_warning = "low"
            
            if position_warning != "safe":
                positions_at_risk[symbol] = {
                    'liquidation_price': liquidation_price,
                    'current_price': current_price,
                    'distance_percent': distance_to_liquidation,
                    'warning_level': position_warning
                }
        
        return {
            'margin_level': margin_level,
            'warning_level': warning_level,
            'liquidation_risk': liquidation_risk,
            'available_balance': available_balance,
            'total_position_value': total_position_value,
            'used_margin': used_margin,
            'total_unrealized_pnl': total_unrealized_pnl,
            'positions_at_risk': positions_at_risk
        }
    
    def export_position_data(self, file_path: Optional[str] = None) -> str:
        """
        Xuất dữ liệu vị thế và lịch sử vào file JSON.
        
        Args:
            file_path: Đường dẫn file (None để tạo tên tự động)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"position_data_{timestamp}.json"
        
        # Lấy dữ liệu vị thế hiện tại
        current_positions = self.get_positions()
        
        # Chuẩn bị dữ liệu để xuất
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'exchange': self.exchange.exchange_id,
            'positions': current_positions,
            'position_history': self.position_history,
            'statistics': self.update_position_statistics(),
            'margin_info': self.check_margin_level(current_positions)
        }
        
        # Lưu vào file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã xuất dữ liệu vị thế vào {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất dữ liệu vị thế: {str(e)}")
            return ""
    
    def import_position_data(self, file_path: str) -> bool:
        """
        Nhập dữ liệu vị thế và lịch sử từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            True nếu nhập thành công, False nếu không
        """
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Kiểm tra tính hợp lệ của dữ liệu
            if 'positions' not in data or 'position_history' not in data:
                self.logger.error(f"File {file_path} không chứa dữ liệu vị thế hợp lệ")
                return False
            
            # Cập nhật cache vị thế
            with self._positions_cache_lock:
                self._positions_cache = data['positions']
                self._positions_cache_time = time.time()
            
            # Cập nhật lịch sử vị thế
            self.position_history = data['position_history']
            
            # Cập nhật thời gian cập nhật
            for symbol in data['positions']:
                self.last_update_time[symbol] = datetime.now()
            
            self.logger.info(f"Đã nhập dữ liệu vị thế từ {file_path}: {len(data['positions'])} vị thế, {len(data['position_history'])} lịch sử")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập dữ liệu vị thế: {str(e)}")
            return False
    
    def calculate_optimal_stop_loss(self, symbol: str, atr_period: int = 14, atr_multiplier: float = 2.0) -> Dict[str, Any]:
        """
        Tính toán mức stop loss tối ưu dựa trên ATR (Average True Range).
        
        Args:
            symbol: Symbol cần tính
            atr_period: Chu kỳ ATR
            atr_multiplier: Số lần nhân ATR
            
        Returns:
            Dict chứa thông tin stop loss
        """
        try:
            # Lấy dữ liệu OHLC
            ohlc_data = self.exchange.fetch_ohlcv(symbol, '1h', limit=atr_period+10)
            
            if not ohlc_data or len(ohlc_data) < atr_period:
                raise ValueError(f"Không đủ dữ liệu OHLC cho {symbol}")
            
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Tính ATR
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=atr_period).mean().fillna(df['tr'])
            
            # Lấy giá và ATR hiện tại
            current_close = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            # Lấy thông tin vị thế hiện tại
            position = self.get_position(symbol)
            
            if not position:
                # Tính stop loss cho cả long và short
                long_stop_loss = current_close - (current_atr * atr_multiplier)
                short_stop_loss = current_close + (current_atr * atr_multiplier)
                
                return {
                    'symbol': symbol,
                    'current_price': current_close,
                    'atr': current_atr,
                    'long_stop_loss': long_stop_loss,
                    'short_stop_loss': short_stop_loss,
                    'long_stop_distance': (current_close - long_stop_loss) / current_close * 100,
                    'short_stop_distance': (short_stop_loss - current_close) / current_close * 100
                }
            else:
                # Tính stop loss dựa trên vị thế hiện tại
                side = position.get('side', 'long')
                entry_price = position.get('entry_price', current_close)
                
                if side == 'long':
                    optimal_stop_loss = current_close - (current_atr * atr_multiplier)
                    # Đảm bảo stop loss mới không thấp hơn stop loss hiện tại (nếu đang dùng trailing stop)
                    current_stop = position.get('trailing_stop', position.get('stop_loss', 0))
                    if current_stop > 0 and optimal_stop_loss < current_stop:
                        optimal_stop_loss = current_stop
                    
                    stop_distance = (current_close - optimal_stop_loss) / current_close * 100
                    risk_percent = (entry_price - optimal_stop_loss) / entry_price * 100 if entry_price > 0 else 0
                else:  # short
                    optimal_stop_loss = current_close + (current_atr * atr_multiplier)
                    # Đảm bảo stop loss mới không cao hơn stop loss hiện tại (nếu đang dùng trailing stop)
                    current_stop = position.get('trailing_stop', position.get('stop_loss', 0))
                    if current_stop > 0 and optimal_stop_loss > current_stop:
                        optimal_stop_loss = current_stop
                    
                    stop_distance = (optimal_stop_loss - current_close) / current_close * 100
                    risk_percent = (optimal_stop_loss - entry_price) / entry_price * 100 if entry_price > 0 else 0
                
                return {
                    'symbol': symbol,
                    'side': side,
                    'current_price': current_close,
                    'entry_price': entry_price,
                    'atr': current_atr,
                    'optimal_stop_loss': optimal_stop_loss,
                    'current_stop_loss': position.get('stop_loss', 0),
                    'stop_distance': stop_distance,
                    'risk_percent': risk_percent
                }
        except Exception as e:
            self.logger.error(f"Lỗi khi tính stop loss tối ưu cho {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def calculate_optimal_take_profit(self, symbol: str, risk_reward_ratio: float = 2.0) -> Dict[str, Any]:
        """
        Tính toán mức take profit tối ưu dựa trên tỷ lệ rủi ro/lợi nhuận.
        
        Args:
            symbol: Symbol cần tính
            risk_reward_ratio: Tỷ lệ lợi nhuận/rủi ro mong muốn
            
        Returns:
            Dict chứa thông tin take profit
        """
        # Lấy thông tin vị thế
        position = self.get_position(symbol)
        
        if not position:
            self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
            return {
                'symbol': symbol,
                'error': 'Không tìm thấy vị thế'
            }
        
        try:
            # Lấy thông tin cần thiết
            side = position.get('side', 'long')
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', 0)
            current_price = position.get('current_price', 0)
            
            if entry_price <= 0 or stop_loss <= 0 or current_price <= 0:
                raise ValueError("Thiếu thông tin giá cần thiết")
            
            # Tính khoảng cách từ entry đến stop loss
            if side == 'long':
                risk_distance = entry_price - stop_loss
                # Tính take profit dựa trên tỷ lệ rủi ro/lợi nhuận
                take_profit = entry_price + (risk_distance * risk_reward_ratio)
                
                # Tính phần trăm
                risk_percent = risk_distance / entry_price * 100
                reward_percent = (take_profit - entry_price) / entry_price * 100
            else:  # short
                risk_distance = stop_loss - entry_price
                # Tính take profit dựa trên tỷ lệ rủi ro/lợi nhuận
                take_profit = entry_price - (risk_distance * risk_reward_ratio)
                
                # Tính phần trăm
                risk_percent = risk_distance / entry_price * 100
                reward_percent = (entry_price - take_profit) / entry_price * 100
            
            # Tính khoảng cách đến take profit
            if side == 'long':
                tp_distance = (take_profit - current_price) / current_price * 100
            else:
                tp_distance = (current_price - take_profit) / current_price * 100
            
            return {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'optimal_take_profit': take_profit,
                'current_take_profit': position.get('take_profit', 0),
                'risk_reward_ratio': risk_reward_ratio,
                'risk_percent': risk_percent,
                'reward_percent': reward_percent,
                'tp_distance': tp_distance
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi tính take profit tối ưu cho {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def set_stop_loss_and_take_profit(self, symbol: str, 
                                     stop_loss: Optional[float] = None,
                                     take_profit: Optional[float] = None,
                                     trailing_stop_enabled: Optional[bool] = None,
                                     trailing_stop_percent: Optional[float] = None) -> Dict[str, Any]:
        """
        Thiết lập stop loss và take profit cho vị thế.
        
        Args:
            symbol: Symbol cần thiết lập
            stop_loss: Giá dừng lỗ mới (None để giữ nguyên)
            take_profit: Giá chốt lời mới (None để giữ nguyên)
            trailing_stop_enabled: Bật/tắt trailing stop (None để giữ nguyên)
            trailing_stop_percent: Phần trăm trailing stop (None để giữ nguyên)
            
        Returns:
            Dict thông tin vị thế đã cập nhật
            
        Raises:
            ValueError: Nếu vị thế không tồn tại
        """
        # Lấy thông tin vị thế
        position = self.get_position(symbol)
        
        if not position:
            raise ValueError(f"Không tìm thấy vị thế cho {symbol}")
        
        # Lưu thông tin cũ
        old_stop_loss = position.get('stop_loss', 0)
        old_take_profit = position.get('take_profit', 0)
        old_trailing_stop_enabled = position.get('trailing_stop_enabled', False)
        
        # Cập nhật nếu được cung cấp
        changes = {}
        
        if stop_loss is not None and stop_loss != old_stop_loss:
            position['stop_loss'] = stop_loss
            changes['stop_loss'] = f"{old_stop_loss} -> {stop_loss}"
            
            # Cập nhật trailing stop nếu được bật
            if position.get('trailing_stop_enabled', False):
                position['trailing_stop'] = stop_loss
                position['initial_stop_loss'] = stop_loss
                changes['trailing_stop'] = f"{position.get('trailing_stop', old_stop_loss)} -> {stop_loss}"
            
            # Thử cập nhật trên sàn giao dịch nếu hỗ trợ
            try:
                if hasattr(self.exchange, 'create_stop_loss_order') and 'order_id' not in position:
                    # Tạo lệnh stop loss mới
                    side = 'sell' if position.get('side', 'long') == 'long' else 'buy'
                    order = self.exchange.create_stop_loss_order(symbol, side, position.get('size', 0), stop_loss)
                    position['stop_loss_order_id'] = order.get('id', '')
                    changes['stop_loss_order'] = f"Đã tạo lệnh stop loss mới với ID {position['stop_loss_order_id']}"
                elif hasattr(self.exchange, 'edit_order') and 'stop_loss_order_id' in position:
                    # Cập nhật lệnh stop loss hiện có
                    self.exchange.edit_order(position['stop_loss_order_id'], symbol, price=stop_loss)
                    changes['stop_loss_order'] = f"Đã cập nhật lệnh stop loss với ID {position['stop_loss_order_id']}"
            except Exception as e:
                self.logger.warning(f"Không thể cập nhật lệnh stop loss trên sàn cho {symbol}: {str(e)}")
        
        if take_profit is not None and take_profit != old_take_profit:
            position['take_profit'] = take_profit
            changes['take_profit'] = f"{old_take_profit} -> {take_profit}"
            
            # Thử cập nhật trên sàn giao dịch nếu hỗ trợ
            try:
                if hasattr(self.exchange, 'create_take_profit_order') and 'take_profit_order_id' not in position:
                    # Tạo lệnh take profit mới
                    side = 'sell' if position.get('side', 'long') == 'long' else 'buy'
                    order = self.exchange.create_take_profit_order(symbol, side, position.get('size', 0), take_profit)
                    position['take_profit_order_id'] = order.get('id', '')
                    changes['take_profit_order'] = f"Đã tạo lệnh take profit mới với ID {position['take_profit_order_id']}"
                elif hasattr(self.exchange, 'edit_order') and 'take_profit_order_id' in position:
                    # Cập nhật lệnh take profit hiện có
                    self.exchange.edit_order(position['take_profit_order_id'], symbol, price=take_profit)
                    changes['take_profit_order'] = f"Đã cập nhật lệnh take profit với ID {position['take_profit_order_id']}"
            except Exception as e:
                self.logger.warning(f"Không thể cập nhật lệnh take profit trên sàn cho {symbol}: {str(e)}")
        
        if trailing_stop_enabled is not None and trailing_stop_enabled != old_trailing_stop_enabled:
            position['trailing_stop_enabled'] = trailing_stop_enabled
            changes['trailing_stop_enabled'] = f"{old_trailing_stop_enabled} -> {trailing_stop_enabled}"
            
            # Nếu bật trailing stop, khởi tạo giá trị ban đầu
            if trailing_stop_enabled:
                position['trailing_stop'] = position.get('stop_loss', 0)
                position['initial_stop_loss'] = position.get('stop_loss', 0)
                
                # Thử cập nhật trên sàn giao dịch nếu hỗ trợ
                try:
                    if hasattr(self.exchange, 'create_trailing_stop_order') and 'trailing_stop_order_id' not in position:
                        # Tạo lệnh trailing stop mới
                        side = 'sell' if position.get('side', 'long') == 'long' else 'buy'
                        
                        # Sử dụng trailing_stop_percent nếu được cung cấp, mặc định là 1%
                        percent = trailing_stop_percent or position.get('trailing_stop_percent', 1.0)
                        
                        order = self.exchange.create_trailing_stop_order(symbol, side, position.get('size', 0), percent)
                        position['trailing_stop_order_id'] = order.get('id', '')
                        changes['trailing_stop_order'] = f"Đã tạo lệnh trailing stop mới với ID {position['trailing_stop_order_id']}"
                except Exception as e:
                    self.logger.warning(f"Không thể cập nhật lệnh trailing stop trên sàn cho {symbol}: {str(e)}")
        
        if trailing_stop_percent is not None:
            position['trailing_stop_percent'] = trailing_stop_percent
            changes['trailing_stop_percent'] = f"{position.get('trailing_stop_percent', 1.0)} -> {trailing_stop_percent}"
        
        # Cập nhật thời gian
        position['last_update_time'] = datetime.now().isoformat()
        self.last_update_time[symbol] = datetime.now()
        
        # Cập nhật cache
        with self._positions_cache_lock:
            self._positions_cache[symbol] = position
            self._positions_cache_time = time.time()
        
        # Log thông tin thay đổi
        if changes:
            changes_str = ", ".join([f"{k}: {v}" for k, v in changes.items()])
            self.logger.info(f"Đã cập nhật SL/TP cho {symbol}: {changes_str}")
        
        return position
    
    def calculate_position_risk(self, symbol: str, account_balance: Optional[float] = None) -> Dict[str, Any]:
        """
        Tính toán mức độ rủi ro của vị thế.
        
        Args:
            symbol: Symbol cần tính
            account_balance: Số dư tài khoản (None để tự động lấy)
            
        Returns:
            Dict chứa thông tin rủi ro
        """
        # Lấy thông tin vị thế
        position = self.get_position(symbol)
        
        if not position:
            return {
                'symbol': symbol,
                'error': 'Không tìm thấy vị thế'
            }
        
        try:
            # Lấy số dư tài khoản nếu không được cung cấp
            if account_balance is None:
                try:
                    # Sàn futures thường có balance riêng
                    if hasattr(self.exchange, 'fetch_balance'):
                        balance = self.exchange.fetch_balance()
                        
                        # Tìm số dư USDT hoặc tương tự
                        for currency in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD']:
                            if currency in balance:
                                account_balance = balance[currency].get('total', 0)
                                break
                except Exception as e:
                    self.logger.warning(f"Không thể lấy số dư: {str(e)}")
                    # Nếu không lấy được, sử dụng giá trị giả định
                    account_balance = 1000.0
            
            # Lấy thông tin cần thiết
            side = position.get('side', 'long')
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', 0)
            stop_loss = position.get('stop_loss', 0)
            size = position.get('size', 0)
            leverage = position.get('leverage', 1)
            
            # Tính giá trị vị thế
            position_value = entry_price * size
            notional_value = current_price * size
            
            # Tính rủi ro khi dừng lỗ
            if stop_loss > 0:
                if side == 'long':
                    risk_amount = (entry_price - stop_loss) * size
                    risk_percent = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
                else:  # short
                    risk_amount = (stop_loss - entry_price) * size
                    risk_percent = (stop_loss - entry_price) / entry_price * 100 if entry_price > 0 else 0
            else:
                # Không có stop loss, giả định rủi ro 100%
                risk_amount = position_value
                risk_percent = 100
            
            # Tính phần trăm rủi ro so với số dư
            account_risk_percent = risk_amount / account_balance * 100 if account_balance > 0 else 0
            
            # Tính R multiple (lợi nhuận hiện tại / rủi ro)
            if risk_amount > 0:
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                
                r_multiple = unrealized_pnl / risk_amount
            else:
                r_multiple = 0
            
            # Tính mức độ rủi ro tổng thể
            # Các yếu tố: account_risk_percent, leverage, size/account_balance, volatility
            # Đơn giản hóa: dựa chủ yếu vào account_risk_percent và leverage
            
            risk_level = "low"
            
            if account_risk_percent > 5:
                risk_level = "extreme"
            elif account_risk_percent > 3:
                risk_level = "high"
            elif account_risk_percent > 1:
                risk_level = "medium"
            
            # Tăng mức độ rủi ro nếu leverage cao
            if leverage > 50:
                risk_level = "extreme"
            elif leverage > 20 and risk_level != "extreme":
                risk_level = "high"
            elif leverage > 10 and risk_level == "low":
                risk_level = "medium"
            
            return {
                'symbol': symbol,
                'side': side,
                'position_value': position_value,
                'notional_value': notional_value,
                'account_balance': account_balance,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'account_risk_percent': account_risk_percent,
                'leverage': leverage,
                'r_multiple': r_multiple,
                'risk_level': risk_level,
                'size_to_balance_ratio': (position_value / account_balance * 100) if account_balance > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi tính rủi ro cho {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def get_position_summary(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Lấy tóm tắt thông tin vị thế.
        
        Args:
            symbol: Symbol cần lấy (None để lấy tất cả)
            
        Returns:
            Dict hoặc List chứa tóm tắt thông tin vị thế
        """
        positions = self.get_positions()
        
        if symbol:
            # Lấy thông tin cho một symbol cụ thể
            position = positions.get(symbol)
            
            if not position:
                return {
                    'symbol': symbol,
                    'error': 'Không tìm thấy vị thế'
                }
            
            # Tính thêm một số thông tin
            entry_price = position.get('entry_price', 0)
            current_price = position.get('current_price', 0)
            side = position.get('side', 'long')
            size = position.get('size', 0)
            leverage = position.get('leverage', 1)
            
            # Tính toán PnL
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) * size
                unrealized_pnl_percent = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            else:  # short
                unrealized_pnl = (entry_price - current_price) * size
                unrealized_pnl_percent = (entry_price - current_price) / entry_price * 100 if entry_price > 0 else 0
            
            # Tính khoảng cách đến stop loss và take profit
            stop_loss = position.get('stop_loss', 0)
            take_profit = position.get('take_profit', 0)
            
            if side == 'long' and current_price > 0:
                sl_distance_percent = (current_price - stop_loss) / current_price * 100 if stop_loss > 0 else None
                tp_distance_percent = (take_profit - current_price) / current_price * 100 if take_profit > 0 else None
            elif side == 'short' and current_price > 0:
                sl_distance_percent = (stop_loss - current_price) / current_price * 100 if stop_loss > 0 else None
                tp_distance_percent = (current_price - take_profit) / current_price * 100 if take_profit > 0 else None
            else:
                sl_distance_percent = None
                tp_distance_percent = None
            
            # Tính thời gian nắm giữ
            creation_time = datetime.fromisoformat(position.get('creation_time', datetime.now().isoformat()))
            hold_time = (datetime.now() - creation_time).total_seconds() / 3600  # Giờ
            
            # Trả về thông tin tóm tắt
            return {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': current_price,
                'size': size,
                'leverage': leverage,
                'position_value': entry_price * size,
                'notional_value': current_price * size,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_percent': unrealized_pnl_percent,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_distance_percent': sl_distance_percent,
                'tp_distance_percent': tp_distance_percent,
                'trailing_stop_enabled': position.get('trailing_stop_enabled', False),
                'trailing_stop': position.get('trailing_stop', 0),
                'creation_time': position.get('creation_time', ''),
                'hold_time_hours': hold_time,
                'hold_time_days': hold_time / 24,
                'status': position.get('status', 'open')
            }
        else:
            # Lấy tóm tắt cho tất cả các vị thế
            summaries = []
            
            for pos_symbol in positions:
                summary = self.get_position_summary(pos_symbol)
                summaries.append(summary)
            
            return summaries