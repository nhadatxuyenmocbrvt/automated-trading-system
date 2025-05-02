"""
Lớp cơ sở cho mô phỏng sàn giao dịch.
File này định nghĩa lớp cơ sở BaseExchangeSimulator cung cấp các phương thức
chung và giao diện thống nhất cho tất cả các lớp mô phỏng sàn giao dịch cụ thể.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
import json
import time
import uuid
import numpy as np

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import OrderType, OrderStatus, PositionSide, TimeInForce, ErrorCode
from config.system_config import get_system_config
from environments.simulators.market_simulator import MarketSimulator

class BaseExchangeSimulator:
    """
    Lớp cơ sở cho mô phỏng sàn giao dịch.
    Định nghĩa giao diện chung và các phương thức cơ bản mà tất cả
    các lớp mô phỏng cụ thể phải triển khai.
    """
    
    def __init__(
        self,
        market_simulator: Optional[MarketSimulator] = None,
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        leverage: float = 1.0,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        min_order_size: float = 0.001,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng sàn giao dịch cơ bản.
        
        Args:
            market_simulator: Mô phỏng thị trường
            initial_balance: Số dư ban đầu theo loại tiền
            leverage: Đòn bẩy mặc định
            maker_fee: Phí maker (%)
            taker_fee: Phí taker (%)
            min_order_size: Kích thước lệnh tối thiểu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("exchange_simulator")
        
        # Lưu trữ các tham số
        self.market_simulator = market_simulator
        self.initial_balance = initial_balance.copy()
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.min_order_size = min_order_size
        
        # Lấy cấu hình hệ thống
        self.config = get_system_config()
        
        # Khởi tạo trạng thái
        self.reset()
        
        self.logger.info(f"Đã khởi tạo BaseExchangeSimulator với số dư {initial_balance}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Đặt lại trạng thái mô phỏng sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái sàn giao dịch sau khi đặt lại
        """
        # Đặt lại số dư
        self.balance = self.initial_balance.copy()
        
        # Đặt lại danh sách lệnh và vị thế
        self.orders = {}  # order_id -> order_info
        self.positions = {}  # symbol -> position_info
        self.order_history = []
        self.trade_history = []
        
        # Đặt lại thời gian
        self.current_timestamp = datetime.now().timestamp() * 1000  # milliseconds
        
        # Đặt lại thống kê
        self.stats = {
            "total_trades": 0,
            "profitable_trades": 0,
            "loss_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": sum(self.balance.values()),
        }
        
        self.logger.info("Đã đặt lại trạng thái BaseExchangeSimulator")
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        return {
            "balance": self.balance.copy(),
            "positions": self.positions.copy(),
            "orders": self.orders.copy(),
            "timestamp": self.current_timestamp,
            "stats": self.stats.copy()
        }
    
    def set_market_simulator(self, market_simulator: MarketSimulator) -> None:
        """
        Thiết lập mô phỏng thị trường để sử dụng.
        
        Args:
            market_simulator: Mô phỏng thị trường
        """
        self.market_simulator = market_simulator
        self.logger.info(f"Đã thiết lập MarketSimulator cho {market_simulator.symbol}")
    
    def update_time(self, timestamp: Optional[float] = None) -> None:
        """
        Cập nhật thời gian hiện tại của sàn giao dịch.
        
        Args:
            timestamp: Thời gian mới (milliseconds)
        """
        if timestamp is not None:
            self.current_timestamp = timestamp
        else:
            self.current_timestamp = datetime.now().timestamp() * 1000
    
    def step(self) -> Dict[str, Any]:
        """
        Tiến hành một bước mô phỏng.
        Cần được triển khai bởi các lớp con.
        
        Returns:
            Dict chứa thông tin trạng thái sau bước mô phỏng
        """
        raise NotImplementedError("Phương thức step() cần được triển khai bởi lớp con")
    
    def submit_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        time_in_force: str = TimeInForce.GTC.value,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gửi một lệnh giao dịch mới.
        Cần được triển khai bởi các lớp con.
        
        Args:
            symbol: Cặp giao dịch
            order_type: Loại lệnh
            side: Phía lệnh (buy/sell)
            amount: Khối lượng
            price: Giá (cho lệnh limit)
            time_in_force: Hiệu lực thời gian
            params: Tham số bổ sung
            
        Returns:
            Dict chứa thông tin lệnh đã gửi
        """
        raise NotImplementedError("Phương thức submit_order() cần được triển khai bởi lớp con")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Hủy một lệnh giao dịch.
        Cần được triển khai bởi các lớp con.
        
        Args:
            order_id: ID lệnh cần hủy
            
        Returns:
            Dict chứa kết quả hủy lệnh
        """
        raise NotImplementedError("Phương thức cancel_order() cần được triển khai bởi lớp con")
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin của một lệnh.
        
        Args:
            order_id: ID lệnh
            
        Returns:
            Dict chứa thông tin lệnh hoặc None nếu không tìm thấy
        """
        return self.orders.get(order_id)
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các lệnh theo điều kiện.
        
        Args:
            symbol: Lọc theo cặp giao dịch
            status: Lọc theo trạng thái
            
        Returns:
            Danh sách các lệnh thỏa mãn điều kiện
        """
        orders = list(self.orders.values())
        
        # Lọc theo symbol
        if symbol is not None:
            orders = [order for order in orders if order["symbol"] == symbol]
        
        # Lọc theo status
        if status is not None:
            orders = [order for order in orders if order["status"] == status]
        
        return orders
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Lấy thông tin vị thế cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Dict chứa thông tin vị thế hoặc None nếu không có vị thế
        """
        return self.positions.get(symbol)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả các vị thế hiện tại.
        
        Returns:
            Danh sách các vị thế
        """
        return list(self.positions.values())
    
    def get_balance(self, currency: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Lấy số dư tài khoản.
        
        Args:
            currency: Loại tiền cần lấy số dư (None để lấy tất cả)
            
        Returns:
            Số dư của loại tiền hoặc dict chứa tất cả số dư
        """
        if currency is not None:
            return self.balance.get(currency, 0.0)
        return self.balance.copy()
    
    def calculate_equity(self) -> float:
        """
        Tính tổng giá trị tài sản (equity) bao gồm số dư và vị thế mở.
        
        Returns:
            Tổng giá trị tài sản
        """
        # Tính tổng giá trị từ số dư
        equity = sum(self.balance.values())
        
        # Cộng thêm giá trị từ các vị thế
        # Các lớp con cần triển khai chi tiết
        return equity
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Lưu trạng thái mô phỏng sàn giao dịch.
        
        Args:
            path: Đường dẫn file
        """
        state = {
            "balance": self.balance,
            "positions": self.positions,
            "orders": self.orders,
            "order_history": self.order_history,
            "trade_history": self.trade_history,
            "current_timestamp": self.current_timestamp,
            "stats": self.stats
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
        
        self.logger.info(f"Đã lưu trạng thái sàn giao dịch vào {path}")
    
    def load_state(self, path: Union[str, Path]) -> bool:
        """
        Tải trạng thái mô phỏng sàn giao dịch.
        
        Args:
            path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            with open(path, 'r') as f:
                state = json.dump(f)
            
            self.balance = state.get("balance", self.initial_balance.copy())
            self.positions = state.get("positions", {})
            self.orders = state.get("orders", {})
            self.order_history = state.get("order_history", [])
            self.trade_history = state.get("trade_history", [])
            self.current_timestamp = state.get("current_timestamp", datetime.now().timestamp() * 1000)
            self.stats = state.get("stats", {})
            
            self.logger.info(f"Đã tải trạng thái sàn giao dịch từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái: {e}")
            return False
    
    def _generate_order_id(self) -> str:
        """
        Tạo ID lệnh duy nhất.
        
        Returns:
            ID lệnh duy nhất
        """
        return str(uuid.uuid4())
    
    def _update_stats(self, trade_info: Dict[str, Any]) -> None:
        """
        Cập nhật thống kê dựa trên thông tin giao dịch.
        
        Args:
            trade_info: Thông tin giao dịch
        """
        # Cập nhật số lượng giao dịch
        self.stats["total_trades"] += 1
        
        # Cập nhật khối lượng giao dịch
        self.stats["total_volume"] += trade_info.get("amount", 0.0) * trade_info.get("price", 0.0)
        
        # Cập nhật phí giao dịch
        self.stats["total_fees"] += trade_info.get("fee", 0.0)
        
        # Cập nhật số lượng giao dịch có lãi/lỗ
        if trade_info.get("realized_pnl", 0.0) > 0:
            self.stats["profitable_trades"] += 1
        elif trade_info.get("realized_pnl", 0.0) < 0:
            self.stats["loss_trades"] += 1
        
        # Cập nhật peak balance và drawdown
        current_balance = sum(self.balance.values())
        if current_balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = current_balance
        else:
            drawdown = (self.stats["peak_balance"] - current_balance) / self.stats["peak_balance"]
            if drawdown > self.stats["max_drawdown"]:
                self.stats["max_drawdown"] = drawdown