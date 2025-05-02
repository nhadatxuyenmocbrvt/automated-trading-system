"""
Quản lý lệnh cho mô phỏng sàn giao dịch.
File này định nghĩa lớp OrderManager để xử lý các lệnh giao dịch 
trong môi trường mô phỏng, bao gồm tạo, cập nhật, và hủy lệnh.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import uuid
import time

from config.logging_config import get_logger
from config.constants import OrderType, OrderStatus, PositionSide, TimeInForce, ErrorCode

class OrderManager:
    """
    Lớp quản lý lệnh giao dịch.
    Cung cấp các phương thức để tạo, cập nhật, và theo dõi lệnh giao dịch.
    """
    
    def __init__(
        self,
        min_order_size: float = 0.001,
        max_orders: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý lệnh.
        
        Args:
            min_order_size: Kích thước lệnh tối thiểu
            max_orders: Số lượng lệnh tối đa
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("order_manager")
        self.min_order_size = min_order_size
        self.max_orders = max_orders
        
        # Khởi tạo danh sách lệnh
        self.orders = {}  # order_id -> order_info
        self.order_history = []
        
        self.logger.info(f"Đã khởi tạo OrderManager với min_order_size={min_order_size}, max_orders={max_orders}")
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái quản lý lệnh.
        """
        self.orders = {}
        self.order_history = []
        self.logger.info("Đã đặt lại OrderManager")
    
    def create_order(
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
        Tạo một lệnh giao dịch mới.
        
        Args:
            symbol: Cặp giao dịch
            order_type: Loại lệnh
            side: Phía lệnh (buy/sell)
            amount: Khối lượng
            price: Giá (cho lệnh limit)
            time_in_force: Hiệu lực thời gian
            params: Tham số bổ sung
            
        Returns:
            Dict chứa thông tin lệnh đã tạo
        """
        # Kiểm tra số lượng lệnh hiện có
        if len(self.orders) >= self.max_orders:
            self.logger.warning(f"Vượt quá số lượng lệnh tối đa ({self.max_orders})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.ORDER_REJECTED.value,
                    "message": f"Vượt quá số lượng lệnh tối đa ({self.max_orders})"
                }
            }
        
        # Kiểm tra kích thước lệnh tối thiểu
        if amount < self.min_order_size:
            self.logger.warning(f"Kích thước lệnh ({amount}) nhỏ hơn kích thước tối thiểu ({self.min_order_size})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Kích thước lệnh nhỏ hơn kích thước tối thiểu"
                }
            }
        
        # Chuẩn hóa các tham số
        order_type = order_type.lower()
        side = side.lower()
        
        # Kiểm tra loại lệnh
        if order_type not in [ot.value for ot in OrderType]:
            self.logger.warning(f"Loại lệnh không hợp lệ: {order_type}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Loại lệnh không hợp lệ: {order_type}"
                }
            }
        
        # Kiểm tra phía lệnh
        if side not in ["buy", "sell"]:
            self.logger.warning(f"Phía lệnh không hợp lệ: {side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Phía lệnh không hợp lệ: {side}"
                }
            }
        
        # Kiểm tra giá cho lệnh limit
        if order_type == OrderType.LIMIT.value and price is None:
            self.logger.warning("Lệnh limit phải có giá")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": "Lệnh limit phải có giá"
                }
            }
        
        # Tạo ID lệnh
        order_id = str(uuid.uuid4())
        
        # Lấy thời gian hiện tại
        timestamp = int(time.time() * 1000)
        
        # Tạo thông tin lệnh
        order = {
            "id": order_id,
            "client_order_id": params.get("client_order_id") if params else None,
            "timestamp": timestamp,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "time_in_force": time_in_force,
            "status": OrderStatus.PENDING.value,
            "filled": 0.0,
            "remaining": amount,
            "cost": 0.0,
            "fee": 0.0,
            "trades": [],
            "params": params
        }
        
        # Lưu lệnh
        self.orders[order_id] = order
        
        self.logger.info(f"Đã tạo lệnh {order_id}: {side} {amount} {symbol} @ {price if price else 'market'}")
        
        return {
            "status": "success",
            "order": order
        }
    
    def update_order(self, order_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật thông tin của một lệnh.
        
        Args:
            order_id: ID lệnh
            update_data: Dữ liệu cập nhật
            
        Returns:
            Dict chứa thông tin lệnh sau khi cập nhật
        """
        # Kiểm tra lệnh tồn tại
        if order_id not in self.orders:
            self.logger.warning(f"Không tìm thấy lệnh {order_id}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.ORDER_REJECTED.value,
                    "message": f"Không tìm thấy lệnh"
                }
            }
        
        # Lấy lệnh
        order = self.orders[order_id]
        
        # Kiểm tra trạng thái lệnh
        if order["status"] not in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]:
            self.logger.warning(f"Không thể cập nhật lệnh với trạng thái {order['status']}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Không thể cập nhật lệnh với trạng thái {order['status']}"
                }
            }
        
        # Cập nhật các trường
        for key, value in update_data.items():
            if key in ["status", "filled", "remaining", "cost", "fee", "trades"]:
                order[key] = value
        
        # Kiểm tra nếu lệnh đã hoàn thành hoặc hủy
        if order["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value]:
            # Chuyển lệnh vào lịch sử
            self.order_history.append(order)
            # Xóa khỏi danh sách lệnh hiện tại
            del self.orders[order_id]
            
            self.logger.info(f"Lệnh {order_id} đã hoàn thành với trạng thái {order['status']}")
        
        return {
            "status": "success",
            "order": order
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Hủy một lệnh giao dịch.
        
        Args:
            order_id: ID lệnh cần hủy
            
        Returns:
            Dict chứa kết quả hủy lệnh
        """
        # Kiểm tra lệnh tồn tại
        if order_id not in self.orders:
            self.logger.warning(f"Không tìm thấy lệnh {order_id}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.ORDER_REJECTED.value,
                    "message": f"Không tìm thấy lệnh"
                }
            }
        
        # Lấy lệnh
        order = self.orders[order_id]
        
        # Kiểm tra trạng thái lệnh
        if order["status"] not in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]:
            self.logger.warning(f"Không thể hủy lệnh với trạng thái {order['status']}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Không thể hủy lệnh với trạng thái {order['status']}"
                }
            }
        
        # Cập nhật trạng thái
        order["status"] = OrderStatus.CANCELED.value
        
        # Chuyển lệnh vào lịch sử
        self.order_history.append(order)
        
        # Xóa khỏi danh sách lệnh hiện tại
        del self.orders[order_id]
        
        self.logger.info(f"Đã hủy lệnh {order_id}")
        
        return {
            "status": "success",
            "order": order
        }
    
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
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử lệnh.
        
        Args:
            symbol: Lọc theo cặp giao dịch
            limit: Số lượng lệnh tối đa trả về
            
        Returns:
            Danh sách lệnh trong lịch sử
        """
        # Lọc theo symbol
        if symbol is not None:
            filtered_history = [order for order in self.order_history if order["symbol"] == symbol]
        else:
            filtered_history = self.order_history.copy()
        
        # Sắp xếp theo thời gian mới nhất
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Giới hạn số lượng
        return filtered_history[:limit]
    
    def match_order(
        self, 
        order_id: str, 
        price: float, 
        amount: float, 
        timestamp: Optional[int] = None,
        fee: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Khớp một lệnh với giá và khối lượng cụ thể.
        
        Args:
            order_id: ID lệnh
            price: Giá khớp lệnh
            amount: Khối lượng khớp
            timestamp: Thời gian khớp lệnh
            fee: Phí giao dịch
            
        Returns:
            Dict chứa kết quả khớp lệnh
        """
        # Kiểm tra lệnh tồn tại
        if order_id not in self.orders:
            self.logger.warning(f"Không tìm thấy lệnh {order_id}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.ORDER_REJECTED.value,
                    "message": f"Không tìm thấy lệnh"
                }
            }
        
        # Lấy lệnh
        order = self.orders[order_id]
        
        # Kiểm tra khối lượng còn lại
        if amount > order["remaining"]:
            self.logger.warning(f"Khối lượng khớp ({amount}) lớn hơn khối lượng còn lại ({order['remaining']})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_ORDER_PARAMS.value,
                    "message": f"Khối lượng khớp lớn hơn khối lượng còn lại"
                }
            }
        
        # Tạo thông tin giao dịch
        trade = {
            "order_id": order_id,
            "timestamp": timestamp or int(time.time() * 1000),
            "symbol": order["symbol"],
            "side": order["side"],
            "price": price,
            "amount": amount,
            "cost": price * amount,
            "fee": fee or 0.0
        }
        
        # Cập nhật thông tin lệnh
        order["filled"] += amount
        order["remaining"] -= amount
        order["cost"] += trade["cost"]
        order["fee"] += trade["fee"]
        order["trades"].append(trade)
        
        # Cập nhật trạng thái lệnh
        if order["remaining"] <= 0:
            order["status"] = OrderStatus.FILLED.value
        else:
            order["status"] = OrderStatus.PARTIALLY_FILLED.value
        
        # Kiểm tra nếu lệnh đã hoàn thành
        if order["status"] == OrderStatus.FILLED.value:
            # Chuyển lệnh vào lịch sử
            self.order_history.append(order)
            # Xóa khỏi danh sách lệnh hiện tại
            del self.orders[order_id]
            
            self.logger.info(f"Lệnh {order_id} đã khớp hoàn toàn: {order['filled']} @ {price}")
        else:
            self.logger.info(f"Lệnh {order_id} đã khớp một phần: {amount} @ {price}, còn lại: {order['remaining']}")
        
        return {
            "status": "success",
            "order": order,
            "trade": trade
        }