"""
Quản lý vị thế cho mô phỏng sàn giao dịch.
File này định nghĩa lớp PositionManager để theo dõi và tính toán
các vị thế trong môi trường mô phỏng, bao gồm tính toán P&L, đòn bẩy, và margin.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import math

from config.logging_config import get_logger
from config.constants import PositionSide, OrderStatus, ErrorCode

class PositionManager:
    """
    Lớp quản lý vị thế giao dịch.
    Cung cấp các phương thức để theo dõi, cập nhật và tính toán P&L cho các vị thế.
    """
    
    def __init__(
        self,
        max_positions: int = 10,
        max_leverage: float = 10.0,
        liquidation_threshold: float = 0.8,  # Margin ratio để thanh lý vị thế (0.8 = 80%)
        initial_margin_ratio: float = 0.1,   # Tỷ lệ margin ban đầu (0.1 = 10%)
        maintenance_margin_ratio: float = 0.05,  # Tỷ lệ margin duy trì (0.05 = 5%)
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý vị thế.
        
        Args:
            max_positions: Số lượng vị thế tối đa
            max_leverage: Đòn bẩy tối đa
            liquidation_threshold: Ngưỡng thanh lý
            initial_margin_ratio: Tỷ lệ margin ban đầu
            maintenance_margin_ratio: Tỷ lệ margin duy trì
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("position_manager")
        self.max_positions = max_positions
        self.max_leverage = max_leverage
        self.liquidation_threshold = liquidation_threshold
        self.initial_margin_ratio = initial_margin_ratio
        self.maintenance_margin_ratio = maintenance_margin_ratio
        
        # Khởi tạo danh sách vị thế
        self.positions = {}  # symbol -> position_info
        self.position_history = []
        
        self.logger.info(f"Đã khởi tạo PositionManager với max_positions={max_positions}, max_leverage={max_leverage}")
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái quản lý vị thế.
        """
        self.positions = {}
        self.position_history = []
        self.logger.info("Đã đặt lại PositionManager")
    
    def create_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Tạo một vị thế mới.
        
        Args:
            symbol: Cặp giao dịch
            side: Phía vị thế (long/short)
            amount: Khối lượng
            entry_price: Giá vào
            leverage: Đòn bẩy
            
        Returns:
            Dict chứa thông tin vị thế đã tạo
        """
        # Kiểm tra số lượng vị thế hiện có
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            self.logger.warning(f"Vượt quá số lượng vị thế tối đa ({self.max_positions})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.POSITION_NOT_FOUND.value,
                    "message": f"Vượt quá số lượng vị thế tối đa ({self.max_positions})"
                }
            }
        
        # Kiểm tra đòn bẩy
        if leverage > self.max_leverage:
            self.logger.warning(f"Đòn bẩy ({leverage}) vượt quá đòn bẩy tối đa ({self.max_leverage})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Đòn bẩy vượt quá giới hạn"
                }
            }
        
        # Chuẩn hóa các tham số
        side = side.lower()
        
        # Kiểm tra phía vị thế
        if side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Phía vị thế không hợp lệ: {side}"
                }
            }
        
        # Nếu đã có vị thế cho symbol, cập nhật vị thế
        if symbol in self.positions:
            return self.update_position(symbol, amount, entry_price, side)
        
        # Tạo vị thế mới
        timestamp = int(time.time() * 1000)
        
        position = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "leverage": leverage,
            "liquidation_price": self._calculate_liquidation_price(side, entry_price, leverage),
            "margin": (amount * entry_price) / leverage,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "timestamp": timestamp,
            "status": "open",
            "trades": []
        }
        
        # Lưu vị thế
        self.positions[symbol] = position
        
        self.logger.info(f"Đã tạo vị thế {symbol}: {side} {amount} @ {entry_price} (x{leverage})")
        
        return {
            "status": "success",
            "position": position
        }
    
    def update_position(
        self,
        symbol: str,
        amount: float,
        price: float,
        side: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật một vị thế hiện có.
        
        Args:
            symbol: Cặp giao dịch
            amount: Khối lượng
            price: Giá hiện tại
            side: Phía vị thế (long/short)
            
        Returns:
            Dict chứa thông tin vị thế sau khi cập nhật
        """
        # Kiểm tra vị thế tồn tại
        if symbol not in self.positions:
            self.logger.warning(f"Không tìm thấy vị thế {symbol}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.POSITION_NOT_FOUND.value,
                    "message": f"Không tìm thấy vị thế"
                }
            }
        
        # Lấy vị thế
        position = self.positions[symbol]
        
        # Chuẩn hóa side nếu được cung cấp
        if side is not None:
            side = side.lower()
            
            # Kiểm tra phía vị thế
            if side not in ["long", "short"]:
                self.logger.warning(f"Phía vị thế không hợp lệ: {side}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Phía vị thế không hợp lệ: {side}"
                    }
                }
        else:
            side = position["side"]
        
        # Lấy thời gian hiện tại
        timestamp = int(time.time() * 1000)
        
        # Tạo thông tin giao dịch
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "price": price,
            "amount": amount
        }
        
        # Tính P&L đã thực hiện nếu đóng hoặc đảo chiều vị thế
        realized_pnl = 0.0
        
        # Nếu side khác với vị thế hiện tại, tính P&L đã thực hiện
        if side != position["side"]:
            # Tính P&L đã thực hiện
            if position["side"] == "long":
                realized_pnl = (price - position["entry_price"]) * min(position["amount"], amount)
            else:  # short
                realized_pnl = (position["entry_price"] - price) * min(position["amount"], amount)
            
            position["realized_pnl"] += realized_pnl
            
            # Nếu đảo chiều vị thế
            if amount > position["amount"]:
                # Đóng vị thế hiện tại
                closed_amount = position["amount"]
                new_amount = amount - position["amount"]
                
                # Ghi nhận giao dịch đóng
                close_trade = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "close",
                    "price": price,
                    "amount": closed_amount,
                    "realized_pnl": realized_pnl
                }
                position["trades"].append(close_trade)
                
                # Tạo vị thế mới với phía ngược lại
                position["side"] = side
                position["amount"] = new_amount
                position["entry_price"] = price
                
                # Ghi nhận giao dịch mở mới
                open_trade = {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "amount": new_amount
                }
                position["trades"].append(open_trade)
                
                self.logger.info(f"Đã đảo chiều vị thế {symbol} từ {position['side']} sang {side}: {new_amount} @ {price}")
            
            # Nếu giảm vị thế
            elif amount < position["amount"]:
                # Giảm khối lượng
                position["amount"] -= amount
                
                # Ghi nhận giao dịch đóng một phần
                trade["realized_pnl"] = realized_pnl
                position["trades"].append(trade)
                
                self.logger.info(f"Đã giảm vị thế {symbol}: -{amount} @ {price}, còn lại: {position['amount']}")
            
            # Nếu đóng hoàn toàn vị thế
            elif amount == position["amount"]:
                # Đóng vị thế
                trade["side"] = "close"
                trade["realized_pnl"] = realized_pnl
                position["trades"].append(trade)
                
                # Chuyển vị thế vào lịch sử
                position["status"] = "closed"
                self.position_history.append(position)
                
                # Xóa khỏi danh sách vị thế hiện tại
                del self.positions[symbol]
                
                self.logger.info(f"Đã đóng vị thế {symbol}: {amount} @ {price}, realized_pnl: {realized_pnl}")
                
                return {
                    "status": "success",
                    "position": None,
                    "realized_pnl": realized_pnl
                }
        
        # Nếu side giống với vị thế hiện tại, tăng vị thế
        else:
            # Tính giá vào trung bình
            total_value = position["entry_price"] * position["amount"] + price * amount
            total_amount = position["amount"] + amount
            
            position["entry_price"] = total_value / total_amount
            position["amount"] = total_amount
            
            # Ghi nhận giao dịch
            position["trades"].append(trade)
            
            self.logger.info(f"Đã tăng vị thế {symbol}: +{amount} @ {price}, total: {position['amount']}, entry: {position['entry_price']}")
        
        # Cập nhật các thông số khác
        position["liquidation_price"] = self._calculate_liquidation_price(position["side"], position["entry_price"], position["leverage"])
        position["margin"] = (position["amount"] * position["entry_price"]) / position["leverage"]
        
        return {
            "status": "success",
            "position": position,
            "realized_pnl": realized_pnl
        }
    
    def close_position(
        self,
        symbol: str,
        price: float,
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Đóng một vị thế.
        
        Args:
            symbol: Cặp giao dịch
            price: Giá đóng
            amount: Khối lượng đóng (None để đóng toàn bộ)
            
        Returns:
            Dict chứa kết quả đóng vị thế
        """
        # Kiểm tra vị thế tồn tại
        if symbol not in self.positions:
            self.logger.warning(f"Không tìm thấy vị thế {symbol}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.POSITION_NOT_FOUND.value,
                    "message": f"Không tìm thấy vị thế"
                }
            }
        
        # Lấy vị thế
        position = self.positions[symbol]
        
        # Nếu không chỉ định khối lượng, đóng toàn bộ
        if amount is None:
            amount = position["amount"]
        
        # Kiểm tra khối lượng
        if amount > position["amount"]:
            self.logger.warning(f"Khối lượng đóng ({amount}) lớn hơn khối lượng vị thế ({position['amount']})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Khối lượng đóng lớn hơn khối lượng vị thế"
                }
            }
        
        # Xác định phía đóng ngược với phía vị thế
        close_side = "short" if position["side"] == "long" else "long"
        
        # Gọi hàm update_position để đóng vị thế
        return self.update_position(symbol, amount, price, close_side)
    
    def liquidate_position(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        Thanh lý một vị thế do không đủ margin.
        
        Args:
            symbol: Cặp giao dịch
            price: Giá thanh lý
            
        Returns:
            Dict chứa kết quả thanh lý
        """
        # Kiểm tra vị thế tồn tại
        if symbol not in self.positions:
            self.logger.warning(f"Không tìm thấy vị thế {symbol}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.POSITION_NOT_FOUND.value,
                    "message": f"Không tìm thấy vị thế"
                }
            }
        
        # Lấy vị thế
        position = self.positions[symbol]
        
        # Lấy thời gian hiện tại
        timestamp = int(time.time() * 1000)
        
        # Tính P&L đã thực hiện
        if position["side"] == "long":
            realized_pnl = (price - position["entry_price"]) * position["amount"]
        else:  # short
            realized_pnl = (position["entry_price"] - price) * position["amount"]
        
        position["realized_pnl"] += realized_pnl
        
        # Ghi nhận giao dịch
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": "liquidation",
            "price": price,
            "amount": position["amount"],
            "realized_pnl": realized_pnl
        }
        position["trades"].append(trade)
        
        # Cập nhật trạng thái vị thế
        position["status"] = "liquidated"
        
        # Chuyển vị thế vào lịch sử
        self.position_history.append(position)
        
        # Xóa khỏi danh sách vị thế hiện tại
        del self.positions[symbol]
        
        self.logger.warning(f"Đã thanh lý vị thế {symbol}: {position['amount']} @ {price}, realized_pnl: {realized_pnl}")
        
        return {
            "status": "success",
            "position": None,
            "realized_pnl": realized_pnl
        }
    
    def update_unrealized_pnl(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Cập nhật lợi nhuận chưa thực hiện của một vị thế.
        
        Args:
            symbol: Cặp giao dịch
            current_price: Giá hiện tại
            
        Returns:
            Dict chứa thông tin vị thế sau khi cập nhật
        """
        # Kiểm tra vị thế tồn tại
        if symbol not in self.positions:
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.POSITION_NOT_FOUND.value,
                    "message": f"Không tìm thấy vị thế"
                }
            }
        
        # Lấy vị thế
        position = self.positions[symbol]
        
        # Tính lợi nhuận chưa thực hiện
        if position["side"] == "long":
            position["unrealized_pnl"] = (current_price - position["entry_price"]) * position["amount"]
        else:  # short
            position["unrealized_pnl"] = (position["entry_price"] - current_price) * position["amount"]
        
        # Kiểm tra nếu cần thanh lý
        if self._check_liquidation(position, current_price):
            return self.liquidate_position(symbol, current_price)
        
        return {
            "status": "success",
            "position": position
        }
    
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
    
    def get_position_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử vị thế.
        
        Args:
            symbol: Lọc theo cặp giao dịch
            limit: Số lượng vị thế tối đa trả về
            
        Returns:
            Danh sách vị thế trong lịch sử
        """
        # Lọc theo symbol
        if symbol is not None:
            filtered_history = [pos for pos in self.position_history if pos["symbol"] == symbol]
        else:
            filtered_history = self.position_history.copy()
        
        # Sắp xếp theo thời gian mới nhất
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Giới hạn số lượng
        return filtered_history[:limit]
    
    def calculate_total_unrealized_pnl(self) -> float:
        """
        Tính tổng lợi nhuận chưa thực hiện của tất cả các vị thế.
        
        Returns:
            Tổng lợi nhuận chưa thực hiện
        """
        return sum(position["unrealized_pnl"] for position in self.positions.values())
    
    def calculate_total_margin(self) -> float:
        """
        Tính tổng margin của tất cả các vị thế.
        
        Returns:
            Tổng margin
        """
        return sum(position["margin"] for position in self.positions.values())
    
    def _calculate_liquidation_price(self, side: str, entry_price: float, leverage: float) -> float:
        """
        Tính giá thanh lý cho một vị thế.
        
        Args:
            side: Phía vị thế (long/short)
            entry_price: Giá vào
            leverage: Đòn bẩy
            
        Returns:
            Giá thanh lý
        """
        # Tính giá thanh lý dựa trên phía vị thế và đòn bẩy
        if side == "long":
            # Giá thanh lý = giá vào - giá vào * (1 / đòn bẩy - maintenance_margin_ratio)
            return entry_price * (1 - (1 / leverage - self.maintenance_margin_ratio))
        else:  # short
            # Giá thanh lý = giá vào + giá vào * (1 / đòn bẩy - maintenance_margin_ratio)
            return entry_price * (1 + (1 / leverage - self.maintenance_margin_ratio))
    
    def _check_liquidation(self, position: Dict[str, Any], current_price: float) -> bool:
        """
        Kiểm tra xem một vị thế có cần thanh lý không.
        
        Args:
            position: Thông tin vị thế
            current_price: Giá hiện tại
            
        Returns:
            True nếu vị thế cần thanh lý, False nếu không
        """
        # Kiểm tra dựa trên phía vị thế và giá thanh lý
        if position["side"] == "long":
            return current_price <= position["liquidation_price"]
        else:  # short
            return current_price >= position["liquidation_price"]