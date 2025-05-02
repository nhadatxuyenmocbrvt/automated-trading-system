"""
Quản lý tài khoản cho mô phỏng sàn giao dịch.
File này định nghĩa lớp AccountManager để quản lý số dư và thống kê tài khoản
trong môi trường mô phỏng sàn giao dịch.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import time
from datetime import datetime

from config.logging_config import get_logger
from config.constants import ErrorCode

class AccountManager:
    """
    Lớp quản lý tài khoản giao dịch.
    Cung cấp các phương thức để quản lý số dư, thống kê tài khoản, và lịch sử giao dịch.
    """
    
    def __init__(
        self,
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        max_withdraw_percentage: float = 0.9,  # Tỷ lệ rút tối đa (90% số dư)
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý tài khoản.
        
        Args:
            initial_balance: Số dư ban đầu theo loại tiền
            max_withdraw_percentage: Tỷ lệ rút tối đa
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("account_manager")
        self.initial_balance = initial_balance.copy()
        self.max_withdraw_percentage = max_withdraw_percentage
        
        # Khởi tạo trạng thái tài khoản
        self.reset()
        
        self.logger.info(f"Đã khởi tạo AccountManager với số dư {initial_balance}")
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái tài khoản.
        """
        self.balance = self.initial_balance.copy()
        self.transaction_history = []
        self.deposit_history = []
        self.withdraw_history = []
        self.transfer_history = []
        
        # Khởi tạo thống kê
        self.stats = {
            "peak_equity": sum(self.balance.values()),
            "min_equity": sum(self.balance.values()),
            "max_drawdown": 0.0,
            "total_deposited": 0.0,
            "total_withdrawn": 0.0,
            "total_traded": 0.0,
            "total_fees": 0.0,
            "total_realized_pnl": 0.0
        }
        
        self.logger.info("Đã đặt lại AccountManager")
    
    def deposit(self, currency: str, amount: float) -> Dict[str, Any]:
        """
        Nạp tiền vào tài khoản.
        
        Args:
            currency: Loại tiền
            amount: Số lượng
            
        Returns:
            Dict chứa kết quả nạp tiền
        """
        # Kiểm tra số lượng
        if amount <= 0:
            self.logger.warning(f"Số lượng nạp không hợp lệ: {amount}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Số lượng nạp phải lớn hơn 0"
                }
            }
        
        # Cập nhật số dư
        if currency in self.balance:
            self.balance[currency] += amount
        else:
            self.balance[currency] = amount
        
        # Tạo bản ghi giao dịch
        transaction = {
            "timestamp": int(time.time() * 1000),
            "type": "deposit",
            "currency": currency,
            "amount": amount,
            "balance_after": self.balance[currency]
        }
        
        # Cập nhật lịch sử
        self.transaction_history.append(transaction)
        self.deposit_history.append(transaction)
        
        # Cập nhật thống kê
        self.stats["total_deposited"] += amount
        self._update_equity_stats()
        
        self.logger.info(f"Đã nạp {amount} {currency}, số dư mới: {self.balance[currency]}")
        
        return {
            "status": "success",
            "transaction": transaction
        }
    
    def withdraw(self, currency: str, amount: float) -> Dict[str, Any]:
        """
        Rút tiền từ tài khoản.
        
        Args:
            currency: Loại tiền
            amount: Số lượng
            
        Returns:
            Dict chứa kết quả rút tiền
        """
        # Kiểm tra số lượng
        if amount <= 0:
            self.logger.warning(f"Số lượng rút không hợp lệ: {amount}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Số lượng rút phải lớn hơn 0"
                }
            }
        
        # Kiểm tra số dư
        if currency not in self.balance:
            self.logger.warning(f"Không có {currency} trong tài khoản")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                    "message": f"Không có {currency} trong tài khoản"
                }
            }
        
        # Kiểm tra số dư đủ
        if self.balance[currency] < amount:
            self.logger.warning(f"Số dư không đủ: {self.balance[currency]} < {amount}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                    "message": "Số dư không đủ"
                }
            }
        
        # Kiểm tra giới hạn rút
        max_withdraw = self.balance[currency] * self.max_withdraw_percentage
        if amount > max_withdraw:
            self.logger.warning(f"Số lượng rút ({amount}) vượt quá giới hạn ({max_withdraw})")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Số lượng rút vượt quá giới hạn {self.max_withdraw_percentage * 100}%"
                }
            }
        
        # Cập nhật số dư
        self.balance[currency] -= amount
        
        # Tạo bản ghi giao dịch
        transaction = {
            "timestamp": int(time.time() * 1000),
            "type": "withdraw",
            "currency": currency,
            "amount": amount,
            "balance_after": self.balance[currency]
        }
        
        # Cập nhật lịch sử
        self.transaction_history.append(transaction)
        self.withdraw_history.append(transaction)
        
        # Cập nhật thống kê
        self.stats["total_withdrawn"] += amount
        self._update_equity_stats()
        
        self.logger.info(f"Đã rút {amount} {currency}, số dư mới: {self.balance[currency]}")
        
        return {
            "status": "success",
            "transaction": transaction
        }
    
    def transfer(self, from_currency: str, to_currency: str, amount: float, rate: float) -> Dict[str, Any]:
        """
        Chuyển đổi giữa các loại tiền.
        
        Args:
            from_currency: Loại tiền nguồn
            to_currency: Loại tiền đích
            amount: Số lượng
            rate: Tỷ giá
            
        Returns:
            Dict chứa kết quả chuyển đổi
        """
        # Kiểm tra số lượng
        if amount <= 0:
            self.logger.warning(f"Số lượng chuyển đổi không hợp lệ: {amount}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Số lượng chuyển đổi phải lớn hơn 0"
                }
            }
        
        # Kiểm tra tỷ giá
        if rate <= 0:
            self.logger.warning(f"Tỷ giá không hợp lệ: {rate}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ giá phải lớn hơn 0"
                }
            }
        
        # Kiểm tra loại tiền nguồn
        if from_currency not in self.balance:
            self.logger.warning(f"Không có {from_currency} trong tài khoản")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                    "message": f"Không có {from_currency} trong tài khoản"
                }
            }
        
        # Kiểm tra số dư đủ
        if self.balance[from_currency] < amount:
            self.logger.warning(f"Số dư không đủ: {self.balance[from_currency]} < {amount}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                    "message": "Số dư không đủ"
                }
            }
        
        # Tính số lượng đích
        to_amount = amount * rate
        
        # Cập nhật số dư
        self.balance[from_currency] -= amount
        
        if to_currency in self.balance:
            self.balance[to_currency] += to_amount
        else:
            self.balance[to_currency] = to_amount
        
        # Tạo bản ghi giao dịch
        transaction = {
            "timestamp": int(time.time() * 1000),
            "type": "transfer",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "from_amount": amount,
            "to_amount": to_amount,
            "rate": rate,
            "from_balance_after": self.balance[from_currency],
            "to_balance_after": self.balance[to_currency]
        }
        
        # Cập nhật lịch sử
        self.transaction_history.append(transaction)
        self.transfer_history.append(transaction)
        
        self.logger.info(f"Đã chuyển đổi {amount} {from_currency} sang {to_amount} {to_currency}")
        
        return {
            "status": "success",
            "transaction": transaction
        }
    
    def trade(
        self,
        currency: str,
        amount: float,
        fee: float,
        realized_pnl: float = 0.0,
        trade_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ghi lại giao dịch và cập nhật số dư.
        
        Args:
            currency: Loại tiền
            amount: Số lượng (dương: tăng, âm: giảm)
            fee: Phí giao dịch
            realized_pnl: Lợi nhuận đã thực hiện
            trade_info: Thông tin giao dịch bổ sung
            
        Returns:
            Dict chứa kết quả giao dịch
        """
        # Kiểm tra phí
        if fee < 0:
            self.logger.warning(f"Phí giao dịch không hợp lệ: {fee}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phí giao dịch không được âm"
                }
            }
        
        # Kiểm tra số lượng
        if amount < 0 and currency in self.balance and abs(amount) > self.balance[currency]:
            self.logger.warning(f"Số dư không đủ: {self.balance.get(currency, 0)} < {abs(amount)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                    "message": "Số dư không đủ"
                }
            }
        
        # Cập nhật số dư
        if currency in self.balance:
            self.balance[currency] += amount - fee
        else:
            self.balance[currency] = amount - fee
        
        # Tạo bản ghi giao dịch
        transaction = {
            "timestamp": int(time.time() * 1000),
            "type": "trade",
            "currency": currency,
            "amount": amount,
            "fee": fee,
            "realized_pnl": realized_pnl,
            "balance_after": self.balance[currency]
        }
        
        # Thêm thông tin giao dịch bổ sung
        if trade_info:
            transaction.update(trade_info)
        
        # Cập nhật lịch sử
        self.transaction_history.append(transaction)
        
        # Cập nhật thống kê
        self.stats["total_traded"] += abs(amount)
        self.stats["total_fees"] += fee
        self.stats["total_realized_pnl"] += realized_pnl
        self._update_equity_stats()
        
        self.logger.info(f"Đã ghi nhận giao dịch: {amount} {currency}, phí: {fee}, realized_pnl: {realized_pnl}")
        
        return {
            "status": "success",
            "transaction": transaction
        }
    
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
    
    def get_equity(self) -> float:
        """
        Tính tổng giá trị tài sản.
        
        Returns:
            Tổng giá trị tài sản
        """
        return sum(self.balance.values())
    
    def get_transaction_history(
        self,
        type_filter: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử giao dịch.
        
        Args:
            type_filter: Lọc theo loại giao dịch
            start_time: Thời gian bắt đầu (milliseconds)
            end_time: Thời gian kết thúc (milliseconds)
            limit: Số lượng giao dịch tối đa trả về
            
        Returns:
            Danh sách các giao dịch
        """
        # Lọc theo loại
        if type_filter is not None:
            filtered_history = [tx for tx in self.transaction_history if tx["type"] == type_filter]
        else:
            filtered_history = self.transaction_history.copy()
        
        # Lọc theo thời gian
        if start_time is not None:
            filtered_history = [tx for tx in filtered_history if tx["timestamp"] >= start_time]
        
        if end_time is not None:
            filtered_history = [tx for tx in filtered_history if tx["timestamp"] <= end_time]
        
        # Sắp xếp theo thời gian mới nhất
        filtered_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Giới hạn số lượng
        return filtered_history[:limit]
    
    def get_stats(self) -> Dict[str, float]:
        """
        Lấy thống kê tài khoản.
        
        Returns:
            Dict chứa thống kê
        """
        return self.stats.copy()
    
    def _update_equity_stats(self) -> None:
        """
        Cập nhật thống kê equity.
        """
        # Tính equity hiện tại
        current_equity = self.get_equity()
        
        # Cập nhật peak equity
        if current_equity > self.stats["peak_equity"]:
            self.stats["peak_equity"] = current_equity
        
        # Cập nhật min equity
        if current_equity < self.stats["min_equity"]:
            self.stats["min_equity"] = current_equity
        
        # Cập nhật max drawdown
        if self.stats["peak_equity"] > 0:
            drawdown = (self.stats["peak_equity"] - current_equity) / self.stats["peak_equity"]
            if drawdown > self.stats["max_drawdown"]:
                self.stats["max_drawdown"] = drawdown