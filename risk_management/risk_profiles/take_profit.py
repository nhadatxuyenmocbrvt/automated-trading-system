"""
Quản lý chốt lời.
File này định nghĩa các phương pháp để tính toán và quản lý chốt lời
cho các vị thế giao dịch, bao gồm các loại chốt lời cố định, di động và nhiều cấp.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import pandas as pd

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import ErrorCode

class TakeProfit:
    """
    Lớp quản lý chốt lời.
    Cung cấp các phương pháp khác nhau để tính toán và cập nhật giá chốt lời
    nhằm tối ưu hóa lợi nhuận và quản lý rủi ro.
    """
    
    def __init__(
        self,
        default_risk_reward_ratio: float = 2.0,
        default_fixed_percent: float = 0.03,
        default_trailing_percent: float = 0.02,
        default_partial_tp_levels: List[float] = [0.5, 0.75, 1.0],
        default_partial_tp_sizes: List[float] = [0.3, 0.3, 0.4],
        use_candle_low_high: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý chốt lời.
        
        Args:
            default_risk_reward_ratio: Tỷ lệ rủi ro/phần thưởng mặc định
            default_fixed_percent: Phần trăm chốt lời cố định mặc định
            default_trailing_percent: Phần trăm chốt lời di động mặc định
            default_partial_tp_levels: Danh sách các mức chốt lời từng phần mặc định
            default_partial_tp_sizes: Danh sách kích thước chốt lời từng phần mặc định
            use_candle_low_high: Sử dụng giá cao/thấp của nến trong tính toán
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("take_profit")
        self.default_risk_reward_ratio = default_risk_reward_ratio
        self.default_fixed_percent = default_fixed_percent
        self.default_trailing_percent = default_trailing_percent
        self.default_partial_tp_levels = default_partial_tp_levels
        self.default_partial_tp_sizes = default_partial_tp_sizes
        self.use_candle_low_high = use_candle_low_high
        
        # Kiểm tra các giá trị mặc định hợp lệ
        if len(default_partial_tp_levels) != len(default_partial_tp_sizes):
            self.logger.warning("Độ dài của danh sách mức chốt lời và kích thước chốt lời không khớp")
        
        if sum(default_partial_tp_sizes) != 1.0:
            self.logger.warning(f"Tổng kích thước chốt lời từng phần ({sum(default_partial_tp_sizes)}) khác 1.0")
        
        self.logger.info(f"Đã khởi tạo TakeProfit với R:R={default_risk_reward_ratio}, fixed_percent={default_fixed_percent}")
    
    def calculate_fixed_take_profit(
        self,
        entry_price: float,
        position_side: str,
        percent: Optional[float] = None,
        price_amount: Optional[float] = None,
        use_market_price: bool = False,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tính giá chốt lời cố định.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            percent: Phần trăm chốt lời (ưu tiên nếu được cung cấp)
            price_amount: Giá trị chốt lời tuyệt đối
            use_market_price: Sử dụng giá thị trường để điều chỉnh
            market_data: Dữ liệu thị trường nếu use_market_price=True
            
        Returns:
            Dict chứa thông tin chốt lời
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        if entry_price <= 0:
            self.logger.warning(f"Giá vào lệnh không hợp lệ: {entry_price}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Giá vào lệnh phải > 0"
                }
            }
        
        try:
            # Nếu sử dụng giá thị trường
            if use_market_price and market_data:
                # Điều chỉnh dựa trên giá thị trường hiện tại
                price_to_use = None
                
                if self.use_candle_low_high and 'low' in market_data and 'high' in market_data:
                    if position_side == "long":
                        price_to_use = market_data.get('high', None)
                    else:
                        price_to_use = market_data.get('low', None)
                
                if price_to_use is None:
                    price_to_use = market_data.get('close', entry_price)
                
                # Nếu là long, chốt lời trên giá thị trường
                if position_side == "long" and price_to_use > entry_price:
                    entry_price = price_to_use
                # Nếu là short, chốt lời dưới giá thị trường
                elif position_side == "short" and price_to_use < entry_price:
                    entry_price = price_to_use
            
            # Tính giá chốt lời
            if price_amount is not None:
                # Giá trị tuyệt đối
                if position_side == "long":
                    take_profit_price = entry_price + price_amount
                else:  # short
                    take_profit_price = entry_price - price_amount
                
                # Tính phần trăm tương ứng
                take_profit_percent = price_amount / entry_price
            
            else:
                # Phần trăm
                if percent is None:
                    percent = self.default_fixed_percent
                
                if position_side == "long":
                    take_profit_price = entry_price * (1 + percent)
                else:  # short
                    take_profit_price = entry_price * (1 - percent)
                
                take_profit_percent = percent
            
            result = {
                "status": "success",
                "take_profit_price": take_profit_price,
                "take_profit_percent": take_profit_percent,
                "take_profit_type": "fixed",
                "position_side": position_side,
                "entry_price": entry_price,
            }
            
            self.logger.info(f"Đã tính chốt lời cố định cho {position_side}: {take_profit_price:.2f} ({take_profit_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời cố định: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời cố định: {str(e)}"
                }
            }
    
    def calculate_risk_reward_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        position_side: str,
        risk_reward_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính giá chốt lời dựa trên tỷ lệ rủi ro/phần thưởng.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            position_side: Phía vị thế ('long' hoặc 'short')
            risk_reward_ratio: Tỷ lệ rủi ro/phần thưởng (mặc định: default_risk_reward_ratio)
            
        Returns:
            Dict chứa thông tin chốt lời dựa trên R:R
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        if risk_reward_ratio is None:
            risk_reward_ratio = self.default_risk_reward_ratio
        
        try:
            # Tính khoảng cách từ giá vào lệnh đến dừng lỗ
            if position_side == "long":
                stop_loss_distance = entry_price - stop_loss_price
                
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_distance <= 0:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế long")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải thấp hơn giá vào cho vị thế long"
                        }
                    }
                
                # Tính khoảng cách đến chốt lời
                take_profit_distance = stop_loss_distance * risk_reward_ratio
                
                # Tính giá chốt lời
                take_profit_price = entry_price + take_profit_distance
                
            else:  # short
                stop_loss_distance = stop_loss_price - entry_price
                
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_distance <= 0:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế short")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải cao hơn giá vào cho vị thế short"
                        }
                    }
                
                # Tính khoảng cách đến chốt lời
                take_profit_distance = stop_loss_distance * risk_reward_ratio
                
                # Tính giá chốt lời
                take_profit_price = entry_price - take_profit_distance
            
            # Tính phần trăm chốt lời
            take_profit_percent = take_profit_distance / entry_price
            
            result = {
                "status": "success",
                "take_profit_price": take_profit_price,
                "take_profit_percent": take_profit_percent,
                "take_profit_type": "risk_reward",
                "position_side": position_side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "risk_reward_ratio": risk_reward_ratio,
                "stop_loss_distance": stop_loss_distance,
                "take_profit_distance": take_profit_distance
            }
            
            self.logger.info(f"Đã tính chốt lời R:R={risk_reward_ratio} cho {position_side}: {take_profit_price:.2f} ({take_profit_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời R:R: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời R:R: {str(e)}"
                }
            }
    
    def calculate_fibonacci_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        position_side: str,
        fib_levels: List[float] = [0.618, 1.0, 1.618, 2.618],
        reference_high: Optional[float] = None,
        reference_low: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính giá chốt lời dựa trên các mức Fibonacci.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            position_side: Phía vị thế ('long' hoặc 'short')
            fib_levels: Danh sách các mức Fibonacci
            reference_high: Giá cao tham chiếu
            reference_low: Giá thấp tham chiếu
            
        Returns:
            Dict chứa thông tin chốt lời dựa trên Fibonacci
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        try:
            # Xác định điểm tham chiếu
            if position_side == "long":
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_price >= entry_price:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế long")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải thấp hơn giá vào cho vị thế long"
                        }
                    }
                
                # Xác định khoảng giá
                if reference_high is not None and reference_low is not None:
                    # Sử dụng khoảng giá tham chiếu
                    price_range = reference_high - reference_low
                    ref_low = reference_low
                else:
                    # Sử dụng entry_price và stop_loss_price
                    price_range = entry_price - stop_loss_price
                    ref_low = stop_loss_price
                
                # Tính các mức chốt lời
                take_profit_prices = []
                for level in fib_levels:
                    tp_price = entry_price + (price_range * level)
                    take_profit_prices.append(tp_price)
                
                # Tính phần trăm chốt lời
                take_profit_percents = [(tp - entry_price) / entry_price for tp in take_profit_prices]
                
            else:  # short
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_price <= entry_price:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế short")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải cao hơn giá vào cho vị thế short"
                        }
                    }
                
                # Xác định khoảng giá
                if reference_high is not None and reference_low is not None:
                    # Sử dụng khoảng giá tham chiếu
                    price_range = reference_high - reference_low
                    ref_high = reference_high
                else:
                    # Sử dụng entry_price và stop_loss_price
                    price_range = stop_loss_price - entry_price
                    ref_high = stop_loss_price
                
                # Tính các mức chốt lời
                take_profit_prices = []
                for level in fib_levels:
                    tp_price = entry_price - (price_range * level)
                    take_profit_prices.append(tp_price)
                
                # Tính phần trăm chốt lời
                take_profit_percents = [(entry_price - tp) / entry_price for tp in take_profit_prices]
            
            result = {
                "status": "success",
                "take_profit_prices": take_profit_prices,
                "take_profit_percents": take_profit_percents,
                "take_profit_type": "fibonacci",
                "position_side": position_side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "fib_levels": fib_levels,
                "price_range": price_range
            }
            
            self.logger.info(f"Đã tính chốt lời Fibonacci cho {position_side}: {take_profit_prices}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời Fibonacci: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời Fibonacci: {str(e)}"
                }
            }
    
    def calculate_trailing_take_profit(
        self,
        current_price: float,
        entry_price: float,
        position_side: str,
        highest_price: float,
        lowest_price: float,
        trailing_percent: Optional[float] = None,
        activation_percent: float = 0.01
    ) -> Dict[str, Any]:
        """
        Tính và cập nhật chốt lời di động (trailing take profit).
        
        Args:
            current_price: Giá hiện tại
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            highest_price: Giá cao nhất kể từ khi vào lệnh
            lowest_price: Giá thấp nhất kể từ khi vào lệnh
            trailing_percent: Phần trăm di động
            activation_percent: Phần trăm để kích hoạt di động
            
        Returns:
            Dict chứa thông tin chốt lời di động
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        try:
            # Thiết lập các giá trị mặc định
            if trailing_percent is None:
                trailing_percent = self.default_trailing_percent
            
            # Tính giá chốt lời ban đầu
            initial_take_profit = self.calculate_fixed_take_profit(
                entry_price=entry_price,
                position_side=position_side,
                percent=self.default_fixed_percent
            )
            initial_tp_price = initial_take_profit.get("take_profit_price", 0)
            
            # Kiểm tra điều kiện kích hoạt
            activation_threshold = entry_price * (1 + activation_percent) if position_side == "long" else entry_price * (1 - activation_percent)
            is_activated = (position_side == "long" and highest_price >= activation_threshold) or (position_side == "short" and lowest_price <= activation_threshold)
            
            # Tính chốt lời di động
            if position_side == "long":
                # Chốt lời ban đầu
                take_profit_price = initial_tp_price
                
                # Nếu đã kích hoạt, tính dựa trên giá cao nhất
                if is_activated:
                    trailing_take_profit = highest_price * (1 - trailing_percent)
                    
                    # Cập nhật nếu trailing_take_profit cao hơn initial_tp_price
                    if trailing_take_profit > take_profit_price:
                        take_profit_price = trailing_take_profit
                
                # Tính phần trăm chốt lời
                take_profit_percent = (take_profit_price - entry_price) / entry_price
                
            else:  # short
                # Chốt lời ban đầu
                take_profit_price = initial_tp_price
                
                # Nếu đã kích hoạt, tính dựa trên giá thấp nhất
                if is_activated:
                    trailing_take_profit = lowest_price * (1 + trailing_percent)
                    
                    # Cập nhật nếu trailing_take_profit thấp hơn initial_tp_price
                    if trailing_take_profit < take_profit_price:
                        take_profit_price = trailing_take_profit
                
                # Tính phần trăm chốt lời
                take_profit_percent = (entry_price - take_profit_price) / entry_price
            
            # Tính khoảng cách từ giá hiện tại đến giá chốt lời
            if position_side == "long":
                distance_percent = (take_profit_price - current_price) / current_price
            else:  # short
                distance_percent = (current_price - take_profit_price) / current_price
            
            # Tính lợi nhuận hiện tại
            if position_side == "long":
                current_profit_percent = (current_price - entry_price) / entry_price
            else:  # short
                current_profit_percent = (entry_price - current_price) / entry_price
            
            result = {
                "status": "success",
                "take_profit_price": take_profit_price,
                "take_profit_percent": take_profit_percent,
                "take_profit_type": "trailing",
                "position_side": position_side,
                "entry_price": entry_price,
                "current_price": current_price,
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "trailing_percent": trailing_percent,
                "initial_take_profit": initial_tp_price,
                "distance_percent": distance_percent,
                "current_profit_percent": current_profit_percent,
                "is_activated": is_activated,
                "activation_threshold": activation_threshold
            }
            
            self.logger.info(f"Đã tính chốt lời di động cho {position_side}: {take_profit_price:.2f} (khoảng cách: {distance_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời di động: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời di động: {str(e)}"
                }
            }
    
    def calculate_partial_take_profits(
        self,
        entry_price: float,
        stop_loss_price: float,
        position_side: str,
        levels: Optional[List[float]] = None,
        sizes: Optional[List[float]] = None,
        risk_reward_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính các mức chốt lời từng phần.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            position_side: Phía vị thế ('long' hoặc 'short')
            levels: Danh sách các mức R:R (ví dụ: [1, 2, 3])
            sizes: Danh sách kích thước từng phần (ví dụ: [0.3, 0.3, 0.4])
            risk_reward_ratio: Tỷ lệ R:R cơ sở (mặc định: default_risk_reward_ratio)
            
        Returns:
            Dict chứa thông tin chốt lời từng phần
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        # Thiết lập các giá trị mặc định
        if levels is None:
            levels = self.default_partial_tp_levels
        
        if sizes is None:
            sizes = self.default_partial_tp_sizes
        
        if risk_reward_ratio is None:
            risk_reward_ratio = self.default_risk_reward_ratio
        
        # Kiểm tra số lượng level và size
        if len(levels) != len(sizes):
            self.logger.warning(f"Số lượng level ({len(levels)}) và size ({len(sizes)}) không khớp")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Số lượng level và size phải bằng nhau"
                }
            }
        
        # Kiểm tra tổng size
        if abs(sum(sizes) - 1.0) > 0.001:  # Cho phép sai số nhỏ
            self.logger.warning(f"Tổng size ({sum(sizes)}) khác 1.0")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tổng size phải bằng 1.0"
                }
            }
        
        try:
            # Tính khoảng cách từ giá vào lệnh đến dừng lỗ
            if position_side == "long":
                stop_loss_distance = entry_price - stop_loss_price
                
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_distance <= 0:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế long")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải thấp hơn giá vào cho vị thế long"
                        }
                    }
                
                # Tính các mức chốt lời
                take_profit_prices = []
                for level in levels:
                    tp_distance = stop_loss_distance * risk_reward_ratio * level
                    tp_price = entry_price + tp_distance
                    take_profit_prices.append(tp_price)
                
                # Tính phần trăm chốt lời
                take_profit_percents = [(tp - entry_price) / entry_price for tp in take_profit_prices]
                
            else:  # short
                stop_loss_distance = stop_loss_price - entry_price
                
                # Kiểm tra dừng lỗ hợp lệ
                if stop_loss_distance <= 0:
                    self.logger.warning(f"Giá dừng lỗ ({stop_loss_price}) không hợp lệ cho vị thế short")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá dừng lỗ phải cao hơn giá vào cho vị thế short"
                        }
                    }
                
                # Tính các mức chốt lời
                take_profit_prices = []
                for level in levels:
                    tp_distance = stop_loss_distance * risk_reward_ratio * level
                    tp_price = entry_price - tp_distance
                    take_profit_prices.append(tp_price)
                
                # Tính phần trăm chốt lời
                take_profit_percents = [(entry_price - tp) / entry_price for tp in take_profit_prices]
            
            # Tạo thông tin chốt lời từng phần
            partial_take_profits = []
            for i in range(len(levels)):
                partial_take_profits.append({
                    "level": levels[i],
                    "price": take_profit_prices[i],
                    "percent": take_profit_percents[i],
                    "size": sizes[i],
                    "r_multiple": levels[i] * risk_reward_ratio
                })
            
            result = {
                "status": "success",
                "take_profit_prices": take_profit_prices,
                "take_profit_percents": take_profit_percents,
                "take_profit_sizes": sizes,
                "take_profit_type": "partial",
                "position_side": position_side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "risk_reward_ratio": risk_reward_ratio,
                "levels": levels,
                "partial_take_profits": partial_take_profits
            }
            
            self.logger.info(f"Đã tính chốt lời từng phần cho {position_side}: {take_profit_prices}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời từng phần: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời từng phần: {str(e)}"
                }
            }
    
    def calculate_support_resistance_take_profit(
        self,
        entry_price: float,
        position_side: str,
        support_levels: List[float],
        resistance_levels: List[float],
        additional_buffer: float = 0.005,
        fallback_percent: float = 0.03
    ) -> Dict[str, Any]:
        """
        Tính giá chốt lời dựa trên các mức hỗ trợ/kháng cự.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            support_levels: Danh sách các mức hỗ trợ
            resistance_levels: Danh sách các mức kháng cự
            additional_buffer: Buffer bổ sung (%)
            fallback_percent: Phần trăm dự phòng nếu không tìm thấy mức hỗ trợ/kháng cự thích hợp
            
        Returns:
            Dict chứa thông tin chốt lời dựa trên hỗ trợ/kháng cự
        """
        position_side = position_side.lower()
        
        # Kiểm tra tham số
        if position_side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {position_side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Phía vị thế phải là 'long' hoặc 'short'"
                }
            }
        
        try:
            # Sắp xếp các mức hỗ trợ và kháng cự
            support_levels = sorted(support_levels)
            resistance_levels = sorted(resistance_levels)
            
            # Tìm mức chốt lời dựa trên hỗ trợ/kháng cự
            take_profit_price = None
            level_used = None
            
            if position_side == "long":
                # Cho long, tìm mức kháng cự gần nhất cao hơn giá vào
                for level in resistance_levels:
                    if level > entry_price:
                        take_profit_price = level * (1 + additional_buffer)
                        level_used = level
                        break
            else:  # short
                # Cho short, tìm mức hỗ trợ gần nhất thấp hơn giá vào
                for level in reversed(support_levels):
                    if level < entry_price:
                        take_profit_price = level * (1 - additional_buffer)
                        level_used = level
                        break
            
            # Nếu không tìm thấy mức thích hợp, sử dụng fallback
            used_fallback = False
            if take_profit_price is None:
                used_fallback = True
                if position_side == "long":
                    take_profit_price = entry_price * (1 + fallback_percent)
                else:  # short
                    take_profit_price = entry_price * (1 - fallback_percent)
                
                self.logger.warning(f"Không tìm thấy mức hỗ trợ/kháng cự thích hợp, sử dụng fallback: {take_profit_price:.2f}")
            
            # Tính phần trăm chốt lời
            if position_side == "long":
                take_profit_percent = (take_profit_price - entry_price) / entry_price
            else:  # short
                take_profit_percent = (entry_price - take_profit_price) / entry_price
            
            result = {
                "status": "success",
                "take_profit_price": take_profit_price,
                "take_profit_percent": take_profit_percent,
                "take_profit_type": "support_resistance",
                "position_side": position_side,
                "entry_price": entry_price,
                "level_used": level_used,
                "used_fallback": used_fallback,
                "additional_buffer": additional_buffer,
                "fallback_percent": fallback_percent
            }
            
            self.logger.info(f"Đã tính chốt lời S/R cho {position_side}: {take_profit_price:.2f} ({take_profit_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chốt lời S/R: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chốt lời S/R: {str(e)}"
                }
            }
    
    def update_take_profit(
        self,
        current_take_profit: Dict[str, Any],
        current_price: float,
        highest_price: float,
        lowest_price: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật chốt lời dựa trên điều kiện thị trường hiện tại.
        
        Args:
            current_take_profit: Thông tin chốt lời hiện tại
            current_price: Giá hiện tại
            highest_price: Giá cao nhất kể từ khi vào lệnh
            lowest_price: Giá thấp nhất kể từ khi vào lệnh
            market_data: Dữ liệu thị trường bổ sung
            
        Returns:
            Dict chứa thông tin chốt lời sau khi cập nhật
        """
        try:
            # Kiểm tra dữ liệu đầu vào
            if current_take_profit.get("status") != "success":
                return current_take_profit
            
            take_profit_type = current_take_profit.get("take_profit_type")
            position_side = current_take_profit.get("position_side")
            entry_price = current_take_profit.get("entry_price")
            
            # Cập nhật dựa trên loại chốt lời
            if take_profit_type == "trailing":
                # Cập nhật chốt lời di động
                return self.calculate_trailing_take_profit(
                    current_price=current_price,
                    entry_price=entry_price,
                    position_side=position_side,
                    highest_price=highest_price,
                    lowest_price=lowest_price,
                    trailing_percent=current_take_profit.get("trailing_percent"),
                    activation_percent=current_take_profit.get("activation_percent", 0.01)
                )
            
            else:
                # Các loại chốt lời khác không cần cập nhật
                return current_take_profit
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật chốt lời: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi cập nhật chốt lời: {str(e)}"
                }
            }
    
    def choose_optimal_take_profit(
        self,
        take_profit_candidates: List[Dict[str, Any]],
        entry_price: float,
        position_side: str,
        risk_preference: str = "balanced"  # "conservative", "balanced", "aggressive"
    ) -> Dict[str, Any]:
        """
        Chọn chốt lời tối ưu từ nhiều ứng viên.
        
        Args:
            take_profit_candidates: Danh sách các chốt lời ứng viên
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            risk_preference: Khẩu vị rủi ro
            
        Returns:
            Dict chứa thông tin chốt lời tối ưu
        """
        position_side = position_side.lower()
        
        # Lọc các ứng viên hợp lệ
        valid_candidates = [tp for tp in take_profit_candidates if tp.get("status") == "success"]
        
        if not valid_candidates:
            self.logger.warning("Không có ứng viên chốt lời hợp lệ")
            # Tạo chốt lời mặc định
            return self.calculate_fixed_take_profit(
                entry_price=entry_price,
                position_side=position_side
            )
        
        try:
            # Tạo điểm đánh giá cho mỗi ứng viên
            for candidate in valid_candidates:
                # Lấy giá chốt lời (nếu là partial, lấy giá cao nhất)
                if candidate.get("take_profit_type") == "partial":
                    tp_price = candidate.get("take_profit_prices", [])[-1]
                    tp_percent = candidate.get("take_profit_percents", [])[-1]
                elif candidate.get("take_profit_type") == "fibonacci":
                    tp_price = candidate.get("take_profit_prices", [])[-1]
                    tp_percent = candidate.get("take_profit_percents", [])[-1]
                else:
                    tp_price = candidate.get("take_profit_price", 0)
                    tp_percent = candidate.get("take_profit_percent", 0)
                
                # Tính khoảng cách từ giá vào lệnh đến chốt lời
                if position_side == "long":
                    distance = (tp_price - entry_price) / entry_price
                else:  # short
                    distance = (entry_price - tp_price) / entry_price
                
                # Điểm đánh giá dựa trên loại chốt lời và khẩu vị rủi ro
                score = 0
                
                # 1. Chốt lời cố định
                if candidate.get("take_profit_type") == "fixed":
                    if risk_preference == "conservative":
                        score = 70
                    elif risk_preference == "balanced":
                        score = 50
                    else:  # aggressive
                        score = 30
                
                # 2. Chốt lời R:R
                elif candidate.get("take_profit_type") == "risk_reward":
                    rr_ratio = candidate.get("risk_reward_ratio", 1.0)
                    if risk_preference == "conservative":
                        score = 50 + min(rr_ratio * 10, 30)
                    elif risk_preference == "balanced":
                        score = 60 + min(rr_ratio * 10, 30)
                    else:  # aggressive
                        score = 70 + min(rr_ratio * 10, 30)
                
                # 3. Chốt lời Fibonacci
                elif candidate.get("take_profit_type") == "fibonacci":
                    if risk_preference == "conservative":
                        score = 60
                    elif risk_preference == "balanced":
                        score = 70
                    else:  # aggressive
                        score = 80
                
                # 4. Chốt lời di động
                elif candidate.get("take_profit_type") == "trailing":
                    if risk_preference == "conservative":
                        score = 60
                    elif risk_preference == "balanced":
                        score = 80
                    else:  # aggressive
                        score = 90
                
                # 5. Chốt lời từng phần
                elif candidate.get("take_profit_type") == "partial":
                    if risk_preference == "conservative":
                        score = 80
                    elif risk_preference == "balanced":
                        score = 90
                    else:  # aggressive
                        score = 70
                
                # 6. Chốt lời S/R
                elif candidate.get("take_profit_type") == "support_resistance":
                    if risk_preference == "conservative":
                        score = 70
                    elif risk_preference == "balanced":
                        score = 80
                    else:  # aggressive
                        score = 70
                    
                    # Trừ điểm nếu sử dụng fallback
                    if candidate.get("used_fallback", False):
                        score -= 20
                
                # Điều chỉnh theo khoảng cách
                if risk_preference == "conservative":
                    # Ưu tiên chốt lời gần hơn
                    score -= min(distance * 100, 50)
                elif risk_preference == "balanced":
                    # Cân bằng
                    pass
                else:  # aggressive
                    # Ưu tiên chốt lời xa hơn
                    score += min(distance * 50, 25)
                
                # Lưu điểm vào ứng viên
                candidate["score"] = score
            
            # Sắp xếp theo điểm
            valid_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Lấy ứng viên tốt nhất
            best_candidate = valid_candidates[0]
            
            # Thêm thông tin so sánh
            best_candidate["compared_candidates"] = len(valid_candidates)
            best_candidate["risk_preference"] = risk_preference
            
            self.logger.info(f"Đã chọn chốt lời tối ưu: {best_candidate.get('take_profit_type')} tại {best_candidate.get('take_profit_price', best_candidate.get('take_profit_prices', [0])[0]):.2f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chọn chốt lời tối ưu: {str(e)}")
            # Trả về ứng viên đầu tiên nếu có lỗi
            return valid_candidates[0] if valid_candidates else self.calculate_fixed_take_profit(entry_price, position_side)