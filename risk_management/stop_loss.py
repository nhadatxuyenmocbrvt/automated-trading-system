"""
Quản lý dừng lỗ.
File này định nghĩa các phương pháp để tính toán và quản lý dừng lỗ
cho các vị thế giao dịch, bao gồm dừng lỗ cố định, di động, và thích ứng.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import pandas as pd

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import ErrorCode

class StopLoss:
    """
    Lớp quản lý dừng lỗ.
    Cung cấp các phương pháp khác nhau để tính toán và cập nhật giá dừng lỗ
    nhằm bảo vệ lợi nhuận và giới hạn rủi ro.
    """
    
    def __init__(
        self,
        default_atr_multiplier: float = 2.0,
        default_fixed_percent: float = 0.02,
        default_trailing_percent: float = 0.01,
        default_risk_reward_ratio: float = 2.0,
        max_stop_loss_percent: float = 0.1,
        use_candle_low_high: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý dừng lỗ.
        
        Args:
            default_atr_multiplier: Hệ số nhân ATR mặc định
            default_fixed_percent: Phần trăm dừng lỗ cố định mặc định
            default_trailing_percent: Phần trăm dừng lỗ di động mặc định
            default_risk_reward_ratio: Tỷ lệ rủi ro/phần thưởng mặc định
            max_stop_loss_percent: Phần trăm dừng lỗ tối đa
            use_candle_low_high: Sử dụng giá cao/thấp của nến trong tính toán
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("stop_loss")
        self.default_atr_multiplier = default_atr_multiplier
        self.default_fixed_percent = default_fixed_percent
        self.default_trailing_percent = default_trailing_percent
        self.default_risk_reward_ratio = default_risk_reward_ratio
        self.max_stop_loss_percent = max_stop_loss_percent
        self.use_candle_low_high = use_candle_low_high
        
        self.logger.info(f"Đã khởi tạo StopLoss với default_fixed_percent={default_fixed_percent}, default_trailing_percent={default_trailing_percent}")
    
    def calculate_fixed_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        percent: Optional[float] = None,
        price_amount: Optional[float] = None,
        use_market_price: bool = False,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tính giá dừng lỗ cố định.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            percent: Phần trăm dừng lỗ (ưu tiên nếu được cung cấp)
            price_amount: Giá trị dừng lỗ tuyệt đối
            use_market_price: Sử dụng giá thị trường để điều chỉnh
            market_data: Dữ liệu thị trường nếu use_market_price=True
            
        Returns:
            Dict chứa thông tin dừng lỗ
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
                        price_to_use = market_data.get('low', None)
                    else:
                        price_to_use = market_data.get('high', None)
                
                if price_to_use is None:
                    price_to_use = market_data.get('close', entry_price)
                
                # Nếu là long, dừng lỗ dưới giá thị trường
                if position_side == "long" and price_to_use < entry_price:
                    entry_price = price_to_use
                # Nếu là short, dừng lỗ trên giá thị trường
                elif position_side == "short" and price_to_use > entry_price:
                    entry_price = price_to_use
            
            # Tính giá dừng lỗ
            if price_amount is not None:
                # Giá trị tuyệt đối
                if position_side == "long":
                    stop_loss_price = entry_price - price_amount
                else:  # short
                    stop_loss_price = entry_price + price_amount
                
                # Tính phần trăm tương ứng
                stop_loss_percent = price_amount / entry_price
            
            else:
                # Phần trăm
                if percent is None:
                    percent = self.default_fixed_percent
                
                # Giới hạn trong max_stop_loss_percent
                percent = min(percent, self.max_stop_loss_percent)
                
                if position_side == "long":
                    stop_loss_price = entry_price * (1 - percent)
                else:  # short
                    stop_loss_price = entry_price * (1 + percent)
                
                stop_loss_percent = percent
            
            result = {
                "status": "success",
                "stop_loss_price": stop_loss_price,
                "stop_loss_percent": stop_loss_percent,
                "stop_loss_type": "fixed",
                "position_side": position_side,
                "entry_price": entry_price,
            }
            
            self.logger.info(f"Đã tính dừng lỗ cố định cho {position_side}: {stop_loss_price:.2f} ({stop_loss_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ cố định: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ cố định: {str(e)}"
                }
            }
    
    def calculate_atr_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        atr_value: float,
        multiplier: Optional[float] = None,
        min_stop_loss_percent: float = 0.005,
        max_stop_loss_percent: Optional[float] = None,
        use_market_price: bool = False,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tính giá dừng lỗ dựa trên ATR (Average True Range).
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            atr_value: Giá trị ATR
            multiplier: Hệ số nhân ATR (mặc định: self.default_atr_multiplier)
            min_stop_loss_percent: Phần trăm dừng lỗ tối thiểu
            max_stop_loss_percent: Phần trăm dừng lỗ tối đa (mặc định: self.max_stop_loss_percent)
            use_market_price: Sử dụng giá thị trường để điều chỉnh
            market_data: Dữ liệu thị trường nếu use_market_price=True
            
        Returns:
            Dict chứa thông tin dừng lỗ dựa trên ATR
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
        
        if atr_value <= 0:
            self.logger.warning(f"Giá trị ATR không hợp lệ: {atr_value}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Giá trị ATR phải > 0"
                }
            }
        
        try:
            # Thiết lập các giá trị mặc định
            if multiplier is None:
                multiplier = self.default_atr_multiplier
                
            if max_stop_loss_percent is None:
                max_stop_loss_percent = self.max_stop_loss_percent
            
            # Nếu sử dụng giá thị trường
            if use_market_price and market_data:
                # Điều chỉnh dựa trên giá thị trường hiện tại
                price_to_use = None
                
                if self.use_candle_low_high and 'low' in market_data and 'high' in market_data:
                    if position_side == "long":
                        price_to_use = market_data.get('low', None)
                    else:
                        price_to_use = market_data.get('high', None)
                
                if price_to_use is None:
                    price_to_use = market_data.get('close', entry_price)
                
                # Nếu là long, dừng lỗ dưới giá thị trường
                if position_side == "long" and price_to_use < entry_price:
                    entry_price = price_to_use
                # Nếu là short, dừng lỗ trên giá thị trường
                elif position_side == "short" and price_to_use > entry_price:
                    entry_price = price_to_use
            
            # Tính khoảng cách dừng lỗ theo ATR
            atr_distance = atr_value * multiplier
            
            # Tính giá dừng lỗ
            if position_side == "long":
                stop_loss_price = entry_price - atr_distance
            else:  # short
                stop_loss_price = entry_price + atr_distance
            
            # Tính phần trăm dừng lỗ
            stop_loss_percent = atr_distance / entry_price
            
            # Đảm bảo dừng lỗ trong giới hạn
            if stop_loss_percent < min_stop_loss_percent:
                # Nếu phần trăm quá nhỏ, sử dụng min_stop_loss_percent
                if position_side == "long":
                    stop_loss_price = entry_price * (1 - min_stop_loss_percent)
                else:  # short
                    stop_loss_price = entry_price * (1 + min_stop_loss_percent)
                
                stop_loss_percent = min_stop_loss_percent
                atr_distance = entry_price * min_stop_loss_percent
            
            if stop_loss_percent > max_stop_loss_percent:
                # Nếu phần trăm quá lớn, sử dụng max_stop_loss_percent
                if position_side == "long":
                    stop_loss_price = entry_price * (1 - max_stop_loss_percent)
                else:  # short
                    stop_loss_price = entry_price * (1 + max_stop_loss_percent)
                
                stop_loss_percent = max_stop_loss_percent
                atr_distance = entry_price * max_stop_loss_percent
            
            result = {
                "status": "success",
                "stop_loss_price": stop_loss_price,
                "stop_loss_percent": stop_loss_percent,
                "stop_loss_type": "atr",
                "position_side": position_side,
                "entry_price": entry_price,
                "atr_value": atr_value,
                "atr_multiplier": multiplier,
                "atr_distance": atr_distance,
                "atr_units": atr_distance / atr_value  # Số ATR units
            }
            
            self.logger.info(f"Đã tính dừng lỗ ATR cho {position_side}: {stop_loss_price:.2f} ({stop_loss_percent:.2%}, {result['atr_units']:.2f} ATR)")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ ATR: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ ATR: {str(e)}"
                }
            }
    
    def calculate_chandelier_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        high_price: float,
        low_price: float,
        atr_value: float,
        multiplier: float = 3.0,
        lookback_periods: int = 22,
        use_entry_price: bool = False
    ) -> Dict[str, Any]:
        """
        Tính giá dừng lỗ Chandelier Exit.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            high_price: Giá cao nhất trong khoảng lookback_periods
            low_price: Giá thấp nhất trong khoảng lookback_periods
            atr_value: Giá trị ATR
            multiplier: Hệ số nhân ATR
            lookback_periods: Số kỳ nhìn lại
            use_entry_price: Sử dụng giá vào lệnh thay vì cao/thấp nhất
            
        Returns:
            Dict chứa thông tin dừng lỗ Chandelier
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
            # Tính Chandelier Exit
            if position_side == "long":
                # Cho long: Chandelier Exit = Highest High - ATR × Multiplier
                if use_entry_price:
                    base_price = entry_price
                else:
                    base_price = high_price
                
                chandelier_exit = base_price - (atr_value * multiplier)
                distance = base_price - chandelier_exit
            else:  # short
                # Cho short: Chandelier Exit = Lowest Low + ATR × Multiplier
                if use_entry_price:
                    base_price = entry_price
                else:
                    base_price = low_price
                
                chandelier_exit = base_price + (atr_value * multiplier)
                distance = chandelier_exit - base_price
            
            # Tính phần trăm dừng lỗ dựa trên entry_price
            stop_loss_percent = distance / entry_price
            
            # Giới hạn trong max_stop_loss_percent
            if stop_loss_percent > self.max_stop_loss_percent:
                if position_side == "long":
                    chandelier_exit = entry_price * (1 - self.max_stop_loss_percent)
                else:  # short
                    chandelier_exit = entry_price * (1 + self.max_stop_loss_percent)
                
                stop_loss_percent = self.max_stop_loss_percent
            
            result = {
                "status": "success",
                "stop_loss_price": chandelier_exit,
                "stop_loss_percent": stop_loss_percent,
                "stop_loss_type": "chandelier",
                "position_side": position_side,
                "entry_price": entry_price,
                "atr_value": atr_value,
                "atr_multiplier": multiplier,
                "base_price": base_price,
                "lookback_periods": lookback_periods,
                "atr_distance": atr_value * multiplier
            }
            
            self.logger.info(f"Đã tính dừng lỗ Chandelier cho {position_side}: {chandelier_exit:.2f} ({stop_loss_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ Chandelier: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ Chandelier: {str(e)}"
                }
            }
    
    def calculate_support_resistance_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        support_levels: List[float],
        resistance_levels: List[float],
        additional_buffer: float = 0.005,
        fallback_percent: float = 0.02
    ) -> Dict[str, Any]:
        """
        Tính giá dừng lỗ dựa trên các mức hỗ trợ/kháng cự.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            support_levels: Danh sách các mức hỗ trợ
            resistance_levels: Danh sách các mức kháng cự
            additional_buffer: Buffer bổ sung (%)
            fallback_percent: Phần trăm dự phòng nếu không tìm thấy mức hỗ trợ/kháng cự thích hợp
            
        Returns:
            Dict chứa thông tin dừng lỗ dựa trên hỗ trợ/kháng cự
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
            
            # Tìm mức dừng lỗ dựa trên hỗ trợ/kháng cự
            stop_loss_price = None
            
            if position_side == "long":
                # Cho long, tìm mức hỗ trợ gần nhất thấp hơn giá vào
                for level in reversed(support_levels):
                    if level < entry_price:
                        stop_loss_price = level * (1 - additional_buffer)
                        break
            else:  # short
                # Cho short, tìm mức kháng cự gần nhất cao hơn giá vào
                for level in resistance_levels:
                    if level > entry_price:
                        stop_loss_price = level * (1 + additional_buffer)
                        break
            
            # Nếu không tìm thấy mức thích hợp, sử dụng fallback
            if stop_loss_price is None:
                if position_side == "long":
                    stop_loss_price = entry_price * (1 - fallback_percent)
                else:  # short
                    stop_loss_price = entry_price * (1 + fallback_percent)
                
                self.logger.warning(f"Không tìm thấy mức hỗ trợ/kháng cự thích hợp, sử dụng fallback: {stop_loss_price:.2f}")
            
            # Tính phần trăm dừng lỗ
            if position_side == "long":
                stop_loss_percent = (entry_price - stop_loss_price) / entry_price
            else:  # short
                stop_loss_percent = (stop_loss_price - entry_price) / entry_price
            
            # Đảm bảo trong giới hạn max_stop_loss_percent
            if stop_loss_percent > self.max_stop_loss_percent:
                if position_side == "long":
                    stop_loss_price = entry_price * (1 - self.max_stop_loss_percent)
                else:  # short
                    stop_loss_price = entry_price * (1 + self.max_stop_loss_percent)
                
                stop_loss_percent = self.max_stop_loss_percent
            
            result = {
                "status": "success",
                "stop_loss_price": stop_loss_price,
                "stop_loss_percent": stop_loss_percent,
                "stop_loss_type": "support_resistance",
                "position_side": position_side,
                "entry_price": entry_price,
                "used_fallback": stop_loss_price is None,
                "additional_buffer": additional_buffer,
                "fallback_percent": fallback_percent
            }
            
            self.logger.info(f"Đã tính dừng lỗ S/R cho {position_side}: {stop_loss_price:.2f} ({stop_loss_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ S/R: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ S/R: {str(e)}"
                }
            }
    
    def calculate_trailing_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        position_side: str,
        highest_price: float,
        lowest_price: float,
        trailing_percent: Optional[float] = None,
        initial_stop_loss: Optional[float] = None,
        activation_percent: float = 0.01
    ) -> Dict[str, Any]:
        """
        Tính và cập nhật dừng lỗ di động (trailing stop loss).
        
        Args:
            current_price: Giá hiện tại
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            highest_price: Giá cao nhất kể từ khi vào lệnh
            lowest_price: Giá thấp nhất kể từ khi vào lệnh
            trailing_percent: Phần trăm di động
            initial_stop_loss: Giá dừng lỗ ban đầu (nếu có)
            activation_percent: Phần trăm để kích hoạt di động
            
        Returns:
            Dict chứa thông tin dừng lỗ di động
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
            
            # Tính dừng lỗ ban đầu nếu không được cung cấp
            if initial_stop_loss is None:
                if position_side == "long":
                    initial_stop_loss = entry_price * (1 - self.default_fixed_percent)
                else:  # short
                    initial_stop_loss = entry_price * (1 + self.default_fixed_percent)
            
            # Kiểm tra điều kiện kích hoạt
            activation_threshold = entry_price * (1 + activation_percent) if position_side == "long" else entry_price * (1 - activation_percent)
            is_activated = (position_side == "long" and highest_price >= activation_threshold) or (position_side == "short" and lowest_price <= activation_threshold)
            
            # Tính dừng lỗ di động
            trailing_stop = initial_stop_loss
            
            if is_activated:
                if position_side == "long":
                    # Cho long: trailing_stop = highest_price * (1 - trailing_percent)
                    new_stop = highest_price * (1 - trailing_percent)
                    # Chỉ cập nhật nếu dừng lỗ mới cao hơn
                    if new_stop > initial_stop_loss:
                        trailing_stop = new_stop
                else:  # short
                    # Cho short: trailing_stop = lowest_price * (1 + trailing_percent)
                    new_stop = lowest_price * (1 + trailing_percent)
                    # Chỉ cập nhật nếu dừng lỗ mới thấp hơn
                    if new_stop < initial_stop_loss:
                        trailing_stop = new_stop
            
            # Tính khoảng cách từ giá hiện tại đến dừng lỗ
            if position_side == "long":
                distance_percent = (current_price - trailing_stop) / current_price
            else:  # short
                distance_percent = (trailing_stop - current_price) / current_price
            
            # Tính lợi nhuận hiện tại
            if position_side == "long":
                current_profit_percent = (current_price - entry_price) / entry_price
            else:  # short
                current_profit_percent = (entry_price - current_price) / entry_price
            
            result = {
                "status": "success",
                "stop_loss_price": trailing_stop,
                "stop_loss_type": "trailing",
                "position_side": position_side,
                "entry_price": entry_price,
                "current_price": current_price,
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "trailing_percent": trailing_percent,
                "initial_stop_loss": initial_stop_loss,
                "distance_percent": distance_percent,
                "current_profit_percent": current_profit_percent,
                "is_activated": is_activated,
                "activation_threshold": activation_threshold
            }
            
            self.logger.info(f"Đã tính dừng lỗ di động cho {position_side}: {trailing_stop:.2f} (khoảng cách: {distance_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ di động: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ di động: {str(e)}"
                }
            }
    
    def calculate_time_based_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        entry_time: datetime,
        current_time: datetime,
        max_holding_time: int = 24,  # hours
        time_decay_factor: float = 0.5,
        initial_stop_percent: float = 0.02
    ) -> Dict[str, Any]:
        """
        Tính dừng lỗ dựa trên thời gian giữ lệnh.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            entry_time: Thời gian vào lệnh
            current_time: Thời gian hiện tại
            max_holding_time: Thời gian giữ lệnh tối đa (giờ)
            time_decay_factor: Hệ số suy giảm theo thời gian
            initial_stop_percent: Phần trăm dừng lỗ ban đầu
            
        Returns:
            Dict chứa thông tin dừng lỗ dựa trên thời gian
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
            # Tính thời gian đã giữ lệnh (giờ)
            holding_time = (current_time - entry_time).total_seconds() / 3600
            
            # Tính hệ số thời gian (time ratio)
            time_ratio = min(holding_time / max_holding_time, 1.0)
            
            # Tính phần trăm dừng lỗ theo công thức:
            # stop_percent = initial_stop_percent * (1 - time_decay_factor * time_ratio)
            stop_percent = initial_stop_percent * (1 - time_decay_factor * time_ratio)
            
            # Tính giá dừng lỗ
            if position_side == "long":
                stop_loss_price = entry_price * (1 - stop_percent)
            else:  # short
                stop_loss_price = entry_price * (1 + stop_percent)
            
            # Nếu đã quá max_holding_time, đề xuất đóng lệnh
            suggest_close = holding_time >= max_holding_time
            
            result = {
                "status": "success",
                "stop_loss_price": stop_loss_price,
                "stop_percent": stop_percent,
                "stop_loss_type": "time_based",
                "position_side": position_side,
                "entry_price": entry_price,
                "holding_time": holding_time,
                "time_ratio": time_ratio,
                "max_holding_time": max_holding_time,
                "suggest_close": suggest_close
            }
            
            self.logger.info(f"Đã tính dừng lỗ theo thời gian cho {position_side}: {stop_loss_price:.2f} ({stop_percent:.2%}, đã giữ: {holding_time:.1f}h/{max_holding_time}h)")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ theo thời gian: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ theo thời gian: {str(e)}"
                }
            }
    
    def calculate_rsi_based_stop_loss(
        self,
        entry_price: float,
        position_side: str,
        rsi_value: float,
        overbought_threshold: float = 70,
        oversold_threshold: float = 30,
        fallback_percent: float = 0.02
    ) -> Dict[str, Any]:
        """
        Tính dừng lỗ dựa trên giá trị RSI.
        
        Args:
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            rsi_value: Giá trị RSI hiện tại
            overbought_threshold: Ngưỡng quá mua
            oversold_threshold: Ngưỡng quá bán
            fallback_percent: Phần trăm dừng lỗ dự phòng
            
        Returns:
            Dict chứa thông tin dừng lỗ dựa trên RSI
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
            # Xác định tín hiệu RSI
            rsi_signal = None
            
            if rsi_value >= overbought_threshold:
                rsi_signal = "overbought"
            elif rsi_value <= oversold_threshold:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
            
            # Xác định nếu RSI đang chuyển thành tín hiệu dừng lỗ
            stop_triggered = False
            
            if position_side == "long" and rsi_signal == "overbought":
                stop_triggered = True
            elif position_side == "short" and rsi_signal == "oversold":
                stop_triggered = True
            
            # Tính phần trăm dừng lỗ dựa trên RSI
            if position_side == "long":
                # Long: RSI càng thấp, dừng lỗ càng thấp
                rsi_factor = min(rsi_value / 50, 1.2)  # Giới hạn rsi_factor
                stop_percent = fallback_percent / rsi_factor
            else:  # short
                # Short: RSI càng cao, dừng lỗ càng cao
                rsi_factor = min((100 - rsi_value) / 50, 1.2)  # Giới hạn rsi_factor
                stop_percent = fallback_percent / rsi_factor
            
            # Giới hạn trong max_stop_loss_percent
            stop_percent = min(stop_percent, self.max_stop_loss_percent)
            
            # Tính giá dừng lỗ
            if position_side == "long":
                stop_loss_price = entry_price * (1 - stop_percent)
            else:  # short
                stop_loss_price = entry_price * (1 + stop_percent)
            
            result = {
                "status": "success",
                "stop_loss_price": stop_loss_price,
                "stop_percent": stop_percent,
                "stop_loss_type": "rsi_based",
                "position_side": position_side,
                "entry_price": entry_price,
                "rsi_value": rsi_value,
                "rsi_signal": rsi_signal,
                "stop_triggered": stop_triggered,
                "rsi_factor": rsi_factor
            }
            
            self.logger.info(f"Đã tính dừng lỗ dựa trên RSI cho {position_side}: {stop_loss_price:.2f} ({stop_percent:.2%}, RSI: {rsi_value:.1f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính dừng lỗ dựa trên RSI: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính dừng lỗ dựa trên RSI: {str(e)}"
                }
            }
    
    def update_stop_loss(
        self,
        current_stop_loss: Dict[str, Any],
        current_price: float,
        highest_price: float,
        lowest_price: float,
        market_data: Optional[Dict[str, Any]] = None,
        rsi_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật dừng lỗ dựa trên điều kiện thị trường hiện tại.
        
        Args:
            current_stop_loss: Thông tin dừng lỗ hiện tại
            current_price: Giá hiện tại
            highest_price: Giá cao nhất kể từ khi vào lệnh
            lowest_price: Giá thấp nhất kể từ khi vào lệnh
            market_data: Dữ liệu thị trường bổ sung
            rsi_value: Giá trị RSI hiện tại
            
        Returns:
            Dict chứa thông tin dừng lỗ sau khi cập nhật
        """
        try:
            # Kiểm tra dữ liệu đầu vào
            if current_stop_loss.get("status") != "success":
                return current_stop_loss
            
            stop_loss_type = current_stop_loss.get("stop_loss_type")
            position_side = current_stop_loss.get("position_side")
            entry_price = current_stop_loss.get("entry_price")
            
            # Cập nhật dựa trên loại dừng lỗ
            if stop_loss_type == "trailing":
                # Cập nhật dừng lỗ di động
                return self.calculate_trailing_stop_loss(
                    current_price=current_price,
                    entry_price=entry_price,
                    position_side=position_side,
                    highest_price=highest_price,
                    lowest_price=lowest_price,
                    trailing_percent=current_stop_loss.get("trailing_percent"),
                    initial_stop_loss=current_stop_loss.get("initial_stop_loss")
                )
            
            elif stop_loss_type == "time_based":
                # Cập nhật dừng lỗ dựa trên thời gian
                entry_time = current_stop_loss.get("entry_time", datetime.now() - (datetime.now() - datetime.now()))
                
                return self.calculate_time_based_stop_loss(
                    entry_price=entry_price,
                    position_side=position_side,
                    entry_time=entry_time,
                    current_time=datetime.now(),
                    max_holding_time=current_stop_loss.get("max_holding_time", 24),
                    initial_stop_percent=current_stop_loss.get("initial_stop_percent", 0.02)
                )
            
            elif stop_loss_type == "rsi_based" and rsi_value is not None:
                # Cập nhật dừng lỗ dựa trên RSI
                return self.calculate_rsi_based_stop_loss(
                    entry_price=entry_price,
                    position_side=position_side,
                    rsi_value=rsi_value,
                    overbought_threshold=current_stop_loss.get("overbought_threshold", 70),
                    oversold_threshold=current_stop_loss.get("oversold_threshold", 30)
                )
            
            elif stop_loss_type == "chandelier":
                # Cập nhật Chandelier Exit
                atr_value = current_stop_loss.get("atr_value", 0)
                multiplier = current_stop_loss.get("atr_multiplier", 3.0)
                
                return self.calculate_chandelier_stop_loss(
                    entry_price=entry_price,
                    position_side=position_side,
                    high_price=highest_price,
                    low_price=lowest_price,
                    atr_value=atr_value,
                    multiplier=multiplier
                )
            
            else:
                # Các loại dừng lỗ khác không cần cập nhật
                return current_stop_loss
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật dừng lỗ: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi cập nhật dừng lỗ: {str(e)}"
                }
            }
    
    def choose_optimal_stop_loss(
        self,
        stop_loss_candidates: List[Dict[str, Any]],
        entry_price: float,
        position_side: str,
        minimize_distance: bool = False
    ) -> Dict[str, Any]:
        """
        Chọn dừng lỗ tối ưu từ nhiều ứng viên.
        
        Args:
            stop_loss_candidates: Danh sách các dừng lỗ ứng viên
            entry_price: Giá vào lệnh
            position_side: Phía vị thế ('long' hoặc 'short')
            minimize_distance: Ưu tiên khoảng cách gần nhất nếu True, xa nhất nếu False
            
        Returns:
            Dict chứa thông tin dừng lỗ tối ưu
        """
        position_side = position_side.lower()
        
        # Lọc các ứng viên hợp lệ
        valid_candidates = [sl for sl in stop_loss_candidates if sl.get("status") == "success"]
        
        if not valid_candidates:
            self.logger.warning("Không có ứng viên dừng lỗ hợp lệ")
            # Tạo dừng lỗ mặc định
            return self.calculate_fixed_stop_loss(
                entry_price=entry_price,
                position_side=position_side
            )
        
        try:
            # Tính khoảng cách dừng lỗ cho mỗi ứng viên
            for candidate in valid_candidates:
                stop_price = candidate.get("stop_loss_price")
                
                if position_side == "long":
                    distance = (entry_price - stop_price) / entry_price
                else:  # short
                    distance = (stop_price - entry_price) / entry_price
                
                candidate["distance_from_entry"] = distance
            
            # Sắp xếp theo khoảng cách
            if minimize_distance:
                # Ưu tiên khoảng cách nhỏ nhất (rủi ro thấp nhất)
                valid_candidates.sort(key=lambda x: x.get("distance_from_entry", float('inf')))
            else:
                # Ưu tiên khoảng cách lớn nhất (rủi ro cao nhất)
                valid_candidates.sort(key=lambda x: x.get("distance_from_entry", 0), reverse=True)
            
            # Lấy ứng viên tốt nhất
            best_candidate = valid_candidates[0]
            
            # Thêm thông tin so sánh
            best_candidate["compared_candidates"] = len(valid_candidates)
            best_candidate["minimized_distance"] = minimize_distance
            
            self.logger.info(f"Đã chọn dừng lỗ tối ưu: {best_candidate.get('stop_loss_type')} tại {best_candidate.get('stop_loss_price'):.2f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chọn dừng lỗ tối ưu: {str(e)}")
            # Trả về ứng viên đầu tiên nếu có lỗi
            return valid_candidates[0] if valid_candidates else self.calculate_fixed_stop_loss(entry_price, position_side)
        
class StopLossManager:
    """
    Quản lý các stop loss cho nhiều vị thế.
    Lớp này đóng gói các phương thức để theo dõi, cập nhật và kích hoạt dừng lỗ cho nhiều vị thế.
    """
    
    def __init__(self, logger=None):
        """
        Khởi tạo StopLossManager.
        
        Args:
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("stop_loss_manager")
        self.stop_loss_calculator = StopLoss()
        self.active_stop_losses = {}  # Dict với key là symbol_position_id
    
    def add_stop_loss(self, position: Dict[str, Any], 
                    stop_loss_price: float, 
                    stop_loss_type: str = "fixed") -> bool:
        """
        Thêm stop loss cho một vị thế.
        
        Args:
            position: Dict thông tin vị thế
            stop_loss_price: Giá dừng lỗ
            stop_loss_type: Loại dừng lỗ ('fixed', 'trailing', 'atr', etc.)
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        try:
            symbol = position.get('symbol', '')
            position_id = position.get('position_id', str(id(position)))
            key = f"{symbol}_{position_id}"
            
            # Kiểm tra vị thế hợp lệ
            if 'side' not in position or 'entry_price' not in position:
                self.logger.warning(f"Vị thế không chứa đủ thông tin: {position}")
                return False
            
            # Tạo đối tượng stop loss
            self.active_stop_losses[key] = {
                'price': stop_loss_price,
                'type': stop_loss_type,
                'symbol': symbol,
                'position_id': position_id,
                'side': position.get('side', ''),
                'entry_price': position.get('entry_price', 0),
                'creation_time': datetime.now().isoformat(),
                'last_update_time': datetime.now().isoformat(),
                'triggered': False
            }
            
            if stop_loss_type == 'trailing':
                # Thêm thông tin cho trailing stop
                self.active_stop_losses[key]['initial_price'] = stop_loss_price
                self.active_stop_losses[key]['highest_price'] = position.get('entry_price', 0)
                self.active_stop_losses[key]['lowest_price'] = position.get('entry_price', 0)
                
            self.logger.info(f"Đã thêm {stop_loss_type} stop loss cho {symbol} tại giá {stop_loss_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm stop loss: {str(e)}")
            return False
    
    def update_stop_loss(self, symbol: str, position_id: str, new_price: Optional[float] = None) -> bool:
        """
        Cập nhật stop loss hiện có.
        
        Args:
            symbol: Symbol của vị thế
            position_id: ID của vị thế
            new_price: Giá dừng lỗ mới
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        try:
            key = f"{symbol}_{position_id}"
            
            if key not in self.active_stop_losses:
                self.logger.warning(f"Không tìm thấy stop loss cho {key}")
                return False
            
            if new_price is not None:
                old_price = self.active_stop_losses[key]['price']
                self.active_stop_losses[key]['price'] = new_price
                self.active_stop_losses[key]['last_update_time'] = datetime.now().isoformat()
                
                self.logger.info(f"Đã cập nhật stop loss cho {symbol} từ {old_price} thành {new_price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật stop loss: {str(e)}")
            return False
    
    def remove_stop_loss(self, symbol: str, position_id: str) -> bool:
        """
        Xóa stop loss.
        
        Args:
            symbol: Symbol của vị thế
            position_id: ID của vị thế
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        try:
            key = f"{symbol}_{position_id}"
            
            if key not in self.active_stop_losses:
                self.logger.warning(f"Không tìm thấy stop loss cho {key}")
                return False
            
            del self.active_stop_losses[key]
            self.logger.info(f"Đã xóa stop loss cho {symbol}_{position_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa stop loss: {str(e)}")
            return False
    
    def check_stop_loss(self, position: Dict[str, Any], current_price: float) -> bool:
        """
        Kiểm tra xem stop loss có kích hoạt hay không.
        
        Args:
            position: Dict thông tin vị thế
            current_price: Giá hiện tại
            
        Returns:
            True nếu stop loss được kích hoạt, False nếu không
        """
        try:
            symbol = position.get('symbol', '')
            position_id = position.get('position_id', str(id(position)))
            key = f"{symbol}_{position_id}"
            
            if key not in self.active_stop_losses:
                return False
            
            stop_loss = self.active_stop_losses[key]
            
            # Kiểm tra nếu đã kích hoạt trước đó
            if stop_loss.get('triggered', False):
                return True
            
            side = position.get('side', '').lower()
            stop_price = stop_loss.get('price', 0)
            
            # Kiểm tra điều kiện kích hoạt
            triggered = False
            
            if side == 'long' and current_price <= stop_price:
                triggered = True
            elif side == 'short' and current_price >= stop_price:
                triggered = True
            
            if triggered:
                self.active_stop_losses[key]['triggered'] = True
                self.active_stop_losses[key]['trigger_price'] = current_price
                self.active_stop_losses[key]['trigger_time'] = datetime.now().isoformat()
                
                self.logger.info(f"Stop loss kích hoạt cho {symbol} tại giá {current_price}")
            
            return triggered
            
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra stop loss: {str(e)}")
            return False
    
    def update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> bool:
        """
        Cập nhật trailing stop dựa trên giá hiện tại.
        
        Args:
            position: Dict thông tin vị thế
            current_price: Giá hiện tại
            
        Returns:
            True nếu cập nhật thành công, False nếu không thay đổi
        """
        try:
            symbol = position.get('symbol', '')
            position_id = position.get('position_id', str(id(position)))
            key = f"{symbol}_{position_id}"
            
            if key not in self.active_stop_losses:
                return False
            
            stop_loss = self.active_stop_losses[key]
            
            # Chỉ cập nhật nếu là trailing stop
            if stop_loss.get('type', '') != 'trailing':
                return False
            
            side = position.get('side', '').lower()
            current_stop = stop_loss.get('price', 0)
            trailing_percent = position.get('trailing_stop_percent', 0.01)
            
            updated = False
            
            if side == 'long':
                # Cập nhật giá cao nhất
                highest_price = stop_loss.get('highest_price', 0)
                
                if current_price > highest_price:
                    stop_loss['highest_price'] = current_price
                    
                    # Tính trailing stop mới
                    new_stop = current_price * (1 - trailing_percent)
                    
                    # Cập nhật nếu stop mới cao hơn stop hiện tại
                    if new_stop > current_stop:
                        stop_loss['price'] = new_stop
                        stop_loss['last_update_time'] = datetime.now().isoformat()
                        updated = True
                        
                        self.logger.info(f"Đã cập nhật trailing stop cho {symbol} từ {current_stop} thành {new_stop}")
            
            elif side == 'short':
                # Cập nhật giá thấp nhất
                lowest_price = stop_loss.get('lowest_price', float('inf'))
                
                if current_price < lowest_price:
                    stop_loss['lowest_price'] = current_price
                    
                    # Tính trailing stop mới
                    new_stop = current_price * (1 + trailing_percent)
                    
                    # Cập nhật nếu stop mới thấp hơn stop hiện tại
                    if new_stop < current_stop:
                        stop_loss['price'] = new_stop
                        stop_loss['last_update_time'] = datetime.now().isoformat()
                        updated = True
                        
                        self.logger.info(f"Đã cập nhật trailing stop cho {symbol} từ {current_stop} thành {new_stop}")
            
            return updated
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật trailing stop: {str(e)}")
            return False
    
    def get_stop_loss(self, symbol: str, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin stop loss cho một vị thế cụ thể.
        
        Args:
            symbol: Symbol của vị thế
            position_id: ID của vị thế
            
        Returns:
            Dict thông tin stop loss hoặc None nếu không tìm thấy
        """
        key = f"{symbol}_{position_id}"
        return self.active_stop_losses.get(key)
    
    def calculate_optimal_stop_loss(self, position: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tính toán stop loss tối ưu cho một vị thế.
        
        Args:
            position: Dict thông tin vị thế
            market_data: Dict dữ liệu thị trường bổ sung
            
        Returns:
            Dict thông tin stop loss tối ưu
        """
        # Sử dụng StopLoss để tính toán
        side = position.get('side', '').lower()
        entry_price = position.get('entry_price', 0)
        
        if market_data and 'atr' in market_data:
            # Nếu có dữ liệu ATR, sử dụng ATR stop loss
            return self.stop_loss_calculator.calculate_atr_stop_loss(
                entry_price=entry_price,
                position_side=side,
                atr_value=market_data['atr'],
                multiplier=2.0
            )
        else:
            # Mặc định sử dụng fixed stop loss
            return self.stop_loss_calculator.calculate_fixed_stop_loss(
                entry_price=entry_price,
                position_side=side,
                percent=0.02
            )