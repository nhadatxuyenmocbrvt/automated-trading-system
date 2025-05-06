"""
Quản lý lệnh giao dịch.
File này cung cấp các chức năng để tạo, cập nhật, hủy và theo dõi lệnh giao dịch
trên các sàn giao dịch tiền điện tử khác nhau.
"""

import time
import logging
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import OrderType, TimeInForce, OrderStatus, PositionSide, ErrorCode
from config.utils.validators import is_valid_order_type, is_valid_trading_pair
from data_collectors.exchange_api.generic_connector import ExchangeConnector, APIError
from risk_management.position_sizer import PositionSizer
from risk_management.stop_loss import StopLoss
from risk_management.take_profit import TakeProfit

class OrderManager:
    """
    Quản lý lệnh giao dịch.
    Cung cấp các phương thức để tạo, cập nhật, hủy và theo dõi lệnh giao dịch 
    trên các sàn giao dịch tiền điện tử.
    """
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        position_sizer: Optional[PositionSizer] = None,
        stop_loss_manager: Optional[StopLoss] = None,
        take_profit_manager: Optional[TakeProfit] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_adjust_quantity: bool = True,
        default_time_in_force: TimeInForce = TimeInForce.GTC,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý lệnh giao dịch.
        
        Args:
            exchange_connector: Kết nối đến sàn giao dịch
            position_sizer: Quản lý kích thước vị thế (tùy chọn)
            stop_loss_manager: Quản lý dừng lỗ (tùy chọn)
            take_profit_manager: Quản lý chốt lời (tùy chọn)
            max_retries: Số lần thử lại tối đa khi gặp lỗi
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            auto_adjust_quantity: Tự động điều chỉnh số lượng theo quy định của sàn
            default_time_in_force: Hiệu lực thời gian mặc định cho lệnh
            logger: Logger tùy chỉnh
        """
        self.exchange = exchange_connector
        self.position_sizer = position_sizer
        self.stop_loss_manager = stop_loss_manager
        self.take_profit_manager = take_profit_manager
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_adjust_quantity = auto_adjust_quantity
        self.default_time_in_force = default_time_in_force
        self.logger = logger or get_logger("order_manager")
        
        # Cache cho trading rules của mỗi symbol
        self._trading_rules_cache = {}
        
        # Lưu trữ lệnh đang chờ xử lý theo ID
        self._pending_orders = {}
        
        # Lưu trữ lệnh đã thực hiện
        self._executed_orders = {}
        
        # Lưu trữ mối quan hệ giữa lệnh chính và lệnh TP/SL
        self._order_relations = {}
        
        self.logger.info(f"Đã khởi tạo OrderManager cho sàn {self.exchange.exchange_id}")
    
    def _get_symbol_rules(self, symbol: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Lấy quy tắc giao dịch cho một symbol.
        
        Args:
            symbol: Symbol cần lấy quy tắc
            force_update: Bắt buộc cập nhật từ sàn thay vì sử dụng cache
            
        Returns:
            Dict chứa quy tắc giao dịch
        """
        if symbol not in self._trading_rules_cache or force_update:
            try:
                # Lấy thông tin thị trường từ exchange_connector
                markets = self.exchange.fetch_markets(force_update=force_update)
                
                for market in markets:
                    if market['symbol'] == symbol:
                        # Tạo quy tắc giao dịch từ thông tin thị trường
                        rules = {
                            'symbol': market['symbol'],
                            'base': market.get('base', ''),
                            'quote': market.get('quote', ''),
                            'min_quantity': float(market.get('limits', {}).get('amount', {}).get('min', 0)),
                            'max_quantity': float(market.get('limits', {}).get('amount', {}).get('max', float('inf'))),
                            'quantity_step': float(market.get('precision', {}).get('amount', 0.00001)),
                            'min_price': float(market.get('limits', {}).get('price', {}).get('min', 0)),
                            'max_price': float(market.get('limits', {}).get('price', {}).get('max', float('inf'))),
                            'price_step': float(market.get('precision', {}).get('price', 0.00001)),
                            'min_notional': float(market.get('limits', {}).get('cost', {}).get('min', 0)),
                            'active': market.get('active', True)
                        }
                        
                        # Lưu vào cache
                        self._trading_rules_cache[symbol] = rules
                        self.logger.debug(f"Đã cập nhật quy tắc giao dịch cho {symbol}")
                        return rules
                
                # Nếu không tìm thấy symbol
                self.logger.warning(f"Không tìm thấy thông tin cho symbol {symbol}")
                return {}
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lấy quy tắc giao dịch cho {symbol}: {str(e)}")
                # Trả về cache nếu có
                return self._trading_rules_cache.get(symbol, {})
        
        return self._trading_rules_cache.get(symbol, {})
    
    def _adjust_quantity_and_price(
        self, 
        symbol: str, 
        quantity: float, 
        price: Optional[float] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Điều chỉnh số lượng và giá theo quy tắc giao dịch của sàn.
        
        Args:
            symbol: Symbol giao dịch
            quantity: Số lượng ban đầu
            price: Giá ban đầu (tùy chọn)
            
        Returns:
            Tuple (adjusted_quantity, adjusted_price)
        """
        if not self.auto_adjust_quantity:
            return quantity, price
        
        rules = self._get_symbol_rules(symbol)
        if not rules:
            self.logger.warning(f"Không tìm thấy quy tắc giao dịch cho {symbol}, không điều chỉnh")
            return quantity, price
        
        try:
            # Điều chỉnh số lượng
            min_quantity = rules.get('min_quantity', 0)
            max_quantity = rules.get('max_quantity', float('inf'))
            quantity_step = rules.get('quantity_step', 0.00001)
            
            # Đảm bảo số lượng không nhỏ hơn mức tối thiểu
            adjusted_quantity = max(min_quantity, quantity)
            
            # Đảm bảo số lượng không lớn hơn mức tối đa
            adjusted_quantity = min(max_quantity, adjusted_quantity)
            
            # Làm tròn theo quantity_step
            if quantity_step > 0:
                decimal_places = self._get_decimal_places(quantity_step)
                adjusted_quantity = self._round_down_to_step(adjusted_quantity, quantity_step, decimal_places)
            
            # Điều chỉnh giá nếu có
            adjusted_price = price
            if price is not None:
                min_price = rules.get('min_price', 0)
                max_price = rules.get('max_price', float('inf'))
                price_step = rules.get('price_step', 0.00001)
                
                # Đảm bảo giá không nhỏ hơn mức tối thiểu
                adjusted_price = max(min_price, price)
                
                # Đảm bảo giá không lớn hơn mức tối đa
                adjusted_price = min(max_price, adjusted_price)
                
                # Làm tròn theo price_step
                if price_step > 0:
                    decimal_places = self._get_decimal_places(price_step)
                    adjusted_price = self._round_down_to_step(adjusted_price, price_step, decimal_places)
            
            # Kiểm tra giá trị notional (quantity * price)
            min_notional = rules.get('min_notional', 0)
            if adjusted_price is not None and adjusted_quantity * adjusted_price < min_notional:
                # Tăng số lượng để đạt được min_notional
                required_quantity = min_notional / adjusted_price
                if required_quantity <= max_quantity:
                    decimal_places = self._get_decimal_places(quantity_step)
                    adjusted_quantity = self._round_down_to_step(required_quantity, quantity_step, decimal_places)
                    self.logger.info(f"Đã điều chỉnh số lượng từ {quantity} lên {adjusted_quantity} để đạt min_notional")
            
            if adjusted_quantity != quantity or (price is not None and adjusted_price != price):
                self.logger.info(f"Đã điều chỉnh lệnh: {quantity}->{adjusted_quantity}, {price}->{adjusted_price}")
            
            return adjusted_quantity, adjusted_price
            
        except Exception as e:
            self.logger.error(f"Lỗi khi điều chỉnh số lượng và giá: {str(e)}")
            return quantity, price
    
    def _get_decimal_places(self, step: float) -> int:
        """
        Lấy số chữ số thập phân từ step.
        
        Args:
            step: Step size
            
        Returns:
            Số chữ số thập phân
        """
        step_str = str(step)
        if '.' in step_str:
            return len(step_str.split('.')[1])
        return 0
    
    def _round_down_to_step(self, value: float, step: float, decimal_places: int) -> float:
        """
        Làm tròn xuống giá trị theo step.
        
        Args:
            value: Giá trị cần làm tròn
            step: Step size
            decimal_places: Số chữ số thập phân
            
        Returns:
            Giá trị đã làm tròn
        """
        multiplier = 10 ** decimal_places
        value_scaled = value * multiplier
        step_scaled = step * multiplier
        
        # Làm tròn xuống đến bội số của step
        result = int(value_scaled / step_scaled) * step_scaled / multiplier
        
        return result
    
    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,  # Giá tham chiếu, không sử dụng cho market order
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh market.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            price: Giá tham chiếu (tùy chọn, chỉ để tính toán)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET.value,
            quantity=quantity,
            price=None,  # Market order không cần giá
            reference_price=price,  # Lưu giá tham chiếu
            params=params
        )
    
    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: Optional[TimeInForce] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh limit.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            price: Giá
            time_in_force: Hiệu lực thời gian
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        order_params = params or {}
        
        # Thêm time_in_force nếu được cung cấp
        if time_in_force:
            order_params['timeInForce'] = self.exchange._time_in_force_map.get(
                time_in_force.value, 
                self.exchange._time_in_force_map.get(self.default_time_in_force.value)
            )
        
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT.value,
            quantity=quantity,
            price=price,
            params=order_params
        )
    
    def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        parent_order_id: Optional[str] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh stop loss.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            stop_price: Giá kích hoạt stop
            limit_price: Giá limit (nếu là stop-limit)
            parent_order_id: ID lệnh cha (nếu có)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        order_params = params or {}
        
        # Thêm stop_price vào params
        order_params['stopPrice'] = stop_price
        
        # Xác định loại lệnh (stop-loss hoặc stop-limit)
        order_type = OrderType.STOP_LOSS.value
        price = None
        
        if limit_price is not None:
            order_type = OrderType.STOP_LIMIT.value
            price = limit_price
        
        result = self._create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            params=order_params
        )
        
        # Lưu mối quan hệ với lệnh cha
        if parent_order_id and result["status"] == "success":
            order_id = result["order"]["id"]
            
            if parent_order_id not in self._order_relations:
                self._order_relations[parent_order_id] = {"stop_loss": [], "take_profit": []}
            
            self._order_relations[parent_order_id]["stop_loss"].append(order_id)
            
            self.logger.info(f"Đã liên kết lệnh stop loss {order_id} với lệnh cha {parent_order_id}")
        
        return result
    
    def create_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        limit_price: Optional[float] = None,
        parent_order_id: Optional[str] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh take profit.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            take_profit_price: Giá kích hoạt take profit
            limit_price: Giá limit (nếu là take-profit-limit)
            parent_order_id: ID lệnh cha (nếu có)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        order_params = params or {}
        
        # Thêm take_profit_price vào params
        order_params['stopPrice'] = take_profit_price
        
        # Xác định loại lệnh (take-profit hoặc take-profit-limit)
        order_type = OrderType.TAKE_PROFIT.value
        price = None
        
        if limit_price is not None:
            order_type = OrderType.TAKE_PROFIT.value  # Nhiều sàn không có take-profit-limit riêng
            price = limit_price
            order_params['type'] = 'TAKE_PROFIT_LIMIT'  # Ghi đè type cho một số sàn
        
        result = self._create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            params=order_params
        )
        
        # Lưu mối quan hệ với lệnh cha
        if parent_order_id and result["status"] == "success":
            order_id = result["order"]["id"]
            
            if parent_order_id not in self._order_relations:
                self._order_relations[parent_order_id] = {"stop_loss": [], "take_profit": []}
            
            self._order_relations[parent_order_id]["take_profit"].append(order_id)
            
            self.logger.info(f"Đã liên kết lệnh take profit {order_id} với lệnh cha {parent_order_id}")
        
        return result
    
    def create_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        activation_price: Optional[float] = None,
        callback_rate: float = 0.01,  # 1%
        parent_order_id: Optional[str] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh trailing stop.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            activation_price: Giá kích hoạt trailing stop
            callback_rate: Tỷ lệ callback (0.01 = 1%)
            parent_order_id: ID lệnh cha (nếu có)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        order_params = params or {}
        
        # Thêm các tham số trailing stop
        if activation_price:
            order_params['activationPrice'] = activation_price
        
        # Chuyển đổi callback_rate thành định dạng phù hợp
        # Một số sàn sử dụng phần trăm (ví dụ: 1.0 = 1%), một số sàn sử dụng tỷ lệ thập phân (0.01 = 1%)
        if 'callbackRate' in self.exchange.exchange_id:
            order_params['callbackRate'] = callback_rate * 100  # Chuyển thành phần trăm
        else:
            order_params['callback'] = callback_rate  # Giữ nguyên tỷ lệ thập phân
        
        result = self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.TRAILING_STOP.value,
            quantity=quantity,
            price=None,
            params=order_params
        )
        
        # Lưu mối quan hệ với lệnh cha
        if parent_order_id and result["status"] == "success":
            order_id = result["order"]["id"]
            
            if parent_order_id not in self._order_relations:
                self._order_relations[parent_order_id] = {"stop_loss": [], "take_profit": []}
            
            self._order_relations[parent_order_id]["stop_loss"].append(order_id)
            
            self.logger.info(f"Đã liên kết lệnh trailing stop {order_id} với lệnh cha {parent_order_id}")
        
        return result
    
    def _create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        reference_price: Optional[float] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Phương thức nội bộ để tạo lệnh.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            order_type: Loại lệnh
            quantity: Số lượng
            price: Giá
            reference_price: Giá tham chiếu (nếu khác với price)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh
        """
        # Kiểm tra tham số
        if not is_valid_trading_pair(symbol):
            self.logger.warning(f"Symbol không hợp lệ: {symbol}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Symbol không hợp lệ: {symbol}"
                }
            }
        
        if not is_valid_order_type(order_type):
            self.logger.warning(f"Loại lệnh không hợp lệ: {order_type}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Loại lệnh không hợp lệ: {order_type}"
                }
            }
        
        if side not in ["buy", "sell"]:
            self.logger.warning(f"Side không hợp lệ: {side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Side phải là 'buy' hoặc 'sell'"
                }
            }
        
        if quantity <= 0:
            self.logger.warning(f"Số lượng không hợp lệ: {quantity}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Số lượng phải > 0"
                }
            }
        
        try:
            # Điều chỉnh số lượng và giá theo quy tắc của sàn
            adjusted_quantity, adjusted_price = self._adjust_quantity_and_price(
                symbol=symbol,
                quantity=quantity,
                price=price
            )
            
            # Chuyển đổi loại lệnh sang định dạng của sàn
            exchange_order_type = self.exchange._order_type_map.get(order_type)
            if not exchange_order_type:
                self.logger.warning(f"Không thể ánh xạ loại lệnh {order_type} sang định dạng sàn")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Không hỗ trợ loại lệnh {order_type}"
                    }
                }
            
            # Khởi tạo thông tin lệnh
            order_info = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "exchange_type": exchange_order_type,
                "quantity": adjusted_quantity,
                "price": adjusted_price,
                "reference_price": reference_price or adjusted_price,
                "status": OrderStatus.PENDING.value,
                "params": params or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "retries": 0
            }
            
            # Gửi lệnh đến sàn với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API tạo lệnh của sàn
                    if exchange_order_type == 'market':
                        # Market order không cần giá
                        order = self.exchange.create_market_order(
                            symbol=symbol,
                            side=side,
                            amount=adjusted_quantity,
                            params=params or {}
                        )
                    else:
                        # Các loại lệnh khác cần giá
                        order = self.exchange.create_order(
                            symbol=symbol,
                            type=exchange_order_type,
                            side=side,
                            amount=adjusted_quantity,
                            price=adjusted_price,
                            params=params or {}
                        )
                    
                    # Xử lý kết quả thành công
                    order_info["exchange_order"] = order
                    order_info["order_id"] = order.get("id")
                    order_info["status"] = self._convert_exchange_status(order.get("status", ""))
                    order_info["filled"] = order.get("filled", 0)
                    order_info["remaining"] = order.get("remaining", adjusted_quantity)
                    order_info["cost"] = order.get("cost", 0)
                    order_info["fee"] = order.get("fee", {})
                    order_info["trades"] = order.get("trades", [])
                    order_info["updated_at"] = datetime.now().isoformat()
                    
                    # Lưu vào pending_orders nếu chưa hoàn thành
                    if order_info["status"] in [OrderStatus.PENDING.value, OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value]:
                        self._pending_orders[order_info["order_id"]] = order_info
                    
                    # Lưu vào executed_orders nếu đã hoàn thành
                    if order_info["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value]:
                        self._executed_orders[order_info["order_id"]] = order_info
                        # Xóa khỏi pending_orders nếu có
                        if order_info["order_id"] in self._pending_orders:
                            del self._pending_orders[order_info["order_id"]]
                    
                    self.logger.info(f"Đã tạo lệnh {order_type} {side} {symbol}: ID={order_info['order_id']}, status={order_info['status']}")
                    
                    return {
                        "status": "success",
                        "order": order_info
                    }
                    
                except APIError as e:
                    order_info["retries"] = attempt
                    order_info["last_error"] = str(e)
                    order_info["updated_at"] = datetime.now().isoformat()
                    
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi tạo lệnh (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi tạo lệnh: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
            
            # Nếu tất cả các lần thử đều thất bại
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể tạo lệnh sau {self.max_retries} lần thử: {order_info.get('last_error', 'Unknown error')}"
                },
                "order_info": order_info
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi tạo lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi tạo lệnh: {str(e)}"
                }
            }
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Hủy lệnh.
        
        Args:
            order_id: ID lệnh cần hủy
            symbol: Symbol giao dịch
            
        Returns:
            Kết quả hủy lệnh
        """
        try:
            # Kiểm tra order_id hợp lệ
            if not order_id:
                self.logger.warning("ID lệnh không hợp lệ")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": "ID lệnh không hợp lệ"
                    }
                }
            
            # Kiểm tra symbol hợp lệ
            if not is_valid_trading_pair(symbol):
                self.logger.warning(f"Symbol không hợp lệ: {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Symbol không hợp lệ: {symbol}"
                    }
                }
            
            # Lấy thông tin lệnh từ cache nếu có
            order_info = self._pending_orders.get(order_id, None)
            
            # Thử hủy lệnh với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API hủy lệnh của sàn
                    result = self.exchange.cancel_order(order_id, symbol)
                    
                    # Cập nhật thông tin lệnh
                    if order_info:
                        order_info["status"] = OrderStatus.CANCELED.value
                        order_info["updated_at"] = datetime.now().isoformat()
                        order_info["cancel_result"] = result
                        
                        # Chuyển từ pending sang executed
                        self._executed_orders[order_id] = order_info
                        if order_id in self._pending_orders:
                            del self._pending_orders[order_id]
                    
                    # Hủy các lệnh con (stop loss, take profit) nếu có
                    if order_id in self._order_relations:
                        related_orders = self._order_relations[order_id]
                        
                        # Hủy lệnh stop loss
                        for sl_id in related_orders.get("stop_loss", []):
                            self.logger.info(f"Hủy lệnh stop loss liên quan: {sl_id}")
                            if sl_id in self._pending_orders:
                                sl_symbol = self._pending_orders[sl_id]["symbol"]
                                self.cancel_order(sl_id, sl_symbol)
                        
                        # Hủy lệnh take profit
                        for tp_id in related_orders.get("take_profit", []):
                            self.logger.info(f"Hủy lệnh take profit liên quan: {tp_id}")
                            if tp_id in self._pending_orders:
                                tp_symbol = self._pending_orders[tp_id]["symbol"]
                                self.cancel_order(tp_id, tp_symbol)
                    
                    self.logger.info(f"Đã hủy lệnh {order_id} thành công")
                    
                    return {
                        "status": "success",
                        "order_id": order_id,
                        "cancel_result": result
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi hủy lệnh {order_id} (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Một số sàn báo lỗi nếu lệnh đã được thực hiện hoặc đã hủy
                    if "order not found" in str(e).lower() or "already" in str(e).lower():
                        # Lệnh có thể đã được thực hiện hoặc đã hủy
                        # Cập nhật trạng thái nếu có thông tin
                        if order_info:
                            self.logger.info(f"Lệnh {order_id} có thể đã được thực hiện hoặc đã hủy")
                            
                            # Lấy thông tin lệnh mới nhất
                            try:
                                updated_order = self.fetch_order(order_id, symbol)
                                if updated_order["status"] == "success":
                                    # Lệnh vẫn tồn tại, cập nhật trạng thái
                                    order_status = updated_order["order"]["status"]
                                    
                                    if order_status in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value]:
                                        # Lệnh đã hoàn thành hoặc đã hủy
                                        if order_info:
                                            order_info["status"] = order_status
                                            order_info["updated_at"] = datetime.now().isoformat()
                                            
                                            # Chuyển từ pending sang executed
                                            self._executed_orders[order_id] = order_info
                                            if order_id in self._pending_orders:
                                                del self._pending_orders[order_id]
                                        
                                        return {
                                            "status": "success",
                                            "order_id": order_id,
                                            "message": f"Lệnh đã ở trạng thái {order_status}"
                                        }
                            except:
                                # Không thể lấy thông tin lệnh, giả định đã hủy thành công
                                pass
                        
                        return {
                            "status": "success",
                            "order_id": order_id,
                            "message": "Lệnh có thể đã được thực hiện hoặc đã hủy"
                        }
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi hủy lệnh: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi hủy lệnh: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể hủy lệnh sau {self.max_retries} lần thử"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi hủy lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi hủy lệnh: {str(e)}"
                }
            }
    
    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Lấy thông tin lệnh.
        
        Args:
            order_id: ID lệnh
            symbol: Symbol giao dịch
            
        Returns:
            Thông tin lệnh
        """
        try:
            # Kiểm tra order_id hợp lệ
            if not order_id:
                self.logger.warning("ID lệnh không hợp lệ")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": "ID lệnh không hợp lệ"
                    }
                }
            
            # Kiểm tra symbol hợp lệ
            if not is_valid_trading_pair(symbol):
                self.logger.warning(f"Symbol không hợp lệ: {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Symbol không hợp lệ: {symbol}"
                    }
                }
            
            # Thử lấy thông tin lệnh với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API lấy thông tin lệnh của sàn
                    order_data = self.exchange.fetch_order(order_id, symbol)
                    
                    # Tạo thông tin lệnh từ dữ liệu sàn
                    order_info = {
                        "order_id": order_data.get("id"),
                        "symbol": order_data.get("symbol"),
                        "side": order_data.get("side"),
                        "type": self._convert_exchange_order_type(order_data.get("type")),
                        "exchange_type": order_data.get("type"),
                        "quantity": order_data.get("amount"),
                        "price": order_data.get("price"),
                        "status": self._convert_exchange_status(order_data.get("status", "")),
                        "filled": order_data.get("filled", 0),
                        "remaining": order_data.get("remaining", 0),
                        "cost": order_data.get("cost", 0),
                        "fee": order_data.get("fee", {}),
                        "trades": order_data.get("trades", []),
                        "updated_at": datetime.now().isoformat(),
                        "exchange_order": order_data
                    }
                    
                    # Cập nhật cache lệnh
                    if order_info["status"] in [OrderStatus.PENDING.value, OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value]:
                        self._pending_orders[order_id] = order_info
                    else:
                        # Lệnh đã hoàn thành, chuyển từ pending sang executed
                        self._executed_orders[order_id] = order_info
                        if order_id in self._pending_orders:
                            del self._pending_orders[order_id]
                    
                    return {
                        "status": "success",
                        "order": order_info
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi lấy thông tin lệnh {order_id} (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e) or "order not found" in str(e).lower():
                        self.logger.error(f"Lỗi không thể khắc phục khi lấy thông tin lệnh: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi lấy thông tin lệnh: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại, kiểm tra trong cache
            # Có thể lệnh đã được lưu trong cache từ trước
            if order_id in self._pending_orders:
                return {
                    "status": "success",
                    "order": self._pending_orders[order_id],
                    "source": "cache_pending"
                }
            
            if order_id in self._executed_orders:
                return {
                    "status": "success",
                    "order": self._executed_orders[order_id],
                    "source": "cache_executed"
                }
            
            # Không tìm thấy lệnh
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.DATA_NOT_FOUND.value,
                    "message": f"Không tìm thấy lệnh {order_id}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi lấy thông tin lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi lấy thông tin lệnh: {str(e)}"
                }
            }
    
    def fetch_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy danh sách lệnh đang mở.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            
        Returns:
            Danh sách lệnh đang mở
        """
        try:
            # Thử lấy danh sách lệnh đang mở với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API lấy danh sách lệnh đang mở của sàn
                    open_orders = self.exchange.fetch_open_orders(symbol)
                    
                    # Chuyển đổi định dạng lệnh
                    converted_orders = []
                    for order_data in open_orders:
                        order_info = {
                            "order_id": order_data.get("id"),
                            "symbol": order_data.get("symbol"),
                            "side": order_data.get("side"),
                            "type": self._convert_exchange_order_type(order_data.get("type")),
                            "exchange_type": order_data.get("type"),
                            "quantity": order_data.get("amount"),
                            "price": order_data.get("price"),
                            "status": self._convert_exchange_status(order_data.get("status", "")),
                            "filled": order_data.get("filled", 0),
                            "remaining": order_data.get("remaining", 0),
                            "cost": order_data.get("cost", 0),
                            "fee": order_data.get("fee", {}),
                            "trades": order_data.get("trades", []),
                            "updated_at": datetime.now().isoformat(),
                            "exchange_order": order_data
                        }
                        
                        # Cập nhật cache lệnh
                        self._pending_orders[order_info["order_id"]] = order_info
                        
                        converted_orders.append(order_info)
                    
                    # Xóa lệnh đã đóng khỏi pending_orders
                    orders_to_remove = []
                    pending_ids = set(self._pending_orders.keys())
                    open_ids = set(order["order_id"] for order in converted_orders)
                    
                    for order_id in pending_ids:
                        if order_id not in open_ids:
                            order_info = self._pending_orders[order_id]
                            # Kiểm tra xem đã được lấy trạng thái mới nhất chưa
                            if "symbol" in order_info:
                                try:
                                    # Lấy thông tin lệnh mới nhất để xác nhận trạng thái
                                    updated_order = self.fetch_order(order_id, order_info["symbol"])
                                    if updated_order["status"] == "success":
                                        # Lệnh vẫn tồn tại, nhưng không còn mở
                                        # Đã được xử lý trong fetch_order
                                        pass
                                    else:
                                        # Không lấy được thông tin, đánh dấu để xóa
                                        orders_to_remove.append(order_id)
                                except:
                                    # Lỗi khi lấy thông tin lệnh, đánh dấu để xóa
                                    orders_to_remove.append(order_id)
                            else:
                                # Không có thông tin symbol, đánh dấu để xóa
                                orders_to_remove.append(order_id)
                    
                    # Xóa các lệnh đã đánh dấu
                    for order_id in orders_to_remove:
                        if order_id in self._pending_orders:
                            # Chuyển sang executed_orders
                            self._executed_orders[order_id] = self._pending_orders[order_id]
                            del self._pending_orders[order_id]
                    
                    self.logger.info(f"Đã lấy {len(converted_orders)} lệnh đang mở" + (f" cho {symbol}" if symbol else ""))
                    
                    return {
                        "status": "success",
                        "orders": converted_orders,
                        "count": len(converted_orders)
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi lấy danh sách lệnh đang mở (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi lấy danh sách lệnh đang mở: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi lấy danh sách lệnh đang mở: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại, trả về từ cache
            # Lọc các lệnh đang mở từ cache theo symbol
            cached_orders = []
            for order_id, order_info in self._pending_orders.items():
                if symbol is None or order_info.get("symbol") == symbol:
                    cached_orders.append(order_info)
            
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể lấy danh sách lệnh đang mở từ sàn sau {self.max_retries} lần thử"
                },
                "cached_orders": cached_orders,
                "count": len(cached_orders),
                "source": "cache"
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi lấy danh sách lệnh đang mở: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi lấy danh sách lệnh đang mở: {str(e)}"
                }
            }
    
    def fetch_closed_orders(
        self, 
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Lấy danh sách lệnh đã đóng.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng lệnh tối đa
            
        Returns:
            Danh sách lệnh đã đóng
        """
        try:
            # Thử lấy danh sách lệnh đã đóng với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API lấy danh sách lệnh đã đóng của sàn
                    closed_orders = self.exchange.fetch_closed_orders(symbol, since, limit)
                    
                    # Chuyển đổi định dạng lệnh
                    converted_orders = []
                    for order_data in closed_orders:
                        order_info = {
                            "order_id": order_data.get("id"),
                            "symbol": order_data.get("symbol"),
                            "side": order_data.get("side"),
                            "type": self._convert_exchange_order_type(order_data.get("type")),
                            "exchange_type": order_data.get("type"),
                            "quantity": order_data.get("amount"),
                            "price": order_data.get("price"),
                            "status": self._convert_exchange_status(order_data.get("status", "")),
                            "filled": order_data.get("filled", 0),
                            "remaining": order_data.get("remaining", 0),
                            "cost": order_data.get("cost", 0),
                            "fee": order_data.get("fee", {}),
                            "trades": order_data.get("trades", []),
                            "updated_at": datetime.now().isoformat(),
                            "exchange_order": order_data
                        }
                        
                        # Cập nhật cache lệnh
                        self._executed_orders[order_info["order_id"]] = order_info
                        # Xóa khỏi pending_orders nếu có
                        if order_info["order_id"] in self._pending_orders:
                            del self._pending_orders[order_info["order_id"]]
                        
                        converted_orders.append(order_info)
                    
                    self.logger.info(f"Đã lấy {len(converted_orders)} lệnh đã đóng" + (f" cho {symbol}" if symbol else ""))
                    
                    return {
                        "status": "success",
                        "orders": converted_orders,
                        "count": len(converted_orders)
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi lấy danh sách lệnh đã đóng (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi lấy danh sách lệnh đã đóng: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi lấy danh sách lệnh đã đóng: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại, trả về từ cache
            # Lọc các lệnh đã đóng từ cache theo symbol
            cached_orders = []
            for order_id, order_info in self._executed_orders.items():
                if symbol is None or order_info.get("symbol") == symbol:
                    cached_orders.append(order_info)
            
            # Lọc theo thời gian nếu có
            if since is not None:
                cached_orders = [
                    order for order in cached_orders 
                    if "exchange_order" in order and order["exchange_order"].get("timestamp", 0) >= since
                ]
            
            # Giới hạn số lượng nếu có
            if limit is not None:
                cached_orders = cached_orders[:limit]
            
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể lấy danh sách lệnh đã đóng từ sàn sau {self.max_retries} lần thử"
                },
                "cached_orders": cached_orders,
                "count": len(cached_orders),
                "source": "cache"
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi lấy danh sách lệnh đã đóng: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi lấy danh sách lệnh đã đóng: {str(e)}"
                }
            }
    
    def fetch_my_trades(
        self, 
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Lấy lịch sử giao dịch của tài khoản.
        
        Args:
            symbol: Symbol giao dịch (tùy chọn)
            since: Thời gian bắt đầu tính từ millisecond epoch
            limit: Số lượng giao dịch tối đa
            
        Returns:
            Lịch sử giao dịch
        """
        try:
            # Thử lấy lịch sử giao dịch với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API lấy lịch sử giao dịch của sàn
                    trades = self.exchange.fetch_my_trades(symbol, since, limit)
                    
                    # Chuyển đổi định dạng giao dịch
                    converted_trades = []
                    for trade_data in trades:
                        trade_info = {
                            "trade_id": trade_data.get("id"),
                            "order_id": trade_data.get("order"),
                            "symbol": trade_data.get("symbol"),
                            "side": trade_data.get("side"),
                            "price": trade_data.get("price"),
                            "amount": trade_data.get("amount"),
                            "cost": trade_data.get("cost"),
                            "fee": trade_data.get("fee"),
                            "timestamp": trade_data.get("timestamp"),
                            "datetime": trade_data.get("datetime"),
                            "exchange_trade": trade_data
                        }
                        
                        converted_trades.append(trade_info)
                    
                    self.logger.info(f"Đã lấy {len(converted_trades)} giao dịch" + (f" cho {symbol}" if symbol else ""))
                    
                    return {
                        "status": "success",
                        "trades": converted_trades,
                        "count": len(converted_trades)
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi lấy lịch sử giao dịch (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi lấy lịch sử giao dịch: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi lấy lịch sử giao dịch: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể lấy lịch sử giao dịch sau {self.max_retries} lần thử"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi lấy lịch sử giao dịch: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi lấy lịch sử giao dịch: {str(e)}"
                }
            }
    
    def create_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh OCO (One-Cancels-the-Other).
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            quantity: Số lượng
            price: Giá limit cho lệnh limit
            stop_price: Giá kích hoạt cho lệnh stop
            stop_limit_price: Giá limit cho lệnh stop (nếu là stop-limit)
            params: Tham số bổ sung
            
        Returns:
            Thông tin lệnh OCO
        """
        order_params = params or {}
        
        # Thêm các tham số OCO
        order_params['stopPrice'] = stop_price
        
        if stop_limit_price is not None:
            order_params['stopLimitPrice'] = stop_limit_price
        
        try:
            # Điều chỉnh số lượng theo quy tắc của sàn
            adjusted_quantity, adjusted_price = self._adjust_quantity_and_price(
                symbol=symbol,
                quantity=quantity,
                price=price
            )
            
            # Gọi API tạo lệnh OCO của sàn
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API tạo lệnh OCO
                    if 'binance' in self.exchange.exchange_id:
                        # Binance có API riêng cho OCO
                        result = self.exchange.create_order_oco(
                            symbol=symbol,
                            side=side,
                            amount=adjusted_quantity,
                            price=adjusted_price,
                            stopPrice=stop_price,
                            stopLimitPrice=stop_limit_price or stop_price,
                            params=order_params
                        )
                    else:
                        # Sàn khác không hỗ trợ OCO, tạo 2 lệnh riêng biệt
                        self.logger.warning(f"Sàn {self.exchange.exchange_id} không hỗ trợ OCO trực tiếp, tạo 2 lệnh riêng biệt")
                        
                        # Tạo lệnh limit
                        limit_result = self.create_limit_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            price=price
                        )
                        
                        # Tạo lệnh stop (nếu lệnh limit thành công)
                        if limit_result["status"] == "success":
                            limit_order_id = limit_result["order"]["order_id"]
                            
                            # Xác định loại lệnh stop
                            if stop_limit_price is not None:
                                stop_result = self.create_stop_loss_order(
                                    symbol=symbol,
                                    side=side,
                                    quantity=quantity,
                                    stop_price=stop_price,
                                    limit_price=stop_limit_price,
                                    parent_order_id=limit_order_id
                                )
                            else:
                                stop_result = self.create_stop_loss_order(
                                    symbol=symbol,
                                    side=side,
                                    quantity=quantity,
                                    stop_price=stop_price,
                                    parent_order_id=limit_order_id
                                )
                            
                            # Trả về kết quả
                            return {
                                "status": "success",
                                "type": "manual_oco",
                                "limit_order": limit_result["order"],
                                "stop_order": stop_result.get("order"),
                                "message": "Đã tạo 2 lệnh riêng biệt thay cho OCO"
                            }
                        else:
                            # Lệnh limit thất bại
                            return limit_result
                    
                    # Xử lý kết quả từ Binance OCO API
                    self.logger.info(f"Đã tạo lệnh OCO thành công cho {symbol}")
                    
                    # Cấu trúc kết quả OCO của Binance
                    return {
                        "status": "success",
                        "type": "native_oco",
                        "oco_result": result,
                        "orders": result.get("orders", []),
                        "order_ids": result.get("orderIds", [])
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi tạo lệnh OCO (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e):
                        self.logger.error(f"Lỗi không thể khắc phục khi tạo lệnh OCO: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
            
            # Nếu tất cả các lần thử đều thất bại
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể tạo lệnh OCO sau {self.max_retries} lần thử"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi tạo lệnh OCO: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi tạo lệnh OCO: {str(e)}"
                }
            }
    
    def create_order_with_tp_sl(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        sl_order_type: str = OrderType.STOP_LOSS.value,
        tp_order_type: str = OrderType.TAKE_PROFIT.value,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Tạo lệnh chính kèm theo stop loss và take profit.
        
        Args:
            symbol: Symbol giao dịch
            side: 'buy' hoặc 'sell'
            order_type: Loại lệnh chính
            quantity: Số lượng
            price: Giá (cho lệnh limit)
            stop_loss_price: Giá dừng lỗ
            take_profit_price: Giá chốt lời
            sl_order_type: Loại lệnh dừng lỗ
            tp_order_type: Loại lệnh chốt lời
            params: Tham số bổ sung
            
        Returns:
            Dict chứa thông tin lệnh chính và các lệnh TP/SL
        """
        # Kiểm tra tham số
        if not is_valid_trading_pair(symbol):
            self.logger.warning(f"Symbol không hợp lệ: {symbol}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Symbol không hợp lệ: {symbol}"
                }
            }
        
        if not is_valid_order_type(order_type):
            self.logger.warning(f"Loại lệnh không hợp lệ: {order_type}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Loại lệnh không hợp lệ: {order_type}"
                }
            }
        
        if side not in ["buy", "sell"]:
            self.logger.warning(f"Side không hợp lệ: {side}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Side phải là 'buy' hoặc 'sell'"
                }
            }
        
        try:
            # 1. Tạo lệnh chính
            if order_type == OrderType.MARKET.value:
                main_order_result = self.create_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,  # Dùng làm giá tham chiếu
                    params=params
                )
            elif order_type == OrderType.LIMIT.value:
                if price is None:
                    self.logger.warning("Giá không được cung cấp cho lệnh limit")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": f"Giá phải được cung cấp cho lệnh limit"
                        }
                    }
                    
                main_order_result = self.create_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    params=params
                )
            else:
                self.logger.warning(f"Loại lệnh không được hỗ trợ cho lệnh chính: {order_type}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Loại lệnh không được hỗ trợ cho lệnh chính: {order_type}"
                    }
                }
            
            # Kiểm tra lệnh chính thành công
            if main_order_result["status"] != "success":
                self.logger.warning(f"Tạo lệnh chính thất bại: {main_order_result.get('error', {}).get('message', 'Unknown error')}")
                return main_order_result
            
            # Lấy thông tin lệnh chính
            main_order = main_order_result["order"]
            main_order_id = main_order["order_id"]
            
            # Xác định side ngược lại cho TP/SL
            opposite_side = "sell" if side == "buy" else "buy"
            
            # Kiểm tra lệnh đã được thực hiện chưa
            is_filled = main_order["status"] == OrderStatus.FILLED.value
            
            # Nếu lệnh market đã được thực hiện (filled), chúng ta có thể tạo TP/SL
            sl_result = None
            tp_result = None
            
            # 2. Tạo lệnh stop loss nếu có
            if stop_loss_price is not None:
                if is_filled or order_type == OrderType.LIMIT.value:
                    # Nếu lệnh chính đã filled hoặc là lệnh limit, tạo lệnh SL
                    sl_result = self.create_stop_loss_order(
                        symbol=symbol,
                        side=opposite_side,
                        quantity=quantity,
                        stop_price=stop_loss_price,
                        parent_order_id=main_order_id
                    )
                    
                    if sl_result["status"] != "success":
                        self.logger.warning(f"Tạo lệnh stop loss thất bại: {sl_result.get('error', {}).get('message', 'Unknown error')}")
            
            # 3. Tạo lệnh take profit nếu có
            if take_profit_price is not None:
                if is_filled or order_type == OrderType.LIMIT.value:
                    # Nếu lệnh chính đã filled hoặc là lệnh limit, tạo lệnh TP
                    tp_result = self.create_take_profit_order(
                        symbol=symbol,
                        side=opposite_side,
                        quantity=quantity,
                        take_profit_price=take_profit_price,
                        parent_order_id=main_order_id
                    )
                    
                    if tp_result["status"] != "success":
                        self.logger.warning(f"Tạo lệnh take profit thất bại: {tp_result.get('error', {}).get('message', 'Unknown error')}")
            
            # Kết quả
            result = {
                "status": "success",
                "main_order": main_order,
                "main_order_id": main_order_id,
                "is_filled": is_filled,
                "has_stop_loss": stop_loss_price is not None,
                "has_take_profit": take_profit_price is not None,
                "stop_loss_created": sl_result is not None and sl_result["status"] == "success",
                "take_profit_created": tp_result is not None and tp_result["status"] == "success"
            }
            
            # Thêm thông tin TP/SL nếu đã tạo
            if sl_result is not None and sl_result["status"] == "success":
                result["stop_loss_order"] = sl_result["order"]
                result["stop_loss_order_id"] = sl_result["order"]["order_id"]
            
            if tp_result is not None and tp_result["status"] == "success":
                result["take_profit_order"] = tp_result["order"]
                result["take_profit_order_id"] = tp_result["order"]["order_id"]
            
            self.logger.info(f"Đã tạo lệnh {order_type} {side} {symbol} kèm TP/SL: ID={main_order_id}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi tạo lệnh với TP/SL: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi tạo lệnh với TP/SL: {str(e)}"
                }
            }
    
    def update_order(
        self,
        order_id: str,
        symbol: str,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật lệnh đang mở.
        
        Args:
            order_id: ID lệnh
            symbol: Symbol giao dịch
            price: Giá mới (tùy chọn)
            quantity: Số lượng mới (tùy chọn)
            params: Tham số bổ sung
            
        Returns:
            Dict chứa thông tin lệnh sau khi cập nhật
        """
        try:
            # Kiểm tra tham số
            if not order_id:
                self.logger.warning("ID lệnh không hợp lệ")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": "ID lệnh không hợp lệ"
                    }
                }
            
            if not is_valid_trading_pair(symbol):
                self.logger.warning(f"Symbol không hợp lệ: {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Symbol không hợp lệ: {symbol}"
                    }
                }
            
            # Điều chỉnh giá và số lượng nếu được cung cấp
            adjusted_quantity = quantity
            adjusted_price = price
            
            if quantity is not None and price is not None:
                adjusted_quantity, adjusted_price = self._adjust_quantity_and_price(
                    symbol=symbol,
                    quantity=quantity,
                    price=price
                )
            elif quantity is not None:
                adjusted_quantity, _ = self._adjust_quantity_and_price(
                    symbol=symbol,
                    quantity=quantity,
                    price=None
                )
            elif price is not None:
                _, adjusted_price = self._adjust_quantity_and_price(
                    symbol=symbol,
                    quantity=0,  # Giá trị giả để tránh lỗi
                    price=price
                )
            
            # Thử cập nhật lệnh với số lần thử lại
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Gọi API cập nhật lệnh của sàn
                    result = self.exchange.edit_order(
                        id=order_id,
                        symbol=symbol,
                        type=None,  # Giữ nguyên loại lệnh
                        side=None,  # Giữ nguyên side
                        amount=adjusted_quantity,
                        price=adjusted_price,
                        params=params or {}
                    )
                    
                    # Cập nhật thông tin lệnh trong cache
                    if order_id in self._pending_orders:
                        order_info = self._pending_orders[order_id]
                        
                        if adjusted_quantity is not None:
                            order_info["quantity"] = adjusted_quantity
                        
                        if adjusted_price is not None:
                            order_info["price"] = adjusted_price
                        
                        order_info["updated_at"] = datetime.now().isoformat()
                        order_info["exchange_order"] = result
                    
                    self.logger.info(f"Đã cập nhật lệnh {order_id}: price={adjusted_price}, quantity={adjusted_quantity}")
                    
                    return {
                        "status": "success",
                        "order_id": order_id,
                        "updated_price": adjusted_price,
                        "updated_quantity": adjusted_quantity,
                        "update_result": result
                    }
                    
                except APIError as e:
                    # Ghi log lỗi
                    self.logger.warning(f"Lỗi khi cập nhật lệnh {order_id} (lần {attempt}/{self.max_retries}): {str(e)}")
                    
                    # Kiểm tra lỗi không thể khắc phục
                    if self._is_fatal_error(e) or "cannot be modified" in str(e).lower():
                        self.logger.error(f"Lỗi không thể khắc phục khi cập nhật lệnh: {str(e)}")
                        break
                    
                    # Chờ trước khi thử lại
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)  # Tăng thời gian chờ theo số lần thử
                except Exception as e:
                    self.logger.error(f"Lỗi không mong đợi khi cập nhật lệnh: {str(e)}")
                    break
            
            # Nếu tất cả các lần thử đều thất bại
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.API_ERROR.value,
                    "message": f"Không thể cập nhật lệnh sau {self.max_retries} lần thử"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi cập nhật lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi cập nhật lệnh: {str(e)}"
                }
            }
    
    def update_all_pending_orders(self) -> Dict[str, Any]:
        """
        Cập nhật trạng thái của tất cả các lệnh đang chờ.
        
        Returns:
            Dict chứa thông tin cập nhật
        """
        try:
            # Lấy danh sách ID lệnh đang chờ
            pending_order_ids = list(self._pending_orders.keys())
            
            if not pending_order_ids:
                return {
                    "status": "success",
                    "message": "Không có lệnh đang chờ cần cập nhật",
                    "count": 0
                }
            
            # Lấy danh sách lệnh đang mở từ sàn
            open_orders_result = self.fetch_open_orders()
            
            # Cập nhật thông tin
            updated_count = 0
            unchanged_count = 0
            error_count = 0
            
            # Danh sách lệnh cần kiểm tra trực tiếp (không có trong danh sách lệnh đang mở)
            orders_to_check = []
            
            if open_orders_result["status"] == "success":
                # Lấy danh sách lệnh đang mở từ kết quả
                open_orders = open_orders_result["orders"]
                
                # Tạo map từ order_id đến order
                open_order_map = {order["order_id"]: order for order in open_orders}
                
                # Cập nhật các lệnh trong cache từ danh sách lệnh đang mở
                for order_id in pending_order_ids:
                    if order_id in open_order_map:
                        # Lệnh vẫn đang mở, cập nhật thông tin
                        self._pending_orders[order_id] = open_order_map[order_id]
                        updated_count += 1
                    else:
                        # Lệnh không còn trong danh sách lệnh đang mở
                        # Cần kiểm tra trực tiếp
                        orders_to_check.append(order_id)
            else:
                # Không lấy được danh sách lệnh đang mở, kiểm tra tất cả
                orders_to_check = pending_order_ids
            
            # Kiểm tra từng lệnh
            for order_id in orders_to_check:
                order_info = self._pending_orders[order_id]
                symbol = order_info.get("symbol")
                
                if not symbol:
                    self.logger.warning(f"Không có thông tin symbol cho lệnh {order_id}")
                    error_count += 1
                    continue
                
                # Lấy thông tin lệnh từ sàn
                order_result = self.fetch_order(order_id, symbol)
                
                if order_result["status"] == "success":
                    order_status = order_result["order"]["status"]
                    
                    if order_status in [OrderStatus.FILLED.value, OrderStatus.CANCELED.value, OrderStatus.REJECTED.value, OrderStatus.EXPIRED.value]:
                        # Lệnh đã hoàn thành, chuyển từ pending sang executed
                        self._executed_orders[order_id] = order_result["order"]
                        del self._pending_orders[order_id]
                        updated_count += 1
                    elif order_status in [OrderStatus.PENDING.value, OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value]:
                        # Lệnh vẫn đang chờ, cập nhật thông tin
                        self._pending_orders[order_id] = order_result["order"]
                        updated_count += 1
                    else:
                        # Trạng thái không thay đổi
                        unchanged_count += 1
                else:
                    # Không lấy được thông tin lệnh
                    self.logger.warning(f"Không lấy được thông tin lệnh {order_id}: {order_result.get('error', {}).get('message', 'Unknown error')}")
                    error_count += 1
            
            return {
                "status": "success",
                "total": len(pending_order_ids),
                "updated": updated_count,
                "unchanged": unchanged_count,
                "errors": error_count,
                "pending_count": len(self._pending_orders),
                "executed_count": len(self._executed_orders)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi không mong đợi khi cập nhật tất cả lệnh đang chờ: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi không mong đợi khi cập nhật tất cả lệnh đang chờ: {str(e)}"
                }
            }
    
    def _convert_exchange_status(self, exchange_status: str) -> str:
        """
        Chuyển đổi trạng thái lệnh từ định dạng sàn sang định dạng chuẩn.
        
        Args:
            exchange_status: Trạng thái từ sàn
            
        Returns:
            Trạng thái chuẩn
        """
        exchange_status = exchange_status.lower()
        
        # Ánh xạ trạng thái
        status_map = {
            "new": OrderStatus.OPEN.value,
            "open": OrderStatus.OPEN.value,
            "partially_filled": OrderStatus.PARTIALLY_FILLED.value,
            "filled": OrderStatus.FILLED.value,
            "canceled": OrderStatus.CANCELED.value,
            "cancelled": OrderStatus.CANCELED.value,
            "rejected": OrderStatus.REJECTED.value,
            "expired": OrderStatus.EXPIRED.value,
            "closed": OrderStatus.FILLED.value,  # Một số sàn sử dụng "closed" thay vì "filled"
        }
        
        # Trả về trạng thái chuẩn nếu có trong ánh xạ
        return status_map.get(exchange_status, OrderStatus.PENDING.value)
    
    def _convert_exchange_order_type(self, exchange_order_type: str) -> str:
        """
        Chuyển đổi loại lệnh từ định dạng sàn sang định dạng chuẩn.
        
        Args:
            exchange_order_type: Loại lệnh từ sàn
            
        Returns:
            Loại lệnh chuẩn
        """
        if not exchange_order_type:
            return OrderType.MARKET.value
            
        exchange_order_type = exchange_order_type.lower()
        
        # Đảo ngược ánh xạ từ order_type_map
        for order_type, mapped_type in self.exchange._order_type_map.items():
            if mapped_type.lower() == exchange_order_type:
                return order_type
        
        # Ánh xạ thủ công cho một số trường hợp đặc biệt
        if "market" in exchange_order_type:
            return OrderType.MARKET.value
        elif "limit" in exchange_order_type:
            if "stop" in exchange_order_type:
                return OrderType.STOP_LIMIT.value
            else:
                return OrderType.LIMIT.value
        elif "stop" in exchange_order_type:
            if "loss" in exchange_order_type:
                return OrderType.STOP_LOSS.value
            else:
                return OrderType.STOP_LOSS.value
        elif "take" in exchange_order_type and "profit" in exchange_order_type:
            return OrderType.TAKE_PROFIT.value
        elif "trailing" in exchange_order_type:
            return OrderType.TRAILING_STOP.value
        
        # Mặc định
        return OrderType.MARKET.value
    
    def _is_fatal_error(self, error: Exception) -> bool:
        """
        Kiểm tra xem lỗi có phải là lỗi không thể khắc phục không.
        
        Args:
            error: Lỗi cần kiểm tra
            
        Returns:
            True nếu là lỗi không thể khắc phục
        """
        error_str = str(error).lower()
        
        # Danh sách các từ khóa chỉ ra lỗi không thể khắc phục
        fatal_keywords = [
            "insufficient balance",
            "insufficient fund",
            "not enough balance",
            "invalid api key",
            "api key expired",
            "api key invalid",
            "authentication",
            "authorized",
            "access denied",
            "permission denied",
            "forbidden",
            "suspended",
            "disabled",
            "invalid signature",
            "invalid symbol",
            "market closed",
            "trading suspended",
            "token invalid",
        ]
        
        # Kiểm tra từng từ khóa
        for keyword in fatal_keywords:
            if keyword in error_str:
                return True
        
        # Kiểm tra mã lỗi nếu là APIError
        if isinstance(error, APIError):
            fatal_error_codes = [
                ErrorCode.PERMISSION_DENIED.value,
                ErrorCode.AUTHENTICATION_FAILED.value,
                ErrorCode.INVALID_PARAMETER.value,
                ErrorCode.INSUFFICIENT_BALANCE.value,
                ErrorCode.MARKET_CLOSED.value
            ]
            
            if error.error_code in fatal_error_codes:
                return True
        
        return False