"""
Mô phỏng sàn giao dịch thực tế.
File này định nghĩa lớp RealisticExchangeSimulator để mô phỏng một sàn giao dịch
với các đặc điểm thực tế như độ trễ, trượt giá, từ chối lệnh, và thanh khoản giới hạn.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import time
import random
import numpy as np
import uuid
import asyncio
from pathlib import Path

from config.logging_config import get_logger
from config.constants import OrderType, OrderStatus, PositionSide, TimeInForce, ErrorCode
from config.system_config import get_system_config
from environments.simulators.market_simulator import MarketSimulator, RealisticMarketSimulator
from environments.simulators.exchange_simulator.base_simulator import BaseExchangeSimulator
from environments.simulators.exchange_simulator.order_manager import OrderManager
from environments.simulators.exchange_simulator.position_manager import PositionManager
from environments.simulators.exchange_simulator.account_manager import AccountManager

class RealisticExchangeSimulator(BaseExchangeSimulator):
    """
    Mô phỏng sàn giao dịch thực tế với các đặc điểm như trượt giá, độ trễ, v.v.
    """
    
    def __init__(
        self,
        market_simulator: Optional[MarketSimulator] = None,
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        leverage: float = 1.0,
        maker_fee: float = 0.0001,  # 0.01%
        taker_fee: float = 0.0005,  # 0.05%
        min_order_size: float = 0.001,
        max_positions: int = 10,
        execution_delay: float = 0.5,  # Độ trễ thực thi (đơn vị là bước)
        order_timeout: int = 60000,  # Thời gian hết hạn lệnh (milliseconds)
        order_rejection_prob: float = 0.03,  # Xác suất từ chối lệnh
        slippage_model: str = "dynamic",  # "fixed", "random", "dynamic"
        fixed_slippage: float = 0.0005,  # Trượt giá cố định (0.05%)
        price_impact_factor: float = 1.0,  # Hệ số tác động giá
        liquidity_factor: float = 1.0,  # Hệ số thanh khoản
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng sàn giao dịch thực tế.
        
        Args:
            market_simulator: Mô phỏng thị trường
            initial_balance: Số dư ban đầu theo loại tiền
            leverage: Đòn bẩy mặc định
            maker_fee: Phí maker (%)
            taker_fee: Phí taker (%)
            min_order_size: Kích thước lệnh tối thiểu
            max_positions: Số vị thế tối đa
            execution_delay: Độ trễ thực thi (đơn vị là bước)
            order_timeout: Thời gian hết hạn lệnh (milliseconds)
            order_rejection_prob: Xác suất từ chối lệnh
            slippage_model: Mô hình trượt giá
            fixed_slippage: Trượt giá cố định
            price_impact_factor: Hệ số tác động giá
            liquidity_factor: Hệ số thanh khoản
            logger: Logger tùy chỉnh
        """
        # Gọi constructor của lớp cha
        super().__init__(
            market_simulator=market_simulator,
            initial_balance=initial_balance,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_order_size=min_order_size,
            logger=logger
        )
        
        # Lưu trữ các tham số bổ sung
        self.execution_delay = execution_delay
        self.order_timeout = order_timeout
        self.order_rejection_prob = order_rejection_prob
        self.slippage_model = slippage_model
        self.fixed_slippage = fixed_slippage
        self.price_impact_factor = price_impact_factor
        self.liquidity_factor = liquidity_factor
        
        # Khởi tạo các quản lý thành phần
        self.order_manager = OrderManager(min_order_size=min_order_size, logger=self.logger)
        self.position_manager = PositionManager(max_positions=max_positions, max_leverage=leverage, logger=self.logger)
        self.account_manager = AccountManager(initial_balance=initial_balance, logger=self.logger)
        
        # Kích hoạt tính năng thực tế nếu market_simulator là RealisticMarketSimulator
        self.realistic_mode = isinstance(market_simulator, RealisticMarketSimulator)
        
        # Hàng đợi lệnh đang xử lý
        self.pending_order_queue = []
        
        # Tăng bộ đếm bước mô phỏng
        self.step_count = 0
        
        self.logger.info(f"Đã khởi tạo RealisticExchangeSimulator với tỷ lệ từ chối lệnh {order_rejection_prob}, độ trễ {execution_delay}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Đặt lại trạng thái mô phỏng sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái sàn giao dịch sau khi đặt lại
        """
        # Đặt lại các quản lý thành phần
        self.order_manager.reset()
        self.position_manager.reset()
        self.account_manager.reset()
        
        # Đặt lại hàng đợi lệnh
        self.pending_order_queue = []
        
        # Đặt lại bộ đếm bước
        self.step_count = 0
        
        # Đặt lại trạng thái khác
        self.current_timestamp = datetime.now().timestamp() * 1000  # milliseconds
        
        # Lấy trạng thái mới
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        market_info = None
        if self.market_simulator:
            market_info = self.market_simulator.get_market_state()
        
        return {
            "timestamp": self.current_timestamp,
            "balance": self.account_manager.get_balance(),
            "positions": self.position_manager.get_positions(),
            "orders": self.order_manager.get_orders(),
            "pending_order_queue": self.pending_order_queue.copy(),
            "market_info": market_info,
            "step_count": self.step_count,
            "stats": self.account_manager.get_stats()
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Tiến hành một bước mô phỏng sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái sau bước mô phỏng
        """
        # Tăng bộ đếm bước
        self.step_count += 1
        
        # Cập nhật thời gian hiện tại
        if self.market_simulator:
            market_state = self.market_simulator.step()
            self.current_timestamp = market_state.get("timestamp", int(time.time() * 1000))
        else:
            self.current_timestamp = int(time.time() * 1000)
        
        # Xử lý hàng đợi lệnh đang chờ
        self._process_pending_orders()
        
        # Cập nhật trạng thái vị thế và giá trị
        self._update_positions()
        
        # Trả về trạng thái mới
        return self.get_state()
    
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
        # Chuẩn hóa tham số
        symbol = symbol.upper()
        order_type = order_type.lower()
        side = side.lower()
        
        # Kiểm tra market_simulator tồn tại
        if not self.market_simulator:
            self.logger.error("Không thể gửi lệnh: Chưa thiết lập market_simulator")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.CONFIGURATION_ERROR.value,
                    "message": "Chưa thiết lập market_simulator"
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
        
        # Lấy thông tin thị trường hiện tại
        market_state = self.market_simulator.get_market_state()
        prices = market_state.get("prices", {})
        
        # Tính toán số tiền cần để đặt lệnh
        quote_currency = symbol.split('/')[1] if '/' in symbol else "USDT"
        
        # Xác định giá thị trường nếu không cung cấp giá
        if price is None:
            if side == "buy":
                price = prices.get("ask", prices.get("close", 0))
            else:
                price = prices.get("bid", prices.get("close", 0))
        
        # Tính số tiền cần thiết
        required_amount = amount * price
        
        # Nếu mua, kiểm tra số dư
        if side == "buy" and order_type != "market":
            balance = self.account_manager.get_balance(quote_currency)
            
            if required_amount > balance:
                self.logger.warning(f"Số dư không đủ: {balance} < {required_amount}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INSUFFICIENT_BALANCE.value,
                        "message": "Số dư không đủ"
                    }
                }
        
        # Tạo lệnh
        order_result = self.order_manager.create_order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
            params=params
        )
        
        if order_result["status"] == "error":
            return order_result
        
        order = order_result["order"]
        
        # Tính thời gian thực thi dự kiến
        expected_execution_step = self.step_count + max(1, int(self.execution_delay * random.uniform(0.8, 1.2)))
        
        # Thêm vào hàng đợi lệnh đang chờ
        pending_order = {
            "order_id": order["id"],
            "expected_execution_step": expected_execution_step,
            "timeout": self.current_timestamp + self.order_timeout
        }
        
        self.pending_order_queue.append(pending_order)
        
        self.logger.info(f"Đã gửi lệnh {order['id']}: {side} {amount} {symbol} @ {price}, thực thi dự kiến tại bước {expected_execution_step}")
        
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
        # Kiểm tra lệnh trong hàng đợi
        for i, pending_order in enumerate(self.pending_order_queue):
            if pending_order["order_id"] == order_id:
                # Xóa khỏi hàng đợi
                self.pending_order_queue.pop(i)
                
                self.logger.info(f"Đã xóa lệnh {order_id} khỏi hàng đợi chờ xử lý")
                
                # Hủy lệnh
                return self.order_manager.cancel_order(order_id)
        
        # Nếu không tìm thấy trong hàng đợi, thử hủy lệnh trực tiếp
        return self.order_manager.cancel_order(order_id)
    
    def _process_pending_orders(self) -> None:
        """
        Xử lý các lệnh đang chờ trong hàng đợi.
        """
        if not self.pending_order_queue:
            return
        
        # Danh sách lệnh đã xử lý để xóa
        processed_orders = []
        
        for pending_order in self.pending_order_queue:
            order_id = pending_order["order_id"]
            expected_execution_step = pending_order["expected_execution_step"]
            timeout = pending_order["timeout"]
            
            # Kiểm tra lệnh đã hết hạn chưa
            if self.current_timestamp > timeout:
                self.logger.warning(f"Lệnh {order_id} đã hết hạn")
                
                # Cập nhật trạng thái lệnh
                order = self.order_manager.get_order(order_id)
                if order:
                    self.order_manager.update_order(
                        order_id=order_id,
                        update_data={"status": OrderStatus.EXPIRED.value}
                    )
                
                processed_orders.append(pending_order)
                continue
            
            # Kiểm tra đã đến thời gian thực thi chưa
            if self.step_count >= expected_execution_step:
                # Xác định xem lệnh có bị từ chối không
                if random.random() < self.order_rejection_prob:
                    self.logger.warning(f"Lệnh {order_id} bị từ chối")
                    
                    # Cập nhật trạng thái lệnh
                    order = self.order_manager.get_order(order_id)
                    if order:
                        self.order_manager.update_order(
                            order_id=order_id,
                            update_data={"status": OrderStatus.REJECTED.value}
                        )
                    
                    processed_orders.append(pending_order)
                    continue
                
                # Xử lý lệnh
                self._execute_order(order_id)
                
                processed_orders.append(pending_order)
        
        # Xóa các lệnh đã xử lý khỏi hàng đợi
        for order in processed_orders:
            if order in self.pending_order_queue:
                self.pending_order_queue.remove(order)
    
    def _execute_order(self, order_id: str) -> None:
        """
        Thực thi một lệnh giao dịch.
        
        Args:
            order_id: ID lệnh cần thực thi
        """
        # Lấy thông tin lệnh
        order = self.order_manager.get_order(order_id)
        if not order:
            self.logger.warning(f"Không tìm thấy lệnh {order_id} để thực thi")
            return
        
        # Kiểm tra market_simulator
        if not self.market_simulator:
            self.logger.error(f"Không thể thực thi lệnh {order_id}: Chưa thiết lập market_simulator")
            self.order_manager.update_order(
                order_id=order_id,
                update_data={"status": OrderStatus.REJECTED.value}
            )
            return
        
        # Lấy thông tin thị trường hiện tại
        market_state = self.market_simulator.get_market_state()
        prices = market_state.get("prices", {})
        
        # Xác định giá thực thi
        execution_price = self._calculate_execution_price(order, prices)
        
        if execution_price <= 0:
            self.logger.warning(f"Không thể thực thi lệnh {order_id}: Giá không hợp lệ")
            return
        
        # Tính khối lượng thực thi
        execution_amount = order["amount"]
        
        # Tính phí
        is_maker = order["type"] != OrderType.MARKET.value
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = execution_amount * execution_price * fee_rate
        
        # Xác định cặp tiền
        symbol = order["symbol"]
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol
        quote_currency = symbol.split('/')[1] if '/' in symbol else "USDT"
        
        # Cập nhật số dư
        if order["side"] == "buy":
            # Kiểm tra số dư
            required_amount = execution_amount * execution_price + fee
            balance = self.account_manager.get_balance(quote_currency)
            
            if required_amount > balance:
                self.logger.warning(f"Không đủ số dư để thực thi lệnh {order_id}: {balance} < {required_amount}")
                self.order_manager.update_order(
                    order_id=order_id,
                    update_data={"status": OrderStatus.REJECTED.value}
                )
                return
            
            # Cập nhật số dư
            self.account_manager.trade(
                currency=quote_currency,
                amount=-required_amount,
                fee=fee,
                trade_info={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "buy",
                    "price": execution_price,
                    "amount": execution_amount
                }
            )
            
            self.account_manager.trade(
                currency=base_currency,
                amount=execution_amount,
                fee=0.0,
                trade_info={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "buy",
                    "price": execution_price,
                    "amount": execution_amount
                }
            )
            
            # Nếu là giao dịch futures, cập nhật vị thế
            if self.leverage > 1.0:
                self.position_manager.create_position(
                    symbol=symbol,
                    side="long",
                    amount=execution_amount,
                    entry_price=execution_price,
                    leverage=self.leverage
                )
            
        else:  # sell
            # Kiểm tra số dư
            balance = self.account_manager.get_balance(base_currency)
            
            if execution_amount > balance and self.leverage <= 1.0:
                self.logger.warning(f"Không đủ số dư để thực thi lệnh {order_id}: {balance} < {execution_amount}")
                self.order_manager.update_order(
                    order_id=order_id,
                    update_data={"status": OrderStatus.REJECTED.value}
                )
                return
            
            # Cập nhật số dư
            self.account_manager.trade(
                currency=base_currency,
                amount=-execution_amount,
                fee=0.0,
                trade_info={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "sell",
                    "price": execution_price,
                    "amount": execution_amount
                }
            )
            
            received_amount = execution_amount * execution_price - fee
            self.account_manager.trade(
                currency=quote_currency,
                amount=received_amount,
                fee=fee,
                trade_info={
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "sell",
                    "price": execution_price,
                    "amount": execution_amount
                }
            )
            
            # Nếu là giao dịch futures, cập nhật vị thế
            if self.leverage > 1.0:
                self.position_manager.create_position(
                    symbol=symbol,
                    side="short",
                    amount=execution_amount,
                    entry_price=execution_price,
                    leverage=self.leverage
                )
        
        # Mô phỏng tác động thị trường
        if self.market_simulator:
            self.market_simulator.simulate_market_impact(
                order_volume=execution_amount,
                side=order["side"],
                order_type=order["type"]
            )
        
        # Cập nhật trạng thái lệnh
        trade = {
            "timestamp": self.current_timestamp,
            "symbol": symbol,
            "side": order["side"],
            "price": execution_price,
            "amount": execution_amount,
            "cost": execution_amount * execution_price,
            "fee": fee
        }
        
        self.order_manager.match_order(
            order_id=order_id,
            price=execution_price,
            amount=execution_amount,
            timestamp=self.current_timestamp,
            fee=fee
        )
        
        self.logger.info(f"Đã thực thi lệnh {order_id}: {order['side']} {execution_amount} {symbol} @ {execution_price}")
    
    def _calculate_execution_price(self, order: Dict[str, Any], prices: Dict[str, float]) -> float:
        """
        Tính giá thực thi cho một lệnh.
        
        Args:
            order: Thông tin lệnh
            prices: Giá thị trường hiện tại
            
        Returns:
            Giá thực thi
        """
        # Lấy giá thị trường
        mid_price = prices.get("mid", prices.get("close", 0))
        bid_price = prices.get("bid", mid_price * 0.999)
        ask_price = prices.get("ask", mid_price * 1.001)
        
        # Xác định giá cơ sở dựa trên loại lệnh
        if order["type"] == OrderType.MARKET.value:
            # Lệnh thị trường
            base_price = ask_price if order["side"] == "buy" else bid_price
            
            # Áp dụng trượt giá
            if self.slippage_model == "fixed":
                # Trượt giá cố định
                slippage = base_price * self.fixed_slippage
                execution_price = base_price + slippage if order["side"] == "buy" else base_price - slippage
            
            elif self.slippage_model == "random":
                # Trượt giá ngẫu nhiên
                max_slippage = base_price * self.fixed_slippage * 2
                slippage = random.uniform(0, max_slippage)
                execution_price = base_price + slippage if order["side"] == "buy" else base_price - slippage
            
            elif self.slippage_model == "dynamic":
                # Trượt giá động dựa trên thanh khoản và khối lượng lệnh
                market_liquidity = self.market_simulator.current_liquidity.get(order["side"], 1.0) * self.liquidity_factor
                volume_ratio = min(1.0, order["amount"] / market_liquidity)
                slippage = base_price * self.fixed_slippage * (1 + volume_ratio * 5)
                execution_price = base_price + slippage if order["side"] == "buy" else base_price - slippage
            
            else:
                execution_price = base_price
        
        elif order["type"] == OrderType.LIMIT.value:
            # Lệnh limit
            limit_price = order["price"]
            
            if order["side"] == "buy":
                # Nếu giá limit cao hơn giá ask, thực thi tại giá ask
                if limit_price >= ask_price:
                    execution_price = ask_price
                else:
                    # Không thực thi
                    return 0.0
            else:  # sell
                # Nếu giá limit thấp hơn giá bid, thực thi tại giá bid
                if limit_price <= bid_price:
                    execution_price = bid_price
                else:
                    # Không thực thi
                    return 0.0
        
        elif order["type"] == OrderType.STOP_LOSS.value:
            # Lệnh stop loss
            stop_price = order["price"]
            
            if order["side"] == "buy":
                # Stop buy kích hoạt khi giá lên trên mức stop
                if mid_price >= stop_price:
                    execution_price = ask_price
                else:
                    # Không thực thi
                    return 0.0
            else:  # sell
                # Stop sell kích hoạt khi giá xuống dưới mức stop
                if mid_price <= stop_price:
                    execution_price = bid_price
                else:
                    # Không thực thi
                    return 0.0
        
        elif order["type"] == OrderType.TAKE_PROFIT.value:
            # Lệnh take profit
            tp_price = order["price"]
            
            if order["side"] == "buy":
                # TP buy kích hoạt khi giá xuống dưới mức TP
                if mid_price <= tp_price:
                    execution_price = ask_price
                else:
                    # Không thực thi
                    return 0.0
            else:  # sell
                # TP sell kích hoạt khi giá lên trên mức TP
                if mid_price >= tp_price:
                    execution_price = bid_price
                else:
                    # Không thực thi
                    return 0.0
        
        else:
            # Loại lệnh không hỗ trợ
            self.logger.warning(f"Loại lệnh không được hỗ trợ: {order['type']}")
            return 0.0
        
        return execution_price
    
    def _update_positions(self) -> None:
        """
        Cập nhật trạng thái các vị thế.
        """
        if self.leverage <= 1.0 or not self.market_simulator:
            return
        
        # Lấy thông tin thị trường hiện tại
        market_state = self.market_simulator.get_market_state()
        prices = market_state.get("prices", {})
        
        # Cập nhật các vị thế
        positions = self.position_manager.get_positions()
        
        for position in positions:
            symbol = position["symbol"]
            
            # Lấy giá hiện tại
            current_price = prices.get("mid", prices.get("close", 0))
            
            # Cập nhật unrealized P&L
            self.position_manager.update_unrealized_pnl(symbol, current_price)