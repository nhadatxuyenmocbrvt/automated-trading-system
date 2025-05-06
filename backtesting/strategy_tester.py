"""
Kiểm tra chiến lược giao dịch.
File này cung cấp các công cụ để kiểm tra và đánh giá hiệu suất của
các chiến lược giao dịch khác nhau trên dữ liệu lịch sử.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import concurrent.futures
from functools import partial

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import OrderStatus, OrderType, PositionStatus, PositionSide
from config.constants import Timeframe, ErrorCode, BacktestMetric
from config.system_config import DATA_DIR, MODEL_DIR, BACKTEST_DIR

# Import các module quản lý rủi ro
from risk_management.position_sizer import PositionSizer
from risk_management.stop_loss import StopLoss
from risk_management.take_profit import TakeProfit
from risk_management.risk_calculator import RiskCalculator

class StrategyTester:
    """
    Lớp kiểm tra chiến lược giao dịch.
    Cung cấp các phương thức để đánh giá hiệu suất của các chiến lược trên dữ liệu lịch sử,
    bao gồm cả việc tạo các chỉ số đánh giá và trực quan hóa.
    """

    def __init__(
        self,
        strategy_func: Optional[Callable] = None,
        strategy_name: str = "default_strategy",
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        leverage: float = 1.0,
        risk_per_trade: float = 0.02,
        allow_short: bool = True,
        max_positions: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo tester cho chiến lược giao dịch.
        
        Args:
            strategy_func: Hàm chiến lược giao dịch
            strategy_name: Tên chiến lược
            data_dir: Thư mục dữ liệu đầu vào
            output_dir: Thư mục đầu ra kết quả kiểm tra
            initial_balance: Số dư ban đầu
            fee_rate: Tỷ lệ phí giao dịch
            slippage: Tỷ lệ trượt giá
            leverage: Đòn bẩy
            risk_per_trade: Rủi ro tối đa cho mỗi giao dịch (% số dư)
            allow_short: Cho phép giao dịch short
            max_positions: Số lượng vị thế mở tối đa
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("strategy_tester")
        
        # Thiết lập tham số
        self.strategy_func = strategy_func
        self.strategy_name = strategy_name
        
        # Thiết lập thư mục dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR
        else:
            self.data_dir = Path(data_dir)
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = BACKTEST_DIR / strategy_name
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập tham số giao dịch
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.allow_short = allow_short
        self.max_positions = max_positions
        
        # Khởi tạo các công cụ quản lý rủi ro
        self.position_sizer = PositionSizer(
            account_balance=initial_balance,
            max_risk_per_trade=risk_per_trade,
            max_leverage=leverage
        )
        self.stop_loss = StopLoss()
        self.take_profit = TakeProfit()
        self.risk_calculator = RiskCalculator()
        
        # Biến theo dõi trạng thái
        self.reset_state()
        
        self.logger.info(f"Đã khởi tạo StrategyTester cho chiến lược '{strategy_name}'")
    
    def reset_state(self) -> None:
        """
        Khởi tạo lại biến trạng thái cho một lần chạy mới.
        """
        # Trạng thái tài khoản
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.used_margin = 0.0
        self.free_margin = self.initial_balance
        
        # Danh sách vị thế
        self.open_positions = {}
        self.closed_positions = []
        
        # Danh sách lệnh
        self.open_orders = {}
        self.filled_orders = []
        self.canceled_orders = []
        
        # Lịch sử giá trị tài khoản
        self.balance_history = {
            "timestamp": [],
            "balance": [],
            "equity": [],
            "used_margin": [],
            "free_margin": []
        }
        
        # Trạng thái kiểm tra
        self.current_date = None
        self.test_completed = False
        self.metrics = {}
    
    def register_strategy(self, strategy_func: Callable, strategy_name: Optional[str] = None) -> None:
        """
        Đăng ký chiến lược giao dịch cho tester.
        
        Args:
            strategy_func: Hàm chiến lược giao dịch
            strategy_name: Tên chiến lược (tùy chọn)
        """
        self.strategy_func = strategy_func
        
        if strategy_name:
            self.strategy_name = strategy_name
            # Cập nhật thư mục đầu ra
            self.output_dir = BACKTEST_DIR / strategy_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Đã đăng ký chiến lược '{self.strategy_name}'")
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        file_path: Optional[Path] = None,
        data_format: str = "parquet"
    ) -> pd.DataFrame:
        """
        Tải dữ liệu lịch sử cho kiểm tra.
        
        Args:
            symbol: Ký hiệu tài sản (ví dụ: "BTC/USDT")
            timeframe: Khung thời gian dữ liệu
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            file_path: Đường dẫn file dữ liệu tùy chỉnh
            data_format: Định dạng dữ liệu ('csv', 'parquet', 'json')
            
        Returns:
            DataFrame dữ liệu lịch sử
        """
        try:
            if file_path:
                # Nếu có đường dẫn file cụ thể
                file_path = Path(file_path)
            else:
                # Nếu không, tạo đường dẫn từ thông tin đã cho
                symbol_safe = symbol.replace('/', '_').lower()
                
                if data_format == "parquet":
                    # Tìm file parquet gần nhất
                    pattern = f"{symbol_safe}_*.parquet"
                    data_files = list(self.data_dir.glob(pattern))
                    data_files.sort(reverse=True)  # Sắp xếp theo thời gian tạo
                    
                    if not data_files:
                        self.logger.error(f"Không tìm thấy file dữ liệu cho {symbol}, timeframe {timeframe}")
                        return pd.DataFrame()
                    
                    file_path = data_files[0]
                else:
                    # Tạo tên file dựa trên thông tin
                    file_name = f"{symbol_safe}_{timeframe}.{data_format}"
                    file_path = self.data_dir / file_name
            
            # Kiểm tra tồn tại
            if not file_path.exists():
                self.logger.error(f"Không tìm thấy file dữ liệu: {file_path}")
                return pd.DataFrame()
            
            # Tải dữ liệu
            if data_format == "csv" or file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
                
                # Chuyển timestamp thành datetime nếu có
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            elif data_format == "parquet" or file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            
            elif data_format == "json" or file_path.suffix == ".json":
                df = pd.read_json(file_path)
                
                # Chuyển timestamp thành datetime nếu có
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            else:
                self.logger.error(f"Định dạng dữ liệu không được hỗ trợ: {data_format}")
                return pd.DataFrame()
            
            # Lọc dữ liệu theo ngày nếu có chỉ định
            if 'timestamp' in df.columns:
                if start_date:
                    df = df[df['timestamp'] >= pd.Timestamp(start_date)]
                
                if end_date:
                    df = df[df['timestamp'] <= pd.Timestamp(end_date)]
                
                # Sắp xếp theo timestamp
                df = df.sort_values('timestamp')
            
            # Kiểm tra dữ liệu trả về trống
            if df.empty:
                self.logger.warning(f"Dữ liệu trống sau khi lọc: {symbol}, timeframe {timeframe}")
                return pd.DataFrame()
            
            self.logger.info(f"Đã tải {len(df)} dòng dữ liệu từ {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu: {str(e)}")
            return pd.DataFrame()
    
    def _process_tick(
        self,
        tick_data: Dict[str, Any],
        strategy_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý mỗi tick dữ liệu.
        
        Args:
            tick_data: Dữ liệu của tick hiện tại
            strategy_args: Tham số bổ sung cho chiến lược
            
        Returns:
            Dict kết quả xử lý tick
        """
        if strategy_args is None:
            strategy_args = {}
        
        # Lấy timestamp hiện tại
        timestamp = tick_data.get('timestamp')
        self.current_date = timestamp
        
        # Cập nhật vị thế đang mở
        self._update_positions(tick_data)
        
        # Cập nhật lệnh đang mở
        self._check_pending_orders(tick_data)
        
        # Gọi hàm chiến lược
        if self.strategy_func:
            try:
                strategy_result = self.strategy_func(
                    tick_data=tick_data,
                    balance=self.balance,
                    equity=self.equity,
                    open_positions=self.open_positions,
                    closed_positions=self.closed_positions,
                    **strategy_args
                )
                
                # Xử lý kết quả từ chiến lược
                if strategy_result and isinstance(strategy_result, dict):
                    self._process_strategy_signals(strategy_result, tick_data)
            
            except Exception as e:
                self.logger.error(f"Lỗi khi chạy chiến lược tại {timestamp}: {str(e)}")
        
        # Cập nhật lịch sử tài khoản
        self._update_account_history(timestamp)
        
        tick_result = {
            "timestamp": timestamp,
            "balance": self.balance,
            "equity": self.equity,
            "used_margin": self.used_margin,
            "free_margin": self.free_margin,
            "open_positions": len(self.open_positions),
            "closed_positions_count": len(self.closed_positions)
        }
        
        return tick_result
    
    def _process_strategy_signals(
        self,
        strategy_result: Dict[str, Any],
        tick_data: Dict[str, Any]
    ) -> None:
        """
        Xử lý các tín hiệu từ chiến lược.
        
        Args:
            strategy_result: Kết quả từ hàm chiến lược
            tick_data: Dữ liệu của tick hiện tại
        """
        # Xử lý tín hiệu mở vị thế
        if "open_position" in strategy_result and strategy_result["open_position"]:
            position_params = strategy_result["open_position"]
            self._open_position(position_params, tick_data)
        
        # Xử lý tín hiệu đóng vị thế
        if "close_position" in strategy_result and strategy_result["close_position"]:
            position_ids = strategy_result["close_position"]
            if isinstance(position_ids, list):
                for position_id in position_ids:
                    self._close_position(position_id, tick_data)
            else:
                self._close_position(position_ids, tick_data)
        
        # Xử lý tín hiệu tạo lệnh
        if "create_order" in strategy_result and strategy_result["create_order"]:
            order_params = strategy_result["create_order"]
            self._create_order(order_params, tick_data)
        
        # Xử lý tín hiệu hủy lệnh
        if "cancel_order" in strategy_result and strategy_result["cancel_order"]:
            order_ids = strategy_result["cancel_order"]
            if isinstance(order_ids, list):
                for order_id in order_ids:
                    self._cancel_order(order_id)
            else:
                self._cancel_order(order_ids)
        
        # Xử lý tín hiệu cập nhật dừng lỗ / chốt lời
        if "update_sl_tp" in strategy_result and strategy_result["update_sl_tp"]:
            sl_tp_params = strategy_result["update_sl_tp"]
            self._update_sl_tp(sl_tp_params, tick_data)
    
    def _open_position(
        self,
        position_params: Dict[str, Any],
        tick_data: Dict[str, Any]
    ) -> str:
        """
        Mở vị thế mới.
        
        Args:
            position_params: Tham số vị thế
            tick_data: Dữ liệu của tick hiện tại
            
        Returns:
            ID vị thế mới
        """
        # Kiểm tra số lượng vị thế đang mở
        if len(self.open_positions) >= self.max_positions:
            self.logger.warning(f"Đã đạt giới hạn vị thế mở tối đa ({self.max_positions})")
            return ""
        
        # Kiểm tra tham số
        if "symbol" not in position_params or "side" not in position_params:
            self.logger.error("Thiếu tham số cần thiết cho vị thế mới")
            return ""
        
        symbol = position_params["symbol"]
        side = position_params["side"].lower()
        
        # Kiểm tra nếu không cho phép short
        if side == "short" and not self.allow_short:
            self.logger.warning("Vị thế short không được phép")
            return ""
        
        # Xác định giá mở vị thế
        if "entry_price" in position_params:
            entry_price = position_params["entry_price"]
        else:
            entry_price = tick_data.get("close", 0)
        
        # Xác định kích thước vị thế
        position_size = position_params.get("size", 0)
        risk_percent = position_params.get("risk_percent", self.risk_per_trade)
        
        if position_size <= 0:
            # Tính toán kích thước dựa trên rủi ro
            if "stop_loss" in position_params:
                stop_loss = position_params["stop_loss"]
                
                # Sử dụng position_sizer để tính kích thước tối ưu
                position_size_result = self.position_sizer.calculate_position_size_fixed_risk(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    risk_percent=risk_percent,
                    leverage=self.leverage,
                    fee_rate=self.fee_rate,
                    slippage_percent=self.slippage
                )
                
                if position_size_result["status"] == "success":
                    position_size = position_size_result["position_size"]
                else:
                    # Sử dụng kích thước mặc định
                    position_size = self.balance * risk_percent * self.leverage
            else:
                # Nếu không có stop_loss, sử dụng % cố định của số dư
                position_size = self.balance * risk_percent * self.leverage
        
        # Tính số lượng coin
        quantity = position_size / entry_price
        
        # Tính margin cần thiết
        required_margin = position_size / self.leverage
        
        # Kiểm tra margin khả dụng
        if required_margin > self.free_margin:
            self.logger.warning(f"Không đủ margin khả dụng: cần {required_margin}, có {self.free_margin}")
            return ""
        
        # Tính phí giao dịch
        fee = position_size * self.fee_rate
        
        # Tạo ID vị thế
        position_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Tạo vị thế mới
        position = {
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "position_size": position_size,
            "leverage": self.leverage,
            "required_margin": required_margin,
            "entry_fee": fee,
            "entry_time": tick_data.get("timestamp", datetime.now()),
            "status": PositionStatus.OPEN.value,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "roi": 0
        }
        
        # Thêm stop_loss và take_profit nếu có
        if "stop_loss" in position_params:
            position["stop_loss"] = position_params["stop_loss"]
        
        if "take_profit" in position_params:
            position["take_profit"] = position_params["take_profit"]
        
        # Cập nhật trạng thái tài khoản
        self.balance -= fee
        self.used_margin += required_margin
        self.free_margin = self.balance - self.used_margin
        self.equity = self.balance + position["unrealized_pnl"]
        
        # Thêm vị thế vào danh sách đang mở
        self.open_positions[position_id] = position
        
        self.logger.info(f"Đã mở vị thế {side} {symbol}: {quantity:.6f} @ {entry_price:.2f}")
        
        return position_id
    
    def _close_position(
        self,
        position_id: str,
        tick_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Đóng vị thế hiện có.
        
        Args:
            position_id: ID vị thế cần đóng
            tick_data: Dữ liệu của tick hiện tại
            
        Returns:
            Dict thông tin vị thế đã đóng
        """
        if position_id not in self.open_positions:
            self.logger.warning(f"Không tìm thấy vị thế {position_id} để đóng")
            return {}
        
        # Lấy thông tin vị thế
        position = self.open_positions[position_id]
        
        # Xác định giá đóng vị thế
        exit_price = tick_data.get("close", 0)
        
        # Tính phí giao dịch
        exit_fee = position["position_size"] * self.fee_rate
        
        # Tính lợi nhuận
        if position["side"] == "long":
            pnl = position["quantity"] * (exit_price - position["entry_price"]) - position["entry_fee"] - exit_fee
        else:  # short
            pnl = position["quantity"] * (position["entry_price"] - exit_price) - position["entry_fee"] - exit_fee
        
        # Tính ROI
        roi = pnl / position["required_margin"]
        
        # Cập nhật thông tin vị thế
        position["exit_price"] = exit_price
        position["exit_time"] = tick_data.get("timestamp", datetime.now())
        position["exit_fee"] = exit_fee
        position["realized_pnl"] = pnl
        position["roi"] = roi
        position["status"] = PositionStatus.CLOSED.value
        position["duration"] = (position["exit_time"] - position["entry_time"]).total_seconds() / 3600  # giờ
        
        # Cập nhật trạng thái tài khoản
        self.balance += position["required_margin"] + pnl
        self.used_margin -= position["required_margin"]
        self.free_margin = self.balance - self.used_margin
        self.equity = self.balance
        
        # Chuyển vị thế từ đang mở sang đã đóng
        self.closed_positions.append(position)
        del self.open_positions[position_id]
        
        self.logger.info(f"Đã đóng vị thế {position['side']} {position['symbol']}: {position['quantity']:.6f} @ {exit_price:.2f}, P&L: {pnl:.2f}, ROI: {roi:.2%}")
        
        return position
    
    def _create_order(
        self,
        order_params: Dict[str, Any],
        tick_data: Dict[str, Any]
    ) -> str:
        """
        Tạo lệnh giao dịch mới.
        
        Args:
            order_params: Tham số lệnh
            tick_data: Dữ liệu của tick hiện tại
            
        Returns:
            ID lệnh mới
        """
        # Kiểm tra tham số
        required_params = ["symbol", "side", "type", "price"]
        if not all(param in order_params for param in required_params):
            self.logger.error(f"Thiếu tham số cần thiết cho lệnh mới: {required_params}")
            return ""
        
        symbol = order_params["symbol"]
        side = order_params["side"].lower()
        order_type = order_params["type"].lower()
        price = order_params["price"]
        
        # Kiểm tra nếu không cho phép short
        if side == "short" and not self.allow_short:
            self.logger.warning("Lệnh short không được phép")
            return ""
        
        # Xác định kích thước lệnh
        order_size = order_params.get("size", 0)
        risk_percent = order_params.get("risk_percent", self.risk_per_trade)
        
        if order_size <= 0:
            # Tính toán kích thước dựa trên rủi ro
            if "stop_loss" in order_params:
                stop_loss = order_params["stop_loss"]
                
                # Sử dụng position_sizer để tính kích thước tối ưu
                order_size_result = self.position_sizer.calculate_position_size_fixed_risk(
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    risk_percent=risk_percent,
                    leverage=self.leverage,
                    fee_rate=self.fee_rate
                )
                
                if order_size_result["status"] == "success":
                    order_size = order_size_result["position_size"]
                else:
                    # Sử dụng kích thước mặc định
                    order_size = self.balance * risk_percent * self.leverage
            else:
                # Nếu không có stop_loss, sử dụng % cố định của số dư
                order_size = self.balance * risk_percent * self.leverage
        
        # Tính số lượng coin
        quantity = order_size / price
        
        # Tính margin cần thiết
        required_margin = order_size / self.leverage
        
        # Kiểm tra margin khả dụng
        if required_margin > self.free_margin:
            self.logger.warning(f"Không đủ margin khả dụng: cần {required_margin}, có {self.free_margin}")
            return ""
        
        # Tạo ID lệnh
        order_id = f"{symbol}_{side}_{order_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Tạo lệnh mới
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": price,
            "quantity": quantity,
            "order_size": order_size,
            "leverage": self.leverage,
            "required_margin": required_margin,
            "create_time": tick_data.get("timestamp", datetime.now()),
            "status": OrderStatus.PENDING.value
        }
        
        # Thêm stop_loss và take_profit nếu có
        if "stop_loss" in order_params:
            order["stop_loss"] = order_params["stop_loss"]
        
        if "take_profit" in order_params:
            order["take_profit"] = order_params["take_profit"]
        
        # Thêm thông tin bổ sung
        if "time_in_force" in order_params:
            order["time_in_force"] = order_params["time_in_force"]
        
        if "expiry_time" in order_params:
            order["expiry_time"] = order_params["expiry_time"]
        
        # Thêm lệnh vào danh sách đang mở
        self.open_orders[order_id] = order
        
        self.logger.info(f"Đã tạo lệnh {side} {order_type} {symbol}: {quantity:.6f} @ {price:.2f}")
        
        return order_id
    
    def _cancel_order(self, order_id: str) -> bool:
        """
        Hủy lệnh giao dịch.
        
        Args:
            order_id: ID lệnh cần hủy
            
        Returns:
            True nếu hủy thành công, False nếu không
        """
        if order_id not in self.open_orders:
            self.logger.warning(f"Không tìm thấy lệnh {order_id} để hủy")
            return False
        
        # Lấy thông tin lệnh
        order = self.open_orders[order_id]
        
        # Cập nhật trạng thái lệnh
        order["status"] = OrderStatus.CANCELED.value
        order["cancel_time"] = datetime.now()
        
        # Chuyển lệnh từ đang mở sang đã hủy
        self.canceled_orders.append(order)
        del self.open_orders[order_id]
        
        self.logger.info(f"Đã hủy lệnh {order['side']} {order['type']} {order['symbol']}")
        
        return True
    
    def _check_pending_orders(self, tick_data: Dict[str, Any]) -> None:
        """
        Kiểm tra và xử lý các lệnh đang chờ.
        
        Args:
            tick_data: Dữ liệu của tick hiện tại
        """
        if not self.open_orders:
            return
        
        # Lấy giá hiện tại
        current_price = tick_data.get("close", 0)
        high_price = tick_data.get("high", current_price)
        low_price = tick_data.get("low", current_price)
        
        # Danh sách lệnh cần xử lý
        orders_to_process = list(self.open_orders.values())
        
        for order in orders_to_process:
            order_id = order["order_id"]
            
            # Kiểm tra hết hạn
            if "expiry_time" in order and tick_data.get("timestamp") > order["expiry_time"]:
                # Lệnh đã hết hạn
                order["status"] = OrderStatus.EXPIRED.value
                order["expire_time"] = tick_data.get("timestamp")
                
                # Chuyển lệnh sang đã hủy
                self.canceled_orders.append(order)
                del self.open_orders[order_id]
                
                self.logger.info(f"Lệnh {order_id} đã hết hạn")
                continue
            
            # Kiểm tra điều kiện khớp lệnh
            matched = False
            
            if order["type"] == OrderType.MARKET.value:
                # Lệnh thị trường luôn khớp ngay lập tức
                matched = True
                order["executed_price"] = current_price
            
            elif order["type"] == OrderType.LIMIT.value:
                # Lệnh limit mua: khớp khi giá thấp hơn hoặc bằng giá limit
                # Lệnh limit bán: khớp khi giá cao hơn hoặc bằng giá limit
                if order["side"] == "long" and low_price <= order["price"]:
                    matched = True
                    order["executed_price"] = order["price"]
                
                elif order["side"] == "short" and high_price >= order["price"]:
                    matched = True
                    order["executed_price"] = order["price"]
            
            elif order["type"] == OrderType.STOP_LOSS.value:
                # Lệnh stop loss mua: khớp khi giá cao hơn hoặc bằng giá stop
                # Lệnh stop loss bán: khớp khi giá thấp hơn hoặc bằng giá stop
                if order["side"] == "long" and high_price >= order["price"]:
                    matched = True
                    order["executed_price"] = order["price"]
                
                elif order["side"] == "short" and low_price <= order["price"]:
                    matched = True
                    order["executed_price"] = order["price"]
            
            # Xử lý lệnh khớp
            if matched:
                # Cập nhật trạng thái lệnh
                order["status"] = OrderStatus.FILLED.value
                order["fill_time"] = tick_data.get("timestamp", datetime.now())
                
                # Mở vị thế mới từ lệnh
                position_params = {
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "entry_price": order["executed_price"],
                    "size": order["order_size"]
                }
                
                # Thêm stop_loss và take_profit nếu có
                if "stop_loss" in order:
                    position_params["stop_loss"] = order["stop_loss"]
                
                if "take_profit" in order:
                    position_params["take_profit"] = order["take_profit"]
                
                # Mở vị thế
                position_id = self._open_position(position_params, tick_data)
                if position_id:
                    order["position_id"] = position_id
                
                # Chuyển lệnh sang đã khớp
                self.filled_orders.append(order)
                del self.open_orders[order_id]
                
                self.logger.info(f"Lệnh {order_id} đã khớp tại {order['executed_price']:.2f}")
    
    def _update_positions(self, tick_data: Dict[str, Any]) -> None:
        """
        Cập nhật trạng thái các vị thế đang mở.
        
        Args:
            tick_data: Dữ liệu của tick hiện tại
        """
        if not self.open_positions:
            return
        
        # Lấy giá hiện tại
        current_price = tick_data.get("close", 0)
        high_price = tick_data.get("high", current_price)
        low_price = tick_data.get("low", current_price)
        
        # Tổng lợi nhuận không thực hiện
        total_unrealized_pnl = 0
        
        # Danh sách vị thế cần đóng
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            # Cập nhật lợi nhuận không thực hiện
            if position["side"] == "long":
                unrealized_pnl = position["quantity"] * (current_price - position["entry_price"]) - position["entry_fee"]
                
                # Kiểm tra dừng lỗ
                if "stop_loss" in position and low_price <= position["stop_loss"]:
                    positions_to_close.append((position_id, "stop_loss", position["stop_loss"]))
                
                # Kiểm tra chốt lời
                if "take_profit" in position and high_price >= position["take_profit"]:
                    positions_to_close.append((position_id, "take_profit", position["take_profit"]))
                
            else:  # short
                unrealized_pnl = position["quantity"] * (position["entry_price"] - current_price) - position["entry_fee"]
                
                # Kiểm tra dừng lỗ
                if "stop_loss" in position and high_price >= position["stop_loss"]:
                    positions_to_close.append((position_id, "stop_loss", position["stop_loss"]))
                
                # Kiểm tra chốt lời
                if "take_profit" in position and low_price <= position["take_profit"]:
                    positions_to_close.append((position_id, "take_profit", position["take_profit"]))
            
            # Cập nhật thông tin vị thế
            position["unrealized_pnl"] = unrealized_pnl
            position["current_price"] = current_price
            
            # Tính ROI hiện tại
            position["unrealized_roi"] = unrealized_pnl / position["required_margin"]
            
            # Cộng vào tổng
            total_unrealized_pnl += unrealized_pnl
        
        # Cập nhật equity
        self.equity = self.balance + total_unrealized_pnl
        
        # Đóng các vị thế kích hoạt stop loss / take profit
        for position_id, trigger_type, trigger_price in positions_to_close:
            # Tạo tick data mới với giá đóng vị thế
            close_tick = tick_data.copy()
            close_tick["close"] = trigger_price
            
            # Đóng vị thế
            self._close_position(position_id, close_tick)
            self.logger.info(f"Vị thế {position_id} đã đóng do kích hoạt {trigger_type} tại {trigger_price:.2f}")
    
    def _update_sl_tp(
        self,
        sl_tp_params: Dict[str, Any],
        tick_data: Dict[str, Any]
    ) -> bool:
        """
        Cập nhật dừng lỗ / chốt lời cho vị thế.
        
        Args:
            sl_tp_params: Tham số cập nhật
            tick_data: Dữ liệu của tick hiện tại
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        if "position_id" not in sl_tp_params:
            self.logger.error("Thiếu position_id trong tham số cập nhật")
            return False
        
        position_id = sl_tp_params["position_id"]
        
        if position_id not in self.open_positions:
            self.logger.warning(f"Không tìm thấy vị thế {position_id} để cập nhật")
            return False
        
        position = self.open_positions[position_id]
        
        # Cập nhật stop_loss
        if "stop_loss" in sl_tp_params:
            position["stop_loss"] = sl_tp_params["stop_loss"]
            self.logger.info(f"Đã cập nhật stop_loss cho vị thế {position_id}: {sl_tp_params['stop_loss']:.2f}")
        
        # Cập nhật take_profit
        if "take_profit" in sl_tp_params:
            position["take_profit"] = sl_tp_params["take_profit"]
            self.logger.info(f"Đã cập nhật take_profit cho vị thế {position_id}: {sl_tp_params['take_profit']:.2f}")
        
        return True
    
    def _update_account_history(self, timestamp) -> None:
        """
        Cập nhật lịch sử giá trị tài khoản.
        
        Args:
            timestamp: Timestamp hiện tại
        """
        self.balance_history["timestamp"].append(timestamp)
        self.balance_history["balance"].append(self.balance)
        self.balance_history["equity"].append(self.equity)
        self.balance_history["used_margin"].append(self.used_margin)
        self.balance_history["free_margin"].append(self.free_margin)
    
    def run_test(
        self,
        data: pd.DataFrame,
        strategy_args: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Chạy kiểm tra chiến lược trên dữ liệu lịch sử.
        
        Args:
            data: DataFrame dữ liệu lịch sử
            strategy_args: Tham số bổ sung cho chiến lược
            verbose: In thông tin chi tiết
            
        Returns:
            Dict kết quả kiểm tra
        """
        if data.empty:
            self.logger.error("Không có dữ liệu để kiểm tra")
            return {"status": "error", "message": "Không có dữ liệu để kiểm tra"}
        
        if self.strategy_func is None:
            self.logger.error("Chưa đăng ký chiến lược giao dịch")
            return {"status": "error", "message": "Chưa đăng ký chiến lược giao dịch"}
        
        # Đặt lại trạng thái
        self.reset_state()
        
        # Chuyển DataFrame thành danh sách dict
        if isinstance(data, pd.DataFrame):
            data_records = data.to_dict("records")
        else:
            data_records = data
        
        # Lưu trữ kết quả tick
        tick_results = []
        
        # Phân tích từng tick
        for i, tick in enumerate(data_records):
            tick_result = self._process_tick(tick, strategy_args)
            tick_results.append(tick_result)
            
            # In trạng thái trong quá trình kiểm tra
            if verbose and (i == 0 or i == len(data_records) - 1 or (i + 1) % 1000 == 0):
                self.logger.info(f"Xử lý {i+1}/{len(data_records)} ticks, balance: {self.balance:.2f}, equity: {self.equity:.2f}")
        
        # Đánh dấu đã hoàn thành
        self.test_completed = True
        
        # Tính toán các chỉ số đánh giá
        self.calculate_metrics()
        
        # Lưu kết quả
        test_result = {
            "status": "success",
            "metrics": self.metrics,
            "balance_history": self.balance_history,
            "closed_positions": self.closed_positions,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "final_equity": self.equity,
            "profit": self.balance - self.initial_balance,
            "roi": (self.balance - self.initial_balance) / self.initial_balance
        }
        
        # In tóm tắt kết quả
        if verbose:
            self._print_summary()
        
        return test_result
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá hiệu suất.
        
        Returns:
            Dict các chỉ số đánh giá
        """
        if not self.test_completed or not self.closed_positions:
            self.logger.warning("Chưa có dữ liệu kiểm tra hoặc không có giao dịch đã đóng")
            return {}
        
        # Chuẩn bị dữ liệu
        equity_series = pd.Series(self.balance_history["equity"], index=self.balance_history["timestamp"])
        
        # Tính lợi nhuận và thua lỗ
        profits = [pos["realized_pnl"] for pos in self.closed_positions if pos["realized_pnl"] > 0]
        losses = [pos["realized_pnl"] for pos in self.closed_positions if pos["realized_pnl"] <= 0]
        
        # Tính số giao dịch
        total_trades = len(self.closed_positions)
        winning_trades = len(profits)
        losing_trades = len(losses)
        
        # Tính tỷ lệ thắng
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Tính các chỉ số cơ bản
        metrics = {
            BacktestMetric.TOTAL_RETURN.value: self.balance - self.initial_balance,
            BacktestMetric.ANNUALIZED_RETURN.value: self._calculate_annualized_return(equity_series),
            BacktestMetric.MAX_DRAWDOWN.value: self._calculate_max_drawdown(equity_series),
            BacktestMetric.WIN_RATE.value: win_rate,
            BacktestMetric.PROFIT_FACTOR.value: sum(profits) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            BacktestMetric.AVERAGE_TRADE.value: sum([pos["realized_pnl"] for pos in self.closed_positions]) / total_trades if total_trades > 0 else 0,
            BacktestMetric.AVERAGE_WIN.value: sum(profits) / winning_trades if winning_trades > 0 else 0,
            BacktestMetric.AVERAGE_LOSS.value: sum(losses) / losing_trades if losing_trades > 0 else 0,
        }
        
        # Tính các chỉ số nâng cao
        if metrics[BacktestMetric.MAX_DRAWDOWN.value] != 0:
            metrics[BacktestMetric.CALMAR_RATIO.value] = metrics[BacktestMetric.ANNUALIZED_RETURN.value] / abs(metrics[BacktestMetric.MAX_DRAWDOWN.value])
        else:
            metrics[BacktestMetric.CALMAR_RATIO.value] = float('inf')
        
        if metrics[BacktestMetric.AVERAGE_LOSS.value] != 0:
            metrics[BacktestMetric.RISK_REWARD_RATIO.value] = metrics[BacktestMetric.AVERAGE_WIN.value] / abs(metrics[BacktestMetric.AVERAGE_LOSS.value])
        else:
            metrics[BacktestMetric.RISK_REWARD_RATIO.value] = float('inf')
        
        # Tính Expectancy
        metrics[BacktestMetric.EXPECTANCY.value] = (win_rate * metrics[BacktestMetric.AVERAGE_WIN.value]) - ((1 - win_rate) * abs(metrics[BacktestMetric.AVERAGE_LOSS.value]))
        
        # Tính Sharpe và Sortino Ratio nếu có đủ dữ liệu
        if len(equity_series) > 1:
            # Tính chuỗi lợi nhuận
            returns = equity_series.pct_change().dropna()
            
            if len(returns) > 0:
                # Sharpe Ratio (giả định lãi suất không rủi ro = 0)
                sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
                metrics[BacktestMetric.SHARPE_RATIO.value] = sharpe_ratio * np.sqrt(252)  # Annualized
                
                # Sortino Ratio (chỉ quan tâm đến phương sai âm)
                negative_returns = returns[returns < 0]
                downside_deviation = negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() != 0 else 1e-9
                sortino_ratio = returns.mean() / downside_deviation
                metrics[BacktestMetric.SORTINO_RATIO.value] = sortino_ratio * np.sqrt(252)  # Annualized
        
        # Lưu metrics
        self.metrics = metrics
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """
        Tính toán drawdown tối đa.
        
        Args:
            equity_series: Series giá trị tài khoản
            
        Returns:
            Phần trăm drawdown tối đa
        """
        # Tính peak-to-trough
        if len(equity_series) <= 1:
            return 0
        
        # Tính cumulative maximum
        cummax = equity_series.cummax()
        
        # Tính drawdown
        drawdown = (equity_series - cummax) / cummax
        
        # Lấy giá trị drawdown tối đa
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _calculate_annualized_return(self, equity_series: pd.Series) -> float:
        """
        Tính toán lợi nhuận hàng năm.
        
        Args:
            equity_series: Series giá trị tài khoản
            
        Returns:
            Phần trăm lợi nhuận hàng năm
        """
        if len(equity_series) <= 1:
            return 0
        
        # Lấy giá trị đầu tiên và cuối cùng
        initial_equity = equity_series.iloc[0]
        final_equity = equity_series.iloc[-1]
        
        # Tính số ngày kiểm tra
        start_date = equity_series.index[0]
        end_date = equity_series.index[-1]
        days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
        
        # Tính lợi nhuận hàng năm
        if days > 0 and initial_equity > 0:
            annualized_return = ((final_equity / initial_equity) ** (365 / days)) - 1
        else:
            annualized_return = 0
        
        return annualized_return
    
    def _print_summary(self) -> None:
        """
        In tóm tắt kết quả kiểm tra.
        """
        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"KẾT QUẢ KIỂM TRA CHIẾN LƯỢC: {self.strategy_name}")
        self.logger.info(f"{'=' * 50}")
        
        self.logger.info(f"Số dư ban đầu: {self.initial_balance:.2f}")
        self.logger.info(f"Số dư cuối cùng: {self.balance:.2f}")
        self.logger.info(f"Tổng lợi nhuận: {self.balance - self.initial_balance:.2f} ({(self.balance - self.initial_balance) / self.initial_balance:.2%})")
        
        self.logger.info(f"\n{'-' * 30}")
        self.logger.info("CÁC CHỈ SỐ ĐÁNH GIÁ:")
        self.logger.info(f"{'-' * 30}")
        
        for metric_name, metric_value in self.metrics.items():
            # Định dạng phần trăm cho các chỉ số tỷ lệ
            if metric_name in [
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.ANNUALIZED_RETURN.value,
                BacktestMetric.MAX_DRAWDOWN.value
            ]:
                self.logger.info(f"{metric_name}: {metric_value:.2%}")
            else:
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
        
        self.logger.info(f"\n{'-' * 30}")
        self.logger.info("THỐNG KÊ GIAO DỊCH:")
        self.logger.info(f"{'-' * 30}")
        
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p["realized_pnl"] > 0])
        losing_trades = len([p for p in self.closed_positions if p["realized_pnl"] <= 0])
        
        self.logger.info(f"Tổng số giao dịch: {total_trades}")
        self.logger.info(f"Giao dịch thắng: {winning_trades} ({winning_trades / total_trades:.2%} nếu total_trades > 0 else 0)")
        self.logger.info(f"Giao dịch thua: {losing_trades} ({losing_trades / total_trades:.2%} nếu total_trades > 0 else 0)")
        
        avg_trade_duration = sum([p["duration"] for p in self.closed_positions]) / total_trades if total_trades > 0 else 0
        self.logger.info(f"Thời gian giữ trung bình: {avg_trade_duration:.2f} giờ")
        
        self.logger.info(f"{'=' * 50}")
    
    def save_results(
        self,
        test_result: Dict[str, Any],
        file_name: Optional[str] = None,
        include_trades: bool = True,
        include_history: bool = True
    ) -> Path:
        """
        Lưu kết quả kiểm tra vào file.
        
        Args:
            test_result: Kết quả kiểm tra
            file_name: Tên file đầu ra
            include_trades: Bao gồm lịch sử giao dịch chi tiết
            include_history: Bao gồm lịch sử giá trị tài khoản
            
        Returns:
            Đường dẫn file kết quả
        """
        if file_name is None:
            # Tạo tên file từ tên chiến lược và timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.strategy_name}_backtest_{timestamp}.json"
        
        file_path = self.output_dir / file_name
        
        # Chuẩn bị dữ liệu để lưu
        save_data = {
            "strategy_name": self.strategy_name,
            "test_date": datetime.now().isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "final_equity": self.equity,
            "profit": self.balance - self.initial_balance,
            "roi": (self.balance - self.initial_balance) / self.initial_balance,
            "metrics": self.metrics,
            "parameters": {
                "fee_rate": self.fee_rate,
                "slippage": self.slippage,
                "leverage": self.leverage,
                "risk_per_trade": self.risk_per_trade,
                "allow_short": self.allow_short,
                "max_positions": self.max_positions
            }
        }
        
        # Thêm lịch sử giao dịch nếu yêu cầu
        if include_trades:
            # Chuyển đổi các đối tượng datetime thành chuỗi ISO format
            closed_positions = []
            for pos in self.closed_positions:
                pos_copy = pos.copy()
                if isinstance(pos_copy.get("entry_time"), datetime):
                    pos_copy["entry_time"] = pos_copy["entry_time"].isoformat()
                if isinstance(pos_copy.get("exit_time"), datetime):
                    pos_copy["exit_time"] = pos_copy["exit_time"].isoformat()
                closed_positions.append(pos_copy)
            
            save_data["trades"] = {
                "total_trades": len(closed_positions),
                "winning_trades": len([p for p in closed_positions if p["realized_pnl"] > 0]),
                "losing_trades": len([p for p in closed_positions if p["realized_pnl"] <= 0]),
                "closed_positions": closed_positions
            }
        
        # Thêm lịch sử tài khoản nếu yêu cầu
        if include_history:
            # Chuyển đổi timestamp thành chuỗi ISO format
            timestamps = [ts.isoformat() if isinstance(ts, datetime) else ts for ts in self.balance_history["timestamp"]]
            
            save_data["balance_history"] = {
                "timestamp": timestamps,
                "balance": self.balance_history["balance"],
                "equity": self.balance_history["equity"]
            }
        
        # Lưu vào file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu kết quả kiểm tra vào {file_path}")
        
        return file_path
    
    def load_results(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Tải kết quả kiểm tra từ file.
        
        Args:
            file_path: Đường dẫn file kết quả
            
        Returns:
            Dict kết quả kiểm tra
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"Không tìm thấy file kết quả: {file_path}")
            return {}
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            self.logger.info(f"Đã tải kết quả kiểm tra từ {file_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kết quả: {str(e)}")
            return {}
    
    def compare_strategies(
        self,
        strategy_results: List[Dict[str, Any]],
        metrics_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        So sánh hiệu suất của nhiều chiến lược.
        
        Args:
            strategy_results: Danh sách kết quả kiểm tra
            metrics_to_compare: Danh sách chỉ số cần so sánh
            
        Returns:
            DataFrame so sánh các chiến lược
        """
        if not strategy_results:
            self.logger.warning("Không có kết quả chiến lược để so sánh")
            return pd.DataFrame()
        
        # Thiết lập các chỉ số mặc định nếu không được chỉ định
        if metrics_to_compare is None:
            metrics_to_compare = [
                BacktestMetric.TOTAL_RETURN.value,
                BacktestMetric.ANNUALIZED_RETURN.value,
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value,
                BacktestMetric.EXPECTANCY.value
            ]
        
        # Chuẩn bị dữ liệu so sánh
        comparison_data = []
        
        for result in strategy_results:
            strategy_name = result.get("strategy_name", "Unnamed")
            metrics = result.get("metrics", {})
            
            row_data = {"strategy_name": strategy_name}
            
            # Thêm các chỉ số
            for metric in metrics_to_compare:
                if metric in metrics:
                    row_data[metric] = metrics[metric]
                else:
                    row_data[metric] = None
            
            # Thêm thông tin cơ bản
            row_data["initial_balance"] = result.get("initial_balance", 0)
            row_data["final_balance"] = result.get("final_balance", 0)
            row_data["profit"] = result.get("profit", 0)
            row_data["roi"] = result.get("roi", 0)
            
            # Thêm thông tin giao dịch
            trades_info = result.get("trades", {})
            row_data["total_trades"] = trades_info.get("total_trades", 0)
            row_data["winning_trades"] = trades_info.get("winning_trades", 0)
            row_data["losing_trades"] = trades_info.get("losing_trades", 0)
            
            comparison_data.append(row_data)
        
        # Tạo DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def run_walk_forward_test(
        self,
        data: pd.DataFrame,
        window_size: int = 90,
        step_size: int = 30,
        retrain_function: Optional[Callable] = None,
        min_window_size: int = 30,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Chạy kiểm tra walk-forward.
        
        Args:
            data: DataFrame dữ liệu lịch sử
            window_size: Kích thước cửa sổ huấn luyện (ngày)
            step_size: Kích thước bước tiến (ngày)
            retrain_function: Hàm huấn luyện lại chiến lược
            min_window_size: Kích thước cửa sổ tối thiểu
            verbose: In thông tin chi tiết
            
        Returns:
            Dict kết quả kiểm tra walk-forward
        """
        if data.empty:
            self.logger.error("Không có dữ liệu để kiểm tra walk-forward")
            return {"status": "error", "message": "Không có dữ liệu để kiểm tra walk-forward"}
        
        if self.strategy_func is None:
            self.logger.error("Chưa đăng ký chiến lược giao dịch")
            return {"status": "error", "message": "Chưa đăng ký chiến lược giao dịch"}
        
        # Đảm bảo có cột timestamp
        if 'timestamp' not in data.columns:
            self.logger.error("Dữ liệu không có cột timestamp")
            return {"status": "error", "message": "Dữ liệu không có cột timestamp"}
        
        # Chuyển đổi timestamp sang datetime nếu cần
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sắp xếp dữ liệu theo timestamp
        data = data.sort_values('timestamp')
        
        # Tính toán số cửa sổ
        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()
        date_range = (end_date - start_date).days
        
        if date_range < min_window_size:
            self.logger.error(f"Khoảng thời gian dữ liệu ({date_range} ngày) không đủ cho kích thước cửa sổ tối thiểu ({min_window_size} ngày)")
            return {"status": "error", "message": f"Khoảng thời gian dữ liệu ({date_range} ngày) không đủ"}
        
        num_windows = max(1, (date_range - window_size) // step_size + 1)
        
        if verbose:
            self.logger.info(f"Bắt đầu kiểm tra walk-forward với {num_windows} cửa sổ")
            self.logger.info(f"Khoảng thời gian dữ liệu: {start_date} - {end_date} ({date_range} ngày)")
        
        # Lưu trữ kết quả cho mỗi cửa sổ
        window_results = []
        all_trades = []
        equity_history = {'timestamp': [], 'equity': []}
        
        # Đặt lại số dư ban đầu
        current_balance = self.initial_balance
        
        # Lặp qua từng cửa sổ
        for i in range(num_windows):
            window_start = start_date + timedelta(days=i * step_size)
            train_end = window_start + timedelta(days=window_size)
            test_end = min(end_date, train_end + timedelta(days=step_size))
            
            # Tạo các tập huấn luyện và kiểm tra
            train_data = data[(data['timestamp'] >= window_start) & (data['timestamp'] < train_end)]
            test_data = data[(data['timestamp'] >= train_end) & (data['timestamp'] < test_end)]
            
            if len(train_data) < min_window_size or len(test_data) == 0:
                continue
            
            if verbose:
                self.logger.info(f"Cửa sổ {i+1}/{num_windows}:")
                self.logger.info(f"  - Huấn luyện: {window_start} - {train_end} ({len(train_data)} mẫu)")
                self.logger.info(f"  - Kiểm tra: {train_end} - {test_end} ({len(test_data)} mẫu)")
            
            # Huấn luyện lại chiến lược nếu được cung cấp hàm
            strategy_args = {}
            if retrain_function:
                try:
                    strategy_args = retrain_function(train_data, window_start, train_end)
                    if verbose:
                        self.logger.info(f"  - Đã huấn luyện lại chiến lược")
                except Exception as e:
                    self.logger.error(f"Lỗi khi huấn luyện lại chiến lược: {str(e)}")
            
            # Tạo một tester mới cho mỗi cửa sổ với số dư hiện tại
            window_tester = StrategyTester(
                strategy_func=self.strategy_func,
                strategy_name=f"{self.strategy_name}_window_{i+1}",
                initial_balance=current_balance,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                leverage=self.leverage,
                risk_per_trade=self.risk_per_trade,
                allow_short=self.allow_short,
                max_positions=self.max_positions,
                logger=self.logger
            )
            
            # Chạy kiểm tra trên dữ liệu cửa sổ
            window_result = window_tester.run_test(
                data=test_data,
                strategy_args=strategy_args,
                verbose=False
            )
            
            # Cập nhật số dư hiện tại
            current_balance = window_result["final_balance"]
            
            # Lưu kết quả cửa sổ
            window_result["window_id"] = i + 1
            window_result["train_start"] = window_start
            window_result["train_end"] = train_end
            window_result["test_start"] = train_end
            window_result["test_end"] = test_end
            window_result["train_size"] = len(train_data)
            window_result["test_size"] = len(test_data)
            
            window_results.append(window_result)
            
            # Lưu các giao dịch
            for trade in window_result.get("closed_positions", []):
                trade["window_id"] = i + 1
                all_trades.append(trade)
            
            # Lưu lịch sử equity
            timestamps = window_result.get("balance_history", {}).get("timestamp", [])
            equity_values = window_result.get("balance_history", {}).get("equity", [])
            equity_history["timestamp"].extend(timestamps)
            equity_history["equity"].extend(equity_values)
            
            if verbose:
                self.logger.info(f"  - Kết quả: balance {window_result['final_balance']:.2f}, ROI {window_result['roi']:.2%}")
        
        # Tính toán các chỉ số tổng hợp
        overall_results = {
            "status": "success",
            "strategy_name": self.strategy_name,
            "initial_balance": self.initial_balance,
            "final_balance": current_balance,
            "roi": (current_balance - self.initial_balance) / self.initial_balance,
            "window_results": window_results,
            "all_trades": all_trades,
            "num_windows": len(window_results),
            "total_trades": len(all_trades),
            "winning_trades": len([t for t in all_trades if t.get("realized_pnl", 0) > 0]),
            "losing_trades": len([t for t in all_trades if t.get("realized_pnl", 0) <= 0])
        }
        
        # Tính các chỉ số đánh giá từ lịch sử equity
        if equity_history["timestamp"]:
            equity_series = pd.Series(equity_history["equity"], index=equity_history["timestamp"])
            
            # Tính lợi nhuận hàng năm
            overall_results["annualized_return"] = self._calculate_annualized_return(equity_series)
            
            # Tính drawdown tối đa
            overall_results["max_drawdown"] = self._calculate_max_drawdown(equity_series)
            
            # Tính Sharpe Ratio
            returns = equity_series.pct_change().dropna()
            if len(returns) > 0 and returns.std() != 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
                overall_results["sharpe_ratio"] = sharpe_ratio
            
            # Tính win rate và profit factor
            if all_trades:
                win_rate = overall_results["winning_trades"] / overall_results["total_trades"]
                overall_results["win_rate"] = win_rate
                
                profit_sum = sum([t.get("realized_pnl", 0) for t in all_trades if t.get("realized_pnl", 0) > 0])
                loss_sum = abs(sum([t.get("realized_pnl", 0) for t in all_trades if t.get("realized_pnl", 0) <= 0]))
                
                if loss_sum > 0:
                    overall_results["profit_factor"] = profit_sum / loss_sum
                else:
                    overall_results["profit_factor"] = float('inf')
        
        # In tóm tắt kết quả
        if verbose:
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"KẾT QUẢ WALK-FORWARD: {self.strategy_name}")
            self.logger.info(f"{'=' * 50}")
            self.logger.info(f"Số cửa sổ: {len(window_results)}")
            self.logger.info(f"Số dư ban đầu: {self.initial_balance:.2f}")
            self.logger.info(f"Số dư cuối cùng: {current_balance:.2f}")
            self.logger.info(f"Tổng lợi nhuận: {current_balance - self.initial_balance:.2f} ({overall_results['roi']:.2%})")
            
            if "annualized_return" in overall_results:
                self.logger.info(f"Lợi nhuận hàng năm: {overall_results['annualized_return']:.2%}")
            
            if "sharpe_ratio" in overall_results:
                self.logger.info(f"Sharpe Ratio: {overall_results['sharpe_ratio']:.4f}")
            
            if "max_drawdown" in overall_results:
                self.logger.info(f"Drawdown tối đa: {overall_results['max_drawdown']:.2%}")
            
            self.logger.info(f"Tổng số giao dịch: {overall_results['total_trades']}")
            
            if overall_results['total_trades'] > 0:
                self.logger.info(f"Tỷ lệ thắng: {overall_results['winning_trades'] / overall_results['total_trades']:.2%}")
            
            if "profit_factor" in overall_results:
                self.logger.info(f"Profit Factor: {overall_results['profit_factor']:.4f}")
        
        return overall_results
    
    def evaluate_monte_carlo(
        self,
        test_result: Dict[str, Any],
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
        calculate_var: bool = True,
        calculate_drawdown: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất bằng phương pháp Monte Carlo.
        
        Args:
            test_result: Kết quả kiểm tra
            num_simulations: Số lần mô phỏng
            confidence_level: Mức độ tin cậy
            calculate_var: Tính Value at Risk
            calculate_drawdown: Tính Drawdown
            
        Returns:
            Dict kết quả đánh giá Monte Carlo
        """
        if "closed_positions" not in test_result or not test_result["closed_positions"]:
            self.logger.error("Không có dữ liệu giao dịch để đánh giá")
            return {"status": "error", "message": "Không có dữ liệu giao dịch để đánh giá"}
        
        # Lấy danh sách ROI của các giao dịch
        trades = test_result["closed_positions"]
        trade_rois = [trade.get("roi", 0) for trade in trades]
        
        if not trade_rois:
            self.logger.error("Không có dữ liệu ROI để đánh giá")
            return {"status": "error", "message": "Không có dữ liệu ROI để đánh giá"}
        
        # Tạo mô phỏng Monte Carlo
        np.random.seed(42)  # Để tái tạo kết quả
        
        # Danh sách kết quả mô phỏng
        simulation_results = []
        
        for _ in range(num_simulations):
            # Lấy mẫu với thay thế từ các ROI giao dịch
            sampled_rois = np.random.choice(trade_rois, size=len(trade_rois), replace=True)
            
            # Tính toán tăng trưởng vốn
            initial_balance = test_result.get("initial_balance", self.initial_balance)
            balance = initial_balance
            equity_curve = [initial_balance]
            
            for roi in sampled_rois:
                # Giả định mỗi giao dịch sử dụng toàn bộ số dư
                balance = balance * (1 + roi)
                equity_curve.append(balance)
            
            # Tính toán các chỉ số
            final_balance = equity_curve[-1]
            total_roi = (final_balance - initial_balance) / initial_balance
            
            # Tính drawdown tối đa
            max_drawdown = 0
            peak = equity_curve[0]
            
            for value in equity_curve:
                peak = max(peak, value)
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Lưu kết quả mô phỏng
            simulation_results.append({
                "final_balance": final_balance,
                "total_roi": total_roi,
                "max_drawdown": max_drawdown,
                "equity_curve": equity_curve
            })
        
        # Tính toán thống kê
        final_balances = [result["final_balance"] for result in simulation_results]
        total_rois = [result["total_roi"] for result in simulation_results]
        max_drawdowns = [result["max_drawdown"] for result in simulation_results]
        
        # Sắp xếp các kết quả để tính phân vị
        final_balances.sort()
        total_rois.sort()
        max_drawdowns.sort()
        
        # Tính chỉ số cho mức độ tin cậy
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_index = int(num_simulations * lower_percentile)
        upper_index = int(num_simulations * upper_percentile)
        
        # Kết quả Monte Carlo
        monte_carlo_results = {
            "status": "success",
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "balance": {
                "mean": np.mean(final_balances),
                "median": np.median(final_balances),
                "std": np.std(final_balances),
                "min": min(final_balances),
                "max": max(final_balances),
                "lower_bound": final_balances[lower_index],
                "upper_bound": final_balances[upper_index]
            },
            "roi": {
                "mean": np.mean(total_rois),
                "median": np.median(total_rois),
                "std": np.std(total_rois),
                "min": min(total_rois),
                "max": max(total_rois),
                "lower_bound": total_rois[lower_index],
                "upper_bound": total_rois[upper_index]
            },
            "drawdown": {
                "mean": np.mean(max_drawdowns),
                "median": np.median(max_drawdowns),
                "std": np.std(max_drawdowns),
                "min": min(max_drawdowns),
                "max": max(max_drawdowns),
                "lower_bound": max_drawdowns[lower_index],
                "upper_bound": max_drawdowns[upper_index]
            }
        }
        
        # Tính Value at Risk nếu được yêu cầu
        if calculate_var:
            var_percentiles = [0.01, 0.05, 0.1]
            var_results = {}
            
            for percentile in var_percentiles:
                var_index = int(num_simulations * percentile)
                var_roi = total_rois[var_index]
                
                var_results[str(int(percentile * 100))] = {
                    "percentile": percentile,
                    "var_roi": var_roi,
                    "var_amount": self.initial_balance * var_roi
                }
            
            monte_carlo_results["var"] = var_results
        
        # Tính xác suất drawdown nếu được yêu cầu
        if calculate_drawdown:
            drawdown_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            drawdown_results = {}
            
            for threshold in drawdown_thresholds:
                exceeded_count = sum(1 for dd in max_drawdowns if dd > threshold)
                probability = exceeded_count / num_simulations
                
                drawdown_results[str(int(threshold * 100))] = {
                    "threshold": threshold,
                    "probability": probability
                }
            
            monte_carlo_results["drawdown_probabilities"] = drawdown_results
        
        # In tóm tắt kết quả
        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"KẾT QUẢ MONTE CARLO: {num_simulations} mô phỏng, {confidence_level:.0%} tin cậy")
        self.logger.info(f"{'=' * 50}")
        
        self.logger.info(f"ROI trung bình: {monte_carlo_results['roi']['mean']:.2%}")
        self.logger.info(f"Khoảng tin cậy ROI: [{monte_carlo_results['roi']['lower_bound']:.2%}, {monte_carlo_results['roi']['upper_bound']:.2%}]")
        self.logger.info(f"Drawdown tối đa trung bình: {monte_carlo_results['drawdown']['mean']:.2%}")
        self.logger.info(f"Khoảng tin cậy Drawdown: [{monte_carlo_results['drawdown']['lower_bound']:.2%}, {monte_carlo_results['drawdown']['upper_bound']:.2%}]")
        
        if calculate_var:
            self.logger.info(f"\nValue at Risk (VaR):")
            for percentile, var_data in monte_carlo_results["var"].items():
                self.logger.info(f"  VaR {percentile}%: {var_data['var_roi']:.2%} ({var_data['var_amount']:.2f})")
        
        if calculate_drawdown:
            self.logger.info(f"\nXác suất Drawdown:")
            for threshold, dd_data in monte_carlo_results["drawdown_probabilities"].items():
                self.logger.info(f"  P(DD > {threshold}%): {dd_data['probability']:.2%}")
        
        return monte_carlo_results
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        metric_to_optimize: str = BacktestMetric.SHARPE_RATIO.value,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa tham số chiến lược.
        
        Args:
            data: DataFrame dữ liệu lịch sử
            param_grid: Grid các tham số cần tối ưu hóa
            metric_to_optimize: Chỉ số cần tối ưu hóa
            max_workers: Số luồng tối đa cho xử lý song song
            
        Returns:
            Dict kết quả tối ưu hóa
        """
        if data.empty:
            self.logger.error("Không có dữ liệu để tối ưu hóa")
            return {"status": "error", "message": "Không có dữ liệu để tối ưu hóa"}
        
        if self.strategy_func is None:
            self.logger.error("Chưa đăng ký chiến lược giao dịch")
            return {"status": "error", "message": "Chưa đăng ký chiến lược giao dịch"}
        
        # Tạo danh sách các tổ hợp tham số
        import itertools
        param_keys = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Bắt đầu tối ưu hóa với {len(param_combinations)} tổ hợp tham số")
        
        # Hàm chạy kiểm tra cho một tổ hợp tham số
        def test_params(params_tuple):
            params_dict = dict(zip(param_keys, params_tuple))
            
            # Tạo một instance mới của tester cho mỗi tổ hợp tham số
            tester = StrategyTester(
                strategy_func=self.strategy_func,
                strategy_name=self.strategy_name,
                initial_balance=self.initial_balance,
                fee_rate=self.fee_rate,
                slippage=self.slippage,
                leverage=self.leverage,
                risk_per_trade=self.risk_per_trade,
                allow_short=self.allow_short,
                max_positions=self.max_positions,
                logger=self.logger
            )
            
            # Chạy kiểm tra với tham số hiện tại
            test_result = tester.run_test(
                data=data,
                strategy_args={"params": params_dict},
                verbose=False
            )
            
            # Lấy giá trị metric cần tối ưu hóa
            if metric_to_optimize in test_result.get("metrics", {}):
                metric_value = test_result["metrics"][metric_to_optimize]
            else:
                # Nếu không có metric, sử dụng ROI
                metric_value = test_result.get("roi", -float('inf'))
            
            # Kết quả cho tổ hợp tham số này
            return {
                "params": params_dict,
                "metric_value": metric_value,
                "test_result": test_result
            }
        
        # Chạy tối ưu hóa song song
        results = []
        
        if max_workers > 1:
            # Chạy song song
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, result in enumerate(executor.map(test_params, param_combinations)):
                    results.append(result)
                    if (i + 1) % 10 == 0 or i == len(param_combinations) - 1:
                        self.logger.info(f"Đã hoàn thành {i+1}/{len(param_combinations)} tổ hợp tham số")
        else:
            # Chạy tuần tự
            for i, params_tuple in enumerate(param_combinations):
                result = test_params(params_tuple)
                results.append(result)
                if (i + 1) % 10 == 0 or i == len(param_combinations) - 1:
                    self.logger.info(f"Đã hoàn thành {i+1}/{len(param_combinations)} tổ hợp tham số")
        
        # Sắp xếp kết quả theo metric
        # Lưu ý: Một số metric là càng lớn càng tốt, một số là càng nhỏ càng tốt
        if metric_to_optimize in [
            BacktestMetric.MAX_DRAWDOWN.value,  # Drawdown càng nhỏ càng tốt
        ]:
            # Sắp xếp tăng dần
            results.sort(key=lambda x: x["metric_value"])
        else:
            # Sắp xếp giảm dần
            results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        # Lấy kết quả tốt nhất
        best_result = results[0] if results else None
        
        if not best_result:
            return {"status": "error", "message": "Không tìm thấy tham số tối ưu"}
        
        # Tạo bảng so sánh các kết quả
        comparison_data = []
        for result in results:
            row = {**result["params"]}
            row["metric_value"] = result["metric_value"]
            row["roi"] = result["test_result"].get("roi", 0)
            
            metrics = result["test_result"].get("metrics", {})
            row["sharpe_ratio"] = metrics.get(BacktestMetric.SHARPE_RATIO.value, 0)
            row["max_drawdown"] = metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0)
            row["win_rate"] = metrics.get(BacktestMetric.WIN_RATE.value, 0)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Lưu kết quả tối ưu hóa
        optimization_results = {
            "status": "success",
            "best_params": best_result["params"],
            "best_metric_value": best_result["metric_value"],
            "best_test_result": best_result["test_result"],
            "comparison_df": comparison_df,
            "all_results": results
        }
        
        # In tham số tối ưu
        self.logger.info(f"Tham số tối ưu: {best_result['params']}")
        self.logger.info(f"Giá trị {metric_to_optimize}: {best_result['metric_value']}")
        
        return optimization_results