"""
Đo lường hiệu suất giao dịch.
File này cung cấp các công cụ để theo dõi, ghi lại, và phân tích các
số liệu trong quá trình giao dịch thực tế, giúp đánh giá hiệu suất
và đưa ra quyết định điều chỉnh chiến lược.
"""

import os
import time
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger, log_trading_metrics
from config.system_config import get_system_config
from config.constants import OrderStatus, PositionSide, OrderType, BacktestMetric
from backtesting.performance_metrics import PerformanceMetrics
from logs.logger import get_trade_logger, TradeLogger

class TradingMetricsTracker:
    """
    Lớp theo dõi và quản lý các số liệu trong quá trình giao dịch.
    Cung cấp các phương thức để ghi nhận, phân tích, và xuất báo cáo
    về hiệu suất giao dịch theo thời gian thực.
    """
    
    def __init__(
        self,
        symbol: str,
        strategy_name: str,
        initial_capital: float,
        output_dir: Optional[Union[str, Path]] = None,
        use_csv: bool = True,
        log_frequency: int = 10,
        logger: Optional[logging.Logger] = None,
        trade_logger: Optional[TradeLogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo theo dõi số liệu giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            strategy_name: Tên chiến lược đang sử dụng
            initial_capital: Vốn ban đầu
            output_dir: Thư mục đầu ra cho số liệu
            use_csv: Lưu số liệu vào file CSV
            log_frequency: Tần suất ghi log (mỗi bao nhiêu giao dịch)
            logger: Logger tùy chỉnh
            trade_logger: Logger giao dịch tùy chỉnh
            config: Cấu hình bổ sung
        """
        # Thiết lập logger
        self.logger = logger or get_logger("trading_metrics")
        self.trade_logger = trade_logger or get_trade_logger(symbol)
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thông tin cơ bản
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.use_csv = use_csv
        self.log_frequency = log_frequency
        
        # Cấu hình theo dõi
        self.config = config or {}
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            logs_dir = Path(self.system_config.get("log_dir", "./logs"))
            output_dir = logs_dir / "trading" / self.symbol.replace("/", "_")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập file CSV
        if self.use_csv:
            current_date = datetime.now().strftime("%Y%m%d")
            self.csv_path = self.output_dir / f"{self.symbol.replace('/', '_')}_{current_date}_metrics.csv"
            
            # Kiểm tra file có tồn tại không
            if not self.csv_path.exists():
                # Tạo DataFrame rỗng với các cột dữ liệu chính
                df = pd.DataFrame(columns=[
                    'timestamp', 'trade_id', 'order_type', 'side', 
                    'entry_price', 'exit_price', 'quantity', 
                    'profit_loss', 'profit_loss_percent', 'duration',
                    'fees', 'current_capital', 'order_status'
                ])
                df.to_csv(self.csv_path, index=False)
                self.logger.info(f"Đã tạo file CSV mới tại {self.csv_path}")
        
        # Khởi tạo biến trạng thái
        self.start_time = time.time()
        self.trade_count = 0
        self.open_positions = {}
        
        # Theo dõi số liệu
        self.metrics_history = {
            "trades": [],                   # Danh sách tất cả các giao dịch
            "capital_history": [],          # Lịch sử vốn theo thời gian
            "profit_loss_history": [],      # Lịch sử lợi nhuận/lỗ
            "win_history": [],              # Lịch sử thắng/thua
            "trade_durations": [],          # Thời gian nắm giữ giao dịch
            "position_sizes": [],           # Kích thước vị thế
            "drawdowns": [],                # Các giai đoạn sụt giảm vốn
            "daily_returns": {},            # Lợi nhuận hàng ngày
            "fees_paid": [],                # Phí giao dịch đã trả
            "position_exposures": []        # Mức độ phơi nhiễm rủi ro
        }
        
        # Thống kê hiệu suất
        self.performance_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "total_profit_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "expectancy": 0.0,
            "average_trade_duration": 0.0,
            "total_fees": 0.0,
            "roi": 0.0
        }
        
        # Ghi log lúc khởi tạo
        self.logger.info(f"Đã khởi tạo TradingMetricsTracker cho {symbol} với chiến lược {strategy_name}")
        self.trade_logger.log_portfolio_update(
            total_value=initial_capital,
            positions={},
            available_balance=initial_capital
        )
    
    def log_order_created(
        self,
        order_id: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận thông tin lệnh giao dịch đã được tạo.
        
        Args:
            order_id: ID của lệnh
            order_type: Loại lệnh (market, limit, etc.)
            side: Phía lệnh (buy, sell)
            quantity: Số lượng
            price: Giá lệnh (cho limit orders)
            timestamp: Thời gian tạo lệnh
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_order_created(
            order_id=order_id,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price
        )
        
        # Lưu thông tin lệnh
        order_info = {
            "order_id": order_id,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "created",
            "timestamp": timestamp.isoformat(),
            "filled_quantity": 0.0,
            "filled_price": 0.0,
            "fees": 0.0
        }
        
        # Thêm vào open_positions nếu là lệnh buy/long
        if side.lower() in ['buy', 'long']:
            position_id = f"position_{len(self.open_positions) + 1}"
            self.open_positions[position_id] = {
                "entry_orders": [order_info],
                "exit_orders": [],
                "position_id": position_id,
                "side": "long",
                "entry_time": timestamp.isoformat(),
                "status": "pending",
                "quantity": quantity,
                "entry_price": price,
                "current_price": price,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            }
        # Nếu là lệnh sell/short, cần xác định xem đây là lệnh thoát hay lệnh mở short mới
        else:
            # Kiểm tra xem có vị thế long đang mở không
            for pos_id, pos_info in self.open_positions.items():
                if pos_info["side"] == "long" and pos_info["status"] in ["open", "pending"]:
                    # Đây là lệnh thoát cho vị thế long
                    pos_info["exit_orders"].append(order_info)
                    return
            
            # Nếu không tìm thấy vị thế long, đây là lệnh mở short mới
            position_id = f"position_{len(self.open_positions) + 1}"
            self.open_positions[position_id] = {
                "entry_orders": [order_info],
                "exit_orders": [],
                "position_id": position_id,
                "side": "short",
                "entry_time": timestamp.isoformat(),
                "status": "pending",
                "quantity": quantity,
                "entry_price": price,
                "current_price": price,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0
            }
    
    def log_order_filled(
        self,
        order_id: str,
        fill_price: float,
        filled_quantity: float,
        fees: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận thông tin lệnh giao dịch đã được thực hiện.
        
        Args:
            order_id: ID của lệnh
            fill_price: Giá thực hiện
            filled_quantity: Số lượng thực hiện
            fees: Phí giao dịch
            timestamp: Thời gian thực hiện lệnh
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if fees is None:
            fees = 0.0
        
        # Ghi log
        self.trade_logger.log_order_filled(
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=filled_quantity,
            fees=fees
        )
        
        # Cập nhật thông tin lệnh trong các vị thế
        for pos_id, pos_info in self.open_positions.items():
            # Kiểm tra trong entry_orders
            for order in pos_info["entry_orders"]:
                if order["order_id"] == order_id:
                    order["status"] = "filled"
                    order["filled_quantity"] = filled_quantity
                    order["filled_price"] = fill_price
                    order["fees"] = fees
                    
                    # Cập nhật thông tin vị thế
                    pos_info["status"] = "open"
                    pos_info["entry_price"] = fill_price
                    pos_info["current_price"] = fill_price
                    
                    # Ghi log
                    self.trade_logger.log_position_opened(
                        position_id=pos_id,
                        side=pos_info["side"],
                        entry_price=fill_price,
                        size=filled_quantity,
                        leverage=1.0  # Mặc định không có đòn bẩy
                    )
                    
                    # Cập nhật vốn
                    position_value = fill_price * filled_quantity
                    self.current_capital -= position_value + fees
                    
                    # Lưu lịch sử vốn
                    self.metrics_history["capital_history"].append({
                        "timestamp": timestamp.isoformat(),
                        "capital": self.current_capital,
                        "event": "entry",
                        "position_id": pos_id
                    })
                    
                    # Lưu phí
                    self.metrics_history["fees_paid"].append({
                        "timestamp": timestamp.isoformat(),
                        "fees": fees,
                        "order_id": order_id,
                        "position_id": pos_id
                    })
                    
                    return
            
            # Kiểm tra trong exit_orders
            for order in pos_info["exit_orders"]:
                if order["order_id"] == order_id:
                    order["status"] = "filled"
                    order["filled_quantity"] = filled_quantity
                    order["filled_price"] = fill_price
                    order["fees"] = fees
                    
                    # Tính P&L
                    entry_price = pos_info["entry_price"]
                    if pos_info["side"] == "long":
                        profit_loss = (fill_price - entry_price) * filled_quantity - fees
                        profit_loss_percent = ((fill_price - entry_price) / entry_price) * 100
                    else:  # short
                        profit_loss = (entry_price - fill_price) * filled_quantity - fees
                        profit_loss_percent = ((entry_price - fill_price) / entry_price) * 100
                    
                    # Cập nhật vị thế
                    pos_info["status"] = "closed"
                    pos_info["realized_pnl"] = profit_loss
                    pos_info["exit_time"] = timestamp.isoformat()
                    pos_info["exit_price"] = fill_price
                    
                    # Tính thời gian nắm giữ
                    entry_time = datetime.fromisoformat(pos_info["entry_time"])
                    duration = (timestamp - entry_time).total_seconds() / 3600  # Giờ
                    
                    # Ghi log
                    self.trade_logger.log_position_closed(
                        position_id=pos_id,
                        exit_price=fill_price,
                        profit_loss=profit_loss,
                        profit_loss_percent=profit_loss_percent,
                        reason="manual"
                    )
                    
                    # Cập nhật vốn
                    position_value = fill_price * filled_quantity
                    self.current_capital += position_value - fees
                    
                    # Lưu lịch sử vốn
                    self.metrics_history["capital_history"].append({
                        "timestamp": timestamp.isoformat(),
                        "capital": self.current_capital,
                        "event": "exit",
                        "position_id": pos_id
                    })
                    
                    # Lưu lịch sử lợi nhuận/lỗ
                    self.metrics_history["profit_loss_history"].append({
                        "timestamp": timestamp.isoformat(),
                        "profit_loss": profit_loss,
                        "profit_loss_percent": profit_loss_percent,
                        "position_id": pos_id
                    })
                    
                    # Lưu phí
                    self.metrics_history["fees_paid"].append({
                        "timestamp": timestamp.isoformat(),
                        "fees": fees,
                        "order_id": order_id,
                        "position_id": pos_id
                    })
                    
                    # Lưu lịch sử thắng/thua
                    self.metrics_history["win_history"].append({
                        "timestamp": timestamp.isoformat(),
                        "is_win": profit_loss > 0,
                        "position_id": pos_id,
                        "profit_loss": profit_loss
                    })
                    
                    # Lưu thời gian nắm giữ
                    self.metrics_history["trade_durations"].append({
                        "position_id": pos_id,
                        "duration": duration,
                        "entry_time": pos_info["entry_time"],
                        "exit_time": timestamp.isoformat()
                    })
                    
                    # Tăng biến đếm giao dịch
                    self.trade_count += 1
                    
                    # Cập nhật thống kê hiệu suất
                    self._update_performance_stats()
                    
                    # Ghi vào CSV nếu cần
                    if self.use_csv:
                        self._append_to_csv({
                            "timestamp": timestamp.isoformat(),
                            "trade_id": pos_id,
                            "order_type": order["order_type"],
                            "side": pos_info["side"],
                            "entry_price": pos_info["entry_price"],
                            "exit_price": fill_price,
                            "quantity": filled_quantity,
                            "profit_loss": profit_loss,
                            "profit_loss_percent": profit_loss_percent,
                            "duration": duration,
                            "fees": fees,
                            "current_capital": self.current_capital,
                            "order_status": "filled"
                        })
                    
                    # Ghi log với tần suất theo cấu hình
                    if self.trade_count % self.log_frequency == 0:
                        self.logger.info(
                            f"Đã hoàn thành {self.trade_count} giao dịch. "
                            f"Vốn hiện tại: {self.current_capital:.2f}, "
                            f"ROI: {(self.current_capital / self.initial_capital - 1) * 100:.2f}%"
                        )
                    
                    return
    
    def log_order_canceled(
        self,
        order_id: str,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận thông tin lệnh giao dịch đã bị hủy.
        
        Args:
            order_id: ID của lệnh
            reason: Lý do hủy lệnh
            timestamp: Thời gian hủy lệnh
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_order_canceled(order_id=order_id, reason=reason)
        
        # Cập nhật thông tin lệnh trong các vị thế
        for pos_id, pos_info in self.open_positions.items():
            # Kiểm tra trong entry_orders
            for order in pos_info["entry_orders"]:
                if order["order_id"] == order_id:
                    order["status"] = "canceled"
                    
                    # Nếu tất cả lệnh vào đều bị hủy, đánh dấu vị thế là đã hủy
                    all_canceled = all(o["status"] == "canceled" for o in pos_info["entry_orders"])
                    if all_canceled and not pos_info["exit_orders"]:
                        pos_info["status"] = "canceled"
                    
                    return
            
            # Kiểm tra trong exit_orders
            for order in pos_info["exit_orders"]:
                if order["order_id"] == order_id:
                    order["status"] = "canceled"
                    return
    
    def log_price_update(
        self,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Cập nhật giá hiện tại và tính toán unrealized P&L.
        
        Args:
            current_price: Giá hiện tại
            timestamp: Thời gian cập nhật
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Cập nhật giá và tính unrealized P&L cho các vị thế mở
        total_unrealized_pnl = 0.0
        
        for pos_id, pos_info in self.open_positions.items():
            if pos_info["status"] == "open":
                pos_info["current_price"] = current_price
                
                entry_price = pos_info["entry_price"]
                quantity = pos_info["quantity"]
                
                if pos_info["side"] == "long":
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                pos_info["unrealized_pnl"] = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl
        
        # Tổng giá trị danh mục
        portfolio_value = self.current_capital + total_unrealized_pnl
        
        # Lưu lịch sử vốn (với tần suất thấp hơn)
        hour_minute = timestamp.strftime("%H%M")
        if hour_minute.endswith("00") or hour_minute.endswith("30"):  # Mỗi 30 phút
            self.metrics_history["capital_history"].append({
                "timestamp": timestamp.isoformat(),
                "capital": self.current_capital,
                "portfolio_value": portfolio_value,
                "unrealized_pnl": total_unrealized_pnl,
                "event": "price_update"
            })
        
        # Cập nhật drawdown
        self._update_drawdown(portfolio_value, timestamp)
    
    def log_stop_loss_triggered(
        self,
        position_id: str,
        stop_price: float,
        original_price: float,
        loss_percent: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận khi stop loss được kích hoạt.
        
        Args:
            position_id: ID của vị thế
            stop_price: Giá dừng lỗ
            original_price: Giá ban đầu
            loss_percent: Phần trăm lỗ
            timestamp: Thời gian kích hoạt
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_stop_loss_triggered(
            position_id=position_id,
            stop_price=stop_price,
            original_price=original_price,
            loss_percent=loss_percent
        )
        
        # Kiểm tra vị thế tồn tại
        if position_id not in self.open_positions:
            self.logger.warning(f"Vị thế {position_id} không tồn tại để kích hoạt stop loss")
            return
        
        pos_info = self.open_positions[position_id]
        
        # Tính P&L
        quantity = pos_info["quantity"]
        entry_price = pos_info["entry_price"]
        
        if pos_info["side"] == "long":
            profit_loss = (stop_price - entry_price) * quantity
        else:  # short
            profit_loss = (entry_price - stop_price) * quantity
        
        profit_loss_percent = loss_percent  # Đã được tính ở ngoài
        
        # Cập nhật vị thế
        pos_info["status"] = "closed"
        pos_info["realized_pnl"] = profit_loss
        pos_info["exit_time"] = timestamp.isoformat()
        pos_info["exit_price"] = stop_price
        pos_info["exit_reason"] = "stop_loss"
        
        # Tính thời gian nắm giữ
        entry_time = datetime.fromisoformat(pos_info["entry_time"])
        duration = (timestamp - entry_time).total_seconds() / 3600  # Giờ
        
        # Cập nhật vốn (giả định không có phí cho stop loss)
        position_value = stop_price * quantity
        self.current_capital += position_value
        
        # Lưu lịch sử vốn
        self.metrics_history["capital_history"].append({
            "timestamp": timestamp.isoformat(),
            "capital": self.current_capital,
            "event": "stop_loss",
            "position_id": position_id
        })
        
        # Lưu lịch sử lợi nhuận/lỗ
        self.metrics_history["profit_loss_history"].append({
            "timestamp": timestamp.isoformat(),
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "position_id": position_id,
            "reason": "stop_loss"
        })
        
        # Lưu lịch sử thắng/thua (stop loss luôn là thua)
        self.metrics_history["win_history"].append({
            "timestamp": timestamp.isoformat(),
            "is_win": False,
            "position_id": position_id,
            "profit_loss": profit_loss,
            "reason": "stop_loss"
        })
        
        # Lưu thời gian nắm giữ
        self.metrics_history["trade_durations"].append({
            "position_id": position_id,
            "duration": duration,
            "entry_time": pos_info["entry_time"],
            "exit_time": timestamp.isoformat(),
            "reason": "stop_loss"
        })
        
        # Tăng biến đếm giao dịch
        self.trade_count += 1
        
        # Cập nhật thống kê hiệu suất
        self._update_performance_stats()
        
        # Ghi vào CSV nếu cần
        if self.use_csv:
            self._append_to_csv({
                "timestamp": timestamp.isoformat(),
                "trade_id": position_id,
                "order_type": "market",  # Stop loss thường thực hiện bằng lệnh market
                "side": "sell" if pos_info["side"] == "long" else "buy",
                "entry_price": entry_price,
                "exit_price": stop_price,
                "quantity": quantity,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "duration": duration,
                "fees": 0.0,  # Giả định không có phí
                "current_capital": self.current_capital,
                "order_status": "filled",
                "reason": "stop_loss"
            })
    
    def log_take_profit_triggered(
        self,
        position_id: str,
        take_profit_price: float,
        original_price: float,
        profit_percent: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận khi take profit được kích hoạt.
        
        Args:
            position_id: ID của vị thế
            take_profit_price: Giá chốt lời
            original_price: Giá ban đầu
            profit_percent: Phần trăm lợi nhuận
            timestamp: Thời gian kích hoạt
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_take_profit_triggered(
            position_id=position_id,
            take_profit_price=take_profit_price,
            original_price=original_price,
            profit_percent=profit_percent
        )
        
        # Kiểm tra vị thế tồn tại
        if position_id not in self.open_positions:
            self.logger.warning(f"Vị thế {position_id} không tồn tại để kích hoạt take profit")
            return
        
        pos_info = self.open_positions[position_id]
        
        # Tính P&L
        quantity = pos_info["quantity"]
        entry_price = pos_info["entry_price"]
        
        if pos_info["side"] == "long":
            profit_loss = (take_profit_price - entry_price) * quantity
        else:  # short
            profit_loss = (entry_price - take_profit_price) * quantity
        
        profit_loss_percent = profit_percent  # Đã được tính ở ngoài
        
        # Cập nhật vị thế
        pos_info["status"] = "closed"
        pos_info["realized_pnl"] = profit_loss
        pos_info["exit_time"] = timestamp.isoformat()
        pos_info["exit_price"] = take_profit_price
        pos_info["exit_reason"] = "take_profit"
        
        # Tính thời gian nắm giữ
        entry_time = datetime.fromisoformat(pos_info["entry_time"])
        duration = (timestamp - entry_time).total_seconds() / 3600  # Giờ
        
        # Cập nhật vốn (giả định không có phí cho take profit)
        position_value = take_profit_price * quantity
        self.current_capital += position_value
        
        # Lưu lịch sử vốn
        self.metrics_history["capital_history"].append({
            "timestamp": timestamp.isoformat(),
            "capital": self.current_capital,
            "event": "take_profit",
            "position_id": position_id
        })
        
        # Lưu lịch sử lợi nhuận/lỗ
        self.metrics_history["profit_loss_history"].append({
            "timestamp": timestamp.isoformat(),
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "position_id": position_id,
            "reason": "take_profit"
        })
        
        # Lưu lịch sử thắng/thua (take profit luôn là thắng)
        self.metrics_history["win_history"].append({
            "timestamp": timestamp.isoformat(),
            "is_win": True,
            "position_id": position_id,
            "profit_loss": profit_loss,
            "reason": "take_profit"
        })
        
        # Lưu thời gian nắm giữ
        self.metrics_history["trade_durations"].append({
            "position_id": position_id,
            "duration": duration,
            "entry_time": pos_info["entry_time"],
            "exit_time": timestamp.isoformat(),
            "reason": "take_profit"
        })
        
        # Tăng biến đếm giao dịch
        self.trade_count += 1
        
        # Cập nhật thống kê hiệu suất
        self._update_performance_stats()
        
        # Ghi vào CSV nếu cần
        if self.use_csv:
            self._append_to_csv({
                "timestamp": timestamp.isoformat(),
                "trade_id": position_id,
                "order_type": "market",  # Take profit thường thực hiện bằng lệnh market
                "side": "sell" if pos_info["side"] == "long" else "buy",
                "entry_price": entry_price,
                "exit_price": take_profit_price,
                "quantity": quantity,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "duration": duration,
                "fees": 0.0,  # Giả định không có phí
                "current_capital": self.current_capital,
                "order_status": "filled",
                "reason": "take_profit"
            })
    
    def log_strategy_signal(
        self,
        strategy_name: str,
        signal_type: str,
        confidence: float,
        parameters: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận tín hiệu từ chiến lược giao dịch.
        
        Args:
            strategy_name: Tên chiến lược
            signal_type: Loại tín hiệu (buy, sell, hold)
            confidence: Độ tin cậy của tín hiệu (0.0-1.0)
            parameters: Các tham số bổ sung
            timestamp: Thời gian phát tín hiệu
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_strategy_signal(
            strategy_name=strategy_name,
            signal_type=signal_type,
            confidence=confidence,
            parameters=parameters
        )
    
    def log_risk_management(
        self,
        action: str,
        value: float,
        original_value: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận hành động quản lý rủi ro.
        
        Args:
            action: Loại hành động (position_size, stop_loss, take_profit)
            value: Giá trị mới
            original_value: Giá trị ban đầu
            timestamp: Thời gian thực hiện
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_risk_management(
            action=action,
            value=value,
            original_value=original_value
        )
    
    def log_portfolio_update(
        self,
        total_value: float,
        positions: Dict[str, Any],
        available_balance: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận cập nhật danh mục đầu tư.
        
        Args:
            total_value: Tổng giá trị danh mục
            positions: Thông tin về các vị thế
            available_balance: Số dư khả dụng
            timestamp: Thời gian cập nhật
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_portfolio_update(
            total_value=total_value,
            positions=positions,
            available_balance=available_balance
        )
        
        # Lưu lịch sử vốn
        self.metrics_history["capital_history"].append({
            "timestamp": timestamp.isoformat(),
            "capital": available_balance,
            "portfolio_value": total_value,
            "position_count": len(positions),
            "event": "portfolio_update"
        })
    
    def log_error(
        self,
        error_msg: str,
        error_code: Optional[int] = None,
        exception: Optional[Exception] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Ghi nhận lỗi giao dịch.
        
        Args:
            error_msg: Thông điệp lỗi
            error_code: Mã lỗi
            exception: Đối tượng Exception
            timestamp: Thời gian xảy ra lỗi
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ghi log
        self.trade_logger.log_error(
            error_msg=error_msg,
            error_code=error_code,
            exception=exception
        )
    
    def _append_to_csv(self, data: Dict[str, Any]) -> None:
        """
        Thêm dữ liệu vào file CSV.
        
        Args:
            data: Dữ liệu cần thêm
        """
        try:
            # Tạo DataFrame mới từ data
            new_data = pd.DataFrame([data])
            
            # Thêm vào file CSV
            new_data.to_csv(self.csv_path, mode='a', header=False, index=False)
            
        except Exception as e:
            self.logger.warning(f"Không thể ghi số liệu vào CSV: {str(e)}")
    
    def _update_performance_stats(self) -> None:
        """
        Cập nhật các thống kê hiệu suất.
        """
        # Số lượng giao dịch
        self.performance_stats["total_trades"] = self.trade_count
        
        # Nếu chưa có giao dịch nào, không cần cập nhật thêm
        if self.trade_count == 0:
            return
        
        # Tính số lượng thắng/thua
        wins = [trade for trade in self.metrics_history["win_history"] if trade["is_win"]]
        losses = [trade for trade in self.metrics_history["win_history"] if not trade["is_win"]]
        
        self.performance_stats["winning_trades"] = len(wins)
        self.performance_stats["losing_trades"] = len(losses)
        
        # Tính win rate
        if self.trade_count > 0:
            self.performance_stats["win_rate"] = len(wins) / self.trade_count
        
        # Tính trung bình lợi nhuận/lỗ
        if len(wins) > 0:
            self.performance_stats["average_profit"] = sum(trade["profit_loss"] for trade in wins) / len(wins)
        
        if len(losses) > 0:
            self.performance_stats["average_loss"] = sum(abs(trade["profit_loss"]) for trade in losses) / len(losses)
        
        # Tính profit factor
        total_profit = sum(trade["profit_loss"] for trade in wins)
        total_loss = sum(abs(trade["profit_loss"]) for trade in losses)
        
        if total_loss > 0:
            self.performance_stats["profit_factor"] = total_profit / total_loss
        else:
            self.performance_stats["profit_factor"] = float('inf') if total_profit > 0 else 0.0
        
        # Tính tổng lợi nhuận/lỗ
        self.performance_stats["total_profit_loss"] = total_profit - total_loss
        
        # Tính ROI
        self.performance_stats["roi"] = (self.current_capital / self.initial_capital) - 1
        
        # Tính thời gian trung bình của giao dịch
        if len(self.metrics_history["trade_durations"]) > 0:
            durations = [trade["duration"] for trade in self.metrics_history["trade_durations"]]
            self.performance_stats["average_trade_duration"] = sum(durations) / len(durations)
        
        # Tính tổng phí
        if len(self.metrics_history["fees_paid"]) > 0:
            self.performance_stats["total_fees"] = sum(fee["fees"] for fee in self.metrics_history["fees_paid"])
        
        # Cập nhật các tỷ lệ khác nếu có đủ dữ liệu
        self._update_drawdown_stats()
        self._update_ratio_stats()
    
    def _update_drawdown_stats(self) -> None:
        """
        Cập nhật thống kê về drawdown.
        """
        if len(self.metrics_history["capital_history"]) < 2:
            return
        
        # Tạo chuỗi vốn
        capital_series = pd.Series([
            entry["capital"] 
            for entry in self.metrics_history["capital_history"]
        ])
        
        # Tính peak-to-trough cho mỗi điểm
        rolling_max = capital_series.cummax()
        drawdown = (capital_series - rolling_max) / rolling_max
        
        # Tìm max drawdown
        max_drawdown = abs(drawdown.min())
        self.performance_stats["max_drawdown"] = max_drawdown * self.initial_capital
        self.performance_stats["max_drawdown_percent"] = max_drawdown
    
    def _update_drawdown(self, current_value: float, timestamp: datetime) -> None:
        """
        Cập nhật thông tin drawdown.
        
        Args:
            current_value: Giá trị danh mục hiện tại
            timestamp: Thời gian cập nhật
        """
        # Nếu chưa có drawdown nào
        if not self.metrics_history["drawdowns"]:
            # Khởi tạo giá trị peak ban đầu
            self.metrics_history["drawdowns"].append({
                "peak_value": current_value,
                "peak_time": timestamp.isoformat(),
                "trough_value": current_value,
                "trough_time": timestamp.isoformat(),
                "recovery_value": None,
                "recovery_time": None,
                "drawdown_percent": 0.0,
                "drawdown_amount": 0.0,
                "duration": 0.0,
                "status": "active"
            })
            return
        
        # Lấy drawdown hiện tại
        current_drawdown = self.metrics_history["drawdowns"][-1]
        
        # Nếu drawdown hiện tại đã kết thúc, tạo mới
        if current_drawdown["status"] == "recovered":
            self.metrics_history["drawdowns"].append({
                "peak_value": current_value,
                "peak_time": timestamp.isoformat(),
                "trough_value": current_value,
                "trough_time": timestamp.isoformat(),
                "recovery_value": None,
                "recovery_time": None,
                "drawdown_percent": 0.0,
                "drawdown_amount": 0.0,
                "duration": 0.0,
                "status": "active"
            })
            return
        
        # Cập nhật peak nếu giá trị hiện tại cao hơn
        if current_value > current_drawdown["peak_value"]:
            current_drawdown["peak_value"] = current_value
            current_drawdown["peak_time"] = timestamp.isoformat()
            current_drawdown["trough_value"] = current_value
            current_drawdown["trough_time"] = timestamp.isoformat()
            current_drawdown["drawdown_percent"] = 0.0
            current_drawdown["drawdown_amount"] = 0.0
            current_drawdown["duration"] = 0.0
            return
        
        # Cập nhật trough nếu giá trị hiện tại thấp hơn
        if current_value < current_drawdown["trough_value"]:
            current_drawdown["trough_value"] = current_value
            current_drawdown["trough_time"] = timestamp.isoformat()
            
            # Tính drawdown
            peak_value = current_drawdown["peak_value"]
            drawdown_amount = peak_value - current_value
            drawdown_percent = drawdown_amount / peak_value
            
            current_drawdown["drawdown_amount"] = drawdown_amount
            current_drawdown["drawdown_percent"] = drawdown_percent
            
            # Tính thời gian
            peak_time = datetime.fromisoformat(current_drawdown["peak_time"])
            duration = (timestamp - peak_time).total_seconds() / (24 * 3600)  # Days
            current_drawdown["duration"] = duration
        
        # Kiểm tra xem đã phục hồi chưa
        if current_value >= current_drawdown["peak_value"] and current_drawdown["drawdown_percent"] > 0:
            current_drawdown["recovery_value"] = current_value
            current_drawdown["recovery_time"] = timestamp.isoformat()
            current_drawdown["status"] = "recovered"
            
            # Tính thời gian phục hồi
            trough_time = datetime.fromisoformat(current_drawdown["trough_time"])
            recovery_duration = (timestamp - trough_time).total_seconds() / (24 * 3600)  # Days
            current_drawdown["recovery_duration"] = recovery_duration
    
    def _update_ratio_stats(self) -> None:
        """
        Cập nhật các tỷ lệ đánh giá hiệu suất.
        """
        # Nếu chưa đủ dữ liệu
        if len(self.metrics_history["capital_history"]) < 10:
            return
        
        try:
            # Tạo chuỗi về lợi nhuận hàng ngày
            # Nhóm theo ngày và lấy giá trị cuối cùng của mỗi ngày
            capital_history = self.metrics_history["capital_history"]
            daily_values = {}
            
            for entry in capital_history:
                date = entry["timestamp"].split("T")[0]
                daily_values[date] = entry["capital"]
            
            # Chuyển thành series
            daily_series = pd.Series(daily_values)
            daily_series.index = pd.to_datetime(daily_series.index)
            daily_series = daily_series.sort_index()
            
            # Tính lợi nhuận hàng ngày
            daily_returns = daily_series.pct_change().dropna()
            
            # Lưu vào metrics_history
            self.metrics_history["daily_returns"] = {
                str(date.date()): return_value 
                for date, return_value in daily_returns.items()
            }
            
            # Tính Sharpe ratio
            risk_free_rate = 0.02 / 252  # Giả định 2% hàng năm
            excess_returns = daily_returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
            self.performance_stats["sharpe_ratio"] = sharpe_ratio
            
            # Tính Sortino ratio
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std() * np.sqrt(252)
                sortino_ratio = (daily_returns.mean() - risk_free_rate) * 252 / downside_deviation
                self.performance_stats["sortino_ratio"] = sortino_ratio
            
            # Tính Calmar ratio
            if self.performance_stats["max_drawdown_percent"] > 0:
                annualized_return = (1 + self.performance_stats["roi"]) ** (252 / len(daily_returns)) - 1
                calmar_ratio = annualized_return / self.performance_stats["max_drawdown_percent"]
                self.performance_stats["calmar_ratio"] = calmar_ratio
            
            # Tính Expectancy
            average_win = self.performance_stats.get("average_profit", 0)
            average_loss = self.performance_stats.get("average_loss", 0)
            win_rate = self.performance_stats.get("win_rate", 0)
            
            if win_rate > 0 and average_loss > 0:
                expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)
                self.performance_stats["expectancy"] = expectancy
        
        except Exception as e:
            self.logger.warning(f"Không thể cập nhật tỷ lệ hiệu suất: {str(e)}")
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Tính toán tất cả các chỉ số hiệu suất giao dịch.
        
        Returns:
            Dict chứa tất cả các chỉ số hiệu suất
        """
        # Cập nhật thống kê hiệu suất
        self._update_performance_stats()
        
        # Tạo dữ liệu để sử dụng với PerformanceMetrics
        if len(self.metrics_history["capital_history"]) > 0:
            # Tạo DataFrame từ lịch sử vốn
            capital_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"]
                }
                for entry in self.metrics_history["capital_history"]
            ])
            
            capital_data["timestamp"] = pd.to_datetime(capital_data["timestamp"])
            capital_data.set_index("timestamp", inplace=True)
            capital_data.sort_index(inplace=True)
            
            # Loại bỏ các dòng trùng lặp (giữ cuối cùng)
            capital_data = capital_data[~capital_data.index.duplicated(keep='last')]
            
            # Tạo DataFrame từ lịch sử giao dịch
            if len(self.metrics_history["profit_loss_history"]) > 0:
                trades_data = pd.DataFrame([
                    {
                        "entry_time": self.open_positions.get(trade["position_id"], {}).get("entry_time", ""),
                        "exit_time": datetime.fromisoformat(trade["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "profit": trade["profit_loss"],
                        "duration": next((d["duration"] for d in self.metrics_history["trade_durations"] 
                                        if d["position_id"] == trade["position_id"]), 0)
                    }
                    for trade in self.metrics_history["profit_loss_history"]
                ])
                
                if not trades_data.empty:
                    # Chuyển đổi thời gian
                    trades_data["entry_time"] = pd.to_datetime(trades_data["entry_time"])
                    trades_data["exit_time"] = pd.to_datetime(trades_data["exit_time"])
                    
                    # Sử dụng PerformanceMetrics
                    try:
                        perf_metrics = PerformanceMetrics(
                            equity_curve=capital_data["capital"],
                            trades=trades_data,
                            initial_capital=self.initial_capital
                        )
                        
                        detailed_metrics = perf_metrics.calculate_all_metrics()
                        
                        # Cập nhật performance_stats với các số liệu chi tiết hơn
                        self.performance_stats.update(detailed_metrics)
                    except Exception as e:
                        self.logger.warning(f"Không thể tính toán chỉ số chi tiết: {str(e)}")
        
        return self.performance_stats
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tóm tắt về quá trình giao dịch.
        
        Returns:
            Dict chứa các thống kê
        """
        # Cập nhật thống kê hiệu suất
        self._update_performance_stats()
        
        summary = {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "roi": self.performance_stats["roi"],
            "roi_percent": self.performance_stats["roi"] * 100,
            "total_trades": self.performance_stats["total_trades"],
            "winning_trades": self.performance_stats["winning_trades"],
            "losing_trades": self.performance_stats["losing_trades"],
            "win_rate": self.performance_stats["win_rate"],
            "win_rate_percent": self.performance_stats["win_rate"] * 100 if self.performance_stats["win_rate"] is not None else None,
            "profit_factor": self.performance_stats["profit_factor"],
            "average_profit": self.performance_stats["average_profit"],
            "average_loss": self.performance_stats["average_loss"],
            "max_drawdown": self.performance_stats["max_drawdown"],
            "max_drawdown_percent": self.performance_stats["max_drawdown_percent"] * 100,
            "average_trade_duration": self.performance_stats["average_trade_duration"],
            "total_fees": self.performance_stats["total_fees"],
            "sharpe_ratio": self.performance_stats.get("sharpe_ratio", None),
            "sortino_ratio": self.performance_stats.get("sortino_ratio", None),
            "calmar_ratio": self.performance_stats.get("calmar_ratio", None),
            "expectancy": self.performance_stats.get("expectancy", None),
            "trading_period": {
                "start_time": self.metrics_history["capital_history"][0]["timestamp"] if len(self.metrics_history["capital_history"]) > 0 else None,
                "end_time": datetime.now().isoformat(),
                "days": (datetime.now() - datetime.fromisoformat(self.metrics_history["capital_history"][0]["timestamp"])).days if len(self.metrics_history["capital_history"]) > 0 else 0
            }
        }
        
        return summary
    
    def save_metrics(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Lưu lịch sử số liệu vào file JSON.
        
        Args:
            path: Đường dẫn file (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo đường dẫn tự động nếu không được cung cấp
        if path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            path = self.output_dir / f"{self.symbol.replace('/', '_')}_{current_date}_metrics.json"
        else:
            path = Path(path)
        
        # Cập nhật thống kê hiệu suất
        self._update_performance_stats()
        
        # Chuẩn bị dữ liệu để lưu
        save_data = {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_trades": self.trade_count,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "trading_duration": time.time() - self.start_time,
            "metrics_history": self.metrics_history,
            "performance_stats": self.performance_stats
        }
        
        # Lưu vào file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu số liệu giao dịch vào {path}")
        return str(path)
    
    def load_metrics(self, path: Union[str, Path]) -> bool:
        """
        Tải số liệu từ file JSON.
        
        Args:
            path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            path = Path(path)
            
            # Đọc file
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cập nhật thông tin
            self.symbol = data.get("symbol", self.symbol)
            self.strategy_name = data.get("strategy_name", self.strategy_name)
            self.initial_capital = data.get("initial_capital", self.initial_capital)
            self.current_capital = data.get("current_capital", self.current_capital)
            self.trade_count = data.get("total_trades", 0)
            
            # Cập nhật lịch sử số liệu
            if "metrics_history" in data:
                self.metrics_history = data["metrics_history"]
            
            # Cập nhật thống kê hiệu suất
            if "performance_stats" in data:
                self.performance_stats = data["performance_stats"]
            
            self.logger.info(f"Đã tải số liệu từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Không thể tải số liệu từ {path}: {str(e)}")
            return False
    
    def plot_metrics(self, save_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Tạo và lưu biểu đồ số liệu giao dịch.
        
        Args:
            save_path: Thư mục lưu biểu đồ (None để sử dụng output_dir)
            
        Returns:
            Danh sách đường dẫn các file biểu đồ đã lưu
        """
        # Kiểm tra nếu không có dữ liệu
        if len(self.metrics_history["capital_history"]) == 0:
            self.logger.warning("Không có dữ liệu để tạo biểu đồ")
            return []
        
        # Thiết lập thư mục lưu
        if save_path is None:
            save_path = self.output_dir / "plots"
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Danh sách để lưu đường dẫn các file
        saved_paths = []
        
        # ---------- 1. Biểu đồ vốn ----------
        try:
            plt.figure(figsize=(12, 6))
            
            # Tạo DataFrame từ lịch sử vốn
            capital_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"]
                }
                for entry in self.metrics_history["capital_history"]
            ])
            
            capital_data["timestamp"] = pd.to_datetime(capital_data["timestamp"])
            capital_data.set_index("timestamp", inplace=True)
            capital_data.sort_index(inplace=True)
            
            # Loại bỏ các dòng trùng lặp (giữ cuối cùng)
            capital_data = capital_data[~capital_data.index.duplicated(keep='last')]
            
            # Tạo biểu đồ
            plt.plot(capital_data.index, capital_data["capital"], 'b-', label='Vốn')
            
            # Đánh dấu các giao dịch
            for trade in self.metrics_history["profit_loss_history"]:
                timestamp = datetime.fromisoformat(trade["timestamp"])
                profit_loss = trade["profit_loss"]
                
                if profit_loss > 0:
                    color = 'g^'  # Tam giác xanh lá cho lợi nhuận
                else:
                    color = 'rv'  # Tam giác đỏ cho lỗ
                
                plt.plot([timestamp], [capital_data.loc[capital_data.index <= timestamp].iloc[-1]["capital"]], color)
            
            plt.grid(True, alpha=0.3)
            plt.title(f'Biến động vốn - {self.symbol}')
            plt.xlabel('Thời gian')
            plt.ylabel('Vốn')
            
            # Thêm chú thích
            plt.legend()
            
            # Lưu biểu đồ
            capital_path = save_path / f"{self.symbol.replace('/', '_')}_capital.png"
            plt.savefig(capital_path)
            plt.close()
            saved_paths.append(capital_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ vốn: {str(e)}")
        
        # ---------- 2. Biểu đồ lợi nhuận/lỗ tích lũy ----------
        try:
            if len(self.metrics_history["profit_loss_history"]) > 0:
                plt.figure(figsize=(12, 6))
                
                # Tạo DataFrame từ lịch sử lợi nhuận/lỗ
                profit_loss_data = pd.DataFrame([
                    {
                        "timestamp": trade["timestamp"],
                        "profit_loss": trade["profit_loss"]
                    }
                    for trade in self.metrics_history["profit_loss_history"]
                ])
                
                profit_loss_data["timestamp"] = pd.to_datetime(profit_loss_data["timestamp"])
                profit_loss_data.set_index("timestamp", inplace=True)
                profit_loss_data.sort_index(inplace=True)
                
                # Tính lợi nhuận/lỗ tích lũy
                profit_loss_data["cumulative"] = profit_loss_data["profit_loss"].cumsum()
                
                # Tạo biểu đồ
                plt.plot(profit_loss_data.index, profit_loss_data["cumulative"], 'g-', label='Lợi nhuận/Lỗ tích lũy')
                
                # Đánh dấu các giao dịch
                for idx, row in profit_loss_data.iterrows():
                    if row["profit_loss"] > 0:
                        color = 'g^'  # Tam giác xanh lá cho lợi nhuận
                    else:
                        color = 'rv'  # Tam giác đỏ cho lỗ
                    
                    plt.plot([idx], [row["cumulative"]], color)
                
                plt.grid(True, alpha=0.3)
                plt.title(f'Lợi nhuận/Lỗ tích lũy - {self.symbol}')
                plt.xlabel('Thời gian')
                plt.ylabel('Lợi nhuận/Lỗ')
                
                # Thêm đường tham chiếu 0
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Thêm chú thích
                plt.legend()
                
                # Lưu biểu đồ
                profit_path = save_path / f"{self.symbol.replace('/', '_')}_profit_loss.png"
                plt.savefig(profit_path)
                plt.close()
                saved_paths.append(profit_path)
        
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ lợi nhuận/lỗ: {str(e)}")
        
        # ---------- 3. Biểu đồ thống kê giao dịch ----------
        try:
            if len(self.metrics_history["win_history"]) > 0:
                plt.figure(figsize=(12, 6))
                
                # Tính tỷ lệ thắng/thua
                win_count = len([trade for trade in self.metrics_history["win_history"] if trade["is_win"]])
                loss_count = len([trade for trade in self.metrics_history["win_history"] if not trade["is_win"]])
                
                # Tạo biểu đồ bánh
                labels = ['Thắng', 'Thua']
                sizes = [win_count, loss_count]
                colors = ['green', 'red']
                explode = (0.1, 0)  # Nhấn mạnh phần thắng
                
                plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
                plt.axis('equal')  # Đảm bảo biểu đồ hình tròn
                
                plt.title(f'Thống kê thắng/thua - {self.symbol}')
                
                # Lưu biểu đồ
                stats_path = save_path / f"{self.symbol.replace('/', '_')}_trade_stats.png"
                plt.savefig(stats_path)
                plt.close()
                saved_paths.append(stats_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ thống kê giao dịch: {str(e)}")
        
        # ---------- 4. Biểu đồ drawdown ----------
        try:
            plt.figure(figsize=(12, 6))
            
            # Tạo DataFrame từ lịch sử vốn
            capital_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"]
                }
                for entry in self.metrics_history["capital_history"]
            ])
            
            capital_data["timestamp"] = pd.to_datetime(capital_data["timestamp"])
            capital_data.set_index("timestamp", inplace=True)
            capital_data.sort_index(inplace=True)
            
            # Loại bỏ các dòng trùng lặp (giữ cuối cùng)
            capital_data = capital_data[~capital_data.index.duplicated(keep='last')]
            
            # Tính drawdown
            capital_series = capital_data["capital"]
            rolling_max = capital_series.cummax()
            drawdown = (capital_series - rolling_max) / rolling_max * 100
            
            # Tạo biểu đồ
            plt.fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
            plt.plot(drawdown.index, drawdown, 'r-', label='Drawdown')
            
            plt.grid(True, alpha=0.3)
            plt.title(f'Drawdown - {self.symbol}')
            plt.xlabel('Thời gian')
            plt.ylabel('Drawdown (%)')
            
            # Thêm chú thích
            plt.legend()
            
            # Lưu biểu đồ
            drawdown_path = save_path / f"{self.symbol.replace('/', '_')}_drawdown.png"
            plt.savefig(drawdown_path)
            plt.close()
            saved_paths.append(drawdown_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ drawdown: {str(e)}")
        
        # ---------- 5. Biểu đồ lợi nhuận phân phối ----------
        try:
            if len(self.metrics_history["profit_loss_history"]) > 10:
                plt.figure(figsize=(12, 6))
                
                # Lấy dữ liệu lợi nhuận/lỗ
                profits = [trade["profit_loss"] for trade in self.metrics_history["profit_loss_history"]]
                
                # Tạo biểu đồ histogram
                plt.hist(profits, bins=20, alpha=0.7, color='blue')
                plt.axvline(0, color='r', linestyle='--')  # Đường phân cách thắng/thua
                
                # Thêm giá trị trung bình
                mean_profit = np.mean(profits)
                plt.axvline(mean_profit, color='g', linestyle='--', label=f'Trung bình: {mean_profit:.2f}')
                
                plt.grid(True, alpha=0.3)
                plt.title(f'Phân phối lợi nhuận/lỗ - {self.symbol}')
                plt.xlabel('Lợi nhuận/Lỗ')
                plt.ylabel('Số lượng giao dịch')
                
                # Thêm chú thích
                plt.legend()
                
                # Lưu biểu đồ
                dist_path = save_path / f"{self.symbol.replace('/', '_')}_profit_distribution.png"
                plt.savefig(dist_path)
                plt.close()
                saved_paths.append(dist_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ phân phối lợi nhuận: {str(e)}")
        
        # ---------- 6. Biểu đồ tổng hợp ----------
        try:
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Biểu đồ vốn
            capital_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"]
                }
                for entry in self.metrics_history["capital_history"]
            ])
            
            capital_data["timestamp"] = pd.to_datetime(capital_data["timestamp"])
            capital_data.set_index("timestamp", inplace=True)
            capital_data.sort_index(inplace=True)
            capital_data = capital_data[~capital_data.index.duplicated(keep='last')]
            
            axs[0, 0].plot(capital_data.index, capital_data["capital"], 'b-')
            axs[0, 0].set_title('Biến động vốn')
            axs[0, 0].grid(True, alpha=0.3)
            
            # 2. Lợi nhuận/lỗ tích lũy
            if len(self.metrics_history["profit_loss_history"]) > 0:
                profit_loss_data = pd.DataFrame([
                    {
                        "timestamp": trade["timestamp"],
                        "profit_loss": trade["profit_loss"]
                    }
                    for trade in self.metrics_history["profit_loss_history"]
                ])
                
                profit_loss_data["timestamp"] = pd.to_datetime(profit_loss_data["timestamp"])
                profit_loss_data.set_index("timestamp", inplace=True)
                profit_loss_data.sort_index(inplace=True)
                profit_loss_data["cumulative"] = profit_loss_data["profit_loss"].cumsum()
                
                axs[0, 1].plot(profit_loss_data.index, profit_loss_data["cumulative"], 'g-')
                axs[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
                axs[0, 1].set_title('Lợi nhuận/Lỗ tích lũy')
                axs[0, 1].grid(True, alpha=0.3)
            
            # 3. Drawdown
            drawdown = (capital_data["capital"] - capital_data["capital"].cummax()) / capital_data["capital"].cummax() * 100
            axs[1, 0].fill_between(drawdown.index, 0, drawdown, color='r', alpha=0.3)
            axs[1, 0].plot(drawdown.index, drawdown, 'r-')
            axs[1, 0].set_title('Drawdown')
            axs[1, 0].grid(True, alpha=0.3)
            
            # 4. Thống kê thắng/thua
            if len(self.metrics_history["win_history"]) > 0:
                win_count = len([trade for trade in self.metrics_history["win_history"] if trade["is_win"]])
                loss_count = len([trade for trade in self.metrics_history["win_history"] if not trade["is_win"]])
                
                labels = ['Thắng', 'Thua']
                sizes = [win_count, loss_count]
                colors = ['green', 'red']
                
                axs[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axs[1, 1].set_title('Thống kê thắng/thua')
                axs[1, 1].axis('equal')
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            summary_path = save_path / f"{self.symbol.replace('/', '_')}_summary.png"
            plt.savefig(summary_path)
            plt.close()
            saved_paths.append(summary_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ tổng hợp: {str(e)}")
        
        return saved_paths
    
    def generate_trading_report(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """
        Tạo báo cáo Markdown về quá trình giao dịch.
        
        Args:
            save_path: Đường dẫn file báo cáo (None để tạo tự động)
            
        Returns:
            Đường dẫn file báo cáo
        """
        # Thiết lập đường dẫn lưu
        if save_path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            save_path = self.output_dir / f"{self.symbol.replace('/', '_')}_{current_date}_report.md"
        else:
            save_path = Path(save_path)
        
        # Tạo thư mục nếu chưa tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo biểu đồ và lưu
        plot_paths = self.plot_metrics()
        
        # Chuẩn bị dữ liệu cho báo cáo
        stats = self.get_summary_stats()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo báo cáo
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Báo cáo giao dịch: {self.symbol}\n\n")
            
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian báo cáo:** {current_time}\n")
            f.write(f"- **Cặp giao dịch:** {self.symbol}\n")
            f.write(f"- **Chiến lược:** {self.strategy_name}\n")
            f.write(f"- **Vốn ban đầu:** {self.initial_capital:.2f}\n")
            f.write(f"- **Vốn hiện tại:** {self.current_capital:.2f}\n")
            f.write(f"- **Tổng số giao dịch:** {stats['total_trades']}\n")
            
            trading_period = stats.get('trading_period', {})
            if trading_period.get('start_time') and trading_period.get('end_time'):
                f.write(f"- **Thời gian giao dịch:** {trading_period.get('days', 0)} ngày\n")
                f.write(f"  - **Bắt đầu:** {datetime.fromisoformat(trading_period['start_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  - **Kết thúc:** {datetime.fromisoformat(trading_period['end_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            f.write("\n## Thống kê hiệu suất\n\n")
            f.write(f"- **ROI:** {stats.get('roi_percent', 0):.2f}%\n")
            f.write(f"- **Win rate:** {stats.get('win_rate_percent', 0):.2f}%\n")
            f.write(f"- **Profit factor:** {stats.get('profit_factor', 0):.2f}\n")
            f.write(f"- **Max drawdown:** {stats.get('max_drawdown_percent', 0):.2f}%\n")
            f.write(f"- **Thời gian giao dịch trung bình:** {stats.get('average_trade_duration', 0):.2f} giờ\n")
            f.write(f"- **Sharpe ratio:** {stats.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"- **Sortino ratio:** {stats.get('sortino_ratio', 0):.2f}\n")
            f.write(f"- **Calmar ratio:** {stats.get('calmar_ratio', 0):.2f}\n")
            f.write(f"- **Expectancy:** {stats.get('expectancy', 0):.2f}\n")
            f.write(f"- **Số giao dịch thắng:** {stats.get('winning_trades', 0)}\n")
            f.write(f"- **Số giao dịch thua:** {stats.get('losing_trades', 0)}\n")
            f.write(f"- **Lợi nhuận trung bình (thắng):** {stats.get('average_profit', 0):.2f}\n")
            f.write(f"- **Lỗ trung bình (thua):** {stats.get('average_loss', 0):.2f}\n")
            f.write(f"- **Tổng phí giao dịch:** {stats.get('total_fees', 0):.2f}\n")
            
            # Hiển thị biểu đồ
            f.write("\n## Biểu đồ phân tích\n\n")
            
            for plot_path in plot_paths:
                rel_path = os.path.relpath(plot_path, save_path.parent)
                plot_name = os.path.splitext(os.path.basename(plot_path))[0].replace(f"{self.symbol.replace('/', '_')}_", "")
                plot_title = ' '.join(word.capitalize() for word in plot_name.split('_'))
                f.write(f"### {plot_title}\n\n")
                f.write(f"![{plot_title}]({rel_path})\n\n")
            
            # Phân tích giao dịch gần đây
            f.write("\n## Phân tích giao dịch gần đây\n\n")
            
            recent_trades = self.metrics_history["profit_loss_history"][-5:] if len(self.metrics_history["profit_loss_history"]) >= 5 else self.metrics_history["profit_loss_history"]
            
            if recent_trades:
                f.write("| Thời gian | Lợi nhuận/Lỗ | % |\n")
                f.write("|-----------|--------------|----|\n")
                
                for trade in reversed(recent_trades):
                    timestamp = datetime.fromisoformat(trade["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    profit_loss = trade["profit_loss"]
                    percent = trade.get("profit_loss_percent", 0)
                    
                    f.write(f"| {timestamp} | {profit_loss:.2f} | {percent:.2f}% |\n")
            else:
                f.write("Chưa có giao dịch nào được thực hiện.\n")
            
            # Thêm phần kết luận
            f.write("\n## Kết luận và đề xuất\n\n")
            
            # Đánh giá hiệu suất
            roi = stats.get('roi', 0)
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            
            if roi > 0.1:  # ROI > 10%
                f.write("Chiến lược đang có hiệu suất **tốt** với ROI dương đáng kể. ")
            elif roi > 0:
                f.write("Chiến lược đang có hiệu suất **khá** với ROI dương nhỏ. ")
            else:
                f.write("Chiến lược đang có hiệu suất **kém** với ROI âm. ")
            
            if win_rate > 0.6:
                f.write(f"Tỷ lệ thắng cao ({win_rate:.2%}) cho thấy chiến lược có độ tin cậy tốt. ")
            elif win_rate > 0.5:
                f.write(f"Tỷ lệ thắng chấp nhận được ({win_rate:.2%}). ")
            else:
                f.write(f"Tỷ lệ thắng thấp ({win_rate:.2%}) cần được cải thiện. ")
            
            if profit_factor > 2:
                f.write(f"Profit factor cao ({profit_factor:.2f}) cho thấy tỷ lệ lợi nhuận/rủi ro rất tốt.\n\n")
            elif profit_factor > 1.5:
                f.write(f"Profit factor khá ({profit_factor:.2f}) cho thấy quản lý rủi ro hiệu quả.\n\n")
            elif profit_factor > 1:
                f.write(f"Profit factor chấp nhận được ({profit_factor:.2f}).\n\n")
            else:
                f.write(f"Profit factor thấp ({profit_factor:.2f}) cảnh báo rủi ro cao hơn lợi nhuận kỳ vọng.\n\n")
            
            # Đề xuất
            f.write("### Đề xuất cải thiện\n\n")
            
            max_drawdown = stats.get('max_drawdown_percent', 0)
            avg_trade_duration = stats.get('average_trade_duration', 0)
            
            if max_drawdown > 20:
                f.write("- **Quản lý rủi ro:** Xem xét thắt chặt stop loss để giảm drawdown tối đa.\n")
            
            if win_rate < 0.5:
                f.write("- **Tỷ lệ thắng:** Đánh giá lại điểm vào lệnh để cải thiện tỷ lệ thắng.\n")
            
            avg_profit = stats.get('average_profit', 0)
            avg_loss = stats.get('average_loss', 0)
            
            if avg_profit > 0 and avg_loss > 0 and avg_profit < avg_loss:
                f.write("- **Tỷ lệ R:R:** Cần giữ lợi nhuận lâu hơn và cắt lỗ sớm hơn để cải thiện tỷ lệ risk:reward.\n")
            
            f.write("- **Khung thời gian:** Đánh giá hiệu suất trên các khung thời gian khác nhau để tìm ra khung thời gian tối ưu.\n")
            f.write("- **Cặp tiền khác:** Thử nghiệm chiến lược trên các cặp tiền khác để đa dạng hóa.\n")
            f.write("- **Tối ưu hóa tham số:** Điều chỉnh các tham số của chiến lược để tối ưu hóa hiệu suất.\n")
        
        self.logger.info(f"Đã tạo báo cáo giao dịch tại {save_path}")
        return str(save_path)
    
    def compare_with_benchmark(
        self,
        benchmark_data: Union[pd.DataFrame, Dict[str, float], List[float]],
        benchmark_name: str = "Benchmark",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất giao dịch với benchmark.
        
        Args:
            benchmark_data: Dữ liệu benchmark (DataFrame với index là timestamp và cột 'value',
                             hoặc Dict với key là timestamp và value là giá trị,
                             hoặc List các giá trị theo thứ tự thời gian)
            benchmark_name: Tên benchmark
            start_date: Ngày bắt đầu so sánh
            end_date: Ngày kết thúc so sánh
            save_path: Đường dẫn lưu biểu đồ
            
        Returns:
            Dict chứa kết quả so sánh
        """
        try:
            # Chuyển đổi benchmark_data sang DataFrame nếu cần
            if isinstance(benchmark_data, dict):
                df = pd.DataFrame({
                    'value': pd.Series(benchmark_data)
                })
            elif isinstance(benchmark_data, list):
                # Nếu là danh sách, giả định các giá trị theo thời gian đều nhau
                if len(self.metrics_history["capital_history"]) < len(benchmark_data):
                    # Không đủ điểm dữ liệu
                    raise ValueError("Không đủ điểm dữ liệu vốn để so sánh với benchmark")
                
                timestamps = [entry["timestamp"] for entry in self.metrics_history["capital_history"]][:len(benchmark_data)]
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'value': benchmark_data
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif isinstance(benchmark_data, pd.DataFrame):
                df = benchmark_data.copy()
                if 'value' not in df.columns:
                    # Giả định cột đầu tiên là giá trị
                    df = df.rename(columns={df.columns[0]: 'value'})
            else:
                raise ValueError("benchmark_data phải là DataFrame, Dict hoặc List")
            
            # Lấy dữ liệu vốn
            capital_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"]
                }
                for entry in self.metrics_history["capital_history"]
            ])
            
            capital_data["timestamp"] = pd.to_datetime(capital_data["timestamp"])
            capital_data.set_index("timestamp", inplace=True)
            capital_data.sort_index(inplace=True)
            
            # Loại bỏ các dòng trùng lặp (giữ cuối cùng)
            capital_data = capital_data[~capital_data.index.duplicated(keep='last')]
            
            # Lọc theo khoảng thời gian nếu cần
            if start_date is not None:
                capital_data = capital_data[capital_data.index >= pd.to_datetime(start_date)]
                df = df[df.index >= pd.to_datetime(start_date)]
            
            if end_date is not None:
                capital_data = capital_data[capital_data.index <= pd.to_datetime(end_date)]
                df = df[df.index <= pd.to_datetime(end_date)]
            
            # Chuẩn hóa để so sánh (giá trị ban đầu = 100)
            initial_capital = capital_data["capital"].iloc[0]
            initial_benchmark = df["value"].iloc[0]
            
            capital_data["normalized"] = capital_data["capital"] / initial_capital * 100
            df["normalized"] = df["value"] / initial_benchmark * 100
            
            # Tính returns
            capital_returns = capital_data["capital"].pct_change().dropna()
            benchmark_returns = df["value"].pct_change().dropna()
            
            # Tính các chỉ số so sánh
            strategy_mean_return = capital_returns.mean()
            benchmark_mean_return = benchmark_returns.mean()
            
            strategy_std = capital_returns.std()
            benchmark_std = benchmark_returns.std()
            
            strategy_sharpe = strategy_mean_return / strategy_std * np.sqrt(252) if strategy_std > 0 else 0
            benchmark_sharpe = benchmark_mean_return / benchmark_std * np.sqrt(252) if benchmark_std > 0 else 0
            
            # Tính alpha, beta
            covariance = capital_returns.cov(benchmark_returns)
            variance = benchmark_returns.var()
            beta = covariance / variance if variance > 0 else 0
            
            risk_free_rate = 0.02 / 252  # Giả định 2% hàng năm
            alpha = strategy_mean_return - (risk_free_rate + beta * (benchmark_mean_return - risk_free_rate))
            alpha_annualized = alpha * 252
            
            # Tính correlation
            correlation = capital_returns.corr(benchmark_returns)
            
            # Tính outperformance
            total_strategy_return = (capital_data["capital"].iloc[-1] / capital_data["capital"].iloc[0]) - 1
            total_benchmark_return = (df["value"].iloc[-1] / df["value"].iloc[0]) - 1
            outperformance = total_strategy_return - total_benchmark_return
            
            # Tạo biểu đồ so sánh
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(12, 6))
                
                plt.plot(capital_data.index, capital_data["normalized"], 'b-', label=f'Chiến lược {self.strategy_name}')
                plt.plot(df.index, df["normalized"], 'r-', label=benchmark_name)
                
                plt.grid(True, alpha=0.3)
                plt.title(f'So sánh hiệu suất - {self.symbol} vs {benchmark_name}')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá trị (chuẩn hóa, ban đầu = 100)')
                plt.legend()
                
                # Lưu biểu đồ
                plt.savefig(save_path)
                plt.close()
            
            # Kết quả so sánh
            result = {
                "strategy_name": self.strategy_name,
                "benchmark_name": benchmark_name,
                "comparison_period": {
                    "start": str(capital_data.index[0]),
                    "end": str(capital_data.index[-1]),
                    "days": (capital_data.index[-1] - capital_data.index[0]).days
                },
                "returns": {
                    "strategy_total_return": total_strategy_return,
                    "benchmark_total_return": total_benchmark_return,
                    "outperformance": outperformance,
                    "strategy_mean_daily_return": strategy_mean_return,
                    "benchmark_mean_daily_return": benchmark_mean_return
                },
                "risk": {
                    "strategy_volatility": strategy_std * np.sqrt(252),
                    "benchmark_volatility": benchmark_std * np.sqrt(252),
                    "beta": beta,
                    "correlation": correlation
                },
                "ratios": {
                    "strategy_sharpe": strategy_sharpe,
                    "benchmark_sharpe": benchmark_sharpe,
                    "alpha": alpha_annualized,
                    "information_ratio": outperformance / (capital_returns - benchmark_returns).std() * np.sqrt(252) if (capital_returns - benchmark_returns).std() > 0 else 0
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi so sánh với benchmark: {str(e)}")
            return {
                "error": str(e),
                "strategy_name": self.strategy_name,
                "benchmark_name": benchmark_name
            }
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của tracker.
        """
        # Reset biến trạng thái
        self.start_time = time.time()
        self.trade_count = 0
        self.current_capital = self.initial_capital
        self.open_positions = {}
        
        # Reset lịch sử số liệu
        for key in self.metrics_history:
            if isinstance(self.metrics_history[key], list):
                self.metrics_history[key] = []
            elif isinstance(self.metrics_history[key], dict):
                self.metrics_history[key] = {}
        
        # Reset thống kê hiệu suất
        self.performance_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "total_profit_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "expectancy": 0.0,
            "average_trade_duration": 0.0,
            "total_fees": 0.0,
            "roi": 0.0
        }
        
        self.logger.info("Đã reset TradingMetricsTracker")
    
    def add_custom_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Thêm số liệu tùy chỉnh.
        
        Args:
            metric_name: Tên số liệu
            value: Giá trị số liệu
            timestamp: Thời gian ghi nhận
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Thêm vào lịch sử nếu chưa có
        if f"custom_{metric_name}" not in self.metrics_history:
            self.metrics_history[f"custom_{metric_name}"] = []
        
        # Thêm giá trị mới
        self.metrics_history[f"custom_{metric_name}"].append({
            "timestamp": timestamp.isoformat(),
            "value": value
        })
        
        self.logger.debug(f"Đã thêm số liệu tùy chỉnh: {metric_name}={value}")


class MultiSymbolTradingMetricsTracker:
    """
    Lớp theo dõi và quản lý các số liệu giao dịch cho nhiều cặp tiền.
    Tổng hợp dữ liệu từ nhiều TradingMetricsTracker riêng lẻ.
    """
    
    def __init__(
        self,
        strategy_name: str,
        initial_capital: float,
        output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo theo dõi số liệu giao dịch cho nhiều cặp tiền.
        
        Args:
            strategy_name: Tên chiến lược đang sử dụng
            initial_capital: Vốn ban đầu
            output_dir: Thư mục đầu ra cho số liệu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("multi_trading_metrics")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thông tin cơ bản
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            logs_dir = Path(self.system_config.get("log_dir", "./logs"))
            output_dir = logs_dir / "trading" / "multi_symbol"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu trữ các trackers cho từng cặp tiền
        self.symbol_trackers: Dict[str, TradingMetricsTracker] = {}
        
        # Khởi tạo biến trạng thái
        self.start_time = time.time()
        self.symbols_allocation: Dict[str, float] = {}  # Phân bổ vốn theo cặp tiền
        
        # Theo dõi số liệu tổng hợp
        self.portfolio_history = []
        self.allocation_history = []
        
        self.logger.info(f"Đã khởi tạo MultiSymbolTradingMetricsTracker cho chiến lược {strategy_name}")
    
    def add_symbol(
        self, 
        symbol: str, 
        allocation: Optional[float] = None
    ) -> TradingMetricsTracker:
        """
        Thêm cặp tiền mới để theo dõi.
        
        Args:
            symbol: Cặp tiền
            allocation: Phân bổ vốn (0.0 - 1.0, None để phân bổ đều)
            
        Returns:
            TradingMetricsTracker cho cặp tiền này
        """
        if symbol in self.symbol_trackers:
            self.logger.warning(f"Cặp tiền {symbol} đã tồn tại, trả về tracker hiện có")
            return self.symbol_trackers[symbol]
        
        # Tính phân bổ nếu không được cung cấp
        if allocation is None:
            # Phân bổ đều cho tất cả các cặp tiền
            num_symbols = len(self.symbol_trackers) + 1
            allocation = 1.0 / num_symbols
            
            # Điều chỉnh lại phân bổ cho các cặp tiền khác
            for sym in self.symbol_trackers:
                self.symbols_allocation[sym] = 1.0 / num_symbols
        
        # Lưu phân bổ
        self.symbols_allocation[symbol] = allocation
        
        # Tính vốn cho cặp tiền này
        symbol_capital = self.initial_capital * allocation
        
        # Tạo thư mục đầu ra cho cặp tiền
        symbol_output_dir = self.output_dir / symbol.replace("/", "_")
        symbol_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tracker cho cặp tiền
        tracker = TradingMetricsTracker(
            symbol=symbol,
            strategy_name=self.strategy_name,
            initial_capital=symbol_capital,
            output_dir=symbol_output_dir,
            logger=self.logger
        )
        
        # Lưu tracker
        self.symbol_trackers[symbol] = tracker
        
        # Ghi log phân bổ vốn
        self.allocation_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "allocation": allocation,
            "capital": symbol_capital,
            "action": "add_symbol"
        })
        
        self.logger.info(f"Đã thêm cặp tiền {symbol} với phân bổ {allocation:.2%}")
        return tracker
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Xóa cặp tiền khỏi danh sách theo dõi.
        
        Args:
            symbol: Cặp tiền cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        if symbol not in self.symbol_trackers:
            self.logger.warning(f"Cặp tiền {symbol} không tồn tại để xóa")
            return False
        
        # Lấy vốn hiện tại của cặp tiền
        tracker = self.symbol_trackers[symbol]
        symbol_capital = tracker.current_capital
        
        # Xóa khỏi danh sách trackers
        del self.symbol_trackers[symbol]
        
        # Xóa khỏi phân bổ
        del self.symbols_allocation[symbol]
        
        # Thêm vốn vào tổng vốn
        self.current_capital += symbol_capital
        
        # Điều chỉnh lại phân bổ cho các cặp tiền còn lại
        if self.symbol_trackers:
            # Phân bổ đều cho tất cả các cặp tiền còn lại
            num_symbols = len(self.symbol_trackers)
            for sym in self.symbol_trackers:
                self.symbols_allocation[sym] = 1.0 / num_symbols
        
        # Ghi log phân bổ vốn
        self.allocation_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "allocation": 0.0,
            "capital": symbol_capital,
            "action": "remove_symbol"
        })
        
        self.logger.info(f"Đã xóa cặp tiền {symbol}, vốn thu hồi: {symbol_capital:.2f}")
        return True
    
    def update_allocation(self, new_allocations: Dict[str, float]) -> bool:
        """
        Cập nhật phân bổ vốn cho các cặp tiền.
        
        Args:
            new_allocations: Dict với key là symbol và value là phân bổ mới
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra tổng phân bổ
        total_allocation = sum(new_allocations.values())
        if not np.isclose(total_allocation, 1.0, atol=0.01):
            self.logger.warning(f"Tổng phân bổ ({total_allocation:.2f}) phải bằng 1.0")
            return False
        
        # Kiểm tra các cặp tiền
        for symbol in new_allocations:
            if symbol not in self.symbol_trackers:
                self.logger.warning(f"Cặp tiền {symbol} không tồn tại trong danh sách theo dõi")
                return False
        
        # Tính tổng vốn hiện tại
        total_capital = self.get_total_capital()
        
        # Cập nhật phân bổ và vốn cho từng cặp tiền
        for symbol, allocation in new_allocations.items():
            # Cập nhật phân bổ
            old_allocation = self.symbols_allocation[symbol]
            self.symbols_allocation[symbol] = allocation
            
            # Tính vốn mới cho cặp tiền
            new_capital = total_capital * allocation
            old_capital = self.symbol_trackers[symbol].current_capital
            
            # Cập nhật vốn cho tracker
            self.symbol_trackers[symbol].current_capital = new_capital
            
            # Ghi log phân bổ vốn
            self.allocation_history.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "old_allocation": old_allocation,
                "new_allocation": allocation,
                "old_capital": old_capital,
                "new_capital": new_capital,
                "action": "update_allocation"
            })
            
            self.logger.info(f"Đã cập nhật phân bổ cho {symbol}: {old_allocation:.2%} -> {allocation:.2%}")
        
        return True
    
    def get_total_capital(self) -> float:
        """
        Lấy tổng vốn hiện tại của tất cả các cặp tiền.
        
        Returns:
            Tổng vốn hiện tại
        """
        return sum(tracker.current_capital for tracker in self.symbol_trackers.values())
    
    def update_portfolio(self, timestamp: Optional[datetime] = None) -> None:
        """
        Cập nhật thông tin danh mục đầu tư.
        
        Args:
            timestamp: Thời gian cập nhật
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Tính tổng vốn
        total_capital = self.get_total_capital()
        self.current_capital = total_capital
        
        # Tạo thông tin danh mục
        portfolio_info = {
            "timestamp": timestamp.isoformat(),
            "total_capital": total_capital,
            "symbols": {}
        }
        
        # Thêm thông tin từng cặp tiền
        for symbol, tracker in self.symbol_trackers.items():
            portfolio_info["symbols"][symbol] = {
                "capital": tracker.current_capital,
                "allocation": self.symbols_allocation[symbol],
                "actual_allocation": tracker.current_capital / total_capital if total_capital > 0 else 0,
                "profit_loss": tracker.current_capital - tracker.initial_capital,
                "roi": (tracker.current_capital / tracker.initial_capital) - 1 if tracker.initial_capital > 0 else 0,
                "trade_count": tracker.trade_count,
                "open_positions": len(tracker.open_positions)
            }
        
        # Lưu thông tin danh mục
        self.portfolio_history.append(portfolio_info)
        
        # Log với tần suất thấp hơn
        hour_minute = timestamp.strftime("%H%M")
        if hour_minute.endswith("00"):  # Mỗi giờ
            self.logger.info(
                f"Cập nhật danh mục: Tổng vốn = {total_capital:.2f}, "
                f"ROI = {(total_capital / self.initial_capital - 1) * 100:.2f}%, "
                f"Số cặp tiền = {len(self.symbol_trackers)}"
            )
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Tính toán tất cả các chỉ số hiệu suất cho toàn bộ danh mục.
        
        Returns:
            Dict chứa tất cả các chỉ số hiệu suất
        """
        # Cập nhật danh mục
        self.update_portfolio()
        
        # Tính toán chỉ số cho từng cặp tiền
        symbols_metrics = {}
        for symbol, tracker in self.symbol_trackers.items():
            symbols_metrics[symbol] = tracker.calculate_all_metrics()
        
        # Tính tổng số giao dịch
        total_trades = sum(metrics["total_trades"] for metrics in symbols_metrics.values())
        winning_trades = sum(metrics["winning_trades"] for metrics in symbols_metrics.values())
        losing_trades = sum(metrics["losing_trades"] for metrics in symbols_metrics.values())
        
        # Tính win rate tổng thể
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Tính profit factor tổng thể
        total_profit = sum(metrics["average_profit"] * metrics["winning_trades"] for metrics in symbols_metrics.values())
        total_loss = sum(metrics["average_loss"] * metrics["losing_trades"] for metrics in symbols_metrics.values())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Tính ROI
        roi = (self.current_capital / self.initial_capital) - 1
        
        # Tạo dict kết quả
        result = {
            "portfolio": {
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "roi": roi,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "symbols_count": len(self.symbol_trackers),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration_days": (datetime.now() - datetime.fromtimestamp(self.start_time)).days
            },
            "symbols": symbols_metrics
        }
        
        # Nếu có dữ liệu đủ để tính drawdown
        if len(self.portfolio_history) > 1:
            # Tạo chuỗi vốn
            capital_values = [entry["total_capital"] for entry in self.portfolio_history]
            
            # Tính drawdown
            max_value = capital_values[0]
            max_drawdown = 0
            max_drawdown_percent = 0
            
            for value in capital_values:
                max_value = max(max_value, value)
                drawdown = max_value - value
                drawdown_percent = drawdown / max_value if max_value > 0 else 0
                
                if drawdown_percent > max_drawdown_percent:
                    max_drawdown = drawdown
                    max_drawdown_percent = drawdown_percent
            
            result["portfolio"]["max_drawdown"] = max_drawdown
            result["portfolio"]["max_drawdown_percent"] = max_drawdown_percent
        
        return result
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê tóm tắt về quá trình giao dịch toàn bộ danh mục.
        
        Returns:
            Dict chứa các thống kê
        """
        # Cập nhật danh mục
        self.update_portfolio()
        
        # Lấy thống kê cho từng cặp tiền
        symbols_stats = {}
        for symbol, tracker in self.symbol_trackers.items():
            symbols_stats[symbol] = tracker.get_summary_stats()
        
        # Tính tổng số giao dịch
        total_trades = sum(stats["total_trades"] for stats in symbols_stats.values())
        winning_trades = sum(stats["winning_trades"] for stats in symbols_stats.values())
        losing_trades = sum(stats["losing_trades"] for stats in symbols_stats.values())
        
        # Tính win rate tổng thể
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Tính profit factor tổng thể
        total_profit = sum(stats["average_profit"] * stats["winning_trades"] for stats in symbols_stats.values() if stats["average_profit"] is not None)
        total_loss = sum(stats["average_loss"] * stats["losing_trades"] for stats in symbols_stats.values() if stats["average_loss"] is not None)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Tính ROI
        roi = (self.current_capital / self.initial_capital) - 1
        
        # Tính thời gian
        start_time = min([datetime.fromtimestamp(self.start_time)] + 
                        [datetime.fromisoformat(stats["trading_period"]["start_time"]) 
                         for stats in symbols_stats.values() 
                         if stats["trading_period"].get("start_time")])
        
        end_time = datetime.now()
        days = (end_time - start_time).days
        
        # Tạo dict kết quả
        result = {
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "roi": roi,
            "roi_percent": roi * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "win_rate_percent": win_rate * 100,
            "profit_factor": profit_factor,
            "symbols_count": len(self.symbol_trackers),
            "symbols": list(self.symbol_trackers.keys()),
            "trading_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "days": days
            },
            "symbols_stats": symbols_stats
        }
        
        # Nếu có dữ liệu đủ để tính drawdown
        if len(self.portfolio_history) > 1:
            # Tạo chuỗi vốn
            capital_values = [entry["total_capital"] for entry in self.portfolio_history]
            
            # Tính drawdown
            max_value = capital_values[0]
            max_drawdown = 0
            max_drawdown_percent = 0
            
            for value in capital_values:
                max_value = max(max_value, value)
                drawdown = max_value - value
                drawdown_percent = drawdown / max_value if max_value > 0 else 0
                
                if drawdown_percent > max_drawdown_percent:
                    max_drawdown = drawdown
                    max_drawdown_percent = drawdown_percent
            
            result["max_drawdown"] = max_drawdown
            result["max_drawdown_percent"] = max_drawdown_percent * 100
        
        return result
    
    def save_metrics(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Lưu lịch sử số liệu vào file JSON.
        
        Args:
            path: Đường dẫn file (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo đường dẫn tự động nếu không được cung cấp
        if path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            path = self.output_dir / f"multi_symbol_{current_date}_metrics.json"
        else:
            path = Path(path)
        
        # Cập nhật thông tin danh mục
        self.update_portfolio()
        
        # Chuẩn bị dữ liệu để lưu
        save_data = {
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "symbols": list(self.symbol_trackers.keys()),
            "symbols_allocation": self.symbols_allocation,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "trading_duration": time.time() - self.start_time,
            "portfolio_history": self.portfolio_history,
            "allocation_history": self.allocation_history,
            "summary_stats": self.get_summary_stats()
        }
        
        # Lưu vào file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu số liệu giao dịch vào {path}")
        
        # Lưu số liệu cho từng cặp tiền
        for symbol, tracker in self.symbol_trackers.items():
            symbol_path = path.parent / f"{symbol.replace('/', '_')}_{current_date}_metrics.json"
            tracker.save_metrics(symbol_path)
        
        return str(path)
    
    def plot_portfolio_metrics(self, save_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Tạo và lưu biểu đồ số liệu danh mục đầu tư.
        
        Args:
            save_path: Thư mục lưu biểu đồ (None để sử dụng output_dir)
            
        Returns:
            Danh sách đường dẫn các file biểu đồ đã lưu
        """
        # Cập nhật thông tin danh mục
        self.update_portfolio()
        
        # Kiểm tra nếu không có dữ liệu
        if len(self.portfolio_history) == 0:
            self.logger.warning("Không có dữ liệu để tạo biểu đồ")
            return []
        
        # Thiết lập thư mục lưu
        if save_path is None:
            save_path = self.output_dir / "plots"
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Danh sách để lưu đường dẫn các file
        saved_paths = []
        
        # ---------- 1. Biểu đồ vốn danh mục ----------
        try:
            plt.figure(figsize=(12, 6))
            
            # Tạo DataFrame từ lịch sử danh mục
            portfolio_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "total_capital": entry["total_capital"]
                }
                for entry in self.portfolio_history
            ])
            
            portfolio_data["timestamp"] = pd.to_datetime(portfolio_data["timestamp"])
            portfolio_data.set_index("timestamp", inplace=True)
            portfolio_data.sort_index(inplace=True)
            
            # Loại bỏ các dòng trùng lặp (giữ cuối cùng)
            portfolio_data = portfolio_data[~portfolio_data.index.duplicated(keep='last')]
            
            # Tạo biểu đồ
            plt.plot(portfolio_data.index, portfolio_data["total_capital"], 'b-', label='Tổng vốn')
            
            # Thêm đường tham chiếu vốn ban đầu
            plt.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Vốn ban đầu')
            
            plt.grid(True, alpha=0.3)
            plt.title(f'Biến động vốn danh mục - {self.strategy_name}')
            plt.xlabel('Thời gian')
            plt.ylabel('Vốn')
            plt.legend()
            
            # Lưu biểu đồ
            capital_path = save_path / f"portfolio_capital.png"
            plt.savefig(capital_path)
            plt.close()
            saved_paths.append(capital_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ vốn danh mục: {str(e)}")
        
        # ---------- 2. Biểu đồ phân bổ hiện tại ----------
        try:
            if len(self.symbol_trackers) > 0:
                plt.figure(figsize=(10, 10))
                
                # Lấy thông tin phân bổ hiện tại
                labels = []
                sizes = []
                for symbol, tracker in self.symbol_trackers.items():
                    labels.append(symbol)
                    sizes.append(tracker.current_capital)
                
                # Tạo biểu đồ bánh
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')  # Đảm bảo biểu đồ hình tròn
                
                plt.title(f'Phân bổ vốn danh mục - {self.strategy_name}')
                
                # Lưu biểu đồ
                allocation_path = save_path / f"portfolio_allocation.png"
                plt.savefig(allocation_path)
                plt.close()
                saved_paths.append(allocation_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ phân bổ hiện tại: {str(e)}")
        
        # ---------- 3. Biểu đồ so sánh hiệu suất các cặp tiền ----------
        try:
            if len(self.symbol_trackers) > 0:
                plt.figure(figsize=(12, 6))
                
                # Chuẩn bị dữ liệu
                data = []
                for symbol, tracker in self.symbol_trackers.items():
                    roi = (tracker.current_capital / tracker.initial_capital - 1) * 100
                    data.append({
                        "symbol": symbol,
                        "roi": roi
                    })
                
                # Sắp xếp theo ROI
                data.sort(key=lambda x: x["roi"], reverse=True)
                
                # Tạo biểu đồ cột
                symbols = [item["symbol"] for item in data]
                roi_values = [item["roi"] for item in data]
                
                # Tạo màu cho các cột
                colors = ['green' if roi >= 0 else 'red' for roi in roi_values]
                
                plt.bar(symbols, roi_values, color=colors)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                plt.grid(True, alpha=0.3, axis='y')
                plt.title(f'So sánh ROI các cặp tiền - {self.strategy_name}')
                plt.xlabel('Cặp tiền')
                plt.ylabel('ROI (%)')
                
                # Xoay nhãn trục x nếu có nhiều cặp tiền
                if len(symbols) > 5:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                
                # Lưu biểu đồ
                roi_comparison_path = save_path / f"portfolio_roi_comparison.png"
                plt.savefig(roi_comparison_path)
                plt.close()
                saved_paths.append(roi_comparison_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ so sánh hiệu suất: {str(e)}")
        
        # ---------- 4. Biểu đồ số lượng giao dịch theo cặp tiền ----------
        try:
            if len(self.symbol_trackers) > 0:
                plt.figure(figsize=(12, 6))
                
                # Chuẩn bị dữ liệu
                data = []
                for symbol, tracker in self.symbol_trackers.items():
                    data.append({
                        "symbol": symbol,
                        "trades": tracker.trade_count,
                        "wins": len([t for t in tracker.metrics_history["win_history"] if t["is_win"]]),
                        "losses": len([t for t in tracker.metrics_history["win_history"] if not t["is_win"]])
                    })
                
                # Sắp xếp theo số lượng giao dịch
                data.sort(key=lambda x: x["trades"], reverse=True)
                
                # Tạo biểu đồ cột chồng
                symbols = [item["symbol"] for item in data]
                wins = [item["wins"] for item in data]
                losses = [item["losses"] for item in data]
                
                width = 0.8
                
                plt.bar(symbols, wins, width, label='Thắng', color='green')
                plt.bar(symbols, losses, width, bottom=wins, label='Thua', color='red')
                
                plt.grid(True, alpha=0.3, axis='y')
                plt.title(f'Số lượng giao dịch theo cặp tiền - {self.strategy_name}')
                plt.xlabel('Cặp tiền')
                plt.ylabel('Số lượng giao dịch')
                plt.legend()
                
                # Xoay nhãn trục x nếu có nhiều cặp tiền
                if len(symbols) > 5:
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                
                # Lưu biểu đồ
                trades_count_path = save_path / f"portfolio_trades_count.png"
                plt.savefig(trades_count_path)
                plt.close()
                saved_paths.append(trades_count_path)
                
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ số lượng giao dịch: {str(e)}")
        
        # ---------- 5. Biểu đồ tổng hợp danh mục ----------
        try:
            # Tạo biểu đồ tổng hợp
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Biến động vốn
            portfolio_data = pd.DataFrame([
                {
                    "timestamp": entry["timestamp"],
                    "total_capital": entry["total_capital"]
                }
                for entry in self.portfolio_history
            ])
            
            portfolio_data["timestamp"] = pd.to_datetime(portfolio_data["timestamp"])
            portfolio_data.set_index("timestamp", inplace=True)
            portfolio_data.sort_index(inplace=True)
            portfolio_data = portfolio_data[~portfolio_data.index.duplicated(keep='last')]
            
            axs[0, 0].plot(portfolio_data.index, portfolio_data["total_capital"], 'b-')
            axs[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
            axs[0, 0].set_title('Biến động vốn danh mục')
            axs[0, 0].grid(True, alpha=0.3)
            
            # 2. Phân bổ hiện tại
            if len(self.symbol_trackers) > 0:
                labels = []
                sizes = []
                for symbol, tracker in self.symbol_trackers.items():
                    labels.append(symbol)
                    sizes.append(tracker.current_capital)
                
                axs[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axs[0, 1].set_title('Phân bổ vốn danh mục')
                axs[0, 1].axis('equal')
            
            # 3. So sánh ROI
            if len(self.symbol_trackers) > 0:
                data = []
                for symbol, tracker in self.symbol_trackers.items():
                    roi = (tracker.current_capital / tracker.initial_capital - 1) * 100
                    data.append({
                        "symbol": symbol,
                        "roi": roi
                    })
                
                data.sort(key=lambda x: x["roi"], reverse=True)
                symbols = [item["symbol"] for item in data]
                roi_values = [item["roi"] for item in data]
                colors = ['green' if roi >= 0 else 'red' for roi in roi_values]
                
                axs[1, 0].bar(symbols, roi_values, color=colors)
                axs[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axs[1, 0].set_title('So sánh ROI các cặp tiền')
                axs[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Xoay nhãn trục x nếu có nhiều cặp tiền
                if len(symbols) > 5:
                    axs[1, 0].set_xticklabels(symbols, rotation=45, ha='right')
            
            # 4. Phân phối giao dịch
            if len(self.symbol_trackers) > 0:
                total_win = 0
                total_loss = 0
                
                for tracker in self.symbol_trackers.values():
                    total_win += len([t for t in tracker.metrics_history["win_history"] if t["is_win"]])
                    total_loss += len([t for t in tracker.metrics_history["win_history"] if not t["is_win"]])
                
                labels = ['Thắng', 'Thua']
                sizes = [total_win, total_loss]
                colors = ['green', 'red']
                
                axs[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axs[1, 1].set_title('Phân phối giao dịch')
                axs[1, 1].axis('equal')
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            summary_path = save_path / f"portfolio_summary.png"
            plt.savefig(summary_path)
            plt.close()
            saved_paths.append(summary_path)
            
        except Exception as e:
            self.logger.warning(f"Không thể tạo biểu đồ tổng hợp danh mục: {str(e)}")
        
        return saved_paths
    
    def generate_portfolio_report(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """
        Tạo báo cáo Markdown về hiệu suất danh mục đầu tư.
        
        Args:
            save_path: Đường dẫn file báo cáo (None để tạo tự động)
            
        Returns:
            Đường dẫn file báo cáo
        """
        # Thiết lập đường dẫn lưu
        if save_path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            save_path = self.output_dir / f"portfolio_{current_date}_report.md"
        else:
            save_path = Path(save_path)
        
        # Tạo thư mục nếu chưa tồn tại
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cập nhật thông tin danh mục
        self.update_portfolio()
        
        # Tạo biểu đồ và lưu
        plot_paths = self.plot_portfolio_metrics()
        
        # Lấy thống kê
        stats = self.get_summary_stats()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo báo cáo
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Báo cáo danh mục đầu tư: {self.strategy_name}\n\n")
            
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian báo cáo:** {current_time}\n")
            f.write(f"- **Chiến lược:** {self.strategy_name}\n")
            f.write(f"- **Vốn ban đầu:** {self.initial_capital:.2f}\n")
            f.write(f"- **Vốn hiện tại:** {self.current_capital:.2f}\n")
            f.write(f"- **ROI:** {stats['roi_percent']:.2f}%\n")
            f.write(f"- **Tổng số giao dịch:** {stats['total_trades']}\n")
            f.write(f"- **Số cặp tiền:** {stats['symbols_count']}\n")
            
            # Thông tin thời gian
            if "trading_period" in stats:
                trading_period = stats["trading_period"]
                f.write(f"- **Thời gian giao dịch:** {trading_period.get('days', 0)} ngày\n")
                f.write(f"  - **Bắt đầu:** {datetime.fromisoformat(trading_period['start_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  - **Kết thúc:** {datetime.fromisoformat(trading_period['end_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            f.write("\n## Thống kê hiệu suất danh mục\n\n")
            f.write(f"- **Win rate:** {stats['win_rate_percent']:.2f}%\n")
            f.write(f"- **Profit factor:** {stats['profit_factor']:.2f}\n")
            
            if "max_drawdown_percent" in stats:
                f.write(f"- **Max drawdown:** {stats['max_drawdown_percent']:.2f}%\n")
            
            f.write(f"- **Số giao dịch thắng:** {stats['winning_trades']}\n")
            f.write(f"- **Số giao dịch thua:** {stats['losing_trades']}\n")
            
            # Hiển thị biểu đồ
            f.write("\n## Biểu đồ phân tích danh mục\n\n")
            
            for plot_path in plot_paths:
                rel_path = os.path.relpath(plot_path, save_path.parent)
                plot_name = os.path.splitext(os.path.basename(plot_path))[0].replace("portfolio_", "")
                plot_title = ' '.join(word.capitalize() for word in plot_name.split('_'))
                f.write(f"### {plot_title}\n\n")
                f.write(f"![{plot_title}]({rel_path})\n\n")
            
            # Thông tin chi tiết theo cặp tiền
            f.write("\n## Thống kê theo cặp tiền\n\n")
            
            f.write("| Cặp tiền | Vốn hiện tại | ROI | Giao dịch | Win rate | Profit factor |\n")
            f.write("|----------|--------------|-----|-----------|----------|---------------|\n")
            
            for symbol, symbol_stats in stats["symbols_stats"].items():
                current_capital = symbol_stats.get("current_capital", 0)
                roi = symbol_stats.get("roi_percent", 0)
                total_trades = symbol_stats.get("total_trades", 0)
                win_rate = symbol_stats.get("win_rate_percent", 0)
                profit_factor = symbol_stats.get("profit_factor", 0)
                
                f.write(f"| {symbol} | {current_capital:.2f} | {roi:.2f}% | {total_trades} | {win_rate:.2f}% | {profit_factor:.2f} |\n")
            
            # Chi tiết các cặp tiền có hiệu suất tốt nhất
            f.write("\n### Cặp tiền hiệu suất tốt nhất\n\n")
            
            # Sắp xếp theo ROI
            sorted_symbols = sorted(
                stats["symbols_stats"].items(), 
                key=lambda x: x[1].get("roi", -float('inf')), 
                reverse=True
            )
            
            if sorted_symbols:
                best_symbol, best_stats = sorted_symbols[0]
                
                f.write(f"**{best_symbol}** đang có hiệu suất tốt nhất với ROI {best_stats.get('roi_percent', 0):.2f}%\n\n")
                f.write(f"- **Win rate:** {best_stats.get('win_rate_percent', 0):.2f}%\n")
                f.write(f"- **Profit factor:** {best_stats.get('profit_factor', 0):.2f}\n")
                f.write(f"- **Số giao dịch:** {best_stats.get('total_trades', 0)}\n")
                
                if "max_drawdown_percent" in best_stats:
                    f.write(f"- **Max drawdown:** {best_stats.get('max_drawdown_percent', 0):.2f}%\n")
            
            # Chi tiết các cặp tiền có hiệu suất kém nhất
            f.write("\n### Cặp tiền hiệu suất kém nhất\n\n")
            
            if sorted_symbols and len(sorted_symbols) > 1:
                worst_symbol, worst_stats = sorted_symbols[-1]
                
                f.write(f"**{worst_symbol}** đang có hiệu suất kém nhất với ROI {worst_stats.get('roi_percent', 0):.2f}%\n\n")
                f.write(f"- **Win rate:** {worst_stats.get('win_rate_percent', 0):.2f}%\n")
                f.write(f"- **Profit factor:** {worst_stats.get('profit_factor', 0):.2f}\n")
                f.write(f"- **Số giao dịch:** {worst_stats.get('total_trades', 0)}\n")
                
                if "max_drawdown_percent" in worst_stats:
                    f.write(f"- **Max drawdown:** {worst_stats.get('max_drawdown_percent', 0):.2f}%\n")
            
            # Thêm phần kết luận
            f.write("\n## Kết luận và đề xuất\n\n")
            
            # Đánh giá hiệu suất
            roi = stats.get('roi', 0)
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            
            if roi > 0.1:  # ROI > 10%
                f.write("Danh mục đang có hiệu suất **tốt** với ROI dương đáng kể. ")
            elif roi > 0:
                f.write("Danh mục đang có hiệu suất **khá** với ROI dương nhỏ. ")
            else:
                f.write("Danh mục đang có hiệu suất **kém** với ROI âm. ")
            
            if win_rate > 0.6:
                f.write(f"Tỷ lệ thắng cao ({win_rate:.2%}) cho thấy chiến lược có độ tin cậy tốt. ")
            elif win_rate > 0.5:
                f.write(f"Tỷ lệ thắng chấp nhận được ({win_rate:.2%}). ")
            else:
                f.write(f"Tỷ lệ thắng thấp ({win_rate:.2%}) cần được cải thiện. ")
            
            if profit_factor > 2:
                f.write(f"Profit factor cao ({profit_factor:.2f}) cho thấy tỷ lệ lợi nhuận/rủi ro rất tốt.\n\n")
            elif profit_factor > 1.5:
                f.write(f"Profit factor khá ({profit_factor:.2f}) cho thấy quản lý rủi ro hiệu quả.\n\n")
            elif profit_factor > 1:
                f.write(f"Profit factor chấp nhận được ({profit_factor:.2f}).\n\n")
            else:
                f.write(f"Profit factor thấp ({profit_factor:.2f}) cảnh báo rủi ro cao hơn lợi nhuận kỳ vọng.\n\n")
            
            # Đánh giá các cặp tiền
            if sorted_symbols and len(sorted_symbols) > 1:
                strong_symbols = [symb for symb, stat in sorted_symbols if stat.get("roi", 0) > 0]
                weak_symbols = [symb for symb, stat in sorted_symbols if stat.get("roi", 0) <= 0]
                
                if strong_symbols:
                    f.write(f"Các cặp tiền có hiệu suất tốt: {', '.join(strong_symbols[:3])}\n\n")
                
                if weak_symbols:
                    f.write(f"Các cặp tiền có hiệu suất kém: {', '.join(weak_symbols[:3])}\n\n")
            
            # Đề xuất
            f.write("### Đề xuất cải thiện\n\n")
            
            # Điều chỉnh phân bổ
            f.write("- **Điều chỉnh phân bổ vốn:** ")
            if sorted_symbols and len(sorted_symbols) > 1:
                best_symbols = [symb for symb, stat in sorted_symbols[:2]]
                worst_symbols = [symb for symb, stat in sorted_symbols[-2:]]
                
                f.write(f"Cân nhắc tăng phân bổ cho {', '.join(best_symbols)} ")
                f.write(f"và giảm phân bổ cho {', '.join(worst_symbols)}.\n")
            else:
                f.write("Xem xét hiệu suất của từng cặp tiền để điều chỉnh phân bổ vốn phù hợp.\n")
            
            # Các đề xuất khác
            f.write("- **Đa dạng hóa:** Xem xét thêm các cặp tiền mới để đa dạng hóa danh mục và giảm rủi ro tương quan.\n")
            f.write("- **Tối ưu hóa chiến lược:** Điều chỉnh tham số của chiến lược dựa trên hiệu suất của từng cặp tiền.\n")
            f.write("- **Quản lý rủi ro:** Xem xét điều chỉnh stop loss và take profit để cải thiện tỷ lệ lợi nhuận/rủi ro.\n")
        
        self.logger.info(f"Đã tạo báo cáo danh mục đầu tư tại {save_path}")
        return str(save_path)
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của multi-tracker.
        """
        # Reset biến trạng thái
        self.start_time = time.time()
        self.current_capital = self.initial_capital
        
        # Reset lịch sử
        self.portfolio_history = []
        self.allocation_history = []
        
        # Reset các trackers
        for tracker in self.symbol_trackers.values():
            tracker.reset()
            tracker.current_capital = self.initial_capital * self.symbols_allocation.get(tracker.symbol, 0)
        
        self.logger.info("Đã reset MultiSymbolTradingMetricsTracker")


def create_trading_metrics_tracker(
    symbol: str,
    strategy_name: str,
    initial_capital: float,
    output_dir: Optional[Union[str, Path]] = None,
    use_csv: bool = True,
    log_frequency: int = 10,
    logger: Optional[logging.Logger] = None
) -> TradingMetricsTracker:
    """
    Hàm tiện ích để tạo TradingMetricsTracker.
    
    Args:
        symbol: Cặp giao dịch
        strategy_name: Tên chiến lược đang sử dụng
        initial_capital: Vốn ban đầu
        output_dir: Thư mục đầu ra cho số liệu
        use_csv: Lưu số liệu vào file CSV
        log_frequency: Tần suất ghi log (mỗi bao nhiêu giao dịch)
        logger: Logger tùy chỉnh
        
    Returns:
        TradingMetricsTracker đã được cấu hình
    """
    return TradingMetricsTracker(
        symbol=symbol,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        output_dir=output_dir,
        use_csv=use_csv,
        log_frequency=log_frequency,
        logger=logger
    )


def create_multi_symbol_tracker(
    strategy_name: str,
    initial_capital: float,
    symbols: Optional[List[str]] = None,
    allocations: Optional[Dict[str, float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> MultiSymbolTradingMetricsTracker:
    """
    Hàm tiện ích để tạo MultiSymbolTradingMetricsTracker.
    
    Args:
        strategy_name: Tên chiến lược đang sử dụng
        initial_capital: Vốn ban đầu
        symbols: Danh sách cặp tiền (None để không thêm cặp tiền nào)
        allocations: Dict với key là symbol và value là phân bổ (None để phân bổ đều)
        output_dir: Thư mục đầu ra cho số liệu
        logger: Logger tùy chỉnh
        
    Returns:
        MultiSymbolTradingMetricsTracker đã được cấu hình
    """
    tracker = MultiSymbolTradingMetricsTracker(
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        output_dir=output_dir,
        logger=logger
    )
    
    # Thêm các cặp tiền nếu được cung cấp
    if symbols:
        # Tính phân bổ nếu không được cung cấp
        if allocations is None:
            # Phân bổ đều
            allocation_value = 1.0 / len(symbols)
            allocations = {symbol: allocation_value for symbol in symbols}
        
        # Kiểm tra tổng phân bổ
        total_allocation = sum(allocations.values())
        if not np.isclose(total_allocation, 1.0, atol=0.01):
            logger.warning(f"Tổng phân bổ ({total_allocation:.2f}) khác 1.0, sẽ được chuẩn hóa")
            scale_factor = 1.0 / total_allocation
            allocations = {symbol: alloc * scale_factor for symbol, alloc in allocations.items()}
        
        # Thêm từng cặp tiền
        for symbol in symbols:
            allocation = allocations.get(symbol, 1.0 / len(symbols))
            tracker.add_symbol(symbol, allocation)
    
    return tracker