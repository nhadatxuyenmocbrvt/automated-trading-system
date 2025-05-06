"""
Thực thi giao dịch tự động.
File này cung cấp các chức năng để thực thi giao dịch dựa trên tín hiệu từ agent,
quản lý vị thế, dừng lỗ, chốt lời và các chiến lược giao dịch khác.
"""

import time
import logging
import threading
import queue
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import OrderType, TimeInForce, OrderStatus, PositionSide, ErrorCode
from config.utils.validators import is_valid_trading_pair, is_valid_order_type
from data_collectors.exchange_api.generic_connector import ExchangeConnector, APIError
from risk_management.position_sizer import PositionSizer
from risk_management.stop_loss import StopLoss
from risk_management.take_profit import TakeProfit
from risk_management.risk_calculator import RiskCalculator
from deployment.exchange_api.order_manager import OrderManager
from deployment.exchange_api.position_tracker import PositionTracker

class TradeExecutor:
    """
    Lớp thực thi giao dịch tự động.
    Cung cấp các phương thức để thực thi giao dịch dựa trên tín hiệu từ agent,
    quản lý vị thế, dừng lỗ, chốt lời và các chiến lược giao dịch khác.
    """
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        order_manager: Optional[OrderManager] = None,
        position_tracker: Optional[PositionTracker] = None,
        position_sizer: Optional[PositionSizer] = None,
        stop_loss_manager: Optional[StopLoss] = None,
        take_profit_manager: Optional[TakeProfit] = None,
        risk_calculator: Optional[RiskCalculator] = None,
        max_active_positions: int = 5,
        max_leverage: float = 5.0,
        max_position_size_percent: float = 0.1,  # 10% vốn tối đa cho mỗi vị thế
        enable_trailing_stop: bool = True,
        default_stop_loss_percent: float = 0.02,  # 2% dừng lỗ mặc định
        default_take_profit_percent: float = 0.04,  # 4% chốt lời mặc định
        default_trailing_stop_percent: float = 0.01,  # 1% trailing stop mặc định
        default_leverage: float = 1.0,
        trade_signals_queue_size: int = 100,
        auto_hedge: bool = False,
        dry_run: bool = False,
        save_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo thực thi giao dịch.
        
        Args:
            exchange_connector: Kết nối đến sàn giao dịch
            order_manager: Quản lý lệnh giao dịch
            position_tracker: Theo dõi vị thế giao dịch
            position_sizer: Quản lý kích thước vị thế
            stop_loss_manager: Quản lý dừng lỗ
            take_profit_manager: Quản lý chốt lời
            risk_calculator: Tính toán mức độ rủi ro
            max_active_positions: Số vị thế hoạt động tối đa
            max_leverage: Đòn bẩy tối đa 
            max_position_size_percent: Phần trăm vốn tối đa cho mỗi vị thế
            enable_trailing_stop: Bật/tắt trailing stop
            default_stop_loss_percent: Phần trăm dừng lỗ mặc định
            default_take_profit_percent: Phần trăm chốt lời mặc định
            default_trailing_stop_percent: Phần trăm trailing stop mặc định
            default_leverage: Đòn bẩy mặc định
            trade_signals_queue_size: Kích thước hàng đợi tín hiệu giao dịch
            auto_hedge: Tự động hedge các vị thế
            dry_run: Chỉ mô phỏng giao dịch, không thực thi thực tế
            save_dir: Thư mục lưu dữ liệu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("trade_executor")
        
        # Lưu trữ tham số cấu hình
        self.exchange = exchange_connector
        self.max_active_positions = max_active_positions
        self.max_leverage = max_leverage
        self.max_position_size_percent = max_position_size_percent
        self.enable_trailing_stop = enable_trailing_stop
        self.default_stop_loss_percent = default_stop_loss_percent
        self.default_take_profit_percent = default_take_profit_percent
        self.default_trailing_stop_percent = default_trailing_stop_percent
        self.default_leverage = default_leverage
        self.auto_hedge = auto_hedge
        self.dry_run = dry_run
        
        # Khởi tạo các đối tượng quản lý nếu chưa được cung cấp
        self.order_manager = order_manager or OrderManager(
            exchange_connector=exchange_connector
        )
        
        self.position_tracker = position_tracker or PositionTracker(
            exchange_connector=exchange_connector
        )
        
        self.position_sizer = position_sizer or PositionSizer()
        self.stop_loss_manager = stop_loss_manager or StopLoss()
        self.take_profit_manager = take_profit_manager or TakeProfit()
        self.risk_calculator = risk_calculator or RiskCalculator()
        
        # Thiết lập thư mục lưu dữ liệu
        if save_dir is None:
            system_config = get_system_config()
            save_dir = Path(system_config.get("deployment.data_dir", "./data/deployment"))
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Hàng đợi tín hiệu giao dịch
        self.trade_signals_queue = queue.Queue(maxsize=trade_signals_queue_size)
        
        # Lưu trữ tín hiệu đã nhận và giao dịch đã thực thi
        self.received_signals = []
        self.executed_trades = []
        
        # Biến kiểm soát cho thread xử lý tín hiệu
        self._stop_signal_processing = False
        self._signal_processing_thread = None
        
        # Lưu trữ chiến lược giao dịch
        self.trading_strategies = {}
        
        # Lịch sử hoạt động giao dịch
        self.trading_activity_log = []
        
        # Kết quả hiệu suất giao dịch
        self.trading_performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Thời gian bắt đầu hoạt động
        self.start_time = datetime.now()
        
        self.logger.info(f"Đã khởi tạo TradeExecutor cho sàn {self.exchange.exchange_id}")
    
    def start(self) -> bool:
        """
        Bắt đầu thực thi giao dịch.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        try:
            # Kiểm tra kết nối đến sàn giao dịch
            if not self._check_exchange_connection():
                self.logger.error("Không thể kết nối đến sàn giao dịch, không bắt đầu thực thi")
                return False
            
            # Bắt đầu theo dõi vị thế
            self.position_tracker.start_tracking()
            
            # Bắt đầu thread xử lý tín hiệu
            self._stop_signal_processing = False
            self._signal_processing_thread = threading.Thread(
                target=self._process_trade_signals_loop,
                daemon=True
            )
            self._signal_processing_thread.start()
            
            self.logger.info(f"Đã bắt đầu thực thi giao dịch trên sàn {self.exchange.exchange_id}")
            
            if self.dry_run:
                self.logger.warning("Đang chạy ở chế độ dry run, không thực thi giao dịch thực tế")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi bắt đầu thực thi giao dịch: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Dừng thực thi giao dịch.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        try:
            # Dừng thread xử lý tín hiệu
            self._stop_signal_processing = True
            
            if self._signal_processing_thread is not None and self._signal_processing_thread.is_alive():
                self._signal_processing_thread.join(timeout=10)
                if self._signal_processing_thread.is_alive():
                    self.logger.warning("Không thể dừng thread xử lý tín hiệu")
                else:
                    self._signal_processing_thread = None
            
            # Dừng theo dõi vị thế
            self.position_tracker.stop_tracking()
            
            # Lưu lịch sử giao dịch
            self._save_trading_history()
            
            self.logger.info(f"Đã dừng thực thi giao dịch trên sàn {self.exchange.exchange_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi dừng thực thi giao dịch: {str(e)}")
            return False
    
    def add_trade_signal(self, trade_signal: Dict[str, Any]) -> bool:
        """
        Thêm tín hiệu giao dịch vào hàng đợi.
        
        Args:
            trade_signal: Thông tin tín hiệu giao dịch
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        # Kiểm tra tín hiệu giao dịch hợp lệ
        if not self._validate_trade_signal(trade_signal):
            self.logger.warning(f"Tín hiệu giao dịch không hợp lệ: {json.dumps(trade_signal)}")
            return False
        
        try:
            # Thêm thời gian nhận vào tín hiệu
            trade_signal["receive_time"] = datetime.now().isoformat()
            
            # Thêm vào hàng đợi nếu chưa đầy
            if self.trade_signals_queue.full():
                self.logger.warning("Hàng đợi tín hiệu giao dịch đã đầy, không thể thêm tín hiệu mới")
                return False
            
            self.trade_signals_queue.put(trade_signal, block=False)
            
            # Lưu vào danh sách các tín hiệu đã nhận
            self.received_signals.append(trade_signal)
            
            # Ghi log
            self.logger.info(f"Đã thêm tín hiệu giao dịch: {trade_signal.get('symbol', 'Unknown')} - {trade_signal.get('action', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm tín hiệu giao dịch: {str(e)}")
            return False
    
    def set_trading_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]) -> bool:
        """
        Thiết lập chiến lược giao dịch.
        
        Args:
            strategy_name: Tên chiến lược
            strategy_config: Cấu hình chiến lược
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        try:
            # Kiểm tra chiến lược hợp lệ
            required_fields = ["take_profit_type", "stop_loss_type", "risk_reward_ratio"]
            for field in required_fields:
                if field not in strategy_config:
                    self.logger.warning(f"Thiếu trường {field} trong cấu hình chiến lược {strategy_name}")
                    return False
            
            # Lưu chiến lược
            self.trading_strategies[strategy_name] = {
                "name": strategy_name,
                "config": strategy_config,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.logger.info(f"Đã thiết lập chiến lược giao dịch: {strategy_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập chiến lược giao dịch: {str(e)}")
            return False
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách vị thế đang hoạt động.
        
        Returns:
            Danh sách vị thế đang hoạt động
        """
        try:
            # Lấy thông tin vị thế từ position_tracker
            positions = self.position_tracker.get_positions(force_update=True)
            
            # Chuyển đổi từ dict sang list
            positions_list = []
            for symbol, position in positions.items():
                positions_list.append(position)
            
            return positions_list
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách vị thế đang hoạt động: {str(e)}")
            return []
    
    def get_trading_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Lấy thông tin hiệu suất giao dịch.
        
        Args:
            days: Số ngày để tính hiệu suất
            
        Returns:
            Thông tin hiệu suất giao dịch
        """
        try:
            # Lấy thông tin hiệu suất từ position_tracker
            performance = self.position_tracker.analyze_position_history(days=days)
            
            # Bổ sung thông tin từ trading_performance
            performance.update(self.trading_performance)
            
            # Bổ sung các chỉ số rủi ro
            active_positions = self.get_active_positions()
            total_position_value = sum(pos.get("notional_value", 0) for pos in active_positions)
            total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in active_positions)
            
            # Tính toán các chỉ số rủi ro
            try:
                margin_info = self.position_tracker.check_margin_level()
                risk_info = {
                    "margin_level": margin_info.get("margin_level", float('inf')),
                    "warning_level": margin_info.get("warning_level", "safe"),
                    "liquidation_risk": margin_info.get("liquidation_risk", 0),
                    "positions_at_risk": margin_info.get("positions_at_risk", {}),
                    "total_position_value": total_position_value,
                    "total_unrealized_pnl": total_unrealized_pnl,
                    "active_positions_count": len(active_positions)
                }
                performance["risk_info"] = risk_info
            except Exception as e:
                self.logger.warning(f"Không thể tính toán chỉ số rủi ro: {str(e)}")
            
            # Thêm thời gian hoạt động
            uptime = datetime.now() - self.start_time
            performance["uptime_hours"] = uptime.total_seconds() / 3600
            performance["uptime_days"] = uptime.total_seconds() / (24 * 3600)
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy thông tin hiệu suất giao dịch: {str(e)}")
            return {"error": str(e)}
    
    def execute_trade(self, trade_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực thi giao dịch dựa trên tín hiệu.
        
        Args:
            trade_signal: Thông tin tín hiệu giao dịch
            
        Returns:
            Kết quả thực thi giao dịch
        """
        try:
            # Kiểm tra tín hiệu giao dịch hợp lệ
            if not self._validate_trade_signal(trade_signal):
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": "Tín hiệu giao dịch không hợp lệ"
                    }
                }
            
            # Lấy thông tin từ tín hiệu
            symbol = trade_signal.get("symbol")
            action = trade_signal.get("action")
            position_size = trade_signal.get("position_size")
            side = trade_signal.get("side", "long" if action == "buy" else "short")
            price = trade_signal.get("price")
            stop_loss = trade_signal.get("stop_loss")
            take_profit = trade_signal.get("take_profit")
            strategy_name = trade_signal.get("strategy", "default")
            leverage = trade_signal.get("leverage", self.default_leverage)
            
            # Kiểm tra giới hạn đòn bẩy
            leverage = min(leverage, self.max_leverage)
            
            # Xác định loại lệnh
            order_type = trade_signal.get("order_type", OrderType.MARKET.value)
            
            # Kiểm tra hạn mức vị thế dựa trên số lượng vị thế hiện tại
            active_positions = self.get_active_positions()
            if len(active_positions) >= self.max_active_positions and action in ["buy", "sell"]:
                self.logger.warning(f"Đã đạt giới hạn vị thế hoạt động tối đa ({self.max_active_positions}), không thể mở vị thế mới")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.LIMIT_EXCEEDED.value,
                        "message": f"Đã đạt giới hạn vị thế hoạt động tối đa ({self.max_active_positions})"
                    }
                }
            
            # Kiểm tra hạn mức kích thước vị thế dựa trên balance
            if position_size is None or position_size <= 0:
                # Tính kích thước vị thế tự động nếu không được cung cấp
                balance = self._get_account_balance()
                position_size = self.position_sizer.calculate_position_size(
                    balance=balance,
                    risk_percent=self.default_stop_loss_percent,
                    entry_price=price,
                    stop_loss_price=stop_loss
                )
            
            # Kiểm tra giới hạn kích thước vị thế tối đa
            max_position_size = self._get_account_balance() * self.max_position_size_percent
            if position_size > max_position_size:
                self.logger.warning(f"Kích thước vị thế ({position_size}) vượt quá giới hạn tối đa ({max_position_size}), đã điều chỉnh xuống")
                position_size = max_position_size
            
            # Tính stop loss và take profit nếu không được cung cấp
            if stop_loss is None and action in ["buy", "sell"]:
                if side == "long":
                    stop_loss = price * (1 - self.default_stop_loss_percent)
                else:  # short
                    stop_loss = price * (1 + self.default_stop_loss_percent)
            
            if take_profit is None and action in ["buy", "sell"]:
                if side == "long":
                    take_profit = price * (1 + self.default_take_profit_percent)
                else:  # short
                    take_profit = price * (1 - self.default_take_profit_percent)
            
            # Lấy chiến lược giao dịch
            strategy = self.trading_strategies.get(strategy_name)
            
            # Ghi log giao dịch sắp thực hiện
            self.logger.info(f"Chuẩn bị thực thi giao dịch: {action} {symbol} {position_size} @ {price} (SL: {stop_loss}, TP: {take_profit})")
            
            # Nếu là chế độ dry run, không thực thi giao dịch thực tế
            if self.dry_run:
                simulated_result = self._simulate_trade_execution(
                    action=action,
                    symbol=symbol,
                    side=side,
                    position_size=position_size,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage
                )
                simulated_result["dry_run"] = True
                return simulated_result
            
            # Thực hiện giao dịch dựa trên action
            if action == "buy" or action == "sell":
                # Mở vị thế mới
                result = self._open_position(
                    symbol=symbol,
                    side=side,
                    position_size=position_size,
                    price=price,
                    order_type=order_type,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                    strategy=strategy
                )
            elif action == "close":
                # Đóng vị thế
                result = self._close_position(
                    symbol=symbol,
                    position_size=position_size,
                    price=price
                )
            elif action == "update_sl":
                # Cập nhật stop loss
                result = self._update_stop_loss(
                    symbol=symbol,
                    new_stop_loss=stop_loss
                )
            elif action == "update_tp":
                # Cập nhật take profit
                result = self._update_take_profit(
                    symbol=symbol,
                    new_take_profit=take_profit
                )
            else:
                self.logger.warning(f"Hành động không được hỗ trợ: {action}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Hành động không được hỗ trợ: {action}"
                    }
                }
            
            # Thêm thông tin tín hiệu vào kết quả
            result["trade_signal"] = trade_signal
            
            # Lưu vào lịch sử giao dịch đã thực thi
            if result["status"] == "success":
                result["execution_time"] = datetime.now().isoformat()
                self.executed_trades.append(result)
                
                # Ghi log thực thi thành công
                self.logger.info(f"Đã thực thi giao dịch thành công: {action} {symbol}")
                
                # Cập nhật lịch sử hoạt động
                self.trading_activity_log.append({
                    "time": datetime.now().isoformat(),
                    "action": action,
                    "symbol": symbol,
                    "side": side,
                    "position_size": position_size,
                    "price": price,
                    "result": "success"
                })
                
                # Cập nhật thông tin hiệu suất giao dịch
                self._update_trading_performance(result)
            else:
                # Ghi log thực thi thất bại
                self.logger.warning(f"Thực thi giao dịch thất bại: {action} {symbol} - {result.get('error', {}).get('message', 'Unknown error')}")
                
                # Cập nhật lịch sử hoạt động
                self.trading_activity_log.append({
                    "time": datetime.now().isoformat(),
                    "action": action,
                    "symbol": symbol,
                    "side": side,
                    "position_size": position_size,
                    "price": price,
                    "result": "error",
                    "error": result.get("error", {}).get("message", "Unknown error")
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực thi giao dịch: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi thực thi giao dịch: {str(e)}"
                }
            }
    
    def close_all_positions(self, reason: str = "manual") -> Dict[str, Any]:
        """
        Đóng tất cả các vị thế hiện tại.
        
        Args:
            reason: Lý do đóng vị thế
            
        Returns:
            Kết quả đóng vị thế
        """
        try:
            # Lấy danh sách vị thế hiện tại
            positions = self.position_tracker.get_positions(force_update=True)
            
            # Kết quả
            results = {
                "status": "success",
                "closed_positions": [],
                "errors": []
            }
            
            # Đóng từng vị thế
            for symbol, position in positions.items():
                try:
                    result = self._close_position(
                        symbol=symbol,
                        position_size=None,  # Đóng toàn bộ
                        price=None,  # Sử dụng giá thị trường
                        reason=reason
                    )
                    
                    if result["status"] == "success":
                        results["closed_positions"].append({
                            "symbol": symbol,
                            "result": result
                        })
                    else:
                        results["errors"].append({
                            "symbol": symbol,
                            "error": result.get("error", {}).get("message", "Unknown error")
                        })
                    
                except Exception as e:
                    results["errors"].append({
                        "symbol": symbol,
                        "error": str(e)
                    })
            
            # Cập nhật trạng thái kết quả
            if len(results["errors"]) > 0:
                if len(results["closed_positions"]) == 0:
                    results["status"] = "error"
                else:
                    results["status"] = "partial"
            
            # Ghi log
            self.logger.info(f"Đã đóng {len(results['closed_positions'])}/{len(positions)} vị thế, {len(results['errors'])} lỗi")
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "close_all",
                "reason": reason,
                "closed_count": len(results["closed_positions"]),
                "error_count": len(results["errors"]),
                "result": results["status"]
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đóng tất cả vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi đóng tất cả vị thế: {str(e)}"
                }
            }
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Hủy tất cả các lệnh đang chờ.
        
        Returns:
            Kết quả hủy lệnh
        """
        try:
            # Lấy danh sách lệnh đang chờ
            open_orders_result = self.order_manager.fetch_open_orders()
            
            if open_orders_result["status"] != "success":
                self.logger.warning(f"Không thể lấy danh sách lệnh đang chờ: {open_orders_result.get('error', {}).get('message', 'Unknown error')}")
                return open_orders_result
            
            open_orders = open_orders_result["orders"]
            
            # Kết quả
            results = {
                "status": "success",
                "canceled_orders": [],
                "errors": []
            }
            
            # Hủy từng lệnh
            for order in open_orders:
                try:
                    order_id = order["order_id"]
                    symbol = order["symbol"]
                    
                    result = self.order_manager.cancel_order(order_id, symbol)
                    
                    if result["status"] == "success":
                        results["canceled_orders"].append({
                            "order_id": order_id,
                            "symbol": symbol,
                            "result": result
                        })
                    else:
                        results["errors"].append({
                            "order_id": order_id,
                            "symbol": symbol,
                            "error": result.get("error", {}).get("message", "Unknown error")
                        })
                    
                except Exception as e:
                    results["errors"].append({
                        "order_id": order.get("order_id", "Unknown"),
                        "symbol": order.get("symbol", "Unknown"),
                        "error": str(e)
                    })
            
            # Cập nhật trạng thái kết quả
            if len(results["errors"]) > 0:
                if len(results["canceled_orders"]) == 0:
                    results["status"] = "error"
                else:
                    results["status"] = "partial"
            
            # Ghi log
            self.logger.info(f"Đã hủy {len(results['canceled_orders'])}/{len(open_orders)} lệnh, {len(results['errors'])} lỗi")
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "cancel_all_orders",
                "canceled_count": len(results["canceled_orders"]),
                "error_count": len(results["errors"]),
                "result": results["status"]
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi hủy tất cả lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi hủy tất cả lệnh: {str(e)}"
                }
            }
    
    def update_risk_parameters(self, risk_params: Dict[str, Any]) -> bool:
        """
        Cập nhật các tham số quản lý rủi ro.
        
        Args:
            risk_params: Các tham số rủi ro mới
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        try:
            # Cập nhật các tham số rủi ro
            if "max_active_positions" in risk_params:
                self.max_active_positions = risk_params["max_active_positions"]
            
            if "max_leverage" in risk_params:
                self.max_leverage = risk_params["max_leverage"]
            
            if "max_position_size_percent" in risk_params:
                self.max_position_size_percent = risk_params["max_position_size_percent"]
            
            if "enable_trailing_stop" in risk_params:
                self.enable_trailing_stop = risk_params["enable_trailing_stop"]
            
            if "default_stop_loss_percent" in risk_params:
                self.default_stop_loss_percent = risk_params["default_stop_loss_percent"]
            
            if "default_take_profit_percent" in risk_params:
                self.default_take_profit_percent = risk_params["default_take_profit_percent"]
            
            if "default_trailing_stop_percent" in risk_params:
                self.default_trailing_stop_percent = risk_params["default_trailing_stop_percent"]
            
            if "default_leverage" in risk_params:
                self.default_leverage = risk_params["default_leverage"]
            
            # Ghi log
            self.logger.info(f"Đã cập nhật tham số rủi ro: {json.dumps(risk_params)}")
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "update_risk_parameters",
                "parameters": risk_params
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật tham số rủi ro: {str(e)}")
            return False
    
    def hedge_position(self, symbol: str, hedge_percent: float = 1.0) -> Dict[str, Any]:
        """
        Tạo vị thế đối ứng để hedge.
        
        Args:
            symbol: Symbol cần hedge
            hedge_percent: Phần trăm hedge (0.0-1.0)
            
        Returns:
            Kết quả hedge
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}, không thể hedge")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Lấy thông tin vị thế
            current_side = position["side"]
            opposite_side = "short" if current_side == "long" else "long"
            position_size = position["size"]
            hedge_size = position_size * hedge_percent
            
            # Lấy giá hiện tại
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            
            # Tạo tín hiệu giao dịch hedge
            hedge_signal = {
                "symbol": symbol,
                "action": "buy" if opposite_side == "long" else "sell",
                "side": opposite_side,
                "position_size": hedge_size,
                "price": current_price,
                "order_type": OrderType.MARKET.value,
                "strategy": "hedge"
            }
            
            # Thực thi lệnh hedge
            result = self.execute_trade(hedge_signal)
            
            # Ghi log
            if result["status"] == "success":
                self.logger.info(f"Đã hedge vị thế {symbol} thành công ({hedge_percent * 100}%)")
            else:
                self.logger.warning(f"Hedge vị thế {symbol} thất bại: {result.get('error', {}).get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi hedge vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi hedge vị thế: {str(e)}"
                }
            }
    
    def export_trading_data(self, file_path: Optional[str] = None) -> str:
        """
        Xuất dữ liệu giao dịch vào file JSON.
        
        Args:
            file_path: Đường dẫn file (None để tạo tên tự động)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        # Tạo tên file tự động nếu không được cung cấp
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = str(self.save_dir / f"trading_data_{timestamp}.json")
        
        try:
            # Lấy dữ liệu để xuất
            trading_data = {
                "timestamp": datetime.now().isoformat(),
                "exchange": self.exchange.exchange_id,
                "trading_performance": self.trading_performance,
                "active_positions": self.get_active_positions(),
                "executed_trades": self.executed_trades,
                "received_signals": self.received_signals,
                "trading_activity_log": self.trading_activity_log,
                "trading_strategies": self.trading_strategies,
                "risk_parameters": {
                    "max_active_positions": self.max_active_positions,
                    "max_leverage": self.max_leverage,
                    "max_position_size_percent": self.max_position_size_percent,
                    "enable_trailing_stop": self.enable_trailing_stop,
                    "default_stop_loss_percent": self.default_stop_loss_percent,
                    "default_take_profit_percent": self.default_take_profit_percent,
                    "default_trailing_stop_percent": self.default_trailing_stop_percent,
                    "default_leverage": self.default_leverage,
                    "auto_hedge": self.auto_hedge,
                    "dry_run": self.dry_run
                }
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(trading_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã xuất dữ liệu giao dịch vào {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất dữ liệu giao dịch: {str(e)}")
            return ""
    
    def import_trading_data(self, file_path: str) -> bool:
        """
        Nhập dữ liệu giao dịch từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            True nếu nhập thành công, False nếu không
        """
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                trading_data = json.load(f)
            
            # Kiểm tra tính hợp lệ của dữ liệu
            required_fields = ["trading_performance", "active_positions", "executed_trades"]
            for field in required_fields:
                if field not in trading_data:
                    self.logger.error(f"File {file_path} không chứa trường dữ liệu bắt buộc: {field}")
                    return False
            
            # Nhập dữ liệu
            self.trading_performance = trading_data["trading_performance"]
            self.executed_trades = trading_data["executed_trades"]
            self.received_signals = trading_data.get("received_signals", [])
            self.trading_activity_log = trading_data.get("trading_activity_log", [])
            self.trading_strategies = trading_data.get("trading_strategies", {})
            
            # Nhập các tham số rủi ro
            risk_params = trading_data.get("risk_parameters", {})
            if risk_params:
                self.update_risk_parameters(risk_params)
            
            # Ghi log
            self.logger.info(f"Đã nhập dữ liệu giao dịch từ {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập dữ liệu giao dịch: {str(e)}")
            return False
    
    def _process_trade_signals_loop(self) -> None:
        """
        Vòng lặp xử lý tín hiệu giao dịch.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý tín hiệu giao dịch")
        
        while not self._stop_signal_processing:
            try:
                # Lấy tín hiệu từ hàng đợi (không block để có thể kiểm tra _stop_signal_processing)
                try:
                    trade_signal = self.trade_signals_queue.get(block=False)
                except queue.Empty:
                    # Không có tín hiệu, chờ rồi tiếp tục vòng lặp
                    time.sleep(0.1)
                    continue
                
                # Xử lý tín hiệu
                self.logger.info(f"Xử lý tín hiệu giao dịch: {trade_signal.get('symbol', 'Unknown')} - {trade_signal.get('action', 'Unknown')}")
                
                # Thực thi giao dịch
                result = self.execute_trade(trade_signal)
                
                # Tự động hedge vị thế nếu cần
                if self.auto_hedge and result["status"] == "success" and trade_signal.get("action") in ["buy", "sell"]:
                    symbol = trade_signal.get("symbol")
                    self.hedge_position(symbol, 1.0)  # Hedge 100%
                
                # Đánh dấu tín hiệu đã xử lý
                self.trade_signals_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý tín hiệu: {str(e)}")
                # Chờ một chút để tránh loop quá nhanh khi có lỗi
                time.sleep(1)
        
        self.logger.info("Đã dừng vòng lặp xử lý tín hiệu giao dịch")
    
    def _validate_trade_signal(self, trade_signal: Dict[str, Any]) -> bool:
        """
        Kiểm tra tín hiệu giao dịch hợp lệ.
        
        Args:
            trade_signal: Thông tin tín hiệu giao dịch
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        # Kiểm tra các trường bắt buộc
        required_fields = ["symbol", "action"]
        for field in required_fields:
            if field not in trade_signal:
                self.logger.warning(f"Thiếu trường bắt buộc trong tín hiệu giao dịch: {field}")
                return False
        
        # Kiểm tra symbol hợp lệ
        symbol = trade_signal.get("symbol")
        if not is_valid_trading_pair(symbol):
            self.logger.warning(f"Symbol không hợp lệ: {symbol}")
            return False
        
        # Kiểm tra action hợp lệ
        action = trade_signal.get("action")
        valid_actions = ["buy", "sell", "close", "update_sl", "update_tp"]
        if action not in valid_actions:
            self.logger.warning(f"Hành động không hợp lệ: {action}")
            return False
        
        # Kiểm tra order_type hợp lệ nếu có
        order_type = trade_signal.get("order_type")
        if order_type is not None and not is_valid_order_type(order_type):
            self.logger.warning(f"Loại lệnh không hợp lệ: {order_type}")
            return False
        
        # Kiểm tra side hợp lệ nếu có
        side = trade_signal.get("side")
        if side is not None and side not in ["long", "short"]:
            self.logger.warning(f"Side không hợp lệ: {side}")
            return False
        
        # Kiểm tra position_size hợp lệ đối với action buy/sell
        if action in ["buy", "sell"]:
            position_size = trade_signal.get("position_size")
            if position_size is not None and position_size <= 0:
                self.logger.warning(f"Kích thước vị thế không hợp lệ: {position_size}")
                return False
        
        return True
    
    def _open_position(
        self,
        symbol: str,
        side: str,
        position_size: float,
        price: Optional[float] = None,
        order_type: str = OrderType.MARKET.value,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Mở vị thế mới.
        
        Args:
            symbol: Symbol giao dịch
            side: Phía vị thế ('long' hoặc 'short')
            position_size: Kích thước vị thế
            price: Giá vào lệnh (tùy chọn cho lệnh limit)
            order_type: Loại lệnh
            stop_loss: Giá dừng lỗ
            take_profit: Giá chốt lời
            leverage: Đòn bẩy
            strategy: Chiến lược giao dịch
            
        Returns:
            Kết quả mở vị thế
        """
        try:
            # Thiết lập leverage trước nếu hỗ trợ
            if hasattr(self.exchange, 'set_leverage'):
                try:
                    self.exchange.set_leverage(leverage, symbol)
                    self.logger.info(f"Đã thiết lập leverage={leverage} cho {symbol}")
                except Exception as e:
                    self.logger.warning(f"Không thể thiết lập leverage cho {symbol}: {str(e)}")
            
            # Chuyển đổi side sang định dạng lệnh
            order_side = "buy" if side == "long" else "sell"
            
            # Tạo lệnh chính kèm stop loss và take profit
            if order_type == OrderType.MARKET.value:
                result = self.order_manager.create_order_with_tp_sl(
                    symbol=symbol,
                    side=order_side,
                    order_type=order_type,
                    quantity=position_size,
                    price=None,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit
                )
            else:  # Limit orders
                if price is None:
                    self.logger.warning("Giá không được cung cấp cho lệnh limit")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá phải được cung cấp cho lệnh limit"
                        }
                    }
                    
                result = self.order_manager.create_order_with_tp_sl(
                    symbol=symbol,
                    side=order_side,
                    order_type=order_type,
                    quantity=position_size,
                    price=price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit
                )
            
            # Kiểm tra kết quả
            if result["status"] != "success":
                return result
            
            # Lấy thông tin lệnh
            main_order = result["main_order"]
            is_filled = result["is_filled"]
            
            # Nếu lệnh thị trường đã được thực hiện, cập nhật vị thế
            if is_filled or order_type == OrderType.MARKET.value:
                # Lấy giá thực hiện từ lệnh
                executed_price = main_order.get("price")
                if not executed_price and "exchange_order" in main_order:
                    executed_price = main_order["exchange_order"].get("price")
                
                # Sử dụng giá tham chiếu nếu không có giá thực hiện
                if not executed_price:
                    executed_price = price
                
                # Lấy stop loss và take profit đã tạo
                actual_stop_loss = stop_loss
                actual_take_profit = take_profit
                
                if "stop_loss_order" in result:
                    actual_stop_loss = result["stop_loss_order"].get("price")
                
                if "take_profit_order" in result:
                    actual_take_profit = result["take_profit_order"].get("price")
                
                # Cập nhật vị thế trong position_tracker
                position = self.position_tracker.open_position(
                    symbol=symbol,
                    side=side,
                    size=position_size,
                    entry_price=executed_price,
                    leverage=leverage,
                    stop_loss=actual_stop_loss,
                    take_profit=actual_take_profit,
                    trailing_stop=self.enable_trailing_stop,
                    trailing_stop_percent=self.default_trailing_stop_percent,
                    order_id=main_order.get("order_id")
                )
                
                # Thêm thông tin vị thế vào kết quả
                result["position"] = position
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi mở vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi mở vị thế: {str(e)}"
                }
            }
    
    def _close_position(
        self,
        symbol: str,
        position_size: Optional[float] = None,
        price: Optional[float] = None,
        reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Đóng vị thế hiện tại.
        
        Args:
            symbol: Symbol giao dịch
            position_size: Kích thước cần đóng (None để đóng toàn bộ)
            price: Giá thoát (None để sử dụng giá thị trường)
            reason: Lý do đóng vị thế
            
        Returns:
            Kết quả đóng vị thế
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Lấy phía vị thế
            side = position["side"]
            opposite_side = "sell" if side == "long" else "buy"
            
            # Xác định kích thước cần đóng
            if position_size is None or position_size >= position["size"]:
                # Đóng toàn bộ vị thế
                size_to_close = position["size"]
                is_partial = False
            else:
                # Đóng một phần vị thế
                size_to_close = position_size
                is_partial = True
            
            # Tạo lệnh đóng vị thế
            close_result = self.order_manager.create_market_order(
                symbol=symbol,
                side=opposite_side,
                quantity=size_to_close,
                price=price
            )
            
            # Kiểm tra kết quả
            if close_result["status"] != "success":
                return close_result
            
            # Lấy thông tin lệnh đóng
            close_order = close_result["order"]
            
            # Lấy giá thoát
            exit_price = close_order.get("price")
            if not exit_price and "exchange_order" in close_order:
                exit_price = close_order["exchange_order"].get("price")
            
            # Nếu không có giá thoát, lấy giá hiện tại
            if not exit_price:
                exit_price = price
            
            # Sử dụng giá thị trường hiện tại nếu không có giá thoát
            if not exit_price:
                ticker = self.exchange.fetch_ticker(symbol)
                exit_price = ticker["last"]
            
            # Cập nhật vị thế trong position_tracker
            position_result = self.position_tracker.close_position(
                symbol=symbol,
                exit_price=exit_price,
                partial_size=size_to_close if is_partial else None,
                reason=reason
            )
            
            # Thêm thông tin vị thế vào kết quả
            close_result["position_result"] = position_result
            close_result["is_partial"] = is_partial
            close_result["reason"] = reason
            
            return close_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đóng vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi đóng vị thế: {str(e)}"
                }
            }
    
    def _update_stop_loss(
        self,
        symbol: str,
        new_stop_loss: float
    ) -> Dict[str, Any]:
        """
        Cập nhật giá dừng lỗ cho vị thế.
        
        Args:
            symbol: Symbol giao dịch
            new_stop_loss: Giá dừng lỗ mới
            
        Returns:
            Kết quả cập nhật
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Kiểm tra stop loss mới có hợp lệ không
            current_price = position.get("current_price", 0)
            side = position.get("side", "long")
            
            if side == "long" and new_stop_loss >= current_price:
                self.logger.warning(f"Stop loss mới ({new_stop_loss}) cao hơn giá hiện tại ({current_price}) cho vị thế long")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Stop loss phải thấp hơn giá hiện tại cho vị thế long"
                    }
                }
            elif side == "short" and new_stop_loss <= current_price:
                self.logger.warning(f"Stop loss mới ({new_stop_loss}) thấp hơn giá hiện tại ({current_price}) cho vị thế short")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Stop loss phải cao hơn giá hiện tại cho vị thế short"
                    }
                }
            
            # Cập nhật stop loss trong position_tracker
            updated_position = self.position_tracker.update_position(
                symbol=symbol,
                stop_loss=new_stop_loss
            )
            
            # Cập nhật stop loss trên sàn nếu hỗ trợ
            exchange_result = None
            try:
                if "stop_loss_order_id" in position:
                    # Hủy lệnh stop loss cũ
                    sl_order_id = position["stop_loss_order_id"]
                    self.order_manager.cancel_order(sl_order_id, symbol)
                    
                    # Tạo lệnh stop loss mới
                    side = "sell" if position["side"] == "long" else "buy"
                    exchange_result = self.order_manager.create_stop_loss_order(
                        symbol=symbol,
                        side=side,
                        quantity=position["size"],
                        stop_price=new_stop_loss
                    )
            except Exception as e:
                self.logger.warning(f"Không thể cập nhật lệnh stop loss trên sàn: {str(e)}")
            
            result = {
                "status": "success",
                "symbol": symbol,
                "old_stop_loss": position.get("stop_loss", 0),
                "new_stop_loss": new_stop_loss,
                "position": updated_position
            }
            
            if exchange_result:
                result["exchange_result"] = exchange_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật stop loss: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi cập nhật stop loss: {str(e)}"
                }
            }
    
    def _update_take_profit(
        self,
        symbol: str,
        new_take_profit: float
    ) -> Dict[str, Any]:
        """
        Cập nhật giá chốt lời cho vị thế.
        
        Args:
            symbol: Symbol giao dịch
            new_take_profit: Giá chốt lời mới
            
        Returns:
            Kết quả cập nhật
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Kiểm tra take profit mới có hợp lệ không
            current_price = position.get("current_price", 0)
            side = position.get("side", "long")
            
            if side == "long" and new_take_profit <= current_price:
                self.logger.warning(f"Take profit mới ({new_take_profit}) thấp hơn giá hiện tại ({current_price}) cho vị thế long")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Take profit phải cao hơn giá hiện tại cho vị thế long"
                    }
                }
            elif side == "short" and new_take_profit >= current_price:
                self.logger.warning(f"Take profit mới ({new_take_profit}) cao hơn giá hiện tại ({current_price}) cho vị thế short")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Take profit phải thấp hơn giá hiện tại cho vị thế short"
                    }
                }
            
            # Cập nhật take profit trong position_tracker
            updated_position = self.position_tracker.update_position(
                symbol=symbol,
                take_profit=new_take_profit
            )
            
            # Cập nhật take profit trên sàn nếu hỗ trợ
            exchange_result = None
            try:
                if "take_profit_order_id" in position:
                    # Hủy lệnh take profit cũ
                    tp_order_id = position["take_profit_order_id"]
                    self.order_manager.cancel_order(tp_order_id, symbol)
                    
                    # Tạo lệnh take profit mới
                    side = "sell" if position["side"] == "long" else "buy"
                    exchange_result = self.order_manager.create_take_profit_order(
                        symbol=symbol,
                        side=side,
                        quantity=position["size"],
                        take_profit_price=new_take_profit
                    )
            except Exception as e:
                self.logger.warning(f"Không thể cập nhật lệnh take profit trên sàn: {str(e)}")
            
            result = {
                "status": "success",
                "symbol": symbol,
                "old_take_profit": position.get("take_profit", 0),
                "new_take_profit": new_take_profit,
                "position": updated_position
            }
            
            if exchange_result:
                result["exchange_result"] = exchange_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật take profit: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi cập nhật take profit: {str(e)}"
                }
            }
    
    def _check_exchange_connection(self) -> bool:
        """
        Kiểm tra kết nối đến sàn giao dịch.
        
        Returns:
            True nếu kết nối thành công, False nếu không
        """
        try:
            # Thử lấy thông tin cơ bản từ sàn
            markets = self.exchange.fetch_markets()
            if markets:
                self.logger.debug(f"Kết nối thành công đến sàn {self.exchange.exchange_id}")
                return True
            else:
                self.logger.warning(f"Không thể lấy thông tin thị trường từ sàn {self.exchange.exchange_id}")
                return False
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra kết nối đến sàn {self.exchange.exchange_id}: {str(e)}")
            return False
    
    def _get_account_balance(self) -> float:
        """
        Lấy số dư tài khoản có sẵn.
        
        Returns:
            Số dư tài khoản (USD hoặc USDT)
        """
        try:
            # Lấy thông tin số dư từ sàn
            balance_info = self.exchange.fetch_balance()
            
            # Ưu tiên các đồng stablecoin
            for currency in ["USDT", "USDC", "BUSD", "DAI", "USD"]:
                if currency in balance_info and "free" in balance_info[currency]:
                    return float(balance_info[currency]["free"])
            
            # Nếu không tìm thấy stablecoin, sử dụng tổng giá trị
            if "total" in balance_info:
                return float(balance_info["total"])
            
            # Nếu không tìm thấy, trả về 0
            self.logger.warning("Không thể xác định số dư tài khoản")
            return 0.0
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy số dư tài khoản: {str(e)}")
            return 0.0
    
    def _update_trading_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Cập nhật thông tin hiệu suất giao dịch.
        
        Args:
            trade_result: Kết quả giao dịch
        """
        try:
            # Kiểm tra kết quả giao dịch
            if trade_result["status"] != "success":
                return
            
            # Lấy thông tin vị thế
            if "position_result" in trade_result:
                # Đóng vị thế
                position_result = trade_result["position_result"]
                
                # Tính profit/loss
                if "realized_pnl" in position_result:
                    pnl = position_result["realized_pnl"]
                    
                    # Cập nhật số lượng giao dịch
                    self.trading_performance["total_trades"] += 1
                    
                    # Cập nhật thắng/thua
                    if pnl > 0:
                        self.trading_performance["winning_trades"] += 1
                        self.trading_performance["total_profit"] += pnl
                    else:
                        self.trading_performance["losing_trades"] += 1
                        self.trading_performance["total_loss"] += abs(pnl)
                    
                    # Cập nhật win rate
                    if self.trading_performance["total_trades"] > 0:
                        self.trading_performance["win_rate"] = (
                            self.trading_performance["winning_trades"] / 
                            self.trading_performance["total_trades"]
                        )
                    
                    # Cập nhật profit factor
                    if self.trading_performance["total_loss"] > 0:
                        self.trading_performance["profit_factor"] = (
                            self.trading_performance["total_profit"] / 
                            self.trading_performance["total_loss"]
                        )
                    
                    # Cập nhật giá trị trung bình
                    if self.trading_performance["winning_trades"] > 0:
                        self.trading_performance["average_win"] = (
                            self.trading_performance["total_profit"] / 
                            self.trading_performance["winning_trades"]
                        )
                    
                    if self.trading_performance["losing_trades"] > 0:
                        self.trading_performance["average_loss"] = (
                            self.trading_performance["total_loss"] / 
                            self.trading_performance["losing_trades"]
                        )
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật hiệu suất giao dịch: {str(e)}")
    
    def _save_trading_history(self) -> bool:
        """
        Lưu lịch sử giao dịch.
        
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Đường dẫn file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.save_dir / f"trading_history_{timestamp}.json"
            
            # Dữ liệu để lưu
            trading_data = {
                "timestamp": datetime.now().isoformat(),
                "exchange": self.exchange.exchange_id,
                "trading_performance": self.trading_performance,
                "executed_trades": self.executed_trades,
                "trading_activity_log": self.trading_activity_log
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(trading_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu lịch sử giao dịch vào {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu lịch sử giao dịch: {str(e)}")
            return False
    
    def _simulate_trade_execution(
        self,
        action: str,
        symbol: str,
        side: str,
        position_size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Mô phỏng thực thi giao dịch trong chế độ dry run.
        
        Args:
            action: Hành động giao dịch
            symbol: Symbol giao dịch
            side: Phía vị thế ('long' hoặc 'short')
            position_size: Kích thước vị thế
            price: Giá vào lệnh
            stop_loss: Giá dừng lỗ
            take_profit: Giá chốt lời
            leverage: Đòn bẩy
            
        Returns:
            Kết quả mô phỏng
        """
        # Lấy giá hiện tại nếu không được cung cấp
        if price is None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker["last"]
            except Exception as e:
                self.logger.warning(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
                price = 0.0
        
        # Tạo ID giả cho lệnh
        simulated_order_id = f"simulated_{int(time.time() * 1000)}_{symbol}"
        
        # Tính toán tỷ lệ risk-reward
        risk_reward_ratio = None
        if stop_loss is not None and take_profit is not None and price != 0:
            if side == "long":
                risk = abs(price - stop_loss) / price
                reward = abs(take_profit - price) / price
            else:  # short
                risk = abs(stop_loss - price) / price
                reward = abs(price - take_profit) / price
                
            if risk != 0:
                risk_reward_ratio = reward / risk
        
        # Tạo kết quả mô phỏng
        result = {
            "status": "success",
            "action": action,
            "symbol": symbol,
            "side": side,
            "position_size": position_size,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "risk_reward_ratio": risk_reward_ratio,
            "simulated": True,
            "time": datetime.now().isoformat(),
            "simulated_order_id": simulated_order_id
        }
        
        # Tính toán thông tin bổ sung dựa trên action
        if action in ["buy", "sell"]:
            # Tính toán notional value
            notional_value = position_size * price
            
            # Tính toán thông tin margin
            margin_required = notional_value / leverage
            
            # Thêm vào kết quả
            result.update({
                "notional_value": notional_value,
                "margin_required": margin_required
            })
            
            # Thêm vào lịch sử giao dịch mô phỏng
            self.executed_trades.append({
                "time": datetime.now().isoformat(),
                "action": action,
                "symbol": symbol,
                "side": side,
                "position_size": position_size,
                "price": price,
                "status": "success",
                "simulated": True,
                "order_id": simulated_order_id
            })
            
            # Ghi log
            self.logger.info(f"[MÔ PHỎNG] Đã thực thi lệnh {action} {symbol} {position_size} @ {price}")
        
        return result

    def activate_trailing_stop(self, symbol: str, percent: Optional[float] = None) -> Dict[str, Any]:
        """
        Kích hoạt trailing stop cho vị thế.
        
        Args:
            symbol: Symbol giao dịch
            percent: Phần trăm trailing stop (None để sử dụng mặc định)
            
        Returns:
            Kết quả kích hoạt
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Sử dụng phần trăm mặc định nếu không được cung cấp
            if percent is None:
                percent = self.default_trailing_stop_percent
            
            # Cập nhật trailing stop trong position_tracker
            updated_position = self.position_tracker.update_position(
                symbol=symbol,
                trailing_stop=True,
                trailing_stop_percent=percent
            )
            
            # Ghi log
            self.logger.info(f"Đã kích hoạt trailing stop {percent * 100}% cho vị thế {symbol}")
            
            # Kết quả
            result = {
                "status": "success",
                "symbol": symbol,
                "trailing_stop_percent": percent,
                "position": updated_position
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi kích hoạt trailing stop: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi kích hoạt trailing stop: {str(e)}"
                }
            }
    
    def modify_position_leverage(self, symbol: str, new_leverage: float) -> Dict[str, Any]:
        """
        Thay đổi đòn bẩy cho vị thế.
        
        Args:
            symbol: Symbol giao dịch
            new_leverage: Đòn bẩy mới
            
        Returns:
            Kết quả thay đổi
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Kiểm tra đòn bẩy mới có hợp lệ không
            if new_leverage <= 0 or new_leverage > self.max_leverage:
                self.logger.warning(f"Đòn bẩy không hợp lệ: {new_leverage} (giới hạn: {self.max_leverage})")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Đòn bẩy phải > 0 và <= {self.max_leverage}"
                    }
                }
            
            # Thử thay đổi đòn bẩy trên sàn nếu hỗ trợ
            exchange_result = None
            try:
                if hasattr(self.exchange, 'set_leverage'):
                    self.exchange.set_leverage(new_leverage, symbol)
                    self.logger.info(f"Đã thiết lập leverage={new_leverage} trên sàn cho {symbol}")
                    exchange_result = {"success": True}
                else:
                    self.logger.warning(f"Sàn {self.exchange.exchange_id} không hỗ trợ thay đổi leverage")
                    exchange_result = {
                        "success": False,
                        "message": f"Sàn {self.exchange.exchange_id} không hỗ trợ thay đổi leverage"
                    }
            except Exception as e:
                self.logger.warning(f"Không thể thay đổi leverage trên sàn: {str(e)}")
                exchange_result = {
                    "success": False,
                    "message": str(e)
                }
            
            # Cập nhật leverage trong position_tracker
            updated_position = self.position_tracker.update_position(
                symbol=symbol,
                leverage=new_leverage
            )
            
            # Ghi log
            self.logger.info(f"Đã thay đổi leverage từ {position.get('leverage', 1.0)} sang {new_leverage} cho vị thế {symbol}")
            
            # Kết quả
            result = {
                "status": "success",
                "symbol": symbol,
                "old_leverage": position.get("leverage", 1.0),
                "new_leverage": new_leverage,
                "position": updated_position
            }
            
            if exchange_result:
                result["exchange_result"] = exchange_result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thay đổi leverage: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi thay đổi leverage: {str(e)}"
                }
            }
    
    def add_strategy_to_position(self, symbol: str, strategy_name: str) -> Dict[str, Any]:
        """
        Thêm chiến lược cho vị thế hiện tại.
        
        Args:
            symbol: Symbol giao dịch
            strategy_name: Tên chiến lược
            
        Returns:
            Kết quả thêm chiến lược
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Kiểm tra chiến lược có tồn tại không
            if strategy_name not in self.trading_strategies:
                self.logger.warning(f"Không tìm thấy chiến lược {strategy_name}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy chiến lược {strategy_name}"
                    }
                }
            
            # Lấy thông tin chiến lược
            strategy = self.trading_strategies[strategy_name]
            
            # Cập nhật vị thế với chiến lược mới
            updated_position = self.position_tracker.update_position(
                symbol=symbol,
                strategy=strategy_name
            )
            
            # Cập nhật stop loss và take profit dựa trên chiến lược nếu cần
            strategy_config = strategy.get("config", {})
            
            # Risk-reward ratio từ chiến lược
            rr_ratio = strategy_config.get("risk_reward_ratio", 2.0)
            
            # Xác định giá hiện tại
            current_price = position.get("current_price", 0)
            if current_price == 0:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker["last"]
                except Exception as e:
                    self.logger.warning(f"Không thể lấy giá hiện tại cho {symbol}: {str(e)}")
            
            # Kiểm tra nếu cần cập nhật stop loss
            if strategy_config.get("apply_stop_loss", True):
                stop_loss_type = strategy_config.get("stop_loss_type", "percent")
                stop_loss_value = strategy_config.get("stop_loss_value", self.default_stop_loss_percent)
                
                # Tính toán stop loss dựa trên loại
                new_stop_loss = None
                if stop_loss_type == "percent":
                    if position["side"] == "long":
                        new_stop_loss = current_price * (1 - stop_loss_value)
                    else:  # short
                        new_stop_loss = current_price * (1 + stop_loss_value)
                elif stop_loss_type == "atr":
                    # Thực hiện tính toán ATR ở đây nếu cần
                    pass
                
                # Cập nhật stop loss nếu đã tính toán
                if new_stop_loss is not None:
                    self._update_stop_loss(symbol, new_stop_loss)
            
            # Kiểm tra nếu cần cập nhật take profit
            if strategy_config.get("apply_take_profit", True):
                take_profit_type = strategy_config.get("take_profit_type", "risk_reward")
                
                # Tính toán take profit dựa trên loại
                new_take_profit = None
                if take_profit_type == "risk_reward":
                    # Sử dụng risk-reward ratio
                    stop_loss = position.get("stop_loss", 0)
                    if stop_loss != 0 and current_price != 0:
                        risk = abs(current_price - stop_loss)
                        if position["side"] == "long":
                            new_take_profit = current_price + (risk * rr_ratio)
                        else:  # short
                            new_take_profit = current_price - (risk * rr_ratio)
                elif take_profit_type == "percent":
                    take_profit_value = strategy_config.get("take_profit_value", self.default_take_profit_percent)
                    if position["side"] == "long":
                        new_take_profit = current_price * (1 + take_profit_value)
                    else:  # short
                        new_take_profit = current_price * (1 - take_profit_value)
                
                # Cập nhật take profit nếu đã tính toán
                if new_take_profit is not None:
                    self._update_take_profit(symbol, new_take_profit)
            
            # Kiểm tra nếu cần kích hoạt trailing stop
            if strategy_config.get("apply_trailing_stop", self.enable_trailing_stop):
                trailing_stop_percent = strategy_config.get("trailing_stop_percent", self.default_trailing_stop_percent)
                self.activate_trailing_stop(symbol, trailing_stop_percent)
            
            # Ghi log
            self.logger.info(f"Đã thêm chiến lược {strategy_name} cho vị thế {symbol}")
            
            # Lấy vị thế đã cập nhật
            final_position = self.position_tracker.get_position(symbol)
            
            # Kết quả
            result = {
                "status": "success",
                "symbol": symbol,
                "strategy": strategy_name,
                "position": final_position
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm chiến lược cho vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi thêm chiến lược cho vị thế: {str(e)}"
                }
            }

    def check_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Kiểm tra điều kiện thị trường cho symbol.
        
        Args:
            symbol: Symbol cần kiểm tra
            
        Returns:
            Thông tin điều kiện thị trường
        """
        try:
            # Lấy thông tin ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Lấy thông tin orderbook
            orderbook = self.exchange.fetch_order_book(symbol)
            
            # Tính toán chỉ số thanh khoản
            bid_volume = sum([bid[1] for bid in orderbook["bids"][:5]])
            ask_volume = sum([ask[1] for ask in orderbook["asks"][:5]])
            
            total_volume = bid_volume + ask_volume
            
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            depth_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Lấy thông tin volatility
            volatility_24h = None
            if "high" in ticker and "low" in ticker and ticker["high"] > 0:
                volatility_24h = (ticker["high"] - ticker["low"]) / ticker["low"]
            
            # Kết quả
            result = {
                "status": "success",
                "symbol": symbol,
                "price": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "spread": (ticker["ask"] - ticker["bid"]) / ticker["bid"] if ticker["bid"] > 0 else 0,
                "volume_24h": ticker.get("volume", 0),
                "change_24h": ticker.get("percentage", 0),
                "volatility_24h": volatility_24h,
                "liquidity": {
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "bid_ask_ratio": bid_ask_ratio,
                    "depth_imbalance": depth_imbalance
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra điều kiện thị trường: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi kiểm tra điều kiện thị trường: {str(e)}"
                }
            }

    def calculate_trade_size(
        self, 
        symbol: str, 
        risk_percent: float, 
        entry_price: float, 
        stop_loss: float
    ) -> Dict[str, Any]:
        """
        Tính toán kích thước giao dịch dựa trên quản lý rủi ro.
        
        Args:
            symbol: Symbol giao dịch
            risk_percent: Phần trăm rủi ro (0.0-1.0)
            entry_price: Giá vào lệnh
            stop_loss: Giá dừng lỗ
            
        Returns:
            Thông tin kích thước giao dịch
        """
        try:
            # Lấy số dư tài khoản
            balance = self._get_account_balance()
            
            # Tính toán kích thước vị thế
            position_size = self.position_sizer.calculate_position_size(
                balance=balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss_price=stop_loss
            )
            
            # Tính toán thông tin rủi ro
            risk_amount = balance * risk_percent
            price_risk = abs(entry_price - stop_loss) / entry_price
            
            # Kiểm tra giới hạn kích thước vị thế tối đa
            max_position_size = balance * self.max_position_size_percent
            
            if position_size > max_position_size:
                position_size = max_position_size
            
            # Kết quả
            result = {
                "status": "success",
                "symbol": symbol,
                "balance": balance,
                "risk_percent": risk_percent,
                "risk_amount": risk_amount,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "price_risk": price_risk,
                "position_size": position_size,
                "max_position_size": max_position_size,
                "notional_value": position_size * entry_price
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán kích thước giao dịch: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính toán kích thước giao dịch: {str(e)}"
                }
            }

    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Lấy trạng thái của lệnh.
        
        Args:
            order_id: ID lệnh
            symbol: Symbol giao dịch
            
        Returns:
            Thông tin trạng thái lệnh
        """
        try:
            # Lấy thông tin lệnh từ order_manager
            order_result = self.order_manager.fetch_order(order_id, symbol)
            
            # Kiểm tra kết quả
            if order_result["status"] != "success":
                return order_result
            
            # Lấy thông tin lệnh
            order = order_result["order"]
            
            # Ghi log
            self.logger.debug(f"Trạng thái lệnh {order_id} cho {symbol}: {order.get('status', 'Unknown')}")
            
            return order_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy trạng thái lệnh: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi lấy trạng thái lệnh: {str(e)}"
                }
            }

    def get_trading_history(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Lấy lịch sử giao dịch.
        
        Args:
            symbol: Symbol giao dịch (None để lấy tất cả)
            days: Số ngày cần lấy
            
        Returns:
            Thông tin lịch sử giao dịch
        """
        try:
            # Lấy lịch sử giao dịch từ exchange_connector
            trades = self.exchange.fetch_my_trades(symbol, limit=100)
            
            # Lọc theo thời gian
            start_time = datetime.now() - timedelta(days=days)
            filtered_trades = []
            
            for trade in trades:
                # Lấy thời gian giao dịch
                trade_time = None
                if "timestamp" in trade:
                    trade_time = datetime.fromtimestamp(trade["timestamp"] / 1000)
                elif "datetime" in trade:
                    try:
                        trade_time = datetime.fromisoformat(trade["datetime"].replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                # Kiểm tra thời gian
                if trade_time and trade_time >= start_time:
                    filtered_trades.append(trade)
                
                # Kiểm tra symbol nếu được chỉ định
                if symbol and trade.get("symbol") != symbol:
                    continue
            
            # Tính toán thống kê
            total_profit = 0
            total_loss = 0
            win_count = 0
            loss_count = 0
            
            for trade in filtered_trades:
                if "realized_pnl" in trade:
                    pnl = trade["realized_pnl"]
                    if pnl > 0:
                        total_profit += pnl
                        win_count += 1
                    else:
                        total_loss += abs(pnl)
                        loss_count += 1
            
            # Tính toán win rate và profit factor
            total_trades = win_count + loss_count
            win_rate = win_count / total_trades if total_trades > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Kết quả
            result = {
                "status": "success",
                "trades": filtered_trades,
                "total_trades": total_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": total_profit - total_loss,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "days": days
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy lịch sử giao dịch: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi lấy lịch sử giao dịch: {str(e)}"
                }
            }
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Tạo báo cáo rủi ro cho các vị thế hiện tại.
        
        Returns:
            Báo cáo rủi ro
        """
        try:
            # Lấy số dư tài khoản
            balance = self._get_account_balance()
            
            # Lấy danh sách vị thế đang hoạt động
            positions = self.position_tracker.get_positions(force_update=True)
            
            # Tạo danh sách vị thế đang hoạt động
            active_positions = []
            for symbol, position in positions.items():
                active_positions.append(position)
            
            # Tính toán tổng giá trị vị thế
            total_position_value = sum(pos.get("notional_value", 0) for pos in active_positions)
            
            # Tính toán tổng rủi ro dừng lỗ
            total_risk = 0
            position_risks = []
            
            for position in active_positions:
                symbol = position.get("symbol", "Unknown")
                entry_price = position.get("entry_price", 0)
                stop_loss = position.get("stop_loss", 0)
                size = position.get("size", 0)
                
                if entry_price > 0 and stop_loss > 0:
                    if position.get("side", "long") == "long":
                        risk_percent = abs(entry_price - stop_loss) / entry_price
                    else:  # short
                        risk_percent = abs(stop_loss - entry_price) / entry_price
                    
                    risk_amount = size * entry_price * risk_percent
                    total_risk += risk_amount
                    
                    position_risks.append({
                        "symbol": symbol,
                        "risk_percent": risk_percent,
                        "risk_amount": risk_amount,
                        "entry_price": entry_price,
                        "current_price": position.get("current_price", 0),
                        "stop_loss": stop_loss,
                        "size": size,
                        "side": position.get("side", "long")
                    })
            
            # Tính toán tỉ lệ margin sử dụng
            margin_usage = total_position_value / balance if balance > 0 else float('inf')
            
            # Kiểm tra mức độ nguy hiểm
            risk_level = "safe"
            if margin_usage > 0.7:
                risk_level = "high"
            elif margin_usage > 0.5:
                risk_level = "medium"
            
            # Tính toán khả năng thanh khoản
            liquidity_risk = total_risk / balance if balance > 0 else float('inf')
            
            # Đánh giá mức độ rủi ro
            risk_assessment = "Thấp"
            if liquidity_risk > 0.2:
                risk_assessment = "Rất cao"
            elif liquidity_risk > 0.1:
                risk_assessment = "Cao"
            elif liquidity_risk > 0.05:
                risk_assessment = "Trung bình"
            
            # Tạo báo cáo rủi ro
            report = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "account_balance": balance,
                "total_position_value": total_position_value,
                "total_risk_amount": total_risk,
                "margin_usage": margin_usage,
                "margin_level": 1 / margin_usage if margin_usage > 0 else float('inf'),
                "liquidity_risk": liquidity_risk,
                "risk_level": risk_level,
                "risk_assessment": risk_assessment,
                "active_positions_count": len(active_positions),
                "position_risks": position_risks,
                "recommendations": []
            }
            
            # Thêm khuyến nghị nếu cần
            if risk_level == "high":
                report["recommendations"].append("Giảm kích thước vị thế để giảm sử dụng margin")
            
            if liquidity_risk > 0.1:
                report["recommendations"].append("Điều chỉnh stop loss để giảm rủi ro thanh khoản")
            
            if len(active_positions) >= self.max_active_positions * 0.8:
                report["recommendations"].append("Số lượng vị thế đang tiếp cận giới hạn, cân nhắc đóng các vị thế không hiệu quả")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo báo cáo rủi ro: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tạo báo cáo rủi ro: {str(e)}"
                }
            }
    
    def scale_in_position(
        self,
        symbol: str,
        additional_size: float,
        price: Optional[float] = None,
        order_type: str = OrderType.MARKET.value
    ) -> Dict[str, Any]:
        """
        Tăng kích thước vị thế (scale in).
        
        Args:
            symbol: Symbol giao dịch
            additional_size: Kích thước bổ sung
            price: Giá vào lệnh (chỉ cho lệnh limit)
            order_type: Loại lệnh
            
        Returns:
            Kết quả thực hiện
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Lấy side vị thế
            side = position["side"]
            order_side = "buy" if side == "long" else "sell"
            
            # Kiểm tra thêm giới hạn kích thước vị thế tối đa
            current_size = position.get("size", 0)
            new_total_size = current_size + additional_size
            
            max_position_size = self._get_account_balance() * self.max_position_size_percent
            if new_total_size * position.get("entry_price", 0) > max_position_size:
                self.logger.warning(f"Kích thước vị thế ({new_total_size}) vượt quá giới hạn tối đa, đã điều chỉnh xuống")
                additional_size = max(0, max_position_size / position.get("entry_price", 0) - current_size)
            
            # Tạo lệnh scale in
            if order_type == OrderType.MARKET.value:
                result = self.order_manager.create_market_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=additional_size
                )
            else:  # Limit orders
                if price is None:
                    self.logger.warning("Giá không được cung cấp cho lệnh limit")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá phải được cung cấp cho lệnh limit"
                        }
                    }
                
                result = self.order_manager.create_limit_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=additional_size,
                    price=price
                )
            
            # Kiểm tra kết quả
            if result["status"] != "success":
                return result
            
            # Lấy thông tin lệnh
            order = result["order"]
            
            # Nếu lệnh thị trường đã được thực hiện, cập nhật vị thế
            if order.get("status") == OrderStatus.FILLED.value or order_type == OrderType.MARKET.value:
                # Lấy giá thực hiện từ lệnh
                executed_price = order.get("price")
                if not executed_price and "exchange_order" in order:
                    executed_price = order["exchange_order"].get("price")
                
                # Sử dụng giá tham chiếu nếu không có giá thực hiện
                if not executed_price:
                    executed_price = price
                
                # Lấy giá hiện tại nếu không có giá thực hiện
                if not executed_price:
                    ticker = self.exchange.fetch_ticker(symbol)
                    executed_price = ticker["last"]
                
                # Cập nhật vị thế trong position_tracker
                updated_position = self.position_tracker.scale_position(
                    symbol=symbol,
                    additional_size=additional_size,
                    additional_price=executed_price
                )
                
                # Thêm thông tin vị thế vào kết quả
                result["position"] = updated_position
                result["old_size"] = current_size
                result["new_size"] = new_total_size
                result["average_price"] = updated_position.get("entry_price", 0)
            
            # Ghi log
            self.logger.info(f"Đã scale in vị thế {symbol}: +{additional_size} @ {price if price else 'market'}")
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "scale_in",
                "symbol": symbol,
                "side": side,
                "additional_size": additional_size,
                "price": price,
                "order_type": order_type,
                "result": "success"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi scale in vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi scale in vị thế: {str(e)}"
                }
            }
    
    def scale_out_position(
        self,
        symbol: str,
        reduce_size: float,
        price: Optional[float] = None,
        order_type: str = OrderType.MARKET.value
    ) -> Dict[str, Any]:
        """
        Giảm kích thước vị thế (scale out).
        
        Args:
            symbol: Symbol giao dịch
            reduce_size: Kích thước cần giảm
            price: Giá thoát (chỉ cho lệnh limit)
            order_type: Loại lệnh
            
        Returns:
            Kết quả thực hiện
        """
        try:
            # Lấy thông tin vị thế hiện tại
            position = self.position_tracker.get_position(symbol)
            
            if not position:
                self.logger.warning(f"Không tìm thấy vị thế cho {symbol}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.DATA_NOT_FOUND.value,
                        "message": f"Không tìm thấy vị thế cho {symbol}"
                    }
                }
            
            # Lấy thông tin vị thế
            current_size = position.get("size", 0)
            side = position.get("side", "long")
            opposite_side = "sell" if side == "long" else "buy"
            
            # Kiểm tra kích thước giảm có hợp lệ không
            if reduce_size <= 0:
                self.logger.warning(f"Kích thước cần giảm ({reduce_size}) phải lớn hơn 0")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": "Kích thước cần giảm phải lớn hơn 0"
                    }
                }
            
            # Kiểm tra kích thước giảm có lớn hơn kích thước hiện tại không
            if reduce_size >= current_size:
                # Đóng toàn bộ vị thế
                return self._close_position(
                    symbol=symbol,
                    price=price,
                    reason="scale_out_all"
                )
            
            # Tạo lệnh scale out
            if order_type == OrderType.MARKET.value:
                result = self.order_manager.create_market_order(
                    symbol=symbol,
                    side=opposite_side,
                    quantity=reduce_size
                )
            else:  # Limit orders
                if price is None:
                    self.logger.warning("Giá không được cung cấp cho lệnh limit")
                    return {
                        "status": "error",
                        "error": {
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Giá phải được cung cấp cho lệnh limit"
                        }
                    }
                
                result = self.order_manager.create_limit_order(
                    symbol=symbol,
                    side=opposite_side,
                    quantity=reduce_size,
                    price=price
                )
            
            # Kiểm tra kết quả
            if result["status"] != "success":
                return result
            
            # Lấy thông tin lệnh
            order = result["order"]
            
            # Nếu lệnh thị trường đã được thực hiện, cập nhật vị thế
            if order.get("status") == OrderStatus.FILLED.value or order_type == OrderType.MARKET.value:
                # Lấy giá thực hiện từ lệnh
                executed_price = order.get("price")
                if not executed_price and "exchange_order" in order:
                    executed_price = order["exchange_order"].get("price")
                
                # Sử dụng giá tham chiếu nếu không có giá thực hiện
                if not executed_price:
                    executed_price = price
                
                # Lấy giá hiện tại nếu không có giá thực hiện
                if not executed_price:
                    ticker = self.exchange.fetch_ticker(symbol)
                    executed_price = ticker["last"]
                
                # Cập nhật vị thế trong position_tracker
                updated_position = self.position_tracker.close_position(
                    symbol=symbol,
                    exit_price=executed_price,
                    partial_size=reduce_size,
                    reason="scale_out"
                )
                
                # Thêm thông tin vị thế vào kết quả
                result["position"] = updated_position
                result["old_size"] = current_size
                result["new_size"] = current_size - reduce_size
                result["realized_pnl"] = updated_position.get("realized_pnl", 0)
            
            # Ghi log
            self.logger.info(f"Đã scale out vị thế {symbol}: -{reduce_size} @ {price if price else 'market'}")
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "scale_out",
                "symbol": symbol,
                "side": side,
                "reduce_size": reduce_size,
                "price": price,
                "order_type": order_type,
                "result": "success"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi scale out vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi scale out vị thế: {str(e)}"
                }
            }
    
    def monitor_margin_level(self) -> Dict[str, Any]:
        """
        Giám sát mức margin và cảnh báo nếu cần.
        
        Returns:
            Thông tin mức margin
        """
        try:
            # Sử dụng position_tracker để kiểm tra mức margin
            margin_info = self.position_tracker.check_margin_level()
            
            # Xử lý theo mức độ cảnh báo
            warning_level = margin_info.get("warning_level", "safe")
            
            if warning_level == "critical":
                # Đóng các vị thế nguy hiểm
                self.logger.warning("Mức margin rất nguy hiểm, đóng các vị thế rủi ro nhất")
                
                positions_at_risk = margin_info.get("positions_at_risk", {})
                for symbol in positions_at_risk:
                    self._close_position(
                        symbol=symbol,
                        reason="margin_critical"
                    )
            
            elif warning_level == "warning":
                # Cảnh báo nhưng không hành động
                self.logger.warning("Mức margin đang ở ngưỡng nguy hiểm, hãy cân nhắc giảm đòn bẩy hoặc đóng một số vị thế")
                
                # Giảm đòn bẩy cho các vị thế rủi ro cao nếu có thể
                positions_at_risk = margin_info.get("positions_at_risk", {})
                for symbol in positions_at_risk:
                    position = self.position_tracker.get_position(symbol)
                    if position and position.get("leverage", 1.0) > 1.0:
                        new_leverage = max(1.0, position.get("leverage", 1.0) / 2)
                        self.modify_position_leverage(symbol, new_leverage)
            
            # Cập nhật lịch sử hoạt động
            self.trading_activity_log.append({
                "time": datetime.now().isoformat(),
                "action": "monitor_margin",
                "warning_level": warning_level,
                "margin_level": margin_info.get("margin_level", float('inf')),
                "liquidation_risk": margin_info.get("liquidation_risk", 0)
            })
            
            return margin_info
            
        except Exception as e:
            self.logger.error(f"Lỗi khi giám sát mức margin: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi giám sát mức margin: {str(e)}"
                }
            }