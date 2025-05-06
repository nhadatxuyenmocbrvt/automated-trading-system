"""
Quản lý danh mục đầu tư.
File này định nghĩa lớp PortfolioManager để quản lý toàn bộ danh mục đầu tư,
bao gồm phân bổ tài sản, theo dõi rủi ro, và tối ưu hóa danh mục.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderStatus, ErrorCode
from risk_management.position_sizer import PositionSizer
from risk_management.risk_calculator import RiskCalculator
from risk_management.stop_loss import StopLossManager
from risk_management.take_profit import TakeProfitManager
from risk_management.drawdown_manager import DrawdownManager

class PortfolioManager:
    """
    Lớp quản lý danh mục đầu tư.
    Cung cấp các phương thức để quản lý, phân bổ và tối ưu hóa danh mục đầu tư.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_profile: str = "moderate",
        risk_per_trade: float = 0.02,
        max_positions: int = 10,
        max_correlated_positions: int = 3,
        correlation_threshold: float = 0.7,
        max_drawdown_percent: float = 0.20,
        rebalance_frequency: str = "weekly",
        target_allocation: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý danh mục đầu tư.
        
        Args:
            initial_capital: Vốn ban đầu
            risk_profile: Hồ sơ rủi ro ('conservative', 'moderate', 'aggressive')
            risk_per_trade: Phần trăm rủi ro trên mỗi giao dịch (0.02 = 2%)
            max_positions: Số lượng vị thế tối đa
            max_correlated_positions: Số lượng vị thế có tương quan cao tối đa
            correlation_threshold: Ngưỡng tương quan để xác định vị thế có liên quan
            max_drawdown_percent: Phần trăm sụt giảm tối đa cho phép
            rebalance_frequency: Tần suất cân bằng lại danh mục ('daily', 'weekly', 'monthly')
            target_allocation: Phân bổ mục tiêu cho các tài sản
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("portfolio_manager")
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_profile = risk_profile
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_correlated_positions = max_correlated_positions
        self.correlation_threshold = correlation_threshold
        self.max_drawdown_percent = max_drawdown_percent
        self.rebalance_frequency = rebalance_frequency
        
        # Phân bổ mục tiêu mặc định nếu không được cung cấp
        self.target_allocation = target_allocation or self._get_default_allocation()
        
        # Khởi tạo các thành phần quản lý rủi ro
        self.position_sizer = PositionSizer(
            risk_per_trade=risk_per_trade,
            position_sizing_method="risk_percentage"
        )
        
        self.risk_calculator = RiskCalculator()
        
        self.stop_loss_manager = StopLossManager(
            max_risk_percent=risk_per_trade * 1.5,  # Stop loss rộng hơn risk per trade
            trailing_stop_enabled=True
        )
        
        self.take_profit_manager = TakeProfitManager(
            risk_reward_ratio=2.0  # Tỷ lệ lợi nhuận/rủi ro mục tiêu
        )
        
        self.drawdown_manager = DrawdownManager(
            max_drawdown_percent=max_drawdown_percent,
            emergency_action="reduce_position_size"
        )
        
        # Khởi tạo danh mục đầu tư
        self.portfolio = {}
        self.positions = {}
        self.position_history = []
        self.asset_stats = {}
        self.correlation_matrix = pd.DataFrame()
        self.last_rebalance_time = datetime.now()
        
        # Khởi tạo các thống kê
        self.stats = {
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "portfolio_value_history": [],
            "daily_returns": [],
            "risk_metrics": {}
        }
        
        self.logger.info(f"Đã khởi tạo PortfolioManager với {initial_capital} vốn ban đầu, risk profile: {risk_profile}")
    
    def reset(self) -> None:
        """
        Đặt lại trạng thái của danh mục đầu tư.
        """
        self.current_capital = self.initial_capital
        self.portfolio = {}
        self.positions = {}
        self.position_history = []
        self.stats = {
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "portfolio_value_history": [],
            "daily_returns": [],
            "risk_metrics": {}
        }
        self.last_rebalance_time = datetime.now()
        self.logger.info("Đã đặt lại PortfolioManager")
    
    def _get_default_allocation(self) -> Dict[str, float]:
        """
        Lấy phân bổ mặc định dựa trên hồ sơ rủi ro.
        
        Returns:
            Dict với key là loại tài sản và value là phần trăm phân bổ
        """
        if self.risk_profile == "conservative":
            return {
                "BTC": 0.30,
                "ETH": 0.20,
                "STABLES": 0.40,
                "OTHERS": 0.10
            }
        elif self.risk_profile == "moderate":
            return {
                "BTC": 0.40,
                "ETH": 0.25,
                "STABLES": 0.20,
                "OTHERS": 0.15
            }
        elif self.risk_profile == "aggressive":
            return {
                "BTC": 0.50,
                "ETH": 0.30,
                "STABLES": 0.0,
                "OTHERS": 0.20
            }
        else:
            return {
                "BTC": 0.40,
                "ETH": 0.30,
                "STABLES": 0.20,
                "OTHERS": 0.10
            }
    
    def add_asset(self, symbol: str, category: str, allocation: float = None) -> Dict[str, Any]:
        """
        Thêm tài sản vào danh mục đầu tư.
        
        Args:
            symbol: Mã tài sản
            category: Loại tài sản ('BTC', 'ETH', 'STABLES', 'OTHERS')
            allocation: Phân bổ mục tiêu (% của danh mục)
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem tài sản đã tồn tại chưa
        if symbol in self.portfolio:
            self.logger.warning(f"Tài sản {symbol} đã tồn tại trong danh mục")
            return {
                "status": "error",
                "message": f"Tài sản {symbol} đã tồn tại trong danh mục"
            }
        
        # Nếu không có phân bổ cụ thể, sử dụng phân bổ mặc định dựa trên category
        if allocation is None:
            if category in self.target_allocation:
                # Chia đều phân bổ cho các tài sản trong cùng category
                assets_in_category = sum(1 for asset_data in self.portfolio.values() if asset_data["category"] == category)
                if assets_in_category > 0:
                    allocation = self.target_allocation[category] / (assets_in_category + 1)
                else:
                    allocation = self.target_allocation[category]
            else:
                allocation = 0.05  # Mặc định 5%
        
        # Tạo thông tin tài sản
        asset_info = {
            "symbol": symbol,
            "category": category,
            "target_allocation": allocation,
            "current_allocation": 0.0,
            "amount": 0.0,
            "value": 0.0,
            "average_price": 0.0,
            "current_price": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_pnl": 0.0,
            "volatility": 0.0,
            "correlation": {},
            "risk_metrics": {},
            "last_updated": datetime.now().isoformat(),
            "added_at": datetime.now().isoformat()
        }
        
        # Thêm vào danh mục
        self.portfolio[symbol] = asset_info
        
        # Cập nhật phân bổ mục tiêu nếu cần
        self._rebalance_target_allocation()
        
        self.logger.info(f"Đã thêm tài sản {symbol} ({category}) vào danh mục với phân bổ mục tiêu {allocation:.2%}")
        
        return {
            "status": "success",
            "message": f"Đã thêm tài sản {symbol} vào danh mục",
            "asset": asset_info
        }
    
    def remove_asset(self, symbol: str, close_positions: bool = True) -> Dict[str, Any]:
        """
        Xóa tài sản khỏi danh mục đầu tư.
        
        Args:
            symbol: Mã tài sản
            close_positions: Đóng các vị thế của tài sản này
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem tài sản có tồn tại không
        if symbol not in self.portfolio:
            self.logger.warning(f"Tài sản {symbol} không tồn tại trong danh mục")
            return {
                "status": "error",
                "message": f"Tài sản {symbol} không tồn tại trong danh mục"
            }
        
        # Lấy thông tin tài sản
        asset_info = self.portfolio[symbol]
        
        # Đóng các vị thế nếu cần
        if close_positions and symbol in self.positions:
            self.close_position(symbol)
        
        # Xóa khỏi danh mục
        del self.portfolio[symbol]
        if symbol in self.positions:
            del self.positions[symbol]
        
        # Cập nhật phân bổ mục tiêu
        self._rebalance_target_allocation()
        
        self.logger.info(f"Đã xóa tài sản {symbol} khỏi danh mục")
        
        return {
            "status": "success",
            "message": f"Đã xóa tài sản {symbol} khỏi danh mục",
            "asset": asset_info
        }
    
    def update_asset_price(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Cập nhật giá hiện tại của tài sản.
        
        Args:
            symbol: Mã tài sản
            current_price: Giá hiện tại
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem tài sản có tồn tại không
        if symbol not in self.portfolio:
            self.logger.warning(f"Tài sản {symbol} không tồn tại trong danh mục")
            return {
                "status": "error",
                "message": f"Tài sản {symbol} không tồn tại trong danh mục"
            }
        
        # Lấy thông tin tài sản
        asset_info = self.portfolio[symbol]
        
        # Lưu giá cũ
        previous_price = asset_info["current_price"]
        
        # Cập nhật giá
        asset_info["current_price"] = current_price
        asset_info["last_updated"] = datetime.now().isoformat()
        
        # Tính lại giá trị và unrealized P&L
        if asset_info["amount"] > 0:
            old_value = asset_info["value"]
            new_value = asset_info["amount"] * current_price
            
            asset_info["value"] = new_value
            asset_info["unrealized_pnl"] = (current_price - asset_info["average_price"]) * asset_info["amount"]
            asset_info["total_pnl"] = asset_info["unrealized_pnl"] + asset_info["realized_pnl"]
            
            # Cập nhật phân bổ hiện tại
            self._update_current_allocation()
            
            # Cập nhật các chỉ số của danh mục
            self._update_portfolio_stats()
            
            # Kiểm tra điều kiện stop loss và take profit
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Kiểm tra stop loss
                if self.stop_loss_manager.check_stop_loss(position, current_price):
                    self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    self.close_position(symbol, current_price, reason="stop_loss")
                
                # Kiểm tra take profit
                elif self.take_profit_manager.check_take_profit(position, current_price):
                    self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                    self.close_position(symbol, current_price, reason="take_profit")
                
                # Cập nhật trailing stop nếu được kích hoạt
                else:
                    self.stop_loss_manager.update_trailing_stop(position, current_price)
        
        # Ghi log nếu giá thay đổi đáng kể
        if previous_price > 0:
            price_change_pct = (current_price - previous_price) / previous_price
            if abs(price_change_pct) > 0.05:  # Thay đổi > 5%
                self.logger.info(f"Giá của {symbol} thay đổi {price_change_pct:.2%} từ {previous_price} lên {current_price}")
        
        return {
            "status": "success",
            "message": f"Đã cập nhật giá của {symbol} thành {current_price}",
            "asset": asset_info
        }
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        risk_amount: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Mở vị thế mới.
        
        Args:
            symbol: Mã tài sản
            side: Phía vị thế ('long' hoặc 'short')
            entry_price: Giá vào
            risk_amount: Số tiền rủi ro (None để sử dụng risk_per_trade)
            stop_loss_price: Giá stop loss
            take_profit_price: Giá take profit
            leverage: Đòn bẩy
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem tài sản có tồn tại không
        if symbol not in self.portfolio:
            self.logger.warning(f"Tài sản {symbol} không tồn tại trong danh mục")
            return {
                "status": "error",
                "message": f"Tài sản {symbol} không tồn tại trong danh mục"
            }
        
        # Kiểm tra số lượng vị thế mở
        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"Đã đạt số lượng vị thế tối đa ({self.max_positions})")
            return {
                "status": "error",
                "message": f"Đã đạt số lượng vị thế tối đa ({self.max_positions})"
            }
        
        # Kiểm tra giới hạn vị thế tương quan
        correlation_positions = self._get_correlated_positions(symbol)
        if len(correlation_positions) >= self.max_correlated_positions:
            self.logger.warning(f"Đã đạt số lượng vị thế tương quan tối đa ({self.max_correlated_positions}) cho {symbol}")
            return {
                "status": "error",
                "message": f"Đã đạt số lượng vị thế tương quan tối đa ({self.max_correlated_positions})"
            }
        
        # Kiểm tra phía vị thế
        side = side.lower()
        if side not in ["long", "short"]:
            self.logger.warning(f"Phía vị thế không hợp lệ: {side}")
            return {
                "status": "error",
                "message": f"Phía vị thế không hợp lệ: {side}"
            }
        
        # Tính size vị thế dựa trên risk
        if risk_amount is None:
            risk_amount = self.current_capital * self.risk_per_trade
        
        # Tính stop loss price nếu không được cung cấp
        if stop_loss_price is None:
            if side == "long":
                # Stop loss 2% dưới giá vào cho vị thế Long
                stop_loss_price = entry_price * 0.98
            else:
                # Stop loss 2% trên giá vào cho vị thế Short
                stop_loss_price = entry_price * 1.02
        
        # Tính take profit price nếu không được cung cấp
        if take_profit_price is None:
            risk_per_share = abs(entry_price - stop_loss_price)
            if side == "long":
                take_profit_price = entry_price + (risk_per_share * self.take_profit_manager.risk_reward_ratio)
            else:
                take_profit_price = entry_price - (risk_per_share * self.take_profit_manager.risk_reward_ratio)
        
        # Tính position size
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            self.logger.warning(f"Risk per share là 0, không thể tính position size")
            return {
                "status": "error",
                "message": "Risk per share là 0, không thể tính position size"
            }
        
        position_size = risk_amount / risk_per_share / leverage
        
        # Kiểm tra số dư
        position_value = position_size * entry_price
        if position_value > self.current_capital:
            self.logger.warning(f"Không đủ vốn để mở vị thế ({position_value} > {self.current_capital})")
            
            # Điều chỉnh position size
            position_size = self.current_capital / entry_price * 0.95  # Giảm 5% để tránh sát biên
            position_value = position_size * entry_price
            
            self.logger.info(f"Đã điều chỉnh position size xuống {position_size}")
        
        # Kiểm tra xem vị thế đã tồn tại chưa
        if symbol in self.positions:
            self.logger.warning(f"Vị thế cho {symbol} đã tồn tại, sẽ đóng vị thế cũ trước")
            self.close_position(symbol, entry_price)
        
        # Tạo thông tin vị thế
        position = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "size": position_size,
            "value": position_value,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_percent": 0.0,
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "status": "open",
            "initial_stop_loss": stop_loss_price,
            "trailing_stop": stop_loss_price
        }
        
        # Thêm vào danh sách vị thế
        self.positions[symbol] = position
        
        # Cập nhật thông tin tài sản
        asset = self.portfolio[symbol]
        
        # Cập nhật average price và amount
        if asset["amount"] == 0:
            asset["average_price"] = entry_price
            asset["amount"] = position_size
        else:
            total_value = asset["average_price"] * asset["amount"] + entry_price * position_size
            total_amount = asset["amount"] + position_size
            asset["average_price"] = total_value / total_amount
            asset["amount"] = total_amount
        
        asset["value"] = asset["amount"] * entry_price
        asset["current_price"] = entry_price
        asset["unrealized_pnl"] = 0.0  # Reset khi mở vị thế mới
        
        # Trừ vốn sử dụng
        self.current_capital -= position_value
        
        # Cập nhật phân bổ hiện tại
        self._update_current_allocation()
        
        # Ghi log
        self.logger.info(f"Đã mở vị thế {side} cho {symbol}: {position_size} @ {entry_price} (Stop: {stop_loss_price}, Take: {take_profit_price})")
        
        return {
            "status": "success",
            "message": f"Đã mở vị thế {side} cho {symbol}",
            "position": position
        }
    
    def close_position(self, symbol: str, exit_price: Optional[float] = None, 
                      amount: Optional[float] = None, reason: str = "manual") -> Dict[str, Any]:
        """
        Đóng vị thế.
        
        Args:
            symbol: Mã tài sản
            exit_price: Giá thoát (None để sử dụng giá hiện tại)
            amount: Số lượng đóng (None để đóng toàn bộ)
            reason: Lý do đóng vị thế
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem vị thế có tồn tại không
        if symbol not in self.positions:
            self.logger.warning(f"Vị thế cho {symbol} không tồn tại")
            return {
                "status": "error",
                "message": f"Vị thế cho {symbol} không tồn tại"
            }
        
        # Lấy thông tin vị thế
        position = self.positions[symbol]
        
        # Lấy thông tin tài sản
        asset = self.portfolio[symbol]
        
        # Sử dụng giá hiện tại nếu không cung cấp giá thoát
        if exit_price is None:
            exit_price = asset["current_price"]
            if exit_price == 0:
                self.logger.warning(f"Giá hiện tại của {symbol} là 0, không thể đóng vị thế")
                return {
                    "status": "error",
                    "message": f"Giá hiện tại của {symbol} là 0, không thể đóng vị thế"
                }
        
        # Sử dụng toàn bộ size nếu không chỉ định amount
        if amount is None:
            amount = position["size"]
        
        # Kiểm tra amount
        if amount > position["size"]:
            self.logger.warning(f"Số lượng đóng ({amount}) lớn hơn kích thước vị thế ({position['size']})")
            amount = position["size"]
        
        # Tính P&L
        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * amount
        else:  # short
            pnl = (position["entry_price"] - exit_price) * amount
        
        # Tạo bản ghi lịch sử vị thế
        position_history = {
            "symbol": symbol,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size": amount,
            "leverage": position["leverage"],
            "entry_time": position["entry_time"],
            "exit_time": datetime.now().isoformat(),
            "pnl": pnl,
            "pnl_percent": pnl / (position["entry_price"] * amount) * 100,
            "duration": (datetime.now() - datetime.fromisoformat(position["entry_time"])).total_seconds() / 3600,  # hours
            "reason": reason,
            "stop_loss": position["stop_loss"],
            "take_profit": position["take_profit"],
            "risk_amount": position["risk_amount"]
        }
        
        # Thêm vào lịch sử
        self.position_history.append(position_history)
        
        # Cập nhật thống kê
        if pnl > 0:
            self.stats["win_count"] += 1
        else:
            self.stats["loss_count"] += 1
        
        self.stats["realized_pnl"] += pnl
        self.stats["total_pnl"] = self.stats["realized_pnl"] + self.stats["unrealized_pnl"]
        
        # Cập nhật thông tin tài sản
        asset["realized_pnl"] += pnl
        asset["total_pnl"] = asset["realized_pnl"] + asset["unrealized_pnl"]
        
        # Giảm amount nếu chỉ đóng một phần
        if amount < position["size"]:
            position["size"] -= amount
            position["value"] = position["size"] * exit_price
            
            asset["amount"] -= amount
            
            self.logger.info(f"Đã đóng một phần vị thế {symbol}: {amount} @ {exit_price}, P&L: {pnl:.2f}")
        else:
            # Xóa khỏi danh sách vị thế nếu đóng toàn bộ
            del self.positions[symbol]
            
            asset["amount"] = 0
            asset["unrealized_pnl"] = 0
            
            self.logger.info(f"Đã đóng toàn bộ vị thế {symbol} @ {exit_price}, P&L: {pnl:.2f}, Lý do: {reason}")
        
        # Cập nhật vốn
        position_value = amount * exit_price
        self.current_capital += position_value + pnl
        
        # Cập nhật giá trị tài sản
        asset["value"] = asset["amount"] * exit_price
        
        # Cập nhật phân bổ hiện tại
        self._update_current_allocation()
        
        # Cập nhật các chỉ số của danh mục
        self._update_portfolio_stats()
        
        return {
            "status": "success",
            "message": f"Đã đóng vị thế {symbol}",
            "pnl": pnl,
            "history": position_history
        }
    
    def _update_current_allocation(self) -> None:
        """
        Cập nhật phân bổ hiện tại của tất cả tài sản trong danh mục.
        """
        # Tính tổng giá trị danh mục
        total_portfolio_value = self.current_capital + sum(asset["value"] for asset in self.portfolio.values())
        
        if total_portfolio_value <= 0:
            self.logger.warning("Tổng giá trị danh mục <= 0, không thể tính phân bổ")
            return
        
        # Cập nhật phân bổ cho từng tài sản
        for symbol, asset in self.portfolio.items():
            asset["current_allocation"] = asset["value"] / total_portfolio_value
    
    def _rebalance_target_allocation(self) -> None:
        """
        Cân bằng lại phân bổ mục tiêu để tổng bằng 100%.
        """
        # Tính tổng phân bổ mục tiêu
        total_target = sum(asset["target_allocation"] for asset in self.portfolio.values())
        
        if total_target == 0:
            self.logger.warning("Tổng phân bổ mục tiêu = 0, không thể cân bằng lại")
            return
        
        # Điều chỉnh nếu cần
        if total_target != 1.0:
            scale_factor = 1.0 / total_target
            
            for asset in self.portfolio.values():
                asset["target_allocation"] *= scale_factor
            
            self.logger.info(f"Đã cân bằng lại phân bổ mục tiêu (scale: {scale_factor:.4f})")
    
    def check_rebalance_needed(self) -> bool:
        """
        Kiểm tra xem có cần cân bằng lại danh mục không.
        
        Returns:
            True nếu cần cân bằng lại, False nếu không
        """
        # Kiểm tra thời gian kể từ lần cân bằng cuối
        time_diff = datetime.now() - self.last_rebalance_time
        
        if self.rebalance_frequency == "daily" and time_diff.days >= 1:
            return True
        elif self.rebalance_frequency == "weekly" and time_diff.days >= 7:
            return True
        elif self.rebalance_frequency == "monthly" and time_diff.days >= 30:
            return True
        
        # Kiểm tra độ lệch so với phân bổ mục tiêu
        max_deviation = 0.0
        
        for symbol, asset in self.portfolio.items():
            deviation = abs(asset["current_allocation"] - asset["target_allocation"])
            max_deviation = max(max_deviation, deviation)
        
        # Cân bằng lại nếu độ lệch > 10%
        return max_deviation > 0.1
    
    def rebalance_portfolio(self, current_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Cân bằng lại danh mục để khớp với phân bổ mục tiêu.
        
        Args:
            current_prices: Dict giá hiện tại của các tài sản
            
        Returns:
            Dict chứa thông tin kết quả
        """
        # Kiểm tra xem có tài sản nào trong danh mục không
        if not self.portfolio:
            self.logger.warning("Không có tài sản nào trong danh mục để cân bằng lại")
            return {
                "status": "error",
                "message": "Không có tài sản nào trong danh mục"
            }
        
        # Cập nhật giá hiện tại nếu được cung cấp
        if current_prices:
            for symbol, price in current_prices.items():
                if symbol in self.portfolio:
                    self.update_asset_price(symbol, price)
        
        # Tính tổng giá trị danh mục
        total_portfolio_value = self.current_capital + sum(asset["value"] for asset in self.portfolio.values())
        
        # Tính giá trị mục tiêu cho mỗi tài sản
        target_values = {}
        rebalance_actions = []
        
        for symbol, asset in self.portfolio.items():
            # Giá trị mục tiêu
            target_value = total_portfolio_value * asset["target_allocation"]
            
            # Giá trị hiện tại
            current_value = asset["value"]
            
            # Độ chênh lệch
            diff = target_value - current_value
            
            # Lưu thông tin
            target_values[symbol] = {
                "target_value": target_value,
                "current_value": current_value,
                "diff": diff
            }
            
            # Xác định hành động cần thực hiện
            if abs(diff) / total_portfolio_value > 0.01:  # Chênh lệch > 1% tổng giá trị
                if diff > 0:
                    # Cần mua thêm
                    # Tính số lượng cần mua
                    quantity = diff / asset["current_price"] if asset["current_price"] > 0 else 0
                    
                    if quantity > 0:
                        rebalance_actions.append({
                            "symbol": symbol,
                            "action": "buy",
                            "amount": quantity,
                            "value": diff,
                            "current_price": asset["current_price"]
                        })
                else:
                    # Cần bán bớt
                    # Tính số lượng cần bán
                    quantity = abs(diff) / asset["current_price"] if asset["current_price"] > 0 else 0
                    
                    if quantity > 0:
                        rebalance_actions.append({
                            "symbol": symbol,
                            "action": "sell",
                            "amount": quantity,
                            "value": abs(diff),
                            "current_price": asset["current_price"]
                        })
        
        # Thực hiện các hành động cân bằng
        executed_actions = []
        
        for action in rebalance_actions:
            symbol = action["symbol"]
            action_type = action["action"]
            amount = action["amount"]
            price = action["current_price"]
            
            if action_type == "buy":
                # Kiểm tra vốn
                if action["value"] > self.current_capital:
                    # Điều chỉnh số lượng mua
                    adjusted_amount = self.current_capital / price if price > 0 else 0
                    action["amount"] = adjusted_amount
                    action["value"] = adjusted_amount * price
                
                # Cập nhật thông tin tài sản
                asset = self.portfolio[symbol]
                
                if asset["amount"] == 0:
                    asset["average_price"] = price
                    asset["amount"] = action["amount"]
                else:
                    total_value = asset["average_price"] * asset["amount"] + price * action["amount"]
                    total_amount = asset["amount"] + action["amount"]
                    asset["average_price"] = total_value / total_amount
                    asset["amount"] = total_amount
                
                asset["value"] = asset["amount"] * price
                
                # Trừ vốn
                self.current_capital -= action["value"]
                
                self.logger.info(f"Rebalance: Mua {action['amount']} {symbol} @ {price}")
                
            elif action_type == "sell":
                # Kiểm tra số lượng hiện có
                asset = self.portfolio[symbol]
                
                if action["amount"] > asset["amount"]:
                    action["amount"] = asset["amount"]
                    action["value"] = asset["amount"] * price
                
                # Tính P&L
                pnl = (price - asset["average_price"]) * action["amount"]
                
                # Cập nhật thông tin tài sản
                asset["amount"] -= action["amount"]
                asset["value"] = asset["amount"] * price
                asset["realized_pnl"] += pnl
                
                # Thêm vốn
                self.current_capital += action["value"]
                
                self.logger.info(f"Rebalance: Bán {action['amount']} {symbol} @ {price}, P&L: {pnl:.2f}")
            
            # Thêm vào danh sách đã thực hiện
            executed_actions.append(action)
        
        # Cập nhật phân bổ hiện tại
        self._update_current_allocation()
        
        # Cập nhật thời gian cân bằng cuối
        self.last_rebalance_time = datetime.now()
        
        return {
            "status": "success",
            "message": f"Đã cân bằng lại danh mục ({len(executed_actions)} hành động)",
            "actions": executed_actions,
            "target_values": target_values
        }
    
    def update_correlation_matrix(self, price_history: Dict[str, pd.DataFrame]) -> None:
        """
        Cập nhật ma trận tương quan giữa các tài sản.
        
        Args:
            price_history: Dict với key là symbol và value là DataFrame giá
        """
        symbols = list(self.portfolio.keys())
        
        if len(symbols) < 2:
            self.logger.info("Cần ít nhất 2 tài sản để tính ma trận tương quan")
            return
        
        # Tạo DataFrame với returns của các tài sản
        returns_data = {}
        
        for symbol in symbols:
            if symbol in price_history:
                df = price_history[symbol]
                
                if 'close' in df.columns:
                    # Tính returns
                    returns = df['close'].pct_change().dropna()
                    returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            self.logger.warning("Không đủ dữ liệu returns để tính ma trận tương quan")
            return
        
        # Tạo DataFrame returns
        returns_df = pd.DataFrame(returns_data)
        
        # Tính ma trận tương quan
        corr_matrix = returns_df.corr()
        
        # Lưu ma trận tương quan
        self.correlation_matrix = corr_matrix
        
        # Cập nhật thông tin tương quan cho mỗi tài sản
        for symbol in symbols:
            if symbol in corr_matrix.index:
                correlations = {}
                
                for other_symbol in symbols:
                    if other_symbol != symbol and other_symbol in corr_matrix.columns:
                        correlations[other_symbol] = corr_matrix.loc[symbol, other_symbol]
                
                # Lưu vào thông tin tài sản
                if symbol in self.portfolio:
                    self.portfolio[symbol]["correlation"] = correlations
        
        self.logger.info(f"Đã cập nhật ma trận tương quan cho {len(returns_data)} tài sản")
    
    def _get_correlated_positions(self, symbol: str, threshold: Optional[float] = None) -> List[str]:
        """
        Lấy danh sách các vị thế có tương quan cao với tài sản đã cho.
        
        Args:
            symbol: Mã tài sản
            threshold: Ngưỡng tương quan (None để sử dụng giá trị mặc định)
            
        Returns:
            Danh sách các symbol có tương quan cao
        """
        if threshold is None:
            threshold = self.correlation_threshold
        
        correlated_positions = []
        
        # Chỉ kiểm tra nếu có ma trận tương quan và tài sản trong danh mục
        if not self.correlation_matrix.empty and symbol in self.portfolio:
            asset_correlations = self.portfolio[symbol].get("correlation", {})
            
            for other_symbol, correlation in asset_correlations.items():
                if abs(correlation) >= threshold and other_symbol in self.positions:
                    correlated_positions.append(other_symbol)
        
        return correlated_positions
    
    def _update_portfolio_stats(self) -> None:
        """
        Cập nhật các thống kê của danh mục đầu tư.
        """
        # Tính unrealized P&L
        total_unrealized_pnl = sum(asset["unrealized_pnl"] for asset in self.portfolio.values())
        self.stats["unrealized_pnl"] = total_unrealized_pnl
        
        # Cập nhật tổng P&L
        self.stats["total_pnl"] = self.stats["realized_pnl"] + self.stats["unrealized_pnl"]
        
        # Tính giá trị danh mục
        portfolio_value = self.current_capital + sum(asset["value"] for asset in self.portfolio.values())
        
        # Lưu vào lịch sử
        timestamp = datetime.now().isoformat()
        self.stats["portfolio_value_history"].append({
            "timestamp": timestamp,
            "value": portfolio_value,
            "capital": self.current_capital,
            "assets_value": portfolio_value - self.current_capital
        })
        
        # Giới hạn kích thước lịch sử
        if len(self.stats["portfolio_value_history"]) > 1000:
            self.stats["portfolio_value_history"] = self.stats["portfolio_value_history"][-1000:]
        
        # Tính win rate
        total_trades = self.stats["win_count"] + self.stats["loss_count"]
        self.stats["win_rate"] = self.stats["win_count"] / total_trades if total_trades > 0 else 0.0
        
        # Tính profit factor nếu có đủ dữ liệu
        if len(self.position_history) >= 10:
            total_profit = sum(pos["pnl"] for pos in self.position_history if pos["pnl"] > 0)
            total_loss = sum(abs(pos["pnl"]) for pos in self.position_history if pos["pnl"] < 0)
            
            self.stats["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Kiểm tra và cập nhật max drawdown
        self.drawdown_manager.update_drawdown_metrics(portfolio_value)
        self.stats["max_drawdown"] = self.drawdown_manager.max_drawdown_percent
        
        # Cập nhật các chỉ số rủi ro
        self.stats["risk_metrics"] = self.risk_calculator.calculate_risk_metrics(
            self.portfolio, self.positions, self.correlation_matrix, portfolio_value
        )
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của danh mục đầu tư.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        # Tính giá trị danh mục
        portfolio_value = self.current_capital + sum(asset["value"] for asset in self.portfolio.values())
        
        # Số lượng tài sản
        asset_count = len(self.portfolio)
        
        # Số lượng vị thế mở
        open_positions_count = len(self.positions)
        
        # Tổng giá trị các vị thế
        positions_value = sum(position["value"] for position in self.positions.values())
        
        # Tạo phân bổ theo category
        category_allocation = {}
        
        for asset in self.portfolio.values():
            category = asset["category"]
            
            if category not in category_allocation:
                category_allocation[category] = 0.0
                
            category_allocation[category] += asset["value"]
        
        # Chuyển sang phần trăm
        for category in category_allocation:
            category_allocation[category] = category_allocation[category] / portfolio_value * 100 if portfolio_value > 0 else 0
        
        return {
            "portfolio_value": portfolio_value,
            "current_capital": self.current_capital,
            "assets_value": portfolio_value - self.current_capital,
            "asset_count": asset_count,
            "open_positions": open_positions_count,
            "positions_value": positions_value,
            "total_pnl": self.stats["total_pnl"],
            "realized_pnl": self.stats["realized_pnl"],
            "unrealized_pnl": self.stats["unrealized_pnl"],
            "win_rate": self.stats["win_rate"] * 100,
            "profit_factor": self.stats["profit_factor"],
            "max_drawdown": self.stats["max_drawdown"] * 100,
            "risk_profile": self.risk_profile,
            "category_allocation": category_allocation,
            "last_rebalance": self.last_rebalance_time.isoformat(),
            "risk_metrics": self.stats["risk_metrics"]
        }
    
    def get_position_summary(self) -> List[Dict[str, Any]]:
        """
        Lấy tóm tắt về các vị thế hiện tại.
        
        Returns:
            Danh sách thông tin tóm tắt vị thế
        """
        position_summary = []
        
        for symbol, position in self.positions.items():
            # Tính P&L
            if symbol in self.portfolio:
                current_price = self.portfolio[symbol]["current_price"]
                
                if position["side"] == "long":
                    unrealized_pnl = (current_price - position["entry_price"]) * position["size"]
                else:  # short
                    unrealized_pnl = (position["entry_price"] - current_price) * position["size"]
                
                # Tính phần trăm P&L
                entry_value = position["entry_price"] * position["size"]
                unrealized_pnl_percent = unrealized_pnl / entry_value * 100 if entry_value > 0 else 0
                
                # Khoảng cách đến stop loss và take profit
                if position["side"] == "long":
                    stop_distance = (current_price - position["stop_loss"]) / current_price * 100 if current_price > 0 else 0
                    tp_distance = (position["take_profit"] - current_price) / current_price * 100 if current_price > 0 else 0
                else:  # short
                    stop_distance = (position["stop_loss"] - current_price) / current_price * 100 if current_price > 0 else 0
                    tp_distance = (current_price - position["take_profit"]) / current_price * 100 if current_price > 0 else 0
                
                # Thời gian nắm giữ
                entry_time = datetime.fromisoformat(position["entry_time"])
                hold_time = (datetime.now() - entry_time).total_seconds() / 3600  # hours
                
                # Thêm vào summary
                position_summary.append({
                    "symbol": symbol,
                    "side": position["side"],
                    "size": position["size"],
                    "entry_price": position["entry_price"],
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": unrealized_pnl_percent,
                    "stop_loss": position["stop_loss"],
                    "take_profit": position["take_profit"],
                    "stop_distance_percent": stop_distance,
                    "tp_distance_percent": tp_distance,
                    "hold_time_hours": hold_time,
                    "leverage": position["leverage"],
                    "value": position["value"],
                    "category": self.portfolio[symbol]["category"]
                })
        
        return position_summary
    
    def get_trade_history_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về lịch sử giao dịch.
        
        Returns:
            Dict chứa thông kê giao dịch
        """
        if not self.position_history:
            return {
                "total_trades": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "average_hold_time": 0.0,
                "total_pnl": 0.0
            }
        
        # Số lượng giao dịch
        total_trades = len(self.position_history)
        
        # Số lượng thắng/thua
        win_trades = [trade for trade in self.position_history if trade["pnl"] > 0]
        loss_trades = [trade for trade in self.position_history if trade["pnl"] <= 0]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        
        # Tỷ lệ thắng
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # Tổng lợi nhuận/lỗ
        total_profit = sum(trade["pnl"] for trade in win_trades)
        total_loss = sum(abs(trade["pnl"]) for trade in loss_trades)
        
        # Hệ số lợi nhuận
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Trung bình thắng/thua
        average_win = total_profit / win_count if win_count > 0 else 0.0
        average_loss = total_loss / loss_count if loss_count > 0 else 0.0
        
        # Thắng/thua lớn nhất
        largest_win = max([trade["pnl"] for trade in win_trades]) if win_trades else 0.0
        largest_loss = max([abs(trade["pnl"]) for trade in loss_trades]) if loss_trades else 0.0
        
        # Thời gian nắm giữ trung bình
        average_hold_time = sum(trade["duration"] for trade in self.position_history) / total_trades
        
        # Tổng P&L
        total_pnl = sum(trade["pnl"] for trade in self.position_history)
        
        # Phân tích theo lý do đóng vị thế
        close_reasons = {}
        
        for trade in self.position_history:
            reason = trade.get("reason", "manual")
            
            if reason not in close_reasons:
                close_reasons[reason] = {
                    "count": 0,
                    "total_pnl": 0.0,
                    "win_count": 0
                }
            
            close_reasons[reason]["count"] += 1
            close_reasons[reason]["total_pnl"] += trade["pnl"]
            
            if trade["pnl"] > 0:
                close_reasons[reason]["win_count"] += 1
        
        # Tính win rate và average pnl cho mỗi lý do
        for reason, stats in close_reasons.items():
            stats["win_rate"] = stats["win_count"] / stats["count"] if stats["count"] > 0 else 0.0
            stats["average_pnl"] = stats["total_pnl"] / stats["count"] if stats["count"] > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "risk_reward_ratio": abs(average_win / average_loss) if average_loss != 0 else float('inf'),
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "average_hold_time": average_hold_time,
            "total_pnl": total_pnl,
            "close_reasons": close_reasons
        }
    
    def export_portfolio_state(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Xuất trạng thái danh mục đầu tư để lưu.
        
        Args:
            file_path: Đường dẫn file để lưu (None để không lưu)
            
        Returns:
            Dict chứa trạng thái danh mục
        """
        # Tạo trạng thái
        state = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self.current_capital + sum(asset["value"] for asset in self.portfolio.values()),
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "risk_profile": self.risk_profile,
            "risk_per_trade": self.risk_per_trade,
            "max_positions": self.max_positions,
            "correlation_threshold": self.correlation_threshold,
            "max_drawdown_percent": self.max_drawdown_percent,
            "rebalance_frequency": self.rebalance_frequency,
            "last_rebalance_time": self.last_rebalance_time.isoformat(),
            "target_allocation": self.target_allocation,
            "assets": self.portfolio,
            "positions": self.positions,
            "position_history": self.position_history,
            "stats": self.stats
        }
        
        # Lưu vào file nếu cần
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=4, ensure_ascii=False)
                    
                self.logger.info(f"Đã xuất trạng thái danh mục đầu tư vào {file_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi xuất trạng thái: {str(e)}")
        
        return state
    
    def import_portfolio_state(self, state: Dict[str, Any] = None, file_path: Optional[str] = None) -> bool:
        """
        Nhập trạng thái danh mục đầu tư.
        
        Args:
            state: Dict chứa trạng thái danh mục
            file_path: Đường dẫn file để đọc (None nếu state đã được cung cấp)
            
        Returns:
            True nếu nhập thành công, False nếu không
        """
        # Đọc từ file nếu cần
        if state is None and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc trạng thái từ file: {str(e)}")
                return False
        
        if not state:
            self.logger.error("Không có dữ liệu trạng thái để nhập")
            return False
        
        try:
            # Cập nhật các thuộc tính cơ bản
            self.current_capital = state.get("current_capital", self.current_capital)
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            self.risk_profile = state.get("risk_profile", self.risk_profile)
            self.risk_per_trade = state.get("risk_per_trade", self.risk_per_trade)
            self.max_positions = state.get("max_positions", self.max_positions)
            self.correlation_threshold = state.get("correlation_threshold", self.correlation_threshold)
            self.max_drawdown_percent = state.get("max_drawdown_percent", self.max_drawdown_percent)
            self.rebalance_frequency = state.get("rebalance_frequency", self.rebalance_frequency)
            
            # Cập nhật target allocation
            self.target_allocation = state.get("target_allocation", self.target_allocation)
            
            # Cập nhật last_rebalance_time
            if "last_rebalance_time" in state:
                self.last_rebalance_time = datetime.fromisoformat(state["last_rebalance_time"])
            
            # Cập nhật các tài sản
            self.portfolio = state.get("assets", {})
            
            # Cập nhật các vị thế
            self.positions = state.get("positions", {})
            
            # Cập nhật lịch sử vị thế
            self.position_history = state.get("position_history", [])
            
            # Cập nhật thống kê
            self.stats = state.get("stats", self.stats)
            
            self.logger.info(f"Đã nhập trạng thái danh mục đầu tư (từ {state.get('timestamp', 'unknown')})")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập trạng thái: {str(e)}")
            return False