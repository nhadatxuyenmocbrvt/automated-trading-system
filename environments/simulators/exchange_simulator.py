"""
Mô phỏng sàn giao dịch.
File này cung cấp lớp chính ExchangeSimulator để mô phỏng sàn giao dịch
và quản lý giao dịch, vị thế và tài khoản. Tích hợp với MarketSimulator
để cung cấp mô phỏng toàn diện cho backtesting và huấn luyện.
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
from environments.simulators.market_simulator import MarketSimulator, HistoricalMarketSimulator, RealisticMarketSimulator
from environments.simulators.exchange_simulator.base_simulator import BaseExchangeSimulator
from environments.simulators.exchange_simulator.realistic_simulator import RealisticExchangeSimulator
from environments.simulators.exchange_simulator.order_manager import OrderManager
from environments.simulators.exchange_simulator.position_manager import PositionManager
from environments.simulators.exchange_simulator.account_manager import AccountManager

class ExchangeSimulator(BaseExchangeSimulator):
    """
    Lớp chính mô phỏng sàn giao dịch.
    Cung cấp các hàm tiện ích để tạo và quản lý các loại mô phỏng khác nhau.
    """
    
    def __init__(
        self,
        market_simulator: Optional[MarketSimulator] = None,
        simulator_type: str = "basic",  # "basic", "historical", "realistic"
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        leverage: float = 1.0,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        min_order_size: float = 0.001,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo mô phỏng sàn giao dịch.
        
        Args:
            market_simulator: Mô phỏng thị trường
            simulator_type: Loại mô phỏng
            initial_balance: Số dư ban đầu theo loại tiền
            leverage: Đòn bẩy mặc định
            maker_fee: Phí maker (%)
            taker_fee: Phí taker (%)
            min_order_size: Kích thước lệnh tối thiểu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("exchange_simulator")
        
        # Khởi tạo mô phỏng sàn giao dịch theo loại
        self.simulator_type = simulator_type
        
        if simulator_type == "realistic":
            self.simulator = RealisticExchangeSimulator(
                market_simulator=market_simulator,
                initial_balance=initial_balance,
                leverage=leverage,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                min_order_size=min_order_size,
                logger=self.logger
            )
        else:
            # Basic hoặc historical
            self.simulator = BaseExchangeSimulator(
                market_simulator=market_simulator,
                initial_balance=initial_balance,
                leverage=leverage,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                min_order_size=min_order_size,
                logger=self.logger
            )
        
        self.logger.info(f"Đã khởi tạo ExchangeSimulator với loại {simulator_type}")
    
    def reset(self) -> Dict[str, Any]:
        """
        Đặt lại trạng thái mô phỏng sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái sàn giao dịch sau khi đặt lại
        """
        return self.simulator.reset()
    
    def step(self) -> Dict[str, Any]:
        """
        Tiến hành một bước mô phỏng.
        
        Returns:
            Dict chứa thông tin trạng thái sau bước mô phỏng
        """
        return self.simulator.step()
    
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
        return self.simulator.submit_order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            time_in_force=time_in_force,
            params=params
        )
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Hủy một lệnh giao dịch.
        
        Args:
            order_id: ID lệnh cần hủy
            
        Returns:
            Dict chứa kết quả hủy lệnh
        """
        return self.simulator.cancel_order(order_id)
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin của một lệnh.
        
        Args:
            order_id: ID lệnh
            
        Returns:
            Dict chứa thông tin lệnh hoặc None nếu không tìm thấy
        """
        return self.simulator.get_order(order_id)
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các lệnh theo điều kiện.
        
        Args:
            symbol: Lọc theo cặp giao dịch
            status: Lọc theo trạng thái
            
        Returns:
            Danh sách các lệnh thỏa mãn điều kiện
        """
        return self.simulator.get_orders(symbol, status)
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Lấy thông tin vị thế cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Dict chứa thông tin vị thế hoặc None nếu không có vị thế
        """
        return self.simulator.get_position(symbol)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả các vị thế hiện tại.
        
        Returns:
            Danh sách các vị thế
        """
        return self.simulator.get_positions()
    
    def get_balance(self, currency: Optional[str] = None) -> Union[Dict[str, float], float]:
        """
        Lấy số dư tài khoản.
        
        Args:
            currency: Loại tiền cần lấy số dư (None để lấy tất cả)
            
        Returns:
            Số dư của loại tiền hoặc dict chứa tất cả số dư
        """
        return self.simulator.get_balance(currency)
    
    def calculate_equity(self) -> float:
        """
        Tính tổng giá trị tài sản (equity) bao gồm số dư và vị thế mở.
        
        Returns:
            Tổng giá trị tài sản
        """
        return self.simulator.calculate_equity()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của sàn giao dịch.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        return self.simulator.get_state()
    
    def set_market_simulator(self, market_simulator: MarketSimulator) -> None:
        """
        Thiết lập mô phỏng thị trường để sử dụng.
        
        Args:
            market_simulator: Mô phỏng thị trường
        """
        self.simulator.set_market_simulator(market_simulator)
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Lưu trạng thái mô phỏng sàn giao dịch.
        
        Args:
            path: Đường dẫn file
        """
        self.simulator.save_state(path)
    
    def load_state(self, path: Union[str, Path]) -> bool:
        """
        Tải trạng thái mô phỏng sàn giao dịch.
        
        Args:
            path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        return self.simulator.load_state(path)
    
    @classmethod
    def create_historical_simulator(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        leverage: float = 1.0,
        min_order_size: float = 0.001,
        start_index: Optional[int] = None
    ) -> 'ExchangeSimulator':
        """
        Tạo một mô phỏng sàn giao dịch lịch sử.
        
        Args:
            data_path: Đường dẫn file dữ liệu
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            initial_balance: Số dư ban đầu
            maker_fee: Phí maker
            taker_fee: Phí taker
            leverage: Đòn bẩy
            min_order_size: Kích thước lệnh tối thiểu
            start_index: Chỉ số bắt đầu trong dữ liệu
            
        Returns:
            Đối tượng ExchangeSimulator
        """
        # Tạo market simulator
        market_simulator = HistoricalMarketSimulator(
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Đặt lại market simulator
        market_simulator.reset(idx=start_index)
        
        # Tạo exchange simulator
        simulator = cls(
            market_simulator=market_simulator,
            simulator_type="historical",
            initial_balance=initial_balance,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_order_size=min_order_size
        )
        
        return simulator
    
    @classmethod
    def create_realistic_simulator(
        cls,
        data_path: Optional[Union[str, Path]] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: Dict[str, float] = {"USDT": 10000.0},
        maker_fee: float = 0.0001,  # 0.01%
        taker_fee: float = 0.0005,  # 0.05%
        leverage: float = 1.0,
        min_order_size: float = 0.001,
        execution_delay: float = 0.5,
        order_rejection_prob: float = 0.03,
        volatility_scale: float = 1.0,
        market_impact_scale: float = 0.0002,
        start_index: Optional[int] = None
    ) -> 'ExchangeSimulator':
        """
        Tạo một mô phỏng sàn giao dịch thực tế.
        
        Args:
            data_path: Đường dẫn file dữ liệu
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            initial_balance: Số dư ban đầu
            maker_fee: Phí maker
            taker_fee: Phí taker
            leverage: Đòn bẩy
            min_order_size: Kích thước lệnh tối thiểu
            execution_delay: Độ trễ thực thi
            order_rejection_prob: Xác suất từ chối lệnh
            volatility_scale: Hệ số biến động
            market_impact_scale: Hệ số tác động thị trường
            start_index: Chỉ số bắt đầu trong dữ liệu
            
        Returns:
            Đối tượng ExchangeSimulator
        """
        # Tạo market simulator
        market_simulator = RealisticMarketSimulator(
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            volatility_scale=volatility_scale,
            market_impact_scale=market_impact_scale,
            execution_delay=execution_delay,
            order_rejection_prob=order_rejection_prob
        )
        
        # Đặt lại market simulator
        market_simulator.reset(idx=start_index)
        
        # Tạo exchange simulator
        simulator = cls(
            market_simulator=market_simulator,
            simulator_type="realistic",
            initial_balance=initial_balance,
            leverage=leverage,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            min_order_size=min_order_size
        )
        
        return simulator
    
    def run_simulation(
        self,
        num_steps: int,
        strategy: Optional[Callable] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Chạy mô phỏng với số lượng bước nhất định.
        
        Args:
            num_steps: Số lượng bước mô phỏng
            strategy: Hàm chiến lược giao dịch (nhận trạng thái, trả về lệnh)
            verbose: In thông tin mô phỏng
            callback: Hàm callback sau mỗi bước
            
        Returns:
            Dict chứa kết quả mô phỏng
        """
        # Đặt lại mô phỏng
        self.reset()
        
        # Lưu trữ lịch sử
        history = {
            "prices": [],
            "balances": [],
            "positions": [],
            "orders": [],
            "equity": []
        }
        
        # Chạy mô phỏng
        for i in range(num_steps):
            # Lấy trạng thái hiện tại
            state = self.get_state()
            
            # Áp dụng chiến lược nếu có
            if strategy:
                strategy_action = strategy(state)
                
                # Thực hiện hành động nếu có
                if strategy_action and "action" in strategy_action:
                    action = strategy_action["action"]
                    
                    if action == "buy":
                        self.submit_order(
                            symbol=strategy_action.get("symbol", "BTC/USDT"),
                            order_type=strategy_action.get("order_type", OrderType.MARKET.value),
                            side="buy",
                            amount=strategy_action.get("amount", 0.01),
                            price=strategy_action.get("price")
                        )
                    elif action == "sell":
                        self.submit_order(
                            symbol=strategy_action.get("symbol", "BTC/USDT"),
                            order_type=strategy_action.get("order_type", OrderType.MARKET.value),
                            side="sell",
                            amount=strategy_action.get("amount", 0.01),
                            price=strategy_action.get("price")
                        )
                    elif action == "close":
                        # Đóng vị thế
                        symbol = strategy_action.get("symbol", "BTC/USDT")
                        position = self.get_position(symbol)
                        
                        if position:
                            close_side = "sell" if position["side"] == "long" else "buy"
                            self.submit_order(
                                symbol=symbol,
                                order_type=OrderType.MARKET.value,
                                side=close_side,
                                amount=position["amount"]
                            )
            
            # Chạy một bước mô phỏng
            new_state = self.step()
            
            # Cập nhật lịch sử
            if "market_info" in new_state and new_state["market_info"]:
                market_prices = new_state["market_info"].get("prices", {})
                history["prices"].append(market_prices.get("close", 0))
            
            history["balances"].append(self.get_balance())
            history["positions"].append(self.get_positions())
            history["orders"].append(self.get_orders())
            history["equity"].append(self.calculate_equity())
            
            # In thông tin nếu cần
            if verbose and i % max(1, num_steps // 10) == 0:
                print(f"Bước {i}/{num_steps} - Equity: {history['equity'][-1]:.2f}")
            
            # Gọi callback nếu có
            if callback:
                callback(new_state, i, history)
        
        # Tính toán kết quả
        if history["equity"]:
            initial_equity = history["equity"][0]
            final_equity = history["equity"][-1]
            
            profit = final_equity - initial_equity
            profit_percent = (profit / initial_equity) * 100 if initial_equity > 0 else 0
            
            # Tính drawdown
            peak = history["equity"][0]
            max_drawdown = 0
            
            for equity in history["equity"]:
                if equity > peak:
                    peak = equity
                
                drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            result = {
                "initial_equity": initial_equity,
                "final_equity": final_equity,
                "profit": profit,
                "profit_percent": profit_percent,
                "max_drawdown": max_drawdown,
                "num_steps": num_steps,
                "history": history
            }
            
            if verbose:
                print(f"\nKết quả mô phỏng:")
                print(f"- Vốn ban đầu: {initial_equity:.2f}")
                print(f"- Vốn cuối: {final_equity:.2f}")
                print(f"- Lợi nhuận: {profit:.2f} ({profit_percent:.2f}%)")
                print(f"- Drawdown tối đa: {max_drawdown*100:.2f}%")
            
            return result
        
        return {"error": "Không có dữ liệu mô phỏng"}